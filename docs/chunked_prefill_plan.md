# Chunked Prefill for Kitty — Deep Implementation Plan

> Branch: `tzj/kvcache-offload` · Worktree: `/mnt/data/tzj/Code/Kitty/.claude/worktrees/kvcache-cpu-offload`
> Status: **PLAN ONLY — no code written yet.** Awaiting review.
> Verify under the **kitty** conda env (`/mnt/data/tzj/anaconda3/envs/kitty/bin/python`, torch 2.4.1+cu121 / transformers 4.57.6 / triton 3.0.0, **no vllm, no flash_attn** — all pinned) on **GPU0** (A100-40GB). The ambient shell python is a different env — do not validate there.

Produced by a fan-out workflow (5 code/research readers → architect → 3 adversarial critics) then reconciled against the source and the **already-committed KV offload** (`abaeb26`). Corrections from review are folded in; an appendix records what changed.

---

## 0. TL;DR

- **Motivation:** after KV offload, the 128k prefill peak is bounded by the **per-layer MLP activation transient** (Qwen ~9–10 GiB, GLM ~10 GiB) + the **attention residual** (~4 GiB @128k) + weights. KV offload can't touch those. **Chunked prefill** processes the prompt in token-chunks so the per-layer activation is `[…, C, …]` not `[…, S, …]`.
- **torch-native, NOT vLLM.** vLLM would force dropping Kitty's 2-bit quant + QUEST (its paged attention assumes uniform fp16/fp8 blocks; QUEST is query-aware sparse selection, not a storage format) and re-platforms the pinned compiled stack. Borrow only vLLM's *idea* (a bounded per-step token budget) — which is ~20 lines of Python for Kitty's B=1 case.
- **Tier 1 — chunk the MLP (ship this).** The MLP is **position-wise**, so chunking along the sequence is **bit-exact** with **zero** cache/kernel/QUEST/RoPE/mask changes. Drops the dominant transient. **Must be layout-aware** (Qwen/Llama `[B,S,H]` → dim 1; GLM `[S,B,H]` → dim 0) and **stacks on `KITTY_OFFLOAD=1`**.
- **Tier 2 — chunk the attention prefill (kernel-grade research, deferred).** Needs a new multi-query Triton kernel + incremental `quantize_prefill` + an explicit prefill/extend/decode phase flag. **Not optional forever:** with offload-on, Tier-1+sdpa fits 256k comfortably but the un-chunkable attention residual still scales, so Tier-1-only re-approaches 40 GB around **~450–550k** (offload-on; ~350–380k offload-off). Beyond that, Tier 2.
- **Key correction vs the first draft:** today's Qwen 128k peak is **~33.4 GiB** (no offload) / **~29.5 GiB** (offload), not "~27"; post-Tier-1 (offload-on) is **~20 GiB** @128k, **~26 GiB** @256k — fits, with real but not infinite headroom.

---

## 1. torch-native vs vLLM — verdict (lead)

**Decision: go torch-native. Do NOT adopt vLLM.** Borrow only its scheduling *idea* (fixed per-step token budget + carry remainder), which is a ~20-line Python loop for Kitty's single-request B=1 case.

1. **vLLM does not preserve Kitty's method — it forces a rewrite.** vLLM chunked prefill is a *scheduler* feature riding on FlashAttention/FlashInfer over a **uniform per-token paged block table**. Kitty's cache is structurally non-uniform: a 32-token fp16 sink buffer, fp16 K/V Q-buffers, an fp16 V local buffer, 2-bit packed pages with out-of-band per-page `KeyPage_Min/Max`, and a `D_BOOSTED` promoted-channel region. vLLM's `KVCacheSpec` assumes "block = N uniform token slots". And QUEST is *query-aware sparse page selection* (budget 2048 → 128 pages), not a storage format — hosting it means reimplementing Kitty's `qk_sparse_kernel`/`sv_sparse_kernel` + the selection scan inside a bespoke vLLM `AttentionBackend`. Net: "adopt vLLM" realistically means **dropping** 2-bit + QUEST.
2. **The env conflict is severe.** The pinned env is torch 2.4.1+cu121 / transformers 4.57.6 / triton 3.0.0, no vllm, no flash_attn (verified). Current vLLM bundles its own torch (~2.7–2.9+), needs CUDA ≥12.8, and re-platforms the compiled stack — forcing re-validation of every hand-written Triton kernel and **invalidating every reproduction number in CLAUDE.md** (GSM8K, LongBench, the 32k probe, the QUEST decode-speed table). A multi-week migration for a scheduling feature worth 20 lines.
3. **The win is cheap and torch-native.** The dominant transient is **position-wise and exactly chunkable** with zero cache/kernel changes. 80/20: nearly all the memory win, almost none of the risk, Kitty's 2-bit + QUEST decode stays bit-identical.

---

## 2. Two-tier design

### Tier 1 — chunk the MLP (position-wise, EXACT, model-agnostic) — SHIP THIS

**Mechanism.** Wrap every `*MLP` module's `forward`: when the input's **sequence** length exceeds `chunk_size`, split the input along the **sequence axis** into C-token slices, run the **original unmodified** `forward` per slice, and concatenate. When the sequence length ≤ `chunk_size` (decode S=1; short prefills), call the original forward verbatim — so **decode is untouched** (zero overhead, identical numerics).

**Why bit-exact:** `Qwen3MLP.forward` (`modeling_qwen3.py` `Qwen3MLP.forward`: `down(act(gate(x)) * up(x))`), stock `LlamaMLP`, and GLM's `MLP.forward` (`modeling_chatglm.py` `MLP`: `dense_h_to_4h` → swiglu `chunk` on the **last** dim → `dense_4h_to_h`) are all **position-wise**: output row *t* depends only on input row *t*. There is **no reduction across the sequence axis**, so per-slice compute + concat reproduces the identical computation. *(Caveat from review: cuBLAS may pick a different GEMM tile for `[…,C,…]` vs `[…,S,…]`, so accept "max|Δ| within a few ×1e-3 fp16 tolerance", not necessarily `torch.equal`. This is numerical-kernel noise, not a math difference.)*

**LAYOUT-AWARE (load-bearing fix).** The hidden-state layout differs by family:
- **Qwen3 / Llama**: batch-first `[B, S, H]` → sequence is **dim 1**.
- **GLM (ChatGLM-4 remote code)**: sequence-first `[S, B, H]` → sequence is **dim 0** (`modeling_chatglm.py` MLP comments `# [s, b, 4hp]`).

The two critics disagreed on GLM's actual runtime layout (comments say seq-first; one critic claimed it observed batch-first). **Resolution: the wrapper self-detects the sequence axis and fails loud**, and M1 must print the real shape on GPU0 to confirm. A naive dim-1 wrapper on GLM would silently **no-op** (slice the batch axis, B=1 ≤ C) — delivering zero memory benefit while passing a bit-exactness test. The wrapper must therefore (a) pick the axis whose length equals the prompt length / exceeds `chunk_size` and is not `hidden_size`, and (b) the M1 GLM gate must assert the chunk branch actually fired (counter > 0) **and** the peak dropped — not just that outputs match.

**What it touches:** nothing in the cache, kernels, QUEST, RoPE, mask, or offload. Attention prefill stays the existing dense `sdpa` over full S; `update`/`quantize_prefill`/the prefill-vs-decode dispatch and the single-token asserts are never exercised differently. **Invariant to preserve:** Tier 1 must keep exactly one full-S `update()` + `quantize_prefill()` per layer per prefill forward and must not change the S=1 decode path (the MLP wrapper is purely intra-forward, downstream of attention, and cannot reorder/duplicate cache calls).

**Covered paths:** real Qwen3/Llama kernel, GLM real kernel, GLM fake-quant, and the sim path — the MLP module is the same regardless of cache. **But** (review): the **sim path's** prefill peak is dominated by its growing **fp16 KV** (it stores full-sequence `key_states.clone()` and only evicts when `KITTY_OFFLOAD=1`), not the MLP — so Tier 1 yields little peak reduction on the sim path. The sim smoke validates **exactness**, not the memory win; the memory win is a **real-kernel + `KITTY_OFFLOAD=1`** property.

### Tier 2 — chunk the attention prefill (query-chunked) — kernel-grade, deferred

Goal: chunk *i*'s queries `[1,H,C,D]` attend to all prior keys `[0, off_i+C)`, causal within the chunk. What the real path needs and why each is hard:

1. **Defeat the 1-token wall.** `KittyCache.update` infers prefill via `Is_Prefill = (Sink_Count==0)`, and the second multi-token call trips `assert key_states.shape[-2]==1` (the decode branch). Tier 2 needs an **explicit phase flag** (`prefill`/`extend`/`decode`) the driver controls + a new **multi-token extend-append** path.
2. **Incrementalize `quantize_prefill`.** Today it is whole-sequence one-shot (derives sink/page/q-buffer boundaries from the global `len_prefill`, packs pages at absolute index 0, fills the q-buffer/local-buffer from the sequence **tail**). The **V local-buffer tail-windowing** (`Local_Buffer_V` from `value_states[:,:,-len_local_v:,:]`, V Q-buffer from `[-len_local_v-len_qbuf_v:-len_local_v]`) is the **highest-risk divergence**: a naive per-chunk append that doesn't reproduce it desyncs `PageCount_K`/`PageCount_V` and silently corrupts decode.
3. **A new multi-query kernel.** Every attention entry is single-query baked (host assert `t_query==1`, sparse paths bail on `t_query!=1`, grid `(B,H_KV)` with the query at a hardcoded `0*q_stride_t`, `attn_score`/`attn_output` with no query axis). Tier 2 needs a real query-block grid `(B,H_KV,ceil(C/BLOCK_Q))`, a `[BLOCK_Q,D]` tile, a `t_query=C` axis, **and an intra-chunk causal mask**. Keep it a **separate** kernel from decode — never weaken the decode assert.
4. **RoPE/mask per chunk: already free.** Positions/`cos`/`sin`/mask are derived per call from `cache_position = arange(past_seen, past_seen+len)` and `get_mask_sizes`. A per-chunk driver gets correct absolute RoPE + a "current chunk attends to full prefix" mask. Only the intra-chunk triangle is new (in the kernel). Qwen3's per-head `q_norm`/`k_norm` must run per chunk before RoPE.
5. **Accuracy semantics shift.** Today prefill attention is **exact fp16** dense; 2-bit only affects decode. If the extend path reads **quantized** prior pages it injects 2-bit error into prefill (validate). The **exact** variant keeps the prefix fp16 (defer `quantize_prefill` to the last chunk) but then does NOT reduce KV memory during prefill (offload already does that) — and re-pressures the offload budget (per-chunk H2D of prior pages).
6. **QUEST interaction.** QUEST is decode-time, single-query. Untouched by Tier 2 if prefill stays full causal. Sparsifying prefill would need a per-chunk **union** of selected pages across C queries — a new contract; do not do unless forced.

**Is Tier 2 needed?** With offload-on, **Tier 1 + sdpa fits 256k comfortably** (§3). The un-chunkable terms (attention residual + resident `KeyPage` metadata) scale linearly, so Tier-1-only re-approaches 40 GB near **~450–550k** (offload-on). **Tier 2 becomes genuinely necessary beyond that, OR offload-off, OR if sdpa's O(S²) compute time (not memory) gates the target.** Ship Tier 1 + force sdpa; treat Tier 2 as a separate kernel-grade project.

---

## 3. Memory math (B=1, 40 GiB; corrected against measured numbers)

Anchored on **measured** values (commit `abaeb26`, GPU0): weights Qwen 15.26 / GLM 17.70 GiB. The peak = weights + the largest single-layer transient stack during prefill. Terms that scale with S and are **NOT** removed by MLP chunking: the **attention residual** and the **resident packed KV** (only removed by `KITTY_OFFLOAD=1`) and the **resident `KeyPage_Min/Max`** QUEST metadata.

**Per-token activation widths (fp16, ×2 B/token):**
- MLP intermediate (chunkable): Qwen `3 × 12288` live ≈ 73728 elems/token; GLM `27392 + 13696` ≈ 41088. At 128k: Qwen ~9–10 GiB, GLM ~10 GiB. **→ chunked to `[…,C,…]`: ~0.3 GiB at C=4k.**
- Attention residual (NOT chunkable by Tier 1): on torch 2.4.1, sdpa GQA is **not** preserved (`use_gqa_in_sdpa` needs torch ≥2.5), so K,V are `repeat_kv`-expanded to full head count → Q + K + V + attn_output + the held fp16 `key_states`/`value_states` ≈ **~4 GiB at 128k**, ~8 GiB at 256k.
- Resident packed KV: ~3.94 GiB @128k (Qwen) — **on GPU if `KITTY_OFFLOAD=0`, on CPU if `=1`**. `KeyPage_Min/Max` ~1.15 GiB @128k stays GPU either way (QUEST).

**Predicted peaks (C=4k):**

| Scenario | Qwen3-8B | GLM-9B |
|---|---|---|
| **128k today, KITTY_OFFLOAD=1** (measured) | **29.47** | **36.70** |
| **128k today, offload=0** (measured) | **33.41** | OOM |
| **128k Tier 1 + offload=1** (predict) | **~20** | **~27** |
| **256k Tier 1 + offload=1** (predict) | **~26** (fits) | **~32** (fits, tight) |
| Tier-1-only ceiling on 40 GB (offload=1) | ~450–550k | ~350–420k |
| Tier-1-only ceiling (offload=0) | ~350–380k | lower |

**Takeaways:** Tier 1 drops the 128k peak by ~9–10 GiB (the MLP term), and with offload-on **256k fits** for both models — but with ~6–8 GiB headroom, not infinite. The "Tier-1 reaches 1M" claim from the draft is **false**: the attention residual + `KeyPage` re-approach 40 GB by ~0.5M; that's the real Tier-2 trigger.

---

## 4. Exact code-change locations (Tier 1 — the change to ship)

### 4.1 The layout-aware wrapper (new helper)

A single helper, e.g. in a new `src/kitty/chunked_prefill.py` (importable from both `kitty.models.*` and `kitty_sim`):

```python
import types, torch

def _seq_axis(x, hidden_size, chunk):
    # sequence axis = the dim that is not the hidden axis and exceeds `chunk`.
    # hidden axis = the dim equal to hidden_size (or its 2x/3x for fused projs is NOT the input).
    cands = [d for d in range(x.dim()) if x.shape[d] != hidden_size and x.shape[d] > chunk]
    if len(cands) != 1:
        raise RuntimeError(f"chunked MLP: ambiguous seq axis for shape {tuple(x.shape)} "
                           f"(hidden={hidden_size}, chunk={chunk}); refusing to guess.")
    return cands[0]

def _chunked_mlp_forward(orig_forward, hidden_size, C, counter):
    def forward(x, *args, **kwargs):
        # x is the MLP input [.., hidden]; decode (any axis == 1) and short prefills no-op.
        if max(x.shape[:-1]) <= C:
            return orig_forward(x, *args, **kwargs)
        ax = _seq_axis(x, hidden_size, C)
        counter[0] += 1
        outs = [orig_forward(t, *args, **kwargs) for t in torch.split(x, C, dim=ax)]
        return torch.cat(outs, dim=ax)
    return forward

def convert_mlps_to_chunked(model, chunk_size=4096):
    targets = ("Qwen3MLP", "LlamaMLP", "MLP")  # "MLP" == GLM remote-code class
    hidden = model.config.hidden_size
    counter = [0]; n = 0
    for m in model.modules():
        if type(m).__name__ in targets and not getattr(m, "_kitty_mlp_chunked", False):
            m.forward = types.MethodType(_chunked_mlp_forward(m.forward, hidden, chunk_size, counter), m)
            m._kitty_mlp_chunked = True; n += 1
    if n == 0:
        raise RuntimeError("convert_mlps_to_chunked: no MLP modules found (model layout drift?)")
    model._kitty_mlp_chunk_counter = counter   # M1 asserts counter[0] > 0 after a long prefill
    return n
```

Notes folded from review: gate on `max(x.shape[:-1])` so it is layout-agnostic (handles `[B,S,H]` and `[S,B,H]`); `_seq_axis` **fails loud** instead of silently slicing the batch dim (the GLM no-op trap); a `counter` so M1 can prove the chunk branch fired. Optional perf variant: pre-allocate `out = torch.empty_like(x)` and write slices to avoid the `cat` copy — but verify it does not pin references that defeat offload eviction (§6).

### 4.2 Install sites (opt-in flag + knob)

- **Real Qwen3/Llama:** call `convert_mlps_to_chunked(model, C)` right after the attention conversion in the `*_Kitty` setup (Llama: alongside `convert_llama_attention_to_kitty`; Qwen3: its analogue).
- **GLM:** install inside `glm_kitty_patch.py` next to the existing `SelfAttention`/`GLMTransformer` patches, with a class-level `_kitty_mlp_chunked_installed` flag; **raise** if zero `MLP` modules found.
- **Driver (covers all paths):** in `src/kitty_sim/longbench/runner.py`, right after `load_model_and_tokenizer`, gated on env (mirrors the existing `KITTY_OFFLOAD` pattern):
  ```python
  _cz = int(os.environ.get("KITTY_PREFILL_CHUNK", "0"))
  if _cz > 0:
      from kitty.chunked_prefill import convert_mlps_to_chunked
      print(f"[chunk] MLP chunked prefill: {convert_mlps_to_chunked(model_obj, _cz)} modules, C={_cz}")
  ```
  Default OFF; `KITTY_PREFILL_CHUNK=4096` turns it on. `model.generate(...)` is unchanged — the chunk lives inside the single prefill forward.

### 4.3 Probe wiring (review fix — the benchmark ignores the env)

`latency_benchmarking/benchmark_kitty.py` does **not** read `KITTY_PREFILL_CHUNK` and is **Qwen3-only**, so the draft's probe commands would silently no-op. Instead **extend `tools/probe_real_kernel.py`** (already handles the real kernel + `--offload`) with a `--chunk N` flag that calls `convert_mlps_to_chunked(model, N)` after load and prints the chunk count; add a `--glm` path (load GLM + `install_glm_real_kitty_kernel` + the MLP chunk) for the GLM probe. Every probe must print the chunk counter so it self-verifies the hook engaged.

### 4.4 Force sdpa (companion, required)

Tier 1's peak is only reached if attention uses **sdpa** (eager materializes `[1,H,S,S]` → fatal at long S). Real-kernel Qwen/Llama already load `attn_implementation="sdpa"`. **GLM uses its own `CoreAttention`/`SdpaAttention`/`FlashAttention2` selected by `config._attn_implementation`** — set `config._attn_implementation="sdpa"` before the patch and **probe GLM's prefill mask separately** (GLM builds a `full_attention_mask` via `get_masks`; don't assume Qwen's "sdpa mask=None" transfers).

---

## 5. Correctness

- **Tier 1 exactness:** MLP body has no cross-token op (`gate/up/down` are per-token linear, `act_fn`/`*` element-wise, GLM swiglu `chunk` on the last dim). Output row *t* depends only on input row *t* → slice/compute/concat is identical math. Accept fp16 cuBLAS-tile tolerance, not necessarily `torch.equal`.
- **Decode untouched:** guaranteed because prefill/decode dispatch keys off `update()`'s return (prefill = first per-layer call, `Sink_Count==0`), and the wrapper is intra-forward, downstream of attention; it must not change the number/shape of `update()`/`quantize_prefill()` calls. The `max(x.shape[:-1]) <= C` guard makes S=1 decode a verbatim passthrough.
- **Stacks on offload:** the memory win needs `KITTY_OFFLOAD=1` (else the resident packed KV is a second large term). The two are orthogonal and compose.
- **Tier 2 (if pursued):** the incremental quantize must produce a **byte-identical final packed cache** to one-shot `quantize_prefill` (same sink=first 32, same page boundaries, same V local-buffer tail-window) — gate behind a per-tensor unit check (`Sink_Buffer_K/V`, `KeyCache`/`ValueCache` pages, `KeyPage_Min/Max`, `Q_Buffer_*`, `Local_Buffer_V`, all counts). Sim-path Tier 2 must route through the sim **prefill block-quantize**, not the decode trailing-window (the sim decode re-quantizes only when `count % buffer_length == 1`).

---

## 6. Risks & failure modes

1. **GLM silent no-op (layout):** dim-1 slice on `[S,B,H]` no-ops. *Mitigation:* layout-aware `_seq_axis` that fails loud; M1 GLM gate asserts `counter>0` AND a peak drop (not just output match). **Empirically print GLM's MLP-input shape on GPU0 in M1.**
2. **Probe runs with hook inactive:** the benchmark ignores the env. *Mitigation:* use the extended `probe_real_kernel.py`; print the chunk counter in every probe.
3. **Memory win absent without offload:** Tier 1 alone leaves resident packed KV. *Mitigation:* set `KITTY_OFFLOAD=1` in every memory probe; document the dependency.
4. **cuBLAS tile mismatch:** `[…,C,…]` vs `[…,S,…]` GEMM may differ at the ULP level. *Mitigation:* tolerance-based gate; pre-allocate `out` to remove `cat` noise.
5. **Throughput cost:** at 256k, C=4k = ~64 sequential MLP calls/layer. *Mitigation:* a throughput probe (prefill tok/s for C∈{2k,4k,8k,16k}) picks the default; C=4k is the starting point (`[1,4k,12288]` fp16 ≈ 96 MiB).
6. **`cat` / pre-alloc holding references that defeat offload eviction:** verify the chosen path doesn't pin evicted layers (peak check with offload on).
7. **GLM remote-code drift:** the `MLP` class name is generic; a `trust_remote_code` update could rename it → silent no-op. *Mitigation:* raise on zero matches; re-entrancy flag; consider checksumming `modeling_chatglm.py`.
8. **Tier-2 V local-buffer desync / 1-token kernel assert / quantized-prefix accuracy** — see §2 Tier 2; all gated behind the byte-identical-packed-cache unit check.

---

## 7. Phased rollout (each = a GPU0 command + pass/fail; all under the kitty env, `KITTY_OFFLOAD=1`)

**Phase 0 — baseline.** Real-kernel Qwen 128k peak via the extended probe (no chunk).
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python tools/probe_real_kernel.py \
  --model /mnt/data/tzj/models/Qwen3-8B --family qwen --context-len 131072 --max-new-tokens 2 --offload
```
Expect peak_alloc ~29.5 GiB (offload on) — the documented baseline.

**Phase 1 — Tier 1 MLP chunk (SHIP). Pass = bit-exact (within fp16 tol) + peak drop + chunk-counter>0, per model family.**
- **1a exactness (gate):** short prompt just above C (e.g. C=512, S=2048), prefill logits chunk vs no-chunk → `max|Δ|` within a few ×1e-3. PASS only if within tolerance.
- **1b peak @128k:** add `--chunk 4096` to the Phase-0 command. PASS: peak ~29.5 → **~20 GiB** AND printed chunk counter > 0.
- **1c 256k fits:** `--context-len 262144 --chunk 4096 --offload`. PASS: completes < 40 GiB (predict ~26 GiB).
- **1d accuracy regression-free (end-to-end):** `KITTY_OFFLOAD=1 KITTY_PREFILL_CHUNK=4096 DATASETS_CSV=trec bash scripts/run_exp.sh qwen --gpu 0 --max-samples 2 --variant quest_kitty_page16_kernel` → predictions identical to the `KITTY_PREFILL_CHUNK=0` run. (Exactness gate; this path's peak is not the MLP-wall test — see §2.)
- **1e GLM parity (the layout gate):** GLM real-kernel probe with `--chunk 4096`; PASS requires **(i)** chunk counter > 0 (proves the layout-aware axis fired on `[S,B,H]`), **(ii)** GLM 128k peak drops ~10 GiB, **(iii)** 256k fits, **(iv)** outputs match within tol. Print the GLM MLP-input shape.
- **1f throughput:** prefill tok/s at 128k for C∈{2k,4k,8k,16k} vs baseline → pick/confirm the default C.

**Phase 2 — force sdpa everywhere.** Confirm no path (esp. GLM) falls back to eager at long S. Metric: Phase-1 peaks hold; attention residual stays ~4 GiB @128k.

**Phase 3 — Tier 2 (only if Phase 1+2 insufficient for the target).** Trigger = target context > Tier-1 ceiling (~450–550k offload-on) OR sdpa O(S²) compute time gates it. Build the multi-query prefill kernel + incremental quantize per §2/§4 sketch, gated by the byte-identical-packed-cache unit check and (if reading quantized prefix) an accuracy re-validation.

---

## 8. Open questions for the user

1. **Target context:** is 256k the requirement (Tier 1 reaches it) or is 0.5M–1M the goal (needs Tier 2)?
2. **Memory-only or also decode/prefill compute-time?** Tier 1 fixes memory exactly; Tier 2's extra value is bounded per-step query activation / sparsified prefill compute.
3. **Default `KITTY_PREFILL_CHUNK`:** ship opt-in (default 4096 when enabled), like `KITTY_OFFLOAD`?
4. **Apply Tier 1 to all four paths now,** or real-kernel Qwen3/Llama/GLM first and sim later (sim's peak is KV-dominated, so it benefits less)?
5. **GLM remote-code pinning:** OK to checksum `/mnt/data/tzj/models/GLM-4-9B-Chat-1M/modeling_chatglm.py` so the `MLP` scan can't silently drift?
6. **Tier-2 accuracy budget (if pursued):** acceptable to inject 2-bit into prefill attention (quantized-prefix), or must Tier 2 stay exact (fp16 prefix, no prefill-KV memory saving)?

---

## Appendix — Corrections folded in from adversarial review

| # | First-draft claim | Correction |
|---|---|---|
| 1 | MLP wrapper slices dim=1 for all models | **GLM is `[S,B,H]` (dim 0)** — dim-1 slice silently no-ops on GLM. Wrapper made **layout-aware + fail-loud**; M1 GLM gate asserts chunk-counter>0 + peak drop. |
| 2 | "KV already offloaded" as ambient fact | Offload is the **committed `KITTY_OFFLOAD` (abaeb26), OFF by default** — Tier 1's memory win **requires `KITTY_OFFLOAD=1`**; probes set it; §3 states the dependency. |
| 3 | Today peak ~27 GiB; post-Tier-1 ~18; reaches ~1M | Measured **~33.4 (offload off) / ~29.5 (on)**; post-Tier-1 **~20 @128k, ~26 @256k**; Tier-1 ceiling **~450–550k** (attention residual + KeyPage scale). |
| 4 | sdpa attention transient ~2 GiB | torch 2.4.1 sdpa **always `repeat_kv`** (GQA needs ≥2.5) → ~**4 GiB @128k**, scales linearly, NOT chunkable by Tier 1. |
| 5 | Probes via `benchmark_kitty.py --max_seq_len … KITTY_PREFILL_CHUNK=…` | That script **ignores the env** and is **Qwen-only** → would silently no-op. Use the extended `tools/probe_real_kernel.py --chunk N` (+ `--glm`), print chunk counter. |
| 6 | Tier-1 bit-exactness = `torch.equal` | cuBLAS tile may differ → **fp16 tolerance**; pre-alloc `out` to remove `cat` noise. |
| 7 | Tier 1 validated on sim LongBench (1d) for memory | Sim prefill peak is **fp16-KV-dominated**, not MLP — 1d is an **exactness** gate; the memory win is a real-kernel + offload property. |
| 8 | Line-number citations | Numbers shifted after the offload edits; **anchored to symbol names** (`KittyCache.update`/`Is_Prefill`/`quantize_prefill`) instead. |
| 9 | Tier 2 "optional" | Tier 2 is **genuinely needed beyond ~450–550k** (offload-on) or offload-off ~350k — reframed as a memory ceiling, not just compute. |
