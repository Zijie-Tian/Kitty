# Layer-wise CPU KV-Cache Offload — Deep Implementation Plan

> Branch: `tzj/kvcache-offload` · Worktree: `/mnt/data/tzj/Code/Kitty/.claude/worktrees/kvcache-cpu-offload`
> Status: **PLAN ONLY — no code written yet.** Awaiting user review before implementation.
> Verification HW: **GPU0** (`CUDA_VISIBLE_DEVICES=0`) — explicit user override of the repo's GPU1-only rule.
> Env `kitty`: torch 2.4.1+cu121, transformers 4.57.6, **tiktoken 0.13.0 present** (GLM tokenizer is NOT blocked), no flash_attn. A100-PCIE-40GB.

This plan was produced by a fan-out workflow (4 code-readers → architect → 3 adversarial critics) and then hand-reconciled against the actual files. The critic corrections are folded in; a "Corrections from review" appendix records what changed from the first draft so the rigor is auditable.

---

## 0. TL;DR

- **What:** add an opt-in `offloading=True` mode to Kitty's KV-cache classes so each layer's KV lives in **pinned host RAM** and is **double-buffered** onto GPU0 one layer ahead of compute. Resident GPU KV drops from "all layers" to a small **working window** (configurable `resident_layers`).
- **Why:** the sim fp16 path holds the full KV on GPU (GLM-9B ~10 GiB @128k; Qwen3-8B ~18 GiB @128k) and OOMs on 40 GB at long context. Offloading the cumulative KV is exactly what lets GLM-9B @128k fit (~38 GiB → **~29 GiB**).
- **Two targets, two payoffs:**
  - **(A) sim fp16 fake-quant path** — offload **buys memory** (the OOM fix). Implement first.
  - **(B) real Qwen3 Triton QUEST path** — 2-bit KV is already small, so the payoff is **fetch-on-select**: keep all packed pages on CPU, fetch only the ~128 QUEST-selected pages per step. Bandwidth-elegant; enables 256k+.
- **The honest cost (sim path ≥128k):** naive full-fp16 offload makes decode **PCIe-bandwidth-bound — roughly 50–77× slower than baseline decode** (~0.86 s/token vs sub-20 ms at 128k). Double-buffering does NOT hide it when copy/layer ≫ compute/layer. **Therefore `resident_layers` partial offload is a v1 requirement, not an optional mitigation, for the sim path at ≥128k.** Frame the sim win as *feasibility* (run-at-all), not speed.
- **Implementation order:** sim Qwen3/Llama (clean, correctness-trivial) → GLM (the headline memory target, but topology-tricky). **The real Triton path (fetch-on-select) is DEFERRED per user decision (2026-06-02): v1 is sim-only.** §5 is retained as the future-work design.

> **Locked decisions (user, 2026-06-02):** (1) **v1 = sim path only** — defer the real Qwen3 Triton M3/M4. (2) **Latency is handled by prefetch/double-buffering** (overlap next-layer H2D with current-layer compute); `resident_layers` is the documented backstop for the ≥128k case where prefetch cannot fully hide the copy. (3) **Host memory = a process-level pinned pool + a `MemAvailable` guard** (preallocate to `max_length`, reuse across samples, abort if the guard trips).

---

## 1. Goal & scope

**Goal.** A per-layer offload-after-use + prefetch-before-use double-buffering layer that keeps only `resident_layers` (default small, e.g. 2–4) layers of KV on the GPU at once, with the rest in pinned host memory, transferred asynchronously on a dedicated CUDA copy stream and ordered by `torch.cuda.Event`.

**Implement FIRST: the sim fp16 path, starting with Qwen3/Llama (not GLM).**
Rationale — the *correctness-trivial* case validates the machinery before the topology-tricky one:

1. Qwen3/Llama sim route KV through the shared HF `Cache.update()` contract (`KittyKVCache.update`, `kitty_simulate.py:231`), called once per layer in strict increasing-`layer_idx` order — a clean place to hang the prefetch/evict pump.
2. With the forced `PostQuant=True` default (`get_kvcache_kitty`, `kitty_simulate.py:334`), `update()` returns **independent clones** (`kitty_simulate.py:253-254` prefill, `:289-290` decode), so the stored slot is safe to evict the instant `update()` returns — no use-after-evict race.
3. GLM is the *motivating memory target* (the actual 128k OOM) but is **NOT correctness-trivial**: each `SelfAttention` owns a private single-layer `KittyKVCache` (always `layer_idx=0`), returns **stored aliases** (`glm_kitty_patch.py:188`), and the vendored attention re-reads the previous step's KV *before* our update runs. It needs a dedicated offload manager and careful hook ordering (§4.2). Deliver it in M2, right after the machinery is proven.

**Two value propositions (kept distinct throughout):**

| Path | KV storage | Offload payoff | Validation route |
|---|---|---|---|
| (A) sim fp16 (`KittyKVCache`, GLM/Llama/Qwen) | fp16, full | **memory** — fixes 128k/256k OOM | `scripts/run_exp.sh` LongBench + peak-mem probe |
| (B) real Qwen3 Triton (`KittyCache`/`KVCache_Layer`) | packed 2-bit/4-bit | **fetch-on-select** bandwidth + 256k+ scaling | `latency_benchmarking/benchmark_kitty.py` only |

**Important routing fact (completeness critic):** the **real `KittyCache` is NOT wired into the LongBench runner** — `runner.py` imports the *sim* cache, and `kitty_page16` is a fake-quant proxy, not the real kernel. The real kernel is **Qwen3-only** (no GLM fork). So **real-path milestones M3/M4 are `benchmark_kitty.py`-only and Qwen3-only**; there is no "real GLM" path.

**Out of scope for v1:** beam search (`num_beams=1`, `runner.py:321`); the `PostQuant=False` sim variant (assert against it when `offloading=True`); `VCache_BitDecoding=True` (forced off); the dense (non-QUEST) real decode path (per-step whole-cache offload not viable there); activation checkpointing for the MLP transient (see §6 the 256k MLP wall); multi-batch B>1 (B=1 scope).

---

## 2. Architecture: offload-after-use + prefetch-before-use double-buffering

Reference semantics: transformers 4.57.6 `OffloadedCache` / `Cache.update` pump (`cache_utils.py:771-781`). We **deliberately diverge in two places the stock code gets wrong**:

1. Stock evicts with `.to("cpu", non_blocking=True)` onto **pageable** host memory (`cache_utils.py:56-60`), which **silently synchronizes** (no real overlap). We use **pre-pinned host mirrors** and `dst.copy_(src, non_blocking=True)`.
2. Stock orders with a coarse `default_stream.wait_stream(prefetch_stream)` (`cache_utils.py:773`). We use **per-layer `torch.cuda.Event`** so a copy for layer L cannot stall an unrelated kernel — required for the real sparse path where the kernel only needs its ~128 fetched pages.

API gate (torch 2.4.1): use `torch.cuda.Stream()` / `torch.cuda.stream(...)`, **not** `torch.Stream` (the latter is a 2.5+ API).

### 2.1 Core invariant

While the compute stream runs layer **L**:

- Layer **L** KV is GPU-resident (guaranteed by `compute_stream.wait_event(_h2d_done[L])`).
- Layer **L+1** H2D copy is in flight on the copy stream (issued during L−1/L).
- Layer **L−1** D2H copy may still be draining on the copy stream.
- All other layers are CPU-resident (pinned).

**Working-window invariant:** at most `resident_layers + 1` layers GPU-resident at once. With `resident_layers=2`, steady state holds `{L, L+1}`; transiently `{L−1, L, L+1}` during evict drain. **`resident_layers` is a first-class knob** (not just 2): at ≥128k on the sim path it is the latency/​memory dial (§7).

### 2.2 State (lives on the cache object, or on a manager for GLM — see §4.2)

```
_offloading: bool
_resident_layers: int                     # working-window size (latency/memory dial)
_copy_stream: torch.cuda.Stream()         # torch 2.4.1 API
_compute_device: torch.device             # captured from first key_states.device
_host_K, _host_V: list[Tensor|None]       # PINNED host mirror per layer, allocated ONCE, reused
_resident: set[int]                       # layer indices currently on GPU
_cur_len: list[int]                        # live token count per layer (sim cat-growth)
_h2d_done:  list[torch.cuda.Event]        # "layer L H2D complete"
_d2h_done:  list[torch.cuda.Event]        # "layer L D2H complete -> host buf reusable"
_compute_consumed: list[torch.cuda.Event] # "compute finished reading layer L" (gates D2H)
```

All of this is allocated **at cache construction, outside `torch.inference_mode()`** (the runner wraps generation in `inference_mode`, `runner.py:317`; the benchmark too, `benchmark_kitty.py:270`). Pinned `torch.empty(..., pin_memory=True)`, `torch.cuda.Stream()`, and `torch.cuda.Event()` must be created before entering inference mode; the hot-path `copy_` / `record` / `wait_event` are data ops and are inference-mode-safe.

### 2.3 The two primitives

**`_prefetch_layer(L)` — H2D, copy stream, before-use:**
```
if L not in _resident and _host_K[L] is not None:
    with torch.cuda.stream(_copy_stream):
        _copy_stream.wait_event(_d2h_done[L])         # don't race a still-draining evict of this slot
        gpuK = empty_like_on(_compute_device, _cur_len[L]); gpuK.copy_(_host_K[L][:_cur_len[L]], non_blocking=True)
        gpuV = ... same ...
        key_cache[L], value_cache[L] = gpuK, gpuV     # rebind slot to GPU tensor
        _h2d_done[L].record(_copy_stream)
        _resident.add(L)
```

**`_evict_layer(L)` — D2H, copy stream, after-use:**
```
if L in _resident:
    with torch.cuda.stream(_copy_stream):
        _copy_stream.wait_event(_compute_consumed[L])  # only after compute READ it
        _host_K[L][:_cur_len[L]].copy_(key_cache[L], non_blocking=True)  # INTO pinned dst
        _host_V[L][:_cur_len[L]].copy_(value_cache[L], non_blocking=True)
        _d2h_done[L].record(_copy_stream)
    key_cache[L], value_cache[L] = _host_K[L], _host_V[L]   # rebind slot to host (drops GPU ref)
    _resident.discard(L)
```

**Two correctness rules (from the correctness critic):**

1. **Never `.to("cpu")` on the hot path** — always `pinned_dst.copy_(src, non_blocking=True)`. `.to("cpu")` allocates pageable memory and synchronizes.
2. **Keep the GPU source tensor referenced until `_d2h_done[L]` is observed.** After eviction the slot is rebound to host immediately on the Python thread, but the byte copy drains async; under the CUDA caching allocator a later same-size alloc could reuse the evicting block if ordering is wrong. The next prefetch of slot L already `wait_event(_d2h_done[L])`; additionally hold the old GPU tensor in a small `_inflight[L]` ref cleared when L is next prefetched. A CUDA sanitizer run (M1 probe) must confirm no mid-copy recycle. *Simplicity fallback:* if events prove fragile, evict on the **default stream** like stock (`cache_utils.py:733-740`) — per-layer decode compute is short, so the lost overlap is small and a whole class of races disappears.

### 2.4 Pinned host buffers & sizing

- **Allocate once, reuse** (`_ensure_host_mirror` at construction, not per evict). The sim runner builds a **fresh `KittyKVCache` per sample** (`runner.py` for-sample loop) and does **not** call `reset()` for non-GLM models — so allocating pinned buffers per evict/per-sample would churn hundreds of large non-pageable allocations and risk host-RAM exhaustion. **Decision: a process-lifetime pinned buffer pool keyed by (layer, shape), reused via `copy_` across samples; freed only at teardown.**
- **Sim decode grows via `torch.cat`** (`kitty_simulate.py:282-283`): the GPU tensor for the active layer grows +1 token/step. Over-allocate each pinned mirror **once** to `[B, H_KV, max_length, head_dim]` and `copy_` only the live `[:, :, :_cur_len[L], :]` slice; track `_cur_len[L]`. `max_length = MAX_MODEL_LEN + MAX_GEN`, known at construction. **Hard `assert _cur_len[L] <= max_length`** with a documented failure if generation exceeds the ceiling (Open Q #3). **Note:** the *active* (resident) layer still carries the full cat-grown fp16 tensor + the PostQuant clone — the memory win is on the **non-resident** layers only.
- **Real path** (`KVCache_Layer`): tensors are static-shaped `(MAX_BS*MAX_PAGE, bytes_per_page)` — allocate the pinned mirror once with the same shape, `copy_` in/out, never reallocate. Natural fit.

### 2.5 Why per-layer events, not `wait_stream`

`compute_stream.wait_event(_h2d_done[L])` asserts exactly "don't launch L's attention until L's H2D is done", without blocking on the unrelated L+1 prefetch that shares the copy stream. The real QUEST sparse kernel needs only its ~128 fetched pages and must not block on a full-arena copy — events give that precision; `wait_stream` would over-serialize.

---

## 3. Prefill vs decode schedules + hook points

Both phases run the same Qwen3 layer loop (`src/kitty/models/qwen3/modeling_qwen3.py:482-498`) and the GLM patched loop (`glm_kitty_patch.py:74-111`). Hooks attach at the **layer-loop level**, not inside `generate()` (which only exposes token boundaries — wrong granularity).

### 3.1 Prefill (one forward over the full prompt)

KV is *born* here; **no H2D needed** (sim prefill constructs from incoming `key_states`, never reads a prior stored slot). Only **D2H after each layer** to keep peak bounded.

```
for L in 0..num_layers-1:
    decoder_layer(L):
        update(k,v,L)            # sim: append clone (:246-247) + quantize settled region
                                 # real: pack (quantize_prefill :247) then free fp16 staging (kitty.py:252-253)
        attention(...)           # sim: on returned clone; real: prefill eager attn on key/value_states
    hidden = layer_outputs[0]    # modeling_qwen3.py:498
    record _compute_consumed[L]; _evict_layer(L)     # D2H this layer's just-born KV
# Prefill peak = weights + (resident window KV) + current-layer transients (MLP, MQA-expand)
```

### 3.2 Decode (one forward per new token)

Each layer **must be resident before its read**, then evicted after.

```
for L in 0..num_layers-1:
    # BEFORE decoder_layer(L):
    compute_stream.wait_event(_h2d_done[L])   # ensure L resident (H2D issued during L-1)
    _prefetch_layer(L+1)                       # wrap L+1 -> 0 on last layer
    decoder_layer(L):
        # SIM decode (kitty_simulate.py:280-310): torch.cat onto resident slot -> returns clone -> attn safe
        # REAL decode: update() writes new token to resident Sink/Q/Local buffers (kitty.py:148-184);
        #              THEN kitty_attention_forward reads packed kv_cache[L] (modeling_qwen3.py:249-254);
        #              THEN quantize_decode writes a new packed page (modeling_qwen3.py:255)
    hidden = layer_outputs[0]                  # modeling_qwen3.py:498
    # AFTER (real path: this MUST be after :255, see below):
    _compute_consumed[L].record(compute_stream)
    _evict_layer(L)
```

### 3.3 Where the hooks attach (corrected file:line)

| Hook | Sim Qwen3/Llama | Sim GLM | Real Qwen3 |
|---|---|---|---|
| Ensure-L-resident + prefetch L+1 | top of `KittyKVCache.update()` `kitty_simulate.py:231` | GLM manager, **before** `layer(...)` `glm_kitty_patch.py:97` (prefetch `index`, overlap `index+1`) | top of `KittyCache.update()` `kitty.py:123` (prefetch only) |
| Evict-L (D2H after use) | end of `update()` (safe: PostQuant clones) `kitty_simulate.py:~312` | GLM manager, **after** `presents += (kv_cache,)` `glm_kitty_patch.py:105` | **modeling loop seam after quantize_decode** `modeling_qwen3.py:498`, keyed on `decoder_layer.self_attn.layer_idx` — **NOT inside `update()`** |

**The single load-bearing real-path rule (correctness critic):** in the real **decode** path the packed `kv_cache[L]` is *read* at `modeling_qwen3.py:249-254` and *written* at `:255`, both **after** `update()` (`:229`) returns. Evicting inside `update()` would race the kernel. So: prefetch/ensure-resident from `update()`-top is fine; **evict only at `modeling_qwen3.py:498`, gated by a `_compute_consumed[L]` event recorded after `:255`.**

For the sim Qwen3/Llama case the read happens *inside* `update()` (the `torch.cat` + clone), so the entire pump can live in `update()` with **zero modeling-file edits** — the cleanest seam. The modeling-loop hook is only needed for the real path and GLM uses its own manager.

---

## 4. Exact code-change locations (edit list)

> All paths relative to the worktree root. The real Qwen3 modeling file is **`src/kitty/models/qwen3/modeling_qwen3.py`** (the first draft mistakenly wrote `src/transformers/...`).

### 4.1 Sim path — `src/kitty_sim/kitty_simulate.py` (`KittyKVCache`, class `:152`)

1. **`KittyKVCacheConfig`** — add `offloading: bool = False`, `resident_layers: int = 2`, `max_length: int | None = None`. (Threading via the config object is required — see 4.3; `get_kvcache_kitty` takes an `argparse.Namespace`, not kwargs.)
2. **`__init__` (`:164-165`)** — when `offloading`, init the §2.2 state (streams/events/pinned pool) **outside inference_mode**.
3. **`update()` (`:231`)** — at top, before the prefill/decode branch (`:241`): capture `_compute_device`; if `offloading`: `torch.cuda.current_stream().wait_event(_h2d_done[layer_idx])` (no-op for prefill) then `_prefetch_layer(layer_idx + 1)` (respect `resident_layers`).
4. **Decode branch (`:280-283`)** — `torch.cat` now safe (slot guaranteed resident by step 3). Add a debug assert `key_cache[layer_idx].device == _compute_device` and a warn-once `.to(device)` fallback for a missed prefetch.
5. **Before the shared return (`:~312`, the `if self.PostQuant:` line)** — if `offloading`: record `_compute_consumed[layer_idx]`, then `_evict_layer(layer_idx)`. **Assert `PostQuant` when `offloading`** (the `PostQuant=False` branch returns stored aliases at `:315` — eviction would corrupt the in-flight attention read).
6. **New methods** `_ensure_host_mirror`, `_prefetch_layer`, `_evict_layer`, `_free_pinned` (§2.3/2.4).
7. **`reset()` (`:225-228`)** — keep the pinned pool, just zero `_cur_len`/`_resident` and re-record events; do **not** free+realloc pinned buffers per sample.
8. **`reorder_cache`/`batch_*` (`:197-223`)** — out of v1 scope (B=1); add a `_page_in_all()` guard that materializes all layers then re-evicts if these are ever called.

### 4.2 GLM — `src/kitty_sim/glm_kitty_patch.py` (the topology-tricky case)

**Topology fact (verified):** each `SelfAttention` owns a private single-layer `KittyKVCache` (`self._kitty_cache`, `:172-175`), always `kc.update(new_key,new_value,0)` (`:182`), returning the **stored aliases** `(kc.key_cache[0], kc.value_cache[0])` at `:188`. The cross-layer loop is `_patched_glmtransformer_forward` (`:74-111`); `presents[index]` (`:105`) is fed back next step as `kv_caches[index]` (`:97-99`) and read **inside the vendored `orig_forward`** (called first at `:163`) *before* our `kc.update` runs.

Consequences: the §2.2 per-layer-list primitives **cannot** live on a single-layer cache. Instead:

1. **New `GLMOffloadManager`** owning per-module-index host buffers + events + resident set, attached to the `GLMTransformer` instance (which has the cross-module loop). `KittyKVCache` stays a pure single-layer storage helper for GLM (no self-prefetch).
2. **Prefetch hook — BEFORE `layer(...)` at `:97`** (memory critic): H2D module `index`'s `_kitty_cache.key_cache[0]/value_cache[0]` so the vendored cat inside `orig_forward` (`:163`) sees a GPU tensor; overlap by also issuing module `index+1`'s prefetch. This fixes the host-vs-GPU `torch.cat` crash that would otherwise occur (the previous step evicted `presents[index]`).
3. **Evict hook — AFTER `presents = presents + (kv_cache,)` at `:105`**: record `_compute_consumed[index]`, D2H module `index`'s `_kitty_cache.key_cache[0]`. Eviction here is safe even with alias-return because both attention and the residual/MLP have consumed the tuple by `:105`. **Never evict before `:188`.**
4. **The vendored `modeling_chatglm.py` cat citation is version-dependent** (loaded via `trust_remote_code`, not in the repo). M2 must **dump the actual `SelfAttention.forward` source** from the loaded model and confirm the `kv_caches[index]` re-read before finalizing the page-in-before-cat guard. Add a device-mismatch assertion at the cat.

### 4.3 Runner — `src/kitty_sim/longbench/runner.py`

1. **`_cache_factory`** — read an opt-in flag (`KITTY_OFFLOAD=1` env and/or `VariantConfig.offloading`); set `offloading`, `resident_layers`, and `max_length = MAX_MODEL_LEN + MAX_GEN` on the `SimpleNamespace` it passes to the factory.
2. **`get_kvcache_kitty` (`kitty_simulate.py:317-337`)** — it takes a single `argparse.Namespace`; read `args.offloading / args.resident_layers / args.max_length` off it and thread them into `KittyKVCacheConfig` / `KittyKVCache`. (Do **not** assume kwargs.)
3. **No change** to the `model.generate(...)` call — the mechanism is internal to the cache/loop.

### 4.4 Real Qwen3 path — `src/kitty/kvcache/`

1. **`utils_kv_per_layer.py` `KVCache_Layer.__init__` (`:48-95`)** — the single site where tensors are hard-coded `device='cuda'`. Add `device`/`offloading`. Keep **GPU-resident**: `KeyPage_Min/Max` (`:55-56`, QUEST selector needs them), `PageTable_K/V` (`:64-67`), `Sink_Buffer_K/V` (`:77-80`), `Q_Buffer_K/V` (`:83-86`), `Local_Buffer_V` (`:90-94`). Make the **offload payload**: `KeyCache`/`KeyCache_metadata` (`:48-51`) and `ValueCache`/`ValueCache_metadata` (`:59-62`). Add `offload()`/`prefetch()` (pinned + event-recorded).
2. **`kitty.py` `KittyCache.__init__` (`:56`, layer build `:107-121`)** — add `offloading`; construct copy stream + event lists; pass through.
3. **`kitty.py` `update()` (`:123`)** — at top: prefetch L+1 (and ensure L resident). **Do NOT evict here.**
4. **`modeling_qwen3.py` loop `:482-498`** — enumerate layers for `layer_idx`; record `_compute_consumed[L]` and `_evict_layer(L)` at `:498`, i.e. **after** `kitty_attention_forward` (`:249`) and `quantize_decode` (`:255`).
5. **`latency_benchmarking/benchmark_kitty.py` `_new_kitty_cache` (`:161-184`)** — add `--offload` (+ `--resident-layers`), thread to `get_kvcache_kitty` (`kitty.py:332`). The manual decode loop (`:291-300`) is the cleanest validation harness.
6. **`reset`/`reorder` on the real cache** — add a `reset()` that re-pins/re-records and a `_page_in_all()` guard (the real cache currently lacks these; needed for multi-sample reuse and any reorder).

---

## 5. QUEST synergy for the real path (fetch-on-select)

This is target (B)'s payoff and the architecturally elegant case.

### 5.1 Why whole-cache offload is marginal but fetch-on-select is compelling

Packed KV **+ metadata** @128k is ~3.94 GiB total (~112 MiB/layer — see §6.3; the first draft's 2.67 GiB omitted the two metadata tensors). Moving it whole-cache adds **~169 ms/decode pass** one-way (§7.2) for little memory gain. But QUEST budget 2048 / page16 selects only **128 of ~8192 logical pages (~1.6%)**. So keep all packed pages CPU-resident and fetch only the selected pages: ~1.78 MB/layer/step, ~64 MB/pass → **~2.56 ms/pass, fully hidden** behind the ~40 ms/token QUEST decode.

### 5.2 Selection point + what stays resident

- **Selection at `kitty_attention.py:928`**: `selected = _select_quest_pages_metadata(query, kv_cache, shared_page_count, topk)` (body `:784-807`). It scans **only `KeyPage_Min/Max`** (`:794-795`), never dequantizes pages. The `selected` tensor `(B, H_KV, topk)` is the exact fetch list, known **before** the `qk_sparse_kernel` launch at `:943`.
- **Must stay GPU-resident:** `KeyPage_Min/Max` (~32 MiB/layer, ~1.12 GiB total @128k), `PageTable_K/V`, `Sink_*`, `Q_Buffer_*`, `Local_Buffer_V`.
- **CPU-resident pinned master:** `KeyCache`, `ValueCache`, `KeyCache_metadata`, `ValueCache_metadata` (all four).

### 5.3 The fetch-on-select mechanism — **stage all four arrays + cover tail pages** (correctness critic)

The sparse kernels indirect through `PageTable_K/V` (logical→physical) and dereference **four** payload tensors plus an **unconditional tail loop** over logical pages `[shared_page_count, page_count)`:
- `qk_sparse_kernel` reads `KeyCache` and page-id-indexed `KeyCache_metadata` (`:485`), with a tail loop (`:513-541`) through the same `page_table_ptr`/`cache_ptr`.
- `sv_sparse_kernel` reads `ValueCache` + `ValueCache_metadata` (`:629`), tail loop (`:646-665`).

So the redirect must:

1. After `:928`, take the **union of `selected` logical pages** (across heads, or per-head) **∪ the recent-tail pages** `[shared_page_count, page_count)` (≤1 page for page16).
2. `copy_` those CPU page rows for **all four arrays** (K bytes, K meta, V bytes, V meta) into **parallel GPU staging arenas** on the copy stream, indexed by the **same staging page-id**.
3. Build **one compact per-step page table** mapping every needed logical page (selected ∪ tail) → its staging row. Pass the four staging tensors as the kernel args (`:954-957`, `:989-992`). **Kernels need no edits** — only the page-table entries and the four tensor pointers change. Verify the staging arena row stride matches the kernel's metadata stride layout.
4. Keep the original logical→physical table **immutable**; use a **per-step scratch staging table** so stale physical rows don't leak across steps.
5. New-page writes (`quantize_decode` `kitty.py:256-300`, `quantize_prefill` `:202-249`) must target the **CPU master** via the `page_offset` arg already threaded through `quantize_pack_*` (`kitty_quant_pack.py:88, 207`): write to a GPU scratch page, evict it to the CPU master, bump `PageCount_*`. (This is a real change, not free — the first draft understated it.)

### 5.4 Intra-layer constraint

QUEST selection for layer L+1 needs **L+1's query**, unavailable until L+1 runs. So the *selected-page fetch* is **intra-layer** (between `:928` and `:943`), not prefetchable a full layer ahead. What you *can* double-buffer a layer ahead is the **query-independent `KeyPage_Min/Max` + constant Sink/Q/Local/tail buffers**. The ~1.78 MB selected fetch (~0.07 ms) overlaps only partially with the same layer's sparse kernel; a small synchronous copy is acceptable.

### 5.5 Memory + bandwidth (fetch-on-select, page16, budget 2048→128 pages, H_KV=8, D=128)

| Quantity | Value |
|---|---|
| Selected K+V payload / layer / step | 128 × 9728 B ≈ 1.22 MB |
| + per-page metadata (K 4096 B, V 512 B) | 128 × 4608 B ≈ 0.576 MB |
| Total / layer / step | ~1.78 MB |
| Across 36 layers / decode step | ~64 MB |
| One-way at 25 GB/s | **~2.56 ms/pass** (16× headroom under ~40 ms/token) |
| GPU staging arena (topk+tail+slack, ×4 arrays) | ~1.3 MB/layer → ~47 MB total |
| `KeyPage_Min/Max` resident (must stay) @128k | ~1.12 GiB total |
| Packed KV + metadata moved off GPU @128k | ~3.94 GiB → host |

---

## 6. Memory math (40 GB A100)

**Confirmed model facts** (Open Q #1 closed): GLM-4-9B-Chat-1M — `num_hidden_layers=40`, `multi_query_group_num=4` (**H_KV=4**), `kv_channels=128`, `ffn_hidden_size=13696`, bf16 weights **~17.7 GiB**. Qwen3-8B — 36 layers, H_KV=8, head_dim=128, weights ~16 GiB.

fp16 KV/token/layer = `2(K+V) × H_KV × head_dim × 2 B`. GLM = 2048 B/token/layer (×40 = 80 KiB/token). Qwen = 4096 B/token/layer (×36 = 144 KiB/token).

Per-layer prefill transients are **freed each layer (NOT cumulative)** — they are the prefill peak's dominant *transient* term but appear once, not ×layers. Two of them for GLM:
- **MLP transient:** `dense_h_to_4h` `[1,S,2·13696]` + swiglu `[1,S,13696]`. @128k = 6.69 + 3.34 = ~10 GiB.
- **MQA head-expansion transient:** GLM expands stored H_KV=4 → H=32 (8×) for the attention input — `~2 GiB/layer @128k` (K+V), created inside `orig_forward` regardless of offload. Offload does NOT remove it. It lives in the *attention phase*; the MLP transient lives in the *MLP phase* — sequential within a layer, so the **peak is `max` of the two phases, not their sum**.
- **PostQuant clone:** `update(PostQuant=True)` returns a full clone of the active layer's KV (+256 MiB GLM / +512 MiB Qwen @128k), live during that layer's update.

### 6.1 GLM-9B, sim fp16 (H_KV=4, 40 layers)

| Ctx | Weights | KV total (no-offload) | Resident KV (offload window+clone) | MLP transient | MQA-expand | Peak no-offload | Peak offload | Fit 40 GB? |
|---|---|---|---|---|---|---|---|---|
| 32k | 17.7 | 2.50 | ~0.2 | ~2.5 | ~0.5 | ~23 | ~21 | both fit |
| 128k | 17.7 | 10.0 | ~0.75 | ~10 | ~2.0 | **~38 (OOM-prone)** | **~29** | offload fits |
| 256k | 17.7 | 20.0 | ~1.5 | ~20 | ~4.0 | ~58 (hard OOM) | **~30–32** | only offload fits |

Decisive 128k line: no-offload ≈ W(17.7) + cumulative KV(10) + current-layer MLP(10) = ~37.7 GiB → dies on the swiglu alloc (matches the measured ~36 GiB-held OOM). With offload, cumulative KV collapses to the window (~0.75 GiB), so the **MLP phase** peak ≈ 17.7 + 0.75 + 10 = **~28.5 GiB**; the attention phase (17.7 + 0.75 + 2.0 MQA + scores) is lower. **Predict ~29 GiB → fits.** Validate empirically (don't trust the table).

### 6.2 Qwen3-8B, sim fp16 (H_KV=8, 36 layers)

| Ctx | Weights | KV total (no-offload) | Resident KV (offload) | MLP transient | Peak no-offload | Peak offload | Fit? |
|---|---|---|---|---|---|---|---|
| 32k | 16.0 | 4.50 | ~0.5 | ~2.5 | ~23 | ~19 | both fit |
| 128k | 16.0 | 18.0 | ~1.5 | ~9 | ~43 (OOM) | **~27** | only offload fits |
| 256k | 16.0 | 36.0 | ~2.5 | ~18 | ~70 (OOM) | **~37** | only offload fits (tight) |

### 6.3 Qwen3-8B, real Triton QUEST+Kitty (packed)

Packed **+ metadata** = K payload 5632 B + V payload 4096 B + K meta 4096 B + V meta 512 B ≈ **14336 B/page**; page16 → ~112 MiB/layer @128k.

| Ctx | Weights | Packed KV+meta total | KeyPage meta (resident) | Staging (fetch-on-select) | Peak no-offload | Peak fetch-on-select | Fit? |
|---|---|---|---|---|---|---|---|
| 32k | 16.0 | 0.98 | 0.28 | ~0.05 | ~19 (env-measured ~19.8 alloc) | ~17 | both fit |
| 128k | 16.0 | 3.94 | 1.12 | ~0.05 | ~22 | ~18 | both fit |
| 256k | 16.0 | 7.88 | 2.25 | ~0.05 | ~26 (+`MAX_PAGE` over-alloc) | ~19 | both fit; offload removes over-alloc |

**Conclusion:** offload is **load-bearing for the sim path** (GLM/Qwen 128k–256k OOM → fit) and a **scaling enabler** for the real path (fits anyway at 128k, but fetch-on-select keeps 256k+ flat and removes the static `MAX_PAGE` over-allocation).

### 6.4 The 256k MLP wall (honest caveat)

Offload removes cumulative KV but NOT the per-layer transients. At 256k the GLM MLP transient alone is ~20 GiB and the Qwen ~18 GiB; with weights that is ~37 GiB — **fits but tight**, and beyond ~256k a single-layer MLP transient + weights exceeds 40 GB. Breaking that needs **chunked/streaming prefill or activation checkpointing — explicitly out of v1 scope.** Validate the §6 256k predictions empirically before claiming the win there.

---

## 7. Bandwidth / perf analysis

Effective PCIe gen4 x16 ≈ 25 GB/s **with pinned memory** (≈half, and non-overlapping, if pageable).

### 7.1 Sim fp16 — the honest, severe cost

| Ctx | KV/layer (fp16) | one-way ms/layer | round-trip ms/layer | full decode pass (round-trip) | vs baseline decode |
|---|---|---|---|---|---|
| 32k (GLM H_KV=4) | 64 MiB | ~2.7 | ~5.4 | ~0.21 s/token | severe |
| 128k (GLM H_KV=4) | 256 MiB | ~10.7 | ~21.5 | **~0.86 s/token** | **~50–77× slower** |

The PCIe copy of 256 MiB (~10.7 ms) is ~77× slower than the A100 HBM read of the same data (~0.14 ms). So full-fp16 offload at ≥128k is not merely "bandwidth-bound" — **decode is ~50–77× slower than baseline** (~0.86 s/token vs sub-20 ms). Double-buffering hides L+1's H2D behind L's compute **only if compute/layer ≳ copy/layer**, which is false at ≥128k (copy ≫ compute).

**Chosen mechanism (user decision): prefetch / double-buffering is primary.** Issue layer L+1's H2D on the copy stream as soon as layer L starts, so the transfer overlaps compute. This fully hides the cost at small/medium context and partially at long context. The mitigations below are the **backstop for the ≥128k case where prefetch alone cannot keep up:**

1. **`resident_layers` partial offload (backstop, default = offload-only-what-overflows)** — keep as many recent layers GPU-resident as the spare headroom allows (at GLM 128k there is ~11 GiB headroom ≈ ~43 layers worth, so in practice almost nothing is offloaded and there is no latency hit). Only when headroom is tight (256k, or a smaller card) does meaningful offload — and its latency — kick in. This is the latency/memory dial; default it to "resident until the working set would OOM, then offload the oldest."
2. **Sink+buffer locality** — keep the hot `sink_length`(32) + `buffer_length`(128) window GPU-resident; offload only the settled/quantized bulk.
3. **Frame as feasibility, not speed** — LongBench is throughput-tolerant; ~0.4–0.9 s/token to *run-at-all* is acceptable. **Never claim free offload for the sim path.**

### 7.2 Real packed — where double-buffering wins

| Mode | bytes/layer/step | one-way ms/layer | full pass (36L) | hidden by compute? |
|---|---|---|---|---|
| whole-layer packed offload | 112 MiB @128k | ~4.7 | ~169 ms | partial; borderline |
| **fetch-on-select** | ~1.78 MB | ~0.07 | **~2.56 ms** | **fully hidden** (16× headroom under ~40 ms/token) |

Fetch-on-select makes the transfer negligible — you pay ~0.07 ms/layer to keep ~3.94 GiB off the GPU. This is why the real path's value is bandwidth-elegant, not memory.

### 7.3 Summary — when double-buffering hides the cost

- **Hides:** real fetch-on-select (always); sim path *only* at small ctx (≤32k) or when just a few overflow layers are offloaded.
- **Does NOT hide:** sim full-fp16 offload at ≥128k (transfer-bound, ~50–77× slower). Win is feasibility, paid in latency — bound it with `resident_layers`.

---

## 8. Risks & failure modes

1. **PCIe-bound sim decode (≥128k):** ~0.86 s/token, ~50–77× slower. *Mitigation:* `resident_layers` partial offload (v1 requirement), sink+buffer locality. *Pass/fail:* measure ms/token; if over budget, reduce offloaded-layer count.
2. **Non-pinned `non_blocking` trap:** `.to("cpu", non_blocking=True)` on pageable dst silently synchronizes. *Mandatory:* pre-pinned mirrors + `dst.copy_(src, non_blocking=True)`; never `.to("cpu")` on the hot path.
3. **Pinned host-RAM pressure & per-sample churn:** runner builds a fresh cache per sample and doesn't `reset()` for non-GLM → naive per-sample pinning churns hundreds of large non-pageable allocs. Worst-case one-time pinned footprint: GLM @256k H_KV=4 ~20 GiB; **Qwen @256k ~40 GiB**. *Mitigation:* process-lifetime pinned **pool** reused via `copy_`; chunked growth; monitor `MemAvailable`/RSS across a **multi-sample** run and abort on pinned-alloc failure.
4. **Aliasing of returned KV:** safe for sim `PostQuant=True` (clones). **Unsafe** for sim `PostQuant=False` (`:315`) and **GLM** (alias return `:188`). *Mitigation:* assert `PostQuant` when sim `offloading`; for GLM evict only after `:105` consumption.
5. **`generate()` compat (transformers 4.57.6):** `get_seq_length` reads `.shape[-2]` (device-independent, safe on CPU slots); length bookkeeping is independent of residency. Use `torch.cuda.Stream`/`Event` (2.4 API). GLM keeps its committed shims (`_extract_past_from_model_output`, `prepare_inputs_for_generation` coercion).
6. **GLM host-cat crash (load-bearing):** the vendored `SelfAttention.forward` cats the previous step's `kv_caches[index]` *before* our update; if it was evicted to host, it's a CPU-vs-GPU `torch.cat` hard crash. *Mitigation:* prefetch module `index` **before `layer(...)` at `:97`**; assert device match at the cat. *Caveat:* the exact vendored line is `trust_remote_code` and version-dependent — M2 must dump the live source.
7. **inference_mode interaction:** allocate pinned mirrors + streams + events **at construction (outside `inference_mode`)**; hot-path `copy_`/`record`/`wait_event` are inference-mode-safe but exercise the path in M0/M1 before claiming byte-identical predictions.
8. **Evict-stream recycle race:** hold the evicting GPU tensor referenced until `_d2h_done[L]` is observed (the caching allocator could otherwise reuse the block mid-copy). *Pass/fail:* CUDA sanitizer + determinism across two identical runs. *Fallback:* evict on the default stream (stock behavior) for simplicity.
9. **Event lifecycle:** events re-recorded on `reset()`; stale events → races. Cover with the determinism test.
10. **Real-path page-table rewrite:** stage **all four** arrays (K/V bytes + K/V metadata) consistently and cover tail pages; keep the original table immutable and use a per-step scratch table.
11. **256k MLP wall:** offload doesn't touch the per-layer MLP/MQA transients; beyond ~256k they + weights exceed 40 GB. Out of scope; validate the §6.4 predictions empirically.
12. **Real cache lacks reset/reorder:** add `reset()` + `_page_in_all()` for multi-sample reuse.

---

## 9. Phased rollout (each with a GPU0 probe + pass/fail)

All verification on **GPU0** (`CUDA_VISIBLE_DEVICES=0`). GPU0 confirmed fully free (40 GB) at plan time.

### M0 — Plumbing, no behavior change
Add config fields + state + no-op `_prefetch/_evict/_ensure_host_mirror` (default off). Exercise under `inference_mode`.
```
KITTY_OFFLOAD=0 bash scripts/run_exp.sh llama32 --gpu 0 --max-samples 2
```
**Pass:** byte-identical predictions to current branch HEAD (regression guard).

### M1 (FIRST WIN) — Sim Qwen3/Llama offload (machinery + memory)
Wire the pump into `KittyKVCache.update()` (§4.1). PostQuant-only, pinned pool, events, copy stream, `resident_layers`. No GLM, no modeling-file edits.
- **Probe A (correctness):** `KITTY_OFFLOAD=1 RUN_MODE=full bash scripts/run_exp.sh llama32 --gpu 0 --max-samples 2` → **predictions identical (fp16 tol) to offload-off.**
- **Probe B (memory):** Qwen3-8B sim @128k, `max_new_tokens=1`, `torch.cuda.max_memory_allocated()` on vs off → **offload peak ≥ 12 GiB lower; no OOM where off OOMs.**
- **Probe C (latency honesty):** decode ms/token at 32k and 128k → **matches §7.1 within ~30%; documented as memory-for-bandwidth.**
- **Probe D (host RAM):** multi-sample run, monitor RSS/`MemAvailable` → **no growth across samples (pool reused).**
- **Probe E (sanitizer):** `compute-sanitizer`/determinism across two identical runs → **no race.**

### M2 — GLM offload (headline OOM target)
`GLMOffloadManager` + hooks at `glm_kitty_patch.py:97`/`:105` + vendored-cat page-in guard (§4.2). **Pre-gate:** dump the live `SelfAttention.forward` source; identical-logits check at a context that fits without offload (e.g. 32k).
```
MAX_MODEL_LEN=131072 KITTY_OFFLOAD=1 RUN_MODE=full \
  bash scripts/run_exp.sh glm --gpu 0 --max-samples 2 --variant kitty
```
**Pass:** completes without OOM; peak ≈ 29 GiB (vs ~38 OOM-prone off); 32k logits match no-offload reference; tiktoken tokenizer loads (it's present).

### M3 — Real path whole-layer offload (benchmark-only, Qwen3-only)
`KVCache_Layer.offload/prefetch` + `KittyCache.update` prefetch + **evict at `modeling_qwen3.py:498` after `:255`** (§4.4). Real cache is not in the LongBench runner → benchmark harness only.
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. python latency_benchmarking/benchmark_kitty.py \
  --model /mnt/data/tzj/models/Qwen3-8B --cache_implementation 0 --page_size 16 \
  --promote_ratio 0.125 --max_seq_len 32768 --max_new_tokens 32 --batch_size 1 \
  --warmup_runs 1 --repeat_runs 3 --compare-quest-kitty --quest-enabled \
  --quest-token-budget 2048 --quest-skip-layers 0 --offload
```
**Pass:** identical generated tokens to off; QUEST labels still `triton_sparse_reduced_budget`; decode within ~169 ms/pass of baseline (whole-layer cost). *(Decide with the user whether to skip M3 and go straight to M4 — M3 may be busywork since 2-bit KV already fits.)*

### M4 — Real path fetch-on-select (the elegant synergy)
Implement §5: CPU master + 4 GPU staging arenas + per-step scratch page table after `:928`/before `:943`; `page_offset` writes to CPU master; keep `KeyPage_Min/Max` resident. Add identical-token oracles.
```
# same as M3, plus --max_seq_len 131072 and 262144 runs; instrument staging bytes/step
```
**Pass:** decode within ~2.5 ms/pass of baseline (fetch fully hidden); packed KV resident ≈ 0 (only staging + KeyPage meta); QUEST labels unchanged; 256k completes where static-`MAX_PAGE` no-offload over-commits.

### M5 — Hardening
`reset`/`reorder`/`_page_in_all` guards, pinned-RAM chunked growth + abort, warn-once device-mismatch fallbacks, sanitizer pass, full LongBench (no `--max-samples`) GLM + Qwen @32k.
**Pass:** scores match published no-offload LongBench within noise; no host-RAM exhaustion.

---

## 10. Decisions (resolved) + remaining open questions

**Resolved:**
1. **GLM `num_key_value_heads`** → H_KV=4, 40 layers, head_dim=128, bf16 (config verified). The §6.1 H_KV=4 table is correct.
2. **Scope** → **v1 = sim path only**; real Qwen3 Triton M3/M4 deferred (user, 2026-06-02).
3. **Latency** → **prefetch/double-buffering primary**; `resident_layers` is the backstop, default "offload only what overflows" (user). No fixed ms/token target required for v1.
4. **Host RAM** → **process-level pinned pool + `MemAvailable` guard** (user); preallocate to `max_length`, reuse across samples, abort if the guard trips.
5. **`max_length` source** → `MAX_MODEL_LEN + MAX_GEN` at construction; `generate()` exceeding it hard-asserts (no dynamic re-pinning in v1).

**Still open (low-stakes, sensible defaults chosen unless you object):**
6. **Opt-in surface:** plan to expose `KITTY_OFFLOAD=1` env + `VariantConfig.offloading` + (later) `--offload`/`--resident-layers`, all feeding one ctor arg. OK?
7. **`resident_layers` default value:** "offload-only-what-overflows" (compute the resident count from free VRAM at construction) vs a fixed small N. Default to the adaptive form.
8. **(Deferred, real path only)** per-head vs union page selection for fetch-on-select — revisit if/when M4 is scheduled.

---

## Appendix B — Implementation status & measured results (node68, GPU1, A100-SXM4-40GB)

Implemented on branch `tzj/kvcache-offload` and validated on node68 GPU1 in the
`kitty` conda env (**transformers 4.53.2** there, not 4.57.6 — the sim path and
the committed GLM patch both work on it; `DynamicCache` uses the classic
`key_cache`/`value_cache` lists, which suits the design).

**Code:** `src/kitty_sim/kv_offload.py` (`LayerKVOffloader` + process-lifetime
pinned pool + MemAvailable guard), `kitty_simulate.py` (offload hooks in
`update()`), `glm_kitty_patch.py` (`GLMOffloadManager` + prefetch/evict in the
patched GLMTransformer loop), `runner.py` (`KITTY_OFFLOAD` opt-in for both
paths), `tools/probe_kv_offload.py` + `tools/probe_glm_offload.py` (probes).

**What is built (v1, synchronous — max memory saving):** M0 plumbing, M1a sim
Qwen/Llama offload, M2 GLM offload. The synchronous design keeps the working set
at ~1 layer, which is exactly right for the OOM-avoidance goal. Outputs are
**bit-identical** to offload-off (offload is a pure CPU↔GPU relocation of fp16 KV).

| Path | Ctx | peak_alloc OFF | peak_alloc ON | Generated ids identical | Note |
|---|---|---|---|---|---|
| Llama-3.2-1B sim | 2k | 2.50 | 2.44 | ✓ | tiny KV |
| Qwen3-8B sim | 32k | 23.03 | 18.53 | ✓ | saved 4.50 GiB = full KV |
| Qwen3-8B sim | 128k | OOM (~43) | **28.33** | — | fits where OFF OOMs |
| GLM-9B sim | 4k | 18.65 | 18.36 | ✓ | tiny KV |
| GLM-9B sim | 128k | OOM (~43) | **36.57** | — | fits (needs expandable_segments) |

End-to-end LongBench (runner) smoke with `KITTY_OFFLOAD=1`: llama32 and GLM both
complete with offload engaged, `trec` score 100.

**Important operational finding — `expandable_segments` for extreme contexts.**
The synchronous offload churns GPU tensors (alloc/free each layer), which
fragments the caching allocator. GLM-9B @128k first OOMed with offload at
peak_alloc 33.26 GiB + 5.56 GiB *reserved-but-unallocated* (fragmentation),
dying on the 3.34 GiB swiglu transient despite ~36.6 GiB being enough. Setting
**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** reclaims the fragmentation
and the run fits (peak_alloc 36.57, reserved 38.96 / 40). So for offload at
≥128k, set `expandable_segments:True` (it must be set before CUDA init, so it
cannot be forced from inside the cache code — document it / set it in the launch
env). At 128k the per-layer MLP transient (~10 GiB) is the remaining wall; offload
removes the cumulative KV, MLP is what keeps the peak near 37 GiB.

**Run commands (GPU1 on node68):**
```bash
# correctness + memory probe (sim Qwen/Llama), off vs on:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python tools/probe_kv_offload.py \
  --model /home/tzj/models/Qwen3-8B --context-len 32768 --max-new-tokens 4
# GLM offload at 128k (needs expandable_segments):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  python tools/probe_glm_offload.py --model /home/tzj/models/GLM-4-9B-Chat-1M \
  --context-len 131072 --max-new-tokens 2 --offload
# end-to-end LongBench with offload:
DATASETS_CSV=trec MAX_MODEL_LEN=8192 KITTY_OFFLOAD=1 bash scripts/run_exp.sh glm --gpu 1 --max-samples 1 --variant kitty
```

**M1b — opt-in double-buffered prefetch (built + measured).** The sim
`LayerKVOffloader` gained an optional prefetch mode (`KITTY_OFFLOAD_PREFETCH=1` /
`--prefetch` / `offload_prefetch`): the next layer's KV is H2D'd one step ahead on
a dedicated copy stream (event-ordered), overlapping the transfer with compute;
evict stays synchronous so the working set grows by only ~1 staged layer.

Measured Qwen3-8B @32k (gen=8), all three bit-identical:

| Mode | time | slowdown vs off | peak_alloc | peak_reserved |
|---|---|---|---|---|
| off | 39.7 s | 1.00× | 23.03 | 23.99 |
| offload, synchronous | 78.5 s | 1.98× | 18.53 | 20.99 |
| offload, **prefetch** | 68.7 s | **1.73×** | 18.53 | 21.37 |

So prefetch cuts ~12% of the offload latency at **no peak_alloc cost** (+0.38 GiB
reserved only). Per §7 the benefit is bounded — at ≥128k copy ≫ compute and the
synchronous evict's D2H is not hidden — so the **default stays synchronous**
(memory-optimal), and prefetch is opt-in for moderate-context / has-headroom
runs. The **GLM path is intentionally synchronous-only** (no prefetch) so the
tight GLM-128k peak (38.96/40) keeps the minimal ~1-layer working set.

---

## Appendix C — Real Triton QUEST+Kitty kernel KV offload (2026-06-03)

After rebasing onto the updated `tzj/kitty` (which added the real Triton QUEST+Kitty
kernel for Llama AND GLM), the offload was adapted to the **real** `KittyCache`
(`src/kitty/kvcache/kitty.py`). The real cache stores **2-bit packed** KV in
statically-allocated per-layer buffers (`KeyCache`, `KeyCache_metadata`,
`ValueCache`, `ValueCache_metadata`); these are offloaded to pinned host RAM, while
`KeyPage_Min/Max` (the QUEST page-selection bounds) stay GPU-resident.

**Implementation is entirely in `kitty.py` (no modeling-file edits).** The decode
kernel reads the packed buffers *between* `update()` and `quantize_decode()`, so the
hooks are: ensure-resident at `update()`-top (decode) + `quantize_prefill()`-top;
evict at `quantize_prefill()`-end. **Key subtlety:** the buffers are allocated upfront
on GPU at cache construction, so a naive progressive evict still leaves the *first*
layer's MLP peak carrying the other layers' packed KV (first attempt saved only
0.22 GiB). The fix evicts **all** layers at `__init__`, then `quantize_prefill` pages
each layer back only to pack and immediately evicts — so any layer's MLP transient
runs with ~0 of the other layers' packed KV resident.

Opt-in: `KITTY_OFFLOAD=1` (runner real-kernel + GLM real-kernel paths) /
`offloading=` (`get_kvcache_kitty`) / `--offload` (`tools/probe_real_kernel.py`).

**Measured @128k, GPU0 (bit-identical off/on; QUEST stays `triton_sparse_reduced_budget`,
128 pages; one-time transfer so no per-token latency cost):**

| Config @128k | peak_alloc | time | fits 40 GB? |
|---|---|---|---|
| Qwen3-8B sim + offload | 28.33 | 295 s | yes |
| Qwen3-8B real kernel, no offload | 33.41 | 48 s | yes |
| **Qwen3-8B real kernel + offload** | **29.47** | 50 s | yes |
| GLM-9B real kernel, no offload | — | — | **OOM (~38.7, swiglu)** |
| **GLM-9B real kernel + offload** | **36.70** | 94 s | **yes** (expandable_segments) |

**Takeaways:**
- Qwen real+offload (29.47, fast) is the sweet spot — low peak AND ~6× faster than sim+offload.
- **GLM real kernel @128k OOMs without offload but FITS (36.70) with it** — the offload is load-bearing for the GLM real kernel.
- The floor is the per-layer **MLP transient** (Qwen ~13, GLM ~19 GiB) + resident `KeyPage` QUEST metadata (~1.15). KV offload removes only the packed KV; it does **not** cut the MLP floor (only chunked prefill does). So GLM real+offload (36.70) ≈ GLM sim+offload (36.57), both MLP-bound — but the real kernel makes long **decode** sparse/fast.

---

## Appendix A — Corrections folded in from adversarial review

| # | First-draft claim | Correction (source) |
|---|---|---|
| 1 | evict real-path KV inside `update()` | **use-after-evict race** — packed KV read at `modeling_qwen3.py:249` & written `:255` after `update()`; evict at `:498` gated post-`:255` (correctness) |
| 2 | GLM offload via per-layer lists on each `_kitty_cache` | GLM = private single-layer caches; needs a **manager on `GLMTransformer`**, prefetch before `:97`, evict after `:105` (correctness + memory) |
| 3 | `src/transformers/modeling_qwen3.py` | real path is **`src/kitty/models/qwen3/modeling_qwen3.py`** (all critics) |
| 4 | fetch-on-select "kernels need no edits", stage K/V only | stage **all 4 arrays + tail pages**, scratch page table, `page_offset` writes to CPU master (correctness + completeness) |
| 5 | pinned buffers a one-time cost | **per-sample churn** — fresh cache/sample, no `reset()` for non-GLM → use a reused **pinned pool**; monitor multi-sample RSS (correctness + completeness) |
| 6 | working window = 2 layers (~0.5 GiB) | add **PostQuant clone** (+256/512 MiB) and **GLM MQA-expand** (~2 GiB/layer @128k); window ≈ 3–4 layer-equiv (memory) |
| 7 | real packed KV 0.67/2.67/5.34 GiB | **+metadata → 0.98/3.94/7.88 GiB**; whole-layer 112 MiB/layer → ~169 ms/pass (memory) |
| 8 | sim ≥128k "bandwidth-bound" | quantified **~50–77× slower** (~0.86 s/token); `resident_layers` partial offload is a **v1 requirement** (memory) |
| 9 | GLM H_KV open question | **closed**: H_KV=4, 40 layers (config verified) |
| 10 | header "no tiktoken" | **tiktoken 0.13.0 present** — GLM tokenizer not blocked (completeness, verified) |
| 11 | real path implies a GLM/LongBench route | real cache **not in the runner**; **Qwen3-only, benchmark-only** (completeness) |
| 12 | `get_kvcache_kitty(**kwargs)` | takes an **`argparse.Namespace`** → thread via config + Namespace attrs (correctness + completeness) |
| 13 | inference_mode unaddressed | allocate pinned/streams/events **outside inference_mode** at construction (correctness) |
