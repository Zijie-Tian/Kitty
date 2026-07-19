# -*- coding: utf-8 -*-
"""Offline codebook-mask calibration for the canonical qlutattn variant.

qlutattn quantizes the post-RoPE K cache per token on the per-channel-mean-
centered residual. Each channel is assigned ONE of two codebooks OFFLINE by a
per-(layer, kv-head, channel) ranking on a calibration corpus:

  ranking signal = sigma^2 x E|q|
    sigma^2 : residual variance of the post-RoPE K channel (how HARD the
              channel is to quantize)
    E|q|    : mean absolute post-RoPE query activation on that channel, query
              heads averaged within each GQA group (how much attention
              actually READS it)

  - the 65% lowest-ranked channels  -> sign (1-bit codeword + per-token
    mean-|r| scale, ~1.25 bit nominal)
  - the 35% highest-ranked channels -> nf2 (symnf2-v1 fixed LUT + per-token
    absmax scale, ~2.25 bit nominal)

Nominal K width = 0.65 * 1.25 + 0.35 * 2.25 = 1.60 bit/value. The split and
the signal are FIXED; they are not knobs. Both statistics are collected in ONE
pass over the corpus (K via the model cache, q via a rope wrap). The runtime
loads the mask and uses it unchanged; there is no online ranking.

The per-channel MEAN is NOT calibrated here -- the runtime subtracts a
per-channel mean self-calibrated from each prompt at prefill (free for
attention, q.mu cancels in softmax). This script only fixes the sign-vs-nf2
CODEBOOK assignment. Note E|q| is corpus-dependent: with the default wikitext
corpus, gains concentrate on QA/retrieval-style tasks.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/calibrate_qlutattn_mask.py \
    --model /path/to/model \
    --calib-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
    --output /path/to/model.qlutattn_mask.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from kitty_sim.qlut_quant import channel_sigma2  # noqa: E402

# The canonical qlutattn codebook pair, split, and ranking signal are fixed;
# they are not knobs.
CODEBOOKS = ("sign", "nf2")
SIGN_FRACTION = 0.65
BITS = {"sign": 1.25, "nf2": 2.25}
NOMINAL_K_BITS = SIGN_FRACTION * BITS["sign"] + (1 - SIGN_FRACTION) * BITS["nf2"]  # 1.60
NF2_IMPL = "symnf2-v1"
RANKING_SIGNAL = "sigma2_x_q"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--calib-data", required=True, help="wikitext train parquet")
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--sample-len", type=int, default=2048)
    p.add_argument("--group-size", type=int, default=128, help="sigma^2 submean group along tokens")
    p.add_argument("--skip-first", type=int, default=32, help="skip the sink window when measuring")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    return p.parse_args()


def load_calib_token_batches(tokenizer, parquet_path, num_samples, sample_len, seed):
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={"train": parquet_path})["train"]
    texts = [t for t in ds["text"] if t and not t.isspace()]
    enc = tokenizer("\n\n".join(texts), return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[0]
    need = num_samples * sample_len
    if ids.numel() < need:
        raise RuntimeError(f"corpus too small: {ids.numel()} < {need}")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - sample_len, (num_samples,), generator=g)
    return torch.stack([ids[s:s + sample_len] for s in starts])


def cache_layer_keys(cache, layer_idx):
    try:
        keys, _ = cache[layer_idx]
        return keys
    except Exception:
        pass
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].keys
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx]
    raise TypeError(f"Unsupported cache type: {type(cache)}")


def rope_module_for(cfg):
    """The modeling module whose apply_rotary_pos_emb the q-tap wraps."""
    if cfg.model_type == "qwen3":
        import transformers.models.qwen3.modeling_qwen3 as ml
    elif cfg.model_type == "llama":
        import transformers.models.llama.modeling_llama as ml
    else:
        raise ValueError(f"unsupported model_type for calibration: {cfg.model_type}")
    return ml


@torch.no_grad()
def collect_statistics(model, rope_mod, batches, skip_first, group_size, device,
                       nl, n_kv, group, head_dim):
    """One pass over the corpus collecting BOTH per-channel statistics:
    sigma2 [nl, n_kv, D] from the cached post-RoPE keys, and q_absmean
    [nl, n_kv, D] from the post-RoPE queries. The rope wrap fires exactly once
    per layer per forward in layer order, so layer = call index % n_layers
    (verified by an exact call-count check)."""
    orig_rope = rope_mod.apply_rotary_pos_emb
    state = {"call": 0}
    q_accum = torch.zeros(nl, n_kv, head_dim, dtype=torch.float64)

    def tapped(q, k, cos, sin, *a, **kw):
        q_emb, k_emb = orig_rope(q, k, cos, sin, *a, **kw)
        li = state["call"] % nl
        state["call"] += 1
        qa = q_emb[0, :, skip_first:, :].abs().mean(dim=1)          # [n_q, D]
        q_accum[li] += qa.reshape(n_kv, group, -1).mean(dim=1).double().cpu()
        return q_emb, k_emb

    s2_accum = None
    n = 0
    rope_mod.apply_rotary_pos_emb = tapped
    try:
        for bi in range(batches.shape[0]):
            ids = batches[bi:bi + 1].to(device)
            cache = model(ids, use_cache=True).past_key_values
            per_layer = []
            for li in range(nl):
                keys = cache_layer_keys(cache, li)                  # [1, n_kv, T, D] post-RoPE
                x = keys[0, :, skip_first:, :].transpose(1, 2)      # [n_kv, D, T]
                per_layer.append(channel_sigma2(x.float().cpu(), group_size))  # [n_kv, D]
            per_layer = torch.stack(per_layer)                      # [nl, n_kv, D]
            s2_accum = per_layer if s2_accum is None else s2_accum + per_layer
            n += 1
            del cache
            if (bi + 1) % 16 == 0:
                print(f"  [calib] {bi + 1}/{batches.shape[0]} samples")
    finally:
        rope_mod.apply_rotary_pos_emb = orig_rope

    expected_calls = n * nl
    if state["call"] != expected_calls:
        raise RuntimeError(
            f"rope call count {state['call']} != expected {expected_calls}; "
            "layer attribution would be wrong (model not plain per-layer rope?)")
    return s2_accum / max(n, 1), (q_accum / max(n, 1)).float()


def build_mask(sigma2, q_absmean):
    """[nl,n_kv,D] stats -> codebook mask [nl,n_kv,D] uint8 (0=sign, 1=nf2).
    Per layer, channels are ranked ASCENDING by sigma2 * E|q| over the n_kv*D
    flattened channels (layer-global: louder heads may take more nf2 budget);
    the lowest round(0.65*N) -> sign, the rest -> nf2."""
    nl, n_kv, D = sigma2.shape
    signal = sigma2.float() * q_absmean.float()
    N = n_kv * D
    k_sign = int(round(SIGN_FRACTION * N))
    mask = torch.ones(nl, n_kv, D, dtype=torch.uint8)               # default nf2(1)
    for li in range(nl):
        order = torch.argsort(signal[li].reshape(-1))               # ascending
        m = mask[li].reshape(-1)
        m[order[:k_sign]] = 0
        mask[li] = m.reshape(n_kv, D)
    return mask


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager").to(args.device).eval()
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    nl = cfg.num_hidden_layers
    print(f"[calib] model={args.model} layers={nl} q={n_q} kv={n_kv} D={head_dim}")
    print(f"[calib] signal={RANKING_SIGNAL} low=sign({BITS['sign']}b)/high=nf2({BITS['nf2']}b, {NF2_IMPL}) "
          f"sign_fraction={SIGN_FRACTION} -> nominal K ~{NOMINAL_K_BITS} bit/value")

    batches = load_calib_token_batches(tok, args.calib_data, args.num_samples, args.sample_len, args.seed)
    sigma2, q_absmean = collect_statistics(
        model, rope_module_for(cfg), batches, args.skip_first, args.group_size,
        args.device, nl, n_kv, n_q // n_kv, head_dim)
    mask = build_mask(sigma2, q_absmean)

    n_lo = int((mask == 0).sum())
    tot = int(mask.numel())
    print(f"[calib] sign={n_lo}/{tot} ({n_lo / tot:.2%}) nf2={tot - n_lo}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "codebook_mask": mask,                  # [nl, n_kv, D] uint8: 0=sign, 1=nf2
        "codebooks": list(CODEBOOKS),
        "low_frac": n_lo / tot,                 # ACTUAL fraction (rounding-exact)
        "nominal_bits": NOMINAL_K_BITS,
        "nf2_impl": NF2_IMPL,
        "ranking_signal": RANKING_SIGNAL,
        "group_size": args.group_size,
        "skip_first": args.skip_first,
        "model": args.model,
        "n_layers": nl,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "sigma2": sigma2,                       # [nl, n_kv, D] for inspection
        "q_absmean": q_absmean,                 # [nl, n_kv, D] for inspection
    }, out)
    print(f"[calib] wrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
