# -*- coding: utf-8 -*-
"""Per-task K/Q statistics collection for the mask-calibratability experiment.

Ported from worktree eval_design_fig_6/overlap_pipeline (2026-07-28 protocol).
Uses repo ``src/kitty_sim`` (no vendored copy). Protocol:

  * rows[row_skip : row_skip + num_prompts] of the task jsonl (shared row ids)
  * prompt = dataset2prompt[task].format(**row); llama3.2 chat wrap unless the
    task is in NO_CHAT_DATASETS (eval-identical construction)
  * tokenize with the model tokenizer (default add_special_tokens, mirroring
    the eval runner); token-level middle-truncate to max_len when longer
    (head max_len//2 + tail). Shorter prompts are kept as-is ("至多 2048").
  * one forward per prompt (bf16, eager, use_cache); collect
      sigma2_i  [nl, n_kv, D] : 128-token block-centered population variance
                of post-RoPE keys (tokens after skip_first; ng = T//G full
                blocks, remainder dropped — identical to qlut_quant._grouped
                for T >= G; for T < G a single partial block over all T
                tokens is used so short tasks like trec stay valid)
      q_absmean_i [nl, n_kv, D] : mean |q| over post-skip tokens, post-RoPE,
                query heads averaged within each GQA group
  * all computation on GPU (fp64 accumulators); per-prompt tensors saved fp32
    plus fp64 means and full provenance metadata.

No text generation, no scoring — forward passes only.
Boundary: mask/statistics only; no accuracy claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from kitty_sim.longbench.config import (  # noqa: E402
    NO_CHAT_DATASETS,
    load_json_config,
)
from kitty_sim.longbench.data import resolve_dataset_file  # noqa: E402
from kitty_sim.qlut_quant import channel_sigma2  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-tag", required=True)
    p.add_argument("--task", required=True)
    p.add_argument(
        "--data-root",
        required=True,
        help="LongBench root containing data/<task>.jsonl "
        "(same convention as scripts/run_exp.sh --data-root)",
    )
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--row-skip", type=int, default=32)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--skip-first", type=int, default=32)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="smoke: only N prompts")
    return p.parse_args(argv)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_prompt_ids(tokenizer, dataset2prompt, task, row, max_len):
    """Eval-identical prompt build + token-level middle truncate to max_len."""
    prompt = dataset2prompt[task].format(**row)
    if task not in NO_CHAT_DATASETS:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    ids = tokenizer(prompt, truncation=False).input_ids
    if len(ids) > max_len:
        half = max_len // 2
        ids = ids[:half] + ids[-(max_len - half):]
    return ids


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
    if cfg.model_type == "llama":
        import transformers.models.llama.modeling_llama as ml
    elif cfg.model_type == "qwen3":
        import transformers.models.qwen3.modeling_qwen3 as ml
    else:
        raise ValueError(f"unsupported model_type: {cfg.model_type}")
    return ml


def channel_sigma2_gpu(x, G):
    """channel_sigma2 with a partial-single-block fallback for T < G.

    x: [H, D, T] float on GPU. For T >= G identical to
    kitty_sim.qlut_quant.channel_sigma2 (ng = T//G full blocks, remainder
    dropped). For 0 < T < G uses one partial block over all T tokens.
    """
    H, D, T = x.shape
    if T >= G:
        return channel_sigma2(x, G)
    if T < 2:
        return torch.full((H, D), float("nan"), device=x.device, dtype=x.dtype)
    xc = x.float() - x.float().mean(-1, keepdim=True)
    return xc.pow(2).mean(dim=-1)


@torch.no_grad()
def main():
    args = parse_args()
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    data_path = resolve_dataset_file(args.task, args.data_root)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    dataset2prompt = load_json_config("dataset2prompt.json")

    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    sel_rows = rows[args.row_skip: args.row_skip + args.num_prompts]
    if len(sel_rows) < args.num_prompts:
        raise RuntimeError(
            f"task {args.task}: only {len(sel_rows)} rows after row_skip="
            f"{args.row_skip} (need {args.num_prompts})")
    if args.limit:
        sel_rows = sel_rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager").to(device).eval()
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    nl = cfg.num_hidden_layers
    group = n_q // n_kv
    print(f"[collect] {args.model_tag}/{args.task}: layers={nl} kv={n_kv} "
          f"D={head_dim} prompts={len(sel_rows)}", flush=True)

    # RoPE tap for post-RoPE q (fires exactly once per layer per forward).
    rope_mod = rope_module_for(cfg)
    orig_rope = rope_mod.apply_rotary_pos_emb
    state = {"call": 0}
    q_accum = torch.zeros(nl, n_kv, head_dim, dtype=torch.float64, device=device)

    def tapped(q, k, cos, sin, *a, **kw):
        q_emb, k_emb = orig_rope(q, k, cos, sin, *a, **kw)
        li = state["call"] % nl
        state["call"] += 1
        T = q_emb.shape[2]
        eff_skip = args.skip_first if T > args.skip_first + 1 else 0
        qa = q_emb[0, :, eff_skip:, :].abs().mean(dim=1)          # [n_q, D]
        q_accum[li] += qa.reshape(n_kv, group, -1).mean(dim=1).double()
        return q_emb, k_emb

    n_prompts = len(sel_rows)
    s2_prompts = torch.empty(n_prompts, nl, n_kv, head_dim, dtype=torch.float32)
    q_prompts = torch.empty(n_prompts, nl, n_kv, head_dim, dtype=torch.float32)
    s2_accum = torch.zeros(nl, n_kv, head_dim, dtype=torch.float64, device=device)
    q_total = torch.zeros(nl, n_kv, head_dim, dtype=torch.float64, device=device)
    token_lens, row_ids, short_flags = [], [], []
    raw_ids = []

    rope_mod.apply_rotary_pos_emb = tapped
    try:
        for pi, row in enumerate(sel_rows):
            ids = build_prompt_ids(tok, dataset2prompt, args.task, row, args.max_len)
            token_lens.append(len(ids))
            row_ids.append(args.row_skip + pi)
            raw_ids.append(str(row.get("_id", "")))
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            q_accum.zero_()
            calls_before = state["call"]
            cache = model(input_ids=input_ids, use_cache=True).past_key_values
            if state["call"] - calls_before != nl:
                raise RuntimeError(
                    f"rope calls {state['call'] - calls_before} != {nl} layers")
            T = input_ids.shape[1]
            eff_skip = args.skip_first if T > args.skip_first + 1 else 0
            short_flags.append(bool(T < args.group_size + eff_skip))
            for li in range(nl):
                keys = cache_layer_keys(cache, li)              # [1, n_kv, T, D]
                x = keys[0, :, eff_skip:, :].transpose(1, 2)    # [n_kv, D, T']
                s2 = channel_sigma2_gpu(x.float(), args.group_size)  # [n_kv, D] GPU
                s2_prompts[pi, li] = s2.cpu()
                s2_accum[li] += s2.double()
            q_prompts[pi] = q_accum.float().cpu()
            q_total += q_accum
            del cache
            if (pi + 1) % 16 == 0:
                print(f"  [collect] {args.task} {pi + 1}/{n_prompts} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        rope_mod.apply_rotary_pos_emb = orig_rope

    sigma2_mean = (s2_accum / n_prompts).cpu()
    q_mean = (q_total / n_prompts).cpu()
    payload = {
        "format": "sensitivity_overlap_task_stats_v1",
        "task": args.task,
        "model": args.model,
        "model_tag": args.model_tag,
        "data_file": str(data_path),
        "data_sha256": sha256_file(data_path),
        "row_skip": args.row_skip,
        "row_ids": row_ids,
        "raw_sample_ids": raw_ids,
        "num_prompts": n_prompts,
        "max_len": args.max_len,
        "skip_first": args.skip_first,
        "group_size": args.group_size,
        "truncation": "token-level middle (head max_len//2 + tail) after chat wrap; "
                      "shorter prompts kept as-is",
        "short_prompt_flags": short_flags,
        "prompt_token_lens": token_lens,
        "n_layers": nl, "n_kv": n_kv, "head_dim": head_dim, "n_q": n_q,
        "sigma2_prompts": s2_prompts,          # fp32 [n, nl, n_kv, D]
        "q_absmean_prompts": q_prompts,        # fp32 [n, nl, n_kv, D]
        "sigma2_mean_fp64": sigma2_mean,       # fp64 [nl, n_kv, D]
        "q_absmean_mean_fp64": q_mean,         # fp64 [nl, n_kv, D]
        "repo_git_head": git_head(REPO),
        "torch": str(torch.__version__),
        "cuda_device": torch.cuda.get_device_name(device),
        "wall_seconds": time.time() - t0,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out / "stats.pt")
    print(f"[collect] wrote {out / 'stats.pt'} "
          f"(T mean {sum(token_lens) / len(token_lens):.0f}, "
          f"short={sum(short_flags)}/{n_prompts}, {time.time() - t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
