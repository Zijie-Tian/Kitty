"""Peak-memory probe for the REAL Triton QUEST+Kitty kernel (Qwen3/Llama).

Loads the architecture-specific *_Kitty model class (real 2-bit paged KittyCache +
QUEST sparse decode kernel) and runs a synthetic prompt, reporting weights-only
memory and the peak during a 128k forward. Lets us compare the real-kernel peak
against the sim+offload peak (the real kernel stores 2-bit KV, so its cumulative
KV is tiny; the per-layer MLP activation transient is the same either way).

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. \
    /mnt/data/tzj/anaconda3/envs/kitty/bin/python tools/probe_real_kernel.py \
    --model /mnt/data/tzj/models/Qwen3-8B --family qwen --context-len 131072 --max-new-tokens 2
"""
from __future__ import annotations

import argparse
import time

import torch


def _model_class(family: str):
    fam = family.lower()
    if "llama" in fam:
        from kitty.models.llama import LlamaForCausalLM_Kitty
        return LlamaForCausalLM_Kitty
    if "qwen" in fam:
        from kitty.models.qwen3 import Qwen3ForCausalLM_Kitty
        return Qwen3ForCausalLM_Kitty
    raise ValueError(f"no real kernel for family={family}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--family", required=True, help="qwen | llama")
    p.add_argument("--context-len", type=int, default=131072)
    p.add_argument("--max-new-tokens", type=int, default=2)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--promote-ratio", type=float, default=0.125)
    p.add_argument("--quest-token-budget", type=int, default=2048)
    p.add_argument("--quest-skip-layers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    assert torch.cuda.is_available()
    print(f"device: {torch.cuda.get_device_name(0)}  real QUEST+Kitty kernel  family={args.family}")
    model_class = _model_class(args.family)
    print(f"loading {args.model} ...")
    model = model_class.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
        local_files_only=True,
    ).eval()
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated() / 2**30
    print(f"WEIGHTS: {weights:.2f} GiB")

    from kitty.kvcache import get_kvcache_kitty as get_real_kvcache_kitty

    config = model.config
    if getattr(config, "head_dim", None) is None:
        config.head_dim = config.hidden_size // config.num_attention_heads
    max_length = args.context_len + args.max_new_tokens
    cache = get_real_kvcache_kitty(
        config,
        1,
        max_length,
        page_size=args.page_size,
        promote_ratio=args.promote_ratio,
        quest_enabled=True,
        quest_token_budget=args.quest_token_budget,
        quest_skip_layers=args.quest_skip_layers,
    )

    vocab = int(config.vocab_size)
    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, vocab, (1, args.context_len), device="cuda")
    eos = config.eos_token_id
    pad = eos if isinstance(eos, int) else (eos[0] if isinstance(eos, list) else 0)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            past_key_values=cache,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    gen = out[0, input_ids.shape[-1]:].detach().cpu().tolist()
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_resv = torch.cuda.max_memory_reserved() / 2**30
    last_path = getattr(cache, "last_quest_path", None)
    print(f"  ctx={args.context_len} gen={args.max_new_tokens} budget={args.quest_token_budget}")
    print(f"  WEIGHTS={weights:.2f}  peak_alloc={peak_alloc:.2f} GiB  peak_reserved={peak_resv:.2f} GiB  "
          f"transient(peak-weights)={peak_alloc - weights:.2f} GiB  time={dt:.2f}s")
    print(f"  last_quest_path={last_path}")
    print(f"GENIDS: {gen}")


if __name__ == "__main__":
    main()
