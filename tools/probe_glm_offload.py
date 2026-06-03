"""Correctness + memory probe for GLM layer-wise CPU KV offload.

GLM uses a legacy tuple cache, so Kitty fake-quant + offload are installed via
install_glm_kitty_fakequant (the SelfAttention/GLMTransformer patch), not via a
KittyKVCache passed to generate(). Run this once with --offload and once without
in separate processes and compare the printed GENIDS (must be identical — offload
is a pure CPU<->GPU relocation of fp16 KV) and the peak GPU memory.

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
    /home/tzj/anaconda3/envs/kitty/bin/python tools/probe_glm_offload.py \
    --model /home/tzj/models/GLM-4-9B-Chat-1M --context-len 4096 --max-new-tokens 8
  # add --offload for the offload run.
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM

from kitty_sim.glm_kitty_patch import install_glm_kitty_fakequant
from kitty_sim.kitty_simulate import KittyKVCacheConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--context-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--offload", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    # Kitty paper-style defaults.
    p.add_argument("--sink-length", type=int, default=32)
    p.add_argument("--buffer-length", type=int, default=128)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--kbits", type=int, default=2)
    p.add_argument("--vbits", type=int, default=2)
    p.add_argument("--promote-ratio", type=float, default=0.125)
    p.add_argument("--promote-bit", type=int, default=4)
    p.add_argument("--channel-selection", type=int, default=1)
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print(f"device: {torch.cuda.get_device_name(0)}  offload={args.offload}")
    print(f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=True,
    ).to("cuda").eval()

    cfg = KittyKVCacheConfig(
        sink_length=args.sink_length,
        buffer_length=args.buffer_length,
        group_size=args.group_size,
        kbits=args.kbits,
        vbits=args.vbits,
        promote_ratio=args.promote_ratio,
        promote_bit=args.promote_bit,
        channel_selection=args.channel_selection,
    )
    max_length = args.context_len + args.max_new_tokens + 8
    stats = install_glm_kitty_fakequant(
        model, cfg, offloading=args.offload, max_length=max_length
    )
    print(f"installed patch: {stats}")

    vocab = int(model.config.padded_vocab_size)
    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, vocab, (1, args.context_len), device="cuda")

    eos = model.config.eos_token_id
    pad = eos if isinstance(eos, int) else (eos[0] if isinstance(eos, list) else 0)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            past_key_values=None,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos,
            pad_token_id=pad,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    gen = out[0, input_ids.shape[-1]:].detach().cpu().tolist()
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_resv = torch.cuda.max_memory_reserved() / 2**30
    print(f"  ctx={args.context_len} gen={args.max_new_tokens} "
          f"calls={stats.get('calls')} prefill={stats.get('prefill_calls')} decode={stats.get('decode_calls')}")
    print(f"  peak_alloc={peak_alloc:.2f} GiB  peak_reserved={peak_resv:.2f} GiB  time={dt:.2f}s")
    print(f"GENIDS: {gen}")


if __name__ == "__main__":
    main()
