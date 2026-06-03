"""Standalone correctness + memory probe for layer-wise CPU KV offload (sim path).

Runs the same model + the same synthetic prompt twice — once with offload OFF and
once with offload ON — and reports (a) whether the generated token ids are
bit-identical (offload is a pure CPU<->GPU relocation of fp16 KV, so they MUST
match) and (b) the peak GPU memory for each mode.

No LongBench data or tokenizer is required (synthetic random input ids), so this
is the fast inner-loop validator for the offload work.

Example (GPU1 via CUDA_VISIBLE_DEVICES=1):

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
    /home/tzj/anaconda3/envs/kitty/bin/python tools/probe_kv_offload.py \
    --model /home/tzj/models/Qwen3-8B --context-len 2048 --max-new-tokens 8

  # memory win at long context:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
    /home/tzj/anaconda3/envs/kitty/bin/python tools/probe_kv_offload.py \
    --model /home/tzj/models/Qwen3-8B --context-len 131072 --max-new-tokens 4
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM

from kitty_sim import get_kvcache_kitty


def _build_cache(args, offloading: bool, max_length: int):
    ns = argparse.Namespace(
        sink_length=args.sink_length,
        buffer_length=args.buffer_length,
        group_size=args.group_size,
        kbits=args.kbits,
        vbits=args.vbits,
        promote_ratio=args.promote_ratio,
        promote_bit=args.promote_bit,
        channel_selection=args.channel_selection,
        offloading=offloading,
        max_length=max_length,
        resident_layers=args.resident_layers,
        offload_prefetch=offloading and args.prefetch,
    )
    return get_kvcache_kitty(ns)


def _run_once(model, input_ids, args, *, offloading: bool, max_length: int):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cache = _build_cache(args, offloading, max_length)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            past_key_values=cache,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=(model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else 0),
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    gen = out[0, input_ids.shape[-1]:].detach().cpu().tolist()
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_resv = torch.cuda.max_memory_reserved() / 2**30
    seq_len = cache.get_seq_length()
    del cache, out
    torch.cuda.empty_cache()
    return {
        "gen": gen,
        "peak_alloc": peak_alloc,
        "peak_resv": peak_resv,
        "seconds": dt,
        "seq_len": seq_len,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--context-len", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--resident-layers", type=int, default=2)
    p.add_argument("--prefetch", action="store_true",
                   help="double-buffered prefetch (overlap next-layer H2D with compute)")
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
    p.add_argument("--modes", default="off,on", help="comma list subset of off,on")
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    assert torch.cuda.is_available(), "CUDA required"
    print(f"device: {torch.cuda.get_device_name(0)}  dtype={args.dtype}")

    print(f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=True,
    ).to("cuda").eval()

    vocab = int(model.config.vocab_size)
    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, vocab, (1, args.context_len), device="cuda")
    max_length = args.context_len + args.max_new_tokens + 8

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = {}
    for mode in modes:
        offloading = mode == "on"
        print(f"\n=== mode={mode} (offloading={offloading}) ctx={args.context_len} "
              f"gen={args.max_new_tokens} resident_layers={args.resident_layers} "
              f"prefetch={offloading and args.prefetch} ===")
        r = _run_once(model, input_ids, args, offloading=offloading, max_length=max_length)
        results[mode] = r
        print(f"  peak_alloc={r['peak_alloc']:.2f} GiB  peak_reserved={r['peak_resv']:.2f} GiB  "
              f"seq_len={r['seq_len']}  time={r['seconds']:.2f}s")
        print(f"  gen_ids[:16]={r['gen'][:16]}")

    if "off" in results and "on" in results:
        match = results["off"]["gen"] == results["on"]["gen"]
        d_alloc = results["off"]["peak_alloc"] - results["on"]["peak_alloc"]
        d_resv = results["off"]["peak_resv"] - results["on"]["peak_resv"]
        print("\n=== SUMMARY ===")
        print(f"  generated ids identical: {match}")
        print(f"  peak_alloc   off={results['off']['peak_alloc']:.2f}  on={results['on']['peak_alloc']:.2f}  "
              f"saved={d_alloc:.2f} GiB")
        print(f"  peak_reserved off={results['off']['peak_resv']:.2f}  on={results['on']['peak_resv']:.2f}  "
              f"saved={d_resv:.2f} GiB")
        print(f"  decode slowdown (on/off time): {results['on']['seconds'] / max(results['off']['seconds'], 1e-9):.2f}x")
        print("RESULT:", "PASS" if match else "FAIL-MISMATCH")


if __name__ == "__main__":
    main()
