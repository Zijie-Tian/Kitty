# -*- coding: utf-8 -*-
"""Microbenchmark for qlutattn-k168v4-pt (_quant_k_pertoken, pertoken_mixed).
Isolates prefill (one big call) vs decode (256 single-token calls) cost, using
the real Llama-3.2-1B per-layer shapes (nh=8 KV heads, D=64), x16 layers/sample.
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/bench_k168_pertoken.py
"""
import sys, time
sys.path.insert(0, "src")
import torch
from kitty_sim.kitty_simulate import KittyKVCache, KittyKVCacheConfig

DEV = "cuda:0"
B, NH, D = 1, 8, 64          # Llama-3.2-1B: 8 KV heads, head_dim 64
NLAYERS = 16
POLICY = ["sign", "sign", "sign", "tern", "nf2", "nf2"]


def make_cache():
    cfg = KittyKVCacheConfig(
        sink_length=32, buffer_length=128, group_size=128, kbits=2, vbits=4,
        promote_ratio=0.0, channel_selection=0, k_quant_mode="per_token",
        k_codebook="qlut", bin_codebooks=POLICY, n_bins=len(POLICY),
        pertoken_mixed=True)
    return KittyKVCache(cfg)


def bench(Tpre, ndec=256, reps=3):
    cache = make_cache()
    torch.manual_seed(0)
    # prefill: one big call over the whole quant region
    pre_t = []
    for r in range(reps):
        cache.k_pc_mean.clear(); cache.k_mix_bins.clear()
        ks = torch.randn(B, NH, Tpre, D, device=DEV, dtype=torch.float16)
        torch.cuda.synchronize(); t = time.time()
        cache._quant_k_pertoken(ks, 0)
        torch.cuda.synchronize(); pre_t.append(time.time() - t)
    # decode: ndec single-token calls (bins cached from prefill above)
    ks1 = torch.randn(B, NH, 1, D, device=DEV, dtype=torch.float16)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(ndec):
        cache._quant_k_pertoken(ks1, 0)
    torch.cuda.synchronize(); dec_t = time.time() - t
    return min(pre_t), dec_t


if __name__ == "__main__":
    print(f"device={torch.cuda.get_device_name(0)}  policy={POLICY}")
    bench(2000, 16)  # warmup
    print(f"{'Tpre':>7} {'prefill/layer':>14} {'decode/step':>12} "
          f"{'==> per-sample (x16 layers, 256 dec)':>38}")
    for Tpre in [4000, 8000, 16000, 31000]:
        pre, dec = bench(Tpre, 256)
        per_sample = (pre + dec) * NLAYERS  # 1 prefill + 256 decode, per layer, x16
        print(f"{Tpre:>7} {pre*1000:>11.1f}ms {dec/256*1000:>10.2f}ms "
              f"{per_sample:>34.1f}s")
