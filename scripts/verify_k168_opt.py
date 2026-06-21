# -*- coding: utf-8 -*-
"""Verify the vectorized decode path of qlutattn-k168v4-pt is numerically
identical to the original per-head extract loop, and measure the decode speedup.
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/verify_k168_opt.py
"""
import sys, time
sys.path.insert(0, "src")
import torch
from kitty_sim.kitty_simulate import KittyKVCache, KittyKVCacheConfig

DEV = "cuda:0"
B, NH, D = 1, 8, 64
POLICY = ["sign", "sign", "sign", "tern", "nf2", "nf2"]


def make():
    cfg = KittyKVCacheConfig(
        sink_length=32, buffer_length=128, group_size=128, kbits=2, vbits=4,
        promote_ratio=0.0, channel_selection=0, k_quant_mode="per_token",
        k_codebook="qlut", bin_codebooks=POLICY, n_bins=len(POLICY),
        pertoken_mixed=True)
    return KittyKVCache(cfg)


def ref_perhead(cache, r, binid, policy):
    """Original per-head x per-bin extract loop (the pre-optimization reference)."""
    nbins = len(policy); _, nh, _, _ = r.shape
    out = torch.empty_like(r)
    for h in range(nh):
        oh = torch.empty_like(r[:, h])
        for bi in range(nbins):
            mask = (binid[h] == bi)
            if not mask.any():
                continue
            oh[:, :, mask] = cache._pure_pt_codebook(r[:, h][:, :, mask], policy[bi])
        out[:, h] = oh
    return out


def vec_decode(cache, r, binid, policy):
    out = torch.zeros_like(r)
    for bi in range(len(policy)):
        m = (binid == bi)
        if not m.any():
            continue
        out = out + cache._pt_codebook_masked(r, m, policy[bi])
    return out


if __name__ == "__main__":
    cache = make()
    torch.manual_seed(0)
    # prefill once to cache per-channel mean + sigma^2 bins for layer 0
    ksp = torch.randn(B, NH, 4000, D, device=DEV, dtype=torch.float16)
    cache._quant_k_pertoken(ksp, 0)
    mu = cache.k_pc_mean[0]; binid = cache.k_mix_bins[0]
    muB = mu[None, :, None, :]

    # ---- numerical equivalence over many random decode tokens ----
    md_total, md_nf2only = 0.0, 0.0
    for _ in range(100):
        ks1 = torch.randn(B, NH, 1, D, device=DEV, dtype=torch.float16)
        r = ks1.float() - muB
        a = vec_decode(cache, r, binid, POLICY)
        b = ref_perhead(cache, r, binid, POLICY)
        md_total = max(md_total, (a - b).abs().max().item())
    # scale of the signal for context
    sig = (ksp.float() - muB).abs().mean().item()
    print(f"signal |r| mean              = {sig:.4f}")
    print(f"decode max|vec - perhead|    = {md_total:.3e}   (over 100 random tokens)")

    # end-to-end through _quant_k_pertoken (uses the new T==1 fast path)
    ks1 = torch.randn(B, NH, 1, D, device=DEV, dtype=torch.float16)
    o_new = cache._quant_k_pertoken(ks1, 0).float()
    r = ks1.float() - muB
    o_ref = (muB + ref_perhead(cache, r, binid, POLICY)).to(torch.float16).float()
    print(f"full _quant_k_pertoken diff  = {(o_new - o_ref).abs().max().item():.3e}")

    # ---- decode speed (new vectorized path) ----
    for _ in range(16):
        cache._quant_k_pertoken(ks1, 0)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(256):
        cache._quant_k_pertoken(ks1, 0)
    torch.cuda.synchronize()
    dms = (time.time() - t) / 256 * 1000
    print(f"\nNEW decode/step = {dms:.2f} ms   (old was ~41 ms -> {41/dms:.1f}x)")
    print(f"est per-sample (16L x [prefill 0.7s + 256 dec]) "
          f"= {(0.043 + dms/1000*256)*16:.1f}s   (old ~169s)")
