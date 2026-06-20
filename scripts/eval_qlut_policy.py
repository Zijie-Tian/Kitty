#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoresearch verify harness: given a sigma^2-binned codebook policy
(configs/qlut_policy.json), load the cached post-RoPE K/Q
(probe_out/kq_cache_<tag>.pt), apply per-bin codebooks, and report:
  - eff_bits : average KV bits/element under the policy (THE metric, lower better)
  - overlap  : top-32 attended-token overlap vs fp16 K, real last-N queries (THE guard)
Writes probe_out/qlut_policy_eval.json.

Policy JSON: {"group_size":128, "bin_codebooks":["meanonly","sign",...]}  (len=n_bins)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_qlut_policy.py \
      --cache probe_out/kq_cache_llama32-1b.pt --policy configs/qlut_policy.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kitty_sim.qlut_quant import (apply_qlut, channel_sigma2, codebook_bits,
                                   effective_bits, sigma2_bins)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="probe_out/kq_cache_llama32-1b.pt")
    ap.add_argument("--policy", default="configs/qlut_policy.json")
    ap.add_argument("--out", default="probe_out/qlut_policy_eval.json")
    args = ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    art = torch.load(args.cache, map_location=dev, weights_only=False)
    meta = art["meta"]
    G, sink, recent, topk = meta["group"], meta["sink"], meta["recent"], meta["topk"]
    pol = json.load(open(args.policy))
    bin_codebooks = pol["bin_codebooks"]
    n_bins = len(bin_codebooks)

    nl = meta["nl"]
    ov_sum, ov_cnt, bits_sum = 0.0, 0, 0.0
    bin_counts = torch.zeros(n_bins)
    for li in range(nl):
        x = art["k_post"][li].to(dev).float()
        q = art["q_real"][li].to(dev).float()
        T = x.shape[-1]
        sig2 = channel_sigma2(x[:, :, sink:T - recent], G)
        binid = sigma2_bins(sig2, n_bins)
        bits_sum += effective_bits(bin_codebooks, binid, G)
        bin_counts += torch.bincount(binid.reshape(-1).cpu(), minlength=n_bins).float()
        xr = apply_qlut(x, bin_codebooks, binid, sink, recent, G)
        s_true = torch.einsum("hqd,hdt->hqt", q, x)
        s_rec = torch.einsum("hqd,hdt->hqt", q, xr)
        t1 = torch.zeros_like(s_true, dtype=torch.bool).scatter(
            -1, s_true.topk(topk, dim=-1).indices, True)
        t2 = torch.zeros_like(s_true, dtype=torch.bool).scatter(
            -1, s_rec.topk(topk, dim=-1).indices, True)
        ov_sum += ((t1 & t2).sum(-1).float() / topk).mean().item()
        ov_cnt += 1

    overlap = ov_sum / ov_cnt
    eff_bits = bits_sum / nl
    frac = (bin_counts / bin_counts.sum()).tolist()
    per_bin = [{"bin": b, "codebook": cb, "bits": round(codebook_bits(cb, G), 4),
                "chan_frac": round(frac[b], 4)} for b, cb in enumerate(bin_codebooks)]
    result = {"eff_bits": round(eff_bits, 5), "overlap": round(overlap, 5),
              "bin_codebooks": bin_codebooks, "per_bin": per_bin,
              "cache": os.path.basename(args.cache), "src": meta["src"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"eff_bits={eff_bits:.5f}  overlap={overlap:.5f}  bins=[" +
          ",".join(bin_codebooks) + "]")


if __name__ == "__main__":
    main()
