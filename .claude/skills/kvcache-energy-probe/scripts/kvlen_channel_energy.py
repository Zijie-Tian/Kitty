#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
channel x kv-len 二维视角：每个 K channel 的 mu^2/sigma^2 能量占比沿 kv 位置是否变化。

回答的问题：channel 的"红/蓝身份"（mu^2 主导 or sigma^2 主导）是 channel 的静态属性，
还是随 kv-len 漂移的动态属性？这决定了按 sigma^2 选 channel 做异构量化时，
选择能否静态化（离线/prefix 校准一次），还是必须逐组在线重选。

在一条 ~32k 真实 LongBench 文档 (post-RoPE K) 上输出：
  A) dim x kv-position 热力图：mu^2 share（聚合 layer/head；行=head_dim 通道,
     列=128-token 组），与既有热力图同色规（红=mu^2 主导, 蓝=sigma^2 主导）；
  B) 漂移图：每行减去该 channel 自己的全程均值，看沿 kv-len 的变化量（发散色图,
     色幅按 p99.5 自适应）；
  C) 代表性 channel 曲线：全程最红/中位/最蓝的 dim, share 随 kv 位置的原始+平滑曲线；
  D) 选择器稳定性（全粒度 L x H x D）：把 32k 切 8 段(各 4096 tok), 每段按 sigma^2
     选 per-head top-12.5% channel, 与全局选择/相邻段选择的重合度。重合度高 =>
     静态选择即可。

用法：
  CUDA_VISIBLE_DEVICES=1 python kvlen_channel_energy.py \
      --model MODEL --seq-len 32768 --group 128 --segments 8 --tag TAG
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _common as c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地路径 / HF repo id / 别名 (见 SKILL.md)")
    ap.add_argument("--longbench-dir", default=None, help="LongBench data 目录；缺省读 KVPROBE_LONGBENCH_DIR / LONGBENCH_DATA_ROOT")
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--promote-frac", type=float, default=0.125)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--outdir", default="probe_out")
    args = ap.parse_args()

    dev = "cuda:0"
    model_path = c.resolve_model_path(args.model)
    lb_dir = c.resolve_longbench_dir(args.longbench_dir)

    model, tok, dims = c.load_model_and_tok(model_path, dev)
    H, D, nl = dims.H, dims.D, dims.nl
    ids, src, full = c.pick_single_doc(lb_dir, tok, args.seq_len)
    print(f"[input] {src}: full {full} tok, using {ids.shape[1]}  (G={args.group})")
    print(f"[model] layers={nl} kv_heads={H} head_dim={D}")

    cache = c.prefill(model, ids, dev, chunk=args.chunk)
    torch.cuda.synchronize()
    print(f"[prefill] max_mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    T = ids.shape[1]
    G = args.group
    ng = T // G
    S = args.segments
    gps = ng // S                                   # groups per segment

    mu2_dp = torch.zeros(D, ng)                     # sum over layers+heads
    ex2_dp = torch.zeros(D, ng)
    sig2_seg = torch.zeros(nl, H, D, S)             # 全粒度选择器用
    sig2_glob = torch.zeros(nl, H, D)
    with torch.inference_mode():
        for li in range(nl):
            K = c.cache_layer_k(cache, li)
            gs = c.group_decompose(K, G)            # mu/ex2/var: [H,D,ng]
            sig2 = gs.var.clamp_min(0)
            mu2_dp += (gs.mu * gs.mu).sum(0).cpu()
            ex2_dp += gs.ex2.sum(0).cpu()
            sig2_seg[li] = sig2.reshape(H, D, S, gps).mean(-1).cpu()
            sig2_glob[li] = sig2.mean(-1).cpu()

    share = (mu2_dp / ex2_dp).numpy()               # [D, ng]
    anom = share - share.mean(axis=1, keepdims=True)

    # ---- 数值: 每 channel 沿 kv-len 的波动幅度 ----
    std_pos = share.std(axis=1)
    rng_pos = share.max(axis=1) - share.min(axis=1)
    print("\n[per-dim mu^2 share variation along kv-len]  (share in [0,1])")
    print(f"  std  across positions: median={np.median(std_pos):.3f}  p90={np.percentile(std_pos, 90):.3f}  max={std_pos.max():.3f}")
    print(f"  range across positions: median={np.median(rng_pos):.3f}  p90={np.percentile(rng_pos, 90):.3f}  max={rng_pos.max():.3f}")
    drift = []
    gi = np.arange(ng)
    for d in range(D):
        cc = np.corrcoef(gi, share[d])[0, 1]
        drift.append(cc)
    drift = np.array(drift)
    n_drift = int(((np.abs(drift) > 0.5) & (rng_pos > 0.10)).sum())
    print(f"  dims with |corr(pos)|>0.5 AND range>0.10: {n_drift}/{D}  (真趋势性漂移的通道数)")

    # ---- 选择器稳定性 (全粒度 L,H,D; per-head top-k by sigma^2) ----
    k = max(1, int(args.promote_frac * D))
    glob_top = c.topk_indices_per_head(sig2_glob, k)                # [nl, H, k]
    seg_top = c.topk_indices_per_head(sig2_seg.permute(3, 0, 1, 2), k)  # [S, nl, H, k]

    seg_vs_glob = [c.overlap_masks(seg_top[s], glob_top, D, k) for s in range(S)]
    adj = [c.overlap_masks(seg_top[s], seg_top[s + 1], D, k) for s in range(S - 1)]
    print(f"\n[selector stability: per-head top-{k}/{D} channels by sigma^2  (frac={args.promote_frac})]")
    print("  segment-vs-global overlap: " + "  ".join(f"s{s}:{v*100:.0f}%" for s, v in enumerate(seg_vs_glob)))
    print(f"  mean={np.mean(seg_vs_glob)*100:.1f}%   first-segment-vs-global={seg_vs_glob[0]*100:.1f}%")
    print(f"  adjacent-segment overlap:  mean={np.mean(adj)*100:.1f}%  min={np.min(adj)*100:.1f}%")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.2 + 0.025 * D),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    ext = (0, T, D - 0.5, -0.5)

    ax = axes[0]
    im0 = c.share_heatmap(ax, share, title="(A) $\\mu^2$ share per channel x kv-position",
                          xlabel="kv position (token)", ylabel="head_dim channel index", extent=ext)
    fig.colorbar(im0, ax=ax, label="$\\mu^2$ share")

    ax = axes[1]
    im1, lim = c.divergent_heatmap(ax, anom, title="(B) deviation from each channel's own mean",
                                   xlabel="kv position (token)", ylabel="head_dim channel index", extent=ext)
    fig.colorbar(im1, ax=ax, label="$\\Delta$ share")

    ax = axes[2]
    mean_share = share.mean(axis=1)
    picks = [int(mean_share.argmax()), int(np.argsort(mean_share)[D // 2]), int(mean_share.argmin())]
    names = ["most $\\mu^2$-dominant", "median", "most $\\sigma^2$-dominant"]
    xpos = (np.arange(ng) + 0.5) * G
    w = max(1, ng // 32)
    kern = np.ones(w) / w
    for d, nm, col in zip(picks, names, ("C3", "C2", "C0")):
        ax.plot(xpos, share[d] * 100, color=col, alpha=0.25, lw=0.8)
        ax.plot(xpos, np.convolve(share[d], kern, mode="same") * 100,
                color=col, lw=2, label=f"dim {d} ({nm})")
    ax.set_xlabel("kv position (token)"); ax.set_ylabel("$\\mu^2$ share (%)")
    ax.set_title("(C) representative channels along kv-len")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(
        f"Per-channel $\\mu^2$/$\\sigma^2$ identity along kv-len  —  "
        f"{c.model_basename(model_path)}, {src}, {T} tok, G={G} "
        f"(rows aggregate {nl} layers x {H} heads)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    c.save_fig(fig, args.outdir, f"kvlen_channel_{args.tag}.png")


if __name__ == "__main__":
    main()
