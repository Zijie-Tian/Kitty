#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
per-channel mu^2/sigma^2 画像是否任务相关：每个 LongBench subtask 取一条真实样本对比。

如果不同任务（英文叙事/论文 QA/摘要/few-shot 分类/代码/中文）的 channel 画像一致，
则"红/蓝身份"是模型属性 -> sigma^2 通道选择可离线一次校准、任务通用；
若任务相关 -> 必须 per-sample prefix 校准。

对每个任务的样本（取前 max_tokens 个 token，统计口径同前：post-RoPE K、
per-channel 取向、G=128 token 组、population）输出：
  1) 整体 K mu^2 share（标量，看任务间漂移幅度）；
  2) dim 画像 [D]（聚合 layer/head 后每 head_dim 通道的 mu^2 share）
     -> 任务 x dim 热力图 + 任务两两 Pearson 相关；
  3) 全粒度 (L,H,D) 的 sigma^2 选择器：per-head top-12.5% 通道，
     任务两两重合度矩阵 + 每任务 vs 共识（全任务平均 sigma^2）的重合度。
     随机基线 = 12.5%。

用法：
  CUDA_VISIBLE_DEVICES=1 python task_channel_profile.py --model MODEL --tag TAG
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
    ap.add_argument("--tasks", default=c.DEFAULT_TASKS)
    ap.add_argument("--min-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--group", type=int, default=128)
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
    G = args.group
    print(f"[model] layers={nl} kv_heads={H} head_dim={D}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    names, n_toks = [], []
    dim_profile = []                      # 每任务 [D] mu^2 share
    overall = []                          # 每任务标量 share
    sig2_full = []                        # 每任务 [nl,H,D] 平均 sigma^2（选择器用）

    for task in tasks:
        ids, ln, full = c.pick_task_sample(lb_dir, task, tok, args.min_tokens, args.max_tokens)
        if ids is None:
            print(f"[skip] {task}: no sample >= {args.min_tokens} tokens")
            continue
        cache = c.prefill(model, ids, dev, chunk=args.chunk)
        mu2_d = torch.zeros(D)
        ex2_d = torch.zeros(D)
        s2f = torch.zeros(nl, H, D)
        with torch.inference_mode():
            for li in range(nl):
                K = c.cache_layer_k(cache, li)
                gs = c.group_decompose(K, G)
                sig2 = gs.var.clamp_min(0)
                mu2_d += (gs.mu * gs.mu).mean(-1).sum(0).cpu()    # 各组等权,够用
                ex2_d += gs.ex2.mean(-1).sum(0).cpu()
                s2f[li] = sig2.mean(-1).cpu()
        del cache
        torch.cuda.empty_cache()
        names.append(task)
        n_toks.append(ids.shape[1])
        dim_profile.append((mu2_d / ex2_d).numpy())
        overall.append((mu2_d.sum() / ex2_d.sum()).item())
        sig2_full.append(s2f)
        print(f"[{task:18s}] line#{ln} tokens={ids.shape[1]:5d} (full {full:6d})  "
              f"K mu^2 share = {overall[-1]*100:5.2f}%")

    nT = len(names)
    P = np.stack(dim_profile)                              # [nT, D]

    # ---- dim 画像两两相关 ----
    corr = np.corrcoef(P)
    print("\n[dim-profile pairwise Pearson r]")
    print(f"  min={corr[np.triu_indices(nT, 1)].min():.3f}  "
          f"mean={corr[np.triu_indices(nT, 1)].mean():.3f}")

    # ---- 选择器跨任务重合 ----
    k = max(1, int(args.promote_frac * D))
    tops = [c.topk_indices_per_head(s, k) for s in sig2_full]  # each [nl,H,k]

    M = np.eye(nT)
    for i in range(nT):
        for j in range(i + 1, nT):
            M[i, j] = M[j, i] = c.overlap_masks(tops[i], tops[j], D, k)
    consensus = c.topk_indices_per_head(torch.stack(sig2_full).mean(0), k)
    vs_cons = [c.overlap_masks(t, consensus, D, k) for t in tops]
    off = M[np.triu_indices(nT, 1)]
    print(f"\n[selector cross-task overlap: per-head top-{k}/{D} by sigma^2 "
          f"(random baseline {k/D*100:.1f}%)]")
    print(f"  pairwise: min={off.min()*100:.1f}%  mean={off.mean()*100:.1f}%  max={off.max()*100:.1f}%")
    print("  task vs consensus: " + "  ".join(
        f"{names[i]}:{vs_cons[i]*100:.0f}%" for i in range(nT)))

    # ---- 图 ----
    fig = plt.figure(figsize=(19, 5.2 + 0.12 * nT))
    gs_fig = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.05, 0.85])

    ax = fig.add_subplot(gs_fig[0])
    im = c.share_heatmap(ax, P, xlabel="head_dim channel index",
                         title="(A) per-dim $\\mu^2$ share, one sample per task\n"
                               "(identical vertical bands = task-independent)",
                         extent=(0, D, nT - 0.5, -0.5))
    ax.set_yticks(range(nT)); ax.set_yticklabels(names, fontsize=9)
    fig.colorbar(im, ax=ax, label="$\\mu^2$ share")

    ax = fig.add_subplot(gs_fig[1])
    im = ax.imshow(M * 100, cmap="viridis", vmin=50, vmax=100)
    ax.set_xticks(range(nT)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(nT)); ax.set_yticklabels(names, fontsize=8)
    for i in range(nT):
        for j in range(nT):
            ax.text(j, i, f"{M[i, j]*100:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < 0.85 else "black")
    ax.set_title(f"(B) cross-task selector overlap (%)\n"
                 f"top-{k}/{D} $\\sigma^2$ channels per head; chance={k/D*100:.0f}%")

    ax = fig.add_subplot(gs_fig[2])
    ax.barh(range(nT), [v * 100 for v in overall], color="C3", alpha=0.8)
    ax.set_yticks(range(nT)); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("overall K $\\mu^2$ share (%)")
    ax.set_title("(C) overall share by task")
    ax.grid(alpha=0.3, axis="x")
    for i, (v, n) in enumerate(zip(overall, n_toks)):
        ax.text(v * 100 + 0.5, i, f"{v*100:.1f} ({n//1024}k)", va="center", fontsize=8)

    fig.suptitle(
        f"Task dependence of per-channel $\\mu^2$/$\\sigma^2$ identity  —  "
        f"{c.model_basename(model_path)}, one real LongBench sample per task, G={G}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    c.save_fig(fig, args.outdir, f"task_channel_{args.tag}.png")


if __name__ == "__main__":
    main()
