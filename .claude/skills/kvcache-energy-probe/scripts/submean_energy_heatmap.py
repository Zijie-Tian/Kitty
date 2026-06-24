#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单条 ~32k 真实 LongBench 文档上的 KV-cache 减均值能量分解热力图。

与 submean_energy.py 相同的统计口径（per-channel 取向、沿 token 轴 G=128 连续分组、
population ddof=0，故 E[x^2] = mu^2 + sigma^2 精确成立），但：
  1) 输入是「一条」真实样本（逐数据集扫描，取第一条 tokenize 后 >= seq_len 的
     context，截前 seq_len 个 token），不做多文档拼接；
  2) 统计粒度细化到 (layer, head, dim)，并归并出两个视角：
         layer x head（每 head 的 D 个通道汇总）
         layer x dim （每通道维在 H 个 head 上汇总，可见 RoPE 频率结构）
  3) 用以 0.5 为中心的发散色图绘制 mu^2 能量占比热力图：
     红 = mu^2 主导（token 轴均值携带能量多），蓝 = sigma^2 主导。
     因占比互补（share_mu + share_sigma = 1），一张图同时表达两者。

用法：
  CUDA_VISIBLE_DEVICES=1 python submean_energy_heatmap.py \
      --model MODEL --seq-len 32768 --group 128 --tag TAG
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import _common as c


@torch.inference_mode()
def channel_stats(t, G):
    """t: [1, H, T, D] -> (sum_mu2, sum_var, sum_ex2)，各 [H, D]（沿 token 轴 G 一组）。"""
    gs = c.group_decompose(t, G)
    return ((gs.mu * gs.mu).sum(-1).cpu(),
            gs.var.clamp_min(0).sum(-1).cpu(),
            gs.ex2.sum(-1).cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地路径 / HF repo id / 别名 (见 SKILL.md)")
    ap.add_argument("--longbench-dir", default=None, help="LongBench data 目录；缺省读 KVPROBE_LONGBENCH_DIR / LONGBENCH_DATA_ROOT")
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--outdir", default="probe_out")
    args = ap.parse_args()

    dev = "cuda:0"
    model_path = c.resolve_model_path(args.model)
    lb_dir = c.resolve_longbench_dir(args.longbench_dir)

    model, tok, dims = c.load_model_and_tok(model_path, dev)
    ids, src, full_len = c.pick_single_doc(lb_dir, tok, args.seq_len)
    print(f"[input] single doc {src}: {full_len} tokens total, using first {ids.shape[1]} "
          f"(G={args.group} -> {ids.shape[1] // args.group} groups/channel)")
    print(f"[model] {model_path}  layers={dims.nl} kv_heads={dims.H}")

    cache = c.prefill(model, ids, dev, chunk=args.chunk)
    torch.cuda.synchronize()
    print(f"[prefill] done, max_mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    nl = c.cache_n_layers(cache)
    stats = {}  # kv -> dict(mu2/var/ex2: [L, H, D])
    for kv in ("K", "V"):
        mu2s, vars_, ex2s = [], [], []
        for li in range(nl):
            k, v = c.cache_layer_kv(cache, li)
            m, va, e = channel_stats(k if kv == "K" else v, args.group)
            mu2s.append(m); vars_.append(va); ex2s.append(e)
        stats[kv] = {
            "mu2": torch.stack(mu2s),   # [L, H, D]
            "var": torch.stack(vars_),
            "ex2": torch.stack(ex2s),
        }
        share = stats[kv]["mu2"].sum() / stats[kv]["ex2"].sum()
        print(f"[{kv}] overall mu^2 share = {share * 100:.2f}%")

    os.makedirs(args.outdir, exist_ok=True)
    torch.save({"stats": stats, "model": model_path, "src": src,
                "seq_len": ids.shape[1], "group": args.group},
               os.path.join(args.outdir, f"submean_heatmap_{args.tag}.pt"))

    # ---------- 绘图：红 = mu^2 主导, 蓝 = sigma^2 主导（share 互补，一图两义） ----------
    L = nl
    fig, axes = plt.subplots(2, 2, figsize=(15, 4 + 0.42 * L),
                             gridspec_kw={"width_ratios": [1, 2.2]})
    cmap = c.CMAP_SHARE
    for row, kv in enumerate(("K", "V")):
        mu2, ex2 = stats[kv]["mu2"], stats[kv]["ex2"]
        by_head = (mu2.sum(-1) / ex2.sum(-1)).numpy()   # [L, H]
        by_dim = (mu2.sum(1) / ex2.sum(1)).numpy()      # [L, D]

        ax = axes[row][0]
        im = ax.imshow(by_head, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{kv} cache: layer x kv-head")
        ax.set_xlabel("kv head"); ax.set_ylabel("layer")
        ax.set_xticks(range(by_head.shape[1]))
        if L <= 20:  # 小模型逐格标数
            for i in range(by_head.shape[0]):
                for j in range(by_head.shape[1]):
                    val = by_head[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(val - 0.5) > 0.3 else "black")

        ax = axes[row][1]
        im = ax.imshow(by_dim, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{kv} cache: layer x channel dim (RoPE pair order)")
        ax.set_xlabel("head_dim channel index"); ax.set_ylabel("layer")

    # 顶部为标题预留的比例随图高自适应（高图标题占比更小，避免压住子图）
    fig_h = 4 + 0.42 * L
    top = 1.0 - 1.1 / fig_h
    fig.suptitle(
        f"$\\mu^2$ energy share of token-axis groups (G={args.group})  —  "
        f"red: $\\mu^2$ dominant, blue: $\\sigma^2$ dominant  "
        f"($\\mathrm{{share}}_{{\\mu^2}}+\\mathrm{{share}}_{{\\sigma^2}}=1$)\n"
        f"{c.model_basename(model_path)}, single LongBench doc {src}, "
        f"{ids.shape[1]} tokens, post-RoPE K / V, per-channel token-axis grouping",
        fontsize=11, y=top + 0.5 * (1 - top))
    fig.tight_layout(rect=(0, 0, 0.92, top))
    cax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$\\mu^2$ share  (1 - $\\sigma^2$ share)")
    png = os.path.join(args.outdir, f"submean_heatmap_{args.tag}.png")
    fig.savefig(png, dpi=150)
    print(f"[saved] {png}")


if __name__ == "__main__":
    main()
