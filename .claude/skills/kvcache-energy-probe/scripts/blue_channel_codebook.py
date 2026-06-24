#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高 sigma^2（蓝）K channel 的最优低-bit 码本对决，按蓝通道子类型拆分。

前轮发现：蓝通道（mu^2 占比<0.5）的 sigma^2 有两种来源——
  - 旋转型(rot)：post/pre-RoPE sigma^2 比值大，sigma^2 主要是 RoPE 快旋转的确定性伪影；
  - 重尾型(tail)：比值≈1，pre 空间也方差大，是真实内容（massive 通道，ρ 小、重尾）。
本探针把蓝 pair（RoPE pair 同治）按 post/pre 比值拆成 rot/tail 两类，在 1 / 1.6 / 2 bit
三档对比 6 种码本，回答"sigma^2 大的 channel 低 bit 用什么"：

  码本（每种都在 post 空间与 de-RoPE(pre)空间各做一遍）：
    sign  : submean + 1-bit sign                         1.25 b
    tern  : submean + 死区三值(τ=0.5)                     1.83 b
    mm2   : 非对称 min-max 2-bit（均匀）                  2.25 b
  de-RoPE 版：逆旋转到 pre-RoPE -> 同码本 -> 旋回（旋转等距，零额外 bit）。

指标：各子类型的 K NMSE（post 空间真值）。判定逻辑：
  - rot-blue：de-RoPE 应大幅降误差，且 de-RoPE-sign(1.25b) 有望优于 post-mm2(2.25b)
    （结构胜位宽）；
  - tail-blue：de-RoPE 收益小，需靠 bit（tern/mm2）——此即"第三类"通道。

用法：
  CUDA_VISIBLE_DEVICES=1 python blue_channel_codebook.py --model MODEL --tag TAG
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
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--sink", type=int, default=32)
    ap.add_argument("--recent", type=int, default=128)
    ap.add_argument("--rot-ratio", type=float, default=2.0, help="post/pre sigma^2 比值阈值")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--outdir", default="probe_out")
    args = ap.parse_args()

    dev = "cuda:0"
    model_path = c.resolve_model_path(args.model)
    lb_dir = c.resolve_longbench_dir(args.longbench_dir)

    model, tok, dims = c.load_model_and_tok(model_path, dev)
    H, D, nl = dims.H, dims.D, dims.nl
    G = args.group
    half = D // 2
    ids, src, full = c.pick_single_doc(lb_dir, tok, args.seq_len)
    print(f"[input] {src}: using {ids.shape[1]} tok (G={G})")
    print(f"[model] layers={nl} kv_heads={H} head_dim={D}")

    T = ids.shape[1]
    cache = c.prefill(model, ids, dev, chunk=args.chunk)
    print(f"[prefill] max_mem={torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

    cosT, sinT = c.rope_tables(model, T, D, dev)                 # [D/2, T]

    KINDS = ["sign", "tern", "mm2"]
    bits = {"sign": 1 + 32 / G, "tern": np.log2(3) + 32 / G, "mm2": 2 + 32 / G}
    # 累加 SSE / energy，按子类 {rot, tail, red}
    classes = ["rot", "tail", "red"]
    sse = {f"{sp}_{k}": {cl: 0.0 for cl in classes}
           for sp in ("post", "pre") for k in KINDS}
    eng = {cl: 0.0 for cl in classes}
    cnt = {cl: 0 for cl in classes}

    q0, q1 = args.sink, T - args.recent
    with torch.inference_mode():
        for li in range(nl):
            K = c.cache_layer_k(cache, li)
            x_post = K[0].permute(0, 2, 1).float()
            x_pre = c.rope_rotate(x_post, cosT, sinT, inverse=True)
            xqp = x_post[:, :, q0:q1]
            xqr = x_pre[:, :, q0:q1]
            ng = (q1 - q0) // G

            # pair 身份 + 旋转比值（量化区）
            segp = xqp[:, :, :ng * G].reshape(H, D, ng, G)
            segr = xqr[:, :, :ng * G].reshape(H, D, ng, G)
            sig2_post = (segp - segp.mean(-1, keepdim=True)).pow(2).mean(dim=(-1, -2))   # [H,D]
            sig2_pre = (segr - segr.mean(-1, keepdim=True)).pow(2).mean(dim=(-1, -2))
            ex2 = (segp * segp).mean(dim=(-1, -2))
            mu2 = segp.mean(-1).pow(2).mean(-1)
            iden_pair = (mu2[:, :half] + mu2[:, half:]) / (ex2[:, :half] + ex2[:, half:]).clamp_min(1e-12)
            ratio_pair = (sig2_post[:, :half] + sig2_post[:, half:]) / \
                         (sig2_pre[:, :half] + sig2_pre[:, half:]).clamp_min(1e-12)
            blue_pair = iden_pair < 0.5
            rot_pair = blue_pair & (ratio_pair > args.rot_ratio)
            tail_pair = blue_pair & (ratio_pair <= args.rot_ratio)
            red_pair = ~blue_pair
            cls_pair = {"rot": rot_pair, "tail": tail_pair, "red": red_pair}     # [H,half]
            cls_chan = {cl: torch.cat([m, m], dim=1) for cl, m in cls_pair.items()}  # [H,D]

            e_tot = (xqp * xqp).sum(-1)                                            # [H,D]
            for cl in classes:
                eng[cl] += e_tot[cls_chan[cl]].sum().item()
                cnt[cl] += int(cls_chan[cl].sum().item())

            for k in KINDS:
                rp = c.submean_codebook(xqp, G, k)
                rr = c.rope_rotate(c.submean_codebook(xqr, G, k), cosT[:, q0:q1], sinT[:, q0:q1])
                ep = (rp - xqp).pow(2).sum(-1)                                     # [H,D]
                er = (rr - xqp).pow(2).sum(-1)
                for cl in classes:
                    sse[f"post_{k}"][cl] += ep[cls_chan[cl]].sum().item()
                    sse[f"pre_{k}"][cl] += er[cls_chan[cl]].sum().item()
    del cache
    torch.cuda.empty_cache()

    tot = sum(cnt.values())
    print(f"\n[channel mix] rot-blue {cnt['rot']/tot*100:.0f}%  tail-blue {cnt['tail']/tot*100:.0f}%  "
          f"red {cnt['red']/tot*100:.0f}%  (rot threshold post/pre>{args.rot_ratio})")
    print(f"\n{'codebook':>14} {'bits':>6} {'rot-blue':>9} {'tail-blue':>10} {'red':>8}")
    order = [("post", "sign"), ("pre", "sign"), ("post", "tern"), ("pre", "tern"),
             ("post", "mm2"), ("pre", "mm2")]
    res = {}
    for sp, k in order:
        name = f"{'deRoPE' if sp == 'pre' else 'post'}-{k}"
        nm = {cl: sse[f"{sp}_{k}"][cl] / max(eng[cl], 1e-9) for cl in classes}
        res[name] = (bits[k], nm)
        print(f"{name:>14} {bits[k]:>6.2f} {nm['rot']:>9.4f} {nm['tail']:>10.4f} {nm['red']:>8.4f}")

    print(f"\n[verdict] rot-blue: deRoPE-sign(1.25b)={res['deRoPE-sign'][1]['rot']:.4f} "
          f"vs post-mm2(2.25b)={res['post-mm2'][1]['rot']:.4f} "
          f"-> {'结构胜位宽 ✓' if res['deRoPE-sign'][1]['rot'] < res['post-mm2'][1]['rot'] else '需位宽'}")
    print(f"[verdict] tail-blue: deRoPE-sign={res['deRoPE-sign'][1]['tail']:.4f} "
          f"vs post-sign={res['post-sign'][1]['tail']:.4f} "
          f"(de-RoPE 增益 {res['post-sign'][1]['tail']/max(res['deRoPE-sign'][1]['tail'],1e-9):.2f}x); "
          f"post-tern={res['post-tern'][1]['tail']:.4f} mm2={res['post-mm2'][1]['tail']:.4f}")

    # ---- 图 ----
    names = list(res.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, cl, ttl in zip(axes, ["rot", "tail", "red"],
                           ["(A) rot-blue: $\\sigma^2$=RoPE artifact",
                            "(B) tail-blue: $\\sigma^2$=genuine heavy-tail",
                            "(C) red (reference)"]):
        vals = [res[n][1][cl] for n in names]
        bb = [res[n][0] for n in names]
        cols = ["C0" if n.startswith("post") else "C1" for n in names]
        ax.bar(range(len(names)), vals, color=cols)
        for i, (v, b) in enumerate(zip(vals, bb)):
            ax.text(i, v, f"{v:.3f}\n{b:.2f}b", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("class K NMSE"); ax.set_title(ttl)
        ax.grid(alpha=0.3, axis="y")
        ax.margins(y=0.18)
    fig.suptitle(
        f"Low-bit codebook for high-σ² (blue) K channels  —  {c.model_basename(model_path)}, "
        f"{src}, {T} tok  |  blue: rot {cnt['rot']/tot*100:.0f}% + tail {cnt['tail']/tot*100:.0f}%  "
        f"(C0=post-RoPE, C1=de-RoPE)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    c.save_fig(fig, args.outdir, f"blue_codebook_{args.tag}.png")


if __name__ == "__main__":
    main()
