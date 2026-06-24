#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分型（typed）超低比特 K 量化：mu^2 主导通道用 kitty_sign，sigma^2 主导通道换方法。

设计依据（本系列前几轮实测）：
  - 红通道（mu^2 主导）：残差小，post-RoPE submean+sign 已是 1-bit Lloyd–Max 最优。
  - 蓝通道（sigma^2 主导）= 快频 RoPE 对：其 sigma^2 的大部分是已知确定性旋转的
    人为产物（post/pre sigma^2 实测 6-8.5x）。旋转相位 theta(t) 由位置唯一决定，
    可在 dequant 时精确重放 -> 把这部分"伪方差"从量化问题里确定性地除掉：
        de-RoPE: k_pre = R(-theta_t) k_post（per RoPE pair 的 2x2 旋转，等距）
        在 pre 空间 submean+sign（同样 1 bit + 同样 2 fp16 侧信息）
        dequant: rec_post = R(theta_t) rec_pre（误差范数不变 = pre 空间误差）
    旋转等距 => post 误差 = pre 误差 = sigma^2_pre (1-rho_pre^2) << post-sign 误差。

对比方案（K only；sink32 + recent128 保 fp16，与 Kitty 窗口一致；V 不动）：
  sign_post   全通道 post-RoPE submean+sign            1.25 b
  tern_post   全通道 post-RoPE submean+tern(τ=0.5)     1.83 b
  typed       蓝 pair -> de-RoPE sign；红 -> post sign  1.25 b（等比特升级！）
  derope_all  全通道 de-RoPE sign                       1.25 b（dequant 全部要旋转）
  typed0      蓝 pair -> de-RoPE 仅存组均值(0 bit/token)；红 -> post sign  ~sub-1b
  typed_orc   每 pair 取 {post-sign, de-RoPE-sign} 误差较小者（oracle 上界） 1.25 b

分型规则：RoPE pair (i, i+D/2) 必须同治（de-RoPE 耦合两维）；pair 身份 =
两维合并的 mu^2 能量占比，蓝 = identity < 0.5（静态、任务无关，前轮已证）。

指标：
  1) K cache NMSE（post 空间），整体 + 分蓝/红类；
  2) top-32 attention overlap（真实最后 256 query，与 kitty_sign 笔记预筛同口径）。

用法：
  CUDA_VISIBLE_DEVICES=1 python typed_channel_quant.py --model MODEL --tag TAG
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
    ap.add_argument("--num-queries", type=int, default=256)
    ap.add_argument("--sink", type=int, default=32)
    ap.add_argument("--recent", type=int, default=128)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--outdir", default="probe_out")
    args = ap.parse_args()

    dev = "cuda:0"
    model_path = c.resolve_model_path(args.model)
    lb_dir = c.resolve_longbench_dir(args.longbench_dir)

    model, tok, dims = c.load_model_and_tok(model_path, dev)
    H, nq_heads, n_rep, D, nl = dims.H, dims.n_q_heads, dims.n_rep, dims.D, dims.nl
    G = args.group
    NQ = args.num_queries
    ids, src, full = c.pick_single_doc(lb_dir, tok, args.seq_len)
    print(f"[input] {src}: full {full} tok, using {ids.shape[1]}  (G={G})")
    print(f"[model] layers={nl} kv_heads={H} head_dim={D} n_rep={n_rep}")

    # prefill + 最后 chunk 抓 q
    qcap, hook_factory = c.gather_q(model)
    T = ids.shape[1]
    cache = c.prefill(model, ids, dev, chunk=args.chunk, hook_factory=hook_factory)
    print(f"[prefill] max_mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    # 全位置 RoPE 表（pair 一半即可）与 query 位置表
    cosT, sinT = c.rope_tables(model, T, D, dev)                 # [D/2, T]
    cosQ, sinQ = c.rope_tables_for_positions(model, T - NQ, T, D, dev)

    SCHEMES = ["sign_post", "tern_post", "typed", "derope_all", "typed0", "typed_orc",
               "typed_tern", "tern_pre_all", "typed_ptm", "ptm_all"]
    sse = {k: 0.0 for k in SCHEMES}                          # 量化区 SSE（post 空间）
    sse_cls = {k: {"blue": 0.0, "red": 0.0} for k in SCHEMES}
    eng = 0.0
    eng_cls = {"blue": 0.0, "red": 0.0}
    ov = {k: [] for k in SCHEMES}
    f_blue_pairs = []

    NQS = 128                                                # 每 kv-head 参与 overlap 的 query 数
    with torch.inference_mode():
        for li in range(nl):
            K = c.cache_layer_k(cache, li)
            x_post = K[0].permute(0, 2, 1).float()           # [H, D, T]
            x_pre = c.rope_rotate(x_post, cosT, sinT, inverse=True)

            # ---- pair 身份（post 空间, 量化区） ----
            q0, q1 = args.sink, T - args.recent              # 量化区 [q0, q1)
            ng = (q1 - q0) // G
            seg = x_post[:, :, q0:q0 + ng * G].reshape(H, D, ng, G)
            mu_g = seg.mean(-1)
            mu2 = (mu_g * mu_g).mean(-1)                     # [H,D]
            ex2 = (seg * seg).mean(dim=(-1, -2))
            half = D // 2
            pair_iden = (mu2[:, :half] + mu2[:, half:]) / (ex2[:, :half] + ex2[:, half:]).clamp_min(1e-12)
            blue_pair = pair_iden < 0.5                      # [H, D/2]
            blue_mask = torch.cat([blue_pair, blue_pair], dim=1)  # [H, D]
            f_blue_pairs.append(blue_pair.float().mean().item())

            # ---- 各方案重建（仅量化区做量化；sink/recent 保真） ----
            xq_post = x_post[:, :, q0:q1]
            xq_pre = x_pre[:, :, q0:q1]
            rec_sign_post = c.submean_sign(xq_post, G, 0.0)
            rec_tern_post = c.submean_sign(xq_post, G, 0.5)
            rec_pre_sign = c.rope_rotate(c.submean_sign(xq_pre, G, 0.0), cosT[:, q0:q1], sinT[:, q0:q1])
            rec_pre_tern = c.rope_rotate(c.submean_sign(xq_pre, G, 0.5), cosT[:, q0:q1], sinT[:, q0:q1])
            rec_pre_mu = c.rope_rotate(c.groupmean_only(xq_pre, G), cosT[:, q0:q1], sinT[:, q0:q1])
            bm = blue_mask[:, :, None]
            recs = {
                "sign_post": rec_sign_post,
                "tern_post": rec_tern_post,
                "typed": torch.where(bm, rec_pre_sign, rec_sign_post),
                "derope_all": rec_pre_sign,
                "typed0": torch.where(bm, rec_pre_mu, rec_sign_post),
                "typed_tern": torch.where(bm, rec_pre_tern, rec_sign_post),
                "tern_pre_all": rec_pre_tern,
            }
            # phase-tracked mu：mu 存 pre 空间（解析旋转、不走数据通路），残差 sign 码
            # 仍在 post 空间 per-channel 仿射量化 -> int/LUT 点积结构完整保留。
            # 侧信息与 sign_post 相同（每通道每组 mu, m 各 1 fp16），bits 同 1.25。
            rec_ptm = rec_pre_mu + c.submean_sign(xq_post - rec_pre_mu, G, 0.0, submean=False)
            recs["typed_ptm"] = torch.where(bm, rec_ptm, rec_sign_post)
            recs["ptm_all"] = rec_ptm
            # oracle：每 pair 选 {post-sign, de-RoPE-sign} 中 SSE 小者
            e_post = (rec_sign_post - xq_post).pow(2).sum(-1)          # [H, D]
            e_pre = (rec_pre_sign - xq_post).pow(2).sum(-1)
            ep = e_post[:, :half] + e_post[:, half:]
            er = e_pre[:, :half] + e_pre[:, half:]
            orc_pair = er < ep
            orc_mask = torch.cat([orc_pair, orc_pair], dim=1)[:, :, None]
            recs["typed_orc"] = torch.where(orc_mask, rec_pre_sign, rec_sign_post)

            e_tot = (xq_post * xq_post).sum(-1)                        # [H, D]
            eng += e_tot.sum().item()
            eng_cls["blue"] += e_tot[blue_mask].sum().item()
            eng_cls["red"] += e_tot[~blue_mask].sum().item()
            for k, rec in recs.items():
                e = (rec - xq_post).pow(2).sum(-1)                     # [H, D]
                sse[k] += e.sum().item()
                sse_cls[k]["blue"] += e[blue_mask].sum().item()
                sse_cls[k]["red"] += e[~blue_mask].sum().item()

            # ---- top-32 overlap（真实 query；sink/recent 用真值拼接） ----
            attn = model.model.layers[li].self_attn
            q = c.add_q_norm_and_rope(attn, qcap[li][0, -NQ:, :].view(NQ, nq_heads, D), cosQ, sinQ)
            qg = q.reshape(H, n_rep * NQ, D)[:, :NQS]                  # [H, NQS, D]

            s_true = torch.einsum("hqd,hdt->hqt", qg, x_post)
            top_true = s_true.topk(args.topk, dim=-1).indices
            t1 = torch.zeros_like(s_true, dtype=torch.bool).scatter(-1, top_true, True)
            for k, rec in recs.items():
                kf = x_post.clone()
                kf[:, :, q0:q1] = rec
                s_rec = torch.einsum("hqd,hdt->hqt", qg, kf)
                top_rec = s_rec.topk(args.topk, dim=-1).indices
                t2 = torch.zeros_like(t1).scatter(-1, top_rec, True)
                ov[k].append((t1 & t2).sum(-1).float().mean().item() / args.topk)
            del recs, s_true, kf, s_rec
    del cache
    torch.cuda.empty_cache()

    f_blue = float(np.mean(f_blue_pairs))
    bits = {
        "sign_post": 1 + 32 / G,
        "tern_post": np.log2(3) + 32 / G,
        "typed": 1 + 32 / G,
        "derope_all": 1 + 32 / G,
        "typed0": (1 - f_blue) * 1 + 32 / G,
        "typed_orc": 1 + 32 / G,
        "typed_tern": (1 - f_blue) * 1 + f_blue * np.log2(3) + 32 / G,
        "tern_pre_all": np.log2(3) + 32 / G,
        "typed_ptm": 1 + 32 / G,
        "ptm_all": 1 + 32 / G,
    }
    print(f"\n[blue pair fraction] {f_blue*100:.1f}% of pairs (identity<0.5, pair-coupled)")
    print(f"\n{'scheme':>12} {'bits':>6} {'K NMSE':>9} {'NMSE blue':>10} {'NMSE red':>9} {'top32 ovlp':>10}")
    for k in SCHEMES:
        nm = sse[k] / eng
        nb = sse_cls[k]["blue"] / eng_cls["blue"]
        nr = sse_cls[k]["red"] / eng_cls["red"]
        print(f"{k:>12} {bits[k]:>6.2f} {nm:>9.4f} {nb:>10.4f} {nr:>9.4f} {np.mean(ov[k]):>10.3f}")
    print(f"\n[mechanism] blue-class NMSE: sign_post {sse_cls['sign_post']['blue']/eng_cls['blue']:.4f} "
          f"-> de-RoPE {sse_cls['derope_all']['blue']/eng_cls['blue']:.4f} "
          f"({sse_cls['sign_post']['blue']/max(sse_cls['derope_all']['blue'],1e-9):.1f}x lower)")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    labels = SCHEMES
    colors = ["0.55", "0.75", "C0", "C9", "C2", "C1", "C4", "C5", "C6", "C8"]

    ax = axes[0]
    vals = [sse_cls[k]["blue"] / eng_cls["blue"] for k in labels]
    vals2 = [sse_cls[k]["red"] / eng_cls["red"] for k in labels]
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.2, vals, 0.38, label="blue channels ($\\sigma^2$-dom)", color="C0")
    ax.bar(xpos + 0.2, vals2, 0.38, label="red channels ($\\mu^2$-dom)", color="C3", alpha=0.75)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("class NMSE"); ax.set_title("(A) per-class error: de-RoPE fixes blue")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    vals = [sse[k] / eng for k in labels]
    ax.bar(xpos, vals, color=colors)
    for i, k in enumerate(labels):
        ax.text(i, vals[i], f"{vals[i]:.4f}\n{bits[k]:.2f}b", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("total K NMSE"); ax.set_title("(B) overall error (bits annotated)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    vals = [np.mean(ov[k]) for k in labels]
    ax.bar(xpos, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(min(vals) - 0.05, 1.0)
    ax.set_ylabel(f"top-{args.topk} attended-token overlap")
    ax.set_title("(C) attention fidelity (real last-256 queries)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Typed channel quantization: red$\\to$sign, blue$\\to$de-RoPE sign  —  "
        f"{c.model_basename(model_path)}, {src}, {T} tok, G={G}, "
        f"blue pairs {f_blue*100:.0f}%",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    c.save_fig(fig, args.outdir, f"typed_quant_{args.tag}.png")


if __name__ == "__main__":
    main()
