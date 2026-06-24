#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QK 分数视角的 channel 重要性 x 能量身份（mu^2/sigma^2）联合分析。

问题：对 attention QK 分数贡献大的 channel，是 mu^2 主导（红）还是 sigma^2 主导（蓝）？

关键理论区分（softmax 平移不变性）：
  score_t = sum_d q_d k_{t,d},  k_{t,d} = mu_d(g) + r_{t,d}
  - 「幅度贡献」 A_d = E|q_d| * E|k_d| ：通道把分数绝对值顶多高。mu 大的红通道天然赢,
    但 q_d*mu_d 在组内对所有 t 是同一常数 -> 不改变 softmax 分布（平移不变）。
  - 「区分度贡献」 B_d = 对 Var_t(score) 的协方差归因 = rowsum(C ∘ M)_d,
      C = Cov_t(k)  (D x D, 可拆 C = C_mu(组均值漂移) + C_res(组内残差)),
      M = E_q[q q^T] (真实 query 的二阶矩)。
    sum_d B_d 恰等于 E_q[Var_t(score)]，是「谁决定 attention 选哪个 token」的精确分解。
  - 「sign 量化噪声注入权重」 N_d = E[q_d^2] * sigma^2_{within,d}：
    sign 误差近独立 per channel，score 噪声方差 = sum_d q_d^2 * MSE_d ∝ N_d。
    用 N_d 排序即 score-噪声意义下的 oracle 选择器（对比 sigma^2-only / E|K|）。

实现：真实 LongBench 单文档 32k prefill；Q 取最后 nq 个位置（decode-like），
通过 q_proj hook 抓 pre-RoPE q，按各架构补 q_norm（Qwen3）并施加 RoPE；
GQA 下 kv-head h 聚合其服务的 n_rep 个 q-head。全部统计 per (layer, kv_head, dim)。

用法：
  CUDA_VISIBLE_DEVICES=1 python qk_channel_importance.py --model MODEL --tag TAG
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
    ap.add_argument("--promote-frac", type=float, default=0.125)
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
    print(f"[model] layers={nl} kv_heads={H} q_heads={nq_heads} (n_rep={n_rep}) head_dim={D}")

    # ---- prefill；最后一个 chunk 挂 q_proj hook 抓 pre-RoPE q ----
    qcap, hook_factory = c.gather_q(model)
    T = ids.shape[1]
    cache = c.prefill(model, ids, dev, chunk=args.chunk, hook_factory=hook_factory)
    torch.cuda.synchronize()
    print(f"[prefill] max_mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    # RoPE cos/sin（绝对位置 T-NQ .. T-1）
    cos, sin = c.rope_tables_for_positions(model, T - NQ, T, D, dev)

    # ---- per (L,H,D) 统计 ----
    rows = dict(identity=[], energy=[], A=[], B=[], B_mu=[], N=[], sig2w=[], eabsk=[])
    tot_B = tot_Bmu = 0.0
    for li in range(nl):
        K = c.cache_layer_k(cache, li)
        x = K[0].permute(0, 2, 1).float()                  # [H,D,T]
        ng = x.shape[-1] // G
        xg = x[:, :, :ng * G].reshape(H, D, ng, G)
        mu_g = xg.mean(-1)                                  # [H,D,ng]
        sig2w = xg.var(dim=-1, unbiased=False).mean(-1)     # [H,D] 组内
        ex2 = (xg * xg).mean(dim=(-1, -2))                  # [H,D]
        mu2 = (mu_g * mu_g).mean(-1)                        # [H,D]
        eabsk = x.abs().mean(-1)                            # [H,D]
        # 协方差 (沿 t) 与组均值协方差
        xc = xg.reshape(H, D, -1)
        xz = xc - xc.mean(-1, keepdim=True)
        C_tot = xz @ xz.transpose(1, 2) / xc.shape[-1]      # [H,D,D]
        mz = mu_g - mu_g.mean(-1, keepdim=True)
        C_mu = mz @ mz.transpose(1, 2) / ng                 # [H,D,D]

        # 真实 q：补 q_norm（若有）+ RoPE，按 kv head 聚合
        attn = model.model.layers[li].self_attn
        q = c.add_q_norm_and_rope(attn, qcap[li][0, -NQ:, :].view(NQ, nq_heads, D), cos, sin)
        q = q.float()                                       # [nq, NQ, D]
        qg = q.reshape(H, n_rep * NQ, D)                    # 每 kv head 的所有 query
        M = qg.transpose(1, 2) @ qg / qg.shape[1]           # [H,D,D] = E[q q^T]
        qabs = qg.abs().mean(1)                             # [H,D]
        q2 = qg.pow(2).mean(1)                              # [H,D]

        B = (C_tot * M).sum(-1)                             # [H,D] 协方差归因
        B_mu = (C_mu * M).sum(-1)
        rows["identity"].append((mu2 / ex2).cpu())
        rows["energy"].append(ex2.cpu())
        rows["A"].append((qabs * eabsk).cpu())
        rows["B"].append(B.cpu())
        rows["B_mu"].append(B_mu.cpu())
        rows["N"].append((q2 * sig2w).cpu())
        rows["sig2w"].append(sig2w.cpu())
        rows["eabsk"].append(eabsk.cpu())
        tot_B += B.sum().item()
        tot_Bmu += B_mu.sum().item()
    del cache
    torch.cuda.empty_cache()

    F = {k: torch.cat([t.detach().reshape(-1) for t in v]).numpy() for k, v in rows.items()}
    C = F["identity"].size
    iden = F["identity"]
    e_sh = F["energy"] / F["energy"].sum()
    a_sh = F["A"] / F["A"].sum()
    b_sh = F["B"] / F["B"].sum()
    n_sh = F["N"] / F["N"].sum()

    # ---- 数值报告 ----
    blue = iden < 0.5
    print(f"\n[mass on blue channels (identity<0.5; {blue.mean()*100:.0f}% of channels)]")
    print(f"  K energy        : {e_sh[blue].sum()*100:5.1f}%")
    print(f"  |score| 幅度 (A) : {a_sh[blue].sum()*100:5.1f}%")
    print(f"  score 方差  (B) : {b_sh[blue].sum()*100:5.1f}%   <- attention 区分度")
    print(f"  sign 噪声   (N) : {n_sh[blue].sum()*100:5.1f}%")
    print(f"\n[score-variance pathway split]  sum_d B_mu / sum_d B = {tot_Bmu/tot_B*100:.1f}% "
          f"via group-mean drift (mu, fp16 无损), {100-tot_Bmu/tot_B*100:.1f}% via residual (量化承压)")

    k = max(1, int(args.promote_frac * C))
    topB = np.argsort(b_sh)[::-1][:k]
    print(f"\n[top-{args.promote_frac*100:.1f}% channels by B (score-variance)]")
    print(f"  median identity = {np.median(iden[topB]):.3f}   blue fraction = {(iden[topB]<0.5).mean()*100:.0f}%")
    print(f"  they hold {b_sh[topB].sum()*100:.1f}% of score variance, {e_sh[topB].sum()*100:.1f}% of K energy")

    # 选择器对比（目标 = sign score-噪声 N_d）
    fracs = [0.0625, 0.125, 0.25, 0.5]

    def removable(key):
        order = np.argsort(key)[::-1]
        return [n_sh[order[:int(f * C)]].sum() for f in fracs]
    rN = removable(F["N"])                  # oracle: q^2*sigma^2
    rS = removable(F["sig2w"])              # sigma^2-only
    rM = removable(F["eabsk"])              # Kitty 现状: E|K|
    ovNS = len(set(np.argsort(F["N"])[::-1][:k]) & set(np.argsort(F["sig2w"])[::-1][:k])) / k
    print(f"\n[removable sign score-noise (share of sum q^2*sigma^2_within)]")
    print(f"  {'k=':>6} {'q2*sig2(oracle)':>16} {'sigma2-only':>12} {'E|K|(mag)':>10}")
    for i, f in enumerate(fracs):
        print(f"  {f*100:5.1f}% {rN[i]*100:>15.1f}% {rS[i]*100:>11.1f}% {rM[i]*100:>9.1f}%")
    print(f"  top-{args.promote_frac*100:.1f}% overlap(q2*sig2 vs sigma2-only) = {ovNS*100:.0f}%")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    bins = np.linspace(0, 1, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    width = 0.028

    ax = axes[0]
    for off, (mass, nm, col) in enumerate([
            (e_sh, "K energy", "0.6"),
            (a_sh, "|score| magnitude (A)", "C3"),
            (b_sh, "score variance (B, discrimination)", "C0")]):
        m = [mass[(iden >= bins[i]) & (iden < bins[i + 1])].sum() * 100 for i in range(10)]
        ax.bar(centers + (off - 1) * width, m, width=width, label=nm, color=col)
    ax.set_xlabel("channel identity: $\\mu^2$ share  (blue side < 0.5 < red side)")
    ax.set_ylabel("% of total mass")
    ax.set_title("(A) where does QK importance mass live\non the $\\sigma^2$↔$\\mu^2$ identity axis")
    ax.axvline(0.5, color="k", lw=0.8, ls="--"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    y = np.maximum(b_sh, 1e-9)
    ax.scatter(iden, y, s=2, alpha=0.15, color="0.5", label="all channels")
    ax.scatter(iden[topB], y[topB], s=4, alpha=0.5, color="C1",
               label=f"top-{args.promote_frac*100:.0f}% by B")
    ax.set_yscale("log")
    ax.set_xlabel("channel identity: $\\mu^2$ share")
    ax.set_ylabel("per-channel share of score variance (B)")
    ax.set_title("(B) discrimination importance vs identity\n"
                 f"top set: median identity {np.median(iden[topB]):.2f}, "
                 f"{(iden[topB]<0.5).mean()*100:.0f}% blue")
    ax.axvline(0.5, color="k", lw=0.8, ls="--"); ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    fx = [f * 100 for f in fracs]
    ax.plot(fx, [v * 100 for v in rN], "o-", label="select by $q^2\\sigma^2$ (oracle)", lw=2)
    ax.plot(fx, [v * 100 for v in rS], "s--", label="select by $\\sigma^2$ only", lw=2)
    ax.plot(fx, [v * 100 for v in rM], "^:", label="select by E|K| (current)", lw=2)
    ax.plot(fx, fx, color="0.7", lw=1, label="chance")
    ax.set_xlabel("% channels promoted"); ax.set_ylabel("% of sign score-noise removed")
    ax.set_title("(C) attention-aware selector payoff\n(target = $\\sum q^2 \\cdot \\mathrm{MSE}$ score noise)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"QK score contribution vs $\\mu^2/\\sigma^2$ identity  —  "
        f"{c.model_basename(model_path)}, {src}, {T} tok, last {NQ} real queries, G={G}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    c.save_fig(fig, args.outdir, f"qk_importance_{args.tag}.png")


if __name__ == "__main__":
    main()
