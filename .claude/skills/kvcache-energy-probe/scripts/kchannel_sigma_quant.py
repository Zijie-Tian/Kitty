#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-cache 的 per-channel sigma^2 结构，以及"对高 sigma^2 channel 换量化方式"是否值得。

在 submean+sign/tern 码本下，组均值 mu 用 fp16 精确存，量化误差全部落在残差 r 上、
正比于 sigma^2 = E[r^2]。因此:
  - mu^2 占比大的红 channel  -> 残差小 -> sign 绝对误差小（sign 很好）
  - sigma^2 占比大的蓝 channel -> 残差大 -> sign 绝对误差大（sign 最吃亏）

本脚本在一条 ~32k 的真实 LongBench 文档上 (post-RoPE K) 直接测：
  1) 每个 (head,dim) channel 的 sign-码本实际 MSE、sigma^2、mu^2，看 sigma^2 的集中度
     与按 head_dim 的剖面（验证"蓝 channel = 高频 RoPE 通道"这一结构假设）；
  2) 当前 Kitty 的 magnitude selector (E|K|) vs variance selector (sigma^2) 选出的
     top-k channel 的重合度——若 magnitude 选的是红 channel，则它在 submean 世界里
     选错了对象；
  3) "保护 top-k channel 到 fp16"能消掉的 cache 总 MSE 占比，按 variance / magnitude /
     random 三种选择排序，量化"换方式"的收益上界与正确的选择信号；
  4) (Llama, 无 k_norm) 同一 channel 的 pre-RoPE sigma^2 vs post-RoPE sigma^2，验证
     蓝 channel 的大 sigma^2 主要是 RoPE 旋转的人为产物 -> 对这些 channel 改在
     pre-RoPE 量化(ShadowKV 式)是"另一种方式"的物理依据。

注：per-channel 五元组在 flat-C [C, n_g, G] 上用一次性 mean(dim=(1,2)) 计算（含 sign
重建），保留本地实现以与原口径逐位一致。

用法：
  CUDA_VISIBLE_DEVICES=1 python kchannel_sigma_quant.py \
      --model MODEL --seq-len 32768 --group 128 --tag TAG --capture-pre-rope
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import _common as c


@torch.inference_mode()
def perchan_sign_stats(K, G):
    """K: [1,H,T,D] post-RoPE -> per-channel [H*D] tensors:
       mse (sign 码本实际 MSE), sig2 (E[r^2]), mu2 (E[mu^2]), energy (E[x^2]), absk (E|x|)."""
    x = K[0].permute(0, 2, 1).reshape(-1, K.shape[2])          # [C, T], C=H*D
    C, T = x.shape
    Tu = (T // G) * G
    xg = x[:, :Tu].reshape(C, Tu // G, G).float()              # [C, n_g, G]
    mu = xg.mean(-1, keepdim=True)
    r = xg - mu
    m = r.abs().mean(-1, keepdim=True)                         # sign 电平 = E|r|
    xhat = mu + m * torch.sign(r)
    mse = (xg - xhat).pow(2).mean(dim=(1, 2))                  # [C]
    sig2 = r.pow(2).mean(dim=(1, 2))
    mu2 = (mu.squeeze(-1) ** 2).mean(-1)
    energy = (xg * xg).mean(dim=(1, 2))
    absk = xg.abs().mean(dim=(1, 2))
    return {k: v.cpu() for k, v in
            dict(mse=mse, sig2=sig2, mu2=mu2, energy=energy, absk=absk).items()}


@torch.inference_mode()
def perchan_sig2_only(K, G):
    x = K[0].permute(0, 2, 1).reshape(-1, K.shape[2])
    C, T = x.shape
    Tu = (T // G) * G
    xg = x[:, :Tu].reshape(C, Tu // G, G).float()
    r = xg - xg.mean(-1, keepdim=True)
    return r.pow(2).mean(dim=(1, 2)).cpu()                     # [C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地路径 / HF repo id / 别名 (见 SKILL.md)")
    ap.add_argument("--longbench-dir", default=None, help="LongBench data 目录；缺省读 KVPROBE_LONGBENCH_DIR / LONGBENCH_DATA_ROOT")
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--outdir", default="probe_out")
    ap.add_argument("--capture-pre-rope", action="store_true",
                    help="hook k_proj for pre-RoPE K (Llama family, no k_norm)")
    args = ap.parse_args()

    dev = "cuda:0"
    model_path = c.resolve_model_path(args.model)
    lb_dir = c.resolve_longbench_dir(args.longbench_dir)

    model, tok, dims = c.load_model_and_tok(model_path, dev)
    H, D, nl = dims.H, dims.D, dims.nl
    ids, src, full = c.pick_single_doc(lb_dir, tok, args.seq_len)
    print(f"[input] {src}: full {full} tok, using {ids.shape[1]}  (G={args.group})")
    print(f"[model] layers={nl} kv_heads={H} head_dim={D}")

    # 可选: hook k_proj 抓 pre-RoPE K（Llama 无 k_norm，k_proj 输出即 pre-RoPE）
    pre_store = None
    hook_factory = None
    if args.capture_pre_rope:
        pre_store, hook_factory = c.gather_pre_rope_k(model)

    cache = c.prefill(model, ids, dev, chunk=args.chunk, hook_factory=hook_factory)
    torch.cuda.synchronize()
    print(f"[prefill] max_mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    # 汇总所有层的 per-channel 统计（K only）
    agg = {k: [] for k in ("mse", "sig2", "mu2", "energy", "absk")}
    pre_sig2_by_dim = torch.zeros(D)
    post_sig2_by_dim = torch.zeros(D)
    npre = 0
    for li in range(nl):
        K = c.cache_layer_k(cache, li)
        st = perchan_sign_stats(K, args.group)
        for k in agg:
            agg[k].append(st[k])
        post_sig2_by_dim += st["sig2"].reshape(H, D).mean(0)
        if pre_store is not None and pre_store.get(li):
            preK = torch.cat(pre_store[li], dim=1)            # [1,T,H*D]
            preK = preK.reshape(1, -1, H, D).permute(0, 2, 1, 3)  # [1,H,T,D]
            pre_sig2_by_dim += perchan_sig2_only(preK, args.group).reshape(H, D).mean(0)
            npre += 1
    agg = {k: torch.cat(v) for k, v in agg.items()}           # each [nl*H*D]
    post_sig2_by_dim /= nl
    if npre:
        pre_sig2_by_dim /= npre

    C = agg["mse"].numel()
    tot_mse = agg["mse"].sum().item()
    tot_energy = agg["energy"].sum().item()
    print(f"\n[K all-sign] cache NMSE = {tot_mse / tot_energy:.4f}   ({C} channels)")

    # --- (1) sigma^2 集中度: top-k% channel 占总残差能量的比例 ---
    sig2_sorted = agg["sig2"].sort(descending=True).values
    cum = sig2_sorted.cumsum(0) / sig2_sorted.sum()
    print("\n[sigma^2 concentration over channels]")
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        idx = int(frac * C) - 1
        print(f"  top {frac*100:4.0f}% channels hold {cum[idx].item()*100:5.1f}% of total residual energy")

    # --- (2) 选择器对比: magnitude (E|K|, 现 Kitty) vs variance (sigma^2) top-k 重合 ---
    print("\n[selector overlap: magnitude(E|K|, current) vs variance(sigma^2)]")
    for frac in (0.0625, 0.125, 0.25):                        # pr=0.0625/0.125/0.25
        k = max(1, int(frac * C))
        ov = c.overlap_sets(agg["absk"].topk(k).indices, agg["sig2"].topk(k).indices, k)
        print(f"  k={frac*100:4.1f}%  overlap = {ov*100:5.1f}%  "
              f"(magnitude picks {(1-ov)*100:.0f}% DIFFERENT channels than variance)")

    # --- (3) 保护 top-k channel 到 fp16 能消掉的 cache MSE（按不同选择器排序）---
    def removable(order):
        # order: channel 索引按某 saliency 降序; 返回 (frac_grid, nmse_after)
        fr = [0, 0.0625, 0.125, 0.25, 0.5]
        out = []
        contrib = agg["mse"][order]
        for f in fr:
            k = int(f * C)
            removed = contrib[:k].sum().item()
            out.append((tot_mse - removed) / tot_energy)
        return fr, out

    fr, nmse_var = removable(agg["sig2"].sort(descending=True).indices)
    _, nmse_mag = removable(agg["absk"].sort(descending=True).indices)
    _, nmse_orc = removable(agg["mse"].sort(descending=True).indices)  # oracle = 直接按 MSE
    print("\n[cache K NMSE after promoting top-k channels to fp16]")
    print(f"  {'k=':>6} {'variance':>10} {'magnitude':>10} {'oracle(MSE)':>12}")
    for i, f in enumerate(fr):
        print(f"  {f*100:5.1f}% {nmse_var[i]:>10.4f} {nmse_mag[i]:>10.4f} {nmse_orc[i]:>12.4f}")

    # --- (4) pre vs post RoPE sigma^2 per dim ---
    if npre:
        print("\n[pre vs post RoPE residual sigma^2, averaged over heads, per dim]")
        ratio = (post_sig2_by_dim / pre_sig2_by_dim.clamp_min(1e-8))
        hi = ratio.topk(6).indices.tolist()
        print(f"  dims where RoPE inflates sigma^2 most (post/pre): "
              + ", ".join(f"d{d}:{ratio[d]:.1f}x" for d in hi))

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))

    ax = axes[0]
    ax.plot([f * 100 for f in [i / C for i in range(C)]], cum.numpy() * 100, lw=2)
    ax.set_xlabel("top-x% channels by $\\sigma^2$"); ax.set_ylabel("% of total residual energy")
    ax.set_title("(1) $\\sigma^2$ concentration across K channels")
    ax.grid(alpha=0.3); ax.set_xlim(0, 50)

    ax = axes[1]
    x = list(range(D))
    ax.plot(x, post_sig2_by_dim.numpy(), label="post-RoPE $\\sigma^2$", lw=2, color="C3")
    if npre:
        ax.plot(x, pre_sig2_by_dim.numpy(), label="pre-RoPE $\\sigma^2$", lw=2, color="C0")
    ax.set_xlabel("head_dim channel index"); ax.set_ylabel("residual $\\sigma^2$ (mean over heads/layers)")
    ax.set_title("(2) per-dim $\\sigma^2$: RoPE inflates the fast (low-index) pairs")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot([f * 100 for f in fr], [v for v in nmse_var], "o-", label="select by $\\sigma^2$ (variance)", lw=2)
    ax.plot([f * 100 for f in fr], [v for v in nmse_mag], "s--", label="select by E|K| (current Kitty)", lw=2)
    ax.set_xlabel("% channels promoted to fp16"); ax.set_ylabel("cache K NMSE (sign elsewhere)")
    ax.set_title("(3) variance-select removes error far faster")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(
        f"K-cache per-channel $\\sigma^2$ & heterogeneous quantization payoff  —  "
        f"{c.model_basename(model_path)}, {src}, {ids.shape[1]} tok, G={args.group}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    c.save_fig(fig, args.outdir, f"kchan_sigma_{args.tag}.png")


if __name__ == "__main__":
    main()
