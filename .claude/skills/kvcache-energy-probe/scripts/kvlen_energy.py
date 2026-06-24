#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-cache 能量 / 减均值分解沿 kv-len（token 位置）的分布：是否集中在某些位置。

在一条 ~32k 真实 LongBench 文档 (post-RoPE K) 上，沿 token 轴看三件事：
  1) 每个 token 的 K 能量 ||k_t||^2（在 channel 上求和），看是否有 attention-sink /
     massive-norm token 把能量集中在序列开头（Kitty 保 sink=32 + recent=128 fp16 的依据）；
  2) 把 token 切成 128 一组（与量化组同尺度），每组的 mu^2 / sigma^2 能量占比沿组位置
     变化——看"残差占比高（难量化）"是否集中在某段 kv-len；
  3) 每组的 sign-码本实际 NMSE 沿组位置变化——看 sign 在哪段 kv-len 误差最大。

mu, sigma 的统计口径与 submean 码本一致：per-channel 取向、每 128 token 一组、population。
报告也给出去掉前 256 个 token（sink 区）后的占比，以区分"开头特殊"与"整体趋势"。

注：sign-NMSE 在 flat-C [C, n_g, G] 上直接计算（并截掉末尾不足一组），与 _common 的
submean_sign（[H,D,T] + 回填末尾）语义不同，故此处的分解保留本地实现（kvlen_decomp）。

用法：
  CUDA_VISIBLE_DEVICES=1 python kvlen_energy.py \
      --model MODEL --seq-len 32768 --group 128 --tag TAG
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import _common as c


@torch.inference_mode()
def kvlen_decomp(K, G):
    """K: [1,H,T,D] post-RoPE.
       返回沿 token 的 per-token 能量 [T]，以及 per-group(沿 kv-len) 的
       mu^2 / sigma^2 / sign-MSE / energy（在 channel 上求和）各 [n_groups]。"""
    x = K[0].permute(0, 2, 1).reshape(-1, K.shape[2]).float()      # [C, T]
    C, T = x.shape
    per_token_energy = (x * x).sum(0)                              # [T]
    Tu = (T // G) * G
    xg = x[:, :Tu].reshape(C, Tu // G, G)                          # [C, n_g, G]
    mu = xg.mean(-1, keepdim=True)
    r = xg - mu
    m = r.abs().mean(-1, keepdim=True)
    xhat = mu + m * torch.sign(r)
    grp = dict(
        mu2=(mu.squeeze(-1) ** 2).sum(0),                          # [n_g] sum over channels
        sig2=r.pow(2).mean(-1).sum(0),
        mse=(xg - xhat).pow(2).mean(-1).sum(0),
        energy=(xg * xg).mean(-1).sum(0),
    )
    return per_token_energy.cpu(), {k: v.cpu() for k, v in grp.items()}


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
    ids, src, full = c.pick_single_doc(lb_dir, tok, args.seq_len)
    print(f"[input] {src}: full {full} tok, using {ids.shape[1]}  (G={args.group})")
    print(f"[model] layers={dims.nl} kv_heads={dims.H}")

    cache = c.prefill(model, ids, dev, chunk=args.chunk)
    torch.cuda.synchronize()
    print(f"[prefill] max_mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    nlay = c.cache_n_layers(cache)
    T = ids.shape[1]
    ng = T // args.group
    tok_e = torch.zeros(T)
    grp_sum = {k: torch.zeros(ng) for k in ("mu2", "sig2", "mse", "energy")}
    share_share = torch.zeros(nlay, ng)                            # 每层每组的 sigma^2 占比
    for li in range(nlay):
        K = c.cache_layer_k(cache, li)
        te, grp = kvlen_decomp(K, args.group)
        tok_e += te
        for k in grp_sum:
            grp_sum[k] += grp[k]
        share_share[li] = grp["sig2"] / grp["energy"]
    mu2_share = grp_sum["mu2"] / grp_sum["energy"]
    sig2_share = grp_sum["sig2"] / grp_sum["energy"]
    nmse = grp_sum["mse"] / grp_sum["energy"]

    # ---- 数值报告 ----
    pos = torch.arange(ng) * args.group
    print("\n[per-token K energy: sink / massive-norm concentration]")
    e = tok_e
    print(f"  token0 energy / median-token energy = {(e[0] / e.median()).item():.1f}x")
    print(f"  first  32 tokens hold {e[:32].sum().item() / e.sum().item() * 100:5.2f}% of total K energy "
          f"({32/T*100:.2f}% of tokens)")
    print(f"  first 128 tokens hold {e[:128].sum().item() / e.sum().item() * 100:5.2f}% of total K energy")
    print(f"  top-1% tokens by energy hold {e.sort(descending=True).values[:int(0.01*T)].sum().item()/e.sum().item()*100:5.2f}%")

    print("\n[sigma^2 (residual) share vs kv position, per 128-group]")
    print(f"  group 0   (tok 0-128)   : sigma^2 share = {sig2_share[0].item()*100:5.1f}%")
    print(f"  group 1   (tok 128-256) : sigma^2 share = {sig2_share[1].item()*100:5.1f}%")
    print(f"  groups 2..end (tok>256) : sigma^2 share = {sig2_share[2:].mean().item()*100:5.1f}% (mean)")
    print(f"  overall                 : sigma^2 share = {sig2_share.mean().item()*100:5.1f}%")
    # 相关性: sigma^2 占比是否随 kv 位置漂移
    gi = torch.arange(2, ng).float()
    sv = sig2_share[2:]
    corr = torch.corrcoef(torch.stack([gi, sv]))[0, 1].item()
    print(f"  corr(sigma^2 share, kv position)  (tok>256) = {corr:+.3f}  "
          f"(~0 => 残差占比沿 kv-len 基本平稳, 不集中)")

    print("\n[sign-codebook NMSE vs kv position]")
    print(f"  group 0   : NMSE = {nmse[0].item():.4f}")
    print(f"  group 1   : NMSE = {nmse[1].item():.4f}")
    print(f"  tok>256   : NMSE = {nmse[2:].mean().item():.4f} (mean)")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))

    ax = axes[0]
    ax.semilogy(torch.arange(T).numpy(), tok_e.numpy(), lw=0.4, color="C0")
    ax.set_xlabel("token position (kv-len)"); ax.set_ylabel("per-token K energy $\\|k_t\\|^2$ (log)")
    ax.set_title("(1) per-token energy: attention-sink spikes at the start")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(pos.numpy(), mu2_share.numpy() * 100, label="$\\mu^2$ share", color="C3", lw=1.5)
    ax.plot(pos.numpy(), sig2_share.numpy() * 100, label="$\\sigma^2$ share", color="C0", lw=1.5)
    ax.set_xlabel("kv position (group start token)"); ax.set_ylabel("energy share (%)")
    ax.set_title("(2) $\\mu^2$/$\\sigma^2$ share vs kv-len: flat after the sink")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 100)

    ax = axes[2]
    im = ax.imshow(share_share.numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=0.6,
                   extent=(0, T, nlay - 0.5, -0.5))
    ax.set_xlabel("kv position (token)"); ax.set_ylabel("layer")
    ax.set_title("(3) $\\sigma^2$ share per layer x kv-position")
    fig.colorbar(im, ax=ax, label="$\\sigma^2$ share")

    fig.suptitle(
        f"K-cache energy decomposition along kv-len  —  "
        f"{c.model_basename(model_path)}, {src}, {T} tok, G={args.group}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    c.save_fig(fig, args.outdir, f"kvlen_energy_{args.tag}.png")


if __name__ == "__main__":
    main()
