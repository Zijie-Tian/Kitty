#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 post-RoPE 空间下，高 sigma^2（蓝）K channel 的低-bit 码本对决（不使用 de-RoPE）。

约束：所有方法都在 post-RoPE 空间量化与重建，不逆旋转。

post-RoPE 下蓝通道为何难：快频 RoPE 通道在 128-token 组内扫过多个旋转周期，
post-RoPE 值近似 A*cos(theta*t+phi)，组内分布是双峰 arcsine（质量堆在 ±A、0 处最稀）——
打在 sign/tern 假设的反面（sign 设单峰零均值；tern 死区把电平放在最稀的 0 处）。
故纯 post-RoPE 的两条杠杆：
  (1) 更小量化组 G：让 per-group mu 分段跟踪正弦（de-RoPE 的数值近似），代价是侧信息 32/G 涨；
  (2) 更优码本电平：per-group Lloyd-Max 最优 4 电平（自然适应双峰），而非 sign/tern/均匀 min-max。

蓝通道子类 = 纯靠 RoPE 频率离线判定（无需 pre-RoPE）：
  fast = 旋转周期 2π/theta_d < 组长 G（组内扫过 >=1 周期 -> arcsine 双峰，难）；
  slow = 周期 >= G（组内近似常数 -> 易，submean 有效）。

码本（effective bit = 码字位 + 32/G 侧信息；Lloyd 记为最优电平上界，可由离线标定的
固定形状码本 + per-group scale 在 2+32/G 位部署）：
  sign  G128/64/32 ; tern G128/64/32 ; uni2 G128 ; lloyd2 G128(最优4电平) ; uni3 G128(天花板)

指标：各子类 post 空间 K NMSE。回答"post-RoPE-only，蓝通道低 bit 用什么、值不值得花在更小 G 还是更多电平"。

用法：
  CUDA_VISIBLE_DEVICES=1 python postrope_blue_codebook.py --model MODEL --tag TAG
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
    ap.add_argument("--rope-base", type=float, default=None)
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

    base = args.rope_base or getattr(model.config, "rope_theta", 10000.0)
    # RoPE 周期（按 pair）：theta_j = base^(-2j/D), period = 2π/theta_j
    j = torch.arange(half)
    theta = base ** (-2.0 * j / D)
    period = 2 * np.pi / theta                                         # [half]
    fast_pair = period < G                                            # 组内 >=1 周期
    print(f"[model] layers={nl} kv_heads={H} head_dim={D} rope_base={base:.0f}; "
          f"fast pairs (period<{G}): {int(fast_pair.sum())}/{half}")

    T = ids.shape[1]
    cache = c.prefill(model, ids, dev, chunk=args.chunk)
    print(f"[prefill] max_mem={torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

    # 码本清单：(label, kind, G)
    SCHEMES = [
        ("sign-G128", "sign", 128), ("sign-G64", "sign", 64), ("sign-G32", "sign", 32),
        ("tern-G128", "tern", 128), ("tern-G64", "tern", 64), ("tern-G32", "tern", 32),
        ("uni2-G128", "uni2", 128), ("uni3-G128", "uni3", 128),
        ("lloyd2-G128", "lloyd2", 128), ("lloyd2-G64", "lloyd2", 64),
    ]

    def bits_of(kind, g):
        cw = {"sign": 1.0, "tern": np.log2(3), "uni2": 2.0, "uni3": 3.0,
              "lloyd2": 2.0}[kind]
        return cw + 32.0 / g

    classes = ["fast", "slow", "red"]
    sse = {lab: {cl: 0.0 for cl in classes} for lab, _, _ in SCHEMES}
    eng = {cl: 0.0 for cl in classes}
    cnt = {cl: 0 for cl in classes}

    q0, q1 = args.sink, T - args.recent
    with torch.inference_mode():
        for li in range(nl):
            K = c.cache_layer_k(cache, li)
            x = K[0].permute(0, 2, 1).float()                        # [H,D,T]
            xq = x[:, :, q0:q1]
            ng = (q1 - q0) // G
            seg = xq[:, :, :ng * G].reshape(H, D, ng, G)
            ex2 = (seg * seg).mean(dim=(-1, -2))                     # [H,D]
            mu2 = seg.mean(-1).pow(2).mean(-1)
            iden_pair = (mu2[:, :half] + mu2[:, half:]) / (ex2[:, :half] + ex2[:, half:]).clamp_min(1e-12)
            blue_pair = iden_pair < 0.5                              # [H,half]
            fp = fast_pair.to(dev)[None].expand(H, half)
            cls_pair = {"fast": blue_pair & fp, "slow": blue_pair & ~fp, "red": ~blue_pair}
            cls_chan = {cl: torch.cat([m, m], dim=1) for cl, m in cls_pair.items()}  # [H,D]

            e_tot = (xq * xq).sum(-1)
            for cl in classes:
                eng[cl] += e_tot[cls_chan[cl]].sum().item()
                cnt[cl] += int(cls_chan[cl].sum().item())

            for lab, kind, g in SCHEMES:
                if kind == "lloyd2":
                    rec = c.lloyd_codebook(xq, g, L=4)
                else:
                    rec = c.submean_codebook(xq, g, kind)
                e = (rec - xq).pow(2).sum(-1)                        # [H,D]
                for cl in classes:
                    sse[lab][cl] += e[cls_chan[cl]].sum().item()
    del cache
    torch.cuda.empty_cache()

    tot = sum(cnt.values())
    print(f"\n[channel mix] fast-blue {cnt['fast']/tot*100:.0f}%  slow-blue {cnt['slow']/tot*100:.0f}%  "
          f"red {cnt['red']/tot*100:.0f}%")
    print(f"\n{'scheme':>13} {'bits':>6} {'fast-blue':>10} {'slow-blue':>10} {'red':>8}")
    res = {}
    for lab, kind, g in SCHEMES:
        b = bits_of(kind, g)
        nm = {cl: sse[lab][cl] / max(eng[cl], 1e-9) for cl in classes}
        res[lab] = (b, nm)
        print(f"{lab:>13} {b:>6.2f} {nm['fast']:>10.4f} {nm['slow']:>10.4f} {nm['red']:>8.4f}")

    # Pareto 前沿（fast-blue）
    print(f"\n[fast-blue Pareto: NMSE at each bit, post-RoPE only]")
    pts = sorted([(res[l][0], res[l][1]['fast'], l) for l in res])
    best = 1e9
    for b, nm, l in pts:
        tag = "  <- Pareto" if nm < best - 1e-9 else ""
        if nm < best:
            best = nm
        print(f"  {b:>5.2f} bit  NMSE={nm:.4f}  {l}{tag}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    cmap = {"sign": "C0", "tern": "C1", "uni2": "C2", "uni3": "C4", "lloyd2": "C3"}
    for ax, cl, ttl in zip(
            axes, classes,
            ["(A) fast-blue (period<G): RoPE-swept, arcsine",
             "(B) slow-blue (period>=G): submean works",
             "(C) red (reference)"]):
        for lab, kind, g in SCHEMES:
            b, nm = res[lab][0], res[lab][1][cl]
            ax.scatter(b, nm, c=cmap[kind], s=60, zorder=3)
            ax.annotate(lab.replace("-G", "·"), (b, nm), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points")
        # Pareto line
        pts = sorted([(res[l][0], res[l][1][cl]) for l in res])
        px, py, bb = [], [], 1e9
        for b, nm in pts:
            if nm < bb - 1e-9:
                bb = nm; px.append(b); py.append(nm)
        ax.plot(px, py, "k--", lw=1, alpha=0.5, zorder=2)
        ax.set_xlabel("effective bits/elem"); ax.set_ylabel("class K NMSE")
        ax.set_title(ttl); ax.grid(alpha=0.3)
    fig.suptitle(
        f"Post-RoPE-only low-bit codebook for high-$\\sigma^2$ K channels  —  "
        f"{c.model_basename(model_path)}, {src}, {T} tok  |  "
        f"blue: fast {cnt['fast']/tot*100:.0f}% + slow {cnt['slow']/tot*100:.0f}%  "
        f"(color: sign/tern/uni2/uni3/lloyd2)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    c.save_fig(fig, args.outdir, f"postrope_blue_{args.tag}.png")


if __name__ == "__main__":
    main()
