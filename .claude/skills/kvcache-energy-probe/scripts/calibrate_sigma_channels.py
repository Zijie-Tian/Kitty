#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线标定高 sigma^2（残差主导）K channel 掩码，并在全 21 个 LongBench 任务上验证泛化。

动机（本系列前几轮实测）：submean+sign 把 mu 精确存，量化误差全在残差、正比于 sigma^2；
高 sigma^2 通道 = sign 最吃亏处，应走"另一种方法"（de-RoPE / tern / promote）。前几轮已证
per-channel 的红/蓝身份跨任务高度一致（dim 画像两两 r>=0.99，选择器重合 ~72-85%）。
本脚本据此把"选哪些通道"做成 OFFLINE 标定：

  标定（域外）：在 wikitext（与 LongBench 完全不同域）上 prefill，逐 (layer, kv_head, dim)
    统计残差 sigma^2（per-channel 取向、128-token 组、submean），按 per-head top-k 选出
    静态高-sigma^2 掩码，存成可加载 artifact。
  验证（任务内、held-out）：对全 21 个 LongBench 任务各取一条真实样本，对比
    - mask overlap：标定掩码 ∩ 该任务自身 top-k(sigma^2) 的比例；
    - energy capture：标定掩码在该任务上抓住的残差能量 / 该任务 oracle 掩码能抓的能量
      （核心指标——切线附近的通道 sigma^2 小，错选代价低，故此值通常远高于 mask overlap）；
    - NMSE retention：把标定掩码通道提到 fp16 后该任务的 K sign-NMSE，对比 oracle 掩码；
    - 随机基线。

判定：energy capture >= 95% 即认为离线标定可部署（静态掩码 ≈ 每任务自适应）。

用法 (GPU1)：
  CUDA_VISIBLE_DEVICES=1 python calibrate_sigma_channels.py \
      --model MODEL \
      --calib-parquet /path/to/wikitext/train-00000-of-00001.parquet --tag TAG
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _common as c


@torch.inference_mode()
def prefill_sigma2(model, ids, dev, G, chunk):
    """返回 per (L,H,D) 的残差 sigma^2、能量 ex2、sign-MSE（沿 token 轴 G 组、submean）。

    sig2/ex2/mse 用一次性 mean(dim=(-1,-2)) 计算（含 sign 重建），与原口径逐位一致。
    """
    cache = c.prefill(model, ids, dev, chunk)
    nl = c.cache_n_layers(cache)
    sig2, ex2, mse = [], [], []
    for li in range(nl):
        K = c.cache_layer_k(cache, li)
        x = K[0].permute(0, 2, 1).float()              # [H,D,T]
        H, D, T = x.shape
        ng = T // G
        xg = x[:, :, :ng * G].reshape(H, D, ng, G)
        mu = xg.mean(-1, keepdim=True)
        r = xg - mu
        m = r.abs().mean(-1, keepdim=True)
        xhat = mu + m * torch.sign(r)
        sig2.append(r.pow(2).mean(dim=(-1, -2)).cpu())          # [H,D]
        ex2.append((xg * xg).mean(dim=(-1, -2)).cpu())
        mse.append((xg - xhat).pow(2).mean(dim=(-1, -2)).cpu())
    del cache
    torch.cuda.empty_cache()
    return torch.stack(sig2), torch.stack(ex2), torch.stack(mse)   # each [L,H,D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地路径 / HF repo id / 别名 (见 SKILL.md)")
    ap.add_argument("--longbench-dir", default=None, help="LongBench data 目录；缺省读 KVPROBE_LONGBENCH_DIR / LONGBENCH_DATA_ROOT")
    ap.add_argument("--calib-parquet", required=True, help="域外标定语料 parquet（如 wikitext）")
    ap.add_argument("--calib-tokens", type=int, default=32768)
    ap.add_argument("--val-tokens", type=int, default=8192)
    ap.add_argument("--val-min-tokens", type=int, default=3072)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--frac", type=float, default=0.125, help="主验证比例")
    ap.add_argument("--frac-grid", default="0.0625,0.125,0.25")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--outdir", default="probe_out")
    args = ap.parse_args()

    dev = "cuda:0"
    os.makedirs(args.outdir, exist_ok=True)
    model_path = c.resolve_model_path(args.model)
    lb_dir = c.resolve_longbench_dir(args.longbench_dir)

    model, tok, dims = c.load_model_and_tok(model_path, dev)
    H, D, nl = dims.H, dims.D, dims.nl
    G = args.group
    print(f"[model] layers={nl} kv_heads={H} head_dim={D}")

    # ---------- 标定（wikitext, 域外） ----------
    text = c.build_calib_text(args.calib_parquet, need_chars=args.calib_tokens * 6)
    cids = tok(text, return_tensors="pt").input_ids
    if cids.shape[1] < args.calib_tokens:
        print(f"[warn] calib only {cids.shape[1]} tok < {args.calib_tokens}")
    cids = cids[:, :args.calib_tokens]
    print(f"[calib] wikitext {cids.shape[1]} tokens")
    sig2_cal, ex2_cal, _ = prefill_sigma2(model, cids, dev, G, args.chunk)
    print(f"[calib] done. overall mu^2 share = {1 - sig2_cal.sum()/ex2_cal.sum():.4f}")

    fracs = [float(x) for x in args.frac_grid.split(",")]
    masks_cal = {f: c.topk_mask(sig2_cal, max(1, int(f * D))) for f in fracs}

    # 存 artifact（可加载用于量化路径）
    half = D // 2
    pair_sig2_cal = sig2_cal[..., :half] + sig2_cal[..., half:]   # [L,H,D/2] pair-coupled
    art = {
        "model": model_path, "calib_source": os.path.basename(args.calib_parquet),
        "calib_tokens": int(cids.shape[1]), "group": G,
        "sigma2_profile": sig2_cal, "energy_profile": ex2_cal,
        "pair_sigma2_profile": pair_sig2_cal,
        "masks": {f"{f}": masks_cal[f] for f in fracs},
        "pair_masks": {f"{f}": c.topk_mask(pair_sig2_cal, max(1, int(f * half))) for f in fracs},
        "layout": "[n_layers, n_kv_heads, head_dim] bool; pair = [.., head_dim/2]",
    }
    art_path = os.path.join(args.outdir, f"sigma_calib_{args.tag}.pt")
    torch.save(art, art_path)
    print(f"[saved] {art_path}")

    # ---------- 验证（LongBench 21 任务, held-out） ----------
    f0 = args.frac
    k0 = max(1, int(f0 * D))
    cal_mask0 = masks_cal[f0]
    g = torch.Generator().manual_seed(0)
    rand_mask0 = c.topk_mask(torch.rand(nl, H, D, generator=g), k0)

    rows = []
    dim_profiles = {}
    print(f"\n[validate] per task @ frac={f0} (k={k0}/{D} per head); "
          f"energy-capture = task residual energy caught by calib-mask / by task-oracle")
    print(f"  {'task':>20} {'muShare':>8} {'maskOv':>7} {'eCap':>6} {'eCapRnd':>8} "
          f"{'NMSE.sign':>10} {'NMSE.cal':>9} {'NMSE.orc':>9}")
    for task in c.VAL_TASKS:
        ids, ntok = c.first_task_sample(lb_dir, task, tok, args.val_min_tokens, args.val_tokens)
        if ids is None:
            print(f"  {task:>20}  [skip: no sample >= {args.val_min_tokens} tok]")
            continue
        sig2_t, ex2_t, mse_t = prefill_sigma2(model, ids, dev, G, args.chunk)
        orc_mask = c.topk_mask(sig2_t, k0)
        mu_share = (1 - sig2_t.sum() / ex2_t.sum()).item()
        # mask overlap (per head 平均)
        ov = (cal_mask0 & orc_mask).sum(-1).float().mean().item() / k0
        # energy capture：用该任务自身 sigma^2 度量
        e_cal = sig2_t[cal_mask0].sum().item()
        e_orc = sig2_t[orc_mask].sum().item()
        e_rnd = sig2_t[rand_mask0].sum().item()
        ecap = e_cal / e_orc
        ecap_rnd = e_rnd / e_orc
        # NMSE retention（提到 fp16 = 去掉该通道 MSE）
        tot_mse, tot_e = mse_t.sum().item(), ex2_t.sum().item()
        nmse_sign = tot_mse / tot_e
        nmse_cal = (tot_mse - mse_t[cal_mask0].sum().item()) / tot_e
        nmse_orc = (tot_mse - mse_t[orc_mask].sum().item()) / tot_e
        rows.append(dict(task=task, ntok=ntok, mu_share=mu_share, overlap=ov,
                         ecap=ecap, ecap_rnd=ecap_rnd, nmse_sign=nmse_sign,
                         nmse_cal=nmse_cal, nmse_orc=nmse_orc))
        dim_profiles[task] = (sig2_t / ex2_t.clamp_min(1e-9)).mean(dim=(0, 1)).numpy()
        print(f"  {task:>20} {mu_share*100:>7.1f}% {ov*100:>6.0f}% {ecap*100:>5.0f}% "
              f"{ecap_rnd*100:>7.0f}% {nmse_sign:>10.4f} {nmse_cal:>9.4f} {nmse_orc:>9.4f}")

    ov_m = np.mean([r["overlap"] for r in rows])
    ec_m = np.mean([r["ecap"] for r in rows])
    ec_r = np.mean([r["ecap_rnd"] for r in rows])
    print(f"\n[aggregate over {len(rows)} tasks @ frac={f0}]")
    print(f"  mask overlap   mean={ov_m*100:.1f}%  (calib vs each task's own top-k)")
    print(f"  energy capture mean={ec_m*100:.1f}%  (random baseline {ec_r*100:.1f}%)  <-- 部署判据")
    print(f"  NMSE: sign-only {np.mean([r['nmse_sign'] for r in rows]):.4f} -> "
          f"calib-promote {np.mean([r['nmse_cal'] for r in rows]):.4f} -> "
          f"oracle-promote {np.mean([r['nmse_orc'] for r in rows]):.4f}")

    json.dump({"frac": f0, "k": k0, "aggregate": dict(
        mask_overlap=ov_m, energy_capture=ec_m, energy_capture_random=ec_r),
        "per_task": rows},
        open(os.path.join(args.outdir, f"sigma_calib_{args.tag}_val.json"), "w"),
        indent=2)

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    ax = axes[0]
    cal_prof = (sig2_cal / ex2_cal.clamp_min(1e-9)).mean(dim=(0, 1)).numpy()
    ax.plot(cal_prof, color="k", lw=2.5, label="CALIB (wikitext)", zorder=5)
    for i, (t, p) in enumerate(list(dim_profiles.items())[:8]):
        ax.plot(p, lw=0.8, alpha=0.7, label=t)
    ax.set_xlabel("head_dim channel index"); ax.set_ylabel("$\\sigma^2$ share (mean over L,H)")
    ax.set_title("(A) calib $\\sigma^2$ dim-profile vs tasks\n(overlap = task-independent)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = axes[1]
    tnames = [r["task"] for r in rows]
    yp = np.arange(len(rows))
    ax.barh(yp, [r["ecap"] * 100 for r in rows], color="C2", label="calib mask")
    ax.plot([r["ecap_rnd"] * 100 for r in rows], yp, "rx", ms=5, label="random")
    ax.axvline(95, color="0.4", ls="--", lw=1, label="95% gate")
    ax.set_yticks(yp); ax.set_yticklabels(tnames, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("energy capture vs task-oracle (%)"); ax.set_xlim(0, 102)
    ax.set_title(f"(B) offline-mask generalization\n@ top-{k0}/{D} per head (frac {f0})")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3, axis="x")

    ax = axes[2]
    ax.plot([r["nmse_sign"] for r in rows], yp, "o-", color="0.5", label="sign only", ms=4)
    ax.plot([r["nmse_cal"] for r in rows], yp, "s-", color="C2", label="calib→fp16", ms=4)
    ax.plot([r["nmse_orc"] for r in rows], yp, "^--", color="C1", label="oracle→fp16", ms=4)
    ax.set_yticks(yp); ax.set_yticklabels(tnames, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("K sign-NMSE"); ax.set_title("(C) NMSE: calib ≈ oracle on held-out tasks")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    fig.suptitle(
        f"Offline $\\sigma^2$-channel calibration (wikitext) → LongBench generalization  —  "
        f"{c.model_basename(model_path)}  |  energy capture {ec_m*100:.1f}% (rand {ec_r*100:.0f}%)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    c.save_fig(fig, args.outdir, f"sigma_calib_{args.tag}.png")


if __name__ == "__main__":
    main()
