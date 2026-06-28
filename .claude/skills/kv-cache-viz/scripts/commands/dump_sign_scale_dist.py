#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump the per-token SIGN-group scale (mag) of snf-pt K-quant and test whether it
can be shared across 16-token blocks (a per-tensor scale over each block).

snf-pt's sign scale is, per (head, token):
    mag = mean_{ch in sign-group} |r|,   r = K - per_channel_mean
where the sign-group = the LOW-sigma^2 half of the head_dim channels (the offline
split: low sigma^2 -> sign). This reproduces kitty_simulate._pt_codebook_masked's
sign branch (mag = masked mean of |r| over the bin's channels) exactly.

We dump mag[L, nh, Tq] and quantify, averaged over layers/heads:
  (1) smoothness of mag along tokens: global CV, median adjacent rel-step,
      CV within each 16-token block;
  (2) extra sign-reconstruction NMSE if 16 tokens SHARE one scale (block-mean
      or block-max) instead of one scale per token;
  (3) extra error if that per-16-token scale is further quantized to 8/4 bit.

Outputs: stats_<tag>.json, sign_scale_<tag>.png, mag_<tag>.pt (for re-analysis).
"""
import argparse
import json
import os

import numpy as np
import torch

from lib.common import (
    add_prefill_args,
    load_model,
    prefill_and_extract,
    save_fig,
    setup_matplotlib,
)


def sign_nmse(r, m, mag):
    # r:[nh,Tq,D], m:[nh,D] bool, mag:[nh,Tq] -> NMSE over sign-group channels
    mb = m[:, None, :]
    rec = torch.sign(r) * mag[:, :, None]
    num = (((r - rec) ** 2) * mb).sum()
    den = ((r ** 2) * mb).sum().clamp(min=1e-9)
    return (num / den).item()


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    add_prefill_args(ap)
    ap.add_argument("--sign-frac", type=float, default=0.5)
    ap.add_argument("--block", type=int, default=16)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    model, tok = load_model(a.model, a.device)
    data = prefill_and_extract(
        model,
        tok,
        os.path.expanduser(a.longbench_dir),
        seq_len=a.seq_len,
        sink=a.sink,
        recent=a.recent,
        chunk=a.chunk,
        device=a.device,
    )
    T = data["T"]
    nl = data["layers"]
    D = data["D"]
    q0, q1 = data["region"]
    print(f"[prefill] {a.tag}: T={T} layers={nl} D={D}", flush=True)

    f, blk = a.sign_frac, a.block

    mags = []
    cv_glob, cv_blk, relstep = [], [], []
    nmse_pt, nmse_mean, nmse_max, nmse_q8, nmse_q4 = [], [], [], [], []
    for li in range(nl):
        K = data["K"][li]  # [nh, Tq, D]
        nh, Tq, _ = K.shape
        mu = K.mean(1, keepdim=True)  # [nh, 1, D] per-channel mean
        r = K - mu  # residual snf-pt quantizes
        sig2 = r.var(1)  # [nh, D] per-channel residual var
        k = max(1, int(round(f * D)))
        order = torch.argsort(sig2, dim=1)  # ascending sigma^2
        m = torch.zeros(nh, D, dtype=torch.bool)
        m.scatter_(1, order[:, :k], True)  # sign-group = low-sigma^2 k channels
        mb = m[:, None, :]
        cnt = m.sum(1).clamp(min=1)[:, None].float()  # [nh, 1]
        mag = (r.abs() * mb).sum(2) / cnt  # [nh, Tq]  <-- THE per-token sign scale
        mags.append(mag.half())

        cv_glob.append((mag.std(1) / mag.mean(1).clamp(min=1e-9)).mean().item())
        dif = (mag[:, 1:] - mag[:, :-1]).abs() / mag[:, :-1].clamp(min=1e-9)
        relstep.append(dif.median().item())

        nb = Tq // blk
        Tt = nb * blk
        magb = mag[:, :Tt].reshape(nh, nb, blk)
        cv_blk.append((magb.std(2) / magb.mean(2).clamp(min=1e-9)).mean().item())

        rT = r[:, :Tt, :]
        nmse_pt.append(sign_nmse(rT, m, mag[:, :Tt]))
        bm = magb.mean(2, keepdim=True).expand(-1, -1, blk).reshape(nh, Tt)
        nmse_mean.append(sign_nmse(rT, m, bm))
        bx = magb.amax(2, keepdim=True).expand(-1, -1, blk).reshape(nh, Tt)
        nmse_max.append(sign_nmse(rT, m, bx))
        for bits, acc in [(8, nmse_q8), (4, nmse_q4)]:
            bmean = magb.mean(2)  # [nh, nb] one scale per block
            smax = bmean.amax(1, keepdim=True).clamp(min=1e-9)
            L = 2 ** bits - 1
            qb = (bmean / smax * L).round().clamp(0, L) / L * smax
            qexp = qb[:, :, None].expand(-1, -1, blk).reshape(nh, Tt)
            acc.append(sign_nmse(rT, m, qexp))

    torch.save(
        {"mag": mags, "D": D, "sign_frac": f, "block": blk, "tag": a.tag},
        f"{a.outdir}/mag_{a.tag}.pt",
    )

    def avg(x):
        return float(np.mean(x))

    stats = {
        "tag": a.tag,
        "model": a.model,
        "T": T,
        "Tq": q1 - q0,
        "layers": nl,
        "D": D,
        "sign_frac": f,
        "block": blk,
        "scale_def": "mag = mean_{ch in low-sigma2 half} |K - per_channel_mean|, per (head,token)",
        "cv_global_mean": avg(cv_glob),
        "rel_adjacent_step_median": avg(relstep),
        "cv_within_16tok_block_mean": avg(cv_blk),
        "sign_recon_nmse": {
            "per_token": avg(nmse_pt),
            "block16_mean": avg(nmse_mean),
            "block16_max": avg(nmse_max),
            "block16_then_8bit": avg(nmse_q8),
            "block16_then_4bit": avg(nmse_q4),
        },
        "nmse_rel_increase_block16_mean_vs_pertoken": avg(nmse_mean)
        / max(avg(nmse_pt), 1e-9)
        - 1,
        "per_layer": {
            "cv_global": cv_glob,
            "cv_block16": cv_blk,
            "nmse_pt": nmse_pt,
            "nmse_block16_mean": nmse_mean,
        },
    }
    json.dump(
        stats,
        open(f"{a.outdir}/stats_{a.tag}.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    print(
        json.dumps(
            {
                k: stats[k]
                for k in [
                    "cv_global_mean",
                    "rel_adjacent_step_median",
                    "cv_within_16tok_block_mean",
                    "sign_recon_nmse",
                    "nmse_rel_increase_block16_mean_vs_pertoken",
                ]
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for li, col in [(0, "#1f77b4"), (nl // 2, "#2ca02c"), (nl - 1, "#d62728")]:
        m0 = mags[li][0].float().numpy()
        axs[0, 0].plot(m0[:2000], col, lw=0.5, label=f"L{li} h0")
    axs[0, 0].set_title("sign scale (mag) along tokens [first 2000]")
    axs[0, 0].set_xlabel("token")
    axs[0, 0].set_ylabel("mag")
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)
    allm = np.concatenate([mags[li][0].float().numpy() for li in range(nl)])
    axs[0, 1].hist(allm, bins=80, color="#1f77b4", alpha=0.8, density=True)
    axs[0, 1].set_title(f"mag distribution, head0 (CV_global={avg(cv_glob):.2f})")
    axs[0, 1].set_xlabel("mag")
    axs[0, 1].set_ylabel("density")
    x = np.arange(nl)
    axs[1, 0].plot(x, cv_blk, "o-", color="#9467bd", label="CV within 16-tok block")
    axs[1, 0].plot(x, cv_glob, "s-", color="#8c564b", label="CV global")
    axs[1, 0].set_title("scale variability: 16-tok block vs global")
    axs[1, 0].set_xlabel("layer")
    axs[1, 0].set_ylabel("CV")
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)
    labels = ["per-token", "16blk-mean", "16blk-max", "16blk+8b", "16blk+4b"]
    vals = [avg(nmse_pt), avg(nmse_mean), avg(nmse_max), avg(nmse_q8), avg(nmse_q4)]
    axs[1, 1].bar(
        labels,
        vals,
        color=["#2ca02c", "#1f77b4", "#1f77b4", "#ff7f0e", "#d62728"],
    )
    axs[1, 1].set_title("sign-recon NMSE: per-token vs 16-tok shared scale")
    axs[1, 1].set_ylabel("NMSE")
    axs[1, 1].tick_params(axis="x", rotation=20)
    fig.suptitle(
        f"snf-pt sign per-token scale study — {a.tag} "
        f"(D={D}, sign-frac={f}, T={T})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, f"{a.outdir}/sign_scale_{a.tag}.png", tight=False)


if __name__ == "__main__":
    run()
