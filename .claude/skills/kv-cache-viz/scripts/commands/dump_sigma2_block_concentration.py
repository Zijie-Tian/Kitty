#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is per-channel K sigma^2 "concentrated" within / across 128-token blocks?

Split the quantized region into consecutive 128-token blocks. For each
(layer, head, block, channel) compute the in-block residual variance (var over
the 128 tokens, i.e. submean-then-var = the qlutattn calibration sigma^2 at
G=128, cf. scripts/calibrate_qlutattn_mask.py).
Then two concentration metrics:

  (A) within-block, across channels: for each (head, block), CV of sigma^2 over
      the D channels  -> are different channels' sigma^2 clustered or spread?
  (B) across-block, per channel: for each (head, channel), CV of sigma^2 over
      blocks -> is a channel's sigma^2 a stable quantity at the 128-token scale?

Also dumps the within-block max/min spread (orders of magnitude across channels).
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


def run(argv=None):
    setup_matplotlib()
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    add_prefill_args(ap)
    ap.add_argument("--block", type=int, default=128)
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
    B = a.block
    print(
        f"[prefill] {a.tag}: T={T} layers={nl} D={D} block={B}",
        flush=True,
    )

    cv_within = []  # (A) per (head,block): CV of sigma^2 across channels
    spread_within = []  # (A) per (head,block): max/min of sigma^2 across channels
    cv_across = []  # (B) per (head,channel): CV of sigma^2 across blocks
    example = None
    for li in range(nl):
        K = data["K"][li].to(a.device)  # [nh, Tq, D] on GPU
        nh, Tq, _ = K.shape
        nb = Tq // B
        Kb = K[:, : nb * B, :].reshape(nh, nb, B, D)
        s2 = Kb.var(2)  # [nh, nb, D] in-block residual var
        # (A) within-block across channels
        m_c = s2.mean(2)
        sd_c = s2.std(2)
        cv_within.append((sd_c / m_c.clamp(min=1e-9)).flatten().cpu())
        spread_within.append(
            (s2.amax(2) / s2.amin(2).clamp(min=1e-9)).flatten().cpu()
        )
        # (B) across blocks per channel
        m_b = s2.mean(1)
        sd_b = s2.std(1)
        cv_across.append((sd_b / m_b.clamp(min=1e-9)).flatten().cpu())
        if li == nl // 2:
            example = s2[0].cpu().numpy()  # [nb, D] head0 of mid layer
        del K, Kb, s2
        torch.cuda.empty_cache()

    cvw = torch.cat(cv_within).numpy()
    spw = torch.cat(spread_within).numpy()
    cva = torch.cat(cv_across).numpy()

    stats = {
        "model": a.tag,
        "block": B,
        "layers": nl,
        "D": D,
        "A_within_block_across_channels": {
            "cv_median": float(np.median(cvw)),
            "cv_mean": float(cvw.mean()),
            "spread_maxmin_median": float(np.median(spw)),
            "spread_maxmin_p90": float(np.percentile(spw, 90)),
        },
        "B_across_blocks_per_channel": {
            "cv_median": float(np.median(cva)),
            "cv_mean": float(cva.mean()),
            "cv_p90": float(np.percentile(cva, 90)),
        },
    }
    json.dump(
        stats,
        open(f"{a.outdir}/sigma2_block_concentration_{a.tag}.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs[0, 0].hist(cvw, bins=80, color="#d62728", alpha=0.85, density=True)
    axs[0, 0].axvline(
        np.median(cvw),
        color="k",
        ls="--",
        lw=0.9,
        label=f"median={np.median(cvw):.2f}",
    )
    axs[0, 0].set_title(
        "(A) within-block, ACROSS channels: CV of sigma^2\n"
        "(low = channels' sigma^2 clustered; high = spread)"
    )
    axs[0, 0].set_xlabel("CV across channels (per 128-tok block)")
    axs[0, 0].set_ylabel("density")
    axs[0, 0].legend()
    axs[0, 1].hist(np.log10(spw), bins=80, color="#d62728", alpha=0.85, density=True)
    axs[0, 1].axvline(
        np.log10(np.median(spw)),
        color="k",
        ls="--",
        lw=0.9,
        label=f"median={np.median(spw):.0f}x",
    )
    axs[0, 1].set_title(
        "(A) within-block sigma^2 spread across channels (max/min)"
    )
    axs[0, 1].set_xlabel("log10(max/min sigma^2 within a block)")
    axs[0, 1].set_ylabel("density")
    axs[0, 1].legend()
    axs[1, 0].hist(cva, bins=80, color="#1f77b4", alpha=0.85, density=True)
    axs[1, 0].axvline(
        np.median(cva),
        color="k",
        ls="--",
        lw=0.9,
        label=f"median={np.median(cva):.2f}",
    )
    axs[1, 0].set_title(
        "(B) across-block, PER channel: CV of sigma^2\n"
        "(low = sigma^2 stable over 128-tok blocks)"
    )
    axs[1, 0].set_xlabel("CV across blocks (per channel)")
    axs[1, 0].set_ylabel("density")
    axs[1, 0].legend()
    if example is not None:
        nb_ex = min(example.shape[0], 120)
        rng = np.random.default_rng(0)
        chans = rng.choice(D, size=min(8, D), replace=False)
        for c in chans:
            axs[1, 1].plot(example[:nb_ex, c], lw=0.7)
        axs[1, 1].set_title(
            "(B) example: 8 channels' sigma^2 vs block index (mid layer, head0)"
        )
        axs[1, 1].set_xlabel("128-tok block index")
        axs[1, 1].set_ylabel("in-block sigma^2")
        axs[1, 1].grid(alpha=0.3)
    fig.suptitle(
        f"Is per-channel K sigma^2 concentrated within / across 128-token blocks? "
        f"({a.tag}, all {nl} layers, D={D})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, f"{a.outdir}/sigma2_block_concentration_{a.tag}.png", tight=False)


if __name__ == "__main__":
    run()
