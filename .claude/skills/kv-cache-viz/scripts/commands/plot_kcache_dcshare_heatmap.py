#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D heatmaps of per-(KV-head, channel) DC-energy share dc_share=mu^2/(mu^2+sigma^2)
of the post-RoPE K-cache, for several layers (one per layer + a grid)."""
import argparse
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
    ap.add_argument("--layers", default="")
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--cache", choices=["K", "V"], default="K")
    ap.add_argument("--cmap", default="coolwarm")
    ap.add_argument(
        "--from-npz",
        default="",
        help="re-plot from a saved matrices .npz (skip the model entirely)",
    )
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    if a.from_npz:
        z = np.load(a.from_npz, allow_pickle=False)
        sel = [int(x) for x in z["sel"]]
        mats = z["mats"]
        dcs = {li: mats[k] for k, li in enumerate(sel)}
        D = int(z["D"])
        nh = int(z["nh"])
        cache = str(z["cache"])
        model_tag = str(z["model_tag"])
        print(
            f"[from-npz] {a.from_npz}  layers={sel} D={D} nh={nh} cache={cache} (no model loaded)",
            flush=True,
        )
    else:
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
        H = data["H"]
        q0, q1 = data["region"]
        print(
            f"[prefill] T={T} layers={nl} D={D} region=[{q0},{q1}) cache={a.cache}",
            flush=True,
        )
        sel = (
            [int(x) for x in a.layers.split(",") if x.strip()]
            if a.layers.strip()
            else sorted(
                set(np.linspace(0, nl - 1, a.num_layers).round().astype(int).tolist())
            )
        )

        def getmat(li):
            X = data["K"][li] if a.cache == "K" else data["V"][li]
            # X is [H, T_region, D] from prefill_and_extract
            mu = X.mean(1)  # [H, D]
            var = X.var(1)  # [H, D]
            return ((mu ** 2) / (mu ** 2 + var + 1e-12)).cpu().numpy()  # [H, D] in [0,1]

        dcs = {li: getmat(li) for li in sel}
        nh = next(iter(dcs.values())).shape[0]
        npz = os.path.join(a.outdir, f"dcshare_{a.cache}_matrices_{a.tag}.npz")
        np.savez(
            npz,
            mats=np.stack([dcs[li] for li in sel]),
            sel=np.array(sel, dtype=int),
            D=D,
            nh=nh,
            cache=a.cache,
            model_tag=a.tag,
        )
        print(
            f"[saved] matrices -> {npz}  (re-plot any cmap/layout with --from-npz {npz}, no model)",
            flush=True,
        )
        cache = a.cache
        model_tag = a.tag

    for li in sel:
        dc = dcs[li]
        print(
            f"  L{li:2d}: DC>0.5={100*(dc>0.5).mean():5.1f}%  median={np.median(dc):.3f}",
            flush=True,
        )

    CMAP = a.cmap

    def plot_one(dc, li, out):
        fig, ax = plt.subplots(figsize=(max(6.5, D * 0.11), nh * 0.42 + 1.8))
        im = ax.imshow(dc, vmin=0, vmax=1, cmap=CMAP, aspect="auto")
        ax.set_xlabel("channel (head_dim index)")
        ax.set_ylabel("KV head")
        ax.set_yticks(range(nh))
        ax.set_title(
            f"{cache}-cache dc_share mu^2/(mu^2+sigma^2)  {model_tag} L{li}  DC>0.5={100*(dc>0.5).mean():.1f}%",
            fontsize=9,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("dc_share")
        save_fig(fig, out, dpi=130)

    for li in sel:
        plot_one(
            dcs[li],
            li,
            os.path.join(a.outdir, f"dcshare_{cache}_L{li:02d}_{model_tag}.png"),
        )

    ncol = int(np.ceil(len(sel) ** 0.5))
    nrow = (len(sel) + ncol - 1) // ncol
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(ncol * 4.4, nrow * 3.4), squeeze=False, constrained_layout=True
    )
    im = None
    for k, li in enumerate(sel):
        ax = axs[k // ncol][k % ncol]
        im = ax.imshow(dcs[li], vmin=0, vmax=1, cmap=CMAP, aspect="auto")
        ax.set_title(f"L{li} DC>0.5:{100*(dcs[li]>0.5).mean():.0f}%", fontsize=9)
        ax.set_xlabel("channel", fontsize=8)
        ax.set_ylabel("KV head", fontsize=8)
        ax.set_yticks(range(nh))
        ax.tick_params(labelsize=6)
    for k in range(len(sel), nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    fig.suptitle(
        f"{cache}-cache per-(head,channel) dc_share across layers ({model_tag}, D={D})",
        fontsize=12,
    )
    fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.025, pad=0.02).set_label("dc_share")
    try:
        fig.get_layout_engine().set(hspace=0.22, wspace=0.06)
    except Exception:
        pass
    out = os.path.join(a.outdir, f"dcshare_{cache}_grid_{model_tag}.png")
    save_fig(fig, out, dpi=130, tight=False, bbox_inches="tight")
    print(f"[done] -> {a.outdir}", flush=True)


if __name__ == "__main__":
    run()
