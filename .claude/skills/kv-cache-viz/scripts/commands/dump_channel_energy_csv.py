#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump per-channel post-RoPE K energy statistics to CSV across ALL layers.

For each (layer, head, channel) over the quantized region [sink, T-recent):
  mu       = per-channel mean over tokens
  mu2      = mu^2                (DC energy)
  sigma2   = per-channel variance over tokens (AC energy = energy after submean)
  energy   = mu2 + sigma2        (total per-channel energy, identity)
  dc_share = mu2 / energy        (how DC-dominated the channel is)
  class    = 1 DC-dominated (dc_share>0.5) | 2 sigma2-low | 3 sigma2-high
             (2/3 split at the per-(layer,head) sigma2 median among non-DC channels,
              matching qlutattn's per-head sigma2 ranking)

This is the per-channel detail behind the qlutattn "energy view" 3-way split.
"""
import argparse
import collections
import csv
import json
import os

import numpy as np
import torch

from lib.common import (
    add_prefill_args,
    load_model,
    prefill_and_extract,
)


def run(argv=None):
    ap = argparse.ArgumentParser()
    add_prefill_args(ap)
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
    print(
        f"[prefill] {a.tag}: T={T} layers={nl} D={D} region=[{q0},{q1})",
        flush=True,
    )

    rows = []
    for li in range(nl):
        K = data["K"][li]  # [nh, Tq, D]
        nh, Tq, _ = K.shape
        mu = K.mean(1)
        var = K.var(1)
        E = mu ** 2 + var
        dc = (mu ** 2) / E
        for h in range(nh):
            nondc = dc[h] <= 0.5
            vmed = var[h][nondc].median() if nondc.any() else var[h].median()
            for c in range(D):
                cls = 1 if dc[h, c] > 0.5 else (2 if var[h, c] <= vmed else 3)
                rows.append(
                    (
                        li,
                        h,
                        c,
                        float(mu[h, c]),
                        float(mu[h, c] ** 2),
                        float(var[h, c]),
                        float(E[h, c]),
                        float(dc[h, c]),
                        cls,
                    )
                )

    out_path = os.path.join(a.outdir, f"channel_energy_{a.tag}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "layer",
                "head",
                "channel",
                "mu",
                "mu2",
                "sigma2",
                "energy",
                "dc_share",
                "class",
            ]
        )
        for li, h, c, mu_, mu2_, s2_, e_, dcv, cls in rows:
            w.writerow(
                [
                    a.tag,
                    li,
                    h,
                    c,
                    f"{mu_:.6f}",
                    f"{mu2_:.6f}",
                    f"{s2_:.6f}",
                    f"{e_:.6f}",
                    f"{dcv:.6f}",
                    cls,
                ]
            )

    cc = collections.Counter(r[8] for r in rows)
    n = len(rows)
    print(f"[csv] {out_path}  rows={n} (layers={nl} heads={nh} D={D})")
    print(
        f"class: (1)DC-dominated={cc[1]} ({100*cc[1]/n:.0f}%)  "
        f"(2)sigma2-low={cc[2]} ({100*cc[2]/n:.0f}%)  "
        f"(3)sigma2-high={cc[3]} ({100*cc[3]/n:.0f}%)",
        flush=True,
    )


if __name__ == "__main__":
    run()
