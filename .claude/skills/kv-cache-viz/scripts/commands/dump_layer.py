#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump the K/V cache for a specific layer to fp16 .pt files.

Generates the files that offline viz commands expect:
  outdir/quant_kvcache_analysis/layer{L}_{K,V}_fp16.pt
"""
import argparse
import os

from lib.common import add_prefill_args, load_model, prefill_and_extract, save_layer_kv


def run(argv=None):
    ap = argparse.ArgumentParser()
    add_prefill_args(ap)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument(
        "--all-layers",
        action="store_true",
        help="Dump every layer instead of just --layer",
    )
    a = ap.parse_args(argv)

    outdir = os.path.join(a.outdir, "quant_kvcache_analysis")
    os.makedirs(outdir, exist_ok=True)

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
    print(
        f"[dump-layer] {a.tag}: T={data['T']} layers={data['layers']} "
        f"region={data['region']} D={data['D']} H={data['H']}",
        flush=True,
    )

    if a.all_layers:
        for li in range(data["layers"]):
            save_layer_kv(data["K"], data["V"], outdir, li)
    else:
        save_layer_kv(data["K"], data["V"], outdir, a.layer)


if __name__ == "__main__":
    run()
