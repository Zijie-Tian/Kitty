#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the lutdecoding-acc-bench methods into one comparison table.

Reads each method's LongBench result.json (21-dataset mean), written by
run_exp.sh after scoring, at:
    longbench_out/[smoke/]<model_slug>_<method_slug>/pred/result.json

and prints + writes (TSV):
    method | variant slug | K bit/value | LongBench avg (21) | retain% vs fp16 | n

Run standalone (driver calls it automatically):
    python collect_lutdecoding_results.py --base longbench_out --layout full \
      --model-slug llama32-1b-instruct
"""
import argparse
import json
import os

# fixed display order; (display, method_slug, default K bit label).
# QLUTATTN's bit column is the K width (nominal 1.75 = 0.5*sign1.25 + 0.5*nf2
# 2.25); its V is the rescued 2-bit tile16c64, like Kitty/KIVI the V width is
# not shown. Q4_0 bit is K=V=4.5 (llama.cpp Q4_0: 32-ch blocks, 4-bit codes +
# fp16 scale).
METHODS = [
    ("F16 FULL", "fp16",                 "16"),
    ("ShadowKV", "shadowkv",             "sparse"),
    ("Kitty",    "kitty-k2b4v2-pr0p125", "~2.5"),
    ("KIVI*-2",  "kivi-star-k2v2",       "2.25"),
    ("KIVI-2",   "kivi-k2v2",            "2.25"),
    ("QLUTATTN", "qlutattn",             "1.75"),
    ("Q4_0",     "llamacpp-q40",         "4.5"),
]


def mean21(path):
    if not os.path.exists(path):
        return None, 0
    d = json.load(open(path))
    s = d.get("scores", d)
    v = [x for x in s.values() if isinstance(x, (int, float))]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="longbench_out")
    ap.add_argument("--layout", choices=["full", "smoke"], default="full")
    ap.add_argument("--model-slug", required=True)
    # --mask kept for driver compatibility; the canonical qlutattn K width is
    # fixed at 1.75 bit so the mask is no longer consulted for the bit column.
    ap.add_argument("--mask", default="")
    args = ap.parse_args()

    root = args.base if args.layout == "full" else os.path.join(args.base, "smoke")

    rows = []
    fp16 = None
    for disp, slug, bit in METHODS:
        sc, n = mean21(os.path.join(root, f"{args.model_slug}_{slug}", "pred", "result.json"))
        if disp == "F16 FULL" and sc is not None:
            fp16 = sc
        rows.append((disp, slug, bit, sc, n))

    print(f"\n=== lutdecoding-acc-bench [{args.layout}]  {args.model_slug} ===")
    print(f"{'method':14s} {'bit':>7s} {'avg21':>7s} {'retain%':>8s} {'n':>3s}  slug")
    print("-" * 68)
    for disp, slug, bit, sc, n in rows:
        if sc is None:
            print(f"{disp:14s} {bit:>7s} {'--':>7s} {'--':>8s} {n:>3d}  {slug}  (MISSING)")
            continue
        ret = f"{100 * sc / fp16:.1f}" if fp16 else "--"
        print(f"{disp:14s} {bit:>7s} {sc:7.2f} {ret:>8s} {n:>3d}  {slug}")

    out = os.path.join(root, f"{args.model_slug}_lutdecoding_bench.tsv")
    with open(out, "w") as f:
        f.write("method\tslug\tKbit\tavg21\tretain_pct\tn\n")
        for disp, slug, bit, sc, n in rows:
            ret = (100 * sc / fp16) if (sc and fp16) else None
            f.write(f"{disp}\t{slug}\t{bit}\t"
                    f"{'' if sc is None else f'{sc:.2f}'}\t"
                    f"{'' if ret is None else f'{ret:.1f}'}\t{n}\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
