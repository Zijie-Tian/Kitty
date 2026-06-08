#!/usr/bin/env python3
# Aggregate full-LongBench per-config averages from res_<dataset>/<tag>.tsv.
# Usage: python3 aggregate.py [tag ...]
import glob, sys
tags = sys.argv[1:] or ["pb2_pr6875", "pb2_pr875", "base2_v4"]
for t in tags:
    sc = {}
    for f in glob.glob("res_*/" + t + ".tsv"):
        ds = f.split("/")[0][4:]
        try:
            sc[ds] = float(open(f).read().rstrip().split("\t")[7])
        except Exception:
            pass
    if sc:
        avg = sum(sc.values()) / len(sc)
        print(f"{t}: full-LB avg={avg:.2f}  n={len(sc)}/21")
        for ds in sorted(sc):
            print(f"    {ds:24s} {sc[ds]}")
    else:
        print(f"{t}: no results yet")
