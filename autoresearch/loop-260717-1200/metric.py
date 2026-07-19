#!/usr/bin/env python3
"""Extract the loop metric for one config: mean score over the requested datasets.

usage: metric.py <slug> [ds1,ds2,...]   (default qasper,multifieldqa_en)
Prints one float. Exits non-zero if result.json or any dataset is missing.
"""
import json
import sys
from pathlib import Path

slug = sys.argv[1]
datasets = (sys.argv[2] if len(sys.argv) > 2 else "qasper,multifieldqa_en").split(",")
root = Path(__file__).resolve().parents[2]
rj = root / "longbench_out" / f"llama32-1b-rm-{slug}_qlutattn" / "pred" / "result.json"
res = json.loads(rj.read_text())
missing = [d for d in datasets if d not in res]
if missing:
    sys.stderr.write(f"missing datasets in {rj}: {missing}\n")
    sys.exit(1)
vals = [float(res[d]) for d in datasets]
per = "  ".join(f"{d}={v:.2f}" for d, v in zip(datasets, vals))
sys.stderr.write(f"[metric] {slug}: {per}\n")
print(f"{sum(vals) / len(vals):.4f}")
