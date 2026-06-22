#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is a method's LongBench output already COMPLETE?

The driver calls this before each method so it can SKIP a method whose datasets
are all done and only (re)run missing/incomplete ones. run_exp.sh resumes at the
dataset level on its own; this adds a method-level skip that avoids even loading
the model + rescanning when nothing is left to do.

Completeness = for EVERY expected dataset, <pred>/<dataset>.manifest.json exists
with status == "ok" (written_samples >= expected_samples, no failed ids). This is
exactly runner.py's own criterion (longbench/runner.py:793 sets status "ok" iff
total_written == expected_samples and not failed).

Expected dataset list = --datasets csv if given, else the repo's authoritative
LONG_BENCH_DATASETS (imported via PYTHONPATH=src so it never drifts; a vendored
copy is used only if the import fails).

Exit: 0 = complete (driver skips) | 2 = incomplete (driver runs/补测) | 1 = usage.
"""
import argparse
import json
import os
import sys

try:  # authoritative source -- keep in sync automatically
    from kitty_sim.longbench.config import LONG_BENCH_DATASETS as _DEFAULT
    _DEFAULT = list(_DEFAULT)
except Exception:  # vendored fallback = src/kitty_sim/longbench/config.py LONG_BENCH_DATASETS
    _DEFAULT = [
        "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa",
        "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news",
        "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_retrieval_en",
        "passage_count", "passage_retrieval_zh", "lcc", "repobench-p",
    ]


def dataset_ok(pred_dir, d):
    """(ok, reason) for one dataset, per the runner's manifest status."""
    mf = os.path.join(pred_dir, f"{d}.manifest.json")
    if not os.path.exists(mf):
        return False, "no-manifest"
    try:
        m = json.load(open(mf))
    except Exception:
        return False, "bad-manifest"
    exp = m.get("expected_samples")
    wr = m.get("written_samples", 0)
    failed = m.get("failed_sample_ids") or []
    status = m.get("status")
    if not failed and (status == "ok" or (exp is not None and wr >= exp)):
        return True, f"ok {wr}/{exp}"
    return False, f"{status or 'partial'} {wr}/{exp} failed={len(failed)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--datasets", default="", help="csv subset; default = all 21")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    datasets = [x.strip() for x in a.datasets.split(",") if x.strip()] or _DEFAULT

    missing = [(d, why) for d in datasets
               for ok, why in [dataset_ok(a.pred_dir, d)] if not ok]

    if not missing:
        if not a.quiet:
            print(f"  [complete] {len(datasets)}/{len(datasets)} datasets ok")
        return 0
    if not a.quiet:
        done = len(datasets) - len(missing)
        head = ", ".join(f"{d}({why})" for d, why in missing[:8])
        print(f"  [incomplete] {done}/{len(datasets)} ok; need: {head}"
              + (" ..." if len(missing) > 8 else ""))
    return 2


if __name__ == "__main__":
    sys.exit(main())
