#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified CLI for the kv-cache-viz skill.

Usage:
  kv-cache-viz probe  <name> [args...]
  kv-cache-viz viz    <name> [args...]
  kv-cache-viz dump-layer [args...]

Examples:
  kv-cache-viz probe mu2sigma2 --model ... --longbench-dir ... --tag llama32-1b
  kv-cache-viz dump-layer --layer 8 --model ... --longbench-dir ...
  kv-cache-viz viz channel-dist --layer 8 --head 0 --tag llama32-1b
"""
import importlib.util
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMANDS_DIR = os.path.join(SCRIPT_DIR, "commands")

# CLI name -> module base name (without dump_/plot_ prefix).  Use aliases so
# command names are short and stable even if the underlying script name varies.
PROBE_ALIASES = {
    "channel-energy": "channel_energy_csv",
    "mu2sigma2": "kv_mu2_sigma2_dist",
    "nf2-pertoken-maxlevel": "nf2_pertoken_maxlevel",
    "sigma2-block-concentration": "sigma2_block_concentration",
    "signpt-dequant": "signpt_dequant_dist",
    "sign-scale": "sign_scale_dist",
    "v-grouping-error": "v_grouping_error",
}

VIZ_ALIASES = {
    "channel-dist": "kv_channel_dist",
    "channel-dist-multi": "kcache_channel_dist_multi",
    "dcshare-heatmap": "kcache_dcshare_heatmap",
    "decomp": "k_decomp_steps",
    "codebook-mu2sigma2": "codebook_mu2_sigma2",
    "e2e-perf": "e2e_perf_bars",
    "kcache-pareto": "kcache_pareto",
    "kcache-pareto-combined": "kcache_pareto_combined",
    "kcache-reorder-2d": "kcache_reorder_2d",
    "kitty-kv-heatmap": "kitty_kv_heatmap",
    "kivistar-heatmap": "kivistar_heatmap",
    "kv-3d-submean": "kv_3d_submean",
    "kv-channel-dist-layers": "kv_channel_dist_layers",
    "kv-memory-bars": "kv_memory_bars",
    "kv-seg-mu2sigma2-3d": "kv_seg_mu2sigma2_3d",
    "pertoken-smooth": "pertoken_smooth",
    "qlutattn-energy-quest1024": "qlutattn_energy_quest1024",
    "reorder-mixed-codebook": "reorder_mixed_codebook",
    "why-sign-beats-minmax": "why_sign_beats_minmax",
    "attn-op-latency": "attn_op_latency_bars",
}


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(f"kv_viz_cmd_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_base(command_type: str, name: str) -> str:
    aliases = PROBE_ALIASES if command_type == "probe" else VIZ_ALIASES
    if name in aliases:
        return aliases[name]
    return name.replace("-", "_")


def _find_module(command_type: str, name: str):
    """Map a CLI name to a command module path."""
    base = _resolve_base(command_type, name)
    if command_type == "probe":
        candidates = [f"dump_{base}.py", f"{base}.py"]
    elif command_type == "viz":
        candidates = [f"plot_{base}.py", f"{base}.py"]
    else:
        candidates = [f"{base}.py"]
    for c in candidates:
        p = os.path.join(COMMANDS_DIR, c)
        if os.path.exists(p):
            return p
    return None


def _usage():
    print("usage: kv-cache-viz <probe|viz|dump-layer> <name> [args...]")
    print("       kv-cache-viz dump-layer [args...]")
    print("")
    print("probe commands (require GPU + LongBench data):")
    print("  " + ", ".join(PROBE_ALIASES.keys()))
    print("")
    print("viz commands (offline plots):")
    print("  " + ", ".join(VIZ_ALIASES.keys()))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _usage()
        sys.exit(0)

    command_type = argv[0]
    if command_type == "dump-layer":
        mod_path = _find_module("dump-layer", "dump_layer")
        rest = argv[1:]
    else:
        if len(argv) < 2:
            _usage()
            sys.exit(1)
        name = argv[1]
        mod_path = _find_module(command_type, name)
        rest = argv[2:]
        if mod_path is None:
            print(f"[err] unknown command: {command_type} {name}")
            _usage()
            sys.exit(1)

    mod_name = "dump_layer" if command_type == "dump-layer" else _resolve_base(command_type, name)
    mod = _load_module(mod_path, mod_name)
    if not hasattr(mod, "run"):
        print(f"[err] {mod_path} has no run() function")
        sys.exit(1)
    mod.run(rest)


if __name__ == "__main__":
    main()
