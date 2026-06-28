#!/usr/bin/env bash
# Unified entry point for the kv-cache-viz skill.
#
# Usage:
#   bash .claude/skills/kv-cache-viz/scripts/run_kv_cache_viz.sh \
#        probe mu2sigma2 --model ... --longbench-dir ... --tag llama32-1b
#
# The skill may also be invoked via the slash command /kv-cache-viz.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$PWD}"

export PYTHONPATH="${SCRIPT_DIR}:${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

python "${SCRIPT_DIR}/kv_cache_viz.py" "$@"
