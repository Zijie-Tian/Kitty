# sensitivity_overlap — QLUTATTN sensitive-channel calibratability

Mask/statistics experiment: cross-task product-mask overlap (M1) and per-layer
NF2 quota structure stability (M2) for Llama-3.2-1B / 3B.

**Boundary:** this experiment produces **no accuracy evidence**. Overlap /
structure stability argue that offline WikiText-2 calibration is feasible; they
are not an accuracy guarantee and were never scored on the same samples.

Ported from worktree
`.claude/worktrees/eval_design_fig_6` (`worktree-eval_design_fig_6`, 2026-07-28).
This tree reuses `src/kitty_sim` — it does **not** vendor a second package copy.

## Protocol (fixed)

| Knob | Value |
| --- | --- |
| Tasks | 20 LongBench tasks (`multifieldqa_en` excluded: &lt;160 rows) |
| Rows | `32..159` (128 prompts, shared row ids across tasks) |
| Prompt | `dataset2prompt` → llama3.2 chat wrap (except `NO_CHAT`) → ≤2048 middle-truncate |
| Stats | `skip_first=32`; σ² = 128-token block-centered population variance; q̄ = post-RoPE \|q\| mean (GQA-averaged) |
| Mask | ρ=0.62 shortest stable prefix on ω=σ²×q̄ (also σ²-only / q-only variants) |
| Bootstrap | 2000 within-task replicates, seed 0 |

**Excluded task:** `multifieldqa_en` (150 rows &lt; 160).

**TREC-like dual scope (pre-registered):** M1 reports both all-task mean and
mean excluding `trec` / `lsht` / `passage_count`.

## Judgment gates (pre-registered)

| Proposition | Gate | Pass label |
| --- | --- | --- |
| M1 product layer-macro Jaccard mean | ≥ 0.75 **and** ≥ prior 6-task mean − 0.05 (= 0.76) | "maintained" |
| M2 Spearman of per-layer NF2 count vectors | median ≥ 0.8 | "stable" |

### Weak-anchor gate (gate6, 1B)

Recompute on the established 6 tasks
(`qasper,2wikimqa,gov_report,trec,passage_retrieval_en,repobench-p`) and check
against the a30 anchors (3 decimal places):

| Anchor | Expected |
| --- | ---: |
| product mean | 0.810 |
| product pairwise range | 0.723–0.914 |
| σ²-only mean | 0.828 |

## Layout

```text
experiments/sensitivity_overlap/
  collect_task_stats.py   # stage 1: per-task forward → stats.pt
  overlap_metrics.py      # stage 2: masks + Jaccard + Spearman + bootstrap
  plot_m2_figure.py       # stage 3: M2 paper-figure candidate
  run_collection.sh       # 6-lane GPU scheduler
  lanes/lane{0..5}.txt    # 40 (model,task) shards
outputs/sensitivity_overlap/<RUN_TAG>/   # gitignored (/outputs)
```

Paths resolve from `.env` / CLI (`KITTY_LLAMA32_*_PATH`, `LONGBENCH_DATA_ROOT`).
`--data-root` is the LongBench **root** containing `data/<task>.jsonl`
(same as `scripts/run_exp.sh`).

WikiText-2 ρ=0.62 reference masks (already in repo root, not copied here):

- 1B: `autoresearch_llama32_1b_topp_p0p62_seed0_v3.pt`
- 3B: `autoresearch_llama32_3b_topp_p062_seed0.pt`

## Smoke (GPU1, 4 prompts)

```bash
cd "$(git rev-parse --show-toplevel)"
source accuracy_simulation/env.sh
DATA_ROOT="${DATA_ROOT:-${LONGBENCH_DATA_ROOT:-${HOME}/data/LongBench}}"
PY="${PYTHON_BIN:-${KITTY_PYTHON_BIN:-python}}"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  "$PY" experiments/sensitivity_overlap/collect_task_stats.py \
  --model "${KITTY_LLAMA32_1B_PATH}" --model-tag llama32-1b --task qasper \
  --data-root "${DATA_ROOT}" \
  --num-prompts 128 --row-skip 32 --max-len 2048 --skip-first 32 --group-size 128 \
  --limit 4 \
  --out outputs/sensitivity_overlap/smoke/llama32-1b/qasper/
# -> outputs/sensitivity_overlap/smoke/llama32-1b/qasper/stats.pt
```

## Full collection (6 GPUs — explicit GPU1-only override)

```bash
cd "$(git rev-parse --show-toplevel)"
source accuracy_simulation/env.sh
# requires KITTY_LLAMA32_1B_PATH, KITTY_LLAMA32_3B_PATH, and LongBench under
# LONGBENCH_DATA_ROOT or $HOME/data/LongBench
RUN_TAG=sensitivity_overlap_ext
for g in 0 1 2 3 4 5; do
  bash experiments/sensitivity_overlap/run_collection.sh \
    "$g" experiments/sensitivity_overlap/lanes/lane${g}.txt &
done
wait
# -> outputs/sensitivity_overlap/sensitivity_overlap_ext/{llama32-1b,llama32-3b}/<task>/stats.pt
```

Single-GPU full collection (stays on GPU1): edit lanes or loop one lane file
with `run_collection.sh 1 …` (serial, much slower).

## Metrics + M2 figure

```bash
cd "$(git rev-parse --show-toplevel)"
source accuracy_simulation/env.sh
ROOT=outputs/sensitivity_overlap/sensitivity_overlap_ext
METRICS="${ROOT}/metrics"
PY="${PYTHON_BIN:-${KITTY_PYTHON_BIN:-python}}"
mkdir -p "${METRICS}"

# weak-anchor gate6 (1B)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  "$PY" experiments/sensitivity_overlap/overlap_metrics.py \
  --stats-root "${ROOT}" --model-tag llama32-1b \
  --tasks qasper,2wikimqa,gov_report,trec,passage_retrieval_en,repobench-p \
  --rho 0.62 --bootstrap 2000 --seed 0 --device cuda:0 \
  --ref-mask autoresearch_llama32_1b_topp_p0p62_seed0_v3.pt \
  --out-dir "${METRICS}" --tag gate6_llama32-1b

# full 1B / 3B
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  "$PY" experiments/sensitivity_overlap/overlap_metrics.py \
  --stats-root "${ROOT}" --model-tag llama32-1b \
  --rho 0.62 --bootstrap 2000 --seed 0 --device cuda:0 \
  --ref-mask autoresearch_llama32_1b_topp_p0p62_seed0_v3.pt \
  --out-dir "${METRICS}" --tag full_llama32-1b

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  "$PY" experiments/sensitivity_overlap/overlap_metrics.py \
  --stats-root "${ROOT}" --model-tag llama32-3b \
  --rho 0.62 --bootstrap 2000 --seed 0 --device cuda:0 \
  --ref-mask autoresearch_llama32_3b_topp_p062_seed0.pt \
  --out-dir "${METRICS}" --tag full_llama32-3b

"$PY" experiments/sensitivity_overlap/plot_m2_figure.py \
  --metrics-dir "${METRICS}" --tag full_llama32-1b \
  --model-label "Llama-3.2-1B-Instruct" \
  --out "${METRICS}/fig_m2_layer_nf2_ratio_llama32-1b.pdf"

"$PY" experiments/sensitivity_overlap/plot_m2_figure.py \
  --metrics-dir "${METRICS}" --tag full_llama32-3b \
  --model-label "Llama-3.2-3B-Instruct" \
  --out "${METRICS}/fig_m2_layer_nf2_ratio_llama32-3b.pdf"
```

Read `summary_*.json` fields `jaccard_product.mean` (M1) and
`spearman.median` (M2).

## Reusing worktree tensors

If you already have
`.claude/worktrees/eval_design_fig_6/longbench_out/channel_overlap_ext_20260728/{llama32-1b,llama32-3b}/*/stats.pt`,
you can symlink or copy them under
`outputs/sensitivity_overlap/sensitivity_overlap_ext/` and skip collection —
only re-run metrics/plot.

## Prior result (2026-07-28, reference)

| Model | M1 product mean | M2 Spearman median | Verdict |
| --- | ---: | ---: | --- |
| 1B (20 tasks) | 0.8033 | 0.9485 | maintained / stable |
| 3B (20 tasks) | 0.8132 | 0.9318 | maintained / stable |
