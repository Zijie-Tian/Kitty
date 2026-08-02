# experiments/

Research-only experiment scripts that are **not** part of the canonical
LongBench / PPL / RULER entry points (`scripts/run_exp.sh`, `run_ppl.sh`,
`run_ruler.sh`).

| Directory | What it is |
| --- | --- |
| [`sensitivity_overlap/`](sensitivity_overlap/) | QLUTATTN sensitive-channel calibratability (mask overlap M1 + per-layer NF2 quota stability M2). Mask/statistics only — no accuracy claims. |

**Output rule:** every experiment under `experiments/<name>/` writes regenerable
artifacts to repo-root `outputs/<name>/` (gitignored via `/outputs`). Scripts
and READMEs in this tree are tracked; do not create `experiments/<name>_out/`.

Reuse `src/kitty_sim` via `PYTHONPATH=src` (or the scripts' own `sys.path`
insert). Do not vend a second copy of the package here.
