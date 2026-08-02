# QLUTATTN sign/nf2 mixed-precision Pareto loop (2026-07-17 → 07-19)

Research archive for the `research_mixed` loop that selected the canonical
QLUTATTN K recipe (`sign`/`nf2`, ranking `σ² × E|q|`, fixed 65/35 → 1.60 bit).
Algorithm semantics live in [`docs/qlutattn.md`](qlutattn.md); this page keeps
the **decision evidence** (kept / discarded / rollout numbers).

Source artifacts lived under `autoresearch/loop-260717-1200/`
(`PLAN.md`, `evals-summary.md`, `handoff.json`, `results.tsv`) and have been
removed from the tree; this page is the retained archive.

Status: **COMPLETE** (handoff 2026-07-19). Not a second public variant.

## Goal and protocol

| Knob | Value |
| --- | --- |
| Goal | sign/nf2-only K mixed-precision Pareto; minimize bits at preserved accuracy |
| Fixed | Q FP16; K per-token + prompt `μ_d`; V rescued tile16c64; sink=32, recent=128 |
| Screening metric | `mean(qasper, multifieldqa_en)`, full samples, Llama-3.2-1B |
| Final metric | full LongBench-21 average |
| Scale | 38 logged runs; canonical f50base reproduced screening 25.055 |

## Kept — shipped into the canonical recipe

These two decisions are the public `qlutattn` K selector
([`docs/qlutattn.md`](qlutattn.md) §1):

1. **Ranking signal = `σ² × E|q|`** (MixKVQ-inspired; offline WikiText `|q|`
   probe). Largest single win on screening vs uniform-`σ²` at matched bits:
   about **+4.08 / +3.08 / +1.51** at 1.50 / 1.60 / 1.75 bit.
   Ablation at 1.50 bit: `σ²`-only ≈ 22.45 ≈ `|q|`-only ≈ 22.38 ≪ product ≈ 26.53.
   Seed-robust (screening Δ ≈ 0.29 across probe seeds).
2. **Interior optimum ≈ 35% nf2 / 65% sign → 1.60 bit/value.** More nf2 past
   that point hurts: sign + mean-|r| fits low-difficulty channels better than
   `symnf2-v1` + absmax. Screening peak: **q65 = 27.795** vs pure-nf2 f00 =
   25.49 @ 2.25 bit and uniform-`σ²` f50base = 25.055 @ 1.75 bit.

## Kept — validated but not the public default

These were positive in the loop but are **not** the shipped public algorithm
(token tiering is explicitly excluded in [`qlutattn.md`](qlutattn.md); band is
a conditional Qwen patch, not a universal default):

3. **Token tiering (2D, `ttm150`)** — observation-window (last-64 queries)
   per-head top-ρ promotion; hi tier must anchor at the channel-optimal mask.
   Screening 27.22 @ 1.50 bit; full-21 ≈ 24.52 (parity with q65 at −0.25 bit vs
   old 1.75 bit canonical). Extends the frontier below the channel-only
   optimum (cliff ~1.40 bit).
4. **Band / per-head exclusion for pathological models (Qwen3)** — extreme
   QK-norm channels poison per-token absmax (reported amplitude ratio ~52× vs
   ~3× on Llama). Conditional only: costs about **−0.89** on healthy Llama-1B.
   Diagnose first via nf2-bin amplitude ratio (CPU-side).

## Discarded (confirmed negative on screening)

| Idea | Approx. screening delta vs matched control |
| --- | ---: |
| Per-head equalized ranking | −1.2 |
| RoPE-pair binding `(i, i+D/2)` | −1.15 |
| Layer sign-frac schedules under the q-signal | −1.07 |
| Token-tier hi = all-nf2 | −2.9 |
| `|q|` tempering α=0.5 | lost QA gains; no zh/code recovery |
| Hi tier above the channel optimum (`ttm160`) | −0.98 |

## Full LongBench-21 numbers

Llama-3.2-1B (same harness):

| Config | K bit (nominal) | Full-21 |
| --- | ---: | ---: |
| canonical f50 (`σ²` 50/50) | 1.75 | 24.23 |
| **q65** (`σ²×E\|q\|`, 65/35) | **1.60** | **24.53** |
| ttm150 (2D token tier) | 1.50 | 24.52 |
| f65 uniform `σ²` control | 1.60 | 24.28 |
| f75 uniform `σ²` control | 1.50 | 23.78 |
| q75 | 1.50 | 24.27 |

Cross-model rollout (q65 @ 1.60 vs prior canonical @ 1.75):

| Model | Canonical | q65 | Δ |
| --- | ---: | ---: | ---: |
| Llama-3.2-1B | 24.23 | 24.53 | +0.30 |
| MiniCPM5-1B | 18.06 | 18.23 | +0.17 |
| Llama-3.2-3B | 34.24 | 34.30 | +0.06 |

ttm150 @ ~1.50 bit held near parity on the same three
(24.52 / 18.10 / 34.18).

Qwen3-4B (collapse + partial repair; structural fixed-LUT misfit vs Lloyd-era
reference remains open):

| Config | Full-21 |
| --- | ---: |
| q65, no band | 12.18 (collapse) |
| q65 + band 0.05 | 24.42 |
| per-head R3 | 25.83 |
| band 0.10 | 26.88 |
| band 0.20 | stopped mid-run |

## Meta observations

- Screening → full dilution ≈ **12×**: gains concentrate in QA/retrieval;
  code/zh often regress (WikiText `|q|` domain bias). Mixed-corpus calibration
  was left as future work.
- Absolute gain shrinks with model size (1B +0.30 → MiniCPM +0.17 → 3B +0.06
  at −0.15 bit).
- Headline under this protocol: about **−0.15 bit** (q65) at parity-or-better
  on llama-family models; **−0.25 bit** if counting ttm150 (not public default).

## What this is not

- Not a claim that token tiering or Qwen band are part of public `qlutattn`.
- Not the later top-p selector study (2026-07-23); that lives in
  [`qlutattn.md`](qlutattn.md) §6.
- Accuracy proxy only (fake-quant); no packed-memory or speed evidence.
