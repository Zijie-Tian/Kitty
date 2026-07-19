# research_mixed loop — evals summary (2026-07-17 → 07-19)

Goal: sign/nf2-only K-cache mixed-precision Pareto (per-token mechanism & V fixed).
Metric: screening = mean(qasper, multifieldqa_en) full samples; final = full LongBench-21.
38 logged runs; 8 experiment commits; guard (wiring tests) green throughout; canonical path bit-identical (f50base reproduced 25.055 = pre-refactor per-dataset values).

## Kept (shipped into the final recipe)
1. **σ²×E|q| channel ranking** (MixKVQ-inspired, offline wikitext |q| probe) — the single
   biggest win: +4.08/+3.08/+1.51 @1.5/1.6/1.75b (screening). Ablation: σ²-only 22.45 ≈
   |q|-only 22.38 << product 26.53. Seed-robust (Δ0.29).
2. **Interior optimum ~35% nf2 (1.6b)** — more nf2 beyond it HURTS (sign+mean|r| fits
   low-difficulty channels better than symnf2+absmax). Final sign fraction: 0.65.
3. **Token tiering (2D)** — observation-window (last-64 queries) per-head top-ρ promotion,
   hi tier MUST anchor at the channel-optimal mask: ttm150 27.22 @1.5b (screen), 24.52 full.
   Extends the frontier below the channel optimum (cliff 1.40b: 25.23 vs 23.63).
4. **Band / per-head exclusion for pathological models (Qwen3)** — conditional patch only
   (costs −0.89 on healthy Llama-1B). Diagnose first: nf2-bin amplitude ratio.

## Discarded (all confirmed negative)
per-head equalized ranking (−1.2) · rope-pair binding (−1.15) · layer-frac schedules under
the q signal (−1.07) · hi=all-nf2 token tier (−2.9) · |q|-power tempering α=0.5 (lost QA
gains, no zh/code recovery) · hi tier above the channel optimum (ttm160 −0.98).

## Full-21 final numbers
Llama-1B: canonical 24.23@1.75 | q65 24.53@1.60 | ttm150 24.52@1.50 | f65 24.28 | f75 23.78 | q75 24.27
MiniCPM5-1B: canonical 18.06 | q65 18.23@1.60 | ttm150 18.10@1.49
Llama-3B: canonical 34.24 | q65 34.30@1.60 | ttm150 34.18@1.50
Qwen3-4B: no-band 12.18 (collapse) | band05 24.42 | R3 25.83 | band10 26.88 | band20 in flight
          (Lloyd-era reference 43.92; fixed-LUT symnf2 misfit = structural, exclusion is partial)

## Trends / meta
- Screening→full dilution ≈ 12×: gains concentrate in QA/retrieval, regress on code/zh
  (wikitext |q| domain bias — mixed-corpus calibration is the open fix).
- Gain shrinks with model size: 1B +0.30, MiniCPM +0.17, 3B +0.06 (at −0.15 bit).
- Headline: −0.25 bit at parity-or-better across all llama-family models (1.50b ≈ old 1.75b).
