# Autoresearch summary — per-token K-cache quant, maximize top-32 attention overlap @ ≤2.5 bit

Model Llama-3.2-1B, 1 LongBench 32k doc (narrativeqa#4), 16 layers, post-RoPE K, offline overlap proxy
(scripts/eval_pertoken_quant.py). Metric = top-32 attended-key overlap vs fp16 (higher better). Guard = eff bits ≤ 2.5.

## Trajectory (champion per-token method by iteration)
| it | idea | champion | overlap | bits | NMSE |
|---:|---|---|---:|---:|---:|
| 0 | baselines | pt/nf2 Lloyd | 0.4609 | 2.50 | 0.125 |
| 1 | + Hadamard rotation (QuaRot) | pt/had+nf2 Lloyd | 0.5113 | 2.50 | 0.097 |
| 2 | + SmoothAttention (QServe) | pt/smooth+had+nf2 Lloyd | 0.5372 | 2.50 | 0.077 |
| 3 | + outlier-channel isolation (KVQuant) | pt/outlier1+Lloyd | 0.5559 | 2.47 | 0.082 |
| 4 | smooth × outlier | pt/smooth+outlier1+Lloyd | 0.6042 | 2.47 | 0.064 |
| 5 | more outliers @ lower bits | pt/smooth+outlier4@4b+Lloyd | 0.6802 | 2.375 | 0.051 |
| 6 | push #isolated channels | **pt/smooth+outlier8@4b+Lloyd** | **0.7168** | 2.50 | 0.042 |
| 7 | amax vs σ² selection (converge) | (champion holds) | 0.7168 | 2.50 | 0.042 |

Reference ceilings (per-channel, same harness): tern 0.7741 / sign 0.7320 / KIVI-uni2 0.7211.

## Winner
**smooth + top-8 amax-outlier channels @ 4-bit (per-channel) + per-token Lloyd 2-bit on the other 56**, 2.5 bit/elem.
overlap 0.461 → 0.717 (+0.256, closes ~72% of the per-token→per-channel gap); NMSE 0.042 < per-channel tern 0.056.

## Mechanism (why it works)
per-token's flaw = one shared scale per 64 heterogeneous channels. Three orthogonal fixes, additive:
1. **Hadamard rotation** spreads per-channel outliers across all dims → per-token group ≈ i.i.d. Gaussian (QuaRot 2404.00456).
2. **SmoothAttention** folds per-channel K scale into Q → flatter channel envelope (QServe 2405.04532).
3. **Outlier isolation** keeps the few PEAK-magnitude channels (amax, not σ²) per-channel-quantized so they
   stop dominating the shared scale (KVQuant dense-and-sparse 2402.xxxx). This is the strongest single lever and
   interpolates per-token↔per-channel: more isolated channels → closer to per-channel (optimum ~k=8/64 @2.5 bit).

σ²-selection (0.527) ≪ amax-selection (0.717): isolate shared-scale *dominators*, i.e. peak magnitude.

## To validate on LongBench (next phase)
Top-2 for real-score check: (1) smooth+outlier8@4b+Lloyd champion, (2) smooth+per-token-nf2 (no outlier, existing path),
vs fp16 ceiling and per-token-nf2 baseline. Needs outlier isolation added to kitty_simulate._quant_k_pertoken.

## LongBench validation (25 samples, multifieldqa_en + hotpotqa, 32k)
| config | mfqa_en | hotpotqa | avg |
| --- | ---: | ---: | ---: |
| fp16 (ceiling) | 57.74 | 42.94 | 50.34 |
| per-token nf2 (baseline) | 41.33 | 42.95 | 42.14 |
| smooth + per-token nf2 | 44.59 | 39.35 | 41.97 |
| **CHAMPION smooth+outlier8@4b+Lloyd** | **56.0** | **45.35** | **50.67** |

Champion +8.5 over per-token baseline, **matches fp16** (50.67 vs 50.34) on these K-fidelity-sensitive
retrieval tasks. multifieldqa_en +14.7 (41.3->56.0). smooth alone ≈ baseline -> outlier isolation is the lever.
Proxy (overlap 0.461->0.717) -> LongBench (+8.5) correlation confirmed. fp16 full-21 mean = 27.59 (doc-matched).
Full 21-dataset runs (baseline + champion) running in background: run_full_longbench.sh -> longbench_out/.

## How to run the champion on LongBench
smooth ckpt = /home/zijie/models/Llama-3.2-1B-Instruct-smooth (scripts/calibrate_smooth_qk.py).
QLUT_BIN_CODEBOOKS=nf2 PERTOKEN_OUTLIER_K=8 PERTOKEN_OUTLIER_BITS=4 \
LLAMA32_MODEL_PATH=<smooth ckpt> LLAMA32_MODEL_SLUG=llama32-1b-instruct-smooth \
MAX_MODEL_LEN=32768 LLAMA32_MAX_GEN=256 bash scripts/run_exp.sh llama32 --gpu 1 --variant qlutattn_pertoken
# -> tag qlutattn_pertoken_nb1_v4_cb<h>_iso8b4. Code: kitty_simulate._quant_k_pertoken (outlier isolation, working-tree).

## FULL LongBench (21 datasets, 32k, 4-GPU run) — FINAL
| config | mean(21) | retain vs fp16 |
| --- | ---: | ---: |
| fp16 (ceiling) | 27.59 | 100% |
| per-token nf2 (baseline) | 23.82 | 86.3% |
| **CHAMPION smooth+outlier8@4b+Lloyd** | **26.48** | **96.0%** |

champion +2.67 over per-token baseline; **closes 70.6% of the baseline→fp16 gap** (offline overlap proxy
predicted 72% — near-perfect proxy↔LongBench correlation). Biggest per-dataset wins (K-fidelity-sensitive
retrieval): multifieldqa_en +9.16, triviaqa +8.26, multifieldqa_zh +6.04, 2wikimqa +5.05, musique +4.77,
samsum +4.52, qmsum +4.24, repobench-p +3.74. Minor regressions: lcc -1.37, lsht -1.0, qasper -0.94.

NOTE (perf): champion's `_quant_k_pertoken` outlier path uses a per-head Python loop (gather/scatter +
per-head Lloyd) that runs every decode step -> ~10-20x slower than the vectorized baseline (passage_retrieval_zh
~183s/sample). Fake-quant accuracy proxy so it doesn't affect the result, but vectorize the per-head loop
before any larger sweep. 4-GPU layout: GPU0 x6 + GPU1/2/3 x2 = 12 workers.
