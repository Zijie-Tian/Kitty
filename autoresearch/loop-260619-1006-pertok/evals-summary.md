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
