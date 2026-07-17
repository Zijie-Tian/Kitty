# -*- coding: utf-8 -*-
"""Build RESEARCH-mode qlutattn codebook masks (runtime requires QLUT_RESEARCH=1).

Reads the per-channel residual sigma^2 stored in an existing calibration blob
(scripts/calibrate_qlutattn_mask.py saves it as `sigma2` [nl, n_kv, D]) and
re-tiers channels into sign / nf2 / int4 codebooks without touching a GPU.

Per layer, channels are ranked ASCENDING by the chosen signal and split by the
requested fractions: lowest -> sign (1-bit), middle -> nf2 (2-bit symnf2-v1),
highest -> int4 (4-bit symmetric absmax, research-only outlier tier).

Signals:
  sigma2       (default) the stored residual sigma^2 ranking (canonical signal)
  sigma2_x_q   sigma2 * E|q_d| (MixKVQ-style query-aware difficulty x relevance;
               needs --qstats from scripts/probe_q_magnitude.py)

Nominal K bits/value = f_sign*1.25 + f_nf2*2.25 + f_int4*4.25 (codeword + one
fp16 per-token scale per tier amortized over 64 channels, matching the
canonical accounting).

Example:
  python scripts/make_research_mask.py \
    --base ~/models/Llama-3.2-1B-Instruct.k188v4pt_f50.pt \
    --fractions sign=0.625,nf2=0.325,int4=0.05 \
    --output ~/models/Llama-3.2-1B-Instruct.rm_mix3_625_325_05.pt
"""
import argparse
from pathlib import Path

import torch

BITS = {"sign": 1.25, "nf2": 2.25, "int4": 4.25}
CODEBOOK_ORDER = ("sign", "nf2", "int4")


def parse_fractions(spec: str) -> dict[str, float]:
    fr = {}
    for part in spec.split(","):
        k, v = part.split("=")
        k = k.strip()
        if k not in CODEBOOK_ORDER:
            raise ValueError(f"unknown codebook {k!r}; allowed {CODEBOOK_ORDER}")
        fr[k] = float(v)
    total = sum(fr.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0; got {total}")
    for k in CODEBOOK_ORDER[:2]:
        fr.setdefault(k, 0.0)
    fr.setdefault("int4", 0.0)
    if any(v < 0 for v in fr.values()):
        raise ValueError("fractions must be non-negative")
    return fr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="calibration blob with sigma2 [nl,n_kv,D]")
    p.add_argument("--fractions", required=True, help="e.g. sign=0.625,nf2=0.325,int4=0.05")
    p.add_argument("--signal", default="sigma2", choices=["sigma2", "sigma2_x_q"])
    p.add_argument("--qstats", default=None, help="q_absmean blob for sigma2_x_q (probe_q_magnitude.py)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    blob = torch.load(args.base, map_location="cpu", weights_only=False)
    sigma2 = blob["sigma2"].float()                                 # [nl, n_kv, D]
    nl, n_kv, D = sigma2.shape
    signal = sigma2
    if args.signal == "sigma2_x_q":
        if not args.qstats:
            raise ValueError("--signal sigma2_x_q requires --qstats")
        qb = torch.load(args.qstats, map_location="cpu", weights_only=False)
        qmag = qb["q_absmean"].float()                              # [nl, n_kv, D]
        if qmag.shape != sigma2.shape:
            raise ValueError(f"qstats shape {tuple(qmag.shape)} != sigma2 {tuple(sigma2.shape)}")
        signal = sigma2 * qmag

    fr = parse_fractions(args.fractions)
    N = n_kv * D
    k_sign = int(round(fr["sign"] * N))
    k_int4 = int(round(fr["int4"] * N))
    k_nf2 = N - k_sign - k_int4
    if k_nf2 < 0:
        raise ValueError("fractions leave a negative nf2 tier")

    mask = torch.empty(nl, n_kv, D, dtype=torch.uint8)
    for li in range(nl):
        order = torch.argsort(signal[li].reshape(-1))               # ascending
        m = torch.empty(N, dtype=torch.uint8)
        m[order[:k_sign]] = 0                                       # lowest -> sign
        m[order[k_sign:k_sign + k_nf2]] = 1                         # middle -> nf2
        if k_int4 > 0:
            m[order[k_sign + k_nf2:]] = 2                           # highest -> int4
        mask[li] = m.reshape(n_kv, D)

    fs, fn, fi = k_sign / N, k_nf2 / N, k_int4 / N
    nominal = fs * BITS["sign"] + fn * BITS["nf2"] + fi * BITS["int4"]
    codebooks = ["sign", "nf2", "int4"] if k_int4 > 0 else ["sign", "nf2"]

    out = dict(blob)
    out.update({
        "codebook_mask": mask,
        "codebooks": codebooks,
        "low_frac": fs,
        "nominal_bits": nominal,
        "research": True,
        "research_signal": args.signal,
        "research_base": str(Path(args.base)),
        "research_fractions": {"sign": fs, "nf2": fn, "int4": fi},
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"[research-mask] {args.output}")
    print(f"[research-mask] signal={args.signal} per-layer tiers: sign={k_sign}/{N} ({fs:.1%}) "
          f"nf2={k_nf2}/{N} ({fn:.1%}) int4={k_int4}/{N} ({fi:.1%}) -> nominal K {nominal:.4f} bit/value")


if __name__ == "__main__":
    main()
