# -*- coding: utf-8 -*-
"""Build RESEARCH-mode qlutattn codebook masks (runtime requires QLUT_RESEARCH=1).

The codebook pair is FIXED to the canonical sign/nf2 (no other codebooks).
What this script varies, per experiment, is the ASSIGNMENT of channels to the
two codebooks:

  * the sign fraction f (nominal K bits = f*1.25 + (1-f)*2.25, so 1.25..2.25)
  * the ranking signal (which channels deserve the 2-bit nf2 budget)
  * optionally a per-layer fraction schedule at a fixed global budget

Reads the per-channel residual sigma^2 stored in an existing calibration blob
(scripts/calibrate_qlutattn_mask.py saves `sigma2` [nl, n_kv, D]); no GPU.
Per layer, channels are ranked ASCENDING by the signal: lowest -> sign,
highest -> nf2.

Signals:
  sigma2       (default) stored residual sigma^2 (canonical signal)
  sigma2_x_q   sigma2 * E|q_d| (MixKVQ-style difficulty x query relevance;
               needs --qstats from scripts/probe_q_magnitude.py)
  q_abs        E|q_d| alone (ablation: pure query relevance)

Per-layer schedule: --layer-fracs "0:0.9,1:0.9,10:0.5,14:0.5" overrides the
global sign fraction for the listed layers; the remaining layers share the
leftover budget so the GLOBAL sign fraction still equals --sign-frac.

Example (uniform f75, 1.5 bit):
  python scripts/make_research_mask.py \
    --base ~/models/Llama-3.2-1B-Instruct.k188v4pt_f50.pt \
    --sign-frac 0.75 --output ~/models/Llama-3.2-1B-Instruct.rm_f75.pt
"""
import argparse
from pathlib import Path

import torch

BITS = {"sign": 1.25, "nf2": 2.25}


def parse_layer_fracs(spec: str | None) -> dict[int, float]:
    if not spec:
        return {}
    out = {}
    for part in spec.split(","):
        k, v = part.split(":")
        out[int(k)] = float(v)
    if any(not (0.0 <= v <= 1.0) for v in out.values()):
        raise ValueError("per-layer sign fractions must be in [0, 1]")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="calibration blob with sigma2 [nl,n_kv,D]")
    p.add_argument("--sign-frac", type=float, required=True, help="global sign fraction in [0,1]")
    p.add_argument("--signal", default="sigma2", choices=["sigma2", "sigma2_x_q", "q_abs"])
    p.add_argument("--qstats", default=None, help="q_absmean blob for query-aware signals")
    p.add_argument("--layer-fracs", default=None,
                   help="per-layer sign-frac overrides 'L:f,L:f'; other layers rebalance to keep the global budget")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if not (0.0 <= args.sign_frac <= 1.0):
        raise ValueError("--sign-frac must be in [0, 1]")

    blob = torch.load(args.base, map_location="cpu", weights_only=False)
    sigma2 = blob["sigma2"].float()                                 # [nl, n_kv, D]
    nl, n_kv, D = sigma2.shape
    signal = sigma2
    if args.signal in ("sigma2_x_q", "q_abs"):
        if not args.qstats:
            raise ValueError(f"--signal {args.signal} requires --qstats")
        qb = torch.load(args.qstats, map_location="cpu", weights_only=False)
        qmag = qb["q_absmean"].float()                              # [nl, n_kv, D]
        if qmag.shape != sigma2.shape:
            raise ValueError(f"qstats shape {tuple(qmag.shape)} != sigma2 {tuple(sigma2.shape)}")
        signal = sigma2 * qmag if args.signal == "sigma2_x_q" else qmag

    N = n_kv * D
    overrides = parse_layer_fracs(args.layer_fracs)
    if any(li < 0 or li >= nl for li in overrides):
        raise ValueError(f"--layer-fracs layer index out of range 0..{nl - 1}")
    # Rebalance the non-override layers so the global sign fraction is honored.
    total_sign = args.sign_frac * nl * N
    override_sign = sum(f * N for f in overrides.values())
    rest = [li for li in range(nl) if li not in overrides]
    if rest:
        rest_frac = (total_sign - override_sign) / (len(rest) * N)
        if not (-1e-9 <= rest_frac <= 1.0 + 1e-9):
            raise ValueError(
                f"--layer-fracs leaves an infeasible remainder fraction {rest_frac:.4f} "
                f"for the {len(rest)} unlisted layers")
        rest_frac = min(max(rest_frac, 0.0), 1.0)
    elif abs(override_sign - total_sign) > 1e-6:
        raise ValueError("--layer-fracs covers all layers but does not meet --sign-frac")

    mask = torch.empty(nl, n_kv, D, dtype=torch.uint8)
    per_layer_sign = []
    for li in range(nl):
        f_li = overrides.get(li, rest_frac if rest else args.sign_frac)
        k_sign = int(round(f_li * N))
        order = torch.argsort(signal[li].reshape(-1))               # ascending
        m = torch.ones(N, dtype=torch.uint8)                        # default nf2(1)
        m[order[:k_sign]] = 0                                       # lowest -> sign
        mask[li] = m.reshape(n_kv, D)
        per_layer_sign.append(k_sign)

    fs = sum(per_layer_sign) / (nl * N)
    nominal = fs * BITS["sign"] + (1 - fs) * BITS["nf2"]

    out = dict(blob)
    out.update({
        "codebook_mask": mask,
        "codebooks": ["sign", "nf2"],
        "low_frac": fs,
        "nominal_bits": nominal,
        "research": True,
        "research_signal": args.signal,
        "research_base": str(Path(args.base)),
        "research_layer_fracs": overrides or None,
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"[research-mask] {args.output}")
    sched = f" layer-overrides={overrides}" if overrides else ""
    print(f"[research-mask] signal={args.signal} global sign={fs:.4f} "
          f"-> nominal K {nominal:.4f} bit/value{sched}")


if __name__ == "__main__":
    main()
