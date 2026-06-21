# -*- coding: utf-8 -*-
"""Oracle / invariant check for the rotated per-token K quant (qlutattn-rotated-*).

Validates the new KittyKVCache._fwht_lastdim and the de-rotate trick against the
NumPy reference's mechanism:
  (1) FWHT == matmul(normalized Hadamard); H Hᵀ = I; FWHT self-inverse.
  (2) On synthetic attention with injected channel outliers, rotation lifts the
      per-token sign/tern attention recovery (reproduces §4 trend gaps).

The de-rotate trick: store k_hat = mu + fwht(quant(fwht(k-mu))); then
score(q, k_hat) = q.mu + (fwht q).quant(...) -> the rotated-basis dot WITHOUT
rotating q (q.mu is a per-row constant, cancels in softmax). So no attention
change is needed -- exactly what the sim variant does.

Run (CPU, no GPU):  CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python scripts/verify_rotated_oracle.py
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import torch

from kitty_sim.kitty_simulate import KittyKVCache

fwht = KittyKVCache._fwht_lastdim


def hadamard(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return torch.tensor(H / np.sqrt(n), dtype=torch.float32)


def rope(x, base=10000.0):
    d = x.shape[-1]; half = d // 2
    inv = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    t = torch.arange(x.shape[0]).float()
    fr = torch.outer(t, inv)
    cos = torch.cat([fr.cos(), fr.cos()], -1)
    sin = torch.cat([fr.sin(), fr.sin()], -1)
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def sm(z):
    z = z - z.max(-1, keepdim=True).values
    e = z.exp()
    return e / e.sum(-1, keepdim=True)


def main():
    torch.manual_seed(0); np.random.seed(0)
    n = 128
    H = hadamard(n)
    x = torch.randn(7, n)
    print("=" * 70)
    print("ROTATED per-token K quant -- oracle / invariants")
    print("=" * 70)
    print(f"[inv1] orthogonal  max|H Hᵀ - I|        = {(H @ H.t() - torch.eye(n)).abs().max():.2e}  (<1e-6)")
    print(f"[inv2] FWHT == matmul(H)  max abs diff  = {(fwht(x) - x @ H).abs().max():.2e}  (<1e-5)")
    print(f"[inv3] self-inverse fwht(fwht(x))=x     = {(fwht(fwht(x)) - x).abs().max():.2e}  (<1e-5)")
    # works on [B,nh,T,D] too (the real call shape)
    x4 = torch.randn(2, 8, 16, n)
    print(f"[inv4] 4D self-inverse                  = {(fwht(fwht(x4)) - x4).abs().max():.2e}  (<1e-5)")

    # --- synthetic outlier attention: naive (k125-style) vs rotated ---
    seq, d = 256, 128
    K = torch.randn(seq, d)
    outlier = np.random.choice(d, 4, replace=False)
    K[:, outlier] *= 12.0
    K[:, outlier[:2]] += 6.0
    targets = torch.randint(0, seq, (seq,))
    Q = 1.4 * K[targets] + torch.randn(seq, d)
    Qr, Kr = rope(Q), rope(K)
    Aref = sm(Qr @ Kr.t() / d ** 0.5)

    def quant(x, cb, rotate):
        mu = x.mean(0, keepdim=True)
        r = x - mu
        if rotate:
            r = fwht(r)
        mag = r.abs().mean(-1, keepdim=True)
        if cb == "tern":
            m = r.abs() > 0.5 * mag
            mag2 = (r.abs() * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp(min=1)
            q = r.sign() * mag2 * m
        else:
            q = r.sign() * mag
        if rotate:
            q = fwht(q)
        return mu + q

    def metrics(Khat):
        A = sm(Qr @ Khat.t() / d ** 0.5)
        top1 = (A.argmax(-1) == Aref.argmax(-1)).float().mean().item()
        tv = (0.5 * (A - Aref).abs().sum(-1)).mean().item()
        return top1, tv

    print("-" * 70)
    print(f"{'scheme':<8}{'naive top1':>12}{'naive TV':>10}{'ROT top1':>11}{'ROT TV':>9}  (per-channel mean on)")
    for cb in ["sign", "tern"]:
        t0, v0 = metrics(quant(Kr, cb, rotate=False))
        t1, v1 = metrics(quant(Kr, cb, rotate=True))
        print(f"{cb:<8}{t0:>12.1%}{v0:>10.3f}{t1:>11.1%}{v1:>9.3f}")
    print("-" * 70)
    print("Expect rotation to LIFT top1 and LOWER TV (sign benefits most), matching")
    print("the reference §4 gaps. naive here already has per-channel mean (k125v4-pt).")


if __name__ == "__main__":
    main()
