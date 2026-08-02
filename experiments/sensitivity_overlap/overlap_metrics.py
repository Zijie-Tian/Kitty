# -*- coding: utf-8 -*-
"""GPU mask/overlap/quota-structure metrics for the calibratability experiment.

Ported from worktree eval_design_fig_6/overlap_pipeline (2026-07-28 protocol).

All heavy computation on CUDA tensors (mask generation = stable desc sort +
fp64 cumsum + searchsorted; layer-macro Jaccard; Spearman of per-layer NF2
count vectors; 2000-replicate within-task bootstrap). CPU only does file IO.

Propositions:
  M1: pairwise layer-macro Jaccard of rho=0.62 product masks across tasks
      (+ sigma2-only and q-only factor variants)
  M2: per-layer NF2 quota structure stability — per-task 16/28-dim NF2 count
      vectors, pairwise Spearman, per-layer cross-task ratio min/max/mean/std
      vs the WikiText-2 reference band

Boundary: these are MASK-level statistics. No accuracy claims anywhere.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

RHO_DEFAULT = 0.62
# 短输入分类任务("TREC 类"),M1 判读的双口径使用(预先登记,机械执行)
TREC_LIKE = ("trec", "lsht", "passage_count")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stats-root", required=True)
    p.add_argument("--model-tag", required=True)
    p.add_argument("--tasks", default="", help="comma list; default = auto-discover dirs")
    p.add_argument("--rho", type=float, default=RHO_DEFAULT)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ref-mask", default="", help="wikitext rho062 mask artifact")
    p.add_argument("--tag", default="")
    return p.parse_args(argv)


def load_task(stats_root, model_tag, task):
    path = Path(stats_root) / model_tag / task / "stats.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload, path


def make_mask(score, rho):
    """score: [..., N] fp64 CUDA → nf2 mask bool [..., N] via shortest stable
    prefix with cumulative mass >= rho * total (ties: ascending flat index,
    guaranteed by stable sort). Vectorized over leading dims."""
    N = score.shape[-1]
    flat = score.reshape(-1, N)
    order = torch.argsort(flat, dim=-1, descending=True, stable=True)
    ordered = torch.gather(flat, -1, order)
    cum = torch.cumsum(ordered, dim=-1, dtype=torch.float64)
    target = cum[:, -1:] * rho
    b = torch.searchsorted(cum, target).clamp(max=N - 1)      # [B, 1]
    ranks = torch.empty_like(order)
    arange = torch.arange(N, device=score.device).expand_as(order)
    ranks.scatter_(-1, order, arange)
    mask = ranks <= b                                          # [B, N] bool
    return mask.reshape(score.shape)


def macro_jaccard(mask_a, mask_b, nl):
    """mask: [..., nl, N] bool → layer-macro Jaccard [...]."""
    inter = (mask_a & mask_b).sum(dim=-1).double()
    union = (mask_a | mask_b).sum(dim=-1).double()
    return (inter / union).mean(dim=-1)


def rankdata_avg(x):
    """Average ranks (1..n) of 1-D CUDA tensor, ties share the mean rank."""
    sorted_x, order = torch.sort(x.double())
    _, inv, counts = torch.unique_consecutive(
        sorted_x, return_inverse=True, return_counts=True)
    cum = torch.cumsum(counts.double(), 0)
    start = cum - counts.double()          # 0-based start of each group
    avg = (start + 1 + cum) / 2.0          # mean of 1-based ranks in group
    ranks_sorted = avg[inv]
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def spearman(x, y):
    rx, ry = rankdata_avg(x), rankdata_avg(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    if denom == 0:
        return float("nan")
    return float((rx @ ry) / denom)


def main():
    args = parse_args()
    dev = torch.device(args.device)
    t0 = time.time()
    stats_root = Path(args.stats_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.model_tag

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        base = stats_root / args.model_tag
        tasks = sorted(p.name for p in base.iterdir()
                       if p.is_dir() and (p / "stats.pt").is_file())
    print(f"[metrics] {args.model_tag}: {len(tasks)} tasks on {dev}", flush=True)

    payloads = {}
    for task in tasks:
        payload, _ = load_task(stats_root, args.model_tag, task)
        payloads[task] = payload
    first = payloads[tasks[0]]
    nl, n_kv, D = first["n_layers"], first["n_kv"], first["head_dim"]
    N = n_kv * D
    print(f"[metrics] layers={nl} kv={n_kv} D={D} N={N} rho={args.rho}", flush=True)

    # ---- point estimates on GPU -------------------------------------------
    s2 = torch.stack([payloads[t]["sigma2_mean_fp64"] for t in tasks]).to(dev)
    qq = torch.stack([payloads[t]["q_absmean_mean_fp64"] for t in tasks]).to(dev)
    T = len(tasks)
    score_prod = (s2 * qq).reshape(T, nl, N)
    masks = {
        "product": make_mask(score_prod, args.rho),
        "sigma2_only": make_mask(s2.reshape(T, nl, N), args.rho),
        "q_only": make_mask(qq.reshape(T, nl, N), args.rho),
    }

    results = {"model_tag": args.model_tag, "tasks": tasks, "rho": args.rho,
               "n_layers": nl, "n_kv": n_kv, "head_dim": D,
               "bootstrap_reps": args.bootstrap, "seed": args.seed,
               "wall_s": None}

    # M1 pairwise Jaccard matrices
    pairwise = {}
    for variant, m in masks.items():
        mb = m.reshape(T, nl, N)
        mat = torch.eye(T, dtype=torch.float64, device=dev)
        for i in range(T):
            for j in range(i + 1, T):
                mat[i, j] = mat[j, i] = macro_jaccard(
                    mb[i], mb[j], nl)
        pairwise[variant] = mat
        off = mat[~torch.eye(T, dtype=torch.bool, device=dev)]
        results[f"jaccard_{variant}"] = {
            "mean": float(off.mean()), "min": float(off.min()),
            "max": float(off.max()),
        }
        print(f"[metrics] jaccard {variant}: mean={off.mean():.4f} "
              f"range=({off.min():.4f},{off.max():.4f})", flush=True)

    # dual-scope means: 含/不含 TREC 类(预先登记)
    trec_idx = [i for i, t in enumerate(tasks) if t in TREC_LIKE]
    mat = pairwise["product"]
    keep = torch.ones(T, T, dtype=torch.bool, device=dev)
    for i in trec_idx:
        keep[i, :] = False
        keep[:, i] = False
    keep.fill_diagonal_(False)
    results["jaccard_product_ex_trec_like"] = {
        "mean": float(mat[keep].mean()),
        "excluded_tasks": [tasks[i] for i in trec_idx],
        "n_pairs": int(keep.sum()) // 2,
    }
    print(f"[metrics] jaccard product 不含 TREC 类 "
          f"{results['jaccard_product_ex_trec_like']['excluded_tasks']}: "
          f"mean={results['jaccard_product_ex_trec_like']['mean']:.4f}",
          flush=True)

    # M2 per-layer NF2 counts + Spearman
    counts = masks["product"].reshape(T, nl, N).sum(dim=-1)  # [T, nl] int64
    ratios = counts.double() / N
    sp_mat = torch.eye(T, dtype=torch.float64, device=dev)
    for i in range(T):
        for j in range(i + 1, T):
            sp_mat[i, j] = sp_mat[j, i] = spearman(counts[i], counts[j])
    sp_off = sp_mat[~torch.eye(T, dtype=torch.bool, device=dev)]
    results["spearman"] = {"median": float(sp_off.median()),
                           "mean": float(sp_off.mean()),
                           "min": float(sp_off.min()),
                           "max": float(sp_off.max())}
    print(f"[metrics] spearman: median={sp_off.median():.4f} "
          f"mean={sp_off.mean():.4f} range=({sp_off.min():.4f},{sp_off.max():.4f})",
          flush=True)
    results["layer_ratio_summary"] = {
        "min": ratios.min(dim=0).values.tolist(),
        "max": ratios.max(dim=0).values.tolist(),
        "mean": ratios.mean(dim=0).tolist(),
        "std": ratios.std(dim=0, unbiased=(T > 1)).tolist(),
    }

    # WikiText 参照逐层占比
    if args.ref_mask:
        ref = torch.load(args.ref_mask, map_location="cpu", weights_only=True)
        ref_ratio = ref["nf2_ratio_per_layer"].double()
        results["wikitext_ref_ratio_per_layer"] = ref_ratio.tolist()
        results["wikitext_ref_band"] = {
            "min": float(ref_ratio.min()), "max": float(ref_ratio.max()),
            "mean": float(ref_ratio.mean())}

    # ---- bootstrap (within-task, 2000 reps, GPU) ---------------------------
    # 重采样均值用"计数权重 × 逐 prompt 张量"的 matmul 实现,避免 materialize
    # [B, n_prompts, nl, H, D](3B 下 ~59GB)。
    B = args.bootstrap
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    n_prompts = first["num_prompts"]
    F = nl * n_kv * D
    s2f = {t: payloads[t]["sigma2_prompts"].reshape(n_prompts, F).double().to(dev)
           for t in tasks}
    qf = {t: payloads[t]["q_absmean_prompts"].reshape(n_prompts, F).double().to(dev)
          for t in tasks}

    def resample_counts():
        idx = torch.randint(0, n_prompts, (B, n_prompts), generator=gen, device=dev)
        cnt = torch.zeros(B, n_prompts, dtype=torch.float64, device=dev)
        cnt.scatter_add_(1, idx, torch.ones(B, n_prompts, dtype=torch.float64, device=dev))
        return cnt / n_prompts

    boot = torch.empty(T, T, B, dtype=torch.float32, device=dev)
    for i in range(T):
        wi = resample_counts()
        s2i = (wi @ s2f[tasks[i]]).reshape(B, nl, N)
        qi = (wi @ qf[tasks[i]]).reshape(B, nl, N)
        mi = make_mask(s2i * qi, args.rho)                        # [B, nl, N]
        for j in range(i + 1, T):
            wj = resample_counts()
            s2j = (wj @ s2f[tasks[j]]).reshape(B, nl, N)
            qj = (wj @ qf[tasks[j]]).reshape(B, nl, N)
            mj = make_mask(s2j * qj, args.rho)
            jb = macro_jaccard(mi, mj, nl)
            boot[i, j] = boot[j, i] = jb.float()
        print(f"[metrics] bootstrap row {i + 1}/{T} ({time.time() - t0:.0f}s)",
              flush=True)
    lo = torch.quantile(boot, 0.025, dim=-1)
    hi = torch.quantile(boot, 0.975, dim=-1)
    results["bootstrap_ci"] = {
        "lo": lo.cpu().tolist(), "hi": hi.cpu().tolist(),
        "mean": boot.mean(dim=-1).cpu().tolist(),
    }

    # ---- write TSV / JSON / artifacts --------------------------------------
    import csv

    def write_matrix_tsv(path, mat):
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["task"] + tasks)
            for i, t in enumerate(tasks):
                w.writerow([t] + [f"{v:.6f}" for v in mat[i].cpu().tolist()])

    write_matrix_tsv(out_dir / f"jaccard_pairwise_product_{tag}.tsv", pairwise["product"])
    write_matrix_tsv(out_dir / f"jaccard_pairwise_sigma2_only_{tag}.tsv", pairwise["sigma2_only"])
    write_matrix_tsv(out_dir / f"jaccard_pairwise_q_only_{tag}.tsv", pairwise["q_only"])
    write_matrix_tsv(out_dir / f"spearman_pairwise_{tag}.tsv", sp_mat)

    with open(out_dir / f"nf2_counts_{tag}.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task"] + [f"layer{i}" for i in range(nl)] + ["total", "mean_ratio"])
        for i, t in enumerate(tasks):
            row = counts[i].cpu().tolist()
            w.writerow([t] + row + [int(counts[i].sum()), f"{ratios[i].mean():.6f}"])

    with open(out_dir / f"layer_ratio_summary_{tag}.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        header = ["layer", "min", "max", "mean", "std"]
        if "wikitext_ref_ratio_per_layer" in results:
            header.append("wikitext_ref")
        w.writerow(header)
        for li in range(nl):
            row = [li,
                   f"{results['layer_ratio_summary']['min'][li]:.6f}",
                   f"{results['layer_ratio_summary']['max'][li]:.6f}",
                   f"{results['layer_ratio_summary']['mean'][li]:.6f}",
                   f"{results['layer_ratio_summary']['std'][li]:.6f}"]
            if "wikitext_ref_ratio_per_layer" in results:
                row.append(f"{results['wikitext_ref_ratio_per_layer'][li]:.6f}")
            w.writerow(row)

    torch.save({
        "masks_product": masks["product"].cpu(),
        "nf2_counts_per_layer": counts.cpu(),
        "tasks": tasks, "rho": args.rho, "model_tag": args.model_tag,
    }, out_dir / f"masks_{tag}.pt")

    results["wall_s"] = time.time() - t0
    with open(out_dir / f"summary_{tag}.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"[metrics] wrote outputs to {out_dir} ({results['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
