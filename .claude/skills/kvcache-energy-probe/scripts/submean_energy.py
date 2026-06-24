#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证减均值分解 E[x^2] = mu^2 + sigma^2 在真实 KV cache 上的能量占比。

对应推导（Notion「为什么 KIVI/smooth/Hadamard 到不了 1-bit」§2.1）：
    组内元素 x_t = mu + r_t,  E[x^2] = mu^2 + 2*mu*E[r] + E[r^2] = mu^2 + sigma^2
其中统计口径为 population（ddof=0），此时恒等式精确成立（仅浮点舍入误差）。

做法：
  1) HF transformers 加载模型（fp16），LongBench 长文本拼接 tokenize 到 seq_len；
  2) 分段 prefill（避免长序列一次性激活峰值），收集每层 post-RoPE K cache 与 V cache；
  3) per-channel 取向（固定 (head, dim) 通道、沿 token 轴），按 G=128 连续分组；
  4) 每组统计 mu_g = mean(x), Ex2_g = mean(x^2), sigma2_g = Ex2_g - mu_g^2；
  5) 能量占比（各组等长，能量加权 = 直接求和）：
         share_mu = sum(mu_g^2) / sum(Ex2_g),  share_sigma = sum(sigma2_g) / sum(Ex2_g)
     并报告恒等式残差 |sum(mu^2)+sum(sigma^2)-sum(Ex2)| / sum(Ex2)（应为 ~1e-7 级）；
  6) 附带残差形状统计 rho = E|r| / sqrt(E[r^2])（pooled）。

参考对照：kitty_sign 笔记实测 Llama-3.2-1B post-RoPE K、G=128 下 mu 分量约占 64.8%。

用法：
  CUDA_VISIBLE_DEVICES=1 python submean_energy.py --model MODEL --seq-len 32768 --group 128
（--model 接受本地路径 / HF id / 别名 KVPROBE_MODEL_<名>；--longbench-dir 缺省读
   KVPROBE_LONGBENCH_DIR / LONGBENCH_DATA_ROOT。）
"""
import argparse
import json

import torch

import _common as c


@torch.inference_mode()
def group_stats(t, G):
    """t: [1, H, T, D] -> per-channel (H*D) 沿 token 轴、G 一组的能量分解。"""
    gs = c.group_decompose(t, G)
    H, D, ng, _ = gs.xg.shape
    r = gs.xg - gs.mu.unsqueeze(-1)                # [H,D,ng,G]
    s_mu2 = (gs.mu * gs.mu).sum().item()
    s_var = gs.var.clamp_min(0).sum().item()
    s_ex2 = gs.ex2.sum().item()
    rho = (r.abs().mean() / r.pow(2).mean().sqrt()).item()
    return {
        "sum_mu2": s_mu2,
        "sum_var": s_var,
        "sum_ex2": s_ex2,
        "share_mu": s_mu2 / s_ex2,
        "share_var": s_var / s_ex2,
        "identity_resid": abs(s_mu2 + s_var - s_ex2) / s_ex2,
        "rho": rho,
        "channels": H * D,
        "n_groups_per_ch": ng,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地路径 / HF repo id / 别名 (见 SKILL.md)")
    ap.add_argument("--longbench-dir", default=None, help="LongBench data 目录；缺省读 KVPROBE_LONGBENCH_DIR / LONGBENCH_DATA_ROOT")
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", default="submean_energy_result.json")
    args = ap.parse_args()

    dev = "cuda:0"
    torch.manual_seed(0)

    model_path = c.resolve_model_path(args.model)
    lb_dir = c.resolve_longbench_dir(args.longbench_dir)

    model, tok, dims = c.load_model_and_tok(model_path, dev)
    text = c.build_concat_text(lb_dir, need_chars=args.seq_len * 16)
    ids = tok(text, return_tensors="pt").input_ids
    assert ids.shape[1] >= args.seq_len, f"text too short: {ids.shape[1]} < {args.seq_len}"
    ids = ids[:, : args.seq_len]
    print(f"[input] {ids.shape[1]} tokens (G={args.group} -> {ids.shape[1] // args.group} groups/channel)")
    print(f"[model] {model_path}  layers={dims.nl} kv_heads={dims.H} head_dim={dims.D}")

    cache = c.prefill(model, ids, dev, chunk=args.chunk)
    torch.cuda.synchronize()
    print(f"[prefill] done, max_mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    nl = c.cache_n_layers(cache)
    per_layer = {"K": [], "V": []}
    tot = {kv: {"sum_mu2": 0.0, "sum_var": 0.0, "sum_ex2": 0.0} for kv in ("K", "V")}

    hdr = f"{'layer':>5} | {'mu^2 share':>10} {'sigma^2 share':>13} {'identity resid':>14} {'rho=E|r|/sigma':>14}"
    for kv in ("K", "V"):
        print(f"\n===== {kv} cache: per-channel, token-axis grouping, G={args.group} =====")
        print(hdr)
        for li in range(nl):
            k, v = c.cache_layer_kv(cache, li)
            st = group_stats(k if kv == "K" else v, args.group)
            per_layer[kv].append(st)
            for key in tot[kv]:
                tot[kv][key] += st[key]
            print(f"{li:>5} | {st['share_mu']:>10.4f} {st['share_var']:>13.4f} "
                  f"{st['identity_resid']:>14.2e} {st['rho']:>14.4f}")
        s = tot[kv]
        share_mu = s["sum_mu2"] / s["sum_ex2"]
        share_var = s["sum_var"] / s["sum_ex2"]
        resid = abs(s["sum_mu2"] + s["sum_var"] - s["sum_ex2"]) / s["sum_ex2"]
        print(f"{'ALL':>5} | {share_mu:>10.4f} {share_var:>13.4f} {resid:>14.2e} "
              f"{'(pooled below)':>14}")
        print(f"[{kv}] overall: mu^2 = {share_mu * 100:.2f}%  sigma^2 = {share_var * 100:.2f}%  "
              f"(identity holds to {resid:.2e})")

    result = {
        "model": model_path, "seq_len": ids.shape[1], "group": args.group,
        "orientation": "per-channel (head,dim) along token axis, post-RoPE K",
        "per_layer": per_layer,
        "overall": {kv: {"share_mu": tot[kv]["sum_mu2"] / tot[kv]["sum_ex2"],
                         "share_var": tot[kv]["sum_var"] / tot[kv]["sum_ex2"]}
                    for kv in ("K", "V")},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
