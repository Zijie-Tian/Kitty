#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kvcache-energy-probe 公共库：把 11 个 KV-cache 减均值能量探针共享的骨架抽出一份。

统一内核（所有探针共用）：per-channel 取向（固定 (head, dim)、沿 token 轴）、按 G
连续分组、population 统计下 ``E[x^2] = mu^2 + sigma^2`` 精确成立。mu^2 = 「红通道」
（组均值携带能量、好量化），sigma^2 = 「蓝通道」（组内残差携带能量、难量化）。

本库只提供「无数值歧义」的公共原语；凡涉及一次性 ``mean(dim=(-1,-2))``、flat-C +
sign 重建、协方差归因等对浮点累加序敏感的逻辑，保留在各薄脚本里本地实现（见各脚本）。

数值等价约定（务必遵守，详见 plan「数值等价 5 高危点」）：
  1. ``group_decompose`` permute 后**立刻 .float()** 再统计（fp16 上先 reduce 会偏 sigma^2）；
  2. 一次性 ``mean(dim=(-1,-2))`` 与链式 ``mean(-1).mean(-1)`` 在浮点下不逐位相等，
     需要一次性形态的脚本请直接对 ``GroupStats.xg`` 本地算，不要走链式 reducer；
  3. ``GroupStats.var`` 返回**未 clamp** 的 population 方差，``clamp_min(0)`` 由调用方按
     原脚本逐处复刻；
  4. flat-C + sign 重建（kchannel_sigma / kvlen_energy）保留本地实现，不复用
     ``submean_sign``（后者在 [H,D,T] 上算且回填末尾，语义不同）；
  5. 原语不擅自搬设备，``.cpu()`` 时机由脚本复刻。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (Agg 必须在 pyplot 之前)
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# ======================================================================== #
# A. 常量                                                                   #
# ======================================================================== #

# 多文档拼接扫描顺序（submean_energy 用；第 3/4 项 = qmsum, musique）。
LB_CONCAT_FILES = [
    "narrativeqa.jsonl", "gov_report.jsonl", "qmsum.jsonl", "musique.jsonl",
    "2wikimqa.jsonl", "hotpotqa.jsonl", "multifieldqa_en.jsonl", "qasper.jsonl",
]
# 单文档扫描顺序（其余单文档探针共用；第 3/4 项 = musique, qmsum）。
# 注意：与 LB_CONCAT_FILES 第 3/4 项顺序不同，混用会改变取样 -> 破坏数值等价。
LB_SINGLE_FILES = [
    "narrativeqa.jsonl", "gov_report.jsonl", "musique.jsonl", "qmsum.jsonl",
    "2wikimqa.jsonl", "hotpotqa.jsonl", "multifieldqa_en.jsonl", "qasper.jsonl",
]
# task_channel_profile 默认 10 个类型差异最大的 subtask。
DEFAULT_TASKS = ("narrativeqa,qasper,hotpotqa,gov_report,trec,triviaqa,"
                 "lcc,repobench-p,dureader,vcsum")
# calibrate 的 21 个 LongBench held-out 验证任务。
VAL_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa",
    "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", "vcsum",
    "trec", "triviaqa", "samsum", "lsht", "passage_count",
    "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p", "dureader",
]
# 模型别名是「运行时配置」的，不在此写死任何项目特定别名/路径：
# resolve_model_path 把别名 X 解析为环境变量 KVPROBE_MODEL_X 的值（见该函数）。

# 共用色规：红 = mu^2 主导, 蓝 = sigma^2 主导（share in [0,1]）。
CMAP_SHARE = "RdBu_r"


# ======================================================================== #
# B. 取样                                                                   #
# ======================================================================== #

def build_concat_text(lb_dir: str, need_chars: int, files=LB_CONCAT_FILES) -> str:
    """多文档拼接：逐文件逐行取 context，凑够 need_chars 字符即返回（"\\n\\n" 连接）。"""
    parts, total = [], 0
    for name in files:
        path = os.path.join(lb_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                ctx = _json_loads(line).get("context", "")
                if not ctx:
                    continue
                parts.append(ctx)
                total += len(ctx)
                if total >= need_chars:
                    return "\n\n".join(parts)
    if not parts:
        raise RuntimeError(f"no usable context found under {lb_dir}")
    return "\n\n".join(parts)


def pick_single_doc(lb_dir: str, tok, seq_len: int,
                    files=LB_SINGLE_FILES, char_ratio: int = 3):
    """返回第一条 tokenize 后 >= seq_len 的单条 context (截前 seq_len)，及出处/全长。

    返回 (ids[:, :seq_len], src="name#ln", full_len)。兜底：无满足者取最长的一条（同样截断）。
    """
    best = None  # (n_tokens, ids_truncated, src)
    for name in files:
        path = os.path.join(lb_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f):
                ctx = _json_loads(line).get("context", "")
                # 字符数粗筛（英文 ~3-4 char/token），避免对短样本白白 tokenize。
                if len(ctx) < seq_len * char_ratio:
                    continue
                ids = tok(ctx, return_tensors="pt").input_ids
                src = f"{name}#{ln}"
                if ids.shape[1] >= seq_len:
                    return ids[:, :seq_len], src, ids.shape[1]
                if best is None or ids.shape[1] > best[0]:
                    best = (ids.shape[1], ids[:, :seq_len], src)
    if best is None:
        raise RuntimeError(f"no single context long enough under {lb_dir}")
    return best[1], best[2], best[0]


def pick_task_sample(lb_dir, task, tok, min_tok, max_tok, scan_lines=300):
    """某任务下第一条 tokenize 后 >= max_tok 的样本；否则前 scan_lines 行里最长的。

    返回 (ids[:, :max_tok] | None, line_no | None, full_len)。
    """
    path = os.path.join(lb_dir, f"{task}.jsonl")
    best = None  # (ntok, ids, line_no)
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f):
            if ln >= scan_lines:
                break
            ctx = _json_loads(line).get("context", "")
            if len(ctx) < min_tok * 2:      # 字符数粗筛
                continue
            ids = tok(ctx, return_tensors="pt").input_ids
            if ids.shape[1] >= max_tok:
                return ids[:, :max_tok], ln, ids.shape[1]
            if best is None or ids.shape[1] > best[0]:
                best = (ids.shape[1], ids, ln)
    if best is None or best[0] < min_tok:
        return None, None, 0
    return best[1][:, :max_tok], best[2], best[0]


def first_task_sample(lb_dir, task, tok, min_tok, max_tok, scan=200):
    """calibrate 用：某任务第一条 >= max_tok 的样本，否则最长的一条。

    返回 (ids | None, ntok)。文件缺失返回 (None, 0)。
    """
    path = os.path.join(lb_dir, f"{task}.jsonl")
    if not os.path.exists(path):
        return None, 0
    best = None
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f):
            if ln >= scan:
                break
            ctx = _json_loads(line).get("context", "")
            if len(ctx) < min_tok * 2:
                continue
            ids = tok(ctx, return_tensors="pt").input_ids
            if ids.shape[1] >= max_tok:
                return ids[:, :max_tok], ids.shape[1]
            if best is None or ids.shape[1] > best[0]:
                best = (ids.shape[1], ids)
    if best is None:
        return None, 0
    return best[1], best[0]


def build_calib_text(parquet: str, need_chars: int) -> str:
    """calibrate 域外标定语料：从 parquet 的 text 列拼足 need_chars（"\\n" 连接）。"""
    import pandas as pd  # lazy：纯函数测试不需要 pandas
    df = pd.read_parquet(parquet)
    col = "text" if "text" in df.columns else df.columns[0]
    parts, total = [], 0
    for s in df[col]:
        if not isinstance(s, str) or not s.strip():
            continue
        parts.append(s)
        total += len(s)
        if total >= need_chars:
            break
    return "\n".join(parts)


def _json_loads(line):
    import json
    return json.loads(line)


# ======================================================================== #
# C. 路径解析（复用 Kitty 的 .env 约定，不硬编码主机路径）                    #
# ======================================================================== #

def resolve_model_path(model_arg: str) -> str:
    """--model：本地路径 / HF repo id / 别名（按此顺序、零写死路径）。

    若设了环境变量 ``KVPROBE_MODEL_<ALIAS>``（ALIAS = model_arg 大写、``-`` 换 ``_``），用
    其值——这让别名机制对任意模型/任意机器通用、可移植，且不在代码里写死任何具体路径。
    否则把 model_arg 按字面当本地路径或 HF repo id 透传给 transformers。

    例：``export KVPROBE_MODEL_M1B=/models/Llama-3.2-1B`` 后 ``--model m1b`` 即解析到该路径；
    或直接 ``--model /models/Llama-3.2-1B`` / ``--model meta-llama/Llama-3.2-1B-Instruct``。
    """
    if model_arg and "/" not in model_arg:
        env = os.environ.get("KVPROBE_MODEL_" + model_arg.upper().replace("-", "_"))
        if env:
            return os.path.expanduser(env)
    return os.path.expanduser(model_arg)


def resolve_longbench_dir(arg) -> str:
    """LongBench 的 data 目录（含 ``<dataset>.jsonl``）。解析优先级，无写死路径 fallback：

      1) ``--longbench-dir`` 参数；
      2) 环境变量 ``KVPROBE_LONGBENCH_DIR``；
      3) 环境变量 ``LONGBENCH_DATA_ROOT``（社区常见约定）下的 ``data/`` 子目录。

    三者都没有则报错、要求显式提供——不假设任何项目特定的相对/绝对路径。
    """
    if arg:
        return os.path.expanduser(arg)
    d = os.environ.get("KVPROBE_LONGBENCH_DIR")
    if d:
        return os.path.expanduser(d)
    root = os.environ.get("LONGBENCH_DATA_ROOT")
    if root:
        return os.path.join(os.path.expanduser(root), "data")
    raise SystemExit(
        "[resolve_longbench_dir] LongBench data dir not set. Pass --longbench-dir "
        "<dir with <dataset>.jsonl>, or set KVPROBE_LONGBENCH_DIR / LONGBENCH_DATA_ROOT."
    )


# ======================================================================== #
# D. 模型与 prefill                                                         #
# ======================================================================== #

@dataclass
class ModelDims:
    nl: int          # num_hidden_layers
    H: int           # num_key_value_heads
    D: int           # head_dim
    n_q_heads: int   # num_attention_heads
    n_rep: int       # n_q_heads // H（GQA group）


def load_model_and_tok(model_path: str, dev: str = "cuda:0"):
    """fp16 + sdpa 加载到 dev 并 eval；返回 (model, tok, ModelDims)。"""
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, attn_implementation="sdpa",
    ).to(dev).eval()
    cfg = model.config
    n_q = cfg.num_attention_heads
    H = cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
    return model, tok, ModelDims(nl=cfg.num_hidden_layers, H=H, D=D,
                                 n_q_heads=n_q, n_rep=n_q // H)


def prefill(model, ids, dev, chunk: int = 2048, cache=None, hook_factory=None):
    """分段 prefill，收集 KV cache（避免长序列一次性激活峰值）。

    hook_factory(step_idx, n_steps, model) -> list[handle]：在指定段注册前向 hook
    （例如 si==n-1 抓最后一段的 q_proj，si==0 全程抓 k_proj）。返回的 handle 在
    prefill 结束统一 remove。
    """
    if cache is None:
        cache = DynamicCache()
    starts = list(range(0, ids.shape[1], chunk))
    handles = []
    with torch.inference_mode():
        for si, s in enumerate(starts):
            if hook_factory is not None:
                handles += hook_factory(si, len(starts), model) or []
            out = model(input_ids=ids[:, s:s + chunk].to(dev),
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
    for h in handles:
        h.remove()
    return cache


# ======================================================================== #
# E. DynamicCache 新旧布局兼容（transformers >= 4.56 用 layers[i].keys）       #
# ======================================================================== #

def cache_n_layers(cache) -> int:
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def cache_layer_k(cache, i):
    return cache.layers[i].keys if hasattr(cache, "layers") else cache.key_cache[i]


def cache_layer_kv(cache, i):
    if hasattr(cache, "layers"):
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


# ======================================================================== #
# F. 能量分解原语（抽库核心）                                                 #
# ======================================================================== #

@dataclass
class GroupStats:
    xg: torch.Tensor    # [H, D, ng, G]  fp32  组内原值（sign/cov/Lloyd 都要它）
    mu: torch.Tensor    # [H, D, ng]     组均值
    ex2: torch.Tensor   # [H, D, ng]     E[x^2]
    var: torch.Tensor   # [H, D, ng]     population var = ex2 - mu^2（**未 clamp**）


def group_decompose(t: torch.Tensor, G: int) -> GroupStats:
    """t: [1, H, T, D] -> permute [H,D,T] -> .float() -> 按 G 连续分组（截掉末尾不足一组）。

    返回 xg/mu/ex2/var。这是全部探针「纯能量统计」的唯一真源；下游粒度
    （flat-C / per-(H,D) / per-(H,D,ng) / per-(D,ng) sum-over-H / per-group sum-over-C）
    都是这 4 个张量的纯 reduce/reshape，不重算 mean/var。
    """
    x = t[0].permute(0, 2, 1).float()                 # [H, D, T]
    H, D, T = x.shape
    ng = T // G
    xg = x[:, :, :ng * G].reshape(H, D, ng, G)
    mu = xg.mean(-1)
    ex2 = (xg * xg).mean(-1)
    return GroupStats(xg=xg, mu=mu, ex2=ex2, var=ex2 - mu * mu)


def per_token_energy(t: torch.Tensor) -> torch.Tensor:
    """t: [1, H, T, D] -> per-token K 能量 ||k_t||^2 [T]（不分组、不截尾）。"""
    x = t[0].permute(0, 2, 1).reshape(-1, t.shape[2]).float()   # [C, T]
    return (x * x).sum(0)


# ======================================================================== #
# G. RoPE 表 & hook 抓取                                                     #
# ======================================================================== #

def rope_rotate(x, cosT, sinT, inverse=False):
    """x: [H, D, T]; cosT/sinT: [D/2, T]（上下半 cos 相同）。pair (i, i+D/2) 的 2x2 旋转。"""
    D = x.shape[1]
    a, b = x[:, : D // 2], x[:, D // 2:]
    c, s = cosT, (-sinT if inverse else sinT)
    return torch.cat([a * c - b * s, b * c + a * s], dim=1)


def rope_tables(model, T, D, dev, dtype=torch.float16):
    """全位置 RoPE 表，返回 cosT, sinT 形如 [D/2, T]（typed/blue 的 de-RoPE 用）。"""
    posT = torch.arange(T, device=dev)[None]
    dummy = torch.zeros(1, T, D, device=dev, dtype=dtype)
    cosF, sinF = model.model.rotary_emb(dummy, posT)
    return cosF[0].T[: D // 2].float(), sinF[0].T[: D // 2].float()


def rope_tables_for_positions(model, start, end, D, dev, dtype=torch.float16):
    """query 末段 [start, end) 的 RoPE cos/sin（原始 [1, end-start, D]，喂 apply_rotary_pos_emb）。"""
    pos = torch.arange(start, end, device=dev)[None]
    dummy = torch.zeros(1, end - start, D, device=dev, dtype=dtype)
    return model.model.rotary_emb(dummy, pos)


def add_q_norm_and_rope(attn, q_NQ_nqh_D, cos, sin):
    """抓到的 pre-RoPE q [NQ, n_q_heads, D] -> 补 q_norm（Qwen3 有）-> RoPE。返回 [n_q_heads, NQ, D] fp32。"""
    q = q_NQ_nqh_D
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    q = q.permute(1, 0, 2)[None]                       # [1, n_q, NQ, D]
    q, _ = apply_rotary_pos_emb(q, q, cos, sin)
    return q[0].float()                                # [n_q, NQ, D]


def gather_q(model):
    """返回 (store, hook_factory)：在**最后一段** prefill 注册 q_proj hook，抓 pre-RoPE q。

    store[li] = q_proj 输出 [1, T_chunk, n_q*D]（覆盖式，仅保留最后一段）。
    """
    store = {}

    def hook_factory(si, n, model):
        if si != n - 1:
            return []
        handles = []
        for li, lyr in enumerate(model.model.layers):
            def mk(li):
                def hook(mod, inp, out):
                    store[li] = out.detach()
                return hook
            handles.append(lyr.self_attn.q_proj.register_forward_hook(mk(li)))
        return handles

    return store, hook_factory


def gather_pre_rope_k(model):
    """返回 (store, hook_factory)：**全程**注册 k_proj hook，逐段 append 抓 pre-RoPE K（Llama 无 k_norm）。

    store[li] = [k_proj 输出 per chunk]，需 torch.cat(dim=1) 拼回 [1, T, H*D]。
    """
    store = {li: [] for li in range(len(model.model.layers))}

    def hook_factory(si, n, model):
        if si != 0:
            return []
        handles = []
        for li, lyr in enumerate(model.model.layers):
            def mk(li):
                def hook(mod, inp, out):
                    store[li].append(out.detach())
                return hook
            handles.append(lyr.self_attn.k_proj.register_forward_hook(mk(li)))
        return handles

    return store, hook_factory


# ======================================================================== #
# H. 量化重建码本                                                            #
# ======================================================================== #

def submean_sign(x, G, threshold=0.0, submean=True):
    """x: [H, D, T] -> submean + sign/tern 重建。threshold=0 即 sign, 0.5 即死区三值。

    submean=False 时不减均值（用于残差已近零均值的 phase-tracked-mu 方案）。
    末尾不足一组：原样保留（与 sink/recent 同理，不计误差）。
    """
    H, D, T = x.shape
    ng = T // G
    xg = x[:, :, : ng * G].reshape(H, D, ng, G)
    mu = xg.mean(-1, keepdim=True) if submean else torch.zeros_like(xg[..., :1])
    r = xg - mu
    t = threshold * r.abs().mean(-1, keepdim=True)
    mask = r.abs() > t
    mag = (r.abs() * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
    rec = (mu + torch.sign(r) * mag * mask).reshape(H, D, ng * G)
    if ng * G < T:
        rec = torch.cat([rec, x[:, :, ng * G:]], dim=-1)
    return rec


def groupmean_only(x, G):
    """x: [H, D, T] -> 每组用组均值常量重建（0 bit/token，仅侧信息）。"""
    H, D, T = x.shape
    ng = T // G
    xg = x[:, :, : ng * G].reshape(H, D, ng, G)
    rec = xg.mean(-1, keepdim=True).expand_as(xg).reshape(H, D, ng * G)
    if ng * G < T:
        rec = torch.cat([rec, x[:, :, ng * G:]], dim=-1)
    return rec


def submean_codebook(x, G, kind):
    """x: [H, D, T]; kind in {sign, tern, mm2, uni2, uni3}.

    sign/tern = submean + 1-bit sign / 死区三值（τ=0.5）；
    mm2/uni2 = 非对称 min-max 4 电平（2-bit）；uni3 = min-max 8 电平（3-bit）。
    （mm2 与 uni2 数值等价，保留两个名字以匹配两个 blue 探针各自的术语。）
    """
    H, D, T = x.shape
    ng = T // G
    xg = x[:, :, : ng * G].reshape(H, D, ng, G)
    if kind in ("sign", "tern"):
        mu = xg.mean(-1, keepdim=True)
        r = xg - mu
        thr = (0.5 if kind == "tern" else 0.0) * r.abs().mean(-1, keepdim=True)
        mask = r.abs() > thr
        mag = (r.abs() * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
        rec = mu + torch.sign(r) * mag * mask
    elif kind in ("mm2", "uni2", "uni3"):
        L = 8 if kind == "uni3" else 4
        mn = xg.min(-1, keepdim=True).values
        mx = xg.max(-1, keepdim=True).values
        scale = (mx - mn).clamp(min=1e-6) / (L - 1)
        q = ((xg - mn) / scale).round().clamp(0, L - 1)
        rec = q * scale + mn
    else:
        raise ValueError(kind)
    rec = rec.reshape(H, D, ng * G)
    if ng * G < T:
        rec = torch.cat([rec, x[:, :, ng * G:]], dim=-1)
    return rec


def lloyd_codebook(x, G, L=4, iters=12):
    """post-RoPE per-group 最优 L-电平标量量化器（1D Lloyd-Max / k-means）。逐 head 控显存。

    返回重建 [H, D, T]（末尾不足一组保真）。
    """
    H, D, T = x.shape
    ng = T // G
    out = x.clone()
    for h in range(H):
        xg = x[h, :, : ng * G].reshape(D, ng, G)                       # [D, ng, G]
        lo = xg.min(-1, keepdim=True).values
        hi = xg.max(-1, keepdim=True).values
        lev = lo + (hi - lo) * (torch.arange(L, device=x.device) + 0.5) / L   # [D, ng, L]
        for _ in range(iters):
            d = (xg.unsqueeze(-1) - lev.unsqueeze(-2)).abs()           # [D, ng, G, L]
            a = d.argmin(-1)                                           # [D, ng, G]
            oh = torch.nn.functional.one_hot(a, L).to(xg.dtype)
            cnt = oh.sum(-2)                                           # [D, ng, L]
            summ = (oh * xg.unsqueeze(-1)).sum(-2)
            lev = torch.where(cnt > 0, summ / cnt.clamp(min=1), lev)
        d = (xg.unsqueeze(-1) - lev.unsqueeze(-2)).abs()
        a = d.argmin(-1)
        rec = torch.gather(lev, 2, a)
        out[h, :, : ng * G] = rec.reshape(D, ng * G).to(out.dtype)
    return out


def sign_level(r):
    """sign 码本电平 = E|r|（沿最后一轴，keepdim）。"""
    return r.abs().mean(-1, keepdim=True)


# ======================================================================== #
# I. 选择器                                                                  #
# ======================================================================== #

def topk_mask(score, k):
    """score: [L, H, D] -> per-head top-k bool mask [L, H, D]。"""
    L, H, D = score.shape
    idx = score.topk(k, dim=-1).indices
    mask = torch.zeros(L, H, D, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def topk_indices_per_head(score, k):
    """score: [..., D] -> top-k indices [..., k]（沿最后一轴）。"""
    return score.topk(k, dim=-1).indices


def overlap_masks(a_idx, b_idx, D, k):
    """a_idx/b_idx: [..., k] 索引张量 -> 平均重合比例（scatter 成 [...,D] bool 取交 / k）。

    支持任意前导 batch 维（用 *a_idx.shape[:-1]）。
    """
    a1 = torch.zeros(*a_idx.shape[:-1], D, dtype=torch.bool).scatter(-1, a_idx, True)
    b1 = torch.zeros(*b_idx.shape[:-1], D, dtype=torch.bool).scatter(-1, b_idx, True)
    return (a1 & b1).sum(-1).float().mean().item() / k


def overlap_sets(a_idx, b_idx, k):
    """flat-C 索引集合重合 / k（a_idx/b_idx 为 1D index 张量或可迭代）。"""
    sa = set(a_idx.tolist() if torch.is_tensor(a_idx) else a_idx)
    sb = set(b_idx.tolist() if torch.is_tensor(b_idx) else b_idx)
    return len(sa & sb) / k


# ======================================================================== #
# J. 画图 & 报告                                                             #
# ======================================================================== #

def model_basename(model_path) -> str:
    return os.path.basename(str(model_path).rstrip("/"))


def save_fig(fig, outdir, name):
    """makedirs + savefig(dpi=150) + print；返回路径。"""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150)
    print(f"[saved] {path}")
    return path


def share_heatmap(ax, mat, title=None, xlabel=None, ylabel=None,
                  vmin=0.0, vmax=1.0, cmap=CMAP_SHARE, extent=None, aspect="auto"):
    """统一色规的 mu^2-share 热力图（红=mu^2 主导, 蓝=sigma^2 主导）。返回 im。"""
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, extent=extent)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return im


def divergent_heatmap(ax, mat, title=None, xlabel=None, ylabel=None,
                      cmap="PuOr_r", pct=99.5, extent=None, aspect="auto"):
    """发散色图（漂移/偏差），色幅按 |mat| 的 pct 百分位自适应。返回 (im, lim)。"""
    lim = max(0.05, float(np.percentile(np.abs(mat), pct)))
    im = ax.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim, aspect=aspect, extent=extent)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return im, lim
