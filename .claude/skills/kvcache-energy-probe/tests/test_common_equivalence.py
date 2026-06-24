#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯函数等价测试：_common.py 的原语 vs 从原 11 个探针脚本逐字复制的 golden 旧实现。

不需要 GPU / 模型。用固定 seed 的随机张量（T=520 非整除 G=128，以测末尾截尾分支），
对能量分解 / sign·tern·minmax·Lloyd 码本 / RoPE 旋转 / 选择器 / per-token 能量逐元素比对。
atol=0 的断言验证「重构未在复制中引入数值偏差」；往返类断言用小 atol（旋转等距）。

跑（kitty conda env，无 pytest）：
  cd .claude/skills/kvcache-energy-probe
  CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -v
或直接：  python tests/test_common_equivalence.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import torch

import _common as c

H, D, T, G = 4, 8, 520, 128          # T 非整除 G -> ng=4, 末尾 8 token 截尾


def kv():       # [1, H, T, D]
    torch.manual_seed(0)
    return torch.randn(1, H, T, D)


def hdt():      # [H, D, T]
    torch.manual_seed(1)
    return torch.randn(H, D, T)


# ----------------------- golden 旧实现（从原脚本逐字复制） ----------------------- #

def old_flatC(t, G):
    """test_kvcache_submean_energy.group_stats 的核心（flat-C 粒度）。"""
    x = t[0].permute(0, 2, 1)
    Hh, Dd, Tt = x.shape
    Tu = (Tt // G) * G
    x = x[:, :, :Tu].reshape(Hh * Dd, Tu // G, G).float()
    mu = x.mean(dim=-1)
    ex2 = (x * x).mean(dim=-1)
    var = ex2 - mu * mu
    return mu, ex2, var


def old_HD(t, G):
    """test_kvcache_submean_energy_heatmap.channel_stats。"""
    x = t[0].permute(0, 2, 1)
    Hh, Dd, Tt = x.shape
    Tu = (Tt // G) * G
    x = x[:, :, :Tu].reshape(Hh, Dd, Tu // G, G).float()
    mu = x.mean(-1)
    ex2 = (x * x).mean(-1)
    var = (ex2 - mu * mu).clamp_min(0)
    return (mu * mu).sum(-1), var.sum(-1), ex2.sum(-1)


def old_per_token(t):
    """probe_kvlen_energy 的 per-token 能量。"""
    x = t[0].permute(0, 2, 1).reshape(-1, t.shape[2]).float()
    return (x * x).sum(0)


def old_submean_sign(x, G, threshold=0.0, submean=True):
    """probe_typed_channel_quant.submean_sign。"""
    Hh, Dd, Tt = x.shape
    ng = Tt // G
    xg = x[:, :, : ng * G].reshape(Hh, Dd, ng, G)
    mu = xg.mean(-1, keepdim=True) if submean else torch.zeros_like(xg[..., :1])
    r = xg - mu
    t = threshold * r.abs().mean(-1, keepdim=True)
    mask = r.abs() > t
    mag = (r.abs() * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
    rec = (mu + torch.sign(r) * mag * mask).reshape(Hh, Dd, ng * G)
    if ng * G < Tt:
        rec = torch.cat([rec, x[:, :, ng * G:]], dim=-1)
    return rec


def old_groupmean_only(x, G):
    """probe_typed_channel_quant.groupmean_only。"""
    Hh, Dd, Tt = x.shape
    ng = Tt // G
    xg = x[:, :, : ng * G].reshape(Hh, Dd, ng, G)
    rec = xg.mean(-1, keepdim=True).expand_as(xg).reshape(Hh, Dd, ng * G)
    if ng * G < Tt:
        rec = torch.cat([rec, x[:, :, ng * G:]], dim=-1)
    return rec


def old_submean_codebook(x, G, kind):
    """probe_blue/postrope 的 submean_codebook（sign/tern/mm2/uni2/uni3 合并）。"""
    Hh, Dd, Tt = x.shape
    ng = Tt // G
    xg = x[:, :, : ng * G].reshape(Hh, Dd, ng, G)
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
    rec = rec.reshape(Hh, Dd, ng * G)
    if ng * G < Tt:
        rec = torch.cat([rec, x[:, :, ng * G:]], dim=-1)
    return rec


def old_lloyd(x, G, L=4, iters=12):
    """probe_postrope_blue_codebook.lloyd_codebook。"""
    Hh, Dd, Tt = x.shape
    ng = Tt // G
    out = x.clone()
    for h in range(Hh):
        xg = x[h, :, : ng * G].reshape(Dd, ng, G)
        lo = xg.min(-1, keepdim=True).values
        hi = xg.max(-1, keepdim=True).values
        lev = lo + (hi - lo) * (torch.arange(L, device=x.device) + 0.5) / L
        for _ in range(iters):
            d = (xg.unsqueeze(-1) - lev.unsqueeze(-2)).abs()
            a = d.argmin(-1)
            oh = torch.nn.functional.one_hot(a, L).to(xg.dtype)
            cnt = oh.sum(-2)
            summ = (oh * xg.unsqueeze(-1)).sum(-2)
            lev = torch.where(cnt > 0, summ / cnt.clamp(min=1), lev)
        d = (xg.unsqueeze(-1) - lev.unsqueeze(-2)).abs()
        a = d.argmin(-1)
        rec = torch.gather(lev, 2, a)
        out[h, :, : ng * G] = rec.reshape(Dd, ng * G).to(out.dtype)
    return out


def old_rope_rotate(x, cosT, sinT, inverse=False):
    Dd = x.shape[1]
    a, b = x[:, : Dd // 2], x[:, Dd // 2:]
    cc, s = cosT, (-sinT if inverse else sinT)
    return torch.cat([a * cc - b * s, b * cc + a * s], dim=1)


def eq(a, b):
    return torch.equal(a, b)


def close(a, b, atol=1e-6):
    # 跨张量形状的 mean 在 CPU 上归约顺序不同 -> ~1e-7 ULP 差异（功能等价、非 bug）。
    # 同形状归约逐位相等，见 test_HD_reduce_matches_old（用 eq/torch.equal）。
    return torch.allclose(a, b, atol=atol, rtol=0)


# ------------------------------------ tests ------------------------------------ #

class TestEnergyPrimitives(unittest.TestCase):
    def test_group_decompose_identity(self):
        gs = c.group_decompose(kv(), G)
        # ex2 = mu^2 + var 数学恒等；fp 非结合 mu^2+(ex2-mu^2)!=ex2 逐位，故 allclose。
        self.assertTrue(close(gs.ex2, gs.mu * gs.mu + gs.var))
        self.assertEqual(tuple(gs.xg.shape), (H, D, T // G, G))

    def test_flatC_matches_old(self):
        t = kv()
        gs = c.group_decompose(t, G)
        ng = T // G
        mu, ex2, var = old_flatC(t, G)
        # group_decompose 在 [H,D,ng,G] 上算、再 reshape 成 flat-C；原脚本直接在
        # [H*D,ng,G] 上算 -> CPU mean 归约形状不同，~1e-7 ULP 差异（功能等价）。
        self.assertTrue(close(gs.mu.reshape(H * D, ng), mu))
        self.assertTrue(close(gs.ex2.reshape(H * D, ng), ex2))
        self.assertTrue(close(gs.var.reshape(H * D, ng), var))

    def test_HD_reduce_matches_old(self):
        t = kv()
        gs = c.group_decompose(t, G)
        mu2, var, ex2 = old_HD(t, G)
        self.assertTrue(eq((gs.mu * gs.mu).sum(-1), mu2))
        self.assertTrue(eq(gs.var.clamp_min(0).sum(-1), var))
        self.assertTrue(eq(gs.ex2.sum(-1), ex2))

    def test_fp16_input_floats_before_reduce(self):
        t16 = kv().half()
        gs = c.group_decompose(t16, G)
        mu, ex2, var = old_flatC(t16, G)
        self.assertEqual(gs.mu.dtype, torch.float32)   # 关键：fp16 -> 统计前已升 fp32
        self.assertTrue(close(gs.mu.reshape(H * D, T // G), mu))
        self.assertTrue(close(gs.var.reshape(H * D, T // G), var))

    def test_per_token_energy(self):
        t = kv()
        self.assertTrue(eq(c.per_token_energy(t), old_per_token(t)))


class TestReconstruction(unittest.TestCase):
    def test_submean_sign_variants(self):
        x = hdt()
        for thr in (0.0, 0.5):
            for sm in (True, False):
                self.assertTrue(eq(c.submean_sign(x, G, thr, sm),
                                   old_submean_sign(x, G, thr, sm)),
                                f"submean_sign thr={thr} submean={sm}")

    def test_groupmean_only(self):
        x = hdt()
        self.assertTrue(eq(c.groupmean_only(x, G), old_groupmean_only(x, G)))

    def test_submean_codebook_kinds(self):
        x = hdt()
        for kind in ("sign", "tern", "mm2", "uni2", "uni3"):
            self.assertTrue(eq(c.submean_codebook(x, G, kind),
                               old_submean_codebook(x, G, kind)), kind)

    def test_mm2_equals_uni2(self):
        x = hdt()
        self.assertTrue(eq(c.submean_codebook(x, G, "mm2"),
                           c.submean_codebook(x, G, "uni2")))

    def test_lloyd_codebook(self):
        x = hdt()
        self.assertTrue(eq(c.lloyd_codebook(x, G, L=4), old_lloyd(x, G, L=4)))

    def test_sign_level(self):
        x = hdt()
        ng = T // G
        xg = x[:, :, :ng * G].reshape(H, D, ng, G)
        r = xg - xg.mean(-1, keepdim=True)
        self.assertTrue(eq(c.sign_level(r), r.abs().mean(-1, keepdim=True)))


class TestRoPE(unittest.TestCase):
    def _tables(self):
        torch.manual_seed(2)
        theta = torch.randn(D // 2, T)
        return torch.cos(theta), torch.sin(theta)

    def test_rope_rotate_matches_old(self):
        x = hdt()
        cosT, sinT = self._tables()
        for inv in (False, True):
            self.assertTrue(eq(c.rope_rotate(x, cosT, sinT, inv),
                               old_rope_rotate(x, cosT, sinT, inv)), f"inverse={inv}")

    def test_rope_roundtrip_is_isometric(self):
        x = hdt()
        cosT, sinT = self._tables()       # 真旋转 (cos^2+sin^2=1)
        back = c.rope_rotate(c.rope_rotate(x, cosT, sinT), cosT, sinT, inverse=True)
        self.assertTrue(torch.allclose(back, x, atol=1e-5))


class TestSelectors(unittest.TestCase):
    def test_topk_mask(self):
        torch.manual_seed(3)
        score = torch.randn(3, 4, D)      # [L,H,D]
        k = 3
        mask = c.topk_mask(score, k)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(eq(mask.sum(-1), torch.full((3, 4), k)))
        # 被选中的应是每行最大的 k 个
        idx = score.topk(k, dim=-1).indices
        ref = torch.zeros(3, 4, D, dtype=torch.bool).scatter(-1, idx, True)
        self.assertTrue(eq(mask, ref))

    def test_topk_indices_per_head(self):
        torch.manual_seed(4)
        score = torch.randn(2, 4, D)
        self.assertTrue(eq(c.topk_indices_per_head(score, 3),
                           score.topk(3, dim=-1).indices))

    def test_overlap_masks_full_and_disjoint(self):
        a = torch.tensor([[0, 1, 2]])     # [1, k]
        self.assertAlmostEqual(c.overlap_masks(a, a, D, 3), 1.0)          # 全重合
        b = torch.tensor([[3, 4, 5]])
        self.assertAlmostEqual(c.overlap_masks(a, b, D, 3), 0.0)          # 不相交
        cc = torch.tensor([[0, 1, 7]])
        self.assertAlmostEqual(c.overlap_masks(a, cc, D, 3), 2.0 / 3.0)   # 2/3 重合

    def test_overlap_masks_batched(self):
        a = torch.tensor([[[0, 1], [2, 3]]])   # [1,2,k]
        self.assertAlmostEqual(c.overlap_masks(a, a, D, 2), 1.0)

    def test_overlap_sets(self):
        a = torch.tensor([0, 1, 2, 3])
        b = torch.tensor([2, 3, 4, 5])
        self.assertAlmostEqual(c.overlap_sets(a, b, 4), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
