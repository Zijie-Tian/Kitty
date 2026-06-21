# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py.
# Author: Haojun Xia (xhjustc@gmail.com)

from typing import Optional, Any
from dataclasses import dataclass
import argparse

import torch
try:
    from transformers.cache_utils import CacheConfig, DynamicCache
except ImportError:  # transformers>=4.57 removed CacheConfig from cache_utils
    from transformers.cache_utils import DynamicCache

    class CacheConfig:  # minimal compatibility shim for Kitty's local config
        def __init__(self, cache_implementation: str | None = None) -> None:
            self.cache_implementation = cache_implementation

from .utils_quant import build_promote_mask, fake_quant_groupwise_lastdim


@dataclass
class KittyKVCacheConfig(CacheConfig):
    """
    Configuration class for Kitty KV cache settings.

    Attributes:
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
    """
    def __init__(
        self,
        sink_length: int = 32,
        buffer_length: int = 128,
        group_size: int = 128,
        kbits: int = 2,
        vbits: int = 2,
        promote_ratio: float = 0.1,
        promote_bit: int = 4,
        promote_ratio_per_layer: Optional[dict] = None,  # {layer_idx: ratio} overriding promote_ratio per layer; None = scalar for every layer
        channel_selection: int = 1,               # -1: Unspecified, 0: Random, 1: Magnitude-based, 3: Cross-head Magnitude (layer-global budget)
        k_quant_mode: str = "per_channel",        # "per_channel": KIVI-style token-axis groups (+promote/qlut codebook); "per_token": K quantized like V along head_dim (uniform, no promote)
        VCache_BitDecoding: bool = False,         # The behavior of Value Cache, set to True means BitDecoding, otherwise KIVI Style Value Cache
        PostQuant: bool = True,                   # Post Quantization is always enabled
        k_codebook: str = "kivi",                 # "kivi" = existing min-max groupwise K quant; "qlut" = per-channel sigma^2-binned codebooks (qlutattn-k1v4)
        bin_codebooks: Optional[list] = None,     # qlut only: list[str] mapping sigma^2-bin -> codebook (meanonly/sign/tern/uni2/nf2/uni3)
        n_bins: int = 6,                          # qlut only: number of per-layer sigma^2 quantile bins
        pertoken_outlier_k: int = 0,              # per_token only: keep top-k peak-|magnitude| channels (per head, fixed) out of the shared per-token scale (dense-and-sparse). 0 = off.
        pertoken_outlier_bits: int = 4,           # per_token only: precision of the kept outlier channels (per-channel along token; >=16 = fp16)
        pertoken_pc_submean: bool = False,        # per_token only: subtract a per-CHANNEL mean (cached at prefill, free for attention) then pure binary/ternary on the residual (qlutattn-k125v4-pt). NOT the per-token submean.
        pertoken_mixed: bool = False,             # per_token only: per-channel submean + sigma^2-binned MIXED codebook (ONLINE sigma^2, legacy k1.68v4-pt path); bin_codebooks = per-bin policy (low sigma^2 -> bin 0).
        pertoken_cb_mask: Optional[str] = None,   # per_token only: path to an OFFLINE per-(layer,head,channel) codebook mask (the corrected k168v4-pt). When set, each channel uses bin_codebooks[mask[c]] FIXED (offline sigma^2 calibration -> sign/tern), no online sigma^2 binning, no nf2.
    ):
        super().__init__("kitty_kv")
        self.sink_length = sink_length
        self.buffer_length = buffer_length
        self.group_size = group_size
        self.kbits = kbits
        self.vbits = vbits
        self.promote_ratio = promote_ratio
        self.promote_bit = promote_bit
        # Per-layer override map {layer_idx: ratio}; layers absent from the map
        # fall back to the scalar promote_ratio above. None => scalar everywhere.
        self.promote_ratio_per_layer = (
            {int(k): float(v) for k, v in promote_ratio_per_layer.items()}
            if promote_ratio_per_layer is not None
            else None
        )
        self.channel_selection = channel_selection
        self.k_quant_mode = k_quant_mode
        self.VCache_BitDecoding = VCache_BitDecoding
        self.PostQuant = PostQuant
        self.k_codebook = k_codebook
        self.bin_codebooks = list(bin_codebooks) if bin_codebooks is not None else None
        self.n_bins = n_bins
        self.pertoken_outlier_k = pertoken_outlier_k
        self.pertoken_outlier_bits = pertoken_outlier_bits
        self.pertoken_pc_submean = pertoken_pc_submean
        self.pertoken_mixed = pertoken_mixed
        self.pertoken_cb_mask = pertoken_cb_mask
        #
        self.validate()

    def validate(self):
        """Validates if the arguments passed are correct"""
        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        if self.k_codebook not in ("kivi", "qlut"):
            raise ValueError(
                incorrect_arg_msg.format(key="k_codebook", correct_value="'kivi' or 'qlut'",
                                         found_value=self.k_codebook))
        if self.k_codebook == "qlut" and not self.bin_codebooks:
            raise ValueError("k_codebook='qlut' requires a non-empty bin_codebooks list")
        if self.k_quant_mode not in ("per_channel", "per_token"):
            raise ValueError(
                incorrect_arg_msg.format(key="k_quant_mode", correct_value="'per_channel' or 'per_token'",
                                         found_value=self.k_quant_mode))
        if self.k_quant_mode == "per_token" and (self.promote_ratio != 0.0 or self.promote_ratio_per_layer is not None):
            raise ValueError(
                "k_quant_mode='per_token' requires promote_ratio=0.0 and no per-layer override "
                "(per-token K has no channel axis at quantization time)")
        if self.channel_selection not in [0, 1, 3]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="channel_selection",
                    correct_value="0, 1 or 3",
                    found_value=self.channel_selection,
                ),
            )
        if self.buffer_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="buffer_length",
                    correct_value="larger than 0",
                    found_value=self.buffer_length,
                ),
            )
        if self.sink_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="sink_length",
                    correct_value="larger than 0",
                    found_value=self.sink_length,
                ),
            )
        if self.group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="group_size",
                    correct_value="larger than 0",
                    found_value=self.group_size,
                ),
            )
        if self.kbits < 1 or self.kbits > 16:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="kbits",
                    correct_value="1 to 16",
                    found_value=self.kbits,
                ),
            )
        if self.vbits < 1 or self.vbits > 16:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="vbits",
                    correct_value="1 to 16",
                    found_value=self.vbits,
                ),
            )
        if self.promote_ratio < 0.0 or self.promote_ratio > 1.0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="promote_ratio",
                    correct_value="between 0.0 and 1.0",
                    found_value=self.promote_ratio,
                ),
            )
        if self.promote_ratio > 0 and self.promote_bit < self.kbits:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="promote_bit",
                    correct_value=f"promote_bit should be larger than kbits ({self.kbits})",
                    found_value=self.promote_bit,
                ),
            )
        if self.promote_bit <= 0 or self.promote_bit > 16:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="promote_bit",
                    correct_value=f"between 1 and 16 (16 = keep promoted channels in fp16)",
                    found_value=self.promote_bit,
                ),
            )
        if self.promote_ratio_per_layer is not None:
            for idx, r in self.promote_ratio_per_layer.items():
                if not (0.0 <= r <= 1.0):
                    raise ValueError(
                        incorrect_arg_msg.format(
                            key=f"promote_ratio_per_layer[{idx}]",
                            correct_value="between 0.0 and 1.0",
                            found_value=r,
                        ),
                    )
            if any(r > 0 for r in self.promote_ratio_per_layer.values()) and self.promote_bit < self.kbits:
                raise ValueError(
                    incorrect_arg_msg.format(
                        key="promote_bit",
                        correct_value=f"promote_bit should be >= kbits ({self.kbits}) when any per-layer promote_ratio > 0",
                        found_value=self.promote_bit,
                    ),
                )
        if self.VCache_BitDecoding not in [False]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="VCache_BitDecoding",
                    correct_value="False",
                    found_value=self.VCache_BitDecoding,
                ),
            )
        if self.group_size > self.buffer_length or self.buffer_length % self.group_size != 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="group_size",
                    correct_value="a factor of buffer_length ({})".format(self.buffer_length),
                    found_value=self.group_size,
                ),
            )
        
class KittyKVCache(DynamicCache):
    """
    A quantizer cache that supports Kitty quantization.
    [batch_size, num_heads, seq_len, head_dim]
    """

    def __init__(self, cache_config: KittyKVCacheConfig) -> None:
        super().__init__()
        # transformers<=4.56 DynamicCache exposed key_cache/value_cache lists.
        # transformers>=4.57 stores layers internally instead. Kitty's quantized
        # cache logic is intentionally list-based, so keep local legacy-style
        # lists and override the small Cache API surface that generation uses.
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        # Initialize Kitty-KV specific configurations
        self.sink_length = cache_config.sink_length
        self.buffer_length = cache_config.buffer_length
        self.group_size = cache_config.group_size
        self.kbits = cache_config.kbits
        self.vbits = cache_config.vbits
        self.promote_ratio = cache_config.promote_ratio
        # Per-layer promote_ratio override {layer_idx: ratio}; None => scalar.
        self.promote_ratio_per_layer = cache_config.promote_ratio_per_layer
        self.promote_bit = cache_config.promote_bit
        self.channel_selection = cache_config.channel_selection
        self.VCache_BitDecoding = cache_config.VCache_BitDecoding
        self.PostQuant = cache_config.PostQuant
        self.cache_implementation = cache_config.cache_implementation
        # QLUT (sigma^2-binned) K codebook support. k_bin_ids[layer_idx] = [nh,D]
        # per-channel bin id, computed once (from the prompt quant region at prefill)
        # then reused for every buffer flush + decode. None for the default 'kivi' path.
        self.k_quant_mode = cache_config.k_quant_mode
        self.k_codebook = cache_config.k_codebook
        self.bin_codebooks = cache_config.bin_codebooks
        self.n_bins = cache_config.n_bins
        self.k_bin_ids: dict[int, torch.Tensor] = {}
        # per_token dense-and-sparse: per-layer fixed outlier channel ids [nh,k]
        self.pertoken_outlier_k = getattr(cache_config, "pertoken_outlier_k", 0)
        self.pertoken_outlier_bits = getattr(cache_config, "pertoken_outlier_bits", 4)
        self.k_outlier_ids: dict[int, torch.Tensor] = {}
        # k125v4-pt: per-channel mean (cached at prefill, reused at decode) for the
        # per-channel-center + per-token pure-binary path. Free for attention.
        self.pertoken_pc_submean = getattr(cache_config, "pertoken_pc_submean", False)
        self.k_pc_mean: dict[int, torch.Tensor] = {}
        # k1.68v4-pt: per-channel submean + sigma^2-binned mixed codebook. k_mix_bins
        # caches the per-layer per-channel sigma^2-bin id [nh,D] (computed once at prefill).
        self.pertoken_mixed = getattr(cache_config, "pertoken_mixed", False)
        self.k_mix_bins: dict[int, torch.Tensor] = {}
        # k168v4-pt (corrected): OFFLINE per-(layer,head,channel) codebook mask (sign/tern),
        # loaded once and used unchanged -- NOT recomputed per prompt. k_cb_mask[layer]=[nh,D].
        self.pertoken_cb_mask_path = getattr(cache_config, "pertoken_cb_mask", None)
        self.pertoken_offline = bool(self.pertoken_cb_mask_path)
        self.k_cb_mask: dict[int, torch.Tensor] = {}
        if self.pertoken_offline:
            _blob = torch.load(self.pertoken_cb_mask_path, map_location="cpu", weights_only=False)
            _m = _blob["codebook_mask"]                        # [nl, n_kv, D] uint8 (0=low,1=high)
            for _li in range(_m.shape[0]):
                self.k_cb_mask[_li] = _m[_li].long()           # [n_kv, D]
            _cbs = _blob.get("codebooks")
            if _cbs:                                           # mask file defines the codebooks (sign/tern, sign/nf2, ...)
                self.bin_codebooks = list(_cbs)
            print(f"[qlutattn-offline] loaded codebook mask {tuple(_m.shape)} from "
                  f"{self.pertoken_cb_mask_path} codebooks={self.bin_codebooks} "
                  f"nominal~{_blob.get('nominal_bits')}")
        #
        #self.query_cache: list[torch.Tensor] = []
        #self.query_score: list[torch.Tensor] = []

    def _layer_pr(self, layer_idx: int) -> float:
        """Promote ratio for this layer: per-layer override if present, else scalar."""
        if self.promote_ratio_per_layer is not None:
            return self.promote_ratio_per_layer.get(layer_idx, self.promote_ratio)
        return self.promote_ratio

    def _ensure_k_bins(self, layer_idx, key_region_t):
        """Cache per-channel sigma^2-bins for a layer from a [B,nh,D,Tq] region.
        Called once with the full prompt quant region at prefill; no-op if set."""
        if self.k_codebook != "qlut" or layer_idx in self.k_bin_ids:
            return
        from .qlut_quant import compute_sigma_bins
        self.k_bin_ids[layer_idx] = compute_sigma_bins(key_region_t, self.group_size, self.n_bins)

    @staticmethod
    def _pure_pt_codebook(sub, cb):
        """Per-token PURE codebook over `sub` [...,nch] (last axis = a sigma^2-bin's
        channels of one token). sign/tern are pure (NO submean -- the residual is
        already per-channel centered); nf2/uni2/etc. go through apply_codebook (Lloyd)."""
        if cb == "sign":
            mag = sub.abs().mean(-1, keepdim=True)
            return torch.sign(sub) * mag
        if cb == "tern":
            mag = sub.abs().mean(-1, keepdim=True)
            mask = sub.abs() > 0.5 * mag
            m2 = (sub.abs() * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
            return torch.sign(sub) * m2 * mask
        from .qlut_quant import apply_codebook
        return apply_codebook(sub, sub.shape[-1], cb)

    @staticmethod
    def _masked_lloyd_lastdim(r, mb, L=4, iters=10):
        """Vectorized masked Lloyd-Max along head_dim, matching qlut_quant._lloyd
        but only over channels where mb is True (per [B,nh,T] row). Lets every head
        run its per-bin nf2 quantizer in one shot (no per-head Python loop).
        r:[B,nh,T,D] float, mb:[1,nh,1,D] bool -> reconstruction [B,nh,T,D] (caller
        zeros the non-bin channels via *mb). Non-bin channels never affect the
        levels (one-hot is masked), so this is numerically identical to extracting
        the bin's channels and calling _lloyd on them."""
        import torch.nn.functional as F
        neg = (~mb).expand_as(r)
        lo = r.masked_fill(neg, float("inf")).amin(-1, keepdim=True)
        hi = r.masked_fill(neg, float("-inf")).amax(-1, keepdim=True)
        empty = lo > hi                                                # all-False row: a head with NO channel in this bin
        lo = torch.where(empty, torch.zeros_like(lo), lo)             # avoid +inf/-inf -> NaN levels (this head is masked out anyway)
        hi = torch.where(empty, torch.zeros_like(hi), hi)
        ar = torch.arange(L, device=r.device, dtype=r.dtype)
        lev = lo + (hi - lo) * (ar + 0.5) / L                          # [B,nh,T,L]
        mbf = mb.to(r.dtype).unsqueeze(-1)                             # [1,nh,1,D,1]
        for _ in range(iters):
            d = (r.unsqueeze(-1) - lev.unsqueeze(-2)).abs()           # [B,nh,T,D,L]
            a = d.argmin(-1)                                          # [B,nh,T,D]
            oh = F.one_hot(a, L).to(r.dtype) * mbf                    # mask non-bin channels
            cnt = oh.sum(-2)                                          # [B,nh,T,L]
            summ = (oh * r.unsqueeze(-1)).sum(-2)
            lev = torch.where(cnt > 0, summ / cnt.clamp(min=1), lev)
        a = (r.unsqueeze(-1) - lev.unsqueeze(-2)).abs().argmin(-1)    # [B,nh,T,D]
        return torch.gather(lev, -1, a)                              # [B,nh,T,D]

    @staticmethod
    def _pt_codebook_masked(r, m, cb):
        """Per-token codebook on r:[B,nh,T,D] over the channels flagged by m:[nh,D],
        VECTORIZED across heads (replaces the per-head Python loop in the decode
        path). Returns a reconstruction that is ZERO outside the bin's channels so
        callers can sum across bins. Numerically matches per-bin
        _pure_pt_codebook(extracted-channels): the masked reductions use exactly the
        bin's channels, head by head."""
        mb = m[None, :, None, :]                                      # [1,nh,1,D] bool
        cnt = m.sum(-1).clamp(min=1)[None, :, None, None].to(r.dtype)  # [1,nh,1,1]
        if cb == "sign":
            mag = (r.abs() * mb).sum(-1, keepdim=True) / cnt          # [B,nh,T,1] masked mean
            return torch.sign(r) * mag * mb
        if cb == "tern":
            mag = (r.abs() * mb).sum(-1, keepdim=True) / cnt
            tmask = (r.abs() > 0.5 * mag) & mb
            denom = tmask.sum(-1, keepdim=True).clamp(min=1).to(r.dtype)
            mag2 = (r.abs() * tmask).sum(-1, keepdim=True) / denom
            return torch.sign(r) * mag2 * tmask
        if cb == "meanonly":
            return ((r * mb).sum(-1, keepdim=True) / cnt) * mb
        if cb in ("uni2", "uni3"):
            L = 4 if cb == "uni2" else 8
            neg = (~mb).expand_as(r)
            mn = r.masked_fill(neg, float("inf")).amin(-1, keepdim=True)
            mx = r.masked_fill(neg, float("-inf")).amax(-1, keepdim=True)
            empty = mn > mx                                           # all-False row guard (no channel in this bin)
            mn = torch.where(empty, torch.zeros_like(mn), mn)
            mx = torch.where(empty, torch.zeros_like(mx), mx)
            scale = (mx - mn).clamp(min=1e-6) / (L - 1)
            q = ((r - mn) / scale).round().clamp(0, L - 1)
            return (q * scale + mn) * mb
        if cb == "nf2":
            return KittyKVCache._masked_lloyd_lastdim(r, mb, L=4, iters=10) * mb
        if cb == "fp16":
            return r * mb
        raise ValueError(cb)

    def _quant_k_pertoken(self, ks, layer_idx=0):
        """Per-token K quant of a [B,nh,T,D] slice: one quantizer per token per
        head along head_dim (like the KIVI-style V cache). k_codebook='qlut'
        applies a SINGLE submean codebook along head_dim -- sigma^2 binning has no
        per-channel axis in per-token mode, so bin_codebooks[0] is used for every
        token (mu = per-token mean over head_dim channels); 'kivi' uses uniform
        min-max. Pair with a SmoothAttention checkpoint to flatten per-channel
        outliers that per-token sharing would otherwise smear into one scale.

        pertoken_outlier_k>0 adds dense-and-sparse isolation (the autoresearch
        per-token champion): the top-k PEAK-|magnitude| channels per head (fixed,
        found once at prefill -- they dominate the shared per-token scale) are
        pulled out and quantized per-channel at pertoken_outlier_bits, while the
        remaining channels get the per-token codebook with a scale computed over
        ONLY them. k=8 @4bit + Lloyd (+ smoothed ckpt) is the winner."""
        if self.k_codebook != "qlut":
            return fake_quant_groupwise_lastdim(ks, self.group_size, self.kbits)
        from .qlut_quant import apply_codebook
        cb = self.bin_codebooks[0]
        B, nh, T, D = ks.shape
        # k168v4-pt (corrected): OFFLINE sign/tern per-channel codebook. The channel->
        # codebook assignment is fixed by offline sigma^2 calibration (self.k_cb_mask),
        # NOT recomputed per prompt; the per-channel MEAN is still self-calibrated at
        # prefill (free for attention). Per-token 2-codebook quant on the residual,
        # vectorized across heads (reuses _pt_codebook_masked). No nf2, no online bins.
        if self.pertoken_offline:
            if layer_idx not in self.k_pc_mean and T >= D:
                self.k_pc_mean[layer_idx] = ks[0].float().mean(dim=1)      # [nh,D] per-channel mean
            mu = self.k_pc_mean.get(layer_idx)
            muB = (mu[None, :, None, :] if mu is not None
                   else ks.float().mean(dim=3, keepdim=True))              # short-prompt fallback
            r = ks.float() - muB
            cb_id = self.k_cb_mask.get(layer_idx)
            if cb_id is None:                                              # layer absent from mask: single codebook
                full = torch.ones(nh, D, dtype=torch.bool, device=ks.device)
                return (muB + self._pt_codebook_masked(r, full, self.bin_codebooks[0])).to(ks.dtype)
            if cb_id.device != ks.device:
                cb_id = cb_id.to(ks.device)
                self.k_cb_mask[layer_idx] = cb_id
            out = torch.zeros_like(r)
            for ci, cbk in enumerate(self.bin_codebooks):                 # e.g. ["sign","tern"]
                m = (cb_id == ci)                                         # [nh,D]
                if not m.any():
                    continue
                out = out + self._pt_codebook_masked(r, m, cbk)
            return (muB + out).to(ks.dtype)
        # k1.68v4-pt: per-channel submean + sigma^2-binned MIXED codebook. Channels are
        # binned once (at prefill) by per-channel residual sigma^2 into len(bin_codebooks)
        # quantile bins; each bin's channels get its codebook, per-token, on the
        # per-channel-centered residual. Mirrors qlutattn-k1v4 but per-token.
        if self.pertoken_mixed:
            from .qlut_quant import sigma2_bins
            policy = self.bin_codebooks
            nbins = len(policy)
            if layer_idx not in self.k_pc_mean and T >= D:
                x0 = ks[0].float()                                          # [nh,T,D]
                mu0 = x0.mean(dim=1)                                        # [nh,D] per-channel mean
                self.k_pc_mean[layer_idx] = mu0
                sig2 = (x0 - mu0[:, None, :]).pow(2).mean(dim=1)           # [nh,D] residual var over tokens
                self.k_mix_bins[layer_idx] = sigma2_bins(sig2, nbins)      # [nh,D] sigma^2 quantile bins
            mu = self.k_pc_mean.get(layer_idx)
            binid = self.k_mix_bins.get(layer_idx)
            if mu is None:                                                  # short-prompt fallback: single codebook
                out = ks.clone()
                for b in range(B):
                    out[b] = apply_codebook(ks[b].float(), D, policy[0]).to(ks.dtype)
                return out
            muB = mu[None, :, None, :]
            r = ks.float() - muB
            # Decode (T==1) is ~99% of the per-token cost: the per-head x per-bin
            # Python loop fires ~750 tiny kernel launches per step. Vectorize across
            # heads with per-bin masks (numerically identical to the per-head extract).
            if T == 1:
                out = torch.zeros_like(r)
                for bi in range(nbins):
                    m = (binid == bi)                                     # [nh,D]
                    if not m.any():
                        continue
                    out = out + self._pt_codebook_masked(r, m, policy[bi])
                return (muB + out).to(ks.dtype)
            # Prefill: one big call. Keep the per-head extract loop -- it is already
            # GPU-efficient on the large tensor and far lighter on memory than a
            # full-head_dim masked Lloyd over ~32k tokens.
            out = torch.empty_like(r)
            for h in range(nh):
                oh = torch.empty_like(r[:, h])                             # [B,T,D]
                for bi in range(nbins):
                    mask = (binid[h] == bi)
                    if not mask.any():
                        continue
                    oh[:, :, mask] = self._pure_pt_codebook(r[:, h][:, :, mask], policy[bi])
                out[:, h] = oh
            return (muB + out).to(ks.dtype)
        # k125v4-pt: subtract a per-CHANNEL mean (cached once at prefill, reused at
        # decode -- free for attention since q.mu is a per-query constant that cancels
        # in softmax), then PURE binary/ternary on the residual (NO second per-token
        # submean). This is the confirmed per-token sign recipe (per-channel center).
        if self.pertoken_pc_submean:
            if layer_idx not in self.k_pc_mean and T >= D:
                self.k_pc_mean[layer_idx] = ks[0].float().mean(dim=1)      # [nh,D] per-channel mean over tokens
            mu = self.k_pc_mean.get(layer_idx)
            muB = (mu[None, :, None, :] if mu is not None
                   else ks.float().mean(dim=3, keepdim=True))              # short-prompt fallback
            r = ks.float() - muB
            mag = r.abs().mean(dim=3, keepdim=True)                        # [B,nh,T,1] per-token scale
            if cb == "tern":
                mask = r.abs() > 0.5 * mag
                mag2 = (r.abs() * mask).sum(3, keepdim=True) / mask.sum(3, keepdim=True).clamp(min=1)
                q = torch.sign(r) * mag2 * mask
            else:                                                          # "sign": pure 1-bit binary
                q = torch.sign(r) * mag
            return (muB + q).to(ks.dtype)
        ok, obits = self.pertoken_outlier_k, self.pertoken_outlier_bits
        if ok <= 0:
            out = ks.clone()
            for b in range(B):
                out[b] = apply_codebook(ks[b].float(), D, cb).to(ks.dtype)
            return out
        # fix the outlier channels once from a long-enough slice (prefill region);
        # reuse for every buffer flush + decode step.
        if layer_idx not in self.k_outlier_ids and T >= D:
            amax = ks[0].abs().amax(dim=1)                      # [nh,D] peak |K| per channel
            self.k_outlier_ids[layer_idx] = amax.topk(ok, dim=1).indices  # [nh,k]
        ids = self.k_outlier_ids.get(layer_idx)
        out = ks.clone()
        for b in range(B):
            x = ks[b].float()                                   # [nh,T,D]
            ob = x.clone()
            for h in range(nh):
                if ids is None:                                 # bins not set yet (region < D); plain per-token
                    ob[h] = apply_codebook(x[h:h + 1], D, cb)[0]
                    continue
                keep = torch.zeros(D, dtype=torch.bool, device=x.device)
                keep[ids[h]] = True
                nb = ~keep
                sub = x[h:h + 1][:, :, nb]                       # [1,T,D-k] non-outlier
                ob[h][:, nb] = apply_codebook(sub, int(nb.sum().item()), cb)[0]
                if obits < 16 and T > 1:                         # outliers: per-channel uniform @obits
                    kept = x[h][:, keep].transpose(0, 1)[None, None]         # [1,1,k,T]
                    ob[h][:, keep] = fake_quant_groupwise_lastdim(
                        kept, min(self.group_size, T), obits)[0, 0].transpose(0, 1)  # [T,k]
            out[b] = ob.to(ks.dtype)
        return out

    def _quant_k_buffer(self, key_slice_t, layer_idx):
        """Quantize a [B,nh,D,buffer] post-RoPE K buffer. 'qlut' uses the layer's
        sigma^2-bins (lazily binned from this buffer if prefill never set them);
        default 'kivi' uses the existing min-max groupwise + promote path."""
        if self.k_codebook == "qlut":
            from .qlut_quant import fake_quant_qlut_buffer
            self._ensure_k_bins(layer_idx, key_slice_t)
            return fake_quant_qlut_buffer(
                key_slice_t, self.k_bin_ids[layer_idx], self.bin_codebooks, self.group_size)
        promote_mask = build_promote_mask(key_slice_t, self._layer_pr(layer_idx), self.channel_selection)
        return fake_quant_groupwise_lastdim(
            key_slice_t, self.group_size, self.kbits, promote_mask, self.promote_bit)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return cached sequence length for transformers cache/mask helpers."""
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """Return dynamic-cache mask dimensions for transformers>=4.57."""
        kv_length = self.get_seq_length(layer_idx) + cache_position.shape[0]
        return kv_length, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Dynamic Kitty caches do not have a fixed maximum length."""
        return -1

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache batch dimension for generation helpers."""
        for layer_idx in range(len(self.key_cache)):
            device_index = beam_idx.to(self.key_cache[layer_idx].device)
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, device_index)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, device_index)

    def crop(self, max_length: int):
        """Crop cache sequence length, matching DynamicCache helper semantics."""
        if max_length < 0:
            max_length = self.get_seq_length() + max_length
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :max_length, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int):
        """Repeat batch entries for generation helpers."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Select batch entries for generation helpers."""
        for layer_idx in range(len(self.key_cache)):
            device_indices = indices.to(self.key_cache[layer_idx].device)
            self.key_cache[layer_idx] = self.key_cache[layer_idx][device_indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][device_indices, ...]

    def reset(self):
        """Clear all cached tensors."""
        self.key_cache.clear()
        self.value_cache.clear()

    # To Do: support prefill length smaller than sink_length
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        #query_states = cache_kwargs.get("query_states", None) if cache_kwargs is not None else None

        if len(self.key_cache) < layer_idx:
            raise ValueError("QuantizedCache does not support model usage where layers are skipped. Use DynamicCache.")
        ################################################## Prefill Phase ##################################################
        elif len(self.key_cache) == layer_idx:
            # Initialize the key and value caches for the layer
            self.key_cache.append(key_states.detach().clone())
            self.value_cache.append(value_states.detach().clone())
            current_key_cache = self.key_cache[layer_idx]
            current_value_cache = self.value_cache[layer_idx]
            current_cache_length = current_key_cache.shape[-2]

            if self.PostQuant:
                keys_to_return = current_key_cache.detach().clone()
                values_to_return = current_value_cache.detach().clone()

            # Need to quantize the middle part of the key and value caches.
            # Short LongBench prompts can be no longer than the sink window; in
            # that case the cache should remain full precision until enough
            # tokens accumulate to fill the sink + quantization buffer.
            if current_cache_length > self.sink_length + self.buffer_length:
                start_idx = self.sink_length
                num_tokens = current_cache_length - self.sink_length
                num_token_to_buffer = num_tokens % self.buffer_length
                num_token_to_quantize = num_tokens - num_token_to_buffer
                end_idx = start_idx + num_token_to_quantize
                # Quantize Key Cache.
                if self.k_quant_mode == "per_token":
                    # Per-token K mirrors the KIVI-style V cache: keep the most
                    # recent buffer_length tokens fp16, quantize the rest with
                    # groups along head_dim (no transpose, no promote). k_codebook
                    # 'qlut' uses a single submean codebook here; 'kivi' uniform.
                    k_end_idx = start_idx + (num_tokens - self.buffer_length)
                    if k_end_idx > start_idx:
                        ks = current_key_cache[:, :, start_idx:k_end_idx, :]
                        ks = self._quant_k_pertoken(ks, layer_idx)
                        current_key_cache[:, :, start_idx:k_end_idx, :] = ks
                else:
                    # QLUT/KIVI per-channel path: bin channels by sigma^2 once over
                    # the full prompt quant region [sink, end_idx) before flushing buffers.
                    self._ensure_k_bins(
                        layer_idx,
                        current_key_cache[:, :, start_idx:end_idx, :].transpose(2, 3).contiguous())
                    for idx in range(start_idx, end_idx, self.buffer_length):
                        key_slice = current_key_cache[:, :, idx:idx+self.buffer_length, :].transpose(2, 3).contiguous()
                        key_slice = self._quant_k_buffer(key_slice, layer_idx).transpose(2, 3).contiguous()
                        current_key_cache[:, :, idx:idx+self.buffer_length, :] = key_slice
                # Quantize Value Cache
                if not self.VCache_BitDecoding:
                    num_token_to_quantize = num_tokens - self.buffer_length   # KIVI Style Value Cache
                    end_idx = start_idx + num_token_to_quantize
                value_slice = current_value_cache[:, :, start_idx:end_idx, :]
                value_slice = fake_quant_groupwise_lastdim(value_slice, self.group_size, self.vbits)
                current_value_cache[:, :, start_idx:end_idx, :] = value_slice
        ################################################## Decoding Phase ##################################################
        else:
            # update the key and value caches
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            current_key_cache = self.key_cache[layer_idx]
            current_value_cache = self.value_cache[layer_idx]
            current_cache_length = current_key_cache.shape[-2]

            if self.PostQuant:
                keys_to_return = current_key_cache.detach().clone()
                values_to_return = current_value_cache.detach().clone()

            # quantize
            num_tokens_kv_to_quantize = current_cache_length - self.sink_length - self.buffer_length
            if self.k_quant_mode == "per_token":
                # Per-token K follows the KIVI-style V schedule: quantize the
                # single token sliding out of the recent fp16 window each step.
                if num_tokens_kv_to_quantize > 0:
                    ks = current_key_cache[:, :, -self.buffer_length-1:-self.buffer_length, :]
                    ks = self._quant_k_pertoken(ks, layer_idx)
                    current_key_cache[:, :, -self.buffer_length-1:-self.buffer_length, :] = ks
            elif num_tokens_kv_to_quantize > 0 and (num_tokens_kv_to_quantize % self.buffer_length == 1):  # need to quantize
                # Quantize Key Cache
                key_slice = current_key_cache[:, :, -self.buffer_length-1:-1, :].transpose(2, 3).contiguous()
                key_slice = self._quant_k_buffer(key_slice, layer_idx).transpose(2, 3).contiguous()
                current_key_cache[:, :, -self.buffer_length-1:-1, :] = key_slice
                # Quantize Value Cache (BitDecoding)
                if self.VCache_BitDecoding:
                    value_slice = current_value_cache[:, :, -self.buffer_length-1:-1, :]
                    value_slice = fake_quant_groupwise_lastdim(value_slice, self.group_size, self.vbits)
                    current_value_cache[:, :, -self.buffer_length-1:-1, :] = value_slice
            # Quantize Value Cache (KIVI Style Value Cache, quantizing a Token each Decoding Step)
            if not self.VCache_BitDecoding:
                if num_tokens_kv_to_quantize > 0:
                    value_slice = current_value_cache[:, :, -self.buffer_length-1:-self.buffer_length, :]
                    value_slice = fake_quant_groupwise_lastdim(value_slice, self.group_size, self.vbits)
                    current_value_cache[:, :, -self.buffer_length-1:-self.buffer_length, :] = value_slice
        ####################################################################################################################
        if self.PostQuant:
            return keys_to_return, values_to_return
        else:
            return current_key_cache, current_value_cache

def get_kvcache_kitty(args: argparse.Namespace) -> KittyKVCache:
    """
    Get the Kitty KVCache object.
    Returns:
        KittyKVCache: The KittyKVCache object.
    """
    #
    cache_config = KittyKVCacheConfig(
        sink_length         = args.sink_length,
        buffer_length       = args.buffer_length,
        group_size          = args.group_size,
        kbits               = args.kbits,
        vbits               = args.vbits,
        promote_ratio       = args.promote_ratio,
        promote_bit         = args.promote_bit,
        promote_ratio_per_layer = getattr(args, "promote_ratio_per_layer", None),
        channel_selection   = args.channel_selection,
        k_quant_mode        = getattr(args, "k_quant_mode", "per_channel"),
        VCache_BitDecoding  = False,  # Using KIVI Style V Cache
        PostQuant           = True,  # Post Quantization is always enabled for Kitty KV Cache
        k_codebook          = getattr(args, "k_codebook", "kivi"),
        bin_codebooks       = getattr(args, "bin_codebooks", None),
        n_bins              = getattr(args, "n_bins", 6),
        pertoken_outlier_k  = getattr(args, "pertoken_outlier_k", 0),
        pertoken_outlier_bits = getattr(args, "pertoken_outlier_bits", 4),
        pertoken_pc_submean = getattr(args, "pertoken_pc_submean", False),
        pertoken_mixed      = getattr(args, "pertoken_mixed", False),
        pertoken_cb_mask    = getattr(args, "pertoken_cb_mask", None),
    )
    #
    return KittyKVCache(cache_config=cache_config)
