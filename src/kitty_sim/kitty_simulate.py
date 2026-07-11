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

from .utils_quant import build_promote_mask, fake_quant_groupwise_lastdim, fake_quant_q4_0_lastdim
from .v_tile_quant import (
    V_MSE_ITERS,
    V_RHT_SEED,
    V_TILE_ALGO_VERSION,
    V_TILE_TOKENS,
    calibrate_and_quantize_first_v_tile_block,
    calibrate_and_quantize_v_tile_prompt,
    fake_quant_v_pertoken2,
    quantize_v_tile_blocks_with_frozen_stats,
)


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
        k_codebook: str = "kivi",                 # "kivi" = existing min-max groupwise K quant; "qlut" = per-channel sigma^2-binned codebooks (qlutattn-k1v4); "q4_0" = llama.cpp Q4_0 (per-token, 32-ch blocks, symmetric d=max/-8; requires k_quant_mode="per_token")
        v_codebook: str = "kivi",                 # "kivi" = existing configured-group V quant; "per_token2" = named whole-head V2; "q4_0" = llama.cpp Q4_0; "tile16_rescued" = rescued tile16cC 2-bit
        bin_codebooks: Optional[list] = None,     # qlut only: list[str] mapping sigma^2-bin -> codebook (meanonly/sign/tern/uni2/nf2/uni3)
        n_bins: int = 6,                          # qlut only: number of per-layer sigma^2 quantile bins
        pertoken_outlier_k: int = 0,              # per_token only: keep top-k peak-|magnitude| channels (per head, fixed) out of the shared per-token scale (dense-and-sparse). 0 = off.
        pertoken_outlier_bits: int = 4,           # per_token only: precision of the kept outlier channels (per-channel along token; >=16 = fp16)
        pertoken_pc_submean: bool = False,        # per_token only: subtract a per-CHANNEL mean (cached at prefill, free for attention) then pure binary/ternary on the residual (qlutattn-k125v4-pt). NOT the per-token submean.
        pertoken_mixed: bool = False,             # per_token only: per-channel submean + sigma^2-binned MIXED codebook (ONLINE sigma^2, legacy k1.68v4-pt path); bin_codebooks = per-bin policy (low sigma^2 -> bin 0).
        pertoken_cb_mask: Optional[str] = None,   # per_token only: path to an OFFLINE per-(layer,head,channel) codebook mask (the corrected k168v4-pt). When set, each channel uses bin_codebooks[mask[c]] FIXED (offline sigma^2 calibration -> sign/tern), no online sigma^2 binning, no nf2.
        pertoken_rotate: bool = False,            # per_token only: Hadamard-rotate (FWHT) the per-channel-centered residual before per-token quant, de-rotate after (FWHT self-inverse). Spreads K outliers -> isotropic -> per-token sign/tern fits one scale. Free for attention (orthogonal; q.mu cancels). qlutattn-rotated-k*v4-pt.
        pertoken_block: int = 1,                  # per_token only: # of consecutive tokens that SHARE one per-token codebook (Lloyd levels / sign-mag). 1 = current per-token (each token its own); >1 = block-shared (side-info amortized block x). Quant axis is unchanged (still head_dim).
        v_tile_tokens: Optional[int] = None,      # tile16_rescued only: fixed 16
        v_tile_channels: Optional[int] = None,    # tile16_rescued only: channel block C
        v_tile_algo_version: Optional[str] = None,# tile16_rescued only: "rht-pcaff-mse1-bias-v1"
        v_rht_seed: Optional[int] = None,         # tile16_rescued only: fixed 20260711
        v_mse_iters: Optional[int] = None,        # tile16_rescued only: fixed 1
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
        self.v_codebook = v_codebook
        self.bin_codebooks = list(bin_codebooks) if bin_codebooks is not None else None
        self.n_bins = n_bins
        self.pertoken_outlier_k = pertoken_outlier_k
        self.pertoken_outlier_bits = pertoken_outlier_bits
        self.pertoken_pc_submean = pertoken_pc_submean
        self.pertoken_mixed = pertoken_mixed
        self.pertoken_cb_mask = pertoken_cb_mask
        self.pertoken_rotate = pertoken_rotate
        self.pertoken_block = int(pertoken_block)
        self.v_tile_tokens = v_tile_tokens
        self.v_tile_channels = v_tile_channels
        self.v_tile_algo_version = v_tile_algo_version
        self.v_rht_seed = v_rht_seed
        self.v_mse_iters = v_mse_iters
        #
        self.validate()

    def validate(self):
        """Validates if the arguments passed are correct"""
        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        if self.k_codebook not in ("kivi", "qlut", "q4_0"):
            raise ValueError(
                incorrect_arg_msg.format(key="k_codebook", correct_value="'kivi', 'qlut' or 'q4_0'",
                                         found_value=self.k_codebook))
        if self.k_codebook == "qlut" and not self.bin_codebooks:
            raise ValueError("k_codebook='qlut' requires a non-empty bin_codebooks list")
        if self.v_codebook not in ("kivi", "per_token2", "q4_0", "tile16_rescued"):
            raise ValueError(
                incorrect_arg_msg.format(
                    key="v_codebook",
                    correct_value="'kivi', 'per_token2', 'q4_0' or 'tile16_rescued'",
                    found_value=self.v_codebook,
                ))
        if self.v_codebook == "tile16_rescued":
            if self.v_tile_tokens != V_TILE_TOKENS:
                raise ValueError(
                    f"v_codebook='tile16_rescued' requires v_tile_tokens={V_TILE_TOKENS}, "
                    f"got {self.v_tile_tokens}"
                )
            if self.v_tile_channels is None or int(self.v_tile_channels) <= 0:
                raise ValueError(
                    "v_codebook='tile16_rescued' requires a positive v_tile_channels (C)"
                )
            if self.v_rht_seed != V_RHT_SEED:
                raise ValueError(
                    "v_codebook='tile16_rescued' requires fixed "
                    f"v_rht_seed={V_RHT_SEED} for rv1, got {self.v_rht_seed}"
                )
            if self.v_mse_iters != V_MSE_ITERS:
                raise ValueError(
                    f"v_codebook='tile16_rescued' requires v_mse_iters={V_MSE_ITERS}, "
                    f"got {self.v_mse_iters}"
                )
            if self.v_tile_algo_version != V_TILE_ALGO_VERSION:
                raise ValueError(
                    f"unsupported v_tile_algo_version={self.v_tile_algo_version!r}; "
                    f"expected {V_TILE_ALGO_VERSION!r}"
                )
            if self.vbits != 2:
                raise ValueError(
                    f"v_codebook='tile16_rescued' requires vbits=2, got {self.vbits}"
                )
        elif self.v_codebook == "per_token2":
            if self.vbits != 2:
                raise ValueError(
                    f"v_codebook='per_token2' requires vbits=2, got {self.vbits}"
                )
            tile_only = (
                self.v_tile_tokens,
                self.v_tile_channels,
                self.v_tile_algo_version,
                self.v_rht_seed,
                self.v_mse_iters,
            )
            if any(v is not None for v in tile_only):
                raise ValueError(
                    "v_codebook='per_token2' must not carry tile-only V fields"
                )
        else:
            tile_only = (
                self.v_tile_tokens,
                self.v_tile_channels,
                self.v_tile_algo_version,
                self.v_rht_seed,
                self.v_mse_iters,
            )
            if any(v is not None for v in tile_only):
                raise ValueError(
                    "tile-only V fields (v_tile_tokens/channels/algo_version/"
                    "v_rht_seed/v_mse_iters) must be None unless "
                    "v_codebook='tile16_rescued'"
                )
        if self.k_quant_mode not in ("per_channel", "per_token"):
            raise ValueError(
                incorrect_arg_msg.format(key="k_quant_mode", correct_value="'per_channel' or 'per_token'",
                                         found_value=self.k_quant_mode))
        if self.k_codebook == "q4_0" and self.k_quant_mode != "per_token":
            raise ValueError(
                "k_codebook='q4_0' quantizes along head_dim (llama.cpp row semantics) and "
                "requires k_quant_mode='per_token'")
        if self.k_quant_mode == "per_token" and (self.promote_ratio != 0.0 or self.promote_ratio_per_layer is not None):
            raise ValueError(
                "k_quant_mode='per_token' requires promote_ratio=0.0 and no per-layer override "
                "(per-token K has no channel axis at quantization time)")
        if self.pertoken_block < 1:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="pertoken_block",
                    correct_value="an integer >= 1 (1 = per-token; >1 = block-shared codebook)",
                    found_value=self.pertoken_block,
                ),
            )
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
        if self.buffer_length == 0:
            # buffer_length=0 = llama.cpp-style "quantize on write": no fp16 recent
            # window at all. Only the per-token K path supports it (the per-channel
            # decode schedule needs `% buffer_length`), so reject per_channel here
            # and skip the group/buffer coupling check (group_size is a head_dim
            # block size in this regime, not a token-axis group).
            if self.k_quant_mode != "per_token":
                raise ValueError(
                    incorrect_arg_msg.format(
                        key="buffer_length",
                        correct_value="> 0 for k_quant_mode='per_channel' (0 is only supported per_token)",
                        found_value=self.buffer_length,
                    ),
                )
        elif self.group_size > self.buffer_length or self.buffer_length % self.group_size != 0:
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
        self.v_codebook = getattr(cache_config, "v_codebook", "kivi")
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
        self.pertoken_rotate = getattr(cache_config, "pertoken_rotate", False)
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
        # per_token block-shared codebook: pertoken_block consecutive tokens share one
        # per-token codebook. k_pt_quant_end[layer] = absolute token index quantized so
        # far (strict block-aligned decode: prefill sets it, decode advances it per block).
        self.pertoken_block = int(getattr(cache_config, "pertoken_block", 1))
        self.k_pt_quant_end: dict[int, int] = {}
        # Rescued V tile16cC state (independent of K).
        self.v_tile_tokens = getattr(cache_config, "v_tile_tokens", None)
        self.v_tile_channels = getattr(cache_config, "v_tile_channels", None)
        self.v_tile_algo_version = getattr(cache_config, "v_tile_algo_version", None)
        self.v_rht_seed = getattr(cache_config, "v_rht_seed", None)
        self.v_mse_iters = getattr(cache_config, "v_mse_iters", None)
        self.v_tile_quant_end: dict[int, int] = {}
        # Named whole-head V PT2 has its own settled-token pointer so a
        # multi-token append quantizes every token that left the recent window.
        # The legacy kivi V path deliberately keeps its historical schedule.
        self.v_pt_quant_end: dict[int, int] = {}
        self.v_pc_mean: dict[int, torch.Tensor] = {}
        self.v_pc_rms: dict[int, torch.Tensor] = {}
        self.v_error_bias: dict[int, torch.Tensor] = {}
        # Engagement counters (read-only observability for LongBench guards).
        self.v_quant_calls = 0
        self.v_quantized_tokens = 0
        self.v_tile_blocks = 0
        self.last_v_quant_mode: Optional[str] = None
        self.last_v_tile_channels: Optional[int] = None
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

    def _pt_codebook_blocked(self, r, m, cb):
        """Block-shared wrapper around _pt_codebook_masked. r:[B,nh,T,D], m:[nh,D].
        pertoken_block<=1 (or T<=1) -> call the core unchanged (bit-identical to the
        per-token path). block>1 -> flatten each block of `block` consecutive tokens'
        (block,D) into the last dim so the core's last-dim reduce / Lloyd is SHARED
        across the block (one codebook per block); the trailing T%block tokens form one
        smaller block. Works for sign/tern/nf2/uni alike because the core only reduces
        and masks along the last axis. Any FWHT rotation is applied by the caller on
        [B,nh,T,D] before this and undone after; this wrapper restores [B,nh,T,D] so it
        stays transparent to rotate/de-rotate."""
        blk = self.pertoken_block
        if blk <= 1 or r.shape[2] <= 1:
            return KittyKVCache._pt_codebook_masked(r, m, cb)
        B, nh, T, D = r.shape
        nb, rem = T // blk, T % blk
        out = torch.empty_like(r)

        def _run(rt, blk_t):                                    # rt:[B,nh,nB,blk_t,D]
            nB = rt.shape[2]
            rf = rt.reshape(B, nh, nB, blk_t * D)
            mf = m[:, None, :].expand(nh, blk_t, D).reshape(nh, blk_t * D)
            rec = KittyKVCache._pt_codebook_masked(rf, mf, cb)  # [B,nh,nB,blk_t*D]
            return rec.reshape(B, nh, nB, blk_t, D)

        if nb > 0:
            main = _run(r[:, :, :nb * blk, :].reshape(B, nh, nb, blk, D), blk)
            out[:, :, :nb * blk, :] = main.reshape(B, nh, nb * blk, D)
        if rem > 0:
            tail = _run(r[:, :, nb * blk:, :].reshape(B, nh, 1, rem, D), rem)
            out[:, :, nb * blk:, :] = tail.reshape(B, nh, rem, D)
        return out

    @staticmethod
    def _fwht_lastdim(x):
        """Normalized fast Walsh-Hadamard transform along the last axis (head_dim,
        a power of 2: 64 on Llama-3.2-1B, 128 on 3B). SELF-INVERSE so de-rotation
        is the same call: fwht(fwht(x)) == x. O(d log d), vectorized (no dense
        d x d matmul). Used by qlutattn-rotated-*: rotate the per-channel-centered
        residual into an isotropic basis before per-token quant, then de-rotate
        the dequantized result so the stored key is in the original basis and the
        rest of attention (q . k_hat) is unchanged -- equivalent to rotating q."""
        n = x.shape[-1]
        if n & (n - 1) != 0:
            raise ValueError(f"FWHT needs head_dim a power of 2, got {n}")
        lead = x.shape[:-1]
        y = x
        h = 1
        while h < n:
            y = y.reshape(*lead, n // (2 * h), 2, h)
            a0 = y[..., 0, :]
            a1 = y[..., 1, :]
            y = torch.stack((a0 + a1, a0 - a1), dim=-2).reshape(*lead, n)
            h *= 2
        return y / (n ** 0.5)

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
        if self.k_codebook == "q4_0":
            # llama.cpp Q4_0 row semantics: symmetric absmax (d=max/-8) per
            # 32-channel block along head_dim. No submean, no promote, no bins.
            return fake_quant_q4_0_lastdim(ks)
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
            if self.pertoken_rotate:                                       # rotate into isotropic basis (mask is sigma^2-calibrated in THIS basis)
                r = self._fwht_lastdim(r)
            cb_id = self.k_cb_mask.get(layer_idx)
            if cb_id is None:                                             # layer absent from mask: single codebook
                full = torch.ones(nh, D, dtype=torch.bool, device=ks.device)
                out = self._pt_codebook_blocked(r, full, self.bin_codebooks[0])
            else:
                if cb_id.device != ks.device:
                    cb_id = cb_id.to(ks.device)
                    self.k_cb_mask[layer_idx] = cb_id
                out = torch.zeros_like(r)
                for ci, cbk in enumerate(self.bin_codebooks):             # e.g. ["sign","tern"] / ["sign","nf2"]
                    m = (cb_id == ci)                                     # [nh,D]
                    if not m.any():
                        continue
                    out = out + self._pt_codebook_blocked(r, m, cbk)
            if self.pertoken_rotate:                                       # de-rotate (FWHT self-inverse) back to original basis
                out = self._fwht_lastdim(out)
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
                    out = out + self._pt_codebook_blocked(r, m, policy[bi])
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
            if self.pertoken_rotate:                                       # Hadamard-rotate residual into isotropic basis
                r = self._fwht_lastdim(r)
            mag = r.abs().mean(dim=3, keepdim=True)                        # [B,nh,T,1] per-token scale
            if cb == "tern":
                mask = r.abs() > 0.5 * mag
                mag2 = (r.abs() * mask).sum(3, keepdim=True) / mask.sum(3, keepdim=True).clamp(min=1)
                q = torch.sign(r) * mag2 * mask
            else:                                                          # "sign": pure 1-bit binary
                q = torch.sign(r) * mag
            if self.pertoken_rotate:                                       # de-rotate (FWHT self-inverse) back to original basis
                q = self._fwht_lastdim(q)
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

    def _quant_v_pertoken(self, value_slice):
        """KIVI-style per-token V quant along head_dim.

        Only the explicit ``per_token2`` codebook locks whole-head grouping.
        Legacy ``kivi`` configurations retain their configured group_size even
        when vbits=2; this avoids silently changing existing Kitty/KIVI runs.
        """
        if self.v_codebook == "per_token2":
            out = fake_quant_v_pertoken2(value_slice)
            self.v_quant_calls += 1
            self.v_quantized_tokens += int(value_slice.shape[-2])
            self.last_v_quant_mode = "per_token2"
            return out
        return fake_quant_groupwise_lastdim(value_slice, self.group_size, self.vbits)

    def _ensure_v_tile_config(self, value_slice):
        D = value_slice.shape[-1]
        C = int(self.v_tile_channels)
        if D % C != 0:
            raise ValueError(
                f"head_dim={D} must be divisible by v_tile_channels={C}"
            )
        # Retain the debug assertion requested by the tile contract without
        # relying on it for production validation under ``python -O``.
        assert D % C == 0, (
            f"head_dim={D} must be divisible by v_tile_channels={C}"
        )
        if D & (D - 1):
            raise ValueError(
                f"Hadamard/RHT requires power-of-two head_dim, got head_dim={D}"
            )
        if value_slice.dtype != torch.float16:
            raise ValueError(
                f"tile16_rescued requires FP16 V tensors, got dtype={value_slice.dtype}"
            )

    def _v_tile_stats_ready(self, layer_idx: int) -> bool:
        return (
            layer_idx in self.v_pc_mean
            and layer_idx in self.v_pc_rms
            and layer_idx in self.v_error_bias
        )

    def _flush_v_tile_blocks(
        self,
        layer_idx: int,
        current_value_cache: torch.Tensor,
        *,
        calibrate_full_prompt: bool = False,
    ):
        """Flush complete 16-token V tiles up to the settled frontier.

        Initial prefill calibrates rv1's prompt statistics over the *entire*
        aligned settled region and quantizes it in one vectorized call.  A short
        prefill has no complete tile; when decode later releases the first tile,
        lazy initialization necessarily calibrates on that single block.  Once
        initialized, all later blocks reuse the frozen prompt statistics.
        """
        self._ensure_v_tile_config(current_value_cache)
        L = current_value_cache.shape[-2]
        sink = self.sink_length
        recent = self.buffer_length
        N = V_TILE_TOKENS
        ready_end = max(sink, L - recent)
        qend = self.v_tile_quant_end.get(layer_idx, sink)
        C = int(self.v_tile_channels)
        seed = int(self.v_rht_seed if self.v_rht_seed is not None else V_RHT_SEED)

        if not self._v_tile_stats_ready(layer_idx):
            if ready_end - qend < N:
                return
            if calibrate_full_prompt:
                aligned_end = qend + ((ready_end - qend) // N) * N
                region = current_value_cache[:, :, qend:aligned_end, :]
                recon, mu, rms, bias = calibrate_and_quantize_v_tile_prompt(
                    region,
                    v_tile_channels=C,
                    v_rht_seed=seed,
                )
                initialized_tokens = aligned_end - qend
            else:
                aligned_end = qend + N
                block = current_value_cache[:, :, qend:aligned_end, :]
                recon, mu, rms, bias = calibrate_and_quantize_first_v_tile_block(
                    block, v_tile_channels=C, v_rht_seed=seed,
                )
                initialized_tokens = N
            current_value_cache[:, :, qend:aligned_end, :] = recon
            self.v_pc_mean[layer_idx] = mu
            self.v_pc_rms[layer_idx] = rms
            self.v_error_bias[layer_idx] = bias
            qend = aligned_end
            self.v_tile_quant_end[layer_idx] = qend
            self.v_quant_calls += 1
            self.v_quantized_tokens += initialized_tokens
            self.v_tile_blocks += initialized_tokens // N
            self.last_v_quant_mode = "tile16_rescued"
            self.last_v_tile_channels = C

        # Decode normally exposes one tile at a time, while a multi-token append
        # may expose several. Quantize every complete block in one vectorized call.
        aligned_end = qend + ((ready_end - qend) // N) * N
        if aligned_end > qend:
            blocks = current_value_cache[:, :, qend:aligned_end, :]
            recon = quantize_v_tile_blocks_with_frozen_stats(
                blocks,
                mu=self.v_pc_mean[layer_idx],
                rms=self.v_pc_rms[layer_idx],
                bias=self.v_error_bias[layer_idx],
                v_tile_channels=C,
                v_rht_seed=seed,
            )
            current_value_cache[:, :, qend:aligned_end, :] = recon
            flushed_tokens = aligned_end - qend
            qend = aligned_end
            self.v_tile_quant_end[layer_idx] = qend
            self.v_quant_calls += 1
            self.v_quantized_tokens += flushed_tokens
            self.v_tile_blocks += flushed_tokens // N
            self.last_v_quant_mode = "tile16_rescued"
            self.last_v_tile_channels = C

    def _quant_v(self, layer_idx, value_slice):
        """Quantize a [B,nh,T,D] V slice. Dispatcher for per-token codebooks."""
        if self.v_codebook == "q4_0":
            return fake_quant_q4_0_lastdim(value_slice)
        if self.v_codebook == "tile16_rescued":
            raise RuntimeError(
                "_quant_v must not be called for tile16_rescued; use "
                "_flush_v_tile_blocks for the strict 16-token schedule"
            )
        return self._quant_v_pertoken(value_slice)

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

    def _index_select_v_tile_stats(self, layer_idx: int, index: torch.Tensor):
        if layer_idx in self.v_pc_mean:
            self.v_pc_mean[layer_idx] = self.v_pc_mean[layer_idx].index_select(0, index)
        if layer_idx in self.v_pc_rms:
            self.v_pc_rms[layer_idx] = self.v_pc_rms[layer_idx].index_select(0, index)
        if layer_idx in self.v_error_bias:
            self.v_error_bias[layer_idx] = self.v_error_bias[layer_idx].index_select(0, index)

    def _repeat_v_tile_stats(self, layer_idx: int, repeats: int):
        if layer_idx in self.v_pc_mean:
            self.v_pc_mean[layer_idx] = self.v_pc_mean[layer_idx].repeat_interleave(repeats, dim=0)
        if layer_idx in self.v_pc_rms:
            self.v_pc_rms[layer_idx] = self.v_pc_rms[layer_idx].repeat_interleave(repeats, dim=0)
        if layer_idx in self.v_error_bias:
            self.v_error_bias[layer_idx] = self.v_error_bias[layer_idx].repeat_interleave(repeats, dim=0)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache batch dimension for generation helpers."""
        for layer_idx in range(len(self.key_cache)):
            device_index = beam_idx.to(self.key_cache[layer_idx].device)
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, device_index)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, device_index)
            self._index_select_v_tile_stats(layer_idx, device_index)

    def crop(self, max_length: int):
        """Crop atomically while keeping K/V settled-token pointers coherent.

        A rescued V tile cannot be cut in the middle: the retained values were
        reconstructed with side parameters shared with the removed values.  All
        layers are therefore validated before any tensor or state is mutated.
        Per-token K/V groups are independent, so their pointers may safely move
        back to the retained length.
        """
        if max_length < 0:
            max_length = self.get_seq_length() + max_length
        new_len = max(0, int(max_length))

        # Validation pass.  Do not mutate any layer if one tile crop is illegal.
        if self.v_codebook == "tile16_rescued":
            for layer_idx in range(len(self.key_cache)):
                old_len = self.key_cache[layer_idx].shape[-2]
                effective_len = min(new_len, old_len)
                sink = self.sink_length
                qend = self.v_tile_quant_end.get(layer_idx, sink)
                if (
                    sink < effective_len < qend
                    and (effective_len - sink) % V_TILE_TOKENS != 0
                ):
                    raise ValueError(
                        f"crop into mid-tile V quantized region is unsupported "
                        f"(layer={layer_idx}, new_len={effective_len}, sink={sink}, "
                        f"v_tile_quant_end={qend}, tile_tokens={V_TILE_TOKENS})"
                    )

        # Mutation pass after every layer has validated.
        for layer_idx in range(len(self.key_cache)):
            old_len = self.key_cache[layer_idx].shape[-2]
            effective_len = min(new_len, old_len)
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :effective_len, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :effective_len, :]
            sink = self.sink_length

            if layer_idx in self.k_pt_quant_end:
                self.k_pt_quant_end[layer_idx] = max(
                    sink, min(self.k_pt_quant_end[layer_idx], effective_len)
                )
            if layer_idx in self.v_pt_quant_end:
                self.v_pt_quant_end[layer_idx] = max(
                    sink, min(self.v_pt_quant_end[layer_idx], effective_len)
                )

            if self.v_codebook == "tile16_rescued":
                qend = self.v_tile_quant_end.get(layer_idx, sink)
                if effective_len <= sink:
                    self.v_tile_quant_end.pop(layer_idx, None)
                    self.v_pc_mean.pop(layer_idx, None)
                    self.v_pc_rms.pop(layer_idx, None)
                    self.v_error_bias.pop(layer_idx, None)
                elif effective_len < qend:
                    # The validation pass proved this is a complete tile boundary.
                    self.v_tile_quant_end[layer_idx] = effective_len

    def batch_repeat_interleave(self, repeats: int):
        """Repeat batch entries for generation helpers."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self._repeat_v_tile_stats(layer_idx, repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Select batch entries for generation helpers."""
        for layer_idx in range(len(self.key_cache)):
            device_indices = indices.to(self.key_cache[layer_idx].device)
            self.key_cache[layer_idx] = self.key_cache[layer_idx][device_indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][device_indices, ...]
            self._index_select_v_tile_stats(layer_idx, device_indices)

    def reset(self):
        """Clear all cached tensors and V tile state."""
        self.key_cache.clear()
        self.value_cache.clear()
        self.v_tile_quant_end.clear()
        self.v_pt_quant_end.clear()
        self.v_pc_mean.clear()
        self.v_pc_rms.clear()
        self.v_error_bias.clear()
        self.k_pt_quant_end.clear()
        self.k_bin_ids.clear()
        self.k_outlier_ids.clear()
        self.k_pc_mean.clear()
        self.k_mix_bins.clear()
        self.v_quant_calls = 0
        self.v_quantized_tokens = 0
        self.v_tile_blocks = 0
        self.last_v_quant_mode = None
        self.last_v_tile_channels = None

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
                # buffer_length=0 = llama.cpp-style quantize-on-write (no recent
                # fp16 window): everything past the sink is quantized right away.
                num_token_to_buffer = num_tokens % self.buffer_length if self.buffer_length > 0 else 0
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
                        # block-shared codebook: remember the absolute token index we
                        # quantized up to, so strict block-aligned decode resumes here.
                        self.k_pt_quant_end[layer_idx] = k_end_idx
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
                if self.v_codebook == "tile16_rescued":
                    self._flush_v_tile_blocks(
                        layer_idx,
                        current_value_cache,
                        calibrate_full_prompt=True,
                    )
                else:
                    if not self.VCache_BitDecoding:
                        num_token_to_quantize = num_tokens - self.buffer_length   # KIVI Style Value Cache
                        end_idx = start_idx + num_token_to_quantize
                    value_slice = current_value_cache[:, :, start_idx:end_idx, :]
                    value_slice = self._quant_v(layer_idx, value_slice)
                    current_value_cache[:, :, start_idx:end_idx, :] = value_slice
                    if self.v_codebook == "per_token2":
                        self.v_pt_quant_end[layer_idx] = end_idx
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
            # The single token to quantize this step: the one sliding out of the
            # recent fp16 window. With buffer_length=0 (llama.cpp-style quantize-
            # on-write) the window is empty and the NEWEST token is quantized --
            # the [-1:0] slice would otherwise be empty and silently skip it.
            quant_slice = (slice(-1, None) if self.buffer_length == 0
                           else slice(-self.buffer_length - 1, -self.buffer_length))
            if self.k_quant_mode == "per_token":
                blk = self.pertoken_block
                if blk <= 1:
                    # Every settled token is independent in the block=1 path.
                    # Use a pointer rather than the historical one-token slice so
                    # chunked/multi-token decode cannot leave an fp16 gap.
                    ready_end = max(self.sink_length, current_cache_length - self.buffer_length)
                    qend = self.k_pt_quant_end.get(layer_idx, self.sink_length)
                    if ready_end > qend:
                        ks = current_key_cache[:, :, qend:ready_end, :]
                        ks = self._quant_k_pertoken(ks, layer_idx)
                        current_key_cache[:, :, qend:ready_end, :] = ks
                        self.k_pt_quant_end[layer_idx] = ready_end
                else:
                    # Strict block-aligned per-token K: accumulate the tokens sliding out
                    # of the recent fp16 window and quantize a full `blk`-token block once
                    # `blk` of them are pending (one shared codebook per block, matching
                    # prefill). The <blk trailing tokens stay fp16 until the next block
                    # fills (pending); generation-end leaves at most blk-1 tokens fp16.
                    qend = self.k_pt_quant_end.get(layer_idx, self.sink_length)
                    ready_end = max(self.sink_length, current_cache_length - self.buffer_length)
                    aligned_end = qend + ((ready_end - qend) // blk) * blk
                    if aligned_end > qend:
                        ks = current_key_cache[:, :, qend:aligned_end, :]
                        ks = self._quant_k_pertoken(ks, layer_idx)
                        current_key_cache[:, :, qend:aligned_end, :] = ks
                        self.k_pt_quant_end[layer_idx] = aligned_end
            elif num_tokens_kv_to_quantize > 0 and (num_tokens_kv_to_quantize % self.buffer_length == 1):  # need to quantize
                # Quantize Key Cache
                key_slice = current_key_cache[:, :, -self.buffer_length-1:-1, :].transpose(2, 3).contiguous()
                key_slice = self._quant_k_buffer(key_slice, layer_idx).transpose(2, 3).contiguous()
                current_key_cache[:, :, -self.buffer_length-1:-1, :] = key_slice
                # Quantize Value Cache (BitDecoding)
                if self.VCache_BitDecoding and self.v_codebook != "tile16_rescued":
                    value_slice = current_value_cache[:, :, -self.buffer_length-1:-1, :]
                    value_slice = self._quant_v(layer_idx, value_slice)
                    current_value_cache[:, :, -self.buffer_length-1:-1, :] = value_slice
            # Quantize Value Cache
            if self.v_codebook == "tile16_rescued":
                self._flush_v_tile_blocks(layer_idx, current_value_cache)
            elif self.v_codebook == "per_token2":
                ready_end = max(self.sink_length, current_cache_length - self.buffer_length)
                qend = self.v_pt_quant_end.get(layer_idx, self.sink_length)
                if ready_end > qend:
                    value_slice = current_value_cache[:, :, qend:ready_end, :]
                    value_slice = self._quant_v(layer_idx, value_slice)
                    current_value_cache[:, :, qend:ready_end, :] = value_slice
                    self.v_pt_quant_end[layer_idx] = ready_end
            elif not self.VCache_BitDecoding:
                if num_tokens_kv_to_quantize > 0:
                    value_slice = current_value_cache[:, :, quant_slice, :]
                    value_slice = self._quant_v(layer_idx, value_slice)
                    current_value_cache[:, :, quant_slice, :] = value_slice
        ####################################################################################################################
        if self.PostQuant:
            return keys_to_return, values_to_return
        else:
            return current_key_cache, current_value_cache

def get_kvcache_kitty(args: argparse.Namespace) -> KittyKVCache:
    """
    Get the Kitty KVCache object.
    Returns:
        KittyKVCache: The Kitty KVCache object.
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
        v_codebook          = getattr(args, "v_codebook", "kivi"),
        bin_codebooks       = getattr(args, "bin_codebooks", None),
        n_bins              = getattr(args, "n_bins", 6),
        pertoken_outlier_k  = getattr(args, "pertoken_outlier_k", 0),
        pertoken_outlier_bits = getattr(args, "pertoken_outlier_bits", 4),
        pertoken_pc_submean = getattr(args, "pertoken_pc_submean", False),
        pertoken_mixed      = getattr(args, "pertoken_mixed", False),
        pertoken_cb_mask    = getattr(args, "pertoken_cb_mask", None),
        pertoken_rotate     = getattr(args, "pertoken_rotate", False),
        pertoken_block      = getattr(args, "pertoken_block", 1),
        v_tile_tokens       = getattr(args, "v_tile_tokens", None),
        v_tile_channels     = getattr(args, "v_tile_channels", None),
        v_tile_algo_version = getattr(args, "v_tile_algo_version", None),
        v_rht_seed          = getattr(args, "v_rht_seed", None),
        v_mse_iters         = getattr(args, "v_mse_iters", None),
    )
    #
    return KittyKVCache(cache_config=cache_config)
