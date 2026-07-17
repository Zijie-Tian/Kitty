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
        k_quant_mode: str = "per_channel",        # "per_channel": KIVI-style token-axis groups (+promote); "per_token": K quantized like V along head_dim (no promote)
        VCache_BitDecoding: bool = False,         # The behavior of Value Cache, set to True means BitDecoding, otherwise KIVI Style Value Cache
        PostQuant: bool = True,                   # Post Quantization is always enabled
        k_codebook: str = "kivi",                 # "kivi" = existing min-max groupwise K quant; "qlut" = canonical qlutattn offline per-channel sign/nf2 mask (per-token); "q4_0" = llama.cpp Q4_0 (per-token, 32-ch blocks, symmetric d=max/-8; requires k_quant_mode="per_token")
        v_codebook: str = "kivi",                 # "kivi" = existing configured-group V quant; "q4_0" = llama.cpp Q4_0; "tile16_rescued" = rescued tile16cC 2-bit (qlutattn)
        bin_codebooks: Optional[list] = None,     # qlut only: per-channel codebook names, mask value -> codebook (canonical: ["sign", "nf2"])
        pertoken_cb_mask: Optional[str] = None,   # qlut only: path to the OFFLINE per-(layer,head,channel) codebook mask. Each channel uses bin_codebooks[mask[c]] FIXED (offline sigma^2 calibration -> sign/nf2); no online binning.
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
        self.pertoken_cb_mask = pertoken_cb_mask
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
        if self.k_codebook == "qlut":
            if not self.bin_codebooks:
                raise ValueError("k_codebook='qlut' requires a non-empty bin_codebooks list")
            if self.k_quant_mode != "per_token":
                raise ValueError(
                    "k_codebook='qlut' (canonical qlutattn) requires k_quant_mode='per_token'")
            if not self.pertoken_cb_mask:
                raise ValueError(
                    "k_codebook='qlut' requires pertoken_cb_mask (the offline "
                    "per-channel codebook mask from scripts/calibrate_qlutattn_mask.py)")
        if self.v_codebook not in ("kivi", "q4_0", "tile16_rescued"):
            raise ValueError(
                incorrect_arg_msg.format(
                    key="v_codebook",
                    correct_value="'kivi', 'q4_0' or 'tile16_rescued'",
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
        self.k_quant_mode = cache_config.k_quant_mode
        self.k_codebook = cache_config.k_codebook
        self.v_codebook = getattr(cache_config, "v_codebook", "kivi")
        self.bin_codebooks = cache_config.bin_codebooks
        # qlutattn: per-channel mean mu_d (cached at prefill, reused at decode)
        # for the per-channel-center + per-token codebook path. Free for
        # attention (q.mu is a per-query constant that cancels in softmax).
        self.k_pc_mean: dict[int, torch.Tensor] = {}
        # qlutattn: OFFLINE per-(layer,head,channel) codebook mask (sign/nf2),
        # loaded once and used unchanged -- NOT recomputed per prompt.
        # k_cb_mask[layer]=[nh,D].
        self.pertoken_cb_mask_path = getattr(cache_config, "pertoken_cb_mask", None)
        self.pertoken_offline = bool(self.pertoken_cb_mask_path)
        self.k_cb_mask: dict[int, torch.Tensor] = {}
        if self.pertoken_offline:
            _blob = torch.load(self.pertoken_cb_mask_path, map_location="cpu", weights_only=False)
            _m = _blob["codebook_mask"]                        # [nl, n_kv, D] uint8 (0=sign,1=nf2)
            for _li in range(_m.shape[0]):
                self.k_cb_mask[_li] = _m[_li].long()           # [n_kv, D]
            _cbs = _blob.get("codebooks")
            if _cbs:                                           # mask file carries its codebook names
                self.bin_codebooks = list(_cbs)
            print(f"[qlutattn-offline] loaded codebook mask {tuple(_m.shape)} from "
                  f"{self.pertoken_cb_mask_path} codebooks={self.bin_codebooks}")
        # per-token K settled-token pointer: prefill sets it, decode advances it,
        # so chunked/multi-token decode cannot leave an fp16 gap.
        self.k_pt_quant_end: dict[int, int] = {}
        # Rescued V tile16cC state (independent of K).
        self.v_tile_tokens = getattr(cache_config, "v_tile_tokens", None)
        self.v_tile_channels = getattr(cache_config, "v_tile_channels", None)
        self.v_tile_algo_version = getattr(cache_config, "v_tile_algo_version", None)
        self.v_rht_seed = getattr(cache_config, "v_rht_seed", None)
        self.v_mse_iters = getattr(cache_config, "v_mse_iters", None)
        self.v_tile_quant_end: dict[int, int] = {}
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

    @staticmethod
    def _masked_nf2sym_lastdim(r, mb):
        """Vectorized masked SYMMETRIC NF2 along head_dim (fixed IR-QLoRA LUT
        {-1,-c,+c,+1}, c=NF2_INNER), only over channels where mb is True (per
        [B,nh,T] row). The residual is already centered (per-channel mu_d on the
        offline path), so there is NO second mean here: one masked absmax scale
        s per group is the only side-info, then the closed-form snap
        sign(r)*s*(c or 1). r:[B,nh,T,D] float, mb:[1,nh,1,D] bool ->
        reconstruction [B,nh,T,D] (caller zeros non-bin channels via *mb).
        Non-bin channels never affect s (masked amax); all-False rows get s=0
        -> reconstruction 0, no inf/NaN (no division anywhere)."""
        from .qlut_quant import NF2_INNER, NF2_THRESH
        neg = (~mb).expand_as(r)
        s = r.abs().masked_fill(neg, float("-inf")).amax(-1, keepdim=True)  # [B,nh,T,1]
        s = torch.where(s.isinf(), torch.zeros_like(s), s)             # all-False row: a head with NO channel in this bin
        level = torch.where(r.abs() > NF2_THRESH * s, s, NF2_INNER * s)
        return torch.sign(r) * level

    @staticmethod
    def _pt_codebook_masked(r, m, cb):
        """Per-token codebook on r:[B,nh,T,D] over the channels flagged by m:[nh,D],
        VECTORIZED across heads. Returns a reconstruction that is ZERO outside the
        bin's channels so callers can sum across bins. The masked reductions use
        exactly the bin's channels, head by head. Canonical qlutattn codebooks:
        'sign' (1-bit, per-token masked-mean |r| scale) and 'nf2' (fixed
        symmetric-NF2 LUT, per-token masked absmax scale). 'int4' (4-bit
        symmetric absmax, per-token step s/7) is research-only and reachable
        exclusively through a QLUT_RESEARCH=1 mask."""
        mb = m[None, :, None, :]                                      # [1,nh,1,D] bool
        if cb == "sign":
            cnt = m.sum(-1).clamp(min=1)[None, :, None, None].to(r.dtype)  # [1,nh,1,1]
            mag = (r.abs() * mb).sum(-1, keepdim=True) / cnt          # [B,nh,T,1] masked mean
            return torch.sign(r) * mag * mb
        if cb == "nf2":
            return KittyKVCache._masked_nf2sym_lastdim(r, mb) * mb
        if cb == "int4":
            neg = (~mb).expand_as(r)
            s = r.abs().masked_fill(neg, float("-inf")).amax(-1, keepdim=True)  # [B,nh,T,1]
            s = torch.where(s.isinf(), torch.zeros_like(s), s)        # all-False row -> 0
            step = s / 7.0
            q = torch.round(r / step.clamp(min=torch.finfo(r.dtype).tiny)).clamp(-7.0, 7.0)
            return q * step * mb                                      # step==0 rows reconstruct 0
        raise ValueError(cb)

    def _quant_k_pertoken(self, ks, layer_idx=0):
        """Per-token K quant of a [B,nh,T,D] slice: one quantizer per token per
        head along head_dim (like the KIVI-style V cache).

        k_codebook='qlut' is the canonical qlutattn path: subtract the
        per-CHANNEL mean mu_d (self-calibrated once from the prompt at prefill,
        reused at decode -- free for attention since q.mu is a per-query
        constant that cancels in softmax), then quantize the residual per token
        with the OFFLINE per-channel codebook mask (self.k_cb_mask): mask 0 ->
        sign (per-token masked-mean |r| scale), mask 1 -> fixed symmetric NF2
        (symnf2-v1 LUT, per-token masked absmax scale, no second mean). The
        assignment is fixed by offline sigma^2 calibration, never recomputed
        per prompt. 'kivi' uses uniform min-max; 'q4_0' llama.cpp semantics."""
        if self.k_codebook == "q4_0":
            # llama.cpp Q4_0 row semantics: symmetric absmax (d=max/-8) per
            # 32-channel block along head_dim. No submean, no promote, no bins.
            return fake_quant_q4_0_lastdim(ks)
        if self.k_codebook != "qlut":
            return fake_quant_groupwise_lastdim(ks, self.group_size, self.kbits)
        B, nh, T, D = ks.shape
        if layer_idx not in self.k_pc_mean and T >= D:
            self.k_pc_mean[layer_idx] = ks[0].float().mean(dim=1)      # [nh,D] per-channel mean
        mu = self.k_pc_mean.get(layer_idx)
        muB = (mu[None, :, None, :] if mu is not None
               else ks.float().mean(dim=3, keepdim=True))              # short-prompt fallback
        r = ks.float() - muB
        cb_id = self.k_cb_mask.get(layer_idx)
        if cb_id is None:
            raise RuntimeError(
                f"qlutattn offline codebook mask has no entry for layer {layer_idx}; "
                "the mask shape must match the model (validated at preflight)"
            )
        if cb_id.device != ks.device:
            cb_id = cb_id.to(ks.device)
            self.k_cb_mask[layer_idx] = cb_id
        out = torch.zeros_like(r)
        for ci, cbk in enumerate(self.bin_codebooks):                  # ["sign", "nf2"]
            m = (cb_id == ci)                                          # [nh,D]
            if not m.any():
                continue
            out = out + self._pt_codebook_masked(r, m, cbk)
        return (muB + out).to(ks.dtype)

    def _quant_k_buffer(self, key_slice_t, layer_idx):
        """Quantize a [B,nh,D,buffer] post-RoPE K buffer with the KIVI-style
        min-max groupwise + promote path (per-channel K variants only; the
        canonical qlutattn K path is per-token and never reaches here)."""
        promote_mask = build_promote_mask(key_slice_t, self._layer_pr(layer_idx), self.channel_selection)
        return fake_quant_groupwise_lastdim(
            key_slice_t, self.group_size, self.kbits, promote_mask, self.promote_bit)

    def _quant_v_pertoken(self, value_slice):
        """KIVI-style per-token V quant along head_dim (configured group_size)."""
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
        self.v_pc_mean.clear()
        self.v_pc_rms.clear()
        self.v_error_bias.clear()
        self.k_pt_quant_end.clear()
        self.k_pc_mean.clear()
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
                    # 'qlut' = canonical qlutattn offline sign/nf2 mask; 'kivi' uniform.
                    k_end_idx = start_idx + (num_tokens - self.buffer_length)
                    if k_end_idx > start_idx:
                        ks = current_key_cache[:, :, start_idx:k_end_idx, :]
                        ks = self._quant_k_pertoken(ks, layer_idx)
                        current_key_cache[:, :, start_idx:k_end_idx, :] = ks
                        # remember the absolute token index we quantized up to,
                        # so the decode pointer resumes here.
                        self.k_pt_quant_end[layer_idx] = k_end_idx
                else:
                    # KIVI per-channel path: flush full buffers over the prompt.
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
                # Every settled token is independent. Use a pointer rather than
                # the historical one-token slice so chunked/multi-token decode
                # cannot leave an fp16 gap.
                ready_end = max(self.sink_length, current_cache_length - self.buffer_length)
                qend = self.k_pt_quant_end.get(layer_idx, self.sink_length)
                if ready_end > qend:
                    ks = current_key_cache[:, :, qend:ready_end, :]
                    ks = self._quant_k_pertoken(ks, layer_idx)
                    current_key_cache[:, :, qend:ready_end, :] = ks
                    self.k_pt_quant_end[layer_idx] = ready_end
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
        pertoken_cb_mask    = getattr(args, "pertoken_cb_mask", None),
        v_tile_tokens       = getattr(args, "v_tile_tokens", None),
        v_tile_channels     = getattr(args, "v_tile_channels", None),
        v_tile_algo_version = getattr(args, "v_tile_algo_version", None),
        v_rht_seed          = getattr(args, "v_rht_seed", None),
        v_mse_iters         = getattr(args, "v_mse_iters", None),
    )
    #
    return KittyKVCache(cache_config=cache_config)
