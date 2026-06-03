"""Make GLM-family models behave like a modern HF model in the kitty_sim path:

1. **Apply Kitty fake-quant** (GLM's legacy remote code never calls Cache.update(),
   so a passed KittyKVCache is a silent no-op -> dense fp16). Verified: update() is
   called 0x for GLM vs 16x for Llama-3.2-1B.
2. **Avoid the huge caching-allocator fragmentation** GLM produces at long context.
   GLM's ``GLMTransformer.forward`` builds the prefill KV cache with a repeated
   ``torch.cat(presents, kv_cache, dim=0)`` (stacking along the layer axis) plus a
   reshape in ``ChatGLMModel.forward`` — an O(n_layers**2) churn of growing buffers
   that bakes ~13 GiB of reserved-but-freed headroom into the allocator (vs ~1.7 GiB
   for Llama-3.1-8B). Llama never does this: its per-layer cache is appended, not
   re-concatenated across layers.
3. **Let generate() run on transformers>=4.57** (GLM's remote
   ``_update_model_kwargs_for_generation`` calls the removed
   ``_extract_past_from_model_output``).

The fixes live entirely in the repo (no edits to the vendored model file):

* ``SelfAttention.forward`` is wrapped so the KV GLM caches is fake-quantized with
  the *same* KittyKVCache logic used for HF-Cache models (per-layer, incremental,
  PostQuant), and is always returned in the per-layer ``(key, value)`` tuple format.
* ``GLMTransformer.forward`` is replaced with a version that appends per-layer
  tuples instead of ``torch.cat``-stacking them — this is the de-fragmentation fix
  and matches how Llama/Qwen accumulate KV.
* ``_extract_past_from_model_output`` is bound onto the model so generate() works.
"""

from __future__ import annotations

from types import MethodType

import torch

from kitty_sim.kitty_simulate import KittyKVCache, KittyKVCacheConfig
from kitty_sim.kv_offload import pinned_pool


def is_glm_family(model_family: str | None) -> bool:
    """Whether a model_family slug refers to a GLM / ChatGLM model."""
    fam = (model_family or "").lower()
    return "glm" in fam or "chatglm" in fam


def cache_config_from_variant(variant) -> KittyKVCacheConfig:
    """Build the same KittyKVCacheConfig that ``_cache_factory`` would produce."""
    return KittyKVCacheConfig(
        sink_length=variant.sink_length,
        buffer_length=variant.buffer_length,
        group_size=variant.group_size,
        kbits=variant.kbits,
        vbits=variant.vbits,
        promote_ratio=variant.promote_ratio,
        promote_bit=variant.promote_bit,
        channel_selection=variant.channel_selection,
    )


def _split_new_kv(new_kv):
    """Return ``(key, value, is_prefill)`` from GLM SelfAttention's cache return.

    GLM uses two formats (modeling_chatglm.py:498-503):
      * prefill (input ``kv_cache`` is None): a stacked tensor
        ``[1, 2, b, n_kv, seq, hn]``
      * decode  (input ``kv_cache`` given):   a tuple ``(key_layer, value_layer)``
    """
    if isinstance(new_kv, tuple):
        return new_kv[0], new_kv[1], False
    return new_kv[0, 0], new_kv[0, 1], True


# --------------------------------------------------------------------------- #
# Layer-wise CPU KV offload for GLM (the de-fragmented per-layer-tuple cache).
#
# GLM does not thread KV through a shared HF Cache; each SelfAttention owns a
# private single-layer KittyKVCache, and the cross-layer cumulative KV lives in
# the ``presents`` tuple the patched GLMTransformer builds (fed back next step as
# ``kv_caches[index]`` and re-cat'd inside the vendored attention BEFORE our
# update runs). So offload is driven from the GLMTransformer loop, not the cache:
#   * before ``layer(index)``: H2D the previous KV so the vendored cat sees GPU,
#     and rebind both ``kv_caches[index]`` and the module's ``_kitty_cache[0]``.
#   * after appending to ``presents``: D2H the new KV, rebind ``presents[index]``
#     AND ``_kitty_cache[0]`` to the host mirror so the GPU tensor is freed.
# Synchronous (no overlap) to keep the working set at ~1 layer (max memory win,
# which is the whole point for GLM's 128k OOM).
# --------------------------------------------------------------------------- #
class GLMOffloadManager:
    def __init__(self, max_length: int) -> None:
        if max_length is None or int(max_length) <= 0:
            raise ValueError("GLM KV offload requires a positive max_length.")
        self.max_length = int(max_length)
        self.device: torch.device | None = None
        self._pool = pinned_pool()

    @staticmethod
    def _attn(layer):
        return getattr(layer, "self_attention", None) or getattr(layer, "self_attn", None)

    def prefetch(self, layer, index: int, kv_tuple):
        """H2D the previous step's KV for ``index`` and sync the module cache."""
        if kv_tuple is None:
            return kv_tuple
        k, v = kv_tuple
        if k.device.type != "cpu":
            return kv_tuple
        assert self.device is not None, "GLM offload device not captured yet"
        gk = k.to(self.device, non_blocking=False).contiguous()
        gv = v.to(self.device, non_blocking=False).contiguous()
        attn = self._attn(layer)
        kc = getattr(attn, "_kitty_cache", None) if attn is not None else None
        if kc is not None and len(kc.key_cache) > 0:
            kc.key_cache[0] = gk
            kc.value_cache[0] = gv
        return (gk, gv)

    def evict(self, layer, index: int, kv_tuple):
        """D2H the new KV for ``index`` to its pinned host mirror; sync caches."""
        if kv_tuple is None:
            return kv_tuple
        k, v = kv_tuple
        if k.device.type == "cpu":
            return kv_tuple
        if self.device is None:
            self.device = k.device
        cur = k.shape[-2]
        if cur > self.max_length:
            raise RuntimeError(
                f"GLM KV offload: layer {index} length {cur} exceeds max_length "
                f"{self.max_length}; raise MAX_MODEL_LEN + MAX_GEN."
            )
        hk = self._pool.mirror(index, "K", k, self.max_length)
        hv = self._pool.mirror(index, "V", v, self.max_length)
        hk[:, :, :cur, :].copy_(k)
        hv[:, :, :cur, :].copy_(v)
        host_k = hk[:, :, :cur, :]
        host_v = hv[:, :, :cur, :]
        attn = self._attn(layer)
        kc = getattr(attn, "_kitty_cache", None) if attn is not None else None
        if kc is not None and len(kc.key_cache) > 0:
            kc.key_cache[0] = host_k
            kc.value_cache[0] = host_v
        return (host_k, host_v)


# --------------------------------------------------------------------------- #
# (2) De-fragmentation: GLMTransformer.forward that appends per-layer tuples
#     instead of torch.cat-stacking them along the layer axis.
# --------------------------------------------------------------------------- #
def _patched_glmtransformer_forward(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
    use_cache=True, output_hidden_states=False,
):
    offloader = getattr(self, "_kitty_offloader", None)
    if not kv_caches:
        kv_caches = [None for _ in range(self.num_layers)]
    elif offloader is not None and not isinstance(kv_caches, list):
        # Need mutability so prefetch can rebind a layer's host KV to GPU.
        kv_caches = list(kv_caches)
    presents = [] if use_cache else None
    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    all_self_attentions = None
    all_hidden_states = () if output_hidden_states else None
    for index in range(self.num_layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer = self._get_layer(index)
        # Offload: bring this layer's previous KV back to GPU before the vendored
        # attention re-cats it (kv_caches[index] was evicted to host last step).
        if offloader is not None and kv_caches[index] is not None:
            kv_caches[index] = offloader.prefetch(layer, index, kv_caches[index])

        if self.gradient_checkpointing and self.training:
            layer_ret = torch.utils.checkpoint.checkpoint(
                layer, hidden_states, attention_mask, rotary_pos_emb,
                kv_caches[index], use_cache, use_reentrant=False,
            )
        else:
            layer_ret = layer(
                hidden_states, attention_mask, rotary_pos_emb,
                kv_cache=kv_caches[index], use_cache=use_cache,
            )
        hidden_states, kv_cache = layer_ret
        if use_cache:
            # Offload: move this layer's new KV to host so `presents` holds the
            # host mirror and the GPU tensor is freed (the memory win).
            if offloader is not None:
                kv_cache = offloader.evict(layer, index, kv_cache)
            # Always append per-layer (key, value); never torch.cat along the layer
            # axis. This is the de-fragmentation fix (matches Llama/Qwen).
            presents.append(kv_cache)

    if use_cache:
        presents = tuple(presents)
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)
    if self.post_layer_norm:
        hidden_states = self.final_layernorm(hidden_states)
    return hidden_states, presents, all_hidden_states, all_self_attentions


# --------------------------------------------------------------------------- #
# (3) generate() compat shim for transformers>=4.57
# --------------------------------------------------------------------------- #
def _extract_past_from_model_output(self, outputs, **kwargs):
    cache_name = "past_key_values"
    past = getattr(outputs, "past_key_values", None)
    if past is None and hasattr(outputs, "mems"):
        cache_name, past = "mems", outputs.mems
    return cache_name, past


def _is_legacy_tuple_cache(past_key_values) -> bool:
    """True only for GLM's own legacy cache: a non-empty tuple/list of per-layer
    ``(key, value)`` tensor tuples. transformers>=4.57 injects an empty
    ``DynamicCache`` on the first generate() step, which GLM's ``get_masks``
    mishandles (``past_key_values[0][0]`` is None); we coerce that to None."""
    if not isinstance(past_key_values, (tuple, list)) or len(past_key_values) == 0:
        return False
    first = past_key_values[0]
    return (
        isinstance(first, (tuple, list))
        and len(first) >= 1
        and torch.is_tensor(first[0])
    )


def install_glm_cache_plumbing(model) -> None:
    """Install the GLM cache de-fragmentation + transformers>=4.57 generate shims.

    These are pure plumbing fixes independent of how each attention layer computes
    (fake-quant or real Triton kernel), so both GLM Kitty installers share them:
      (2) replace ``GLMTransformer.forward`` to append per-layer tuples (de-frag);
      (3) restore ``_extract_past_from_model_output`` and coerce the empty
          ``DynamicCache`` transformers injects on the first step to None.
    """
    transformer_modules = [m for m in model.modules() if type(m).__name__ == "GLMTransformer"]
    if not transformer_modules:
        raise RuntimeError(
            "install_glm_cache_plumbing: no GLMTransformer module found; cannot apply "
            "the de-fragmentation fix."
        )
    tr_cls = type(transformer_modules[0])
    if not getattr(tr_cls, "_kitty_defrag_installed", False):
        tr_cls._kitty_orig_forward = tr_cls.forward
        tr_cls.forward = _patched_glmtransformer_forward
        tr_cls._kitty_defrag_installed = True

    if not hasattr(model, "_extract_past_from_model_output"):
        model._extract_past_from_model_output = MethodType(_extract_past_from_model_output, model)

    gen_cls = type(model)
    if not getattr(gen_cls, "_kitty_prepare_inputs_installed", False):
        orig_prepare = gen_cls.prepare_inputs_for_generation

        def patched_prepare_inputs(self, input_ids, past_key_values=None, **kwargs):
            if past_key_values is not None and not _is_legacy_tuple_cache(past_key_values):
                past_key_values = None
            return orig_prepare(self, input_ids, past_key_values=past_key_values, **kwargs)

        gen_cls.prepare_inputs_for_generation = patched_prepare_inputs
        gen_cls._kitty_orig_prepare_inputs = orig_prepare
        gen_cls._kitty_prepare_inputs_installed = True


def install_glm_kitty_fakequant(
    model,
    cache_config: KittyKVCacheConfig,
    stats=None,
    *,
    offloading: bool = False,
    max_length: int | None = None,
):
    """Install all GLM fixes (fake-quant + de-fragmentation + generate shim) on a
    loaded GLM model. Returns a stats dict and raises if the expected modules are
    not found (so a layout change cannot silently disable Kitty quantization).

    When ``offloading`` is True, a GLMOffloadManager is attached to the
    GLMTransformer so each layer's KV is mirrored to pinned host RAM (layer-wise
    CPU offload) — this is what lets GLM-9B fit long contexts on a 40 GB card."""
    if stats is None:
        stats = {"installed": 0, "calls": 0, "prefill_calls": 0, "decode_calls": 0}

    attn_modules = [m for m in model.modules() if type(m).__name__ == "SelfAttention"]
    if not attn_modules:
        raise RuntimeError(
            "install_glm_kitty_fakequant: no SelfAttention modules found on the model; "
            "the GLM remote modeling layout may have changed. Refusing to silently "
            "skip Kitty quantization."
        )

    # (1) Wrap SelfAttention.forward: fake-quant the cached KV (per-layer, PostQuant)
    #     and always return the (key, value) tuple format.
    attn_cls = type(attn_modules[0])
    if not getattr(attn_cls, "_kitty_fakequant_installed", False):
        orig_forward = attn_cls.forward

        def patched_forward(self, hidden_states, attention_mask, rotary_pos_emb,
                            kv_cache=None, use_cache=True):
            output, new_kv = orig_forward(
                self, hidden_states, attention_mask, rotary_pos_emb,
                kv_cache=kv_cache, use_cache=use_cache,
            )
            if not use_cache or new_kv is None:
                return output, new_kv

            key, value, is_prefill = _split_new_kv(new_kv)

            kc = getattr(self, "_kitty_cache", None)
            if kc is None:
                kc = KittyKVCache(cache_config)
                self._kitty_cache = kc
            if kv_cache is None:  # fresh prefill -> new sequence
                kc.reset()

            prev_len = kc.get_seq_length()
            new_key = key[:, :, prev_len:, :].contiguous()
            new_value = value[:, :, prev_len:, :].contiguous()
            kc.update(new_key, new_value, 0)  # quantizes settled region in place

            stats["calls"] += 1
            stats["prefill_calls" if is_prefill else "decode_calls"] += 1
            # Always return the per-layer tuple format (no stacked tensor), so
            # GLMTransformer can append instead of torch.cat-stacking.
            return output, (kc.key_cache[0], kc.value_cache[0])

        attn_cls.forward = patched_forward
        attn_cls._kitty_fakequant_installed = True
        attn_cls._kitty_orig_forward = orig_forward

    # (2)+(3) de-fragmentation + generate() compat (shared plumbing).
    install_glm_cache_plumbing(model)

    # (2b) Attach the layer-wise CPU KV offload manager (per GLMTransformer
    #      instance) when offload is enabled. _patched_glmtransformer_forward
    #      (installed by the plumbing) reads ``self._kitty_offloader``.
    if offloading:
        manager = GLMOffloadManager(max_length)
        for tm in (m for m in model.modules() if type(m).__name__ == "GLMTransformer"):
            tm._kitty_offloader = manager
        stats["offloading"] = True
        stats["offload_max_length"] = manager.max_length

    stats["installed"] = len(attn_modules)
    return stats


# --------------------------------------------------------------------------- #
# Real Triton QUEST + Kitty kernel for GLM (vs. the fake-quant path above).
#
# GLM's SelfAttention.forward is REPLACED (not wrapped): we redo GLM's QKV split +
# RoPE, drive a per-layer real paged ``KittyCache`` (prefill: dense GLM
# core_attention + quantize_prefill; decode: Triton ``kitty_attention_forward`` with
# query-aware QUEST page selection + quantize_decode), and return a shape-only
# placeholder tuple so GLM's legacy-cache loop / get_masks / generate keep working
# (the real cache lives on each module, off to the side of GLM's tuple plumbing).
# --------------------------------------------------------------------------- #
def _glm_real_kitty_attention_forward(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    # Real kernel + cache are imported lazily so the fake-quant/sim path never pulls
    # in Triton.
    from kitty.kvcache import get_kvcache_kitty as _get_real_kvcache_kitty
    from kitty.kvcache.kernels.kitty_attention import kitty_attention_forward as _kitty_attn

    stats = self._kitty_stats
    cfg = self._kitty_kernel_cfg

    # ---- QKV projection + split + RoPE (mirror modeling_chatglm.py:455-491) ----
    if not self.multi_query_attention:
        raise NotImplementedError(
            "GLM real Kitty kernel currently supports multi_query_attention=True only."
        )
    nh = self.num_attention_heads_per_partition
    ng = self.num_multi_query_groups_per_partition
    hd = self.hidden_size_per_attention_head
    mixed = self.query_key_value(hidden_states)
    query_layer, key_layer, value_layer = mixed.split([nh * hd, ng * hd, ng * hd], dim=-1)
    query_layer = query_layer.view(query_layer.size()[:-1] + (nh, hd))
    key_layer = key_layer.view(key_layer.size()[:-1] + (ng, hd))
    value_layer = value_layer.view(value_layer.size()[:-1] + (ng, hd))
    # [b, sq, heads, hd] -> [b, heads, sq, hd]
    query_layer = query_layer.transpose(1, 2)
    key_layer = key_layer.transpose(1, 2)
    value_layer = value_layer.transpose(1, 2)
    if rotary_pos_emb is not None:
        query_layer = self._kitty_apply_rope(query_layer, rotary_pos_emb)
        key_layer = self._kitty_apply_rope(key_layer, rotary_pos_emb)

    # ---- per-sample real paged KittyCache (rebuilt on each fresh prefill) ----
    if kv_cache is None:  # fresh prefill => new sequence
        max_len = self._kitty_state.get("max_length")
        if max_len is None:
            max_len = int(key_layer.shape[2]) + 1024  # safe fallback margin
        self._kitty_cache = _get_real_kvcache_kitty(
            self._kitty_shim_config, 1, int(max_len),
            page_size=cfg["page_size"], promote_ratio=cfg["promote_ratio"],
            quest_enabled=cfg["quest_enabled"], quest_token_budget=cfg["quest_token_budget"],
            quest_skip_layers=cfg["quest_skip_layers"],
        )
    kc = self._kitty_cache
    is_prefill = kc.update(key_layer, value_layer, 0, None)

    if is_prefill:
        # Dense prefill over MQA-expanded full-precision K/V (mirror :507-527).
        rep = nh // ng
        key_exp = key_layer.unsqueeze(2).expand(-1, -1, rep, -1, -1)
        key_exp = key_exp.contiguous().view(key_layer.size(0), nh, key_layer.size(2), hd)
        val_exp = value_layer.unsqueeze(2).expand(-1, -1, rep, -1, -1)
        val_exp = val_exp.contiguous().view(value_layer.size(0), nh, value_layer.size(2), hd)
        context_layer = self.core_attention(query_layer, key_exp, val_exp, attention_mask)
        kc.quantize_prefill(0)
        stats["prefill_calls"] += 1
    else:
        # Decode: real Triton QUEST kernel over the quantized paged cache.
        attn_output, _ = _kitty_attn(self, query_layer.contiguous(), kc.kv_cache[0], scaling=hd ** -0.5)
        kc.quantize_decode(0)
        b, sq = attn_output.size(0), attn_output.size(1)
        context_layer = attn_output.reshape(b, sq, -1).contiguous()  # [b, sq, hp]
        layer0 = kc.kv_cache[0]
        path = str(getattr(layer0, "last_quest_path", "unknown"))
        stats["paths"][path] = stats["paths"].get(path, 0) + 1
        sel = getattr(layer0, "last_selected_pages", None)
        if sel is not None and getattr(sel, "ndim", 0) > 0:
            stats["last_selected_pages"] = int(sel.shape[-1])
        stats["decode_calls"] += 1
    stats["calls"] += 1

    output = self.dense(context_layer)

    if not use_cache:
        return output, None
    # Shape-only placeholder: GLM get_masks only reads past[0][0].shape[2] (=past len).
    total_len = int(kc.get_seq_length())
    placeholder = output.new_empty((output.size(0), ng, total_len, 0))
    return output, (placeholder, placeholder)


def install_glm_real_kitty_kernel(
    model,
    *,
    page_size: int = 16,
    promote_ratio: float = 0.125,
    quest_enabled: bool = True,
    quest_token_budget: int | None = 2048,
    quest_skip_layers: int = 0,
    stats=None,
):
    """Install the REAL Triton QUEST+Kitty decode kernel on a loaded GLM model.

    Per-layer Method A: each ``SelfAttention`` owns a 1-layer real ``KittyCache``
    (addressed at layer_idx=0). Reuses the shared GLM cache plumbing. Returns a stats
    dict (``calls``/``decode_calls``/``paths``/...) used by the runner guardrail.
    Pair with ``set_glm_real_kitty_sample_length`` to size the cache per sample.
    """
    import sys
    from types import SimpleNamespace

    if stats is None:
        stats = {
            "installed": 0, "calls": 0, "prefill_calls": 0, "decode_calls": 0,
            "paths": {}, "last_selected_pages": None,
        }

    attn_modules = [m for m in model.modules() if type(m).__name__ == "SelfAttention"]
    if not attn_modules:
        raise RuntimeError(
            "install_glm_real_kitty_kernel: no SelfAttention modules found; the GLM "
            "remote modeling layout may have changed. Refusing to silently skip QUEST."
        )

    attn_cls = type(attn_modules[0])
    glm_mod = sys.modules[attn_cls.__module__]
    apply_rope = glm_mod.apply_rotary_pos_emb

    shared_state = {"max_length": None}
    kernel_cfg = {
        "page_size": page_size, "promote_ratio": promote_ratio,
        "quest_enabled": quest_enabled, "quest_token_budget": quest_token_budget,
        "quest_skip_layers": quest_skip_layers,
    }

    for attn in attn_modules:
        # kitty_attention_forward reads these off the module.
        attn.num_attention_heads = attn.num_attention_heads_per_partition
        attn.num_key_value_heads = attn.num_multi_query_groups_per_partition
        attn._kitty_shim_config = SimpleNamespace(
            num_hidden_layers=1,
            head_dim=attn.hidden_size_per_attention_head,
            num_key_value_heads=attn.num_multi_query_groups_per_partition,
        )
        attn._kitty_kernel_cfg = kernel_cfg
        attn._kitty_state = shared_state
        attn._kitty_stats = stats
        attn._kitty_apply_rope = apply_rope
        attn._kitty_cache = None

    model._glm_kitty_state = shared_state
    model._glm_kitty_stats = stats

    if not getattr(attn_cls, "_kitty_real_kernel_installed", False):
        attn_cls._kitty_real_orig_forward = attn_cls.forward
        attn_cls.forward = _glm_real_kitty_attention_forward
        attn_cls._kitty_real_kernel_installed = True

    install_glm_cache_plumbing(model)
    stats["installed"] = len(attn_modules)
    return stats


def set_glm_real_kitty_sample_length(model, max_length: int) -> None:
    """Size the per-layer GLM KittyCache for the next sequence (context + max_gen)."""
    state = getattr(model, "_glm_kitty_state", None)
    if state is not None:
        state["max_length"] = int(max_length)
