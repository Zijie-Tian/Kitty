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
# (2) De-fragmentation: GLMTransformer.forward that appends per-layer tuples
#     instead of torch.cat-stacking them along the layer axis.
# --------------------------------------------------------------------------- #
def _patched_glmtransformer_forward(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
    use_cache=True, output_hidden_states=False,
):
    if not kv_caches:
        kv_caches = [None for _ in range(self.num_layers)]
    presents = () if use_cache else None
    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    all_self_attentions = None
    all_hidden_states = () if output_hidden_states else None
    for index in range(self.num_layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer = self._get_layer(index)
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
            # Always append per-layer (key, value); never torch.cat along the layer
            # axis. This is the de-fragmentation fix (matches Llama/Qwen).
            presents = presents + (kv_cache,)

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


def install_glm_kitty_fakequant(model, cache_config: KittyKVCacheConfig, stats=None):
    """Install all GLM fixes (fake-quant + de-fragmentation + generate shim) on a
    loaded GLM model. Returns a stats dict and raises if the expected modules are
    not found (so a layout change cannot silently disable Kitty quantization)."""
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

    # (2) Replace GLMTransformer.forward to append per-layer tuples (de-frag).
    transformer_modules = [m for m in model.modules() if type(m).__name__ == "GLMTransformer"]
    if not transformer_modules:
        raise RuntimeError(
            "install_glm_kitty_fakequant: no GLMTransformer module found; cannot apply "
            "the de-fragmentation fix."
        )
    tr_cls = type(transformer_modules[0])
    if not getattr(tr_cls, "_kitty_defrag_installed", False):
        tr_cls._kitty_orig_forward = tr_cls.forward
        tr_cls.forward = _patched_glmtransformer_forward
        tr_cls._kitty_defrag_installed = True

    # (3) generate() compat for transformers>=4.57:
    #   (a) restore the removed _extract_past_from_model_output (bound to instance);
    #   (b) coerce the empty DynamicCache transformers injects on the first step to
    #       None, so GLM uses its own legacy tuple cache (which get_masks expects).
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

    stats["installed"] = len(attn_modules)
    return stats
