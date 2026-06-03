"""Layer-wise CPU KV-cache offload for the Kitty sim (fake-quant) path.

Motivation: the sim path stores fp16 KV for every layer on the GPU, so long
contexts OOM on a 40 GB card (GLM-9B ~10 GiB @128k, Qwen3-8B ~18 GiB @128k).
Offloading each layer's stored KV to pinned host RAM and only keeping a small
working window on the GPU lets those runs fit.

This module is the **synchronous (M1a)** implementation: after a layer's KV is
produced/used it is copied to a pinned host mirror and the GPU tensor is
dropped; before a layer is needed again it is copied back. It is numerically a
no-op (fp16 bytes are moved CPU<->GPU unchanged), so offload-on must produce
bit-identical outputs to offload-off.

Pinned mirrors live in a process-lifetime pool so the per-sample KittyKVCache
churn in the LongBench runner does not re-pin host memory every sample.

Double-buffered prefetch (overlapping the copy with compute on a dedicated CUDA
stream) is layered on top of this in a later milestone; the public
``evict`` / ``ensure_resident`` API is designed to stay stable across that work.
"""
from __future__ import annotations

import torch

try:  # psutil is in the documented kitty stack; degrade gracefully if absent
    import psutil

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PSUTIL = False

_HOST_MARGIN_BYTES = 4 << 30  # keep 4 GiB host headroom before pinning more


class _PinnedHostPool:
    """Process-lifetime pool of pinned host KV mirrors.

    Keyed by ``(layer_idx, which)`` and sized once to the ``max_length`` ceiling
    so the per-sample cache instances the runner creates reuse the same buffers
    instead of re-pinning (pinning is slow and non-pageable).
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple[int, str], torch.Tensor] = {}

    def mirror(self, layer_idx: int, which: str, ref: torch.Tensor, max_length: int) -> torch.Tensor:
        b, h, _seq, d = ref.shape
        want = (b, h, max_length, d)
        key = (layer_idx, which)
        buf = self._buffers.get(key)
        if buf is None or tuple(buf.shape) != want or buf.dtype != ref.dtype:
            self._guard(b * h * max_length * d * ref.element_size())
            buf = torch.empty(want, dtype=ref.dtype, device="cpu", pin_memory=True)
            self._buffers[key] = buf
        return buf

    @staticmethod
    def _guard(nbytes: int) -> None:
        if not _HAS_PSUTIL:
            return
        avail = psutil.virtual_memory().available
        if nbytes + _HOST_MARGIN_BYTES > avail:
            raise MemoryError(
                f"KV offload: refusing to pin {nbytes / 2**30:.2f} GiB host RAM; only "
                f"{avail / 2**30:.2f} GiB available (need +{_HOST_MARGIN_BYTES / 2**30:.0f} GiB margin)."
            )

    def clear(self) -> None:
        self._buffers.clear()


_POOL = _PinnedHostPool()


def pinned_pool() -> _PinnedHostPool:
    return _POOL


class LayerKVOffloader:
    """Offload policy for one ``KittyKVCache``.

    Pinned mirrors come from the shared process-lifetime pool, so distinct cache
    instances (one per LongBench sample) reuse the same host buffers.
    """

    def __init__(
        self,
        max_length: int | None,
        resident_layers: int = 2,
        prefetch: bool = False,
    ) -> None:
        if max_length is None or int(max_length) <= 0:
            raise ValueError(
                "KV offload requires a positive max_length ceiling "
                "(set it to MAX_MODEL_LEN + MAX_GEN)."
            )
        self.max_length = int(max_length)
        # resident_layers is the working-window / latency dial used by the
        # double-buffered milestone; the synchronous path keeps only the active
        # layer resident and ignores it beyond bookkeeping.
        self.resident_layers = int(resident_layers)
        self.device: torch.device | None = None
        # Optional double-buffered prefetch: H2D the *next* layer one step ahead
        # on a dedicated copy stream so the transfer overlaps the current layer's
        # compute. Evict stays synchronous, so the working set grows by only ~1
        # layer (the staged next layer). Opt-in: it trades a little memory for
        # latency and helps mainly at moderate contexts (where copy ~ compute);
        # at >=128k the copy dominates and the overlap is partial.
        self.prefetch_enabled = bool(prefetch)
        self._copy_stream: "torch.cuda.Stream | None" = None
        self._h2d_done: dict[int, "torch.cuda.Event"] = {}
        self._staged: dict[int, tuple] = {}

    # ------------------------------------------------------------------ #
    # synchronous primitives (M1a)
    # ------------------------------------------------------------------ #
    def capture_device(self, ref: torch.Tensor) -> None:
        if self.device is None:
            self.device = ref.device

    def evict(self, key_cache: list, value_cache: list, layer_idx: int) -> None:
        """Move a layer's stored KV from GPU to its pinned host mirror."""
        k = key_cache[layer_idx]
        if k.device.type == "cpu":
            return  # already host-resident
        v = value_cache[layer_idx]
        cur = k.shape[-2]
        if cur > self.max_length:
            raise RuntimeError(
                f"KV offload: layer {layer_idx} length {cur} exceeds max_length "
                f"{self.max_length}; raise MAX_MODEL_LEN + MAX_GEN."
            )
        hk = _POOL.mirror(layer_idx, "K", k, self.max_length)
        hv = _POOL.mirror(layer_idx, "V", v, self.max_length)
        hk[:, :, :cur, :].copy_(k)
        hv[:, :, :cur, :].copy_(v)
        # Rebind the list slot to a host view; drops the GPU tensor reference so
        # the caching allocator can reclaim it.
        key_cache[layer_idx] = hk[:, :, :cur, :]
        value_cache[layer_idx] = hv[:, :, :cur, :]

    def ensure_resident(self, key_cache: list, value_cache: list, layer_idx: int) -> None:
        """Bring a layer's KV back to the compute device if it was evicted.

        If the layer was prefetched (staged on the copy stream), adopt it after a
        single event wait; otherwise fall back to a synchronous H2D.
        """
        k = key_cache[layer_idx]
        if k.device.type != "cpu":
            return
        dev = self.device
        if layer_idx in self._staged:
            torch.cuda.current_stream().wait_event(self._h2d_done[layer_idx])
            gk, gv = self._staged.pop(layer_idx)
            key_cache[layer_idx] = gk
            value_cache[layer_idx] = gv
            return
        key_cache[layer_idx] = k.to(dev, non_blocking=False).contiguous()
        value_cache[layer_idx] = value_cache[layer_idx].to(dev, non_blocking=False).contiguous()

    def prefetch_next(self, key_cache: list, value_cache: list, layer_idx: int) -> None:
        """Issue an async H2D for the layer after ``layer_idx`` (wrapping to 0 on
        the last layer, so the next decode step's first layer is ready) on the
        copy stream, overlapping it with this layer's compute. No-op unless
        prefetch is enabled."""
        if not self.prefetch_enabled:
            return
        num_layers = len(key_cache)
        if num_layers <= 1 or self.device is None:
            return
        nxt = (layer_idx + 1) % num_layers
        k = key_cache[nxt]
        if k.device.type != "cpu" or nxt in self._staged:
            return  # already resident or already staged
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.device)
        v = value_cache[nxt]
        with torch.cuda.stream(self._copy_stream):
            gk = k.to(self.device, non_blocking=True)
            gv = v.to(self.device, non_blocking=True)
            ev = self._h2d_done.get(nxt)
            if ev is None:
                ev = torch.cuda.Event()
                self._h2d_done[nxt] = ev
            ev.record(self._copy_stream)
        self._staged[nxt] = (gk, gv)
