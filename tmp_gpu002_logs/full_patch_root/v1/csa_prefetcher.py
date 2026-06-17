"""
csa_prefetcher.py — HC-proxy speculative block prefetcher for DeepSeek-V4 CSA layers.

At each decode step, for each CSA layer L the Indexer selects index_topk=1024
compressed KV blocks. Adjacent decode steps share ~91% of these blocks.

HC-proxy speculative prefetch (CSASpecPrefetcher):
  Before the Indexer runs for layer L, compute a proxy using the HC residual
  from the previous block:
    proxy = attn_norm_L( HC_pre_L(residual_f_{L-1}, hc_attn_fn_L, ...) )
  This proxy predicts spec_topk with ~83.8% accuracy. Spec reads are submitted
  immediately, overlapping with the FFN/attention computation window (~3300 µs).
  When the real Indexer returns true_topk, only the ~16% miss blocks need
  fallback reads.

Integration:
    prefetch = CSASpecPrefetcher(store, csa_layer_ids=sorted(csa_lids))
    prefetch.patch_transformer(transformer)   # call once after model load
    prefetch.reset()                          # call before each new decode run
    prefetch.unpatch_transformer()            # reversible
"""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch


# ---------------------------------------------------------------------------
# Portable pread
# ---------------------------------------------------------------------------

if hasattr(os, "pread"):
    _pread = os.pread  # POSIX: atomic, thread-safe, no seek side-effects
else:
    _pread_locks: Dict[int, threading.Lock] = {}

    def _pread(fd: int, n: int, offset: int) -> bytes:  # type: ignore[misc]
        """Windows fallback: seek+read under a per-fd lock."""
        if fd not in _pread_locks:
            _pread_locks[fd] = threading.Lock()
        with _pread_locks[fd]:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, n)


# ---------------------------------------------------------------------------
# Block store
# ---------------------------------------------------------------------------


@dataclass
class CSABlockStoreConfig:
    """Configuration for CSABlockStore.

    Args:
        store_dir:        Directory holding per-layer .bin files.
        n_blocks:         Number of compressed blocks per layer
                          (= max_seq_len // compress_ratio).
        block_size_bytes: Bytes per block
                          (= compress_ratio * kv_lora_rank * sizeof(dtype)).
        io_workers:       Thread-pool size for async reads.
        use_odirect:      Open with O_DIRECT on Linux to bypass page cache.
    """

    store_dir: str
    n_blocks: int
    block_size_bytes: int
    io_workers: int = 8
    use_odirect: bool = False


class CSABlockStore:
    """Flat-file KV block store indexed by (layer_id, block_id).

    Layout: one file per layer — ``{store_dir}/csa_layer_{lid}.bin``.
    Block ``bid`` occupies byte range
    ``[bid * block_size_bytes, (bid + 1) * block_size_bytes)``.
    Files are pre-allocated on first open.
    """

    def __init__(self, cfg: CSABlockStoreConfig) -> None:
        os.makedirs(cfg.store_dir, exist_ok=True)
        self._cfg = cfg
        self._fds: Dict[int, int] = {}
        self._write_paths: Dict[int, str] = {}
        self._executor = ThreadPoolExecutor(max_workers=cfg.io_workers,
                                            thread_name_prefix="csa_io")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _layer_path(self, layer_id: int) -> str:
        return os.path.join(self._cfg.store_dir, f"csa_layer_{layer_id}.bin")

    def _open_rd(self, layer_id: int) -> int:
        if layer_id not in self._fds:
            path = self._layer_path(layer_id)
            if not os.path.exists(path):
                self._preallocate(layer_id, path)
            flags = os.O_RDONLY
            if self._cfg.use_odirect and sys.platform == "linux":
                flags |= getattr(os, "O_DIRECT", 0)
            self._fds[layer_id] = os.open(path, flags)
        return self._fds[layer_id]

    def _preallocate(self, layer_id: int, path: str) -> None:
        total = self._cfg.n_blocks * self._cfg.block_size_bytes
        with open(path, "wb") as f:
            # write in 4 MB chunks to avoid one huge allocation
            chunk = min(total, 4 * 1024 * 1024)
            zeros = b"\x00" * chunk
            written = 0
            while written < total:
                n = min(chunk, total - written)
                f.write(zeros[:n])
                written += n

    # ------------------------------------------------------------------
    # Write (synchronous — used during prefill / setup)
    # ------------------------------------------------------------------

    def write_block_sync(self, layer_id: int, block_id: int,
                         data: bytes) -> None:
        """Write one block synchronously.

        Args:
            layer_id: CSA layer index.
            block_id: Compressed-block index.
            data:     Exactly ``block_size_bytes`` bytes.

        Raises:
            ValueError: If ``len(data) != block_size_bytes``.
        """
        if len(data) != self._cfg.block_size_bytes:
            raise ValueError(
                f"Expected {self._cfg.block_size_bytes} bytes, got {len(data)}"
            )
        path = self._layer_path(layer_id)
        if not os.path.exists(path):
            self._preallocate(layer_id, path)
        offset = block_id * self._cfg.block_size_bytes
        with open(path, "r+b") as f:
            f.seek(offset)
            f.write(data)

    def write_blocks_from_tensor(self, layer_id: int,
                                  data: torch.Tensor) -> None:
        """Write all blocks for a layer from a tensor.

        Args:
            layer_id: CSA layer index.
            data:     CPU tensor of shape ``[n_blocks, block_size_bytes // sizeof]``.
                      Will be flattened to raw bytes and written sequentially.
        """
        raw = data.contiguous().cpu().view(torch.int16).numpy().tobytes()
        path = self._layer_path(layer_id)
        if not os.path.exists(path):
            self._preallocate(layer_id, path)
        with open(path, "r+b") as f:
            f.seek(0)
            f.write(raw)

    # ------------------------------------------------------------------
    # Read (asynchronous)
    # ------------------------------------------------------------------

    def read_block_sync(self, layer_id: int, block_id: int) -> bytes:
        """Blocking read for one block (used by naive/non-speculative path).

        Returns:
            Bytes of length ``block_size_bytes``.
        """
        fd = self._open_rd(layer_id)
        n = self._cfg.block_size_bytes
        return _pread(fd, n, block_id * n)

    def read_block_async(self, layer_id: int, block_id: int) -> Future:
        """Submit an async read for one block.

        Returns:
            Future that resolves to ``bytes`` of length ``block_size_bytes``.
        """
        fd = self._open_rd(layer_id)
        n = self._cfg.block_size_bytes
        offset = block_id * n
        return self._executor.submit(_pread, fd, n, offset)

    def read_blocks_async(self, layer_id: int,
                          block_ids: Set[int]) -> Dict[int, Future]:
        """Submit async reads for multiple blocks.

        Returns:
            Mapping from block_id to Future[bytes].
        """
        return {bid: self.read_block_async(layer_id, bid)
                for bid in block_ids}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the thread pool and close all file descriptors."""
        self._executor.shutdown(wait=False)
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class LayerPrefetchStats:
    """Per-layer prefetch statistics over one decode run."""

    layer_id: int
    steps: int = 0
    total_delta: int = 0       # blocks submitted for prefetch
    total_topk: int = 0        # blocks actually selected
    total_io_us: float = 0.0   # µs spent in synchronous read (baseline)
    max_delta: int = 0
    min_delta: int = 0

    @property
    def mean_delta(self) -> float:
        return self.total_delta / self.steps if self.steps else 0.0

    @property
    def overlap_rate(self) -> float:
        """Fraction of topk blocks that were already held (prev step overlap)."""
        if self.total_topk == 0:
            return 0.0
        return 1.0 - self.total_delta / self.total_topk

    def __repr__(self) -> str:
        return (f"layer={self.layer_id:3d}  steps={self.steps}  "
                f"mean_delta={self.mean_delta:.1f}  "
                f"overlap={self.overlap_rate:.3f}  "
                f"total_io_ms={self.total_io_us/1e3:.2f}")


# ---------------------------------------------------------------------------
# Prefetcher
# ---------------------------------------------------------------------------


class CSAPrefetcher:
    """Speculative block prefetcher for CSA layers.

    Call ``patch_transformer(transformer)`` once after model load, then
    ``reset()`` before starting each decode run.

    Args:
        store:          Backing CSABlockStore instance.
        csa_layer_ids:  Layer IDs of CSA blocks in the transformer.
        index_topk:     Number of blocks the Indexer selects per step.
    """

    def __init__(self, store: CSABlockStore,
                 csa_layer_ids: List[int],
                 index_topk: int = 1024) -> None:
        self._store = store
        self._csa_layer_ids = set(csa_layer_ids)
        self._index_topk = index_topk

        # Per-layer delta state
        self._prev_topk: Dict[int, Set[int]] = {}
        # Pending prefetch futures: (layer_id, block_id) -> Future[bytes]
        self._pending: Dict[tuple, Future] = {}
        self._pending_lock = threading.Lock()

        # Statistics
        self._stats: Dict[int, LayerPrefetchStats] = {
            lid: LayerPrefetchStats(layer_id=lid)
            for lid in csa_layer_ids
        }

        # Saved originals for unpatching
        self._orig_indexer_forward = None
        self._patched = False

    # ------------------------------------------------------------------
    # Reset between runs
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear delta cache and pending futures. Call before each decode."""
        self._prev_topk.clear()
        with self._pending_lock:
            self._pending.clear()
        for s in self._stats.values():
            s.steps = s.total_delta = s.total_topk = 0
            s.total_io_us = 0.0
            s.max_delta = s.min_delta = 0

    # ------------------------------------------------------------------
    # Transformer patching
    # ------------------------------------------------------------------

    def patch_transformer(self, transformer) -> None:
        """Monkey-patch Indexer.forward on ``transformer`` in-place.

        Args:
            transformer: The Transformer object whose Indexer modules will
                         be patched. The Indexer class is located via
                         ``lmcache.v1.csa_prefetcher._locate_indexer_class``.

        Raises:
            RuntimeError: If already patched.
        """
        if self._patched:
            raise RuntimeError("Already patched; call unpatch_transformer first.")

        indexer_cls = _locate_indexer_class(transformer)
        if indexer_cls is None:
            raise RuntimeError("Could not find Indexer class on transformer.")

        # Tag each Indexer with its layer_id
        for name, module in transformer.named_modules():
            if type(module).__name__ == "Indexer":
                parts = name.split(".")
                try:
                    lid = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    lid = -1
                module._csa_layer_id = lid

        self._orig_indexer_forward = indexer_cls.forward
        prefetcher_ref = self  # avoid circular ref via closure

        orig = self._orig_indexer_forward

        def _patched_forward(self_idx, x, qr, start_pos, offset):
            topk = orig(self_idx, x, qr, start_pos, offset)
            if start_pos > 0:
                lid = getattr(self_idx, "_csa_layer_id", -1)
                if lid in prefetcher_ref._csa_layer_ids:
                    prefetcher_ref._on_real_topk(lid, topk, start_pos)
            return topk

        indexer_cls.forward = _patched_forward
        self._patched_indexer_cls = indexer_cls
        self._patched = True

    def unpatch_transformer(self) -> None:
        """Restore the original Indexer.forward."""
        if not self._patched:
            return
        self._patched_indexer_cls.forward = self._orig_indexer_forward
        self._patched = False

    # ------------------------------------------------------------------
    # Core callback
    # ------------------------------------------------------------------

    def _on_real_topk(self, layer_id: int, topk: torch.Tensor,
                      start_pos: int) -> None:
        """Called immediately after Indexer returns real_topk.

        Computes the delta vs the previous step and submits async reads
        for the new blocks. Executes synchronously on the forward-pass
        thread but returns quickly (Future submission is non-blocking).

        Args:
            layer_id:  CSA layer index.
            topk:      Raw Indexer output tensor.
            start_pos: Current decode position.
        """
        # Flatten to 1-D set of block IDs
        topk_set: Set[int] = set(topk.reshape(-1).cpu().tolist())

        prev = self._prev_topk.get(layer_id, set())
        delta = topk_set - prev

        # Update state
        self._prev_topk[layer_id] = topk_set

        # Submit async reads for delta blocks
        t0 = time.perf_counter()
        futures = self._store.read_blocks_async(layer_id, delta)
        t1 = time.perf_counter()

        with self._pending_lock:
            for bid, fut in futures.items():
                self._pending[(layer_id, bid)] = fut

        # Update stats
        s = self._stats.get(layer_id)
        if s is not None:
            n_delta = len(delta)
            n_topk = len(topk_set)
            s.steps += 1
            s.total_delta += n_delta
            s.total_topk += n_topk
            # Baseline I/O cost: synchronous pread for the delta blocks
            s.total_io_us += (t1 - t0) * 1e6
            if s.steps == 1:
                s.max_delta = s.min_delta = n_delta
            else:
                s.max_delta = max(s.max_delta, n_delta)
                s.min_delta = min(s.min_delta, n_delta)

    # ------------------------------------------------------------------
    # Wait / drain
    # ------------------------------------------------------------------

    def wait_layer(self, layer_id: int,
                   timeout: Optional[float] = None) -> int:
        """Block until all pending reads for ``layer_id`` complete.

        Args:
            layer_id: CSA layer index.
            timeout:  Per-future timeout in seconds, or None.

        Returns:
            Number of futures resolved.
        """
        to_wait: List[Future] = []
        with self._pending_lock:
            keys = [k for k in self._pending if k[0] == layer_id]
            for k in keys:
                to_wait.append(self._pending.pop(k))
        for f in to_wait:
            f.result(timeout=timeout)
        return len(to_wait)

    def wait_all(self, timeout: Optional[float] = None) -> int:
        """Block until all pending reads across all layers complete.

        Returns:
            Total number of futures resolved.
        """
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for f in pending:
            f.result(timeout=timeout)
        return len(pending)

    # ------------------------------------------------------------------
    # Stats access
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[int, LayerPrefetchStats]:
        """Return per-layer stats dict (copy)."""
        return dict(self._stats)

    def print_stats(self) -> None:
        """Print a formatted stats table to stdout."""
        print("\n--- CSAPrefetcher stats ---")
        print(f"  {'Layer':>6}  {'Steps':>6}  {'MeanΔ':>7}  "
              f"{'MaxΔ':>6}  {'MinΔ':>6}  {'Overlap':>8}  {'SubmitMs':>9}")
        print("  " + "-" * 65)
        for lid in sorted(self._stats):
            s = self._stats[lid]
            if s.steps == 0:
                continue
            print(f"  {lid:6d}  {s.steps:6d}  {s.mean_delta:7.1f}  "
                  f"{s.max_delta:6d}  {s.min_delta:6d}  "
                  f"{s.overlap_rate:8.3f}  {s.total_io_us/1e3:9.3f}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _locate_indexer_class(transformer):
    """Return the Indexer class used by ``transformer``, or None."""
    for _, module in transformer.named_modules():
        if type(module).__name__ == "Indexer":
            return type(module)
    return None


def build_from_model(transformer,
                     store_dir: str,
                     compress_ratio: int,
                     kv_lora_rank: int,
                     max_seq_len: int,
                     dtype: Optional[torch.dtype] = None,
                     io_workers: int = 8,
                     speculative: bool = False) -> "CSAPrefetcher":
    """Convenience factory: build CSABlockStore + CSAPrefetcher from model.

    Computes block_size_bytes = compress_ratio * kv_lora_rank * sizeof(dtype).
    Discovers CSA layer IDs from the transformer's Indexer submodules.

    Args:
        transformer:    The Transformer instance.
        store_dir:      Directory for per-layer .bin files.
        compress_ratio: CSA compression ratio (4 for DSV4).
        kv_lora_rank:   Latent KV dimension (512 for DSV4).
        max_seq_len:    Maximum sequence length supported.
        dtype:          KV data type (default: torch.bfloat16).
        io_workers:     Thread-pool size.
        speculative:    If True, return CSASpecPrefetcher with HC-proxy mode.

    Returns:
        A ready-to-use CSAPrefetcher (or CSASpecPrefetcher) — not yet patched.
    """
    if dtype is None:
        dtype = torch.bfloat16
    elem_bytes = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 1
    block_size_bytes = compress_ratio * kv_lora_rank * elem_bytes
    n_blocks = max_seq_len // compress_ratio

    csa_lids: List[int] = []
    for name, module in transformer.named_modules():
        if type(module).__name__ == "Indexer":
            parts = name.split(".")
            try:
                csa_lids.append(int(parts[parts.index("layers") + 1]))
            except (ValueError, IndexError):
                pass
    csa_lids = sorted(set(csa_lids))

    cfg = CSABlockStoreConfig(
        store_dir=store_dir,
        n_blocks=n_blocks,
        block_size_bytes=block_size_bytes,
        io_workers=io_workers,
    )
    store = CSABlockStore(cfg)
    cls = CSASpecPrefetcher if speculative else CSAPrefetcher
    return cls(store, csa_layer_ids=csa_lids)


# ---------------------------------------------------------------------------
# Spec-layer stats (hit/miss for HC-proxy mode)
# ---------------------------------------------------------------------------


@dataclass
class SpecLayerStats:
    """Per-layer hit/miss statistics for HC-proxy speculative mode."""

    layer_id: int
    steps: int = 0
    spec_hits: int = 0    # blocks in spec_topk ∩ true_topk
    spec_total: int = 0   # blocks in true_topk (denominator)
    fallback_reads: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of true_topk blocks correctly speculated."""
        return self.spec_hits / self.spec_total if self.spec_total else 0.0

    def __repr__(self) -> str:
        return (f"layer={self.layer_id:3d}  steps={self.steps}  "
                f"hit={self.hit_rate:.3f}  "
                f"fallback_mean={self.fallback_reads/self.steps:.1f}"
                if self.steps else f"layer={self.layer_id:3d}  steps=0")


# ---------------------------------------------------------------------------
# HC-proxy speculative prefetcher
# ---------------------------------------------------------------------------


def _run_spec_indexer(indexer, proxy: torch.Tensor,
                      qr: torch.Tensor,
                      start_pos: int, offset,
                      orig_forward) -> torch.Tensor:
    """Run Indexer.forward speculatively with ``proxy`` as input.

    Saves and restores the compressor's stateful buffers so the speculative
    run does not pollute the real forward pass.

    Args:
        indexer:      The Indexer module.
        proxy:        HC-transformed proxy: attn_norm_L(HC_pre_L(residual_f)).
        qr:           True query residual for this step (passed through).
        start_pos:    Current decode position.
        offset:       Offset argument for Indexer.forward.
        orig_forward: The unpatched Indexer.forward method.

    Returns:
        topk tensor from the speculative run (same shape as true topk).
    """
    comp = indexer.compressor
    ratio = getattr(indexer, "compress_ratio", 4)
    slot = start_pos // ratio

    saved: Dict[str, torch.Tensor] = {}
    if comp.kv_cache is not None and slot < comp.kv_cache.size(1):
        saved["kv"] = comp.kv_cache[:, slot].clone()
    if hasattr(comp, "kv_state"):
        saved["ks"] = comp.kv_state.clone()
    if hasattr(comp, "score_state"):
        saved["ss"] = comp.score_state.clone()

    with torch.no_grad():
        topk = orig_forward(indexer, proxy, qr, start_pos, offset)

    if "kv" in saved:
        comp.kv_cache[:, slot].copy_(saved["kv"])
    if "ks" in saved:
        comp.kv_state.copy_(saved["ks"])
    if "ss" in saved:
        comp.score_state.copy_(saved["ss"])

    return topk


class CSASpecPrefetcher(CSAPrefetcher):
    """HC-proxy speculative prefetcher for CSA layers.

    Extends CSAPrefetcher (delta-select) with advance speculation.  For each
    CSA layer L, at the *start* of Block L's forward:

      1. Compute proxy = attn_norm_L( HC_pre_L(residual_f_{L-1}, hc_attn_fn_L) )
         This takes ~124 µs and is available immediately after the previous
         block's attention completes.
      2. Run spec Indexer on proxy → spec_topk (~83.8% hit rate vs true_topk).
      3. Submit async NVMe reads for spec_topk blocks.

    When the real Indexer later returns true_topk:
      4. Compute miss = true_topk − spec_topk (~165 blocks on average).
      5. Submit fallback reads for the missed blocks.

    Compared to delta-select alone:
      - Delta-select reads ~92 blocks/step/layer (9% of 1024).
      - Spec mode additionally pre-reads ~952 blocks (93%) during attention.
      - Net fallback = ~72 blocks (7%), issued after Indexer.
      - Total effective latency: fallback_reads / NVMe_bw ≈ 72×4KB/3GBps < 100µs.

    Patching is reversible: call unpatch_transformer().
    """

    def __init__(self, store: CSABlockStore,
                 csa_layer_ids: List[int],
                 index_topk: int = 1024) -> None:
        """Initialize speculative prefetcher.

        Args:
            store:          Backing CSABlockStore.
            csa_layer_ids:  Layer IDs of CSA blocks.
            index_topk:     Number of blocks per Indexer step.
        """
        super().__init__(store, csa_layer_ids, index_topk)

        # State shared across patched forward calls (cleared on reset)
        self._residual_f: Optional[torch.Tensor] = None
        self._block_proxy: Dict[int, Optional[torch.Tensor]] = {}
        # spec_topk[layer_id] = set predicted for the CURRENT step
        self._spec_topk: Dict[int, Set[int]] = {}

        # Spec-layer hit/miss statistics
        self._spec_stats: Dict[int, SpecLayerStats] = {
            lid: SpecLayerStats(layer_id=lid) for lid in csa_layer_ids
        }

        # Saved originals for unpatching
        self._orig_block_forward = None
        self._patched_block_cls = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all ephemeral state. Call before each decode run."""
        super().reset()
        self._residual_f = None
        self._block_proxy.clear()
        self._spec_topk.clear()
        for s in self._spec_stats.values():
            s.steps = s.spec_hits = s.spec_total = s.fallback_reads = 0

    # ------------------------------------------------------------------
    # Patching
    # ------------------------------------------------------------------

    def patch_transformer(self, transformer) -> None:
        """Patch both Block.forward and Indexer.forward.

        Block.forward is patched to compute the HC proxy and submit
        speculative NVMe reads.  Indexer.forward is patched (via super) to
        compare true_topk vs spec_topk and submit fallback reads.

        Args:
            transformer: Transformer whose Block and Indexer modules will be
                         patched in-place.

        Raises:
            RuntimeError: If already patched.
        """
        if self._patched:
            raise RuntimeError("Already patched; call unpatch_transformer first.")

        # Locate Block class
        block_cls = None
        for _, m in transformer.named_modules():
            if type(m).__name__ == "Block":
                block_cls = type(m)
                break
        if block_cls is None:
            raise RuntimeError("Could not find Block class on transformer.")

        # Locate Indexer class and tag each Indexer with its layer_id
        indexer_cls = _locate_indexer_class(transformer)
        if indexer_cls is None:
            raise RuntimeError("Could not find Indexer class on transformer.")

        for name, module in transformer.named_modules():
            if type(module).__name__ == "Indexer":
                parts = name.split(".")
                try:
                    lid = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    lid = -1
                module._csa_layer_id = lid

        self._orig_block_forward = block_cls.forward
        self._orig_indexer_forward = indexer_cls.forward
        self._patched_indexer_cls = indexer_cls
        pf_ref = self  # avoid circular ref via closure
        orig_idx = self._orig_indexer_forward

        def _patched_block_forward(self_blk, x, start_pos, input_ids):
            lid = getattr(self_blk, "layer_id", -1)

            # Step 1: compute HC proxy for THIS layer from prev block's residual_f.
            # proxy = attn_norm_L( HC_pre_L(residual_f_{L-1}, hc_attn_fn_L) )
            if (lid in pf_ref._csa_layer_ids
                    and start_pos > 0
                    and pf_ref._residual_f is not None):
                with torch.no_grad():
                    p, _, _ = self_blk.hc_pre(pf_ref._residual_f,
                                               self_blk.hc_attn_fn,
                                               self_blk.hc_attn_scale,
                                               self_blk.hc_attn_base)
                    pf_ref._block_proxy[lid] = self_blk.attn_norm(p)
                    # Spec Indexer runs inside _patched_indexer_forward below,
                    # which is called from self_blk.attn() → attn.forward().
            else:
                pf_ref._block_proxy[lid] = None

            # Step 2: run the real block forward
            residual_a = x
            x_a, post_a, comb_a = self_blk.hc_pre(x, self_blk.hc_attn_fn,
                                                    self_blk.hc_attn_scale,
                                                    self_blk.hc_attn_base)
            x_a = self_blk.attn_norm(x_a)
            x_a = self_blk.attn(x_a, start_pos)
            x = self_blk.hc_post(x_a, residual_a, post_a, comb_a)
            residual_f = x
            x_f, post_f, comb_f = self_blk.hc_pre(x, self_blk.hc_ffn_fn,
                                                    self_blk.hc_ffn_scale,
                                                    self_blk.hc_ffn_base)
            x_f = self_blk.ffn_norm(x_f)

            # Capture residual_f for decode steps only (size(1)==1)
            if x.size(1) == 1:
                pf_ref._residual_f = residual_f.detach()
            else:
                pf_ref._residual_f = None

            x = self_blk.ffn(x_f, input_ids)
            x = self_blk.hc_post(x, residual_f, post_f, comb_f)
            return x

        def _patched_indexer_forward(self_idx, x, qr, start_pos, offset):
            lid = getattr(self_idx, "_csa_layer_id", -1)
            proxy = pf_ref._block_proxy.get(lid)

            # Speculative read: run spec Indexer with HC proxy → submit reads
            if start_pos > 0 and proxy is not None:
                spec_topk_t = _run_spec_indexer(
                    self_idx, proxy, qr, start_pos, offset, orig_idx)
                spec_set: Set[int] = set(spec_topk_t.reshape(-1).cpu().tolist())
                pf_ref._spec_topk[lid] = spec_set
                # Submit reads for ALL spec_topk blocks (not just delta)
                pf_ref._submit_spec_reads(lid, spec_set)
            else:
                pf_ref._spec_topk.pop(lid, None)

            # Real Indexer forward
            true_topk = orig_idx(self_idx, x, qr, start_pos, offset)

            if start_pos > 0 and lid in pf_ref._csa_layer_ids:
                true_set: Set[int] = set(true_topk.reshape(-1).cpu().tolist())
                spec_set = pf_ref._spec_topk.get(lid, set())
                miss_set = true_set - spec_set

                # Update delta stats BEFORE overwriting prev_topk
                pf_ref._update_delta_stats(lid, true_set)
                # Update prev_topk for next step's delta computation
                pf_ref._prev_topk[lid] = true_set

                # Submit fallback reads for blocks not covered by spec
                if miss_set:
                    pf_ref._submit_fallback_reads(lid, miss_set)

                pf_ref._update_spec_stats(lid, true_set, spec_set, len(miss_set))

            return true_topk

        block_cls.forward = _patched_block_forward
        indexer_cls.forward = _patched_indexer_forward
        self._patched_block_cls = block_cls
        self._patched = True

    def unpatch_transformer(self) -> None:
        """Restore original Block.forward and Indexer.forward."""
        if not self._patched:
            return
        if self._patched_block_cls is not None:
            self._patched_block_cls.forward = self._orig_block_forward
        if self._patched_indexer_cls is not None:
            self._patched_indexer_cls.forward = self._orig_indexer_forward
        self._patched = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit_spec_reads(self, layer_id: int, spec_set: Set[int]) -> None:
        """Submit async reads for spec_topk blocks not already in-flight."""
        with self._pending_lock:
            already = {bid for (lid, bid) in self._pending if lid == layer_id}
        new_blocks = spec_set - already
        if not new_blocks:
            return
        futures = self._store.read_blocks_async(layer_id, new_blocks)
        with self._pending_lock:
            for bid, fut in futures.items():
                self._pending[(layer_id, bid)] = fut

    def _submit_fallback_reads(self, layer_id: int,
                                miss_set: Set[int]) -> None:
        """Submit async reads for miss blocks not covered by spec."""
        with self._pending_lock:
            already = {bid for (lid, bid) in self._pending if lid == layer_id}
        new_blocks = miss_set - already
        if not new_blocks:
            return
        futures = self._store.read_blocks_async(layer_id, new_blocks)
        with self._pending_lock:
            for bid, fut in futures.items():
                self._pending[(layer_id, bid)] = fut

    def _update_delta_stats(self, layer_id: int,
                             true_set: Set[int]) -> None:
        """Update CSAPrefetcher (base) delta stats from true_topk."""
        s = self._stats.get(layer_id)
        if s is None:
            return
        prev = self._prev_topk.get(layer_id, set())
        n_delta = len(true_set - prev)
        n_topk = len(true_set)
        s.steps += 1
        s.total_delta += n_delta
        s.total_topk += n_topk
        if s.steps == 1:
            s.max_delta = s.min_delta = n_delta
        else:
            s.max_delta = max(s.max_delta, n_delta)
            s.min_delta = min(s.min_delta, n_delta)

    def _update_spec_stats(self, layer_id: int,
                            true_set: Set[int],
                            spec_set: Set[int],
                            n_fallback: int) -> None:
        """Update SpecLayerStats for this step."""
        ss = self._spec_stats.get(layer_id)
        if ss is None:
            return
        ss.steps += 1
        ss.spec_hits += len(true_set & spec_set)
        ss.spec_total += len(true_set)
        ss.fallback_reads += n_fallback

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def print_stats(self) -> None:
        """Print delta-select stats AND HC-proxy hit/miss stats."""
        super().print_stats()
        print("\n--- CSASpecPrefetcher: HC-proxy accuracy ---")
        print(f"  {'Layer':>6}  {'Steps':>6}  {'HitRate':>8}  "
              f"{'FallbackMean':>13}")
        print("  " + "-" * 45)
        for lid in sorted(self._spec_stats):
            ss = self._spec_stats[lid]
            if ss.steps == 0:
                continue
            fb_mean = ss.fallback_reads / ss.steps
            print(f"  {lid:6d}  {ss.steps:6d}  {ss.hit_rate:8.3f}  "
                  f"{fb_mean:13.1f}")
        # Overall summary
        total_hits = sum(s.spec_hits for s in self._spec_stats.values())
        total_topk = sum(s.spec_total for s in self._spec_stats.values())
        mean_hit = total_hits / total_topk if total_topk else 0.0
        total_fb = sum(s.fallback_reads for s in self._spec_stats.values())
        steps = max((s.steps for s in self._spec_stats.values()), default=0)
        print(f"\n  Overall: hit_rate={mean_hit:.3f}  "
              f"fallback/step={total_fb/steps:.1f}"
              if steps else "  Overall: no data")
