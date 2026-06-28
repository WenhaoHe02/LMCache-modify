# SPDX-License-Identifier: Apache-2.0
"""Deterministic HCA prefetch manager for DeepSeek V4.

DeepSeek V4 HCA layers use ``compress_ratio == 128`` and do not run the
learned Lightning Indexer used by CSA layers. The set of compressed KV entries
needed by a decode step is deterministic from the sequence position:

``compressed_ids = range((position + 1) // 128)``.

This manager is an environment-gated prototype that wires that deterministic
plan into the same FFN-window used by CSA speculative prefetch. It does not
replace official vLLM C128A metadata or attention indices; it only preloads the
corresponding compressed KV rows back into the layer's normal vLLM KV cache
before the target HCA attention executes.
"""

from __future__ import annotations

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, PinMemoryAllocator

logger = init_logger(__name__)

_READY_LOCK = threading.Lock()
_READY_EXPECTED: set[int] = set()
_READY_SEEDED: set[int] = set()
_ACTIVE_MANAGER: Any = None


def hca_prefill_store_ready() -> bool:
    """Return whether every registered HCA layer has a seeded flat store."""
    with _READY_LOCK:
        return bool(_READY_EXPECTED) and _READY_EXPECTED.issubset(_READY_SEEDED)


def get_hca_prefetch_manager() -> Any:
    """Return the process-local HCA prefetch manager, if one is attached."""
    return _ACTIVE_MANAGER


def _register_ready_layer(layer_id: int) -> None:
    with _READY_LOCK:
        _READY_EXPECTED.add(layer_id)


def _mark_ready_layer(layer_id: int) -> None:
    with _READY_LOCK:
        _READY_SEEDED.add(layer_id)


def _env_flag(name: str) -> bool:
    """Return whether an environment variable is set to a truthy value."""
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _env_flag_default(name: str, default: bool) -> bool:
    """Return an environment flag, or ``default`` when it is unset."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable.

    Args:
        name: Environment variable name.
        default: Value to return when the variable is unset or invalid.

    Returns:
        Parsed integer or ``default``.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d", name, value, default)
        return default


def _align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to ``alignment`` bytes."""
    return ((value + alignment - 1) // alignment) * alignment


def hca_prefetch_enabled() -> bool:
    """Return whether deterministic HCA prefetch is enabled.

    Returns:
        ``True`` only when ``LMCACHE_HCA_ENABLE_PREFETCH`` is truthy.
    """
    return _env_flag("LMCACHE_HCA_ENABLE_PREFETCH")


def _pread_into(fd: int, buffer: memoryview, size: int, offset: int) -> int:
    """Read up to ``size`` bytes from ``fd`` into ``buffer`` at ``offset``."""
    view = buffer.cast("B") if buffer.format != "B" else buffer
    pos = 0
    while pos < size:
        chunk = os.pread(fd, size - pos, offset + pos)
        if not chunk:
            break
        n = len(chunk)
        view[pos : pos + n] = chunk
        pos += n
    return pos


def _pwrite(fd: int, data: bytes, offset: int) -> None:
    """Write ``data`` to ``fd`` at ``offset``."""
    view = memoryview(data)
    pos = 0
    while pos < len(data):
        pos += os.pwrite(fd, view[pos:], offset + pos)


class HCACompressedStore:
    """Flat SSD file for one HCA layer's compressed KV rows."""

    def __init__(
        self,
        store_dir: str,
        layer_id: int,
        row_bytes: int,
        max_rows: int,
    ) -> None:
        """Create or open the HCA layer store.

        Args:
            store_dir: Directory for HCA prefetch files.
            layer_id: Transformer layer id.
            row_bytes: Number of bytes in one compressed KV row.
            max_rows: Maximum number of compressed rows to address.
        """
        Path(store_dir).mkdir(parents=True, exist_ok=True)
        self._path = Path(store_dir) / f"hca_layer_{layer_id:03d}.bin"
        self._row_bytes = row_bytes
        self._fd: int | None = None
        if not self._path.exists():
            with open(self._path, "wb") as f:
                f.seek(max(0, max_rows * row_bytes - 1))
                f.write(b"\x00")

    def read_row(self, row_id: int) -> bytes:
        """Read one compressed KV row.

        Args:
            row_id: Logical compressed row id.

        Returns:
            Raw row bytes.
        """
        fd = self._open()
        buf = bytearray(self._row_bytes)
        n = _pread_into(fd, memoryview(buf), self._row_bytes, row_id * self._row_bytes)
        return bytes(buf[:n])

    def read_rows_into(
        self,
        start_row_id: int,
        num_rows: int,
        buffer: memoryview,
    ) -> int:
        """Read contiguous compressed rows into a caller-owned buffer.

        Args:
            start_row_id: First logical compressed row id.
            num_rows: Number of rows to read.
            buffer: Writable transient buffer.

        Returns:
            Number of rows fully read into ``buffer``.
        """
        if num_rows <= 0:
            return 0
        size = num_rows * self._row_bytes
        if len(buffer) < size:
            raise ValueError(
                f"HCA pinned buffer too small: need {size}, got {len(buffer)}"
            )
        fd = self._open()
        n = _pread_into(fd, buffer, size, start_row_id * self._row_bytes)
        return n // self._row_bytes

    def write_rows_contiguous(self, start_row_id: int, data: bytes) -> None:
        """Write contiguous compressed rows.

        Args:
            start_row_id: First logical compressed row id.
            data: Raw bytes whose length is a multiple of ``row_bytes``.

        Raises:
            ValueError: If ``data`` is not row-aligned.
        """
        if len(data) % self._row_bytes != 0:
            raise ValueError(
                f"Expected byte length divisible by {self._row_bytes}, "
                f"got {len(data)}"
            )
        fd = self._open()
        _pwrite(fd, data, start_row_id * self._row_bytes)

    def close(self) -> None:
        """Close the backing file descriptor."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def _open(self) -> int:
        if self._fd is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT)
        return self._fd


@dataclass
class HCALayerState:
    """Runtime state for one HCA layer."""

    layer_id: int
    layer: Any
    kv_cache: torch.Tensor
    compress_ratio: int
    block_size: int
    row_bytes: int
    store: HCACompressedStore
    initialized_rows: int = 0


@dataclass
class HCAPendingRead:
    """One transient HCA prefetch backed by a pinned I/O buffer."""

    start_row_id: int
    slot_ids: list[int]
    buffer_obj: Any | None
    future: Future[Any]
    object_layout: "HCAObjectReadLayout | HCAObjectBatchLayout | None" = None


@dataclass
class HCAMemoryRange:
    """CPU-resident HCA rows from one LMCache retrieve chunk."""

    start_row_id: int
    rows: torch.Tensor


@dataclass
class HCAObjectChunk:
    """Per-layer KV-object source for a logical row range."""

    start_row_id: int
    rows: int
    key: Any
    path: str
    record: Any
    slab_layer_index: int | None = None
    slab_num_layers: int = 0


@dataclass
class HCAObjectSource:
    """Object-store backed HCA source installed by CacheEngine.retrieve."""

    loader: Any
    loader_lock: Any | None
    chunks: list[HCAObjectChunk]


@dataclass
class HCAObjectReadLayout:
    """Object-source read layout for one pending HCA drain."""

    rows: int
    hidden_bytes: int
    k_prefix_bytes: int
    v_prefix_bytes: int
    k_dma_bytes: int
    v_dma_bytes: int
    row_major: bool = False
    row_prefix_bytes: int = 0
    row_dma_bytes: int = 0
    row_bytes: int = 0


@dataclass
class HCAObjectBatchSpan:
    """One chunk result inside a batched object-source HCA read."""

    slot_offset: int
    rows: int
    layout: HCAObjectReadLayout
    memory_index: int | None = None


@dataclass
class HCAObjectBatchLayout:
    """Drain layout for one batched object-source HCA read."""

    spans: list[HCAObjectBatchSpan]


@dataclass
class HCAObjectBatchReadSpec:
    """Tutti read metadata for one object chunk inside a batched read."""

    chunk: HCAObjectChunk
    read_ranges: tuple[Any, ...]
    nbytes: int
    rows: int


@dataclass
class HCAObjectBatchResult:
    """HBM staging objects returned by one batched object-source read."""

    rows: int
    memory_objs: list[Any | None]


class HCAPrefetchManager:
    """Environment-gated deterministic prefetch manager for DSv4 HCA layers."""

    def __init__(
        self,
        store_dir: str,
        max_seq_len: int,
        io_workers: int,
        resident_budget_blocks: int,
        prefetch_window_tokens: int,
    ) -> None:
        """Initialize the manager.

        Args:
            store_dir: Directory for HCA prefetch backing files.
            max_seq_len: Maximum logical sequence length.
            io_workers: Number of background read threads.
            resident_budget_blocks: Metadata budget for resident compressed rows.
                ``0`` means unlimited metadata residency.
            prefetch_window_tokens: Tail logical-token window to prefetch.
                ``0`` means all deterministic compressed rows for the position.
        """
        self._store_dir = store_dir
        self._max_seq_len = max_seq_len
        self._resident_budget_blocks = max(0, resident_budget_blocks)
        self._prefetch_window_tokens = max(0, prefetch_window_tokens)
        self._blocking_drain = _env_flag_default("LMCACHE_HCA_BLOCKING_DRAIN", False)
        self._async_hbm_drain = _env_flag("LMCACHE_HCA_ENABLE_ASYNC_HBM_DRAIN")
        self._gpu_ordered_drain = _env_flag("LMCACHE_HCA_GPU_ORDERED_DRAIN")
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, io_workers),
            thread_name_prefix="lmcache-hca-prefetch",
        )
        self._layers: dict[int, HCALayerState] = {}
        self._pending: dict[int, list[HCAPendingRead]] = {}
        self._drain_futures: dict[int, Future[None] | None] = {}
        self._drain_streams: dict[int, Any] = {}
        self._copy_releases: dict[int, list[tuple[Any, list[Any]]]] = {}
        self._resident_hbm: dict[int, set[int]] = {}
        self._inflight_hbm: dict[int, set[int]] = {}
        self._active_slots: dict[int, torch.Tensor] = {}
        self._active_fire_plans: dict[int, tuple[list[int], list[int]]] = {}
        self._active_fired_layers: set[int] = set()
        self._active_request_suspended = False
        self._ready_layers: set[int] = set()
        self._memory_backing: dict[int, dict[int, HCAMemoryRange]] = {}
        self._memory_backed_layers: set[int] = set()
        self._object_sources: dict[int, HCAObjectSource] = {}
        self._lock = threading.RLock()
        self._debug_fire_logged: set[int] = set()
        self._debug_drain_logged: set[int] = set()
        self._debug_seed_logged: set[int] = set()
        self._debug_skip_logged: set[tuple[int, str]] = set()
        self._debug_object_source_logged: set[int] = set()
        self._debug_object_submit_logged: set[int] = set()
        self._debug_expire_logged: set[int] = set()
        self._memory_backing_enabled = _env_flag("LMCACHE_HCA_ENABLE_MEMORY_BACKING")
        self._gpu_backing_enabled = _env_flag("LMCACHE_HCA_ENABLE_GPU_BACKING")
        self._gpu_backing_rows: dict[int, torch.Tensor] = {}
        self._gpu_backing_end: dict[int, int] = {}
        self._timing_enabled = _env_flag("LMCACHE_HCA_TIMING") or _env_flag(
            "LMCACHE_INDEXER_TIMING"
        )
        self._timing_verbose = _env_flag("LMCACHE_HCA_TIMING_VERBOSE")
        self._timing_seed_verbose = _env_flag("LMCACHE_HCA_TIMING_SEED_VERBOSE")
        self._object_source_enabled = _env_flag("LMCACHE_HCA_ENABLE_OBJECT_SOURCE")
        self._object_source_log_logged = False
        self._timing_seen = 0
        self._timing_limit = max(0, _env_int("LMCACHE_HCA_TIMING_LIMIT", 512))
        pinned_mb = max(1, _env_int("LMCACHE_HCA_PINNED_BUFFER_MB", 64))
        self._pinned_allocator = PinMemoryAllocator(pinned_mb * 1024 * 1024)
        logger.info(
            "HCAPrefetchManager: using transient pinned I/O slab size=%d MiB; "
            "this is not a CPU KV cache",
            pinned_mb,
        )
        logger.info(
            "HCAPrefetchManager: blocking_drain=%s async_hbm_drain=%s "
            "gpu_ordered_drain=%s object_source=%s gpu_backing=%s",
            self._blocking_drain,
            self._async_hbm_drain,
            self._gpu_ordered_drain,
            self._object_source_enabled,
            self._gpu_backing_enabled,
        )
        global _ACTIVE_MANAGER
        _ACTIVE_MANAGER = self

    def register_hca_layer(self, layer_id: int, layer: Any) -> None:
        """Register one HCA attention layer.

        Args:
            layer_id: Transformer layer id.
            layer: vLLM ``DeepseekV4MLAAttention``-like object with
                ``compress_ratio == 128`` and a populated ``kv_cache`` tensor.
        """
        kv_cache = getattr(layer, "kv_cache", None)
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
            return
        compress_ratio = int(getattr(layer, "compress_ratio", 1))
        if compress_ratio != 128:
            return
        if kv_cache.ndim < 3:
            logger.warning(
                "HCAPrefetchManager: skip layer %d with unexpected kv shape %s",
                layer_id,
                tuple(kv_cache.shape),
            )
            return
        block_size = int(kv_cache.shape[1])
        row_bytes = int(kv_cache[0, 0].numel() * kv_cache.element_size())
        max_rows = (self._max_seq_len + compress_ratio - 1) // compress_ratio
        state = HCALayerState(
            layer_id=layer_id,
            layer=layer,
            kv_cache=kv_cache,
            compress_ratio=compress_ratio,
            block_size=block_size,
            row_bytes=row_bytes,
            store=HCACompressedStore(
                self._store_dir,
                layer_id,
                row_bytes,
                max_rows,
            ),
        )
        self._layers[layer_id] = state
        self._pending.setdefault(layer_id, [])
        self._drain_futures.setdefault(layer_id, None)
        self._resident_hbm.setdefault(layer_id, set())
        self._inflight_hbm.setdefault(layer_id, set())
        self._memory_backing.setdefault(layer_id, {})
        _register_ready_layer(layer_id)

    def registered_layer_ids(self) -> tuple[int, ...]:
        """Return HCA transformer layer ids registered in this process."""
        return tuple(sorted(self._layers))

    def object_source_enabled(self) -> bool:
        """Return whether HCA should read rows from LMCache KV objects."""
        return self._object_source_enabled

    def set_layer_object_source(
        self,
        layer_id: int,
        chunks: Sequence[HCAObjectChunk],
        loader: Any,
        loader_lock: Any | None = None,
    ) -> None:
        """Install object-store backed rows for one active HCA layer.

        Args:
            layer_id: Transformer HCA layer id.
            chunks: Logical row ranges and object records for the current hit.
            loader: Tutti direct loader used to DMA object ranges into HBM.
            loader_lock: Optional process-local lock protecting ``loader``.
        """
        if not self._object_source_enabled or layer_id not in self._layers:
            return
        valid_chunks = [
            chunk
            for chunk in sorted(chunks, key=lambda item: item.start_row_id)
            if chunk.rows > 0 and getattr(chunk.record, "length", 0) > 0
        ]
        if not valid_chunks:
            return
        with self._lock:
            self._object_sources[layer_id] = HCAObjectSource(
                loader=loader,
                loader_lock=loader_lock,
                chunks=valid_chunks,
            )
            state = self._layers[layer_id]
            source_start = min(chunk.start_row_id for chunk in valid_chunks)
            source_rows = max(
                chunk.start_row_id + chunk.rows for chunk in valid_chunks
            )
            state.initialized_rows = max(state.initialized_rows, source_rows)
            self._resident_hbm.setdefault(layer_id, set()).clear()
            self._inflight_hbm.setdefault(layer_id, set()).clear()
            self._active_fired_layers.discard(layer_id)
            _mark_ready_layer(layer_id)
        if not self._object_source_log_logged:
            self._object_source_log_logged = True
            logger.info(
                "HCAPrefetchManager: object-source HCA reads enabled; "
                "SSD rows will be loaded from per-layer KV objects"
            )
        if layer_id not in self._debug_object_source_logged:
            self._debug_object_source_logged.add(layer_id)
            logger.info(
                "HCAPrefetchManager: object-source layer=%d chunks=%d "
                "rows=%d range=[%d,%d)",
                layer_id,
                len(valid_chunks),
                sum(chunk.rows for chunk in valid_chunks),
                source_start,
                source_rows,
            )

    def replace_object_sources(
        self,
        entries: Sequence[tuple[int, Sequence[HCAObjectChunk], Any, Any | None]],
    ) -> int:
        """Atomically replace active object-store backed rows.

        Args:
            entries: Per-layer object-source tuples of ``(layer_id, chunks,
                loader, loader_lock)`` for the current LMCache hit.

        Returns:
            Number of layers whose object source was installed.
        """
        if not self._object_source_enabled:
            return 0
        prepared: list[tuple[int, HCAObjectSource, int, int, int, int]] = []
        for layer_id, chunks, loader, loader_lock in entries:
            if layer_id not in self._layers:
                continue
            valid_chunks = [
                chunk
                for chunk in sorted(chunks, key=lambda item: item.start_row_id)
                if chunk.rows > 0 and getattr(chunk.record, "length", 0) > 0
            ]
            if not valid_chunks:
                continue
            source_start = min(chunk.start_row_id for chunk in valid_chunks)
            source_rows = max(
                chunk.start_row_id + chunk.rows for chunk in valid_chunks
            )
            prepared.append(
                (
                    layer_id,
                    HCAObjectSource(
                        loader=loader,
                        loader_lock=loader_lock,
                        chunks=valid_chunks,
                    ),
                    len(valid_chunks),
                    sum(chunk.rows for chunk in valid_chunks),
                    source_start,
                    source_rows,
                )
            )
        if not prepared:
            return 0
        with self._lock:
            self._object_sources.clear()
            for (
                layer_id,
                source,
                _chunk_count,
                _row_count,
                _source_start,
                source_rows,
            ) in prepared:
                self._object_sources[layer_id] = source
                state = self._layers[layer_id]
                state.initialized_rows = max(state.initialized_rows, source_rows)
                self._resident_hbm.setdefault(layer_id, set()).clear()
                self._inflight_hbm.setdefault(layer_id, set()).clear()
                self._active_fired_layers.discard(layer_id)
                _mark_ready_layer(layer_id)
        if not self._object_source_log_logged:
            self._object_source_log_logged = True
            logger.info(
                "HCAPrefetchManager: object-source HCA reads enabled; "
                "SSD rows will be loaded from per-layer KV objects"
            )
        for (
            layer_id,
            _source,
            chunk_count,
            row_count,
            source_start,
            source_rows,
        ) in prepared:
            if layer_id in self._debug_object_source_logged:
                continue
            self._debug_object_source_logged.add(layer_id)
            logger.info(
                "HCAPrefetchManager: object-source layer=%d chunks=%d "
                "rows=%d range=[%d,%d)",
                layer_id,
                chunk_count,
                row_count,
                source_start,
                source_rows,
            )
        return len(prepared)

    def has_object_source(self, layer_id: int) -> bool:
        """Return whether ``layer_id`` has an active object-backed source."""
        with self._lock:
            return layer_id in self._object_sources

    def layer_fired_for_active_request(self, layer_id: int) -> bool:
        """Return whether active-request HCA reads were already fired."""
        with self._lock:
            return layer_id in self._active_fired_layers

    def clear_object_sources(self) -> None:
        """Clear active object-backed sources between requests."""
        with self._lock:
            self._object_sources.clear()

    def submit_seed_after_reuse(
        self,
        layer_id: int,
        kv_cache_cpu: torch.Tensor,
        compressed_seq_len: int,
        slot_mapping_cpu: torch.Tensor,
        fire_seq_len: int | None = None,
    ) -> None:
        """Seed the HCA SSD store from a reused prefix already loaded to HBM.

        Args:
            layer_id: HCA layer id.
            kv_cache_cpu: CPU copy of the layer's vLLM HCA KV cache.
            compressed_seq_len: Number of valid compressed rows.
            slot_mapping_cpu: Physical slot id for each logical compressed row.
            fire_seq_len: Optional logical sequence length whose deterministic
                HCA rows should be prefetched immediately after the seed
                completes. This avoids racing an async seed with an immediate
                fire attempt from the caller.
        """
        state = self._layers.get(layer_id)
        if state is None or compressed_seq_len <= 0:
            return
        compressed_seq_len = min(compressed_seq_len, slot_mapping_cpu.numel())

        def _seed() -> None:
            timing = self._timing_enabled
            t0 = time.perf_counter() if timing else 0.0
            t_read0 = time.perf_counter() if timing else 0.0
            rows = self._read_rows_by_slots(
                kv_cache_cpu,
                slot_mapping_cpu[:compressed_seq_len],
                state.block_size,
            )
            read_ms = (time.perf_counter() - t_read0) * 1000.0 if timing else 0.0
            t_write0 = time.perf_counter() if timing else 0.0
            state.store.write_rows_contiguous(0, rows.contiguous().numpy().tobytes())
            write_ms = (time.perf_counter() - t_write0) * 1000.0 if timing else 0.0
            with self._lock:
                state.initialized_rows = compressed_seq_len
                # The flat SSD store is now populated, but CPU/pinned/SSD state
                # is not HBM residency. The next request may receive different
                # vLLM physical slots, so only drain_for_layer() may mark rows
                # resident for the current forward.
                self._memory_backed_layers.discard(layer_id)
                self._memory_backing[layer_id] = {}
                self._object_sources.pop(layer_id, None)
                self._resident_hbm.setdefault(layer_id, set()).clear()
                self._active_fired_layers.discard(layer_id)
                _mark_ready_layer(layer_id)
            if layer_id not in self._debug_seed_logged:
                self._debug_seed_logged.add(layer_id)
                logger.info(
                    "HCAPrefetchManager: seeded layer %d rows=%d row_bytes=%d",
                    layer_id,
                    compressed_seq_len,
                    state.row_bytes,
                )
            if timing:
                self._log_timing(
                    "seed",
                    layer_id,
                    total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
                    read_ms=f"{read_ms:.3f}",
                    write_ms=f"{write_ms:.3f}",
                    rows=compressed_seq_len,
                )
            if fire_seq_len is not None and fire_seq_len > 0:
                self.fire_for_seq_len(layer_id, fire_seq_len)

        self._executor.submit(_seed)

    def seed_range_from_lmcache_group(
        self,
        layer_ids: Sequence[int],
        memory_tensor: torch.Tensor,
        start: int,
        end: int,
    ) -> int:
        """Seed HCA flat stores directly from an LMCache retrieve chunk.

        Args:
            layer_ids: Transformer HCA layer ids in the same order as the
                LMCache HCA group layer axis.
            memory_tensor: LMCache HCA group tensor with layout
                ``[kv_size, num_layers, physical_rows, hidden_dim]``.
            start: Logical token start offset of this LMCache chunk.
            end: Logical token end offset of this LMCache chunk.

        Returns:
            Number of HCA layers whose flat store was updated.

        Raises:
            ValueError: If the tensor shape or token range is incompatible with
                the registered HCA layers.
        """
        if not layer_ids or memory_tensor.numel() == 0 or end <= start:
            return 0
        if memory_tensor.ndim != 4:
            raise ValueError(
                "HCA LMCache group tensor must have shape "
                "[kv_size, num_layers, rows, hidden_dim], got "
                f"{tuple(memory_tensor.shape)}"
            )
        first_state = self._layers.get(int(layer_ids[0]))
        if first_state is None:
            return 0
        compress_ratio = first_state.compress_ratio
        if start % compress_ratio != 0 or end % compress_ratio != 0:
            raise ValueError(
                f"HCA LMCache range [{start}, {end}) is not aligned to "
                f"compress_ratio {compress_ratio}"
            )
        start_row = start // compress_ratio
        expected_rows = (end - start) // compress_ratio
        rows_in_tensor = int(memory_tensor.shape[2])
        rows_to_seed = min(expected_rows, rows_in_tensor)
        if rows_to_seed <= 0:
            return 0
        if len(layer_ids) > int(memory_tensor.shape[1]):
            raise ValueError(
                f"HCA LMCache group has {memory_tensor.shape[1]} layers but "
                f"{len(layer_ids)} layer ids were provided"
            )

        timing = self._timing_enabled
        t0 = time.perf_counter() if timing else 0.0
        gpu_backed = (
            self._gpu_backing_enabled
            and memory_tensor.is_cuda
        )
        if gpu_backed:
            tensor_gpu = memory_tensor.detach().contiguous()
            cpu_ms = 0.0
            contiguous_ms = 0.0
            tensor_cpu = None
        else:
            t_cpu0 = time.perf_counter() if timing else 0.0
            tensor_cpu = memory_tensor.detach()
            if tensor_cpu.device.type != "cpu":
                tensor_cpu = tensor_cpu.to(device="cpu", non_blocking=False)
            cpu_ms = (time.perf_counter() - t_cpu0) * 1000.0 if timing else 0.0
            t_contig0 = time.perf_counter() if timing else 0.0
            tensor_cpu = tensor_cpu.contiguous()
            contiguous_ms = (
                (time.perf_counter() - t_contig0) * 1000.0 if timing else 0.0
            )

        seeded = 0
        reshape_ms = 0.0
        write_ms = 0.0
        state_ms = 0.0
        memory_backed = self._memory_backing_enabled
        for group_layer_idx, layer_id in enumerate(layer_ids):
            state = self._layers.get(int(layer_id))
            if state is None:
                continue
            if state.compress_ratio != compress_ratio:
                raise ValueError(
                    "All HCA layers in one LMCache group must share "
                    f"compress_ratio {compress_ratio}; layer {layer_id} has "
                    f"{state.compress_ratio}"
                )
            t_reshape0 = time.perf_counter() if timing else 0.0
            if gpu_backed:
                source_tensor = tensor_gpu
            else:
                source_tensor = tensor_cpu
            layer_rows = source_tensor[
                :,
                group_layer_idx,
                :rows_to_seed,
                :,
            ].permute(1, 0, 2)
            flat_rows = layer_rows.contiguous().view(rows_to_seed, -1)
            if timing:
                reshape_ms += (time.perf_counter() - t_reshape0) * 1000.0
            row_bytes = flat_rows.shape[1] * flat_rows.element_size()
            if row_bytes != state.row_bytes:
                raise ValueError(
                    f"HCA layer {layer_id} row size mismatch: LMCache row "
                    f"has {row_bytes} bytes, registered HCA row has "
                    f"{state.row_bytes} bytes"
                )
            t_write0 = time.perf_counter() if timing else 0.0
            if gpu_backed:
                flat_rows_u8 = flat_rows.view(torch.uint8).reshape(
                    rows_to_seed, state.row_bytes
                )
                self._ensure_gpu_backing(
                    state,
                    flat_rows_u8.device,
                    start_row + rows_to_seed,
                )
                gpu_buf = self._gpu_backing_rows[int(layer_id)]
                gpu_buf[start_row : start_row + rows_to_seed].copy_(
                    flat_rows_u8, non_blocking=True
                )
                with self._lock:
                    self._gpu_backing_end[int(layer_id)] = max(
                        self._gpu_backing_end.get(int(layer_id), 0),
                        start_row + rows_to_seed,
                    )
            elif memory_backed:
                with self._lock:
                    if start_row == 0:
                        self._memory_backing[int(layer_id)] = {}
                    self._memory_backing[int(layer_id)][start_row] = HCAMemoryRange(
                        start_row_id=start_row,
                        rows=flat_rows.clone(),
                    )
                    self._memory_backed_layers.add(int(layer_id))
            else:
                state.store.write_rows_contiguous(
                    start_row,
                    flat_rows.numpy().tobytes(),
                )
            if timing:
                write_ms += (time.perf_counter() - t_write0) * 1000.0
            t_state0 = time.perf_counter() if timing else 0.0
            with self._lock:
                state.initialized_rows = max(
                    state.initialized_rows,
                    start_row + rows_to_seed,
                )
                self._object_sources.pop(layer_id, None)
                self._resident_hbm.setdefault(layer_id, set()).clear()
                self._active_fired_layers.discard(layer_id)
                _mark_ready_layer(layer_id)
            if timing:
                state_ms += (time.perf_counter() - t_state0) * 1000.0
            seeded += 1

        if seeded and timing:
            self._log_timing(
                "seed_lmcache",
                -1,
                total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
                cpu_ms=f"{cpu_ms:.3f}",
                contiguous_ms=f"{contiguous_ms:.3f}",
                reshape_ms=f"{reshape_ms:.3f}",
                write_ms=f"{write_ms:.3f}",
                state_ms=f"{state_ms:.3f}",
                layers=seeded,
                start_row=start_row,
                rows=rows_to_seed,
                backing="memory" if memory_backed else "ssd",
            )
        return seeded

    def has_seeded_rows(self, layer_id: int, compressed_seq_len: int) -> bool:
        """Return whether ``layer_id`` has enough HCA rows in its flat store."""
        state = self._layers.get(layer_id)
        return state is not None and state.initialized_rows >= compressed_seq_len

    def begin_active_request(self) -> None:
        """Reset current-request HCA prefetch plans before slot maps are installed."""
        self._release_completed_copies(blocking=False)
        with self._lock:
            self._active_slots.clear()
            self._active_fire_plans.clear()
            self._active_fired_layers.clear()
            self._active_request_suspended = False

    def suspend_active_request(self) -> None:
        """Disable HCA fire/drain work for the current active request."""
        self._release_completed_copies(blocking=False)
        with self._lock:
            self._active_slots.clear()
            self._active_fire_plans.clear()
            self._active_fired_layers.clear()
            self._ready_layers.clear()
            self._pending.clear()
            self._active_request_suspended = True

    def active_request_suspended(self) -> bool:
        """Return whether current-request HCA fire/drain work is suspended."""
        with self._lock:
            return self._active_request_suspended

    def set_active_request_slots(
        self,
        layer_id: int,
        compressed_seq_len: int,
        slot_mapping_cpu: torch.Tensor,
    ) -> None:
        """Record current-request HCA row -> physical slot mapping.

        Args:
            layer_id: HCA layer id.
            compressed_seq_len: Number of valid compressed rows.
            slot_mapping_cpu: CPU tensor whose index is logical compressed row
                id and value is the current request's vLLM physical slot.
        """
        state = self._layers.get(layer_id)
        if state is None or compressed_seq_len <= 0:
            return
        slots = slot_mapping_cpu[:compressed_seq_len].to(device="cpu", dtype=torch.long)
        self._active_slots[layer_id] = slots
        row_ids = self._deterministic_row_ids_from_compressed_rows(
            min(compressed_seq_len, state.initialized_rows),
            state,
        )
        slot_ids: list[int] = []
        if row_ids:
            selected = slots[row_ids[0] : row_ids[-1] + 1]
            if not bool(torch.any(selected < 0).item()):
                slot_ids = [int(slot_id) for slot_id in selected.tolist()]
            else:
                row_ids = []
        self._active_fire_plans[layer_id] = (row_ids, slot_ids)
        with self._lock:
            if layer_id not in self._active_fired_layers:
                self._resident_hbm.setdefault(layer_id, set()).clear()
                self._inflight_hbm.setdefault(layer_id, set()).clear()

    def fire_active_request_layers(self) -> int:
        """Fire all current-request HCA plans as soon as slots are known.

        Returns:
            Number of layers with a non-empty plan submitted or already covered.
        """
        if self.active_request_suspended():
            return 0
        prefire_limit = max(0, _env_int("LMCACHE_HCA_ACTIVE_PREFIRE_LIMIT", 0))
        layer_ids = []
        for layer_id in sorted(self._active_fire_plans):
            if prefire_limit > 0 and len(layer_ids) >= prefire_limit:
                break
            row_ids, slot_ids = self._active_fire_plans[layer_id]
            if row_ids and slot_ids:
                layer_ids.append(layer_id)
        return self.fire_async_for_layers(layer_ids, positions=None)

    def prefire_first_hca(self, seq_len: int) -> None:
        """Optionally fire the first HCA layer before forward for diagnostics."""
        if not _env_flag("LMCACHE_HCA_ALLOW_PREFORWARD_FALLBACK"):
            self._log_skip_once(
                -1,
                "prefire_disabled",
                "HCAPrefetchManager: first-layer pre-forward fire is disabled; "
                "full-hit HCA prefetch fires after current-request slot "
                "mapping is ready",
            )
            return
        if not self._layers:
            return
        first_layer_id = min(self._layers)
        self.fire_for_seq_len(first_layer_id, seq_len)

    def fire_for_seq_len(self, layer_id: int, seq_len: int) -> None:
        """Fire deterministic HCA reads for a prefix length."""
        if self.active_request_suspended():
            return
        state = self._layers.get(layer_id)
        if state is None or seq_len <= 0 or state.initialized_rows <= 0:
            return
        row_ids = self._deterministic_row_ids_from_seq_len(seq_len, state)
        self._fire_rows(layer_id, state, row_ids)

    def fire_async_for_layer(
        self,
        layer_id: int,
        positions: torch.Tensor | None,
    ) -> None:
        """Fire deterministic HCA reads for ``layer_id``.

        Args:
            layer_id: Target HCA layer.
            positions: Current vLLM positions tensor. The last position derives
                the deterministic compressed row range. This works for both
                prefill-hit chunks and one-token decode.
        """
        if self.active_request_suspended():
            return
        state = self._layers.get(layer_id)
        if state is None:
            return
        if state.initialized_rows <= 0:
            return
        with self._lock:
            if layer_id in self._active_fired_layers:
                return
        timing = self._timing_enabled
        t0 = time.perf_counter() if timing else None
        active_plan = self._active_fire_plans.get(layer_id)
        if active_plan is not None:
            row_ids, slot_ids = active_plan
            with self._lock:
                self._active_fired_layers.add(layer_id)
            fired = self._fire_rows(
                layer_id,
                state,
                row_ids,
                start_time=t0,
                precomputed_slot_ids=slot_ids,
            )
            if not fired:
                with self._lock:
                    self._active_fired_layers.discard(layer_id)
            return
        if positions is None or positions.numel() == 0:
            return
        if positions.reshape(-1).numel() == 1 and not _env_flag(
            "LMCACHE_HCA_ENABLE_DECODE_HOOK"
        ):
            return
        row_ids = self._deterministic_row_ids(positions, state)
        with self._lock:
            self._active_fired_layers.add(layer_id)
        fired = self._fire_rows(layer_id, state, row_ids, start_time=t0)
        if not fired:
            with self._lock:
                self._active_fired_layers.discard(layer_id)

    def fire_async_for_layers(
        self,
        layer_ids: Sequence[int],
        positions: torch.Tensor | None,
    ) -> int:
        """Fire HCA reads for several layers in one overlap window.

        Args:
            layer_ids: Target HCA layer ids for the current FFN window.
            positions: Current vLLM positions tensor.

        Returns:
            Number of layers that were submitted or already covered.
        """
        if self.active_request_suspended():
            return 0
        ordered_layer_ids = tuple(dict.fromkeys(int(v) for v in layer_ids))
        if not ordered_layer_ids:
            return 0
        if not self._object_source_enabled or len(ordered_layer_ids) == 1:
            fired = 0
            for layer_id in ordered_layer_ids:
                before = self.layer_fired_for_active_request(layer_id)
                self.fire_async_for_layer(layer_id, positions)
                if before or self.layer_fired_for_active_request(layer_id):
                    fired += 1
            return fired

        entries: list[
            tuple[int, HCALayerState, list[int], list[tuple[int, int]]]
        ] = []
        covered_layer_ids: set[int] = set()
        for layer_id in ordered_layer_ids:
            with self._lock:
                if layer_id in self._active_fired_layers:
                    covered_layer_ids.add(layer_id)
                    continue
            state = self._layers.get(layer_id)
            if state is None or state.initialized_rows <= 0:
                continue
            active_plan = self._active_fire_plans.get(layer_id)
            if active_plan is not None:
                row_ids, slot_ids = active_plan
            else:
                if positions is None or positions.numel() == 0:
                    continue
                if positions.reshape(-1).numel() == 1 and not _env_flag(
                    "LMCACHE_HCA_ENABLE_DECODE_HOOK"
                ):
                    continue
                row_ids = self._deterministic_row_ids(positions, state)
                slot_ids = self._global_slots_for_rows(layer_id, row_ids, state)
                if slot_ids is None:
                    continue
            missing = self._missing_rows_for_fire(
                layer_id,
                state,
                row_ids,
                precomputed_slot_ids=slot_ids,
            )
            if missing is None:
                continue
            missing_rows, _slots_ms, _filter_ms = missing
            covered_layer_ids.add(layer_id)
            if missing_rows:
                entries.append((layer_id, state, row_ids, missing_rows))

        if covered_layer_ids:
            with self._lock:
                self._active_fired_layers.update(covered_layer_ids)

        submitted = self._submit_object_multi_layer_reads(entries)
        fallback_layers = {
            layer_id
            for layer_id, pending_reads in submitted.items()
            if not pending_reads
        }
        for layer_id, state, row_ids, missing in entries:
            pending_reads = submitted.get(layer_id, [])
            if pending_reads:
                with self._lock:
                    inflight = self._inflight_hbm.setdefault(layer_id, set())
                    for pending_read in pending_reads:
                        inflight.update(
                            range(
                                pending_read.start_row_id,
                                pending_read.start_row_id
                                + len(pending_read.slot_ids),
                            )
                        )
                    self._pending.setdefault(layer_id, []).extend(pending_reads)
                    self._active_fired_layers.add(layer_id)
                self._attach_ready_callbacks(layer_id, pending_reads)
                continue
            if layer_id in fallback_layers:
                fired = self._fire_rows(layer_id, state, row_ids)
                if fired:
                    with self._lock:
                        self._active_fired_layers.add(layer_id)

        for layer_id in covered_layer_ids:
            if layer_id not in fallback_layers:
                with self._lock:
                    self._active_fired_layers.add(layer_id)
        return len(covered_layer_ids)

    def _fire_rows(
        self,
        layer_id: int,
        state: HCALayerState,
        row_ids: list[int],
        start_time: float | None = None,
        precomputed_slot_ids: list[int] | None = None,
    ) -> bool:
        if not row_ids:
            return False
        timing = self._timing_enabled
        t0 = time.perf_counter() if timing and start_time is None else start_time
        missing_result = self._missing_rows_for_fire(
            layer_id,
            state,
            row_ids,
            precomputed_slot_ids=precomputed_slot_ids,
            timing=timing,
        )
        if missing_result is None:
            return False
        missing, slots_ms, filter_ms = missing_result
        if not missing:
            if timing and self._timing_verbose and t0 is not None:
                self._log_timing(
                    "fire",
                    layer_id,
                    total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
                    slots_ms="0.000",
                    filter_ms=f"{filter_ms:.3f}",
                    submit_ms="0.000",
                    rows=len(row_ids),
                    missing=0,
                    pending=0,
                )
            return True

        if self._gpu_backing_enabled:
            t_gpu0 = time.perf_counter() if timing else 0.0
            if self._fire_via_gpu_backing(layer_id, state, missing):
                if timing and t0 is not None:
                    self._log_timing(
                        "fire",
                        layer_id,
                        total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
                        slots_ms=f"{slots_ms:.3f}",
                        filter_ms=f"{filter_ms:.3f}",
                        submit_ms=f"{(time.perf_counter() - t_gpu0) * 1000.0:.3f}",
                        rows=len(row_ids),
                        missing=len(missing),
                        pending=0,
                        mode="gpu_backing",
                    )
                return True

        t_submit0 = time.perf_counter() if timing else 0.0
        pending_reads = self._build_pending_reads(state, missing)
        with self._lock:
            inflight = self._inflight_hbm.setdefault(layer_id, set())
            for pending_read in pending_reads:
                inflight.update(
                    range(
                        pending_read.start_row_id,
                        pending_read.start_row_id + len(pending_read.slot_ids),
                    )
                )
            self._pending.setdefault(layer_id, []).extend(pending_reads)
        self._attach_ready_callbacks(layer_id, pending_reads)
        submit_ms = (time.perf_counter() - t_submit0) * 1000.0 if timing else 0.0
        if missing and layer_id not in self._debug_fire_logged:
            self._debug_fire_logged.add(layer_id)
            logger.debug(
                "HCAPrefetchManager: fire layer=%d rows=%d missing=%d "
                "mode=pinned_transient",
                layer_id,
                len(row_ids),
                len(missing),
            )
        if timing and t0 is not None and (missing or self._timing_verbose):
            self._log_timing(
                "fire",
                layer_id,
                total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
                slots_ms=f"{slots_ms:.3f}",
                filter_ms=f"{filter_ms:.3f}",
                submit_ms=f"{submit_ms:.3f}",
                rows=len(row_ids),
                missing=len(missing),
                pending=len(pending_reads),
            )
        return True

    def _missing_rows_for_fire(
        self,
        layer_id: int,
        state: HCALayerState,
        row_ids: list[int],
        *,
        precomputed_slot_ids: list[int] | None = None,
        timing: bool = False,
    ) -> tuple[list[tuple[int, int]], float, float] | None:
        """Return row/slot pairs that still need an HCA load."""
        if not row_ids:
            return ([], 0.0, 0.0)
        t_filter0 = time.perf_counter() if timing else 0.0
        with self._lock:
            resident = self._resident_hbm[layer_id]
            inflight = self._inflight_hbm[layer_id]
            missing_row_ids = [
                row_id
                for row_id in row_ids
                if row_id not in resident and row_id not in inflight
            ]
        filter_ms = (time.perf_counter() - t_filter0) * 1000.0 if timing else 0.0
        if not missing_row_ids:
            return ([], 0.0, filter_ms)

        if precomputed_slot_ids is not None:
            if len(precomputed_slot_ids) != len(row_ids):
                return None
            row_start = row_ids[0]
            slot_ids = [
                precomputed_slot_ids[row_id - row_start]
                for row_id in missing_row_ids
            ]
            slots_ms = 0.0
        else:
            t_slots0 = time.perf_counter() if timing else 0.0
            slot_ids = self._global_slots_for_rows(layer_id, missing_row_ids, state)
            if slot_ids is None:
                return None
            slots_ms = (time.perf_counter() - t_slots0) * 1000.0 if timing else 0.0
        with self._lock:
            resident = self._resident_hbm[layer_id]
            inflight = self._inflight_hbm[layer_id]
            missing = [
                (row_id, slot_id)
                for row_id, slot_id in zip(
                    missing_row_ids,
                    slot_ids,
                    strict=False,
                )
                if row_id not in resident and row_id not in inflight
            ]
        return (missing, slots_ms, filter_ms)

    def prepare_layer_async(self, layer_id: int) -> None:
        """Enqueue ready HCA rows for copy-back without waiting.

        Args:
            layer_id: Target HCA layer whose pending reads should be drained
                from the caller's CUDA context when their I/O futures are ready.
        """
        if self.active_request_suspended():
            return
        if not self._async_hbm_drain:
            return
        if layer_id not in self._layers:
            return
        self._release_completed_copies(layer_id, blocking=False)
        if not self._has_ready_pending(layer_id):
            with self._lock:
                self._ready_layers.discard(layer_id)
            return
        self._drain_for_layer_once(
            layer_id,
            blocking=False,
            synchronize_copy=False,
        )

    def prepare_layers_async(self, layer_ids: Sequence[int]) -> None:
        """Enqueue ready HCA rows for several layers without waiting.

        Args:
            layer_ids: Target HCA layer ids whose pending reads should be
                drained from the caller's CUDA context when their I/O futures
                are already complete.
        """
        if self.active_request_suspended():
            return
        if not self._async_hbm_drain:
            return
        if not layer_ids:
            return
        self._release_completed_copies(blocking=False)
        ready_layer_ids: list[int] = []
        seen: set[int] = set()
        for layer_id in layer_ids:
            if layer_id in seen or layer_id not in self._layers:
                continue
            seen.add(layer_id)
            if self._has_ready_pending(layer_id):
                ready_layer_ids.append(layer_id)
        for layer_id in ready_layer_ids:
            self._drain_for_layer_once(
                layer_id,
                blocking=False,
                synchronize_copy=False,
            )

    def prepare_active_request_layers(self, limit: int = 0) -> int:
        """Enqueue ready copy-back work for layers already fired in this request.

        Args:
            limit: Maximum number of fired layers to inspect. ``0`` means all
                fired layers.

        Returns:
            Number of fired layers considered for opportunistic copy-back.
        """
        if self.active_request_suspended():
            return 0
        if not self._async_hbm_drain:
            return 0
        with self._lock:
            layer_ids = sorted(self._active_fired_layers)
        if limit > 0:
            layer_ids = layer_ids[:limit]
        if not layer_ids:
            return 0
        self.prepare_layers_async(layer_ids)
        return len(layer_ids)

    def prepare_ready_layers(self, limit: int = 0) -> int:
        """Enqueue copy-back for HCA layers whose read futures completed.

        I/O worker callbacks only mark layer IDs as ready; this method is
        called from the model thread and is the only place that launches CUDA
        scatter work for those completions.

        Args:
            limit: Maximum number of ready layers to drain. ``0`` means all.

        Returns:
            Number of ready layers considered for copy-back.
        """
        if self.active_request_suspended():
            return 0
        if not self._async_hbm_drain:
            return 0
        with self._lock:
            layer_ids = sorted(self._ready_layers)
            if limit > 0:
                layer_ids = layer_ids[:limit]
            for layer_id in layer_ids:
                self._ready_layers.discard(layer_id)
        if not layer_ids:
            return 0
        self.prepare_layers_async(layer_ids)
        return len(layer_ids)

    def drain_for_layer(self, layer_id: int, blocking: bool | None = None) -> None:
        """Drain pending HCA reads into the target layer's vLLM KV cache.

        Args:
            layer_id: Target HCA layer id.
            blocking: Override for whether this drain should wait for pending
                reads. ``None`` uses the manager's configured default.
        """
        if self.active_request_suspended():
            return
        with self._lock:
            drain_future = self._drain_futures.get(layer_id)
        if drain_future is not None:
            drain_future.result()
            with self._lock:
                if self._drain_futures.get(layer_id) is drain_future:
                    self._drain_futures[layer_id] = None
        effective_blocking = self._blocking_drain if blocking is None else blocking
        gpu_ordered = self._async_hbm_drain and self._gpu_ordered_drain
        self._release_completed_copies(layer_id, blocking=False)
        with self._lock:
            has_pending = bool(self._pending.get(layer_id))
            has_copy_work = bool(self._copy_releases.get(layer_id))
        if not effective_blocking and not has_pending and not has_copy_work:
            return
        self._drain_for_layer_once(
            layer_id,
            blocking=effective_blocking,
            synchronize_copy=not gpu_ordered,
        )
        if not effective_blocking:
            self._expire_pending_reads_for_layer(layer_id)
        if gpu_ordered:
            self._order_pending_copies_on_current_stream(layer_id)
            self._release_completed_copies(layer_id, blocking=False)
        else:
            self._release_completed_copies(layer_id, blocking=effective_blocking)

    def _drain_for_layer_once(
        self,
        layer_id: int,
        blocking: bool | None = None,
        *,
        synchronize_copy: bool = True,
    ) -> None:
        """Drain currently ready HCA reads without waiting for future drains."""
        state = self._layers.get(layer_id)
        if state is None:
            return
        wait_for_pending = self._blocking_drain if blocking is None else blocking
        timing = self._timing_enabled
        t0 = time.perf_counter() if timing else 0.0
        t_select0 = time.perf_counter() if timing else 0.0
        with self._lock:
            pending = self._pending.get(layer_id, [])
            if not wait_for_pending:
                ready = [
                    pending_read
                    for pending_read in pending
                    if pending_read.future.done()
                ]
                self._pending[layer_id] = [
                    pending_read
                    for pending_read in pending
                    if not pending_read.future.done()
                ]
                pending = ready
            else:
                self._pending[layer_id] = []
        select_ms = (time.perf_counter() - t_select0) * 1000.0 if timing else 0.0
        wait_ms = 0.0
        write_ms = 0.0
        written = 0
        copy_stream = self._drain_stream_for_state(state)
        copy_device_context = nullcontext()
        if copy_stream is not None and state.kv_cache.device.index is not None:
            copy_device_context = torch.cuda.device(
                int(state.kv_cache.device.index)
            )
        copy_submitted = False
        release_objs: list[Any] = []
        for pending_read in pending:
            t_wait0 = time.perf_counter() if timing else 0.0
            row_span = range(
                pending_read.start_row_id,
                pending_read.start_row_id + len(pending_read.slot_ids),
            )
            buffer_obj = pending_read.buffer_obj
            try:
                read_result = pending_read.future.result()
                if isinstance(read_result, HCAObjectBatchResult):
                    rows_read = read_result.rows
                    buffer_obj = read_result
                elif isinstance(read_result, tuple):
                    rows_read = int(read_result[0])
                    buffer_obj = read_result[1]
                else:
                    rows_read = int(read_result)
                if timing:
                    wait_ms += (time.perf_counter() - t_wait0) * 1000.0
                t_write0 = time.perf_counter() if timing else 0.0
                if buffer_obj is None:
                    rows_written = 0
                else:
                    context = (
                        torch.cuda.stream(copy_stream)
                        if copy_stream is not None
                        else nullcontext()
                    )
                    with copy_device_context:
                        with context:
                            rows_written = self._write_pending_to_kv_cache(
                                state,
                                pending_read.slot_ids,
                                buffer_obj,
                                rows_read,
                                object_layout=pending_read.object_layout,
                            )
                    copy_submitted = copy_submitted or rows_written > 0
                if timing:
                    write_ms += (time.perf_counter() - t_write0) * 1000.0
                if rows_written > 0:
                    with self._lock:
                        resident = self._resident_hbm[layer_id]
                        resident.update(
                            range(
                                pending_read.start_row_id,
                                pending_read.start_row_id + rows_written,
                            )
                        )
                        if self._resident_budget_blocks > 0 and len(resident) > (
                            self._resident_budget_blocks * 2
                        ):
                            keep_start = max(
                                0,
                                pending_read.start_row_id
                                + rows_written
                                - self._resident_budget_blocks,
                            )
                            self._resident_hbm[layer_id] = {
                                rid for rid in resident if rid >= keep_start
                            }
                written += rows_written
            finally:
                with self._lock:
                    self._inflight_hbm[layer_id].difference_update(row_span)
                if buffer_obj is not None:
                    release_objs.append(buffer_obj)
        if copy_stream is not None and copy_submitted:
            if synchronize_copy:
                copy_stream.synchronize()
            else:
                self._defer_buffer_releases(layer_id, copy_stream, release_objs)
                release_objs = []
        for obj in release_objs:
            self._release_buffer_obj(obj)
        if written and layer_id not in self._debug_drain_logged:
            self._debug_drain_logged.add(layer_id)
            logger.debug(
                "HCAPrefetchManager: drain layer=%d written=%d "
                "mode=pinned_transient",
                layer_id,
                written,
            )
        if timing and pending:
            self._log_timing(
                "drain",
                layer_id,
                total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
                select_ms=f"{select_ms:.3f}",
                wait_ms=f"{wait_ms:.3f}",
                write_ms=f"{write_ms:.3f}",
                pending=len(pending),
                written=written,
            )

    def close(self) -> None:
        """Close files and shut down the background executor."""
        self._executor.shutdown(wait=True)
        self._release_completed_copies(blocking=True)
        self._pinned_allocator.memcheck()
        for state in self._layers.values():
            state.store.close()

    def _has_ready_pending(self, layer_id: int) -> bool:
        """Return whether ``layer_id`` has completed reads waiting to drain."""
        with self._lock:
            return any(
                pending_read.future.done()
                for pending_read in self._pending.get(layer_id, [])
            )

    def _expire_pending_reads_for_layer(self, layer_id: int) -> None:
        """Drop HCA reads that missed their attention deadline."""
        with self._lock:
            expired = self._pending.pop(layer_id, [])
        if not expired:
            return
        expired_rows = 0
        release_objs: list[Any] = []
        for pending_read in expired:
            row_span = range(
                pending_read.start_row_id,
                pending_read.start_row_id + len(pending_read.slot_ids),
            )
            expired_rows += len(pending_read.slot_ids)
            if pending_read.future.done():
                buffer_obj = self._buffer_obj_from_pending_result(pending_read)
                if buffer_obj is not None:
                    release_objs.append(buffer_obj)
            else:
                pending_read.future.add_done_callback(
                    self._release_expired_pending_callback(pending_read)
                )
            with self._lock:
                self._inflight_hbm[layer_id].difference_update(row_span)
        for obj in release_objs:
            self._release_buffer_obj(obj)
        if layer_id not in self._debug_expire_logged:
            self._debug_expire_logged.add(layer_id)
            logger.info(
                "HCAPrefetchManager: expired %d pending rows for layer=%d "
                "after nonblocking attention deadline",
                expired_rows,
                layer_id,
            )

    def _release_expired_pending_callback(
        self,
        pending_read: HCAPendingRead,
    ) -> Callable[[Future[Any]], None]:
        """Return a callback that releases a late HCA read after expiration."""

        def _release_expired(_future: Future[Any]) -> None:
            buffer_obj = self._buffer_obj_from_pending_result(pending_read)
            if buffer_obj is not None:
                self._release_buffer_obj(buffer_obj)

        return _release_expired

    def _attach_ready_callbacks(
        self,
        layer_id: int,
        pending_reads: Sequence[HCAPendingRead],
    ) -> None:
        """Mark ``layer_id`` ready when pending HCA I/O futures complete."""
        if not pending_reads:
            return
        for pending_read in pending_reads:
            pending_read.future.add_done_callback(
                lambda _future, ready_layer_id=layer_id: self._mark_layer_ready(
                    ready_layer_id
                )
            )

    def _mark_layer_ready(self, layer_id: int) -> None:
        """Record that a layer has completed reads awaiting model-thread drain."""
        with self._lock:
            if layer_id in self._layers and self._pending.get(layer_id):
                self._ready_layers.add(layer_id)

    @staticmethod
    def _buffer_obj_from_pending_result(pending_read: HCAPendingRead) -> Any | None:
        """Return the ref-counted buffer object held by a completed pending read."""
        try:
            read_result = pending_read.future.result()
        except Exception:
            return pending_read.buffer_obj
        if isinstance(read_result, HCAObjectBatchResult):
            return read_result
        if isinstance(read_result, tuple):
            return read_result[1]
        return pending_read.buffer_obj

    def _drain_stream_for_state(self, state: HCALayerState) -> Any | None:
        """Return the CUDA stream used for HCA KV copy-back, if applicable."""
        if not state.kv_cache.is_cuda or not torch.cuda.is_available():
            return None
        device = state.kv_cache.device
        device_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        with self._lock:
            stream = self._drain_streams.get(device_index)
            if stream is None:
                with torch.cuda.device(device_index):
                    stream = torch.cuda.Stream()
                self._drain_streams[device_index] = stream
            return stream

    def _defer_buffer_releases(
        self,
        layer_id: int,
        copy_stream: Any,
        release_objs: list[Any],
    ) -> None:
        """Release HCA buffers after the copy stream reaches a recorded event."""
        if not release_objs:
            return
        event = torch.cuda.Event()
        event.record(copy_stream)
        with self._lock:
            self._copy_releases.setdefault(layer_id, []).append(
                (event, release_objs)
            )

    def _release_completed_copies(
        self,
        layer_id: int | None = None,
        *,
        blocking: bool,
    ) -> None:
        """Release buffers whose asynchronous copy-back has completed."""
        with self._lock:
            if layer_id is None:
                items = list(self._copy_releases.items())
                self._copy_releases.clear()
            else:
                items = [(layer_id, self._copy_releases.pop(layer_id, []))]
        keep_by_layer: dict[int, list[tuple[Any, list[Any]]]] = {}
        for current_layer_id, pending in items:
            for event, release_objs in pending:
                if blocking:
                    event.synchronize()
                    ready = True
                else:
                    ready = event.query()
                if ready:
                    for obj in release_objs:
                        self._release_buffer_obj(obj)
                else:
                    keep_by_layer.setdefault(current_layer_id, []).append(
                        (event, release_objs)
                    )
        if keep_by_layer:
            with self._lock:
                for current_layer_id, pending in keep_by_layer.items():
                    self._copy_releases.setdefault(current_layer_id, []).extend(
                        pending
                    )

    def _order_pending_copies_on_current_stream(self, layer_id: int) -> bool:
        """Order pending HCA copy-back before the current CUDA stream.

        GPU-ordered drain must not release staging buffers immediately after
        ``current_stream.wait_event(copy_event)``. The wait only enqueues a GPU
        dependency; CPU execution continues. This method therefore replaces
        each copy event with a release event recorded on the current stream
        after the wait, letting non-blocking release polling retire buffers once
        attention has passed the dependency.
        """
        state = self._layers.get(layer_id)
        if state is None or not state.kv_cache.is_cuda or not torch.cuda.is_available():
            return False
        with self._lock:
            pending = self._copy_releases.pop(layer_id, [])
        if not pending:
            return False
        device = state.kv_cache.device
        device_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        ordered: list[tuple[Any, list[Any]]] = []
        with torch.cuda.device(device_index):
            current_stream = torch.cuda.current_stream()
            for event, release_objs in pending:
                current_stream.wait_event(event)
                release_event = torch.cuda.Event()
                release_event.record(current_stream)
                ordered.append((release_event, release_objs))
        with self._lock:
            self._copy_releases.setdefault(layer_id, []).extend(ordered)
        return True

    @staticmethod
    def _release_buffer_obj(buffer_obj: Any) -> None:
        """Release a flat or object-batch HCA drain buffer."""
        if isinstance(buffer_obj, HCAObjectBatchResult):
            for memory_obj in buffer_obj.memory_objs:
                if memory_obj is not None:
                    memory_obj.ref_count_down()
            return
        buffer_obj.ref_count_down()

    def _ensure_gpu_backing(
        self,
        state: HCALayerState,
        device: torch.device,
        required_rows: int,
    ) -> None:
        """Lazily allocate a per-layer HBM-resident HCA row buffer.

        Args:
            state: HCA layer state whose row data should live in HBM.
            device: CUDA device to host the backing buffer.
            required_rows: Minimum number of compressed rows the buffer must
                cover; used to size the lazy allocation.
        """
        layer_id = state.layer_id
        with self._lock:
            existing = self._gpu_backing_rows.get(layer_id)
        if existing is not None and existing.shape[0] >= required_rows:
            return
        target_rows = max(required_rows, self._max_seq_len // state.compress_ratio)
        target_rows = max(target_rows, required_rows)
        if existing is not None and existing.shape[0] >= target_rows:
            return
        if device.type != "cuda":
            return
        new_buf = torch.empty(
            (target_rows, state.row_bytes),
            dtype=torch.uint8,
            device=device,
        )
        if existing is not None:
            new_buf[: existing.shape[0]].copy_(existing, non_blocking=True)
        with self._lock:
            self._gpu_backing_rows[layer_id] = new_buf

    def _gpu_backed_end_row(self, layer_id: int) -> int:
        """Return the exclusive end row currently populated in GPU backing."""
        with self._lock:
            return int(self._gpu_backing_end.get(layer_id, 0))

    def _fire_via_gpu_backing(
        self,
        layer_id: int,
        state: HCALayerState,
        missing: list[tuple[int, int]],
    ) -> bool:
        """Fast-path fire: scatter from HBM-resident HCA rows into KV slots.

        Args:
            layer_id: Target HCA layer id.
            state: Layer state with HBM-resident gpu_rows backing.
            missing: Row id / slot id pairs that still need KV scatter.

        Returns:
            ``True`` when all missing rows were scattered through HBM-to-HBM
            copy on the drain stream and the row ids were registered as
            resident; ``False`` if the backing buffer does not cover the
            requested rows (caller falls back to the pinned-slab path).
        """
        if not missing:
            return True
        end_row = self._gpu_backed_end_row(layer_id)
        if end_row <= 0:
            return False
        max_row = -1
        for row_id, _slot_id in missing:
            if row_id > max_row:
                max_row = row_id
        if max_row >= end_row:
            return False
        with self._lock:
            gpu_buf = self._gpu_backing_rows.get(layer_id)
        if gpu_buf is None:
            return False
        if not state.kv_cache.is_cuda:
            return False
        if state.kv_cache.dtype != torch.uint8:
            return False

        device = state.kv_cache.device
        row_ids_list = [row_id for row_id, _slot_id in missing]
        slot_ids_list = [slot_id for _row_id, slot_id in missing]
        device_index = int(device.index) if device.index is not None else 0
        with torch.cuda.device(device_index):
            row_ids_tensor = torch.tensor(
                row_ids_list, dtype=torch.long, device=device
            )
            src = gpu_buf.index_select(0, row_ids_tensor)
            if state.kv_cache.is_contiguous():
                slot_tensor = torch.tensor(
                    slot_ids_list, dtype=torch.long, device=device
                )
                flat_kv = state.kv_cache.reshape(
                    state.kv_cache.shape[0] * state.block_size, -1
                )
                flat_kv.index_copy_(0, slot_tensor, src)
            else:
                self._write_row_major_to_kv_cache(
                    state,
                    slot_ids_list,
                    src,
                    len(slot_ids_list),
                )
        with self._lock:
            resident = self._resident_hbm.setdefault(layer_id, set())
            resident.update(row_id for row_id, _slot_id in missing)
        return True

    @staticmethod
    def _dedupe_lba_cache(records_by_path: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Remove duplicated raw LBA records before registering a cache."""
        deduped: dict[str, list[Any]] = {}
        for path, records in records_by_path.items():
            seen: set[tuple[int, int, int]] = set()
            unique_records: list[Any] = []
            for record in records:
                key = (
                    int(record.file_offset),
                    int(record.slba),
                    int(record.n_sectors),
                )
                if key in seen:
                    continue
                seen.add(key)
                unique_records.append(record)
            if unique_records:
                deduped[path] = unique_records
        return deduped

    def _build_pending_reads(
        self,
        state: HCALayerState,
        missing: list[tuple[int, int]],
    ) -> list[HCAPendingRead]:
        if not missing:
            return []
        reads: list[HCAPendingRead] = []
        run_start = missing[0][0]
        run_slots: list[int] = [missing[0][1]]
        last_row = run_start
        for row_id, slot_id in missing[1:]:
            if row_id == last_row + 1:
                run_slots.append(slot_id)
            else:
                reads.extend(
                    self._submit_range_reads(state, run_start, run_slots)
                )
                run_start = row_id
                run_slots = [slot_id]
            last_row = row_id
        reads.extend(self._submit_range_reads(state, run_start, run_slots))
        return reads

    def _submit_range_reads(
        self,
        state: HCALayerState,
        start_row_id: int,
        slot_ids: list[int],
    ) -> list[HCAPendingRead]:
        if self._object_source_enabled:
            pending = self._submit_object_range_reads(
                state,
                start_row_id,
                slot_ids,
            )
            if pending:
                return pending
        pending = self._submit_flat_range_read(state, start_row_id, slot_ids)
        return [pending] if pending is not None else []

    def _submit_flat_range_read(
        self,
        state: HCALayerState,
        start_row_id: int,
        slot_ids: list[int],
    ) -> HCAPendingRead | None:
        size = len(slot_ids) * state.row_bytes
        buffer_obj = self._pinned_allocator.allocate(
            torch.Size([size]),
            torch.uint8,
            fmt=MemoryFormat.BINARY,
        )
        if buffer_obj is None:
            logger.warning(
                "HCAPrefetchManager: pinned transient slab exhausted; "
                "skip layer=%d rows=%d",
                state.layer_id,
                len(slot_ids),
            )
            return None
        future = self._executor.submit(
            self._read_rows_into_buffer,
            state,
            start_row_id,
            slot_ids,
            buffer_obj,
        )
        return HCAPendingRead(
            start_row_id=start_row_id,
            slot_ids=slot_ids,
            buffer_obj=buffer_obj,
            future=future,
        )

    def _submit_object_multi_layer_reads(
        self,
        entries: list[
            tuple[int, HCALayerState, list[int], list[tuple[int, int]]]
        ],
    ) -> dict[int, list[HCAPendingRead]]:
        """Submit one Tutti object-source batch for several HCA layers."""
        slab_submitted = self._submit_object_slab_multi_layer_reads(entries)
        if slab_submitted is not None:
            return slab_submitted

        submitted: dict[int, list[HCAPendingRead]] = {}
        grouped: dict[
            tuple[int, int],
            tuple[
                HCAObjectSource,
                list[tuple[Future[Any], list[HCAObjectBatchReadSpec]]],
            ],
        ] = {}
        total_specs = 0
        total_rows = 0
        total_bytes = 0
        for layer_id, state, _row_ids, missing in entries:
            run_start = missing[0][0]
            if any(
                row_id != run_start + offset
                for offset, (row_id, _slot_id) in enumerate(missing)
            ):
                submitted[layer_id] = []
                continue
            run_slots = [slot_id for _row_id, slot_id in missing]
            plan = self._build_object_batch_plan(state, run_start, run_slots)
            if plan is None:
                submitted[layer_id] = []
                continue
            source, specs, layout = plan
            future: Future[Any] = Future()
            pending_read = HCAPendingRead(
                start_row_id=run_start,
                slot_ids=run_slots,
                buffer_obj=None,
                future=future,
                object_layout=layout,
            )
            group_key = (id(source.loader), id(source.loader_lock))
            if group_key not in grouped:
                grouped[group_key] = (source, [])
            grouped[group_key][1].append((future, specs))
            total_specs += len(specs)
            total_rows += len(run_slots)
            total_bytes += sum(spec.nbytes for spec in specs)
            submitted[layer_id] = [pending_read]

        if not grouped:
            return submitted
        for source, plans in grouped.values():
            self._executor.submit(
                self._resolve_object_multi_layer_batch,
                source,
                plans,
            )
        logger.info(
            "HCAPrefetchManager: object-source multi-submit layers=%d "
            "pending=%d specs=%d rows=%d read_mb=%.3f groups=%d",
            sum(1 for reads in submitted.values() if reads),
            sum(len(reads) for reads in submitted.values()),
            total_specs,
            total_rows,
            total_bytes / 1024**2,
            len(grouped),
        )
        return submitted

    def _submit_object_slab_multi_layer_reads(
        self,
        entries: list[
            tuple[int, HCALayerState, list[int], list[tuple[int, int]]]
        ],
    ) -> dict[int, list[HCAPendingRead]] | None:
        """Submit one packed slab-slice Tutti batch for several HCA layers."""
        if len(entries) <= 1:
            return None

        layer_plans: list[
            tuple[
                int,
                int,
                list[int],
                list[HCAObjectBatchReadSpec],
                HCAObjectBatchLayout,
            ]
        ] = []
        reference_source: HCAObjectSource | None = None
        reference_group_key: tuple[int, int] | None = None
        reference_specs: list[HCAObjectBatchReadSpec] | None = None
        for layer_id, state, _row_ids, missing in entries:
            run_start = missing[0][0]
            if any(
                row_id != run_start + offset
                for offset, (row_id, _slot_id) in enumerate(missing)
            ):
                return None
            run_slots = [slot_id for _row_id, slot_id in missing]
            plan = self._build_object_slab_batch_plan(
                state,
                run_start,
                run_slots,
            )
            if plan is None:
                return None
            source, specs, layout = plan
            group_key = (id(source.loader), id(source.loader_lock))
            if reference_source is None:
                reference_source = source
                reference_group_key = group_key
                reference_specs = specs
            elif (
                group_key != reference_group_key
                or reference_specs is None
                or not self._object_batch_specs_share_slab(reference_specs, specs)
            ):
                return None

            layer_plans.append((layer_id, run_start, run_slots, specs, layout))

        if (
            reference_source is None
            or reference_specs is None
            or not layer_plans
        ):
            return None

        combined_specs: list[HCAObjectBatchReadSpec] = []
        spans_by_layer: dict[int, list[HCAObjectBatchSpan]] = {
            layer_id: [] for layer_id, _run_start, _run_slots, _specs, _layout in layer_plans
        }
        total_bytes = 0
        total_ranges = 0
        for spec_index, reference_spec in enumerate(reference_specs):
            packed_ranges: list[Any] = []
            target_offset = 0
            for layer_id, _run_start, _run_slots, specs, layout in layer_plans:
                spec = specs[spec_index]
                span = layout.spans[spec_index]
                packed_ranges.extend(
                    self._shift_object_read_ranges(
                        spec.read_ranges,
                        target_offset,
                    )
                )
                spans_by_layer[layer_id].append(
                    HCAObjectBatchSpan(
                        slot_offset=span.slot_offset,
                        rows=span.rows,
                        layout=self._shift_object_layout(
                            span.layout,
                            target_offset,
                        ),
                        memory_index=spec_index,
                    )
                )
                target_offset += spec.nbytes
            combined_specs.append(
                HCAObjectBatchReadSpec(
                    chunk=reference_spec.chunk,
                    read_ranges=tuple(packed_ranges),
                    nbytes=target_offset,
                    rows=reference_spec.rows,
                )
            )
            total_bytes += target_offset
            total_ranges += len(packed_ranges)

        submitted: dict[int, list[HCAPendingRead]] = {}
        futures: list[Future[Any]] = []
        total_rows = 0
        for layer_id, run_start, run_slots, _specs, _layout in layer_plans:
            future: Future[Any] = Future()
            futures.append(future)
            submitted[layer_id] = [
                HCAPendingRead(
                    start_row_id=run_start,
                    slot_ids=run_slots,
                    buffer_obj=None,
                    future=future,
                    object_layout=HCAObjectBatchLayout(
                        spans=spans_by_layer[layer_id],
                    ),
                )
            ]
            total_rows += len(run_slots)

        self._executor.submit(
            self._resolve_object_slab_multi_layer_batch,
            reference_source,
            combined_specs,
            futures,
        )
        logger.info(
            "HCAPrefetchManager: object-source slab-submit layers=%d "
            "pending=%d specs=%d ranges=%d rows=%d shared_read_mb=%.3f",
            sum(1 for reads in submitted.values() if reads),
            sum(len(reads) for reads in submitted.values()),
            len(combined_specs),
            total_ranges,
            total_rows,
            total_bytes / 1024**2,
        )
        return submitted

    def _build_object_batch_plan(
        self,
        state: HCALayerState,
        start_row_id: int,
        slot_ids: list[int],
    ) -> tuple[
        HCAObjectSource,
        list[HCAObjectBatchReadSpec],
        HCAObjectBatchLayout,
    ] | None:
        """Build a fully object-backed read plan for one contiguous row range."""
        with self._lock:
            source = self._object_sources.get(state.layer_id)
        if source is None or not slot_ids:
            return None

        request_end = start_row_id + len(slot_ids)
        cursor = start_row_id
        specs: list[HCAObjectBatchReadSpec] = []
        batch_spans: list[HCAObjectBatchSpan] = []
        for chunk in source.chunks:
            chunk_end = chunk.start_row_id + chunk.rows
            if chunk_end <= cursor:
                continue
            if chunk.start_row_id >= request_end:
                break
            if chunk.start_row_id > cursor:
                return None
            cover_start = max(cursor, chunk.start_row_id)
            cover_end = min(request_end, chunk_end)
            if cover_end <= cover_start:
                continue
            local_start = cover_start - chunk.start_row_id
            rows = cover_end - cover_start
            slot_offset = cover_start - start_row_id
            layout_and_ranges = self._object_read_layout_and_ranges(
                state,
                chunk,
                local_start,
                rows,
            )
            if layout_and_ranges is None:
                return None
            layout, read_ranges, nbytes = layout_and_ranges
            specs.append(
                HCAObjectBatchReadSpec(
                    chunk=chunk,
                    read_ranges=read_ranges,
                    nbytes=nbytes,
                    rows=rows,
                )
            )
            batch_spans.append(
                HCAObjectBatchSpan(
                    slot_offset=slot_offset,
                    rows=rows,
                    layout=layout,
                )
            )
            cursor = cover_end
            if cursor >= request_end:
                break
        if cursor < request_end or not specs:
            return None
        return source, specs, HCAObjectBatchLayout(spans=batch_spans)

    def _build_object_slab_batch_plan(
        self,
        state: HCALayerState,
        start_row_id: int,
        slot_ids: list[int],
    ) -> tuple[
        HCAObjectSource,
        list[HCAObjectBatchReadSpec],
        HCAObjectBatchLayout,
    ] | None:
        """Build a shared slab read plan for one HCA layer."""
        with self._lock:
            source = self._object_sources.get(state.layer_id)
        if source is None or not slot_ids:
            return None
        spans = self._object_source_covering_spans(
            source,
            start_row_id,
            len(slot_ids),
        )
        if not spans:
            return None
        specs: list[HCAObjectBatchReadSpec] = []
        batch_spans: list[HCAObjectBatchSpan] = []
        for spec_index, (chunk, local_start, rows, slot_offset) in enumerate(spans):
            layout_and_ranges = self._object_slab_slice_layout_and_ranges(
                state,
                chunk,
                local_start,
                rows,
            )
            if layout_and_ranges is None:
                return None
            layout, read_ranges, nbytes = layout_and_ranges
            specs.append(
                HCAObjectBatchReadSpec(
                    chunk=chunk,
                    read_ranges=read_ranges,
                    nbytes=nbytes,
                    rows=rows,
                )
            )
            batch_spans.append(
                HCAObjectBatchSpan(
                    slot_offset=slot_offset,
                    rows=rows,
                    layout=layout,
                    memory_index=spec_index,
                )
            )
        return source, specs, HCAObjectBatchLayout(spans=batch_spans)

    def _object_source_covering_spans(
        self,
        source: HCAObjectSource,
        start_row_id: int,
        rows: int,
    ) -> list[tuple[HCAObjectChunk, int, int, int]] | None:
        """Return source chunks that exactly cover a contiguous HCA row range."""
        if rows <= 0:
            return None
        request_end = start_row_id + rows
        cursor = start_row_id
        spans: list[tuple[HCAObjectChunk, int, int, int]] = []
        for chunk in source.chunks:
            chunk_end = chunk.start_row_id + chunk.rows
            if chunk_end <= cursor:
                continue
            if chunk.start_row_id >= request_end:
                break
            if chunk.start_row_id > cursor:
                return None
            cover_start = max(cursor, chunk.start_row_id)
            cover_end = min(request_end, chunk_end)
            if cover_end <= cover_start:
                continue
            spans.append(
                (
                    chunk,
                    cover_start - chunk.start_row_id,
                    cover_end - cover_start,
                    cover_start - start_row_id,
                )
            )
            cursor = cover_end
            if cursor >= request_end:
                break
        if cursor < request_end:
            return None
        return spans

    def _resolve_object_multi_layer_batch(
        self,
        source: HCAObjectSource,
        plans: list[tuple[Future[Any], list[HCAObjectBatchReadSpec]]],
    ) -> None:
        """Resolve per-layer futures from one multi-layer Tutti batch."""
        all_specs: list[HCAObjectBatchReadSpec] = []
        slices: list[tuple[Future[Any], int, int]] = []
        for future, specs in plans:
            start = len(all_specs)
            all_specs.extend(specs)
            slices.append((future, start, len(specs)))
        try:
            batch_result = self._read_rows_from_object_source_batch(
                source,
                all_specs,
            )
            if len(batch_result.memory_objs) != len(all_specs):
                raise RuntimeError(
                    "multi-layer object batch returned "
                    f"{len(batch_result.memory_objs)} objects for "
                    f"{len(all_specs)} specs"
                )
            for future, start, count in slices:
                specs = all_specs[start : start + count]
                future.set_result(
                    HCAObjectBatchResult(
                        rows=sum(spec.rows for spec in specs),
                        memory_objs=batch_result.memory_objs[
                            start : start + count
                        ],
                    )
                )
        except Exception as exc:
            logger.warning(
                "HCAPrefetchManager: object-source multi-submit failed: %r",
                exc,
            )
            for future, _start, _count in slices:
                if not future.done():
                    future.set_result(HCAObjectBatchResult(rows=0, memory_objs=[]))

    def _resolve_object_slab_multi_layer_batch(
        self,
        source: HCAObjectSource,
        specs: list[HCAObjectBatchReadSpec],
        futures: list[Future[Any]],
    ) -> None:
        """Resolve per-layer futures from one shared HCA slab Tutti batch."""
        try:
            batch_result = self._read_rows_from_object_source_batch(
                source,
                specs,
            )
            if len(batch_result.memory_objs) != len(specs):
                raise RuntimeError(
                    "slab object batch returned "
                    f"{len(batch_result.memory_objs)} objects for "
                    f"{len(specs)} specs"
                )
            extra_owners = max(0, len(futures) - 1)
            if extra_owners:
                for memory_obj in batch_result.memory_objs:
                    if memory_obj is None:
                        continue
                    for _ in range(extra_owners):
                        memory_obj.ref_count_up()
            rows = sum(spec.rows for spec in specs)
            for future in futures:
                future.set_result(
                    HCAObjectBatchResult(
                        rows=rows,
                        memory_objs=batch_result.memory_objs,
                    )
                )
        except Exception as exc:
            logger.warning(
                "HCAPrefetchManager: object-source slab-submit failed: %r",
                exc,
            )
            for future in futures:
                if not future.done():
                    future.set_result(HCAObjectBatchResult(rows=0, memory_objs=[]))

    @staticmethod
    def _object_batch_specs_match(
        left: list[HCAObjectBatchReadSpec],
        right: list[HCAObjectBatchReadSpec],
    ) -> bool:
        """Return whether two per-layer plans reference the same slab reads."""
        if len(left) != len(right):
            return False
        for left_spec, right_spec in zip(left, right, strict=True):
            if left_spec.nbytes != right_spec.nbytes:
                return False
            if left_spec.rows != right_spec.rows:
                return False
            if left_spec.chunk.key != right_spec.chunk.key:
                return False
            if left_spec.chunk.path != right_spec.chunk.path:
                return False
            if left_spec.read_ranges != right_spec.read_ranges:
                return False
        return True

    @staticmethod
    def _object_batch_specs_share_slab(
        left: list[HCAObjectBatchReadSpec],
        right: list[HCAObjectBatchReadSpec],
    ) -> bool:
        """Return whether two plans address the same physical slab chunks."""
        if len(left) != len(right):
            return False
        for left_spec, right_spec in zip(left, right, strict=True):
            if left_spec.rows != right_spec.rows:
                return False
            if left_spec.chunk.path != right_spec.chunk.path:
                return False
            if left_spec.chunk.start_row_id != right_spec.chunk.start_row_id:
                return False
            if left_spec.chunk.rows != right_spec.chunk.rows:
                return False
            if left_spec.chunk.slab_num_layers != right_spec.chunk.slab_num_layers:
                return False
            left_record = getattr(left_spec.chunk.record, "object_id", None)
            right_record = getattr(right_spec.chunk.record, "object_id", None)
            if left_record != right_record:
                return False
        return True

    @staticmethod
    def _shift_object_read_ranges(
        read_ranges: tuple[Any, ...],
        target_offset: int,
    ) -> tuple[Any, ...]:
        """Return byte ranges shifted inside a packed object-source read."""
        shifted = []
        for byte_range in read_ranges:
            shifted.append(
                type(byte_range)(
                    offset=int(byte_range.offset),
                    length=int(byte_range.length),
                    target_offset=int(byte_range.target_offset) + target_offset,
                )
            )
        return tuple(shifted)

    @staticmethod
    def _shift_object_layout(
        layout: HCAObjectReadLayout,
        target_offset: int,
    ) -> HCAObjectReadLayout:
        """Shift an object-source layout into a packed staging object."""
        if layout.row_major:
            return HCAObjectReadLayout(
                rows=layout.rows,
                hidden_bytes=layout.hidden_bytes,
                k_prefix_bytes=layout.k_prefix_bytes,
                v_prefix_bytes=layout.v_prefix_bytes,
                k_dma_bytes=layout.k_dma_bytes,
                v_dma_bytes=layout.v_dma_bytes,
                row_major=True,
                row_prefix_bytes=layout.row_prefix_bytes + target_offset,
                row_dma_bytes=layout.row_dma_bytes,
                row_bytes=layout.row_bytes,
            )
        return HCAObjectReadLayout(
            rows=layout.rows,
            hidden_bytes=layout.hidden_bytes,
            k_prefix_bytes=layout.k_prefix_bytes + target_offset,
            v_prefix_bytes=layout.v_prefix_bytes,
            k_dma_bytes=layout.k_dma_bytes + target_offset,
            v_dma_bytes=layout.v_dma_bytes,
            row_major=False,
            row_prefix_bytes=layout.row_prefix_bytes,
            row_dma_bytes=layout.row_dma_bytes,
            row_bytes=layout.row_bytes,
        )

    def _submit_object_range_reads(
        self,
        state: HCALayerState,
        start_row_id: int,
        slot_ids: list[int],
    ) -> list[HCAPendingRead]:
        with self._lock:
            source = self._object_sources.get(state.layer_id)
        if not slot_ids:
            return []
        if source is None:
            self._log_skip_once(
                state.layer_id,
                "object_no_source",
                "HCAPrefetchManager: object-source skip layer=%d "
                "reason=no_source request=[%d,%d)",
                start_row_id,
                start_row_id + len(slot_ids),
            )
            return []

        request_end = start_row_id + len(slot_ids)
        pending: list[HCAPendingRead] = []
        cursor = start_row_id
        covered_rows = 0
        object_run_start: int | None = None
        object_run_slots: list[int] = []
        object_spans: list[tuple[HCAObjectChunk, int, int, int]] = []

        def _flush_object_run() -> None:
            nonlocal object_run_start, object_run_slots, object_spans
            if object_run_start is None or not object_run_slots or not object_spans:
                object_run_start = None
                object_run_slots = []
                object_spans = []
                return
            object_pending = self._submit_object_batch_read(
                state,
                source,
                object_run_start,
                object_run_slots,
                object_spans,
            )
            if object_pending is None:
                object_pending = self._submit_flat_range_read(
                    state,
                    object_run_start,
                    object_run_slots,
                )
            if object_pending is not None:
                pending.append(object_pending)
            object_run_start = None
            object_run_slots = []
            object_spans = []

        def _append_flat_segment(segment_start: int, segment_end: int) -> None:
            if segment_end <= segment_start:
                return
            _flush_object_run()
            segment_offset = segment_start - start_row_id
            segment_rows = segment_end - segment_start
            flat_pending = self._submit_flat_range_read(
                state,
                segment_start,
                slot_ids[segment_offset : segment_offset + segment_rows],
            )
            if flat_pending is not None:
                pending.append(flat_pending)

        for chunk in source.chunks:
            chunk_end = chunk.start_row_id + chunk.rows
            if chunk_end <= cursor:
                continue
            if chunk.start_row_id >= request_end:
                break
            if chunk.start_row_id > cursor:
                gap_end = min(chunk.start_row_id, request_end)
                _append_flat_segment(cursor, gap_end)
                cursor = gap_end
                if cursor >= request_end:
                    break
            cover_start = max(cursor, chunk.start_row_id)
            cover_end = min(request_end, chunk_end)
            if cover_end <= cover_start:
                continue
            expected_start = (
                object_run_start + len(object_run_slots)
                if object_run_start is not None
                else cover_start
            )
            if object_run_start is None:
                object_run_start = cover_start
            elif cover_start != expected_start:
                _flush_object_run()
                object_run_start = cover_start
            slot_offset = cover_start - object_run_start
            local_start = cover_start - chunk.start_row_id
            take = cover_end - cover_start
            cover_offset = cover_start - start_row_id
            object_run_slots.extend(
                slot_ids[cover_offset : cover_offset + take]
            )
            object_spans.append(
                (
                    chunk,
                    local_start,
                    take,
                    slot_offset,
                )
            )
            cursor = cover_end
            covered_rows += take
            if cursor >= request_end:
                break
        _flush_object_run()
        if covered_rows <= 0:
            source_start = min(chunk.start_row_id for chunk in source.chunks)
            source_end = max(
                chunk.start_row_id + chunk.rows for chunk in source.chunks
            )
            self._log_skip_once(
                state.layer_id,
                "object_no_intersection",
                "HCAPrefetchManager: object-source skip layer=%d "
                "reason=no_intersection request=[%d,%d) "
                "source=[%d,%d) chunks=%d",
                start_row_id,
                request_end,
                source_start,
                source_end,
                len(source.chunks),
            )
        if cursor < request_end:
            _append_flat_segment(cursor, request_end)
        return pending

    def _submit_object_batch_read(
        self,
        state: HCALayerState,
        source: HCAObjectSource,
        start_row_id: int,
        slot_ids: list[int],
        spans: list[tuple[HCAObjectChunk, int, int, int]],
    ) -> HCAPendingRead | None:
        specs: list[HCAObjectBatchReadSpec] = []
        batch_spans: list[HCAObjectBatchSpan] = []
        total_bytes = 0
        slab_slice_specs = 0
        for chunk, local_start, rows, slot_offset in spans:
            layout_and_ranges = self._object_slab_slice_layout_and_ranges(
                state,
                chunk,
                local_start,
                rows,
            )
            used_slab_slice = layout_and_ranges is not None
            if layout_and_ranges is None:
                layout_and_ranges = self._object_read_layout_and_ranges(
                    state,
                    chunk,
                    local_start,
                    rows,
                )
            if layout_and_ranges is None:
                return None
            if used_slab_slice:
                slab_slice_specs += 1
            layout, read_ranges, nbytes = layout_and_ranges
            specs.append(
                HCAObjectBatchReadSpec(
                    chunk=chunk,
                    read_ranges=read_ranges,
                    nbytes=nbytes,
                    rows=rows,
                )
            )
            batch_spans.append(
                HCAObjectBatchSpan(
                    slot_offset=slot_offset,
                    rows=rows,
                    layout=layout,
                )
            )
            total_bytes += nbytes
        if not specs:
            return None
        future = self._executor.submit(
            self._read_rows_from_object_source_batch,
            source,
            specs,
        )
        if state.layer_id not in self._debug_object_submit_logged:
            self._debug_object_submit_logged.add(state.layer_id)
            logger.info(
                "HCAPrefetchManager: object-source submit layer=%d rows=%d "
                "spans=%d read_mb=%.3f slab_slice=%s",
                state.layer_id,
                len(slot_ids),
                len(specs),
                total_bytes / 1024**2,
                slab_slice_specs > 0,
            )
        return HCAPendingRead(
            start_row_id=start_row_id,
            slot_ids=slot_ids,
            buffer_obj=None,
            future=future,
            object_layout=HCAObjectBatchLayout(spans=batch_spans),
        )

    def _object_read_layout_and_ranges(
        self,
        state: HCALayerState,
        chunk: HCAObjectChunk,
        local_start: int,
        rows: int,
    ) -> tuple[HCAObjectReadLayout, tuple[Any, ...], int] | None:
        if rows <= 0 or state.row_bytes <= 0 or state.row_bytes % 2 != 0:
            self._log_skip_once(
                state.layer_id,
                "object_layout_invalid_shape",
                "HCAPrefetchManager: object-source skip layer=%d "
                "reason=invalid_shape rows=%d row_bytes=%d",
                rows,
                state.row_bytes,
            )
            return None
        raw_ranges = tuple(getattr(chunk.record, "read_ranges", ()))
        try:
            ranges = sorted(raw_ranges, key=lambda item: item.target_offset)
        except AttributeError:
            self._log_skip_once(
                state.layer_id,
                "object_layout_bad_range",
                "HCAPrefetchManager: object-source skip layer=%d "
                "reason=bad_range range_type=%s path=%s",
                type(raw_ranges[0]).__name__ if raw_ranges else "none",
                chunk.path,
            )
            return None
        if not ranges:
            self._log_skip_once(
                state.layer_id,
                "object_layout_no_ranges",
                "HCAPrefetchManager: object-source skip layer=%d "
                "reason=no_ranges raw_extents=%d path=%s",
                len(tuple(getattr(chunk.record, "raw_extents", ()))),
                chunk.path,
            )
            return None

        try:
            from lmcache.v1.kv_object_store import KVObjectByteRange
        except ImportError:
            self._log_skip_once(
                state.layer_id,
                "object_layout_import",
                "HCAPrefetchManager: object-source skip layer=%d "
                "reason=import_kv_object_range",
            )
            return None

        def _aligned_range(
            source_offset: int,
            target_offset: int,
            payload_bytes: int,
        ) -> tuple[KVObjectByteRange, int, int]:
            aligned_offset = (source_offset // 512) * 512
            prefix_bytes = source_offset - aligned_offset
            dma_bytes = _align_up(prefix_bytes + payload_bytes, 512)
            return (
                KVObjectByteRange(
                    offset=aligned_offset,
                    length=dma_bytes,
                    target_offset=target_offset,
                ),
                prefix_bytes,
                dma_bytes,
            )

        row_payload_bytes = rows * state.row_bytes
        if len(ranges) == 1:
            if row_payload_bytes <= 0:
                self._log_skip_once(
                    state.layer_id,
                    "object_layout_empty_row_payload",
                    "HCAPrefetchManager: object-source skip layer=%d "
                    "reason=empty_row_payload rows=%d row_bytes=%d",
                    rows,
                    state.row_bytes,
                )
                return None
            source = int(ranges[0].offset) + local_start * state.row_bytes
            row_range, row_prefix, row_dma = _aligned_range(
                source,
                0,
                row_payload_bytes,
            )
            layout = HCAObjectReadLayout(
                rows=rows,
                hidden_bytes=state.row_bytes // 2,
                k_prefix_bytes=0,
                v_prefix_bytes=0,
                k_dma_bytes=0,
                v_dma_bytes=0,
                row_major=True,
                row_prefix_bytes=row_prefix,
                row_dma_bytes=row_dma,
                row_bytes=state.row_bytes,
            )
            return layout, (row_range,), row_dma

        hidden_bytes = state.row_bytes // 2
        payload_bytes = rows * hidden_bytes
        if payload_bytes <= 0:
            self._log_skip_once(
                state.layer_id,
                "object_layout_empty_payload",
                "HCAPrefetchManager: object-source skip layer=%d "
                "reason=empty_payload rows=%d hidden_bytes=%d",
                rows,
                hidden_bytes,
            )
            return None

        k_source = int(ranges[0].offset) + local_start * hidden_bytes
        k_range, k_prefix, k_dma = _aligned_range(k_source, 0, payload_bytes)
        v_source = int(ranges[1].offset) + local_start * hidden_bytes
        v_range, v_prefix, v_dma = _aligned_range(
            v_source,
            k_dma,
            payload_bytes,
        )
        layout = HCAObjectReadLayout(
            rows=rows,
            hidden_bytes=hidden_bytes,
            k_prefix_bytes=k_prefix,
            v_prefix_bytes=v_prefix,
            k_dma_bytes=k_dma,
            v_dma_bytes=v_dma,
        )
        return layout, (k_range, v_range), k_dma + v_dma

    def _object_slab_layout_and_ranges(
        self,
        state: HCALayerState,
        chunk: HCAObjectChunk,
        local_start: int,
        rows: int,
    ) -> tuple[HCAObjectReadLayout, tuple[Any, ...], int] | None:
        """Return one shared slab range plus this layer's slice layout."""
        if (
            rows <= 0
            or state.row_bytes <= 0
            or state.row_bytes % 2 != 0
            or chunk.slab_layer_index is None
            or chunk.slab_num_layers <= 0
        ):
            return None
        if local_start != 0 or rows != chunk.rows:
            return None

        raw_ranges = tuple(getattr(chunk.record, "read_ranges", ()))
        try:
            ranges = sorted(raw_ranges, key=lambda item: item.target_offset)
        except AttributeError:
            return None
        if len(ranges) != 1:
            return None

        hidden_bytes = state.row_bytes // 2
        layer_index = int(chunk.slab_layer_index)
        num_layers = int(chunk.slab_num_layers)
        if layer_index < 0 or layer_index >= num_layers:
            return None
        layer_plane_bytes = rows * hidden_bytes
        kv_plane_bytes = num_layers * layer_plane_bytes
        payload_bytes = 2 * kv_plane_bytes
        if payload_bytes <= 0 or int(ranges[0].length) != payload_bytes:
            return None

        try:
            from lmcache.v1.kv_object_store import KVObjectByteRange
        except ImportError:
            return None

        source_offset = int(ranges[0].offset)
        aligned_offset = (source_offset // 512) * 512
        prefix_bytes = source_offset - aligned_offset
        dma_bytes = _align_up(prefix_bytes + payload_bytes, 512)
        read_range = KVObjectByteRange(
            offset=aligned_offset,
            length=dma_bytes,
            target_offset=0,
        )
        layer_offset = layer_index * layer_plane_bytes
        layout = HCAObjectReadLayout(
            rows=rows,
            hidden_bytes=hidden_bytes,
            k_prefix_bytes=prefix_bytes + layer_offset,
            v_prefix_bytes=layer_offset,
            k_dma_bytes=prefix_bytes + kv_plane_bytes,
            v_dma_bytes=0,
        )
        return layout, (read_range,), dma_bytes

    def _object_slab_slice_layout_and_ranges(
        self,
        state: HCALayerState,
        chunk: HCAObjectChunk,
        local_start: int,
        rows: int,
    ) -> tuple[HCAObjectReadLayout, tuple[Any, ...], int] | None:
        """Return row-major ranges for this layer's slice inside an HCA slab."""
        if (
            rows <= 0
            or state.row_bytes <= 0
            or chunk.slab_layer_index is None
            or chunk.slab_num_layers <= 0
            or local_start < 0
            or local_start + rows > chunk.rows
        ):
            return None

        raw_ranges = tuple(getattr(chunk.record, "read_ranges", ()))
        try:
            ranges = sorted(raw_ranges, key=lambda item: item.target_offset)
        except AttributeError:
            return None
        if len(ranges) != 1:
            return None

        row_bytes = state.row_bytes
        layer_index = int(chunk.slab_layer_index)
        num_layers = int(chunk.slab_num_layers)
        if layer_index < 0 or layer_index >= num_layers:
            return None

        slab_payload_bytes = int(ranges[0].length)
        slab_rows_denominator = num_layers * row_bytes
        if (
            slab_payload_bytes <= 0
            or slab_rows_denominator <= 0
            or slab_payload_bytes % slab_rows_denominator != 0
        ):
            return None
        slab_rows = slab_payload_bytes // slab_rows_denominator
        if slab_rows <= 0:
            return None
        row_offset = (chunk.start_row_id % slab_rows) + local_start
        if row_offset < 0 or row_offset + rows > slab_rows:
            return None

        try:
            from lmcache.v1.kv_object_store import KVObjectByteRange
        except ImportError:
            return None

        def _aligned_range(
            source_offset: int,
            target_offset: int,
            payload_bytes: int,
        ) -> tuple[KVObjectByteRange, int, int]:
            aligned_offset = (source_offset // 512) * 512
            prefix_bytes = source_offset - aligned_offset
            dma_bytes = _align_up(prefix_bytes + payload_bytes, 512)
            return (
                KVObjectByteRange(
                    offset=aligned_offset,
                    length=dma_bytes,
                    target_offset=target_offset,
                ),
                prefix_bytes,
                dma_bytes,
            )

        row_payload_bytes = rows * row_bytes
        if row_payload_bytes <= 0:
            return None
        base_offset = int(ranges[0].offset)
        layer_plane_bytes = slab_rows * row_bytes
        layer_offset = layer_index * layer_plane_bytes
        local_offset = row_offset * row_bytes
        row_source = base_offset + layer_offset + local_offset
        row_range, row_prefix, row_dma = _aligned_range(
            row_source,
            0,
            row_payload_bytes,
        )
        layout = HCAObjectReadLayout(
            rows=rows,
            hidden_bytes=row_bytes // 2,
            k_prefix_bytes=0,
            v_prefix_bytes=0,
            k_dma_bytes=0,
            v_dma_bytes=0,
            row_major=True,
            row_prefix_bytes=row_prefix,
            row_dma_bytes=row_dma,
            row_bytes=row_bytes,
        )
        return layout, (row_range,), row_dma

    def _deterministic_row_ids(
        self,
        positions: torch.Tensor,
        state: HCALayerState,
    ) -> list[int]:
        position = int(positions.reshape(-1)[-1].detach().cpu().item())
        return self._deterministic_row_ids_from_seq_len(position + 1, state)

    def _deterministic_row_ids_from_seq_len(
        self,
        seq_len: int,
        state: HCALayerState,
    ) -> list[int]:
        num_rows = min(
            seq_len // state.compress_ratio,
            state.initialized_rows,
        )
        return self._deterministic_row_ids_from_compressed_rows(num_rows, state)

    def _deterministic_row_ids_from_compressed_rows(
        self,
        num_rows: int,
        state: HCALayerState,
    ) -> list[int]:
        if num_rows <= 0:
            return []
        start = 0
        if self._prefetch_window_tokens > 0:
            window_rows = max(
                1,
                (self._prefetch_window_tokens + state.compress_ratio - 1)
                // state.compress_ratio,
            )
            start = max(0, num_rows - window_rows)
        max_blocks = _env_int("LMCACHE_HCA_PREFETCH_MAX_BLOCKS", 0)
        if max_blocks > 0:
            start = max(start, num_rows - max_blocks)
        return list(range(start, num_rows))

    def _global_slots_for_rows(
        self,
        layer_id: int,
        row_ids: list[int],
        state: HCALayerState,
    ) -> list[int] | None:
        active_slots = self._active_slots.get(layer_id)
        if active_slots is not None:
            if not row_ids:
                return []
            if row_ids[-1] >= active_slots.numel() or row_ids[0] < 0:
                return None
            if row_ids[-1] - row_ids[0] + 1 == len(row_ids):
                selected = active_slots[row_ids[0] : row_ids[-1] + 1]
            else:
                selected = active_slots.index_select(
                    0,
                    torch.tensor(row_ids, dtype=torch.long),
                )
            if bool(torch.any(selected < 0).item()):
                return None
            return [int(slot_id) for slot_id in selected.tolist()]

        try:
            from vllm.forward_context import get_forward_context
        except ImportError:
            return None
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None
        prefix = getattr(state.layer, "prefix", None)
        metadata = attn_metadata.get(prefix) if prefix is not None else None
        block_table = getattr(metadata, "block_table", None)
        if not isinstance(block_table, torch.Tensor) or block_table.numel() == 0:
            return None
        if block_table.ndim == 2:
            if block_table.shape[0] != 1:
                self._log_skip_once(
                    layer_id,
                    "multi_seq_block_table",
                    "HCAPrefetchManager: skip layer %d because prototype only "
                    "supports a single block table row",
                )
                return None
            block_row = block_table[0]
        else:
            block_row = block_table
        slot_ids: list[int] = []
        for row_id in row_ids:
            block_idx = row_id // state.block_size
            block_offset = row_id % state.block_size
            if block_idx >= block_row.numel():
                return None
            physical_block = int(block_row[block_idx].detach().cpu().item())
            if physical_block < 0:
                return None
            slot_ids.append(physical_block * state.block_size + block_offset)
        return slot_ids

    def _mark_resident(self, layer_id: int, row_id: int) -> None:
        with self._lock:
            resident = self._resident_hbm[layer_id]
            resident.add(row_id)
            if self._resident_budget_blocks > 0 and len(resident) > (
                self._resident_budget_blocks * 2
            ):
                keep_start = max(0, row_id - self._resident_budget_blocks + 1)
                self._resident_hbm[layer_id] = {
                    rid for rid in resident if rid >= keep_start
                }

    def _seed_resident_ids(
        self,
        compressed_seq_len: int,
        state: HCALayerState,
    ) -> range:
        """Return rows to mark resident after seed.

        Kept for compatibility with old notes; the pinned-transient path no
        longer uses seed-time residency because SSD/pinned state is not HBM.
        """
        if self._resident_budget_blocks <= 0:
            return range(0)
        start = max(0, compressed_seq_len - self._resident_budget_blocks)
        return range(start, start)

    @staticmethod
    def _read_rows_by_slots(
        kv_cache_cpu: torch.Tensor,
        slot_mapping_cpu: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        slots = slot_mapping_cpu.to(dtype=torch.long, device="cpu")
        if bool((slots < 0).any().item()):
            raise ValueError("HCA compressed slot mapping contains negative slots")
        flat = kv_cache_cpu.reshape(kv_cache_cpu.shape[0] * block_size, -1)
        return flat[slots].contiguous()

    def _write_pending_to_kv_cache(
        self,
        state: HCALayerState,
        slot_ids: list[int],
        buffer_obj: Any,
        rows_read: int,
        object_layout: HCAObjectReadLayout | HCAObjectBatchLayout | None = None,
    ) -> int:
        rows_to_write = min(rows_read, len(slot_ids))
        if rows_to_write <= 0:
            return 0
        if isinstance(object_layout, HCAObjectBatchLayout):
            if not isinstance(buffer_obj, HCAObjectBatchResult):
                return 0
            return self._write_object_batch_pending_to_kv_cache(
                state,
                slot_ids,
                buffer_obj,
                rows_to_write,
                object_layout,
            )
        tensor = buffer_obj.tensor
        if tensor is None:
            return 0
        if object_layout is not None:
            return self._write_object_pending_to_kv_cache(
                state,
                slot_ids,
                tensor,
                rows_to_write,
                object_layout,
            )
        raw = tensor[: rows_to_write * state.row_bytes].view(torch.uint8)
        src = raw.view(rows_to_write, state.row_bytes)
        return self._write_row_major_to_kv_cache(state, slot_ids, src, rows_to_write)

    def _write_object_batch_pending_to_kv_cache(
        self,
        state: HCALayerState,
        slot_ids: list[int],
        batch_result: HCAObjectBatchResult,
        rows_to_write: int,
        layout: HCAObjectBatchLayout,
    ) -> int:
        """Write batched object-source chunks into vLLM HCA KV cache."""
        written = 0
        for span_index, span in enumerate(layout.spans):
            memory_index = (
                span.memory_index
                if span.memory_index is not None
                else span_index
            )
            if memory_index < 0 or memory_index >= len(batch_result.memory_objs):
                continue
            memory_obj = batch_result.memory_objs[memory_index]
            if memory_obj is None or span.slot_offset >= rows_to_write:
                continue
            tensor = memory_obj.tensor
            if tensor is None:
                continue
            span_rows = min(span.rows, rows_to_write - span.slot_offset)
            span_slots = slot_ids[span.slot_offset : span.slot_offset + span_rows]
            written += self._write_object_pending_to_kv_cache(
                state,
                span_slots,
                tensor,
                span_rows,
                span.layout,
            )
        return written

    def _write_object_pending_to_kv_cache(
        self,
        state: HCALayerState,
        slot_ids: list[int],
        tensor: torch.Tensor,
        rows_to_write: int,
        layout: HCAObjectReadLayout,
    ) -> int:
        """Write object-source rows into vLLM HCA KV cache."""
        rows_to_write = min(rows_to_write, layout.rows, len(slot_ids))
        if rows_to_write <= 0:
            return 0
        raw = tensor.view(torch.uint8)
        if layout.row_major:
            row_bytes = layout.row_bytes or state.row_bytes
            start = layout.row_prefix_bytes
            end = start + rows_to_write * row_bytes
            if row_bytes != state.row_bytes or end > raw.numel():
                return 0
            src = raw[start:end].view(rows_to_write, row_bytes)
            return self._write_row_major_to_kv_cache(
                state,
                slot_ids,
                src,
                rows_to_write,
            )
        hidden = layout.hidden_bytes
        k_start = layout.k_prefix_bytes
        v_start = layout.k_dma_bytes + layout.v_prefix_bytes
        k_end = k_start + rows_to_write * hidden
        v_end = v_start + rows_to_write * hidden
        if k_end > raw.numel() or v_end > raw.numel():
            return 0
        k_rows = raw[k_start:k_end].view(rows_to_write, hidden)
        v_rows = raw[v_start:v_end].view(rows_to_write, hidden)
        src = torch.empty(
            (rows_to_write, state.row_bytes),
            dtype=torch.uint8,
            device=raw.device,
        )
        src[:, :hidden].copy_(k_rows, non_blocking=True)
        src[:, hidden:].copy_(v_rows, non_blocking=True)
        return self._write_row_major_to_kv_cache(
            state,
            slot_ids,
            src,
            rows_to_write,
        )

    def _write_row_major_to_kv_cache(
        self,
        state: HCALayerState,
        slot_ids: list[int],
        src: torch.Tensor,
        rows_to_write: int,
    ) -> int:
        """Copy row-major HCA rows into their current vLLM physical slots."""
        if state.kv_cache.dtype != torch.uint8:
            raise TypeError(
                "HCAPrefetchManager currently expects uint8 HCA KV rows, "
                f"got {state.kv_cache.dtype}"
            )
        if not state.kv_cache.is_contiguous():
            written = 0
            for idx, slot_id in enumerate(slot_ids[:rows_to_write]):
                block_idx = slot_id // state.block_size
                block_offset = slot_id % state.block_size
                if block_idx < 0 or block_idx >= state.kv_cache.shape[0]:
                    continue
                state.kv_cache[block_idx, block_offset].reshape(-1).copy_(
                    src[idx],
                    non_blocking=True,
                )
                written += 1
            return written
        flat_kv = state.kv_cache.reshape(
            state.kv_cache.shape[0] * state.block_size,
            -1,
        )
        max_slot = flat_kv.shape[0]
        written = 0
        run_src_start = -1
        run_slot_start = -1
        run_len = 0

        def _flush_run() -> None:
            nonlocal written, run_src_start, run_slot_start, run_len
            if run_len <= 0:
                return
            dst = flat_kv[run_slot_start : run_slot_start + run_len]
            dst.copy_(
                src[run_src_start : run_src_start + run_len],
                non_blocking=True,
            )
            written += run_len
            run_src_start = -1
            run_slot_start = -1
            run_len = 0

        for src_idx, slot_id in enumerate(slot_ids[:rows_to_write]):
            if slot_id < 0 or slot_id >= max_slot:
                _flush_run()
                continue
            if run_len > 0 and slot_id == run_slot_start + run_len:
                run_len += 1
                continue
            _flush_run()
            run_src_start = src_idx
            run_slot_start = slot_id
            run_len = 1
        _flush_run()
        return written

    def _read_rows_from_object_source_batch(
        self,
        source: HCAObjectSource,
        specs: list[HCAObjectBatchReadSpec],
    ) -> HCAObjectBatchResult:
        """Read multiple object-source spans in one Tutti loader call."""
        if not specs:
            return HCAObjectBatchResult(rows=0, memory_objs=[])
        try:
            from lmcache.utils import DiskCacheMetadata
            from lmcache.v1.gpu_connector.tutti_direct_loader import LbaRecord
        except ImportError:
            return HCAObjectBatchResult(rows=0, memory_objs=[])

        keys = []
        disk_metas = []
        read_ranges_per_key = []
        raw_lba_cache: dict[str, list[LbaRecord]] = {}
        for spec in specs:
            if spec.nbytes <= 0 or spec.rows <= 0:
                return HCAObjectBatchResult(rows=0, memory_objs=[])
            shape = torch.Size([spec.nbytes])
            disk_metas.append(
                DiskCacheMetadata(
                    path=spec.chunk.path,
                    size=spec.nbytes,
                    shape=shape,
                    dtype=torch.uint8,
                    fmt=MemoryFormat.BINARY,
                    shapes=[shape],
                    dtypes=[torch.uint8],
                )
            )
            keys.append(spec.chunk.key)
            read_ranges_per_key.append(spec.read_ranges)
            raw_extents = getattr(spec.chunk.record, "raw_extents", ())
            if raw_extents:
                raw_lba_cache.setdefault(spec.chunk.path, []).extend(
                    LbaRecord(
                        file_offset=file_offset,
                        slba=slba,
                        n_sectors=n_sectors,
                    )
                    for file_offset, slba, n_sectors in raw_extents
                )

        raw_lba_cache = self._dedupe_lba_cache(raw_lba_cache)
        if source.loader_lock is None:
            if raw_lba_cache:
                source.loader.register_lba_cache(raw_lba_cache)
            loaded = source.loader.load_chunks_to_hbm(
                keys,
                disk_metas,
                read_ranges_per_key=read_ranges_per_key,
            )
        else:
            with source.loader_lock:
                if raw_lba_cache:
                    source.loader.register_lba_cache(raw_lba_cache)
                loaded = source.loader.load_chunks_to_hbm(
                    keys,
                    disk_metas,
                    read_ranges_per_key=read_ranges_per_key,
                )
        if len(loaded) != len(specs):
            for memory_obj in loaded:
                if memory_obj is not None:
                    memory_obj.ref_count_down()
            return HCAObjectBatchResult(rows=0, memory_objs=[])
        return HCAObjectBatchResult(
            rows=sum(spec.rows for spec in specs),
            memory_objs=loaded,
        )

    def _read_rows_from_object_source(
        self,
        source: HCAObjectSource,
        chunk: HCAObjectChunk,
        read_ranges: tuple[Any, ...],
        nbytes: int,
        rows: int,
    ) -> tuple[int, Any | None]:
        """Read one object-source row span into a GPU TensorMemoryObj."""
        if nbytes <= 0 or rows <= 0:
            return 0, None
        try:
            from lmcache.utils import DiskCacheMetadata
            from lmcache.v1.gpu_connector.tutti_direct_loader import LbaRecord
        except ImportError:
            return 0, None

        shape = torch.Size([nbytes])
        disk_meta = DiskCacheMetadata(
            path=chunk.path,
            size=nbytes,
            shape=shape,
            dtype=torch.uint8,
            fmt=MemoryFormat.BINARY,
            shapes=[shape],
            dtypes=[torch.uint8],
        )
        raw_lba_cache = {}
        raw_extents = getattr(chunk.record, "raw_extents", ())
        if raw_extents:
            raw_lba_cache[chunk.path] = [
                LbaRecord(
                    file_offset=file_offset,
                    slba=slba,
                    n_sectors=n_sectors,
                )
                for file_offset, slba, n_sectors in raw_extents
            ]
        raw_lba_cache = self._dedupe_lba_cache(raw_lba_cache)
        if source.loader_lock is None:
            if raw_lba_cache:
                source.loader.register_lba_cache(raw_lba_cache)
            loaded = source.loader.load_chunks_to_hbm(
                [chunk.key],
                [disk_meta],
                read_ranges_per_key=[read_ranges],
            )
        else:
            with source.loader_lock:
                if raw_lba_cache:
                    source.loader.register_lba_cache(raw_lba_cache)
                loaded = source.loader.load_chunks_to_hbm(
                    [chunk.key],
                    [disk_meta],
                    read_ranges_per_key=[read_ranges],
                )
        if not loaded or loaded[0] is None:
            return 0, None
        return rows, loaded[0]

    def _read_rows_into_buffer(
        self,
        state: HCALayerState,
        start_row_id: int,
        slot_ids: list[int],
        buffer_obj: Any,
    ) -> int:
        """Read HCA rows into a pinned buffer.

        GPU KV writes are deliberately performed by ``drain_for_layer`` on the
        model thread by default.  PyTorch CUDA work launched from this Python
        executor can race vLLM's graph/MoE execution and surface as unrelated
        illegal-memory-access failures.
        """
        tensor = buffer_obj.tensor
        if tensor is None:
            return 0
        buffer = buffer_obj.byte_array
        if self._memory_backing_enabled:
            with self._lock:
                memory_backed = state.layer_id in self._memory_backed_layers
            rows = self._read_rows_from_memory_backing(
                state,
                start_row_id,
                len(slot_ids),
                buffer,
            )
            if rows > 0 or memory_backed:
                return rows
        return state.store.read_rows_into(
            start_row_id,
            len(slot_ids),
            buffer,
        )

    def _read_rows_from_memory_backing(
        self,
        state: HCALayerState,
        start_row_id: int,
        count: int,
        buffer: memoryview,
    ) -> int:
        """Copy a contiguous HCA row range from CPU memory backing."""
        if count <= 0:
            return 0
        with self._lock:
            ranges = list(self._memory_backing.get(state.layer_id, {}).values())
        if not ranges:
            return 0
        ranges.sort(key=lambda item: item.start_row_id)
        dst = buffer.cast("B") if buffer.format != "B" else buffer
        row_id = start_row_id
        copied = 0
        while copied < count:
            source: HCAMemoryRange | None = None
            for memory_range in ranges:
                range_end = memory_range.start_row_id + int(memory_range.rows.shape[0])
                if memory_range.start_row_id <= row_id < range_end:
                    source = memory_range
                    break
            if source is None:
                break
            local_start = row_id - source.start_row_id
            available = int(source.rows.shape[0]) - local_start
            take = min(count - copied, available)
            byte_start = copied * state.row_bytes
            byte_end = byte_start + take * state.row_bytes
            src = (
                source.rows[local_start : local_start + take]
                .contiguous()
                .view(torch.uint8)
                .numpy()
            )
            dst[byte_start:byte_end] = memoryview(src).cast("B")
            copied += take
            row_id += take
        return copied

    def _log_timing(self, event: str, layer_id: int, **fields: Any) -> None:
        if not self._timing_enabled:
            return
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        if event == "seed_lmcache" and not self._timing_seed_verbose:
            logger.debug(
                "HCAPrefetchTiming: event=%s layer=%d %s",
                event,
                layer_id,
                detail,
            )
            return
        if self._timing_seen >= self._timing_limit:
            return
        self._timing_seen += 1
        logger.info(
            "HCAPrefetchTiming: event=%s layer=%d %s",
            event,
            layer_id,
            detail,
        )

    def _log_skip_once(
        self,
        layer_id: int,
        reason: str,
        message: str,
        *args: Any,
    ) -> None:
        key = (layer_id, reason)
        if key in self._debug_skip_logged:
            return
        self._debug_skip_logged.add(key)
        logger.info(message, layer_id, *args)
