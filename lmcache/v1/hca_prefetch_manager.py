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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
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
    buffer_obj: Any
    future: Future[int]


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
        self._blocking_drain = _env_flag("LMCACHE_HCA_BLOCKING_DRAIN")
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, io_workers),
            thread_name_prefix="lmcache-hca-prefetch",
        )
        self._layers: dict[int, HCALayerState] = {}
        self._pending: dict[int, list[HCAPendingRead]] = {}
        self._resident_hbm: dict[int, set[int]] = {}
        self._active_slots: dict[int, torch.Tensor] = {}
        self._lock = threading.Lock()
        self._debug_fire_logged: set[int] = set()
        self._debug_drain_logged: set[int] = set()
        self._debug_seed_logged: set[int] = set()
        self._debug_skip_logged: set[tuple[int, str]] = set()
        self._timing_enabled = _env_flag("LMCACHE_HCA_TIMING") or _env_flag(
            "LMCACHE_INDEXER_TIMING"
        )
        self._timing_seen = 0
        self._timing_limit = max(0, _env_int("LMCACHE_HCA_TIMING_LIMIT", 512))
        pinned_mb = max(1, _env_int("LMCACHE_HCA_PINNED_BUFFER_MB", 64))
        self._pinned_allocator = PinMemoryAllocator(pinned_mb * 1024 * 1024)
        logger.info(
            "HCAPrefetchManager: using transient pinned I/O slab size=%d MiB; "
            "this is not a CPU KV cache",
            pinned_mb,
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
        self._resident_hbm.setdefault(layer_id, set())
        _register_ready_layer(layer_id)

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
                self._resident_hbm[layer_id].clear()
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
            layer_rows = tensor_cpu[
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
                self._resident_hbm[layer_id].clear()
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
            )
        return seeded

    def has_seeded_rows(self, layer_id: int, compressed_seq_len: int) -> bool:
        """Return whether ``layer_id`` has enough HCA rows in its flat store."""
        state = self._layers.get(layer_id)
        return state is not None and state.initialized_rows >= compressed_seq_len

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
        self._resident_hbm[layer_id].clear()

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
        state = self._layers.get(layer_id)
        if state is None or positions is None or positions.numel() == 0:
            return
        if positions.reshape(-1).numel() == 1 and not _env_flag(
            "LMCACHE_HCA_ENABLE_DECODE_HOOK"
        ):
            return
        if state.initialized_rows <= 0:
            return
        timing = self._timing_enabled
        t0 = time.perf_counter() if timing else None
        row_ids = self._deterministic_row_ids(positions, state)
        self._fire_rows(layer_id, state, row_ids, start_time=t0)

    def _fire_rows(
        self,
        layer_id: int,
        state: HCALayerState,
        row_ids: list[int],
        start_time: float | None = None,
    ) -> None:
        if not row_ids:
            return
        timing = self._timing_enabled
        t0 = time.perf_counter() if timing and start_time is None else start_time
        t_slots0 = time.perf_counter() if timing else 0.0
        slot_ids = self._global_slots_for_rows(layer_id, row_ids, state)
        if slot_ids is None:
            return
        slots_ms = (time.perf_counter() - t_slots0) * 1000.0 if timing else 0.0
        t_filter0 = time.perf_counter() if timing else 0.0
        with self._lock:
            resident = self._resident_hbm[layer_id]
            pending_ids = {
                row_id
                for pending in self._pending[layer_id]
                for row_id in range(
                    pending.start_row_id,
                    pending.start_row_id + len(pending.slot_ids),
                )
            }
            missing = [
                (row_id, slot_id)
                for row_id, slot_id in zip(row_ids, slot_ids, strict=False)
                if row_id not in resident and row_id not in pending_ids
            ]
            filter_ms = (
                (time.perf_counter() - t_filter0) * 1000.0 if timing else 0.0
            )
            t_submit0 = time.perf_counter() if timing else 0.0
            pending_reads = self._build_pending_reads(state, missing)
            submit_ms = (
                (time.perf_counter() - t_submit0) * 1000.0 if timing else 0.0
            )
            self._pending[layer_id].extend(pending_reads)
        if missing and layer_id not in self._debug_fire_logged:
            self._debug_fire_logged.add(layer_id)
            logger.info(
                "HCAPrefetchManager: fire layer=%d rows=%d missing=%d "
                "mode=pinned_transient",
                layer_id,
                len(row_ids),
                len(missing),
            )
        if timing and t0 is not None:
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

    def prepare_layer_async(self, layer_id: int) -> None:
        """Submit a pending HCA drain for asynchronous execution.

        Args:
            layer_id: Target HCA layer whose pending reads should be drained
                in the background.
        """
        if layer_id not in self._layers:
            return
        self._executor.submit(self.drain_for_layer, layer_id)

    def drain_for_layer(self, layer_id: int) -> None:
        """Drain pending HCA reads into the target layer's vLLM KV cache.

        Args:
            layer_id: Target HCA layer id.
        """
        state = self._layers.get(layer_id)
        if state is None:
            return
        timing = self._timing_enabled
        t0 = time.perf_counter() if timing else 0.0
        t_select0 = time.perf_counter() if timing else 0.0
        with self._lock:
            pending = self._pending.get(layer_id, [])
            if not self._blocking_drain:
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
        for pending_read in pending:
            t_wait0 = time.perf_counter() if timing else 0.0
            rows_written = pending_read.future.result()
            if timing:
                wait_ms += (time.perf_counter() - t_wait0) * 1000.0
            for row_id in range(
                pending_read.start_row_id,
                pending_read.start_row_id + rows_written,
            ):
                self._mark_resident(layer_id, row_id)
            written += rows_written
            pending_read.buffer_obj.ref_count_down()
        if written and layer_id not in self._debug_drain_logged:
            self._debug_drain_logged.add(layer_id)
            logger.info(
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
        self._pinned_allocator.memcheck()
        for state in self._layers.values():
            state.store.close()

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
                pending = self._submit_range_read(state, run_start, run_slots)
                if pending is not None:
                    reads.append(pending)
                run_start = row_id
                run_slots = [slot_id]
            last_row = row_id
        pending = self._submit_range_read(state, run_start, run_slots)
        if pending is not None:
            reads.append(pending)
        return reads

    def _submit_range_read(
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
            self._read_rows_into_kv_cache,
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
            slot_ids: list[int] = []
            for row_id in row_ids:
                if row_id >= active_slots.numel():
                    return None
                slot_id = int(active_slots[row_id].item())
                if slot_id < 0:
                    return None
                slot_ids.append(slot_id)
            return slot_ids

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
    ) -> int:
        rows_to_write = min(rows_read, len(slot_ids))
        if rows_to_write <= 0:
            return 0
        tensor = buffer_obj.tensor
        if tensor is None:
            return 0
        raw = tensor[: rows_to_write * state.row_bytes].view(torch.uint8)
        src = raw.view(rows_to_write, state.row_bytes)
        for idx, slot_id in enumerate(slot_ids[:rows_to_write]):
            block_idx = slot_id // state.block_size
            block_offset = slot_id % state.block_size
            if block_idx >= state.kv_cache.shape[0]:
                continue
            row = state.kv_cache[block_idx, block_offset]
            if row.dtype != torch.uint8:
                raise TypeError(
                    "HCAPrefetchManager currently expects uint8 HCA KV rows, "
                    f"got {row.dtype}"
            )
            row.reshape(-1).copy_(src[idx].to(device=row.device, non_blocking=True))
        return rows_to_write

    def _read_rows_into_kv_cache(
        self,
        state: HCALayerState,
        start_row_id: int,
        slot_ids: list[int],
        buffer_obj: Any,
    ) -> int:
        """Read HCA rows into a pinned buffer and copy them into KV cache."""
        tensor = buffer_obj.tensor
        if tensor is None:
            return 0
        buffer = buffer_obj.byte_array
        rows_read = state.store.read_rows_into(
            start_row_id,
            len(slot_ids),
            buffer,
        )
        return self._write_pending_to_kv_cache(
            state,
            slot_ids,
            buffer_obj,
            rows_read,
        )

    def _log_timing(self, event: str, layer_id: int, **fields: Any) -> None:
        if not self._timing_enabled:
            return
        if self._timing_seen >= self._timing_limit:
            return
        self._timing_seen += 1
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info("HCAPrefetchTiming: event=%s layer=%d %s", event, layer_id, detail)

    def _log_skip_once(self, layer_id: int, reason: str, message: str) -> None:
        key = (layer_id, reason)
        if key in self._debug_skip_logged:
            return
        self._debug_skip_logged.add(key)
        logger.info(message, layer_id)
