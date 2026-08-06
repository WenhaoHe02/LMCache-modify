# SPDX-License-Identifier: Apache-2.0
"""Low-interference CPU writes into pre-resolved raw NVMe extents."""

from __future__ import annotations

# Standard
from collections.abc import Callable, Sequence
import mmap
import os
import threading
import time


_LBA_BYTES = 512
_MIB = 1024**2
_DEFAULT_BLOCK_BYTES = 4 * _MIB

RawExtent = tuple[int, int, int]
WriteAdmissionWaiter = Callable[[int, float], None]


class CPURawBlockWriter:
    """Persist packed KV bytes through the kernel NVMe queue.

    The writer keeps one raw block-device descriptor and one page-aligned
    staging buffer alive for its full lifetime. Writes are deliberately QD1:
    before every bounded block the caller-provided admission hook can yield to
    demand reads, and rate limiting leaves queue-idle gaps for GPU-direct reads.

    Args:
        device_path: Kernel block device exposed by snvme.
        target_mib_s: Maximum sustained write rate. Zero disables throttling.
        block_bytes: Maximum bytes in one non-preemptible kernel write.
        wait_for_admission: Optional callback invoked before every block. It
            receives the block size and the monotonic time at which the full
            wave started waiting.

    Raises:
        ValueError: If rate or block geometry is invalid.
        OSError: If the block device cannot be opened with ``O_DIRECT``.
    """

    def __init__(
        self,
        device_path: str,
        *,
        target_mib_s: float = 512.0,
        block_bytes: int = _DEFAULT_BLOCK_BYTES,
        wait_for_admission: WriteAdmissionWaiter | None = None,
    ) -> None:
        if target_mib_s < 0:
            raise ValueError("target_mib_s must be non-negative")
        if block_bytes <= 0 or block_bytes % _LBA_BYTES:
            raise ValueError("block_bytes must be a positive multiple of 512")
        direct_flag = getattr(os, "O_DIRECT", 0)
        if direct_flag == 0:
            raise OSError("O_DIRECT is unavailable on this platform")
        self._device_path = str(device_path)
        self._target_bytes_s = target_mib_s * _MIB
        self._block_bytes = int(block_bytes)
        self._wait_for_admission = wait_for_admission
        self._fd = os.open(self._device_path, os.O_WRONLY | direct_flag)
        self._buffer = mmap.mmap(-1, self._block_bytes)
        self._lock = threading.Lock()
        self._closed = False

    @property
    def device_path(self) -> str:
        """Return the opened kernel block-device path."""
        return self._device_path

    @property
    def target_mib_s(self) -> float:
        """Return the configured per-device bandwidth limit."""
        return self._target_bytes_s / _MIB

    @property
    def block_bytes(self) -> int:
        """Return the maximum non-preemptible write size."""
        return self._block_bytes

    def write(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        raw_extents: Sequence[RawExtent],
        base_file_offset: int,
        logical_nbytes: int,
    ) -> tuple[tuple[RawExtent, ...], float]:
        """Write one logical wave and return its normalized raw extents.

        Args:
            payload: Packed logical bytes. Missing alignment tail bytes are
                zero-filled.
            raw_extents: Physical extents as ``(file_offset, slba, sectors)``.
            base_file_offset: Logical pool offset corresponding to payload byte
                zero.
            logical_nbytes: Aligned reservation length to persist.

        Returns:
            Normalized extents covering the wave and elapsed milliseconds.

        Raises:
            ValueError: If arguments are empty, unaligned, or inconsistent.
            RuntimeError: If extents do not cover the wave or a short write
                occurs.
        """
        payload_view = memoryview(payload).cast("B")
        payload_nbytes = len(payload_view)
        if base_file_offset < 0 or base_file_offset % _LBA_BYTES:
            raise ValueError("base_file_offset must be non-negative and aligned")
        if logical_nbytes <= 0 or logical_nbytes % _LBA_BYTES:
            raise ValueError("logical_nbytes must be a positive LBA multiple")
        if payload_nbytes <= 0 or payload_nbytes > logical_nbytes:
            raise ValueError("payload length must be within logical_nbytes")
        normalized = self._normalize_extents(
            raw_extents,
            base_file_offset=base_file_offset,
            logical_nbytes=logical_nbytes,
        )
        wait_started_s = time.perf_counter()
        started_s = wait_started_s
        with self._lock:
            if self._closed:
                raise RuntimeError("CPU raw writer is closed")
            for file_offset, slba, n_sectors in normalized:
                extent_nbytes = n_sectors * _LBA_BYTES
                extent_cursor = 0
                while extent_cursor < extent_nbytes:
                    io_nbytes = min(
                        self._block_bytes,
                        extent_nbytes - extent_cursor,
                    )
                    if self._wait_for_admission is not None:
                        self._wait_for_admission(io_nbytes, wait_started_s)
                    source_offset = file_offset + extent_cursor - base_file_offset
                    source_nbytes = min(
                        io_nbytes,
                        max(0, payload_nbytes - source_offset),
                    )
                    if source_nbytes > 0:
                        self._buffer[:source_nbytes] = payload_view[
                            source_offset : source_offset + source_nbytes
                        ]
                    if source_nbytes < io_nbytes:
                        self._buffer[source_nbytes:io_nbytes] = b"\0" * (
                            io_nbytes - source_nbytes
                        )
                    block_started_s = time.perf_counter()
                    block_view = memoryview(self._buffer)[:io_nbytes]
                    try:
                        written = os.pwrite(
                            self._fd,
                            block_view,
                            slba * _LBA_BYTES + extent_cursor,
                        )
                    finally:
                        block_view.release()
                    if written != io_nbytes:
                        raise RuntimeError(
                            f"short CPU raw write: {written}/{io_nbytes} bytes"
                        )
                    self._throttle(io_nbytes, block_started_s)
                    extent_cursor += io_nbytes
        return normalized, (time.perf_counter() - started_s) * 1000.0

    def close(self) -> None:
        """Close the persistent aligned buffer and block-device descriptor."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._buffer.close()
            os.close(self._fd)

    def _throttle(self, nbytes: int, block_started_s: float) -> None:
        if self._target_bytes_s <= 0:
            return
        minimum_s = nbytes / self._target_bytes_s
        delay_s = minimum_s - (time.perf_counter() - block_started_s)
        if delay_s > 0:
            time.sleep(delay_s)

    @staticmethod
    def _normalize_extents(
        raw_extents: Sequence[RawExtent],
        *,
        base_file_offset: int,
        logical_nbytes: int,
    ) -> tuple[RawExtent, ...]:
        object_end = base_file_offset + logical_nbytes
        normalized: list[RawExtent] = []
        covered_nbytes = 0
        logical_cursor = base_file_offset
        for file_offset, slba, n_sectors in sorted(raw_extents):
            extent_nbytes = int(n_sectors) * _LBA_BYTES
            extent_end = int(file_offset) + extent_nbytes
            write_start = max(base_file_offset, int(file_offset))
            write_end = min(object_end, extent_end)
            if write_start >= write_end:
                continue
            if write_start != logical_cursor:
                raise RuntimeError("CPU raw extents contain a logical gap")
            extent_skip = write_start - int(file_offset)
            write_nbytes = write_end - write_start
            if extent_skip % _LBA_BYTES or write_nbytes % _LBA_BYTES:
                raise ValueError("CPU raw extent is not LBA aligned")
            normalized.append(
                (
                    write_start,
                    int(slba) + extent_skip // _LBA_BYTES,
                    write_nbytes // _LBA_BYTES,
                )
            )
            covered_nbytes += write_nbytes
            logical_cursor = write_end
        if covered_nbytes != logical_nbytes:
            raise RuntimeError(
                "CPU raw extents do not cover the logical wave: "
                f"{covered_nbytes}/{logical_nbytes} bytes"
            )
        return tuple(normalized)

    def __enter__(self) -> "CPURawBlockWriter":
        """Return this writer for context-manager use."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close this writer when leaving a context manager."""
        self.close()
