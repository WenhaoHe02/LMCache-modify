# SPDX-License-Identifier: Apache-2.0
# Standard
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import os
import threading
import time

# First Party
from lmcache.v1.kv_object_store.record import KVObjectRecord


_FD_LOCKS: dict[int, threading.Lock] = {}
_FD_LOCKS_GUARD = threading.Lock()


def _fd_lock(fd: int) -> threading.Lock:
    """Return a process-local lock for fd-position fallback I/O."""
    with _FD_LOCKS_GUARD:
        if fd not in _FD_LOCKS:
            _FD_LOCKS[fd] = threading.Lock()
        return _FD_LOCKS[fd]


def _pread(fd: int, length: int, offset: int) -> bytes:
    """Read bytes at offset without changing fd position when supported."""
    if hasattr(os, "pread"):
        return os.pread(fd, length, offset)
    chunks: list[bytes] = []
    remaining = length
    cursor = offset
    with _fd_lock(fd):
        os.lseek(fd, offset, os.SEEK_SET)
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            read_len = len(chunk)
            remaining -= read_len
            cursor += read_len
            os.lseek(fd, cursor, os.SEEK_SET)
    return b"".join(chunks)


def _preadv(fd: int, buffers: Sequence[memoryview], offset: int) -> int:
    """Read into buffers at offset without changing fd position when supported."""
    if hasattr(os, "preadv"):
        return os.preadv(fd, buffers, offset)
    total = 0
    cursor = offset
    for buffer in buffers:
        payload = _pread(fd, len(buffer), cursor)
        read_len = len(payload)
        buffer[:read_len] = payload
        total += read_len
        cursor += read_len
        if read_len != len(buffer):
            break
    return total


def _pwritev(fd: int, buffers: Sequence[memoryview], offset: int) -> int:
    """Write buffers at offset without changing fd position when supported."""
    if hasattr(os, "pwritev"):
        return os.pwritev(fd, buffers, offset)
    total = 0
    cursor = offset
    with _fd_lock(fd):
        os.lseek(fd, offset, os.SEEK_SET)
        for buffer in buffers:
            view = memoryview(buffer).cast("B")
            written = 0
            while written < len(view):
                nbytes = os.write(fd, view[written:])
                if nbytes == 0:
                    break
                written += nbytes
                total += nbytes
                cursor += nbytes
                os.lseek(fd, cursor, os.SEEK_SET)
            if written != len(view):
                break
    return total


@dataclass(frozen=True, slots=True)
class KVObjectReadBatch:
    """Result and profile data for a batch object read.

    Args:
        payloads: Read payloads in the same order as requested records.
        elapsed_ms: Wall-clock time spent in the batch read.
        bytes_read: Total bytes returned to the caller.
    """

    payloads: list[bytes]
    elapsed_ms: float
    bytes_read: int


class KVObjectPoolIO:
    """Blocking file I/O implementation for KV object pool records.

    This class is deliberately small and synchronous. It gives the higher layer
    a concrete object-store read/write contract that can later be implemented by
    Tutti, GDS, io_uring, or a CPU helper kernel without changing object
    metadata.
    """

    def __init__(self, pool_paths: Mapping[str, Path]) -> None:
        """Create a pool I/O helper.

        Args:
            pool_paths: Mapping from logical ``pool_id`` to pool file path.

        Raises:
            ValueError: If no pools are provided.
        """
        if not pool_paths:
            raise ValueError("pool_paths must be non-empty")
        self.pool_paths = dict(pool_paths)

    def write_object(self, record: KVObjectRecord, payload: bytes) -> float:
        """Write one object's bytes to its recorded pool offset.

        Args:
            record: Object metadata containing pool id, offset and length.
            payload: Exact bytes for the object.

        Returns:
            Wall-clock write time in milliseconds.

        Raises:
            KeyError: If the record's pool id is unknown.
            ValueError: If payload length does not match record length.
            OSError: If the underlying file write fails.
        """
        return self.write_many([record], [payload])

    def read_object(self, record: KVObjectRecord) -> bytes:
        """Read one object's bytes from its recorded pool offset.

        Args:
            record: Object metadata containing pool id, offset and length.

        Returns:
            The object payload.

        Raises:
            KeyError: If the record's pool id is unknown.
            OSError: If the underlying file read fails or is short.
        """
        return self.read_many([record]).payloads[0]

    def write_many(
        self,
        records: Sequence[KVObjectRecord],
        payloads: Sequence[bytes | memoryview],
    ) -> float:
        """Write many object payloads in request order.

        Args:
            records: Metadata records to write.
            payloads: Exact payloads matching ``records`` by index.

        Returns:
            Wall-clock write time in milliseconds.

        Raises:
            ValueError: If record and payload counts differ, or a payload length
                does not match its record length.
            KeyError: If any record's pool id is unknown.
            OSError: If any underlying file write fails.
        """
        if len(records) != len(payloads):
            raise ValueError("records and payloads must have the same length")
        for record, payload in zip(records, payloads, strict=True):
            if len(payload) != record.length:
                raise ValueError(
                    f"payload length {len(payload)} does not match "
                    f"record length {record.length}"
                )

        start = time.perf_counter()
        fds = self._open_pool_fds(records, os.O_RDWR)
        try:
            for record, payload in zip(records, payloads, strict=True):
                payload_view = memoryview(payload).cast("B")
                pad_len = record.aligned_length - len(payload_view)
                write_buffers = [payload_view]
                if pad_len > 0:
                    write_buffers.append(memoryview(bytes(pad_len)))
                written = _pwritev(
                    fds[record.pool_id],
                    write_buffers,
                    record.offset,
                )
                if written != record.aligned_length:
                    raise OSError(
                        f"short write for {record.object_id.to_key()}: "
                        f"{written} of {record.aligned_length} bytes"
                    )
        finally:
            self._close_pool_fds(fds)
        return (time.perf_counter() - start) * 1000.0

    def read_many(self, records: Sequence[KVObjectRecord]) -> KVObjectReadBatch:
        """Read many object payloads in request order.

        Args:
            records: Metadata records to read.

        Returns:
            A batch result containing payloads and read profile data.

        Raises:
            KeyError: If any record's pool id is unknown.
            OSError: If any underlying file read fails or is short.
        """
        start = time.perf_counter()
        payloads: list[bytes] = []
        bytes_read = 0
        fds = self._open_pool_fds(records, os.O_RDONLY)
        try:
            for record in records:
                payload_buffer = bytearray(record.length)
                for byte_range in record.read_ranges:
                    payload = _pread(
                        fds[record.pool_id],
                        byte_range.length,
                        byte_range.offset,
                    )
                    if len(payload) != byte_range.length:
                        raise OSError(
                            f"short read for {record.object_id.to_key()}: "
                            f"{len(payload)} of {byte_range.length} bytes"
                        )
                    target_start = byte_range.target_offset
                    target_end = target_start + byte_range.length
                    payload_buffer[target_start:target_end] = payload
                payload = bytes(payload_buffer)
                payloads.append(payload)
                bytes_read += len(payload)
        finally:
            self._close_pool_fds(fds)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return KVObjectReadBatch(
            payloads=payloads,
            elapsed_ms=elapsed_ms,
            bytes_read=bytes_read,
        )

    def read_into_many(
        self,
        records: Sequence[KVObjectRecord],
        buffers: Sequence[memoryview],
    ) -> KVObjectReadBatch:
        """Read many object payloads directly into caller-provided buffers.

        Args:
            records: Metadata records to read.
            buffers: Writable buffers matching records by index.

        Returns:
            A batch result with profile data. ``payloads`` is empty because data
            is written into ``buffers`` in place.

        Raises:
            ValueError: If counts or buffer lengths do not match records.
            KeyError: If any record's pool id is unknown.
            OSError: If any underlying file read fails or is short.
        """
        if len(records) != len(buffers):
            raise ValueError("records and buffers must have the same length")
        cast_buffers = [memoryview(buffer).cast("B") for buffer in buffers]
        for record, buffer in zip(records, cast_buffers, strict=True):
            if len(buffer) != record.length:
                raise ValueError(
                    f"buffer length {len(buffer)} does not match "
                    f"record length {record.length}"
                )

        start = time.perf_counter()
        bytes_read = 0
        fds = self._open_pool_fds(records, os.O_RDONLY)
        try:
            for record, buffer in zip(records, cast_buffers, strict=True):
                read_total = 0
                for byte_range in record.read_ranges:
                    target_start = byte_range.target_offset
                    target_end = target_start + byte_range.length
                    read = _preadv(
                        fds[record.pool_id],
                        [buffer[target_start:target_end]],
                        byte_range.offset,
                    )
                    if read != byte_range.length:
                        raise OSError(
                            f"short read for {record.object_id.to_key()}: "
                            f"{read} of {byte_range.length} bytes"
                        )
                    read_total += read
                bytes_read += read_total
        finally:
            self._close_pool_fds(fds)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return KVObjectReadBatch(
            payloads=[],
            elapsed_ms=elapsed_ms,
            bytes_read=bytes_read,
        )

    def _open_pool_fds(
        self,
        records: Sequence[KVObjectRecord],
        flags: int,
    ) -> dict[str, int]:
        pool_ids = {record.pool_id for record in records}
        fds: dict[str, int] = {}
        try:
            for pool_id in pool_ids:
                path = self.pool_paths[pool_id]
                fds[pool_id] = os.open(path, flags | getattr(os, "O_BINARY", 0))
        except Exception:
            self._close_pool_fds(fds)
            raise
        return fds

    def _close_pool_fds(self, fds: Mapping[str, int]) -> None:
        for fd in fds.values():
            try:
                os.close(fd)
            finally:
                with _FD_LOCKS_GUARD:
                    _FD_LOCKS.pop(fd, None)
