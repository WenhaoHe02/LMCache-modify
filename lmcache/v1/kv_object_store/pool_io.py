# SPDX-License-Identifier: Apache-2.0
# Standard
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import os
import time

# First Party
from lmcache.v1.kv_object_store.record import KVObjectRecord


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
                written = os.pwritev(
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
                payload = os.pread(fds[record.pool_id], record.length, record.offset)
                if len(payload) != record.length:
                    raise OSError(
                        f"short read for {record.object_id.to_key()}: "
                        f"{len(payload)} of {record.length} bytes"
                    )
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
                read = os.preadv(fds[record.pool_id], [buffer], record.offset)
                if read != record.length:
                    raise OSError(
                        f"short read for {record.object_id.to_key()}: "
                        f"{read} of {record.length} bytes"
                    )
                bytes_read += read
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
                fds[pool_id] = os.open(path, flags)
        except Exception:
            self._close_pool_fds(fds)
            raise
        return fds

    def _close_pool_fds(self, fds: Mapping[str, int]) -> None:
        for fd in fds.values():
            os.close(fd)
