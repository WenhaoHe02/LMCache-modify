# SPDX-License-Identifier: Apache-2.0
# Standard
from __future__ import annotations

from pathlib import Path
import math
import os
import threading

# First Party
from lmcache.v1.kv_object_store.object_id import KVObjectId
from lmcache.v1.kv_object_store.record import KVObjectRecord


class KVObjectPoolFullError(RuntimeError):
    """Raised when a KV object pool has no slot for a new object."""


class KVObjectPoolLayout:
    """Fixed-slot file layout for KV objects in one logical pool.

    This first MVP keeps allocation deterministic and cheap: each object gets one
    aligned slot in a sparse pool file. Later integration can replace allocation
    with a free-list or extent allocator without changing the object identifier
    and metadata contract.
    """

    def __init__(
        self,
        *,
        pool_id: str,
        pool_path: Path,
        slot_bytes: int,
        capacity: int,
        alignment: int = 4096,
        dense: bool = False,
    ) -> None:
        """Create a fixed-slot pool layout.

        Args:
            pool_id: Logical identifier used in metadata records.
            pool_path: File path for the sparse pool.
            slot_bytes: Maximum bytes per slot before alignment.
            capacity: Number of object slots in the pool.
            alignment: Byte alignment for each slot.
            dense: When true, allocate objects back-to-back by aligned object
                length instead of fixed slot stride. This is the preferred mode
                for Tutti FIEMAP because it avoids sparse holes.

        Raises:
            ValueError: If sizing arguments are invalid.
        """
        if not pool_id:
            raise ValueError("pool_id must be non-empty")
        if slot_bytes <= 0:
            raise ValueError("slot_bytes must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if alignment <= 0:
            raise ValueError("alignment must be positive")

        self.pool_id = pool_id
        self.pool_path = pool_path
        self.slot_bytes = slot_bytes
        self.capacity = capacity
        self.alignment = alignment
        self.aligned_slot_bytes = self.align_length(slot_bytes)
        self.dense = dense
        self._next_slot = 0
        self._next_offset = 0
        self._lock = threading.Lock()

    def ensure_file(self) -> None:
        """Create or resize the sparse pool file to the required capacity."""
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        with self.pool_path.open("ab"):
            pass
        if self.dense:
            return
        required_bytes = self.aligned_slot_bytes * self.capacity
        current_bytes = os.path.getsize(self.pool_path)
        if current_bytes < required_bytes:
            with self.pool_path.open("r+b") as pool_file:
                pool_file.truncate(required_bytes)

    def allocate(
        self,
        object_id: KVObjectId,
        *,
        length: int,
        shape: tuple[int, ...],
        dtype: str,
    ) -> KVObjectRecord:
        """Allocate one fixed slot and return its metadata record.

        Args:
            object_id: Object identifier for the new slot.
            length: Actual object byte length.
            shape: Tensor shape represented by the object.
            dtype: Tensor dtype string.

        Returns:
            An allocated metadata record.

        Raises:
            ValueError: If the object is larger than one slot.
            KVObjectPoolFullError: If the pool has no remaining slot.
        """
        if length <= 0:
            raise ValueError("length must be positive")
        if length > self.slot_bytes:
            raise ValueError("length exceeds fixed slot size")
        self.ensure_file()
        aligned_length = self.align_length(length)
        with self._lock:
            if self._next_slot >= self.capacity:
                raise KVObjectPoolFullError(
                    f"KV object pool {self.pool_id!r} is full"
                )
            slot = self._next_slot
            self._next_slot += 1
            if self.dense:
                offset = self._next_offset
                self._next_offset += aligned_length
            else:
                offset = slot * self.aligned_slot_bytes
        if self.dense:
            with self.pool_path.open("r+b") as pool_file:
                pool_file.truncate(self._next_offset)
        else:
            offset = slot * self.aligned_slot_bytes
        return KVObjectRecord(
            object_id=object_id,
            pool_id=self.pool_id,
            offset=offset,
            length=length,
            aligned_length=aligned_length,
            shape=shape,
            dtype=dtype,
        )

    def pool_size_bytes(self) -> int:
        """Return the required pool file size in bytes."""
        if self.dense:
            return self._next_offset
        return self.aligned_slot_bytes * self.capacity

    def align_length(self, length: int) -> int:
        """Round a byte length up to this pool's alignment.

        Args:
            length: Raw byte length.

        Returns:
            Aligned byte length.
        """
        if length <= 0:
            raise ValueError("length must be positive")
        return math.ceil(length / self.alignment) * self.alignment
