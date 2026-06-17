# SPDX-License-Identifier: Apache-2.0
# Standard
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

# First Party
from lmcache.v1.kv_object_store.object_id import KVObjectId


class KVObjectState(str, Enum):
    """Lifecycle state for an object-store metadata record."""

    ALLOCATED = "allocated"
    READY = "ready"
    EVICTED = "evicted"


@dataclass(frozen=True, slots=True)
class KVObjectByteRange:
    """One byte range backing a logical KV object.

    Args:
        offset: Byte offset inside the pool or raw-region namespace.
        length: Number of payload bytes in this range.
        target_offset: Byte offset inside the logical object payload where this
            range should be placed.
    """

    offset: int
    length: int
    target_offset: int = 0

    def __post_init__(self) -> None:
        """Validate range bounds."""
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.target_offset < 0:
            raise ValueError("target_offset must be non-negative")

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible representation."""
        return {
            "offset": self.offset,
            "length": self.length,
            "target_offset": self.target_offset,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KVObjectByteRange":
        """Build a byte range from a JSON-compatible dictionary.

        Args:
            value: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed byte range.
        """
        return cls(
            offset=int(value["offset"]),
            length=int(value["length"]),
            target_offset=int(value.get("target_offset", 0)),
        )


@dataclass(frozen=True, slots=True)
class KVObjectRecord:
    """Location and tensor metadata for one KV object.

    Args:
        object_id: Stable layer/block identifier.
        pool_id: Logical pool that contains the object bytes.
        offset: Byte offset within the pool file.
        length: Actual object length in bytes.
        aligned_length: Reserved byte length after alignment.
        shape: Tensor shape represented by this object.
        dtype: Tensor dtype string, for example ``torch.bfloat16``.
        state: Metadata lifecycle state.
        raw_extents: Optional raw LBA extents represented as
            ``(file_offset, slba, n_sectors)`` tuples.  These are used by
            Tutti raw-object storage after the filesystem has been unmounted.
        byte_ranges: Optional byte ranges backing this logical object.  Empty
            means the object is stored as one contiguous range at ``offset``.
    """

    object_id: KVObjectId
    pool_id: str
    offset: int
    length: int
    aligned_length: int
    shape: tuple[int, ...]
    dtype: str
    state: KVObjectState = KVObjectState.ALLOCATED
    raw_extents: tuple[tuple[int, int, int], ...] = ()
    byte_ranges: tuple[KVObjectByteRange, ...] = ()

    def __post_init__(self) -> None:
        """Validate byte ranges and tensor metadata."""
        if not self.pool_id:
            raise ValueError("pool_id must be non-empty")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.aligned_length < self.length:
            raise ValueError("aligned_length must be at least length")
        if any(dim <= 0 for dim in self.shape):
            raise ValueError("shape dimensions must be positive")
        if not self.dtype:
            raise ValueError("dtype must be non-empty")
        for file_offset, slba, n_sectors in self.raw_extents:
            if file_offset < 0:
                raise ValueError("raw extent file_offset must be non-negative")
            if slba < 0:
                raise ValueError("raw extent slba must be non-negative")
            if n_sectors <= 0:
                raise ValueError("raw extent n_sectors must be positive")
        if not self.byte_ranges:
            return
        covered_until = 0
        for byte_range in sorted(self.byte_ranges, key=lambda item: item.target_offset):
            if byte_range.target_offset != covered_until:
                raise ValueError("byte_ranges must exactly cover the logical payload")
            covered_until += byte_range.length
        if covered_until != self.length:
            raise ValueError("byte_ranges must cover exactly record.length bytes")

    @property
    def read_ranges(self) -> tuple[KVObjectByteRange, ...]:
        """Return byte ranges to read for this record.

        Contiguous legacy records expose a synthetic one-range view.
        """
        if self.byte_ranges:
            return self.byte_ranges
        return (KVObjectByteRange(offset=self.offset, length=self.length),)

    @property
    def is_contiguous(self) -> bool:
        """Return True when the logical payload is backed by one byte range."""
        ranges = self.read_ranges
        return len(ranges) == 1 and ranges[0].target_offset == 0

    def mark_ready(self) -> "KVObjectRecord":
        """Return a copy of this record marked as ready for retrieval."""
        return KVObjectRecord(
            object_id=self.object_id,
            pool_id=self.pool_id,
            offset=self.offset,
            length=self.length,
            aligned_length=self.aligned_length,
            shape=self.shape,
            dtype=self.dtype,
            state=KVObjectState.READY,
            raw_extents=self.raw_extents,
            byte_ranges=self.byte_ranges,
        )

    def mark_evicted(self) -> "KVObjectRecord":
        """Return a copy of this record marked as evicted."""
        return KVObjectRecord(
            object_id=self.object_id,
            pool_id=self.pool_id,
            offset=self.offset,
            length=self.length,
            aligned_length=self.aligned_length,
            shape=self.shape,
            dtype=self.dtype,
            state=KVObjectState.EVICTED,
            raw_extents=self.raw_extents,
            byte_ranges=self.byte_ranges,
        )

    def with_raw_extents(
        self,
        raw_extents: Sequence[tuple[int, int, int]],
    ) -> "KVObjectRecord":
        """Return a copy of this record with Tutti raw LBA extents attached.

        Args:
            raw_extents: Sequence of ``(file_offset, slba, n_sectors)`` tuples.

        Returns:
            A metadata record with the same logical object fields and the
            supplied raw extents.
        """
        return KVObjectRecord(
            object_id=self.object_id,
            pool_id=self.pool_id,
            offset=self.offset,
            length=self.length,
            aligned_length=self.aligned_length,
            shape=self.shape,
            dtype=self.dtype,
            state=self.state,
            raw_extents=tuple(
                (int(file_offset), int(slba), int(n_sectors))
                for file_offset, slba, n_sectors in raw_extents
            ),
            byte_ranges=self.byte_ranges,
        )

    def with_byte_ranges(
        self,
        byte_ranges: Sequence[KVObjectByteRange],
        *,
        length: int | None = None,
    ) -> "KVObjectRecord":
        """Return a logical view record backed by explicit byte ranges.

        Args:
            byte_ranges: Byte ranges that cover the logical payload.
            length: Optional logical payload length.  When omitted, this is
                derived from the ranges.

        Returns:
            A metadata record with the supplied byte-range view.
        """
        normalized = tuple(byte_ranges)
        view_length = (
            int(length)
            if length is not None
            else sum(byte_range.length for byte_range in normalized)
        )
        return KVObjectRecord(
            object_id=self.object_id,
            pool_id=self.pool_id,
            offset=normalized[0].offset if normalized else self.offset,
            length=view_length,
            aligned_length=self.aligned_length,
            shape=(view_length,),
            dtype=self.dtype,
            state=self.state,
            raw_extents=self.raw_extents,
            byte_ranges=normalized,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible metadata representation."""
        return {
            "object_id": self.object_id.to_dict(),
            "pool_id": self.pool_id,
            "offset": self.offset,
            "length": self.length,
            "aligned_length": self.aligned_length,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "state": self.state.value,
            "raw_extents": [
                [file_offset, slba, n_sectors]
                for file_offset, slba, n_sectors in self.raw_extents
            ],
            "byte_ranges": [
                byte_range.to_dict() for byte_range in self.byte_ranges
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KVObjectRecord":
        """Build a record from a JSON-compatible dictionary.

        Args:
            value: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed metadata record.

        Raises:
            KeyError: If a required field is missing.
            ValueError: If a field value is invalid.
        """
        return cls(
            object_id=KVObjectId.from_dict(value["object_id"]),
            pool_id=str(value["pool_id"]),
            offset=int(value["offset"]),
            length=int(value["length"]),
            aligned_length=int(value["aligned_length"]),
            shape=tuple(int(dim) for dim in value["shape"]),
            dtype=str(value["dtype"]),
            state=KVObjectState(str(value["state"])),
            raw_extents=tuple(
                (int(item[0]), int(item[1]), int(item[2]))
                for item in value.get("raw_extents", [])
            ),
            byte_ranges=tuple(
                KVObjectByteRange.from_dict(item)
                for item in value.get("byte_ranges", [])
            ),
        )
