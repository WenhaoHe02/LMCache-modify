# SPDX-License-Identifier: Apache-2.0
"""GPU-direct NVMe backend for CSA Indexer per-token storage.

Replaces the legacy per-CSA-layer ``indexer_layer_*.bin`` files +
``os.pread``/``os.pwrite`` path in :class:`IndexerSSDManager`. Storage lives
in a pre-reserved raw NVMe region addressable through Tutti's snvme
GPU-direct DMA path:

* Writes (LMCache retrieve seed; prefill new-token persistence)::

    CPU bytes
       │ store_bytes_to_raw_extents()
       ▼  (one-shot CPU→Tutti staging→NVMe via P2P)
    NVMe raw extent

* Reads (CSA spec prefetch + true LI miss correction)::

    NVMe raw extent
       │ load_chunks_to_hbm() with synthetic key + byte_range
       ▼  (snvme GPU-direct DMA, no CPU staging)
    HBM staging slot
       │ G2G copy
       ▼
    IndexerHBMPool.pool_tensor

Layout
------
One contiguous raw region per rank, partitioned into ``num_csa_layers``
fixed-size slots.  Each slot holds the full ``max_seq_len`` tokens for one
CSA layer in flat ``[token_id, token_bytes]`` order, matching the
file-backed :class:`IndexerBlockStore` semantics so the rest of
:class:`IndexerSSDManager` (HBM pool, two-tier LRU, speculation logic) is
unchanged.

Slot ``i``'s byte range within the raw region is::

    [i * slot_bytes, (i + 1) * slot_bytes)
    slot_bytes = max_seq_len * token_bytes

The CSA-layer-id → slot-index mapping is contiguous in CSA layer order
(layer ids sorted ascending → slot indices 0..29).
"""
from __future__ import annotations

# Standard
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.kv_object_store import KVObjectByteRange
from lmcache.v1.memory_management import MemoryFormat

if TYPE_CHECKING:
    from lmcache.v1.gpu_connector.tutti_direct_loader import (
        LbaRecord,
        TuttiDirectLoader,
    )

logger = init_logger(__name__)

# NVMe sector size enforced by snvme.
_NVME_LBS: int = 512


def _align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to ``alignment`` bytes."""
    return ((value + alignment - 1) // alignment) * alignment


def _align_down(value: int, alignment: int) -> int:
    """Round ``value`` down to ``alignment`` bytes."""
    return (value // alignment) * alignment


@dataclass(frozen=True, slots=True)
class TuttiIndexerLayerSlot:
    """Byte range carved out of the raw region for one CSA layer.

    Args:
        layer_id: Transformer-side CSA layer index.
        slot_index: Position within the raw region; slot_index ∈ [0, num_layers).
        file_offset: Byte offset of the slot's first byte inside the raw region.
        slot_bytes: Capacity of the slot in bytes
            (= ``max_seq_len`` × ``token_bytes``, sector-aligned).
        token_bytes: Bytes per logical token; matches IndexerSSDManager's
            ``token_bytes`` (132 for FP8 + 4 scale).
        max_seq_len: Logical token capacity of the slot.
    """

    layer_id: int
    slot_index: int
    file_offset: int
    slot_bytes: int
    token_bytes: int
    max_seq_len: int

    def token_byte_offset(self, token_id: int) -> int:
        """Return slot-relative byte offset for ``token_id``.

        Args:
            token_id: Global token position index inside this CSA layer.

        Returns:
            Byte offset relative to the start of this layer's slot.

        Raises:
            ValueError: If ``token_id`` is outside the slot capacity.
        """
        if token_id < 0 or token_id >= self.max_seq_len:
            raise ValueError(
                f"token_id {token_id} outside CSA slot capacity {self.max_seq_len}"
            )
        return token_id * self.token_bytes

    def absolute_byte_offset(self, token_id: int) -> int:
        """Return raw-region byte offset for ``token_id``."""
        return self.file_offset + self.token_byte_offset(token_id)


class TuttiIndexerStorage:
    """Owner of the indexer raw region; produces per-layer slot proxies.

    The storage is a single contiguous raw NVMe region addressable through
    Tutti.  Reads and writes operate at slot byte offsets; the higher-level
    :class:`IndexerSSDManager` uses one
    :class:`TuttiIndexerBlockStore` instance per CSA layer that targets a
    single :class:`TuttiIndexerLayerSlot` inside this region.
    """

    def __init__(
        self,
        tutti_loader: "TuttiDirectLoader",
        raw_region_path: str,
        raw_region_extents: Sequence[tuple[int, int, int]],
        layer_ids: Sequence[int],
        token_bytes: int,
        max_seq_len: int,
    ) -> None:
        """Initialise the shared indexer raw region.

        Args:
            tutti_loader: Active Tutti direct loader bound to the rank's NVMe
                device.  Used for all read/write submissions.
            raw_region_path: Synthetic Tutti path for the indexer region
                (typically ``tutti://csa_indexer_rank_<R>``).  This is the key
                under which ``raw_region_extents`` is registered in the
                loader's LBA cache.
            raw_region_extents: ``(file_offset, slba, n_sectors)`` extents
                covering the entire indexer region.  The first extent's
                ``file_offset`` must be 0 (the region is treated as a logical
                file starting at offset 0).
            layer_ids: CSA layer ids in ascending order.  ``len(layer_ids)``
                slots are carved out of the region.
            token_bytes: Bytes per token K vector (FP8 + scale = 132).
            max_seq_len: Maximum logical token capacity per layer.

        Raises:
            ValueError: If sizing or extent arguments are invalid, or if the
                supplied extents are too small for all slots.
        """
        if tutti_loader is None:
            raise ValueError("tutti_loader is required for Tutti indexer storage")
        if not raw_region_path:
            raise ValueError("raw_region_path must be non-empty")
        if not raw_region_extents:
            raise ValueError("raw_region_extents must be non-empty")
        if not layer_ids:
            raise ValueError("layer_ids must be non-empty")
        if token_bytes <= 0:
            raise ValueError("token_bytes must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        slot_bytes = _align_up(max_seq_len * token_bytes, _NVME_LBS)
        sorted_layers = sorted(int(lid) for lid in layer_ids)
        total_required = slot_bytes * len(sorted_layers)
        provided_bytes = sum(
            int(n_sectors) * _NVME_LBS
            for _, _, n_sectors in raw_region_extents
        )
        if provided_bytes < total_required:
            raise ValueError(
                f"Tutti indexer raw region is too small: have {provided_bytes} bytes, "
                f"need {total_required} bytes for {len(sorted_layers)} slots of "
                f"{slot_bytes} bytes each"
            )

        # The first extent must cover offset 0 so that token_id 0 maps to a
        # known LBA via the file_offset-based extent lookup in Tutti.
        first_offset = int(raw_region_extents[0][0])
        if first_offset != 0:
            raise ValueError(
                "First raw_region_extent must start at file_offset 0; got "
                f"{first_offset}"
            )

        self._tutti_loader = tutti_loader
        self._raw_region_path = raw_region_path
        self._raw_region_extents = [
            (int(file_offset), int(slba), int(n_sectors))
            for file_offset, slba, n_sectors in raw_region_extents
        ]
        self._token_bytes = int(token_bytes)
        self._max_seq_len = int(max_seq_len)
        self._slot_bytes = int(slot_bytes)
        self._slots: dict[int, TuttiIndexerLayerSlot] = {}
        for slot_index, layer_id in enumerate(sorted_layers):
            file_offset = slot_index * slot_bytes
            self._slots[layer_id] = TuttiIndexerLayerSlot(
                layer_id=layer_id,
                slot_index=slot_index,
                file_offset=file_offset,
                slot_bytes=slot_bytes,
                token_bytes=self._token_bytes,
                max_seq_len=self._max_seq_len,
            )

        # Register the raw region with Tutti so subsequent reads via
        # load_chunks_to_hbm can resolve raw_region_path to LBA extents.
        from lmcache.v1.gpu_connector.tutti_direct_loader import LbaRecord

        records = [
            LbaRecord(slba=slba, n_sectors=n_sectors, file_offset=file_offset)
            for file_offset, slba, n_sectors in self._raw_region_extents
        ]
        self._tutti_loader.register_lba_cache({self._raw_region_path: records})
        self._write_lock = threading.Lock()

        logger.info(
            "TuttiIndexerStorage: path=%s layers=%d slot_bytes=%d total_bytes=%d "
            "first_slba=%d extents=%d",
            self._raw_region_path,
            len(sorted_layers),
            self._slot_bytes,
            total_required,
            self._raw_region_extents[0][1],
            len(self._raw_region_extents),
        )

    @property
    def token_bytes(self) -> int:
        """Bytes per token K vector."""
        return self._token_bytes

    @property
    def max_seq_len(self) -> int:
        """Maximum logical token capacity per CSA layer."""
        return self._max_seq_len

    @property
    def slot_bytes(self) -> int:
        """Capacity of one CSA-layer slot in bytes (sector-aligned)."""
        return self._slot_bytes

    @property
    def raw_region_path(self) -> str:
        """Synthetic Tutti path under which this region is registered."""
        return self._raw_region_path

    def slot_for_layer(self, layer_id: int) -> TuttiIndexerLayerSlot:
        """Return the :class:`TuttiIndexerLayerSlot` for ``layer_id``.

        Raises:
            KeyError: If ``layer_id`` was not registered at construction.
        """
        if layer_id not in self._slots:
            raise KeyError(f"CSA layer id {layer_id} not registered in indexer storage")
        return self._slots[layer_id]

    def write_bytes(
        self,
        slot: TuttiIndexerLayerSlot,
        start_token_id: int,
        payload: bytes,
    ) -> None:
        """Persist ``payload`` to ``slot`` starting at ``start_token_id``.

        Args:
            slot: Target CSA-layer slot.
            start_token_id: Global token position the payload starts at.
            payload: Contiguous bytes; length must be a multiple of
                ``slot.token_bytes`` and fit within slot capacity.

        Raises:
            ValueError: If payload size or offset is invalid.
            RuntimeError: If Tutti reports an NVMe error.
        """
        if not payload:
            return
        if len(payload) % slot.token_bytes != 0:
            raise ValueError(
                f"payload length {len(payload)} is not a multiple of token_bytes "
                f"{slot.token_bytes}"
            )
        n_tokens = len(payload) // slot.token_bytes
        if start_token_id < 0 or start_token_id + n_tokens > slot.max_seq_len:
            raise ValueError(
                f"write range [{start_token_id}, {start_token_id + n_tokens}) "
                f"outside slot capacity {slot.max_seq_len}"
            )

        with self._write_lock:
            absolute_offset = slot.absolute_byte_offset(start_token_id)
            payload_end = absolute_offset + len(payload)
            aligned_start = _align_down(absolute_offset, _NVME_LBS)
            aligned_end = _align_up(payload_end, _NVME_LBS)
            write_payload = payload
            if aligned_start != absolute_offset or aligned_end != payload_end:
                existing = self._read_aligned_bytes(
                    slot,
                    aligned_start,
                    aligned_end - aligned_start,
                    start_token_id,
                    n_tokens,
                )
                merged = bytearray(existing)
                payload_start = absolute_offset - aligned_start
                merged[payload_start : payload_start + len(payload)] = payload
                write_payload = bytes(merged)
            sub_extents = self._extents_for_range(
                aligned_start,
                len(write_payload),
            )
            self._tutti_loader.store_bytes_to_raw_extents(
                write_payload,
                raw_extents=sub_extents,
                base_file_offset=aligned_start,
                logical_nbytes=len(write_payload),
            )

    def _read_aligned_bytes(
        self,
        slot: TuttiIndexerLayerSlot,
        absolute_offset: int,
        nbytes: int,
        start_token_id: int,
        n_tokens: int,
    ) -> bytes:
        """Read an aligned range used to preserve edge sectors on writes."""
        synthetic_key = _synthesize_indexer_read_key(
            slot.layer_id,
            start_token_id,
            start_token_id + n_tokens - 1,
        )
        disk_meta = DiskCacheMetadata(
            path=self._raw_region_path,
            size=nbytes,
            fmt=MemoryFormat.BINARY_BUFFER,
            shape=torch.Size((nbytes,)),
            dtype=torch.uint8,
        )
        memory_objs = self._tutti_loader.load_chunks_to_hbm(
            [synthetic_key],
            [disk_meta],
            shapes_per_key=None,
            file_offsets=[0],
            read_ranges_per_key=[
                (
                    KVObjectByteRange(
                        offset=absolute_offset,
                        length=nbytes,
                        target_offset=0,
                    ),
                )
            ],
            io_priority="demand",
        )
        if not memory_objs or memory_objs[0] is None:
            raise RuntimeError("Tutti edge-sector read returned no payload")
        memory_obj = memory_objs[0]
        try:
            tensor = memory_obj.raw_tensor
            if tensor is None:
                raise RuntimeError("Tutti edge-sector read has no raw tensor")
            host = tensor.reshape(-1)[:nbytes].cpu().numpy().tobytes()
        finally:
            ref_count_down = getattr(memory_obj, "ref_count_down", None)
            if callable(ref_count_down):
                ref_count_down()
        if len(host) != nbytes:
            raise RuntimeError(
                f"Tutti edge-sector read returned {len(host)}/{nbytes} bytes"
            )
        return host

    def build_read_request(
        self,
        slot: TuttiIndexerLayerSlot,
        token_ids: Sequence[int],
    ) -> "TuttiIndexerReadRequest":
        """Return a read request descriptor for ``token_ids`` in ``slot``.

        The descriptor packs the byte ranges, a synthetic CacheEngineKey, and
        a :class:`DiskCacheMetadata` instance so the caller can invoke
        :meth:`TuttiDirectLoader.load_chunks_to_hbm` directly.

        Args:
            slot: Target CSA-layer slot.
            token_ids: Token ids to read.  Empty lists return ``None``.

        Returns:
            A :class:`TuttiIndexerReadRequest` ready for Tutti.  Returns
            ``None`` if ``token_ids`` is empty.
        """
        if not token_ids:
            return TuttiIndexerReadRequest.empty()
        sorted_ids = sorted(set(int(tid) for tid in token_ids))
        coalesced = _coalesce_token_runs(sorted_ids)
        ranges: List[KVObjectByteRange] = []
        cursor = 0
        token_runs: List[tuple[int, int]] = []  # (first_token_id, n_tokens)
        for start_token, n_tokens in coalesced:
            absolute_offset = slot.absolute_byte_offset(start_token)
            length = n_tokens * slot.token_bytes
            ranges.append(
                KVObjectByteRange(
                    offset=absolute_offset,
                    length=length,
                    target_offset=cursor,
                )
            )
            token_runs.append((start_token, n_tokens))
            cursor += length

        logical_nbytes = cursor
        synthetic_key = _synthesize_indexer_read_key(
            slot.layer_id, sorted_ids[0], sorted_ids[-1]
        )
        disk_meta = DiskCacheMetadata(
            path=self._raw_region_path,
            size=logical_nbytes,
            fmt=MemoryFormat.BINARY_BUFFER,
            shape=torch.Size((logical_nbytes,)),
            dtype=torch.uint8,
        )
        return TuttiIndexerReadRequest(
            key=synthetic_key,
            disk_meta=disk_meta,
            file_offset=0,
            read_ranges=tuple(ranges),
            token_runs=tuple(token_runs),
            token_bytes=slot.token_bytes,
            total_nbytes=logical_nbytes,
        )

    def load_read_request(
        self,
        request: "TuttiIndexerReadRequest",
        *,
        io_priority: str = "demand",
        should_continue: Optional[Callable[[], bool]] = None,
        deadline_monotonic: Optional[float] = None,
    ) -> list[Any]:
        """Load one prepared indexer request through the shared Tutti loader.

        Args:
            request: Descriptor returned by :meth:`build_read_request`.
            io_priority: Tutti admission class, either ``demand`` or
                ``speculative``.
            should_continue: Optional cancellation predicate checked before
                each speculative microbatch.
            deadline_monotonic: Optional absolute ``time.perf_counter()``
                deadline for speculative submission.

        Returns:
            Loader results parallel to the request's synthetic key list.

        Raises:
            RuntimeError: If Tutti reports a submission or completion error.
        """
        if request.is_empty:
            return []
        return self._tutti_loader.load_chunks_to_hbm(
            [request.key],
            [request.disk_meta],
            shapes_per_key=None,
            file_offsets=[request.file_offset],
            read_ranges_per_key=[request.read_ranges],
            io_priority=io_priority,
            max_batch_ios=8 if io_priority == "speculative" else None,
            should_continue=should_continue,
            deadline_monotonic=deadline_monotonic,
        )

    def _extents_for_range(
        self,
        absolute_offset: int,
        length: int,
    ) -> list["LbaRecord"]:
        """Return raw LBA extents covering ``[absolute_offset, +length)``.

        Args:
            absolute_offset: Byte offset inside the raw region (matches
                ``file_offset`` semantics of :class:`LbaRecord`).
            length: Number of bytes to cover.

        Returns:
            Ordered list of :class:`LbaRecord` instances whose union covers
            the requested range.  Each record's ``file_offset`` field reflects
            the absolute byte offset inside the raw region.

        Raises:
            ValueError: If the requested range exceeds the raw region.
        """
        from lmcache.v1.gpu_connector.tutti_direct_loader import LbaRecord

        if length <= 0:
            raise ValueError("length must be positive")
        end_offset = absolute_offset + length
        result: list[LbaRecord] = []
        covered = 0
        for file_offset, slba, n_sectors in self._raw_region_extents:
            extent_start = file_offset
            extent_end = file_offset + n_sectors * _NVME_LBS
            write_start = max(absolute_offset, extent_start)
            write_end = min(end_offset, extent_end)
            if write_end <= write_start:
                if covered >= length:
                    break
                continue
            sub_slba = slba + (write_start - extent_start) // _NVME_LBS
            sub_sectors = _align_up(write_end - write_start, _NVME_LBS) // _NVME_LBS
            result.append(
                LbaRecord(
                    slba=sub_slba,
                    n_sectors=sub_sectors,
                    file_offset=write_start,
                )
            )
            covered += write_end - write_start
            if covered >= length:
                break
        if covered < length:
            raise ValueError(
                f"raw region does not cover [{absolute_offset}, {end_offset}); "
                f"covered {covered} of {length} bytes"
            )
        return result


@dataclass(frozen=True, slots=True)
class TuttiIndexerReadRequest:
    """Read request prepared for :meth:`TuttiDirectLoader.load_chunks_to_hbm`.

    Args:
        key: Synthetic :class:`CacheEngineKey` identifying this request.
        disk_meta: Metadata pointing at the indexer raw region path; ``size``
            is the request's total payload length.
        file_offset: Base file offset (always 0 for raw-region reads; the
            actual per-range offsets live in ``read_ranges``).
        read_ranges: Byte ranges relative to the raw region; target_offset is
            the byte offset within the request's logical payload.
        token_runs: ``(first_token_id, n_tokens)`` pairs in the same order as
            ``read_ranges``.  The caller uses these to map ranges back to
            individual token K vectors.
        token_bytes: Bytes per token K vector.
        total_nbytes: Total logical payload size for this request.
    """

    key: CacheEngineKey
    disk_meta: DiskCacheMetadata
    file_offset: int
    read_ranges: tuple[KVObjectByteRange, ...]
    token_runs: tuple[tuple[int, int], ...]
    token_bytes: int
    total_nbytes: int

    @classmethod
    def empty(cls) -> "TuttiIndexerReadRequest":
        """Return an empty request used when there are no tokens to read."""
        return cls(
            key=_synthesize_indexer_read_key(-1, 0, 0),
            disk_meta=DiskCacheMetadata(path="", size=0),
            file_offset=0,
            read_ranges=(),
            token_runs=(),
            token_bytes=0,
            total_nbytes=0,
        )

    @property
    def is_empty(self) -> bool:
        """Return True when the request carries no read ranges."""
        return self.total_nbytes == 0


def _coalesce_token_runs(sorted_ids: Sequence[int]) -> list[tuple[int, int]]:
    """Group consecutive token ids into ``(start, length)`` runs.

    Args:
        sorted_ids: Token ids in ascending order; assumed unique.

    Returns:
        Ordered list of runs covering exactly the input ids.
    """
    runs: list[tuple[int, int]] = []
    if not sorted_ids:
        return runs
    run_start = sorted_ids[0]
    run_len = 1
    for tid in sorted_ids[1:]:
        if tid == run_start + run_len:
            run_len += 1
            continue
        runs.append((run_start, run_len))
        run_start = tid
        run_len = 1
    runs.append((run_start, run_len))
    return runs


def _synthesize_indexer_read_key(
    layer_id: int,
    first_token_id: int,
    last_token_id: int,
) -> CacheEngineKey:
    """Return a synthetic :class:`CacheEngineKey` for an indexer read.

    The synthetic key never collides with real LMCache content keys because
    its ``model_name`` component is namespaced to ``lmcache::indexer``.

    Args:
        layer_id: CSA layer id covered by the read.
        first_token_id: Lowest token id in the read.
        last_token_id: Highest token id in the read.

    Returns:
        A :class:`CacheEngineKey` instance suitable for passing to
        :meth:`TuttiDirectLoader.load_chunks_to_hbm`.
    """
    import torch

    chunk_hash = (
        (int(layer_id) & 0xFFFF) << 48
        | (int(first_token_id) & 0xFFFFFFFF) << 16
        | (int(last_token_id) & 0xFFFF)
    )
    return CacheEngineKey(
        model_name="lmcache::indexer",
        world_size=1,
        worker_id=int(layer_id),
        chunk_hash=int(chunk_hash),
        dtype=torch.uint8,
    )
