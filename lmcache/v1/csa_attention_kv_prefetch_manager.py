# SPDX-License-Identifier: Apache-2.0
"""On-demand prefetcher for DSv4 CSA attention KV (~100 MiB / 24K prefix).

The DeepSeek V4 CSA layers use ``compress_ratio == 4`` MLA-style attention
KV.  At baseline, LMCache scatters the entire compressed KV (``584 B`` per
compressed entry, ``S/4`` entries per layer, 21 sparse layers for V4-Flash) into
24K prefix) into vLLM's MLA K cache during ``retrieve``.  Most of those
bytes are never read by a single decode step: the Lightning Indexer picks
``top-K = 1024`` compressed blocks per layer (out of ~6K for 24K prefix),
so ~85 % of the 100 MiB is wasted.

This manager removes the synchronous scatter by:

* Filtering ``csa_attention_kv`` out of the retrieve shape table so vLLM's
  K cache slots stay empty after retrieve.  This is the ``-100 MiB``
  retrieve-time saving.
* Pre-fetching the predicted ``top-K`` compressed blocks for the next CSA
  layer in the FFN/MoE window, using the same HC-proxy that drives the
  Indexer SSD manager.  Tutti reads land directly into vLLM's K cache
  slots via the standard ``load_chunks_to_hbm`` + ``read_ranges_per_key``
  path (GPU-direct DMA, no CPU bounce).
* Issuing miss-correction reads after the true Lightning Indexer outputs
  its top-K.  The Indexer.forward call site is monkey-patched so that
  sparse attention only sees the post-corrected ``true_topk`` and waits
  on a single drain event before reading the K cache.

The manager does NOT shrink vLLM's MLA K cache capacity: vLLM still
allocates a full-prefix-sized tensor at engine startup, and this
prefetcher writes into those existing slots on demand.  Shrinking the K
cache is a vLLM-side change and is out of scope here.

Data source
-----------
The compressed attention KV bytes are read from LMCache's existing chunk
storage (group role ``csa_attention_kv``, one of the 7 DSv4 groups).  No
new SSD storage is introduced.  At retrieve time the cache engine builds
a per-CSA-layer chunk map mapping compressed block ids back to the
``CacheEngineKey`` + byte offset inside the chunk's csa_attention_kv
slab.  The manager owns this map for the request's lifetime and uses it
to assemble Tutti read ranges.

This manager is wired together with the Indexer SSD manager through the single
``LMCACHE_INDEXER_ENABLE_PREFETCH`` feature gate. When disabled, the legacy
retrieve scatter path is preserved.
"""

from __future__ import annotations

# Standard
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

# Third Party
import torch

# First Party
try:
    import lmcache.c_ops as _csa_c_ops
except ImportError:
    _csa_c_ops = None  # type: ignore[assignment]

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.csa_pipeline_nvtx import (
    CsaNvtxEvent,
    CsaNvtxRange,
    csa_pipeline_nvtx,
)
from lmcache.v1.kv_object_store import KVObjectByteRange

if TYPE_CHECKING:
    from lmcache.v1.gpu_connector.tutti_direct_loader import TuttiDirectLoader

logger = init_logger(__name__)

_ACTIVE_MANAGER: Optional["CSAAttentionKVPrefetchManager"] = None


_ACTIVE_MANAGER_LOCK = threading.Lock()


def _timing_enabled() -> bool:
    """Return True when CSA attention KV prefetch timing logs are enabled."""
    value = os.environ.get("LMCACHE_CSA_ATTENTION_KV_TIMING", "")
    return value.lower() in {"1", "true", "yes", "on"}


def get_csa_attention_kv_prefetch_manager() -> Optional[
    "CSAAttentionKVPrefetchManager"
]:
    """Return the process-local CSA attention KV prefetcher, if any."""
    with _ACTIVE_MANAGER_LOCK:
        return _ACTIVE_MANAGER


def set_csa_attention_kv_prefetch_manager(
    manager: Optional["CSAAttentionKVPrefetchManager"],
) -> None:
    """Replace the process-local CSA attention KV prefetcher."""
    global _ACTIVE_MANAGER
    with _ACTIVE_MANAGER_LOCK:
        _ACTIVE_MANAGER = manager


@dataclass(frozen=True, slots=True)
class CSAAttentionKVChunkLoc:
    """LMCache-chunk descriptor for one block of csa_attention_kv bytes.

    Args:
        first_compressed_block: Compressed block id (= compressed token id //
            block_size) of this chunk's first compressed block.
        n_compressed_blocks: Number of consecutive compressed blocks covered by
            this chunk.  The chunk's compressed range is
            ``[first_compressed_block, first_compressed_block +
            n_compressed_blocks)``.
        key: LMCache cache engine key identifying the chunk.
        disk_meta: Cached :class:`DiskCacheMetadata` pointing at a synthetic
            ``tutti://<pool_id>`` path that resolves through the Tutti
            loader's ``_lba_cache``.  Reads against this path bypass the
            filesystem and read directly from the pre-FIEMAP'd raw NVMe
            extents (raw_extents).
        layer_byte_offset: **Pool-absolute** byte offset where the chunk's
            csa_attention_kv slab for this CSA layer begins.  Already
            includes ``record.offset``; the prefetcher passes it straight
            into :class:`KVObjectByteRange`.``offset`` because Tutti's
            ``_logical_read_ranges`` ignores ``file_offsets`` when explicit
            read_ranges are provided, so the byte addressing must already
            be in the same coordinate system as the registered LBA extents'
            ``file_offset`` fields.
        bytes_per_block: ``block_size * token_bytes`` for the csa_attention_kv
            group (block_size == 64 typically, token_bytes == 584).
        raw_extents: ``(file_offset, slba, n_sectors)`` extents covering the
            chunk's slot inside the pool.  Registered once with the Tutti
            loader in ``register_request_chunks``; the union of every
            chunk's extents is what makes the synthetic path resolvable for
            every byte range the prefetcher asks for.
        physical_block_ids: vLLM K-cache physical row indices that this
            chunk's csa_attention_kv slab must be written into, one per
            compressed block covered by the chunk.  Derived from the
            retrieve's ``slot_mapping`` so the destination row matches what
            the sparse-attention kernel will read.  An empty tuple means
            the caller did not provide ``slot_mapping`` (legacy path); the
            prefetcher then falls back to the sequence-position index,
            which is correct ONLY for fresh per-request CSA caches without
            a block table indirection.
        layer_major: Whether this descriptor is one contiguous full-prefix
            HCA layer object.  Such objects bypass per-entry range planning
            and are loaded with one logical read plus one fused scatter.
    """

    first_compressed_block: int
    n_compressed_blocks: int
    key: CacheEngineKey
    disk_meta: DiskCacheMetadata
    layer_byte_offset: int
    bytes_per_block: int
    raw_extents: tuple[tuple[int, int, int], ...] = ()
    physical_block_ids: tuple[int, ...] = ()
    # V28 (HCA rows): the Tutti loader rejects non-512B-aligned reads, and
    # HCA's per-layer stride (2*584 bytes) never aligns.  When set,
    # ``layer_byte_offset`` is already rounded DOWN to a 512B boundary,
    # ``read_length`` is the rounded-UP read size, and the real payload
    # starts ``payload_skip`` bytes into the returned buffer.  Zero values
    # (CSA chunks) mean "payload starts at 0, read n_blocks*bytes_per_block".
    payload_skip: int = 0
    read_length: int = 0
    layer_major: bool = False

    @property
    def end_compressed_block(self) -> int:
        """Exclusive upper bound of compressed blocks covered by this chunk."""
        return self.first_compressed_block + self.n_compressed_blocks

    def contains(self, compressed_block_id: int) -> bool:
        """Return True if this chunk covers ``compressed_block_id``."""
        return (
            self.first_compressed_block
            <= compressed_block_id
            < self.end_compressed_block
        )

    def chunk_byte_offset_for(self, compressed_block_id: int) -> int:
        """Return absolute byte offset inside the chunk for one block id.

        For 512B-rounded chunks (``payload_skip > 0``) the returned offset
        addresses the TRUE payload byte (rounded base + skip + local
        stride), so per-block miss reads still start at the real data; the
        caller must round again for the Tutti alignment requirement.

        Raises:
            ValueError: If ``compressed_block_id`` is outside this chunk.
        """
        if not self.contains(compressed_block_id):
            raise ValueError(
                f"compressed_block_id {compressed_block_id} outside chunk range "
                f"[{self.first_compressed_block}, {self.end_compressed_block})"
            )
        local = compressed_block_id - self.first_compressed_block
        return self.layer_byte_offset + self.payload_skip + local * self.bytes_per_block


@dataclass(slots=True)
class CSAAttentionKVLayerState:
    """Runtime state for one CSA layer.

    Args:
        layer_id: Transformer-side CSA layer id (e.g., 2, 4, ..., 60).
        compressed_block_size: Compressed entries per K cache block; matches
            vLLM's IndexerCache layout (64).
        token_bytes: Bytes per compressed entry (584 for fp8_ds_mla).
        k_cache_tensor: vLLM's MLA Attention K cache tensor for this layer,
            shape ``[num_blocks, compressed_block_size, token_bytes]``.
            Reads land directly into slices of this tensor.
        in_pool_bitmap: GPU bool tensor of length
            ``ceil(max_seq_len / compress_ratio / compressed_block_size)``
            where ``True`` at index ``b`` means the layer's K cache slot for
            compressed block ``b`` is currently populated.  This replaces a
            Python set so the miss-vs-pool check can run entirely on the GPU
            and avoid a per-layer GPU→CPU sync of the full top-K.
        chunks: Ordered list of LMCache chunks covering the active request's
            prefix for this CSA layer.  Populated by
            :meth:`register_request_chunks`.
        pending_reads_lock: Guards mutation of ``pending_reads_bitmap`` /
            ``last_drain_event`` / ``pending_drains``.
        pending_reads_bitmap: CPU bool bitmap of compressed block ids whose
            Tutti read is currently in-flight or queued. Tensor indexing
            replaces serial Python set scans, including for small layers.
        resident_blocks_bitmap: CPU mirror of blocks whose NVMe read completed
            and whose scatter is ordered on the layer's final CUDA event. This
            mirror closes the gap between clearing pending state and a GPU
            consumer observing ``in_pool_bitmap``.
        pending_read_count: Number of true entries in
            ``pending_reads_bitmap``; lets condition waits avoid rescanning
            the bitmap.
        last_drain_event: Optional CUDA event recording the completion of
            the latest read submission.  Sparse attention waits on this
            event during :meth:`CSAAttentionKVPrefetchManager.drain_for_layer`.
        pending_drains: List of ``(event, memory_objs, nvtx_range, op_id)``
            tuples accumulated by ``_issue_reads``. The staging-buffer
            ``ref_count_down`` is
            deferred until after the event has been synchronized in
            ``drain_for_layer`` to prevent freeing Tutti HBM staging buffers
            while the async ``non_blocking=True`` CUDA copies are still
            in-flight.  Guarded by ``pending_reads_lock``.
    """

    layer_id: int
    compressed_block_size: int
    token_bytes: int
    k_cache_tensor: torch.Tensor
    in_pool_bitmap: torch.Tensor
    chunks: List[CSAAttentionKVChunkLoc]
    pending_reads_lock: threading.Condition
    pending_reads_bitmap: torch.Tensor
    resident_blocks_bitmap: torch.Tensor
    pending_read_count: int
    last_drain_event: Optional[torch.cuda.Event]
    pending_drains: List[
        Tuple[Optional[torch.cuda.Event], List[Any], Optional[CsaNvtxRange], str]
    ]
    # V28 (HCA): vLLM's HCA K cache tensor is a NON-CONTIGUOUS slice of a
    # larger buffer, so a flat [rows, bytes] view is not materialisable
    # in-place (view raises; reshape would silently scatter into a COPY).
    # When True the tensor stays 3-D [num_blocks, block_slot_size,
    # token_bytes] and scatters use two-index ``index_put_`` with
    # ``(row // block_slot_size, row % block_slot_size)``.
    block_slot_scatter: bool = False
    block_slot_size: int = 0
    # Optional CSA fast-path tables. Each logical compressed block indexes one
    # physical SLBA and one destination K-cache row. They remain on the GPU so
    # Tutti can submit an arbitrary block subset without rebuilding Python
    # range/descriptors for every layer invocation.
    indexed_slba_table: Optional[torch.Tensor] = None
    indexed_dst_rows_table: Optional[torch.Tensor] = None
    layer_major_dst_rows_table: Optional[torch.Tensor] = None


class CSAAttentionKVPrefetchManager:
    """Tutti-backed on-demand loader for DSv4 CSA attention KV bytes.

    The manager exposes a small public API mirroring
    :class:`IndexerSSDManager`'s lifecycle:

    * :meth:`register_layer` — register one CSA layer's vLLM K cache tensor.
    * :meth:`register_request_chunks` — register the per-request LMCache
      chunk locations for csa_attention_kv (called from the cache engine
      at retrieve time, once the filter has zero-shaped the group).
    * :meth:`fire_predicted_reads` — submit Tutti reads for the predicted
      top-K of a layer (called from the Indexer SSD manager when its
      HC-proxy has produced a prediction).
    * :meth:`submit_miss_reads` — submit Tutti reads for the true_topk
      blocks not covered by ``fire_predicted_reads`` (called from the
      patched Indexer.forward right after the true Lightning Indexer
      returns).
    * :meth:`drain_for_layer` — wait until the layer's pending reads are
      done.  Called from the patched Indexer.forward before returning the
      true_topk to the sparse attention kernel.
    * :meth:`patch_indexer_forward` — monkey-patch one CSA Indexer
      module's ``forward`` so true_topk triggers miss-correction + drain.
    """

    def __init__(
        self,
        tutti_loader: "TuttiDirectLoader",
        csa_layer_ids: Sequence[int],
        compressed_block_size: int = 64,
        token_bytes: int = 584,
    ) -> None:
        """Initialise the manager.

        Args:
            tutti_loader: Active Tutti direct loader bound to the rank's
                NVMe device.
            csa_layer_ids: CSA layer ids in ascending order.
            compressed_block_size: Compressed entries per K cache block.
                Defaults to 64 (vLLM IndexerCache packed layout).
            token_bytes: Bytes per compressed entry.  Defaults to 584
                (DSv4 fp8_ds_mla format).
        """
        if tutti_loader is None:
            raise ValueError("tutti_loader is required for CSA attention KV prefetch")
        if not csa_layer_ids:
            raise ValueError("csa_layer_ids must be non-empty")
        if compressed_block_size <= 0:
            raise ValueError("compressed_block_size must be positive")
        if token_bytes <= 0:
            raise ValueError("token_bytes must be positive")

        self._tutti_loader = tutti_loader
        self._compressed_block_size = int(compressed_block_size)
        self._token_bytes = int(token_bytes)
        self._bytes_per_block = self._compressed_block_size * self._token_bytes
        # Per-request LBA cache snapshot.  Other paths (e.g.,
        # ``_tutti_batched_get``) overwrite ``self._tutti_loader._lba_cache``
        # with the FILTERED ``read_ranges`` extents, which exclude
        # csa_attention_kv.  We cache the union here and re-register before
        # every ``_issue_reads`` to ensure the loader sees our full-record
        # extents at the moment Tutti dispatches.
        self._pending_raw_lba_cache: dict[str, list[Any]] = {}
        self._layers: Dict[int, CSAAttentionKVLayerState] = {}
        self._csa_layer_ids = tuple(sorted(int(lid) for lid in csa_layer_ids))
        # HCA layers use deterministic staged submissions and are gated via
        # wait_for_layer instead of a patched indexer forward.
        self._hca_layer_ids: Tuple[int, ...] = ()
        self._patched_modules: List[Tuple[Any, str, Callable]] = []
        self._patch_lock = threading.Lock()
        self._active_request_id: Optional[str] = None
        self._full_nsys_seen_request_id = ""
        self._full_nsys_seen_requests = 0
        self._full_nsys_capture_active = False
        self._full_nsys_capture_complete = False
        # Private stream for streaming-scatter copies (staging -> K cache).
        # Serialized by Tutti's _io_lock (one scatter runs at a time), so a
        # single stream per device suffices.  Kept off the default stream to
        # avoid cross-rank deadlocks with forward collectives.
        self._scatter_streams: Dict[int, torch.cuda.Stream] = {}
        self._scatter_streams_lock = threading.Lock()
        # Selected-id uploads are independent of NVMe queue ownership. Keep
        # them off the I/O stream so a prediction can prepare its GPU plan
        # while an earlier request is polling/scattering on Tutti.
        self._prepare_streams: Dict[int, torch.cuda.Stream] = {}
        self._prepare_streams_lock = threading.Lock()
        self._prediction_waiter: Optional[Callable[[int], bool]] = None
        self._scheduled_layer_futures: Dict[int, Any] = {}
        self._scheduled_layer_futures_lock = threading.Lock()
        self._closed = False

    def _scatter_stream_for(self, device: torch.device) -> torch.cuda.Stream:
        """Return the stream used for K-cache scatter copies.

        Reuses the Tutti loader's ``io_stream`` when available: that
        stream already orders the NVMe submit/poll spin kernels, so scatter
        kernels enqueued on it are strictly ordered after the DMA-completion
        poll of their staging bytes AND never interleave with the store
        path's H2D staging uploads (which are ordered on the same stream).
        The V13/V14 crashes (async illegal access surfacing in NCCL/MoE
        sync) appeared exactly when a separate scatter stream ran while
        store_raw H2D traffic hammered staging; the old, long-stable clone
        path always used io_stream.  Falls back to a private stream when
        the loader has none (CPU-only tests).
        """
        io_stream = self._tutti_loader.io_stream
        if io_stream is not None:
            return io_stream
        index = device.index if device.index is not None else 0
        with self._scatter_streams_lock:
            stream = self._scatter_streams.get(index)
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                self._scatter_streams[index] = stream
            return stream

    def _prepare_stream_for(self, device: torch.device) -> torch.cuda.Stream:
        """Return the side stream used to upload indexed selection plans."""
        index = device.index if device.index is not None else 0
        with self._prepare_streams_lock:
            stream = self._prepare_streams.get(index)
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                self._prepare_streams[index] = stream
            return stream

    @property
    def csa_layer_ids(self) -> Tuple[int, ...]:
        """Return registered CSA layer ids in ascending order."""
        return self._csa_layer_ids

    @property
    def active_request_id(self) -> str:
        """Return the req_id whose chunks are currently registered.

        Used by :class:`IndexerSSDManager` to scope its fire gate: proxy
        scoring runs once per (request, layer) because the NVMe-resident
        prefix is fixed at registration time, so later chunks' fires would
        recompute an identical prediction.  Empty string when no request
        is registered.
        """
        return self._active_request_id or ""

    @property
    def full_nsys_capture_active(self) -> bool:
        """Return whether this worker currently owns an active full capture."""
        return self._full_nsys_capture_active

    @property
    def bytes_per_block(self) -> int:
        """Bytes per compressed K cache block (block_size × token_bytes)."""
        return self._bytes_per_block

    @property
    def hca_layer_ids(self) -> Tuple[int, ...]:
        """Transformer layer ids registered via :meth:`register_hca_layer`."""
        return self._hca_layer_ids

    def register_layer(
        self,
        layer_id: int,
        k_cache_tensor: torch.Tensor,
    ) -> None:
        """Register one CSA layer's vLLM K cache tensor.

        Args:
            layer_id: Transformer-side CSA layer id.
            k_cache_tensor: vLLM's MLA Attention K cache tensor, shape
                ``[num_blocks, compressed_block_size, token_bytes]``.  Reads
                land into slices of this tensor without copy.

        Raises:
            ValueError: If ``layer_id`` is not in the registered set or the
                tensor shape is incompatible.
        """
        if int(layer_id) not in self._csa_layer_ids:
            raise ValueError(
                f"layer_id {layer_id} not in registered CSA layer ids "
                f"{self._csa_layer_ids}"
            )
        if k_cache_tensor.ndim != 3:
            raise ValueError(
                "k_cache_tensor must be 3-D [num_blocks, block_size, token_bytes]; "
                f"got shape {tuple(k_cache_tensor.shape)}"
            )
        if int(k_cache_tensor.shape[1]) != self._compressed_block_size:
            raise ValueError(
                f"k_cache_tensor block_size {int(k_cache_tensor.shape[1])} != "
                f"manager compressed_block_size {self._compressed_block_size}"
            )
        if int(k_cache_tensor.shape[2]) != self._token_bytes:
            raise ValueError(
                f"k_cache_tensor token_bytes {int(k_cache_tensor.shape[2])} != "
                f"manager token_bytes {self._token_bytes}"
            )
        self._register_layer_state(
            int(layer_id),
            k_cache_tensor,
            compressed_block_size=self._compressed_block_size,
            token_bytes=self._token_bytes,
        )

    def register_hca_layer(
        self,
        layer_id: int,
        k_cache_tensor: torch.Tensor,
    ) -> None:
        """Register one HCA layer's vLLM K cache tensor.

        HCA compresses 128:1 and vLLM packs its K cache as
        ``[num_blocks, hca_block_size, token_bytes]`` with a small block
        size (8 entries). Reads address compressed-entry rows:
        chunk descriptors count entries and carry flattened row ids
        ``block_id * hca_block_size + slot``.  The vLLM tensor is a
        non-contiguous slice of a larger paged buffer, so it CANNOT be
        flattened with ``view``; the layer state keeps the 3-D tensor and
        sets ``block_slot_scatter`` so scatters decompose each flat row id
        into ``(row // hca_block_size, row % hca_block_size)``.

        Args:
            layer_id: Transformer-side HCA layer id.
            k_cache_tensor: vLLM's HCA K cache tensor, 3-D as above.

        Raises:
            ValueError: If the tensor is not 3-D or its token_bytes differ
                from the manager's.
        """
        if k_cache_tensor.ndim != 3:
            raise ValueError(
                "HCA k_cache_tensor must be 3-D [num_blocks, block_size, "
                f"token_bytes]; got shape {tuple(k_cache_tensor.shape)}"
            )
        if int(k_cache_tensor.shape[2]) != self._token_bytes:
            raise ValueError(
                f"HCA k_cache_tensor token_bytes {int(k_cache_tensor.shape[2])} "
                f"!= manager token_bytes {self._token_bytes}"
            )
        hca_block_size = int(k_cache_tensor.shape[1])
        self._hca_layer_ids = tuple(sorted(set(self._hca_layer_ids) | {int(layer_id)}))
        self._register_layer_state(
            int(layer_id),
            k_cache_tensor,
            compressed_block_size=1,
            token_bytes=self._token_bytes,
            block_slot_size=hca_block_size,
        )

    def _register_layer_state(
        self,
        layer_id: int,
        k_cache_tensor: torch.Tensor,
        compressed_block_size: int,
        token_bytes: int,
        block_slot_size: int = 0,
    ) -> None:
        """Create the per-layer runtime state shared by CSA and HCA layers.

        Args:
            layer_id: Transformer-side layer id.
            k_cache_tensor: Scatter target.  CSA layers flatten it to
                ``[shape[0], -1]`` rows; HCA layers keep it 3-D and use
                block/slot decomposition (``block_slot_size > 0``).
            compressed_block_size: Entries per addressable row (64 for CSA
                blocks, 1 for HCA flattened entries).
            token_bytes: Bytes per compressed entry.
            block_slot_size: Entries per physical block when the tensor is
                non-contiguous (HCA); 0 selects the flat CSA addressing.
        """
        # Addressable-row count bounds the ids chunk descriptors may carry;
        # for block/slot layers that is blocks * slots.
        if block_slot_size > 0:
            num_rows = int(k_cache_tensor.shape[0]) * block_slot_size
        else:
            num_rows = int(k_cache_tensor.shape[0])
        in_pool_bitmap = torch.zeros(
            num_rows,
            dtype=torch.bool,
            device=k_cache_tensor.device,
        )
        pending_reads_bitmap = torch.zeros(
            num_rows,
            dtype=torch.bool,
            device="cpu",
        )
        resident_blocks_bitmap = torch.zeros_like(pending_reads_bitmap)
        self._layers[int(layer_id)] = CSAAttentionKVLayerState(
            layer_id=int(layer_id),
            compressed_block_size=compressed_block_size,
            token_bytes=token_bytes,
            k_cache_tensor=k_cache_tensor,
            in_pool_bitmap=in_pool_bitmap,
            chunks=[],
            pending_reads_lock=threading.Condition(),
            pending_reads_bitmap=pending_reads_bitmap,
            resident_blocks_bitmap=resident_blocks_bitmap,
            pending_read_count=0,
            last_drain_event=None,
            pending_drains=[],
            block_slot_scatter=block_slot_size > 0,
            block_slot_size=block_slot_size,
        )

    def register_request_chunks(
        self,
        req_id: str,
        chunks_by_layer: Dict[int, List[CSAAttentionKVChunkLoc]],
    ) -> None:
        """Register LMCache chunk locations for an active request.

        The cache engine calls this at retrieve time, after determining
        which chunks contain the request's prefix.  The chunks must cover
        consecutive compressed block ranges starting at 0; gaps trigger a
        warning because they would surface as silent miss-on-fetch later.

        For every distinct ``disk_meta.path`` referenced by the supplied
        chunks, the chunk's ``raw_extents`` are pushed into the Tutti
        loader's ``_lba_cache`` so subsequent ``load_chunks_to_hbm`` calls
        can resolve the synthetic path without FIEMAP.  Re-registering the
        same path is idempotent in Tutti.

        Args:
            req_id: Request identifier (advisory; the manager only keeps
                one active request at a time in the initial implementation).
            chunks_by_layer: Mapping from CSA layer id to ordered chunk
                descriptors.  Empty entries clear the layer's chunk list.
        """
        is_new_request = str(req_id) != self._active_request_id
        self._active_request_id = str(req_id)
        if is_new_request:
            self.start_full_nsys_capture_for_request(str(req_id))
        if is_new_request:
            with self._scheduled_layer_futures_lock:
                self._scheduled_layer_futures.clear()
        if not is_new_request:
            # Second/later chunk-step of the SAME request (multi-step hits:
            # 16K+ increments).  The NVMe-resident prefix is fixed at first
            # registration, so rebuilding chunk maps is pure churn — and it
            # Replacing state.chunks/physical_block_ids while staged reads are
            # in flight would desynchronise their scatter plan. Keep the first
            # registration and only refresh the LBA cache union.
            if self._pending_raw_lba_cache:
                try:
                    self._tutti_loader.register_lba_cache(self._pending_raw_lba_cache)
                except Exception:
                    logger.exception(
                        "CSAAttentionKVPrefetchManager: LBA cache re-register "
                        "failed on repeat registration"
                    )
            return

        # Accumulate every (path, raw_extents) pair carried by the chunk
        # map and push the union into the Tutti loader's LBA cache.  All
        # chunks belonging to the same rank-local pool share one synthetic
        # path (e.g. ``tutti://rank2-full``), but each chunk only carries
        # the raw extents covering ITS slab of the pool.  Naively
        # deduplicating by path would register one chunk's extents and drop
        # every other chunk's coverage — exactly the
        # ``Tutti extents ... cover 0/N bytes`` bug.  Dedup at the
        # (path, file_offset, slba, n_sectors) tuple level instead.
        try:
            from lmcache.v1.gpu_connector.tutti_direct_loader import LbaRecord
        except ImportError:
            LbaRecord = None  # type: ignore[assignment]
        if LbaRecord is not None:
            seen: set[tuple[str, int, int, int]] = set()
            raw_lba_cache: dict[str, list["LbaRecord"]] = {}
            for chunks in chunks_by_layer.values():
                for chunk in chunks:
                    path = chunk.disk_meta.path if chunk.disk_meta else None
                    if not path or not chunk.raw_extents:
                        continue
                    bucket = raw_lba_cache.setdefault(path, [])
                    for file_offset, slba, n_sectors in chunk.raw_extents:
                        key = (
                            path,
                            int(file_offset),
                            int(slba),
                            int(n_sectors),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        bucket.append(
                            LbaRecord(
                                file_offset=int(file_offset),
                                slba=int(slba),
                                n_sectors=int(n_sectors),
                            )
                        )
            if raw_lba_cache:
                # Stash for re-registration on every read (see comment on
                # ``_pending_raw_lba_cache``).
                self._pending_raw_lba_cache = raw_lba_cache
                try:
                    self._tutti_loader.register_lba_cache(raw_lba_cache)
                except Exception:
                    logger.exception(
                        "CSAAttentionKVPrefetchManager: failed to register %d "
                        "LBA cache entries for request %s",
                        len(raw_lba_cache),
                        req_id,
                    )

        for layer_id, state in self._layers.items():
            chunks = chunks_by_layer.get(int(layer_id), [])
            ordered = sorted(chunks, key=lambda chunk: chunk.first_compressed_block)
            # Validate contiguity – the read path assumes any compressed
            # block id falls inside exactly one chunk and chunks cover
            # ``[0, total_compressed_blocks)`` densely.
            expected = 0
            for chunk in ordered:
                if chunk.first_compressed_block != expected:
                    logger.warning(
                        "CSAAttentionKVPrefetchManager: layer %d chunk %s "
                        "starts at compressed_block_id %d, expected %d; "
                        "gap may cause silent miss",
                        layer_id,
                        chunk.key.to_string()
                        if hasattr(chunk.key, "to_string")
                        else repr(chunk.key),
                        chunk.first_compressed_block,
                        expected,
                    )
                expected = chunk.end_compressed_block
            state.chunks = ordered
            self._compile_indexed_layer_tables(state)
            self._compile_layer_major_table(state)
            # Only reset drain and bitmap state for genuinely new requests.
            # Per-microbatch re-registration (same req_id) must preserve the
            # in_pool_bitmap and last_drain_event so staging slots from reads
            # already in flight are not orphaned (leak → pool exhaustion).
            if is_new_request:
                state.in_pool_bitmap.zero_()
                with state.pending_reads_lock:
                    state.pending_reads_bitmap.zero_()
                    state.resident_blocks_bitmap.zero_()
                    state.pending_read_count = 0
                    state.last_drain_event = None

    def start_full_nsys_capture_for_request(self, request_id: str) -> None:
        """Start one full-forward Nsight capture for a selected cache hit.

        Args:
            request_id: Newly registered LMCache request identifier.

        Notes:
            ``LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS`` counts cache-hit
            requests registered by this manager, allowing warmup hits to be
            skipped without tracing cold-store model execution.
        """
        enabled = os.getenv("LMCACHE_NSYS_FULL_CAPTURE", "0").lower() in {
            "1",
            "on",
            "true",
            "yes",
        }
        if not enabled:
            return
        if self._full_nsys_capture_active or self._full_nsys_capture_complete:
            return
        if request_id != self._full_nsys_seen_request_id:
            self._full_nsys_seen_request_id = request_id
            self._full_nsys_seen_requests += 1
        try:
            skip = max(
                0,
                int(os.getenv("LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS", "0")),
            )
        except ValueError:
            skip = 0
        if self._full_nsys_seen_requests <= skip:
            return
        try:
            torch.cuda.profiler.start()
            self._full_nsys_capture_active = True
            logger.info(
                "CSAAttentionKVPrefetchManager: NSYS_FULL_CAPTURE start "
                "request=%s request_index=%d",
                request_id,
                self._full_nsys_seen_requests,
            )
        except Exception:
            self._full_nsys_capture_complete = True
            logger.exception(
                "CSAAttentionKVPrefetchManager: NSYS_FULL_CAPTURE start failed"
            )

    def finish_full_nsys_capture(self) -> None:
        """Stop an active full-forward capture after the final decoder layer."""
        if not self._full_nsys_capture_active:
            return
        try:
            torch.cuda.profiler.stop()
            logger.info(
                "CSAAttentionKVPrefetchManager: NSYS_FULL_CAPTURE stop request=%s",
                self.active_request_id,
            )
        except Exception:
            logger.exception(
                "CSAAttentionKVPrefetchManager: NSYS_FULL_CAPTURE stop failed"
            )
        finally:
            self._full_nsys_capture_active = False
            self._full_nsys_capture_complete = True

    def _compile_indexed_layer_tables(
        self,
        state: CSAAttentionKVLayerState,
    ) -> None:
        """Compile one-range CSA blocks into persistent GPU lookup tables."""
        state.indexed_slba_table = None
        state.indexed_dst_rows_table = None
        if state.block_slot_scatter or not state.chunks:
            return
        io_nbytes = state.compressed_block_size * state.token_bytes
        if io_nbytes <= 0 or io_nbytes % 512:
            return

        table_size = int(state.chunks[-1].end_compressed_block)
        slbas = torch.full((table_size,), -1, dtype=torch.int64)
        dst_rows = torch.full((table_size,), -1, dtype=torch.int64)
        n_rows = int(state.k_cache_tensor.shape[0])
        for chunk in state.chunks:
            if not chunk.raw_extents or chunk.bytes_per_block != io_nbytes:
                return
            # Resolve every block in this chunk as one vector operation. This
            # removes the remaining per-block Python loop even though a layer
            # has only ~1.8K blocks. searchsorted performs the monotonic
            # block-to-extent mapping in native parallel code.
            extent_table = torch.as_tensor(
                chunk.raw_extents,
                dtype=torch.int64,
            ).reshape(-1, 3)
            extent_table = extent_table.index_select(
                0,
                torch.argsort(extent_table[:, 0]),
            )
            extent_offsets = extent_table[:, 0]
            extent_slbas = extent_table[:, 1]
            extent_ends = extent_offsets + extent_table[:, 2] * 512
            block_ids = torch.arange(
                chunk.first_compressed_block,
                chunk.end_compressed_block,
                dtype=torch.int64,
            )
            local_ids = block_ids - int(chunk.first_compressed_block)
            file_offsets = (
                int(chunk.layer_byte_offset)
                + int(chunk.payload_skip)
                + local_ids * io_nbytes
            )
            if bool(torch.any(torch.remainder(file_offsets, 512))):
                return
            extent_ids = torch.searchsorted(
                extent_ends,
                file_offsets,
                right=True,
            )
            if bool(torch.any(extent_ids >= int(extent_table.shape[0]))):
                return
            selected_offsets = extent_offsets.index_select(0, extent_ids)
            selected_ends = extent_ends.index_select(0, extent_ids)
            if bool(
                torch.any(
                    (file_offsets < selected_offsets)
                    | (file_offsets + io_nbytes > selected_ends)
                )
            ):
                # A block crossing an extent remains correct through the
                # general range resolver; the indexed op intentionally
                # handles exactly one NVMe command per selected block.
                return
            chunk_slbas = extent_slbas.index_select(0, extent_ids) + torch.div(
                file_offsets - selected_offsets,
                512,
                rounding_mode="floor",
            )
            if chunk.physical_block_ids:
                if len(chunk.physical_block_ids) != int(block_ids.numel()):
                    return
                chunk_dst_rows = torch.as_tensor(
                    chunk.physical_block_ids,
                    dtype=torch.int64,
                )
            else:
                chunk_dst_rows = block_ids
            if bool(torch.any((chunk_dst_rows < 0) | (chunk_dst_rows >= n_rows))):
                return
            slbas[block_ids] = chunk_slbas
            dst_rows[block_ids] = chunk_dst_rows
        if bool(torch.any(slbas < 0)) or bool(torch.any(dst_rows < 0)):
            return

        device = state.k_cache_tensor.device
        with torch.inference_mode():
            if device.type == "cuda":
                io_stream = self._scatter_stream_for(device)
                with torch.cuda.stream(io_stream):
                    state.indexed_slba_table = slbas.to(device, non_blocking=True)
                    state.indexed_dst_rows_table = dst_rows.to(
                        device,
                        non_blocking=True,
                    )
            else:
                state.indexed_slba_table = slbas.to(device)
                state.indexed_dst_rows_table = dst_rows.to(device)
        logger.info(
            "CSAAttentionKVPrefetchManager: compiled indexed Tutti table "
            "layer=%d blocks=%d bytes_per_block=%d",
            state.layer_id,
            table_size,
            io_nbytes,
        )

    def _compile_layer_major_table(
        self,
        state: CSAAttentionKVLayerState,
    ) -> None:
        """Upload the destination rows for a full-prefix layer object."""
        state.layer_major_dst_rows_table = None
        if len(state.chunks) != 1 or not state.chunks[0].layer_major:
            return
        chunk = state.chunks[0]
        if len(chunk.physical_block_ids) != chunk.n_compressed_blocks:
            return
        state.layer_major_dst_rows_table = torch.as_tensor(
            chunk.physical_block_ids,
            dtype=torch.int64,
            device=state.k_cache_tensor.device,
        )

    def fire_predicted_reads(
        self,
        layer_id: int,
        compressed_block_ids: Sequence[int] | torch.Tensor,
        prefetch_level: int = 2,
    ) -> None:
        """Submit Tutti reads for the predicted ``top-K`` of one CSA layer.

        Called from :class:`IndexerSSDManager` after its HC-proxy emits a
        prediction for ``layer_id``.  Block ids already in the pool or with
        an in-flight read are skipped.

        Args:
            layer_id: Transformer-side CSA layer id.
            compressed_block_ids: Predicted compressed block ids.
            prefetch_level: Must be ``2`` for the one early L2 prediction.

        Raises:
            ValueError: If ``prefetch_level`` is not two.
        """
        if prefetch_level != 2:
            raise ValueError("prefetch_level must be 2")
        self._submit_reads(
            layer_id,
            compressed_block_ids,
            label=f"predicted_l{prefetch_level}",
            io_priority="speculative",
        )

    def set_prediction_waiter(
        self,
        waiter: Optional[Callable[[int], bool]],
    ) -> None:
        """Set the target-layer gate for asynchronous proxy prediction.

        The waiter is invoked only after the official target-layer indexer
        has finished. This preserves overlap between prediction, Tutti I/O,
        and true-indexer compute while ensuring miss correction observes all
        prediction submissions before it decides which blocks to read.

        Args:
            waiter: Callable accepting a transformer layer id and returning
                whether its prediction futures completed, or ``None`` to
                remove the gate.
        """
        self._prediction_waiter = waiter

    def fire_deterministic_layer(self, layer_id: int) -> bool:
        """Submit every covered block for a deterministic HCA layer.

        HCA has no sparse indexer: every compressed entry is consumed by its
        attention.  The preceding layer can therefore move this read into its
        FFN/MoE window without prediction or correctness risk.

        Args:
            layer_id: Registered HCA transformer layer id.

        Returns:
            ``True`` when the layer has a registered non-empty chunk map.
        """
        state = self._layers.get(int(layer_id))
        if state is None or not state.chunks:
            return False
        covered_end = int(state.chunks[-1].end_compressed_block)
        self._submit_reads(
            int(layer_id),
            range(covered_end),
            label="hca_deterministic",
            raise_on_error=True,
        )
        return True

    def track_layer_submission(self, layer_id: int, future: Any) -> None:
        """Track a background submission that must precede a layer gate.

        Args:
            layer_id: Target transformer layer id.
            future: Future whose result means Tutti/scatter work was enqueued.
        """
        with self._scheduled_layer_futures_lock:
            self._scheduled_layer_futures[int(layer_id)] = future

    def layer_submission_ready(self, layer_id: int) -> bool:
        """Return whether a tracked background layer submission is complete."""
        with self._scheduled_layer_futures_lock:
            future = self._scheduled_layer_futures.get(int(layer_id))
        return future is None or bool(future.done())

    def submit_miss_reads(
        self,
        layer_id: int,
        compressed_block_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Submit Tutti reads for blocks not covered by the prediction.

        Called from the patched ``DeepseekV4Indexer.forward`` after the true
        Lightning Indexer returns ``true_topk``.

        Args:
            layer_id: Transformer-side CSA layer id.
            compressed_block_ids: ``true_topk`` block ids whose K cache
                slots are not yet populated.
        """
        self._submit_reads(
            layer_id,
            compressed_block_ids,
            label="miss",
            raise_on_error=True,
        )

    def wait_for_layer(self, layer_id: int, timeout_s: float = 2.0) -> bool:
        """Block until scheduled and in-flight reads for a layer complete.

        HCA layers have no indexer forward to
        patch, so the vLLM connector's per-layer ``wait_for_layer_load``
        hook calls this before the layer's attention runs.

        Args:
            layer_id: Transformer-side layer id.
            timeout_s: Upper bound on the wait.  On expiry the method
                returns ``False`` and the caller must treat the layer as
                potentially stale.

        Returns:
            ``True`` when the layer has no pending reads (fully landed or
            never registered), ``False`` on timeout.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._scheduled_layer_futures_lock:
            scheduled = self._scheduled_layer_futures.pop(int(layer_id), None)
        if scheduled is not None:
            try:
                scheduled.result(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                logger.exception(
                    "CSAAttentionKVPrefetchManager: scheduled layer %d "
                    "submission failed or timed out",
                    layer_id,
                )
                return False
        state = self._layers.get(int(layer_id))
        if state is None:
            return True
        with state.pending_reads_lock:
            while state.pending_read_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "CSAAttentionKVPrefetchManager: wait_for_layer %d "
                        "timed out with %d reads still pending",
                        layer_id,
                        state.pending_read_count,
                    )
                    return False
                state.pending_reads_lock.wait(remaining)
        self.drain_for_layer(int(layer_id))
        return True

    def drain_for_layer(self, layer_id: int) -> None:
        """Block until all pending reads for ``layer_id`` have completed.

        Records the latest CUDA event into ``last_drain_event`` so callers
        can also do a non-blocking wait via stream synchronisation if
        desired.

        Args:
            layer_id: Transformer-side CSA layer id.
        """
        state = self._layers.get(int(layer_id))
        if state is None:
            return
        with state.pending_reads_lock:
            event = state.last_drain_event
            # Take a snapshot and clear pending_drains atomically with reading
            # last_drain_event. Any (ev, objs, range, op_id) tuple in
            # pending_drains was
            # appended together with setting last_drain_event = ev (under the
            # same lock), so synchronizing last_drain_event covers all of them.
            pending = state.pending_drains[:]
            state.pending_drains.clear()
        if event is not None:
            with csa_pipeline_nvtx.range(
                CsaNvtxEvent.TARGET_GATE_WAIT,
                layer_id=int(layer_id),
                request_id=self.active_request_id,
            ):
                event.synchronize()
        # Release staging buffers now that the CUDA stream has confirmed all
        # non_blocking copies into the K cache have completed.
        for _, memory_objs, io_range, operation_id in pending:
            csa_pipeline_nvtx.finish_io(
                io_range,
                layer_id=int(layer_id),
                target_layer_id=int(layer_id),
                operation_id=operation_id,
                request_id=self.active_request_id,
            )
            for memory_obj in memory_objs:
                ref_count_down = getattr(memory_obj, "ref_count_down", None)
                if callable(ref_count_down):
                    ref_count_down()

    def patch_indexer_forward(
        self,
        indexer_module: Any,
        layer_id: int,
    ) -> None:
        """Monkey-patch one CSA :class:`DeepseekV4Indexer`'s ``forward``.

        The wrapped forward delegates to the original Lightning Indexer to
        get ``true_topk``, computes the miss set against the pool, submits
        miss reads, drains, and returns ``true_topk`` unchanged so the
        downstream sparse attention kernel reads correct K bytes.

        Args:
            indexer_module: The :class:`DeepseekV4Indexer` instance.
            layer_id: CSA layer id this indexer belongs to.

        Raises:
            RuntimeError: If the module has already been patched by this
                manager.
        """
        with self._patch_lock:
            for module, attr, _orig in self._patched_modules:
                if module is indexer_module and attr == "forward":
                    raise RuntimeError(
                        f"indexer module {indexer_module!r} already patched"
                    )
            orig_forward = indexer_module.forward
            mgr = self

            def _patched_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                t0 = time.perf_counter() if _timing_enabled() else 0.0
                wait_ms = 0.0
                # Never wait for speculative proxy work before the
                # official Lightning Indexer.  Running true scoring first
                # overlaps its GPU work with pending I/O and preserves the
                # exact target-layer semantics on every request.
                reused_residual = False
                true_indexer_start = time.perf_counter() if _timing_enabled() else 0.0
                with csa_pipeline_nvtx.range(
                    CsaNvtxEvent.TRUE_INDEXER,
                    layer_id=layer_id,
                    request_id=mgr.active_request_id,
                ):
                    true_topk = orig_forward(*args, **kwargs)
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None and args:
                    hidden_states = args[0]
                active_rows = (
                    min(int(hidden_states.shape[0]), int(true_topk.shape[0]))
                    if isinstance(hidden_states, torch.Tensor) and true_topk.ndim >= 2
                    else int(true_topk.shape[0])
                    if true_topk.ndim >= 2
                    else 1
                )
                active_topk = (
                    true_topk[:active_rows] if true_topk.ndim >= 2 else true_topk
                )
                true_indexer_ms = (
                    (time.perf_counter() - true_indexer_start) * 1000.0
                    if _timing_enabled()
                    else 0.0
                )
                try:
                    # Prediction remains asynchronous through the complete
                    # official indexer. Join only now, immediately before
                    # miss filtering needs a stable pending/resident view.
                    prediction_wait_start = (
                        time.perf_counter() if _timing_enabled() else 0.0
                    )
                    waiter = mgr._prediction_waiter
                    prediction_ready = waiter is None or waiter(layer_id)
                    wait_ms = (
                        (time.perf_counter() - prediction_wait_start) * 1000.0
                        if _timing_enabled()
                        else 0.0
                    )
                    # The true Lightning Indexer output is the live source of
                    # truth. Record accuracy only after the asynchronous proxy
                    # has joined, so its block set is complete and stable.
                    from lmcache.v1.indexer_ssd_manager import (
                        get_indexer_ssd_manager,
                    )

                    indexer_manager = get_indexer_ssd_manager()
                    if indexer_manager is not None:
                        indexer_manager.record_csa_prediction_accuracy(
                            layer_id,
                            active_topk,
                        )
                    first_drain_ms = 0.0
                    t_miss0 = time.perf_counter() if _timing_enabled() else 0.0
                    miss_ids = mgr._miss_ids_for_topk(layer_id, active_topk)
                    miss_ms = (
                        (time.perf_counter() - t_miss0) * 1000.0
                        if _timing_enabled()
                        else 0.0
                    )
                    if miss_ids.numel():
                        mgr.submit_miss_reads(layer_id, miss_ids)
                    t_drain1 = time.perf_counter() if _timing_enabled() else 0.0
                    mgr.drain_for_layer(layer_id)
                    second_drain_ms = (
                        (time.perf_counter() - t_drain1) * 1000.0
                        if _timing_enabled()
                        else 0.0
                    )
                    if indexer_manager is not None:
                        indexer_manager.finish_nsys_capture_for_layer(layer_id)
                    if _timing_enabled():
                        logger.info(
                            "CSAAttentionKVPrefetchManager: correction "
                            "device=%s layer=%d active_query_rows=%d "
                            "raw_selected_entries=%d miss_blocks=%d "
                            "reused_residual=%d "
                            "prediction_ready=%d true_indexer_ms=%.3f wait_ms=%.3f "
                            "first_drain_ms=%.3f "
                            "miss_filter_ms=%.3f second_drain_ms=%.3f "
                            "total_ms=%.3f",
                            str(true_topk.device),
                            layer_id,
                            active_rows,
                            int(active_topk.numel()),
                            len(miss_ids),
                            int(reused_residual),
                            int(prediction_ready),
                            true_indexer_ms,
                            wait_ms,
                            first_drain_ms,
                            miss_ms,
                            second_drain_ms,
                            (time.perf_counter() - t0) * 1000.0,
                        )
                except Exception as exc:
                    logger.exception(
                        "CSAAttentionKVPrefetchManager: miss correction failed "
                        "for layer %d; aborting request before attention",
                        layer_id,
                    )
                    raise RuntimeError(
                        f"CSA attention KV correction failed for layer {layer_id}"
                    ) from exc
                return true_topk

            indexer_module.forward = _patched_forward
            indexer_module._lmcache_csa_attention_kv_original_forward = orig_forward
            self._patched_modules.append((indexer_module, "forward", orig_forward))

    def unpatch(self) -> None:
        """Restore all patched indexer forwards."""
        with self._patch_lock:
            for module, attr, original in self._patched_modules:
                setattr(module, attr, original)
                try:
                    delattr(module, "_lmcache_csa_attention_kv_original_forward")
                except AttributeError:
                    pass
            self._patched_modules.clear()

    def close(self) -> None:
        """Release resources held by this manager."""
        if self._closed:
            return
        self._closed = True
        self.unpatch()
        self._layers.clear()
        self._active_request_id = None
        self._prediction_waiter = None
        with self._scheduled_layer_futures_lock:
            self._scheduled_layer_futures.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit_reads(
        self,
        layer_id: int,
        compressed_block_ids: Sequence[int] | torch.Tensor,
        label: str,
        io_priority: str = "demand",
        *,
        raise_on_error: bool = False,
    ) -> bool:
        """Submit reads and publish only blocks that actually completed.

        Args:
            layer_id: Target transformer layer id.
            compressed_block_ids: Logical block ids requested by prediction
                or exact correction.
            label: Profiling label for the submission.
            io_priority: ``demand`` requires complete execution;
                ``speculative`` permits a cancelled or partial result.
            raise_on_error: Re-raise submission and completion failures.

        Returns:
            ``True`` when the submission itself completed safely.  A
            speculative partial completion is safe and returns ``True``;
            only its actual completed ids become resident.

        Raises:
            RuntimeError: If strict error propagation is requested and the
                layer is unavailable or its I/O fails.
            ValueError: If ``io_priority`` is invalid.
        """
        if io_priority not in {"demand", "speculative"}:
            raise ValueError("io_priority must be demand or speculative")
        state = self._layers.get(int(layer_id))
        if state is None:
            if raise_on_error:
                raise RuntimeError(f"layer {layer_id} is not registered")
            return False
        candidate_ids = torch.as_tensor(
            compressed_block_ids,
            dtype=torch.int64,
            device="cpu",
        ).reshape(-1)
        if candidate_ids.numel() == 0:
            return True
        candidate_count = int(candidate_ids.numel())
        if _timing_enabled():
            logger.info(
                "CSAAttentionKVPrefetchManager: _submit_reads label=%s "
                "layer=%d candidates=%d",
                label,
                layer_id,
                candidate_count,
            )
        if not state.chunks:
            if raise_on_error:
                raise RuntimeError(f"layer {layer_id} has no read plan")
            logger.warning(
                "CSAAttentionKVPrefetchManager: no chunks registered for "
                "layer %d at %s submission; skipping %d ids",
                layer_id,
                label,
                candidate_count,
            )
            return False

        # Decide which block ids are net-new without a GPU round trip. All
        # per-id work is expressed as native tensor operations, including for
        # the small (~1.8K) one-layer set; Python only submits the whole batch.
        pool_size = int(state.in_pool_bitmap.shape[0])
        # Clamp to the chunk map's covered compressed-block range as well:
        # the HC-proxy prediction routinely emits ids past the cached prefix
        # (short/partial prefixes cover only the first few blocks), and
        # ``_issue_reads`` cannot read uncovered ids.  The miss path already
        # drops them in ``_miss_ids_for_topk``; without this clamp the
        # predicted path books uncovered ids into pending state and
        # floods per-id "not covered by any chunk" warnings.
        covered_end = int(state.chunks[-1].end_compressed_block)
        limit = min(pool_size, covered_end)
        candidate_ids = torch.unique(
            candidate_ids[(candidate_ids >= 0) & (candidate_ids < limit)],
            sorted=True,
        )
        accepted_count = int(candidate_ids.numel())
        if accepted_count < candidate_count and _timing_enabled():
            logger.info(
                "CSAAttentionKVPrefetchManager: _submit_reads label=%s "
                "layer=%d dropped %d/%d ids outside covered range [0, %d)",
                label,
                layer_id,
                candidate_count - accepted_count,
                candidate_count,
                limit,
            )
        if candidate_ids.numel() == 0:
            return True
        with state.pending_reads_lock:
            # Exact-lookahead scheduling has one predictor per target. Miss
            # correction waits for predicted in-flight blocks at the final
            # drain instead of promoting or resubmitting them.
            new_ids = candidate_ids[
                ~state.resident_blocks_bitmap[candidate_ids]
                & ~state.pending_reads_bitmap[candidate_ids]
            ]
            if new_ids.numel() == 0:
                if _timing_enabled():
                    logger.info(
                        "CSAAttentionKVPrefetchManager: _submit_reads "
                        "label=%s layer=%d new=0 reason=resident_or_pending",
                        label,
                        layer_id,
                    )
                return True
            # These request bitmaps are allocated while vLLM is inside
            # inference mode. Predicted I/O completes on a background thread,
            # so explicitly re-enter inference mode for every in-place update.
            with torch.inference_mode():
                state.pending_reads_bitmap[new_ids] = True
            state.pending_read_count += int(new_ids.numel())
        new_count = int(new_ids.numel())
        if _timing_enabled():
            logger.info(
                "CSAAttentionKVPrefetchManager: _submit_reads label=%s layer=%d new=%d",
                label,
                layer_id,
                new_count,
            )

        operation_id = f"{label}-{time.monotonic_ns()}"
        io_range = csa_pipeline_nvtx.start_io(
            layer_id=int(layer_id),
            target_layer_id=int(layer_id),
            operation_id=operation_id,
            request_id=self.active_request_id,
            attributes={"kind": label, "blocks": new_count},
        )
        try:
            # Predicted reads often finish on a background thread, outside
            # vLLM's model-forward inference_mode context.  The target K cache
            # and resident bitmap may be inference tensors, so all in-place GPU
            # updates must re-enter inference_mode here as well.
            with torch.inference_mode():
                event, issued_memory_objs, completed_ids = self._issue_reads(
                    state,
                    new_ids,
                    io_priority=io_priority,
                )
                completed_ids = torch.unique(
                    completed_ids.to(device="cpu", dtype=torch.int64),
                    sorted=True,
                )
                if io_priority == "demand" and not torch.equal(
                    completed_ids,
                    new_ids,
                ):
                    raise RuntimeError(
                        f"demand read completed {int(completed_ids.numel())}/"
                        f"{new_count} blocks for layer {layer_id}"
                    )
        except Exception as exc:
            csa_pipeline_nvtx.finish_io(
                io_range,
                layer_id=int(layer_id),
                target_layer_id=int(layer_id),
                operation_id=operation_id,
                request_id=self.active_request_id,
                status="error",
            )
            with state.pending_reads_lock:
                with torch.inference_mode():
                    state.pending_reads_bitmap[new_ids] = False
                state.pending_read_count = max(
                    0,
                    state.pending_read_count - new_count,
                )
                state.pending_reads_lock.notify_all()
            logger.exception(
                "CSAAttentionKVPrefetchManager: failed to issue %s reads for "
                "layer %d (%d ids)",
                label,
                layer_id,
                new_count,
            )
            if raise_on_error:
                raise RuntimeError(f"failed to materialize layer {layer_id}") from exc
            return False
        with state.pending_reads_lock:
            # Publish the CPU resident view before clearing pending state.
            # The K-cache scatter and GPU bitmap update are both ordered by
            # ``event``; the target layer waits for that event at its gate.
            if completed_ids.numel():
                with torch.inference_mode():
                    state.resident_blocks_bitmap[completed_ids] = True
            with torch.inference_mode():
                state.pending_reads_bitmap[new_ids] = False
            state.pending_read_count = max(
                0,
                state.pending_read_count - new_count,
            )
            state.last_drain_event = event
            # Store (event, memory_objs) atomically with last_drain_event so
            # drain_for_layer can safely release buffers after synchronizing.
            state.pending_drains.append(
                (event, issued_memory_objs, io_range, operation_id)
            )
            # Wake any miss-correction waiting for these blocks to land.
            state.pending_reads_lock.notify_all()
        return True

    def _issue_reads(
        self,
        state: CSAAttentionKVLayerState,
        sorted_block_ids: Sequence[int] | torch.Tensor,
        *,
        io_priority: str = "demand",
    ) -> Tuple[Optional[torch.cuda.Event], List[Any], torch.Tensor]:
        """Issue Tutti reads grouped by chunk and copy bytes into the K cache.

        Each LMCache chunk that the block ids touch becomes one
        ``CacheEngineKey`` in the ``load_chunks_to_hbm`` call.  Per-key
        ``read_ranges_per_key`` entries select only the bytes for the
        actually-needed blocks.  After Tutti returns the GPU-resident
        :class:`MemoryObj`, the bytes are copied into the registered K
        cache tensor at the correct ``[block_idx, *, *]`` rows.

        Args:
            state: Per-layer state with chunk map and K cache tensor.
            sorted_block_ids: Block ids to read, ascending.  The caller
                already deduplicated against the pool and pending sets.
            io_priority: Tutti admission class for the I/O submission.

        Returns:
            Tuple of the final scatter event, retained staging objects, and
            the CPU block ids whose bytes were actually scattered.
            The indexed path returns an event so the target layer waits only
            at its consumption gate instead of synchronizing during prefetch.
        """
        if len(sorted_block_ids) == 0:
            return None, [], torch.empty(0, dtype=torch.int64)
        if (
            len(state.chunks) == 1
            and state.chunks[0].layer_major
            and state.layer_major_dst_rows_table is not None
        ):
            selected = torch.as_tensor(
                sorted_block_ids,
                dtype=torch.int64,
                device="cpu",
            ).reshape(-1)
            n_blocks = state.chunks[0].n_compressed_blocks
            if int(selected.numel()) == n_blocks and torch.equal(
                selected,
                torch.arange(n_blocks, dtype=torch.int64),
            ):
                return self._issue_layer_major_full_read(
                    state,
                    io_priority=io_priority,
                )
        if (
            state.indexed_slba_table is not None
            and state.indexed_dst_rows_table is not None
            and _csa_c_ops is not None
            and hasattr(_csa_c_ops, "tutti_submit_indexed_sgl_read")
            and hasattr(self._tutti_loader, "load_indexed_chunks_to_hbm")
        ):
            return self._issue_indexed_reads(
                state,
                sorted_block_ids,
                io_priority=io_priority,
            )
        if isinstance(sorted_block_ids, torch.Tensor):
            sorted_block_ids = sorted_block_ids.tolist()
        # Group block ids by chunk and coalesce consecutive ids per chunk.
        chunks_used: Dict[int, Tuple[CSAAttentionKVChunkLoc, List[int]]] = {}
        chunk_iter = iter(enumerate(state.chunks))
        current_chunk_idx, current_chunk = next(chunk_iter, (None, None))
        for block_id in sorted_block_ids:
            while (
                current_chunk is not None
                and block_id >= current_chunk.end_compressed_block
            ):
                current_chunk_idx, current_chunk = next(chunk_iter, (None, None))
            if current_chunk is None or not current_chunk.contains(block_id):
                logger.warning(
                    "CSAAttentionKVPrefetchManager: layer %d block_id %d not "
                    "covered by any chunk; skipping",
                    state.layer_id,
                    block_id,
                )
                continue
            existing = chunks_used.setdefault(
                current_chunk_idx,
                (current_chunk, []),
            )
            existing[1].append(block_id)

        keys: List[CacheEngineKey] = []
        disk_metas: List[Optional[DiskCacheMetadata]] = []
        file_offsets: List[int] = []
        read_ranges_per_key: List[Optional[Tuple[KVObjectByteRange, ...]]] = []
        # Vectorized-scatter plan: staging bytes for each chunk are densely
        # packed in sorted-block order (cursor advances bytes_per_block per
        # block), so the whole chunk is one [n_blocks, block_size,
        # token_bytes] tensor.  Precompute every block's destination K-cache
        # row here (CPU, cheap) so the hot callback is a single index_copy_
        # per chunk with zero per-block Python work.
        dst_rows_all: List[int] = []
        comp_ids_all: List[int] = []
        completed_comp_ids: List[int] = []
        chunk_row_spans: List[Tuple[int, int]] = []
        chunk_payload_skips: List[int] = []

        for chunk_idx in sorted(chunks_used.keys()):
            chunk, ids_in_chunk = chunks_used[chunk_idx]
            ranges: List[KVObjectByteRange] = []
            cursor = 0
            sorted_ids_in_chunk = sorted(ids_in_chunk)
            span_start = len(dst_rows_all)
            if state.block_slot_scatter:
                n_rows = int(state.k_cache_tensor.shape[0]) * state.block_slot_size
            else:
                n_rows = int(state.k_cache_tensor.shape[0])
            for block_id in sorted_ids_in_chunk:
                if chunk.physical_block_ids:
                    local_idx = int(block_id) - chunk.first_compressed_block
                    if 0 <= local_idx < len(chunk.physical_block_ids):
                        dst_row = int(chunk.physical_block_ids[local_idx])
                    else:
                        dst_row = int(block_id)
                else:
                    dst_row = int(block_id)
                if not 0 <= dst_row < n_rows:
                    logger.warning(
                        "CSAAttentionKVPrefetchManager: dropping write for "
                        "layer %d compressed_block_id=%d dst_row=%d outside "
                        "k_cache_tensor[0:%d]",
                        state.layer_id,
                        int(block_id),
                        dst_row,
                        n_rows,
                    )
                    continue
                dst_rows_all.append(dst_row)
                comp_ids_all.append(int(block_id))
            complete_chunk = (
                len(sorted_ids_in_chunk) == chunk.n_compressed_blocks
                and sorted_ids_in_chunk[0] == chunk.first_compressed_block
                and sorted_ids_in_chunk[-1] == chunk.end_compressed_block - 1
            )
            if complete_chunk and chunk.read_length > 0:
                ranges.append(
                    KVObjectByteRange(
                        offset=int(chunk.layer_byte_offset),
                        length=int(chunk.read_length),
                        target_offset=0,
                    )
                )
                cursor = int(chunk.read_length)
                payload_skip = int(chunk.payload_skip)
            else:
                run_start: Optional[int] = None
                run_length = 0
                for block_id in sorted_ids_in_chunk + [None]:
                    if run_start is None:
                        if block_id is None:
                            continue
                        run_start = int(block_id)
                        run_length = 1
                        continue
                    if block_id is not None and int(block_id) == run_start + run_length:
                        run_length += 1
                        continue
                    first_offset = chunk.chunk_byte_offset_for(run_start)
                    length = run_length * chunk.bytes_per_block
                    ranges.append(
                        KVObjectByteRange(
                            offset=first_offset,
                            length=length,
                            target_offset=cursor,
                        )
                    )
                    cursor += length
                    if block_id is None:
                        break
                    run_start = int(block_id)
                    run_length = 1
                payload_skip = 0
            keys.append(chunk.key)
            disk_metas.append(chunk.disk_meta)
            file_offsets.append(0)
            read_ranges_per_key.append(tuple(ranges))
            chunk_row_spans.append((span_start, len(dst_rows_all)))
            chunk_payload_skips.append(payload_skip)

        if not keys:
            return None, []

        # Re-register our full-record LBA extents.  ``_tutti_batched_get``
        # in the retrieve path overwrites ``_lba_cache`` with extents that
        # respect ``record.read_ranges``, which excludes csa_attention_kv
        # when the filter is enabled.  Without this re-registration, Tutti
        # reports "extents cover 0/N bytes" on every csa byte_range because
        # the cache only covers the prior groups.
        if self._pending_raw_lba_cache:
            try:
                self._tutti_loader.register_lba_cache(self._pending_raw_lba_cache)
            except Exception:
                logger.exception(
                    "CSAAttentionKVPrefetchManager: re-register LBA cache "
                    "failed in _issue_reads"
                )

        # Vectorized streaming scatter: each staging batch is consumed inside
        # Tutti's on_batch_loaded callback with ONE index_copy_ kernel per
        # chunk.  The staging layout guarantees the chunk's blocks are densely
        # packed in sorted order, so the whole chunk is a single
        # [n_blocks, blk_bytes] source tensor; destination rows were resolved
        # to physical K-cache indices at plan time and uploaded to the GPU
        # once for the entire call.  No per-block Python, no per-block kernel
        # launches.
        #
        # Copies run on a private scatter stream and only that stream is
        # synchronized before the callback returns (staging slots are
        # recycled immediately after).  Never the default stream: a fire
        # thread waiting on forward collectives while holding _io_lock is
        # the cross-rank deadlock we already debugged once.
        blk_bytes = state.compressed_block_size * state.token_bytes
        if state.block_slot_scatter:
            n_rows = int(state.k_cache_tensor.shape[0]) * state.block_slot_size
            k_cache_flat = state.k_cache_tensor.view(torch.uint8)
        else:
            n_rows = int(state.k_cache_tensor.shape[0])
            # Byte view so index_copy_ sees identical dtype/shape on both
            # sides regardless of the K cache's declared element type.
            k_cache_flat = state.k_cache_tensor.view(torch.uint8).reshape(n_rows, -1)
        scatter_stream = self._scatter_stream_for(state.k_cache_tensor.device)
        with torch.inference_mode():
            # Upload on the same stream that consumes it so the destination
            # rows cannot race the scatter operation.
            with torch.cuda.stream(scatter_stream):
                dst_rows_gpu = torch.as_tensor(
                    dst_rows_all,
                    dtype=torch.int64,
                ).to(state.k_cache_tensor.device, non_blocking=True)
            scatter_stream.synchronize()

        def _scatter_batch(
            batch_start: int,
            batch_results: List[Optional[Any]],
        ) -> None:
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                for offset_in_batch, memory_obj in enumerate(batch_results):
                    key_index = batch_start + offset_in_batch
                    if memory_obj is None:
                        raise RuntimeError(
                            "load_chunks_to_hbm returned no payload for CSA "
                            "attention KV read"
                        )
                    tensor = memory_obj.raw_tensor
                    if tensor is None:
                        raise RuntimeError(
                            "load_chunks_to_hbm returned MemoryObj without raw_tensor"
                        )
                    span_start, span_end = chunk_row_spans[key_index]
                    n_blocks = span_end - span_start
                    if n_blocks <= 0:
                        continue
                    payload_skip = chunk_payload_skips[key_index]
                    flat = tensor.view(torch.uint8).reshape(-1)[payload_skip:]
                    usable = min(n_blocks, int(flat.numel()) // blk_bytes)
                    if usable < n_blocks:
                        # Short Tutti read (partial chunk clipped at
                        # aligned_length); scatter what arrived.
                        logger.warning(
                            "CSAAttentionKVPrefetchManager: staging buffer "
                            "short for layer %d chunk %d (have %d/%d blocks)",
                            state.layer_id,
                            key_index,
                            usable,
                            n_blocks,
                        )
                    if usable <= 0:
                        continue
                    src = flat[: usable * blk_bytes].view(usable, blk_bytes)
                    rows = dst_rows_gpu[span_start : span_start + usable]
                    if state.block_slot_scatter:
                        slot_size = state.block_slot_size
                        blocks_idx = torch.div(rows, slot_size, rounding_mode="floor")
                        slots_idx = rows - blocks_idx * slot_size
                        k_cache_flat.index_put_((blocks_idx, slots_idx), src)
                    else:
                        k_cache_flat.index_copy_(0, rows, src)
                    completed_comp_ids.extend(
                        comp_ids_all[span_start : span_start + usable]
                    )
                # Staging slots are recycled as soon as this callback returns;
                # retire the scatter kernels first (private stream only).
            scatter_stream.synchronize()

        raw_batch_enabled = _csa_c_ops is not None and hasattr(
            _csa_c_ops, "scatter_rows_from_object_ptrs"
        )

        def _scatter_raw_batch(
            batch_start: int,
            completed_indices: List[int],
            completed_offsets: List[int],
            completed_nbytes: List[int],
            staging: torch.Tensor,
        ) -> None:
            if _csa_c_ops is None:
                raise RuntimeError("raw Tutti scatter requires lmcache.c_ops")
            materialize = getattr(
                _csa_c_ops,
                "scatter_rows_from_object_ptrs",
                None,
            )
            if materialize is None:
                raise RuntimeError("lmcache.c_ops lacks scatter_rows_from_object_ptrs")

            # Each selected chunk's possibly-disjoint SSD ranges were packed
            # densely by target_offset in plan order. Adjacent chunks with the
            # same selected-row count can therefore share one pointer-batched
            # scatter launch without requiring physically contiguous I/O.
            raw_runs: List[Tuple[int, List[int], int, int]] = []
            staging_base = int(staging.data_ptr())
            for local_index, chunk_offset, nbytes in zip(
                completed_indices,
                completed_offsets,
                completed_nbytes,
                strict=True,
            ):
                key_index = batch_start + local_index
                span_start, span_end = chunk_row_spans[key_index]
                planned_rows = span_end - span_start
                payload_skip = chunk_payload_skips[key_index]
                usable = min(
                    planned_rows,
                    max(0, int(nbytes) - payload_skip) // blk_bytes,
                )
                if usable <= 0:
                    continue
                completed_comp_ids.extend(
                    comp_ids_all[span_start : span_start + usable]
                )
                source_ptr = staging_base + int(chunk_offset) + payload_skip
                usable_end = span_start + usable
                if raw_runs:
                    last_rows, last_ptrs, last_start, last_end = raw_runs[-1]
                    if last_rows == usable and last_end == span_start:
                        last_ptrs.append(source_ptr)
                        raw_runs[-1] = (
                            last_rows,
                            last_ptrs,
                            last_start,
                            usable_end,
                        )
                        continue
                raw_runs.append(
                    (
                        usable,
                        [source_ptr],
                        span_start,
                        usable_end,
                    )
                )

            pointer_tensors: List[torch.Tensor] = []
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                for rows_per_object, source_ptrs, span_start, span_end in raw_runs:
                    source_ptrs_gpu = torch.as_tensor(
                        source_ptrs,
                        dtype=torch.int64,
                    ).to(state.k_cache_tensor.device, non_blocking=True)
                    pointer_tensors.append(source_ptrs_gpu)
                    materialize(
                        source_ptrs_gpu,
                        k_cache_flat,
                        dst_rows_gpu[span_start:span_end],
                        rows_per_object,
                        blk_bytes,
                        state.block_slot_size if state.block_slot_scatter else 0,
                        all(pointer % 8 == 0 for pointer in source_ptrs),
                    )
            scatter_stream.synchronize()

        load_kwargs: Dict[str, Any] = {
            "shapes_per_key": None,
            "file_offsets": file_offsets,
            "read_ranges_per_key": read_ranges_per_key,
            "io_priority": io_priority,
            # A predicted union can span hundreds of tiny ranges. Release the
            # single Tutti queue between bounded speculative batches so HCA
            # and true-topK demand reads never queue behind the whole walk.
            "lock_per_batch": io_priority == "speculative",
        }
        if raw_batch_enabled:
            load_kwargs["on_raw_batch_loaded"] = _scatter_raw_batch
        else:
            load_kwargs["on_batch_loaded"] = _scatter_batch
        self._tutti_loader.load_chunks_to_hbm(
            keys,
            disk_metas,
            **load_kwargs,
        )
        completed_ids = torch.as_tensor(
            completed_comp_ids,
            dtype=torch.int64,
        )
        if completed_ids.numel():
            completed_ids = torch.unique(completed_ids, sorted=True)
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                bitmap_ids = completed_ids.to(
                    device=state.in_pool_bitmap.device,
                    non_blocking=True,
                )
                state.in_pool_bitmap.index_fill_(0, bitmap_ids, True)
            scatter_stream.synchronize()
        # All copies are already synchronized; no deferred staging buffers to
        # release and no drain event is needed.
        return None, [], completed_ids

    def _issue_layer_major_full_read(
        self,
        state: CSAAttentionKVLayerState,
        *,
        io_priority: str,
    ) -> Tuple[Optional[torch.cuda.Event], List[Any], torch.Tensor]:
        """Read and scatter one complete layer-major attention KV object."""
        chunk = state.chunks[0]
        dst_rows = state.layer_major_dst_rows_table
        if dst_rows is None:
            raise RuntimeError("layer-major destination rows are unavailable")
        n_blocks = int(chunk.n_compressed_blocks)
        block_nbytes = int(chunk.bytes_per_block)
        payload_nbytes = n_blocks * block_nbytes
        read_nbytes = int(chunk.read_length or payload_nbytes)
        if payload_nbytes <= 0 or read_nbytes < payload_nbytes:
            raise RuntimeError("invalid layer-major read length")

        if self._pending_raw_lba_cache:
            self._tutti_loader.register_lba_cache(self._pending_raw_lba_cache)

        if state.block_slot_scatter:
            k_cache = state.k_cache_tensor.view(torch.uint8)
        else:
            k_cache = state.k_cache_tensor.view(torch.uint8).reshape(
                int(state.k_cache_tensor.shape[0]),
                -1,
            )
        scatter_stream = self._scatter_stream_for(state.k_cache_tensor.device)
        completed = False

        def _scatter_tensor_batch(
            _batch_start: int,
            batch_results: List[Optional[Any]],
        ) -> None:
            nonlocal completed
            if not batch_results or batch_results[0] is None:
                raise RuntimeError("layer-major Tutti read returned no payload")
            memory_obj = batch_results[0]
            assert memory_obj is not None
            tensor = memory_obj.raw_tensor
            if tensor is None:
                raise RuntimeError("layer-major Tutti read has no raw tensor")
            source = (
                tensor.view(torch.uint8)
                .reshape(-1)[:payload_nbytes]
                .view(
                    n_blocks,
                    block_nbytes,
                )
            )
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                if state.block_slot_scatter:
                    slot_size = state.block_slot_size
                    block_ids = torch.div(
                        dst_rows,
                        slot_size,
                        rounding_mode="floor",
                    )
                    slot_ids = dst_rows - block_ids * slot_size
                    k_cache.index_put_((block_ids, slot_ids), source)
                else:
                    k_cache.index_copy_(0, dst_rows, source)
            scatter_stream.synchronize()
            completed = True

        def _scatter_raw_batch(
            _batch_start: int,
            completed_indices: List[int],
            completed_offsets: List[int],
            completed_nbytes: List[int],
            staging: torch.Tensor,
        ) -> None:
            nonlocal completed
            if not completed_indices:
                raise RuntimeError("layer-major raw Tutti read did not complete")
            if int(completed_nbytes[0]) < payload_nbytes:
                raise RuntimeError(
                    "layer-major raw Tutti read returned a short payload"
                )
            if _csa_c_ops is None:
                raise RuntimeError("layer-major raw scatter requires lmcache.c_ops")
            scatter = getattr(
                _csa_c_ops,
                "scatter_rows_from_object_ptrs",
                None,
            )
            if scatter is None:
                raise RuntimeError("layer-major fused scatter op is unavailable")
            source_ptr = int(staging.data_ptr()) + int(completed_offsets[0])
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                source_ptrs = torch.tensor(
                    [source_ptr],
                    dtype=torch.int64,
                    device=state.k_cache_tensor.device,
                )
                scatter(
                    source_ptrs,
                    k_cache,
                    dst_rows,
                    n_blocks,
                    block_nbytes,
                    state.block_slot_size if state.block_slot_scatter else 0,
                    source_ptr % 8 == 0,
                )
            scatter_stream.synchronize()
            completed = True

        load_kwargs: Dict[str, Any] = {
            "shapes_per_key": None,
            "file_offsets": [0],
            "read_ranges_per_key": [
                (
                    KVObjectByteRange(
                        offset=int(chunk.layer_byte_offset),
                        length=read_nbytes,
                        target_offset=0,
                    ),
                )
            ],
            "io_priority": io_priority,
        }
        if _csa_c_ops is not None and hasattr(
            _csa_c_ops, "scatter_rows_from_object_ptrs"
        ):
            load_kwargs["on_raw_batch_loaded"] = _scatter_raw_batch
        else:
            load_kwargs["on_batch_loaded"] = _scatter_tensor_batch
        self._tutti_loader.load_chunks_to_hbm(
            [chunk.key],
            [chunk.disk_meta],
            **load_kwargs,
        )
        completed_ids = (
            torch.arange(n_blocks, dtype=torch.int64)
            if completed
            else torch.empty(0, dtype=torch.int64)
        )
        if completed:
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                state.in_pool_bitmap[:n_blocks].fill_(True)
            scatter_stream.synchronize()
        return None, [], completed_ids

    def _issue_indexed_reads(
        self,
        state: CSAAttentionKVLayerState,
        sorted_block_ids: Sequence[int] | torch.Tensor,
        *,
        io_priority: str,
    ) -> Tuple[Optional[torch.cuda.Event], List[Any], torch.Tensor]:
        """Read one-range CSA blocks through the fused indexed CUDA path."""
        profile_start = time.perf_counter()
        slba_table = state.indexed_slba_table
        dst_rows_table = state.indexed_dst_rows_table
        if slba_table is None or dst_rows_table is None:
            raise RuntimeError("indexed CSA tables are unavailable")

        device = state.k_cache_tensor.device
        io_nbytes = state.compressed_block_size * state.token_bytes
        io_stream = self._scatter_stream_for(device)
        prepare_stream = self._prepare_stream_for(device)
        prepare_start = time.perf_counter()
        with torch.cuda.stream(prepare_stream):
            selected_ids = torch.as_tensor(
                sorted_block_ids,
                dtype=torch.int64,
                device=device,
            )
            input_ready_event = torch.cuda.Event()
            input_ready_event.record(prepare_stream)
        k_cache_flat = state.k_cache_tensor.view(torch.uint8).reshape(
            int(state.k_cache_tensor.shape[0]),
            -1,
        )
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0
        last_scatter_event: Optional[torch.cuda.Event] = None

        def _scatter_indexed_batch(
            _batch_start: int,
            batch_ids: torch.Tensor,
            staging_stride: int,
            logical_nbytes: int,
            staging: torch.Tensor,
        ) -> None:
            nonlocal last_scatter_event
            n_selected = int(batch_ids.numel())
            if n_selected == 0:
                return
            with torch.inference_mode(), torch.cuda.stream(io_stream):
                source = torch.as_strided(
                    staging,
                    size=(n_selected, logical_nbytes),
                    stride=(staging_stride, 1),
                )
                rows = dst_rows_table.index_select(0, batch_ids)
                k_cache_flat.index_copy_(0, rows, source)
                last_scatter_event = torch.cuda.Event()
                last_scatter_event.record(io_stream)
            # The next indexed submit is enqueued on this same stream, so it
            # cannot let NVMe overwrite staging until this scatter completes.
            # The final event is returned to the target-layer consumption gate.

        load_start = time.perf_counter()
        self._tutti_loader.load_indexed_chunks_to_hbm(
            selected_ids,
            slba_table,
            io_nbytes,
            _scatter_indexed_batch,
            io_priority=io_priority,
            profile_layer_id=state.layer_id,
            input_ready_event=input_ready_event,
        )
        # Reuse the already-uploaded ids. This avoids the old second H2D copy
        # and keeps the resident bitmap ordered after every batch scatter.
        with torch.inference_mode(), torch.cuda.stream(io_stream):
            state.in_pool_bitmap.index_fill_(0, selected_ids, True)
            last_scatter_event = torch.cuda.Event()
            last_scatter_event.record(io_stream)
        load_ms = (time.perf_counter() - load_start) * 1000.0
        if _timing_enabled():
            logger.info(
                "CSA_LAYER_PROFILE device=%d layer=%d blocks=%d "
                "bytes_mib=%.3f prepare_ms=%.3f loader_ms=%.3f "
                "g2g_async=1 total_ms=%.3f",
                int(device.index or 0),
                state.layer_id,
                len(sorted_block_ids),
                len(sorted_block_ids) * io_nbytes / 1024**2,
                prepare_ms,
                load_ms,
                (time.perf_counter() - profile_start) * 1000.0,
            )
        completed_ids = torch.as_tensor(
            sorted_block_ids,
            dtype=torch.int64,
            device="cpu",
        ).reshape(-1)
        return last_scatter_event, [], completed_ids

    def _miss_ids_for_topk(
        self,
        layer_id: int,
        true_topk: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``true_topk`` block ids missing from the pool.

        Args:
            layer_id: Transformer-side CSA layer id.
            true_topk: Output tensor of the true Lightning Indexer, shape
                ``[num_queries, top_k]`` or ``[top_k]``.  Values are
                compressed *entry* ids in ``[0, S/compress_ratio)``; they are
                converted to block ids by integer division with
                ``compressed_block_size`` (matches vLLM IndexerCache's
                ``[num_blocks, 64, token_bytes]`` layout).

        Returns:
            Sorted unique CPU tensor of block ids not yet in the layer's pool
            and within the chunk map's registered compressed-block range. The
            indexer often emits block ids past the current prefix (sentinel
            padding); skipping them here avoids noisy
            "block_id N not covered by any chunk" warnings downstream.

        Implementation note: the entire filter runs on the GPU so we do not
        pay a per-layer ``true_topk.cpu()`` sync.  The Python sync at the
        tail of the function only touches the (small) miss-set, which is
        typically a single-digit number of block ids once the HC-proxy
        prediction has primed the pool.  When the miss-set is empty the
        function returns ``[]`` without ever materialising it on the host.
        """
        state = self._layers.get(int(layer_id))
        if state is None or not state.chunks:
            return torch.empty(0, dtype=torch.int64)
        entries = true_topk.reshape(-1)
        if entries.numel() == 0:
            return torch.empty(0, dtype=torch.int64)
        device = state.in_pool_bitmap.device
        if entries.device != device:
            entries = entries.to(device)
        entries = entries.to(torch.int64)
        block_ids = entries // state.compressed_block_size
        # Clip to the chunk-map's registered range (sentinel padding past the
        # cached prefix) and to the bitmap capacity (mirrors vLLM's K cache
        # num_blocks) in a single validity mask; sentinel/negative entries
        # are dropped by the same mask.
        max_block_id = state.chunks[-1].end_compressed_block
        bitmap_len = int(state.in_pool_bitmap.shape[0])
        limit = min(int(max_block_id), bitmap_len)
        if limit <= 0:
            return torch.empty(0, dtype=torch.int64)
        # Deduplicate via a scatter into a [limit] bool mask instead of
        # torch.unique(): unique() is sort-based (O(N log N)) and costs
        # ~130 ms per call at 33M padded top-K entries (64K-token chunks),
        # while the scatter is a single O(N) elementwise pass.  This also
        # avoids the two .item() early-exit syncs the old path paid before
        # ever reaching the bitmap lookup.
        valid = (entries >= 0) & (block_ids < limit)
        seen = torch.zeros(limit, dtype=torch.bool, device=device)
        seen[block_ids[valid]] = True
        miss_mask = seen & ~state.in_pool_bitmap[:limit]
        return miss_mask.nonzero(as_tuple=False).reshape(-1).cpu()
