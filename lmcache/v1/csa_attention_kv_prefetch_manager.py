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
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

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
    detailed_io_nvtx,
)
from lmcache.v1.kv_object_store import KVObjectByteRange
from lmcache.v1.ssd_tp_sharded_prefetch import (
    CollectiveDescriptor,
    SSDReadMode,
    SSDTPShardedPrefetchConfig,
    ShardCollectiveError,
    ShardGatherTransport,
    ShardPrefetchDecisionTable,
    bucket_prefetch_key,
    compile_cp_read_plan,
    partition_block_union,
)

if TYPE_CHECKING:
    from lmcache.v1.gpu_connector.tutti_direct_loader import TuttiDirectLoader

logger = init_logger(__name__)

_ACTIVE_MANAGER: Optional["CSAAttentionKVPrefetchManager"] = None

_DSV4_CSA_COMPRESS_RATIO = 4


_ACTIVE_MANAGER_LOCK = threading.Lock()

# Reuse the exact list objects for immutable generation extent tables across
# requests.  Tutti's ensure_lba_cache() has an identity fast path; recreating
# equal lists otherwise forces an unnecessary sort/index rebuild on every hit.
_SHARED_LBA_TABLE_CACHE: dict[
    tuple[tuple[str, int, int, int], ...], dict[str, list[Any]]
] = {}
_SHARED_LBA_TABLE_CACHE_LOCK = threading.Lock()
_SHARED_LBA_TABLE_CACHE_LIMIT = 8


def _timing_enabled() -> bool:
    """Return True when CSA attention KV prefetch timing logs are enabled."""
    value = os.environ.get("LMCACHE_CSA_ATTENTION_KV_TIMING", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _io_profile_kind(label: str) -> str:
    """Return a stable HCA/CSA timeline kind for an internal read label."""

    if label.startswith("hca_deterministic"):
        return "hca_deterministic"
    if label == "indexer_native_stream":
        return "dsa_indexer"
    if label.startswith("predicted_"):
        return "csa_predicted"
    if label == "miss" or label.startswith("exact_chunk_"):
        return "csa_correction"
    if label == "dense":
        return "csa_dense"
    return label


def _exact_chunk_prefetch_enabled() -> bool:
    """Return whether exact first-chunk true-topK reads are enabled."""
    value = os.environ.get("LMCACHE_CSA_EXACT_CHUNK_PREFETCH", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _exact_chunk_prefetch_limit() -> int:
    """Return the maximum number of true-topK chunks prefetched per layer."""
    try:
        return max(
            0,
            int(os.environ.get("LMCACHE_CSA_EXACT_CHUNK_PREFETCH_MAX_CHUNKS", "1")),
        )
    except ValueError:
        return 1


def _coalesce_physically_contiguous_extents(
    extents: Sequence[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Merge extent boundaries that are contiguous in file and on NVMe.

    FIEMAP may split a physically contiguous allocation into adjacent records.
    Treating those records as hard boundaries rejects an otherwise valid
    one-command indexed block whenever the block straddles the metadata-only
    split.

    Args:
        extents: ``(file_offset, slba, n_sectors)`` records.

    Returns:
        Sorted records with exactly adjacent logical and physical runs merged.
    """
    ordered = sorted(
        (int(offset), int(slba), int(sectors))
        for offset, slba, sectors in extents
        if int(sectors) > 0
    )
    merged: list[tuple[int, int, int]] = []
    for offset, slba, sectors in ordered:
        if merged:
            prev_offset, prev_slba, prev_sectors = merged[-1]
            if (
                prev_offset + prev_sectors * 512 == offset
                and prev_slba + prev_sectors == slba
            ):
                merged[-1] = (
                    prev_offset,
                    prev_slba,
                    prev_sectors + sectors,
                )
                continue
        merged.append((offset, slba, sectors))
    return merged


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


def build_shared_raw_lba_cache(
    chunk_maps: Sequence[Mapping[int, Sequence[CSAAttentionKVChunkLoc]]],
) -> dict[str, list[Any]]:
    """Build one immutable Tutti extent table shared by stream consumers.

    Args:
        chunk_maps: CSA, HCA, and native-indexer layer plans whose raw
            extents must remain simultaneously addressable.

    Returns:
        A path-keyed, de-duplicated list of Tutti ``LbaRecord`` objects. The
        returned list objects are intentionally shared by every consumer so
        ``ensure_lba_cache`` can use identity checks instead of repeatedly
        sorting the same pool extents on the layer hot path.
    """
    seen: set[tuple[str, int, int, int]] = set()
    for chunks_by_layer in chunk_maps:
        for chunks in chunks_by_layer.values():
            for chunk in chunks:
                path = chunk.disk_meta.path if chunk.disk_meta else None
                if not path or not chunk.raw_extents:
                    continue
                for file_offset, slba, n_sectors in chunk.raw_extents:
                    seen.add(
                        (
                            path,
                            int(file_offset),
                            int(slba),
                            int(n_sectors),
                        )
                    )
    signature = tuple(sorted(seen))
    if not signature:
        return {}
    with _SHARED_LBA_TABLE_CACHE_LOCK:
        cached = _SHARED_LBA_TABLE_CACHE.get(signature)
        if cached is not None:
            return cached

    from lmcache.v1.gpu_connector.tutti_direct_loader import LbaRecord

    raw_lba_cache: dict[str, list[Any]] = {}
    for path, file_offset, slba, n_sectors in signature:
        raw_lba_cache.setdefault(path, []).append(
            LbaRecord(
                file_offset=file_offset,
                slba=slba,
                n_sectors=n_sectors,
            )
        )
    with _SHARED_LBA_TABLE_CACHE_LOCK:
        cached = _SHARED_LBA_TABLE_CACHE.get(signature)
        if cached is not None:
            return cached
        if len(_SHARED_LBA_TABLE_CACHE) >= _SHARED_LBA_TABLE_CACHE_LIMIT:
            _SHARED_LBA_TABLE_CACHE.clear()
        _SHARED_LBA_TABLE_CACHE[signature] = raw_lba_cache
    return raw_lba_cache


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
    true_selected_blocks_bitmap: Optional[torch.Tensor] = None
    true_selected_covers_cached_prefix: bool = False
    streamed_selected_blocks_bitmap: Optional[torch.Tensor] = None
    streamed_selected_rows: int = 0
    streamed_selected_chunks: int = 0
    streamed_selected_request_token: Optional[Tuple[str, int]] = None
    streamed_selected_event: Optional[torch.cuda.Event] = None
    streamed_selected_failed: bool = False


@dataclass(slots=True)
class _DeferredShardGather:
    """Rank-local read whose private collective is deferred to a model gate."""

    state: CSAAttentionKVLayerState
    descriptor: CollectiveDescriptor
    selected: torch.Tensor
    owned: torch.Tensor
    local_ready_event: Optional[torch.cuda.Event]
    local_objects: List[Any]
    local_complete: bool
    local_capability: bool
    pending_ids: Optional[torch.Tensor] = None
    io_range: Optional[CsaNvtxRange] = None
    operation_id: str = ""
    profile_kind: str = "unknown"
    profile_source_layer: Optional[int] = None
    request_id: Optional[str] = None
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock)
    lifecycle_claimed: bool = False


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
        *,
        data_group: str = "csa",
        shard_config: Optional[SSDTPShardedPrefetchConfig] = None,
        shard_transport: Optional[ShardGatherTransport] = None,
        cp_rank: int = 0,
        cp_world_size: int = 1,
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
            data_group: ``"csa"`` for attention KV or ``"indexer"`` for the
                compact native Indexer-K restore helper.
            shard_config: Centralized feature gates and cost parameters.
            shard_transport: Optional independent collective transport.
            cp_rank: Rank in the native Indexer context-parallel group.
            cp_world_size: Native Indexer context-parallel group size.
        """
        if tutti_loader is None:
            raise ValueError("tutti_loader is required for CSA attention KV prefetch")
        if not csa_layer_ids:
            raise ValueError("csa_layer_ids must be non-empty")
        if compressed_block_size <= 0:
            raise ValueError("compressed_block_size must be positive")
        if token_bytes <= 0:
            raise ValueError("token_bytes must be positive")
        if data_group not in {"csa", "indexer"}:
            raise ValueError("data_group must be csa or indexer")
        if cp_world_size <= 0 or cp_rank < 0 or cp_rank >= cp_world_size:
            raise ValueError("invalid CP rank or world size")

        self._tutti_loader = tutti_loader
        self._compressed_block_size = int(compressed_block_size)
        self._token_bytes = int(token_bytes)
        self._bytes_per_block = self._compressed_block_size * self._token_bytes
        self._data_group = data_group
        self._shard_config = shard_config or SSDTPShardedPrefetchConfig.from_env()
        self._shard_transport = shard_transport
        self._shard_decisions = ShardPrefetchDecisionTable(self._shard_config)
        self._shard_collective_lock = threading.Lock()
        self._cp_rank = int(cp_rank)
        self._cp_world_size = int(cp_world_size)
        self._cp_fallback_reasons: set[tuple[int, str]] = set()
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
        self._external_kv_forward_active = False
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
        self._request_transition_lock = threading.RLock()
        self._request_state = threading.Condition()
        self._active_submissions = 0
        self._request_generation = 0
        self._request_cleanup_failed = False
        self._request_lifecycle = "inactive"
        self._closed = False
        # Physical destination rows are normally identical across requests
        # when vLLM reuses the same paged-cache layout.  Keep a small,
        # process-local GPU table cache so the first cache hit after cold
        # admission pays the upload once and later request registrations only
        # attach an existing tensor.  The complete row tuple is part of the
        # key, so a different allocation layout cannot reuse a stale table.
        self._layer_major_dst_rows_table_cache: dict[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = {}
        self._layer_major_dst_rows_table_cache_limit = 16
        # vLLM's single-request block allocator normally returns one
        # contiguous physical range.  Build the largest identity row table
        # while KV tensors are registered at server startup, so the first
        # cache hit can attach a slice without a CUDA allocation or H2D copy.
        # Non-contiguous layouts are detected exactly and use the tuple cache
        # above; correctness never depends on the contiguous fast path.
        self._identity_rows_by_device: dict[str, torch.Tensor] = {}
        # The vLLM indexer produces an 8192-row prefill in multiple query
        # chunks.  The first chunk already identifies most of the final block
        # union, so submit those exact blocks on a background thread while the
        # remaining chunks are scored.  The normal full-output correction is
        # retained as the authoritative final gate.
        self._exact_chunk_prefetch_enabled = _exact_chunk_prefetch_enabled()
        self._exact_chunk_prefetch_limit = _exact_chunk_prefetch_limit()
        self._exact_chunk_prefetch_executor: Optional[ThreadPoolExecutor] = None
        if self._exact_chunk_prefetch_enabled and self._exact_chunk_prefetch_limit:
            self._exact_chunk_prefetch_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="lmcache-csa-exact-chunk",
            )
        self._exact_chunk_futures: Dict[int, List[Future[bool]]] = {}
        self._exact_chunk_futures_lock = threading.Lock()
        self._exact_chunk_hook_prefixes: Dict[Any, str] = {}

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
        """Return successfully registered CSA consumer layers."""
        return tuple(
            layer_id for layer_id in self._csa_layer_ids if layer_id in self._layers
        )

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

    def warm_runtime_resources(self) -> None:
        """Create reusable CUDA streams before the first cache hit.

        Layer registration supplies the actual CUDA devices used by the model.
        Resolving the scatter and indexed-plan streams here moves their one-time
        construction out of request latency. When CSA sharding is enabled, the
        same startup hook allocates and prewarms the bounded gather ring. The
        method is idempotent and performs no SSD I/O or cache mutation.
        """
        devices = {
            state.k_cache_tensor.device
            for state in self._layers.values()
            if state.k_cache_tensor.device.type == "cuda"
        }
        for device in devices:
            with torch.cuda.device(device):
                self._scatter_stream_for(device)
                self._prepare_stream_for(device)
        transport = self._shard_transport
        if (
            self._data_group == "csa"
            and self._shard_config.enabled
            and self._shard_config.csa_enabled
            and transport is not None
        ):
            registered_blocks = max(
                (int(state.in_pool_bitmap.numel()) for state in self._layers.values()),
                default=0,
            )
            # The registered bitmap spans the complete model KV pool (and can
            # therefore be several GiB per layer).  The gather ring only needs
            # to cover one bounded union.  Cap its warm allocation to the
            # largest world-size-aligned union that fits in a staging slot;
            # larger request unions are rejected by preflight and use the
            # LOCAL_DIRECT fallback.
            blocks_per_rank = self._shard_config.staging_slot_bytes // (
                transport.world_size * self._bytes_per_block
            )
            staging_blocks = blocks_per_rank * transport.world_size
            max_blocks = min(registered_blocks, staging_blocks)
            if max_blocks > 0 and devices:
                # A TP rank's registered CSA tensors all live on one device.
                # Multiple devices would require one communicator per device,
                # which is deliberately outside the first implementation.
                if len(devices) != 1:
                    raise RuntimeError(
                        "shard-gather manager requires exactly one CUDA device"
                    )
                transport.warm(
                    max_union_blocks=max_blocks,
                    block_bytes=self._bytes_per_block,
                    device=next(iter(devices)),
                )
        if devices:
            logger.info(
                "CSAAttentionKVPrefetchManager: warmed runtime streams devices=%s",
                sorted(str(device) for device in devices),
            )

    @property
    def active_request_token(self) -> Tuple[str, int]:
        """Return the active request id and its plan generation."""
        with self._request_state:
            if self._active_request_id is None:
                return "", -1
            return self._active_request_id, self._request_generation

    def set_external_kv_forward_active(self, active: bool) -> None:
        """Enable correction work only for the external-KV prefill forward.

        Args:
            active: Whether the current model forward consumes external KV.
        """
        self._external_kv_forward_active = bool(active)

    def request_stream_available(self) -> bool:
        """Return whether a prior request cleanup left the consumer usable."""
        with self._request_state:
            return not self._request_cleanup_failed and self._request_lifecycle in {
                "inactive",
                "active",
            }

    def request_chunks_match(
        self,
        req_id: str,
        chunks_by_layer: Dict[int, List[CSAAttentionKVChunkLoc]],
    ) -> bool:
        """Return whether a repeated request proposes the registered plan.

        Args:
            req_id: Request identifier whose plan is being checked.
            chunks_by_layer: Newly constructed per-layer chunk descriptors.

        Returns:
            ``True`` only when the request id, layer set, byte locations, and
            destination rows exactly match the currently registered plan.
        """
        if str(req_id) != self.active_request_id:
            return False
        if set(chunks_by_layer) != set(self._layers):
            return False

        def signature(chunk: CSAAttentionKVChunkLoc) -> tuple[Any, ...]:
            disk_path = chunk.disk_meta.path if chunk.disk_meta is not None else None
            return (
                int(chunk.first_compressed_block),
                int(chunk.n_compressed_blocks),
                chunk.key,
                disk_path,
                int(chunk.layer_byte_offset),
                int(chunk.bytes_per_block),
                tuple(chunk.raw_extents),
                tuple(chunk.physical_block_ids),
                int(chunk.payload_skip),
                int(chunk.read_length),
                bool(chunk.layer_major),
            )

        for layer_id, state in self._layers.items():
            proposed = sorted(
                chunks_by_layer.get(int(layer_id), ()),
                key=lambda chunk: int(chunk.first_compressed_block),
            )
            existing = sorted(
                state.chunks,
                key=lambda chunk: int(chunk.first_compressed_block),
            )
            if tuple(map(signature, proposed)) != tuple(map(signature, existing)):
                return False
        return True

    def deactivate_request(self, timeout_s: float = 30.0) -> bool:
        """Drain and clear the active request plan without unpatching layers.

        Args:
            timeout_s: Total time allowed for scheduled and in-flight reads.

        Returns:
            ``True`` when no old-request I/O can still write the registered
            cache tensors. ``False`` means the caller must abort before model
            forward starts.
        """
        with self._request_transition_lock:
            return self._deactivate_request_locked(timeout_s)

    def _deactivate_request_locked(self, timeout_s: float = 30.0) -> bool:
        """Deactivate one request while holding the lifecycle transition lock."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._request_state:
            self._request_cleanup_failed = True
            self._request_lifecycle = "draining"
            self._active_request_id = None
            while self._active_submissions:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "CSAAttentionKVPrefetchManager: deactivate timed out "
                        "with active_submissions=%d",
                        self._active_submissions,
                    )
                    return False
                self._request_state.wait(remaining)
        with self._scheduled_layer_futures_lock:
            scheduled = tuple(self._scheduled_layer_futures.items())
        for _layer_id, future in scheduled:
            cancel = getattr(future, "cancel", None)
            if callable(cancel):
                cancel()
        for layer_id, future in scheduled:
            result = getattr(future, "result", None)
            try:
                if not callable(result):
                    raise RuntimeError("tracked submission has no result method")
                completed_work = result(timeout=max(0.0, deadline - time.monotonic()))
                self.discard_deferred_shard_gather(
                    completed_work,
                    status="request_deactivated",
                )
            except Exception:
                cancelled = bool(getattr(future, "cancelled", lambda: False)())
                done = bool(getattr(future, "done", lambda: False)())
                if not cancelled and not done:
                    logger.error(
                        "CSAAttentionKVPrefetchManager: deactivate timed out "
                        "waiting for scheduled layer=%d",
                        layer_id,
                    )
                    return False
                if not cancelled:
                    logger.exception(
                        "CSAAttentionKVPrefetchManager: old request "
                        "submission failed while deactivating"
                    )
            with self._scheduled_layer_futures_lock:
                if self._scheduled_layer_futures.get(layer_id) is future:
                    self._scheduled_layer_futures.pop(layer_id, None)

        if not self._drain_exact_topk_futures(deadline):
            logger.error(
                "CSAAttentionKVPrefetchManager: deactivate timed out waiting "
                "for exact top-K chunk prefetch"
            )
            return False

        for layer_id, state in self._layers.items():
            with state.pending_reads_lock:
                while state.pending_read_count:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.error(
                            "CSAAttentionKVPrefetchManager: deactivate timed "
                            "out layer=%d pending=%d",
                            layer_id,
                            state.pending_read_count,
                        )
                        return False
                    state.pending_reads_lock.wait(remaining)
            if not self._drain_for_layer_until(int(layer_id), deadline):
                logger.error(
                    "CSAAttentionKVPrefetchManager: deactivate timed out "
                    "waiting for CUDA drain layer=%d",
                    layer_id,
                )
                return False

        self._pending_raw_lba_cache = {}
        for state in self._layers.values():
            state.chunks = []
            state.in_pool_bitmap.zero_()
            with state.pending_reads_lock:
                state.pending_reads_bitmap.zero_()
                state.resident_blocks_bitmap.zero_()
                state.pending_read_count = 0
                state.last_drain_event = None
                state.pending_drains.clear()
            state.indexed_slba_table = None
            state.indexed_dst_rows_table = None
            state.layer_major_dst_rows_table = None
            state.true_selected_blocks_bitmap = None
            state.true_selected_covers_cached_prefix = False
            state.streamed_selected_blocks_bitmap = None
            state.streamed_selected_rows = 0
            state.streamed_selected_chunks = 0
            state.streamed_selected_request_token = None
            state.streamed_selected_event = None
            state.streamed_selected_failed = False
        with self._request_state:
            self._request_cleanup_failed = False
            self._request_lifecycle = "inactive"
        return True

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

    @property
    def dense_shard_layer_ids(self) -> Tuple[int, ...]:
        """Return registered CSA layers eligible for dense shard-gather."""
        if (
            not self._shard_config.enabled
            or not self._shard_config.csa_enabled
            or self._shard_transport is None
        ):
            return ()
        return tuple(
            layer_id
            for layer_id in self.csa_layer_ids
            if layer_id in self._shard_config.dense_layers
            and layer_id not in self._shard_config.disabled_layers
        )

    def owns_k_cache(self, k_cache_tensor: torch.Tensor) -> bool:
        """Return whether ``k_cache_tensor`` belongs to a registered CSA layer.

        The vLLM sparse-prefill patch uses this public probe to enable compact
        top-K gathering only for caches whose bytes are supplied by this
        manager.  Comparing storage pointers avoids parsing model-specific
        layer names and remains valid for strided views of the same cache.

        Args:
            k_cache_tensor: Candidate vLLM compressed K-cache tensor.

        Returns:
            ``True`` when the tensor shares storage with a registered CSA
            layer, otherwise ``False``.
        """
        if not isinstance(k_cache_tensor, torch.Tensor):
            return False
        candidate_ptr = int(k_cache_tensor.untyped_storage().data_ptr())
        return any(
            int(state.k_cache_tensor.untyped_storage().data_ptr()) == candidate_ptr
            for layer_id, state in self._layers.items()
            if layer_id in self._csa_layer_ids
        )

    def true_selected_blocks_for_cache(
        self,
        k_cache_tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return the exact latest true-topK page union for one CSA cache.

        Args:
            k_cache_tensor: Exact compressed K-cache view used by attention.

        Returns:
            A CUDA int32 bitmap, or ``None`` when no exact union is available.
        """
        if not isinstance(k_cache_tensor, torch.Tensor):
            return None
        candidate_ptr = int(k_cache_tensor.data_ptr())
        for layer_id, state in self._layers.items():
            if layer_id not in self._csa_layer_ids:
                continue
            if int(state.k_cache_tensor.data_ptr()) == candidate_ptr:
                return state.true_selected_blocks_bitmap
        return None

    def true_selected_covers_cached_prefix_for_cache(
        self,
        k_cache_tensor: torch.Tensor,
    ) -> bool:
        """Return whether true top-K selected every cached-prefix page.

        Args:
            k_cache_tensor: Exact compressed K-cache view used by attention.

        Returns:
            ``True`` only when the latest correction proved that every page
            backed by LMCache is selected and resident.
        """
        if not isinstance(k_cache_tensor, torch.Tensor):
            return False
        candidate_ptr = int(k_cache_tensor.data_ptr())
        for layer_id, state in self._layers.items():
            if layer_id not in self._csa_layer_ids:
                continue
            if int(state.k_cache_tensor.data_ptr()) == candidate_ptr:
                return state.true_selected_covers_cached_prefix
        return False

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
        device_key = str(k_cache_tensor.device)
        identity_rows = self._identity_rows_by_device.get(device_key)
        if identity_rows is None or int(identity_rows.numel()) < num_rows:
            self._identity_rows_by_device[device_key] = torch.arange(
                num_rows,
                dtype=torch.int64,
                device=k_cache_tensor.device,
            )
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
        *,
        start_profile_capture: bool = True,
        shared_raw_lba_cache: Optional[dict[str, list[Any]]] = None,
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
            start_profile_capture: Whether this manager owns the request-level
                profiler trigger. Auxiliary managers sharing the same Tutti
                loader must disable it so only one component starts capture.
            shared_raw_lba_cache: Optional immutable union of the CSA, HCA,
                and native-indexer extents. All consumers of one Tutti loader
                must receive the same object so layer reads never replace and
                re-sort each other's extent table.
        """
        with self._request_transition_lock:
            try:
                self._register_request_chunks_locked(
                    req_id,
                    chunks_by_layer,
                    start_profile_capture=start_profile_capture,
                    shared_raw_lba_cache=shared_raw_lba_cache,
                )
            except Exception:
                with self._request_state:
                    preparing = self._request_lifecycle == "preparing"
                if preparing and not self._deactivate_request_locked():
                    logger.error(
                        "CSAAttentionKVPrefetchManager: failed to roll back "
                        "partial request plan"
                    )
                raise

    def _register_request_chunks_locked(
        self,
        req_id: str,
        chunks_by_layer: Dict[int, List[CSAAttentionKVChunkLoc]],
        *,
        start_profile_capture: bool,
        shared_raw_lba_cache: Optional[dict[str, list[Any]]],
    ) -> None:
        """Install one request plan while holding the transition lock."""
        request_id = str(req_id)
        is_new_request = request_id != self._active_request_id
        if is_new_request and not self.deactivate_request():
            raise RuntimeError("previous CSA/HCA request could not be deactivated")
        if is_new_request:
            with self._request_state:
                self._request_lifecycle = "preparing"
        if is_new_request and start_profile_capture:
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
                    self._tutti_loader.ensure_lba_cache(self._pending_raw_lba_cache)
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
        raw_lba_cache = shared_raw_lba_cache
        if raw_lba_cache is None:
            raw_lba_cache = build_shared_raw_lba_cache((chunks_by_layer,))
        if raw_lba_cache:
            # Keep the exact shared lists. Tutti's identity fast path then
            # makes every per-layer ensure call O(paths), with no copy/sort.
            self._pending_raw_lba_cache = raw_lba_cache
            try:
                self._tutti_loader.ensure_lba_cache(raw_lba_cache)
            except Exception:
                logger.exception(
                    "CSAAttentionKVPrefetchManager: failed to register %d "
                    "LBA cache entries for request %s",
                    len(raw_lba_cache),
                    req_id,
                )

        layer_major_table_cache: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
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
            self._compile_layer_major_table(state, layer_major_table_cache)
            # Only reset drain and bitmap state for genuinely new requests.
            # Per-microbatch re-registration (same req_id) must preserve the
            # in_pool_bitmap and last_drain_event so staging slots from reads
            # already in flight are not orphaned (leak → pool exhaustion).
            if is_new_request:
                state.in_pool_bitmap.zero_()
                state.true_selected_blocks_bitmap = None
                state.true_selected_covers_cached_prefix = False
                state.streamed_selected_blocks_bitmap = None
                state.streamed_selected_rows = 0
                state.streamed_selected_chunks = 0
                state.streamed_selected_request_token = None
                state.streamed_selected_event = None
                state.streamed_selected_failed = False
                with state.pending_reads_lock:
                    state.pending_reads_bitmap.zero_()
                    state.resident_blocks_bitmap.zero_()
                    state.pending_read_count = 0
                    state.last_drain_event = None
        if is_new_request:
            with self._request_state:
                self._request_generation += 1
                self._active_request_id = request_id
                self._request_lifecycle = "active"

    def start_full_nsys_capture_for_request(self, request_id: str) -> None:
        """Start one full-forward Nsight capture for a selected cache hit.

        Args:
            request_id: Newly registered LMCache request identifier.

        Notes:
            ``LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS`` counts cache-hit
            requests registered by this manager, allowing warmup hits to be
            skipped without tracing cold-store model execution.
        """
        if detailed_io_nvtx.enabled:
            logger.info(
                "LMCACHE_CSA_DETAILED_IO_NVTX requires one externally "
                "coordinated Nsys window spanning HTTP send through first "
                "token; per-rank cudaProfilerApi capture is disabled"
            )
            return
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
        if (
            state.block_slot_scatter
            or not state.chunks
            or (len(state.chunks) == 1 and state.chunks[0].layer_major)
        ):
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
                _coalesce_physically_contiguous_extents(chunk.raw_extents),
                dtype=torch.int64,
            ).reshape(-1, 3)
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
        shared_tables: dict[tuple[str, tuple[int, ...]], torch.Tensor],
    ) -> None:
        """Attach one shared destination-row table for layer-major objects."""
        state.layer_major_dst_rows_table = None
        if not state.chunks or any(not chunk.layer_major for chunk in state.chunks):
            return
        expected = 0
        row_parts: list[tuple[int, ...]] = []
        for chunk in state.chunks:
            if (
                chunk.first_compressed_block != expected
                or len(chunk.physical_block_ids) != chunk.n_compressed_blocks
            ):
                return
            row_parts.append(chunk.physical_block_ids)
            expected = chunk.end_compressed_block
        rows = (
            row_parts[0]
            if len(row_parts) == 1
            else tuple(row for part in row_parts for row in part)
        )
        # Every layer in one sidecar group uses the same immutable physical
        # row tuple. Upload it once per device instead of issuing 20/21 tiny,
        # synchronising H2D allocations during every 480K cache hit.
        cache_key = (
            str(state.k_cache_tensor.device),
            tuple(id(part) for part in row_parts),
        )
        table = shared_tables.get(cache_key)
        if table is None:
            persistent_key = (
                str(state.k_cache_tensor.device),
                rows,
            )
            first_row = rows[0] if rows else 0
            identity_rows = self._identity_rows_by_device.get(persistent_key[0])
            is_contiguous = bool(
                rows
                and first_row >= 0
                and identity_rows is not None
                and first_row + len(rows) <= int(identity_rows.numel())
                and all(row == first_row + index for index, row in enumerate(rows))
            )
            table = (
                identity_rows[first_row : first_row + len(rows)]
                if is_contiguous and identity_rows is not None
                else self._layer_major_dst_rows_table_cache.get(persistent_key)
            )
            if table is None:
                table = torch.as_tensor(
                    rows,
                    dtype=torch.int64,
                    device=state.k_cache_tensor.device,
                )
                if (
                    len(self._layer_major_dst_rows_table_cache)
                    >= self._layer_major_dst_rows_table_cache_limit
                ):
                    self._layer_major_dst_rows_table_cache.clear()
                self._layer_major_dst_rows_table_cache[persistent_key] = table
            shared_tables[cache_key] = table
        state.layer_major_dst_rows_table = table

    def fire_predicted_reads(
        self,
        layer_id: int,
        compressed_block_ids: Sequence[int] | torch.Tensor,
        prefetch_level: int = 2,
        *,
        request_token: Optional[Tuple[str, int]] = None,
        profile_source_layer: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
        profile_kind: Optional[str] = None,
    ) -> bool | _DeferredShardGather:
        """Submit Tutti reads for the predicted ``top-K`` of one CSA layer.

        Called from :class:`IndexerSSDManager` after its HC-proxy emits a
        prediction for ``layer_id``.  Block ids already in the pool or with
        an in-flight read are skipped.

        Args:
            layer_id: Transformer-side CSA layer id.
            compressed_block_ids: Predicted compressed block ids.
            prefetch_level: Source-to-target distance, either one or two.
            request_token: Request generation captured when work was queued.
            profile_source_layer: Optional true prediction source layer.
            profile_operation_id: Optional parent operation correlation id.
            profile_kind: Optional profile-only I/O classification.

        Raises:
            ValueError: If ``prefetch_level`` is unsupported.
        """
        if prefetch_level not in (1, 2):
            raise ValueError("prefetch_level must be 1 or 2")
        state = self._layers.get(int(layer_id))
        active_sharding = bool(
            state is not None
            and self._shard_transport is not None
            and self._shard_config.enabled
            and self._shard_config.csa_enabled
        )
        if active_sharding:
            return self._prepare_predicted_shard_gather(
                state,
                compressed_block_ids,
                label=f"predicted_l{prefetch_level}",
                request_token=request_token,
                profile_source_layer=profile_source_layer,
                profile_operation_id=profile_operation_id,
                profile_kind=profile_kind,
            )
        return self._submit_reads(
            layer_id,
            compressed_block_ids,
            label=f"predicted_l{prefetch_level}",
            io_priority="lookahead",
            request_token=request_token,
            source_layer_id=(
                int(profile_source_layer)
                if profile_source_layer is not None
                else int(layer_id) - int(prefetch_level)
            ),
            profile_operation_id=profile_operation_id,
            profile_kind=profile_kind,
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

    def fire_deterministic_layer(
        self,
        layer_id: int,
        *,
        label: str = "hca_deterministic",
        request_token: Optional[Tuple[str, int]] = None,
        source_layer_id: Optional[int] = None,
    ) -> bool:
        """Submit every covered block for a deterministic HCA layer.

        HCA has no sparse indexer: every compressed entry is consumed by its
        attention.  The preceding layer can therefore move this read into its
        FFN/MoE window without prediction or correctness risk.

        Args:
            layer_id: Registered HCA transformer layer id.
            label: Profiling label attached to the Tutti submission.
            request_token: Request generation captured when work was queued.

        Returns:
            ``True`` when the layer has a registered non-empty chunk map.
        """
        state = self._layers.get(int(layer_id))
        if state is None or not state.chunks:
            return False
        covered_end = int(state.chunks[-1].end_compressed_block)
        selected_blocks: Sequence[int] = range(covered_end)
        if self._data_group == "indexer":
            selected_blocks = self._indexer_cp_owned_blocks(
                int(layer_id),
                covered_end,
            )
        self._submit_reads(
            int(layer_id),
            selected_blocks,
            label=label,
            raise_on_error=True,
            request_token=request_token,
            source_layer_id=source_layer_id,
        )
        return True

    def fire_dense_layer(
        self,
        layer_id: int,
        *,
        request_token: Optional[Tuple[str, int]] = None,
        source_layer_id: Optional[int] = None,
    ) -> bool | _DeferredShardGather:
        """Prepare a complete CSA layer for dense shard-gather.

        Args:
            layer_id: Registered CSA transformer layer id.
            request_token: Request generation captured by the source hook.

        Returns:
            Deferred gate work when TP sharding is active, or ``True`` when a
            non-empty local layer plan was submitted. The background producer
            reads only this rank's owner shard; private-NCCL consensus and
            gather run later after all model ranks reach the target-layer gate.
        """
        if self._data_group != "csa":
            return False
        state = self._layers.get(int(layer_id))
        if state is None or not state.chunks:
            return False
        covered_end = int(state.chunks[-1].end_compressed_block)
        active_sharding = bool(
            self._shard_transport is not None
            and self._shard_config.enabled
            and self._shard_config.csa_enabled
        )
        if active_sharding:
            return self._prepare_shard_gather(
                state,
                range(covered_end),
                mode=SSDReadMode.SHARD_GATHER_DENSE,
                io_priority="lookahead",
                request_token=request_token,
                profile_kind="csa_dense",
                profile_source_layer=source_layer_id,
            )
        self._submit_reads(
            int(layer_id),
            range(covered_end),
            label="dense",
            io_priority="lookahead",
            raise_on_error=True,
            request_token=request_token,
            source_layer_id=source_layer_id,
        )
        return True

    def track_layer_submission(
        self,
        layer_id: int,
        future: Any,
        *,
        request_token: Optional[Tuple[str, int]] = None,
    ) -> None:
        """Track a background submission that must precede a layer gate.

        Args:
            layer_id: Target transformer layer id.
            future: Future whose result means Tutti/scatter work was enqueued.
            request_token: Request generation captured when work was queued.
        """
        with self._request_state:
            current_token = (
                self._active_request_id or "",
                self._request_generation,
            )
            valid = self._request_lifecycle == "active" and (
                request_token is None or request_token == current_token
            )
        if not valid:
            cancel = getattr(future, "cancel", None)
            if callable(cancel):
                cancel()
            return
        with self._scheduled_layer_futures_lock:
            self._scheduled_layer_futures[int(layer_id)] = future

    def layer_submission_ready(self, layer_id: int) -> bool:
        """Return whether a tracked background layer submission is complete."""
        with self._scheduled_layer_futures_lock:
            future = self._scheduled_layer_futures.get(int(layer_id))
        return future is None or bool(future.done())

    def wait_for_tracked_submission(
        self,
        layer_id: int,
        timeout_s: float = 30.0,
    ) -> bool:
        """Join a tracked producer before miss filtering inspects bitmaps.

        Args:
            layer_id: Target transformer layer id.
            timeout_s: Maximum seconds to wait for the producer.

        Returns:
            ``True`` when no producer is tracked or it completed successfully.
        """
        with self._scheduled_layer_futures_lock:
            future = self._scheduled_layer_futures.pop(int(layer_id), None)
        if future is None:
            return True
        try:
            result = future.result(timeout=max(0.0, timeout_s))
            if isinstance(result, _DeferredShardGather):
                self.finalize_deferred_shard_gather(result)
            return True
        except Exception:
            logger.exception(
                "CSAAttentionKVPrefetchManager: tracked layer %d submission "
                "failed or timed out",
                layer_id,
            )
            return False

    def finalize_deferred_shard_gather(self, work: Any) -> bool:
        """Finish a deferred TP shard gather at an aligned model-layer gate.

        Args:
            work: Value returned by a background dense or predicted read.

        Returns:
            ``True`` when ``work`` was a deferred gather and was finalized.
        """
        if not isinstance(work, _DeferredShardGather):
            return False
        self._finalize_shard_gather(work)
        return True

    def discard_deferred_shard_gather(
        self,
        work: Any,
        *,
        status: str = "cancelled",
    ) -> bool:
        """Release unconsumed rank-local prediction work without a collective.

        Args:
            work: Value returned by a deferred dense or predicted submission.
            status: Completion status written to the associated profile range.

        Returns:
            ``True`` when ``work`` was deferred shard work. Repeated finalize
            or discard calls are harmless.

        Notes:
            Request teardown uses this path for predictions whose target layer
            was never consumed. It waits only for already-enqueued local CUDA
            writes before releasing staging references; no TP collective is
            introduced on the teardown path.
        """
        if not isinstance(work, _DeferredShardGather):
            return False
        if self._claim_deferred_shard_gather(work):
            self._release_deferred_shard_gather(work, status=status)
        return True

    def uses_gate_aligned_shard_gather(self) -> bool:
        """Return whether predicted TP collectives must run at model gates."""
        return bool(
            self._shard_transport is not None
            and self._shard_config.enabled
            and self._shard_config.csa_enabled
        )

    def submit_miss_reads(
        self,
        layer_id: int,
        compressed_block_ids: Sequence[int] | torch.Tensor,
        *,
        request_token: Optional[Tuple[str, int]] = None,
        profile_source_layer: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
        profile_kind: Optional[str] = None,
    ) -> None:
        """Submit Tutti reads for blocks not covered by the prediction.

        Called from the patched ``DeepseekV4Indexer.forward`` after the true
        Lightning Indexer returns ``true_topk``.

        Args:
            layer_id: Transformer-side CSA layer id.
            compressed_block_ids: ``true_topk`` block ids whose K cache
                slots are not yet populated.
            request_token: Request generation captured before true-indexer work.
            profile_source_layer: Optional authoritative indexer source layer.
            profile_operation_id: Optional parent operation correlation id.
            profile_kind: Optional profile-only I/O classification.
        """
        self._submit_reads(
            layer_id,
            compressed_block_ids,
            label="miss",
            raise_on_error=True,
            request_token=request_token,
            source_layer_id=(
                int(profile_source_layer)
                if profile_source_layer is not None
                else int(layer_id)
            ),
            profile_operation_id=profile_operation_id,
            profile_kind=profile_kind,
        )

    def submit_topk_miss_reads(
        self,
        layer_id: int,
        true_topk: torch.Tensor,
        *,
        request_token: Optional[Tuple[str, int]] = None,
        profile_source_layer: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
        profile_kind: Optional[str] = None,
    ) -> None:
        """Submit only non-resident blocks selected by an indexer result.

        Args:
            layer_id: Transformer-side sparse-attention layer id.
            true_topk: Authoritative indexer output containing compressed-entry
                ids, with shape ``[num_queries, top_k]`` or ``[top_k]``.
            request_token: Optional request generation captured before the
                indexer ran. A stale generation is rejected by the normal
                submission path.
            profile_source_layer: Optional authoritative indexer source layer.
            profile_operation_id: Optional parent operation correlation id.
            profile_kind: Optional profile-only I/O classification.

        Notes:
            Miss selection stays on the GPU and copies only the compact miss
            set to the CPU. This avoids synchronizing the complete top-K tensor
            once per Full layer, which is especially expensive during decode.
        """
        missing_ids = self._miss_ids_for_topk(layer_id, true_topk)
        if missing_ids.numel() == 0:
            return
        self.submit_miss_reads(
            layer_id,
            missing_ids,
            request_token=request_token,
            profile_source_layer=profile_source_layer,
            profile_operation_id=profile_operation_id,
            profile_kind=profile_kind,
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
        if not detailed_io_nvtx.enabled:
            return self._wait_for_layer_impl(layer_id, timeout_s, "disabled")
        kind = (
            "dsa_indexer"
            if self._data_group == "indexer"
            else ("hca_deterministic" if int(layer_id) in self.hca_layer_ids else "csa")
        )
        with detailed_io_nvtx.range(
            CsaNvtxEvent.CONSUMER_WAIT,
            layer_id=int(layer_id),
            target_layer_id=int(layer_id),
            request_id=self.active_request_id,
            attributes={"kind": kind},
        ):
            return self._wait_for_layer_impl(layer_id, timeout_s, kind)

    def _wait_for_layer_impl(
        self,
        layer_id: int,
        timeout_s: float,
        profile_kind: str,
    ) -> bool:
        """Implement :meth:`wait_for_layer` under its detailed NVTX range."""
        profile = _timing_enabled()
        wait_start = time.perf_counter()
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._scheduled_layer_futures_lock:
            scheduled = self._scheduled_layer_futures.pop(int(layer_id), None)
        scheduled_present = scheduled is not None
        scheduled_start = time.perf_counter()
        if scheduled is not None:
            try:
                with detailed_io_nvtx.range(
                    CsaNvtxEvent.FUTURE_WAIT,
                    layer_id=int(layer_id),
                    target_layer_id=int(layer_id),
                    request_id=self.active_request_id,
                    attributes={"kind": profile_kind},
                ):
                    result = scheduled.result(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                # A gate-aligned shard gather books ``pending_read_count`` when
                # it is prepared and only releases it when finalized.  Waiting
                # on the future alone leaves that booking outstanding, so the
                # loop below would block until the deadline on work that has
                # already landed.  Mirror ``wait_for_tracked_submission``.
                if isinstance(result, _DeferredShardGather):
                    self.finalize_deferred_shard_gather(result)
            except Exception:
                logger.exception(
                    "CSAAttentionKVPrefetchManager: scheduled layer %d "
                    "submission failed or timed out",
                    layer_id,
                )
                return False
        scheduled_wait_ms = (time.perf_counter() - scheduled_start) * 1000.0
        state = self._layers.get(int(layer_id))
        if state is None:
            return True
        pending_start = time.perf_counter()
        with detailed_io_nvtx.range(
            CsaNvtxEvent.CONDITION_WAIT,
            layer_id=int(layer_id),
            target_layer_id=int(layer_id),
            request_id=self.active_request_id,
            attributes={"kind": profile_kind},
        ):
            with state.pending_reads_lock:
                pending_before = int(state.pending_read_count)
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
        pending_wait_ms = (time.perf_counter() - pending_start) * 1000.0
        drain_start = time.perf_counter()
        self.drain_for_layer(int(layer_id))
        drain_ms = (time.perf_counter() - drain_start) * 1000.0
        if profile and self.active_request_id:
            logger.info(
                "CSA_GATE_PROFILE request=%s layer=%d scheduled=%d "
                "pending_before=%d scheduled_wait_ms=%.3f "
                "pending_wait_ms=%.3f drain_ms=%.3f total_ms=%.3f",
                self.active_request_id,
                int(layer_id),
                int(scheduled_present),
                pending_before,
                scheduled_wait_ms,
                pending_wait_ms,
                drain_ms,
                (time.perf_counter() - wait_start) * 1000.0,
            )
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
            gate_attributes = {
                "kind": (
                    "dsa_indexer"
                    if self._data_group == "indexer"
                    else (
                        "hca_deterministic"
                        if int(layer_id) in self.hca_layer_ids
                        else "csa"
                    )
                )
            }
            with csa_pipeline_nvtx.range(
                CsaNvtxEvent.TARGET_GATE_WAIT,
                layer_id=int(layer_id),
                target_layer_id=int(layer_id),
                request_id=self.active_request_id,
                attributes=gate_attributes,
            ):
                with detailed_io_nvtx.range(
                    CsaNvtxEvent.CUDA_EVENT_WAIT,
                    layer_id=int(layer_id),
                    target_layer_id=int(layer_id),
                    request_id=self.active_request_id,
                    attributes=gate_attributes,
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

    def _drain_for_layer_until(self, layer_id: int, deadline: float) -> bool:
        """Release one layer's staging buffers without exceeding a deadline."""
        state = self._layers.get(int(layer_id))
        if state is None:
            return True
        with state.pending_reads_lock:
            event = state.last_drain_event
        if event is not None:
            query = getattr(event, "query", None)
            if not callable(query):
                return False
            while not bool(query()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                time.sleep(min(0.001, remaining))
        with state.pending_reads_lock:
            pending = state.pending_drains[:]
            state.pending_drains.clear()
            state.last_drain_event = None
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
        return True

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

            try:
                from vllm.model_executor.layers.sparse_attn_indexer import (
                    register_lmcache_topk_chunk_callback,
                )

                cache = getattr(indexer_module, "k_cache", None)
                hook_prefix = str(cache.prefix)

                def _observe_topk_chunk(
                    topk_indices: torch.Tensor,
                    chunk_index: int,
                    *,
                    _layer_id: int = int(layer_id),
                ) -> None:
                    mgr._accumulate_true_topk_chunk(
                        _layer_id,
                        topk_indices,
                        chunk_index,
                    )
                    mgr._schedule_exact_topk_chunk(
                        _layer_id,
                        topk_indices,
                        chunk_index,
                    )

                register_lmcache_topk_chunk_callback(
                    hook_prefix,
                    _observe_topk_chunk,
                )
                self._exact_chunk_hook_prefixes[indexer_module] = hook_prefix
                logger.info(
                    "CSAAttentionKVPrefetchManager: streamed true-topK union "
                    "attached layer=%d prefix=%s exact_prefetch_chunks=%d",
                    layer_id,
                    hook_prefix,
                    (
                        self._exact_chunk_prefetch_limit
                        if self._exact_chunk_prefetch_executor is not None
                        else 0
                    ),
                )
            except (AttributeError, ImportError, TypeError):
                logger.exception(
                    "CSAAttentionKVPrefetchManager: failed to attach true-topK "
                    "chunk observer for layer %d; final correction remains active",
                    layer_id,
                )

            def _patched_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                if not getattr(mgr, "_external_kv_forward_active", True):
                    return orig_forward(*args, **kwargs)
                request_token = mgr.active_request_token
                if not request_token[0]:
                    return orig_forward(*args, **kwargs)
                t0 = time.perf_counter() if _timing_enabled() else 0.0
                wait_ms = 0.0
                # The native Lightning Indexer reads its own compact K cache
                # before producing ``true_topk``.  The connector's optional
                # per-layer callback is not a reliable consumption boundary,
                # so gate the native-cache stream here.  Besides correctness,
                # this advances the bounded read window and drains completed
                # staging buffers for the early CSA layers that have no L2
                # proxy trigger.
                from lmcache.v1.indexer_ssd_manager import (
                    get_indexer_ssd_manager,
                )

                indexer_manager = get_indexer_ssd_manager()
                if indexer_manager is not None:
                    wait_native = getattr(
                        indexer_manager,
                        "wait_for_native_indexer_layer",
                        None,
                    )
                    if callable(wait_native) and not wait_native(layer_id):
                        raise RuntimeError(
                            f"native indexer cache is unavailable for layer {layer_id}"
                        )
                # Dense shard-gather owns an independent NCCL communicator,
                # but its background producer can arrive with inter-rank
                # skew.  Join it before the official indexer's forward
                # collectives: otherwise vLLM 0.26 can enqueue the two
                # communicator domains in a different order across ranks and
                # deadlock at the third dense layer on a reused request.
                # This is still the end of the two-layer FFN prefetch window;
                # speculative proxy work remains asynchronous below.
                if hasattr(mgr, "_scheduled_layer_futures_lock") and not (
                    mgr.wait_for_tracked_submission(layer_id)
                ):
                    raise RuntimeError(
                        f"tracked shard submission failed for layer {layer_id}"
                    )
                reused_residual = False
                true_indexer_start = time.perf_counter() if _timing_enabled() else 0.0
                with csa_pipeline_nvtx.range(
                    CsaNvtxEvent.TRUE_INDEXER,
                    layer_id=layer_id,
                    request_id=request_token[0],
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
                exact_chunk_wait_start = (
                    time.perf_counter() if _timing_enabled() else 0.0
                )
                mgr._wait_for_exact_topk_chunks(layer_id)
                exact_chunk_wait_ms = (
                    (time.perf_counter() - exact_chunk_wait_start) * 1000.0
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
                    if indexer_manager is not None:
                        indexer_manager.record_csa_prediction_accuracy(
                            layer_id,
                            active_topk,
                        )
                    first_drain_ms = 0.0
                    t_miss0 = time.perf_counter() if _timing_enabled() else 0.0
                    # The paired vLLM patch compacts the compressed-K gather
                    # through a top-K-derived block table.  Therefore prefill
                    # and decode both need only the physical pages referenced
                    # by the true indexer output; loading a whole layer here
                    # defeats sparse I/O and makes the 8192-row union approach
                    # the complete prefix.
                    miss_ids = mgr._miss_ids_for_topk(layer_id, active_topk)
                    miss_ms = (
                        (time.perf_counter() - t_miss0) * 1000.0
                        if _timing_enabled()
                        else 0.0
                    )
                    if miss_ids.numel():
                        mgr.submit_miss_reads(
                            layer_id,
                            miss_ids,
                            request_token=request_token,
                        )
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
                            "full_cached_prefix_selected=%d "
                            "reused_residual=%d "
                            "prediction_ready=%d true_indexer_ms=%.3f wait_ms=%.3f "
                            "exact_chunk_wait_ms=%.3f "
                            "first_drain_ms=%.3f "
                            "miss_filter_ms=%.3f second_drain_ms=%.3f "
                            "total_ms=%.3f",
                            str(true_topk.device),
                            layer_id,
                            active_rows,
                            int(active_topk.numel()),
                            len(miss_ids),
                            int(
                                mgr._layers[
                                    int(layer_id)
                                ].true_selected_covers_cached_prefix
                            ),
                            int(reused_residual),
                            int(prediction_ready),
                            true_indexer_ms,
                            wait_ms,
                            exact_chunk_wait_ms,
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
                hook_prefixes = getattr(self, "_exact_chunk_hook_prefixes", {})
                hook_prefix = hook_prefixes.pop(module, None)
                if hook_prefix is not None:
                    try:
                        from vllm.model_executor.layers.sparse_attn_indexer import (
                            unregister_lmcache_topk_chunk_callback,
                        )

                        unregister_lmcache_topk_chunk_callback(hook_prefix)
                    except (ImportError, TypeError):
                        logger.exception(
                            "CSAAttentionKVPrefetchManager: failed to remove "
                            "exact top-K chunk hook prefix=%s",
                            hook_prefix,
                        )
                try:
                    delattr(module, "_lmcache_csa_attention_kv_original_forward")
                except AttributeError:
                    pass
            self._patched_modules.clear()

    def close(self) -> None:
        """Release resources held by this manager."""
        with self._request_transition_lock:
            if self._closed:
                return
            if not self._deactivate_request_locked():
                logger.error(
                    "CSAAttentionKVPrefetchManager: close left resources "
                    "attached because request I/O did not drain"
                )
                return
            self._closed = True
            with self._request_state:
                self._request_lifecycle = "closed"
            self.unpatch()
            if getattr(self, "_exact_chunk_prefetch_executor", None) is not None:
                self._exact_chunk_prefetch_executor.shutdown(wait=True)
                self._exact_chunk_prefetch_executor = None
            shard_transport = getattr(self, "_shard_transport", None)
            if shard_transport is not None:
                shard_transport.close()
                self._shard_transport = None
            self._layers.clear()
            self._prediction_waiter = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _accumulate_true_topk_chunk(
        self,
        layer_id: int,
        topk_indices: torch.Tensor,
        chunk_index: int,
    ) -> None:
        """Accumulate one completed true-topK chunk into a page bitmap."""
        marker = getattr(_csa_c_ops, "mark_csa_selected_blocks_into", None)
        state = self._layers.get(int(layer_id))
        request_token = self.active_request_token
        if (
            not callable(marker)
            or state is None
            or not request_token[0]
            or not isinstance(topk_indices, torch.Tensor)
            or not topk_indices.is_cuda
            or topk_indices.numel() == 0
        ):
            return
        expected_chunk = state.streamed_selected_chunks
        same_request = state.streamed_selected_request_token == request_token
        if chunk_index == 0:
            same_request = False
        elif not same_request or int(chunk_index) != expected_chunk:
            state.streamed_selected_failed = True
            return

        device = state.in_pool_bitmap.device
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(topk_indices.device))
        prepare_stream = self._prepare_stream_for(device)
        with (
            torch.cuda.device(device),
            torch.inference_mode(),
            torch.cuda.stream(prepare_stream),
        ):
            prepare_stream.wait_event(ready_event)
            if not same_request:
                state.streamed_selected_blocks_bitmap = torch.zeros(
                    int(state.in_pool_bitmap.numel()),
                    dtype=torch.int32,
                    device=device,
                )
                state.streamed_selected_rows = 0
                state.streamed_selected_chunks = 0
                state.streamed_selected_request_token = request_token
                state.streamed_selected_failed = False
            bitmap = state.streamed_selected_blocks_bitmap
            if bitmap is None:
                state.streamed_selected_failed = True
                return
            topk_indices.record_stream(prepare_stream)
            marker(
                topk_indices.contiguous(),
                bitmap,
                int(bitmap.numel()),
                state.compressed_block_size,
            )
            done_event = torch.cuda.Event()
            done_event.record(prepare_stream)
        state.streamed_selected_rows += int(topk_indices.shape[0])
        state.streamed_selected_chunks += 1
        state.streamed_selected_event = done_event

    def _streamed_true_topk_union(
        self,
        state: CSAAttentionKVLayerState,
        active_rows: int,
    ) -> Optional[torch.Tensor]:
        """Return the complete streamed page union, or ``None`` to fall back."""
        bitmap = state.streamed_selected_blocks_bitmap
        event = state.streamed_selected_event
        if (
            state.streamed_selected_failed
            or state.streamed_selected_request_token != self.active_request_token
            or state.streamed_selected_rows != int(active_rows)
            or state.streamed_selected_chunks <= 0
            or bitmap is None
            or event is None
        ):
            return None
        torch.cuda.current_stream(bitmap.device).wait_event(event)
        return bitmap

    def _schedule_exact_topk_chunk(
        self,
        layer_id: int,
        topk_indices: torch.Tensor,
        chunk_index: int,
    ) -> None:
        """Queue one exact true-topK chunk without blocking model execution."""
        executor = self._exact_chunk_prefetch_executor
        if (
            executor is None
            or chunk_index < 0
            or chunk_index >= self._exact_chunk_prefetch_limit
            or not isinstance(topk_indices, torch.Tensor)
            or not topk_indices.is_cuda
            or topk_indices.numel() == 0
        ):
            return
        request_token = self.active_request_token
        if not request_token[0]:
            return
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(topk_indices.device))
        future = executor.submit(
            self._run_exact_topk_chunk,
            int(layer_id),
            topk_indices,
            int(chunk_index),
            ready_event,
            request_token,
        )
        with self._exact_chunk_futures_lock:
            self._exact_chunk_futures.setdefault(int(layer_id), []).append(future)

    def _run_exact_topk_chunk(
        self,
        layer_id: int,
        topk_indices: torch.Tensor,
        chunk_index: int,
        ready_event: torch.cuda.Event,
        request_token: Tuple[str, int],
    ) -> bool:
        """Select and read exact blocks from one completed indexer chunk."""
        state = self._layers.get(int(layer_id))
        if state is None:
            return False
        with self._request_state:
            current_token = (
                self._active_request_id or "",
                self._request_generation,
            )
            if self._request_lifecycle != "active" or request_token != current_token:
                return False
        device = state.in_pool_bitmap.device
        prepare_stream = self._prepare_stream_for(device)
        with (
            torch.cuda.device(device),
            torch.inference_mode(),
            torch.cuda.stream(prepare_stream),
        ):
            prepare_stream.wait_event(ready_event)
            missing_ids = self._missing_ids_for_exact_topk_chunk(
                int(layer_id),
                topk_indices,
            )
        if missing_ids.numel() == 0:
            return True
        return self._submit_reads(
            int(layer_id),
            missing_ids,
            label=f"exact_chunk_{chunk_index}",
            io_priority="demand",
            raise_on_error=False,
            request_token=request_token,
        )

    def _wait_for_exact_topk_chunks(self, layer_id: int) -> None:
        """Join progressive reads for one layer before final correction."""
        lock = getattr(self, "_exact_chunk_futures_lock", None)
        if lock is None:
            return
        with lock:
            futures = self._exact_chunk_futures.pop(int(layer_id), [])
        for future in futures:
            try:
                future.result()
            except Exception:
                # Progressive I/O is an optimization only. The authoritative
                # full-topK correction immediately following this join will
                # retry every block that did not become resident.
                logger.exception(
                    "CSAAttentionKVPrefetchManager: exact top-K chunk "
                    "prefetch failed for layer %d; using final correction",
                    layer_id,
                )

    def _drain_exact_topk_futures(self, deadline: float) -> bool:
        """Cancel or join every progressive future before request teardown."""
        lock = getattr(self, "_exact_chunk_futures_lock", None)
        if lock is None:
            return True
        with lock:
            futures = [
                future
                for layer_futures in self._exact_chunk_futures.values()
                for future in layer_futures
            ]
            self._exact_chunk_futures.clear()
        for future in futures:
            future.cancel()
        for future in futures:
            try:
                future.result(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                if not future.done():
                    return False
        return True

    def _missing_ids_for_exact_topk_chunk(
        self,
        layer_id: int,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return unique cached-prefix blocks selected by one true-topK chunk."""
        state = self._layers.get(int(layer_id))
        if state is None or not state.chunks:
            return torch.empty(0, dtype=torch.int64)
        entries = topk_indices.reshape(-1)
        device = state.in_pool_bitmap.device
        if entries.device != device:
            entries = entries.to(device)
        limit = min(
            int(state.chunks[-1].end_compressed_block),
            int(state.in_pool_bitmap.shape[0]),
        )
        if limit <= 0:
            return torch.empty(0, dtype=torch.int64)
        native_select = getattr(_csa_c_ops, "select_missing_csa_blocks", None)
        if callable(native_select) and entries.is_cuda and state.in_pool_bitmap.is_cuda:
            return native_select(
                entries.contiguous(),
                state.in_pool_bitmap,
                limit,
                state.compressed_block_size,
            ).cpu()
        entries = entries.to(torch.int64)
        block_ids = entries // state.compressed_block_size
        valid = (entries >= 0) & (block_ids < limit)
        seen = torch.zeros(limit, dtype=torch.bool, device=device)
        seen[block_ids[valid]] = True
        return (seen & ~state.in_pool_bitmap[:limit]).nonzero().reshape(-1).cpu()

    def _submit_reads(
        self,
        layer_id: int,
        compressed_block_ids: Sequence[int] | torch.Tensor,
        label: str,
        io_priority: str = "demand",
        *,
        raise_on_error: bool = False,
        request_token: Optional[Tuple[str, int]] = None,
        source_layer_id: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
        profile_kind: Optional[str] = None,
    ) -> bool:
        """Run one read submission while pinning the active request state."""
        with self._request_state:
            current_token = (
                self._active_request_id or "",
                self._request_generation,
            )
            if (
                self._request_lifecycle != "active"
                or self._active_request_id is None
                or (request_token is not None and request_token != current_token)
            ):
                if raise_on_error:
                    raise RuntimeError("request read plan is inactive or stale")
                return False
            self._active_submissions += 1
        try:
            return self._submit_reads_active(
                layer_id,
                compressed_block_ids,
                label,
                io_priority,
                raise_on_error=raise_on_error,
                source_layer_id=source_layer_id,
                profile_operation_id=profile_operation_id,
                profile_kind=profile_kind,
            )
        finally:
            with self._request_state:
                self._active_submissions -= 1
                self._request_state.notify_all()

    def _submit_reads_active(
        self,
        layer_id: int,
        compressed_block_ids: Sequence[int] | torch.Tensor,
        label: str,
        io_priority: str = "demand",
        *,
        raise_on_error: bool = False,
        source_layer_id: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
        profile_kind: Optional[str] = None,
    ) -> bool:
        """Submit reads and publish only blocks that actually completed.

        Args:
            layer_id: Target transformer layer id.
            compressed_block_ids: Logical block ids requested by prediction
                or exact correction.
            label: Profiling label for the submission.
            io_priority: ``demand`` requires complete execution;
                ``lookahead`` is required to reach Tutti but may fall back to
                exact correction, and ``speculative`` permits cancellation.
            raise_on_error: Re-raise submission and completion failures.
            source_layer_id: Optional layer that produced this I/O request.
            profile_operation_id: Optional parent operation correlation id.
            profile_kind: Optional profile-only I/O classification.

        Returns:
            ``True`` when the submission itself completed safely.  A
            speculative partial completion is safe and returns ``True``;
            only its actual completed ids become resident.

        Raises:
            RuntimeError: If strict error propagation is requested and the
                layer is unavailable or its I/O fails.
            ValueError: If ``io_priority`` is invalid.
        """
        if io_priority not in {"demand", "lookahead", "speculative"}:
            raise ValueError("io_priority must be demand, lookahead, or speculative")
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
        active_sharding = bool(
            getattr(self, "_shard_transport", None) is not None
            and getattr(self, "_shard_config", None) is not None
            and self._shard_config.enabled
            and self._shard_config.csa_enabled
            and (label == "dense" or label.startswith("predicted_"))
        )
        # Sharded participants must not return independently after the local
        # loaded/pending filter: one rank may already own every requested row
        # while another still needs data. All ranks retain the canonical
        # request union for metadata consensus; ``new_ids`` below is used only
        # by the LOCAL_DIRECT fallback and pending-state accounting.
        if candidate_ids.numel() == 0 and not active_sharding:
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
            if new_ids.numel() == 0 and not active_sharding:
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

        operation_id = profile_operation_id or f"{label}-{time.monotonic_ns()}"
        effective_profile_kind = profile_kind or _io_profile_kind(label)
        source_layer = (
            int(source_layer_id) if source_layer_id is not None else int(layer_id)
        )
        issue_profile_kwargs = (
            {
                "profile_kind": effective_profile_kind,
                "profile_source_layer": source_layer,
                "profile_operation_id": operation_id,
            }
            if csa_pipeline_nvtx.enabled or detailed_io_nvtx.enabled
            else {}
        )
        io_range = csa_pipeline_nvtx.start_io(
            layer_id=source_layer,
            target_layer_id=int(layer_id),
            operation_id=operation_id,
            request_id=self.active_request_id,
            attributes={
                "kind": effective_profile_kind,
                "detail": label,
                "blocks": new_count,
                "source_known": int(source_layer_id is not None),
            },
        )
        try:
            # Predicted reads often finish on a background thread, outside
            # vLLM's model-forward inference_mode context.  The target K cache
            # and resident bitmap may be inference tensors, so all in-place GPU
            # updates must re-enter inference_mode here as well.
            # ThreadPoolExecutor workers do not inherit vLLM's per-rank CUDA
            # current-device state.  Without an explicit device guard, ranks
            # other than zero enter the correct private stream but restore a
            # device-0 stream on exit; the error is then reported as an
            # asynchronous illegal access and invalidates every later layer.
            k_cache_tensor = getattr(state, "k_cache_tensor", None)
            device_guard = (
                torch.cuda.device(k_cache_tensor.device)
                if isinstance(k_cache_tensor, torch.Tensor) and k_cache_tensor.is_cuda
                else nullcontext()
            )
            with torch.inference_mode(), device_guard:
                preferred_mode = None
                issue_ids = new_ids
                if active_sharding:
                    preferred_mode = (
                        SSDReadMode.SHARD_GATHER_DENSE
                        if label == "dense"
                        else SSDReadMode.SHARD_GATHER_PREDICTED
                    )
                    issue_ids = candidate_ids
                with detailed_io_nvtx.range(
                    CsaNvtxEvent.IO_LOADER_CALL,
                    layer_id=source_layer,
                    target_layer_id=int(layer_id),
                    operation_id=operation_id,
                    request_id=self.active_request_id,
                    attributes={
                        "kind": effective_profile_kind,
                        "detail": label,
                        "blocks": new_count,
                    },
                ):
                    if active_sharding:
                        event, issued_memory_objs, completed_ids = self._issue_reads(
                            state,
                            issue_ids,
                            io_priority=io_priority,
                            local_fallback_ids=new_ids,
                            preferred_mode=preferred_mode,
                            **issue_profile_kwargs,
                        )
                    else:
                        # Preserve the original call contract for local reads and
                        # integrations that replace the issue hook in tests.
                        event, issued_memory_objs, completed_ids = self._issue_reads(
                            state,
                            issue_ids,
                            io_priority=io_priority,
                            **issue_profile_kwargs,
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
            if isinstance(exc, ShardCollectiveError) and exc.data_submitted:
                raise RuntimeError(
                    f"submitted shard-gather failed for layer {layer_id}"
                ) from exc
            if raise_on_error:
                raise RuntimeError(f"failed to materialize layer {layer_id}") from exc
            return False
        if completed_ids.numel() == 0:
            csa_pipeline_nvtx.finish_io(
                io_range,
                layer_id=int(layer_id),
                target_layer_id=int(layer_id),
                operation_id=operation_id,
                request_id=self.active_request_id,
                status="not_submitted",
            )
            with state.pending_reads_lock:
                with torch.inference_mode():
                    state.pending_reads_bitmap[new_ids] = False
                state.pending_read_count = max(
                    0,
                    state.pending_read_count - new_count,
                )
                state.pending_reads_lock.notify_all()
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
        local_fallback_ids: Optional[Sequence[int] | torch.Tensor] = None,
        preferred_mode: Optional[SSDReadMode] = None,
        profile_kind: Optional[str] = None,
        profile_source_layer: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
    ) -> Tuple[Optional[torch.cuda.Event], List[Any], torch.Tensor]:
        """Choose shard-gather or the preserved rank-local read path."""
        if (
            getattr(self, "_shard_transport", None) is None
            or not getattr(self, "_shard_config", None)
            or preferred_mode is None
        ):
            return self._issue_local_reads(
                state,
                local_fallback_ids
                if local_fallback_ids is not None
                else sorted_block_ids,
                io_priority=io_priority,
                profile_kind=profile_kind,
                profile_source_layer=profile_source_layer,
                profile_operation_id=profile_operation_id,
            )
        local_capability = self._shard_capability_for(
            state,
            sorted_block_ids,
            mode=preferred_mode,
        )
        # A process group requires one total collective order per rank. Proxy
        # I/O uses multiple workers, so serialize metadata and data collectives
        # while retaining overlap between SSD reads and model compute.
        with self._shard_collective_lock:
            return self._issue_shard_gather(
                state,
                sorted_block_ids,
                mode=preferred_mode,
                io_priority=io_priority,
                local_fallback_ids=(
                    local_fallback_ids
                    if local_fallback_ids is not None
                    else sorted_block_ids
                ),
                local_capability=local_capability,
                profile_kind=profile_kind,
                profile_source_layer=profile_source_layer,
                profile_operation_id=profile_operation_id,
            )

    def _issue_local_reads(
        self,
        state: CSAAttentionKVLayerState,
        sorted_block_ids: Sequence[int] | torch.Tensor,
        *,
        io_priority: str = "demand",
        profile_kind: Optional[str] = None,
        profile_source_layer: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
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
        selected_cpu = torch.as_tensor(
            sorted_block_ids,
            dtype=torch.int64,
            device="cpu",
        ).reshape(-1)
        profile_kwargs = (
            {
                "profile_kind": profile_kind,
                "profile_source_layer": profile_source_layer,
                "profile_operation_id": profile_operation_id,
            }
            if any(
                value is not None
                for value in (
                    profile_kind,
                    profile_source_layer,
                    profile_operation_id,
                )
            )
            else {}
        )
        full_layer_major = bool(
            len(state.chunks) > 1
            and all(chunk.layer_major for chunk in state.chunks)
            and state.layer_major_dst_rows_table is not None
            and selected_cpu.numel() == int(state.chunks[-1].end_compressed_block)
            and int(selected_cpu[0]) == 0
            and int(selected_cpu[-1]) == int(state.chunks[-1].end_compressed_block) - 1
        )
        if full_layer_major:
            return self._issue_full_multi_layer_major_read(
                state,
                io_priority=io_priority,
                **profile_kwargs,
            )
        if (
            len(state.chunks) == 1
            and state.chunks[0].layer_major
            and state.layer_major_dst_rows_table is not None
        ):
            return self._issue_layer_major_read(
                state,
                selected_cpu,
                io_priority=io_priority,
                **profile_kwargs,
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
                **profile_kwargs,
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

        # Restore our full-record extent snapshot only after Tutti owns its
        # queue lock.  Registering before lock acquisition races the ordinary
        # retrieve path, which can replace the same synthetic pool path with
        # a filtered extent table while this request waits for the queue.
        def _restore_lba_cache() -> None:
            if self._pending_raw_lba_cache:
                self._tutti_loader.ensure_lba_cache(self._pending_raw_lba_cache)

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
            "before_batch": _restore_lba_cache,
            # A predicted union can span hundreds of tiny ranges. Release the
            # single Tutti queue between bounded speculative batches so HCA
            # and true-topK demand reads never queue behind the whole walk.
            "lock_per_batch": io_priority == "speculative",
            "profile_layer_id": getattr(state, "layer_id", -1),
            "profile_kind": profile_kind,
            "profile_operation_id": profile_operation_id,
            "profile_source_layer": profile_source_layer,
        }
        if io_priority == "speculative":
            # A long-prefix prediction can touch ranges in every layer-major
            # segment. Submit the bounded selected ranges in large batches and
            # rely on the demand gate for preemption instead of spending the
            # two-layer compute window in artificial throttling.
            load_kwargs.update(
                max_batch_bytes=128 * 1024**2,
                max_batch_ios=256,
                throttle_speculative=False,
                wait_for_active_io=True,
            )
        elif io_priority == "lookahead":
            # A required prediction must not be silently dropped, but the
            # old speculative per-batch path can expose an incomplete NVMe
            # batch to the polling kernel. Wait behind already-announced HCA
            # or indexer demand, then use the same whole-call geometry as the
            # proven miss-correction path. The predicted union is bounded, so
            # a newly arriving demand read waits for only this small call.
            load_kwargs["wait_for_active_io"] = True
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

    def _issue_full_multi_layer_major_read(
        self,
        state: CSAAttentionKVLayerState,
        *,
        io_priority: str,
        profile_kind: Optional[str] = None,
        profile_source_layer: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
    ) -> Tuple[Optional[torch.cuda.Event], List[Any], torch.Tensor]:
        """Restore a complete layer from several admitted generation objects.

        Native indexer restore consumes every block of a layer. After deferred
        suffix admission, that layer is represented by two or more contiguous
        layer-major objects. The general sparse path needlessly walks every
        block to rediscover two full-object ranges. This path submits one range
        per generation and performs one fused scatter per returned object.

        Args:
            state: Registered layer whose chunks densely cover the layer.
            io_priority: Tutti admission class for this demand read.

        Returns:
            A completed synchronous read tuple compatible with
            :meth:`_issue_reads`.

        Raises:
            RuntimeError: If the layer-major plan or returned payload is
                incomplete.
        """
        chunks = state.chunks
        dst_rows_table = state.layer_major_dst_rows_table
        if len(chunks) <= 1 or dst_rows_table is None:
            raise RuntimeError("multi-generation layer-major plan is unavailable")
        if any(not chunk.layer_major for chunk in chunks):
            raise RuntimeError("multi-generation plan contains a non-layer object")

        block_nbytes = int(state.compressed_block_size * state.token_bytes)
        if block_nbytes <= 0:
            raise RuntimeError("invalid multi-generation layer-major block size")
        read_ranges: list[tuple[KVObjectByteRange, ...]] = []
        payload_skips: list[int] = []
        expected = 0
        for chunk in chunks:
            if chunk.first_compressed_block != expected:
                raise RuntimeError("multi-generation layer-major plan has a gap")
            payload_nbytes = int(chunk.n_compressed_blocks) * block_nbytes
            read_nbytes = int(chunk.read_length) or (
                int(chunk.payload_skip) + payload_nbytes
            )
            if read_nbytes < int(chunk.payload_skip) + payload_nbytes:
                raise RuntimeError("multi-generation layer-major read is truncated")
            read_ranges.append(
                (
                    KVObjectByteRange(
                        offset=int(chunk.layer_byte_offset),
                        length=read_nbytes,
                        target_offset=0,
                    ),
                )
            )
            payload_skips.append(int(chunk.payload_skip))
            expected = chunk.end_compressed_block

        scatter_stream = self._scatter_stream_for(state.k_cache_tensor.device)
        if state.block_slot_scatter:
            k_cache = state.k_cache_tensor.view(torch.uint8)
        else:
            k_cache = state.k_cache_tensor.view(torch.uint8).reshape(
                int(state.k_cache_tensor.shape[0]),
                -1,
            )

        def _restore_lba_cache() -> None:
            if self._pending_raw_lba_cache:
                self._tutti_loader.ensure_lba_cache(self._pending_raw_lba_cache)

        def _scatter_tensor_batch(
            batch_start: int,
            batch_results: List[Optional[Any]],
        ) -> None:
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                for offset_in_batch, memory_obj in enumerate(batch_results):
                    if memory_obj is None or memory_obj.raw_tensor is None:
                        raise RuntimeError(
                            "multi-generation layer-major read returned no payload"
                        )
                    chunk_index = batch_start + offset_in_batch
                    chunk = chunks[chunk_index]
                    payload_nbytes = int(chunk.n_compressed_blocks) * block_nbytes
                    payload_skip = payload_skips[chunk_index]
                    flat = memory_obj.raw_tensor.view(torch.uint8).reshape(-1)
                    if int(flat.numel()) < payload_skip + payload_nbytes:
                        raise RuntimeError(
                            "multi-generation layer-major tensor is truncated"
                        )
                    source = flat[payload_skip : payload_skip + payload_nbytes].view(
                        int(chunk.n_compressed_blocks), block_nbytes
                    )
                    rows = dst_rows_table[
                        chunk.first_compressed_block : chunk.end_compressed_block
                    ]
                    if state.block_slot_scatter:
                        slot_size = state.block_slot_size
                        block_ids = torch.div(rows, slot_size, rounding_mode="floor")
                        slot_ids = rows - block_ids * slot_size
                        k_cache.index_put_((block_ids, slot_ids), source)
                    else:
                        k_cache.index_copy_(0, rows, source)
            scatter_stream.synchronize()

        def _scatter_raw_batch(
            batch_start: int,
            completed_indices: List[int],
            completed_offsets: List[int],
            completed_nbytes: List[int],
            staging: torch.Tensor,
        ) -> None:
            if _csa_c_ops is None:
                raise RuntimeError("multi-generation raw scatter requires c_ops")
            scatter = getattr(_csa_c_ops, "scatter_rows_from_object_ptrs", None)
            if scatter is None:
                raise RuntimeError("multi-generation fused scatter is unavailable")
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                for local_index, offset, nbytes in zip(
                    completed_indices,
                    completed_offsets,
                    completed_nbytes,
                    strict=True,
                ):
                    chunk_index = batch_start + local_index
                    chunk = chunks[chunk_index]
                    payload_skip = payload_skips[chunk_index]
                    payload_nbytes = int(chunk.n_compressed_blocks) * block_nbytes
                    if int(nbytes) < payload_skip + payload_nbytes:
                        raise RuntimeError(
                            "multi-generation raw layer-major read is truncated"
                        )
                    source_ptr = int(staging.data_ptr()) + int(offset) + payload_skip
                    source_ptrs = torch.tensor(
                        [source_ptr],
                        dtype=torch.int64,
                        device=state.k_cache_tensor.device,
                    )
                    rows = dst_rows_table[
                        chunk.first_compressed_block : chunk.end_compressed_block
                    ]
                    scatter(
                        source_ptrs,
                        k_cache,
                        rows,
                        int(chunk.n_compressed_blocks),
                        block_nbytes,
                        state.block_slot_size if state.block_slot_scatter else 0,
                        source_ptr % 8 == 0,
                    )
            scatter_stream.synchronize()

        load_kwargs: Dict[str, Any] = {
            "shapes_per_key": None,
            "file_offsets": [0] * len(chunks),
            "read_ranges_per_key": read_ranges,
            "io_priority": io_priority,
            "before_batch": _restore_lba_cache,
            "max_batch_ios": 256,
            "max_batch_bytes": 128 * 1024**2,
            "wait_for_active_io": io_priority == "lookahead",
            "profile_layer_id": getattr(state, "layer_id", -1),
            "profile_kind": profile_kind,
            "profile_operation_id": profile_operation_id,
            "profile_source_layer": profile_source_layer,
        }
        if _csa_c_ops is not None and hasattr(
            _csa_c_ops,
            "scatter_rows_from_object_ptrs",
        ):
            load_kwargs["on_raw_batch_loaded"] = _scatter_raw_batch
        else:
            load_kwargs["on_batch_loaded"] = _scatter_tensor_batch
        self._tutti_loader.load_chunks_to_hbm(
            [chunk.key for chunk in chunks],
            [chunk.disk_meta for chunk in chunks],
            **load_kwargs,
        )
        completed_ids = torch.arange(expected, dtype=torch.int64)
        with torch.inference_mode(), torch.cuda.stream(scatter_stream):
            bitmap_ids = completed_ids.to(
                device=state.in_pool_bitmap.device,
                non_blocking=True,
            )
            state.in_pool_bitmap.index_fill_(0, bitmap_ids, True)
        scatter_stream.synchronize()
        return None, [], completed_ids

    def _issue_layer_major_read(
        self,
        state: CSAAttentionKVLayerState,
        selected_block_ids: torch.Tensor,
        *,
        io_priority: str,
        profile_kind: Optional[str] = None,
        profile_source_layer: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
    ) -> Tuple[Optional[torch.cuda.Event], List[Any], torch.Tensor]:
        """Read and scatter selected rows from a layer-major KV object.

        A long-prefix layer can resolve to more physical extents than the
        Tutti NVMe queue can hold.  Submit bounded logical block segments
        instead of one oversized key so the loader never launches a submit
        kernel with more commands than its SQ/CQ and status buffers.
        """
        chunk = state.chunks[0]
        dst_rows_table = state.layer_major_dst_rows_table
        if dst_rows_table is None:
            raise RuntimeError("layer-major destination rows are unavailable")
        selected = selected_block_ids.to(device="cpu", dtype=torch.int64).reshape(-1)
        available_blocks = int(chunk.n_compressed_blocks)
        block_nbytes = int(chunk.bytes_per_block)
        if selected.numel() == 0:
            return None, [], selected
        if block_nbytes <= 0:
            raise RuntimeError("invalid layer-major block size")
        if int(selected[0]) < 0 or int(selected[-1]) >= available_blocks:
            raise RuntimeError("layer-major selected block id is out of range")

        # One compressed block normally resolves to at most a handful of
        # extents.  A 256-block bound stays well below the 1023 usable entries
        # of the production Tutti queue even on fragmented object-store files,
        # while keeping the number of callbacks small for a 480K prefix.
        # Keep each loader key contiguous.  Besides avoiding one I/O command
        # per sparse block, this lets us sector-align the physical read while
        # retaining one simple payload offset for the scatter callback.
        # ``bytes_per_block`` is 8448 for the compact DSV4 indexer, so an odd
        # run start is only 256B-aligned even though the layer object itself
        # starts on a 4KiB boundary.
        segment_bounds: List[Tuple[int, int]] = []
        segment_start = 0
        selected_ids = selected.tolist()
        for index in range(1, len(selected_ids) + 1):
            at_end = index == len(selected_ids)
            run_broken = not at_end and int(selected_ids[index]) != (
                int(selected_ids[index - 1]) + 1
            )
            size_limit = index - segment_start == 256
            if at_end or run_broken or size_limit:
                segment_bounds.append((segment_start, index))
                segment_start = index
        segments = [selected[start:end] for start, end in segment_bounds]
        scatter_stream = self._scatter_stream_for(state.k_cache_tensor.device)
        if dst_rows_table.device.type == "cuda":
            # Build destination rows on the same private stream that consumes
            # them.  Creating these tensors on the caller's current stream and
            # immediately launching the fused scatter on ``scatter_stream``
            # leaves no CUDA dependency between the producer and consumer.
            # The fastest rank can then dereference an unfinished index tensor;
            # the illegal access is reported later by the next I/O-stream sync.
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                segment_rows = [
                    dst_rows_table.index_select(
                        0,
                        segment.to(
                            device=dst_rows_table.device,
                            non_blocking=True,
                        ),
                    )
                    for segment in segments
                ]
            scatter_stream.synchronize()
        else:
            segment_rows = [
                dst_rows_table.index_select(0, segment) for segment in segments
            ]

        def _read_plan_for_segment(
            segment: torch.Tensor,
        ) -> Tuple[Tuple[KVObjectByteRange, ...], int]:
            """Return a direct-I/O range and payload skip for one block run."""
            payload_offset = int(chunk.layer_byte_offset) + (
                int(segment[0]) * block_nbytes
            )
            payload_length = int(segment.numel()) * block_nbytes
            aligned_offset = payload_offset - payload_offset % 512
            aligned_end = ((payload_offset + payload_length + 511) // 512) * 512
            payload_skip = payload_offset - aligned_offset
            return (
                (
                    KVObjectByteRange(
                        offset=aligned_offset,
                        length=aligned_end - aligned_offset,
                        target_offset=0,
                    ),
                ),
                payload_skip,
            )

        segment_read_plans = [_read_plan_for_segment(segment) for segment in segments]

        def _restore_lba_cache() -> None:
            if self._pending_raw_lba_cache:
                self._tutti_loader.ensure_lba_cache(self._pending_raw_lba_cache)

        if state.block_slot_scatter:
            k_cache = state.k_cache_tensor.view(torch.uint8)
        else:
            k_cache = state.k_cache_tensor.view(torch.uint8).reshape(
                int(state.k_cache_tensor.shape[0]),
                -1,
            )
        completed_block_ids: List[int] = []

        def _scatter_tensor_batch(
            batch_start: int,
            batch_results: List[Optional[Any]],
        ) -> None:
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                for offset_in_batch, memory_obj in enumerate(batch_results):
                    if memory_obj is None:
                        raise RuntimeError(
                            "layer-major Tutti segment returned no payload"
                        )
                    key_index = batch_start + offset_in_batch
                    segment = segments[key_index]
                    segment_blocks = int(segment.numel())
                    segment_nbytes = segment_blocks * block_nbytes
                    payload_skip = segment_read_plans[key_index][1]
                    tensor = memory_obj.raw_tensor
                    if tensor is None:
                        raise RuntimeError(
                            "layer-major Tutti segment has no raw tensor"
                        )
                    flat = tensor.view(torch.uint8).reshape(-1)
                    if int(flat.numel()) < payload_skip + segment_nbytes:
                        raise RuntimeError(
                            "layer-major Tutti segment returned a short payload"
                        )
                    source = flat[payload_skip : payload_skip + segment_nbytes].view(
                        segment_blocks,
                        block_nbytes,
                    )
                    rows = segment_rows[key_index]
                    if state.block_slot_scatter:
                        slot_size = state.block_slot_size
                        block_ids = torch.div(
                            rows,
                            slot_size,
                            rounding_mode="floor",
                        )
                        slot_ids = rows - block_ids * slot_size
                        k_cache.index_put_((block_ids, slot_ids), source)
                    else:
                        k_cache.index_copy_(0, rows, source)
                    completed_block_ids.extend(int(value) for value in segment.tolist())
            scatter_stream.synchronize()

        def _scatter_raw_batch(
            batch_start: int,
            completed_indices: List[int],
            completed_offsets: List[int],
            completed_nbytes: List[int],
            staging: torch.Tensor,
        ) -> None:
            if not completed_indices:
                raise RuntimeError("layer-major raw Tutti read did not complete")
            if _csa_c_ops is None:
                raise RuntimeError("layer-major raw scatter requires lmcache.c_ops")
            scatter = getattr(
                _csa_c_ops,
                "scatter_rows_from_object_ptrs",
                None,
            )
            if scatter is None:
                raise RuntimeError("layer-major fused scatter op is unavailable")
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                for local_index, offset, nbytes in zip(
                    completed_indices,
                    completed_offsets,
                    completed_nbytes,
                    strict=True,
                ):
                    key_index = batch_start + local_index
                    segment = segments[key_index]
                    segment_blocks = int(segment.numel())
                    segment_nbytes = segment_blocks * block_nbytes
                    payload_skip = segment_read_plans[key_index][1]
                    if int(nbytes) < payload_skip + segment_nbytes:
                        raise RuntimeError(
                            "layer-major raw Tutti segment returned a short payload"
                        )
                    source_ptr = int(staging.data_ptr()) + int(offset) + payload_skip
                    source_ptrs = torch.tensor(
                        [source_ptr],
                        dtype=torch.int64,
                        device=state.k_cache_tensor.device,
                    )
                    scatter(
                        source_ptrs,
                        k_cache,
                        segment_rows[key_index],
                        segment_blocks,
                        block_nbytes,
                        state.block_slot_size if state.block_slot_scatter else 0,
                        source_ptr % 8 == 0,
                    )
                    completed_block_ids.extend(int(value) for value in segment.tolist())
            scatter_stream.synchronize()

        load_kwargs: Dict[str, Any] = {
            "shapes_per_key": None,
            "file_offsets": [0] * len(segments),
            "read_ranges_per_key": [read_plan[0] for read_plan in segment_read_plans],
            "io_priority": io_priority,
            "before_batch": _restore_lba_cache,
            "max_batch_ios": 256,
            "max_batch_bytes": 128 * 1024**2,
            "wait_for_active_io": io_priority == "lookahead",
            "profile_layer_id": getattr(state, "layer_id", -1),
            "profile_kind": profile_kind,
            "profile_operation_id": profile_operation_id,
            "profile_source_layer": profile_source_layer,
        }
        if _csa_c_ops is not None and hasattr(
            _csa_c_ops, "scatter_rows_from_object_ptrs"
        ):
            load_kwargs["on_raw_batch_loaded"] = _scatter_raw_batch
        else:
            load_kwargs["on_batch_loaded"] = _scatter_tensor_batch
        self._tutti_loader.load_chunks_to_hbm(
            [chunk.key] * len(segments),
            [chunk.disk_meta] * len(segments),
            **load_kwargs,
        )
        completed_ids = torch.as_tensor(
            sorted(set(completed_block_ids)),
            dtype=torch.int64,
        )
        if completed_ids.numel():
            with torch.inference_mode(), torch.cuda.stream(scatter_stream):
                bitmap_ids = completed_ids.to(
                    device=state.in_pool_bitmap.device,
                    non_blocking=True,
                )
                state.in_pool_bitmap.index_fill_(0, bitmap_ids, True)
            scatter_stream.synchronize()
        return None, [], completed_ids

    def _issue_indexed_reads(
        self,
        state: CSAAttentionKVLayerState,
        sorted_block_ids: Sequence[int] | torch.Tensor,
        *,
        io_priority: str,
        profile_kind: Optional[str] = None,
        profile_source_layer: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
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
            with csa_pipeline_nvtx.range(
                CsaNvtxEvent.IO_SCATTER,
                layer_id=(
                    int(profile_source_layer)
                    if profile_source_layer is not None
                    else state.layer_id
                ),
                target_layer_id=state.layer_id,
                request_id=self.active_request_id,
                attributes={
                    "kind": profile_kind or "unknown",
                    "blocks": n_selected,
                },
            ):
                with detailed_io_nvtx.range(
                    CsaNvtxEvent.IO_MATERIALIZE,
                    layer_id=(
                        int(profile_source_layer)
                        if profile_source_layer is not None
                        else state.layer_id
                    ),
                    target_layer_id=state.layer_id,
                    request_id=self.active_request_id,
                    operation_id=profile_operation_id,
                    attributes={
                        "kind": profile_kind or "unknown",
                        "blocks": n_selected,
                    },
                ):
                    with detailed_io_nvtx.range(
                        CsaNvtxEvent.IO_SCATTER,
                        layer_id=(
                            int(profile_source_layer)
                            if profile_source_layer is not None
                            else state.layer_id
                        ),
                        target_layer_id=state.layer_id,
                        request_id=self.active_request_id,
                        operation_id=profile_operation_id,
                        attributes={
                            "kind": profile_kind or "unknown",
                            "blocks": n_selected,
                        },
                    ):
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
            profile_layer_id=getattr(state, "layer_id", -1),
            profile_kind=profile_kind,
            profile_operation_id=profile_operation_id,
            profile_source_layer=profile_source_layer,
            input_ready_event=input_ready_event,
            wait_for_active_io=io_priority == "lookahead",
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
        # Never let an empty or malformed later call reuse a prior query's
        # compact-gather coverage proof.
        state.true_selected_blocks_bitmap = None
        state.true_selected_covers_cached_prefix = False
        entries = true_topk.reshape(-1)
        if entries.numel() == 0:
            return torch.empty(0, dtype=torch.int64)
        device = state.in_pool_bitmap.device
        if entries.device != device:
            entries = entries.to(device)
        # Clip to the chunk-map's registered range (sentinel padding past the
        # cached prefix) and to the bitmap capacity (mirrors vLLM's K cache
        # num_blocks) in a single validity mask; sentinel/negative entries
        # are dropped by the same mask.
        max_block_id = state.chunks[-1].end_compressed_block
        bitmap_len = int(state.in_pool_bitmap.shape[0])
        limit = min(int(max_block_id), bitmap_len)
        if limit <= 0:
            return torch.empty(0, dtype=torch.int64)
        active_rows = int(true_topk.shape[0]) if true_topk.ndim >= 2 else 1
        streamed_union = self._streamed_true_topk_union(state, active_rows)
        if streamed_union is not None:
            selected_prefix = streamed_union[:limit].ne(0)
            state.true_selected_blocks_bitmap = streamed_union
            state.true_selected_covers_cached_prefix = bool(
                selected_prefix.all().item()
            )
            return (
                (selected_prefix & ~state.in_pool_bitmap[:limit])
                .nonzero(as_tuple=False)
                .reshape(-1)
                .cpu()
            )
        native_select = getattr(
            _csa_c_ops,
            "select_missing_csa_blocks",
            None,
        )
        native_select_with_seen = getattr(
            _csa_c_ops,
            "select_missing_csa_blocks_with_seen",
            None,
        )
        if (
            (callable(native_select_with_seen) or callable(native_select))
            and entries.is_cuda
            and state.in_pool_bitmap.is_cuda
        ):
            # Prefill commonly carries 8192 x 512 = 4.19M selected entries.
            # A single native atomic marker scans them without materialising
            # the 4M-entry int64 block-id and validity tensors created by the
            # generic PyTorch path. Only the <=1.9K compact miss ids cross to
            # the CPU for Tutti submission.
            if callable(native_select_with_seen):
                suffix_blocks = (
                    active_rows
                    + state.compressed_block_size * _DSV4_CSA_COMPRESS_RATIO
                    - 1
                ) // (state.compressed_block_size * _DSV4_CSA_COMPRESS_RATIO)
                selected_limit = min(bitmap_len, limit + suffix_blocks)
                outputs = native_select_with_seen(
                    entries.contiguous(),
                    state.in_pool_bitmap,
                    limit,
                    selected_limit,
                    state.compressed_block_size,
                )
                state.true_selected_blocks_bitmap = outputs[1]
                missing_cpu = outputs[0].cpu()
                state.true_selected_covers_cached_prefix = bool(
                    outputs[1][:limit].all().item()
                )
                return missing_cpu
            assert callable(native_select)
            return native_select(
                entries.contiguous(),
                state.in_pool_bitmap,
                limit,
                state.compressed_block_size,
            ).cpu()
        entries = entries.to(torch.int64)
        block_ids = entries // state.compressed_block_size
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

    def _shard_capability_for(
        self,
        state: CSAAttentionKVLayerState,
        block_ids: Sequence[int] | torch.Tensor,
        *,
        mode: SSDReadMode,
    ) -> bool:
        """Return this rank's capability/cost vote for a proposed mode."""
        transport = self._shard_transport
        if (
            self._data_group != "csa"
            or transport is None
            or not self._shard_config.enabled
            or not self._shard_config.csa_enabled
        ):
            return False
        selected = torch.as_tensor(
            block_ids,
            dtype=torch.int64,
            device="cpu",
        ).reshape(-1)
        selected = torch.unique(selected, sorted=True)
        if selected.numel() == 0:
            return False
        covered_end = int(state.chunks[-1].end_compressed_block) if state.chunks else 0
        complete_layer = bool(
            int(selected.numel()) == covered_end
            and int(selected[0]) == 0
            and int(selected[-1]) == covered_end - 1
        )
        mode_matches_request = bool(
            mode == SSDReadMode.SHARD_GATHER_PREDICTED
            or (
                mode == SSDReadMode.SHARD_GATHER_DENSE
                and complete_layer
                and state.layer_id in self._shard_config.dense_layers
            )
        )
        logical_rows = self._logical_destination_rows(state)
        capability_ok = bool(
            mode_matches_request
            and self._shard_config.csa_replica_verified
            and transport.healthy
            and transport.world_size == self._shard_config.cp_size
            and logical_rows is not None
            and not state.block_slot_scatter
            and int(selected[-1]) < int(state.in_pool_bitmap.numel())
            and int(selected[-1]) < int(logical_rows.numel())
        )
        context_tokens = (
            covered_end * state.compressed_block_size * _DSV4_CSA_COMPRESS_RATIO
        )
        key = bucket_prefetch_key(
            group="csa",
            layer_id=state.layer_id,
            context_tokens=context_tokens,
            query_tokens=0,
            union_blocks=int(selected.numel()),
        )
        return bool(
            self._shard_decisions.choose(
                key,
                union_blocks=int(selected.numel()),
                block_bytes=self._bytes_per_block,
                world_size=transport.world_size,
                shard_mode=mode,
                capability_ok=capability_ok,
            )
            == mode
        )

    def _prepare_predicted_shard_gather(
        self,
        state: CSAAttentionKVLayerState,
        block_ids: Sequence[int] | torch.Tensor,
        *,
        label: str,
        request_token: Optional[Tuple[str, int]],
        profile_source_layer: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
        profile_kind: Optional[str] = None,
    ) -> bool | _DeferredShardGather:
        """Book predicted blocks and read only this rank's owner shard."""
        selected = torch.as_tensor(
            block_ids,
            dtype=torch.int64,
            device="cpu",
        ).reshape(-1)
        limit = min(
            int(state.chunks[-1].end_compressed_block),
            int(state.in_pool_bitmap.numel()),
        )
        selected = torch.unique(
            selected[(selected >= 0) & (selected < limit)],
            sorted=True,
        )
        if selected.numel() == 0:
            return True
        with state.pending_reads_lock:
            pending_ids = selected[
                ~state.resident_blocks_bitmap[selected]
                & ~state.pending_reads_bitmap[selected]
            ]
            with torch.inference_mode():
                state.pending_reads_bitmap[pending_ids] = True
            state.pending_read_count += int(pending_ids.numel())
        operation_id = profile_operation_id or f"{label}-{time.monotonic_ns()}"
        source_layer = (
            int(profile_source_layer)
            if profile_source_layer is not None
            else state.layer_id
        )
        if profile_source_layer is None and label.startswith("predicted_l"):
            try:
                source_layer -= int(label.rsplit("l", 1)[1])
            except ValueError:
                pass
        effective_profile_kind = profile_kind or _io_profile_kind(label)
        io_range = csa_pipeline_nvtx.start_io(
            layer_id=source_layer,
            target_layer_id=state.layer_id,
            operation_id=operation_id,
            request_id=self.active_request_id,
            attributes={
                "kind": effective_profile_kind,
                "detail": label,
                "blocks": int(pending_ids.numel()),
            },
        )
        try:
            work = self._prepare_shard_gather(
                state,
                selected,
                mode=SSDReadMode.SHARD_GATHER_PREDICTED,
                io_priority="lookahead",
                request_token=request_token,
                profile_operation_id=operation_id,
                profile_kind=effective_profile_kind,
                profile_source_layer=source_layer,
            )
        except Exception:
            with state.pending_reads_lock:
                with torch.inference_mode():
                    state.pending_reads_bitmap[pending_ids] = False
                state.pending_read_count = max(
                    0,
                    state.pending_read_count - int(pending_ids.numel()),
                )
                state.pending_reads_lock.notify_all()
            csa_pipeline_nvtx.finish_io(
                io_range,
                layer_id=state.layer_id,
                target_layer_id=state.layer_id,
                operation_id=operation_id,
                request_id=self.active_request_id,
                status="error",
            )
            raise
        work.pending_ids = pending_ids
        work.io_range = io_range
        work.operation_id = operation_id
        return work

    def _prepare_shard_gather(
        self,
        state: CSAAttentionKVLayerState,
        block_ids: Sequence[int] | torch.Tensor,
        *,
        mode: SSDReadMode,
        io_priority: str,
        request_token: Optional[Tuple[str, int]],
        profile_operation_id: Optional[str] = None,
        profile_kind: Optional[str] = None,
        profile_source_layer: Optional[int] = None,
    ) -> _DeferredShardGather:
        """Read this rank's owner shard without launching a collective."""
        with self._request_state:
            current_token = (
                self._active_request_id or "",
                self._request_generation,
            )
            if self._request_lifecycle != "active" or (
                request_token is not None and request_token != current_token
            ):
                raise RuntimeError("request shard read plan is inactive or stale")
            self._active_submissions += 1
        try:
            transport = self._shard_transport
            if transport is None:
                raise RuntimeError("shard transport is unavailable")
            selected = torch.as_tensor(
                block_ids,
                dtype=torch.int64,
                device="cpu",
            ).reshape(-1)
            selected = torch.unique(selected, sorted=True)
            partition = partition_block_union(selected.tolist(), transport.world_size)
            descriptor = CollectiveDescriptor(
                request_generation=int(self._request_generation),
                layer_id=state.layer_id,
                phase=2 if mode == SSDReadMode.SHARD_GATHER_DENSE else 1,
                mode=mode,
                partition=partition,
            )
            local_capability = self._shard_capability_for(
                state,
                selected,
                mode=mode,
            )
            owned = torch.as_tensor(
                partition.blocks_for_rank(transport.rank),
                dtype=torch.int64,
                device="cpu",
            )
            local_event: Optional[torch.cuda.Event] = None
            local_objects: List[Any] = []
            local_completed = torch.empty(0, dtype=torch.int64)
            local_error: Optional[Exception] = None
            detailed_operation_id = profile_operation_id or (
                f"shard-{mode.value}-{time.monotonic_ns()}"
            )
            detailed_kind = profile_kind or (
                "csa_dense"
                if mode == SSDReadMode.SHARD_GATHER_DENSE
                else "csa_predicted"
            )
            detailed_source_layer = (
                int(profile_source_layer)
                if profile_source_layer is not None
                else state.layer_id
            )
            if local_capability:
                try:
                    with detailed_io_nvtx.range(
                        CsaNvtxEvent.IO_LOADER_CALL,
                        layer_id=detailed_source_layer,
                        target_layer_id=state.layer_id,
                        operation_id=detailed_operation_id,
                        request_id=self.active_request_id,
                        attributes={
                            "kind": detailed_kind,
                            "detail": mode.value,
                            "blocks": int(owned.numel()),
                        },
                    ):
                        local_event, local_objects, local_completed = (
                            self._issue_local_reads(
                                state,
                                owned,
                                io_priority=io_priority,
                                profile_kind=detailed_kind,
                                profile_source_layer=detailed_source_layer,
                                profile_operation_id=detailed_operation_id,
                            )
                        )
                except Exception as exc:
                    local_error = exc
            local_complete = bool(
                local_error is None
                and torch.equal(
                    torch.unique(local_completed.cpu(), sorted=True),
                    owned,
                )
            )
            if local_complete and owned.numel():
                with state.pending_reads_lock, torch.inference_mode():
                    state.resident_blocks_bitmap[owned] = True
            return _DeferredShardGather(
                state=state,
                descriptor=descriptor,
                selected=selected,
                owned=owned,
                local_ready_event=local_event,
                local_objects=local_objects,
                local_complete=local_complete,
                local_capability=local_capability,
                operation_id=detailed_operation_id,
                profile_kind=detailed_kind,
                profile_source_layer=detailed_source_layer,
                request_id=self.active_request_id,
            )
        finally:
            with self._request_state:
                self._active_submissions -= 1
                self._request_state.notify_all()

    def _finalize_shard_gather(
        self,
        prepared: _DeferredShardGather,
    ) -> None:
        """Run consensus/gather after all model ranks reach the layer gate."""
        if not self._claim_deferred_shard_gather(prepared):
            return
        state = prepared.state
        transport = self._shard_transport
        descriptor = prepared.descriptor
        device = state.k_cache_tensor.device
        retained_objects = list(prepared.local_objects)
        completed = False
        try:
            if transport is None:
                raise RuntimeError("shard transport disappeared before its gate")
            with self._shard_collective_lock:
                agreed = transport.preflight(
                    descriptor,
                    local_capability=prepared.local_capability,
                    device=device,
                )
                ready_descriptor = CollectiveDescriptor(
                    request_generation=descriptor.request_generation,
                    layer_id=descriptor.layer_id,
                    phase=descriptor.phase + 128,
                    mode=descriptor.mode,
                    partition=descriptor.partition,
                )
                all_reads_ready = agreed and transport.preflight(
                    ready_descriptor,
                    local_capability=prepared.local_complete,
                    device=device,
                )
                if not all_reads_ready:
                    missing = prepared.selected[
                        ~state.resident_blocks_bitmap[prepared.selected]
                    ]
                    fallback_operation_id = (
                        f"{prepared.operation_id or 'shard'}-fallback"
                    )
                    fallback_source = (
                        prepared.profile_source_layer
                        if prepared.profile_source_layer is not None
                        else state.layer_id
                    )
                    with detailed_io_nvtx.range(
                        CsaNvtxEvent.IO_LOADER_CALL,
                        layer_id=int(fallback_source),
                        target_layer_id=state.layer_id,
                        operation_id=fallback_operation_id,
                        request_id=self.active_request_id,
                        attributes={
                            "kind": prepared.profile_kind,
                            "detail": "shard_fallback",
                            "blocks": int(missing.numel()),
                        },
                    ):
                        event, objects, completed = self._issue_local_reads(
                            state,
                            missing,
                            io_priority="demand",
                            profile_kind=prepared.profile_kind,
                            profile_source_layer=int(fallback_source),
                            profile_operation_id=fallback_operation_id,
                        )
                    retained_objects.extend(objects)
                    if event is not None:
                        event.synchronize()
                    if not torch.equal(
                        torch.unique(completed.cpu(), sorted=True),
                        torch.unique(missing.cpu(), sorted=True),
                    ):
                        raise RuntimeError(
                            f"shard fallback is incomplete for layer {state.layer_id}"
                        )
                    with state.pending_reads_lock, torch.inference_mode():
                        state.resident_blocks_bitmap[prepared.selected] = True
                    completed = True
                    return
                logical_rows = self._logical_destination_rows(state)
                if logical_rows is None or state.block_slot_scatter:
                    raise RuntimeError("shard destination layout is unavailable")
                k_cache_rows = state.k_cache_tensor.view(torch.uint8).reshape(
                    int(state.k_cache_tensor.shape[0]),
                    -1,
                )
                gather_event = transport.gather_into(
                    descriptor,
                    source_rows=k_cache_rows,
                    logical_destination_rows=logical_rows,
                    destination_rows=k_cache_rows,
                    local_ready_event=prepared.local_ready_event,
                    resident_bitmap=state.in_pool_bitmap,
                )
                gather_event.synchronize()
                with state.pending_reads_lock, torch.inference_mode():
                    state.resident_blocks_bitmap[prepared.selected] = True
                completed = True
                partition = descriptor.partition
                logger.info(
                    "CSA_SHARD_GATHER layer=%d mode=%s union_blocks=%d "
                    "owned_blocks=%d ssd_bytes=%d gather_send_bytes=%d "
                    "gather_total_bytes=%d union_hash=%d sequence=%d "
                    "gate_aligned=1",
                    state.layer_id,
                    descriptor.mode.value,
                    len(partition.union),
                    len(partition.blocks_for_rank(transport.rank)),
                    len(partition.blocks_for_rank(transport.rank))
                    * self._bytes_per_block,
                    partition.padded_blocks * self._bytes_per_block,
                    partition.padded_blocks
                    * partition.world_size
                    * self._bytes_per_block,
                    partition.union_hash,
                    descriptor.sequence_number,
                )
        finally:
            prepared.local_objects = retained_objects
            self._release_deferred_shard_gather(
                prepared,
                status="complete" if completed else "error",
            )

    @staticmethod
    def _claim_deferred_shard_gather(prepared: _DeferredShardGather) -> bool:
        """Claim exactly one finalize-or-discard path for deferred work."""
        with prepared.lifecycle_lock:
            if prepared.lifecycle_claimed:
                return False
            prepared.lifecycle_claimed = True
            return True

    def _release_deferred_shard_gather(
        self,
        prepared: _DeferredShardGather,
        *,
        status: str,
    ) -> None:
        """Retire local CUDA work, pending bits, references, and profile state."""
        state = prepared.state
        event_error: Optional[BaseException] = None
        if prepared.local_ready_event is not None:
            try:
                prepared.local_ready_event.synchronize()
            except BaseException as exc:
                event_error = exc
            finally:
                prepared.local_ready_event = None
        memory_objects = prepared.local_objects
        prepared.local_objects = []
        for memory_obj in memory_objects:
            ref_count_down = getattr(memory_obj, "ref_count_down", None)
            if callable(ref_count_down):
                try:
                    ref_count_down()
                except Exception:
                    logger.exception(
                        "CSAAttentionKVPrefetchManager: deferred staging "
                        "release failed layer=%d",
                        state.layer_id,
                    )
        pending_ids = prepared.pending_ids
        prepared.pending_ids = None
        if pending_ids is not None:
            with state.pending_reads_lock:
                with torch.inference_mode():
                    state.pending_reads_bitmap[pending_ids] = False
                state.pending_read_count = max(
                    0,
                    state.pending_read_count - int(pending_ids.numel()),
                )
                state.pending_reads_lock.notify_all()
        io_range = prepared.io_range
        prepared.io_range = None
        if io_range is not None:
            csa_pipeline_nvtx.finish_io(
                io_range,
                layer_id=state.layer_id,
                target_layer_id=state.layer_id,
                operation_id=prepared.operation_id,
                request_id=prepared.request_id,
                status=status,
            )
        if event_error is not None:
            raise event_error

    def _issue_shard_gather(
        self,
        state: CSAAttentionKVLayerState,
        block_ids: Sequence[int] | torch.Tensor,
        *,
        mode: SSDReadMode,
        io_priority: str,
        local_fallback_ids: Sequence[int] | torch.Tensor,
        local_capability: bool,
        profile_kind: Optional[str] = None,
        profile_source_layer: Optional[int] = None,
        profile_operation_id: Optional[str] = None,
    ) -> Tuple[Optional[torch.cuda.Event], List[Any], torch.Tensor]:
        """Read this rank's owner slice and gather the complete union."""
        transport = self._shard_transport
        logical_rows = self._logical_destination_rows(state)
        if transport is None:
            return self._issue_local_reads(
                state,
                local_fallback_ids,
                io_priority=io_priority,
                profile_kind=profile_kind,
                profile_source_layer=profile_source_layer,
                profile_operation_id=profile_operation_id,
            )
        selected = torch.as_tensor(
            block_ids,
            dtype=torch.int64,
            device="cpu",
        ).reshape(-1)
        selected = torch.unique(selected, sorted=True)
        partition = partition_block_union(selected.tolist(), transport.world_size)
        phase = 2 if mode == SSDReadMode.SHARD_GATHER_DENSE else 1
        descriptor = CollectiveDescriptor(
            request_generation=int(self._request_generation),
            layer_id=state.layer_id,
            phase=phase,
            mode=mode,
            partition=partition,
        )
        device = state.k_cache_tensor.device
        try:
            agreed = transport.preflight(
                descriptor,
                local_capability=local_capability,
                device=device,
            )
        except ShardCollectiveError:
            logger.exception(
                "CSAAttentionKVPrefetchManager: preflight failed layer=%d; "
                "using LOCAL_DIRECT",
                state.layer_id,
            )
            return self._issue_local_reads(
                state,
                local_fallback_ids,
                io_priority=io_priority,
                profile_kind=profile_kind,
                profile_source_layer=profile_source_layer,
                profile_operation_id=profile_operation_id,
            )
        if not agreed:
            logger.warning(
                "CSAAttentionKVPrefetchManager: shard consensus rejected "
                "layer=%d mode=%s union_blocks=%d; using LOCAL_DIRECT",
                state.layer_id,
                mode.value,
                int(selected.numel()),
            )
            return self._issue_local_reads(
                state,
                local_fallback_ids,
                io_priority=io_priority,
                profile_kind=profile_kind,
                profile_source_layer=profile_source_layer,
                profile_operation_id=profile_operation_id,
            )
        if logical_rows is None:
            raise RuntimeError(
                "shard-gather consensus accepted without a destination-row table"
            )

        owned = torch.as_tensor(
            partition.blocks_for_rank(transport.rank),
            dtype=torch.int64,
            device="cpu",
        )
        local_event: Optional[torch.cuda.Event] = None
        local_objects: List[Any] = []
        local_completed = torch.empty(0, dtype=torch.int64)
        local_error: Optional[Exception] = None
        try:
            local_event, local_objects, local_completed = self._issue_local_reads(
                state,
                owned,
                io_priority=io_priority,
                profile_kind=profile_kind,
                profile_source_layer=profile_source_layer,
                profile_operation_id=profile_operation_id,
            )
        except Exception as exc:
            local_error = exc
        local_complete = bool(
            local_error is None
            and torch.equal(
                torch.unique(local_completed.cpu(), sorted=True),
                owned,
            )
        )
        ready_descriptor = CollectiveDescriptor(
            request_generation=descriptor.request_generation,
            layer_id=descriptor.layer_id,
            phase=descriptor.phase + 128,
            mode=descriptor.mode,
            partition=descriptor.partition,
        )
        try:
            all_reads_ready = transport.preflight(
                ready_descriptor,
                local_capability=local_complete,
                device=device,
            )
        except ShardCollectiveError:
            all_reads_ready = False
        if not all_reads_ready:
            logger.warning(
                "CSAAttentionKVPrefetchManager: owner read readiness rejected "
                "layer=%d owned=%d; using LOCAL_DIRECT",
                state.layer_id,
                int(owned.numel()),
            )
            fallback_event, fallback_objects, completed = self._issue_local_reads(
                state,
                local_fallback_ids,
                io_priority=io_priority,
                profile_kind=profile_kind,
                profile_source_layer=profile_source_layer,
                profile_operation_id=(
                    f"{profile_operation_id}-fallback"
                    if profile_operation_id is not None
                    else None
                ),
            )
            return (
                fallback_event or local_event,
                [*local_objects, *fallback_objects],
                completed,
            )

        if state.block_slot_scatter:
            raise RuntimeError("CSA shard-gather does not support block-slot scatter")
        k_cache_rows = state.k_cache_tensor.view(torch.uint8).reshape(
            int(state.k_cache_tensor.shape[0]),
            -1,
        )
        gather_event = transport.gather_into(
            descriptor,
            source_rows=k_cache_rows,
            logical_destination_rows=logical_rows,
            destination_rows=k_cache_rows,
            local_ready_event=local_event,
            resident_bitmap=state.in_pool_bitmap,
        )
        logger.info(
            "CSA_SHARD_GATHER layer=%d mode=%s union_blocks=%d "
            "owned_blocks=%d ssd_bytes=%d gather_send_bytes=%d "
            "gather_total_bytes=%d union_hash=%d sequence=%d",
            state.layer_id,
            mode.value,
            len(partition.union),
            len(partition.blocks_for_rank(transport.rank)),
            len(partition.blocks_for_rank(transport.rank)) * self._bytes_per_block,
            partition.padded_blocks * self._bytes_per_block,
            partition.padded_blocks * partition.world_size * self._bytes_per_block,
            partition.union_hash,
            descriptor.sequence_number,
        )
        return gather_event, local_objects, selected

    @staticmethod
    def _logical_destination_rows(
        state: CSAAttentionKVLayerState,
    ) -> Optional[torch.Tensor]:
        """Return the logical-block to physical-row table for a layer."""
        if state.layer_major_dst_rows_table is not None:
            return state.layer_major_dst_rows_table
        return state.indexed_dst_rows_table

    def _indexer_cp_owned_blocks(
        self,
        layer_id: int,
        covered_blocks: int,
    ) -> Sequence[int]:
        """Compile native Indexer-K CP ownership or return the full fallback."""
        config = self._shard_config
        if (
            not config.enabled
            or not config.indexer_enabled
            or not config.indexer_cp_verified
            or layer_id in config.disabled_layers
        ):
            return range(covered_blocks)
        try:
            if self._cp_world_size != config.cp_size:
                raise ValueError(
                    f"configured_cp={config.cp_size},runtime_cp={self._cp_world_size}"
                )
            plan = compile_cp_read_plan(
                total_rows=covered_blocks * self._compressed_block_size,
                rank=self._cp_rank,
                world_size=self._cp_world_size,
                interleave_rows=config.cp_interleave,
                block_rows=self._compressed_block_size,
                row_bytes=self._token_bytes,
            )
        except ValueError as exc:
            reason = str(exc)
            key = (int(layer_id), reason)
            if key not in self._cp_fallback_reasons:
                self._cp_fallback_reasons.add(key)
                logger.warning(
                    "INDEXER_CP_LOCAL_FALLBACK layer=%d reason=%s planned_mode=%s",
                    layer_id,
                    reason,
                    SSDReadMode.LOCAL_DIRECT.value,
                )
            return range(covered_blocks)
        logger.info(
            "INDEXER_CP_LOCAL_READ layer=%d rank=%d world_size=%d "
            "planned_rows=%d row_ranges=%d owned_blocks=%d ssd_bytes=%d "
            "kernel_row_min=%d kernel_row_max=%d mode=%s",
            layer_id,
            plan.rank,
            plan.world_size,
            plan.planned_rows,
            len(plan.row_ranges),
            len(plan.block_ids),
            plan.ssd_bytes,
            plan.row_ranges[0].start if plan.row_ranges else -1,
            plan.row_ranges[-1].end if plan.row_ranges else -1,
            SSDReadMode.CP_LOCAL_INDEXER.value,
        )
        return plan.block_ids
