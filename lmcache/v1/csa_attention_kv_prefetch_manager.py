# SPDX-License-Identifier: Apache-2.0
"""On-demand prefetcher for DSv4 CSA attention KV (~100 MiB / 24K prefix).

The DeepSeek V4 CSA layers use ``compress_ratio == 4`` MLA-style attention
KV.  At baseline, LMCache scatters the entire compressed KV (``584 B`` per
compressed entry, ``S/4`` entries per layer, 30 layers ≈ 100 MiB for a
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

This manager is wired in via ``LMCACHE_INDEXER_FULL_OVERLAP=1`` together
with the Indexer SSD manager.  When disabled, the legacy retrieve scatter
path is preserved.
"""
from __future__ import annotations

# Standard
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

# First Party
import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.kv_object_store import KVObjectByteRange
from lmcache.v1.memory_management import MemoryFormat

if TYPE_CHECKING:
    from lmcache.v1.gpu_connector.tutti_direct_loader import TuttiDirectLoader

logger = init_logger(__name__)

_ACTIVE_MANAGER: Optional["CSAAttentionKVPrefetchManager"] = None
_ACTIVE_MANAGER_LOCK = threading.Lock()


def _timing_enabled() -> bool:
    """Return True when CSA attention KV prefetch timing logs are enabled."""
    value = os.environ.get("LMCACHE_CSA_ATTENTION_KV_TIMING", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _bulk_predicted_enabled() -> bool:
    """Return True when predicted fires collapse into one bulk read.

    At long contexts the HC-proxy's block-level prediction covers the whole
    prefix (union of top-K over tens of thousands of query rows saturates
    every block), so per-layer predicted reads degenerate into each of the
    ~21 CSA layers re-reading the full prefix: ~11x NVMe read amplification
    measured at 384K.  Bulk mode reads every layer's slab once, in a single
    Tutti pass, on the first fire of the request.
    """
    value = os.environ.get("LMCACHE_CSA_BULK_PREDICTED", "1")
    return value.lower() in {"1", "true", "yes", "on"}


def _walker_resident_skip_enabled() -> bool:
    """Return True when the walker may skip chunks already resident in HBM.

    When the previous request's walk landed a chunk's csa_attention_kv slab
    into the SAME physical K-cache rows (same chunk key, same
    ``physical_block_ids``), those rows still hold the bytes: the retrieve
    path never writes the csa_attention_kv group while the prefetcher is
    attached (LMCACHE_DSV4_CSA_ATTENTION_KV_FILTER), decode only appends to
    newly allocated rows, and only the immediately previous request's
    signatures are kept (any interposed registration replaces them).  Under
    a serial workload the signature match is therefore content-exact and
    the chunk's 1.5 GB read + SM-contending scatter can be skipped.

    Known limitation: a request that runs WITHOUT registering (pure cold
    miss) can dirty freed rows invisibly; its chunk keys differ, so a
    same-key+same-rows match after it requires the allocator to hand the
    same rows back to the same content — benign for serial benchmarks,
    but concurrent production workloads should flip this off until row
    epochs are wired through the vLLM block manager.
    """
    value = os.environ.get("LMCACHE_CSA_WALKER_RESIDENT_SKIP", "1")
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
    """

    first_compressed_block: int
    n_compressed_blocks: int
    key: CacheEngineKey
    disk_meta: DiskCacheMetadata
    layer_byte_offset: int
    bytes_per_block: int
    raw_extents: tuple[tuple[int, int, int], ...] = ()
    physical_block_ids: tuple[int, ...] = ()

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

        Raises:
            ValueError: If ``compressed_block_id`` is outside this chunk.
        """
        if not self.contains(compressed_block_id):
            raise ValueError(
                f"compressed_block_id {compressed_block_id} outside chunk range "
                f"[{self.first_compressed_block}, {self.end_compressed_block})"
            )
        local = compressed_block_id - self.first_compressed_block
        return self.layer_byte_offset + local * self.bytes_per_block


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
        pending_reads_lock: Guards mutation of ``pending_reads`` /
            ``last_drain_event`` / ``pending_drains``.
        pending_reads: Set of compressed block ids whose Tutti read is
            currently in-flight or queued.
        last_drain_event: Optional CUDA event recording the completion of
            the latest read submission.  Sparse attention waits on this
            event during :meth:`CSAAttentionKVPrefetchManager.drain_for_layer`.
        pending_drains: List of ``(event, memory_objs)`` pairs accumulated by
            ``_issue_reads``.  The staging-buffer ``ref_count_down`` is
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
    pending_reads: set[int]
    last_drain_event: Optional[torch.cuda.Event]
    pending_drains: List[Tuple[Optional[torch.cuda.Event], List[Any]]]


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
        self._patched_modules: List[Tuple[Any, str, Callable]] = []
        self._patch_lock = threading.Lock()
        self._active_request_id: Optional[str] = None
        # Private stream for streaming-scatter copies (staging -> K cache).
        # Serialized by Tutti's _io_lock (one scatter runs at a time), so a
        # single stream per device suffices.  Kept off the default stream to
        # avoid cross-rank deadlocks with forward collectives.
        self._scatter_streams: Dict[int, torch.cuda.Stream] = {}
        self._scatter_streams_lock = threading.Lock()
        # Bulk read-ahead worker: one daemon thread per request walks every
        # CSA layer in ascending order and reads its whole covered slab.
        self._bulk_thread: Optional[threading.Thread] = None
        # Lazy-start state: the walk is ARMED at registration (CPU-only) and
        # STARTED at the first CSA gate (= compute phase, NVMe queue idle).
        self._bulk_arm_lock = threading.Lock()
        self._armed_bulk_req_id: Optional[str] = None
        self._bulk_started_req_id: Optional[str] = None
        # Layers whose full covered range the walker has landed for the
        # active request; gates skip the top-K miss scan for these.
        self._fully_resident_layers: set[int] = set()
        # (layer_id, chunk key str) -> (first_compressed_block,
        # n_compressed_blocks, layer_byte_offset, physical_block_ids) for
        # chunks whose slab FULLY landed in a walker pass.  A later request
        # presenting the identical signature has its bytes already in the
        # same K-cache rows (retrieve never writes the csa_attention_kv
        # group while the filter is active), so the walker skips the read
        # AND the SM-contending scatter.  Replaced (not merged) on every
        # new-request registration so entries never outlive one request
        # cycle; see ``_walker_resident_skip_enabled`` for the safety
        # argument.
        self._landed_chunk_signatures: Dict[
            Tuple[int, str], Tuple[int, int, int, Tuple[int, ...]]
        ] = {}
        # Per-layer chunk-key strings of the ACTIVE request whose bytes are
        # already resident (signature match); the walker and the arm step
        # exclude their blocks.
        self._resident_chunk_keys: Dict[int, set[str]] = {}

    def _scatter_stream_for(self, device: torch.device) -> torch.cuda.Stream:
        """Return the stream used for K-cache scatter copies.

        Reuses the Tutti loader's private ``io_stream`` when available: that
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
        io_stream = getattr(self._tutti_loader, "_io_stream", None)
        if io_stream is not None:
            return io_stream
        index = device.index if device.index is not None else 0
        with self._scatter_streams_lock:
            stream = self._scatter_streams.get(index)
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                self._scatter_streams[index] = stream
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
    def bulk_read_ahead_active(self) -> bool:
        """Return True when bulk layer-cadence read-ahead owns prefill reads.

        The IndexerSSDManager checks this to skip the prefill proxy fire
        path entirely (proxy scoring + microbatch dispatch are pure overhead
        when the walker reads every layer's slab anyway).
        """
        return _bulk_predicted_enabled()

    @property
    def bytes_per_block(self) -> int:
        """Bytes per compressed K cache block (block_size × token_bytes)."""
        return self._bytes_per_block

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
        # vLLM's MLA K cache tensor has ``num_blocks`` slots; the same slot
        # count is the upper bound on compressed block ids the indexer can
        # ever emit for this layer.  Allocate a bool bitmap sized to match
        # so ``_miss_ids_for_topk`` can answer "is this block already in
        # pool" entirely on the GPU.
        num_blocks = int(k_cache_tensor.shape[0])
        in_pool_bitmap = torch.zeros(
            num_blocks,
            dtype=torch.bool,
            device=k_cache_tensor.device,
        )
        self._layers[int(layer_id)] = CSAAttentionKVLayerState(
            layer_id=int(layer_id),
            compressed_block_size=self._compressed_block_size,
            token_bytes=self._token_bytes,
            k_cache_tensor=k_cache_tensor,
            in_pool_bitmap=in_pool_bitmap,
            chunks=[],
            pending_reads_lock=threading.Condition(),
            pending_reads=set(),
            last_drain_event=None,
            pending_drains=[],
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
        if not is_new_request:
            # Second/later chunk-step of the SAME request (multi-step hits:
            # 16K+ increments).  The NVMe-resident prefix is fixed at first
            # registration, so rebuilding chunk maps is pure churn — and it
            # RACES the in-flight bulk walker: replacing state.chunks /
            # physical_block_ids mid-walk desynchronizes its scatter plan
            # (stale rows -> illegal access family), and the accompanying
            # retrieve's _tutti_batched_get overwrites the loader's LBA
            # cache with CSA-filtered extents the walker can't resolve.
            # Keep the first registration; just refresh the LBA cache union.
            if self._pending_raw_lba_cache:
                try:
                    self._tutti_loader.register_lba_cache(
                        self._pending_raw_lba_cache
                    )
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
                        chunk.key.to_string() if hasattr(chunk.key, "to_string")
                        else repr(chunk.key),
                        chunk.first_compressed_block,
                        expected,
                    )
                expected = chunk.end_compressed_block
            state.chunks = ordered
            # Only reset drain and bitmap state for genuinely new requests.
            # Per-microbatch re-registration (same req_id) must preserve the
            # in_pool_bitmap and last_drain_event so staging slots from reads
            # already in flight are not orphaned (leak → pool exhaustion).
            if is_new_request:
                state.in_pool_bitmap.zero_()
                with state.pending_reads_lock:
                    state.pending_reads.clear()
                    state.last_drain_event = None

        # Resident-chunk skip (V26): a chunk whose (key, physical rows,
        # offsets) signature matches a slab FULLY landed for the immediately
        # previous request still holds its bytes in the same K-cache rows —
        # the retrieve path never writes the csa_attention_kv group while
        # the prefetcher is attached, decode appends only to newly allocated
        # rows, and unmatched signatures are dropped right here so no entry
        # survives a request that could have dirtied its rows.  Matched
        # chunks skip the walker's NVMe read AND its SM-contending scatter
        # (the measured 0.7-1.0 s steady-state tax).
        self._match_resident_chunks(req_id)

        # Layer-cadence bulk read-ahead (LMCACHE_CSA_BULK_PREDICTED=1,
        # default on): during prefill the per-query top-K union saturates
        # the prefix (measured candidates == all chunks), so per-layer
        # proxy-predicted fires degenerate into 21 full-prefix reads with
        # fire-tax on top.  Instead, one background thread walks the CSA
        # layers in ascending order and reads each layer's whole covered
        # slab exactly once, racing ahead of the forward pass: layer L's
        # bytes land while the GPU is still computing layers < L.  The
        # patched indexer forward's drain/miss correction remains the
        # correctness net: any block the walker has not landed yet is
        # fetched by the miss path.
        if is_new_request and _bulk_predicted_enabled():
            self._arm_bulk_read_ahead(str(req_id))

    def _layer_fully_resident(self, layer_id: int) -> bool:
        """Return True when the walker landed this layer's full range.

        Set by the layer-major walk after a layer's every covered block has
        been scattered and verified (landed == covered).  Cleared on each
        new request registration.  When True the true-topK miss scan is
        skipped entirely: any in-range top-K entry is resident by
        construction, and out-of-range entries are sentinel padding that
        ``_miss_ids_for_topk`` would drop anyway.
        """
        with self._bulk_arm_lock:
            return int(layer_id) in self._fully_resident_layers

    def _mark_layer_fully_resident(self, layer_id: int) -> None:
        with self._bulk_arm_lock:
            self._fully_resident_layers.add(int(layer_id))

    def _clear_fully_resident(self) -> None:
        with self._bulk_arm_lock:
            self._fully_resident_layers.clear()

    def _chunk_signature(
        self, chunk: CSAAttentionKVChunkLoc
    ) -> Tuple[int, int, int, Tuple[int, ...]]:
        """Return the identity tuple used for resident-chunk matching.

        Two registrations of the same content in the same K-cache rows
        produce equal signatures: the chunk key string (captured in the
        dict key) pins the content, ``physical_block_ids`` pins the
        destination rows, and the offsets pin the on-disk slab.
        """
        return (
            int(chunk.first_compressed_block),
            int(chunk.n_compressed_blocks),
            int(chunk.layer_byte_offset),
            tuple(int(r) for r in chunk.physical_block_ids),
        )

    @staticmethod
    def _chunk_key_str(chunk: CSAAttentionKVChunkLoc) -> str:
        """Return a stable string identity for the chunk's cache key."""
        key = chunk.key
        return key.to_string() if hasattr(key, "to_string") else repr(key)

    def _match_resident_chunks(self, req_id: str) -> None:
        """Mark or relocate this request's chunks already resident in HBM.

        Compares every registered chunk against ``_landed_chunk_signatures``
        (slabs fully landed for the immediately previous request).  Chunk
        keys are content hashes, so a key hit means the bytes this chunk
        wants are ALREADY in the K cache — the only question is which rows:

        * rows equal → mark resident directly (no work at all);
        * rows differ (the common case: vLLM's block allocator hands the
          same prefix different physical rows every request, measured
          100% churn) → relocate on the GPU: gather the old rows to a
          temporary, scatter to the new rows.  This runs at registration
          time (retrieve phase, SMs idle waiting on NVMe), costs ~2 ms of
          HBM bandwidth per request, and replaces both the walker's NVMe
          re-read and its compute-phase scatter (the measured 0.7-1.0 s
          SM-contention tax).

        Matched chunks' blocks are flipped on in ``in_pool_bitmap`` and
        their keys recorded in ``_resident_chunk_keys`` so the walker skips
        them; their signatures are re-recorded with the NEW rows so the
        next request matches against where the bytes now live.  Unmatched
        previous signatures are dropped — residency claims never survive a
        request whose chunks landed elsewhere.

        Safety: gather-before-scatter through a temporary makes arbitrary
        old/new row aliasing safe; each layer completes its gather before
        its scatter, and layers use disjoint tensors.  Source rows are
        intact because the retrieve path never writes the csa_attention_kv
        group while the filter is active and the previous request's
        increment/decode tokens went to rows disjoint from its prefix rows.
        Serial workloads only — see ``_walker_resident_skip_enabled``.

        Args:
            req_id: Active request id (logging only).
        """
        with self._bulk_arm_lock:
            prev_signatures = dict(self._landed_chunk_signatures)
        self._resident_chunk_keys = {}
        if not prev_signatures or not _walker_resident_skip_enabled():
            with self._bulk_arm_lock:
                self._landed_chunk_signatures = {}
            return
        matched: Dict[Tuple[int, str], Tuple[int, int, int, Tuple[int, ...]]] = {}
        total_chunks = 0
        key_hits = 0
        same_row_hits = 0
        relocated_chunks = 0
        relocated_rows = 0
        t0 = time.perf_counter()
        for layer_id, state in self._layers.items():
            resident_keys: set[str] = set()
            resident_block_ids: List[int] = []
            # Relocation plan for this layer: parallel old/new row lists.
            old_rows: List[int] = []
            new_rows: List[int] = []
            relocate_chunks: List[
                Tuple[str, CSAAttentionKVChunkLoc]
            ] = []
            n_rows_cache = int(state.k_cache_tensor.shape[0])
            for chunk in state.chunks:
                total_chunks += 1
                # Row-fallback chunks (no slot_mapping) use sequence-position
                # rows that silently collide across requests; never treat
                # them as resident.
                if not chunk.physical_block_ids:
                    continue
                key_str = self._chunk_key_str(chunk)
                sig_key = (int(layer_id), key_str)
                prev = prev_signatures.get(sig_key)
                if prev is None:
                    continue
                sig_now = self._chunk_signature(chunk)
                # Same slab identity (block span + byte offset) required;
                # only the destination rows may differ.
                if prev[:3] != sig_now[:3] or len(prev[3]) != len(sig_now[3]):
                    continue
                key_hits += 1
                if prev[3] == sig_now[3]:
                    same_row_hits += 1
                    resident_keys.add(key_str)
                    matched[sig_key] = sig_now
                    resident_block_ids.extend(
                        range(
                            chunk.first_compressed_block,
                            chunk.end_compressed_block,
                        )
                    )
                    continue
                # Row churn: plan a gather/scatter relocation.  Validate
                # bounds now; one bad row disqualifies the chunk (walker
                # covers it instead).
                if not all(
                    0 <= int(r) < n_rows_cache for r in prev[3]
                ) or not all(0 <= int(r) < n_rows_cache for r in sig_now[3]):
                    continue
                old_rows.extend(int(r) for r in prev[3])
                new_rows.extend(int(r) for r in sig_now[3])
                relocate_chunks.append((key_str, chunk))
            if relocate_chunks:
                device = state.k_cache_tensor.device
                scatter_stream = self._scatter_stream_for(device)
                k_flat = state.k_cache_tensor.view(torch.uint8).reshape(
                    n_rows_cache, -1
                )
                with torch.inference_mode(), torch.cuda.stream(
                    scatter_stream
                ):
                    # Upload on the consuming stream (V17 lesson: a
                    # cross-stream upload races into half-written indices).
                    old_t = torch.as_tensor(old_rows, dtype=torch.int64).to(
                        device, non_blocking=True
                    )
                    new_t = torch.as_tensor(new_rows, dtype=torch.int64).to(
                        device, non_blocking=True
                    )
                    # Gather fully before scatter: temp copy makes any
                    # old/new aliasing (allocator reshuffle) safe.
                    temp = k_flat.index_select(0, old_t)
                    k_flat.index_copy_(0, new_t, temp)
                scatter_stream.synchronize()
                relocated_rows += len(old_rows)
                for key_str, chunk in relocate_chunks:
                    relocated_chunks += 1
                    resident_keys.add(key_str)
                    matched[
                        (int(layer_id), key_str)
                    ] = self._chunk_signature(chunk)
                    resident_block_ids.extend(
                        range(
                            chunk.first_compressed_block,
                            chunk.end_compressed_block,
                        )
                    )
            if resident_keys:
                self._resident_chunk_keys[int(layer_id)] = resident_keys
                with torch.inference_mode():
                    ids_tensor = torch.as_tensor(
                        resident_block_ids,
                        dtype=torch.int64,
                        device=state.in_pool_bitmap.device,
                    )
                    state.in_pool_bitmap[ids_tensor] = True
        with self._bulk_arm_lock:
            self._landed_chunk_signatures = matched
        if matched:
            logger.info(
                "CSAAttentionKVPrefetchManager: resident-chunk skip req=%s "
                "matched=%d/%d chunks (same_row=%d relocated=%d rows=%d) "
                "in %.1f ms",
                req_id,
                len(matched),
                total_chunks,
                same_row_hits,
                relocated_chunks,
                relocated_rows,
                (time.perf_counter() - t0) * 1000.0,
            )
        elif prev_signatures:
            # Signatures existed but nothing matched — log why so the field
            # diagnosis (key churn vs slab churn) is one grep away.
            logger.info(
                "CSAAttentionKVPrefetchManager: resident-chunk skip req=%s "
                "matched=0/%d (prev_sigs=%d key_hits=%d)",
                req_id,
                total_chunks,
                len(prev_signatures),
                key_hits,
            )

    def _chunk_is_resident(
        self, layer_id: int, chunk: CSAAttentionKVChunkLoc
    ) -> bool:
        """Return True when this chunk's slab already sits in its K rows."""
        keys = self._resident_chunk_keys.get(int(layer_id))
        if not keys:
            return False
        return self._chunk_key_str(chunk) in keys

    def _arm_bulk_read_ahead(self, req_id: str) -> None:
        """Arm (but do not start) the bulk walker for this request.

        Timeline insight (V19-V23 all fought the same structural fight):
        for a cache-hit request the base is retrieved, so vLLM's forward
        computes ONLY the increment, in the FINAL scheduler step, AFTER all
        retrieve steps.  The NVMe queue is busy for the whole retrieve
        phase and idle for the whole compute phase.  A walker started at
        registration time (= start of retrieve) contends with the retrieve
        no matter how the lock is sliced (whole-call starves the walker;
        per-batch fragments the retrieve from 11.7 to 2 GB/s).

        So: prepare everything on the CPU now, and START the NVMe reads at
        the first CSA gate arrival (= compute start, queue idle).  At an
        idle-queue 11.7 GB/s the whole 1.5 GB walk takes ~130 ms and the
        first layer's ~60 ms read is satisfied while the first CSA layer's
        attention is still upstream of the gate.
        """
        with self._bulk_arm_lock:
            self._armed_bulk_req_id = req_id
            self._bulk_started_req_id = None
            self._fully_resident_layers.clear()
        # Mark every covered block of every layer as in-flight NOW, not at
        # walk start: the first gate's miss correction consults
        # pending_reads before the lazily-started walker has a chance to
        # mark anything; unmarked blocks would be self-read in full
        # (duplicate 1.5 GB).  The walk (and its finally-clause) clears
        # these marks as bytes land or on failure.  Resident chunks
        # (signature match from the previous request) are excluded: their
        # bytes are already in the K rows and the walker will not touch
        # them — marking them pending would stall gates for the full miss
        # grace.  A layer whose EVERY chunk is resident is fully resident
        # before the walk even starts.
        for lid in self._csa_layer_ids:
            state = self._layers.get(int(lid))
            if state is None or not state.chunks:
                continue
            covered = set()
            all_resident = True
            for chunk in state.chunks:
                if self._chunk_is_resident(int(lid), chunk):
                    continue
                all_resident = False
                covered.update(
                    range(
                        chunk.first_compressed_block,
                        chunk.end_compressed_block,
                    )
                )
            if all_resident:
                self._mark_layer_fully_resident(int(lid))
                continue
            with state.pending_reads_lock:
                state.pending_reads.update(covered)

    def ensure_bulk_started(self) -> None:
        """Start the armed bulk walk if it has not started yet.

        Called from the patched indexer forward (first CSA gate) on the
        main thread; the walk itself still runs on a daemon thread so the
        gate only pays thread-spawn latency (~100 us), then proceeds into
        drain/miss logic that the walker's per-layer notifications feed.
        """
        with self._bulk_arm_lock:
            req_id = self._armed_bulk_req_id
            if not req_id or self._bulk_started_req_id == req_id:
                return
            self._bulk_started_req_id = req_id
        self._start_bulk_read_ahead(req_id)

    def _start_bulk_read_ahead(self, req_id: str) -> None:
        """Spawn the per-request bulk walker thread (replaces prior one)."""
        thread = threading.Thread(
            target=self._bulk_read_ahead,
            args=(req_id,),
            name=f"csa-bulk-read-{req_id[-8:]}",
            daemon=True,
        )
        self._bulk_thread = thread
        thread.start()

    def _bulk_read_ahead(self, req_id: str) -> None:
        """Layer-major bulk read: complete layers in consumption order.

        The forward pass consumes CSA layers in ascending order, so the
        walker reads layer L's whole slab set (one contiguous byte range
        per chunk, batched into a few large Tutti calls) and only then
        moves to layer L+1.  The first CSA layer's bytes land in a few
        hundred ms and every later layer's gate finds its bytes already
        resident.  The previous chunk-major walk completed each layer only
        at the END of the whole walk, so every gate waited on the full
        walk and the miss path duplicated reads — hit degenerated to
        retrieve + walk + compute instead of retrieve + max(walk, compute).

        All blocks of all layers are marked in-flight up front so the miss
        path waits for landing bytes instead of issuing duplicates; each
        staging batch clears its blocks and wakes waiters as it lands.
        """
        t0 = time.perf_counter()
        layer_states = [
            (lid, self._layers[lid])
            for lid in self._csa_layer_ids
            if lid in self._layers and self._layers[lid].chunks
        ]
        if not layer_states:
            return
        # Mark every covered block of every layer as in-flight (resident
        # chunks excluded — their bytes are already in the K rows; see
        # _match_resident_chunks).  Layers whose every chunk is resident
        # drop out of the walk entirely.
        pending_by_layer: Dict[int, set[int]] = {}
        skipped_resident_chunks = 0
        active_layer_states = []
        for lid, state in layer_states:
            covered = set()
            has_work = False
            for chunk in state.chunks:
                if self._chunk_is_resident(int(lid), chunk):
                    skipped_resident_chunks += 1
                    continue
                has_work = True
                covered.update(
                    range(chunk.first_compressed_block, chunk.end_compressed_block)
                )
            if not has_work:
                self._mark_layer_fully_resident(int(lid))
                continue
            active_layer_states.append((lid, state))
            pending_by_layer[lid] = covered
            with state.pending_reads_lock:
                state.pending_reads.update(covered)
        if skipped_resident_chunks and _timing_enabled():
            logger.info(
                "CSAAttentionKVPrefetchManager: walker skipping %d resident "
                "chunks for req=%s",
                skipped_resident_chunks,
                req_id,
            )
        layer_states = active_layer_states
        if not layer_states:
            logger.info(
                "CSAAttentionKVPrefetchManager: bulk read-ahead req=%s fully "
                "resident; nothing to read",
                req_id,
            )
            return

        def _clear_pending(lids_blocks: Dict[int, List[int]]) -> None:
            for lid, blocks in lids_blocks.items():
                state = self._layers.get(lid)
                if state is None:
                    continue
                with state.pending_reads_lock:
                    state.pending_reads.difference_update(blocks)
                    state.pending_reads_lock.notify_all()

        try:
            if self._pending_raw_lba_cache:
                try:
                    self._tutti_loader.register_lba_cache(
                        self._pending_raw_lba_cache
                    )
                except Exception:
                    logger.exception(
                        "CSAAttentionKVPrefetchManager: bulk LBA re-register "
                        "failed"
                    )

            device = layer_states[0][1].k_cache_tensor.device
            scatter_stream = self._scatter_stream_for(device)

            def _reregister_lba() -> None:
                # An interleaved retrieve overwrites the loader's extent
                # table for our pool path with CSA-filtered ranges between
                # walker batches (multi-step hits); restore the full-record
                # union before resolving this batch's extents.
                # ensure_lba_cache short-circuits on identity, so this is
                # near-free when nothing overwrote our table (the common
                # case) instead of re-sorting ~30k extents per batch —
                # measured 2.8 s absorbed by the first layer under V19.
                if self._pending_raw_lba_cache:
                    ensure = getattr(
                        self._tutti_loader, "ensure_lba_cache", None
                    )
                    if callable(ensure):
                        ensure(self._pending_raw_lba_cache)
                    else:
                        self._tutti_loader.register_lba_cache(
                            self._pending_raw_lba_cache
                        )

            total_chunks = 0
            # Single-layer-per-call ONLY: every attempt to batch multiple
            # layers into one Tutti call (chunk-major V10-V18, first-4-group
            # V22) reintroduced the mixed-layer scatter race (illegal access
            # family).  Layer isolation per call is the stable shape.
            layer_groups: List[List[Tuple[int, Any]]] = [
                [entry] for entry in layer_states
            ]
            for group in layer_groups:
                if str(self._active_request_id or "") != req_id:
                    logger.info(
                        "CSAAttentionKVPrefetchManager: layer-major walk for "
                        "%s aborted (request changed)",
                        req_id,
                    )
                    return
                group_t0 = time.perf_counter()
                keys: List[CacheEngineKey] = []
                disk_metas: List[Optional[DiskCacheMetadata]] = []
                file_offsets: List[int] = []
                read_ranges_per_key: List[
                    Optional[Tuple[KVObjectByteRange, ...]]
                ] = []
                # Per key: (layer_id, n_blocks, blk_bytes, comp_ids)
                plans: List[Tuple[int, int, int, List[int]]] = []
                chunk_refs: List[CSAAttentionKVChunkLoc] = []
                dst_rows_all: List[int] = []
                row_spans: List[Tuple[int, int]] = []
                k_flat_by_layer: Dict[int, torch.Tensor] = {}
                for lid, state in group:
                    n_rows_cache = int(state.k_cache_tensor.shape[0])
                    k_flat_by_layer[lid] = state.k_cache_tensor.view(
                        torch.uint8
                    ).reshape(n_rows_cache, -1)
                    for chunk in state.chunks:
                        if self._chunk_is_resident(int(lid), chunk):
                            continue
                        dst_rows: List[int] = []
                        comp_ids: List[int] = []
                        for local_idx in range(chunk.n_compressed_blocks):
                            comp_id = (
                                chunk.first_compressed_block + local_idx
                            )
                            if chunk.physical_block_ids and local_idx < len(
                                chunk.physical_block_ids
                            ):
                                row = int(
                                    chunk.physical_block_ids[local_idx]
                                )
                            else:
                                row = comp_id
                            if 0 <= row < n_rows_cache:
                                dst_rows.append(row)
                                comp_ids.append(comp_id)
                        span_start = len(dst_rows_all)
                        dst_rows_all.extend(dst_rows)
                        row_spans.append((span_start, len(dst_rows_all)))
                        keys.append(chunk.key)
                        disk_metas.append(chunk.disk_meta)
                        file_offsets.append(0)
                        read_ranges_per_key.append(
                            (
                                KVObjectByteRange(
                                    offset=chunk.layer_byte_offset,
                                    length=chunk.n_compressed_blocks
                                    * chunk.bytes_per_block,
                                    target_offset=0,
                                ),
                            )
                        )
                        plans.append(
                            (
                                lid,
                                len(dst_rows),
                                chunk.bytes_per_block,
                                comp_ids,
                            )
                        )
                        chunk_refs.append(chunk)
                if not keys:
                    continue

                with torch.inference_mode():
                    # Upload the row plan ON the scatter stream (same-stream
                    # upload/consume; a cross-stream default-stream upload
                    # raced into half-written indices -> illegal access).
                    with torch.cuda.stream(scatter_stream):
                        rows_gpu = torch.as_tensor(
                            dst_rows_all,
                            dtype=torch.int64,
                        ).to(device, non_blocking=True)
                    scatter_stream.synchronize()

                landed_by_layer: Dict[int, List[int]] = {}
                chunks_fully_landed: set[int] = set()

                def _scatter_batch(
                    batch_start: int,
                    batch_results: List[Optional[Any]],
                ) -> None:
                    batch_landed: Dict[int, List[int]] = {}
                    with torch.inference_mode(), torch.cuda.stream(
                        scatter_stream
                    ):
                        for offset_in_batch, memory_obj in enumerate(
                            batch_results
                        ):
                            chunk_idx = batch_start + offset_in_batch
                            plan_lid, n_blocks, blk_bytes, comp_ids = plans[
                                chunk_idx
                            ]
                            if (
                                memory_obj is None
                                or memory_obj.raw_tensor is None
                            ):
                                logger.warning(
                                    "CSAAttentionKVPrefetchManager: layer %d "
                                    "chunk %d returned no payload; miss path "
                                    "will cover it",
                                    plan_lid,
                                    chunk_idx,
                                )
                                continue
                            flat = memory_obj.raw_tensor.view(
                                torch.uint8
                            ).reshape(-1)
                            have = int(flat.numel())
                            if n_blocks <= 0:
                                continue
                            usable = min(n_blocks, max(0, have // blk_bytes))
                            if usable <= 0:
                                continue
                            src = flat[: usable * blk_bytes].view(
                                usable, blk_bytes
                            )
                            span_s, _span_e = row_spans[chunk_idx]
                            rows = rows_gpu[span_s : span_s + usable]
                            k_flat_by_layer[plan_lid].index_copy_(
                                0, rows, src
                            )
                            batch_landed.setdefault(plan_lid, []).extend(
                                comp_ids[:usable]
                            )
                            if usable == n_blocks:
                                chunks_fully_landed.add(chunk_idx)
                    scatter_stream.synchronize()
                    for lid_landed, comp_ids in batch_landed.items():
                        lstate = self._layers[lid_landed]
                        with torch.inference_mode():
                            ids_tensor = torch.as_tensor(
                                comp_ids,
                                dtype=torch.int64,
                                device=lstate.in_pool_bitmap.device,
                            )
                            lstate.in_pool_bitmap[ids_tensor] = True
                        with lstate.pending_reads_lock:
                            lstate.pending_reads.difference_update(comp_ids)
                            lstate.pending_reads_lock.notify_all()
                        landed_by_layer.setdefault(lid_landed, []).extend(
                            comp_ids
                        )

                with torch.inference_mode():
                    self._tutti_loader.load_chunks_to_hbm(
                        keys,
                        disk_metas,
                        shapes_per_key=None,
                        file_offsets=file_offsets,
                        read_ranges_per_key=read_ranges_per_key,
                        on_batch_loaded=_scatter_batch,
                        lock_per_batch=True,
                        before_batch=_reregister_lba,
                    )
                for lid_done, lstate in group:
                    landed = landed_by_layer.get(lid_done, [])
                    remaining = pending_by_layer.pop(lid_done, set())
                    remaining.difference_update(landed)
                    # Whatever this layer failed to land must not block
                    # its gate.
                    if remaining:
                        with lstate.pending_reads_lock:
                            lstate.pending_reads.difference_update(remaining)
                            lstate.pending_reads_lock.notify_all()
                    else:
                        # Full coverage landed: this layer's gate can skip
                        # the 33M-entry top-K miss scan entirely.
                        self._mark_layer_fully_resident(lid_done)
                # Record fully-landed chunk signatures so an identical
                # follow-up request (same key, same rows) skips both the
                # read and the SM-contending scatter.  Row-fallback chunks
                # (no physical_block_ids) are never recorded.  Skipped when
                # the active request changed mid-group: a stale write here
                # could poison the NEXT registration's match table with
                # rows another request may have since dirtied.
                if (
                    _walker_resident_skip_enabled()
                    and str(self._active_request_id or "") == req_id
                ):
                    with self._bulk_arm_lock:
                        for chunk_idx in chunks_fully_landed:
                            chunk = chunk_refs[chunk_idx]
                            if not chunk.physical_block_ids:
                                continue
                            plan_lid, plan_n_blocks = plans[chunk_idx][:2]
                            # A plan that dropped out-of-range rows covers
                            # fewer blocks than the chunk claims; recording
                            # it would assert residency for rows never
                            # written.
                            if plan_n_blocks != int(chunk.n_compressed_blocks):
                                continue
                            self._landed_chunk_signatures[
                                (int(plan_lid), self._chunk_key_str(chunk))
                            ] = self._chunk_signature(chunk)
                total_chunks += len(keys)
                if _timing_enabled():
                    logger.info(
                        "CSAAttentionKVPrefetchManager: layer-major walk "
                        "layers=%s chunks=%d landed=%d group_ms=%.1f",
                        [lid for lid, _ in group],
                        len(keys),
                        sum(len(v) for v in landed_by_layer.values()),
                        (time.perf_counter() - group_t0) * 1000.0,
                    )
        except Exception:
            logger.exception(
                "CSAAttentionKVPrefetchManager: bulk read-ahead failed for "
                "%s; miss correction will cover the remainder",
                req_id,
            )
        finally:
            # Whatever did not land must not leave the miss path waiting.
            _clear_pending(
                {lid: list(blocks) for lid, blocks in pending_by_layer.items()}
            )
        logger.info(
            "CSAAttentionKVPrefetchManager: bulk read-ahead req=%s "
            "chunks=%d layers=%d total_ms=%.1f",
            req_id,
            total_chunks,
            len(layer_states),
            (time.perf_counter() - t0) * 1000.0,
        )

    def fire_predicted_reads(
        self,
        layer_id: int,
        compressed_block_ids: Sequence[int],
    ) -> None:
        """Submit Tutti reads for the predicted ``top-K`` of one CSA layer.

        Called from :class:`IndexerSSDManager` after its HC-proxy emits a
        prediction for ``layer_id``.  Block ids already in the pool or with
        an in-flight read are skipped.

        Args:
            layer_id: Transformer-side CSA layer id.
            compressed_block_ids: Predicted compressed block ids.
        """
        if _bulk_predicted_enabled():
            # The bulk walker owns prefill read-ahead; per-layer predicted
            # fires would only duplicate its in-flight reads (measured: the
            # prefill top-K union saturates the prefix, so "predicted" ==
            # "everything" anyway).  Miss correction remains active.
            return
        self._submit_reads(layer_id, compressed_block_ids, label="predicted")

    def submit_miss_reads(
        self,
        layer_id: int,
        compressed_block_ids: Sequence[int],
    ) -> None:
        """Submit Tutti reads for blocks not covered by the prediction.

        Called from the patched ``DeepseekV4Indexer.forward`` after the true
        Lightning Indexer returns ``true_topk``.

        Args:
            layer_id: Transformer-side CSA layer id.
            compressed_block_ids: ``true_topk`` block ids whose K cache
                slots are not yet populated.
        """
        self._submit_reads(layer_id, compressed_block_ids, label="miss")

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
            # last_drain_event.  Any (ev, objs) pair in pending_drains was
            # appended together with setting last_drain_event = ev (under the
            # same lock), so synchronizing last_drain_event covers all of them.
            pending = state.pending_drains[:]
            state.pending_drains.clear()
        if event is not None:
            event.synchronize()
        # Release staging buffers now that the CUDA stream has confirmed all
        # non_blocking copies into the K cache have completed.
        for _, memory_objs in pending:
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
            indexer_prefetch_manager = getattr(indexer_module, "ssd_manager", None)
            mgr = self

            def _patched_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                t0 = time.perf_counter() if _timing_enabled() else 0.0
                wait_ms = 0.0
                # Lazy-start the bulk walk at the FIRST gate of the compute
                # phase: the NVMe queue is idle now (all retrieve steps are
                # done), so the walk runs at full bandwidth instead of
                # contending with the retrieve (V19-V23 lesson).
                try:
                    mgr.ensure_bulk_started()
                except Exception:
                    logger.exception(
                        "CSAAttentionKVPrefetchManager: bulk lazy-start "
                        "failed; miss correction will cover"
                    )
                try:
                    wait_predicted = getattr(
                        indexer_prefetch_manager,
                        "wait_for_csa_attention_kv_prediction",
                        None,
                    )
                    if callable(wait_predicted):
                        t_wait = time.perf_counter() if _timing_enabled() else 0.0
                        wait_predicted(layer_id)
                        wait_ms = (
                            (time.perf_counter() - t_wait) * 1000.0
                            if _timing_enabled()
                            else 0.0
                        )
                except Exception:
                    logger.exception(
                        "CSAAttentionKVPrefetchManager: prediction wait failed "
                        "for layer %d; falling back to true-topK correction",
                        layer_id,
                    )
                true_topk = orig_forward(*args, **kwargs)
                try:
                    t_drain0 = time.perf_counter() if _timing_enabled() else 0.0
                    mgr.drain_for_layer(layer_id)
                    first_drain_ms = (
                        (time.perf_counter() - t_drain0) * 1000.0
                        if _timing_enabled()
                        else 0.0
                    )
                    t_miss0 = time.perf_counter() if _timing_enabled() else 0.0
                    # Fast path: when the walker has landed this layer's
                    # whole covered range (pending empty + full-coverage
                    # flag), the miss set is empty by construction — skip
                    # the 33M-entry top-K scan (measured 46 ms x 21 layers
                    # ≈ 1 s per hit, the single largest ON-only tax).
                    if mgr._layer_fully_resident(layer_id):
                        miss_ids = []
                    else:
                        miss_ids = mgr._miss_ids_for_topk(layer_id, true_topk)
                    miss_ms = (
                        (time.perf_counter() - t_miss0) * 1000.0
                        if _timing_enabled()
                        else 0.0
                    )
                    if miss_ids:
                        mgr.submit_miss_reads(layer_id, miss_ids)
                    t_drain1 = time.perf_counter() if _timing_enabled() else 0.0
                    mgr.drain_for_layer(layer_id)
                    second_drain_ms = (
                        (time.perf_counter() - t_drain1) * 1000.0
                        if _timing_enabled()
                        else 0.0
                    )
                    if _timing_enabled():
                        logger.info(
                            "CSAAttentionKVPrefetchManager: correction "
                            "layer=%d true_entries=%d miss_blocks=%d "
                            "wait_ms=%.3f first_drain_ms=%.3f "
                            "miss_filter_ms=%.3f second_drain_ms=%.3f "
                            "total_ms=%.3f",
                            layer_id,
                            int(true_topk.numel()),
                            len(miss_ids),
                            wait_ms,
                            first_drain_ms,
                            miss_ms,
                            second_drain_ms,
                            (time.perf_counter() - t0) * 1000.0,
                        )
                except Exception:
                    logger.exception(
                        "CSAAttentionKVPrefetchManager: miss correction failed "
                        "for layer %d; downstream attention may read stale data",
                        layer_id,
                    )
                return true_topk

            indexer_module.forward = _patched_forward
            indexer_module._lmcache_csa_attention_kv_original_forward = orig_forward
            self._patched_modules.append(
                (indexer_module, "forward", orig_forward)
            )

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
        self.unpatch()
        self._layers.clear()
        self._active_request_id = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit_reads(
        self,
        layer_id: int,
        compressed_block_ids: Sequence[int],
        label: str,
    ) -> None:
        """Common path for predicted and miss read submission."""
        state = self._layers.get(int(layer_id))
        if state is None:
            return
        if not compressed_block_ids:
            return
        if _timing_enabled():
            logger.info(
                "CSAAttentionKVPrefetchManager: _submit_reads label=%s "
                "layer=%d candidates=%d",
                label,
                layer_id,
                len(compressed_block_ids),
            )
        if not state.chunks:
            logger.warning(
                "CSAAttentionKVPrefetchManager: no chunks registered for "
                "layer %d at %s submission; skipping %d ids",
                layer_id,
                label,
                len(compressed_block_ids),
            )
            return

        # Decide which block ids are net-new (not already in pool, not
        # already in flight).  ``in_pool_bitmap`` lives on the GPU so the
        # cross-check stays a CPU-only set lookup against the small
        # ``pending_reads`` set; the bitmap read happens after we've
        # narrowed the candidate list.
        pool_size = int(state.in_pool_bitmap.shape[0])
        # Clamp to the chunk map's covered compressed-block range as well:
        # the HC-proxy prediction routinely emits ids past the cached prefix
        # (short/partial prefixes cover only the first few blocks), and
        # ``_issue_reads`` cannot read uncovered ids.  The miss path already
        # drops them in ``_miss_ids_for_topk``; without this clamp the
        # predicted path books uncovered ids into ``pending_reads`` and
        # floods per-id "not covered by any chunk" warnings.
        covered_end = int(state.chunks[-1].end_compressed_block)
        limit = min(pool_size, covered_end)
        unique_ids = set(int(b) for b in compressed_block_ids)
        candidate_ids = sorted(bid for bid in unique_ids if 0 <= bid < limit)
        if len(candidate_ids) < len(unique_ids) and _timing_enabled():
            logger.info(
                "CSAAttentionKVPrefetchManager: _submit_reads label=%s "
                "layer=%d dropped %d/%d ids outside covered range [0, %d)",
                label,
                layer_id,
                len(unique_ids) - len(candidate_ids),
                len(unique_ids),
                limit,
            )
        with state.pending_reads_lock:
            if label == "miss":
                # Miss correction needs these bytes NOW.  Give in-flight bulk
                # reads a short grace to land, then read the still-pending
                # blocks ourselves.  With the LAYER-MAJOR walker the current
                # layer's slabs land within a few hundred ms of its gate
                # (layers complete in consumption order), so a 1 s grace
                # almost always avoids duplicate reads; the bound still
                # protects against a stalled walker (EP divergence killed
                # runs when this wait was unbounded).
                deadline = time.monotonic() + 1.0
                while any(
                    bid in state.pending_reads for bid in candidate_ids
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        still = sum(
                            1 for bid in candidate_ids
                            if bid in state.pending_reads
                        )
                        logger.info(
                            "CSAAttentionKVPrefetchManager: miss correction "
                            "layer %d proceeding to self-read %d blocks still "
                            "in flight on the walker",
                            layer_id,
                            still,
                        )
                        break
                    state.pending_reads_lock.wait(remaining)
                # Pass every candidate to the bitmap filter below: blocks the
                # walker already landed are in in_pool_bitmap and get dropped
                # there; blocks still in flight are (by definition) not in
                # the bitmap and get duplicated — benign, same bytes to same
                # rows.  (An earlier variant returned only still-pending ids
                # here, which silently skipped blocks the walker had FAILED
                # to land — stale K-cache rows.)
                ids_after_flight = list(candidate_ids)
            else:
                in_flight = state.pending_reads
                ids_after_flight = [
                    bid for bid in candidate_ids if bid not in in_flight
                ]
            if not ids_after_flight:
                return
        ids_tensor = torch.as_tensor(
            ids_after_flight,
            dtype=torch.int64,
            device=state.in_pool_bitmap.device,
        )
        in_pool_mask = state.in_pool_bitmap[ids_tensor]
        new_ids_tensor = ids_tensor[~in_pool_mask]
        if new_ids_tensor.numel() == 0:
            if _timing_enabled():
                logger.info(
                    "CSAAttentionKVPrefetchManager: _submit_reads label=%s "
                    "layer=%d new=0 reason=resident_or_inflight",
                    label,
                    layer_id,
                )
            return
        new_ids = new_ids_tensor.cpu().tolist()
        with state.pending_reads_lock:
            new_ids = [bid for bid in new_ids if bid not in state.pending_reads]
            if not new_ids:
                if _timing_enabled():
                    logger.info(
                        "CSAAttentionKVPrefetchManager: _submit_reads "
                        "label=%s layer=%d new=0 reason=pending_race",
                        label,
                        layer_id,
                    )
                return
            state.pending_reads.update(new_ids)
        if _timing_enabled():
            logger.info(
                "CSAAttentionKVPrefetchManager: _submit_reads label=%s "
                "layer=%d new=%d",
                label,
                layer_id,
                len(new_ids),
            )

        try:
            # Predicted reads often finish on a background thread, outside
            # vLLM's model-forward inference_mode context.  The target K cache
            # and resident bitmap may be inference tensors, so all in-place GPU
            # updates must re-enter inference_mode here as well.
            with torch.inference_mode():
                event, issued_memory_objs = self._issue_reads(state, sorted(new_ids))
        except Exception:
            with state.pending_reads_lock:
                state.pending_reads.difference_update(new_ids)
                state.pending_reads_lock.notify_all()
            logger.exception(
                "CSAAttentionKVPrefetchManager: failed to issue %s reads for "
                "layer %d (%d ids)",
                label,
                layer_id,
                len(new_ids),
            )
            return

        with state.pending_reads_lock:
            state.pending_reads.difference_update(new_ids)
            state.last_drain_event = event
            # Store (event, memory_objs) atomically with last_drain_event so
            # drain_for_layer can safely release buffers after synchronizing.
            state.pending_drains.append((event, issued_memory_objs))
            # Wake any miss-correction waiting for these blocks to land.
            state.pending_reads_lock.notify_all()
        if new_ids:
            # Update GPU bitmap in bulk so subsequent miss filtering sees
            # these block ids as already-resident.
            with torch.inference_mode():
                ids_tensor = torch.as_tensor(
                    new_ids,
                    dtype=torch.int64,
                    device=state.in_pool_bitmap.device,
                )
                state.in_pool_bitmap.index_fill_(0, ids_tensor, True)

    def _issue_reads(
        self,
        state: CSAAttentionKVLayerState,
        sorted_block_ids: List[int],
    ) -> Tuple[Optional[torch.cuda.Event], List[Any]]:
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

        Returns:
            Tuple of ``(event, memory_objs)`` kept for interface stability;
            always ``(None, [])`` now.  Scatter copies are performed inside
            Tutti's ``on_batch_loaded`` callback and synchronized before the
            call returns, so there is nothing left to drain: no CUDA event
            is pending and no staging buffer references are retained.
        """
        if not sorted_block_ids:
            return None, []
        # Group block ids by chunk and coalesce consecutive ids per chunk.
        chunks_used: Dict[int, Tuple[CSAAttentionKVChunkLoc, List[int]]] = {}
        chunk_iter = iter(enumerate(state.chunks))
        current_chunk_idx, current_chunk = next(chunk_iter, (None, None))
        for block_id in sorted_block_ids:
            while current_chunk is not None and block_id >= current_chunk.end_compressed_block:
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
        chunk_row_spans: List[Tuple[int, int]] = []

        for chunk_idx in sorted(chunks_used.keys()):
            chunk, ids_in_chunk = chunks_used[chunk_idx]
            ranges: List[KVObjectByteRange] = []
            cursor = 0
            sorted_ids_in_chunk = sorted(ids_in_chunk)
            span_start = len(dst_rows_all)
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
            keys.append(chunk.key)
            disk_metas.append(chunk.disk_meta)
            file_offsets.append(0)
            read_ranges_per_key.append(tuple(ranges))
            chunk_row_spans.append((span_start, len(dst_rows_all)))

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
        n_rows = int(state.k_cache_tensor.shape[0])
        # Byte view so index_copy_ sees identical dtype/shape on both sides
        # regardless of the K cache's declared element type.
        k_cache_flat = state.k_cache_tensor.view(torch.uint8).reshape(n_rows, -1)
        scatter_stream = self._scatter_stream_for(state.k_cache_tensor.device)
        with torch.inference_mode():
            # Upload on the SAME stream that consumes it (see bulk walker:
            # default-stream upload + scatter-stream consume raced).
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
                            "load_chunks_to_hbm returned MemoryObj without "
                            "raw_tensor"
                        )
                    span_start, span_end = chunk_row_spans[key_index]
                    n_blocks = span_end - span_start
                    if n_blocks <= 0:
                        continue
                    flat = tensor.view(torch.uint8).reshape(-1)
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
                    k_cache_flat.index_copy_(0, rows, src)
                # Staging slots are recycled as soon as this callback returns;
                # retire the scatter kernels first (private stream only).
            scatter_stream.synchronize()

        self._tutti_loader.load_chunks_to_hbm(
            keys,
            disk_metas,
            shapes_per_key=None,
            file_offsets=file_offsets,
            read_ranges_per_key=read_ranges_per_key,
            on_batch_loaded=_scatter_batch,
        )
        if comp_ids_all:
            with torch.inference_mode():
                bitmap_ids = torch.as_tensor(
                    comp_ids_all,
                    dtype=torch.int64,
                    device=state.in_pool_bitmap.device,
                )
                state.in_pool_bitmap[bitmap_ids] = True
        # All copies are already synchronized; no deferred staging buffers to
        # release and no drain event is needed.
        return None, []

    def _miss_ids_for_topk(
        self,
        layer_id: int,
        true_topk: torch.Tensor,
    ) -> List[int]:
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
            Sorted unique list of block ids not yet in the layer's pool and
            within the chunk map's registered compressed-block range.  The
            indexer often emits block ids past the current prefix
            (sentinel padding); skipping them here avoids noisy
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
            return []
        entries = true_topk.reshape(-1)
        if entries.numel() == 0:
            return []
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
            return []
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
        miss_ids = miss_mask.nonzero(as_tuple=False).reshape(-1)
        if miss_ids.numel() == 0:
            return []
        return miss_ids.cpu().tolist()
