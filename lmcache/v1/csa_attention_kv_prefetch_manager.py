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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import os
import threading

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
        disk_meta: Cached :class:`DiskCacheMetadata` pointing at the chunk's
            on-disk location.  Reused verbatim for Tutti reads.
        layer_byte_offset: Byte offset inside the chunk's payload where the
            csa_attention_kv slab for this CSA layer begins.  Tutti reads
            stage a ``read_ranges_per_key`` entry at this offset plus the
            block-relative offset.
        bytes_per_block: ``block_size * token_bytes`` for the csa_attention_kv
            group (block_size == 64 typically, token_bytes == 584).
    """

    first_compressed_block: int
    n_compressed_blocks: int
    key: CacheEngineKey
    disk_meta: DiskCacheMetadata
    layer_byte_offset: int
    bytes_per_block: int

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
        compressed_block_ids_in_pool: Set of compressed block ids whose K
            cache slot currently holds valid data.  Used to dedup predicted
            and miss reads.
        chunks: Ordered list of LMCache chunks covering the active request's
            prefix for this CSA layer.  Populated by
            :meth:`register_request_chunks`.
        pending_reads_lock: Guards mutation of ``pending_reads``.
        pending_reads: Set of compressed block ids whose Tutti read is
            currently in-flight or queued.
        last_drain_event: Optional CUDA event recording the completion of
            the latest read submission.  Sparse attention waits on this
            event during :meth:`CSAAttentionKVPrefetchManager.drain_for_layer`.
    """

    layer_id: int
    compressed_block_size: int
    token_bytes: int
    k_cache_tensor: torch.Tensor
    compressed_block_ids_in_pool: set[int]
    chunks: List[CSAAttentionKVChunkLoc]
    pending_reads_lock: threading.Lock
    pending_reads: set[int]
    last_drain_event: Optional[torch.cuda.Event]


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
        self._layers: Dict[int, CSAAttentionKVLayerState] = {}
        self._csa_layer_ids = tuple(sorted(int(lid) for lid in csa_layer_ids))
        self._patched_modules: List[Tuple[Any, str, Callable]] = []
        self._patch_lock = threading.Lock()
        self._active_request_id: Optional[str] = None

    @property
    def csa_layer_ids(self) -> Tuple[int, ...]:
        """Return registered CSA layer ids in ascending order."""
        return self._csa_layer_ids

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
        self._layers[int(layer_id)] = CSAAttentionKVLayerState(
            layer_id=int(layer_id),
            compressed_block_size=self._compressed_block_size,
            token_bytes=self._token_bytes,
            k_cache_tensor=k_cache_tensor,
            compressed_block_ids_in_pool=set(),
            chunks=[],
            pending_reads_lock=threading.Lock(),
            pending_reads=set(),
            last_drain_event=None,
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

        Args:
            req_id: Request identifier (advisory; the manager only keeps
                one active request at a time in the initial implementation).
            chunks_by_layer: Mapping from CSA layer id to ordered chunk
                descriptors.  Empty entries clear the layer's chunk list.
        """
        self._active_request_id = str(req_id)
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
            state.compressed_block_ids_in_pool.clear()
            with state.pending_reads_lock:
                state.pending_reads.clear()
                state.last_drain_event = None

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
        if event is not None:
            event.synchronize()

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
                true_topk = orig_forward(*args, **kwargs)
                try:
                    miss_ids = mgr._miss_ids_for_topk(layer_id, true_topk)
                    if miss_ids:
                        mgr.submit_miss_reads(layer_id, miss_ids)
                    mgr.drain_for_layer(layer_id)
                except Exception:
                    logger.exception(
                        "CSAAttentionKVPrefetchManager: miss correction failed "
                        "for layer %d; downstream attention may read stale data",
                        layer_id,
                    )
                return true_topk

            indexer_module.forward = _patched_forward
            self._patched_modules.append(
                (indexer_module, "forward", orig_forward)
            )

    def unpatch(self) -> None:
        """Restore all patched indexer forwards."""
        with self._patch_lock:
            for module, attr, original in self._patched_modules:
                setattr(module, attr, original)
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
        if not state.chunks:
            logger.warning(
                "CSAAttentionKVPrefetchManager: no chunks registered for "
                "layer %d at %s submission; skipping %d ids",
                layer_id,
                label,
                len(compressed_block_ids),
            )
            return

        with state.pending_reads_lock:
            new_ids = [
                int(bid)
                for bid in compressed_block_ids
                if int(bid) not in state.compressed_block_ids_in_pool
                and int(bid) not in state.pending_reads
            ]
            if not new_ids:
                return
            state.pending_reads.update(new_ids)

        try:
            event = self._issue_reads(state, sorted(new_ids))
        except Exception:
            with state.pending_reads_lock:
                state.pending_reads.difference_update(new_ids)
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
            state.compressed_block_ids_in_pool.update(new_ids)
            state.last_drain_event = event

    def _issue_reads(
        self,
        state: CSAAttentionKVLayerState,
        sorted_block_ids: List[int],
    ) -> Optional[torch.cuda.Event]:
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
            CUDA event recording completion of the staging copies, or
            ``None`` if no reads were issued.
        """
        if not sorted_block_ids:
            return None
        # Group block ids by chunk and coalesce consecutive ids per chunk.
        chunks_used: Dict[int, Tuple[CSAAttentionKVChunkLoc, List[int]]] = {}
        chunk_index_by_block: Dict[int, int] = {}
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
            chunk_index_by_block[block_id] = current_chunk_idx

        keys: List[CacheEngineKey] = []
        disk_metas: List[Optional[DiskCacheMetadata]] = []
        file_offsets: List[int] = []
        read_ranges_per_key: List[Optional[Tuple[KVObjectByteRange, ...]]] = []
        target_offsets_per_chunk: List[List[Tuple[int, int]]] = []

        for chunk_idx in sorted(chunks_used.keys()):
            chunk, ids_in_chunk = chunks_used[chunk_idx]
            ranges: List[KVObjectByteRange] = []
            target_offsets: List[Tuple[int, int]] = []
            cursor = 0
            sorted_ids_in_chunk = sorted(ids_in_chunk)
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
                for offset_in_run in range(run_length):
                    target_offsets.append(
                        (run_start + offset_in_run, cursor + offset_in_run * chunk.bytes_per_block)
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
            target_offsets_per_chunk.append(target_offsets)

        if not keys:
            return None

        memory_objs = self._tutti_loader.load_chunks_to_hbm(
            keys,
            disk_metas,
            shapes_per_key=None,
            file_offsets=file_offsets,
            read_ranges_per_key=read_ranges_per_key,
        )
        event = torch.cuda.Event(blocking=False)
        try:
            for memory_obj, target_offsets in zip(
                memory_objs,
                target_offsets_per_chunk,
                strict=True,
            ):
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
                flat = tensor.reshape(-1).contiguous().view(torch.uint8)
                for compressed_block_id, byte_offset in target_offsets:
                    dst_block_idx = compressed_block_id // state.compressed_block_size
                    entry_idx = compressed_block_id % state.compressed_block_size
                    if entry_idx != 0:
                        # The current implementation assumes block_size-aligned
                        # reads (entries are batched per K cache block).  This
                        # is true for the standard DSv4 layout where one
                        # csa_block_id maps 1:1 to one entry inside the K cache
                        # block.  Reaching this branch indicates a layout
                        # mismatch.
                        raise RuntimeError(
                            f"Unexpected non-aligned compressed_block_id "
                            f"{compressed_block_id} (entry_idx={entry_idx})"
                        )
                    dst_slot = state.k_cache_tensor[dst_block_idx]
                    src_slice = flat[
                        byte_offset : byte_offset
                        + state.compressed_block_size * state.token_bytes
                    ].view(
                        state.compressed_block_size,
                        state.token_bytes,
                    )
                    dst_slot.copy_(src_slice, non_blocking=True)
        finally:
            for memory_obj in memory_objs:
                if memory_obj is None:
                    continue
                ref_count_down = getattr(memory_obj, "ref_count_down", None)
                if callable(ref_count_down):
                    ref_count_down()
        event.record()
        return event

    def _miss_ids_for_topk(
        self,
        layer_id: int,
        true_topk: torch.Tensor,
    ) -> List[int]:
        """Return ``true_topk`` block ids missing from the pool.

        Args:
            layer_id: Transformer-side CSA layer id.
            true_topk: Output tensor of the true Lightning Indexer, shape
                ``[num_queries, top_k]`` or ``[top_k]``.

        Returns:
            Sorted unique list of block ids not yet in the layer's pool.
        """
        state = self._layers.get(int(layer_id))
        if state is None:
            return []
        block_ids = sorted(set(int(v) for v in true_topk.reshape(-1).cpu().tolist()))
        return [bid for bid in block_ids if bid not in state.compressed_block_ids_in_pool]
