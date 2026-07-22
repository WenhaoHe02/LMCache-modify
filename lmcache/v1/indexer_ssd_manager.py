# SPDX-License-Identifier: Apache-2.0
"""
IndexerSSDManager — SSD-backed HBM pool for DeepSeek V4 CSA Indexer cache.

Architecture
------------
vLLM's SparseAttnIndexer remains the source of truth for decode top-K. This
module uses a small HBM pool only to prefetch data before the target CSA layer
needs it, then fills any true-topK misses after the official Lightning Indexer
result is known:

  * Per CSA layer: a fixed-size HBM pool in vLLM's packed IndexerCache layout.
    Each 64-token block stores all value bytes first, followed by per-token
    scale bytes. SSD files keep one interleaved value+scale record per logical
    token, so pool writes/readbacks convert between the two layouts.
  * All other token K-vectors are stored on SSD (one flat file per CSA layer).
  * Two-tier LRU:
      _lru_resident — the prefill seed blocks (top-1024 from initial prefill),
                      evicted last.
      _lru_ordinary — speculatively prefetched / new decode blocks, evicted
                      first.
  * Speculative prefetch: the caller (DeepseekV4DecoderLayer) fires
    fire_async_for_layer() before MoE FFN.  The FFN window (~3 ms) overlaps
    with NVMe reads, but predicted IDs never replace the true top-K.

Integration with vLLM
---------------------
  1. IndexerSSDManager is created by LMCache's vllm_v1_adapter when it detects
     DSv4 CSA indexer-cache layers.
  2. DeepseekV4DecoderLayer.forward() calls:
       mgr.fire_async_for_layer(next_csa_layer_id, residual_f, positions)
       # before FFN
  3. SparseAttnIndexer.forward_cuda() runs the official true LI, then calls:
       mgr.correct_true_topk(layer_id, true_topk)

The optional pool-only scorer is diagnostic-only and must be enabled explicitly
in vLLM with LMCACHE_INDEXER_ENABLE_POOL_SCORING=1.

Only single-sequence decode is supported in this version (batch_size = 1).
Multi-sequence support can be added by keying pool state per seq_id.
"""

from __future__ import annotations

import os
import inspect
import threading
import time
from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

from lmcache.logging import init_logger
from lmcache.v1.csa_pipeline_nvtx import CsaNvtxEvent, csa_pipeline_nvtx

logger = init_logger(__name__)
DEEPGEMM_PAGED_BLOCK_SIZE = 64
_COLD_PROXY_WARM_ROWS = 8192
_INDEXER_SSD_MANAGER: Optional["IndexerSSDManager"] = None


def get_indexer_ssd_manager() -> Optional["IndexerSSDManager"]:
    """Return the process-local CSA indexer SSD manager, if one is attached."""
    return _INDEXER_SSD_MANAGER


def set_indexer_ssd_manager(manager: Optional["IndexerSSDManager"]) -> None:
    """Set the process-local CSA indexer SSD manager used by GPU connectors."""
    global _INDEXER_SSD_MANAGER
    _INDEXER_SSD_MANAGER = manager


class _SpeculativeReadCancelled(RuntimeError):
    """Raised when Tutti declines an unsubmitted speculative batch."""


def _future_list_item(future: "Future[List[bytes]]", index: int) -> bytes:
    """Return one item from a shared batch-read future."""
    return future.result()[index]


def _proxy_num_rows(proxy_state: Optional[torch.Tensor]) -> int:
    """Return the number of token rows represented by a proxy tensor."""
    if proxy_state is None or proxy_state.ndim == 0:
        return 0
    if proxy_state.ndim >= 3 and int(proxy_state.shape[0]) == 1:
        return int(proxy_state.shape[1])
    return int(proxy_state.shape[0])


def _flatten_proxy_state_for_positions(
    proxy_state: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return proxy rows aligned to flattened position rows."""
    flat_positions = positions.reshape(-1)
    if proxy_state.ndim >= 2:
        if int(proxy_state.shape[0]) == int(flat_positions.shape[0]):
            return proxy_state, flat_positions, int(flat_positions.shape[0])
        flat_proxy = proxy_state.reshape(-1, int(proxy_state.shape[-1]))
        max_rows = min(int(flat_proxy.shape[0]), int(flat_positions.shape[0]))
        return (
            flat_proxy[:max_rows].contiguous(),
            flat_positions[:max_rows].contiguous(),
            max_rows,
        )
    max_rows = min(int(proxy_state.shape[0]), int(flat_positions.shape[0]))
    return proxy_state[:max_rows], flat_positions[:max_rows].contiguous(), max_rows


def _align_proxy_rows(
    proxy_state: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Align residual-proxy rows with position rows without dropping tokens.

    DeepSeek V4 ``hc_pre`` expects the residual to keep its final
    ``[hc_mult, hidden]`` dimensions. The residual-proxy prediction must run
    over every available token row for the chunk; this helper only handles the
    defensive case where the residual and position tensors disagree.
    """
    return _flatten_proxy_state_for_positions(proxy_state, positions)


def _env_flag(name: str) -> bool:
    """Return True when environment variable *name* is set to a truthy value."""
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable with a safe fallback."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d", name, value, default)
        return default


def _deepseek_indexer_cache_classes() -> tuple[type[torch.nn.Module], ...]:
    """Return DeepSeek V4 indexer-cache classes available in this vLLM build."""
    class_specs = (
        (
            "vllm.models.deepseek_v4.nvidia.ops.attention",
            "DeepseekV4IndexerCache",
        ),
        (
            "vllm.model_executor.models.deepseek_v4",
            "DeepseekV4IndexerCache",
        ),
        # Some older NVIDIA vLLM trees place the shared indexer-cache class in
        # deepseek_v2.py even when it backs the DSv4-compatible sparse indexer.
        # This fallback is only for cache discovery; residual-proxy hooks remain
        # DSv4-specific.
        (
            "vllm.model_executor.models.deepseek_v2",
            "DeepseekV32IndexerCache",
        ),
    )
    classes: list[type[torch.nn.Module]] = []
    for module_name, class_name in class_specs:
        try:
            module = __import__(module_name, fromlist=[class_name])
        except ImportError:
            continue
        cache_cls = getattr(module, class_name, None)
        if isinstance(cache_cls, type):
            classes.append(cache_cls)
    if not classes:
        raise ImportError("No DeepSeek V4-compatible indexer cache class found")
    return tuple(classes)


def _profile_accuracy_enabled() -> bool:
    """Return True when the lightweight accuracy profiler is enabled."""
    return _env_flag("LMCACHE_INDEXER_PROFILE_ACCURACY") or os.path.exists(
        "/tmp/lmcache_indexer_profile_accuracy"
    )


# ---------------------------------------------------------------------------
# Environment flag: set LMCACHE_DISABLE_RESIDENT_INDEXER=1 to use flat LRU
# ---------------------------------------------------------------------------
_RESIDENT_ENABLED: bool = os.environ.get("LMCACHE_DISABLE_RESIDENT_INDEXER", "0") != "1"

_TIMING_LIMIT: int = max(0, _env_int("LMCACHE_INDEXER_TIMING_LIMIT", 512))
_TIMING_SEED_VERBOSE = _env_flag("LMCACHE_INDEXER_TIMING_SEED_VERBOSE")


def _timing_enabled() -> bool:
    """Return True when lightweight timing diagnostics are enabled."""
    return _env_flag("LMCACHE_INDEXER_TIMING") or os.path.exists(
        "/tmp/lmcache_indexer_timing"
    )


def _select_rank_local_proxy_blocks(
    token_ids: torch.Tensor,
    cursor: int,
    num_blocks: int,
    block_budget: int,
) -> torch.Tensor:
    """Select a fixed-budget rank-local block list from proxy token IDs.

    The per-query proxy output is reduced to block occurrence counts locally.
    Only the most frequent blocks survive; no rank collective or persistent
    score vector is required. Empty slots are represented by ``-1`` so the
    result always has a host-known size and can be copied asynchronously.

    Args:
        token_ids: Proxy-selected logical token IDs on CPU or CUDA.
        cursor: Exclusive upper bound of initialized logical token IDs.
        num_blocks: Number of initialized compressed blocks.
        block_budget: Maximum number of blocks to return.

    Returns:
        One-dimensional int64 tensor of length
        ``min(num_blocks, block_budget)``. Valid block IDs precede optional
        ``-1`` padding in unspecified order.

    Raises:
        ValueError: If ``num_blocks`` or ``block_budget`` is not positive.
    """
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if block_budget <= 0:
        raise ValueError("block_budget must be positive")
    flat = token_ids.reshape(-1).to(dtype=torch.int64)
    valid = flat[(flat >= 0) & (flat < int(cursor))]
    block_ids = torch.div(
        valid,
        DEEPGEMM_PAGED_BLOCK_SIZE,
        rounding_mode="floor",
    )
    counts = torch.bincount(block_ids, minlength=int(num_blocks))
    budget = min(int(num_blocks), int(block_budget))
    top_counts, top_blocks = torch.topk(counts, k=budget, sorted=False)
    return torch.where(
        top_counts > 0,
        top_blocks,
        torch.full_like(top_blocks, -1),
    )


def _weighted_predicted_block_hits(
    entries: torch.Tensor,
    valid: torch.Tensor,
    predicted_blocks: Set[int],
    num_blocks: int,
) -> tuple[int, int]:
    """Count true top-K entries covered by predicted compressed blocks.

    Unlike unique-block recall, this metric preserves query/top-K frequency:
    a block selected by many query rows contributes once per selected entry.
    That is the relevant coverage signal for a bounded speculative I/O budget
    when the union of all query rows spans most of a long prefix.

    Args:
        entries: Flattened true compressed-entry IDs.
        valid: Boolean mask selecting entries inside the cached prefix.
        predicted_blocks: Predicted compressed-block IDs.
        num_blocks: Number of compressed blocks in the cached prefix.

    Returns:
        ``(covered_entries, valid_entries)``.
    """
    total = int(valid.sum().item())
    if total <= 0 or not predicted_blocks or num_blocks <= 0:
        return 0, total
    predicted_bitmap = torch.zeros(
        num_blocks,
        dtype=torch.bool,
        device=entries.device,
    )
    predicted_ids = torch.tensor(
        sorted(predicted_blocks),
        dtype=torch.int64,
        device=entries.device,
    )
    predicted_ids = predicted_ids[(predicted_ids >= 0) & (predicted_ids < num_blocks)]
    if predicted_ids.numel() == 0:
        return 0, total
    predicted_bitmap[predicted_ids] = True
    valid_blocks = entries[valid] // DEEPGEMM_PAGED_BLOCK_SIZE
    hits = int(predicted_bitmap[valid_blocks].sum().item())
    return hits, total


# ---------------------------------------------------------------------------
# Low-level pread helper (Linux/macOS)
# ---------------------------------------------------------------------------


def _pread(fd: int, size: int, offset: int) -> bytes:
    """Read *size* bytes from *fd* at *offset* without moving the file cursor."""
    buf = bytearray(size)
    view = memoryview(buf)
    pos = 0
    while pos < size:
        chunk = os.pread(fd, size - pos, offset + pos)
        if not chunk:
            break
        n = len(chunk)
        view[pos : pos + n] = chunk
        pos += n
    return bytes(buf[:pos])


def _pwrite(fd: int, data: bytes, offset: int) -> None:
    """Write *data* to *fd* at *offset* without moving the file cursor."""
    view = memoryview(data)
    pos = 0
    while pos < len(data):
        n = os.pwrite(fd, view[pos:], offset + pos)
        pos += n


# ---------------------------------------------------------------------------
# SSD block store — flat file, one entry per token
# ---------------------------------------------------------------------------


class IndexerBlockStore:
    """Per-CSA-layer flat SSD file storing FP8-quantized K vectors.

    Layout: token_id * token_bytes  →  raw bytes (uint8, head_dim_with_scale
    bytes per token).  The file is pre-sized lazily (sparse file on Linux).
    """

    def __init__(
        self,
        store_dir: str,
        layer_id: int,
        token_bytes: int,
        max_seq_len: int,
    ) -> None:
        """
        Args:
            store_dir: Directory for SSD files.
            layer_id: CSA layer index (used in filename).
            token_bytes: Bytes per token (head_dim_with_scale).
            max_seq_len: Maximum context length (determines file capacity).
        """
        store_path = Path(store_dir)
        # The per-rank directories are normally created before snvme binds
        # and unmounts the filesystem.  Avoid a redundant mkdir syscall here:
        # during the lazy first-request attach it can race that unmount and
        # surface as EPERM even though the directory already exists.
        if not store_path.is_dir():
            store_path.mkdir(parents=True, exist_ok=True)
        self._path = store_path / f"indexer_layer_{layer_id:03d}.bin"
        self._token_bytes = token_bytes
        self._max_seq_len = max_seq_len
        self._fd: Optional[int] = None
        # Pre-create the sparse file AND open the fd eagerly: on the Tutti
        # deployment the store lives on a drive that snvme later unmounts for
        # GPU-direct bind.  An already-open fd keeps working on the detached
        # filesystem, but any lazy mkdir/open after the unmount lands on the
        # bare mountpoint (root-fs leak) or raises FileNotFoundError and
        # aborts the whole retrieve batch.
        self._ensure_file()
        self._open()

    def _open(self) -> int:
        if self._fd is None:
            self._ensure_file()
            self._fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT)
        return self._fd

    def _ensure_file(self) -> None:
        """Ensure the backing directory and sparse file exist."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            return
        with open(self._path, "wb") as f:
            f.seek(self._max_seq_len * self._token_bytes - 1)
            f.write(b"\x00")

    def read_token(self, token_id: int) -> bytes:
        """Synchronously read one token's K vector from SSD."""
        fd = self._open()
        return _pread(fd, self._token_bytes, token_id * self._token_bytes)

    def write_token(self, token_id: int, data: bytes) -> None:
        """Write one token's K vector to SSD."""
        if len(data) != self._token_bytes:
            raise ValueError(f"Expected {self._token_bytes} bytes, got {len(data)}")
        fd = self._open()
        _pwrite(fd, data, token_id * self._token_bytes)

    def write_tokens_contiguous(self, start_token_id: int, data: bytes) -> None:
        """Write contiguous token bytes starting at *start_token_id*."""
        if len(data) % self._token_bytes != 0:
            raise ValueError(
                f"Expected byte length divisible by {self._token_bytes}, "
                f"got {len(data)}"
            )
        fd = self._open()
        _pwrite(fd, data, start_token_id * self._token_bytes)

    def read_tokens_batch(self, token_ids: List[int]) -> List[bytes]:
        """Read multiple token K vectors from SSD (sequential pread calls)."""
        return [self.read_token(tid) for tid in token_ids]

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


# ---------------------------------------------------------------------------
# Per-layer HBM pool
# ---------------------------------------------------------------------------


class IndexerStorageProtocol:
    """Per-CSA-layer indexer storage interface.

    Two implementations live alongside :class:`IndexerSSDManager`:

    * :class:`IndexerBlockStore` — legacy per-layer ``.bin`` file via
      ``os.pread``/``os.pwrite``.  Used when ``LMCACHE_INDEXER_BACKEND``
      is unset or ``file``.
    * :class:`TuttiIndexerBlockStore` — GPU-direct NVMe path through
      :class:`lmcache.v1.indexer_tutti_backend.TuttiIndexerStorage`.  Used when
      ``LMCACHE_INDEXER_BACKEND=tutti`` and a Tutti loader is attached.

    The protocol is informational only; both implementations are duck-typed.
    Methods:

    * ``read_token(token_id) -> bytes``
    * ``write_token(token_id, data: bytes) -> None``
    * ``write_tokens_contiguous(start_token_id, data: bytes) -> None``
    * ``read_tokens_batch(token_ids: List[int]) -> List[bytes]``
    * ``close() -> None``
    """


class TuttiIndexerBlockStore:
    """Per-CSA-layer indexer block store backed by Tutti raw NVMe extents.

    Thin adapter that satisfies the same interface as
    :class:`IndexerBlockStore` but routes I/O through a shared
    :class:`lmcache.v1.indexer_tutti_backend.TuttiIndexerStorage` instance.
    The storage class owns the raw region and the Tutti loader reference;
    one ``TuttiIndexerBlockStore`` exists per CSA layer and targets one
    slot inside that region.

    Demand reads support true-topK correction. Predicted reads use bounded
    speculative batches so foreground retrieve and store announcements
    can cancel unsubmitted indexer work. The write path (LMCache retrieve seed
    + prefill new-token persistence) is wired through
    :meth:`TuttiIndexerStorage.write_bytes`.
    """

    def __init__(
        self,
        tutti_storage: Any,
        layer_id: int,
    ) -> None:
        """Bind this store to one CSA layer slot inside the Tutti region.

        Args:
            tutti_storage: Shared :class:`TuttiIndexerStorage` owning the raw
                region and the Tutti loader.
            layer_id: CSA layer id for this store; must be a key in
                ``tutti_storage._slots``.

        Raises:
            KeyError: If ``layer_id`` was not registered with
                ``tutti_storage`` at construction time.
        """
        self._storage = tutti_storage
        self._slot = tutti_storage.slot_for_layer(int(layer_id))
        self._token_bytes = self._slot.token_bytes
        self._max_seq_len = self._slot.max_seq_len

    def read_token(self, token_id: int) -> bytes:
        """Synchronously read one token's K vector from the Tutti raw region.

        Args:
            token_id: Global token position index.

        Returns:
            Raw uint8 bytes of length ``token_bytes``.

        Raises:
            ValueError: If ``token_id`` is outside slot capacity.
        """
        results = self.read_tokens_batch([int(token_id)])
        if not results:
            raise RuntimeError(
                f"TuttiIndexerBlockStore.read_token returned no bytes for "
                f"token_id={token_id}"
            )
        return results[0]

    def read_tokens_batch(
        self,
        token_ids: List[int],
        *,
        io_priority: str = "demand",
        should_continue: Optional[Callable[[], bool]] = None,
        deadline_monotonic: Optional[float] = None,
    ) -> List[bytes]:
        """Read multiple token K vectors via Tutti GPU-direct DMA.

        Each batch issues a single ``load_chunks_to_hbm`` call against the
        shared indexer raw region.  Adjacent ``token_ids`` are coalesced into
        contiguous byte ranges so NVMe issues large sequential reads instead
        of many sector-sized random reads.

        Args:
            token_ids: Token ids to read.  Returned bytes follow the input
                order (not sorted).
            io_priority: Tutti admission class for this read.
            should_continue: Optional speculative cancellation predicate.
            deadline_monotonic: Optional absolute speculative deadline.

        Returns:
            List of bytes, one entry per ``token_id`` with length
            ``token_bytes`` each.  Returns an empty list if ``token_ids`` is
            empty.

        Raises:
            ValueError: If any ``token_id`` is outside slot capacity.
            RuntimeError: If Tutti reports a read failure.
        """
        if not token_ids:
            return []
        request = self._storage.build_read_request(self._slot, token_ids)
        if request.is_empty:
            return [b""] * len(token_ids)

        # Allocate ephemeral disk_metadata + key bundle for Tutti.  The
        # synthetic CacheEngineKey carries the layer+token-range identity in
        # its chunk_hash so logging is debuggable even if the request is
        # batched with others later.
        memory_objs = self._storage.load_read_request(
            request,
            io_priority=io_priority,
            should_continue=should_continue,
            deadline_monotonic=deadline_monotonic,
        )
        if not memory_objs or memory_objs[0] is None:
            if io_priority == "speculative":
                raise _SpeculativeReadCancelled(
                    "Tutti declined an unsubmitted speculative indexer read"
                )
            raise RuntimeError(
                f"Tutti load_chunks_to_hbm returned no payload for indexer "
                f"layer {self._slot.layer_id} request "
                f"({len(request.read_ranges)} ranges, "
                f"{request.total_nbytes} bytes)"
            )
        memory_obj = memory_objs[0]
        try:
            tensor = memory_obj.raw_tensor
            if tensor is None:
                raise RuntimeError(
                    "TuttiDirectLoader returned a MemoryObj without raw_tensor"
                )
            # Flatten to 1-D uint8 and copy to CPU bytes.  This intermediate
            # CPU bounce exists only to satisfy the IndexerBlockStore bytes
            # API; the higher-level IndexerSSDManager will be refactored to
            # consume the GPU tensor directly in a later commit.
            flat = tensor.reshape(-1).contiguous()
            host = flat.cpu().numpy().tobytes()
        finally:
            ref_count_down = getattr(memory_obj, "ref_count_down", None)
            if callable(ref_count_down):
                ref_count_down()

        if len(host) < request.total_nbytes:
            raise RuntimeError(
                f"Tutti read returned {len(host)} bytes, expected "
                f"{request.total_nbytes} for layer {self._slot.layer_id}"
            )

        # Reconstruct per-token bytes from the read payload.  Token order in
        # the request follows token_runs (sorted, deduplicated).  Map back to
        # the caller's original token_ids order.
        per_token: Dict[int, bytes] = {}
        cursor = 0
        for first_token, n_tokens in request.token_runs:
            run_bytes = host[cursor : cursor + n_tokens * request.token_bytes]
            cursor += n_tokens * request.token_bytes
            for offset in range(n_tokens):
                tid = first_token + offset
                start = offset * request.token_bytes
                end = start + request.token_bytes
                per_token[tid] = run_bytes[start:end]

        return [per_token[int(tid)] for tid in token_ids]

    def write_token(self, token_id: int, data: bytes) -> None:
        """Persist one token K vector to the Tutti raw region.

        Args:
            token_id: Global token position index.
            data: Raw uint8 bytes of length ``token_bytes``.

        Raises:
            ValueError: If ``data`` size or ``token_id`` is invalid.
        """
        if len(data) != self._token_bytes:
            raise ValueError(f"Expected {self._token_bytes} bytes, got {len(data)}")
        self._storage.write_bytes(self._slot, int(token_id), data)

    def write_tokens_contiguous(self, start_token_id: int, data: bytes) -> None:
        """Persist contiguous token K vectors starting at ``start_token_id``.

        Args:
            start_token_id: Global token position of the first token.
            data: Raw uint8 bytes; length must be a multiple of ``token_bytes``.

        Raises:
            ValueError: If ``data`` size or ``start_token_id`` is invalid.
        """
        if len(data) % self._token_bytes != 0:
            raise ValueError(
                f"Expected byte length divisible by {self._token_bytes}, "
                f"got {len(data)}"
            )
        self._storage.write_bytes(self._slot, int(start_token_id), data)

    def close(self) -> None:
        """No-op; the underlying Tutti loader is owned by the cache engine."""


class IndexerHBMPool:
    """Fixed-size HBM tensor holding FP8-quantized K vectors for one CSA layer.

    Pool layout matches vLLM's packed IndexerCache:
    ``[num_blocks, 64, token_bytes]``. Within each 64-token block, value bytes
    for every token come first and scale bytes are stored at the end of the
    block. Public insert/read helpers accept and return interleaved
    ``value+scale`` token records because that is the SSD file layout.

    LRU eviction uses two tiers when _RESIDENT_ENABLED:
      _lru_resident: prefill-seed slots (lowest eviction priority)
      _lru_ordinary: all other slots
    """

    def __init__(
        self,
        pool_size: int,
        token_bytes: int,
        device: torch.device,
    ) -> None:
        """
        Args:
            pool_size: Maximum number of tokens held in HBM.
            token_bytes: Bytes per token (head_dim_with_scale).
            device: CUDA device.
        """
        block_size = DEEPGEMM_PAGED_BLOCK_SIZE
        num_blocks = (pool_size + block_size - 1) // block_size
        self._pool_size = num_blocks * block_size
        self._token_bytes = token_bytes
        self._scale_bytes = 4
        self._head_dim = token_bytes - self._scale_bytes
        self.block_size = block_size

        # Same packed block layout as vLLM IndexerCache.
        self.pool_tensor: torch.Tensor = torch.zeros(
            num_blocks,
            block_size,
            token_bytes,
            dtype=torch.uint8,
            device=device,
        )
        # pool slot → global token ID (-1 = empty)
        self.pool_ids: torch.Tensor = torch.full(
            (self._pool_size,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        # block_table for fp8_fp4_paged_mqa_logits: [1, pool_size] int32
        self.block_table: torch.Tensor = torch.arange(
            num_blocks,
            dtype=torch.int32,
            device=device,
        ).unsqueeze(0)

        # CPU-side index structures
        self._id_to_slot: Dict[int, int] = {}
        self._slot_to_id: List[int] = [-1] * self._pool_size
        self._free: List[int] = list(range(self._pool_size))
        self._lru_ordinary: OrderedDict[int, None] = OrderedDict()
        self._lru_resident: OrderedDict[int, None] = OrderedDict()
        self._resident_slots: Set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contains(self, token_id: int) -> bool:
        """Return True if *token_id* is currently in the HBM pool."""
        return token_id in self._id_to_slot

    def insert(self, token_id: int, data: bytes) -> int:
        """Insert *token_id* with raw *data* bytes; return its pool slot.

        If *token_id* is already present, promote it in LRU and return its
        existing slot without writing data.

        Args:
            token_id: Global token position index.
            data: Raw uint8 bytes of length token_bytes.

        Returns:
            Pool slot index.
        """
        if token_id in self._id_to_slot:
            slot = self._id_to_slot[token_id]
            self._touch(slot)
            return slot

        slot = self._free.pop() if self._free else self._evict_one()
        with torch.inference_mode():
            self._write_slot(slot, data)
            self.pool_ids[slot] = token_id
        self._slot_to_id[slot] = int(token_id)
        self._id_to_slot[token_id] = slot
        self._lru_ordinary[slot] = None
        return slot

    def load_tokens(self, token_ids: List[int], token_bytes: torch.Tensor) -> None:
        """Bulk-load interleaved token records into an empty HBM pool.

        Args:
            token_ids: Global token IDs to load into pool slots.
            token_bytes: CPU uint8 tensor ``[N, token_bytes]`` containing
                interleaved value+scale records for the corresponding IDs.
        """
        if len(token_ids) != int(token_bytes.shape[0]):
            raise ValueError(
                f"token_ids length {len(token_ids)} != token_bytes rows "
                f"{int(token_bytes.shape[0])}"
            )
        if token_bytes.ndim != 2 or int(token_bytes.shape[1]) != self._token_bytes:
            raise ValueError(
                f"Expected token_bytes [N, {self._token_bytes}], got "
                f"{tuple(token_bytes.shape)}"
            )
        if len(token_ids) > self._pool_size:
            raise ValueError(
                f"Cannot load {len(token_ids)} tokens into pool size {self._pool_size}"
            )
        if not token_ids:
            return

        with torch.inference_mode():
            rows = token_bytes.to(device=self.pool_tensor.device, dtype=torch.uint8)
            n = len(token_ids)
            slots = torch.arange(n, dtype=torch.long, device=self.pool_tensor.device)
            block_idx = slots // self.block_size
            block_offset = slots % self.block_size
            value_offsets = block_offset.unsqueeze(1) * self._head_dim + torch.arange(
                self._head_dim, dtype=torch.long, device=self.pool_tensor.device
            )
            scale_offsets = (
                self.block_size * self._head_dim
                + block_offset.unsqueeze(1) * self._scale_bytes
                + torch.arange(
                    self._scale_bytes,
                    dtype=torch.long,
                    device=self.pool_tensor.device,
                )
            )
            flat_blocks = self.pool_tensor.view(self.pool_tensor.shape[0], -1)
            flat_blocks[block_idx.unsqueeze(1), value_offsets] = rows[
                :, : self._head_dim
            ]
            flat_blocks[block_idx.unsqueeze(1), scale_offsets] = rows[
                :, self._head_dim :
            ]

            self.pool_ids[:n] = torch.tensor(
                token_ids,
                dtype=self.pool_ids.dtype,
                device=self.pool_ids.device,
            )
            self.pool_ids[n:] = -1
        self._slot_to_id = [int(token_id) for token_id in token_ids] + [-1] * (
            self._pool_size - n
        )
        self._id_to_slot = {int(tid): idx for idx, tid in enumerate(token_ids)}
        self._free = list(range(n, self._pool_size))
        self._lru_ordinary = OrderedDict((idx, None) for idx in range(n))
        self._lru_resident.clear()
        self._resident_slots.clear()

    def protect(self, token_id: int) -> None:
        """Mark *token_id* as a resident (prefill-seed) block.

        Resident blocks are moved to the resident LRU tier and evicted only
        after all ordinary blocks have been evicted.  No-op if
        LMCACHE_DISABLE_RESIDENT_INDEXER=1.

        Args:
            token_id: Global token position to protect.
        """
        if not _RESIDENT_ENABLED:
            return
        slot = self._id_to_slot.get(token_id)
        if slot is None or slot in self._resident_slots:
            return
        self._lru_ordinary.pop(slot, None)
        self._lru_resident[slot] = None
        self._resident_slots.add(slot)

    def protect_only(self, token_ids: List[int]) -> None:
        """Make in-pool *token_ids* the resident set and demote all others.

        Args:
            token_ids: Global token positions that should remain protected.
        """
        if not _RESIDENT_ENABLED:
            return
        keep_slots = {
            self._id_to_slot[int(token_id)]
            for token_id in token_ids
            if int(token_id) in self._id_to_slot
        }
        ordinary_items = [
            (slot, None) for slot in self._lru_ordinary if slot not in keep_slots
        ]
        ordinary_items.extend(
            (slot, None) for slot in self._lru_resident if slot not in keep_slots
        )
        resident_items = [(slot, None) for slot in keep_slots]
        self._lru_ordinary = OrderedDict(ordinary_items)
        self._lru_resident = OrderedDict(resident_items)
        self._resident_slots = set(keep_slots)

    def reset(self) -> None:
        """Clear all pool metadata before loading a new sequence."""
        # ``pool_ids`` is created while the model executes under
        # ``torch.inference_mode``. Reset can run later on the asynchronous
        # prefill-eviction worker, where inference mode is thread-local and
        # therefore no longer inherited.
        with torch.inference_mode():
            self.pool_ids.fill_(-1)
        self._slot_to_id = [-1] * self._pool_size
        self._id_to_slot.clear()
        self._free = list(range(self._pool_size))
        self._lru_ordinary.clear()
        self._lru_resident.clear()
        self._resident_slots.clear()

    def get_slot(self, token_id: int) -> int:
        """Return pool slot for *token_id*; raise KeyError if not present."""
        return self._id_to_slot[token_id]

    def valid_slot_count(self) -> int:
        """Return the number of populated slots using CPU-side metadata."""
        return len(self._id_to_slot)

    def read_slot(self, slot: int) -> torch.Tensor:
        """Read one pool slot as interleaved ``value+scale`` token bytes."""
        block_idx = slot // self.block_size
        block_offset = slot % self.block_size
        value_start = block_offset * self._head_dim
        scale_start = self.block_size * self._head_dim + (
            block_offset * self._scale_bytes
        )
        flat_block = self.pool_tensor[block_idx].view(-1)
        value = flat_block[value_start : value_start + self._head_dim]
        scale = flat_block[scale_start : scale_start + self._scale_bytes]
        return torch.cat((value, scale))

    def evict_to_ssd(self, store: IndexerBlockStore) -> None:
        """Write every in-pool token to SSD (called after prefill)."""
        for token_id, slot in self._id_to_slot.items():
            raw = self.read_slot(slot).cpu().numpy().tobytes()
            store.write_token(token_id, raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _touch(self, slot: int) -> None:
        if slot in self._resident_slots:
            self._lru_resident.move_to_end(slot)
        elif slot in self._lru_ordinary:
            self._lru_ordinary.move_to_end(slot)

    def _write_slot(self, slot: int, data: bytes) -> None:
        """Write interleaved token bytes into the packed block layout."""
        if len(data) != self._token_bytes:
            raise ValueError(f"Expected {self._token_bytes} bytes, got {len(data)}")
        raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        block_idx = slot // self.block_size
        block_offset = slot % self.block_size
        value_start = block_offset * self._head_dim
        scale_start = self.block_size * self._head_dim + (
            block_offset * self._scale_bytes
        )
        self.pool_tensor[block_idx].view(-1)[
            value_start : value_start + self._head_dim
        ] = raw[: self._head_dim]
        self.pool_tensor[block_idx].view(-1)[
            scale_start : scale_start + self._scale_bytes
        ] = raw[self._head_dim :]

    def _evict_one(self) -> int:
        if self._lru_ordinary:
            slot, _ = self._lru_ordinary.popitem(last=False)
        elif self._lru_resident:
            slot, _ = self._lru_resident.popitem(last=False)
            self._resident_slots.discard(slot)
        else:
            raise RuntimeError("IndexerHBMPool: both LRU queues are empty")
        old_id = self._slot_to_id[slot]
        if old_id >= 0:
            del self._id_to_slot[old_id]
        self._slot_to_id[slot] = -1
        return slot


# ---------------------------------------------------------------------------
# Top-level manager
# ---------------------------------------------------------------------------


class IndexerSSDManager:
    """Manages SSD-backed HBM pools for all CSA Indexer layers in DeepSeek V4.

    One instance is shared across all CSA layers.  Each CSA layer has its own
    :class:`IndexerHBMPool` and :class:`IndexerBlockStore`.

    The manager is invoked from two call sites in the vLLM forward pass:

    1. **Before MoE FFN** (``DeepseekV4DecoderLayer.forward``):
       :meth:`fire_async_for_layer` kicks off async NVMe reads for the
       predicted delta blocks of the next CSA layer.

    2. **Inside SparseAttnIndexer scoring** (``sparse_attn_indexer.py``):
       :meth:`prepare_pool` drains pending async reads, then returns the pool
       tensor + a block_table for the current sequence.  After the topk kernel
       runs, :meth:`record_topk` saves the result for the next step.
    """

    def __init__(
        self,
        csa_layer_ids: List[int],
        store_dir: str,
        pool_size: int,
        token_bytes: int,
        max_seq_len: int,
        io_workers: int,
        device: torch.device,
        tutti_storage: Optional[Any] = None,
    ) -> None:
        """
        Args:
            csa_layer_ids: Sorted list of CSA layer indices in the model.
            store_dir: Directory for SSD backing files (file backend) or for
                the synthetic raw region path metadata (Tutti backend).
            pool_size: HBM pool capacity in tokens per CSA layer.
            token_bytes: Bytes per token (head_dim + scale overhead).
            max_seq_len: Maximum context length for SSD file sizing.
            io_workers: Number of async I/O threads.
            device: CUDA device for HBM pools.
            tutti_storage: Optional shared
                :class:`lmcache.v1.indexer_tutti_backend.TuttiIndexerStorage`
                owning a pre-reserved Tutti raw region.  When provided, this
                manager uses :class:`TuttiIndexerBlockStore` for every CSA
                layer (GPU-direct NVMe path).  When ``None``, the legacy
                per-layer ``.bin`` file backend is used.
        """
        self._csa_layer_ids = csa_layer_ids
        self._store_dir = store_dir
        self._pool_size = pool_size
        self._token_bytes = token_bytes
        self._scale_bytes = 4
        self._head_dim = token_bytes - self._scale_bytes
        self._device = device
        self._prefill_ready_timeout_s = float(
            _env_int("LMCACHE_INDEXER_PREFILL_READY_TIMEOUT_SEC", 600)
        )
        self._prediction_gate_timeout_s = float(
            _env_int("LMCACHE_CSA_PREDICTION_GATE_TIMEOUT_SEC", 5)
        )
        self._tutti_storage = tutti_storage

        # Per-layer HBM pool
        self._pools: Dict[int, IndexerHBMPool] = {
            lid: IndexerHBMPool(pool_size, token_bytes, device) for lid in csa_layer_ids
        }
        # Per-layer SSD store; backend depends on whether a Tutti storage is
        # attached.  The legacy IndexerBlockStore writes per-layer .bin files
        # via os.pread/pwrite; the Tutti backend routes I/O through Tutti's
        # GPU-direct NVMe path against a shared pre-reserved raw region.
        if tutti_storage is not None:
            self._stores: Dict[int, Any] = {
                lid: TuttiIndexerBlockStore(tutti_storage, lid) for lid in csa_layer_ids
            }
            logger.info(
                "IndexerSSDManager: using Tutti GPU-direct backend; layers=%d "
                "raw_region_path=%s slot_bytes=%d",
                len(csa_layer_ids),
                tutti_storage.raw_region_path,
                tutti_storage.slot_bytes,
            )
        else:
            self._stores = {
                lid: IndexerBlockStore(store_dir, lid, token_bytes, max_seq_len)
                for lid in csa_layer_ids
            }
            logger.info(
                "IndexerSSDManager: using file backend; layers=%d store_dir=%s",
                len(csa_layer_ids),
                store_dir,
            )

        # Async I/O
        self._executor = ThreadPoolExecutor(max_workers=io_workers)
        # A single worker preserves per-layer CPU collective ordering across
        # ranks while keeping proxy completion off the model-forward thread.
        self._proxy_executor = ThreadPoolExecutor(max_workers=1)
        # Do not run Tutti I/O on the ordered collective worker.  A speculative
        # read synchronizes its private CUDA I/O stream batch by batch; keeping
        # that work on _proxy_executor serializes later layers behind the full
        # NVMe walk even after their proxy kernels have completed.
        self._proxy_io_executor = ThreadPoolExecutor(
            max_workers=max(2, min(io_workers, 4))
        )
        # Direct LMCache seed persistence is latency-insensitive and may wait
        # for Tutti's idle-write window. Keep it off the general I/O executor:
        # HCA submissions and HBM-pool readiness must never queue behind the
        # 50 ms write-slack waits.
        self._persistence_executor = ThreadPoolExecutor(max_workers=1)
        # Native indexer-cache restore is deliberately ordered by transformer
        # layer. The first layer is submitted as Stage0 before model forward;
        # its existing consumption gate waits only if layers 0-1 did not hide
        # the read. Queue the complete ordered walk by default: the executor
        # still issues one layer at a time and releases Tutti's queue between
        # layers, while avoiding Python scheduling gaps at later layer gates.
        self._native_indexer_stream_executor = ThreadPoolExecutor(max_workers=1)
        self._native_indexer_cache_manager: Optional[Any] = None
        self._native_indexer_stream_request_id = ""
        self._native_indexer_stream_active = False
        self._native_indexer_stream_cleanup_failed = False
        self._native_indexer_stage0_layers = max(
            1,
            min(
                len(self._csa_layer_ids),
                _env_int("LMCACHE_NATIVE_INDEXER_STAGE0_LAYERS", 1),
            ),
        )
        self._native_indexer_window_layers = max(
            1,
            min(
                len(self._csa_layer_ids),
                _env_int(
                    "LMCACHE_NATIVE_INDEXER_WINDOW_LAYERS",
                    len(self._csa_layer_ids),
                ),
            ),
        )
        self._native_indexer_scheduled_layers: Set[int] = set()
        self._closed = False
        self._lock = threading.Lock()
        # pending async read results: layer_id → list of (token_id, Future[bytes])
        self._pending: Dict[int, List[Tuple[int, "Future[bytes]"]]] = {
            lid: [] for lid in csa_layer_ids
        }
        self._inflight_tokens: Dict[int, Set[int]] = {
            lid: set() for lid in csa_layer_ids
        }
        # Per-layer speculative topk from previous step: token IDs
        self._prev_topk: Dict[int, Optional[List[int]]] = {
            lid: None for lid in csa_layer_ids
        }
        self._last_proxy_topk: Dict[int, Optional[List[int]]] = {
            lid: None for lid in csa_layer_ids
        }
        self._last_proxy_blocks: Dict[int, Optional[List[int]]] = {
            lid: None for lid in csa_layer_ids
        }
        self._decoder_layers: Dict[int, Any] = {}

        # Future tracking for post-prefill async SSD eviction.
        # prepare_pool blocks on this before scoring if it's not yet done.
        self._ready_futures: Dict[int, Optional[Future[None]]] = {
            lid: None for lid in csa_layer_ids
        }
        self._ready_cuda_events: Dict[int, Optional[torch.cuda.Event]] = {
            lid: None for lid in csa_layer_ids
        }
        self._drain_futures: Dict[int, Optional[Future[None]]] = {
            lid: None for lid in csa_layer_ids
        }
        self._drain_cuda_events: Dict[int, Optional[torch.cuda.Event]] = {
            lid: None for lid in csa_layer_ids
        }
        self._proxy_futures: Dict[int, List[Future[None]]] = {
            lid: [] for lid in csa_layer_ids
        }
        self._expired_proxy_layers: Set[int] = set()
        self._proxy_block_budget = max(
            1,
            _env_int("LMCACHE_CSA_PREFETCH_BLOCK_BUDGET", 256),
        )
        self._cp_exchange_proxy_ids = (
            _env_int("LMCACHE_CSA_PREFETCH_CP_EXCHANGE_IDS", 1) != 0
        )
        # One reusable proxy stream per target layer. Different target layers
        # still overlap, while repeated requests avoid creating a fresh CUDA
        # stream for every fire. A single per-device stream is intentionally
        # not used: it serializes independent lookahead levels and previously
        # caused a large steady-state TTFT regression.
        self._proxy_streams: Dict[int, torch.cuda.Stream] = {}
        self._proxy_streams_lock = threading.Lock()
        self._proxy_cpu_selection_pool: Dict[int, List[torch.Tensor]] = {
            lid: [] for lid in csa_layer_ids
        }
        self._proxy_buffers_lock = threading.Lock()
        self._direct_seed_tail_buffers: Dict[int, List[Tuple[int, torch.Tensor]]] = {
            lid: [] for lid in csa_layer_ids
        }

        # Next sequential token ID for new decode-step tokens, per layer.
        # Set to seq_len by evict_after_prefill; incremented on each decode step.
        # Stays 0 until eviction runs (acts as "SSD uninitialized" guard).
        self._decode_cursor: Dict[int, int] = {lid: 0 for lid in csa_layer_ids}
        # Layers already fired for the current request.  The NVMe-resident
        # prefix is fixed when the CSA manager registers a request's chunks,
        # so one proxy prediction per (request, layer) covers it; re-firing
        # on every prefill chunk recomputes an O(prefix) scoring pass for a
        # prediction the resident bitmap then discards (measured: 38-68 ms
        # per fire at 24K prefix, ~50 s of pure recompute per request, and
        # the per-fire stream-private intermediates are what exhaust GPU
        # memory on the second request).  True-topK miss correction remains
        # the correctness net for any blocks the single fire missed.
        self._csa_fired_request_id: str = ""
        self._csa_fired_levels: Set[Tuple[int, int]] = set()
        self._prefetch_lookahead: Dict[int, int] = {lid: 0 for lid in csa_layer_ids}
        self._cp_proxy_fallback_logged: Set[Tuple[int, str]] = set()
        self._hca_fired_request_id: str = ""
        self._hca_fired_layers: Set[int] = set()

        # CSA layer index → position in csa_layer_ids list (for "next CSA layer" lookup)
        self._csa_pos: Dict[int, int] = {lid: i for i, lid in enumerate(csa_layer_ids)}
        self._debug_evict_logged: set[Tuple[str, int]] = set()
        self._debug_prepare_logged: set[int] = set()
        self._debug_insert_logged: set[int] = set()
        self._debug_topk_logged: set[int] = set()
        self._debug_fire_no_prev_logged: set[int] = set()
        self._debug_fire_active_logged: set[int] = set()
        self._debug_residual_proxy_attempt_logged: set[int] = set()
        self._debug_residual_proxy_skip_logged: set[Tuple[int, str]] = set()
        self._debug_residual_proxy_logged: set[int] = set()
        self._debug_attention_topk_logged: set[int] = set()
        self._debug_lmcache_seed_logged = False
        self._proxy_profile_limit = _env_int(
            "LMCACHE_INDEXER_RESIDUAL_PROFILE_LIMIT", 128
        )
        self._proxy_profile_seen = 0
        self._proxy_profile_hits = 0
        self._proxy_profile_total = 0
        self._attention_profile_seen = 0
        self._attention_profile_hits = 0
        self._attention_profile_total = 0
        self._timing_seen = 0
        self._nsys_seen_request_id = ""
        self._nsys_seen_requests = 0
        self._nsys_capture_active = False
        self._nsys_capture_complete = False
        logger.info("IndexerSSDManager: canonical L2 pipeline enabled")

    def _log_timing(self, event: str, layer_id: int, **fields: Any) -> None:
        """Emit one lightweight timing line when timing diagnostics are enabled."""
        if not _timing_enabled():
            return
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        if event == "seed_lmcache" and not _TIMING_SEED_VERBOSE:
            logger.debug(
                "IndexerSSDTiming: event=%s layer=%d %s",
                event,
                layer_id,
                detail,
            )
            return
        if self._timing_seen >= _TIMING_LIMIT:
            return
        self._timing_seen += 1
        logger.info(
            "IndexerSSDTiming: event=%s layer=%d %s",
            event,
            layer_id,
            detail,
        )

    def register_decoder_layer(self, layer_id: int, decoder_layer: Any) -> None:
        """Register a decoder layer for optional residual proxy prefetch."""
        self._decoder_layers[layer_id] = decoder_layer

    def configure_prefetch_lookahead(self, by_target_layer: Dict[int, int]) -> None:
        """Configure demand-only or two-layer prefetch per target CSA.

        Args:
            by_target_layer: Mapping from target transformer layer id to a
                lookahead of zero or two layers. Zero disables proxy
                prediction while preserving true-indexer miss correction.

        Raises:
            ValueError: If a target is unknown or lookahead is not zero or two.
        """
        configured = dict(self._prefetch_lookahead)
        for layer_id, lookahead in by_target_layer.items():
            target = int(layer_id)
            level = int(lookahead)
            if target not in self._csa_pos:
                raise ValueError(f"unknown CSA target layer {target}")
            if level not in (0, 2):
                raise ValueError("CSA prefetch lookahead must be 0 or 2")
            configured[target] = level
        self._prefetch_lookahead = configured

    def csa_attention_kv_prefetch_attached(self) -> bool:
        """Return True when CSA attention-KV prefetch is attached."""
        return getattr(self, "_csa_attention_kv_manager", None) is not None

    def start_nsys_capture_for_layer(self, layer_id: int, request_id: str) -> None:
        """Start one profiler-gated capture at the first configured L2 target.

        Args:
            layer_id: CSA target whose proxy prediction is being submitted.
            request_id: Active request identifier used to skip warmup requests.
        """
        if not _env_flag("LMCACHE_NSYS_CAPTURE"):
            return
        if self._nsys_capture_active or self._nsys_capture_complete:
            return
        enabled_targets = sorted(
            target
            for target, lookahead in self._prefetch_lookahead.items()
            if lookahead == 2
        )
        if not enabled_targets or int(layer_id) != enabled_targets[0]:
            return
        if request_id != self._nsys_seen_request_id:
            self._nsys_seen_request_id = request_id
            self._nsys_seen_requests += 1
        skip = max(0, _env_int("LMCACHE_NSYS_CAPTURE_SKIP_REQUESTS", 0))
        if self._nsys_seen_requests <= skip:
            return
        try:
            torch.cuda.profiler.start()
            self._nsys_capture_active = True
            logger.info(
                "IndexerSSDManager: NSYS_CAPTURE start target_layer=%d "
                "request_index=%d",
                layer_id,
                self._nsys_seen_requests,
            )
        except Exception:
            self._nsys_capture_complete = True
            logger.exception("IndexerSSDManager: NSYS_CAPTURE start failed")

    def finish_nsys_capture_for_layer(self, layer_id: int) -> None:
        """Stop a gated Nsight Systems capture after the final CSA target.

        Args:
            layer_id: CSA target whose true-indexer correction just completed.

        Notes:
            This is a no-op unless ``LMCACHE_NSYS_CAPTURE`` is enabled and a
            capture was started by the first configured L2 target.
        """
        if not self._nsys_capture_active:
            return
        enabled_targets = sorted(
            target
            for target, lookahead in self._prefetch_lookahead.items()
            if lookahead == 2
        )
        if not enabled_targets or int(layer_id) != enabled_targets[-1]:
            return
        try:
            torch.cuda.profiler.stop()
            logger.info(
                "IndexerSSDManager: NSYS_CAPTURE stop target_layer=%d",
                layer_id,
            )
        except Exception:
            logger.exception("IndexerSSDManager: NSYS_CAPTURE stop failed")
        finally:
            self._nsys_capture_active = False
            self._nsys_capture_complete = True

    def layer_initialized(self, layer_id: int) -> bool:
        """Return True when *layer_id* has an initialized SSD/pool cursor."""
        return self._decode_cursor.get(layer_id, 0) > 0

    def has_layer_rows(self, layer_id: int, seq_len: int) -> bool:
        """Return True when *layer_id* has direct-seeded rows through seq_len.

        Args:
            layer_id: CSA layer index.
            seq_len: Required compressed sequence length.

        Returns:
            True if the layer's SSD/pool cursor covers at least seq_len rows.
        """
        return self._decode_cursor.get(layer_id, 0) >= seq_len

    def seed_range_from_lmcache_group(
        self,
        layer_ids: Sequence[int],
        memory_tensor: torch.Tensor,
        start: int,
        end: int,
        total_logical_tokens: Optional[int] = None,
    ) -> int:
        """Seed CSA SSD/HBM pools directly from one LMCache retrieve chunk.

        Args:
            layer_ids: Transformer CSA layer ids in the same order as the
                LMCache CSA-indexer group layer axis.
            memory_tensor: LMCache CSA group tensor with shape
                ``[kv_size, num_layers, rows, hidden_dim]``.
            start: Logical token start offset of this LMCache chunk.
            end: Logical token end offset of this LMCache chunk.
            total_logical_tokens: Total logical tokens being restored for the
                request. When provided, only the final tail window needed to
                seed the HBM pool is initialized synchronously.

        Returns:
            Number of CSA layers whose SSD/HBM state was updated.

        Raises:
            ValueError: If the tensor shape or token range is incompatible
                with the registered CSA layers.
        """
        if not layer_ids or memory_tensor.numel() == 0 or end <= start:
            return 0
        if memory_tensor.ndim != 4:
            raise ValueError(
                "CSA LMCache group tensor must have shape "
                "[kv_size, num_layers, rows, hidden_dim], got "
                f"{tuple(memory_tensor.shape)}"
            )
        if start % 4 != 0 or end % 4 != 0:
            raise ValueError(
                f"CSA LMCache range [{start}, {end}) is not aligned to compress_ratio 4"
            )
        if len(layer_ids) > int(memory_tensor.shape[1]):
            raise ValueError(
                f"CSA LMCache group has {memory_tensor.shape[1]} layers but "
                f"{len(layer_ids)} layer ids were provided"
            )

        seq_start = start // 4
        expected_rows = (end - start) // 4
        rows_to_seed = min(expected_rows, int(memory_tensor.shape[2]))
        if rows_to_seed <= 0:
            return 0
        total_rows = (total_logical_tokens or end) // 4
        if total_rows <= 0:
            return 0
        tail_rows = self._pool_size
        tail_rows = min(tail_rows, total_rows)
        tail_start = total_rows - tail_rows
        seq_end = seq_start + rows_to_seed
        overlap_start = max(seq_start, tail_start)
        overlap_end = min(seq_end, total_rows)
        if overlap_start >= overlap_end:
            return 0
        local_start = overlap_start - seq_start
        local_rows = overlap_end - overlap_start

        timing = _timing_enabled()
        t0 = time.perf_counter() if timing else 0.0
        t_cpu0 = time.perf_counter() if timing else 0.0
        tensor_cpu = memory_tensor.detach()
        if tensor_cpu.device.type != "cpu":
            tensor_cpu = tensor_cpu.to(device="cpu", non_blocking=False)
        cpu_ms = (time.perf_counter() - t_cpu0) * 1000.0 if timing else 0.0
        t_contig0 = time.perf_counter() if timing else 0.0
        tensor_cpu = tensor_cpu.contiguous()
        contiguous_ms = (time.perf_counter() - t_contig0) * 1000.0 if timing else 0.0

        seeded = 0
        reshape_ms = 0.0
        write_ms = 0.0
        load_ms = 0.0
        state_ms = 0.0
        for group_layer_idx, layer_id in enumerate(layer_ids):
            layer_id = int(layer_id)
            if layer_id not in self._pools or layer_id not in self._stores:
                continue
            t_reshape0 = time.perf_counter() if timing else 0.0
            layer_rows = tensor_cpu[
                :,
                group_layer_idx,
                :rows_to_seed,
                :,
            ].permute(1, 0, 2)
            token_bytes = (
                layer_rows[local_start : local_start + local_rows]
                .contiguous()
                .view(local_rows, -1)
            )
            if timing:
                reshape_ms += (time.perf_counter() - t_reshape0) * 1000.0
            if int(token_bytes.shape[1]) != self._token_bytes:
                raise ValueError(
                    f"CSA layer {layer_id} row size mismatch: LMCache row has "
                    f"{int(token_bytes.shape[1])} bytes, manager expects "
                    f"{self._token_bytes} bytes"
                )

            t_load0 = time.perf_counter() if timing else 0.0
            load_ids: List[int] = []
            tail_complete = False
            finished_tail = overlap_end >= total_rows
            with self._lock:
                if overlap_start == tail_start:
                    self._direct_seed_tail_buffers[layer_id] = []
                self._direct_seed_tail_buffers[layer_id].append(
                    (overlap_start, token_bytes.clone())
                )
                tail_chunks = list(self._direct_seed_tail_buffers[layer_id])
                if finished_tail:
                    self._direct_seed_tail_buffers[layer_id] = []

            if finished_tail and tail_chunks:
                tail_chunks.sort(key=lambda item: item[0])
                chunks = []
                expected_start = tail_start
                for chunk_start, chunk_bytes in tail_chunks:
                    if chunk_start > expected_start:
                        break
                    skip = max(0, expected_start - chunk_start)
                    if skip < int(chunk_bytes.shape[0]):
                        chunks.append(chunk_bytes[skip:])
                        expected_start = chunk_start + int(chunk_bytes.shape[0])
                    if expected_start >= total_rows:
                        break
                if chunks and expected_start >= total_rows:
                    tail_complete = True
                    pool = self._pools[layer_id]
                    pool.reset()
                    load_ids = list(range(tail_start, total_rows))
                    tail_bytes = torch.cat(chunks, dim=0)[: len(load_ids)]
                    # Publish HBM readiness before persistence. Tutti writes
                    # intentionally wait for a request-idle window; making the
                    # true indexer join that write produced one 1.67 s stall
                    # followed by a 50 ms bubble per layer. The complete tail
                    # is resident and protected in HBM, so persistence can run
                    # independently without delaying this request's consumer.
                    pool.load_tokens(load_ids, tail_bytes.contiguous())
                    pool.protect_only(load_ids)
                    t_write0 = time.perf_counter() if timing else 0.0
                    payload = tail_bytes.contiguous().numpy().tobytes()
                    if self._tutti_storage is not None:
                        store = self._stores[layer_id]

                        def _persist_tail(
                            target_store: Any = store,
                            target_layer: int = layer_id,
                            target_start: int = tail_start,
                            target_payload: bytes = payload,
                        ) -> None:
                            try:
                                target_store.write_tokens_contiguous(
                                    target_start,
                                    target_payload,
                                )
                            except Exception:
                                logger.exception(
                                    "IndexerSSDManager: deferred LMCache tail "
                                    "persistence failed for layer %d rows=[%d,%d)",
                                    target_layer,
                                    target_start,
                                    target_start
                                    + len(target_payload) // self._token_bytes,
                                )

                        self._persistence_executor.submit(_persist_tail)
                    else:
                        self._stores[layer_id].write_tokens_contiguous(
                            tail_start,
                            payload,
                        )
                    if timing:
                        write_ms += (time.perf_counter() - t_write0) * 1000.0
            if timing:
                load_ms += (time.perf_counter() - t_load0) * 1000.0

            t_state0 = time.perf_counter() if timing else 0.0
            if finished_tail and tail_complete:
                with self._lock:
                    self._decode_cursor[layer_id] = max(
                        self._decode_cursor.get(layer_id, 0),
                        total_rows,
                    )
                    self._prev_topk[layer_id] = list(load_ids[:1024])
            if timing:
                state_ms += (time.perf_counter() - t_state0) * 1000.0
            seeded += 1

        if seeded > 0 and not self._debug_lmcache_seed_logged:
            self._debug_lmcache_seed_logged = True
            logger.info(
                "IndexerSSDManager: direct LMCache seed enabled for %d CSA "
                "layers; first tail rows=[%d,%d) total_rows=%d",
                seeded,
                overlap_start,
                overlap_end,
                total_rows,
            )
        if seeded and timing:
            self._log_timing(
                "seed_lmcache",
                -1,
                total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
                cpu_ms=f"{cpu_ms:.3f}",
                contiguous_ms=f"{contiguous_ms:.3f}",
                reshape_ms=f"{reshape_ms:.3f}",
                write_ms=f"{write_ms:.3f}",
                load_ms=f"{load_ms:.3f}",
                state_ms=f"{state_ms:.3f}",
                layers=seeded,
                start_row=seq_start,
                rows=rows_to_seed,
            )
        return seeded

    def submit_seed_range_from_lmcache_group(
        self,
        layer_ids: Sequence[int],
        memory_tensor: torch.Tensor,
        start: int,
        end: int,
        total_logical_tokens: Optional[int] = None,
    ) -> Optional[Future[int]]:
        """Snapshot a retrieve staging view and seed it after the callback.

        Tutti invokes the GPU connector while holding its loader I/O lock. A
        direct call to :meth:`seed_range_from_lmcache_group` can therefore
        deadlock when the indexer store is backed by the same loader: the seed
        writes raw extents and tries to acquire that lock again. This method
        synchronously copies only the final HBM-tail overlap to owned CPU
        memory, then performs all store writes and state updates on the
        manager's executor after the retrieve callback can return.

        Args:
            layer_ids: Transformer CSA layer ids in LMCache group order.
            memory_tensor: Retrieve staging tensor. It may become invalid as
                soon as the current streaming callback returns.
            start: Logical token start offset of the retrieve chunk.
            end: Logical token end offset of the retrieve chunk.
            total_logical_tokens: Total logical tokens restored for the
                request. Defaults to ``end``.

        Returns:
            A future for the number of seeded layers, or ``None`` when this
            chunk does not overlap the configured direct-seed tail.

        Raises:
            ValueError: If the tensor shape or token range is incompatible.
        """
        if not layer_ids or memory_tensor.numel() == 0 or end <= start:
            return None
        if memory_tensor.ndim != 4:
            raise ValueError(
                "CSA LMCache group tensor must have shape "
                "[kv_size, num_layers, rows, hidden_dim], got "
                f"{tuple(memory_tensor.shape)}"
            )
        if start % 4 != 0 or end % 4 != 0:
            raise ValueError(
                f"CSA LMCache range [{start}, {end}) is not aligned to compress_ratio 4"
            )
        layer_ids_tuple = tuple(int(layer_id) for layer_id in layer_ids)
        if len(layer_ids_tuple) > int(memory_tensor.shape[1]):
            raise ValueError(
                f"CSA LMCache group has {memory_tensor.shape[1]} layers but "
                f"{len(layer_ids_tuple)} layer ids were provided"
            )

        seq_start = start // 4
        rows_in_chunk = min((end - start) // 4, int(memory_tensor.shape[2]))
        total_rows = int(total_logical_tokens or end) // 4
        if rows_in_chunk <= 0 or total_rows <= 0:
            return None
        tail_rows = self._pool_size
        tail_start = total_rows - min(tail_rows, total_rows)
        overlap_start = max(seq_start, tail_start)
        overlap_end = min(seq_start + rows_in_chunk, total_rows)
        if overlap_start >= overlap_end:
            return None

        local_start = overlap_start - seq_start
        local_end = local_start + (overlap_end - overlap_start)
        staging_slice = memory_tensor[
            :,
            : len(layer_ids_tuple),
            local_start:local_end,
            :,
        ].detach()
        # ``copy=True`` is required for CPU-backed staging too: contiguous()
        # alone may return an alias whose storage is recycled after callback.
        snapshot = staging_slice.to(
            device="cpu",
            non_blocking=False,
            copy=True,
        ).contiguous()

        def _seed_after_previous(
            previous_futures: tuple[Future[Any], ...],
        ) -> int:
            for previous_future in previous_futures:
                previous_future.result(timeout=self._prefill_ready_timeout_s)
            for layer_id in layer_ids_tuple:
                self._wait_for_ready_cuda_event(layer_id)
            try:
                seeded = self.seed_range_from_lmcache_group(
                    layer_ids_tuple,
                    snapshot,
                    overlap_start * 4,
                    overlap_end * 4,
                    total_logical_tokens=total_rows * 4,
                )
                self._record_ready_cuda_event(layer_ids_tuple)
                return seeded
            except OSError as exc:
                # Preserve the old synchronous path's best-effort contract:
                # an unavailable indexer store must not invalidate an LMCache
                # hit. The official indexer can repopulate its pool later.
                logger.warning(
                    "IndexerSSDManager: deferred direct seed skipped because "
                    "the indexer store is unavailable: %s",
                    exc,
                )
                return 0

        with self._lock:
            previous_futures = tuple(
                dict.fromkeys(
                    previous
                    for layer_id in layer_ids_tuple
                    if (previous := self._ready_futures.get(layer_id)) is not None
                )
            )
            # Capture and publish the dependency atomically so concurrent
            # retrieve callbacks cannot fork two writers from the same tail.
            future = self._executor.submit(
                _seed_after_previous,
                previous_futures,
            )
            for layer_id in layer_ids_tuple:
                if layer_id in self._ready_futures:
                    self._ready_futures[layer_id] = future
        return future

    def wait_for_seed(self, layer_id: int) -> bool:
        """Wait for the latest deferred LMCache seed of one CSA layer.

        Args:
            layer_id: Target CSA layer id.

        Returns:
            ``True`` when no seed is pending or the latest seed completed.

        Raises:
            TimeoutError: If the configured prefill-ready timeout expires.
            Exception: Propagates a deferred seed failure to the consumer.
        """
        with self._lock:
            future = self._ready_futures.get(layer_id)
        if future is None:
            return True
        future.result(timeout=self._prefill_ready_timeout_s)
        self._wait_for_ready_cuda_event(layer_id)
        with self._lock:
            if self._ready_futures.get(layer_id) is future:
                self._ready_futures[layer_id] = None
        return True

    # ------------------------------------------------------------------
    # Called from DeepseekV4DecoderLayer.forward, before self.ffn()
    # ------------------------------------------------------------------

    def fire_async_for_layer(
        self,
        layer_id: int,
        residual_f: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prefetch_level: int = 2,
    ) -> None:
        """Schedule the canonical two-layer CSA prediction.

        The decoder calls this method once, from the FFN-entry hook exactly two
        transformer layers before the target CSA layer. GPU proxy scoring is
        submitted on a side stream and completion/I/O dispatch is handled by
        the proxy executor, so the source layer never waits for SSD I/O.

        Args:
            layer_id: Target CSA transformer layer.
            residual_f: Source layer post-attention residual.
            positions: Query positions aligned with the residual.
            llama_4_scaling: Optional compatibility input for DSv4 variants.
            prefetch_level: Must be two. No L1 prediction path exists.

        Raises:
            ValueError: If the caller requests a non-L2 prediction.
        """
        if prefetch_level != 2:
            raise ValueError("prefetch_level must be 2")
        configured_lookahead = self._prefetch_lookahead.get(int(layer_id), 0)
        if configured_lookahead != 2:
            self._log_timing(
                "prefill_fire_async_skip",
                layer_id,
                reason="prediction_disabled",
                configured_lookahead=configured_lookahead,
            )
            return

        manager = getattr(self, "_csa_attention_kv_manager", None)
        if manager is None or residual_f is None or positions is None:
            self._log_residual_proxy_skip(layer_id, "canonical_l2_inputs_missing")
            return
        residual_f, positions, aligned_rows = _flatten_proxy_state_for_positions(
            residual_f,
            positions,
        )
        if aligned_rows <= 1:
            # The production L2 pipeline is for cache-hit prefill/recompute.
            # Decode continues through the official indexer path.
            self._log_residual_proxy_skip(layer_id, "canonical_l2_requires_prefill")
            return
        if self._decode_cursor.get(layer_id, 0) <= 0:
            self._warm_cold_proxy_kernels(layer_id, residual_f, positions)
            self._log_residual_proxy_skip(layer_id, "ssd_uninitialized")
            return
        # L2 proxy scoring reads the target layer's native indexer K cache two
        # transformer layers before true scoring. The compact stream therefore
        # needs its correctness gate here, not only at target consumption.
        # With the two-layer rolling window this is normally event-only; any
        # exposed wait is an explicit, measurable prefetch bubble.
        native_wait_start = time.perf_counter()
        if not self.wait_for_native_indexer_layer(int(layer_id)):
            self._log_residual_proxy_skip(layer_id, "native_indexer_stream_late")
            return
        native_wait_ms = (time.perf_counter() - native_wait_start) * 1000.0
        self._log_timing(
            "native_indexer_proxy_gate",
            int(layer_id),
            wait_ms=f"{native_wait_ms:.3f}",
        )
        request_id = str(getattr(manager, "active_request_id", ""))
        self.start_nsys_capture_for_layer(layer_id, request_id)
        fire_key = (int(layer_id), 2)
        with self._lock:
            if request_id != self._csa_fired_request_id:
                self._csa_fired_request_id = request_id
                self._csa_fired_levels.clear()
                self._expired_proxy_layers.clear()
                for target_layer_id in self._csa_layer_ids:
                    self._last_proxy_blocks[target_layer_id] = None
            if request_id and fire_key in self._csa_fired_levels:
                return
            # Reserve before launching so duplicate decoder hooks cannot race
            # into two predictions for the same request and target.
            if request_id:
                self._csa_fired_levels.add(fire_key)

        t0 = time.perf_counter()
        submitted = self._submit_csa_attention_kv_proxy_async(
            layer_id,
            residual_f,
            positions,
            llama_4_scaling,
            fire_start=t0,
            prefetch_level=2,
        )
        if not submitted:
            with self._lock:
                self._csa_fired_levels.discard(fire_key)
            self._log_residual_proxy_skip(layer_id, "canonical_l2_submit_failed")
            return
        self._log_timing(
            "prefill_fire_async",
            layer_id,
            total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
            rows=aligned_rows,
            mode="canonical_l2_async_submit",
            prefetch_level=2,
        )

    def attach_csa_attention_kv_manager(self, manager: Optional[Any]) -> None:
        """Attach (or detach) the CSA attention KV prefetcher.

        When attached, every predicted top-K computed in
        :meth:`fire_async_for_layer` is also forwarded to the attention KV
        prefetcher so it can issue Tutti reads in parallel with the
        indexer-cache reads.  Passing ``None`` detaches the prefetcher.

        Args:
            manager: A
                :class:`lmcache.v1.csa_attention_kv_prefetch_manager.CSAAttentionKVPrefetchManager`
                instance, or ``None``.
        """
        previous = getattr(self, "_csa_attention_kv_manager", None)
        if previous is not None and previous is not manager:
            clear_waiter = getattr(previous, "set_prediction_waiter", None)
            if callable(clear_waiter):
                clear_waiter(None)
        self._csa_attention_kv_manager = manager
        if manager is not None:
            set_waiter = getattr(manager, "set_prediction_waiter", None)
            if callable(set_waiter):
                set_waiter(self.wait_for_csa_attention_kv_prediction)

    def attach_native_indexer_cache_loader(
        self,
        tutti_loader: Any,
        layer_tensors: Dict[int, torch.Tensor],
    ) -> None:
        """Attach the compact layer-major loader for native indexer caches.

        The native vLLM indexer cache remains the true-scoring source. This
        loader only changes how its cached prefix is restored: compact
        per-layer sidecars are streamed directly into the existing tensors
        instead of retrieving the padded LMCache group as one large object.

        Args:
            tutti_loader: Active Tutti direct-NVMe loader for this rank.
            layer_tensors: Native indexer K-cache tensors keyed by transformer
                layer id. Every configured CSA layer must be present.

        Raises:
            ValueError: If a configured layer or tensor is missing.
        """
        if self._native_indexer_cache_manager is not None:
            return
        missing = sorted(set(self._csa_layer_ids) - set(layer_tensors))
        if missing:
            raise ValueError(f"native indexer cache tensors missing layers {missing}")
        from lmcache.v1.csa_attention_kv_prefetch_manager import (
            CSAAttentionKVPrefetchManager,
        )

        loader = CSAAttentionKVPrefetchManager(
            tutti_loader=tutti_loader,
            csa_layer_ids=self._csa_layer_ids,
            compressed_block_size=DEEPGEMM_PAGED_BLOCK_SIZE,
            token_bytes=self._token_bytes,
        )
        for layer_id in self._csa_layer_ids:
            loader.register_layer(int(layer_id), layer_tensors[int(layer_id)])
        self._native_indexer_cache_manager = loader
        logger.info(
            "IndexerSSDManager: compact native indexer loader attached "
            "layers=%d token_bytes=%d stage0_layers=%d",
            len(self._csa_layer_ids),
            self._token_bytes,
            self._native_indexer_stage0_layers,
        )

    def register_native_indexer_stream(
        self,
        req_id: str,
        chunks_by_layer: Dict[int, List[Any]],
        *,
        shared_raw_lba_cache: Optional[dict[str, list[Any]]] = None,
    ) -> bool:
        """Register and start an ordered compact native-indexer restore.

        The first layer forms Stage0 and is submitted before model forward.
        All layers continue on one ordered worker and are gated immediately
        before their transformer layer consumes them. The gate preserves
        correctness if an unusually slow read exhausts the overlap window.

        Args:
            req_id: Active request identifier.
            chunks_by_layer: Complete compact layer-major read plan.
            shared_raw_lba_cache: Immutable union of all request-sidecar
                extents shared with the CSA/HCA stream manager.

        Returns:
            ``True`` only when every configured CSA layer has a non-empty,
            safely registered plan and Stage0 submitted.
        """
        loader = self._native_indexer_cache_manager
        if loader is None:
            return False
        request_id = str(req_id)
        if (
            self._native_indexer_stream_active
            and request_id == self._native_indexer_stream_request_id
        ):
            if self.native_indexer_stream_matches(request_id, chunks_by_layer):
                return True
            logger.warning(
                "IndexerSSDManager: repeated native indexer plan changed "
                "request=%s; refusing stale stream reuse",
                request_id,
            )
            if not self.deactivate_native_indexer_stream():
                raise RuntimeError("stale native indexer stream could not drain")
            return False
        if not self.deactivate_native_indexer_stream():
            raise RuntimeError("previous native indexer stream could not drain")
        missing = [
            layer_id
            for layer_id in self._csa_layer_ids
            if not chunks_by_layer.get(int(layer_id))
        ]
        if missing:
            logger.warning(
                "IndexerSSDManager: compact native indexer plan incomplete "
                "request=%s missing_layers=%s; retaining synchronous restore",
                request_id,
                missing,
            )
            return False
        # Publish provisional ownership before registration so a partial
        # loader mutation can be rolled back through the normal deactivation
        # path if plan compilation raises.
        self._native_indexer_stream_request_id = request_id
        self._native_indexer_stream_active = True
        try:
            loader.register_request_chunks(
                request_id,
                chunks_by_layer,
                start_profile_capture=False,
                shared_raw_lba_cache=shared_raw_lba_cache,
            )
        except Exception:
            if not self.deactivate_native_indexer_stream():
                logger.error(
                    "IndexerSSDManager: partial native indexer registration "
                    "did not drain request=%s",
                    request_id,
                )
            raise
        self._native_indexer_scheduled_layers.clear()
        covered_rows: Optional[int] = None
        for layer_id in self._csa_layer_ids:
            layer_chunks = chunks_by_layer[int(layer_id)]
            layer_rows = (
                int(layer_chunks[-1].end_compressed_block) * DEEPGEMM_PAGED_BLOCK_SIZE
            )
            if covered_rows is None:
                covered_rows = layer_rows
            elif covered_rows != layer_rows:
                self.deactivate_native_indexer_stream()
                raise RuntimeError(
                    "compact native indexer layers have inconsistent coverage"
                )
        assert covered_rows is not None and covered_rows > 0
        with self._lock:
            for layer_id in self._csa_layer_ids:
                self._decode_cursor[int(layer_id)] = covered_rows
        stage0 = self._csa_layer_ids[: self._native_indexer_stage0_layers]
        self._schedule_native_indexer_through(len(stage0) - 1)
        # Keep only a bounded number of post-Stage0 layers queued. Each CSA
        # consumption gate advances this window by one, leaving Tutti queue
        # opportunities for the nearer HCA and predicted CSA-KV reads.
        self._schedule_native_indexer_through(
            len(stage0) + self._native_indexer_window_layers - 1
        )
        if _env_flag("LMCACHE_TUTTI_PROFILE") or int(self._device.index or 0) == 0:
            logger.info(
                "IndexerSSDManager: compact native indexer stream started "
                "request=%s layers=%d stage0=%s window=%d",
                request_id,
                len(self._csa_layer_ids),
                list(stage0),
                self._native_indexer_window_layers,
            )
        return True

    def native_indexer_stream_active(self) -> bool:
        """Return whether the active request may skip full indexer restore."""
        return bool(self._native_indexer_stream_active)

    def native_indexer_stream_available(self) -> bool:
        """Return whether the compact native-indexer consumer is attached."""
        return bool(
            self._native_indexer_cache_manager is not None
            and not self._native_indexer_stream_cleanup_failed
        )

    def native_indexer_stream_matches(
        self,
        req_id: str,
        chunks_by_layer: Dict[int, List[Any]],
    ) -> bool:
        """Return whether the active compact stream uses this exact plan."""
        loader = self._native_indexer_cache_manager
        if (
            loader is None
            or not self._native_indexer_stream_active
            or str(req_id) != self._native_indexer_stream_request_id
        ):
            return False
        matches = getattr(loader, "request_chunks_match", None)
        return bool(callable(matches) and matches(str(req_id), chunks_by_layer))

    def deactivate_native_indexer_stream(self, timeout_s: float = 30.0) -> bool:
        """Drain and disable the active compact native-indexer stream.

        Args:
            timeout_s: Total time allowed for old-request I/O to finish.

        Returns:
            ``True`` when no previous stream can still write indexer caches.
        """
        # The first cache hit has an attached loader but no active request.
        # Treat that state as already drained; calling the loader's strict
        # request deactivation API here reports False for "nothing active"
        # and incorrectly rejects the first stream registration.
        if not self._native_indexer_stream_active:
            return not self._native_indexer_stream_cleanup_failed
        self._native_indexer_stream_active = False
        self._native_indexer_stream_cleanup_failed = True
        loader = self._native_indexer_cache_manager
        deactivate = getattr(loader, "deactivate_request", None)
        if loader is None:
            drained = True
        elif not callable(deactivate):
            drained = False
        else:
            try:
                drained = bool(deactivate(timeout_s=timeout_s))
            except Exception:
                logger.exception(
                    "IndexerSSDManager: native indexer deactivation failed"
                )
                drained = False
        self._native_indexer_stream_request_id = ""
        self._native_indexer_scheduled_layers.clear()
        with self._lock:
            for layer_id in self._csa_layer_ids:
                self._decode_cursor[int(layer_id)] = 0
        self._native_indexer_stream_cleanup_failed = not drained
        return drained

    def wait_for_native_indexer_layer(self, layer_id: int) -> bool:
        """Wait until one streamed native indexer layer is safe to consume.

        Args:
            layer_id: Transformer layer about to run its true indexer.

        Returns:
            ``True`` when no compact stream is active or the layer landed;
            ``False`` if its I/O did not complete within the safety timeout.
        """
        if not self._native_indexer_stream_active:
            return True
        loader = self._native_indexer_cache_manager
        if loader is None or int(layer_id) not in self._csa_pos:
            return True
        position = self._csa_pos[int(layer_id)]
        # A missing tracked future means "not scheduled", not "already
        # complete".  Explicitly queue through the demanded layer before
        # consulting the loader.  This prevents a late proxy gate (for
        # example layer 26) from skipping its own restore and then flooding
        # every intervening layer into the single I/O worker at once.
        self._schedule_native_indexer_through(position)
        completed = bool(loader.wait_for_layer(int(layer_id), timeout_s=30.0))
        if completed:
            self._schedule_native_indexer_through(
                position + self._native_indexer_window_layers
            )
        return completed

    def _schedule_native_indexer_through(self, position: int) -> None:
        """Queue unscheduled compact indexer layers through one position."""
        loader = self._native_indexer_cache_manager
        if loader is None or not self._native_indexer_stream_active:
            return
        stop = min(len(self._csa_layer_ids), max(0, int(position) + 1))
        request_token = loader.active_request_token
        for layer_id in self._csa_layer_ids[:stop]:
            target = int(layer_id)
            if target in self._native_indexer_scheduled_layers:
                continue
            future = self._native_indexer_stream_executor.submit(
                loader.fire_deterministic_layer,
                target,
                label="indexer_native_stream",
                request_token=request_token,
            )
            self._native_indexer_scheduled_layers.add(target)
            loader.track_layer_submission(
                target,
                future,
                request_token=request_token,
            )

    def fire_residual_prefetch_for_layer(
        self,
        layer_id: int,
        residual_f: torch.Tensor,
        positions: torch.Tensor,
        *,
        lookahead: int,
    ) -> None:
        """Fire the one canonical L2 residual prediction for a target CSA.

        Args:
            layer_id: Target CSA transformer layer id.
            residual_f: Source layer's post-attention residual.
            positions: Query positions aligned with ``residual_f``.
            lookahead: Source-to-target distance. It must be two.

        Raises:
            ValueError: If ``lookahead`` is not two.
        """
        if lookahead != 2:
            raise ValueError("lookahead must be 2")
        configured_lookahead = self._prefetch_lookahead.get(int(layer_id), 0)
        if configured_lookahead != lookahead:
            self._log_timing(
                "residual_prefetch_skip",
                layer_id,
                reason=(
                    "prediction_disabled"
                    if configured_lookahead == 0
                    else "non_configured_lookahead"
                ),
                configured_lookahead=configured_lookahead,
                lookahead=lookahead,
            )
            return
        self.wait_for_seed(layer_id)
        self._log_timing(
            "l2_submit",
            layer_id,
            rows=_proxy_num_rows(residual_f),
        )
        with csa_pipeline_nvtx.range(
            CsaNvtxEvent.L2_PROXY,
            layer_id=layer_id - 2,
            target_layer_id=layer_id,
            request_id=str(
                getattr(self._csa_attention_kv_manager, "active_request_id", "")
            ),
            attributes={"phase": "submit"},
        ):
            self.fire_async_for_layer(
                layer_id,
                residual_f=residual_f,
                positions=positions,
                prefetch_level=2,
            )

    def layer_fired_for_active_request(self, layer_id: int) -> bool:
        """Return whether deterministic HCA I/O was scheduled this request."""
        manager = getattr(self, "_csa_attention_kv_manager", None)
        request_id = str(getattr(manager, "active_request_id", ""))
        with self._lock:
            return (
                request_id == self._hca_fired_request_id
                and int(layer_id) in self._hca_fired_layers
            )

    def fire_async_for_layers(
        self,
        layer_ids: Sequence[int],
        positions: Optional[torch.Tensor] = None,
    ) -> None:
        """Schedule deterministic HCA attention-KV reads in the FFN window.

        Args:
            layer_ids: Upcoming registered HCA transformer layer ids.
            positions: Unused compatibility argument for decoder hooks.
        """
        del positions
        manager = getattr(self, "_csa_attention_kv_manager", None)
        fire_layer = getattr(manager, "fire_deterministic_layer", None)
        track = getattr(manager, "track_layer_submission", None)
        if not callable(fire_layer):
            return
        request_id = str(getattr(manager, "active_request_id", ""))
        request_token = getattr(manager, "active_request_token", (request_id, -1))
        for raw_layer_id in layer_ids:
            layer_id = int(raw_layer_id)
            with self._lock:
                if request_id != self._hca_fired_request_id:
                    self._hca_fired_request_id = request_id
                    self._hca_fired_layers.clear()
                if layer_id in self._hca_fired_layers:
                    continue
                self._hca_fired_layers.add(layer_id)

            def _fire(
                target_layer_id: int = layer_id,
                token: Tuple[str, int] = request_token,
            ) -> None:
                fire_layer(target_layer_id, request_token=token)

            future = self._executor.submit(_fire)
            if callable(track):
                track(layer_id, future, request_token=request_token)
            self._log_timing(
                "hca_deterministic_submit",
                layer_id,
                request_id=request_id,
            )

    def drain_for_layer(
        self,
        layer_id: int,
        blocking: bool = True,
    ) -> bool:
        """Delegate an HCA consumption gate to the unified KV manager.

        Args:
            layer_id: Target HCA transformer layer id.
            blocking: When ``False``, only report whether a scheduled future
                is already complete; the production attention gate passes
                ``True`` to preserve correctness.

        Returns:
            ``True`` when the layer is ready for attention.
        """
        manager = getattr(self, "_csa_attention_kv_manager", None)
        wait_for_layer = getattr(manager, "wait_for_layer", None)
        if not callable(wait_for_layer):
            return True
        if not blocking:
            ready = getattr(manager, "layer_submission_ready", None)
            return bool(ready(int(layer_id))) if callable(ready) else False
        return bool(wait_for_layer(int(layer_id)))

    def prepare_layer_async(self, layer_id: int) -> None:
        """Submit CSA ready/drain work before the target layer consumes it.

        This lets the caller place the wait and HBM-pool insertion in the MoE
        window instead of paying it at ``SparseAttnIndexer.prepare_pool()``.

        Args:
            layer_id: CSA layer whose ready future and pending reads should be
                progressed in the background.
        """
        if layer_id not in self._pools:
            return
        with self._lock:
            ready_fut = self._ready_futures.get(layer_id)
            previous_drain = self._drain_futures.get(layer_id)
            has_pending = bool(self._pending.get(layer_id))
            if previous_drain is not None and not previous_drain.done():
                return
            if ready_fut is None and not has_pending and previous_drain is None:
                return

        def _prepare() -> None:
            if ready_fut is not None:
                ready_fut.result(timeout=self._prefill_ready_timeout_s)
                self._wait_for_ready_cuda_event(layer_id)
                with self._lock:
                    if self._ready_futures.get(layer_id) is ready_fut:
                        self._ready_futures[layer_id] = None
            self._drain(layer_id)
            self._record_drain_cuda_event(layer_id)

        drain_future = self._executor.submit(_prepare)
        with self._lock:
            self._drain_futures[layer_id] = drain_future

        def _clear_done(fut: Future[None]) -> None:
            try:
                fut.result()
            except Exception:
                logger.exception(
                    "IndexerSSDManager: async prepare failed for layer %d",
                    layer_id,
                )
            with self._lock:
                if self._drain_futures.get(layer_id) is fut:
                    self._drain_futures[layer_id] = None

        drain_future.add_done_callback(_clear_done)

    def _submit_csa_attention_kv_proxy_async(
        self,
        layer_id: int,
        residual_f: torch.Tensor,
        positions: torch.Tensor,
        llama_4_scaling: Optional[torch.Tensor],
        *,
        fire_start: float,
        prefetch_level: int,
    ) -> bool:
        """Submit one all-row CSA proxy and finish it asynchronously."""
        manager = getattr(self, "_csa_attention_kv_manager", None)
        if manager is None:
            logger.info(
                "IndexerSSDManager: CSA attention-KV proxy unavailable "
                "layer=%d manager=%s",
                layer_id,
                manager is not None,
            )
            return False
        if not residual_f.is_cuda:
            self._log_residual_proxy_skip(
                layer_id,
                "csa_attention_kv_proxy_requires_cuda_residual",
            )
            return False
        decoder_layer = self._decoder_layers.get(layer_id)
        if decoder_layer is None or not self._is_deepseek_v4_layer(decoder_layer):
            logger.info(
                "IndexerSSDManager: CSA attention-KV proxy missing decoder "
                "layer=%d registered=%s is_v4=%s residual_shape=%s "
                "positions_shape=%s",
                layer_id,
                decoder_layer is not None,
                (
                    self._is_deepseek_v4_layer(decoder_layer)
                    if decoder_layer is not None
                    else False
                ),
                tuple(residual_f.shape),
                tuple(positions.shape),
            )
            return False
        device_index = (
            int(residual_f.device.index)
            if residual_f.device.index is not None
            else int(torch.cuda.current_device())
        )
        batches: list[tuple[torch.Tensor, Any, Any, Any, dict[str, Any]]] = []
        selected_rows = 0
        with torch.cuda.device(device_index):
            # Reuse one stream per target layer. This preserves concurrency
            # across lookahead targets without paying stream creation on every
            # request. Repeated work for the same target remains correctly
            # ordered on that target's stream.
            proxy_stream = self._proxy_stream_for(layer_id)
            current_stream = torch.cuda.current_stream()
            proxy_stream.wait_stream(current_stream)
            try:
                with torch.no_grad():
                    with torch.cuda.stream(proxy_stream):
                        residual_f.record_stream(proxy_stream)
                        if positions.is_cuda:
                            positions.record_stream(proxy_stream)
                        proxy_rows, proxy_positions, selected_rows = _align_proxy_rows(
                            residual_f,
                            positions,
                        )
                        # Keep the target indexer call at the original chunk
                        # granularity.  DeepGEMM's prefill metadata is built
                        # for the current forward chunk, so slicing rows here
                        # makes seq_len disagree with cu_seq_len_k_start.
                        timing_enabled = _timing_enabled()
                        proxy_start = (
                            torch.cuda.Event(enable_timing=True)
                            if timing_enabled
                            else None
                        )
                        proxy_done = (
                            torch.cuda.Event(enable_timing=True)
                            if timing_enabled
                            else None
                        )
                        copy_done = torch.cuda.Event(enable_timing=timing_enabled)
                        phase_events: dict[str, Any] = {}
                        if proxy_start is not None:
                            proxy_start.record(proxy_stream)
                        topk_buf, num_rows, cp_context = self._residual_proxy_topk_gpu(
                            layer_id,
                            decoder_layer,
                            proxy_rows,
                            proxy_positions,
                            llama_4_scaling,
                            enable_prefill_cp=True,
                            timing_events=phase_events,
                        )
                        if proxy_done is not None:
                            proxy_done.record(proxy_stream)
                        selected_topk = topk_buf[:num_rows]
                        # Reduce the rank-local query shard to a bounded block
                        # list, then exchange only those fixed-size IDs. Enqueue
                        # this tiny collective here, while every model rank is
                        # executing the same decoder hook, rather than from a
                        # background CPU worker. Background Gloo calls can enter
                        # in different orders across ranks and deadlock at the
                        # target gate. The GPU exchange preserves the semantic
                        # union without the prefix-sized bitmap AllReduce.
                        cursor = self._decode_cursor.get(layer_id, 0)
                        num_blocks = (
                            cursor + DEEPGEMM_PAGED_BLOCK_SIZE - 1
                        ) // DEEPGEMM_PAGED_BLOCK_SIZE
                        if num_blocks > 0:
                            selected_blocks = _select_rank_local_proxy_blocks(
                                selected_topk,
                                cursor,
                                num_blocks,
                                self._proxy_block_budget,
                            ).to(torch.int32)
                            if cp_context is not None and self._cp_exchange_proxy_ids:
                                import torch.distributed as dist
                                from vllm.distributed import get_tp_group

                                cp_world_size = int(cp_context[1])
                                selected_blocks = selected_blocks.contiguous()
                                exchanged_blocks = torch.empty(
                                    int(selected_blocks.numel()) * cp_world_size,
                                    dtype=selected_blocks.dtype,
                                    device=selected_blocks.device,
                                )
                                dist.all_gather_into_tensor(
                                    exchanged_blocks,
                                    selected_blocks,
                                    group=get_tp_group().device_group,
                                )
                                selected_blocks = exchanged_blocks
                            # Tutti's indexed-read ABI is widened only after
                            # the bounded IDs have reached the CPU worker.
                            selected_cpu = self._acquire_proxy_cpu_selection(
                                layer_id,
                                int(selected_blocks.numel()),
                            )
                            selected_cpu.copy_(selected_blocks, non_blocking=True)
                            copy_done.record(proxy_stream)
                            batches.append(
                                (
                                    selected_cpu,
                                    proxy_start,
                                    proxy_done,
                                    copy_done,
                                    phase_events,
                                )
                            )
            except Exception as exc:
                logger.warning(
                    "IndexerSSDManager: CSA attention-KV proxy "
                    "submit failed for layer %d residual_shape=%s "
                    "positions_shape=%s selected_rows=%d cursor=%d "
                    "exc=%s",
                    layer_id,
                    tuple(residual_f.shape),
                    tuple(positions.shape),
                    selected_rows,
                    self._decode_cursor.get(layer_id, 0),
                    repr(exc),
                    exc_info=True,
                )
                return False

        if not batches:
            logger.info(
                "IndexerSSDManager: CSA attention-KV proxy produced no "
                "topk layer=%d residual_shape=%s positions_shape=%s "
                "selected_rows=%d cursor=%d",
                layer_id,
                tuple(residual_f.shape),
                tuple(positions.shape),
                selected_rows,
                self._decode_cursor.get(layer_id, 0),
            )
            self._log_residual_proxy_skip(layer_id, "csa_attention_kv_empty_topk")
            return False

        cursor = self._decode_cursor.get(layer_id, 0)
        request_id = str(getattr(manager, "active_request_id", ""))
        request_token = getattr(manager, "active_request_token", (request_id, -1))
        # Tutti's indexed API currently validates CQ status synchronously on
        # the host. Reuse the existing I/O executor for that host-only bridge;
        # proxy scoring itself is already enqueued on its CUDA side stream.
        future = self._proxy_executor.submit(
            self._finish_csa_attention_kv_proxy,
            layer_id,
            batches,
            cursor,
            selected_rows,
            fire_start,
            prefetch_level,
            request_id,
            request_token,
        )
        with self._lock:
            self._proxy_futures[layer_id].append(future)

        def _clear_done(done_future: Future[None]) -> None:
            try:
                if not done_future.cancelled():
                    done_future.result()
            except Exception:
                logger.exception(
                    "IndexerSSDManager: CSA attention-KV proxy failed for layer %d",
                    layer_id,
                )
            with self._lock:
                futures = self._proxy_futures.get(layer_id)
                if futures is not None and done_future in futures:
                    futures.remove(done_future)

        future.add_done_callback(_clear_done)
        return True

    def _finish_csa_attention_kv_proxy(
        self,
        layer_id: int,
        batches: list[tuple[torch.Tensor, Any, Any, Any, dict[str, Any]]],
        cursor: int,
        selected_rows: int,
        fire_start: float,
        prefetch_level: int,
        request_id: str,
        request_token: Tuple[str, int],
    ) -> None:
        """Dispatch the bounded union of all rank-local query predictions."""
        manager = getattr(self, "_csa_attention_kv_manager", None)
        if manager is None:
            for (
                selected_cpu,
                _proxy_start,
                _proxy_done,
                _copy_done,
                _phase_events,
            ) in batches:
                self._release_proxy_cpu_selection(layer_id, selected_cpu)
            return
        t0 = time.perf_counter()
        selected_batches: list[torch.Tensor] = []
        event_wait_ms = 0.0
        proxy_gpu_ms = 0.0
        d2h_gpu_ms = 0.0
        proxy_total_gpu_ms = 0.0
        phase_gpu_ms: dict[str, float] = {
            "hc_pre": 0.0,
            "indexer_inputs": 0.0,
            "q_quant": 0.0,
            "cp_score": 0.0,
        }
        for (
            selected_cpu,
            proxy_start,
            proxy_done,
            copy_done,
            phase_events,
        ) in batches:
            t_wait0 = time.perf_counter()
            copy_done.synchronize()
            event_wait_ms += (time.perf_counter() - t_wait0) * 1000.0
            if proxy_start is not None and proxy_done is not None:
                try:
                    proxy_gpu_ms += float(proxy_start.elapsed_time(proxy_done))
                    d2h_gpu_ms += float(proxy_done.elapsed_time(copy_done))
                    proxy_total_gpu_ms += float(proxy_start.elapsed_time(copy_done))
                    phase_order = (
                        ("hc_pre", proxy_start, phase_events.get("hidden_done")),
                        (
                            "indexer_inputs",
                            phase_events.get("hidden_done"),
                            phase_events.get("inputs_done"),
                        ),
                        (
                            "q_quant",
                            phase_events.get("inputs_done"),
                            phase_events.get("q_quant_done"),
                        ),
                        (
                            "cp_score",
                            phase_events.get("q_quant_done"),
                            proxy_done,
                        ),
                    )
                    for phase_name, phase_start, phase_done in phase_order:
                        if phase_start is not None and phase_done is not None:
                            phase_gpu_ms[phase_name] += float(
                                phase_start.elapsed_time(phase_done)
                            )
                except RuntimeError:
                    pass
            selected_batches.append(selected_cpu[selected_cpu >= 0].to(torch.int64))
            self._release_proxy_cpu_selection(layer_id, selected_cpu)
        block_ids_tensor = torch.unique(
            torch.cat(selected_batches)
            if selected_batches
            else torch.empty(0, dtype=torch.int64),
            sorted=True,
        )
        dispatched_blocks = int(block_ids_tensor.numel())
        dispatched_batches = 0
        with self._lock:
            expired = (
                int(layer_id) in self._expired_proxy_layers
                or request_id != self._csa_fired_request_id
            )
            self._last_proxy_blocks[int(layer_id)] = (
                [int(block_id) for block_id in block_ids_tensor.tolist()]
                if not expired
                else None
            )
        if dispatched_blocks and not expired:
            self._log_timing(
                "dispatch_csa_attention_kv_predicted",
                layer_id,
                predicted_tokens=(dispatched_blocks * DEEPGEMM_PAGED_BLOCK_SIZE),
                blocks=dispatched_blocks,
                mode="single_layer_batch",
            )
            io_future = self._proxy_io_executor.submit(
                manager.fire_predicted_reads,
                layer_id,
                block_ids_tensor,
                prefetch_level=prefetch_level,
                request_token=request_token,
            )
            with self._lock:
                self._proxy_futures[layer_id].append(io_future)

            def _clear_io_done(done_future: Future[None]) -> None:
                try:
                    if not done_future.cancelled():
                        done_future.result()
                except Exception:
                    logger.exception(
                        "IndexerSSDManager: CSA attention-KV predicted I/O "
                        "failed for layer %d",
                        layer_id,
                    )
                with self._lock:
                    futures = self._proxy_futures.get(layer_id)
                    if futures is not None and done_future in futures:
                        futures.remove(done_future)

            io_future.add_done_callback(_clear_io_done)
            dispatched_batches = 1
        proxy_ms = (time.perf_counter() - t0) * 1000.0
        self._log_timing(
            "prefill_fire_async",
            layer_id,
            total_ms=f"{(time.perf_counter() - fire_start) * 1000.0:.3f}",
            proxy_ms=f"{proxy_ms:.3f}",
            filter_ms="0.000",
            submit_ms="0.000",
            rows=selected_rows,
            batches=len(batches),
            dispatched_batches=dispatched_batches,
            blocks=dispatched_blocks,
            event_wait_ms=f"{event_wait_ms:.3f}",
            proxy_gpu_ms=f"{proxy_gpu_ms:.3f}",
            id_exchange_ms="0.000",
            d2h_gpu_ms=f"{d2h_gpu_ms:.3f}",
            proxy_total_gpu_ms=f"{proxy_total_gpu_ms:.3f}",
            hc_pre_gpu_ms=f"{phase_gpu_ms['hc_pre']:.3f}",
            indexer_inputs_gpu_ms=f"{phase_gpu_ms['indexer_inputs']:.3f}",
            q_quant_gpu_ms=f"{phase_gpu_ms['q_quant']:.3f}",
            cp_score_gpu_ms=f"{phase_gpu_ms['cp_score']:.3f}",
            mode="canonical_l2_async_finish",
            prefetch_level=prefetch_level,
        )

    def _residual_proxy_topk_gpu(
        self,
        layer_id: int,
        decoder_layer: Any,
        residual_f: torch.Tensor,
        positions: torch.Tensor,
        llama_4_scaling: Optional[torch.Tensor],
        *,
        enable_prefill_cp: bool = False,
        timing_events: Optional[dict[str, Any]] = None,
        metadata_query_row_start: Optional[int] = None,
        runtime_info: Optional[dict[str, int]] = None,
    ) -> tuple[torch.Tensor, int, Optional[Tuple[int, int, int, int]]]:
        """Run the V4 proxy indexer and return the GPU top-k buffer."""
        del llama_4_scaling
        residual_f, positions, selected_rows = _align_proxy_rows(
            residual_f,
            positions,
        )
        proxy_hidden = self._v4_attention_proxy_hidden(decoder_layer, residual_f)
        if timing_events is not None:
            timing_events["hidden_done"] = torch.cuda.Event(enable_timing=True)
            timing_events["hidden_done"].record()
        qr, weights, indexer, rotary_emb = self._v4_indexer_inputs(
            decoder_layer,
            proxy_hidden,
        )
        if timing_events is not None:
            timing_events["inputs_done"] = torch.cuda.Event(enable_timing=True)
            timing_events["inputs_done"].record()
        cp_context = self._prefill_cp_context(layer_id) if enable_prefill_cp else None
        topk_buf, cp_used = self._v4_proxy_topk_direct(
            layer_id,
            proxy_hidden,
            qr,
            weights,
            positions,
            indexer,
            rotary_emb,
            prefill_cp_context=cp_context,
            timing_events=timing_events,
            metadata_query_row_start=metadata_query_row_start,
            runtime_info=runtime_info,
        )
        if cp_used and cp_context is not None:
            rank, world_size, interleave, oversubscribe = cp_context
            self._log_timing(
                "prefill_cp_proxy",
                layer_id,
                rows=selected_rows,
                rank=rank,
                world_size=world_size,
                interleave=interleave,
                oversubscribe=oversubscribe,
            )
        return topk_buf, selected_rows, cp_context if cp_used else None

    @staticmethod
    def _is_deepseek_v4_layer(decoder_layer: Any) -> bool:
        """Return True when *decoder_layer* exposes the DeepSeek V4 HC API."""
        required = (
            "hc_pre",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_attn_base",
            "attn_norm",
            "attn",
        )
        return all(hasattr(decoder_layer, name) for name in required)

    @staticmethod
    def _v4_attention_proxy_hidden(
        decoder_layer: Any,
        residual_f: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``attn_norm(HC_pre(residual_f, hc_attn_*))`` for V4."""
        attn_norm = decoder_layer.attn_norm
        norm_weight = getattr(attn_norm, "weight", None)
        norm_eps = getattr(attn_norm, "variance_epsilon", None)
        if norm_eps is None:
            norm_eps = getattr(decoder_layer, "rms_norm_eps", 1e-6)
        if norm_weight is not None:
            try:
                proxy_hidden, _, _ = decoder_layer.hc_pre(
                    residual_f,
                    decoder_layer.hc_attn_fn,
                    decoder_layer.hc_attn_scale,
                    decoder_layer.hc_attn_base,
                    norm_weight=norm_weight.data,
                    norm_eps=float(norm_eps),
                )
                return proxy_hidden
            except TypeError:
                pass
        proxy_hidden, _, _ = decoder_layer.hc_pre(
            residual_f,
            decoder_layer.hc_attn_fn,
            decoder_layer.hc_attn_scale,
            decoder_layer.hc_attn_base,
        )
        return attn_norm(proxy_hidden)

    @staticmethod
    def _v4_indexer_inputs(
        decoder_layer: Any,
        proxy_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
        """Build read-only V4 proxy scoring inputs for a proxy hidden state."""
        attn = decoder_layer.attn
        indexer = getattr(attn, "indexer", None)
        if indexer is None:
            mla_attn = getattr(attn, "mla_attn", None)
            indexer = getattr(mla_attn, "indexer", None)
        if indexer is None:
            raise RuntimeError("DeepSeek V4 CSA layer has no indexer")

        mla_attn = getattr(attn, "mla_attn", attn)
        fused_wqa_wkv = getattr(mla_attn, "fused_wqa_wkv", None)
        weights_proj = getattr(indexer, "weights_proj", None)
        q_norm = getattr(attn, "q_norm", getattr(mla_attn, "q_norm", None))
        if not callable(fused_wqa_wkv):
            raise RuntimeError("DeepSeek V4 MLA fused_wqa_wkv is missing")
        if not callable(weights_proj):
            raise RuntimeError("DeepSeek V4 CSA weights_proj is missing")
        if not callable(q_norm):
            raise RuntimeError("DeepSeek V4 MLA q_norm is missing")

        qr_kv, _ = fused_wqa_wkv(proxy_hidden)
        q_lora_rank = getattr(attn, "q_lora_rank", None)
        if q_lora_rank is None:
            q_lora_rank = mla_attn.q_lora_rank
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            head_dim = mla_attn.head_dim
        qr, _ = qr_kv.split([int(q_lora_rank), int(head_dim)], dim=-1)
        qr = q_norm(qr)
        indexer_weights, _ = weights_proj(proxy_hidden)
        rotary_emb = getattr(attn, "rotary_emb", None)
        if rotary_emb is None:
            rotary_emb = mla_attn.rotary_emb
        return qr, indexer_weights, indexer, rotary_emb

    def _v4_proxy_topk_direct(
        self,
        layer_id: int,
        proxy_hidden: torch.Tensor,
        qr: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        indexer: Any,
        rotary_emb: Any,
        *,
        prefill_cp_context: Optional[Tuple[int, int, int, int]] = None,
        timing_events: Optional[dict[str, Any]] = None,
        metadata_query_row_start: Optional[int] = None,
        runtime_info: Optional[dict[str, int]] = None,
    ) -> tuple[torch.Tensor, bool]:
        """Compute V4 proxy top-K without running ``DeepseekV4Indexer.forward``.

        The official V4 indexer forward first runs the target layer compressor,
        which writes compressor state and indexer K-cache rows.  Proxy prefetch
        must be read-only: it should score the target layer's existing indexer
        K cache with the proxy query, then return a private top-K buffer.
        """
        indexer_op = getattr(indexer, "indexer_op", None)
        if indexer_op is None:
            self._log_residual_proxy_skip(layer_id, "indexer_op_missing")
            raise RuntimeError("DeepSeek V4 CSA indexer_op is missing")

        reference_topk = getattr(indexer_op, "topk_indices_buffer", None)
        if not isinstance(reference_topk, torch.Tensor):
            self._log_residual_proxy_skip(layer_id, "topk_buffer_missing")
            raise RuntimeError("DeepSeek V4 CSA topk buffer is missing")

        try:
            from vllm.v1.attention.ops.deepseek_v4_ops import (
                fused_indexer_q_rope_quant,
            )
        except ImportError as exc:
            self._log_residual_proxy_skip(layer_id, "fused_indexer_q_missing")
            raise RuntimeError(
                "DeepSeek V4 fused_indexer_q_rope_quant is missing"
            ) from exc

        wq_b = getattr(indexer, "wq_b", None)
        if not callable(wq_b):
            self._log_residual_proxy_skip(layer_id, "indexer_wq_b_missing")
            raise RuntimeError("DeepSeek V4 CSA indexer wq_b is missing")

        q, _ = wq_b(qr)
        n_head = int(indexer.n_head)
        head_dim = int(indexer.head_dim)
        q = q.view(-1, n_head, head_dim)
        q_quant, score_weights = fused_indexer_q_rope_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            indexer_weights,
            float(indexer.softmax_scale),
            n_head**-0.5,
            use_fp4=bool(getattr(indexer, "use_fp4_kv", False)),
        )
        if timing_events is not None:
            timing_events["q_quant_done"] = torch.cuda.Event(enable_timing=True)
            timing_events["q_quant_done"].record()
        topk_buf = torch.empty(
            (int(proxy_hidden.shape[0]), int(reference_topk.shape[1])),
            dtype=reference_topk.dtype,
            device=reference_topk.device,
        )
        old_topk = getattr(indexer_op, "topk_indices_buffer", None)
        has_skip_insert_attr = hasattr(indexer_op, "skip_k_cache_insert")
        old_skip_insert = (
            indexer_op.skip_k_cache_insert if has_skip_insert_attr else None
        )
        cp_used = False
        try:
            indexer_op.topk_indices_buffer = topk_buf
            if has_skip_insert_attr:
                indexer_op.skip_k_cache_insert = True
            if prefill_cp_context is not None:
                rank, world_size, interleave_size, oversubscribe = prefill_cp_context
                try:
                    from lmcache.v1.csa_prefill_cp_scorer import (
                        score_prefill_proxy_rank_local,
                    )

                    score_prefill_proxy_rank_local(
                        indexer_op,
                        proxy_hidden,
                        q_quant,
                        score_weights,
                        topk_buf,
                        rank=rank,
                        world_size=world_size,
                        interleave_size=interleave_size,
                        oversubscribe=oversubscribe,
                        metadata_query_row_start=metadata_query_row_start,
                        runtime_info=runtime_info,
                    )
                    cp_used = True
                except Exception as exc:
                    self._log_prefill_cp_fallback(layer_id, repr(exc))
                    if metadata_query_row_start is not None:
                        raise
            if not cp_used:
                self._call_v4_indexer_op_read_only(
                    layer_id,
                    indexer_op,
                    proxy_hidden,
                    q_quant,
                    score_weights,
                    has_skip_insert_attr=has_skip_insert_attr,
                )
        finally:
            indexer_op.topk_indices_buffer = old_topk
            if has_skip_insert_attr:
                indexer_op.skip_k_cache_insert = old_skip_insert
        return topk_buf, cp_used

    def _prefill_cp_context(
        self,
        layer_id: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return TP rank data for prefill-only speculative CP scoring."""
        requested = _env_int("LMCACHE_CSA_PREFETCH_CP_SIZE", 1)
        if requested <= 1:
            return None
        interleave = _env_int("LMCACHE_CSA_PREFETCH_CP_INTERLEAVE", 64)
        oversubscribe = _env_int("LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE", 1)
        try:
            from vllm.distributed import get_tp_group

            tp_group = get_tp_group()
            native_world = int(tp_group.world_size)
            native_rank = int(tp_group.rank_in_group)
        except Exception as exc:
            self._log_prefill_cp_fallback(layer_id, f"tp_group={exc!r}")
            return None
        if (
            native_world != requested
            or native_rank < 0
            or native_rank >= native_world
            or interleave <= 0
            or oversubscribe <= 0
        ):
            reason = (
                f"requested={requested},native_world={native_world},"
                f"rank={native_rank},interleave={interleave},"
                f"oversubscribe={oversubscribe}"
            )
            key = (layer_id, reason)
            if key not in self._cp_proxy_fallback_logged:
                self._cp_proxy_fallback_logged.add(key)
                logger.warning(
                    "IndexerSSDManager: prefill-only CP proxy disabled for "
                    "layer %d (%s); "
                    "falling back to the exact global proxy",
                    layer_id,
                    reason,
                )
            return None
        return native_rank, native_world, interleave, oversubscribe

    def _log_prefill_cp_fallback(self, layer_id: int, reason: str) -> None:
        """Log one prefill-only CP fallback reason per target layer."""
        key = (int(layer_id), str(reason))
        if key in self._cp_proxy_fallback_logged:
            return
        self._cp_proxy_fallback_logged.add(key)
        logger.warning(
            "IndexerSSDManager: prefill-only CP proxy failed for layer %d "
            "(%s); using the original full-K proxy",
            layer_id,
            reason,
        )

    def _call_v4_indexer_op_read_only(
        self,
        layer_id: int,
        indexer_op: Any,
        proxy_hidden: torch.Tensor,
        q_quant: torch.Tensor,
        score_weights: torch.Tensor,
        *,
        has_skip_insert_attr: bool,
    ) -> None:
        """Call the V4 sparse indexer op without inserting proxy K rows."""
        forward = getattr(
            indexer_op,
            "_lmcache_csa_attention_kv_original_forward",
            None,
        )
        if not callable(forward):
            forward = getattr(indexer_op, "forward", None)
        if not callable(forward):
            self._log_residual_proxy_skip(layer_id, "indexer_op_forward_missing")
            raise RuntimeError("DeepSeek V4 CSA indexer_op forward is missing")
        try:
            signature = inspect.signature(forward)
        except (AttributeError, TypeError, ValueError):
            signature = None
        supports_skip_kw = False
        if signature is not None:
            kwargs: dict[str, Any] = {}
            for name in (
                "skip_k_cache_insert",
                "skip_kv_cache_insert",
                "skip_cache_insert",
            ):
                if name in signature.parameters:
                    kwargs[name] = True
                    supports_skip_kw = True
                    break
            try:
                if not has_skip_insert_attr and not supports_skip_kw:
                    self._log_residual_proxy_skip(
                        layer_id,
                        "indexer_op_read_only_unsupported",
                    )
                    raise RuntimeError(
                        "DeepSeek V4 CSA indexer_op cannot run read-only proxy"
                    )
                forward(
                    proxy_hidden,
                    q_quant,
                    None,
                    score_weights,
                    **kwargs,
                )
                return
            except TypeError:
                if kwargs:
                    self._log_residual_proxy_skip(
                        layer_id,
                        "indexer_op_skip_kw_unsupported",
                    )
                else:
                    raise
        if not has_skip_insert_attr:
            self._log_residual_proxy_skip(
                layer_id,
                "indexer_op_read_only_unsupported",
            )
            raise RuntimeError("DeepSeek V4 CSA indexer_op cannot run read-only proxy")
        forward(proxy_hidden, q_quant, None, score_weights)

    def _log_residual_proxy_skip(self, layer_id: int, reason: str) -> None:
        """Log each residual-proxy skip reason once per layer."""
        key = (layer_id, reason)
        if key in self._debug_residual_proxy_skip_logged:
            return
        self._debug_residual_proxy_skip_logged.add(key)
        logger.info(
            "IndexerSSDManager: residual_proxy_skip layer %d reason=%s",
            layer_id,
            reason,
        )

    # ------------------------------------------------------------------
    # Called from SparseAttnIndexer, before scoring
    # ------------------------------------------------------------------

    def prepare_pool(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Drain pending async reads into the HBM pool for *layer_id*.

        Blocks if a post-prefill SSD eviction future is still running so the
        pool is populated before scoring begins.

        Returns:
            pool_tensor: ``[num_blocks, 64, token_bytes]`` uint8 on CUDA.
            block_table: ``[1, num_blocks]`` int32 on CUDA.
        """
        t0 = time.perf_counter()
        proxy_wait_ms = 0.0
        ready_ms = 0.0
        drain_ms = 0.0
        sync_ms = 0.0
        waited_ready = False
        # Residual-proxy work targets attention KV and is purely speculative.
        # The official indexer must run immediately instead of joining it here;
        # the patched target-layer gate checks readiness only after true scoring.
        with self._lock:
            ready_fut = self._ready_futures.get(layer_id)
            drain_fut = self._drain_futures.get(layer_id)
        if ready_fut is not None:
            if not ready_fut.done():
                waited_ready = True
                logger.info(
                    "IndexerSSDManager: waiting for post-prefill SSD init layer %d",
                    layer_id,
                )
            t_ready0 = time.perf_counter()
            ready_fut.result(timeout=self._prefill_ready_timeout_s)
            ready_ms = (time.perf_counter() - t_ready0) * 1000.0
            self._wait_for_ready_cuda_event(layer_id)
            with self._lock:
                if self._ready_futures.get(layer_id) is ready_fut:
                    self._ready_futures[layer_id] = None
        t_drain0 = time.perf_counter()
        if drain_fut is not None:
            drain_fut.result(timeout=self._prefill_ready_timeout_s)
            with self._lock:
                if self._drain_futures.get(layer_id) is drain_fut:
                    self._drain_futures[layer_id] = None
        self._wait_for_drain_cuda_event(layer_id)
        self._drain(layer_id)
        drain_ms = (time.perf_counter() - t_drain0) * 1000.0
        pool = self._pools[layer_id]
        if layer_id not in self._debug_prepare_logged:
            self._debug_prepare_logged.add(layer_id)
            valid_slots = pool.valid_slot_count()
            logger.info(
                "IndexerSSDManager: prepare_pool layer %d valid_slots=%d",
                layer_id,
                valid_slots,
            )
        self._log_timing(
            "prepare_pool",
            layer_id,
            total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
            proxy_wait_ms=f"{proxy_wait_ms:.3f}",
            ready_ms=f"{ready_ms:.3f}",
            drain_ms=f"{drain_ms:.3f}",
            sync_ms=f"{sync_ms:.3f}",
            waited_ready=int(waited_ready),
            valid_slots=pool.valid_slot_count(),
        )
        return pool.pool_tensor, pool.block_table

    def wait_for_csa_attention_kv_prediction(self, layer_id: int) -> bool:
        """Join any prediction that could not be cancelled at the target gate.

        The cache-hit prefill path fires the target CSA layer's HC-proxy top-K
        from the previous decoder layer's FFN/MoE window. Work that is still
        queued at the target is cancelled so true-topK correction can take
        over. A task that has already entered the loader cannot be cancelled;
        it must finish before miss filtering observes the resident bitmap.
        Otherwise the target can skip a block merely because it is pending and
        run attention before that block has landed.

        Args:
            layer_id: CSA layer whose async proxy prediction should be joined.

        Returns:
            ``True`` when every prediction completed, or ``False`` when at
            least one queued prediction was cancelled.

        Raises:
            RuntimeError: If running prediction work fails or does not finish
                within the configured target-gate timeout.
        """
        gate_start = time.perf_counter()
        with self._lock:
            futures = tuple(self._proxy_futures.get(int(layer_id), ()))
            prediction_submitted = bool(futures) or bool(
                self._last_proxy_blocks.get(int(layer_id))
            )
            ready = prediction_submitted and all(future.done() for future in futures)
            if not ready:
                self._expired_proxy_layers.add(int(layer_id))
        if not prediction_submitted:
            if _profile_accuracy_enabled():
                logger.info(
                    "IndexerSSDManager: prediction_target_gate layer %d "
                    "submitted=0 was_ready=0 cancelled=0 running=0 wait_ms=0.000",
                    layer_id,
                )
            return False
        if ready and not futures:
            if _profile_accuracy_enabled():
                logger.info(
                    "IndexerSSDManager: prediction_target_gate layer %d "
                    "submitted=1 was_ready=1 cancelled=0 running=0 wait_ms=0.000",
                    layer_id,
                )
            return True

        cancelled = False
        running: list[Future[None]] = []
        for future in futures:
            if future.done():
                continue
            if future.cancel():
                cancelled = True
            else:
                running.append(future)
        if cancelled:
            logger.debug(
                "IndexerSSDManager: cancelled queued CSA prediction for layer %d",
                layer_id,
            )
        timeout_s = max(
            0.0,
            float(getattr(self, "_prediction_gate_timeout_s", 5.0)),
        )
        deadline = time.monotonic() + timeout_s
        completed = [future for future in futures if not future.cancelled()]
        for future in completed:
            try:
                future.result(timeout=max(0.0, deadline - time.monotonic()))
            except CancelledError:
                cancelled = True
            except TimeoutError as exc:
                raise RuntimeError(
                    f"CSA prediction for layer {layer_id} exceeded the "
                    f"{timeout_s:.1f}s target-gate timeout"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"CSA prediction for layer {layer_id} failed at target gate"
                ) from exc
        if _profile_accuracy_enabled():
            logger.info(
                "IndexerSSDManager: prediction_target_gate layer %d "
                "submitted=1 was_ready=%d cancelled=%d running=%d wait_ms=%.3f",
                layer_id,
                int(ready),
                int(cancelled),
                len(running),
                (time.perf_counter() - gate_start) * 1000.0,
            )
        return not cancelled

    def pool_ids_for_layer(self, layer_id: int) -> torch.Tensor:
        """Return pool_ids tensor [pool_size] int64 for *layer_id*.

        Used to translate pool-slot topk indices back to global token IDs.
        """
        return self._pools[layer_id].pool_ids

    def pool_slot_bytes(self, layer_id: int, slot: int) -> torch.Tensor:
        """Return one HBM pool slot as interleaved ``value+scale`` bytes."""
        return self._pools[layer_id].read_slot(slot)

    # ------------------------------------------------------------------
    # Called from SparseAttnIndexer, after scoring
    # ------------------------------------------------------------------

    def translate_pool_slots(
        self, layer_id: int, topk_pool_slots: torch.Tensor
    ) -> torch.Tensor:
        """Translate pool-slot top-k indices to global token IDs.

        Args:
            layer_id: CSA layer that just ran scoring.
            topk_pool_slots: 1-D int32 tensor of pool slot indices (on CUDA).

        Returns:
            Tensor with the same shape as ``topk_pool_slots`` containing global
            token IDs. Empty pool slots are translated to ``-1``.
        """
        pool = self._pools[layer_id]
        flat_slots = topk_pool_slots.to(device=pool.pool_ids.device, dtype=torch.long)
        valid = (flat_slots >= 0) & (flat_slots < pool.pool_ids.numel())
        translated = torch.full_like(flat_slots, -1, dtype=pool.pool_ids.dtype)
        if valid.any():
            translated[valid] = pool.pool_ids[flat_slots[valid]]
        return translated.reshape(topk_pool_slots.shape).to(
            device=topk_pool_slots.device
        )

    def record_topk(self, layer_id: int, topk_pool_slots: torch.Tensor) -> None:
        """Record top-k pool slots for use as next step's prediction.

        Args:
            layer_id: CSA layer that just ran scoring.
            topk_pool_slots: 1-D int32 tensor of pool slot indices.
        """
        global_topk = self.translate_pool_slots(layer_id, topk_pool_slots)
        self.record_global_topk(layer_id, global_topk)

    def record_global_topk(self, layer_id: int, token_ids_tensor: torch.Tensor) -> None:
        """Record global token IDs for use as next step's prediction.

        Args:
            layer_id: CSA layer that just ran scoring.
            token_ids_tensor: Tensor containing global token IDs.
        """
        token_ids = []
        for tid in token_ids_tensor.reshape(-1).cpu().tolist():
            tid_int = int(tid)
            if tid_int >= 0:
                token_ids.append(tid_int)
        self._prev_topk[layer_id] = token_ids
        self._record_residual_proxy_accuracy(layer_id, token_ids)
        if layer_id not in self._debug_topk_logged:
            self._debug_topk_logged.add(layer_id)
            logger.info(
                "IndexerSSDManager: record_global_topk layer %d count=%d",
                layer_id,
                len(token_ids),
            )

    def correct_true_topk(self, layer_id: int, token_ids_tensor: torch.Tensor) -> None:
        """Drain speculative reads and synchronously fill true-topK misses.

        Args:
            layer_id: CSA layer that just ran the true Lightning Indexer.
            token_ids_tensor: True global token IDs emitted by the official
                sparse indexer op.
        """
        t0 = time.perf_counter()
        collect_ms = 0.0
        drain_ms = 0.0
        miss_ms = 0.0
        read_ms = 0.0
        insert_ms = 0.0

        t_collect0 = time.perf_counter()
        token_ids: List[int] = []
        seen: Set[int] = set()
        for tid in token_ids_tensor.reshape(-1).detach().cpu().tolist():
            tid_int = int(tid)
            if tid_int < 0 or tid_int in seen:
                continue
            seen.add(tid_int)
            token_ids.append(tid_int)
        collect_ms = (time.perf_counter() - t_collect0) * 1000.0
        if not token_ids:
            self.record_global_topk(layer_id, token_ids_tensor)
            return

        cursor = self._decode_cursor.get(layer_id, 0)
        if cursor <= 0:
            self.record_global_topk(layer_id, token_ids_tensor)
            return

        t_drain0 = time.perf_counter()
        self._drain(layer_id)
        drain_ms = (time.perf_counter() - t_drain0) * 1000.0

        pool = self._pools[layer_id]
        store = self._stores[layer_id]

        t_miss0 = time.perf_counter()
        missing = [
            token_id
            for token_id in token_ids
            if 0 <= token_id < cursor and not pool.contains(token_id)
        ]
        miss_ms = (time.perf_counter() - t_miss0) * 1000.0

        for token_id in missing:
            try:
                t_read0 = time.perf_counter()
                data = store.read_token(token_id)
                read_ms += (time.perf_counter() - t_read0) * 1000.0
                if not data:
                    continue
                t_insert0 = time.perf_counter()
                pool.insert(token_id, data)
                insert_ms += (time.perf_counter() - t_insert0) * 1000.0
            except Exception as exc:
                logger.error(
                    "IndexerSSDManager: true-topK fallback read failed for "
                    "token %d layer %d: %r",
                    token_id,
                    layer_id,
                    exc,
                )

        pool.protect_only([token_id for token_id in token_ids if token_id < cursor])
        self.record_global_topk(layer_id, token_ids_tensor)
        self._log_timing(
            "correct_true_topk",
            layer_id,
            total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
            collect_ms=f"{collect_ms:.3f}",
            drain_ms=f"{drain_ms:.3f}",
            miss_ms=f"{miss_ms:.3f}",
            read_ms=f"{read_ms:.3f}",
            insert_ms=f"{insert_ms:.3f}",
            true=len(token_ids),
            missing=len(missing),
        )

    def record_csa_prediction_accuracy(
        self,
        layer_id: int,
        true_topk: torch.Tensor,
    ) -> None:
        """Compare the last CSA proxy prediction with the true indexer output.

        Args:
            layer_id: Transformer-side CSA layer id.
            true_topk: True Lightning Indexer output containing compressed
                entry ids. Values outside the initialized prefix are ignored.

        Notes:
            This diagnostic intentionally synchronizes the small block bitmap
            to the CPU. It is active only when
            ``LMCACHE_INDEXER_PROFILE_ACCURACY`` is enabled.
        """
        if not _profile_accuracy_enabled():
            return
        with self._lock:
            if self._attention_profile_seen >= self._proxy_profile_limit:
                return
            cursor = self._decode_cursor.get(int(layer_id), 0)
            predicted_blocks = set(self._last_proxy_blocks.get(int(layer_id)) or [])
        if cursor <= 0 or true_topk.numel() == 0:
            return

        entries = true_topk.detach().reshape(-1).to(dtype=torch.int64)
        valid = (entries >= 0) & (entries < cursor)
        num_blocks = (
            cursor + DEEPGEMM_PAGED_BLOCK_SIZE - 1
        ) // DEEPGEMM_PAGED_BLOCK_SIZE
        true_bitmap = torch.zeros(
            num_blocks,
            dtype=torch.bool,
            device=entries.device,
        )
        true_bitmap[entries[valid] // DEEPGEMM_PAGED_BLOCK_SIZE] = True
        true_blocks = set(
            true_bitmap.nonzero(as_tuple=False).reshape(-1).cpu().tolist()
        )
        if not true_blocks:
            return

        block_hits = len(true_blocks & predicted_blocks)
        block_misses = len(true_blocks - predicted_blocks)
        block_recall = float(block_hits) / float(len(true_blocks))
        weighted_block_hits, weighted_block_total = _weighted_predicted_block_hits(
            entries,
            valid,
            predicted_blocks,
            num_blocks,
        )
        weighted_block_recall = (
            float(weighted_block_hits) / float(weighted_block_total)
            if weighted_block_total
            else 0.0
        )
        with self._lock:
            if self._attention_profile_seen >= self._proxy_profile_limit:
                return
            self._attention_profile_seen += 1
            self._attention_profile_hits += block_hits
            self._attention_profile_total += len(true_blocks)
            sample = self._attention_profile_seen
            avg_block_recall = float(self._attention_profile_hits) / float(
                self._attention_profile_total
            )
        logger.info(
            "IndexerSSDManager: attention_true_topk_profile layer %d "
            "sample=%d true_tokens=%d true_blocks=%d predicted_blocks=%d "
            "block_hits=%d block_misses=%d block_recall=%.4f "
            "avg_block_recall=%.4f weighted_block_hits=%d "
            "weighted_block_total=%d weighted_block_recall=%.4f",
            layer_id,
            sample,
            int(valid.sum().item()),
            len(true_blocks),
            len(predicted_blocks),
            block_hits,
            block_misses,
            block_recall,
            avg_block_recall,
            weighted_block_hits,
            weighted_block_total,
            weighted_block_recall,
        )

    def record_attention_topk_slots(
        self,
        layer_id: int,
        logical_ids_tensor: torch.Tensor,
        slot_ids_tensor: torch.Tensor,
        block_size: int,
    ) -> None:
        """Record true sparse-MLA KV slots consumed by attention.

        Args:
            layer_id: CSA layer whose sparse attention is about to run.
            logical_ids_tensor: Logical true top-K IDs produced by the official
                Lightning Indexer before block-table translation.
            slot_ids_tensor: Global physical KV slot IDs after block-table
                translation, with ``-1`` entries already marking invalid slots.
            block_size: Attention KV cache block size used to derive block IDs.
        """
        first_log = layer_id not in self._debug_attention_topk_logged
        should_profile = _profile_accuracy_enabled() and (
            self._attention_profile_seen < self._proxy_profile_limit
        )
        should_prefill_correct = (
            self.prefill_proxy_enabled() and self.layer_initialized(layer_id)
        )
        should_collect = should_profile or _timing_enabled() or should_prefill_correct
        if not should_collect:
            if first_log:
                self._debug_attention_topk_logged.add(layer_id)
                cols = int(slot_ids_tensor.shape[1]) if slot_ids_tensor.ndim > 1 else 1
                logger.info(
                    "IndexerSSDManager: attention_true_topk layer %d "
                    "rows=%d cols=%d block_size=%d",
                    layer_id,
                    int(slot_ids_tensor.shape[0]),
                    cols,
                    int(block_size),
                )
            return

        logical_ids: List[int] = []
        logical_seen: Set[int] = set()
        for tid in logical_ids_tensor.reshape(-1).detach().cpu().tolist():
            tid_int = int(tid)
            if tid_int < 0 or tid_int in logical_seen:
                continue
            logical_seen.add(tid_int)
            logical_ids.append(tid_int)

        slot_ids: List[int] = []
        slot_seen: Set[int] = set()
        for slot in slot_ids_tensor.reshape(-1).detach().cpu().tolist():
            slot_int = int(slot)
            if slot_int < 0 or slot_int in slot_seen:
                continue
            slot_seen.add(slot_int)
            slot_ids.append(slot_int)

        block_ids = {slot // int(block_size) for slot in slot_ids}
        true_set = set(logical_ids)
        true_blocks = {token_id // DEEPGEMM_PAGED_BLOCK_SIZE for token_id in true_set}
        predicted_blocks = set(self._last_proxy_blocks.get(layer_id) or [])
        spec_hits = len(true_blocks & predicted_blocks)
        true_misses = len(true_blocks - predicted_blocks)
        recall = float(spec_hits) / float(len(true_blocks)) if true_blocks else 0.0
        if should_prefill_correct and logical_ids:
            self._correct_prefill_true_topk(layer_id, logical_ids)

        if first_log:
            self._debug_attention_topk_logged.add(layer_id)
            logger.info(
                "IndexerSSDManager: attention_true_topk layer %d unique_slots=%d "
                "unique_tokens=%d unique_blocks=%d block_size=%d",
                layer_id,
                len(slot_ids),
                len(true_set),
                len(block_ids),
                int(block_size),
            )
        if should_profile and true_blocks:
            self._attention_profile_seen += 1
            self._attention_profile_hits += spec_hits
            self._attention_profile_total += len(true_blocks)
            avg_recall = (
                float(self._attention_profile_hits)
                / float(self._attention_profile_total)
                if self._attention_profile_total > 0
                else 0.0
            )
            logger.info(
                "IndexerSSDManager: attention_true_topk_profile layer %d "
                "sample=%d true_tokens=%d true_blocks=%d predicted_blocks=%d "
                "block_hits=%d block_misses=%d block_recall=%.4f "
                "avg_block_recall=%.4f physical_blocks=%d",
                layer_id,
                self._attention_profile_seen,
                len(true_set),
                len(true_blocks),
                len(predicted_blocks),
                spec_hits,
                true_misses,
                recall,
                avg_recall,
                len(block_ids),
            )
        self._log_timing(
            "attention_true_topk",
            layer_id,
            rows=int(slot_ids_tensor.shape[0]),
            cols=int(slot_ids_tensor.shape[1]) if slot_ids_tensor.ndim > 1 else 1,
            block_size=int(block_size),
            true_tokens=len(true_set),
            true_blocks=len(true_blocks),
            predicted_blocks=len(predicted_blocks),
            block_hits=spec_hits,
            block_misses=true_misses,
            physical_blocks=len(block_ids),
        )

    def _correct_prefill_true_topk(
        self,
        layer_id: int,
        token_ids: List[int],
    ) -> None:
        """Drain prefill proxy reads and synchronously fill true-topK misses."""
        t0 = time.perf_counter()
        cursor = self._decode_cursor.get(layer_id, 0)
        if cursor <= 0:
            self._log_timing(
                "prefill_correct_true_topk_skip",
                layer_id,
                reason="ssd_uninitialized",
                true=len(token_ids),
                cursor=cursor,
            )
            return

        t_drain0 = time.perf_counter()
        self._drain(layer_id)
        drain_ms = (time.perf_counter() - t_drain0) * 1000.0

        pool = self._pools[layer_id]
        store = self._stores[layer_id]

        t_miss0 = time.perf_counter()
        missing = [
            token_id
            for token_id in token_ids
            if 0 <= token_id < cursor and not pool.contains(token_id)
        ]
        miss_ms = (time.perf_counter() - t_miss0) * 1000.0

        read_ms = 0.0
        insert_ms = 0.0
        for token_id in missing:
            try:
                t_read0 = time.perf_counter()
                data = store.read_token(token_id)
                read_ms += (time.perf_counter() - t_read0) * 1000.0
                if not data:
                    continue
                t_insert0 = time.perf_counter()
                pool.insert(token_id, data)
                insert_ms += (time.perf_counter() - t_insert0) * 1000.0
            except Exception as exc:
                logger.error(
                    "IndexerSSDManager: prefill true-topK fallback read "
                    "failed for token %d layer %d: %r",
                    token_id,
                    layer_id,
                    exc,
                )

        pool.protect_only([token_id for token_id in token_ids if token_id < cursor])
        self._prev_topk[layer_id] = list(token_ids)
        self._log_timing(
            "prefill_correct_true_topk",
            layer_id,
            total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
            drain_ms=f"{drain_ms:.3f}",
            miss_ms=f"{miss_ms:.3f}",
            read_ms=f"{read_ms:.3f}",
            insert_ms=f"{insert_ms:.3f}",
            true=len(token_ids),
            missing=len(missing),
        )

    def _record_residual_proxy_accuracy(
        self,
        layer_id: int,
        true_token_ids: List[int],
    ) -> None:
        """Log residual-proxy recall against the true top-k when profiling."""
        if not _profile_accuracy_enabled():
            return
        if self._proxy_profile_seen >= self._proxy_profile_limit:
            return
        spec_token_ids = self._last_proxy_topk.get(layer_id)
        if not spec_token_ids or not true_token_ids:
            return
        compare_k = min(len(spec_token_ids), len(true_token_ids))
        if compare_k <= 0:
            return
        spec_set = set(spec_token_ids[:compare_k])
        hits = sum(1 for tid in true_token_ids[:compare_k] if tid in spec_set)
        self._proxy_profile_seen += 1
        self._proxy_profile_hits += hits
        self._proxy_profile_total += compare_k
        recall = float(hits) / float(compare_k)
        avg = float(self._proxy_profile_hits) / float(self._proxy_profile_total)
        logger.info(
            "IndexerSSDManager: residual_proxy_accuracy layer %d sample=%d "
            "k=%d recall=%.4f avg_recall=%.4f",
            layer_id,
            self._proxy_profile_seen,
            compare_k,
            recall,
            avg,
        )

    def insert_decode_token(self, layer_id: int, data: bytes) -> Optional[int]:
        """Insert the current decode token into the HBM pool and SSD store.

        The decode cursor is initialized by :meth:`evict_after_prefill`. If the
        layer has not been initialized yet, this method is a no-op.

        Args:
            layer_id: CSA layer index.
            data: Raw uint8 bytes in the same layout as the IndexerCache.

        Returns:
            Global token ID assigned to the inserted decode token, or ``None``
            when SSD state has not been initialized.
        """
        if len(data) != self._token_bytes:
            raise ValueError(f"Expected {self._token_bytes} bytes, got {len(data)}")
        with self._lock:
            token_id = self._decode_cursor.get(layer_id, 0)
            if token_id <= 0:
                return None
            self._decode_cursor[layer_id] = token_id + 1
        self._pools[layer_id].insert(token_id, data)
        self._stores[layer_id].write_token(token_id, data)
        if layer_id not in self._debug_insert_logged:
            self._debug_insert_logged.add(layer_id)
            logger.info(
                "IndexerSSDManager: insert_decode_token layer %d token_id=%d",
                layer_id,
                token_id,
            )
        return token_id

    # ------------------------------------------------------------------
    # Called after prefill to evict IndexerCache to SSD
    # ------------------------------------------------------------------

    def evict_after_prefill(
        self,
        layer_id: int,
        kv_cache_tensor: torch.Tensor,
        seq_len: int,
        seed_token_ids: Optional[List[int]] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        timing_event: str = "evict_after_prefill",
        persist_to_store: bool = True,
    ) -> None:
        """Populate pool + SSD from vLLM's IndexerCache tensor after prefill.

        Reads all *seq_len* token K-vectors from *kv_cache_tensor* (vLLM's
        paged IndexerCache), writes them to SSD, and loads
        *pool_size* of them into the HBM pool (using the seed tokens if
        provided, otherwise the first pool_size tokens).

        Args:
            layer_id: CSA layer index.
            kv_cache_tensor: vLLM's IndexerCache tensor
                ``[num_blocks, block_size, 1, token_bytes]`` uint8 on CPU
                (caller must transfer to CPU before calling).
            seq_len: Number of valid tokens (context length after prefill).
            seed_token_ids: Sequence positions (0..seq_len-1) to prioritize in
                HBM pool (prefill topk); marked as resident (lower eviction priority).
            slot_mapping: Optional tensor [seq_len] mapping sequence position i
                to its physical vLLM slot in *kv_cache_tensor*.  When provided,
                token i is read from kv_cache_tensor[slot//block_size, slot%block_size].
                When None, sequential layout is assumed: token i maps to
                block i//block_size at offset i%block_size.
            block_table: Optional tensor mapping logical block IDs to physical vLLM
                block IDs.  When provided, it is expanded into a slot mapping for
                the full compressed IndexerCache sequence.
            timing_event: Timing event name used to distinguish post-prefill
                eviction from LMCache-hit reuse seeding.
            persist_to_store: Whether to rewrite the complete SSD layer. Reuse
                seeding disables this because the cold-store pass already
                persisted the prefix; only the selected HBM seed rows need to
                be gathered again.
        """
        t0 = time.perf_counter()
        pool = self._pools[layer_id]
        store = self._stores[layer_id]
        block_size = kv_cache_tensor.shape[1]
        t_reset0 = time.perf_counter()
        pool.reset()
        reset_ms = (time.perf_counter() - t_reset0) * 1000.0

        map_ms = 0.0
        if block_table is not None:
            t_map0 = time.perf_counter()
            slot_mapping = self._slot_mapping_from_block_table(
                block_table, seq_len, block_size
            )
            map_ms = (time.perf_counter() - t_map0) * 1000.0

        # Resolve the HBM seed set before gathering cache rows. On a reuse hit
        # the SSD layer already contains the cold prefix, so gathering and
        # rewriting all ``seq_len`` rows would serialize prediction behind a
        # redundant 15+ MiB write per CSA layer.
        load_ids: List[int] = []
        if seed_token_ids:
            load_ids = seed_token_ids[: self._pool_size]
        if len(load_ids) < self._pool_size:
            seed_set = set(load_ids)
            extra = [tid for tid in range(seq_len) if tid not in seed_set]
            load_ids += extra[: self._pool_size - len(load_ids)]

        t_read0 = time.perf_counter()
        if persist_to_store:
            token_bytes = self._read_packed_tokens(
                kv_cache_tensor,
                seq_len,
                slot_mapping,
            )
            seed_bytes = token_bytes[load_ids] if load_ids else token_bytes[:0]
        else:
            seed_bytes = self._read_packed_token_ids(
                kv_cache_tensor,
                load_ids,
                slot_mapping,
                seq_len,
            )
            token_bytes = None
        read_ms = (time.perf_counter() - t_read0) * 1000.0

        t_write0 = time.perf_counter()
        if persist_to_store:
            assert token_bytes is not None
            store.write_tokens_contiguous(
                0,
                token_bytes.contiguous().numpy().tobytes(),
            )
        write_ms = (time.perf_counter() - t_write0) * 1000.0

        # Load seed tokens into HBM pool first.
        t_load0 = time.perf_counter()
        if load_ids:
            pool.load_tokens(load_ids, seed_bytes.contiguous())

        # Protect seed tokens so they are evicted last
        if seed_token_ids:
            for tid in seed_token_ids[: self._pool_size]:
                pool.protect(tid)
        load_ms = (time.perf_counter() - t_load0) * 1000.0

        # Initialize prev_topk and decode cursor
        if seed_token_ids:
            self._prev_topk[layer_id] = list(seed_token_ids[:1024])
        with self._lock:
            self._decode_cursor[layer_id] = seq_len
        log_key = (timing_event, layer_id)
        if log_key not in self._debug_evict_logged:
            self._debug_evict_logged.add(log_key)
            logger.info(
                "IndexerSSDManager: %s complete layer %d seq_len=%d seed_tokens=%d",
                timing_event,
                layer_id,
                seq_len,
                len(seed_token_ids or []),
            )
        self._log_timing(
            timing_event,
            layer_id,
            total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
            reset_ms=f"{reset_ms:.3f}",
            map_ms=f"{map_ms:.3f}",
            read_ms=f"{read_ms:.3f}",
            write_ms=f"{write_ms:.3f}",
            load_ms=f"{load_ms:.3f}",
            seq_len=seq_len,
            seed_tokens=len(seed_token_ids or []),
            load_ids=len(load_ids),
            persist_to_store=int(persist_to_store),
        )

    def submit_evict_after_prefill(
        self,
        layer_id: int,
        kv_cache_cpu: torch.Tensor,
        seq_len: int,
        seed_token_ids: Optional[List[int]] = None,
        slot_mapping_cpu: Optional[torch.Tensor] = None,
        block_table_cpu: Optional[torch.Tensor] = None,
        timing_event: str = "evict_after_prefill",
        persist_to_store: bool = True,
    ) -> None:
        """Submit :meth:`evict_after_prefill` to the I/O thread pool.

        The tensors must already be on CPU (caller's responsibility to call
        ``.cpu()`` on CUDA tensors before invoking this method).
        :meth:`prepare_pool` will block on the resulting future so the pool
        is populated before the first decode step scores against it.

        Args:
            layer_id: CSA layer index.
            kv_cache_cpu: IndexerCache tensor already on CPU.
            seq_len: Number of valid prefill tokens.
            seed_token_ids: Prefill topk sequence positions.
            slot_mapping_cpu: Slot mapping tensor already on CPU.
            block_table_cpu: Block table tensor already on CPU.
            timing_event: Timing event name passed to
                :meth:`evict_after_prefill`.
            persist_to_store: Whether the worker must rewrite the complete SSD
                layer before publishing the HBM seed pool.
        """
        with self._lock:
            previous_future = self._ready_futures.get(layer_id)

        def _run_after_previous() -> None:
            if previous_future is not None:
                try:
                    previous_future.result(timeout=self._prefill_ready_timeout_s)
                except Exception:
                    logger.exception(
                        "IndexerSSDManager: previous prefill eviction failed "
                        "before layer %d seq_len=%d; continuing with the "
                        "newer eviction because it rewrites layer state",
                        layer_id,
                        seq_len,
                    )
            try:
                self.evict_after_prefill(
                    layer_id,
                    kv_cache_cpu,
                    seq_len,
                    seed_token_ids,
                    slot_mapping_cpu,
                    block_table_cpu,
                    timing_event=timing_event,
                    persist_to_store=persist_to_store,
                )
                self._record_ready_cuda_event((layer_id,))
            except Exception:
                logger.exception(
                    "IndexerSSDManager: %s failed layer %d "
                    "seq_len=%d seed_tokens=%d kv_shape=%s "
                    "slot_mapping_shape=%s block_table_shape=%s",
                    timing_event,
                    layer_id,
                    seq_len,
                    len(seed_token_ids or []),
                    tuple(kv_cache_cpu.shape),
                    tuple(slot_mapping_cpu.shape)
                    if slot_mapping_cpu is not None
                    else None,
                    tuple(block_table_cpu.shape)
                    if block_table_cpu is not None
                    else None,
                )
                raise

        future = self._executor.submit(_run_after_previous)
        with self._lock:
            self._ready_cuda_events[layer_id] = None
            self._ready_futures[layer_id] = future
        logger.info(
            "IndexerSSDManager: submit_%s layer %d seq_len=%d seed_tokens=%d",
            timing_event,
            layer_id,
            seq_len,
            len(seed_token_ids or []),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slot_mapping_from_block_table(
        block_table: torch.Tensor, seq_len: int, block_size: int
    ) -> torch.Tensor:
        """Expand a vLLM block table row into per-token physical slot IDs."""
        if block_table.ndim == 2:
            block_row = block_table[0]
        elif block_table.ndim == 1:
            block_row = block_table
        else:
            raise ValueError(
                f"Expected 1D/2D block_table, got shape {tuple(block_table.shape)}"
            )
        num_blocks = (seq_len + block_size - 1) // block_size
        block_row = block_row[:num_blocks].to(dtype=torch.long, device="cpu")
        offsets = torch.arange(block_size, dtype=torch.long).repeat(num_blocks)
        slots = block_row.repeat_interleave(block_size) * block_size + offsets
        return slots[:seq_len]

    def _read_packed_token(
        self,
        kv_cache_tensor: torch.Tensor,
        block_idx: int,
        block_offset: int,
    ) -> torch.Tensor:
        """Read one token from an IndexerCache packed block.

        ``kv_cache_tensor[block, offset]`` is not a token because each block is
        laid out as all values followed by all scales.  The SSD store keeps one
        token as interleaved ``value+scale`` bytes, so eviction and decode
        insertion use this conversion.
        """
        block_size = kv_cache_tensor.shape[1]
        flat_block = kv_cache_tensor[block_idx].reshape(-1)
        value_start = block_offset * self._head_dim
        scale_start = block_size * self._head_dim + (block_offset * self._scale_bytes)
        value = flat_block[value_start : value_start + self._head_dim]
        scale = flat_block[scale_start : scale_start + self._scale_bytes]
        return torch.cat((value, scale))

    def _read_packed_tokens(
        self,
        kv_cache_tensor: torch.Tensor,
        seq_len: int,
        slot_mapping: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Read the first *seq_len* logical tokens from packed IndexerCache.

        Returns a CPU tensor ``[seq_len, token_bytes]`` in interleaved
        ``value+scale`` layout used by the SSD store.
        """
        block_size = kv_cache_tensor.shape[1]
        device = kv_cache_tensor.device
        if slot_mapping is None:
            slots = torch.arange(seq_len, dtype=torch.long, device=device)
        else:
            slots = slot_mapping[:seq_len].to(device=device, dtype=torch.long)
            if bool((slots < 0).any().item()):
                bad_slot = int(slots[slots < 0][0].item())
                raise ValueError(f"Invalid negative slot {bad_slot}")

        block_ids = slots // block_size
        block_offsets = slots % block_size
        flat_blocks = kv_cache_tensor.reshape(kv_cache_tensor.shape[0], -1)

        value_offsets = block_offsets.unsqueeze(1) * self._head_dim + torch.arange(
            self._head_dim, dtype=torch.long, device=device
        )
        scale_offsets = (
            block_size * self._head_dim
            + block_offsets.unsqueeze(1) * self._scale_bytes
            + torch.arange(self._scale_bytes, dtype=torch.long, device=device)
        )
        values = flat_blocks[block_ids.unsqueeze(1), value_offsets]
        scales = flat_blocks[block_ids.unsqueeze(1), scale_offsets]
        return torch.cat((values, scales), dim=1)

    def _read_packed_token_ids(
        self,
        kv_cache_tensor: torch.Tensor,
        token_ids: Sequence[int],
        slot_mapping: Optional[torch.Tensor],
        seq_len: int,
    ) -> torch.Tensor:
        """Gather selected logical tokens from a packed IndexerCache."""
        device = kv_cache_tensor.device
        logical_ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)
        if logical_ids.numel() == 0:
            return torch.empty(
                (0, self._token_bytes),
                dtype=torch.uint8,
                device=device,
            )
        if bool(((logical_ids < 0) | (logical_ids >= seq_len)).any().item()):
            raise ValueError("seed token id is outside the reused prefix")
        if slot_mapping is None:
            slots = logical_ids
        else:
            mapping = slot_mapping.to(device=device, dtype=torch.long)
            slots = mapping.index_select(0, logical_ids)
            if bool((slots < 0).any().item()):
                raise ValueError("seed token maps to a negative physical slot")

        block_size = int(kv_cache_tensor.shape[1])
        block_ids = torch.div(slots, block_size, rounding_mode="floor")
        block_offsets = slots - block_ids * block_size
        flat_blocks = kv_cache_tensor.reshape(kv_cache_tensor.shape[0], -1)
        value_offsets = block_offsets.unsqueeze(1) * self._head_dim + torch.arange(
            self._head_dim,
            dtype=torch.long,
            device=device,
        )
        scale_offsets = (
            block_size * self._head_dim
            + block_offsets.unsqueeze(1) * self._scale_bytes
            + torch.arange(self._scale_bytes, dtype=torch.long, device=device)
        )
        values = flat_blocks[block_ids.unsqueeze(1), value_offsets]
        scales = flat_blocks[block_ids.unsqueeze(1), scale_offsets]
        return torch.cat((values, scales), dim=1)

    def _tail_seed_token_ids(self, start: int, rows: int) -> List[int]:
        pool_size = max(1, self._pool_size)
        tail = min(rows, pool_size)
        tail_start = start + max(0, rows - tail)
        return list(range(tail_start, start + rows))

    def _drain(self, layer_id: int) -> None:
        """Wait for all pending async reads for *layer_id* and insert into pool."""
        t0 = time.perf_counter()
        with self._lock:
            pending = self._pending.get(layer_id, [])
            self._pending[layer_id] = []
        pool = self._pools[layer_id]
        store = self._stores[layer_id]
        wait_ms = 0.0
        insert_ms = 0.0
        fallback_count = 0
        for tid, fut in pending:
            try:
                t_wait0 = time.perf_counter()
                data = fut.result()
                wait_ms += (time.perf_counter() - t_wait0) * 1000.0
                if data:
                    t_insert0 = time.perf_counter()
                    pool.insert(tid, data)
                    insert_ms += (time.perf_counter() - t_insert0) * 1000.0
            except Exception as exc:
                if isinstance(exc, _SpeculativeReadCancelled):
                    logger.debug(
                        "IndexerSSDManager: speculative read cancelled for "
                        "token %d layer %d",
                        tid,
                        layer_id,
                    )
                    continue
                logger.warning(
                    "IndexerSSDManager: async read failed for token %d layer %d: %r",
                    tid,
                    layer_id,
                    exc,
                )
                # Fallback: synchronous read
                try:
                    fallback_count += 1
                    t_wait0 = time.perf_counter()
                    data = store.read_token(tid)
                    wait_ms += (time.perf_counter() - t_wait0) * 1000.0
                    if data:
                        t_insert0 = time.perf_counter()
                        pool.insert(tid, data)
                        insert_ms += (time.perf_counter() - t_insert0) * 1000.0
                except Exception as exc2:
                    logger.error(
                        "IndexerSSDManager: fallback sync read also failed: %s", exc2
                    )
            finally:
                with self._lock:
                    self._inflight_tokens[layer_id].discard(tid)
        self._log_timing(
            "drain",
            layer_id,
            total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
            wait_ms=f"{wait_ms:.3f}",
            insert_ms=f"{insert_ms:.3f}",
            pending=len(pending),
            fallback=fallback_count,
        )

    def reset(self) -> None:
        """Reset all per-step state (prev_topk, pending reads, cursors).

        Call between benchmark runs or when starting a new sequence.
        """
        with self._lock:
            for lid in self._csa_layer_ids:
                self._prev_topk[lid] = None
                self._pending[lid] = []
                self._inflight_tokens[lid].clear()
                self._ready_futures[lid] = None
                self._ready_cuda_events[lid] = None
                self._drain_futures[lid] = None
                self._drain_cuda_events[lid] = None
                self._proxy_futures[lid] = []
                self._decode_cursor[lid] = 0
                self._last_proxy_blocks[lid] = None

    def close(self) -> None:
        """Shut down I/O thread pool and close SSD files."""
        if self._closed:
            return
        self._closed = True
        self._proxy_executor.shutdown(wait=True)
        self._proxy_io_executor.shutdown(wait=True)
        self._executor.shutdown(wait=True)
        self._persistence_executor.shutdown(wait=True)
        self._native_indexer_stream_executor.shutdown(wait=True)
        native_loader = self._native_indexer_cache_manager
        if native_loader is not None:
            native_loader.close()
        for store in self._stores.values():
            store.close()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_vllm_model(
        cls,
        model: torch.nn.Module,
        store_dir: str,
        pool_size: int = 2048,
        io_workers: int = 8,
        max_seq_len: int = 131072,
    ) -> "IndexerSSDManager":
        """Construct an IndexerSSDManager by inspecting a vLLM DeepSeek model.

        Scans the model's layers for DSv4 indexer-cache modules to discover
        CSA layer IDs and pool configuration.

        Args:
            model: vLLM DeepSeek V4 model or compatible module tree.
            store_dir: Directory for SSD backing files.
            pool_size: HBM pool capacity in tokens per CSA layer.
            io_workers: Async I/O thread pool size.
            max_seq_len: Maximum context length.

        Returns:
            Configured :class:`IndexerSSDManager` instance.

        Raises:
            ValueError: If no CSA layers are found in the model.
        """
        indexer_cache_classes = _deepseek_indexer_cache_classes()

        csa_layer_ids: List[int] = []
        token_bytes: Optional[int] = None
        device: Optional[torch.device] = None

        for name, module in model.named_modules():
            if isinstance(module, indexer_cache_classes):
                # Extract the layer index from a prefix such as
                # "model.layers.3.self_attn.indexer.k_cache".
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            lid = int(parts[i + 1])
                            csa_layer_ids.append(lid)
                        except ValueError:
                            pass
                        break
                if token_bytes is None and module.kv_cache.numel() > 0:
                    token_bytes = module.head_dim
                if device is None and module.kv_cache.numel() > 0:
                    device = module.kv_cache.device

        if not csa_layer_ids:
            raise ValueError(
                "IndexerSSDManager.from_vllm_model: no DSv4 indexer-cache "
                "layers found in model"
            )

        csa_layer_ids = sorted(set(csa_layer_ids))
        if token_bytes is None:
            # Infer from model config
            token_bytes = 144  # FP8 head_dim=128 + 16-byte scale overhead (heuristic)
        if device is None:
            device = torch.device("cuda")

        return cls(
            csa_layer_ids=csa_layer_ids,
            store_dir=store_dir,
            pool_size=pool_size,
            token_bytes=token_bytes,
            max_seq_len=max_seq_len,
            io_workers=io_workers,
            device=device,
        )

    def warm_runtime_resources(self) -> None:
        """Create reusable proxy resources before the first cache hit.

        The residual proxy path otherwise creates one CUDA stream and one
        pinned block-selection buffer per CSA target on first use. For long
        prefixes that setup is visible in first-hit TTFT even though later
        requests reuse the same resources. This method is idempotent and may
        be called during adapter attachment after CUDA has been initialised.
        """
        if self._device.type != "cuda" or not torch.cuda.is_available():
            return

        with torch.cuda.device(self._device):
            for layer_id in self._csa_layer_ids:
                self._proxy_stream_for(layer_id)

        selection_sizes = [self._proxy_block_budget]
        cp_size = _env_int("LMCACHE_CSA_PREFETCH_CP_SIZE", 1)
        if self._cp_exchange_proxy_ids and cp_size > 1:
            selection_sizes.append(self._proxy_block_budget * cp_size)
        with self._proxy_buffers_lock:
            for layer_id in self._csa_layer_ids:
                pool = self._proxy_cpu_selection_pool.setdefault(layer_id, [])
                for selection_size in selection_sizes:
                    if any(int(buffer.numel()) == selection_size for buffer in pool):
                        continue
                    pool.append(
                        torch.empty(
                            (selection_size,),
                            dtype=torch.int32,
                            device="cpu",
                            pin_memory=True,
                        )
                    )

        logger.info(
            "IndexerSSDManager: warmed proxy resources layers=%d "
            "selection_sizes=%s device=%s",
            len(self._csa_layer_ids),
            selection_sizes,
            self._device,
        )

    def _warm_cold_proxy_kernels(
        self,
        layer_id: int,
        residual_f: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Warm one target's exact 8192-row proxy shape for each cold chunk.

        Cold admission normally uses much larger scheduler chunks, so its
        official indexer execution does not initialise the kernels selected by
        the 8192-row cache-hit proxy. The warmup is read-only, discards top-k,
        and deliberately orders the model stream after its completion. Running
        once per chunk is intentional: DeepGEMM specializes the scorer for the
        active total K length, so only the final cold chunk warms the exact
        full-prefix shape needed by the first hit. This moves shape-specific
        CUDA setup into cold admission without issuing speculative I/O or
        racing the target layer's cache writes.
        """
        if not residual_f.is_cuda or int(residual_f.shape[0]) < _COLD_PROXY_WARM_ROWS:
            return
        decoder_layer = self._decoder_layers.get(int(layer_id))
        if decoder_layer is None or not self._is_deepseek_v4_layer(decoder_layer):
            return
        try:
            with torch.cuda.device(residual_f.device):
                model_stream = torch.cuda.current_stream(residual_f.device)
                proxy_stream = self._proxy_stream_for(int(layer_id))
                proxy_stream.wait_stream(model_stream)
                with torch.no_grad(), torch.cuda.stream(proxy_stream):
                    warm_residual = residual_f[:_COLD_PROXY_WARM_ROWS]
                    warm_positions = positions[:_COLD_PROXY_WARM_ROWS]
                    runtime_info: dict[str, int] = {}
                    warm_residual.record_stream(proxy_stream)
                    if warm_positions.is_cuda:
                        warm_positions.record_stream(proxy_stream)
                    topk_buf, rows, cp_context = self._residual_proxy_topk_gpu(
                        int(layer_id),
                        decoder_layer,
                        warm_residual,
                        warm_positions,
                        None,
                        enable_prefill_cp=True,
                        metadata_query_row_start=0,
                        runtime_info=runtime_info,
                    )
                    total_seq_lens = int(runtime_info.get("total_seq_lens", 0))
                    if total_seq_lens > 0:
                        num_blocks = (
                            total_seq_lens + DEEPGEMM_PAGED_BLOCK_SIZE - 1
                        ) // DEEPGEMM_PAGED_BLOCK_SIZE
                        selected_blocks = _select_rank_local_proxy_blocks(
                            topk_buf[:rows],
                            total_seq_lens,
                            num_blocks,
                            self._proxy_block_budget,
                        ).to(torch.int32)
                        if cp_context is not None and self._cp_exchange_proxy_ids:
                            import torch.distributed as dist
                            from vllm.distributed import get_tp_group

                            cp_world_size = int(cp_context[1])
                            selected_blocks = selected_blocks.contiguous()
                            exchanged_blocks = torch.empty(
                                int(selected_blocks.numel()) * cp_world_size,
                                dtype=selected_blocks.dtype,
                                device=selected_blocks.device,
                            )
                            dist.all_gather_into_tensor(
                                exchanged_blocks,
                                selected_blocks,
                                group=get_tp_group().device_group,
                            )
                            selected_blocks = exchanged_blocks
                        selected_cpu = self._acquire_proxy_cpu_selection(
                            int(layer_id),
                            int(selected_blocks.numel()),
                        )
                        selected_cpu.copy_(selected_blocks, non_blocking=True)
                        self._release_proxy_cpu_selection(
                            int(layer_id),
                            selected_cpu,
                        )
                    topk_buf.record_stream(proxy_stream)
                    warm_done = torch.cuda.Event()
                    warm_done.record(proxy_stream)
                model_stream.wait_event(warm_done)
            logger.info(
                "IndexerSSDManager: cold proxy kernels warmed layer=%d "
                "rows=%d total_k=%d cp=%s io_dispatched=0",
                int(layer_id),
                int(rows),
                total_seq_lens,
                cp_context is not None,
            )
        except Exception:
            logger.exception(
                "IndexerSSDManager: cold proxy kernel warmup failed layer=%d",
                int(layer_id),
            )

    def _record_ready_cuda_event(self, layer_ids: Sequence[int]) -> None:
        """Publish a stream-local completion event for asynchronous HBM seeds."""
        if self._device.type != "cuda" or not torch.cuda.is_available():
            return
        with torch.cuda.device(self._device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self._device))
        with self._lock:
            for layer_id in layer_ids:
                if int(layer_id) in self._ready_cuda_events:
                    self._ready_cuda_events[int(layer_id)] = event

    def _wait_for_ready_cuda_event(self, layer_id: int) -> None:
        """Order the current stream after a completed asynchronous HBM seed."""
        with self._lock:
            event = self._ready_cuda_events.get(int(layer_id))
        if event is None:
            return
        with torch.cuda.device(self._device):
            torch.cuda.current_stream(self._device).wait_event(event)
        with self._lock:
            if self._ready_cuda_events.get(int(layer_id)) is event:
                self._ready_cuda_events[int(layer_id)] = None

    def _record_drain_cuda_event(self, layer_id: int) -> None:
        """Publish completion of background pool preparation on its stream."""
        if self._device.type != "cuda" or not torch.cuda.is_available():
            return
        with torch.cuda.device(self._device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self._device))
        with self._lock:
            self._drain_cuda_events[int(layer_id)] = event

    def _wait_for_drain_cuda_event(self, layer_id: int) -> None:
        """Order the consumer stream after background pool preparation."""
        with self._lock:
            event = self._drain_cuda_events.get(int(layer_id))
        if event is None:
            return
        with torch.cuda.device(self._device):
            torch.cuda.current_stream(self._device).wait_event(event)
        with self._lock:
            if self._drain_cuda_events.get(int(layer_id)) is event:
                self._drain_cuda_events[int(layer_id)] = None

    def _proxy_stream_for(self, layer_id: int) -> torch.cuda.Stream:
        """Return the reusable CUDA proxy stream for a target layer.

        A per-layer stream keeps independent lookahead targets concurrent while
        naturally ordering repeated work for the same target. Streams are
        created lazily on the manager's active CUDA device and retained for the
        manager lifetime.

        Args:
            layer_id: Target CSA layer whose proxy work will use the stream.

        Returns:
            Reusable CUDA stream dedicated to ``layer_id``.
        """
        with self._proxy_streams_lock:
            stream = self._proxy_streams.get(int(layer_id))
            if stream is None:
                stream = torch.cuda.Stream()
                self._proxy_streams[int(layer_id)] = stream
            return stream

    def _acquire_proxy_cpu_selection(
        self,
        layer_id: int,
        selection_size: int,
    ) -> torch.Tensor:
        """Acquire a pinned fixed-size block list for the Tutti bridge.

        Args:
            layer_id: Target CSA layer id.
            selection_size: Required number of block-id slots.

        Returns:
            A pinned CPU int32 tensor with exactly ``selection_size`` entries.
        """
        with self._proxy_buffers_lock:
            pool = self._proxy_cpu_selection_pool.setdefault(int(layer_id), [])
            for index, buffer in enumerate(pool):
                if int(buffer.numel()) == int(selection_size):
                    pool.pop(index)
                    return buffer
        return torch.empty(
            (int(selection_size),),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

    def _release_proxy_cpu_selection(
        self,
        layer_id: int,
        buffer: torch.Tensor,
    ) -> None:
        """Return one completed pinned block list to the bounded layer pool.

        Args:
            layer_id: Target CSA layer id.
            buffer: Selection whose asynchronous device copy has completed.
        """
        with self._proxy_buffers_lock:
            pool = self._proxy_cpu_selection_pool.setdefault(int(layer_id), [])
            pool[:] = [buffer]
