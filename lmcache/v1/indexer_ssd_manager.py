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
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from lmcache.logging import init_logger

logger = init_logger(__name__)
DEEPGEMM_PAGED_BLOCK_SIZE = 64
_INDEXER_SSD_MANAGER: Optional["IndexerSSDManager"] = None


def get_indexer_ssd_manager() -> Optional["IndexerSSDManager"]:
    """Return the process-local CSA indexer SSD manager, if one is attached."""
    return _INDEXER_SSD_MANAGER


def set_indexer_ssd_manager(manager: Optional["IndexerSSDManager"]) -> None:
    """Set the process-local CSA indexer SSD manager used by GPU connectors."""
    global _INDEXER_SSD_MANAGER
    _INDEXER_SSD_MANAGER = manager


def _proxy_num_rows(proxy_state: Optional[torch.Tensor]) -> int:
    """Return the number of token rows represented by a proxy tensor."""
    if proxy_state is None or proxy_state.ndim == 0:
        return 0
    return int(proxy_state.shape[0])


def _select_last_proxy_row(
    proxy_state: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one tail token row for residual-proxy indexer execution.

    DeepSeek V4 ``hc_pre`` expects the residual to keep its final
    ``[hc_mult, hidden]`` dimensions. Prefill proxy therefore slices only the
    token dimension and keeps the tail token as ``[1, hc_mult, hidden]``.
    """
    row = proxy_state[-1:].contiguous()
    pos = positions.reshape(-1)[-1:].contiguous()
    return row, pos


def _tail_topk_rows(
    topk_buffer: torch.Tensor,
    num_rows: int,
    tail_rows: Optional[int],
) -> torch.Tensor:
    """Return the top-k rows that should drive speculative prefetch."""
    rows = topk_buffer[:num_rows]
    if tail_rows is None:
        return rows
    tail = max(1, min(int(tail_rows), int(rows.shape[0])))
    return rows[-tail:]


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
_RESIDUAL_PROXY_ENABLED: bool = _env_flag("LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY")
_DECODE_PREFETCH_ENABLED: bool = _env_flag(
    "LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH"
)
_PREFILL_RESIDUAL_PROXY_ENV = os.environ.get(
    "LMCACHE_INDEXER_ENABLE_PREFILL_RESIDUAL_PROXY"
)
_PREFILL_RESIDUAL_PROXY_ENABLED: bool = (
    _env_flag("LMCACHE_INDEXER_ENABLE_PREFILL_RESIDUAL_PROXY")
    if _PREFILL_RESIDUAL_PROXY_ENV is not None
    else (
        _RESIDUAL_PROXY_ENABLED
        and _env_flag("LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH")
    )
)
_PREFILL_PROXY_ROWS: int = max(
    0,
    _env_int(
        "LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS",
        1 if _PREFILL_RESIDUAL_PROXY_ENABLED else 0,
    ),
)
if _PREFILL_RESIDUAL_PROXY_ENABLED and _PREFILL_PROXY_ROWS <= 0:
    _PREFILL_PROXY_ROWS = 1
_PREFILL_EVICTION_ENABLED: bool = _env_flag(
    "LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION"
)
_PREFILL_EVICTION_FLAG_FILE = "/tmp/lmcache_indexer_enable_prefill_eviction"
_TIMING_LIMIT: int = max(0, _env_int("LMCACHE_INDEXER_TIMING_LIMIT", 512))
_TIMING_SEED_VERBOSE = _env_flag("LMCACHE_INDEXER_TIMING_SEED_VERBOSE")
_PREFILL_NATIVE_PROXY_TOPK_ENABLED = _env_flag(
    "LMCACHE_INDEXER_ENABLE_PREFILL_NATIVE_PROXY_TOPK"
)
_PREFILL_OVERLAP_PROFILE_FLAG_FILE = "/tmp/lmcache_indexer_prefill_overlap_profile"
_PREFILL_OVERLAP_PROFILE_LIMIT: int = max(
    0, _env_int("LMCACHE_INDEXER_PREFILL_OVERLAP_PROFILE_LIMIT", 512)
)
_PREFILL_OVERLAP_READ_LIMIT: int = max(
    1, _env_int("LMCACHE_INDEXER_PREFILL_OVERLAP_READ_LIMIT", 256)
)
_PROXY_ASYNC_ENABLED = _env_flag("LMCACHE_INDEXER_PROXY_ASYNC")
_PROXY_ASYNC_WORKERS = max(
    1, _env_int("LMCACHE_INDEXER_PROXY_ASYNC_WORKERS", 2)
)


def _timing_enabled() -> bool:
    """Return True when lightweight timing diagnostics are enabled."""
    return _env_flag("LMCACHE_INDEXER_TIMING") or os.path.exists(
        "/tmp/lmcache_indexer_timing"
    )


def _prefill_overlap_profile_enabled() -> bool:
    """Return True when CSA prefill overlap timing diagnostics are enabled."""
    return _env_flag("LMCACHE_INDEXER_PREFILL_OVERLAP_PROFILE") or os.path.exists(
        _PREFILL_OVERLAP_PROFILE_FLAG_FILE
    )


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
        Path(store_dir).mkdir(parents=True, exist_ok=True)
        self._path = Path(store_dir) / f"indexer_layer_{layer_id:03d}.bin"
        self._token_bytes = token_bytes
        self._max_seq_len = max_seq_len
        self._fd: Optional[int] = None
        # Pre-create sparse file
        self._ensure_file()

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
            raise ValueError(
                f"Expected {self._token_bytes} bytes, got {len(data)}"
            )
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

    Read path (CSA spec prefetch and true-LI miss correction) is wired in a
    follow-up commit; this scaffold raises :class:`NotImplementedError`
    for reads so callers fail loudly instead of silently returning empty
    bytes.  The write path (LMCache retrieve seed + prefill new-token
    persistence) is fully wired through
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

    def read_tokens_batch(self, token_ids: List[int]) -> List[bytes]:
        """Read multiple token K vectors via Tutti GPU-direct DMA.

        Each batch issues a single ``load_chunks_to_hbm`` call against the
        shared indexer raw region.  Adjacent ``token_ids`` are coalesced into
        contiguous byte ranges so NVMe issues large sequential reads instead
        of many sector-sized random reads.

        Args:
            token_ids: Token ids to read.  Returned bytes follow the input
                order (not sorted).

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
        loader = self._storage._tutti_loader  # noqa: SLF001 — single owner
        memory_objs = loader.load_chunks_to_hbm(
            [request.key],
            [request.disk_meta],
            shapes_per_key=None,
            file_offsets=[request.file_offset],
            read_ranges_per_key=[request.read_ranges],
        )
        if not memory_objs or memory_objs[0] is None:
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
            raise ValueError(
                f"Expected {self._token_bytes} bytes, got {len(data)}"
            )
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
            (self._pool_size,), -1, dtype=torch.int64, device=device,
        )
        # block_table for fp8_fp4_paged_mqa_logits: [1, pool_size] int32
        self.block_table: torch.Tensor = torch.arange(
            num_blocks, dtype=torch.int32, device=device,
        ).unsqueeze(0)

        # CPU-side index structures
        self._id_to_slot: Dict[int, int] = {}
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
        self._write_slot(slot, data)
        self.pool_ids[slot] = token_id
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

        rows = token_bytes.to(device=self.pool_tensor.device, dtype=torch.uint8)
        n = len(token_ids)
        slots = torch.arange(n, dtype=torch.long, device=self.pool_tensor.device)
        block_idx = slots // self.block_size
        block_offset = slots % self.block_size
        value_offsets = (
            block_offset.unsqueeze(1) * self._head_dim
            + torch.arange(self._head_dim, dtype=torch.long, device=self.pool_tensor.device)
        )
        scale_offsets = (
            self.block_size * self._head_dim
            + block_offset.unsqueeze(1) * self._scale_bytes
            + torch.arange(
                self._scale_bytes, dtype=torch.long, device=self.pool_tensor.device
            )
        )
        flat_blocks = self.pool_tensor.view(self.pool_tensor.shape[0], -1)
        flat_blocks[block_idx.unsqueeze(1), value_offsets] = rows[:, : self._head_dim]
        flat_blocks[block_idx.unsqueeze(1), scale_offsets] = rows[:, self._head_dim :]

        self.pool_ids[:n] = torch.tensor(
            token_ids, dtype=self.pool_ids.dtype, device=self.pool_ids.device
        )
        self.pool_ids[n:] = -1
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
        self.pool_ids.fill_(-1)
        self._id_to_slot.clear()
        self._free = list(range(self._pool_size))
        self._lru_ordinary.clear()
        self._lru_resident.clear()
        self._resident_slots.clear()

    def get_slot(self, token_id: int) -> int:
        """Return pool slot for *token_id*; raise KeyError if not present."""
        return self._id_to_slot[token_id]

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
            raise ValueError(
                f"Expected {self._token_bytes} bytes, got {len(data)}"
            )
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
        old_id = int(self.pool_ids[slot].item())
        if old_id >= 0:
            del self._id_to_slot[old_id]
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
        self._tutti_storage = tutti_storage

        # Per-layer HBM pool
        self._pools: Dict[int, IndexerHBMPool] = {
            lid: IndexerHBMPool(pool_size, token_bytes, device)
            for lid in csa_layer_ids
        }
        # Per-layer SSD store; backend depends on whether a Tutti storage is
        # attached.  The legacy IndexerBlockStore writes per-layer .bin files
        # via os.pread/pwrite; the Tutti backend routes I/O through Tutti's
        # GPU-direct NVMe path against a shared pre-reserved raw region.
        if tutti_storage is not None:
            self._stores: Dict[int, Any] = {
                lid: TuttiIndexerBlockStore(tutti_storage, lid)
                for lid in csa_layer_ids
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
        self._proxy_executor: Optional[ThreadPoolExecutor] = None
        if _PROXY_ASYNC_ENABLED:
            self._proxy_executor = ThreadPoolExecutor(max_workers=_PROXY_ASYNC_WORKERS)
        self._lock = threading.Lock()
        # pending async read results: layer_id → list of (token_id, Future[bytes])
        self._pending: Dict[int, List[Tuple[int, "Future[bytes]"]]] = {
            lid: [] for lid in csa_layer_ids
        }
        self._inflight_tokens: Dict[int, Set[int]] = {
            lid: set() for lid in csa_layer_ids
        }
        self._prefill_overlap_pending: Dict[
            int, List[Tuple[int, "Future[Tuple[bytes, float, float, float]]"]]
        ] = {lid: [] for lid in csa_layer_ids}
        self._prefill_overlap_seen = 0

        # Per-layer speculative topk from previous step: token IDs
        self._prev_topk: Dict[int, Optional[List[int]]] = {
            lid: None for lid in csa_layer_ids
        }
        self._last_proxy_topk: Dict[int, Optional[List[int]]] = {
            lid: None for lid in csa_layer_ids
        }
        self._last_attention_true_ids: Dict[int, Optional[Set[int]]] = {
            lid: None for lid in csa_layer_ids
        }
        self._last_attention_block_ids: Dict[int, Optional[Set[int]]] = {
            lid: None for lid in csa_layer_ids
        }
        self._decoder_layers: Dict[int, Any] = {}

        # Future tracking for post-prefill async SSD eviction.
        # prepare_pool blocks on this before scoring if it's not yet done.
        self._ready_futures: Dict[int, Optional[Future[None]]] = {
            lid: None for lid in csa_layer_ids
        }
        self._drain_futures: Dict[int, Optional[Future[None]]] = {
            lid: None for lid in csa_layer_ids
        }
        self._proxy_futures: Dict[int, List[Future[None]]] = {
            lid: [] for lid in csa_layer_ids
        }
        self._direct_seed_tail_buffers: Dict[int, List[Tuple[int, torch.Tensor]]] = {
            lid: [] for lid in csa_layer_ids
        }

        # Next sequential token ID for new decode-step tokens, per layer.
        # Set to seq_len by evict_after_prefill; incremented on each decode step.
        # Stays 0 until eviction runs (acts as "SSD uninitialized" guard).
        self._decode_cursor: Dict[int, int] = {lid: 0 for lid in csa_layer_ids}

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
        logger.info(
            "IndexerSSDManager: config residual_proxy=%s reuse_prefetch=%s "
            "decode_prefetch=%s prefill_proxy=%s prefill_rows=%d "
            "proxy_async=%s proxy_async_workers=%d",
            _RESIDUAL_PROXY_ENABLED,
            self.reuse_prefetch_enabled(),
            _DECODE_PREFETCH_ENABLED,
            _PREFILL_RESIDUAL_PROXY_ENABLED,
            _PREFILL_PROXY_ROWS,
            _PROXY_ASYNC_ENABLED,
            _PROXY_ASYNC_WORKERS,
        )

    @staticmethod
    def _profile_read_token(
        store: IndexerBlockStore,
        token_id: int,
    ) -> Tuple[bytes, float, float, float]:
        """Read one token and return payload plus timing metadata."""
        start = time.perf_counter()
        data = store.read_token(token_id)
        end = time.perf_counter()
        return data, start, end, (end - start) * 1000.0

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

    def prefill_eviction_enabled(self) -> bool:
        """Return True when post-prefill SSD/pool initialization is enabled."""
        return _PREFILL_EVICTION_ENABLED or os.path.exists(
            _PREFILL_EVICTION_FLAG_FILE
        )

    def reuse_prefetch_enabled(self) -> bool:
        """Return True when LMCache-hit prefill may seed CSA prefetch state."""
        return _env_flag("LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH")

    def decode_prefetch_enabled(self) -> bool:
        """Return True when per-token decode CSA prefetch/correction is enabled."""
        return _DECODE_PREFETCH_ENABLED

    def prefill_overlap_profile_enabled(self) -> bool:
        """Return True when prefill-overlap timing diagnostics are enabled."""
        return _prefill_overlap_profile_enabled()

    def prefill_proxy_enabled(self) -> bool:
        """Return True when prefill-stage CSA residual proxy is enabled."""
        return _PREFILL_RESIDUAL_PROXY_ENABLED and _PREFILL_PROXY_ROWS > 0

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
                f"CSA LMCache range [{start}, {end}) is not aligned to "
                "compress_ratio 4"
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
        tail_rows = max(
            1,
            _env_int("LMCACHE_INDEXER_DIRECT_SEED_TAIL_ROWS", self._pool_size),
        )
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
        contiguous_ms = (
            (time.perf_counter() - t_contig0) * 1000.0 if timing else 0.0
        )

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

            store = self._stores[layer_id]
            t_write0 = time.perf_counter() if timing else 0.0
            store.write_tokens_contiguous(
                overlap_start,
                token_bytes.contiguous().numpy().tobytes(),
            )
            if timing:
                write_ms += (time.perf_counter() - t_write0) * 1000.0

            t_load0 = time.perf_counter() if timing else 0.0
            load_ids: List[int] = []
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
                    pool = self._pools[layer_id]
                    pool.reset()
                    load_ids = list(range(tail_start, total_rows))
                    tail_bytes = torch.cat(chunks, dim=0)[: len(load_ids)]
                    pool.load_tokens(load_ids, tail_bytes.contiguous())
                    pool.protect_only(load_ids)
            if timing:
                load_ms += (time.perf_counter() - t_load0) * 1000.0

            t_state0 = time.perf_counter() if timing else 0.0
            if finished_tail:
                with self._lock:
                    self._decode_cursor[layer_id] = max(
                        self._decode_cursor.get(layer_id, 0),
                        total_rows,
                    )
                    self._prev_topk[layer_id] = list(load_ids[:1024])
                    self._ready_futures[layer_id] = None
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

    # ------------------------------------------------------------------
    # Called from DeepseekV4DecoderLayer.forward, before self.ffn()
    # ------------------------------------------------------------------

    def fire_async_for_layer(
        self,
        layer_id: int,
        residual_f: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        residual_proxy: Optional[torch.Tensor] = None,
    ) -> None:
        """Fire async NVMe reads for the predicted delta of *layer_id*.

        When ``LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=1``, DeepSeek V4 passes
        the attention HC-post state from the previous block. The manager runs
        the target CSA layer's V4 ``HC_pre(...hc_attn...)`` + attention norm +
        indexer projection on that proxy state to predict token IDs. If the
        proxy is unavailable, it falls back to the previous step's true top-k
        for *layer_id*. Tokens already in the HBM pool are skipped. Final
        correctness is handled later by :meth:`correct_true_topk`, after the
        target layer's true Lightning Indexer result is known.

        Args:
            layer_id: CSA layer for which to prefetch.
            residual_f: DeepSeek V4 HC state after attention and before FFN.
            positions: Decode positions for the current token.
            llama_4_scaling: Unused compatibility argument for older call sites.
            residual_proxy: Deprecated compatibility alias for ``residual_f``.
        """
        proxy_state = residual_f if residual_f is not None else residual_proxy
        proxy_rows = _proxy_num_rows(proxy_state)
        is_prefill_proxy = proxy_rows > 1
        if is_prefill_proxy:
            if not _PREFILL_RESIDUAL_PROXY_ENABLED or _PREFILL_PROXY_ROWS <= 0:
                reason = (
                    "prefill_proxy_experimental_disabled"
                    if _PREFILL_PROXY_ROWS > 0
                    else "prefill_proxy_disabled"
                )
                self._log_residual_proxy_skip(layer_id, reason)
                return
        elif not _DECODE_PREFETCH_ENABLED:
            return
        prev = None
        t0 = time.perf_counter()
        proxy_ms = 0.0
        filter_ms = 0.0
        submit_ms = 0.0
        topk_tail_rows: Optional[int] = None
        cursor = self._decode_cursor.get(layer_id, 0)
        if cursor <= 0:
            if layer_id not in self._debug_fire_no_prev_logged:
                self._debug_fire_no_prev_logged.add(layer_id)
                logger.debug(
                    "IndexerSSDManager: fire_async_for_layer layer %d skipped "
                    "because SSD state is not initialized yet",
                    layer_id,
                )
            return
        if _RESIDUAL_PROXY_ENABLED and proxy_state is not None and positions is not None:
            if proxy_state.shape[0] != positions.shape[0]:
                min_rows = min(proxy_state.shape[0], positions.shape[0])
                proxy_state = proxy_state[-min_rows:]
                positions = positions[-min_rows:]
            run_native_proxy = True
            if is_prefill_proxy:
                original_shape = tuple(proxy_state.shape)
                topk_tail_rows = min(
                    max(1, _PREFILL_PROXY_ROWS),
                    int(proxy_state.shape[0]),
                    int(positions.shape[0]),
                )
                # vLLM's prefill sparse-indexer kernels validate against the
                # full prefill metadata chunk. Running the native indexer on
                # only the tail token breaks that invariant, so the default
                # prefill path uses corrected previous top-K stability for I/O
                # overlap. The full native proxy remains available as an
                # explicit ablation because it is metadata-safe but expensive.
                run_native_proxy = _PREFILL_NATIVE_PROXY_TOPK_ENABLED
                if layer_id not in self._debug_residual_proxy_attempt_logged:
                    logger.debug(
                        "IndexerSSDManager: residual_proxy prefill layer %d "
                        "using_tail_rows=%d original_shape=%s compute_shape=%s "
                        "native_topk=%s",
                        layer_id,
                        topk_tail_rows,
                        original_shape,
                        tuple(proxy_state.shape),
                        run_native_proxy,
                    )
            if layer_id not in self._debug_residual_proxy_attempt_logged:
                self._debug_residual_proxy_attempt_logged.add(layer_id)
                logger.debug(
                    "IndexerSSDManager: residual_proxy_attempt layer %d "
                    "proxy_shape=%s positions_shape=%s",
                    layer_id,
                    tuple(proxy_state.shape),
                    tuple(positions.shape),
                )
            if run_native_proxy:
                t_proxy0 = time.perf_counter()
                try:
                    if self._should_run_proxy_async(proxy_state):
                        if self._submit_residual_proxy_topk_async(
                            layer_id,
                            proxy_state,
                            positions,
                            llama_4_scaling,
                            topk_tail_rows=topk_tail_rows,
                            is_prefill_proxy=is_prefill_proxy,
                            fire_start=t0,
                        ):
                            proxy_ms = (time.perf_counter() - t_proxy0) * 1000.0
                            self._log_timing(
                                "prefill_fire_async"
                                if is_prefill_proxy
                                else "fire_async",
                                layer_id,
                                total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
                                proxy_ms=f"{proxy_ms:.3f}",
                                filter_ms="0.000",
                                submit_ms="0.000",
                                prev=0,
                                missing=0,
                                mode="proxy_async_submit",
                            )
                            prev = self._prev_topk.get(layer_id)
                            if prev is None:
                                return
                            self._submit_predicted_reads(
                                layer_id,
                                prev,
                                cursor,
                                is_prefill_proxy=is_prefill_proxy,
                                fire_start=t0,
                                proxy_ms=proxy_ms,
                                mode="proxy_async_prev_fallback",
                            )
                            return
                    prev = self._residual_proxy_topk(
                        layer_id,
                        proxy_state,
                        positions,
                        llama_4_scaling,
                        topk_tail_rows=topk_tail_rows,
                    )
                except Exception as exc:
                    self._log_residual_proxy_skip(
                        layer_id,
                        f"native_proxy_exception={type(exc).__name__}",
                    )
                    logger.debug(
                        "IndexerSSDManager: native residual proxy failed for "
                        "layer %d",
                        layer_id,
                        exc_info=True,
                    )
                    prev = None
                proxy_ms = (time.perf_counter() - t_proxy0) * 1000.0
                self._last_proxy_topk[layer_id] = prev
            elif is_prefill_proxy:
                self._last_proxy_topk[layer_id] = None
        if prev is None:
            self._last_proxy_topk[layer_id] = None
            prev = self._prev_topk.get(layer_id)
        self._submit_predicted_reads(
            layer_id,
            prev,
            cursor,
            is_prefill_proxy=is_prefill_proxy,
            fire_start=t0,
            proxy_ms=proxy_ms,
            mode="sync",
        )

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
            if (
                ready_fut is None
                and not has_pending
                and previous_drain is None
            ):
                return

        def _prepare() -> None:
            if ready_fut is not None:
                ready_fut.result(timeout=self._prefill_ready_timeout_s)
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                with self._lock:
                    if self._ready_futures.get(layer_id) is ready_fut:
                        self._ready_futures[layer_id] = None
            self._drain(layer_id)
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)

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

    def _prefill_overlap_profile_fire(
        self,
        layer_id: int,
        proxy_state: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        llama_4_scaling: Optional[torch.Tensor],
    ) -> None:
        """Submit profile-only CSA reads during prefill without changing output."""
        if self._prefill_overlap_seen >= _PREFILL_OVERLAP_PROFILE_LIMIT:
            return
        if proxy_state is None or positions is None:
            return
        if proxy_state.shape[0] != positions.shape[0]:
            min_rows = min(proxy_state.shape[0], positions.shape[0])
            proxy_state = proxy_state[-min_rows:]
            positions = positions[-min_rows:]
        if proxy_state.shape[0] <= 1:
            return

        rows = max(1, _env_int("LMCACHE_INDEXER_PREFILL_OVERLAP_ROWS", 1))
        rows = min(rows, int(proxy_state.shape[0]))
        proxy_state = proxy_state[-rows:].contiguous()
        positions = positions[-rows:].contiguous()

        t0 = time.perf_counter()
        proxy_ms = 0.0
        predicted_set = self._last_attention_true_ids.get(layer_id)
        if not predicted_set:
            return
        predicted = list(predicted_set)
        self._last_proxy_topk[layer_id] = predicted
        if not predicted:
            return

        token_ids: List[int] = []
        seen: Set[int] = set()
        for tid in predicted:
            tid_int = int(tid)
            if tid_int < 0 or tid_int in seen:
                continue
            seen.add(tid_int)
            token_ids.append(tid_int)
            if len(token_ids) >= _PREFILL_OVERLAP_READ_LIMIT:
                break
        if not token_ids:
            return

        t_submit0 = time.perf_counter()
        store = self._stores[layer_id]
        futures = [
            (tid, self._executor.submit(self._profile_read_token, store, tid))
            for tid in token_ids
        ]
        submit_ms = (time.perf_counter() - t_submit0) * 1000.0
        with self._lock:
            self._prefill_overlap_pending[layer_id].extend(futures)
        self._prefill_overlap_seen += 1
        self._log_timing(
            "prefill_overlap_fire",
            layer_id,
            total_ms=f"{(time.perf_counter() - t0) * 1000.0:.3f}",
            proxy_ms=f"{proxy_ms:.3f}",
            submit_ms=f"{submit_ms:.3f}",
            rows=rows,
            predicted=len(predicted),
            submitted=len(futures),
        )

    def _should_run_proxy_async(self, proxy_state: torch.Tensor) -> bool:
        """Return whether residual-proxy top-k should use async completion."""
        return (
            _PROXY_ASYNC_ENABLED
            and self._proxy_executor is not None
            and proxy_state.is_cuda
            and torch.cuda.is_available()
        )

    def _submit_predicted_reads(
        self,
        layer_id: int,
        prev: Optional[List[int]],
        cursor: int,
        *,
        is_prefill_proxy: bool,
        fire_start: float,
        proxy_ms: float,
        mode: str,
    ) -> bool:
        """Filter predicted token IDs and submit missing CSA SSD reads.

        Args:
            layer_id: CSA layer for which reads should be submitted.
            prev: Predicted token IDs, or ``None`` when no prediction exists.
            cursor: Exclusive upper bound for initialized token IDs.
            is_prefill_proxy: Whether the caller is a prefill proxy path.
            fire_start: Start timestamp for timing diagnostics.
            proxy_ms: Time already spent computing/submitting proxy work.
            mode: Short label recorded in timing logs.

        Returns:
            True when the prediction was usable, even if all tokens were
            already resident; False when no prediction existed.
        """
        if prev is None:
            self._last_proxy_topk[layer_id] = None
            if layer_id not in self._debug_fire_no_prev_logged:
                self._debug_fire_no_prev_logged.add(layer_id)
                logger.debug(
                    "IndexerSSDManager: fire_async_for_layer layer %d has no "
                    "previous topk yet",
                    layer_id,
                )
            return False

        t_filter0 = time.perf_counter()
        filtered = [tid for tid in prev if 0 <= tid < cursor]
        if not filtered:
            return True
        pool = self._pools[layer_id]
        store = self._stores[layer_id]
        with self._lock:
            pending_token_ids = set(self._inflight_tokens[layer_id])
        missing = [
            tid
            for tid in filtered
            if not pool.contains(tid) and tid not in pending_token_ids
        ]
        filter_ms = (time.perf_counter() - t_filter0) * 1000.0
        if not missing:
            if layer_id not in self._debug_fire_active_logged:
                self._debug_fire_active_logged.add(layer_id)
                logger.debug(
                    "IndexerSSDManager: fire_async_for_layer layer %d active, "
                    "prev_topk=%d, all already resident",
                    layer_id,
                    len(filtered),
                )
            self._log_timing(
                "prefill_fire_async" if is_prefill_proxy else "fire_async",
                layer_id,
                total_ms=f"{(time.perf_counter() - fire_start) * 1000.0:.3f}",
                proxy_ms=f"{proxy_ms:.3f}",
                filter_ms=f"{filter_ms:.3f}",
                submit_ms="0.000",
                prev=len(filtered),
                missing=0,
                mode=mode,
            )
            return True
        if layer_id not in self._debug_fire_active_logged:
            self._debug_fire_active_logged.add(layer_id)
            logger.debug(
                "IndexerSSDManager: fire_async_for_layer layer %d active, "
                "prev_topk=%d, missing=%d",
                layer_id,
                len(filtered),
                len(missing),
            )

        t_submit0 = time.perf_counter()
        futures = [
            (tid, self._executor.submit(store.read_token, tid))
            for tid in missing
        ]
        submit_ms = (time.perf_counter() - t_submit0) * 1000.0
        with self._lock:
            self._pending[layer_id].extend(futures)
            self._inflight_tokens[layer_id].update(tid for tid, _ in futures)
        self.prepare_layer_async(layer_id)
        self._log_timing(
            "prefill_fire_async" if is_prefill_proxy else "fire_async",
            layer_id,
            total_ms=f"{(time.perf_counter() - fire_start) * 1000.0:.3f}",
            proxy_ms=f"{proxy_ms:.3f}",
            filter_ms=f"{filter_ms:.3f}",
            submit_ms=f"{submit_ms:.3f}",
            prev=len(filtered),
            missing=len(missing),
            mode=mode,
        )
        return True

    def _submit_residual_proxy_topk_async(
        self,
        layer_id: int,
        residual_f: torch.Tensor,
        positions: torch.Tensor,
        llama_4_scaling: Optional[torch.Tensor],
        *,
        topk_tail_rows: Optional[int],
        is_prefill_proxy: bool,
        fire_start: float,
    ) -> bool:
        """Submit residual-proxy top-k work on a side stream.

        The CUDA work is enqueued immediately, then a background thread waits
        on the recorded event, copies top-k IDs to CPU, and submits missing SSD
        reads. This keeps the decoder hook from synchronizing on
        ``valid.cpu().tolist()`` while MoE/FFN can continue on the default
        stream.
        """
        proxy_executor = self._proxy_executor
        if proxy_executor is None:
            return False
        decoder_layer = self._decoder_layers.get(layer_id)
        if decoder_layer is None or not self._is_deepseek_v4_layer(decoder_layer):
            return False
        device_index = (
            int(residual_f.device.index)
            if residual_f.device.index is not None
            else int(torch.cuda.current_device())
        )
        with torch.cuda.device(device_index):
            proxy_stream = torch.cuda.Stream()
            current_stream = torch.cuda.current_stream()
            proxy_stream.wait_stream(current_stream)
            try:
                with torch.no_grad():
                    with torch.cuda.stream(proxy_stream):
                        residual_f.record_stream(proxy_stream)
                        positions.record_stream(proxy_stream)
                        topk_buf, num_rows = self._residual_proxy_topk_gpu(
                            layer_id,
                            decoder_layer,
                            residual_f,
                            positions,
                            llama_4_scaling,
                            topk_tail_rows=topk_tail_rows,
                        )
                        selected_topk = _tail_topk_rows(
                            topk_buf,
                            num_rows,
                            topk_tail_rows,
                        )
                        valid = selected_topk.reshape(-1)
                        valid = valid[valid >= 0]
                        valid_cpu = torch.empty(
                            (int(valid.numel()),),
                            dtype=valid.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        valid_cpu.copy_(valid, non_blocking=True)
                        event = torch.cuda.Event()
                        event.record(proxy_stream)
            except Exception:
                logger.debug(
                    "IndexerSSDManager: async residual proxy submit failed "
                    "for layer %d",
                    layer_id,
                    exc_info=True,
                )
                return False

        cursor = self._decode_cursor.get(layer_id, 0)
        future = proxy_executor.submit(
            self._finish_residual_proxy_topk_async,
            layer_id,
            valid_cpu,
            event,
            cursor,
            is_prefill_proxy,
            fire_start,
        )
        with self._lock:
            self._proxy_futures[layer_id].append(future)

        def _clear_done(done_future: Future[None]) -> None:
            try:
                done_future.result()
            except Exception:
                logger.exception(
                    "IndexerSSDManager: async residual proxy failed for layer %d",
                    layer_id,
                )
            with self._lock:
                futures = self._proxy_futures.get(layer_id)
                if futures is not None and done_future in futures:
                    futures.remove(done_future)

        future.add_done_callback(_clear_done)
        return True

    def _finish_residual_proxy_topk_async(
        self,
        layer_id: int,
        valid_cpu: torch.Tensor,
        event: Any,
        cursor: int,
        is_prefill_proxy: bool,
        fire_start: float,
    ) -> None:
        """Finish side-stream residual proxy and submit predicted reads."""
        t0 = time.perf_counter()
        event.synchronize()
        token_ids = self._token_ids_from_cpu_topk(valid_cpu)
        proxy_ms = (time.perf_counter() - t0) * 1000.0
        self._last_proxy_topk[layer_id] = token_ids
        if not token_ids:
            self._log_residual_proxy_skip(layer_id, "empty_token_ids")
            return
        if layer_id not in self._debug_residual_proxy_logged:
            self._debug_residual_proxy_logged.add(layer_id)
            logger.info(
                "IndexerSSDManager: residual_proxy layer %d spec_tokens=%d "
                "mode=async",
                layer_id,
                len(token_ids),
            )
        self._submit_predicted_reads(
            layer_id,
            token_ids,
            cursor,
            is_prefill_proxy=is_prefill_proxy,
            fire_start=fire_start,
            proxy_ms=proxy_ms,
            mode="proxy_async_finish",
        )

    @staticmethod
    def _token_ids_from_cpu_topk(valid_cpu: torch.Tensor) -> List[int]:
        """Convert a CPU top-k tensor to a de-duplicated token-id list."""
        token_ids: List[int] = []
        seen: Set[int] = set()
        for tid in valid_cpu.tolist():
            tid_int = int(tid)
            if tid_int in seen:
                continue
            seen.add(tid_int)
            token_ids.append(tid_int)
            if len(token_ids) >= 1024:
                break
        return token_ids

    def _residual_proxy_topk(
        self,
        layer_id: int,
        residual_f: torch.Tensor,
        positions: torch.Tensor,
        llama_4_scaling: Optional[torch.Tensor],
        topk_tail_rows: Optional[int] = None,
    ) -> Optional[List[int]]:
        """Run the next CSA layer's V4 indexer on attention HC-post state."""
        decoder_layer = self._decoder_layers.get(layer_id)
        if decoder_layer is None:
            self._log_residual_proxy_skip(layer_id, "decoder_layer_missing")
            return None
        if not self._is_deepseek_v4_layer(decoder_layer):
            self._log_residual_proxy_skip(
                layer_id,
                f"not_v4_layer={type(decoder_layer).__name__}",
            )
            return None

        with torch.no_grad():
            topk_buf, num_rows = self._residual_proxy_topk_gpu(
                layer_id,
                decoder_layer,
                residual_f,
                positions,
                llama_4_scaling,
                topk_tail_rows=topk_tail_rows,
            )
            selected_topk = _tail_topk_rows(topk_buf, num_rows, topk_tail_rows)
            valid = selected_topk.reshape(-1)
            valid = valid[valid >= 0]
            token_ids = self._token_ids_from_cpu_topk(valid.cpu())
        if not token_ids:
            self._log_residual_proxy_skip(layer_id, "empty_token_ids")
        if layer_id not in self._debug_residual_proxy_logged:
            self._debug_residual_proxy_logged.add(layer_id)
            logger.info(
                "IndexerSSDManager: residual_proxy layer %d spec_tokens=%d",
                layer_id,
                len(token_ids),
            )
        return token_ids

    def _residual_proxy_topk_gpu(
        self,
        layer_id: int,
        decoder_layer: Any,
        residual_f: torch.Tensor,
        positions: torch.Tensor,
        llama_4_scaling: Optional[torch.Tensor],
        *,
        topk_tail_rows: Optional[int],
    ) -> tuple[torch.Tensor, int]:
        """Run the V4 proxy indexer and return the GPU top-k buffer."""
        del llama_4_scaling, topk_tail_rows
        proxy_hidden = self._v4_attention_proxy_hidden(decoder_layer, residual_f)
        qr, kv_score, weights, indexer, rotary_emb = self._v4_indexer_inputs(
            decoder_layer,
            proxy_hidden,
        )
        indexer_op = getattr(indexer, "indexer_op", None)
        if indexer_op is None:
            self._log_residual_proxy_skip(layer_id, "indexer_op_missing")
            raise RuntimeError("DeepSeek V4 CSA indexer_op is missing")
        old_ssd_manager = getattr(indexer_op, "ssd_manager", None)
        indexer_op.ssd_manager = None
        saved_cache_rows = self._save_v4_proxy_cache_rows(indexer, indexer_op)
        try:
            indexer(proxy_hidden, qr, kv_score, weights, positions, rotary_emb)
        finally:
            self._restore_cache_rows(saved_cache_rows)
            indexer_op.ssd_manager = old_ssd_manager
        topk_buf = getattr(indexer_op, "topk_indices_buffer", None)
        if topk_buf is None:
            self._log_residual_proxy_skip(layer_id, "topk_buffer_missing")
            raise RuntimeError("DeepSeek V4 CSA topk buffer is missing")
        return topk_buf, int(proxy_hidden.shape[0])

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
        attn_norm = getattr(decoder_layer, "attn_norm")
        norm_weight = getattr(attn_norm, "weight", None)
        norm_eps = getattr(attn_norm, "variance_epsilon", None)
        if norm_eps is None:
            norm_eps = getattr(decoder_layer, "rms_norm_eps", 1e-6)
        if norm_weight is not None:
            try:
                proxy_hidden, _, _ = decoder_layer.hc_pre(
                    residual_f,
                    getattr(decoder_layer, "hc_attn_fn"),
                    getattr(decoder_layer, "hc_attn_scale"),
                    getattr(decoder_layer, "hc_attn_base"),
                    norm_weight=norm_weight.data,
                    norm_eps=float(norm_eps),
                )
                return proxy_hidden
            except TypeError:
                pass
        proxy_hidden, _, _ = decoder_layer.hc_pre(
            residual_f,
            getattr(decoder_layer, "hc_attn_fn"),
            getattr(decoder_layer, "hc_attn_scale"),
            getattr(decoder_layer, "hc_attn_base"),
        )
        return attn_norm(proxy_hidden)

    @staticmethod
    def _v4_indexer_inputs(
        decoder_layer: Any,
        proxy_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any]:
        """Build the V4 ``Indexer.forward`` inputs for a proxy hidden state."""
        attn = getattr(decoder_layer, "attn")
        indexer = getattr(attn, "indexer", None)
        if indexer is None:
            mla_attn = getattr(attn, "mla_attn", None)
            indexer = getattr(mla_attn, "indexer", None)
        if indexer is None:
            raise RuntimeError("DeepSeek V4 CSA layer has no indexer")

        qr_kv, _, indexer_kv_score, indexer_weights = (
            attn.mla_attn.attn_gemm_parallel_execute(proxy_hidden)
        )
        qr, _ = qr_kv.split([int(attn.q_lora_rank), int(attn.head_dim)], dim=-1)
        qr = attn.q_norm(qr)
        return qr, indexer_kv_score, indexer_weights, indexer, attn.rotary_emb

    @staticmethod
    def _save_v4_proxy_cache_rows(
        indexer: Any,
        indexer_op: Any,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Save V4 proxy-write cache rows so the real KV state stays intact."""
        try:
            from vllm.forward_context import get_forward_context
        except ImportError:
            return []

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return []

        saved: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        compressor = getattr(indexer, "compressor", None)
        state_cache = getattr(getattr(compressor, "state_cache", None), "kv_cache", None)
        state_prefix = getattr(getattr(compressor, "state_cache", None), "prefix", None)
        state_meta = attn_metadata.get(state_prefix) if state_prefix is not None else None
        if isinstance(state_cache, torch.Tensor):
            slot_mapping = getattr(state_meta, "slot_mapping", None)
            state_saved = IndexerSSDManager._save_cache_rows(state_cache, slot_mapping)
            if state_saved is not None:
                saved.append(state_saved)

        k_cache = getattr(getattr(indexer_op, "k_cache", None), "kv_cache", None)
        k_prefix = getattr(compressor, "k_cache_prefix", None)
        k_meta = attn_metadata.get(k_prefix) if k_prefix is not None else None
        if isinstance(k_cache, torch.Tensor):
            slot_mapping = getattr(k_meta, "slot_mapping", None)
            k_saved = IndexerSSDManager._save_cache_rows(k_cache, slot_mapping)
            if k_saved is not None:
                saved.append(k_saved)
        return saved

    @staticmethod
    def _save_cache_rows(
        cache: torch.Tensor,
        slot_mapping: Any,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Clone cache rows addressed by a vLLM slot mapping."""
        if not isinstance(slot_mapping, torch.Tensor):
            return None
        if cache.numel() == 0 or cache.ndim < 3 or cache.shape[1] <= 0:
            return None
        slots = slot_mapping.reshape(-1).to(device=cache.device, dtype=torch.long)
        slots = slots[slots >= 0]
        if slots.numel() == 0:
            return None
        block_size = cache.shape[1]
        blocks = slots // block_size
        offsets = slots % block_size
        valid = blocks < cache.shape[0]
        if not valid.all():
            blocks = blocks[valid]
            offsets = offsets[valid]
        if blocks.numel() == 0:
            return None
        return cache, blocks, offsets, cache[blocks, offsets].clone()

    @staticmethod
    def _restore_cache_rows(
        saved: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        """Restore cache rows saved by :meth:`_save_cache_rows`."""
        for cache, blocks, offsets, values in reversed(saved):
            cache[blocks, offsets].copy_(values)

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

    def prepare_pool(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        t_proxy0 = time.perf_counter()
        self._wait_proxy_futures(layer_id)
        proxy_wait_ms = (time.perf_counter() - t_proxy0) * 1000.0
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
            if self._device.type == "cuda":
                t_sync0 = time.perf_counter()
                torch.cuda.synchronize(self._device)
                sync_ms += (time.perf_counter() - t_sync0) * 1000.0
            with self._lock:
                if self._ready_futures.get(layer_id) is ready_fut:
                    self._ready_futures[layer_id] = None
        t_drain0 = time.perf_counter()
        if drain_fut is not None:
            drain_fut.result(timeout=self._prefill_ready_timeout_s)
            with self._lock:
                if self._drain_futures.get(layer_id) is drain_fut:
                    self._drain_futures[layer_id] = None
        self._drain(layer_id)
        drain_ms = (time.perf_counter() - t_drain0) * 1000.0
        if self._device.type == "cuda":
            t_sync0 = time.perf_counter()
            torch.cuda.synchronize(self._device)
            sync_ms += (time.perf_counter() - t_sync0) * 1000.0
        pool = self._pools[layer_id]
        if layer_id not in self._debug_prepare_logged:
            self._debug_prepare_logged.add(layer_id)
            valid_slots = int((pool.pool_ids >= 0).sum().item())
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
            valid_slots=int((pool.pool_ids >= 0).sum().item()),
        )
        return pool.pool_tensor, pool.block_table

    def _wait_proxy_futures(self, layer_id: int) -> None:
        """Wait for side-stream residual proxy work targeting ``layer_id``."""
        with self._lock:
            futures = list(self._proxy_futures.get(layer_id, ()))
        for future in futures:
            future.result(timeout=self._prefill_ready_timeout_s)

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
        return translated.reshape(topk_pool_slots.shape).to(device=topk_pool_slots.device)

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
        if not _DECODE_PREFETCH_ENABLED:
            return
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
        should_prefill_correct = self.prefill_proxy_enabled() and self.layer_initialized(
            layer_id
        )
        should_collect = (
            should_profile
            or _timing_enabled()
            or _prefill_overlap_profile_enabled()
            or should_prefill_correct
        )
        if not should_collect:
            if first_log:
                self._debug_attention_topk_logged.add(layer_id)
                valid_slots = int((slot_ids_tensor >= 0).sum().item())
                cols = int(slot_ids_tensor.shape[1]) if slot_ids_tensor.ndim > 1 else 1
                logger.info(
                    "IndexerSSDManager: attention_true_topk layer %d "
                    "rows=%d cols=%d valid_slots=%d block_size=%d",
                    layer_id,
                    int(slot_ids_tensor.shape[0]),
                    cols,
                    valid_slots,
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
        predicted = self._last_proxy_topk.get(layer_id) or []
        predicted_set = set(predicted)
        spec_hits = len(true_set & predicted_set)
        true_misses = len(true_set - predicted_set)
        recall = float(spec_hits) / float(len(true_set)) if true_set else 0.0
        if should_prefill_correct and logical_ids:
            self._correct_prefill_true_topk(layer_id, logical_ids)
        self._prefill_overlap_profile_drain(layer_id, true_set)

        self._last_attention_true_ids[layer_id] = true_set
        self._last_attention_block_ids[layer_id] = block_ids

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
        if should_profile and true_set:
            self._attention_profile_seen += 1
            self._attention_profile_hits += spec_hits
            self._attention_profile_total += len(true_set)
            avg_recall = (
                float(self._attention_profile_hits)
                / float(self._attention_profile_total)
                if self._attention_profile_total > 0
                else 0.0
            )
            logger.info(
                "IndexerSSDManager: attention_true_topk_profile layer %d "
                "sample=%d true=%d predicted=%d spec_hits=%d true_misses=%d "
                "recall=%.4f avg_recall=%.4f physical_blocks=%d",
                layer_id,
                self._attention_profile_seen,
                len(true_set),
                len(predicted_set),
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
            true=len(true_set),
            predicted=len(predicted_set),
            spec_hits=spec_hits,
            true_misses=true_misses,
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

    def _prefill_overlap_profile_drain(
        self,
        layer_id: int,
        true_set: Set[int],
    ) -> None:
        """Drain prefill-overlap profile reads and log wait/hidden time."""
        if not _prefill_overlap_profile_enabled():
            return
        with self._lock:
            pending = self._prefill_overlap_pending[layer_id]
            self._prefill_overlap_pending[layer_id] = []
        if not pending:
            return

        t_wait0 = time.perf_counter()
        read_starts: List[float] = []
        read_ends: List[float] = []
        read_ms_values: List[float] = []
        token_ids: List[int] = []
        failed = 0
        for tid, future in pending:
            try:
                _data, read_start, read_end, read_ms = future.result()
            except Exception:
                failed += 1
                continue
            token_ids.append(tid)
            read_starts.append(read_start)
            read_ends.append(read_end)
            read_ms_values.append(read_ms)
        wait_ms = (time.perf_counter() - t_wait0) * 1000.0
        if read_starts and read_ends:
            async_span_ms = (max(read_ends) - min(read_starts)) * 1000.0
            first_submit_gap_ms = (t_wait0 - min(read_starts)) * 1000.0
        else:
            async_span_ms = 0.0
            first_submit_gap_ms = 0.0
        hidden_ms = max(0.0, async_span_ms - wait_ms)
        hidden_ratio = hidden_ms / async_span_ms if async_span_ms > 0 else 0.0
        token_set = set(token_ids)
        true_hits = len(token_set & true_set)
        true_misses = len(true_set - token_set) if true_set else 0
        self._log_timing(
            "prefill_overlap_drain",
            layer_id,
            wait_ms=f"{wait_ms:.3f}",
            async_span_ms=f"{async_span_ms:.3f}",
            window_ms=f"{first_submit_gap_ms:.3f}",
            hidden_ms=f"{hidden_ms:.3f}",
            hidden_ratio=f"{hidden_ratio:.4f}",
            read_ms_sum=f"{sum(read_ms_values):.3f}",
            read_ms_max=f"{max(read_ms_values) if read_ms_values else 0.0:.3f}",
            pending=len(pending),
            completed=len(token_ids),
            failed=failed,
            true=len(true_set),
            submitted_true_hits=true_hits,
            submitted_true_misses=true_misses,
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
            raise ValueError(
                f"Expected {self._token_bytes} bytes, got {len(data)}"
            )
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
                When None, sequential layout (token i = block i//bs, offset i%bs) is assumed.
            block_table: Optional tensor mapping logical block IDs to physical vLLM
                block IDs.  When provided, it is expanded into a slot mapping for
                the full compressed IndexerCache sequence.
            timing_event: Timing event name used to distinguish post-prefill
                eviction from LMCache-hit reuse seeding.
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

        t_read0 = time.perf_counter()
        token_bytes = self._read_packed_tokens(
            kv_cache_tensor,
            seq_len,
            slot_mapping,
        )
        read_ms = (time.perf_counter() - t_read0) * 1000.0

        # Write all tokens to SSD as one contiguous range.  Per-token pwrite
        # calls dominate long-prefill latency.
        t_write0 = time.perf_counter()
        store.write_tokens_contiguous(0, token_bytes.contiguous().numpy().tobytes())
        write_ms = (time.perf_counter() - t_write0) * 1000.0

        # Load seed tokens into HBM pool first
        t_load0 = time.perf_counter()
        load_ids: List[int] = []
        if seed_token_ids:
            load_ids = seed_token_ids[: self._pool_size]
        if len(load_ids) < self._pool_size:
            seed_set = set(load_ids)
            extra = [tid for tid in range(seq_len) if tid not in seed_set]
            load_ids += extra[: self._pool_size - len(load_ids)]

        if load_ids:
            pool.load_tokens(load_ids, token_bytes[load_ids].contiguous())

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
                "IndexerSSDManager: %s complete layer %d seq_len=%d "
                "seed_tokens=%d",
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
        )

    def submit_seed_after_reuse(
        self,
        layer_id: int,
        kv_cache_cpu: torch.Tensor,
        seq_len: int,
        seed_token_ids: Optional[List[int]] = None,
        slot_mapping_cpu: Optional[torch.Tensor] = None,
        block_table_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        """Submit CSA SSD/HBM pool initialization after LMCache full-hit reuse.

        Args:
            layer_id: CSA layer index.
            kv_cache_cpu: Retrieved IndexerCache tensor already on CPU.
            seq_len: Number of valid compressed CSA tokens in the reused prefix.
            seed_token_ids: Token IDs to prioritize in the HBM pool.
            slot_mapping_cpu: Slot mapping tensor already on CPU.
            block_table_cpu: Block table tensor already on CPU.
        """
        self.submit_evict_after_prefill(
            layer_id,
            kv_cache_cpu,
            seq_len,
            seed_token_ids,
            slot_mapping_cpu,
            block_table_cpu,
            timing_event="reuse_prefetch_seed",
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
                )
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
            self._ready_futures[layer_id] = future
        logger.info(
            "IndexerSSDManager: submit_%s layer %d seq_len=%d "
            "seed_tokens=%d",
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
        scale_start = block_size * self._head_dim + (
            block_offset * self._scale_bytes
        )
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

        value_offsets = (
            block_offsets.unsqueeze(1) * self._head_dim
            + torch.arange(self._head_dim, dtype=torch.long, device=device)
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
                logger.warning(
                    "IndexerSSDManager: async read failed for token %d layer %d: %r",
                    tid, layer_id, exc,
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
                self._drain_futures[lid] = None
                self._proxy_futures[lid] = []
                self._decode_cursor[lid] = 0

    def close(self) -> None:
        """Shut down I/O thread pool and close SSD files."""
        if self._proxy_executor is not None:
            self._proxy_executor.shutdown(wait=True)
        self._executor.shutdown(wait=True)
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
                # Extract layer index from prefix like "model.layers.3.self_attn.indexer.k_cache"
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
