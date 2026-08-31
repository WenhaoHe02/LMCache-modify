# SPDX-License-Identifier: Apache-2.0
"""Rank-local prefill scorer used only by speculative CSA prefetch."""

# Standard
import os
import threading
from typing import Any, Literal, cast

# Third Party
import torch


_PROXY_K_WORKSPACE_LOCK = threading.Lock()
_PROXY_K_WORKSPACES: dict[
    tuple[str, int | None, int, int, bool, torch.dtype, int],
    tuple[torch.Tensor, torch.Tensor],
] = {}

PrefillCPMode = Literal[
    "query",
    "key_history_only",
    "key_sharded_early",
    "key_sharded_union",
    "key_sharded_local",
    "key_contiguous_union",
    "key_contiguous_replicated_append",
    "key_sharded_owner",
    "key_contiguous_owner",
    "key_replicated_append",
    "key_sharded_append",
]
_PREFILL_CP_MODES = frozenset(
    {
        "query",
        "key_history_only",
        "key_sharded_early",
        "key_sharded_union",
        "key_sharded_local",
        "key_contiguous_union",
        "key_contiguous_replicated_append",
        "key_sharded_owner",
        "key_contiguous_owner",
        "key_replicated_append",
        "key_sharded_append",
    }
)


def resolve_prefill_cp_mode(value: str | None = None) -> PrefillCPMode:
    """Resolve the opt-in speculative CP partition mode.

    Args:
        value: Explicit mode, or ``None`` to read
            ``LMCACHE_CSA_PREFETCH_CP_MODE``.

    Returns:
        Validated partition mode. The default remains ``query``.

    Raises:
        ValueError: If the configured mode is unsupported.
    """
    configured = (
        os.getenv("LMCACHE_CSA_PREFETCH_CP_MODE", "query") if value is None else value
    )
    normalized = configured.strip().lower()
    if normalized not in _PREFILL_CP_MODES:
        supported = ", ".join(sorted(_PREFILL_CP_MODES))
        raise ValueError(
            f"unsupported speculative CP mode {configured!r}; expected {supported}"
        )
    return cast(PrefillCPMode, normalized)


def prefill_cp_requires_id_exchange(value: str | None = None) -> bool:
    """Return whether a partition mode requires cross-rank candidate union."""
    return resolve_prefill_cp_mode(value) in {
        "query",
        "key_sharded_union",
        "key_sharded_local",
        "key_contiguous_union",
        "key_contiguous_replicated_append",
    }


def prefill_cp_reads_rank_local(value: str | None = None) -> bool:
    """Return whether a partition mode keeps prediction I/O fully rank-local.

    The rank-local mode never enters a collective on the prediction path:
    each rank scores its own K shard, reads only its own candidate blocks
    from its own SSD replica, and lets authoritative miss correction cover
    every block the local shard could not predict. It requires byte-identical
    CSA replicas across ranks (``LMCACHE_SSD_TP_CSA_REPLICA_VERIFIED``).

    Args:
        value: Explicit mode, or ``None`` to read
            ``LMCACHE_CSA_PREFETCH_CP_MODE``.

    Returns:
        ``True`` only for ``key_sharded_local``.
    """
    return resolve_prefill_cp_mode(value) == "key_sharded_local"


PredictionGatePolicy = Literal["join", "fail_open"]
_PREDICTION_GATE_POLICIES = frozenset({"join", "fail_open"})


def resolve_prediction_gate_policy(
    value: str | None = None,
) -> PredictionGatePolicy:
    """Resolve how the target-layer gate treats an unfinished prediction.

    ``join`` preserves the deployed behavior: the gate blocks until proxy
    scoring and prediction I/O futures complete, so the resident view is
    final before miss filtering. ``fail_open`` never blocks on speculative
    scoring: the gate closes the layer's submission window, waits only for
    prediction I/O that already reached the storage layer, and lets
    authoritative miss correction demand-read everything else.

    ``fail_open`` is valid only for prediction paths without a deferred
    KV-row collective, because skipping a collective on a subset of ranks
    would deadlock the transport. The rank-local partition mode satisfies
    this by construction.

    Args:
        value: Explicit policy, or ``None`` to read
            ``LMCACHE_CSA_PREDICTION_GATE``. When the variable is unset, the
            policy defaults to ``fail_open`` for the rank-local partition
            mode and ``join`` otherwise.

    Returns:
        Validated gate policy.

    Raises:
        ValueError: If the configured policy is unsupported.
    """
    configured = (
        os.getenv("LMCACHE_CSA_PREDICTION_GATE", "") if value is None else value
    )
    normalized = configured.strip().lower()
    if not normalized:
        return "fail_open" if prefill_cp_reads_rank_local() else "join"
    if normalized not in _PREDICTION_GATE_POLICIES:
        supported = ", ".join(sorted(_PREDICTION_GATE_POLICIES))
        raise ValueError(
            f"unsupported prediction gate policy {configured!r}; expected {supported}"
        )
    return cast(PredictionGatePolicy, normalized)


def _private_proxy_k_workspace(
    *,
    device: torch.device,
    total_capacity: int,
    head_dim: int,
    use_fp4: bool,
    fp8_dtype: torch.dtype,
    slot: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K scratch owned exclusively by the speculative proxy path.

    Args:
        device: CUDA device that owns the workspace.
        total_capacity: Maximum K rows the workspace must hold.
        head_dim: Indexer head dimension.
        use_fp4: Whether the packed K layout is FP4.
        fp8_dtype: Platform FP8 dtype for the non-FP4 layout.
        slot: Concurrency slot. Callers that score different target layers
            concurrently must pass distinct slots so one scorer's gather
            cannot overwrite another's K rows mid-kernel.

    Returns:
        Value and scale tensors reused across calls with the same key.
    """
    key = (
        device.type,
        device.index,
        int(total_capacity),
        int(head_dim),
        bool(use_fp4),
        fp8_dtype,
        int(slot),
    )
    with _PROXY_K_WORKSPACE_LOCK:
        cached = _PROXY_K_WORKSPACES.get(key)
        if cached is not None:
            return cached
        if use_fp4:
            values = torch.empty(
                (total_capacity, head_dim // 2),
                dtype=torch.uint8,
                device=device,
            )
            scales = torch.empty(
                (total_capacity, head_dim // 32),
                dtype=torch.uint8,
                device=device,
            )
        else:
            values = torch.empty(
                (total_capacity, head_dim),
                dtype=fp8_dtype,
                device=device,
            )
            scales = torch.empty(
                (total_capacity, 4),
                dtype=torch.uint8,
                device=device,
            )
        cached = (values, scales)
        _PROXY_K_WORKSPACES[key] = cached
        return cached


def _block_cyclic_indices(
    length: int,
    rank: int,
    world_size: int,
    interleave_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build one rank's block-cyclic offsets without a dynamic CUDA shape."""
    cycle = int(world_size) * int(interleave_size)
    rank_start = int(rank) * int(interleave_size)
    full_cycles, remainder = divmod(int(length), cycle)
    tail = max(0, min(int(interleave_size), remainder - rank_start))
    count = full_cycles * int(interleave_size) + tail
    if count == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    local = torch.arange(count, dtype=torch.int64, device=device)
    return (
        torch.div(local, int(interleave_size), rounding_mode="floor") * cycle
        + rank_start
        + torch.remainder(local, int(interleave_size))
    )


def prefill_cp_key_indices(
    total_tokens: int,
    rank: int,
    world_size: int,
    interleave_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return global K-token ids owned by one speculative prefill CP rank.

    Args:
        total_tokens: Number of gathered indexer K rows.
        rank: Rank in the speculative prefetch group.
        world_size: Number of ranks sharing the proxy scorer.
        interleave_size: Consecutive K rows assigned to one rank at a time.
        device: Device on which to construct the indices.

    Returns:
        Sorted global row indices assigned in block-cyclic order.

    Raises:
        ValueError: If a partition argument is invalid.
    """
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be within the prefetch CP group")
    if interleave_size <= 0:
        raise ValueError("interleave_size must be positive")
    return _block_cyclic_indices(
        total_tokens,
        rank,
        world_size,
        interleave_size,
        device,
    )


def prefill_cp_key_indices_with_append(
    total_tokens: int,
    append_start: int,
    rank: int,
    world_size: int,
    interleave_size: int,
    device: torch.device,
    *,
    replicate_append: bool,
) -> torch.Tensor:
    """Return one rank's K rows for a history-plus-append prefill.

    History is always block-cyclically sharded. Append K rows are either
    replicated on every rank or sharded with the same global ownership rule.

    Args:
        total_tokens: Total K rows visible to the active prefill chunk.
        append_start: First K row produced by the active append chunk.
        rank: Rank in the speculative prefetch group.
        world_size: Number of ranks sharing proxy scoring.
        interleave_size: Consecutive K rows assigned to one rank.
        device: Device on which to construct the indices.
        replicate_append: Whether every rank keeps all append K rows.

    Returns:
        Sorted global K-row ids owned by this rank.

    Raises:
        ValueError: If ``append_start`` is outside ``[0, total_tokens]`` or a
            partition argument is invalid.
    """
    if append_start < 0 or append_start > total_tokens:
        raise ValueError("append_start must be within the K-token range")
    if not replicate_append:
        return prefill_cp_key_indices(
            total_tokens,
            rank,
            world_size,
            interleave_size,
            device,
        )
    history = prefill_cp_key_indices(
        append_start,
        rank,
        world_size,
        interleave_size,
        device,
    )
    append = torch.arange(
        append_start,
        total_tokens,
        dtype=torch.int64,
        device=device,
    )
    return torch.cat((history, append))


def gather_prefill_cp_local_k_rows(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    global_key_indices: torch.Tensor,
    *,
    value_bytes: int,
    scale_bytes: int,
    value_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only one speculative rank's K rows from a paged cache.

    Args:
        kv_cache: Byte-sized paged indexer cache shaped
            ``[blocks, block_size, stride]``. Each native block stores all K
            value bytes first, followed by all scale bytes (not interleaved).
        block_table: Single-sequence logical-to-physical block table.
        global_key_indices: Sorted global token rows owned by this rank.
        value_bytes: Packed value bytes per cache row.
        scale_bytes: Packed scale bytes per cache row.
        value_dtype: Dtype used to reinterpret the packed value bytes.

    Returns:
        Contiguous value and scale tensors containing only the selected rows.

    Raises:
        ValueError: If a tensor layout or byte width is unsupported.

    Notes:
        This performs no cross-rank communication and does not materialize the
        full logical K sequence before selecting the local partition.
    """
    if kv_cache.ndim != 3:
        raise ValueError("kv_cache must have block, row, and byte dimensions")
    if kv_cache.element_size() != 1:
        raise ValueError("paged K-cache storage must be byte-sized")
    if kv_cache.stride(2) != 1 or kv_cache.stride(1) != kv_cache.shape[2]:
        raise ValueError("each paged K-cache block must be contiguous")
    if block_table.ndim != 2 or int(block_table.shape[0]) != 1:
        raise ValueError("block_table must describe exactly one sequence")
    if global_key_indices.ndim != 1:
        raise ValueError("global_key_indices must be one-dimensional")
    if value_bytes <= 0 or scale_bytes <= 0:
        raise ValueError("K row byte widths must be positive")
    row_bytes = value_bytes + scale_bytes
    if int(kv_cache.shape[2]) < row_bytes:
        raise ValueError("paged K-cache row is smaller than the requested layout")

    block_size = int(kv_cache.shape[1])
    logical_blocks = torch.div(
        global_key_indices,
        block_size,
        rounding_mode="floor",
    ).to(torch.int64)
    block_offsets = torch.remainder(global_key_indices, block_size).to(torch.int64)
    physical_blocks = block_table[0].index_select(0, logical_blocks).to(torch.int64)
    # Native indexer_k_quant_and_cache packs separate value/scale planes.
    # These are views: gather only the selected rows, never entire K blocks.
    num_blocks = int(kv_cache.shape[0])
    packed_blocks = kv_cache.view(num_blocks, block_size * kv_cache.shape[2])
    values_end = block_size * value_bytes
    value_plane = packed_blocks[:, :values_end].view(
        num_blocks, block_size, value_bytes
    )
    scale_plane = packed_blocks[:, values_end : block_size * row_bytes].view(
        num_blocks, block_size, scale_bytes
    )
    values = value_plane[physical_blocks, block_offsets].contiguous()
    if values.dtype != value_dtype:
        values = values.view(value_dtype)
    scales = scale_plane[physical_blocks, block_offsets].contiguous()
    return values, scales


def prefill_cp_key_block_partition(
    total_tokens: int,
    rank: int,
    world_size: int,
    interleave_blocks: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one rank's logical blocks and expanded global K rows.

    Args:
        total_tokens: Number of valid logical K rows.
        rank: Rank in the speculative prefetch group.
        world_size: Number of ranks sharing K scoring.
        interleave_blocks: Consecutive cache blocks assigned to one rank.
        block_size: Token rows stored in each cache block.
        device: Device on which to construct the partition.

    Returns:
        Selected logical block ids and their valid global token-row ids.

    Raises:
        ValueError: If a length or partition argument is invalid.
    """
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    num_blocks = (total_tokens + block_size - 1) // block_size
    logical_blocks = _block_cyclic_indices(
        num_blocks,
        rank,
        world_size,
        interleave_blocks,
        device,
    )
    offsets = torch.arange(block_size, dtype=torch.int64, device=device)
    global_rows = (logical_blocks[:, None] * block_size + offsets).reshape(-1)
    return logical_blocks, global_rows[global_rows < total_tokens]


def prefill_cp_contiguous_key_block_partition(
    total_tokens: int,
    rank: int,
    world_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one rank's contiguous, non-overlapping K-block partition.

    Args:
        total_tokens: Number of valid logical K rows.
        rank: Rank in the speculative prefetch group.
        world_size: Number of ranks sharing K scoring.
        block_size: Token rows stored in each cache block.
        device: Device on which to construct the partition.

    Returns:
        A contiguous logical-block slice and its valid global token-row ids.

    Raises:
        ValueError: If a length or partition argument is invalid.

    Notes:
        Uneven tails are assigned to the lowest ranks, so all blocks are
        covered exactly once without padding or cross-rank synchronization.
    """
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be within the prefetch CP group")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    num_blocks = (total_tokens + block_size - 1) // block_size
    blocks_per_rank, remainder = divmod(num_blocks, world_size)
    block_start = rank * blocks_per_rank + min(rank, remainder)
    block_count = blocks_per_rank + int(rank < remainder)
    logical_blocks = torch.arange(
        block_start,
        block_start + block_count,
        dtype=torch.int64,
        device=device,
    )
    offsets = torch.arange(block_size, dtype=torch.int64, device=device)
    global_rows = (logical_blocks[:, None] * block_size + offsets).reshape(-1)
    return logical_blocks, global_rows[global_rows < total_tokens]


def prefill_cp_contiguous_history_with_replicated_append(
    total_tokens: int,
    append_start: int,
    rank: int,
    world_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition history contiguously while replicating append K blocks.

    Args:
        total_tokens: Total history-plus-append K rows.
        append_start: First append K row.
        rank: Rank in the speculative prefetch group.
        world_size: Number of ranks sharing K scoring.
        block_size: Token rows stored in each cache block.
        device: Device on which to construct the partition.

    Returns:
        Logical blocks to gather and their sorted global token-row ids.

    Raises:
        ValueError: If the range is invalid or append does not begin on a
            cache-block boundary.

    Notes:
        The block-alignment requirement matches LMCache's admitted prefix
        granularity and keeps the direct paged-cache gather copy-free.
    """
    if append_start < 0 or append_start > total_tokens:
        raise ValueError("append_start must be within the K-token range")
    if append_start % block_size != 0:
        raise ValueError("replicated append must begin on a K-cache block")
    history_blocks, history_rows = prefill_cp_contiguous_key_block_partition(
        append_start,
        rank,
        world_size,
        block_size,
        device,
    )
    append_block_start = append_start // block_size
    total_blocks = (total_tokens + block_size - 1) // block_size
    append_blocks = torch.arange(
        append_block_start,
        total_blocks,
        dtype=torch.int64,
        device=device,
    )
    offsets = torch.arange(block_size, dtype=torch.int64, device=device)
    append_rows = (append_blocks[:, None] * block_size + offsets).reshape(-1)
    append_rows = append_rows[append_rows < total_tokens]
    return (
        torch.cat((history_blocks, append_blocks)),
        torch.cat((history_rows, append_rows)),
    )


def prefill_cp_query_indices(
    token_start: int,
    token_end: int,
    rank: int,
    world_size: int,
    interleave_size: int,
    device: torch.device,
    sample_stride: int = 1,
) -> torch.Tensor:
    """Return query rows owned by one speculative prefill rank.

    Args:
        token_start: Inclusive first proxy query row.
        token_end: Exclusive final proxy query row.
        rank: Rank in the speculative prefetch group.
        world_size: Number of ranks sharing proxy scoring.
        interleave_size: Consecutive query rows assigned to one rank.
        device: Device on which to construct the indices.
        sample_stride: Fractional query sampling stride. A value of two scores
            one half of the rows while keeping the work balanced across ranks.

    Returns:
        Sorted query-row indices assigned in block-cyclic order.

    Raises:
        ValueError: If a range or partition argument is invalid.
    """
    if token_start < 0 or token_end < token_start:
        raise ValueError("invalid query token range")
    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be within the prefetch CP group")
    if interleave_size <= 0:
        raise ValueError("interleave_size must be positive")
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    return _block_cyclic_indices(
        token_end - token_start,
        rank * sample_stride,
        world_size * sample_stride,
        interleave_size,
        device,
    ).add_(int(token_start))


def prefill_cp_sampled_query_indices(
    token_start: int,
    token_end: int,
    sample_stride: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the common sampled Q rows used by every K-sharded rank.

    Args:
        token_start: Inclusive first global query-token offset.
        token_end: Exclusive final global query-token offset.
        sample_stride: Keep one query row per this many rows.
        device: Device on which to construct the indices.

    Returns:
        Sampled global query-token ids.

    Raises:
        ValueError: If the token range or stride is invalid.
    """
    if token_start < 0 or token_end < token_start:
        raise ValueError("invalid query token range")
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    return torch.arange(
        token_start,
        token_end,
        sample_stride,
        dtype=torch.int64,
        device=device,
    )


def globalize_prefill_cp_topk(
    local_topk: torch.Tensor,
    global_key_indices: torch.Tensor,
) -> torch.Tensor:
    """Map rank-local top-k column ids back to global K-token ids.

    Args:
        local_topk: Top-k indices produced against the rank-local K matrix.
        global_key_indices: Global K row corresponding to each local column.

    Returns:
        Tensor with the same shape and dtype as ``local_topk``. Invalid
        negative or out-of-range entries become ``-1``.
    """
    result = torch.full_like(local_topk, -1)
    valid = (local_topk >= 0) & (local_topk < global_key_indices.shape[0])
    if global_key_indices.numel() > 0:
        result[valid] = global_key_indices[local_topk[valid].long()].to(result.dtype)
    return result


def localize_prefill_cp_bounds(
    global_key_indices: torch.Tensor,
    global_starts: torch.Tensor,
    global_ends: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global causal K bounds into one sorted rank-local K matrix.

    Args:
        global_key_indices: Sorted global K ids present on the rank.
        global_starts: Inclusive global K starts for query rows.
        global_ends: Exclusive global K ends for query rows.

    Returns:
        Inclusive starts and exclusive ends in the compact local K matrix,
        preserving the input bounds' dtype.

    Raises:
        ValueError: If an input is not one-dimensional or the bound shapes
            differ.
    """
    if global_key_indices.ndim != 1:
        raise ValueError("global_key_indices must be one-dimensional")
    if global_starts.ndim != 1 or global_ends.ndim != 1:
        raise ValueError("prefill K bounds must be one-dimensional")
    if global_starts.shape != global_ends.shape:
        raise ValueError("prefill K bound shapes must match")
    local_starts = torch.searchsorted(
        global_key_indices,
        global_starts.to(torch.int64),
        right=False,
    ).to(global_starts.dtype)
    local_ends = torch.searchsorted(
        global_key_indices,
        global_ends.to(torch.int64),
        right=False,
    ).to(global_ends.dtype)
    return local_starts, local_ends


def prefill_cp_local_topk_tokens(
    global_topk_tokens: int,
    world_size: int,
    oversubscribe: int = 1,
) -> int:
    """Return each rank's local candidate quota for a bounded union.

    Args:
        global_topk_tokens: Original full-K top-k width.
        world_size: Number of speculative prefill CP ranks.
        oversubscribe: Integer multiplier used to trade I/O for recall.

    Returns:
        Per-rank top-k width, capped by the original global width.

    Raises:
        ValueError: If an argument is not positive.
    """
    if global_topk_tokens <= 0:
        raise ValueError("global_topk_tokens must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if oversubscribe <= 0:
        raise ValueError("oversubscribe must be positive")
    quota = (global_topk_tokens + world_size - 1) // world_size
    return min(global_topk_tokens, quota * oversubscribe)


def prefill_cp_query_ranges(
    token_start: int,
    token_end: int,
    local_k_tokens: int,
    max_logits_bytes: int,
    logits_element_size: int = 4,
) -> list[tuple[int, int]]:
    """Split contiguous proxy queries against the rank-local K matrix.

    Args:
        token_start: Inclusive first query-token offset.
        token_end: Exclusive final query-token offset.
        local_k_tokens: Number of K rows owned by this rank.
        max_logits_bytes: Maximum temporary logits storage for one call.
        logits_element_size: Bytes used by one logits element. DeepGEMM's
            prefill MQA logits output uses four-byte elements.

    Returns:
        Contiguous ``(start, end)`` token ranges covering the input exactly.

    Raises:
        ValueError: If a range or size is invalid, or the budget cannot hold
            one query row.

    Notes:
        The official indexer chunks queries using the full K width. Speculative
        CP owns only a subset of K, so this function recomputes larger query
        ranges without modifying the official metadata.
    """
    if token_start < 0 or token_end < token_start:
        raise ValueError("invalid query token range")
    if local_k_tokens <= 0:
        raise ValueError("local_k_tokens must be positive")
    if max_logits_bytes <= 0:
        raise ValueError("max_logits_bytes must be positive")
    if logits_element_size <= 0:
        raise ValueError("logits_element_size must be positive")

    bytes_per_query = local_k_tokens * logits_element_size
    max_query_tokens = max_logits_bytes // bytes_per_query
    if max_query_tokens <= 0:
        raise ValueError("CP logits budget cannot hold one query row")
    return [
        (start, min(start + max_query_tokens, token_end))
        for start in range(token_start, token_end, max_query_tokens)
    ]


def _prefill_cp_max_logits_bytes(vllm_default_mb: int) -> int:
    configured = os.getenv(
        "LMCACHE_CSA_PREFETCH_CP_MAX_LOGITS_MB",
        str(vllm_default_mb),
    )
    try:
        max_logits_mb = int(configured)
    except ValueError as exc:
        raise RuntimeError(
            "LMCACHE_CSA_PREFETCH_CP_MAX_LOGITS_MB must be an integer"
        ) from exc
    if max_logits_mb <= 0:
        raise RuntimeError("LMCACHE_CSA_PREFETCH_CP_MAX_LOGITS_MB must be positive")
    return max_logits_mb * 1024 * 1024


def score_prefill_proxy_rank_local(
    indexer_op: Any,
    hidden_states: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    output: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    interleave_size: int,
    oversubscribe: int = 1,
    partition_mode: PrefillCPMode | str = "query",
    workspace_slot: int = 0,
    topk_tokens_override: int | None = None,
    metadata_query_row_start: int | None = None,
    preselected_query_rows: torch.Tensor | None = None,
    preselected_query_span: int | None = None,
    runtime_info: dict[str, int] | None = None,
) -> torch.Tensor:
    """Run one optional speculative query- or K-sharded indexer path.

    The official indexer op and its metadata are not modified. This function
    mirrors its prefill gather/scoring path. ``query`` preserves the existing
    query-sharded behavior. The ``key_*`` modes use the same sampled query rows
    on every rank, score only one rank's K partition, emit a bounded local
    candidate set, and intentionally avoid an exact global Top-K
    synchronization. ``key_history_only`` excludes append K because append KV
    is already resident and therefore cannot create an SSD miss.
    ``key_sharded_early`` retains the original fully sharded K candidate set
    while moving shared query sampling ahead of proxy Q preparation.
    ``key_sharded_union`` exchanges only the bounded approximate candidate
    IDs. ``key_contiguous_union`` does the same exchange after assigning each
    rank one contiguous K slice. ``key_contiguous_replicated_append`` keeps
    those contiguous history shards but scores the complete append on every
    rank. The two owner modes leave IDs rank-local so each rank can read its
    own KV rows before the consumer-gate AllGather; ``key_contiguous_owner``
    uses contiguous instead of block-cyclic K ownership.

    Args:
        indexer_op: Active vLLM ``SparseAttnIndexer`` instance.
        hidden_states: Proxy hidden rows for the current prefill chunk.
        q_quant: Quantized proxy queries, optionally with FP4 scales.
        weights: Per-query indexer head weights.
        output: Private top-k output buffer.
        rank: Rank in the speculative prefetch group.
        world_size: Number of ranks sharing proxy scoring.
        interleave_size: Consecutive query rows assigned per rank.
        oversubscribe: Multiplier for the per-rank K-shard candidate quota.
            Query sharding continues to use the full proxy top-k width.
        partition_mode: ``query`` (existing behavior),
            ``key_history_only`` (sharded history K with append excluded),
            ``key_sharded_early`` (sharded history and append K with early Q),
            ``key_sharded_union`` (direct K shards plus approximate ID union),
            ``key_contiguous_union`` (contiguous K shards plus ID union),
            ``key_contiguous_replicated_append`` (contiguous history shards,
            replicated append, and ID union),
            ``key_sharded_owner`` (direct K shards plus owner-local KV reads),
            ``key_contiguous_owner`` (contiguous K shards plus owner-local
            KV reads),
            ``key_sharded_local`` (direct K shards with fully rank-local
            SSD reads and no prediction-path collective),
            ``key_replicated_append`` (sharded history plus full append K), or
            ``key_sharded_append`` (sharded history and append K).
        topk_tokens_override: Optional wider speculative top-k width. This
            changes only proxy coverage; the official indexer's output width
            and sparse-attention semantics remain unchanged.
        metadata_query_row_start: Optional row offset into the active prefill
            metadata. When set, ``q_quant``, ``weights``, and ``output`` are a
            compact slice beginning at this metadata row. This is used only
            to warm the exact cache-hit proxy shape during cold admission.
        preselected_query_rows: Optional original row indices represented by
            compact proxy inputs. When supplied, expensive HC/Q preparation
            has already been restricted to this rank's sampled query rows.
        preselected_query_span: Original row count before compact sampling.
        runtime_info: Optional output mapping populated with resolved scoring
            dimensions for cold-admission warmup bookkeeping.

    Returns:
        ``output`` filled with global token ids for this rank's candidates.

    Raises:
        RuntimeError: If active metadata is not a supported prefill-only,
            single-sequence indexer request.
    """
    # Third Party
    from vllm import _custom_ops as ops
    from vllm import envs
    from vllm.forward_context import get_forward_context
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    from vllm.utils.torch_utils import _encode_layer_name, _resolve_layer_name
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadata,
    )

    metadata_by_layer = get_forward_context().attn_metadata
    if not isinstance(metadata_by_layer, dict):
        raise RuntimeError("prefill CP proxy requires active indexer metadata")
    prefix = _resolve_layer_name(_encode_layer_name(indexer_op.k_cache.prefix))
    metadata = metadata_by_layer[prefix]
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        raise RuntimeError("unsupported indexer metadata for prefill CP proxy")
    if metadata.num_prefills <= 0 or metadata.num_decodes > 0:
        raise RuntimeError("prefill CP proxy only supports prefill-only batches")
    prefill = metadata.prefill
    if prefill is None:
        raise RuntimeError("prefill CP proxy metadata is missing")

    if isinstance(q_quant, tuple):
        q_values, q_scale = q_quant
    else:
        q_values, q_scale = q_quant, None
    use_fp4 = bool(getattr(indexer_op, "use_fp4_cache", False))
    if use_fp4 != (q_scale is not None):
        raise RuntimeError("proxy Q format does not match the indexer K cache")

    output[: hidden_states.shape[0]] = -1
    if oversubscribe <= 0:
        raise RuntimeError("prefill CP oversubscribe must be positive")
    mode = resolve_prefill_cp_mode(str(partition_mode))
    topk_tokens = (
        int(topk_tokens_override)
        if topk_tokens_override is not None
        else int(indexer_op.topk_tokens)
    )
    if topk_tokens <= 0:
        raise RuntimeError("proxy top-k width must be positive")
    required_output_width = topk_tokens
    if mode != "query":
        required_output_width = prefill_cp_local_topk_tokens(
            topk_tokens,
            world_size,
            oversubscribe,
        )
    if required_output_width > int(output.shape[1]):
        raise RuntimeError("proxy top-k width does not fit the output buffer")
    fp8_dtype = current_platform.fp8_dtype()
    head_dim = int(indexer_op.head_dim)

    chunks = prefill.chunks
    if not chunks:
        return output
    first_token = int(chunks[0].token_start)
    final_token = first_token
    total_seq_lens = int(chunks[0].total_seq_lens)
    if runtime_info is not None:
        runtime_info["total_seq_lens"] = total_seq_lens
    gather_chunk = None
    global_ks_parts = []
    global_ke_parts = []
    for chunk in chunks:
        if int(chunk.cu_seq_lens.shape[0]) != 2:
            raise RuntimeError("prefill CP proxy currently requires one sequence")
        if int(chunk.total_seq_lens) != total_seq_lens:
            raise RuntimeError("prefill CP query chunks must share one K range")
        token_start = int(chunk.token_start)
        token_end = int(chunk.token_end)
        if token_start != final_token or token_end < token_start:
            raise RuntimeError("prefill CP query chunks must be contiguous")
        if int(chunk.cu_seqlen_ks.numel()) != token_end - token_start:
            raise RuntimeError("prefill CP K-start metadata does not match queries")
        if int(chunk.cu_seqlen_ke.numel()) != token_end - token_start:
            raise RuntimeError("prefill CP K-end metadata does not match queries")
        final_token = token_end
        global_ks_parts.append(chunk.cu_seqlen_ks)
        global_ke_parts.append(chunk.cu_seqlen_ke)
        if not chunk.skip_kv_gather:
            if gather_chunk is not None:
                raise RuntimeError("prefill CP metadata requests multiple K gathers")
            gather_chunk = chunk
    if gather_chunk is None:
        raise RuntimeError("prefill CP metadata has no K-gather chunk")

    global_ks = torch.cat(global_ks_parts)
    global_ke = torch.cat(global_ke_parts)
    query_first = first_token
    query_final = final_token
    compact_query_base = 0
    if metadata_query_row_start is not None:
        if metadata_query_row_start < 0:
            raise RuntimeError("metadata query row start must be non-negative")
        query_first = first_token + metadata_query_row_start
        query_final = query_first + int(
            preselected_query_span
            if preselected_query_span is not None
            else hidden_states.shape[0]
        )
        if query_final > final_token:
            raise RuntimeError("compact proxy query slice exceeds prefill metadata")
        compact_query_base = query_first
    try:
        query_sample_stride = int(
            os.getenv("LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE", "1")
        )
    except ValueError as exc:
        raise RuntimeError(
            "LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE must be an integer"
        ) from exc
    if query_sample_stride <= 0:
        raise RuntimeError(
            "LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE must be positive"
        )
    if preselected_query_rows is not None:
        if preselected_query_rows.ndim != 1:
            raise RuntimeError("preselected proxy query rows must be one-dimensional")
        if preselected_query_rows.device != hidden_states.device:
            raise RuntimeError("preselected proxy query rows must use the proxy device")
        if preselected_query_rows.numel() != hidden_states.shape[0]:
            raise RuntimeError(
                "preselected proxy query rows do not match compact inputs"
            )
        if preselected_query_rows.numel() > 0:
            first_row = int(preselected_query_rows[0].item())
            final_row = int(preselected_query_rows[-1].item())
            if first_row < 0 or final_row >= query_final - query_first:
                raise RuntimeError("preselected proxy query row is outside metadata")
        local_query_ids = preselected_query_rows.to(torch.int64) + query_first
        input_query_rows = torch.arange(
            int(preselected_query_rows.numel()),
            dtype=torch.int64,
            device=hidden_states.device,
        )
    elif mode != "query":
        local_query_ids = prefill_cp_sampled_query_indices(
            query_first,
            query_final,
            query_sample_stride,
            hidden_states.device,
        )
        input_query_rows = local_query_ids - query_first
    else:
        local_query_ids = prefill_cp_query_indices(
            query_first,
            query_final,
            rank,
            world_size,
            interleave_size,
            hidden_states.device,
            sample_stride=query_sample_stride,
        )
        input_query_rows = local_query_ids - compact_query_base
    if runtime_info is not None:
        runtime_info["query_sample_stride"] = query_sample_stride
        runtime_info["local_query_rows"] = int(local_query_ids.numel())
    if local_query_ids.numel() == 0:
        return output

    if mode != "query":
        append_rows = final_token - first_token
        append_start = total_seq_lens - append_rows
        if append_start < 0:
            raise RuntimeError("prefill metadata has more append rows than K rows")
        if mode in {"key_history_only", "key_sharded_early"}:
            partition_tokens = (
                append_start if mode == "key_history_only" else total_seq_lens
            )
            global_key_indices = prefill_cp_key_indices(
                partition_tokens,
                rank,
                world_size,
                interleave_size,
                hidden_states.device,
            )
            if global_key_indices.numel() == 0:
                return output
            total_capacity = int(indexer_op.max_total_seq_len)
            k_quant_full, k_scale_full = _private_proxy_k_workspace(
                device=hidden_states.device,
                total_capacity=total_capacity,
                head_dim=head_dim,
                use_fp4=use_fp4,
                fp8_dtype=fp8_dtype,
                slot=workspace_slot,
            )
            gathered_k_quant = k_quant_full[:total_seq_lens]
            gathered_k_scale = k_scale_full[:total_seq_lens]
            ops.cp_gather_indexer_k_quant_cache(
                indexer_op.k_cache.kv_cache,
                gathered_k_quant,
                gathered_k_scale,
                gather_chunk.block_table,
                gather_chunk.cu_seq_lens,
            )
            k_quant = gathered_k_quant.index_select(0, global_key_indices)
            k_scale = gathered_k_scale.index_select(0, global_key_indices)
        elif mode in {
            "key_sharded_append",
            "key_sharded_union",
            "key_sharded_local",
            "key_contiguous_union",
            "key_contiguous_replicated_append",
            "key_sharded_owner",
            "key_contiguous_owner",
        }:
            cache_block_size = int(indexer_op.k_cache.kv_cache.shape[1])
            if mode in {"key_contiguous_union", "key_contiguous_owner"}:
                logical_blocks, global_key_indices = (
                    prefill_cp_contiguous_key_block_partition(
                        total_seq_lens,
                        rank,
                        world_size,
                        cache_block_size,
                        hidden_states.device,
                    )
                )
            elif mode == "key_contiguous_replicated_append":
                logical_blocks, global_key_indices = (
                    prefill_cp_contiguous_history_with_replicated_append(
                        total_seq_lens,
                        append_start,
                        rank,
                        world_size,
                        cache_block_size,
                        hidden_states.device,
                    )
                )
            else:
                logical_blocks, global_key_indices = prefill_cp_key_block_partition(
                    total_seq_lens,
                    rank,
                    world_size,
                    interleave_size,
                    cache_block_size,
                    hidden_states.device,
                )
            if global_key_indices.numel() == 0:
                return output
            local_block_table = gather_chunk.block_table.index_select(
                1,
                logical_blocks.to(torch.int64),
            )
            local_cu_seq_lens = torch.tensor(
                [0, int(global_key_indices.numel())],
                dtype=gather_chunk.cu_seq_lens.dtype,
                device=hidden_states.device,
            )
            local_capacity = (
                int(indexer_op.max_total_seq_len) + world_size - 1
            ) // world_size + cache_block_size
            if mode == "key_contiguous_replicated_append":
                required_capacity = int(global_key_indices.numel())
                capacity_quantum = 8192
                local_capacity = max(
                    local_capacity,
                    ((required_capacity + capacity_quantum - 1) // capacity_quantum)
                    * capacity_quantum,
                )
            k_quant_full, k_scale_full = _private_proxy_k_workspace(
                device=hidden_states.device,
                total_capacity=local_capacity,
                head_dim=head_dim,
                use_fp4=use_fp4,
                fp8_dtype=fp8_dtype,
                slot=workspace_slot,
            )
            k_quant = k_quant_full[: global_key_indices.numel()]
            k_scale = k_scale_full[: global_key_indices.numel()]
            ops.cp_gather_indexer_k_quant_cache(
                indexer_op.k_cache.kv_cache,
                k_quant,
                k_scale,
                local_block_table,
                local_cu_seq_lens,
            )
        else:
            global_key_indices = prefill_cp_key_indices_with_append(
                total_seq_lens,
                append_start,
                rank,
                world_size,
                interleave_size,
                hidden_states.device,
                replicate_append=True,
            )
            if global_key_indices.numel() == 0:
                return output
            value_bytes = head_dim // 2 if use_fp4 else head_dim
            scale_bytes = head_dim // 32 if use_fp4 else 4
            k_quant, k_scale = gather_prefill_cp_local_k_rows(
                indexer_op.k_cache.kv_cache,
                gather_chunk.block_table,
                global_key_indices,
                value_bytes=value_bytes,
                scale_bytes=scale_bytes,
                value_dtype=torch.uint8 if use_fp4 else fp8_dtype,
            )
        local_topk_tokens = min(
            int(global_key_indices.numel()),
            prefill_cp_local_topk_tokens(
                topk_tokens,
                world_size,
                oversubscribe,
            ),
        )
        if runtime_info is not None:
            runtime_info["append_start"] = int(append_start)
            runtime_info["local_key_rows"] = int(global_key_indices.numel())
            runtime_info["local_topk_tokens"] = int(local_topk_tokens)
    else:
        total_capacity = int(indexer_op.max_total_seq_len)
        k_quant_full, k_scale_full = _private_proxy_k_workspace(
            device=hidden_states.device,
            total_capacity=total_capacity,
            head_dim=head_dim,
            use_fp4=use_fp4,
            fp8_dtype=fp8_dtype,
            slot=workspace_slot,
        )
        k_quant = k_quant_full[:total_seq_lens]
        k_scale = k_scale_full[:total_seq_lens]
        ops.cp_gather_indexer_k_quant_cache(
            indexer_op.k_cache.kv_cache,
            k_quant,
            k_scale,
            gather_chunk.block_table,
            gather_chunk.cu_seq_lens,
        )
        global_key_indices = torch.arange(
            total_seq_lens,
            dtype=torch.int64,
            device=k_quant.device,
        )
        local_topk_tokens = topk_tokens

    max_logits_bytes = _prefill_cp_max_logits_bytes(
        envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
    )
    max_query_tokens = max_logits_bytes // max(1, int(k_quant.shape[0]) * 4)
    if max_query_tokens <= 0:
        raise RuntimeError("CP logits budget cannot hold one full-K query row")

    if use_fp4:
        k_cast = k_quant.view(torch.int8)
        scale_cast = k_scale.view(torch.int32).squeeze(-1)
    else:
        k_cast = k_quant
        scale_cast = k_scale.view(torch.float32).squeeze(-1)

    for query_start in range(0, int(local_query_ids.numel()), max_query_tokens):
        query_ids = local_query_ids[query_start : query_start + max_query_tokens]
        metadata_ids = query_ids - first_token
        query_row_ids = input_query_rows[query_start : query_start + max_query_tokens]
        selected_global_ks = global_ks.index_select(0, metadata_ids)
        selected_global_ke = global_ke.index_select(0, metadata_ids)
        if mode == "query":
            local_ks = selected_global_ks
            local_ke = selected_global_ke
        else:
            local_ks, local_ke = localize_prefill_cp_bounds(
                global_key_indices,
                selected_global_ks,
                selected_global_ke,
            )
        q_slice = q_values.index_select(0, query_row_ids)
        q_scale_slice = (
            q_scale.index_select(0, query_row_ids) if q_scale is not None else None
        )
        if use_fp4:
            q_cast = q_slice.view(torch.int8)
        else:
            q_cast = q_slice
        logits = fp8_fp4_mqa_logits(
            (q_cast, q_scale_slice),
            (k_cast, scale_cast),
            weights.index_select(0, query_row_ids),
            local_ks,
            local_ke,
            clean_logits=False,
        )
        local_output = torch.empty(
            (int(query_ids.numel()), local_topk_tokens),
            dtype=output.dtype,
            device=output.device,
        )
        torch.ops._C.top_k_per_row_prefill(
            logits,
            local_ks,
            local_ke,
            local_output,
            int(logits.shape[0]),
            int(logits.stride(0)),
            int(logits.stride(1)),
            local_topk_tokens,
        )
        if mode != "query":
            local_output = globalize_prefill_cp_topk(
                local_output,
                global_key_indices,
            )
            output[query_row_ids, :local_topk_tokens] = local_output
        else:
            output.index_copy_(0, query_row_ids, local_output)
    return output
