# SPDX-License-Identifier: Apache-2.0
"""Rank-local prefill scorer used only by speculative CSA prefetch."""

# Standard
import os
from typing import Any

# Third Party
import torch


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


def prefill_cp_query_indices(
    token_start: int,
    token_end: int,
    rank: int,
    world_size: int,
    interleave_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return query rows owned by one speculative prefill rank.

    Args:
        token_start: Inclusive first proxy query row.
        token_end: Exclusive final proxy query row.
        rank: Rank in the speculative prefetch group.
        world_size: Number of ranks sharing proxy scoring.
        interleave_size: Consecutive query rows assigned to one rank.
        device: Device on which to construct the indices.

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
    return _block_cyclic_indices(
        token_end - token_start,
        rank,
        world_size,
        interleave_size,
        device,
    ).add_(int(token_start))


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
    topk_tokens_override: int | None = None,
    metadata_query_row_start: int | None = None,
    runtime_info: dict[str, int] | None = None,
) -> torch.Tensor:
    """Run the speculative indexer for this rank's query shard over full K.

    The official indexer op and its metadata are not modified. This function
    mirrors its prefill gather/scoring path, selects one block-cyclic subset
    of query rows, and computes the original full-width top-k against every K
    row. It intentionally rejects decode and multi-sequence metadata.

    Args:
        indexer_op: Active vLLM ``SparseAttnIndexer`` instance.
        hidden_states: Proxy hidden rows for the current prefill chunk.
        q_quant: Quantized proxy queries, optionally with FP4 scales.
        weights: Per-query indexer head weights.
        output: Private top-k output buffer.
        rank: Rank in the speculative prefetch group.
        world_size: Number of ranks sharing proxy scoring.
        interleave_size: Consecutive query rows assigned per rank.
        oversubscribe: Reserved compatibility argument. Query sharding always
            uses the official full top-k width.
        topk_tokens_override: Optional wider speculative top-k width. This
            changes only proxy coverage; the official indexer's output width
            and sparse-attention semantics remain unchanged.
        metadata_query_row_start: Optional row offset into the active prefill
            metadata. When set, ``q_quant``, ``weights``, and ``output`` are a
            compact slice beginning at this metadata row. This is used only
            to warm the exact cache-hit proxy shape during cold admission.
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
    from vllm.v1.worker.workspace import current_workspace_manager

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
    topk_tokens = (
        int(topk_tokens_override)
        if topk_tokens_override is not None
        else int(indexer_op.topk_tokens)
    )
    if topk_tokens <= 0 or topk_tokens > int(output.shape[1]):
        raise RuntimeError("proxy top-k width does not fit the output buffer")
    workspace = current_workspace_manager()
    fp8_dtype = current_platform.fp8_dtype()
    head_dim = int(indexer_op.head_dim)
    total_capacity = int(indexer_op.max_total_seq_len)
    if use_fp4:
        values_spec = ((total_capacity, head_dim // 2), torch.uint8)
        scales_spec = ((total_capacity, head_dim // 32), torch.uint8)
    else:
        values_spec = ((total_capacity, head_dim), fp8_dtype)
        scales_spec = ((total_capacity, 4), torch.uint8)
    k_quant_full, k_scale_full = workspace.get_simultaneous(
        values_spec,
        scales_spec,
    )

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

    k_quant = k_quant_full[:total_seq_lens]
    k_scale = k_scale_full[:total_seq_lens]
    ops.cp_gather_indexer_k_quant_cache(
        indexer_op.k_cache.kv_cache,
        k_quant,
        k_scale,
        gather_chunk.block_table,
        gather_chunk.cu_seq_lens,
    )
    query_first = first_token
    query_final = final_token
    compact_query_base = 0
    if metadata_query_row_start is not None:
        if metadata_query_row_start < 0:
            raise RuntimeError("metadata query row start must be non-negative")
        query_first = first_token + metadata_query_row_start
        query_final = query_first + int(hidden_states.shape[0])
        if query_final > final_token:
            raise RuntimeError("compact proxy query slice exceeds prefill metadata")
        compact_query_base = query_first
    local_query_ids = prefill_cp_query_indices(
        query_first,
        query_final,
        rank,
        world_size,
        interleave_size,
        k_quant.device,
    )
    if local_query_ids.numel() == 0:
        return output

    # Every rank sees the complete gathered K matrix but scores only 1/N query
    # rows. This keeps MQA FLOPs equal to K sharding while making each local
    # prediction semantically complete and eliminating the rank union.
    global_ks = torch.cat(global_ks_parts)
    global_ke = torch.cat(global_ke_parts)
    max_logits_bytes = _prefill_cp_max_logits_bytes(
        envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
    )
    max_query_tokens = max_logits_bytes // max(1, total_seq_lens * 4)
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
        query_row_ids = query_ids - compact_query_base
        local_ks = global_ks.index_select(0, metadata_ids)
        local_ke = global_ke.index_select(0, metadata_ids)
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
            (int(query_ids.numel()), topk_tokens),
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
            topk_tokens,
        )
        output.index_copy_(0, query_row_ids, local_output)
    return output
