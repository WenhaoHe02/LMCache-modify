# SPDX-License-Identifier: Apache-2.0
"""Helpers for vLLM DeepSeek-V4 compact CSA prefill gathers."""

from __future__ import annotations

# Standard
from collections.abc import Sequence

# Third Party
import torch

try:
    import lmcache.c_ops as _lmc_ops
except ImportError:
    _lmc_ops = None


def native_compact_prefill_available() -> bool:
    """Return whether the CUDA compact-prefill planner is installed.

    Returns:
        ``True`` when ``lmcache.c_ops`` exports the native planner.
    """
    return callable(
        getattr(_lmc_ops, "build_compact_csa_prefill_gather_plan", None)
    )


def build_compact_csa_prefill_gather_plan(
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    query_row_offsets: Sequence[int] | torch.Tensor,
    block_size: int,
    selected_page_bitmap: torch.Tensor | None = None,
    cached_prefix_fully_selected: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build paged gather metadata containing only true-topK pages.

    Each request may have many prefill query rows. The function unions their
    selected logical pages, translates those pages through vLLM's physical
    block table, and remaps the original entry ids into the compact per-request
    workspace. Invalid or padded top-K ids remain ``-1``.

    Args:
        topk_indices: Per-query compressed entry ids with shape
            ``[num_query_rows, top_k]``.
        block_table: Per-request physical page ids with shape
            ``[num_requests, max_logical_pages]``.
        compressed_seq_lens: Valid compressed entry count for each request.
        query_row_offsets: Prefix sum mapping requests to rows in
            ``topk_indices``. It must contain ``num_requests + 1`` entries.
            The first value may be a non-zero common row base.
            CUDA production calls should pass a CUDA tensor to avoid a host
            copy or synchronization.
        block_size: Number of compressed entries in one physical page.
        selected_page_bitmap: Optional exact int32 logical-page union produced
            by LMCache correction. It is used only for one-request CUDA plans;
            other layouts rescan ``topk_indices`` as before.
        cached_prefix_fully_selected: Whether LMCache correction proved every
            cached-prefix page is selected and resident. For a one-request
            plan, the original vLLM gather is then already safe and avoids all
            compact-table construction and top-K remapping.

    Returns:
        A tuple of ``(compact_block_table, compact_seq_lens, remapped_topk)``.

    Raises:
        ValueError: If tensor ranks, request counts, row offsets, or block size
            are inconsistent.
    """
    if topk_indices.ndim != 2:
        raise ValueError("topk_indices must be a 2D tensor")
    if block_table.ndim != 2:
        raise ValueError("block_table must be a 2D tensor")
    if compressed_seq_lens.ndim != 1:
        raise ValueError("compressed_seq_lens must be a 1D tensor")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    num_requests = int(block_table.shape[0])
    if int(compressed_seq_lens.shape[0]) != num_requests:
        raise ValueError("compressed_seq_lens must contain one value per request")
    if isinstance(query_row_offsets, torch.Tensor):
        if query_row_offsets.ndim != 1:
            raise ValueError("query_row_offsets must be a 1D tensor")
        if int(query_row_offsets.numel()) != num_requests + 1:
            raise ValueError(
                "query_row_offsets must contain num_requests + 1 entries"
            )
    elif len(query_row_offsets) != num_requests + 1:
        raise ValueError("query_row_offsets must contain num_requests + 1 entries")
    if cached_prefix_fully_selected and num_requests == 1:
        return block_table, compressed_seq_lens, topk_indices
    any_cuda = bool(
        topk_indices.is_cuda
        or block_table.is_cuda
        or compressed_seq_lens.is_cuda
        or (
            isinstance(query_row_offsets, torch.Tensor)
            and query_row_offsets.is_cuda
        )
    )
    if any_cuda:
        if not (
            topk_indices.is_cuda
            and block_table.is_cuda
            and compressed_seq_lens.is_cuda
            and isinstance(query_row_offsets, torch.Tensor)
            and query_row_offsets.is_cuda
        ):
            raise ValueError(
                "CUDA compact prefill requires every input, including "
                "query_row_offsets, on the same CUDA device"
            )
        reuse_seen = bool(
            num_requests == 1
            and isinstance(selected_page_bitmap, torch.Tensor)
            and selected_page_bitmap.is_cuda
            and selected_page_bitmap.device == topk_indices.device
            and selected_page_bitmap.dtype == torch.int32
        )
        native_op = getattr(
            _lmc_ops,
            "build_compact_csa_prefill_gather_plan_from_page_seen"
            if reuse_seen
            else "build_compact_csa_prefill_gather_plan",
            None,
        )
        if not callable(native_op):
            raise RuntimeError(
                "CUDA compact prefill requires the LMCache native extension; "
                "rebuild lmcache.c_ops"
            )
        if not (
            topk_indices.device == block_table.device
            == compressed_seq_lens.device
            == query_row_offsets.device
        ):
            raise ValueError("all compact prefill tensors must use one CUDA device")
        args = (
            topk_indices.contiguous(),
            block_table.contiguous(),
            compressed_seq_lens.contiguous(),
            query_row_offsets.contiguous(),
            block_size,
        )
        if reuse_seen:
            assert selected_page_bitmap is not None
            max_pages = int(block_table.shape[1])
            flat_seen = selected_page_bitmap.reshape(-1)
            if int(flat_seen.numel()) >= max_pages:
                page_seen = flat_seen[:max_pages].reshape(1, max_pages).contiguous()
            else:
                page_seen = torch.zeros(
                    (1, max_pages),
                    dtype=torch.int32,
                    device=topk_indices.device,
                )
                page_seen[0, : flat_seen.numel()] = flat_seen
            outputs = native_op(*args, page_seen)
        else:
            outputs = native_op(*args)
        return outputs[0], outputs[1], outputs[2]

    raw_offsets = tuple(
        int(value)
        for value in (
            query_row_offsets.tolist()
            if isinstance(query_row_offsets, torch.Tensor)
            else query_row_offsets
        )
    )
    row_base = raw_offsets[0]
    normalized_offsets = tuple(value - row_base for value in raw_offsets)
    if (
        normalized_offsets[0] != 0
        or normalized_offsets[-1] != int(topk_indices.shape[0])
        or any(
            start > end
            for start, end in zip(
                normalized_offsets,
                normalized_offsets[1:],
                strict=False,
            )
        )
    ):
        raise ValueError("query_row_offsets must exactly partition topk_indices")

    remapped_topk = torch.full_like(topk_indices, -1)
    selected_physical_tables: list[torch.Tensor] = []
    selected_entry_lens: list[int] = []
    max_selected_pages = 1

    for request_index in range(num_requests):
        row_start = normalized_offsets[request_index]
        row_end = normalized_offsets[request_index + 1]
        request_topk = topk_indices[row_start:row_end]
        compressed_len = int(compressed_seq_lens[request_index].item())
        if compressed_len < 0:
            raise ValueError("compressed sequence lengths must be non-negative")
        num_logical_pages = (compressed_len + block_size - 1) // block_size
        if num_logical_pages > int(block_table.shape[1]):
            raise ValueError("block_table does not cover the compressed sequence")
        if request_topk.numel() == 0 or num_logical_pages == 0:
            selected_physical_tables.append(block_table.new_empty((0,)))
            selected_entry_lens.append(0)
            continue

        entries = request_topk.to(torch.int64)
        page_ids = torch.div(entries, block_size, rounding_mode="floor")
        valid = (entries >= 0) & (entries < compressed_len)
        page_seen = torch.zeros(
            num_logical_pages,
            dtype=torch.bool,
            device=entries.device,
        )
        page_seen[page_ids[valid]] = True
        selected_pages = page_seen.nonzero(as_tuple=False).reshape(-1)
        selected_count = int(selected_pages.numel())
        if selected_count == 0:
            selected_physical_tables.append(block_table.new_empty((0,)))
            selected_entry_lens.append(0)
            continue

        physical_pages = block_table[request_index].index_select(
            0,
            selected_pages.to(block_table.device),
        )
        selected_physical_tables.append(physical_pages)
        selected_entry_lens.append(selected_count * block_size)
        max_selected_pages = max(max_selected_pages, selected_count)

        compact_page_ids = torch.full(
            (num_logical_pages,),
            -1,
            dtype=torch.int64,
            device=entries.device,
        )
        compact_page_ids[selected_pages] = torch.arange(
            selected_count,
            dtype=torch.int64,
            device=entries.device,
        )
        compact_entries = (
            compact_page_ids[page_ids[valid]] * block_size
            + torch.remainder(entries[valid], block_size)
        )
        request_remapped = remapped_topk[row_start:row_end]
        request_remapped[valid] = compact_entries.to(request_remapped.dtype)

    compact_block_table = torch.zeros(
        (num_requests, max_selected_pages),
        dtype=block_table.dtype,
        device=block_table.device,
    )
    for request_index, physical_pages in enumerate(selected_physical_tables):
        if physical_pages.numel():
            compact_block_table[
                request_index,
                : physical_pages.numel(),
            ] = physical_pages
    compact_seq_lens = torch.tensor(
        selected_entry_lens,
        dtype=compressed_seq_lens.dtype,
        device=compressed_seq_lens.device,
    )
    return compact_block_table, compact_seq_lens, remapped_topk


def build_compact_csa_prefill_page_plan(
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    query_row_offsets: torch.Tensor,
    block_size: int,
    selected_page_bitmap: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compact page metadata without materializing remapped top-K IDs.

    The returned page map is consumed by the final sparse-index combiner, which
    translates logical top-K entries while it already visits them.
    """
    if (
        topk_indices.ndim != 2
        or block_table.ndim != 2
        or compressed_seq_lens.ndim != 1
        or query_row_offsets.ndim != 1
    ):
        raise ValueError("compact page-plan tensors have incompatible ranks")
    if int(block_table.shape[0]) != 1:
        raise ValueError("streamed compact page plans require one request")
    if int(compressed_seq_lens.numel()) != 1 or int(query_row_offsets.numel()) != 2:
        raise ValueError("compact page plan requires one sequence and two offsets")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    tensors = (
        topk_indices,
        block_table,
        compressed_seq_lens,
        query_row_offsets,
        selected_page_bitmap,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("compact page plan requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("compact page-plan tensors must use one CUDA device")
    if selected_page_bitmap.dtype != torch.int32:
        raise ValueError("selected_page_bitmap must use int32 dtype")
    native_op = getattr(
        _lmc_ops,
        "build_compact_csa_prefill_gather_plan_from_page_seen",
        None,
    )
    if not callable(native_op):
        raise RuntimeError("compact page plan requires the LMCache native extension")
    max_pages = int(block_table.shape[1])
    flat_seen = selected_page_bitmap.reshape(-1)
    if int(flat_seen.numel()) < max_pages:
        raise ValueError("selected page bitmap does not cover the block table")
    page_seen = flat_seen[:max_pages].reshape(1, max_pages).contiguous()
    outputs = native_op(
        topk_indices.contiguous(),
        block_table.contiguous(),
        compressed_seq_lens.contiguous(),
        query_row_offsets.contiguous(),
        block_size,
        page_seen,
        False,
    )
    return outputs[0], outputs[1], outputs[3]
