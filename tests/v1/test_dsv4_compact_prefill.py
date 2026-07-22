# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.dsv4_compact_prefill import (
    build_compact_csa_prefill_gather_plan,
    native_compact_prefill_available,
)


def test_compact_plan_unions_pages_and_remaps_entries() -> None:
    """Selected entries retain their values after compact page remapping."""
    block_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)
    topk = torch.tensor([[1, 9, -1], [2, 8, 99]], dtype=torch.int32)

    compact_table, compact_lens, remapped = (
        build_compact_csa_prefill_gather_plan(
            topk,
            block_table,
            torch.tensor([10], dtype=torch.int32),
            [0, 2],
            block_size=4,
        )
    )

    assert compact_table.tolist() == [[10, 12]]
    assert compact_lens.tolist() == [8]
    assert remapped.tolist() == [[1, 5, -1], [2, 4, -1]]


def test_compact_plan_keeps_request_page_tables_independent() -> None:
    """Each request receives its own compact table and index namespace."""
    block_table = torch.tensor(
        [[10, 11, 12], [20, 21, 22]],
        dtype=torch.int32,
    )
    topk = torch.tensor([[0, 7], [4, -1], [8, 1]], dtype=torch.int64)

    compact_table, compact_lens, remapped = (
        build_compact_csa_prefill_gather_plan(
            topk,
            block_table,
            torch.tensor([8, 9], dtype=torch.int32),
            [0, 2, 3],
            block_size=4,
        )
    )

    assert compact_table.tolist() == [[10, 11], [20, 22]]
    assert compact_lens.tolist() == [8, 8]
    assert remapped.tolist() == [[0, 7], [4, -1], [4, 1]]


def test_compact_plan_accepts_tensor_row_offsets() -> None:
    """The fallback matches vLLM's tensor-form row metadata contract."""
    compact_table, compact_lens, remapped = build_compact_csa_prefill_gather_plan(
        torch.tensor([[1, 9], [2, 8]], dtype=torch.int32),
        torch.tensor([[10, 11, 12]], dtype=torch.int32),
        torch.tensor([10], dtype=torch.int32),
        torch.tensor([5, 7], dtype=torch.int32),
        block_size=4,
    )

    assert compact_table.tolist() == [[10, 12]]
    assert compact_lens.tolist() == [8]
    assert remapped.tolist() == [[1, 5], [2, 4]]


def test_full_cached_prefix_selection_returns_original_plan() -> None:
    """A fully resident cached prefix needs no table or top-K remapping."""
    topk = torch.tensor([[1, 9], [2, 8]], dtype=torch.int32)
    block_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)
    seq_lens = torch.tensor([10], dtype=torch.int32)

    table_out, lens_out, topk_out = build_compact_csa_prefill_gather_plan(
        topk,
        block_table,
        seq_lens,
        [0, 2],
        block_size=4,
        cached_prefix_fully_selected=True,
    )

    assert table_out is block_table
    assert lens_out is seq_lens
    assert topk_out is topk


def test_full_prefix_hint_falls_back_for_multiple_requests() -> None:
    """A request-aggregated hint cannot bypass per-request compaction."""
    table, lens, remapped = build_compact_csa_prefill_gather_plan(
        torch.tensor([[0], [4]], dtype=torch.int32),
        torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        torch.tensor([4, 8], dtype=torch.int32),
        [0, 1, 2],
        block_size=4,
        cached_prefix_fully_selected=True,
    )

    assert table.tolist() == [[10], [21]]
    assert lens.tolist() == [4, 4]
    assert remapped.tolist() == [[0], [0]]


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not native_compact_prefill_available(),
    reason="LMCache compact prefill CUDA extension is unavailable",
)
def test_compact_plan_cuda_matches_cpu_without_host_metadata() -> None:
    """The production CUDA path preserves CPU reference remapping."""
    topk_cpu = torch.tensor([[0, 7], [4, -1], [8, 1]], dtype=torch.int32)
    table_cpu = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
    lens_cpu = torch.tensor([8, 9], dtype=torch.int32)
    offsets_cpu = torch.tensor([10, 12, 13], dtype=torch.int32)
    expected_table, expected_lens, expected_remapped = (
        build_compact_csa_prefill_gather_plan(
            topk_cpu,
            table_cpu,
            lens_cpu,
            offsets_cpu,
            block_size=4,
        )
    )

    compact_table, compact_lens, remapped = build_compact_csa_prefill_gather_plan(
        topk_cpu.cuda(),
        table_cpu.cuda(),
        lens_cpu.cuda(),
        offsets_cpu.cuda(),
        block_size=4,
    )

    assert torch.equal(compact_lens.cpu(), expected_lens)
    assert torch.equal(remapped.cpu(), expected_remapped)
    assert torch.equal(
        compact_table.cpu()[:, : expected_table.shape[1]],
        expected_table,
    )


@pytest.mark.parametrize(
    ("offsets", "block_size"),
    [([1, 1], 4), ([0, 2], 4), ([0, 1], 0)],
)
def test_compact_plan_rejects_invalid_layouts(
    offsets: list[int],
    block_size: int,
) -> None:
    """Malformed vLLM metadata fails before any gather is launched."""
    with pytest.raises(ValueError):
        build_compact_csa_prefill_gather_plan(
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[5]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            offsets,
            block_size,
        )
