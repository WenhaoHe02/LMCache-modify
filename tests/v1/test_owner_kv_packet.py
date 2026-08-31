# SPDX-License-Identifier: Apache-2.0
"""Byte-level contract for combined owner ID+KV transport packets."""

# Third Party
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("failed_rank", [False, True])
def test_owner_packets_preserve_rows_and_deduplicate(failed_rank: bool) -> None:
    """Padding/failure/duplicates cannot overwrite unrelated or valid rows."""
    from lmcache.v1.owner_kv_packet import pack_owner_rows, scatter_owner_packets

    source = (
        torch.arange(10 * 48, device="cuda", dtype=torch.int32)
        .to(torch.uint8)
        .reshape(10, 48)[:, 8:40]
    )
    row_map = torch.arange(9, -1, -1, device="cuda", dtype=torch.int64)
    ids = ([2, 5], [5, 8], [])
    packets = []
    for rank, values in enumerate(ids):
        packet = torch.empty((3, 40), dtype=torch.uint8, device="cuda")
        pack_owner_rows(
            source,
            row_map,
            torch.tensor(values, dtype=torch.int64, device="cuda"),
            packet,
            local_ready=not (failed_rank and rank == 1),
        )
        packets.append(packet)
    received = torch.cat(packets)
    storage = torch.full((10, 48), 255, dtype=torch.uint8, device="cuda")
    destination = storage[:, 8:40]
    resident = torch.zeros(10, dtype=torch.bool, device="cuda")
    resident[1] = True
    selected = scatter_owner_packets(
        received,
        torch.empty(10, dtype=torch.int32, device="cuda"),
        row_map,
        destination,
        resident,
    )
    wanted = [2, 5] if failed_rank else [2, 5, 8]
    assert selected[selected >= 0].cpu().tolist() == wanted
    assert resident.nonzero().flatten().cpu().tolist() == sorted([1, *wanted])
    expected = torch.full_like(storage, 255)
    for block in wanted:
        expected[9 - block, 8:40] = source[9 - block]
    assert torch.equal(storage, expected)
