# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for TP-sharded SSD prefetch planning."""

# Standard
from types import SimpleNamespace
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.kv_object_store import KVObjectByteRange
from lmcache.v1.ssd_tp_sharded_prefetch import (
    SSDReadMode,
    SSDTPShardedPrefetchConfig,
    ShardPrefetchDecisionTable,
    TorchDistributedShardGather,
    bucket_prefetch_key,
    compile_layer_major_read_ranges,
    compile_cp_read_plan,
    cp_owned_row_ranges,
    dense_rank_major_metadata,
    deterministic_block_union,
    owner_gather_receive_positions,
    owner_gpu_route,
    owner_gpu_padded_blocks,
    parse_layer_ranges,
    partition_block_union,
    partition_rank_local_blocks,
    rank_major_inverse_indices,
    stable_union_hash,
)


def test_owner_gpu_route_handles_padding_and_duplicate_ids() -> None:
    """GPU route keeps the first owner row and ignores rank padding."""
    payloads = torch.tensor(
        [[2, 8, 2, -1], [3, 7, 2, 9], [0, -1, -1, -1]],
        dtype=torch.int64,
    )

    selected, positions, all_ready = owner_gpu_route(
        payloads.reshape(-1), world_size=3, padded_blocks=3
    )

    assert selected.tolist() == [2, 7, 8, 9]
    assert positions.tolist() == [1, 3, 0, 5]
    assert bool(all_ready)


def test_owner_gpu_route_excludes_failed_rank_but_keeps_ready_ranks() -> None:
    """A negative count publishes no rows from that rank for later correction."""
    payloads = torch.tensor([[2, 4, 6], [-1, 8, 10], [1, 12, -1]], dtype=torch.int64)

    selected, positions, all_ready = owner_gpu_route(
        payloads.reshape(-1), world_size=3, padded_blocks=2
    )

    assert selected.tolist() == [4, 6, 12]
    assert positions.tolist() == [0, 1, 4]
    assert not bool(all_ready)


@pytest.mark.parametrize(
    ("covered_end", "expected"),
    [(384, 48), (512, 64), (1920, 240), (4096, 512)],
)
def test_owner_gpu_padding_tracks_covered_prefix(
    covered_end: int, expected: int
) -> None:
    """All ranks derive the same bounded width from shared prefix geometry."""
    assert (
        owner_gpu_padded_blocks(covered_end, world_size=8, configured_cap=512)
        == expected
    )
    assert {
        owner_gpu_padded_blocks(covered_end, world_size=8, configured_cap=512)
        for _rank in range(8)
    } == {expected}


def test_owner_gpu_padding_reserves_suffix_blocks_and_honors_cap() -> None:
    """Append reserve prevents one suffix block from failing owner readiness."""
    assert (
        owner_gpu_padded_blocks(
            384, world_size=8, configured_cap=512, append_reserve_blocks=1
        )
        == 49
    )
    assert (
        owner_gpu_padded_blocks(
            384, world_size=8, configured_cap=512, append_reserve_blocks=32
        )
        == 80
    )
    assert (
        owner_gpu_padded_blocks(
            4096, world_size=8, configured_cap=512, append_reserve_blocks=32
        )
        == 512
    )


def test_dense_gpu_metadata_masks_failed_rank_and_padding() -> None:
    """Dense fixed metadata maps only valid rows from ready owner ranks."""
    partition = partition_block_union([2, 4, 6, 8, 10], 3)
    ids, positions, owners = dense_rank_major_metadata(partition)
    ready = torch.tensor([1, 0, 1], dtype=torch.bool)
    publish = ready[torch.tensor(owners)]

    assert partition.padded_blocks == 2
    assert torch.tensor(ids)[publish].tolist() == [2, 4, 10]
    assert torch.tensor(positions)[publish].tolist() == [0, 1, 4]


@pytest.mark.parametrize(
    ("available_blocks", "block_nbytes", "aligned_length"),
    [
        (20, 1168, 23_552),
        (5, 8448, 42_496),
    ],
)
def test_layer_major_tail_uses_sector_padded_allocation(
    available_blocks: int,
    block_nbytes: int,
    aligned_length: int,
) -> None:
    """Short HCA and odd Indexer payloads include only their tail padding."""
    ranges = compile_layer_major_read_ranges(
        list(range(available_blocks)),
        available_blocks=available_blocks,
        block_nbytes=block_nbytes,
        layer_byte_offset=4096,
        aligned_read_length=aligned_length,
    )

    assert ranges == (
        KVObjectByteRange(
            offset=4096,
            length=aligned_length,
            target_offset=0,
        ),
    )


@pytest.mark.parametrize(
    ("block_ids", "world_size", "expected_counts"),
    [
        ([], 8, (0, 0, 0, 0, 0, 0, 0, 0)),
        ([7], 8, (1, 0, 0, 0, 0, 0, 0, 0)),
        ([4, 1, 4], 8, (1, 1, 0, 0, 0, 0, 0, 0)),
        (list(range(10)), 4, (3, 3, 2, 2)),
        (list(range(1875)), 8, (235, 235, 235, 234, 234, 234, 234, 234)),
    ],
)
def test_deterministic_partition_balances_union(
    block_ids: list[int],
    world_size: int,
    expected_counts: tuple[int, ...],
) -> None:
    """Empty, small, uneven, and full unions keep deterministic balance."""
    partition = partition_block_union(block_ids, world_size)

    assert partition.counts == expected_counts
    assert max(partition.counts, default=0) - min(partition.counts, default=0) <= 1
    assert (
        tuple(
            block_id
            for rank_blocks in partition.blocks_by_rank
            for block_id in rank_blocks
        )
        == partition.union
    )


def test_union_deduplicates_with_process_independent_hash() -> None:
    """Input order and duplicates do not affect the canonical union digest."""
    first = deterministic_block_union([9, 2, 9, 1, 2])
    second = deterministic_block_union([1, 2, 9])

    assert first == second == (1, 2, 9)
    assert stable_union_hash(first) == stable_union_hash(second)
    with pytest.raises(ValueError, match="non-negative"):
        deterministic_block_union([1, -1])


def test_rank_major_inverse_skips_padding() -> None:
    """Rank-major gathered rows map back to canonical union order."""
    partition = partition_block_union([11, 3, 7, 5, 13], 3)

    assert partition.union == (3, 5, 7, 11, 13)
    assert partition.counts == (2, 2, 1)
    assert partition.padded_blocks == 2
    assert rank_major_inverse_indices(partition) == (0, 1, 2, 3, 4)


def test_rank_local_partition_preserves_owner_and_deduplicates() -> None:
    """Owner-compute candidates stay with their rank for the SSD read."""
    partition = partition_rank_local_blocks(
        ((8, 2, 8), (7, 2, 9), (), (5,)),
    )

    assert partition.union == (2, 5, 7, 8, 9)
    assert partition.blocks_by_rank == ((2, 8), (7, 9), (), (5,))
    assert partition.counts == (2, 2, 0, 1)
    assert partition.padded_blocks == 2
    assert rank_major_inverse_indices(partition) == (0, 6, 2, 1, 3)


def test_owner_receive_positions_use_sent_offsets_and_effective_width() -> None:
    """Dedup holes retain their original send rows under a shrunken stride."""
    sent = ((2, 8), (2, 7, 9), (), (5,))
    partition = partition_rank_local_blocks(sent)

    assert owner_gather_receive_positions(
        sent, partition.blocks_by_rank, effective_padded=3
    ) == (0, 1, 4, 5, 9)


def test_owner_receive_positions_reject_invalid_metadata() -> None:
    """Invalid metadata fails before selecting the wrong KV receive row."""
    with pytest.raises(ValueError, match="absent"):
        owner_gather_receive_positions(((2,),), ((3,),), effective_padded=1)
    with pytest.raises(ValueError, match="exceed"):
        owner_gather_receive_positions(((2, 3),), ((2,),), effective_padded=1)


def test_owner_gather_uses_actual_max_width_and_duplicate_send_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KV gather shrinks to max count and scatters the first duplicate owner."""
    transport = object.__new__(TorchDistributedShardGather)
    transport._healthy = True
    transport._world_size = 3
    transport._rank = 0
    transport._lock = threading.Lock()
    transport._stream = object()
    transport._metadata_stream = SimpleNamespace(synchronize=lambda: None)
    transport._metadata_process_group = object()
    transport._process_group = object()
    transport._slot_cursor = 0
    transport._config = SimpleNamespace()
    slot = SimpleNamespace(
        send=torch.zeros((8, 1), dtype=torch.uint8),
        receive=torch.zeros((24, 1), dtype=torch.uint8),
        send_ids=torch.full((9,), -1, dtype=torch.int64),
        receive_ids=torch.full((27,), -1, dtype=torch.int64),
        completion_event=None,
        completion_work=None,
    )
    transport._slots = [slot]
    gathered_widths: list[int] = []

    class _Context:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class _Work:
        def wait(self) -> None:
            return None

    class _Event:
        def record(self, _stream: object) -> None:
            return None

    payloads = torch.full((3, 9), -1, dtype=torch.int64)
    payloads[0, :3] = torch.tensor([2, 2, 8])
    payloads[1, :4] = torch.tensor([3, 2, 7, 9])
    payloads[2, 0] = 0
    gathered_rows = torch.tensor(
        [[20], [80], [0], [200], [70], [90], [0], [0], [0]],
        dtype=torch.uint8,
    )

    def _all_gather(
        output: torch.Tensor, input_: torch.Tensor, **_kwargs: object
    ) -> _Work:
        gathered_widths.append(int(input_.shape[0]))
        if input_.dtype == torch.int64:
            output.copy_(payloads.reshape(-1))
        else:
            output.copy_(gathered_rows)
        return _Work()

    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: _Context())
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", _all_gather)
    destination = torch.zeros((10, 1), dtype=torch.uint8)
    resident = torch.zeros(10, dtype=torch.bool)

    _event, partition, ready = transport.gather_owner_rows_into(
        local_block_ids=torch.tensor([2, 8]),
        local_ready=True,
        padded_blocks=8,
        source_rows=torch.arange(10, dtype=torch.uint8).reshape(10, 1),
        logical_destination_rows=torch.arange(10),
        destination_rows=destination,
        local_ready_event=None,
        resident_bitmap=resident,
    )

    assert ready
    assert partition.blocks_by_rank == ((2, 8), (7, 9), ())
    assert gathered_widths == [9, 3]
    assert destination[[2, 7, 8, 9], 0].tolist() == [20, 70, 80, 90]
    assert resident.nonzero().reshape(-1).tolist() == [2, 7, 8, 9]


def test_rank_local_partition_applies_global_staging_limit() -> None:
    """A global cap is identical across ranks without changing ownership."""
    partition = partition_rank_local_blocks(
        ((9, 1), (7, 3), (5,)),
        max_union_blocks=4,
    )

    assert partition.union == (1, 3, 5, 7)
    assert partition.blocks_by_rank == ((1,), (3, 7), (5,))


def test_cp_row_ownership_covers_context_without_overlap() -> None:
    """Block-cyclic CP row ranges cover every row exactly once."""
    ranges_by_rank = [cp_owned_row_ranges(29, rank, 4, 3) for rank in range(4)]
    owned = [
        row
        for ranges in ranges_by_rank
        for row_range in ranges
        for row in range(row_range.start, row_range.end)
    ]

    assert sorted(owned) == list(range(29))
    assert len(owned) == len(set(owned))


def test_cp_read_plan_compiles_aligned_whole_blocks() -> None:
    """CP8 with one 64-row interleave emits coalesced whole-block reads."""
    plan = compile_cp_read_plan(
        total_rows=1875 * 64,
        rank=3,
        world_size=8,
        interleave_rows=64,
        block_rows=64,
        row_bytes=132,
    )

    assert plan.planned_rows == 234 * 64
    assert len(plan.block_ids) == 234
    assert plan.block_ids[:3] == (3, 11, 19)
    assert plan.ssd_bytes == 234 * 64 * 132


def test_cp_read_plan_fails_closed_on_partial_block_ownership() -> None:
    """A CP boundary inside an SSD block is rejected before I/O planning."""
    with pytest.raises(ValueError, match="inside an SSD block"):
        compile_cp_read_plan(
            total_rows=1024,
            rank=1,
            world_size=8,
            interleave_rows=32,
            block_rows=64,
            row_bytes=132,
        )


def test_decision_table_selects_large_union_and_rejects_small_union() -> None:
    """Calibrated p90 cost chooses gather only when it beats local I/O."""
    config = SSDTPShardedPrefetchConfig(
        enabled=True,
        csa_replica_verified=True,
        min_union_blocks=128,
    )
    table = ShardPrefetchDecisionTable(config)
    large_key = bucket_prefetch_key(
        group="csa",
        layer_id=12,
        context_tokens=480_000,
        query_tokens=8_000,
        union_blocks=1875,
    )
    small_key = bucket_prefetch_key(
        group="csa",
        layer_id=12,
        context_tokens=32_000,
        query_tokens=2_000,
        union_blocks=32,
    )

    assert (
        table.choose(
            large_key,
            union_blocks=1875,
            block_bytes=37_376,
            world_size=8,
            shard_mode=SSDReadMode.SHARD_GATHER_DENSE,
            capability_ok=True,
        )
        == SSDReadMode.SHARD_GATHER_DENSE
    )
    assert (
        table.choose(
            small_key,
            union_blocks=32,
            block_bytes=37_376,
            world_size=8,
            shard_mode=SSDReadMode.SHARD_GATHER_PREDICTED,
            capability_ok=True,
        )
        == SSDReadMode.LOCAL_DIRECT
    )


def test_capability_and_layer_kill_switch_force_local_mode() -> None:
    """Capability failure and per-layer disablement retain local fallback."""
    config = SSDTPShardedPrefetchConfig(
        enabled=True,
        csa_replica_verified=True,
        min_union_blocks=1,
        disabled_layers=frozenset({8}),
    )
    table = ShardPrefetchDecisionTable(config)
    disabled = bucket_prefetch_key(
        group="csa",
        layer_id=8,
        context_tokens=480_000,
        query_tokens=8_000,
        union_blocks=1875,
    )
    enabled = bucket_prefetch_key(
        group="csa",
        layer_id=10,
        context_tokens=480_000,
        query_tokens=8_000,
        union_blocks=1875,
    )

    assert (
        table.choose(
            disabled,
            union_blocks=1875,
            block_bytes=37_376,
            world_size=8,
            shard_mode=SSDReadMode.SHARD_GATHER_DENSE,
            capability_ok=True,
        )
        == SSDReadMode.LOCAL_DIRECT
    )
    assert (
        table.choose(
            enabled,
            union_blocks=1875,
            block_bytes=37_376,
            world_size=8,
            shard_mode=SSDReadMode.SHARD_GATHER_DENSE,
            capability_ok=False,
        )
        == SSDReadMode.LOCAL_DIRECT
    )


def test_online_p90_ignores_cold_start_and_uses_max_rank_skew() -> None:
    """Steady-state calibration is bounded and excludes cold samples."""
    config = SSDTPShardedPrefetchConfig(
        enabled=True,
        csa_replica_verified=True,
        min_union_blocks=1,
    )
    table = ShardPrefetchDecisionTable(config)
    key = bucket_prefetch_key(
        group="csa",
        layer_id=10,
        context_tokens=480_000,
        query_tokens=8_000,
        union_blocks=1875,
    )

    table.record_sample(
        key,
        SSDReadMode.LOCAL_DIRECT,
        500.0,
        cold_start=True,
    )
    assert table.observed_p90(key, SSDReadMode.LOCAL_DIRECT) is None

    for elapsed_ms in range(1, 11):
        table.record_sample(
            key,
            SSDReadMode.LOCAL_DIRECT,
            float(elapsed_ms),
            max_rank_skew_ms=1.0,
        )
    assert table.observed_p90(key, SSDReadMode.LOCAL_DIRECT) == 10.0

    with pytest.raises(ValueError, match="finite and non-negative"):
        table.record_sample(key, SSDReadMode.LOCAL_DIRECT, -1.0)


def test_central_config_parses_ranges_and_engine_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment and engine configuration share one validated schema."""
    monkeypatch.delenv("LMCACHE_SSD_TP_DENSE_LAYERS", raising=False)
    assert SSDTPShardedPrefetchConfig.from_env().dense_layers == frozenset()
    assert LMCacheEngineConfig.from_defaults().ssd_tp_dense_layers == ""

    monkeypatch.setenv("LMCACHE_SSD_TP_SHARDED_PREFETCH", "1")
    monkeypatch.setenv("LMCACHE_SSD_TP_DENSE_LAYERS", "2-4,8")
    env_config = SSDTPShardedPrefetchConfig.from_env()

    assert env_config.enabled
    assert env_config.dense_layers == frozenset({2, 3, 4, 8})
    assert parse_layer_ranges("") == frozenset()

    engine_config = LMCacheEngineConfig.from_defaults(
        ssd_tp_sharded_prefetch=True,
        ssd_tp_dense_layers="6-8",
        ssd_tp_staging_slots=3,
    )
    resolved = SSDTPShardedPrefetchConfig.from_engine_config(engine_config)

    assert resolved.enabled
    assert resolved.dense_layers == frozenset({6, 7, 8})
    assert resolved.staging_slots == 3
