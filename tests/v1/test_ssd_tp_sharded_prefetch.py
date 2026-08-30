# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for TP-sharded SSD prefetch planning."""

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
    bucket_prefetch_key,
    compile_layer_major_read_ranges,
    compile_cp_read_plan,
    cp_owned_row_ranges,
    deterministic_block_union,
    owner_gpu_route,
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
