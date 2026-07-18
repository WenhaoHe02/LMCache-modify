# SPDX-License-Identifier: Apache-2.0
"""Tests for the public Index-to-I/O logical planning interface."""

# Standard
from dataclasses import FrozenInstanceError
import random

# Third Party
import pytest

# First Party
from lmcache.v1.index_to_io_plan import (
    Confidence,
    ContentKey,
    Correctness,
    Coverage,
    DecisionKey,
    LayerPlan,
    LayerRole,
    LogicalBlockSource,
    LogicalCodec,
    LogicalSelection,
    LoweringCostModel,
    OrderedMerklePrefix,
    PhysicalRange,
    PlacementKind,
    PlanRole,
    bind_logical_selection,
    canonical_checksum,
    logical_missing,
    lower_layer_plan,
)


def _decision_key(*, query_digest: str = "query-a") -> DecisionKey:
    return DecisionKey(
        content=ContentKey(
            model_fingerprint="model-a",
            prefix_root="prefix-a",
            token_count=480_000,
            layout_fingerprint="csa-4x64-fp8",
        ),
        layer_id=4,
        layer_role=LayerRole.CSA,
        plan_role=PlanRole.ATTENTION_KV,
        query_start=480_000,
        query_count=256,
        query_digest=query_digest,
    )


def _layer_plan(ids: tuple[int, ...], universe_size: int = 16) -> LayerPlan:
    selection = LogicalSelection.from_ids(ids, universe_size)
    return LayerPlan(
        key=_decision_key(),
        correctness=Correctness.SPECULATIVE,
        selection=selection,
        coverage=Coverage(len(ids), universe_size),
        confidence=Confidence(0.9, 12),
    )


@pytest.mark.parametrize(
    "codec",
    [LogicalCodec.LIST, LogicalCodec.BITMAP, LogicalCodec.RANGES],
)
def test_logical_codec_randomized_round_trip(codec: LogicalCodec) -> None:
    """Every standalone logical codec preserves randomized sets."""
    rng = random.Random(7)
    for universe_size in range(0, 130):
        ids = tuple(
            logical_id for logical_id in range(universe_size) if rng.random() < 0.35
        )
        selection = LogicalSelection.from_ids(ids, universe_size, codec)
        assert selection.ids() == ids
        assert selection.coverage() == Coverage(len(ids), universe_size)


def test_xor_delta_round_trip_and_base_validation() -> None:
    """XOR deltas reconstruct targets only with their canonical base."""
    base = LogicalSelection.from_ids((1, 2, 8, 15), 20, LogicalCodec.BITMAP)
    target_ids = (2, 3, 8, 12, 19)
    delta = LogicalSelection.from_ids(
        target_ids,
        20,
        LogicalCodec.XOR_DELTA,
        base=base,
    )

    assert delta.ids(base=base) == target_ids
    with pytest.raises(ValueError, match="requires its base"):
        delta.ids()
    wrong_base = LogicalSelection.from_ids((1, 2, 9, 15), 20)
    with pytest.raises(ValueError, match="checksum mismatch"):
        delta.ids(base=wrong_base)
    with pytest.raises(ValueError, match="universe_size mismatch"):
        LogicalSelection.from_ids(
            target_ids,
            20,
            LogicalCodec.XOR_DELTA,
            base=LogicalSelection.from_ids((), 19),
        )


def test_logical_missing_excludes_resident_and_inflight() -> None:
    """Missing selection computes S minus both runtime membership sets."""
    selection = LogicalSelection.from_ids((1, 2, 3, 5, 8), 10)

    missing = logical_missing(
        selection,
        resident=(1, 8),
        inflight=(2, 7),
        codec=LogicalCodec.BITMAP,
    )

    assert missing.ids() == (3, 5)


def test_ordered_merkle_append_prefix_root_and_lcp() -> None:
    """Ordered roots support immutable append and longest-common-prefix."""
    empty = OrderedMerklePrefix()
    left = empty.extend(("a", "b", "c"))
    right = empty.extend(("a", "b", "d", "e"))
    reordered = empty.extend(("b", "a", "c"))

    assert empty.chunk_count == 0
    assert left.chunk_count == 3
    assert left.longest_common_prefix(right) == 2
    assert left.longest_common_prefix(reordered) == 0
    assert left.prefix_root(2) == right.prefix_root(2)
    assert left.root != reordered.root
    assert empty.chunk_count == 0


def test_canonical_plan_hash_and_exact_key_mismatch() -> None:
    """Canonical hashes are stable and exact plans reject another key."""
    plan = _layer_plan((1, 3, 7))
    identical = _layer_plan((1, 3, 7))

    assert plan.plan_hash == identical.plan_hash
    assert plan.plan_hash == canonical_checksum(plan)
    plan.require_key(_decision_key())
    with pytest.raises(ValueError, match="key mismatch"):
        plan.require_key(_decision_key(query_digest="query-b"))


def test_canonical_checksum_is_mapping_order_independent() -> None:
    """Equivalent nested mappings and sequences have one stable digest."""
    left = {
        "model": "glm",
        "shape": {"layers": 4, "types": ["full", "shared"]},
    }
    right = {
        "shape": {"types": ["full", "shared"], "layers": 4},
        "model": "glm",
    }

    assert canonical_checksum(left) == canonical_checksum(right)


def test_canonical_checksum_rejects_non_string_mapping_keys() -> None:
    """Mappings with ambiguous JSON key coercion are rejected."""
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        canonical_checksum({1: "layer"})


def test_public_values_are_immutable() -> None:
    """Plans, keys, selections, and ranges cannot be mutated after creation."""
    plan = _layer_plan((1, 2))
    read_range = PhysicalRange("raw", "chunk-a", 0, 512, 12, 132, (1,))

    with pytest.raises(FrozenInstanceError):
        plan.correctness = Correctness.EXACT  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.key.layer_id = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        read_range.length = 1024  # type: ignore[misc]
    assert not hasattr(plan, "__dict__")


def test_binding_alignment_payload_skip_and_safe_merge() -> None:
    """Binding aligns reads and merges only contiguous same-chunk payloads."""
    selection = LogicalSelection.from_ids((0, 1, 2, 3), 4)
    sources = {
        0: LogicalBlockSource(0, "raw-a", "chunk-a", 100, 132),
        1: LogicalBlockSource(1, "raw-a", "chunk-a", 232, 132),
        2: LogicalBlockSource(2, "raw-a", "chunk-b", 364, 132),
        3: LogicalBlockSource(3, "raw-b", "chunk-b", 496, 132),
    }

    ranges = bind_logical_selection(
        selection,
        sources,
        alignment=512,
        merge_ranges=True,
    )

    assert ranges == (
        PhysicalRange("raw-a", "chunk-a", 0, 512, 100, 264, (0, 1)),
        PhysicalRange("raw-a", "chunk-b", 0, 512, 364, 132, (2,)),
        PhysicalRange("raw-b", "chunk-b", 0, 1024, 496, 132, (3,)),
    )
    represented = {logical_id for item in ranges for logical_id in item.logical_ids}
    assert represented == {0, 1, 2, 3}


def test_binding_rejects_missing_or_mismatched_sources() -> None:
    """Physical binding cannot silently drop or alias selected logical ids."""
    selection = LogicalSelection.from_ids((1, 2), 4)
    with pytest.raises(ValueError, match="missing physical source"):
        bind_logical_selection(
            selection,
            {1: LogicalBlockSource(1, "raw", "a", 0, 64)},
            alignment=64,
        )
    with pytest.raises(ValueError, match="does not match"):
        bind_logical_selection(
            selection,
            {
                1: LogicalBlockSource(1, "raw", "a", 0, 64),
                2: LogicalBlockSource(3, "raw", "a", 64, 64),
            },
            alignment=64,
        )


def test_lowering_cost_model_selects_list_range_or_bulk() -> None:
    """Injected costs deterministically select each lowering candidate."""
    plan = _layer_plan((0, 1, 4), universe_size=6)
    sources = {
        logical_id: LogicalBlockSource(
            logical_id,
            "raw",
            "chunk",
            logical_id * 64,
            64,
        )
        for logical_id in range(6)
    }
    list_plan = lower_layer_plan(
        plan,
        sources,
        alignment=64,
        cost_model=LoweringCostModel(
            list_per_range=0,
            range_base=10,
            bulk_base=10,
        ),
    )
    range_plan = lower_layer_plan(
        plan,
        sources,
        alignment=64,
        cost_model=LoweringCostModel(
            list_base=10,
            range_per_range=0,
            bulk_base=10,
        ),
    )
    bulk_plan = lower_layer_plan(
        plan,
        sources,
        alignment=64,
        cost_model=LoweringCostModel(
            list_base=10,
            range_base=10,
            bulk_per_range=0,
        ),
    )

    assert list_plan.placement is PlacementKind.LIST
    assert range_plan.placement is PlacementKind.RANGE
    assert bulk_plan.placement is PlacementKind.BULK
    assert bulk_plan.ranges[0].length == 6 * 64
    for bound in (list_plan, range_plan, bulk_plan):
        assert {
            logical_id
            for read_range in bound.ranges
            for logical_id in read_range.logical_ids
        } == {0, 1, 4}
        assert len(bound.checksum) == 64


def test_cost_model_charges_bulk_overread() -> None:
    """A configurable overread penalty can reject an otherwise cheap bulk read."""
    plan = _layer_plan((0, 5), universe_size=6)
    sources = {
        logical_id: LogicalBlockSource(
            logical_id,
            "raw",
            "chunk",
            logical_id * 64,
            64,
        )
        for logical_id in range(6)
    }

    bound = lower_layer_plan(
        plan,
        sources,
        alignment=64,
        cost_model=LoweringCostModel(
            list_per_range=1,
            range_per_range=1,
            bulk_per_range=0,
            overread_cost=1,
        ),
    )

    assert bound.placement is PlacementKind.LIST
