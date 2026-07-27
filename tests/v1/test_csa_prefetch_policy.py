# SPDX-License-Identifier: Apache-2.0
"""Tests for the public per-layer CSA prefetch policy."""

# Third Party
import pytest

# First Party
from lmcache.v1.csa_prefetch_policy import (
    CSAPrefetchLookaheadPolicy,
    build_residual_prefetch_sources,
)


def test_profile80_enables_only_profiled_deep_targets() -> None:
    """The checked-in profile disables shallow and enables deep L2 targets."""
    policy = CSAPrefetchLookaheadPolicy("profile80")

    assert policy.lookahead_for(24) == 0
    assert policy.lookahead_for(26) == 2
    assert policy.lookahead_for(42) == 2
    assert policy.lookahead_for(44) == 0
    assert policy.two_layer_targets([24, 26, 27, 42, 44]) == {26, 27, 42}
    assert policy.disabled_targets([24, 26, 27, 42, 44]) == {24, 44}


def test_profile80_hybrid_moves_only_correction_heavy_targets_to_l1() -> None:
    """The optimized profile preserves one fire while moving two targets."""
    policy = CSAPrefetchLookaheadPolicy("profile80_hybrid")
    targets = list(range(2, 43, 2))

    assert policy.one_layer_targets(targets) == {36, 42}
    assert policy.two_layer_targets(targets) == {26, 28, 30, 32, 34, 38, 40}
    assert policy.disabled_targets(targets) == set(range(2, 25, 2))
    sources = build_residual_prefetch_sources(targets, policy)
    assert sources[35] == (36, 1)
    assert sources[41] == (42, 1)


def test_explicit_policy_defaults_unlisted_targets_to_demand_only() -> None:
    """Omitted targets never silently re-enable a legacy prediction path."""
    policy = CSAPrefetchLookaheadPolicy("26-42:2")

    assert policy.lookahead_for(2) == 0
    assert policy.lookahead_for(25) == 0
    assert policy.lookahead_for(30) == 2
    assert policy.lookahead_for(44) == 0


def test_explicit_default_zero_disables_unlisted_targets() -> None:
    """A zero default keeps prediction limited to selected deep layers."""
    policy = CSAPrefetchLookaheadPolicy("default:0,26-42:2")

    assert policy.lookahead_for(24) == 0
    assert policy.lookahead_for(30) == 2
    assert policy.lookahead_for(44) == 0


def test_recall_profile_uses_inclusive_eighty_percent_threshold() -> None:
    """A layer at exactly 80% recall qualifies for early L2 prefetch."""
    policy = CSAPrefetchLookaheadPolicy.from_recall_profile(
        {24: 0.7999, 26: 0.8, 28: 0.91}
    )

    assert policy.lookahead_for(24) == 0
    assert policy.lookahead_for(26) == 2
    assert policy.lookahead_for(28) == 2


def test_source_mapping_uses_exactly_one_l2_for_each_deep_target() -> None:
    """Disabled shallow targets have no hook and deep targets have only L2."""
    policy = CSAPrefetchLookaheadPolicy("default:0,26-42:2")

    sources = build_residual_prefetch_sources([24, 26, 28], policy)

    assert sources == {
        24: (26, 2),
        26: (28, 2),
    }


def test_source_mapping_supports_selected_one_layer_targets() -> None:
    """A target may use one closer source without adding a second fire."""
    policy = CSAPrefetchLookaheadPolicy("default:0,26:2,28:1")

    assert policy.lookahead_for(26) == 2
    assert policy.lookahead_for(28) == 1
    assert build_residual_prefetch_sources([26, 28], policy) == {
        24: (26, 2),
        27: (28, 1),
    }


def test_dsv4_flash_21_layer_profile_has_nine_deep_targets_without_conflicts() -> None:
    """V4-Flash has 21 even CSA layers, not the legacy 30-layer geometry."""
    csa_layer_ids = list(range(2, 43, 2))
    policy = CSAPrefetchLookaheadPolicy("profile80")

    assert len(csa_layer_ids) == 21
    assert policy.disabled_targets(csa_layer_ids) == set(range(2, 25, 2))
    assert policy.two_layer_targets(csa_layer_ids) == set(range(26, 43, 2))

    sources = build_residual_prefetch_sources(csa_layer_ids, policy)
    assert set(sources) == set(range(24, 41, 2))
    for source in range(24, 41, 2):
        assert sources[source] == (source + 2, 2)


@pytest.mark.parametrize(
    "specification",
    [
        "2:3",
        "default:3",
        "default:0,default:1",
        "4-2:2",
        "bad",
        "2-4:1,3:2",
    ],
)
def test_invalid_policy_is_rejected(specification: str) -> None:
    """Malformed, out-of-range, reversed, and conflicting entries fail."""
    with pytest.raises(ValueError):
        CSAPrefetchLookaheadPolicy(specification)
