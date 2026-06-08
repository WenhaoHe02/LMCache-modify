# SPDX-License-Identifier: Apache-2.0
# First Party
from test_csa_selection_stability import (
    compute_cross_layer,
    compute_cross_step,
    compute_random_baseline,
    compute_unique_pool,
)


def test_compute_cross_step_overlap() -> None:
    """Cross-step overlap is measured against the previous same-layer set."""
    selections = {
        3: [{1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 6, 7}],
        7: [{10, 11}, {10, 12}],
    }

    by_layer, all_overlaps = compute_cross_step(selections)

    assert by_layer[3]["mean"] == 0.625
    assert by_layer[7]["mean"] == 0.5
    assert all_overlaps == [0.75, 0.5, 0.5]


def test_compute_cross_layer_adjacent_overlap() -> None:
    """Cross-layer overlap compares adjacent CSA layers at the same step."""
    selections = {
        3: [{1, 2, 3, 4}, {10, 11, 12, 13}],
        7: [{3, 4, 5, 6}, {12, 13, 14, 15}],
        9: [{4, 5, 6, 7}, {14, 15, 16, 17}],
    }

    by_pair, directed, jaccards = compute_cross_layer(selections)

    assert by_pair["3->7"]["overlap_mean"] == 0.5
    assert by_pair["7->9"]["overlap_mean"] == 0.5
    assert directed == [0.5, 0.5, 0.5, 0.5]
    assert jaccards == [1 / 3, 1 / 3, 1 / 3, 1 / 3]


def test_compute_unique_pool_reuse_factor() -> None:
    """Unified-pool reuse factor is naive per-layer blocks divided by uniques."""
    selections = {
        3: [{1, 2, 3, 4}, {10, 11, 12, 13}],
        7: [{3, 4, 5, 6}, {12, 13, 14, 15}],
        9: [{4, 5, 6, 7}, {14, 15, 16, 17}],
    }

    summary = compute_unique_pool(selections)

    assert summary["layers"] == 3.0
    assert summary["steps"] == 2.0
    assert summary["naive_blocks_mean"] == 12.0
    assert summary["unique_blocks_mean"] == 7.0
    assert summary["reuse_factor_mean"] == 12 / 7
    assert summary["unique_ratio_mean"] == 7 / 12


def test_compute_random_baseline() -> None:
    """Random baseline shows how much reuse appears from finite block space."""
    baseline = compute_random_baseline(
        n_blocks=8192,
        topk_per_layer=1024,
        n_layers=30,
    )

    assert baseline["adjacent_overlap_mean"] == 0.125
    assert baseline["adjacent_jaccard_mean"] == 0.125 / 1.875
    assert baseline["naive_blocks_mean"] == 30720.0
    assert baseline["reuse_factor_mean"] > 3.8
