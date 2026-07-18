# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for static Index-to-I/O profiles."""

# Third Party
import pytest

# First Party
from lmcache.v1.index_to_io_profile import (
    IndexReuseKind,
    IndexerMode,
    PrefetchCandidateMeasurement,
    PrefetchStrategy,
    WorkloadBucket,
    extract_glm_dsa_topology,
    select_static_prefetch_profile,
)


def _glm_config() -> dict[str, object]:
    return {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 8,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_topk_pattern": None,
        "index_skip_topk_offset": 3,
        "indexer_rope_interleave": True,
        "index_share_for_mtp_iteration": True,
        "indexer_types": [
            "full",
            "full",
            "full",
            "shared",
            "shared",
            "shared",
            "full",
            "shared",
        ],
    }


def test_extract_glm_dsa_topology_preserves_indexshare_groups() -> None:
    profile = extract_glm_dsa_topology(_glm_config(), "glm-weights-a")

    assert profile.index_topk == 2048
    assert profile.shares_mtp_index is True
    assert [group.source_layer for group in profile.index_groups] == [0, 1, 2, 6]
    assert [group.consumer_layers for group in profile.index_groups] == [
        (0,),
        (1,),
        (2, 3, 4, 5),
        (6, 7),
    ]
    assert all(
        group.reuse_kind is IndexReuseKind.ARCHITECTURAL
        for group in profile.index_groups
    )
    assert profile.layers[2].indexer_mode is IndexerMode.FULL
    assert profile.layers[5].indexer_mode is IndexerMode.SHARED
    assert profile.layers[5].index_source_layer == 2
    assert profile.layers[5].index_group_id == profile.layers[2].index_group_id


def test_extract_glm_dsa_topology_hash_changes_with_structure() -> None:
    original = extract_glm_dsa_topology(_glm_config(), "glm-weights-a")
    changed_config = _glm_config()
    changed_config["index_topk"] = 1024
    changed = extract_glm_dsa_topology(changed_config, "glm-weights-a")

    assert original.config_digest != changed.config_digest
    assert original.profile_hash != changed.profile_hash


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"model_type": "other"}, "model_type"),
        ({"indexer_types": ["shared"] * 8}, "precede"),
        ({"num_hidden_layers": 7}, "match"),
        ({"indexer_types": ["full"] * 7 + ["unknown"]}, "unsupported"),
    ],
)
def test_extract_glm_dsa_topology_rejects_invalid_structure(
    update: dict[str, object], message: str
) -> None:
    config = _glm_config()
    config.update(update)

    with pytest.raises(ValueError, match=message):
        extract_glm_dsa_topology(config, "glm-weights-a")


def test_static_profiler_prefers_hidden_low_byte_candidate() -> None:
    topology = extract_glm_dsa_topology(_glm_config(), "glm-weights-a")
    bucket = WorkloadBucket(
        prefix_tokens_min=400_000,
        prefix_tokens_max=520_000,
        query_tokens_min=8_000,
        query_tokens_max=16_000,
    )
    sparse = PrefetchCandidateMeasurement(
        strategy=PrefetchStrategy.SPARSE,
        lookahead_layers=2,
        service_us_p95=800.0,
        overlap_us_p05=900.0,
        read_bytes_p95=32 << 20,
        hbm_bytes_p95=24 << 20,
        samples=30,
    )
    bulk = PrefetchCandidateMeasurement(
        strategy=PrefetchStrategy.BULK,
        lookahead_layers=1,
        service_us_p95=600.0,
        overlap_us_p05=700.0,
        read_bytes_p95=256 << 20,
        hbm_bytes_p95=256 << 20,
        samples=30,
    )

    profile = select_static_prefetch_profile(
        topology,
        "h200-tp8-long-prefill-v1",
        {(5, bucket): (bulk, sparse)},
    )

    selected = profile.layers[0]
    assert selected.preferred_strategy is PrefetchStrategy.SPARSE
    assert selected.preferred_lookahead_layers == 2
    assert selected.candidates == (bulk, sparse)
    assert profile.topology_hash == topology.profile_hash


def test_static_profiler_minimizes_predicted_gate_stall_first() -> None:
    topology = extract_glm_dsa_topology(_glm_config(), "glm-weights-a")
    bucket = WorkloadBucket(0, 128_000, 1, 4_096)
    low_bytes_but_stalls = PrefetchCandidateMeasurement(
        strategy=PrefetchStrategy.SPARSE,
        lookahead_layers=1,
        service_us_p95=900.0,
        overlap_us_p05=400.0,
        read_bytes_p95=8 << 20,
        hbm_bytes_p95=8 << 20,
        samples=10,
    )
    higher_bytes_hidden = PrefetchCandidateMeasurement(
        strategy=PrefetchStrategy.RANGE,
        lookahead_layers=2,
        service_us_p95=700.0,
        overlap_us_p05=800.0,
        read_bytes_p95=32 << 20,
        hbm_bytes_p95=16 << 20,
        samples=10,
    )

    profile = select_static_prefetch_profile(
        topology,
        "calibration-a",
        {(3, bucket): (low_bytes_but_stalls, higher_bytes_hidden)},
    )

    assert profile.layers[0].preferred_strategy is PrefetchStrategy.RANGE


def test_static_profiler_rejects_unknown_layer() -> None:
    topology = extract_glm_dsa_topology(_glm_config(), "glm-weights-a")
    bucket = WorkloadBucket(0, 1, 0, 1)
    candidate = PrefetchCandidateMeasurement(
        strategy=PrefetchStrategy.DEMAND,
        lookahead_layers=0,
        service_us_p95=1.0,
        overlap_us_p05=0.0,
        read_bytes_p95=1,
        hbm_bytes_p95=1,
        samples=1,
    )

    with pytest.raises(ValueError, match="unknown layer"):
        select_static_prefetch_profile(
            topology, "calibration-a", {(8, bucket): (candidate,)}
        )
