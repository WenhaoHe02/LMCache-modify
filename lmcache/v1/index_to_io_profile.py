# SPDX-License-Identifier: Apache-2.0
"""Static model and prefetch profiles for the Index-to-I/O compiler.

The structural profile captures model-defined sparse-attention semantics, such
as GLM IndexShare ownership.  The calibrated profile captures offline latency
measurements without turning workload-dependent observations into correctness
claims.  Both profiles are immutable and independent of CUDA and storage
runtime state.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

# First Party
from lmcache.v1.index_to_io_plan import canonical_checksum


class AttentionFamily(str, Enum):
    """Attention implementation used by a transformer layer."""

    DENSE = "dense"
    DSA = "dsa"
    CSA = "csa"
    HCA = "hca"
    SWA = "swa"


class IndexerMode(str, Enum):
    """How a layer obtains its sparse-attention index."""

    NONE = "none"
    FULL = "full"
    SHARED = "shared"


class IndexReuseKind(str, Enum):
    """Authority under which an index is reused across layers."""

    PER_LAYER = "per_layer"
    ARCHITECTURAL = "architectural"
    QUALITY_GUARDED = "quality_guarded"


class PrefetchStrategy(str, Enum):
    """Physical strategy evaluated by an offline static profiler."""

    DEMAND = "demand"
    SPARSE = "sparse"
    RANGE = "range"
    BULK = "bulk"


@dataclass(frozen=True, slots=True)
class LayerTopology:
    """Structural sparse-attention metadata for one layer.

    Args:
        layer_id: Zero-based transformer layer id.
        attention_family: Attention implementation used by the layer.
        indexer_mode: Whether the layer computes or shares an index.
        index_group_id: Stable id of the cross-layer index group, if any.
        index_source_layer: Layer that computes the group's index, if any.
    """

    layer_id: int
    attention_family: AttentionFamily
    indexer_mode: IndexerMode
    index_group_id: str | None
    index_source_layer: int | None

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        has_index = self.indexer_mode is not IndexerMode.NONE
        if has_index != (self.index_group_id is not None):
            raise ValueError("indexed layers require an index_group_id")
        if has_index != (self.index_source_layer is not None):
            raise ValueError("indexed layers require an index_source_layer")
        if self.indexer_mode is IndexerMode.FULL:
            if self.index_source_layer != self.layer_id:
                raise ValueError("full indexers must own their index group")
        if self.indexer_mode is IndexerMode.SHARED:
            if self.index_source_layer is None:
                raise ValueError("shared layers require an index source")
            if self.index_source_layer >= self.layer_id:
                raise ValueError("shared layers must use a preceding index source")


@dataclass(frozen=True, slots=True)
class IndexGroupTopology:
    """One index computation and all attention layers that consume it.

    Args:
        group_id: Stable group identifier.
        source_layer: Full layer that computes the index.
        consumer_layers: Sorted layers that consume the index, including the
            source layer.
        reuse_kind: Authority that permits the cross-layer reuse.
    """

    group_id: str
    source_layer: int
    consumer_layers: tuple[int, ...]
    reuse_kind: IndexReuseKind

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        if self.source_layer < 0:
            raise ValueError("source_layer must be non-negative")
        if not self.consumer_layers:
            raise ValueError("consumer_layers must be non-empty")
        if tuple(sorted(set(self.consumer_layers))) != self.consumer_layers:
            raise ValueError("consumer_layers must be sorted and unique")
        if self.source_layer not in self.consumer_layers:
            raise ValueError("source_layer must be a consumer")


@dataclass(frozen=True, slots=True)
class ModelTopologyProfile:
    """Immutable model topology consumed by the Index-to-I/O compiler.

    Args:
        schema_version: Profile schema version.
        model_fingerprint: Digest of the exact model weights and adapters.
        model_type: Model architecture identifier from the model config.
        config_digest: Digest of the structural config fields used here.
        index_topk: Number of tokens selected by each sparse index.
        layers: Per-layer topology in layer order.
        index_groups: Cross-layer index ownership groups.
        shares_mtp_index: Whether MTP iterations reuse the model index.
    """

    schema_version: int
    model_fingerprint: str
    model_type: str
    config_digest: str
    index_topk: int
    layers: tuple[LayerTopology, ...]
    index_groups: tuple[IndexGroupTopology, ...]
    shares_mtp_index: bool

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if not self.model_fingerprint:
            raise ValueError("model_fingerprint must be non-empty")
        if not self.model_type:
            raise ValueError("model_type must be non-empty")
        if not self.config_digest:
            raise ValueError("config_digest must be non-empty")
        if self.index_topk <= 0:
            raise ValueError("index_topk must be positive")
        expected_ids = tuple(range(len(self.layers)))
        if tuple(layer.layer_id for layer in self.layers) != expected_ids:
            raise ValueError("layers must be dense and ordered by layer_id")
        group_ids = tuple(group.group_id for group in self.index_groups)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("index group ids must be unique")
        groups = {group.group_id: group for group in self.index_groups}
        for layer in self.layers:
            if layer.index_group_id is None:
                continue
            group = groups.get(layer.index_group_id)
            if group is None:
                raise ValueError("layer references an unknown index group")
            if layer.layer_id not in group.consumer_layers:
                raise ValueError("layer is absent from its index group")
            if layer.index_source_layer != group.source_layer:
                raise ValueError("layer and group disagree on index ownership")

    @property
    def profile_hash(self) -> str:
        """Return a canonical digest of the complete structural profile."""
        return canonical_checksum(self)


@dataclass(frozen=True, slots=True)
class WorkloadBucket:
    """Context and query-size interval represented by offline measurements.

    Args:
        prefix_tokens_min: Inclusive lower prefix length.
        prefix_tokens_max: Inclusive upper prefix length.
        query_tokens_min: Inclusive lower query length.
        query_tokens_max: Inclusive upper query length.
    """

    prefix_tokens_min: int
    prefix_tokens_max: int
    query_tokens_min: int
    query_tokens_max: int

    def __post_init__(self) -> None:
        if self.prefix_tokens_min < 0 or self.query_tokens_min < 0:
            raise ValueError("bucket lower bounds must be non-negative")
        if self.prefix_tokens_max < self.prefix_tokens_min:
            raise ValueError("invalid prefix-token interval")
        if self.query_tokens_max < self.query_tokens_min:
            raise ValueError("invalid query-token interval")


@dataclass(frozen=True, slots=True)
class PrefetchCandidateMeasurement:
    """Robust offline measurement of one prefetch candidate.

    Args:
        strategy: Physical I/O strategy under test.
        lookahead_layers: Number of layers between submission and consumption.
        service_us_p95: P95 I/O plus materialization service time.
        overlap_us_p05: Conservative P05 compute window available for hiding it.
        read_bytes_p95: P95 physical SSD bytes read.
        hbm_bytes_p95: P95 destination or staging HBM footprint.
        samples: Number of observations in the bucket.
    """

    strategy: PrefetchStrategy
    lookahead_layers: int
    service_us_p95: float
    overlap_us_p05: float
    read_bytes_p95: int
    hbm_bytes_p95: int
    samples: int

    def __post_init__(self) -> None:
        if self.lookahead_layers < 0:
            raise ValueError("lookahead_layers must be non-negative")
        if self.service_us_p95 < 0 or self.overlap_us_p05 < 0:
            raise ValueError("latencies must be non-negative")
        if self.read_bytes_p95 < 0 or self.hbm_bytes_p95 < 0:
            raise ValueError("byte counts must be non-negative")
        if self.samples <= 0:
            raise ValueError("samples must be positive")

    @property
    def predicted_gate_stall_us(self) -> float:
        """Return conservative target-layer stall after compute/I/O overlap."""
        return max(0.0, self.service_us_p95 - self.overlap_us_p05)


@dataclass(frozen=True, slots=True)
class LayerPrefetchProfile:
    """Static prefetch prior for one layer and workload bucket.

    Args:
        layer_id: Target attention layer.
        bucket: Context/query interval covered by the measurements.
        candidates: Measurements retained for runtime fallback and admission.
        preferred_strategy: Candidate selected by the static profiler.
        preferred_lookahead_layers: Selected submission distance.
    """

    layer_id: int
    bucket: WorkloadBucket
    candidates: tuple[PrefetchCandidateMeasurement, ...]
    preferred_strategy: PrefetchStrategy
    preferred_lookahead_layers: int

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if not self.candidates:
            raise ValueError("candidates must be non-empty")
        matches = [
            candidate
            for candidate in self.candidates
            if candidate.strategy is self.preferred_strategy
            and candidate.lookahead_layers == self.preferred_lookahead_layers
        ]
        if len(matches) != 1:
            raise ValueError("preferred candidate must appear exactly once")


@dataclass(frozen=True, slots=True)
class StaticPrefetchProfile:
    """Versioned offline calibration artifact for one model topology.

    Args:
        schema_version: Calibration schema version.
        topology_hash: Hash of the structural profile used during calibration.
        calibration_id: Stable identifier of the workload and hardware run.
        layers: Per-layer and per-bucket static prefetch priors.
    """

    schema_version: int
    topology_hash: str
    calibration_id: str
    layers: tuple[LayerPrefetchProfile, ...]

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if not self.topology_hash:
            raise ValueError("topology_hash must be non-empty")
        if not self.calibration_id:
            raise ValueError("calibration_id must be non-empty")
        keys = tuple((profile.layer_id, profile.bucket) for profile in self.layers)
        if len(keys) != len(set(keys)):
            raise ValueError("layer/bucket calibration keys must be unique")

    @property
    def profile_hash(self) -> str:
        """Return a canonical digest of the calibration artifact."""
        return canonical_checksum(self)


def extract_glm_dsa_topology(
    config: Mapping[str, Any], model_fingerprint: str
) -> ModelTopologyProfile:
    """Extract GLM DSA and IndexShare topology from a model config.

    The extractor prefers the model's explicit ``indexer_types``.  vLLM 0.26
    also accepts ``index_topk_pattern`` or the frequency/offset form, so those
    native rules are used when the explicit list is absent. A ``full`` layer
    starts a new group; following ``shared`` layers consume that Full result.

    Args:
        config: Parsed GLM model ``config.json`` mapping.
        model_fingerprint: Digest of the exact model weights and adapters.

    Returns:
        Validated immutable model topology profile.

    Raises:
        ValueError: If required fields are absent or structurally inconsistent.
    """
    model_type = str(config.get("model_type", ""))
    if model_type != "glm_moe_dsa":
        raise ValueError("expected model_type='glm_moe_dsa'")
    if not model_fingerprint:
        raise ValueError("model_fingerprint must be non-empty")

    layer_count = _require_int(config, "num_hidden_layers")
    index_topk = _require_int(config, "index_topk")
    raw_types = config.get("indexer_types")
    if isinstance(raw_types, Sequence) and not isinstance(raw_types, (str, bytes)):
        indexer_types = tuple(str(value).lower() for value in raw_types)
        # GLM-5.2 publishes one additional entry for its MTP decoder while
        # ``num_hidden_layers`` describes only the main decoder stack. The MTP
        # layer reuses the final main-stack index, so it must not create an
        # extra prefetch edge or require another registered decoder here.
        if (
            len(indexer_types) == layer_count + 1
            and bool(config.get("index_share_for_mtp_iteration", False))
            and config.get("num_nextn_predict_layers") == 1
        ):
            indexer_types = indexer_types[:layer_count]
        if len(indexer_types) != layer_count:
            raise ValueError("indexer_types must match num_hidden_layers")
    else:
        raw_pattern = config.get("index_topk_pattern")
        if isinstance(raw_pattern, Sequence) and not isinstance(
            raw_pattern, (str, bytes)
        ):
            if len(raw_pattern) != layer_count:
                raise ValueError("index_topk_pattern must match num_hidden_layers")
            indexer_types = tuple(
                "shared" if str(value).upper() == "S" else "full"
                for value in raw_pattern
            )
        else:
            frequency = config.get("index_topk_freq", 1)
            offset = config.get("index_skip_topk_offset", 2)
            if not isinstance(frequency, int) or frequency <= 0:
                raise ValueError("index_topk_freq must be a positive integer")
            if not isinstance(offset, int):
                raise ValueError("index_skip_topk_offset must be an integer")
            indexer_types = tuple(
                "shared"
                if max(layer_id - offset + 1, 0) % frequency != 0
                else "full"
                for layer_id in range(layer_count)
            )

    layers: list[LayerTopology] = []
    group_consumers: list[list[int]] = []
    group_sources: list[int] = []
    current_group_index: int | None = None

    for layer_id, indexer_type in enumerate(indexer_types):
        if indexer_type == "full":
            current_group_index = len(group_sources)
            group_sources.append(layer_id)
            group_consumers.append([layer_id])
            source_layer = layer_id
            indexer_mode = IndexerMode.FULL
        elif indexer_type == "shared":
            if current_group_index is None:
                raise ValueError("shared indexer cannot precede the first full indexer")
            source_layer = group_sources[current_group_index]
            group_consumers[current_group_index].append(layer_id)
            indexer_mode = IndexerMode.SHARED
        else:
            raise ValueError(f"unsupported GLM indexer type: {indexer_type!r}")

        group_id = f"glm-index-{source_layer}"
        layers.append(
            LayerTopology(
                layer_id=layer_id,
                attention_family=AttentionFamily.DSA,
                indexer_mode=indexer_mode,
                index_group_id=group_id,
                index_source_layer=source_layer,
            )
        )

    groups = tuple(
        IndexGroupTopology(
            group_id=f"glm-index-{source_layer}",
            source_layer=source_layer,
            consumer_layers=tuple(consumers),
            reuse_kind=IndexReuseKind.ARCHITECTURAL,
        )
        for source_layer, consumers in zip(group_sources, group_consumers, strict=True)
    )
    structural_config = {
        "model_type": model_type,
        "num_hidden_layers": layer_count,
        "index_topk": index_topk,
        "index_topk_freq": config.get("index_topk_freq"),
        "index_topk_pattern": config.get("index_topk_pattern"),
        "index_skip_topk_offset": config.get("index_skip_topk_offset"),
        "indexer_rope_interleave": config.get("indexer_rope_interleave"),
        "indexer_types": indexer_types,
        "index_share_for_mtp_iteration": config.get(
            "index_share_for_mtp_iteration", False
        ),
    }
    return ModelTopologyProfile(
        schema_version=1,
        model_fingerprint=model_fingerprint,
        model_type=model_type,
        config_digest=canonical_checksum(structural_config),
        index_topk=index_topk,
        layers=tuple(layers),
        index_groups=groups,
        shares_mtp_index=bool(config.get("index_share_for_mtp_iteration", False)),
    )


def select_static_prefetch_profile(
    topology: ModelTopologyProfile,
    calibration_id: str,
    measurements: Mapping[
        tuple[int, WorkloadBucket], Sequence[PrefetchCandidateMeasurement]
    ],
) -> StaticPrefetchProfile:
    """Select a conservative static prefetch prior from offline measurements.

    Candidates first minimize predicted target-layer gate stall.  Ties prefer
    fewer SSD bytes, less HBM, and then shorter lookahead.  The full candidate
    set remains in the artifact so the runtime can reject the static choice
    when current residency, congestion, or memory pressure differs.

    Args:
        topology: Structural model profile used by the calibration run.
        calibration_id: Stable workload/hardware calibration identifier.
        measurements: Candidate measurements keyed by target layer and bucket.

    Returns:
        Immutable static prefetch profile.

    Raises:
        ValueError: If a key references an unknown layer or has no candidates.
    """
    profiles: list[LayerPrefetchProfile] = []
    for (layer_id, bucket), raw_candidates in sorted(
        measurements.items(), key=lambda item: (item[0][0], repr(item[0][1]))
    ):
        if layer_id < 0 or layer_id >= len(topology.layers):
            raise ValueError("measurement references an unknown layer")
        candidates = tuple(raw_candidates)
        if not candidates:
            raise ValueError("each measurement key requires candidates")
        identities = tuple(
            (candidate.strategy, candidate.lookahead_layers)
            for candidate in candidates
        )
        if len(identities) != len(set(identities)):
            raise ValueError("candidate strategy/lookahead pairs must be unique")
        preferred = min(
            candidates,
            key=lambda candidate: (
                candidate.predicted_gate_stall_us,
                candidate.read_bytes_p95,
                candidate.hbm_bytes_p95,
                candidate.lookahead_layers,
                candidate.service_us_p95,
                candidate.strategy.value,
            ),
        )
        profiles.append(
            LayerPrefetchProfile(
                layer_id=layer_id,
                bucket=bucket,
                candidates=candidates,
                preferred_strategy=preferred.strategy,
                preferred_lookahead_layers=preferred.lookahead_layers,
            )
        )
    return StaticPrefetchProfile(
        schema_version=1,
        topology_hash=topology.profile_hash,
        calibration_id=calibration_id,
        layers=tuple(profiles),
    )


def _require_int(config: Mapping[str, Any], field: str) -> int:
    value = config.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value
