# SPDX-License-Identifier: Apache-2.0
"""Planning and transport primitives for TP-sharded SSD prefetch.

The module deliberately separates deterministic CPU planning from the CUDA
and NCCL transport.  Request hot paths can therefore validate a complete plan
before submitting SSD I/O, while unit tests exercise the same ownership and
inverse-mapping contracts without requiring a GPU.
"""

from __future__ import annotations

# Standard
from collections import deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import os
import threading
from typing import Any, Mapping, Optional, Protocol, Sequence

# Third Party
import torch

# First Party
from lmcache.v1.kv_object_store import KVObjectByteRange


class SSDReadMode(str, Enum):
    """Supported SSD read modes for one logical cache group."""

    LOCAL_DIRECT = "local_direct"
    CP_LOCAL_INDEXER = "cp_local_indexer"
    SHARD_GATHER_DENSE = "shard_gather_dense"
    SHARD_GATHER_PREDICTED = "shard_gather_predicted"


@dataclass(frozen=True, slots=True)
class RowRange:
    """One half-open, physically coalescible logical row range.

    Args:
        start: Inclusive logical row.
        end: Exclusive logical row.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate the half-open range."""
        if self.start < 0 or self.end < self.start:
            raise ValueError("row range must satisfy 0 <= start <= end")

    @property
    def length(self) -> int:
        """Return the number of rows in the range."""
        return self.end - self.start


def compile_layer_major_read_ranges(
    block_ids: Sequence[int],
    *,
    available_blocks: int,
    block_nbytes: int,
    layer_byte_offset: int,
    aligned_read_length: int,
) -> tuple[KVObjectByteRange, ...]:
    """Compile sector-aligned ranges for one layer-major object segment.

    A full-layer tail read includes the allocation's final sector padding;
    consumers still materialize only the model payload.
    """
    ids = tuple(int(block_id) for block_id in block_ids)
    if (
        not ids
        or available_blocks <= 0
        or block_nbytes <= 0
        or layer_byte_offset < 0
        or aligned_read_length <= 0
    ):
        raise ValueError("invalid layer-major range geometry")
    if any(
        current <= previous for previous, current in zip(ids, ids[1:], strict=False)
    ):
        raise ValueError("layer-major block ids must be strictly increasing")
    if ids[0] < 0 or ids[-1] >= available_blocks:
        raise ValueError("layer-major block id is out of range")
    payload_nbytes = available_blocks * block_nbytes
    if aligned_read_length < payload_nbytes:
        raise ValueError("aligned layer-major object truncates its payload")
    if layer_byte_offset % 512 or aligned_read_length % 512:
        raise ValueError("layer-major object allocation is not sector aligned")

    ranges: list[KVObjectByteRange] = []
    target_offset = 0
    run_start = ids[0]
    run_length = 1
    for block_id in (*ids[1:], None):
        if block_id is not None and block_id == run_start + run_length:
            run_length += 1
            continue
        length = run_length * block_nbytes
        if (
            block_id is None
            and run_start + run_length == available_blocks
            and aligned_read_length > payload_nbytes
        ):
            length = aligned_read_length - run_start * block_nbytes
        source_offset = layer_byte_offset + run_start * block_nbytes
        if source_offset % 512 or length % 512 or target_offset % 512:
            raise ValueError(
                "layer-major selected run cannot be represented by "
                "sector-aligned Tutti I/O"
            )
        if source_offset + length > layer_byte_offset + aligned_read_length:
            raise ValueError("layer-major selected run exceeds its allocation")
        ranges.append(
            KVObjectByteRange(
                offset=source_offset,
                length=length,
                target_offset=target_offset,
            )
        )
        target_offset += length
        if block_id is not None:
            run_start = block_id
            run_length = 1
    return tuple(ranges)


def owner_gpu_route(
    gathered_ids: torch.Tensor,
    *,
    world_size: int,
    padded_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a duplicate-free owner route without copying IDs to the CPU."""
    payloads = gathered_ids.reshape(world_size, padded_blocks + 1)
    counts = payloads[:, 0]
    all_ready = (counts >= 0).all()
    offsets = torch.arange(padded_blocks, device=gathered_ids.device)
    valid = offsets.unsqueeze(0) < counts.clamp(min=0, max=padded_blocks).unsqueeze(1)
    block_ids = payloads[:, 1:].reshape(-1)
    # A failed rank contributes no valid IDs. Successful ranks are still
    # published; authoritative correction later fills the failed rank's IDs.
    valid = valid.reshape(-1) & (block_ids >= 0)
    sentinel = torch.iinfo(torch.int64).max
    sortable = torch.where(valid, block_ids, sentinel)
    sorted_ids, source_positions = torch.sort(sortable, stable=True)
    unique = sorted_ids != sentinel
    if sorted_ids.numel() > 1:
        unique[1:] &= sorted_ids[1:] != sorted_ids[:-1]
    return sorted_ids[unique], source_positions[unique], all_ready


def owner_gpu_padded_blocks(
    covered_end: int,
    *,
    world_size: int,
    configured_cap: int,
    append_reserve_blocks: int = 0,
) -> int:
    """Return the safe per-rank owner width for a globally covered prefix."""
    if world_size <= 0 or configured_cap <= 0 or append_reserve_blocks < 0:
        raise ValueError("owner padding geometry must be positive")
    history_shard = math.ceil(max(0, covered_end) / world_size)
    return max(1, min(configured_cap, history_shard + append_reserve_blocks))


def dense_rank_major_metadata(
    partition: BlockPartition,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return dense IDs, rank-major receive positions, and owner ranks."""
    ids: list[int] = []
    positions: list[int] = []
    owners: list[int] = []
    for rank, blocks in enumerate(partition.blocks_by_rank):
        ids.extend(blocks)
        positions.extend(
            rank * partition.padded_blocks + offset for offset in range(len(blocks))
        )
        owners.extend([rank] * len(blocks))
    return tuple(ids), tuple(positions), tuple(owners)


def owner_gather_receive_positions(
    sent_blocks_by_rank: Sequence[Sequence[int]],
    owned_blocks_by_rank: Sequence[Sequence[int]],
    effective_padded: int,
) -> tuple[int, ...]:
    """Map deduplicated owner blocks to their rank-major receive rows."""
    if effective_padded <= 0:
        raise ValueError("effective owner gather width must be positive")
    if len(sent_blocks_by_rank) != len(owned_blocks_by_rank):
        raise ValueError("sent and owned rank counts must match")
    positions: list[int] = []
    for rank, (sent_blocks, owned_blocks) in enumerate(
        zip(sent_blocks_by_rank, owned_blocks_by_rank, strict=True)
    ):
        if len(sent_blocks) > effective_padded:
            raise ValueError("sent owner blocks exceed effective gather width")
        sent_offsets = {
            int(block_id): offset for offset, block_id in enumerate(sent_blocks)
        }
        for block_id in owned_blocks:
            try:
                offset = sent_offsets[int(block_id)]
            except KeyError as exc:
                raise ValueError(
                    "owned block is absent from its rank send list"
                ) from exc
            positions.append(rank * effective_padded + offset)
    return tuple(positions)


@dataclass(frozen=True, slots=True)
class CPReadPlan:
    """Compiled CP ownership expressed as coalesced rows and SSD blocks.

    Args:
        rank: Rank in the context-parallel group.
        world_size: Number of context-parallel ranks.
        total_rows: Number of logical Indexer-K rows.
        row_ranges: Coalesced half-open row ranges owned by ``rank``.
        block_ids: Layer-major SSD blocks that contain exactly those rows.
        row_bytes: Bytes in one logical Indexer-K row.
        block_rows: Logical rows in one layer-major SSD block.
    """

    rank: int
    world_size: int
    total_rows: int
    row_ranges: tuple[RowRange, ...]
    block_ids: tuple[int, ...]
    row_bytes: int
    block_rows: int

    @property
    def planned_rows(self) -> int:
        """Return the number of rows owned by this rank."""
        return sum(row_range.length for row_range in self.row_ranges)

    @property
    def ssd_bytes(self) -> int:
        """Return the exact bytes represented by the selected SSD blocks."""
        return len(self.block_ids) * self.block_rows * self.row_bytes


@dataclass(frozen=True, slots=True)
class BlockPartition:
    """Deterministic, balanced partition of one globally ordered block union.

    Args:
        union: Sorted, duplicate-free logical block ids.
        blocks_by_rank: Contiguous union slices assigned to each rank.
        padded_blocks: Fixed send count used by the data collective.
        union_hash: Stable process-independent digest of ``union``.
    """

    union: tuple[int, ...]
    blocks_by_rank: tuple[tuple[int, ...], ...]
    padded_blocks: int
    union_hash: int

    @property
    def world_size(self) -> int:
        """Return the number of owner ranks."""
        return len(self.blocks_by_rank)

    @property
    def counts(self) -> tuple[int, ...]:
        """Return the valid block count for each owner rank."""
        return tuple(len(blocks) for blocks in self.blocks_by_rank)

    def blocks_for_rank(self, rank: int) -> tuple[int, ...]:
        """Return the ordered blocks owned by ``rank``.

        Args:
            rank: Rank in ``[0, world_size)``.

        Returns:
            The rank's contiguous slice of the global union.

        Raises:
            ValueError: If ``rank`` is outside the partition.
        """
        if rank < 0 or rank >= self.world_size:
            raise ValueError("rank is outside the block partition")
        return self.blocks_by_rank[rank]


@dataclass(frozen=True, slots=True)
class CollectiveDescriptor:
    """Metadata that all ranks must agree on before data all-gather.

    Args:
        request_generation: Monotonic request generation local to the engine.
        layer_id: Transformer layer that consumes the gathered bytes.
        phase: Stable phase number within the request/layer pair.
        mode: Dense or predicted shard-gather mode.
        partition: Deterministic partition used for this collective.
    """

    request_generation: int
    layer_id: int
    phase: int
    mode: SSDReadMode
    partition: BlockPartition

    @property
    def sequence_number(self) -> int:
        """Return a deterministic collective sequence number."""
        if self.request_generation < 0 or self.layer_id < 0 or self.phase < 0:
            raise ValueError("collective sequence fields must be non-negative")
        return (
            (int(self.request_generation) << 32)
            | (int(self.layer_id) << 8)
            | int(self.phase)
        )


@dataclass(frozen=True, slots=True)
class SSDTPShardedPrefetchConfig:
    """Centralized configuration for TP-sharded SSD prefetch.

    Environment variables are parsed only by :meth:`from_env`; managers
    receive this immutable object and do not maintain independent kill
    switches.

    Args:
        enabled: Global kill switch.
        indexer_enabled: Enable CP-local Indexer-K reads.
        csa_enabled: Enable CSA SSD shard-gather.
        dense_layers: Layers eligible for dense shard-gather.
        disabled_layers: Per-layer kill switch.
        debug_verify: Enable additional union/checksum verification.
        csa_replica_verified: Operator attestation that CSA logical blocks are
            byte-identical across candidate source ranks.
        indexer_cp_verified: Operator attestation that the active Indexer
            kernel consumes only rows owned by this CP rank.
        cp_size: Required context-parallel group size.
        cp_interleave: Consecutive Indexer-K rows assigned to a rank.
        min_union_blocks: Lower bound for shard-gather consideration.
        staging_slot_bytes: Maximum gathered bytes in one staging slot.
        staging_slots: Number of reusable staging slots.
        early_lookahead: Layers by which dense work is issued early.
        margin_ratio: Safety margin applied to shard p90 estimates.
        hysteresis_ms: Minimum benefit retained when changing a cached mode.
        ssd_fixed_ms: SSD submit/poll fixed cost.
        ssd_block_us: SSD service time per block at calibrated queue depth.
        gather_fixed_ms: Collective launch and synchronization fixed cost.
        nvlink_gbps: Effective one-direction NVLink bandwidth.
        interference_ms: Calibrated resource-interference penalty.
    """

    enabled: bool = False
    indexer_enabled: bool = True
    csa_enabled: bool = True
    dense_layers: frozenset[int] = frozenset(range(2, 25))
    disabled_layers: frozenset[int] = frozenset()
    debug_verify: bool = False
    csa_replica_verified: bool = False
    indexer_cp_verified: bool = False
    cp_size: int = 8
    cp_interleave: int = 64
    min_union_blocks: int = 128
    staging_slot_bytes: int = 128 * 1024**2
    staging_slots: int = 2
    early_lookahead: int = 2
    margin_ratio: float = 0.10
    hysteresis_ms: float = 0.25
    ssd_fixed_ms: float = 0.5
    ssd_block_us: float = 3.2
    gather_fixed_ms: float = 0.3
    nvlink_gbps: float = 300.0
    interference_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration boundaries."""
        if self.cp_size <= 0:
            raise ValueError("cp_size must be positive")
        if self.cp_interleave <= 0:
            raise ValueError("cp_interleave must be positive")
        if self.min_union_blocks < 0:
            raise ValueError("min_union_blocks must be non-negative")
        if self.staging_slot_bytes <= 0:
            raise ValueError("staging_slot_bytes must be positive")
        if self.staging_slots < 2:
            raise ValueError("staging_slots must be at least two")
        if self.early_lookahead not in (1, 2):
            raise ValueError("early_lookahead must be one or two")
        if self.margin_ratio < 0.0:
            raise ValueError("margin_ratio must be non-negative")
        if self.hysteresis_ms < 0.0:
            raise ValueError("hysteresis_ms must be non-negative")
        if self.ssd_fixed_ms < 0.0 or self.ssd_block_us <= 0.0:
            raise ValueError("SSD timing parameters must be positive")
        if self.gather_fixed_ms < 0.0 or self.nvlink_gbps <= 0.0:
            raise ValueError("gather timing parameters must be positive")
        if self.interference_ms < 0.0:
            raise ValueError("interference_ms must be non-negative")

    @classmethod
    def from_env(cls) -> "SSDTPShardedPrefetchConfig":
        """Build the centralized config from LMCache environment variables.

        Returns:
            A validated immutable configuration object.

        Raises:
            ValueError: If a numeric or layer-range value is invalid.
        """

        def _flag(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            normalized = raw.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{name} must be a boolean")

        def _int(name: str, default: int) -> int:
            return int(os.getenv(name, str(default)))

        def _float(name: str, default: float) -> float:
            return float(os.getenv(name, str(default)))

        legacy_cp_size = int(os.getenv("LMCACHE_CSA_PREFETCH_CP_SIZE", "8"))
        legacy_cp_interleave = int(
            os.getenv("LMCACHE_CSA_PREFETCH_CP_INTERLEAVE", "64")
        )
        return cls(
            enabled=_flag("LMCACHE_SSD_TP_SHARDED_PREFETCH", False),
            indexer_enabled=_flag("LMCACHE_SSD_TP_SHARD_INDEXER", True),
            csa_enabled=_flag("LMCACHE_SSD_TP_SHARD_CSA", True),
            dense_layers=parse_layer_ranges(
                os.getenv("LMCACHE_SSD_TP_DENSE_LAYERS", "")
            ),
            disabled_layers=parse_layer_ranges(
                os.getenv("LMCACHE_SSD_TP_DISABLED_LAYERS", "")
            ),
            debug_verify=_flag("LMCACHE_SSD_TP_DEBUG_VERIFY", False),
            csa_replica_verified=_flag(
                "LMCACHE_SSD_TP_CSA_REPLICA_VERIFIED",
                False,
            ),
            indexer_cp_verified=_flag(
                "LMCACHE_SSD_TP_INDEXER_CP_VERIFIED",
                False,
            ),
            cp_size=_int("LMCACHE_SSD_TP_CP_SIZE", legacy_cp_size),
            cp_interleave=_int(
                "LMCACHE_SSD_TP_CP_INTERLEAVE",
                legacy_cp_interleave,
            ),
            min_union_blocks=_int("LMCACHE_SSD_TP_MIN_UNION_BLOCKS", 128),
            staging_slot_bytes=_int(
                "LMCACHE_SSD_TP_STAGING_SLOT_BYTES",
                128 * 1024**2,
            ),
            staging_slots=_int("LMCACHE_SSD_TP_STAGING_SLOTS", 2),
            early_lookahead=_int("LMCACHE_SSD_TP_EARLY_LOOKAHEAD", 2),
            margin_ratio=_float("LMCACHE_SSD_TP_MARGIN_RATIO", 0.10),
            hysteresis_ms=_float("LMCACHE_SSD_TP_HYSTERESIS_MS", 0.25),
            ssd_fixed_ms=_float("LMCACHE_SSD_TP_SSD_FIXED_MS", 0.5),
            ssd_block_us=_float("LMCACHE_SSD_TP_SSD_BLOCK_US", 3.2),
            gather_fixed_ms=_float("LMCACHE_SSD_TP_GATHER_FIXED_MS", 0.3),
            nvlink_gbps=_float("LMCACHE_SSD_TP_NVLINK_GBPS", 300.0),
            interference_ms=_float(
                "LMCACHE_SSD_TP_INTERFERENCE_MS",
                0.0,
            ),
        )

    @classmethod
    def from_engine_config(cls, config: Any) -> "SSDTPShardedPrefetchConfig":
        """Build from public fields on an ``LMCacheEngineConfig`` object.

        Args:
            config: Engine configuration exposing the ``ssd_tp_*`` fields.

        Returns:
            A validated immutable sharded-prefetch configuration.
        """
        return cls(
            enabled=bool(config.ssd_tp_sharded_prefetch),
            indexer_enabled=bool(config.ssd_tp_shard_indexer),
            csa_enabled=bool(config.ssd_tp_shard_csa),
            dense_layers=parse_layer_ranges(config.ssd_tp_dense_layers),
            disabled_layers=parse_layer_ranges(config.ssd_tp_disabled_layers),
            debug_verify=bool(config.ssd_tp_debug_verify),
            csa_replica_verified=bool(config.ssd_tp_csa_replica_verified),
            indexer_cp_verified=bool(config.ssd_tp_indexer_cp_verified),
            cp_size=int(config.ssd_tp_cp_size),
            cp_interleave=int(config.ssd_tp_cp_interleave),
            min_union_blocks=int(config.ssd_tp_min_union_blocks),
            staging_slot_bytes=int(config.ssd_tp_staging_slot_bytes),
            staging_slots=int(config.ssd_tp_staging_slots),
            early_lookahead=int(config.ssd_tp_early_lookahead),
            margin_ratio=float(config.ssd_tp_margin_ratio),
            hysteresis_ms=float(config.ssd_tp_hysteresis_ms),
            ssd_fixed_ms=float(config.ssd_tp_ssd_fixed_ms),
            ssd_block_us=float(config.ssd_tp_ssd_block_us),
            gather_fixed_ms=float(config.ssd_tp_gather_fixed_ms),
            nvlink_gbps=float(config.ssd_tp_nvlink_gbps),
            interference_ms=float(config.ssd_tp_interference_ms),
        )


@dataclass(frozen=True, slots=True)
class PrefetchDecisionKey:
    """Bucketed key used by the request-hot-path decision table.

    Args:
        group: Logical cache group, such as ``csa``.
        layer_id: Transformer layer id.
        context_bucket: Rounded context length.
        query_bucket: Rounded incremental query length.
        union_bucket: Rounded union block count.
    """

    group: str
    layer_id: int
    context_bucket: int
    query_bucket: int
    union_bucket: int


@dataclass(frozen=True, slots=True)
class PrefetchCostEstimate:
    """P90 cost comparison for local and sharded reads.

    Args:
        local_ms: Estimated p90 local-direct service time.
        shard_ms: Estimated p90 shard-gather service time including margin.
        benefit_ms: ``local_ms - shard_ms``.
    """

    local_ms: float
    shard_ms: float
    benefit_ms: float


class ShardPrefetchDecisionTable:
    """Bounded lookup table with p90 margin and mode hysteresis."""

    def __init__(self, config: SSDTPShardedPrefetchConfig) -> None:
        """Initialize an empty decision table.

        Args:
            config: Centralized cost and threshold configuration.
        """
        self._config = config
        self._modes: dict[PrefetchDecisionKey, SSDReadMode] = {}
        self._samples: dict[tuple[PrefetchDecisionKey, SSDReadMode], deque[float]] = {}
        self._lock = threading.Lock()

    def estimate(
        self,
        union_blocks: int,
        block_bytes: int,
        world_size: int,
    ) -> PrefetchCostEstimate:
        """Estimate local and sharded p90 service times.

        Args:
            union_blocks: Number of unique blocks required by the consumer.
            block_bytes: Bytes in one logical block.
            world_size: Number of shard owners.

        Returns:
            The calibrated p90 comparison.

        Raises:
            ValueError: If a geometry argument is invalid.
        """
        if union_blocks < 0 or block_bytes <= 0 or world_size <= 0:
            raise ValueError("invalid cost-model geometry")
        local_ms = self._config.ssd_fixed_ms + (
            union_blocks * self._config.ssd_block_us / 1000.0
        )
        owned = math.ceil(union_blocks / world_size)
        ssd_ms = self._config.ssd_fixed_ms + (
            owned * self._config.ssd_block_us / 1000.0
        )
        receive_bytes = (world_size - 1) / world_size * union_blocks * block_bytes
        transfer_ms = receive_bytes / (self._config.nvlink_gbps * 1_000_000.0)
        raw_shard_ms = (
            ssd_ms
            + self._config.gather_fixed_ms
            + transfer_ms
            + self._config.interference_ms
        )
        shard_ms = raw_shard_ms * (1.0 + self._config.margin_ratio)
        return PrefetchCostEstimate(
            local_ms=local_ms,
            shard_ms=shard_ms,
            benefit_ms=local_ms - shard_ms,
        )

    def choose(
        self,
        key: PrefetchDecisionKey,
        *,
        union_blocks: int,
        block_bytes: int,
        world_size: int,
        shard_mode: SSDReadMode,
        capability_ok: bool,
    ) -> SSDReadMode:
        """Choose a mode using only a lookup, estimate, and boundary checks.

        Args:
            key: Pre-bucketed decision key.
            union_blocks: Number of required unique blocks.
            block_bytes: Bytes in one block.
            world_size: Number of participating ranks.
            shard_mode: Dense or predicted shard mode under consideration.
            capability_ok: Whether layout, communicator, and staging gates pass.

        Returns:
            ``LOCAL_DIRECT`` or ``shard_mode``.

        Raises:
            ValueError: If ``shard_mode`` is not a shard-gather mode.
        """
        if shard_mode not in {
            SSDReadMode.SHARD_GATHER_DENSE,
            SSDReadMode.SHARD_GATHER_PREDICTED,
        }:
            raise ValueError("shard_mode must be dense or predicted")
        if (
            not self._config.enabled
            or not self._config.csa_enabled
            or not capability_ok
            or key.layer_id in self._config.disabled_layers
            or union_blocks < self._config.min_union_blocks
        ):
            return SSDReadMode.LOCAL_DIRECT
        estimate = self.estimate(union_blocks, block_bytes, world_size)
        with self._lock:
            previous = self._modes.get(key, SSDReadMode.LOCAL_DIRECT)
            local_p90 = self._observed_p90_locked(key, SSDReadMode.LOCAL_DIRECT)
            shard_p90 = self._observed_p90_locked(key, shard_mode)
            benefit_ms = estimate.benefit_ms
            if local_p90 is not None and shard_p90 is not None:
                benefit_ms = local_p90 - shard_p90 * (1.0 + self._config.margin_ratio)
            threshold = (
                -self._config.hysteresis_ms
                if previous == shard_mode
                else self._config.hysteresis_ms
            )
            selected = (
                shard_mode if benefit_ms > threshold else SSDReadMode.LOCAL_DIRECT
            )
            self._modes[key] = selected
        return selected

    def record_sample(
        self,
        key: PrefetchDecisionKey,
        mode: SSDReadMode,
        elapsed_ms: float,
        *,
        cold_start: bool = False,
        max_rank_skew_ms: float = 0.0,
    ) -> None:
        """Record one bounded steady-state timing sample.

        Args:
            key: Bucketed decision key used for the request.
            mode: Mode whose end-to-end service time was observed.
            elapsed_ms: Measured prepare-through-ready time in milliseconds.
            cold_start: Exclude this sample from steady-state calibration.
            max_rank_skew_ms: Non-negative max-rank safety penalty.

        Raises:
            ValueError: If a timing value is negative or non-finite.
        """
        if (
            not math.isfinite(elapsed_ms)
            or not math.isfinite(max_rank_skew_ms)
            or elapsed_ms < 0.0
            or max_rank_skew_ms < 0.0
        ):
            raise ValueError("timing samples must be finite and non-negative")
        if cold_start:
            return
        # Cap pathological samples so one request cannot permanently poison
        # the table. A 60-second layer operation is already a request failure.
        bounded_ms = min(60_000.0, elapsed_ms + max_rank_skew_ms)
        with self._lock:
            samples = self._samples.setdefault((key, mode), deque(maxlen=64))
            samples.append(bounded_ms)

    def observed_p90(
        self,
        key: PrefetchDecisionKey,
        mode: SSDReadMode,
    ) -> Optional[float]:
        """Return the current steady-state p90 for a key and mode.

        Args:
            key: Bucketed decision key.
            mode: Observed read mode.

        Returns:
            Nearest-rank p90 in milliseconds, or ``None`` before sampling.
        """
        with self._lock:
            return self._observed_p90_locked(key, mode)

    def _observed_p90_locked(
        self,
        key: PrefetchDecisionKey,
        mode: SSDReadMode,
    ) -> Optional[float]:
        samples = self._samples.get((key, mode))
        if not samples:
            return None
        ordered = sorted(samples)
        index = max(0, math.ceil(0.90 * len(ordered)) - 1)
        return ordered[index]


class ShardCollectiveError(RuntimeError):
    """Collective failure carrying whether the data all-gather was submitted."""

    def __init__(self, message: str, *, data_submitted: bool) -> None:
        """Initialize the collective failure.

        Args:
            message: Human-readable failure detail.
            data_submitted: Whether rank-local fallback is no longer safe.
        """
        super().__init__(message)
        self.data_submitted = bool(data_submitted)


class ShardGatherTransport(Protocol):
    """Public transport contract consumed by the CSA prefetch manager."""

    @property
    def rank(self) -> int:
        """Return this process's rank in the prefetch group."""

    @property
    def world_size(self) -> int:
        """Return the prefetch group size."""

    @property
    def healthy(self) -> bool:
        """Return whether future shard-gather submissions are allowed."""

    def warm(
        self,
        *,
        max_union_blocks: int,
        block_bytes: int,
        device: torch.device,
    ) -> None:
        """Allocate and prewarm fixed staging resources."""

    def preflight(
        self,
        descriptor: CollectiveDescriptor,
        *,
        local_capability: bool,
        device: torch.device,
    ) -> bool:
        """Reach metadata consensus before any rank submits SSD I/O."""

    def exchange_block_union(
        self,
        local_block_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Exchange rank-local predictions and return one global block union."""

    def exchange_block_partition(
        self,
        local_block_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> BlockPartition:
        """Exchange rank-local predictions while preserving their owners."""

    def gather_owner_rows_into(
        self,
        *,
        local_block_ids: torch.Tensor,
        local_ready: bool,
        padded_blocks: int,
        source_rows: torch.Tensor,
        logical_destination_rows: torch.Tensor,
        destination_rows: torch.Tensor,
        local_ready_event: Optional[torch.cuda.Event],
        resident_bitmap: torch.Tensor,
    ) -> tuple[torch.cuda.Event, BlockPartition, bool]:
        """AllGather owner KV rows with an in-stream block-ID sidecar."""

    def gather_owner_rows_gpu_into(
        self,
        *,
        local_block_ids: torch.Tensor,
        local_ready: bool,
        padded_blocks: int,
        source_rows: torch.Tensor,
        logical_destination_rows: torch.Tensor,
        destination_rows: torch.Tensor,
        local_ready_event: Optional[torch.cuda.Event],
        resident_bitmap: torch.Tensor,
    ) -> tuple[torch.cuda.Event, torch.Tensor, torch.Tensor]:
        """AllGather and route owner rows using GPU-resident metadata."""

    def gather_dense_rows_gpu_into(
        self,
        descriptor: CollectiveDescriptor,
        *,
        local_ready: bool,
        source_rows: torch.Tensor,
        logical_destination_rows: torch.Tensor,
        destination_rows: torch.Tensor,
        local_ready_event: Optional[torch.cuda.Event],
        resident_bitmap: torch.Tensor,
    ) -> tuple[torch.cuda.Event, torch.Tensor]:
        """Gather a fixed dense partition with GPU-resident readiness."""

    def gather_into(
        self,
        descriptor: CollectiveDescriptor,
        *,
        source_rows: torch.Tensor,
        logical_destination_rows: torch.Tensor,
        destination_rows: torch.Tensor,
        local_ready_event: Optional[torch.cuda.Event],
        resident_bitmap: torch.Tensor,
    ) -> torch.cuda.Event:
        """Gather owner rows and materialize the complete union."""

    def close(self) -> None:
        """Release transport-owned resources."""


@dataclass(slots=True)
class _GatherSlot:
    send: torch.Tensor
    receive: torch.Tensor
    send_ids: torch.Tensor
    receive_ids: torch.Tensor
    completion_event: Optional[torch.cuda.Event] = None
    completion_work: Any = None


class TorchDistributedShardGather:
    """NCCL shard-gather transport with fixed double-buffered staging."""

    def __init__(
        self,
        process_group: Any,
        *,
        metadata_process_group: Any | None = None,
        rank: int,
        world_size: int,
        config: SSDTPShardedPrefetchConfig,
        owns_process_group: bool = False,
        owns_metadata_process_group: bool = False,
    ) -> None:
        """Initialize a transport around an independent process group.

        Args:
            process_group: Dedicated torch.distributed process group.
            rank: Rank within ``process_group``.
            world_size: Size of ``process_group``.
            config: Centralized staging and safety configuration.
            owns_process_group: Destroy the group from :meth:`close`.
        """
        if process_group is None:
            raise ValueError("process_group is required")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be within the process group")
        self._process_group = process_group
        self._metadata_process_group = metadata_process_group or process_group
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._config = config
        self._owns_process_group = bool(owns_process_group)
        self._owns_metadata_process_group = bool(
            owns_metadata_process_group
            and self._metadata_process_group is not self._process_group
        )
        self._healthy = True
        self._stream: Optional[torch.cuda.Stream] = None
        self._metadata_stream: Optional[torch.cuda.Stream] = None
        self._slots: list[_GatherSlot] = []
        self._slot_cursor = 0
        self._max_union_blocks = 0
        self._block_bytes = 0
        self._lock = threading.Lock()
        self._metadata_lock = threading.Lock()

    @classmethod
    def from_model_group(
        cls,
        model_group: Any,
        config: SSDTPShardedPrefetchConfig,
    ) -> "TorchDistributedShardGather":
        """Create an independent NCCL group with the model group's ranks.

        Args:
            model_group: Existing model TP process group, used only to discover
                global ranks.
            config: Centralized sharded-prefetch configuration.

        Returns:
            A transport owning a newly-created NCCL communicator.

        Raises:
            RuntimeError: If torch.distributed is unavailable or uninitialized.
        """
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")
        ranks = dist.get_process_group_ranks(model_group)
        process_group = dist.new_group(ranks=ranks, backend="nccl")
        metadata_process_group = dist.new_group(ranks=ranks, backend="nccl")
        return cls(
            process_group,
            metadata_process_group=metadata_process_group,
            rank=dist.get_rank(group=process_group),
            world_size=dist.get_world_size(group=process_group),
            config=config,
            owns_process_group=True,
            owns_metadata_process_group=True,
        )

    @property
    def rank(self) -> int:
        """Return this process's rank in the prefetch group."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Return the prefetch group size."""
        return self._world_size

    @property
    def healthy(self) -> bool:
        """Return whether the communicator accepts new work."""
        return self._healthy

    def warm(
        self,
        *,
        max_union_blocks: int,
        block_bytes: int,
        device: torch.device,
    ) -> None:
        """Allocate fixed staging slots and prewarm the private CUDA stream.

        Args:
            max_union_blocks: Largest complete union supported by the manager.
            block_bytes: Bytes in one logical block.
            device: CUDA device used by the model rank.

        Raises:
            ValueError: If geometry is invalid or exceeds staging capacity.
        """
        if max_union_blocks <= 0 or block_bytes <= 0:
            raise ValueError("staging geometry must be positive")
        if device.type != "cuda":
            raise ValueError("NCCL shard-gather requires a CUDA device")
        padded = math.ceil(max_union_blocks / self._world_size)
        gathered_bytes = padded * self._world_size * block_bytes
        if gathered_bytes > self._config.staging_slot_bytes:
            raise ValueError(
                f"gathered layer needs {gathered_bytes} bytes, exceeding "
                f"staging slot {self._config.staging_slot_bytes}"
            )
        with self._lock, torch.cuda.device(device), torch.inference_mode():
            if (
                self._slots
                and self._max_union_blocks >= max_union_blocks
                and self._block_bytes == block_bytes
            ):
                return
            self._stream = torch.cuda.Stream(device=device)
            self._metadata_stream = torch.cuda.Stream(device=device)
            self._slots = [
                _GatherSlot(
                    send=torch.empty(
                        (padded, block_bytes),
                        dtype=torch.uint8,
                        device=device,
                    ),
                    receive=torch.empty(
                        (padded * self._world_size, block_bytes),
                        dtype=torch.uint8,
                        device=device,
                    ),
                    send_ids=torch.empty(
                        padded + 1,
                        dtype=torch.int64,
                        device=device,
                    ),
                    receive_ids=torch.empty(
                        self._world_size * (padded + 1),
                        dtype=torch.int64,
                        device=device,
                    ),
                )
                for _ in range(self._config.staging_slots)
            ]
            self._slot_cursor = 0
            self._max_union_blocks = int(max_union_blocks)
            self._block_bytes = int(block_bytes)
            # A tiny all-reduce forces communicator and stream initialization
            # out of the first request. All ranks call warm at engine attach.
            warm_value = torch.ones(1, dtype=torch.int32, device=device)
            metadata_warm_value = torch.ones(
                1,
                dtype=torch.int32,
                device=device,
            )
            with torch.cuda.stream(self._stream):
                torch.distributed.all_reduce(
                    warm_value,
                    group=self._process_group,
                )
            with torch.cuda.stream(self._metadata_stream):
                torch.distributed.all_reduce(
                    metadata_warm_value,
                    group=self._metadata_process_group,
                )
            self._stream.synchronize()
            self._metadata_stream.synchronize()

    def exchange_block_union(
        self,
        local_block_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Exchange local predicted block IDs on the metadata communicator.

        Every rank participates, including ranks with an empty local set. The
        returned CPU tensor is identical on every rank and is capped to the
        configured global staging budget.

        Args:
            local_block_ids: Rank-local predicted logical block IDs.
            device: CUDA device used by this model rank.

        Returns:
            Sorted, duplicate-free global block IDs on CPU.

        Raises:
            ShardCollectiveError: If the metadata exchange fails.
        """
        try:
            rank_blocks = self._exchange_rank_block_ids(local_block_ids, device=device)
            union = deterministic_block_union(
                block_id for blocks in rank_blocks for block_id in blocks
            )
            return torch.as_tensor(
                union[: self._max_union_blocks],
                dtype=torch.int64,
            )
        except Exception as exc:
            self._healthy = False
            raise ShardCollectiveError(
                "predicted block-union exchange failed",
                data_submitted=False,
            ) from exc

    def exchange_block_partition(
        self,
        local_block_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> BlockPartition:
        """Exchange candidate IDs and retain the rank that selected each ID.

        Duplicate IDs are assigned to the lowest selecting rank. The returned
        descriptor is identical on every rank and can therefore drive the
        existing KV-row AllGather without an exact distributed Top-K merge.
        """
        try:
            rank_blocks = self._exchange_rank_block_ids(local_block_ids, device=device)
            return partition_rank_local_blocks(
                rank_blocks,
                max_union_blocks=self._max_union_blocks,
            )
        except Exception as exc:
            self._healthy = False
            raise ShardCollectiveError(
                "predicted owner-partition exchange failed",
                data_submitted=False,
            ) from exc

    def _exchange_rank_block_ids(
        self,
        local_block_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[tuple[int, ...], ...]:
        """Return each rank's canonical candidate IDs on the CPU."""
        if not self._healthy:
            raise ShardCollectiveError(
                "shard-gather metadata communicator is unhealthy",
                data_submitted=False,
            )
        max_blocks = int(self._max_union_blocks)
        if max_blocks <= 0:
            raise ShardCollectiveError(
                "shard-gather metadata resources are not warmed",
                data_submitted=False,
            )
        metadata_stream = self._metadata_stream or torch.cuda.current_stream(device)
        with (
            self._metadata_lock,
            torch.cuda.device(device),
            torch.inference_mode(),
            torch.cuda.stream(metadata_stream),
        ):
            local = torch.as_tensor(
                local_block_ids,
                dtype=torch.int64,
                device=device,
            ).reshape(-1)
            local = torch.unique(local[local >= 0], sorted=True)[:max_blocks]
            payload = torch.full(
                (max_blocks + 1,),
                -1,
                dtype=torch.int64,
                device=device,
            )
            payload[0] = int(local.numel())
            if local.numel():
                payload[1 : 1 + int(local.numel())] = local
            gathered = torch.empty(
                self._world_size * (max_blocks + 1),
                dtype=torch.int64,
                device=device,
            )
            torch.distributed.all_gather_into_tensor(
                gathered,
                payload,
                group=self._metadata_process_group,
            )
        metadata_stream.synchronize()
        gathered_cpu = gathered.cpu().reshape(self._world_size, max_blocks + 1)
        blocks_by_rank: list[tuple[int, ...]] = []
        for rank_payload in gathered_cpu:
            length = max(0, min(int(rank_payload[0].item()), max_blocks))
            blocks_by_rank.append(
                deterministic_block_union(rank_payload[1 : 1 + length].tolist())
            )
        return tuple(blocks_by_rank)

    def gather_owner_rows_into(
        self,
        *,
        local_block_ids: torch.Tensor,
        local_ready: bool,
        padded_blocks: int,
        source_rows: torch.Tensor,
        logical_destination_rows: torch.Tensor,
        destination_rows: torch.Tensor,
        local_ready_event: Optional[torch.cuda.Event],
        resident_bitmap: torch.Tensor,
    ) -> tuple[torch.cuda.Event, BlockPartition, bool]:
        """AllGather owner rows and routing IDs without a pre-read union.

        Every rank always enters both collectives. A negative sidecar count
        advertises a failed local read; callers then use authoritative local
        correction instead of publishing gathered rows.
        """
        if (
            not self._healthy
            or not self._slots
            or self._stream is None
            or padded_blocks <= 0
        ):
            raise ShardCollectiveError(
                "owner gather resources are unavailable",
                data_submitted=False,
            )
        data_submitted = False
        try:
            metadata_stream = self._metadata_stream
            if metadata_stream is None:
                raise RuntimeError("owner gather metadata stream is unavailable")
            with self._lock, torch.inference_mode():
                slot = self._slots[self._slot_cursor]
                self._slot_cursor = (self._slot_cursor + 1) % len(self._slots)
                if padded_blocks > int(slot.send.shape[0]):
                    raise ValueError("owner gather exceeds warmed row capacity")
                local_cpu = torch.as_tensor(
                    local_block_ids,
                    dtype=torch.int64,
                    device="cpu",
                ).reshape(-1)
                local_cpu = torch.unique(local_cpu[local_cpu >= 0], sorted=True)
                rank_ready = bool(
                    local_ready and int(local_cpu.numel()) <= padded_blocks
                )
                local_gpu = local_cpu.to(source_rows.device)
                completion_event = slot.completion_event
                id_width = padded_blocks + 1
                with torch.cuda.stream(metadata_stream):
                    if completion_event is not None:
                        metadata_stream.wait_event(completion_event)
                    slot.send_ids[:id_width].fill_(-1)
                    slot.send_ids[0] = int(local_cpu.numel()) if rank_ready else -1
                    if rank_ready and local_gpu.numel():
                        slot.send_ids[1 : 1 + int(local_gpu.numel())] = local_gpu
                    id_work = torch.distributed.all_gather_into_tensor(
                        slot.receive_ids[: self._world_size * id_width],
                        slot.send_ids[:id_width],
                        group=self._metadata_process_group,
                        async_op=True,
                    )
                with torch.cuda.stream(self._stream):
                    if completion_event is not None:
                        self._stream.wait_event(completion_event)
                    if local_ready_event is not None:
                        self._stream.wait_event(local_ready_event)
                    slot.send[:padded_blocks].zero_()
                    if rank_ready and local_gpu.numel():
                        local_rows = logical_destination_rows.index_select(
                            0,
                            local_gpu,
                        )
                        torch.index_select(
                            source_rows,
                            0,
                            local_rows,
                            out=slot.send[: int(local_gpu.numel())],
                        )
                id_work.wait()
            metadata_stream.synchronize()

            payloads = (
                slot.receive_ids[: self._world_size * (padded_blocks + 1)]
                .cpu()
                .reshape(self._world_size, padded_blocks + 1)
            )
            all_ready = all(int(payload[0].item()) >= 0 for payload in payloads)
            rank_blocks: list[tuple[int, ...]] = []
            for payload in payloads:
                count = max(0, min(int(payload[0].item()), padded_blocks))
                rank_blocks.append(
                    deterministic_block_union(payload[1 : 1 + count].tolist())
                )
            # The gathered ID payload is identical on every rank, so every
            # rank derives the same effective width. Shrinking the KV
            # collective to the actual maximum advertised count (instead of
            # the fixed warmed capacity) removes the dominant fixed cost of
            # the owner gather: with a 512-block capacity but e.g. 80
            # candidates per rank, the transfer drops by 6x. NCCL still sees
            # equal send sizes on all ranks.
            effective_padded = max(
                1,
                min(
                    padded_blocks,
                    max(
                        (len(blocks) for blocks in rank_blocks),
                        default=0,
                    ),
                ),
            )
            gathered_rows = effective_padded * self._world_size
            with self._lock, torch.inference_mode(), torch.cuda.stream(self._stream):
                # Every rank reaches this collective unconditionally, failed
                # local reads included (they advertised -1 and send stale
                # rows that no receiver scatters).
                data_submitted = True
                data_work = torch.distributed.all_gather_into_tensor(
                    slot.receive[:gathered_rows],
                    slot.send[:effective_padded],
                    group=self._process_group,
                    async_op=True,
                )
            partition = partition_rank_local_blocks(rank_blocks)
            with torch.inference_mode(), torch.cuda.stream(self._stream):
                # The data collective was enqueued first on this same stream;
                # scatter and the completion event are naturally ordered
                # after it. The metadata D2H/partition work above overlaps the
                # KV transfer instead of forcing a full data-stream sync.
                if all_ready and partition.union:
                    # Each rank sent its complete candidate list in sorted
                    # order, but ownership dedup may drop interior entries
                    # (a duplicate is owned by the lowest selecting rank).
                    # Row offsets must therefore come from each owned block's
                    # position within the rank's SENT list, not from the
                    # owned list's enumeration order. The receive layout
                    # stride is the shrunken effective width, not the warmed
                    # capacity.
                    positions = owner_gather_receive_positions(
                        rank_blocks,
                        partition.blocks_by_rank,
                        effective_padded,
                    )
                    position_tensor = torch.as_tensor(
                        positions,
                        dtype=torch.int64,
                        device=source_rows.device,
                    )
                    union_tensor = torch.as_tensor(
                        tuple(
                            block_id
                            for blocks in partition.blocks_by_rank
                            for block_id in blocks
                        ),
                        dtype=torch.int64,
                        device=source_rows.device,
                    )
                    gathered = slot.receive[:gathered_rows].index_select(
                        0,
                        position_tensor,
                    )
                    destination = logical_destination_rows.index_select(
                        0,
                        union_tensor,
                    )
                    destination_rows.index_copy_(0, destination, gathered)
                    resident_bitmap[union_tensor] = True
                event = torch.cuda.Event()
                event.record(self._stream)
                slot.completion_event = event
                slot.completion_work = data_work
            return event, partition, all_ready
        except Exception as exc:
            self._healthy = False
            raise ShardCollectiveError(
                "owner KV gather failed",
                data_submitted=data_submitted,
            ) from exc

    def gather_owner_rows_gpu_into(
        self,
        *,
        local_block_ids: torch.Tensor,
        local_ready: bool,
        padded_blocks: int,
        source_rows: torch.Tensor,
        logical_destination_rows: torch.Tensor,
        destination_rows: torch.Tensor,
        local_ready_event: Optional[torch.cuda.Event],
        resident_bitmap: torch.Tensor,
    ) -> tuple[torch.cuda.Event, torch.Tensor, torch.Tensor]:
        """Gather owner rows without a host metadata barrier or partition."""
        if (
            not self._healthy
            or not self._slots
            or self._stream is None
            or self._metadata_stream is None
            or padded_blocks <= 0
        ):
            raise ShardCollectiveError(
                "GPU owner gather resources are unavailable",
                data_submitted=False,
            )
        data_submitted = False
        try:
            with self._lock, torch.inference_mode():
                slot = self._slots[self._slot_cursor]
                self._slot_cursor = (self._slot_cursor + 1) % len(self._slots)
                if padded_blocks > int(slot.send.shape[0]):
                    raise ValueError("owner gather exceeds warmed row capacity")
                local_cpu = torch.as_tensor(
                    local_block_ids,
                    dtype=torch.int64,
                    device="cpu",
                ).reshape(-1)
                local_cpu = torch.unique(local_cpu[local_cpu >= 0], sorted=True)
                rank_ready = bool(local_ready and local_cpu.numel() <= padded_blocks)
                local_gpu = local_cpu.to(source_rows.device)
                completion_event = slot.completion_event
                id_width = padded_blocks + 1
                with torch.cuda.stream(self._metadata_stream):
                    if completion_event is not None:
                        self._metadata_stream.wait_event(completion_event)
                    slot.send_ids[:id_width].fill_(-1)
                    slot.send_ids[0] = int(local_gpu.numel()) if rank_ready else -1
                    if rank_ready and local_gpu.numel():
                        slot.send_ids[1 : 1 + local_gpu.numel()] = local_gpu
                    torch.distributed.all_gather_into_tensor(
                        slot.receive_ids[: self._world_size * id_width],
                        slot.send_ids[:id_width],
                        group=self._metadata_process_group,
                        async_op=True,
                    )
                    metadata_ready = torch.cuda.Event()
                    metadata_ready.record(self._metadata_stream)
                gathered_rows = padded_blocks * self._world_size
                with torch.cuda.stream(self._stream):
                    if completion_event is not None:
                        self._stream.wait_event(completion_event)
                    if local_ready_event is not None:
                        self._stream.wait_event(local_ready_event)
                    slot.send[:padded_blocks].zero_()
                    if rank_ready and local_gpu.numel():
                        local_rows = logical_destination_rows.index_select(0, local_gpu)
                        torch.index_select(
                            source_rows,
                            0,
                            local_rows,
                            out=slot.send[: local_gpu.numel()],
                        )
                    data_submitted = True
                    data_work = torch.distributed.all_gather_into_tensor(
                        slot.receive[:gathered_rows],
                        slot.send[:padded_blocks],
                        group=self._process_group,
                        async_op=True,
                    )
                    self._stream.wait_event(metadata_ready)
                    selected, source_positions, all_ready = owner_gpu_route(
                        slot.receive_ids[: self._world_size * id_width],
                        world_size=self._world_size,
                        padded_blocks=padded_blocks,
                    )
                    valid = (
                        (selected >= 0)
                        & (selected < int(logical_destination_rows.numel()))
                        & (source_positions >= 0)
                        & (source_positions < gathered_rows)
                    )
                    all_ready = all_ready & torch.all(valid)
                    selected = selected[valid]
                    source_positions = source_positions[valid]
                    gathered = slot.receive[:gathered_rows].index_select(
                        0, source_positions
                    )
                    destination = logical_destination_rows.index_select(0, selected)
                    destination_rows.index_copy_(0, destination, gathered)
                    resident_bitmap[selected] = True
                    event = torch.cuda.Event()
                    event.record(self._stream)
                    slot.completion_event = event
                    slot.completion_work = data_work
            return event, selected, all_ready
        except Exception as exc:
            self._healthy = False
            raise ShardCollectiveError(
                "GPU owner KV gather failed",
                data_submitted=data_submitted,
            ) from exc

    def preflight(
        self,
        descriptor: CollectiveDescriptor,
        *,
        local_capability: bool,
        device: torch.device,
    ) -> bool:
        """Reach all-rank plan and capability consensus before SSD I/O.

        Args:
            descriptor: Request/layer/phase and union metadata.
            local_capability: Rank-local layout, staging, and health result.
            device: CUDA device used for the metadata collective.

        Returns:
            ``True`` only when every rank supplied identical metadata and a
            positive capability flag.

        Raises:
            ShardCollectiveError: If the metadata collective itself fails.
        """
        if not self._healthy:
            return False
        partition = descriptor.partition
        mode_code = {
            SSDReadMode.SHARD_GATHER_DENSE: 1,
            SSDReadMode.SHARD_GATHER_PREDICTED: 2,
        }.get(descriptor.mode, 0)
        capacity_ok = (
            bool(self._slots)
            and len(partition.union) <= self._max_union_blocks
            and partition.padded_blocks * self._world_size * self._block_bytes
            <= self._config.staging_slot_bytes
        )
        try:
            # Never enqueue an auxiliary NCCL collective on the model's
            # default stream.  Background dense-prefetch threads can reach
            # this point with small inter-rank skew; mixing their metadata
            # all-gather with forward collectives on the shared default
            # stream creates a cross-communicator dependency cycle.  Keep
            # both metadata phases on the transport's ordered private stream,
            # just like the data gather below.
            collective_stream = self._stream or torch.cuda.current_stream(device)
            with (
                self._lock,
                torch.cuda.device(device),
                torch.inference_mode(),
                torch.cuda.stream(collective_stream),
            ):
                metadata = torch.tensor(
                    [
                        descriptor.sequence_number,
                        descriptor.layer_id,
                        mode_code,
                        len(partition.union),
                        partition.union_hash,
                        partition.padded_blocks,
                        int(local_capability and capacity_ok and self._healthy),
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                gathered = torch.empty(
                    (self._world_size, int(metadata.numel())),
                    dtype=metadata.dtype,
                    device=device,
                )
                torch.distributed.all_gather_into_tensor(
                    gathered,
                    metadata,
                    group=self._process_group,
                )
                reference = gathered[0]
                metadata_match_tensor = torch.all(gathered[:, :-1] == reference[:-1])
                all_capable_tensor = torch.all(gathered[:, -1] == 1)
            collective_stream.synchronize()
            metadata_match = bool(metadata_match_tensor.item())
            all_capable = bool(all_capable_tensor.item())
            return metadata_match and all_capable
        except Exception as exc:
            self._healthy = False
            raise ShardCollectiveError(
                "shard-gather metadata consensus failed",
                data_submitted=False,
            ) from exc

    def gather_dense_rows_gpu_into(
        self,
        descriptor: CollectiveDescriptor,
        *,
        local_ready: bool,
        source_rows: torch.Tensor,
        logical_destination_rows: torch.Tensor,
        destination_rows: torch.Tensor,
        local_ready_event: Optional[torch.cuda.Event],
        resident_bitmap: torch.Tensor,
    ) -> tuple[torch.cuda.Event, torch.Tensor]:
        """Gather dense rows with one data collective and GPU ready flags."""
        partition = descriptor.partition
        if descriptor.mode is not SSDReadMode.SHARD_GATHER_DENSE:
            raise ValueError("GPU dense gather requires a dense descriptor")
        if partition.world_size != self._world_size:
            raise ValueError("partition world size does not match transport")
        if not self._healthy or not self._slots or self._stream is None:
            raise ShardCollectiveError(
                "GPU dense gather resources are unavailable",
                data_submitted=False,
            )
        if self._metadata_stream is None:
            raise ShardCollectiveError(
                "GPU dense metadata stream is unavailable",
                data_submitted=False,
            )
        data_submitted = False
        try:
            with self._lock, torch.inference_mode():
                slot = self._slots[self._slot_cursor]
                self._slot_cursor = (self._slot_cursor + 1) % len(self._slots)
                local_blocks = partition.blocks_for_rank(self._rank)
                ready = bool(
                    local_ready and len(local_blocks) <= partition.padded_blocks
                )
                ready_send = torch.tensor(
                    [int(ready)], dtype=torch.int32, device=source_rows.device
                )
                ready_receive = torch.empty(
                    self._world_size, dtype=torch.int32, device=source_rows.device
                )
                with torch.cuda.stream(self._metadata_stream):
                    torch.distributed.all_gather_into_tensor(
                        ready_receive,
                        ready_send,
                        group=self._metadata_process_group,
                        async_op=True,
                    )
                    metadata_ready = torch.cuda.Event()
                    metadata_ready.record(self._metadata_stream)
                with torch.cuda.stream(self._stream):
                    if slot.completion_event is not None:
                        self._stream.wait_event(slot.completion_event)
                    if ready and local_ready_event is not None:
                        self._stream.wait_event(local_ready_event)
                    slot.send[: partition.padded_blocks].zero_()
                    local_ids = torch.as_tensor(
                        local_blocks, dtype=torch.int64, device=source_rows.device
                    )
                    if ready and local_ids.numel():
                        local_rows = logical_destination_rows.index_select(0, local_ids)
                        torch.index_select(
                            source_rows,
                            0,
                            local_rows,
                            out=slot.send[: local_ids.numel()],
                        )
                    gathered_rows = partition.padded_blocks * self._world_size
                    data_submitted = True
                    data_work = torch.distributed.all_gather_into_tensor(
                        slot.receive[:gathered_rows],
                        slot.send[: partition.padded_blocks],
                        group=self._process_group,
                        async_op=True,
                    )
                    self._stream.wait_event(metadata_ready)
                    ids, positions, owners = dense_rank_major_metadata(partition)
                    ids_gpu = torch.as_tensor(
                        ids, dtype=torch.int64, device=source_rows.device
                    )
                    positions_gpu = torch.as_tensor(
                        positions, dtype=torch.int64, device=source_rows.device
                    )
                    owners_gpu = torch.as_tensor(
                        owners, dtype=torch.int64, device=source_rows.device
                    )
                    publish = ready_receive.index_select(0, owners_gpu).bool()
                    selected = ids_gpu[publish]
                    source_positions = positions_gpu[publish]
                    gathered = slot.receive[:gathered_rows].index_select(
                        0, source_positions
                    )
                    destination = logical_destination_rows.index_select(0, selected)
                    destination_rows.index_copy_(0, destination, gathered)
                    resident_bitmap[selected] = True
                    event = torch.cuda.Event()
                    event.record(self._stream)
                    slot.completion_event = event
                    slot.completion_work = data_work
            return event, selected
        except Exception as exc:
            self._healthy = False
            raise ShardCollectiveError(
                "GPU dense gather failed", data_submitted=data_submitted
            ) from exc

    def gather_into(
        self,
        descriptor: CollectiveDescriptor,
        *,
        source_rows: torch.Tensor,
        logical_destination_rows: torch.Tensor,
        destination_rows: torch.Tensor,
        local_ready_event: Optional[torch.cuda.Event],
        resident_bitmap: torch.Tensor,
    ) -> torch.cuda.Event:
        """All-gather owner rows and scatter them to union destinations.

        Args:
            descriptor: Descriptor previously accepted by :meth:`preflight`.
            source_rows: Flat byte view of the rank-local K cache.
            logical_destination_rows: Logical block id to physical row table.
            destination_rows: Flat byte view receiving the complete union.
            local_ready_event: Event recorded after this rank's SSD scatter.
            resident_bitmap: GPU bitmap updated for every gathered block.

        Returns:
            CUDA event recorded after gather, inverse mapping, and bitmap update.

        Raises:
            ShardCollectiveError: If a submitted data collective fails.
            ValueError: If tensor or partition geometry is incompatible.
        """
        partition = descriptor.partition
        if partition.world_size != self._world_size:
            raise ValueError("partition world size does not match transport")
        if not self._healthy or not self._slots or self._stream is None:
            raise ShardCollectiveError(
                "shard-gather communicator is unavailable",
                data_submitted=False,
            )
        local_blocks = partition.blocks_for_rank(self._rank)
        if len(local_blocks) > partition.padded_blocks:
            raise ValueError("local partition exceeds padded collective count")
        data_submitted = False
        try:
            with self._lock, torch.inference_mode(), torch.cuda.stream(self._stream):
                slot = self._slots[self._slot_cursor]
                self._slot_cursor = (self._slot_cursor + 1) % len(self._slots)
                if slot.completion_event is not None:
                    self._stream.wait_event(slot.completion_event)
                if local_ready_event is not None:
                    self._stream.wait_event(local_ready_event)
                local_ids = torch.as_tensor(
                    local_blocks,
                    dtype=torch.int64,
                    device=source_rows.device,
                )
                local_rows = logical_destination_rows.index_select(0, local_ids)
                if local_ids.numel():
                    torch.index_select(
                        source_rows,
                        0,
                        local_rows,
                        out=slot.send[: int(local_ids.numel())],
                    )
                if int(local_ids.numel()) < partition.padded_blocks:
                    slot.send[int(local_ids.numel()) : partition.padded_blocks].zero_()
                data_submitted = True
                gathered_rows = partition.padded_blocks * self._world_size
                work = torch.distributed.all_gather_into_tensor(
                    slot.receive[:gathered_rows],
                    slot.send[: partition.padded_blocks],
                    group=self._process_group,
                    async_op=True,
                )
                work.wait()
                if self._config.debug_verify:
                    checksum = torch.sum(
                        slot.receive[:gathered_rows],
                        dtype=torch.int64,
                    ).reshape(1)
                    checksums = torch.empty(
                        self._world_size,
                        dtype=torch.int64,
                        device=source_rows.device,
                    )
                    torch.distributed.all_gather_into_tensor(
                        checksums,
                        checksum,
                        group=self._process_group,
                    )
                    if not bool(torch.all(checksums == checksums[0])):
                        raise RuntimeError(
                            "gathered destination checksum differs across ranks"
                        )
                materialization_error: Optional[Exception] = None
                try:
                    for owner_rank, owner_blocks in enumerate(partition.blocks_by_rank):
                        if not owner_blocks:
                            continue
                        owner_ids = torch.as_tensor(
                            owner_blocks,
                            dtype=torch.int64,
                            device=source_rows.device,
                        )
                        owner_rows = logical_destination_rows.index_select(
                            0,
                            owner_ids,
                        )
                        receive_start = owner_rank * partition.padded_blocks
                        receive_end = receive_start + len(owner_blocks)
                        destination_rows.index_copy_(
                            0,
                            owner_rows,
                            slot.receive[receive_start:receive_end],
                        )
                    union_ids = torch.as_tensor(
                        partition.union,
                        dtype=torch.int64,
                        device=resident_bitmap.device,
                    )
                    if union_ids.numel():
                        resident_bitmap.index_fill_(0, union_ids, True)
                except Exception as exc:
                    materialization_error = exc
                # Once the data all-gather has been submitted, a rank-local
                # scatter failure must fail every participant and disable the
                # feature consistently. Otherwise healthy ranks could enter a
                # later collective while the failed rank has fallen back.
                materialization_status = torch.tensor(
                    [int(materialization_error is None)],
                    dtype=torch.int32,
                    device=source_rows.device,
                )
                torch.distributed.all_reduce(
                    materialization_status,
                    op=torch.distributed.ReduceOp.MIN,
                    group=self._process_group,
                )
                if not bool(materialization_status.item()):
                    raise RuntimeError(
                        "at least one rank failed shard materialization"
                    ) from materialization_error
                completion = torch.cuda.Event()
                completion.record(self._stream)
                slot.completion_event = completion
                return completion
        except Exception as exc:
            self._healthy = False
            raise ShardCollectiveError(
                "shard-gather data collective or materialization failed",
                data_submitted=data_submitted,
            ) from exc

    def close(self) -> None:
        """Synchronize staging and release the owned process group."""
        with self._metadata_lock:
            if self._metadata_stream is not None:
                self._metadata_stream.synchronize()
            self._metadata_stream = None
        with self._lock:
            for slot in self._slots:
                if slot.completion_event is not None:
                    slot.completion_event.synchronize()
            self._slots.clear()
            self._stream = None
        if self._owns_process_group:
            try:
                torch.distributed.destroy_process_group(self._process_group)
            finally:
                self._owns_process_group = False
        if self._owns_metadata_process_group:
            try:
                torch.distributed.destroy_process_group(self._metadata_process_group)
            finally:
                self._owns_metadata_process_group = False


def parse_layer_ranges(specification: str) -> frozenset[int]:
    """Parse a comma-separated layer/range specification.

    Args:
        specification: Entries such as ``"2-24,30,32-34"``.

    Returns:
        The expanded non-negative layer ids.

    Raises:
        ValueError: If an entry is malformed, negative, or reversed.
    """
    layers: set[int] = set()
    for raw_entry in specification.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        bounds = entry.split("-", maxsplit=1)
        try:
            start = int(bounds[0])
            end = int(bounds[1]) if len(bounds) == 2 else start
        except ValueError as exc:
            raise ValueError(f"invalid layer range {entry!r}") from exc
        if start < 0 or end < start:
            raise ValueError(f"invalid layer range {entry!r}")
        layers.update(range(start, end + 1))
    return frozenset(layers)


def stable_union_hash(block_ids: Sequence[int]) -> int:
    """Return a stable signed-63-bit digest for ordered block ids.

    Args:
        block_ids: Ordered, process-independent logical block ids.

    Returns:
        A digest safe to transmit in a signed int64 metadata tensor.

    Raises:
        ValueError: If a block id is negative.
    """
    digest = hashlib.blake2b(digest_size=8, person=b"lmcache1")
    digest.update(len(block_ids).to_bytes(8, byteorder="little", signed=False))
    for block_id in block_ids:
        value = int(block_id)
        if value < 0:
            raise ValueError("block ids must be non-negative")
        digest.update(value.to_bytes(8, byteorder="little", signed=False))
    return int.from_bytes(digest.digest(), byteorder="little") & ((1 << 63) - 1)


def deterministic_block_union(block_ids: Sequence[int]) -> tuple[int, ...]:
    """Deduplicate block ids into the canonical ascending union.

    Args:
        block_ids: Possibly duplicated logical ids in arbitrary order.

    Returns:
        Sorted, duplicate-free block ids.

    Raises:
        ValueError: If any id is negative.
    """
    normalized = tuple(int(block_id) for block_id in block_ids)
    if any(block_id < 0 for block_id in normalized):
        raise ValueError("block ids must be non-negative")
    return tuple(sorted(set(normalized)))


def partition_block_union(
    block_ids: Sequence[int],
    world_size: int,
) -> BlockPartition:
    """Partition a canonical block union into balanced contiguous slices.

    Args:
        block_ids: Possibly duplicated logical block ids.
        world_size: Number of participating ranks.

    Returns:
        A deterministic partition whose rank counts differ by at most one.

    Raises:
        ValueError: If ``world_size`` is not positive or an id is negative.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    union = deterministic_block_union(block_ids)
    base, remainder = divmod(len(union), world_size)
    partitions: list[tuple[int, ...]] = []
    cursor = 0
    for rank in range(world_size):
        count = base + int(rank < remainder)
        partitions.append(union[cursor : cursor + count])
        cursor += count
    padded = base + int(remainder > 0)
    return BlockPartition(
        union=union,
        blocks_by_rank=tuple(partitions),
        padded_blocks=padded,
        union_hash=stable_union_hash(union),
    )


def partition_rank_local_blocks(
    rank_block_ids: Sequence[Sequence[int]],
    *,
    max_union_blocks: Optional[int] = None,
) -> BlockPartition:
    """Preserve rank-local candidate ownership in one collective descriptor.

    Args:
        rank_block_ids: Candidate block IDs selected independently by each rank.
        max_union_blocks: Optional global staging limit. The canonical union is
            truncated before ownership is materialized.

    Returns:
        A deterministic partition. Duplicate candidates are owned by the
        lowest rank that selected them.

    Raises:
        ValueError: If there are no ranks, the limit is invalid, or an ID is
            negative.
    """
    if not rank_block_ids:
        raise ValueError("rank_block_ids must contain at least one rank")
    if max_union_blocks is not None and max_union_blocks <= 0:
        raise ValueError("max_union_blocks must be positive")
    canonical_by_rank = tuple(
        deterministic_block_union(block_ids) for block_ids in rank_block_ids
    )
    union = deterministic_block_union(
        tuple(block_id for blocks in canonical_by_rank for block_id in blocks)
    )
    if max_union_blocks is not None:
        union = union[:max_union_blocks]
    retained = set(union)
    claimed: set[int] = set()
    owners: list[tuple[int, ...]] = []
    for blocks in canonical_by_rank:
        owned = tuple(
            block_id
            for block_id in blocks
            if block_id in retained and block_id not in claimed
        )
        claimed.update(owned)
        owners.append(owned)
    padded = max((len(blocks) for blocks in owners), default=0)
    return BlockPartition(
        union=union,
        blocks_by_rank=tuple(owners),
        padded_blocks=padded,
        union_hash=stable_union_hash(union),
    )


def rank_major_inverse_indices(partition: BlockPartition) -> tuple[int, ...]:
    """Return rank-major padded positions that restore union order.

    Args:
        partition: Deterministic block partition.

    Returns:
        For each union position, the corresponding flattened rank-major
        collective-buffer index. Padding positions are never returned.
    """
    inverse: dict[int, int] = {}
    for rank, blocks in enumerate(partition.blocks_by_rank):
        base = rank * partition.padded_blocks
        for local_index, block_id in enumerate(blocks):
            inverse[block_id] = base + local_index
    return tuple(inverse[block_id] for block_id in partition.union)


def cp_owned_row_ranges(
    total_rows: int,
    rank: int,
    world_size: int,
    interleave_rows: int,
) -> tuple[RowRange, ...]:
    """Return block-cyclic context rows owned by one CP rank.

    Args:
        total_rows: Total logical context rows.
        rank: Rank in the CP group.
        world_size: Number of CP ranks.
        interleave_rows: Consecutive rows assigned to one rank per cycle.

    Returns:
        Sorted, coalesced half-open ranges.

    Raises:
        ValueError: If partition geometry is invalid.
    """
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid CP rank or world size")
    if interleave_rows <= 0:
        raise ValueError("interleave_rows must be positive")
    cycle = world_size * interleave_rows
    start = rank * interleave_rows
    ranges: list[RowRange] = []
    while start < total_rows:
        end = min(start + interleave_rows, total_rows)
        if ranges and ranges[-1].end == start:
            ranges[-1] = RowRange(ranges[-1].start, end)
        else:
            ranges.append(RowRange(start, end))
        start += cycle
    return tuple(ranges)


def compile_cp_read_plan(
    *,
    total_rows: int,
    rank: int,
    world_size: int,
    interleave_rows: int,
    block_rows: int,
    row_bytes: int,
) -> CPReadPlan:
    """Compile CP ownership to whole layer-major SSD blocks.

    The first implementation intentionally fails closed when an ownership
    boundary cuts through a non-tail SSD block. Reading that block would load
    rows owned by another rank and invalidate the advertised 1/P byte model.

    Args:
        total_rows: Total Indexer-K logical rows.
        rank: Rank in the CP group.
        world_size: Number of CP ranks.
        interleave_rows: Consecutive rows assigned per cycle.
        block_rows: Rows stored in one layer-major SSD block.
        row_bytes: Bytes in one logical row.

    Returns:
        Coalesced row ranges and their exact whole-block SGL selection.

    Raises:
        ValueError: If geometry is invalid or ownership cuts through a block.
    """
    if block_rows <= 0 or row_bytes <= 0:
        raise ValueError("block_rows and row_bytes must be positive")
    ranges = cp_owned_row_ranges(
        total_rows,
        rank,
        world_size,
        interleave_rows,
    )
    blocks: list[int] = []
    for row_range in ranges:
        if row_range.start % block_rows:
            raise ValueError("CP ownership starts inside an SSD block")
        if row_range.end != total_rows and row_range.end % block_rows:
            raise ValueError("CP ownership ends inside an SSD block")
        block_start = row_range.start // block_rows
        block_end = math.ceil(row_range.end / block_rows)
        blocks.extend(range(block_start, block_end))
    return CPReadPlan(
        rank=rank,
        world_size=world_size,
        total_rows=total_rows,
        row_ranges=ranges,
        block_ids=tuple(blocks),
        row_bytes=row_bytes,
        block_rows=block_rows,
    )


def bucket_prefetch_key(
    *,
    group: str,
    layer_id: int,
    context_tokens: int,
    query_tokens: int,
    union_blocks: int,
    buckets: Mapping[str, int] | None = None,
) -> PrefetchDecisionKey:
    """Build a stable coarse key for decision-table lookup.

    Args:
        group: Logical cache group.
        layer_id: Transformer layer id.
        context_tokens: Cached context length.
        query_tokens: Incremental query length.
        union_blocks: Required union block count.
        buckets: Optional bucket widths for ``context``, ``query``, and
            ``union``. Defaults are 32K, 1K, and 128.

    Returns:
        A rounded-up decision key.

    Raises:
        ValueError: If an input or bucket width is invalid.
    """
    widths = {"context": 32768, "query": 1024, "union": 128}
    if buckets is not None:
        widths.update({name: int(value) for name, value in buckets.items()})
    if (
        layer_id < 0
        or context_tokens < 0
        or query_tokens < 0
        or union_blocks < 0
        or any(value <= 0 for value in widths.values())
    ):
        raise ValueError("invalid decision-table bucket input")

    def _ceil_bucket(value: int, width: int) -> int:
        return math.ceil(value / width) * width if value else 0

    return PrefetchDecisionKey(
        group=str(group),
        layer_id=int(layer_id),
        context_bucket=_ceil_bucket(context_tokens, widths["context"]),
        query_bucket=_ceil_bucket(query_tokens, widths["query"]),
        union_bucket=_ceil_bucket(union_blocks, widths["union"]),
    )
