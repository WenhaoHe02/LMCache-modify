# SPDX-License-Identifier: Apache-2.0
"""Predictive prefetch control plane for GLM DSA IndexShare groups.

This module deliberately does not import vLLM.  It owns the model-topology
schedule, request lifecycle, prediction/correction events, and lightweight
top-K agreement telemetry.  A version-specific adapter supplies the predictor
and the physical KV prefetch submitter.
"""

from __future__ import annotations

# Standard
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from threading import RLock
import time
from typing import Any

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.csa_pipeline_nvtx import CsaNvtxEvent, csa_pipeline_nvtx
from lmcache.v1.index_to_io_profile import ModelTopologyProfile

logger = init_logger(__name__)


def _timing_enabled() -> bool:
    """Return whether detailed attention-KV timing is enabled."""
    return os.environ.get("LMCACHE_CSA_ATTENTION_KV_TIMING", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _accuracy_profile_enabled() -> bool:
    """Return whether prediction agreement telemetry is enabled."""
    return os.environ.get("LMCACHE_INDEXER_PROFILE_ACCURACY", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _decode_coverage_profile_enabled() -> bool:
    """Return whether read-only decode residency telemetry is enabled."""
    return os.environ.get(
        "LMCACHE_GLM_DSA_DECODE_COVERAGE_PROFILE",
        "0",
    ).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class GLMDSAPredictionSchedule:
    """One early-prediction edge for an IndexShare group.

    Args:
        source_layer: Decoder layer whose post-MLP state triggers prediction.
        target_layer: Full-indexer layer being predicted.
        group_id: Stable IndexShare group identifier.
        consumer_layers: Layers that consume the target Full index.
        is_bootstrap: Whether this is the special early ``L0 -> L2`` edge.
    """

    source_layer: int
    target_layer: int
    group_id: str
    consumer_layers: tuple[int, ...]
    is_bootstrap: bool

    def __post_init__(self) -> None:
        if self.source_layer < 0 or self.target_layer < 0:
            raise ValueError("prediction layers must be non-negative")
        if self.source_layer >= self.target_layer:
            raise ValueError("prediction source must precede its target")
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        if self.target_layer not in self.consumer_layers:
            raise ValueError("target layer must consume its predicted index")


@dataclass(frozen=True, slots=True)
class GLMDSAPrefetchEvent:
    """A predicted or corrected IndexShare prefetch submission.

    Args:
        request_id: Active LMCache request identifier.
        schedule: Prediction edge that produced this event.
        topk_indices: Predicted or true token indices, shaped ``[rows, topk]``.
        correction: ``False`` for speculative prefetch and ``True`` after the
            target Full indexer produces authoritative indices.
        profile_operation_id: Optional profile-only correlation identifier.
    """

    request_id: str
    schedule: GLMDSAPredictionSchedule
    topk_indices: torch.Tensor
    correction: bool
    profile_operation_id: str | None = None


@dataclass(frozen=True, slots=True)
class _StagedPrediction:
    """One Shared-consumer prediction waiting for a decoder progress gate."""

    request_id: str
    source_layer: int
    target_layer: int
    group_id: str
    consumer_layer: int
    blocks: torch.Tensor
    level: int
    request_token: Any
    profile_operation_id: str | None = None


def _glm_profile_operation_id(
    event: GLMDSAPrefetchEvent,
    phase: str,
    consumer_layer: int,
) -> str | None:
    """Return one profile-only operation id without touching the disabled path."""
    if not csa_pipeline_nvtx.enabled:
        return None
    base = event.profile_operation_id or (
        f"glm-{phase}-{event.schedule.source_layer}-"
        f"{event.schedule.target_layer}-{time.monotonic_ns()}"
    )
    if int(consumer_layer) == event.schedule.target_layer:
        return base
    return f"{base}-consumer-{int(consumer_layer)}"


def _glm_io_profile_kwargs(
    event: GLMDSAPrefetchEvent,
    phase: str,
    consumer_layer: int,
) -> dict[str, object]:
    """Build optional source/op metadata accepted by the Tutti manager."""
    operation_id = _glm_profile_operation_id(event, phase, consumer_layer)
    if operation_id is None:
        return {}
    source_layer = (
        event.schedule.target_layer
        if phase == "correction"
        else event.schedule.source_layer
    )
    return {
        "profile_source_layer": int(source_layer),
        "profile_operation_id": operation_id,
        "profile_kind": f"glm_dsa_{phase}",
    }


@dataclass(frozen=True, slots=True)
class TopKAgreement:
    """Sampled set agreement between predicted and authoritative top-K rows.

    Args:
        sampled_rows: Number of tail query rows compared.
        intersection: Valid true indices also present in the predicted row.
        predicted: Number of valid predicted indices in sampled rows.
        actual: Number of valid authoritative indices in sampled rows.
    """

    sampled_rows: int
    intersection: int
    predicted: int
    actual: int

    @property
    def recall(self) -> float:
        """Return the fraction of authoritative indices predicted early."""
        return self.intersection / self.actual if self.actual else 1.0

    @property
    def precision(self) -> float:
        """Return the fraction of predicted indices that were authoritative."""
        return self.intersection / self.predicted if self.predicted else 1.0


@dataclass(frozen=True, slots=True)
class GLMDSALayerStats:
    """Cumulative agreement counters for one target Full layer."""

    observations: int
    sampled_rows: int
    intersection: int
    predicted: int
    actual: int

    @property
    def recall(self) -> float:
        """Return cumulative sampled top-K recall."""
        return self.intersection / self.actual if self.actual else 1.0

    @property
    def precision(self) -> float:
        """Return cumulative sampled top-K precision."""
        return self.intersection / self.predicted if self.predicted else 1.0


PredictTopK = Callable[
    [GLMDSAPredictionSchedule, torch.Tensor, torch.Tensor], torch.Tensor
]
SubmitPrefetch = Callable[[GLMDSAPrefetchEvent], None]


class GLMDSAPhysicalPrefetchSink:
    """Translate GLM DSA token indices into real layer-major KV reads.

    The sink reuses the production CSA attention-KV Tutti manager. Predicted
    reads are submitted on background workers, authoritative top-K output
    joins the prediction submission before issuing misses, and the current
    consumer is gated until its CUDA scatter has landed.

    Args:
        attention_kv_manager: Active layer-major Tutti KV prefetch manager.
        schedules: Prediction edges whose consumer groups share one index.
        index_groups: Mapping from Full indexer layer to consumer layers.
        compressed_block_size: vLLM sparse MLA cache page size in tokens.
        io_workers: Number of background prediction/correction submitters.
        enable_prediction: Whether speculative reads are issued before the
            authoritative top-K arrives. ``None`` reads the environment and
            defaults to enabled.
    """

    def __init__(
        self,
        attention_kv_manager: Any,
        schedules: tuple[GLMDSAPredictionSchedule, ...],
        index_groups: Mapping[int, tuple[int, ...]],
        compressed_block_size: int,
        *,
        io_workers: int = 4,
        enable_prediction: bool | None = None,
    ) -> None:
        if compressed_block_size <= 0:
            raise ValueError("compressed_block_size must be positive")
        if io_workers <= 0:
            raise ValueError("io_workers must be positive")
        self._attention_kv_manager = attention_kv_manager
        self._scheduled_targets = {schedule.target_layer for schedule in schedules}
        self._index_groups = {
            int(layer): tuple(int(value) for value in consumers)
            for layer, consumers in index_groups.items()
        }
        self._compressed_block_size = int(compressed_block_size)
        try:
            self._prediction_block_budget = max(
                1,
                int(
                    os.getenv(
                        "LMCACHE_GLM_DSA_PREFETCH_BLOCK_BUDGET",
                        "2048",
                    )
                ),
            )
        except ValueError as exc:
            raise ValueError(
                "LMCACHE_GLM_DSA_PREFETCH_BLOCK_BUDGET must be an integer"
            ) from exc
        if enable_prediction is None:
            enable_prediction = os.getenv(
                "LMCACHE_GLM_DSA_PHYSICAL_PREDICTION",
                "1",
            ).strip().lower() in {"1", "true", "yes", "on"}
        self._enable_prediction = bool(enable_prediction)
        self._early_dense_group = os.getenv(
            "LMCACHE_GLM_DSA_EARLY_DENSE_GROUP",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        shared_prediction_value = (
            os.getenv(
                "LMCACHE_GLM_DSA_PREDICT_SHARED_CONSUMERS",
                "1",
            )
            .strip()
            .lower()
        )
        shared_prediction_modes = {
            "1": "all",
            "true": "all",
            "yes": "all",
            "on": "all",
            "all": "all",
            "0": "none",
            "false": "none",
            "no": "none",
            "off": "none",
            "none": "none",
            "staged": "staged",
        }
        try:
            self._shared_prediction_mode = shared_prediction_modes[
                shared_prediction_value
            ]
        except KeyError as exc:
            raise ValueError(
                "LMCACHE_GLM_DSA_PREDICT_SHARED_CONSUMERS must be one of "
                "all, staged, or none"
            ) from exc
        self._executor = ThreadPoolExecutor(
            max_workers=io_workers,
            thread_name_prefix="lmcache-glm-dsa-kv",
        )
        # Sharded predicted reads execute CP collectives.  A multi-worker
        # executor can start adjacent consumer layers in a different order on
        # each TP rank, so one rank may enter L76 while another enters L75.
        # Keep collective-producing prediction work FIFO while retaining the
        # general executor above for local miss-correction submissions.
        self._prediction_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lmcache-glm-dsa-predict",
        )
        self._lock = RLock()
        self._predicted_blocks: dict[int, torch.Tensor | None] = {}
        self._staged_request_id: str | None = None
        self._pending_prediction_stages: dict[int, list[_StagedPrediction]] = (
            defaultdict(list)
        )
        self._staged_prediction_futures: dict[int, Future[Any]] = {}
        self._decode_selected_blocks: dict[int, set[int]] = defaultdict(set)
        self._decode_missing_blocks: dict[int, set[int]] = defaultdict(set)
        logger.info(
            "GLM DSA physical prediction reads enabled=%s block_budget=%d "
            "shared_prediction_mode=%s early_dense_group=%s",
            self._enable_prediction,
            self._prediction_block_budget,
            self._shared_prediction_mode,
            self._early_dense_group,
        )

    def submit(self, event: GLMDSAPrefetchEvent) -> None:
        """Submit predicted blocks or authoritative miss correction.

        Args:
            event: Request-scoped prediction or correction event.
        """
        manager = self._attention_kv_manager
        if str(getattr(manager, "active_request_id", "")) != event.request_id:
            return
        if not event.correction and not self._enable_prediction:
            return
        consumers = event.schedule.consumer_layers
        request_token = getattr(manager, "active_request_token", None)
        if not event.correction and self._early_dense_group:
            # Dense-group overlap is an opt-in alternative to sparse physical
            # prediction.  The model predictor still decides *when* the group
            # becomes available, while the storage path uses complete
            # layer-major reads.  This avoids one fragmented indexed-read
            # submission per predicted block and lets independent local layers
            # overlap on the regular I/O executor.  TP-sharded deployments
            # should leave this mode disabled because dense shard gather owns
            # collective ordering separately.
            with self._lock:
                if self._staged_request_id != event.request_id:
                    self._pending_prediction_stages.clear()
                    self._staged_prediction_futures.clear()
                    self._staged_request_id = event.request_id
                self._predicted_blocks[event.schedule.target_layer] = None
            for layer_id in consumers:
                future = self._executor.submit(
                    manager.fire_dense_layer,
                    int(layer_id),
                    request_token=request_token,
                    source_layer_id=event.schedule.source_layer,
                )
                manager.track_layer_submission(
                    int(layer_id),
                    future,
                    request_token=request_token,
                )
            return
        if event.correction and self._early_dense_group:
            # ``fire_dense_layer`` covers every registered block, so the
            # authoritative top-K cannot discover a physical miss.  Avoid
            # flattening/uniquing the full top-K tensor and rescanning the
            # resident bitmap once per consumer.  Only the Full target is on
            # the current critical path; later consumers keep their tracked
            # dense futures and join them at their own attention gates.  In
            # particular, do not eagerly join the whole group here, which
            # would serialize otherwise-overlappable reads behind this Full
            # layer.
            target_layer = int(event.schedule.target_layer)
            with self._lock:
                self._predicted_blocks.pop(target_layer, None)
                self._drop_pending_target_locked(target_layer)
            if target_layer not in consumers:
                return
            if not manager.wait_for_layer(
                target_layer,
                timeout_s=self._gate_timeout_s(),
            ):
                raise RuntimeError(
                    f"GLM DSA dense-group read timed out at layer {target_layer}"
                )
            return
        blocks = self._topk_to_blocks(
            event.topk_indices,
            block_budget=(None if event.correction else self._prediction_block_budget),
        )
        if event.correction:
            with self._lock:
                predicted = self._predicted_blocks.pop(
                    event.schedule.target_layer,
                    None,
                )
                self._drop_pending_target_locked(event.schedule.target_layer)
            self._log_block_agreement(event, predicted, blocks)
            for layer_id in consumers:
                if int(layer_id) == event.schedule.target_layer:
                    correction_start = time.perf_counter()
                    join_start = correction_start
                    prediction_ready = manager.wait_for_tracked_submission(
                        int(layer_id)
                    )
                    join_ms = (time.perf_counter() - join_start) * 1000.0
                    submit_start = time.perf_counter()
                    manager.submit_miss_reads(
                        int(layer_id),
                        blocks,
                        request_token=request_token,
                        **_glm_io_profile_kwargs(
                            event,
                            "correction",
                            int(layer_id),
                        ),
                    )
                    submit_ms = (time.perf_counter() - submit_start) * 1000.0
                    gate_start = time.perf_counter()
                    gate_ready = manager.wait_for_layer(
                        int(layer_id),
                        timeout_s=self._gate_timeout_s(),
                    )
                    gate_ms = (time.perf_counter() - gate_start) * 1000.0
                    if _timing_enabled():
                        logger.info(
                            "GLM_DSA_CORRECTION_PROFILE request=%s target=L%d "
                            "blocks=%d prediction_ready=%d join_ms=%.3f "
                            "submit_ms=%.3f gate_ms=%.3f total_ms=%.3f",
                            event.request_id,
                            int(layer_id),
                            int(blocks.numel()),
                            int(prediction_ready),
                            join_ms,
                            submit_ms,
                            gate_ms,
                            (time.perf_counter() - correction_start) * 1000.0,
                        )
                    if not gate_ready:
                        raise RuntimeError(
                            f"GLM DSA KV correction timed out at layer {layer_id}"
                        )
                    continue
                shared_join_start = time.perf_counter()
                with self._lock:
                    staged_future = self._staged_prediction_futures.pop(
                        int(layer_id),
                        None,
                    )
                # Both modes must *join* the staged prediction through the
                # manager, not merely observe that its future is done.  The
                # manager tracks the same future and a sharded prediction
                # returns deferred gather work that only
                # ``wait_for_tracked_submission`` finalizes; polling ``done()``
                # leaves that work booked against the layer's pending-read
                # count and the consumer gate then blocks until it times out.
                shared_prediction_ready = manager.wait_for_tracked_submission(
                    int(layer_id)
                )
                shared_join_ms = (time.perf_counter() - shared_join_start) * 1000.0
                shared_submit_start = time.perf_counter()
                if self._shared_prediction_mode == "staged":
                    # ``wait_for_tracked_submission`` already consumed the
                    # tracked future above, so pass ``None`` rather than
                    # re-awaiting the popped one.
                    del staged_future
                    future = self._executor.submit(
                        self._submit_after_prediction,
                        None,
                        int(layer_id),
                        blocks,
                        request_token,
                        event,
                    )
                else:
                    future = self._executor.submit(
                        manager.submit_miss_reads,
                        int(layer_id),
                        blocks,
                        request_token=request_token,
                        **_glm_io_profile_kwargs(
                            event,
                            "correction",
                            int(layer_id),
                        ),
                    )
                manager.track_layer_submission(
                    int(layer_id),
                    future,
                    request_token=request_token,
                )
                if _timing_enabled():
                    logger.info(
                        "GLM_DSA_SHARED_CORRECTION_PROFILE request=%s "
                        "target=L%d consumer=L%d blocks=%d "
                        "prediction_ready=%d join_ms=%.3f enqueue_ms=%.3f",
                        event.request_id,
                        event.schedule.target_layer,
                        int(layer_id),
                        int(blocks.numel()),
                        int(shared_prediction_ready),
                        shared_join_ms,
                        (time.perf_counter() - shared_submit_start) * 1000.0,
                    )
            return

        with self._lock:
            if self._staged_request_id != event.request_id:
                self._pending_prediction_stages.clear()
                self._staged_prediction_futures.clear()
                self._staged_request_id = event.request_id
            self._predicted_blocks[event.schedule.target_layer] = blocks
        level = min(
            2,
            event.schedule.target_layer - event.schedule.source_layer,
        )
        prediction_consumers = (
            consumers
            if self._shared_prediction_mode == "all"
            else (event.schedule.target_layer,)
        )
        for layer_id in prediction_consumers:
            self._submit_predicted_layer(
                int(layer_id),
                blocks,
                level,
                request_token,
                event=event,
            )
        if self._shared_prediction_mode == "staged":
            self._stage_shared_predictions(
                event,
                blocks,
                level,
                request_token,
            )

    def submit_authoritative(
        self,
        target_layer: int,
        true_topk: torch.Tensor,
    ) -> None:
        """Demand-load correction-only early Full groups (L0 and L1).

        Args:
            target_layer: Full indexer layer producing ``true_topk``.
            true_topk: Authoritative token indices for the group.
        """
        target = int(target_layer)
        # ON receives scheduled Full groups through prediction correction.
        # Layer-wise OFF has no producer event, so the authoritative observer
        # owns the same consumer group directly.
        if target in self._scheduled_targets and self._enable_prediction:
            return
        consumers = self._index_groups.get(target)
        if not consumers:
            return
        manager = self._attention_kv_manager
        # Cold-store requests have no SSD read plan.  Their Full indexers still
        # execute (and therefore still reach this observer), but attempting a
        # demand read would incorrectly turn a normal cold forward into a
        # fail-closed stale-plan error.
        if not str(getattr(manager, "active_request_id", "")):
            return
        request_token = getattr(manager, "active_request_token", None)
        submit_topk_misses = getattr(manager, "submit_topk_miss_reads", None)
        blocks = None
        if not callable(submit_topk_misses):
            # Compatibility fallback for external/fake managers. Production
            # uses submit_topk_miss_reads so only its compact miss set crosses
            # the CUDA-to-CPU boundary.
            blocks = self._topk_to_blocks(true_topk, block_budget=None)
        for layer_id in consumers:
            profile_kwargs = (
                {
                    "profile_source_layer": target,
                    "profile_operation_id": (
                        f"glm-authoritative-{target}-{int(layer_id)}-"
                        f"{time.monotonic_ns()}"
                    ),
                    "profile_kind": "glm_dsa_correction",
                }
                if csa_pipeline_nvtx.enabled
                else {}
            )
            if int(layer_id) == target:
                if callable(submit_topk_misses):
                    submit_topk_misses(
                        target,
                        true_topk,
                        request_token=request_token,
                        **profile_kwargs,
                    )
                else:
                    manager.submit_miss_reads(
                        target,
                        blocks,
                        request_token=request_token,
                        **profile_kwargs,
                    )
                if not manager.wait_for_layer(
                    target,
                    timeout_s=self._gate_timeout_s(),
                ):
                    raise RuntimeError(
                        f"GLM DSA KV demand load timed out at layer {target}"
                    )
                continue
            if callable(submit_topk_misses):
                future = self._executor.submit(
                    submit_topk_misses,
                    int(layer_id),
                    true_topk,
                    request_token=request_token,
                    **profile_kwargs,
                )
            else:
                future = self._executor.submit(
                    manager.submit_miss_reads,
                    int(layer_id),
                    blocks,
                    request_token=request_token,
                    **profile_kwargs,
                )
            manager.track_layer_submission(
                int(layer_id),
                future,
                request_token=request_token,
            )

    def profile_decode_authoritative(
        self,
        target_layer: int,
        true_topk: torch.Tensor,
    ) -> None:
        """Accumulate decode coverage relative to append-prefill residency.

        Args:
            target_layer: Full indexer layer producing ``true_topk``.
            true_topk: Authoritative decode top-K indices for this group.

        Notes:
            Enabled only by ``LMCACHE_GLM_DSA_DECODE_COVERAGE_PROFILE``. This
            method performs no I/O and never changes the resident bitmap.
        """
        if not _decode_coverage_profile_enabled():
            return
        consumers = self._index_groups.get(int(target_layer), ())
        manager = self._attention_kv_manager
        if not consumers or not str(getattr(manager, "active_request_id", "")):
            return
        query = getattr(manager, "profile_topk_residency", None)
        if not callable(query):
            return
        for layer_id in consumers:
            selected, missing = query(int(layer_id), true_topk)
            with self._lock:
                self._decode_selected_blocks[int(layer_id)].update(
                    int(value) for value in selected.tolist()
                )
                self._decode_missing_blocks[int(layer_id)].update(
                    int(value) for value in missing.tolist()
                )

    def wait_for_consumer(self, layer_id: int) -> bool:
        """Wait until one consumer layer's predicted/corrected KV has landed.

        Args:
            layer_id: Transformer layer about to consume sparse MLA KV.

        Returns:
            Whether every tracked read completed before the timeout.
        """
        manager = self._attention_kv_manager
        with csa_pipeline_nvtx.range(
            CsaNvtxEvent.CONSUMER_WAIT,
            layer_id=int(layer_id),
            target_layer_id=int(layer_id),
            request_id=str(getattr(manager, "active_request_id", "")) or None,
            attributes={"kind": "glm_dsa"},
        ):
            self._release_prediction_stages(int(layer_id))
            return bool(
                manager.wait_for_layer(
                    int(layer_id),
                    timeout_s=self._gate_timeout_s(),
                )
            )

    def finish_request(self, request_id: str) -> bool:
        """Drain physical I/O owned by a completed request.

        Args:
            request_id: Exact active LMCache request identifier.

        Returns:
            ``True`` when the request had no active physical plan or all of
            its tracked reads and scatters completed before teardown.
        """
        manager = self._attention_kv_manager
        if manager.active_request_id != str(request_id):
            return True
        with self._lock:
            selected = sum(
                len(values) for values in self._decode_selected_blocks.values()
            )
            missing = sum(
                len(values) for values in self._decode_missing_blocks.values()
            )
            per_layer = {
                str(layer_id): {
                    "selected": len(values),
                    "missing": len(self._decode_missing_blocks.get(layer_id, ())),
                }
                for layer_id, values in sorted(self._decode_selected_blocks.items())
            }
        if selected:
            logger.info(
                "GLM_DSA_DECODE_COVERAGE request=%s selected_unique_blocks=%d "
                "missing_unique_blocks=%d resident_unique_blocks=%d "
                "resident_fraction=%.9f per_layer=%s",
                request_id,
                selected,
                missing,
                selected - missing,
                (selected - missing) / selected,
                per_layer,
            )
        drained = manager.deactivate_request(timeout_s=self._gate_timeout_s())
        if not drained:
            return False
        with self._lock:
            self._predicted_blocks.clear()
            self._pending_prediction_stages.clear()
            self._staged_prediction_futures.clear()
            self._staged_request_id = None
            self._decode_selected_blocks.clear()
            self._decode_missing_blocks.clear()
        return True

    def close(self) -> None:
        """Drain and close background submission workers."""
        self._prediction_executor.shutdown(wait=True, cancel_futures=False)
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _topk_to_blocks(
        self,
        topk_indices: torch.Tensor,
        *,
        block_budget: int | None,
    ) -> torch.Tensor:
        valid = topk_indices.detach().reshape(-1).to(dtype=torch.int64)
        valid = valid[valid >= 0]
        if valid.numel() == 0:
            return valid
        block_ids = torch.div(
            valid,
            self._compressed_block_size,
            rounding_mode="floor",
        )
        if block_budget is None:
            return torch.unique(block_ids, sorted=True)

        counts = torch.bincount(block_ids)
        budget = min(int(block_budget), int(counts.numel()))
        if budget <= 0:
            return torch.empty(0, dtype=torch.int64, device=valid.device)
        top_counts, top_blocks = torch.topk(counts, k=budget, sorted=False)
        selected = top_blocks[top_counts > 0]
        return torch.sort(selected).values

    def _log_block_agreement(
        self,
        event: GLMDSAPrefetchEvent,
        predicted: torch.Tensor | None,
        actual: torch.Tensor,
    ) -> None:
        if not _accuracy_profile_enabled() or predicted is None or actual.numel() == 0:
            return
        predicted = predicted.to(device=actual.device, dtype=torch.int64)
        locations = torch.searchsorted(predicted, actual).clamp_max(
            max(0, int(predicted.numel()) - 1)
        )
        intersection = (
            int((predicted.index_select(0, locations) == actual).sum().item())
            if predicted.numel()
            else 0
        )
        logger.info(
            "GLM DSA physical block coverage request=%s target=L%d "
            "predicted=%d actual=%d intersection=%d recall=%.6f",
            event.request_id,
            event.schedule.target_layer,
            int(predicted.numel()),
            int(actual.numel()),
            intersection,
            intersection / int(actual.numel()),
        )

    def _stage_shared_predictions(
        self,
        event: GLMDSAPrefetchEvent,
        blocks: torch.Tensor,
        level: int,
        request_token: Any,
    ) -> None:
        shared_consumers = tuple(
            layer
            for layer in event.schedule.consumer_layers
            if layer != event.schedule.target_layer
        )
        gap = event.schedule.target_layer - event.schedule.source_layer
        release_layers: list[int] = []
        with self._lock:
            for ordinal, consumer_layer in enumerate(shared_consumers, start=1):
                release_layer = event.schedule.source_layer + math.ceil(
                    gap * ordinal / (len(shared_consumers) + 1)
                )
                release_layer = min(
                    event.schedule.target_layer - 1,
                    max(event.schedule.source_layer + 1, release_layer),
                )
                self._pending_prediction_stages[release_layer].append(
                    _StagedPrediction(
                        request_id=event.request_id,
                        source_layer=event.schedule.source_layer,
                        target_layer=event.schedule.target_layer,
                        group_id=event.schedule.group_id,
                        consumer_layer=int(consumer_layer),
                        blocks=blocks,
                        level=level,
                        request_token=request_token,
                        profile_operation_id=_glm_profile_operation_id(
                            event,
                            "prediction",
                            int(consumer_layer),
                        ),
                    )
                )
                release_layers.append(release_layer)
        if _timing_enabled():
            logger.info(
                "GLM_DSA_STAGED_PREDICTION request=%s source=L%d target=L%d "
                "release_layers=%s consumers=%s",
                event.request_id,
                event.schedule.source_layer,
                event.schedule.target_layer,
                release_layers,
                list(shared_consumers),
            )

    def _release_prediction_stages(self, progress_layer: int) -> None:
        if self._shared_prediction_mode != "staged":
            return
        manager = self._attention_kv_manager
        active_request_id = str(getattr(manager, "active_request_id", ""))
        with self._lock:
            release_layers = sorted(
                layer
                for layer in self._pending_prediction_stages
                if layer <= progress_layer
            )
            stages = [
                (release_layer, stage)
                for release_layer in release_layers
                for stage in self._pending_prediction_stages.pop(release_layer)
                if stage.request_id == active_request_id
            ]
        for release_layer, stage in stages:
            csa_pipeline_nvtx.mark(
                CsaNvtxEvent.GLM_DSA_STAGE_RELEASE,
                layer_id=int(release_layer),
                target_layer_id=stage.consumer_layer,
                request_id=stage.request_id,
                operation_id=stage.profile_operation_id,
                attributes={
                    "kind": "glm_dsa_prediction",
                    "source": stage.source_layer,
                    "full_target": stage.target_layer,
                    "group": stage.group_id,
                },
            )
            future = self._submit_predicted_layer(
                stage.consumer_layer,
                stage.blocks,
                stage.level,
                stage.request_token,
                profile_source_layer=stage.source_layer,
                profile_operation_id=stage.profile_operation_id,
            )
            with self._lock:
                self._staged_prediction_futures[stage.consumer_layer] = future

    def _submit_predicted_layer(
        self,
        layer_id: int,
        blocks: torch.Tensor,
        level: int,
        request_token: Any,
        *,
        event: GLMDSAPrefetchEvent | None = None,
        profile_source_layer: int | None = None,
        profile_operation_id: str | None = None,
    ) -> Future[Any]:
        manager = self._attention_kv_manager
        profile_kwargs: dict[str, object] = {}
        if csa_pipeline_nvtx.enabled:
            source_layer = (
                int(profile_source_layer)
                if profile_source_layer is not None
                else int(event.schedule.source_layer)
                if event is not None
                else int(layer_id) - int(level)
            )
            operation_id = profile_operation_id
            if operation_id is None and event is not None:
                operation_id = _glm_profile_operation_id(
                    event,
                    "prediction",
                    int(layer_id),
                )
            profile_kwargs = {
                "profile_source_layer": source_layer,
                "profile_operation_id": operation_id,
                "profile_kind": "glm_dsa_prediction",
            }
        future = self._prediction_executor.submit(
            manager.fire_predicted_reads,
            layer_id,
            blocks,
            level,
            request_token=request_token,
            **profile_kwargs,
        )
        manager.track_layer_submission(
            layer_id,
            future,
            request_token=request_token,
        )
        return future

    def _submit_after_prediction(
        self,
        prediction_future: Future[Any] | None,
        layer_id: int,
        blocks: torch.Tensor,
        request_token: Any,
        event: GLMDSAPrefetchEvent,
    ) -> None:
        if prediction_future is not None:
            prediction_future.result()
        self._attention_kv_manager.submit_miss_reads(
            layer_id,
            blocks,
            request_token=request_token,
            **_glm_io_profile_kwargs(
                event,
                "correction",
                int(layer_id),
            ),
        )

    def _drop_pending_target_locked(self, target_layer: int) -> None:
        empty_release_layers: list[int] = []
        for release_layer, stages in self._pending_prediction_stages.items():
            self._pending_prediction_stages[release_layer] = [
                stage for stage in stages if stage.target_layer != target_layer
            ]
            if not self._pending_prediction_stages[release_layer]:
                empty_release_layers.append(release_layer)
        for release_layer in empty_release_layers:
            self._pending_prediction_stages.pop(release_layer, None)

    @staticmethod
    def _gate_timeout_s() -> float:
        try:
            return max(
                0.1,
                float(os.getenv("LMCACHE_GLM_DSA_GATE_TIMEOUT_SEC", "30")),
            )
        except ValueError:
            return 30.0


def build_glm_dsa_prediction_schedule(
    topology: ModelTopologyProfile,
    *,
    bootstrap_source_layer: int = 0,
    steady_full_layer_lookahead: int = 1,
) -> tuple[GLMDSAPredictionSchedule, ...]:
    """Build the GLM-5.2 early-Full prediction schedule.

    The first predictable IndexShare group uses ``L0 -> L2`` by default. With
    one-group lookahead, the remaining groups yield ``L2 -> L6``, ``L6 ->
    L10``, and so on. Larger lookaheads ramp up without duplicating bootstrap
    work: with two groups, L6 uses L1, then the steady pattern becomes ``L2 ->
    L10``, ``L6 -> L14``, and so on.

    Args:
        topology: Validated GLM DSA topology extracted from model config.
        bootstrap_source_layer: Source for the Full group owned by layer 2.
        steady_full_layer_lookahead: Number of Full groups to predict ahead
            after the bootstrap edge.

    Returns:
        Ordered immutable prediction edges.

    Raises:
        ValueError: If the topology or requested lookahead is invalid.
    """
    if topology.model_type != "glm_moe_dsa":
        raise ValueError("prediction schedule requires a GLM DSA topology")
    if bootstrap_source_layer not in (0, 1):
        raise ValueError("bootstrap_source_layer must be 0 or 1")
    if steady_full_layer_lookahead <= 0:
        raise ValueError("steady_full_layer_lookahead must be positive")

    schedules: list[GLMDSAPredictionSchedule] = []
    layer_count = len(topology.layers)
    predictable_groups = [
        group for group in topology.index_groups if group.source_layer >= 2
    ]
    for group_index, group in enumerate(predictable_groups):
        target = group.source_layer
        if target == 2:
            source = bootstrap_source_layer
            is_bootstrap = True
        else:
            if group_index < steady_full_layer_lookahead:
                # Fill the early pipeline from consecutive decoder layers so
                # no source runs two proxy indexers. For lookahead two this is
                # L0->L2, L1->L6, L2->L10, then one prediction per Full layer.
                source = min(
                    bootstrap_source_layer + group_index,
                    target - 1,
                )
            else:
                source_index = group_index - steady_full_layer_lookahead
                source = predictable_groups[source_index].source_layer
            is_bootstrap = False
        if source < 0 or source >= layer_count:
            raise ValueError("prediction source is outside the model topology")
        schedules.append(
            GLMDSAPredictionSchedule(
                source_layer=source,
                target_layer=target,
                group_id=group.group_id,
                consumer_layers=group.consumer_layers,
                is_bootstrap=is_bootstrap,
            )
        )
    return tuple(schedules)


def post_decoder_layer_activation(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """Return the logical post-MLP residual state of a vLLM decoder layer.

    vLLM 0.20.2 returns the MLP delta and residual accumulator separately.
    Their sum is the state that the next layer's input RMSNorm observes.

    Args:
        hidden_states: Decoder layer MLP output.
        residual: Decoder residual accumulator, if the model exposes one.

    Returns:
        Combined post-layer activation without mutating either input.
    """
    if residual is None:
        return hidden_states
    if hidden_states.shape != residual.shape:
        raise ValueError("hidden_states and residual must have identical shapes")
    return hidden_states + residual


def compare_topk_rows(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    *,
    sample_rows: int = 32,
) -> TopKAgreement:
    """Compare tail top-K rows without constructing a quadratic match matrix.

    Invalid negative padding entries are excluded.  Each predicted row is
    sorted and searched with ``torch.searchsorted`` so memory scales linearly
    in sampled rows times top-K width.

    Args:
        predicted: Predicted indices shaped ``[rows, topk]``.
        actual: Authoritative indices shaped ``[rows, topk]``.
        sample_rows: Maximum number of tail query rows to compare.

    Returns:
        Integer agreement counters and derived recall/precision properties.

    Raises:
        ValueError: If tensors are not two-dimensional or sampling is invalid.
    """
    if predicted.ndim != 2 or actual.ndim != 2:
        raise ValueError("top-K tensors must be two-dimensional")
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive")
    rows = min(int(predicted.shape[0]), int(actual.shape[0]), sample_rows)
    if rows == 0:
        return TopKAgreement(0, 0, 0, 0)

    predicted_tail = predicted[-rows:].to(dtype=torch.int64)
    actual_tail = actual[-rows:].to(
        device=predicted_tail.device,
        dtype=torch.int64,
    )
    predicted_valid = predicted_tail >= 0
    actual_valid = actual_tail >= 0
    sentinel = torch.iinfo(torch.int64).max
    sorted_predicted = torch.sort(
        torch.where(predicted_valid, predicted_tail, sentinel), dim=1
    ).values
    locations = torch.searchsorted(sorted_predicted, actual_tail)
    locations = locations.clamp_max(sorted_predicted.shape[1] - 1)
    matches = (
        torch.gather(sorted_predicted, 1, locations) == actual_tail
    ) & actual_valid
    return TopKAgreement(
        sampled_rows=rows,
        intersection=int(matches.sum().item()),
        predicted=int(predicted_valid.sum().item()),
        actual=int(actual_valid.sum().item()),
    )


class GLMDSAPredictivePrefetchManager:
    """Coordinate request-scoped GLM DSA prediction and true-index correction.

    Args:
        schedules: Early prediction edges to execute.
        predictor: Version-specific read-only target-indexer proxy.
        submit_prefetch: Optional physical KV prefetch sink.  Omitting it keeps
            prediction and accuracy measurement active without issuing I/O.
        sample_rows: Tail query rows used for low-overhead agreement telemetry.
        enable_prediction: Whether source-layer predictor computation and
            speculative events are enabled. Disabled mode retains only the
            authoritative layer-wise demand path.
    """

    def __init__(
        self,
        schedules: tuple[GLMDSAPredictionSchedule, ...],
        predictor: PredictTopK,
        submit_prefetch: SubmitPrefetch | None = None,
        *,
        sample_rows: int = 32,
        enable_prediction: bool = True,
    ) -> None:
        if not schedules:
            raise ValueError("at least one prediction schedule is required")
        if sample_rows <= 0:
            raise ValueError("sample_rows must be positive")
        sources: dict[int, list[GLMDSAPredictionSchedule]] = defaultdict(list)
        targets: dict[int, GLMDSAPredictionSchedule] = {}
        for schedule in schedules:
            if schedule.target_layer in targets:
                raise ValueError("target Full layers must be unique")
            sources[schedule.source_layer].append(schedule)
            targets[schedule.target_layer] = schedule

        self.schedules = schedules
        self._predictor = predictor
        self._submit_prefetch = submit_prefetch
        self._sample_rows = sample_rows
        self._enable_prediction = bool(enable_prediction)
        self._early_dense_group_trigger = os.getenv(
            "LMCACHE_GLM_DSA_EARLY_DENSE_GROUP",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._by_source = dict(sources)
        self._by_target = targets
        self._lock = RLock()
        self._request_id: str | None = None
        self._predictions: dict[int, torch.Tensor] = {}
        self._stats: dict[int, list[int]] = {}

    def begin_request(self, request_id: str) -> None:
        """Start a single cache-hit request and discard stale predictions.

        Args:
            request_id: Non-empty request identifier.
        """
        if not request_id:
            raise ValueError("request_id must be non-empty")
        with self._lock:
            self._request_id = request_id
            self._predictions.clear()

    def end_request(self, request_id: str) -> None:
        """End a request if it still owns this manager's active generation.

        Args:
            request_id: Request identifier previously passed to begin_request.
        """
        with self._lock:
            if request_id != self._request_id:
                return
            self._request_id = None
            self._predictions.clear()

    def cancel_active_request(self) -> None:
        """Disable prediction and discard all request-scoped state."""
        with self._lock:
            self._request_id = None
            self._predictions.clear()

    def set_submit_prefetch(self, submit_prefetch: SubmitPrefetch | None) -> None:
        """Install or remove the physical prefetch event sink.

        Args:
            submit_prefetch: Request-scoped event consumer, or ``None`` to
                retain prediction/accuracy measurement without physical I/O.
        """
        with self._lock:
            self._submit_prefetch = submit_prefetch

    def after_source_layer(
        self,
        source_layer: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[int, ...]:
        """Predict every Full group triggered by a completed decoder layer.

        Args:
            source_layer: Layer whose attention and MLP have completed.
            hidden_states: Layer MLP output.
            residual: Layer residual accumulator.
            positions: Token positions for this model invocation.

        Returns:
            Target Full layer ids successfully predicted.
        """
        schedules = self._by_source.get(source_layer, ())
        if not schedules or not self._enable_prediction:
            return ()
        with self._lock:
            request_id = self._request_id
        if request_id is None:
            return ()

        # Dense-group triggering depends only on the static layer schedule.
        # Do not materialize the full post-MLP residual tensor when no proxy
        # indexer will consume it; at long prefill lengths that addition is a
        # large GPU kernel and allocation of its own.
        activation = (
            None
            if self._early_dense_group_trigger
            else post_decoder_layer_activation(hidden_states, residual)
        )
        activation_rows = (
            hidden_states.shape[0] if activation is None else activation.shape[0]
        )
        if activation_rows != positions.shape[0]:
            logger.warning(
                "GLM DSA prediction skipped at L%d: activation rows=%d, "
                "position rows=%d",
                source_layer,
                activation_rows,
                positions.shape[0],
            )
            return ()

        predicted_targets: list[int] = []
        for schedule in schedules:
            operation_id = (
                f"glm-prediction-{source_layer}-{schedule.target_layer}-"
                f"{time.monotonic_ns()}"
                if csa_pipeline_nvtx.enabled
                else None
            )
            if self._early_dense_group_trigger:
                # The physical sink reads the complete consumer group, so its
                # launch depends only on the deterministic schedule. Avoid
                # computing proxy top-K that this mode deliberately ignores.
                topk = torch.empty(
                    (0, 0),
                    dtype=torch.int64,
                    device=positions.device,
                )
            else:
                assert activation is not None
                with csa_pipeline_nvtx.range(
                    CsaNvtxEvent.GLM_DSA_PREDICTION,
                    layer_id=int(source_layer),
                    target_layer_id=schedule.target_layer,
                    request_id=request_id,
                    operation_id=operation_id,
                    attributes={"kind": "glm_dsa", "group": schedule.group_id},
                ):
                    topk = self._predictor(schedule, activation, positions)
            if topk.ndim != 2:
                raise ValueError("predictor must return a two-dimensional tensor")
            topk = topk.detach()
            with self._lock:
                if request_id != self._request_id:
                    return tuple(predicted_targets)
                self._predictions[schedule.target_layer] = topk
            self._submit(
                GLMDSAPrefetchEvent(
                    request_id=request_id,
                    schedule=schedule,
                    topk_indices=topk,
                    correction=False,
                    profile_operation_id=operation_id,
                )
            )
            predicted_targets.append(schedule.target_layer)
        return tuple(predicted_targets)

    def observe_true_topk(
        self,
        target_layer: int,
        true_topk: torch.Tensor,
    ) -> TopKAgreement | None:
        """Measure a prediction and submit authoritative correction I/O.

        Args:
            target_layer: Full-indexer layer producing the true indices.
            true_topk: Authoritative top-K indices from vLLM.

        Returns:
            Sampled agreement when a matching prediction exists, otherwise
            ``None``.
        """
        schedule = self._by_target.get(target_layer)
        if schedule is None:
            return None
        with self._lock:
            request_id = self._request_id
            predicted = self._predictions.pop(target_layer, None)
        if request_id is None or predicted is None:
            return None
        if true_topk.ndim != 2:
            raise ValueError("true top-K tensor must be two-dimensional")

        agreement = None
        if _accuracy_profile_enabled():
            agreement = compare_topk_rows(
                predicted,
                true_topk,
                sample_rows=self._sample_rows,
            )
            with self._lock:
                counters = self._stats.setdefault(target_layer, [0, 0, 0, 0, 0])
                counters[0] += 1
                counters[1] += agreement.sampled_rows
                counters[2] += agreement.intersection
                counters[3] += agreement.predicted
                counters[4] += agreement.actual
            logger.info(
                "GLM DSA prediction accuracy request=%s source=L%d target=L%d "
                "rows=%d recall=%.6f precision=%.6f",
                request_id,
                schedule.source_layer,
                target_layer,
                agreement.sampled_rows,
                agreement.recall,
                agreement.precision,
            )
        operation_id = (
            f"glm-correction-{schedule.source_layer}-{target_layer}-"
            f"{time.monotonic_ns()}"
            if csa_pipeline_nvtx.enabled
            else None
        )
        with csa_pipeline_nvtx.range(
            CsaNvtxEvent.GLM_DSA_CORRECTION,
            layer_id=int(target_layer),
            target_layer_id=int(target_layer),
            request_id=request_id,
            operation_id=operation_id,
            attributes={
                "kind": "glm_dsa",
                "source": schedule.source_layer,
                "group": schedule.group_id,
            },
        ):
            self._submit(
                GLMDSAPrefetchEvent(
                    request_id=request_id,
                    schedule=schedule,
                    topk_indices=true_topk.detach(),
                    correction=True,
                    profile_operation_id=operation_id,
                )
            )
        return agreement

    def stats_snapshot(self) -> Mapping[int, GLMDSALayerStats]:
        """Return an immutable copy of cumulative per-target telemetry."""
        with self._lock:
            return {
                target: GLMDSALayerStats(*counters)
                for target, counters in self._stats.items()
            }

    def _submit(self, event: GLMDSAPrefetchEvent) -> None:
        if self._submit_prefetch is not None:
            self._submit_prefetch(event)
