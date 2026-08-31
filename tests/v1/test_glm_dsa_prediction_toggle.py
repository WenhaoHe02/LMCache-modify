# SPDX-License-Identifier: Apache-2.0
"""Prediction toggles must preserve every authoritative physical consumer."""

# Standard
from concurrent.futures import Future
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.glm_dsa_predictive_prefetch import (
    GLMDSAPhysicalPrefetchSink,
    GLMDSAPredictionSchedule,
    GLMDSAPredictivePrefetchManager,
)


class DemandManager:
    """Physical-manager contract with tracked asynchronous submissions."""

    def __init__(self) -> None:
        self.active_request_id = ""
        self.active_request_token = ("request", 1)
        self.loaded: set[int] = set()
        self.futures: dict[int, Future[Any]] = {}

    def submit_topk_miss_reads(
        self, layer: int, topk: torch.Tensor, **kwargs: Any
    ) -> None:
        """Record an authoritative demand submission."""
        self.loaded.add(layer)

    def track_layer_submission(
        self, layer: int, future: Future[Any], **kwargs: Any
    ) -> None:
        """Retain completion until the matching consumer gate."""
        self.futures[layer] = future

    def wait_for_layer(self, layer: int, timeout_s: float) -> bool:
        """Join tracked demand before allowing attention to consume it."""
        future = self.futures.pop(layer, None)
        if future is not None:
            future.result(timeout=timeout_s)
        return True


def test_toggle_off_preserves_all_78_authoritative_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ON-to-OFF loads every group's consumers, not just L0/L1."""
    monkeypatch.setenv("LMCACHE_GLM_DSA_EARLY_DENSE_GROUP", "0")
    groups = {0: (0,), 1: (1,)}
    groups.update({layer: tuple(range(layer, layer + 4)) for layer in range(2, 78, 4)})
    schedules = tuple(
        GLMDSAPredictionSchedule(
            0 if layer == 2 else layer - 4,
            layer,
            f"full-{layer}",
            consumers,
            layer == 2,
        )
        for layer, consumers in groups.items()
        if layer >= 2
    )
    physical = DemandManager()
    sink = GLMDSAPhysicalPrefetchSink(
        physical, schedules, groups, 64, enable_prediction=True
    )
    manager = GLMDSAPredictivePrefetchManager(
        schedules, lambda *_args: pytest.fail("disabled predictor ran"), sink.submit
    )
    try:
        manager.set_prediction_enabled(False)
        assert not manager.prediction_enabled and not sink.prediction_enabled
        physical.active_request_id = "request"
        manager.begin_request("request")
        assert (
            manager.after_source_layer(0, torch.zeros((1, 1)), None, torch.zeros(1))
            == ()
        )
        for full_layer in groups:
            sink.submit_authoritative(full_layer, torch.tensor([[0, 64]]))
        for consumer in range(78):
            assert sink.wait_for_consumer(consumer)
        assert physical.loaded == set(range(78))
        manager.end_request("request")
        physical.active_request_id = ""
        manager.set_prediction_enabled(True)
        assert manager.prediction_enabled and sink.prediction_enabled
        physical.loaded.clear()
        physical.active_request_id = "next"
        sink.submit_authoritative(2, torch.tensor([[0]]))
        assert not physical.loaded  # ON scheduled group belongs to correction
    finally:
        manager.close()
        sink.close()


@pytest.mark.parametrize("active_component", ["predictor", "physical"])
def test_busy_toggle_leaves_both_flags_unchanged(active_component: str) -> None:
    """Reject a live-request toggle before mutating either component."""
    schedule = GLMDSAPredictionSchedule(0, 2, "full-2", (2,), True)
    physical = DemandManager()
    sink = GLMDSAPhysicalPrefetchSink(
        physical, (schedule,), {2: (2,)}, 64, enable_prediction=True
    )
    manager = GLMDSAPredictivePrefetchManager(
        (schedule,), lambda *_args: torch.tensor([[0]]), sink.submit
    )
    try:
        if active_component == "predictor":
            manager.begin_request("request")
        else:
            physical.active_request_id = "request"
        with pytest.raises(RuntimeError, match="cannot switch prediction"):
            manager.set_prediction_enabled(False)
        assert manager.prediction_enabled and sink.prediction_enabled
    finally:
        manager.close()
        sink.close()
