# SPDX-License-Identifier: Apache-2.0
"""Public lifecycle tests for optional GLM private-stream prediction."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.glm_dsa_predictive_prefetch import (
    GLMDSAPredictionSchedule,
    GLMDSAPredictivePrefetchManager,
)


def _schedule() -> GLMDSAPredictionSchedule:
    return GLMDSAPredictionSchedule(0, 2, "group-2", (2,), False)


def test_async_prediction_joins_before_event_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The target join observes CUDA completion before publishing prefetch."""
    monkeypatch.setenv("LMCACHE_GLM_DSA_ASYNC_PREDICTION", "1")
    completed: list[str] = []
    submitted: list[torch.Tensor] = []

    class _Event:
        def synchronize(self) -> None:
            completed.append("cuda")

    def _async_predict(*_args: object) -> object:
        return SimpleNamespace(
            topk_indices=torch.tensor([[7, 9]], dtype=torch.int64),
            done_event=_Event(),
        )

    manager = GLMDSAPredictivePrefetchManager(
        (_schedule(),),
        lambda *_args: pytest.fail("synchronous predictor ran"),
        lambda event: submitted.append(event.topk_indices.clone()),
        async_predictor=_async_predict,
    )
    try:
        manager.begin_request("request-a")
        assert manager.after_source_layer(
            0, torch.zeros((1, 2)), None, torch.zeros(1, dtype=torch.int64)
        ) == (2,)
        assert manager.wait_for_prediction(2)
        assert completed == ["cuda"]
        assert len(submitted) == 1
        assert submitted[0].tolist() == [[7, 9]]
    finally:
        manager.close()


def test_async_prediction_gate_propagates_failure_and_keeps_all_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target joins every queued chunk and propagates async CUDA failure."""
    monkeypatch.setenv("LMCACHE_GLM_DSA_ASYNC_PREDICTION", "1")
    calls = 0

    class _Event:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def synchronize(self) -> None:
            if self.fail:
                raise RuntimeError("prediction CUDA failure")

    def _async_predict(*_args: object) -> object:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            topk_indices=torch.tensor([[calls]], dtype=torch.int64),
            done_event=_Event(calls == 2),
        )

    manager = GLMDSAPredictivePrefetchManager(
        (_schedule(),),
        lambda *_args: pytest.fail("synchronous predictor ran"),
        async_predictor=_async_predict,
    )
    manager.begin_request("request-a")
    manager.after_source_layer(0, torch.zeros((1, 2)), None, torch.zeros(1))
    manager.after_source_layer(0, torch.zeros((1, 2)), None, torch.zeros(1))

    with pytest.raises(RuntimeError, match="prediction CUDA failure"):
        manager.wait_for_prediction(2)
    manager.close()


def test_async_prediction_is_disabled_by_default() -> None:
    """Without the opt-in environment the established synchronous path runs."""
    submitted: list[torch.Tensor] = []
    manager = GLMDSAPredictivePrefetchManager(
        (_schedule(),),
        lambda *_args: torch.tensor([[3]], dtype=torch.int64),
        lambda event: submitted.append(event.topk_indices.clone()),
        async_predictor=lambda *_args: pytest.fail("async predictor ran"),
    )
    try:
        manager.begin_request("request-a")
        manager.after_source_layer(0, torch.zeros((1, 2)), None, torch.zeros(1))
        assert submitted[0].tolist() == [[3]]
    finally:
        manager.close()
