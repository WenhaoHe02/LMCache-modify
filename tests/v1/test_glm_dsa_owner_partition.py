# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for GLM rank-local prediction ownership."""

from __future__ import annotations

# Standard
from concurrent.futures import Future
import threading
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.glm_dsa_predictive_prefetch import (
    GLMDSAPhysicalPrefetchSink,
    GLMDSAPredictionSchedule,
    GLMDSAPrefetchEvent,
)


@pytest.mark.parametrize("enabled", [False, True])
def test_glm_prediction_can_preserve_rank_local_ownership(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    """Forward the opt-in owner-partition policy to predicted SSD reads."""
    monkeypatch.setenv(
        "LMCACHE_GLM_DSA_OWNER_PARTITION",
        "1" if enabled else "0",
    )
    monkeypatch.setenv("LMCACHE_CSA_OWNER_BLOCKS_PER_RANK", "2")

    class FakeAttentionKVManager:
        def __init__(self) -> None:
            self.active_request_id = "request"
            self.active_request_token = ("request", 1)
            self.submissions: list[tuple[bool, list[int]]] = []
            self.futures: dict[int, Future[Any]] = {}

        def fire_predicted_reads(
            self,
            _layer_id: int,
            _blocks: torch.Tensor,
            _level: int,
            *,
            preserve_owner_partition: bool,
            **_kwargs: object,
        ) -> bool:
            self.submissions.append((preserve_owner_partition, _blocks.tolist()))
            return True

        def track_layer_submission(
            self,
            layer_id: int,
            future: Future[Any],
            **_kwargs: object,
        ) -> None:
            self.futures[layer_id] = future

        def wait_for_layer(self, layer_id: int, timeout_s: float) -> bool:
            del timeout_s
            self.futures.pop(layer_id).result()
            return True

    schedule = GLMDSAPredictionSchedule(
        source_layer=0,
        target_layer=2,
        group_id="full-2",
        consumer_layers=(2,),
        is_bootstrap=True,
    )
    manager = FakeAttentionKVManager()
    sink = GLMDSAPhysicalPrefetchSink(
        manager,
        (schedule,),
        {2: (2,)},
        compressed_block_size=64,
        io_workers=1,
    )
    sink.submit(
        GLMDSAPrefetchEvent(
            request_id="request",
            schedule=schedule,
            topk_indices=torch.tensor([[0, 64, 128]]),
            correction=False,
        )
    )

    assert sink.wait_for_consumer(2)
    sink.close()
    expected_blocks = [0, 1] if enabled else [0, 1, 2]
    assert manager.submissions == [(enabled, expected_blocks)]


@pytest.mark.parametrize("prediction_mode", ["all", "staged"])
@pytest.mark.parametrize("failed_join", [False, True])
@pytest.mark.parametrize("gpu_filter", [False, True])
def test_shared_correction_waits_only_at_its_consumer(
    monkeypatch: pytest.MonkeyPatch,
    prediction_mode: str,
    failed_join: bool,
    gpu_filter: bool,
) -> None:
    """A Full gate never joins future consumers; misses follow their gather."""
    monkeypatch.setenv("LMCACHE_GLM_DSA_OWNER_PARTITION", "1")
    monkeypatch.setenv("LMCACHE_GLM_DSA_SHARED_CORRECTION_AT_CONSUMER", "1")
    monkeypatch.setenv("LMCACHE_GLM_DSA_PREDICT_SHARED_CONSUMERS", prediction_mode)
    monkeypatch.setenv("LMCACHE_INDEXER_PROFILE_ACCURACY", "0")
    monkeypatch.setenv("LMCACHE_CSA_ATTENTION_KV_TIMING", "0")

    class Manager:
        def __init__(self) -> None:
            self.active_request_id = "request"
            self.active_request_token = ("request", 1)
            self.futures: dict[int, Future[Any]] = {}
            self.events: list[tuple[str, int]] = []
            if gpu_filter:
                self.submit_topk_miss_reads = self.submit_true_topk_misses

        def fire_predicted_reads(self, layer: int, *args: Any, **kwargs: Any) -> int:
            return layer

        def track_layer_submission(
            self, layer: int, future: Future[Any], **kwargs: Any
        ) -> None:
            self.futures[layer] = future

        def wait_for_tracked_submission(
            self, layer: int, timeout_s: float = 30.0
        ) -> bool:
            future = self.futures.pop(layer, None)
            if future is not None:
                assert future.result(timeout=timeout_s) == layer
                self.events.append(("gather", layer))
            return not (failed_join and layer == 3)

        def submit_miss_reads(self, layer: int, *args: Any, **kwargs: Any) -> None:
            assert not gpu_filter, "GPU routing must not consult a stale CPU shadow"
            assert ("gather", layer) in self.events
            self.events.append(("miss", layer))

        def submit_true_topk_misses(
            self, layer: int, topk: torch.Tensor, **kwargs: Any
        ) -> None:
            assert ("gather", layer) in self.events
            assert topk.tolist() == [[0, 64, 128]]
            self.events.append(("miss", layer))

        def wait_for_layer(self, layer: int, timeout_s: float) -> bool:
            return self.wait_for_tracked_submission(layer, timeout_s)

        def deactivate_request(self, timeout_s: float) -> bool:
            self.futures.clear()
            self.active_request_id = ""
            return True

    schedule = GLMDSAPredictionSchedule(0, 2, "group-2", (2, 3, 4, 5), True)
    manager = Manager()
    sink = GLMDSAPhysicalPrefetchSink(
        manager, (schedule,), {2: (2, 3, 4, 5)}, compressed_block_size=64
    )
    try:
        for _chunk in range(2):
            manager.events.clear()
            sink.submit(
                GLMDSAPrefetchEvent(
                    "request", schedule, torch.tensor([[0, 64]]), correction=False
                )
            )
            assert sink.wait_for_consumer(1)  # release staged shared predictions
            sink.submit(
                GLMDSAPrefetchEvent(
                    "request", schedule, torch.tensor([[0, 64, 128]]), correction=True
                )
            )
            assert manager.events == [("gather", 2), ("miss", 2)]
            assert set(manager.futures) == {3, 4, 5}
            if failed_join:
                with pytest.raises(RuntimeError, match="shared prediction failed"):
                    sink.wait_for_consumer(3)
                assert ("miss", 3) not in manager.events
                break
            for layer in (3, 4, 5):
                assert sink.wait_for_consumer(layer)
                assert manager.events[-2:] == [("gather", layer), ("miss", layer)]
            assert not manager.futures
        assert sink.finish_request("request")
        manager.events.clear()
        manager.active_request_id = "next"
        manager.active_request_token = ("next", 2)
        assert sink.wait_for_consumer(4)
        assert not manager.events  # no old correction survives teardown
    finally:
        sink.close()


def test_async_shared_correction_owns_prediction_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared gathers run FIFO off the model thread without self-joining."""
    monkeypatch.setenv("LMCACHE_GLM_DSA_OWNER_PARTITION", "1")
    monkeypatch.setenv("LMCACHE_GLM_DSA_ASYNC_SHARED_CORRECTION", "1")
    monkeypatch.setenv("LMCACHE_GLM_DSA_PREDICT_SHARED_CONSUMERS", "all")
    release = threading.Event()
    model_thread = threading.get_ident()
    gathered: list[tuple[int, int]] = []
    corrected: list[int] = []

    class Manager:
        active_request_id = "request"
        active_request_token = ("request", 1)

        def __init__(self) -> None:
            self.futures: dict[int, Future[Any]] = {}

        def fire_predicted_reads(self, layer: int, *args: Any, **kwargs: Any) -> int:
            return layer

        def track_layer_submission(
            self, layer: int, future: Future[Any], **kwargs: Any
        ) -> None:
            self.futures[layer] = future

        def take_layer_submission(self, layer: int) -> Future[Any] | None:
            return self.futures.pop(layer, None)

        def finalize_deferred_shard_gather(self, work: Any) -> bool:
            if not isinstance(work, int):
                return False
            if work != 2:
                assert release.wait(2), "Full submission blocked on a Shared gather"
            gathered.append((work, threading.get_ident()))
            return True

        def wait_for_tracked_submission(self, layer: int, timeout_s: float = 2) -> bool:
            future = self.take_layer_submission(layer)
            if future is not None:
                self.finalize_deferred_shard_gather(future.result(timeout_s))
            return True

        def wait_for_layer(self, layer: int, timeout_s: float) -> bool:
            return self.wait_for_tracked_submission(layer, timeout_s)

        def submit_miss_reads(self, layer: int, *args: Any, **kwargs: Any) -> None:
            assert layer in [entry[0] for entry in gathered]
            corrected.append(layer)

    schedule = GLMDSAPredictionSchedule(0, 2, "group-2", (2, 3, 4, 5), True)
    sink = GLMDSAPhysicalPrefetchSink(
        Manager(), (schedule,), {2: (2, 3, 4, 5)}, compressed_block_size=64
    )
    try:
        sink.submit(
            GLMDSAPrefetchEvent(
                "request", schedule, torch.tensor([[0, 64]]), correction=False
            )
        )
        sink.submit(
            GLMDSAPrefetchEvent(
                "request", schedule, torch.tensor([[0, 64, 128]]), correction=True
            )
        )
        assert corrected == [2]
        release.set()
        for layer in (3, 4, 5):
            assert sink.wait_for_consumer(layer)
        assert corrected == [2, 3, 4, 5]
        assert gathered[0] == (2, model_thread)
        assert [layer for layer, _thread in gathered] == [2, 3, 4, 5]
        assert all(thread != model_thread for _layer, thread in gathered[1:])
    finally:
        release.set()
        sink.close()


def test_gate_aligned_owner_reads_can_prepare_four_consumers_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only read preparation runs concurrently; its futures remain per layer."""
    monkeypatch.setenv("LMCACHE_GLM_DSA_OWNER_PARTITION", "1")
    monkeypatch.setenv("LMCACHE_GLM_DSA_PREDICT_SHARED_CONSUMERS", "all")
    monkeypatch.setenv("LMCACHE_GLM_DSA_OWNER_READ_WORKERS", "4")
    arrived = threading.Barrier(4)

    class Manager:
        active_request_id = "request"
        active_request_token = ("request", 1)

        def __init__(self) -> None:
            self.futures: dict[int, Future[Any]] = {}

        def uses_gate_aligned_shard_gather(self) -> bool:
            return True

        def fire_predicted_reads(self, layer: int, *args: Any, **kwargs: Any) -> int:
            arrived.wait(timeout=3)
            return layer

        def track_layer_submission(
            self, layer: int, future: Future[Any], **kwargs: Any
        ) -> None:
            self.futures[layer] = future

    manager = Manager()
    schedule = GLMDSAPredictionSchedule(0, 2, "group-2", (2, 3, 4, 5), True)
    sink = GLMDSAPhysicalPrefetchSink(
        manager, (schedule,), {2: (2, 3, 4, 5)}, compressed_block_size=64
    )
    try:
        sink.submit(
            GLMDSAPrefetchEvent(
                "request", schedule, torch.tensor([[0, 64]]), correction=False
            )
        )
        assert {
            layer: future.result(timeout=4) for layer, future in manager.futures.items()
        } == {2: 2, 3: 3, 4: 4, 5: 5}
    finally:
        sink.close()


def test_early_shared_misses_retain_the_prediction_and_final_exact_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Full-layer miss is a prefetch hint, not proof of Shared residency."""
    monkeypatch.setenv("LMCACHE_GLM_DSA_OWNER_PARTITION", "1")
    monkeypatch.setenv("LMCACHE_GLM_DSA_PREFETCH_SHARED_MISSES", "1")
    monkeypatch.setenv("LMCACHE_GLM_DSA_SHARED_CORRECTION_AT_CONSUMER", "1")
    monkeypatch.setenv("LMCACHE_GLM_DSA_ASYNC_SHARED_CORRECTION", "0")
    monkeypatch.setenv("LMCACHE_GLM_DSA_PREDICT_SHARED_CONSUMERS", "all")
    events: list[tuple[str, int]] = []

    class Manager:
        active_request_id = "request"
        active_request_token = ("request", 1)

        def __init__(self) -> None:
            self.futures: dict[int, Future[Any]] = {}

        def fire_predicted_reads(self, layer: int, *args: Any, **kwargs: Any) -> int:
            return layer

        def track_layer_submission(
            self, layer: int, future: Future[Any], **kwargs: Any
        ) -> None:
            self.futures[layer] = future

        def prepare_topk_block_selection(
            self, layer: int, topk: torch.Tensor
        ) -> object:
            return object()

        def submit_selection_miss_reads(
            self, layer: int, selection: object, **kwargs: Any
        ) -> torch.Tensor:
            events.append(("exact", layer))
            return torch.tensor([9])

        def submit_miss_reads(
            self, layer: int, ids: torch.Tensor, **kwargs: Any
        ) -> None:
            assert ids.tolist() == [9]
            events.append(("early", layer))

        def drain_for_layer(self, layer: int) -> None:
            events.append(("early_complete", layer))

        def wait_for_tracked_submission(self, layer: int, timeout_s: float = 2) -> bool:
            future = self.futures.pop(layer, None)
            if future is not None:
                future.result(timeout=timeout_s)
                events.append(("gather", layer))
            return True

        def wait_for_layer(self, layer: int, timeout_s: float) -> bool:
            return self.wait_for_tracked_submission(layer, timeout_s)

    manager = Manager()
    schedule = GLMDSAPredictionSchedule(0, 2, "group-2", (2, 3), True)
    sink = GLMDSAPhysicalPrefetchSink(
        manager, (schedule,), {2: (2, 3)}, compressed_block_size=64
    )
    try:
        topk = torch.tensor([[0, 64]])
        sink.submit(GLMDSAPrefetchEvent("request", schedule, topk, correction=False))
        sink.submit(GLMDSAPrefetchEvent("request", schedule, topk, correction=True))
        assert 3 in manager.futures
        assert sink.wait_for_consumer(3)
        assert events.index(("early_complete", 3)) < events.index(("gather", 3))
        assert events.index(("gather", 3)) < events.index(("exact", 3))
    finally:
        sink.close()
