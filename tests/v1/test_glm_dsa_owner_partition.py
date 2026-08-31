# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for GLM rank-local prediction ownership."""

from __future__ import annotations

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
