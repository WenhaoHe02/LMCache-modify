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
