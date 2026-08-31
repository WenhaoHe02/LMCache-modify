# SPDX-License-Identifier: Apache-2.0
"""Public hook ordering for overlapping Full-indexer compute with KV reads."""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import torch
import pytest

# First Party
from lmcache.integration.vllm.glm_dsa_vllm_0202 import VLLM0202GLMDSAHooks
from lmcache.v1.glm_dsa_predictive_prefetch import GLMDSAPredictionSchedule


def test_full_indexer_precedes_kv_join_and_proxy_does_not_join() -> None:
    """Real scoring overlaps reads; speculative scoring has no observer side effects."""
    events: list[tuple[str, int]] = []

    class Indexer(torch.nn.Module):
        def __init__(self, layer: int) -> None:
            super().__init__()
            self.layer = layer
            self.indexer_op = SimpleNamespace(skip_k_cache_insert=False)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            events.append(("indexer", self.layer))
            return torch.zeros((hidden_states.shape[0], 2), dtype=torch.int32)

    class Decoder:
        def __init__(self, layer: int) -> None:
            self.layer_idx = layer
            self.self_attn = SimpleNamespace(indexer=Indexer(layer))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            result = self.self_attn.indexer(hidden_states)
            events.append(("attention", self.layer_idx))
            return result

    class Manager:
        schedules = (GLMDSAPredictionSchedule(0, 2, "full-2", (2,), True),)

        def wait_for_prediction(self, layer: int) -> bool:
            events.append(("prediction_join", layer))
            return True

        def observe_true_topk(self, layer: int, topk: torch.Tensor) -> None:
            events.append(("authoritative", layer))

    def wait_for_consumer(layer: int) -> bool:
        events.append(("kv_join", layer))
        return True

    layers: dict[int, Any] = {layer: Decoder(layer) for layer in range(3)}
    hooks = VLLM0202GLMDSAHooks(
        layers,
        Manager(),
        indexer_types=("full", "full", "full"),
        consumer_waiter=wait_for_consumer,
    )
    hooks.attach()
    try:
        hidden = torch.zeros((3, 4))
        layers[2].forward(hidden)
        assert events == [
            ("indexer", 2),
            ("prediction_join", 2),
            ("kv_join", 2),
            ("authoritative", 2),
            ("attention", 2),
        ]
        events.clear()
        layers[2].self_attn.indexer.indexer_op.skip_k_cache_insert = True
        layers[2].self_attn.indexer(hidden)
        assert events == [("indexer", 2)]
    finally:
        hooks.close()


def test_owner_gather_launch_is_ordered_after_source_before_next_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Early communication does not bypass the real consumer's completion gate."""
    monkeypatch.setenv("LMCACHE_GLM_DSA_PREFIRE_OWNER_GATHER", "1")
    events: list[tuple[str, int]] = []

    class Decoder:
        def __init__(self, layer: int) -> None:
            self.layer_idx = layer
            self.self_attn = SimpleNamespace(indexer=None)

        def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            events.append(("compute", self.layer_idx))
            return positions, positions

    class Manager:
        schedules = (GLMDSAPredictionSchedule(0, 2, "group", (2, 3), True),)

        def wait_for_prediction(self, layer: int) -> bool:
            events.append(("prediction_ready", layer))
            return True

        def after_source_layer(self, layer: int, *args: Any) -> None:
            events.append(("predict", layer))

    layers = {layer: Decoder(layer) for layer in range(4)}
    hooks = VLLM0202GLMDSAHooks(
        layers,
        Manager(),
        indexer_types=("full", "full", "full", "shared"),
        consumer_waiter=lambda layer: events.append(("join", layer)) or True,
        consumer_starter=lambda layer: events.append(("launch", layer)) or True,
    )
    hooks.attach()
    try:
        layers[2].forward(torch.arange(3))
        layers[3].forward(torch.arange(3))
        assert events.index(("compute", 2)) < events.index(("launch", 3))
        assert events.index(("launch", 3)) < events.index(("join", 3))
        assert events.index(("join", 3)) < events.index(("compute", 3))
    finally:
        hooks.close()
