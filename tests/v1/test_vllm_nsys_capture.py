# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
import lmcache.integration.vllm.vllm_v1_adapter as adapter
from lmcache.integration.vllm.vllm_v1_adapter import (
    _FullNsysCaptureController,
)


def test_generic_full_capture_skips_warmup_and_stops_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMCACHE_NSYS_FULL_CAPTURE", "1")
    monkeypatch.setenv("LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS", "1")
    monkeypatch.setenv("LMCACHE_INDEXER_ENABLE_PREFETCH", "0")
    events: list[str] = []
    monkeypatch.setattr(
        torch.cuda.profiler,
        "start",
        lambda: events.append("start"),
    )
    monkeypatch.setattr(
        torch.cuda.profiler,
        "stop",
        lambda: events.append("stop"),
    )
    controller = _FullNsysCaptureController()

    controller.start_for_request("warmup")
    controller.start_for_request("warmup")
    assert events == []

    controller.start_for_request("target")
    controller.start_for_request("later")
    assert events == ["start"]
    assert controller.active

    controller.finish()
    controller.finish()
    assert events == ["start", "stop"]
    assert controller.complete
    assert not controller.active


def test_generic_full_capture_is_disabled_when_prefetch_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMCACHE_NSYS_FULL_CAPTURE", "1")
    monkeypatch.setenv("LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS", "0")
    monkeypatch.setenv("LMCACHE_INDEXER_ENABLE_PREFETCH", "1")
    events: list[str] = []
    monkeypatch.setattr(
        torch.cuda.profiler,
        "start",
        lambda: events.append("start"),
    )
    controller = _FullNsysCaptureController()

    controller.start_for_request("target")

    assert events == []
    assert not controller.active


def test_decoder_forward_waits_for_registered_hca_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeManager:
        hca_layer_ids = (3,)

        def __init__(self) -> None:
            self.waited: list[int] = []

        def wait_for_layer(self, layer_id: int) -> bool:
            self.waited.append(layer_id)
            return True

    class FakeDecoder:
        layer_idx = 3

        def forward(self, value: int) -> int:
            return value + 1

    manager = FakeManager()
    decoder = FakeDecoder()
    monkeypatch.setattr(adapter, "_CSA_ATTENTION_KV_PREFETCH_MANAGER", manager)

    assert adapter._install_decoder_forward_position_hook(decoder)
    assert decoder.forward(4) == 5
    assert manager.waited == [3]
