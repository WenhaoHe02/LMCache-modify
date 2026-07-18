# SPDX-License-Identifier: Apache-2.0

# Standard
from types import MethodType, SimpleNamespace
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.csa_attention_kv_prefetch_manager import (
    CSAAttentionKVPrefetchManager,
)


def _minimal_manager_with_partial_issue() -> CSAAttentionKVPrefetchManager:
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._active_request_id = "request-a"
    manager._layers = {
        2: SimpleNamespace(
            chunks=[SimpleNamespace(end_compressed_block=4)],
            in_pool_bitmap=torch.zeros(4, dtype=torch.bool),
            resident_blocks_bitmap=torch.zeros(4, dtype=torch.bool),
            pending_reads_bitmap=torch.zeros(4, dtype=torch.bool),
            pending_reads_lock=threading.Condition(),
            pending_read_count=0,
            last_drain_event=None,
            pending_drains=[],
        )
    }

    def _partial_issue(
        _self: CSAAttentionKVPrefetchManager,
        _state: object,
        _ids: object,
        *,
        io_priority: str,
    ) -> tuple[None, list[object], torch.Tensor]:
        del io_priority
        return None, [], torch.tensor([0], dtype=torch.int64)

    manager._issue_reads = MethodType(_partial_issue, manager)
    return manager


def test_demand_partial_completion_fails_closed() -> None:
    """Demand correction must not publish a partially completed read set."""
    manager = _minimal_manager_with_partial_issue()

    with pytest.raises(RuntimeError, match="failed to materialize layer 2"):
        manager.submit_miss_reads(2, torch.tensor([0, 1]))

    state = manager._layers[2]
    assert not bool(torch.any(state.resident_blocks_bitmap))
    assert not bool(torch.any(state.pending_reads_bitmap))
    assert state.pending_read_count == 0


def test_speculative_partial_completion_marks_only_completed_blocks() -> None:
    """Cancelled speculative work may complete partially but cannot lie."""
    manager = _minimal_manager_with_partial_issue()

    manager.fire_predicted_reads(2, torch.tensor([0, 1]), prefetch_level=2)

    state = manager._layers[2]
    assert state.resident_blocks_bitmap.tolist() == [True, False, False, False]
    assert not bool(torch.any(state.pending_reads_bitmap))


def test_background_prediction_updates_inference_bitmaps() -> None:
    """Async prediction may update bitmaps allocated in inference mode."""
    manager = _minimal_manager_with_partial_issue()
    state = manager._layers[2]
    with torch.inference_mode():
        state.in_pool_bitmap = torch.zeros(4, dtype=torch.bool)
        state.resident_blocks_bitmap = torch.zeros(4, dtype=torch.bool)
        state.pending_reads_bitmap = torch.zeros(4, dtype=torch.bool)

    manager.fire_predicted_reads(2, torch.tensor([0, 1]), prefetch_level=2)

    assert state.resident_blocks_bitmap.tolist() == [True, False, False, False]
    assert not bool(torch.any(state.pending_reads_bitmap))


def test_late_prediction_falls_back_to_true_topk_without_blocking() -> None:
    """A late speculative result must not block true-topK correction."""
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._patch_lock = threading.Lock()
    manager._patched_modules = []
    manager._active_request_id = "request-a"
    manager._prediction_waiter = lambda _layer_id: False
    manager._miss_ids_for_topk = MethodType(
        lambda _self, _layer_id, _topk: torch.empty(0, dtype=torch.int64),
        manager,
    )
    manager.drain_for_layer = MethodType(lambda _self, _layer_id: None, manager)
    indexer = SimpleNamespace(forward=lambda: torch.tensor([1, 2]))
    manager.patch_indexer_forward(indexer, 2)

    assert indexer.forward().tolist() == [1, 2]


def test_demand_only_layer_reads_true_indexer_misses() -> None:
    """A layer without prediction must demand-read only its true miss set."""
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._patch_lock = threading.Lock()
    manager._patched_modules = []
    manager._active_request_id = "request-a"
    manager._prediction_waiter = None
    calls: list[tuple[str, object]] = []

    def _miss_ids(
        _self: CSAAttentionKVPrefetchManager,
        layer_id: int,
        true_topk: torch.Tensor,
    ) -> torch.Tensor:
        calls.append(("miss_filter", (layer_id, true_topk.tolist())))
        return torch.tensor([3, 7], dtype=torch.int64)

    def _submit(
        _self: CSAAttentionKVPrefetchManager,
        layer_id: int,
        miss_ids: torch.Tensor,
    ) -> None:
        calls.append(("submit", (layer_id, miss_ids.tolist())))

    def _drain(
        _self: CSAAttentionKVPrefetchManager,
        layer_id: int,
    ) -> None:
        calls.append(("drain", layer_id))

    manager._miss_ids_for_topk = MethodType(_miss_ids, manager)
    manager.submit_miss_reads = MethodType(_submit, manager)
    manager.drain_for_layer = MethodType(_drain, manager)
    indexer = SimpleNamespace(forward=lambda: torch.tensor([3, 5, 7]))

    manager.patch_indexer_forward(indexer, 2)

    assert indexer.forward().tolist() == [3, 5, 7]
    assert calls == [
        ("miss_filter", (2, [3, 5, 7])),
        ("submit", (2, [3, 7])),
        ("drain", 2),
    ]


def test_correction_ignores_preallocated_topk_tail_rows() -> None:
    """Only rows represented by hidden states may drive demand I/O."""
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._patch_lock = threading.Lock()
    manager._patched_modules = []
    manager._active_request_id = "request-a"
    manager._prediction_waiter = None
    observed: list[torch.Tensor] = []

    def _miss_ids(
        _self: CSAAttentionKVPrefetchManager,
        _layer_id: int,
        active_topk: torch.Tensor,
    ) -> torch.Tensor:
        observed.append(active_topk.clone())
        return torch.empty(0, dtype=torch.int64)

    manager._miss_ids_for_topk = MethodType(_miss_ids, manager)
    manager.drain_for_layer = MethodType(lambda _self, _layer_id: None, manager)
    workspace = torch.tensor([[1, 2], [3, 4], [99, 100], [101, 102]])
    indexer = SimpleNamespace(forward=lambda _hidden: workspace)
    manager.patch_indexer_forward(indexer, 36)

    result = indexer.forward(torch.zeros((2, 8)))

    assert result is workspace
    assert len(observed) == 1
    assert observed[0].tolist() == [[1, 2], [3, 4]]
