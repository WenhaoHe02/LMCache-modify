# SPDX-License-Identifier: Apache-2.0

# Standard
from types import MethodType, SimpleNamespace
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import DiskCacheMetadata
from lmcache.v1.csa_attention_kv_prefetch_manager import (
    CSAAttentionKVChunkLoc,
    CSAAttentionKVPrefetchManager,
    build_shared_raw_lba_cache,
)


def _init_active_request_state(
    manager: CSAAttentionKVPrefetchManager,
) -> None:
    manager._active_request_id = "request-a"
    manager._request_transition_lock = threading.RLock()
    manager._request_state = threading.Condition()
    manager._active_submissions = 0
    manager._request_generation = 1
    manager._request_cleanup_failed = False
    manager._request_lifecycle = "active"


def _minimal_manager_with_partial_issue() -> CSAAttentionKVPrefetchManager:
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    _init_active_request_state(manager)
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


def _minimal_lifecycle_manager() -> CSAAttentionKVPrefetchManager:
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    _init_active_request_state(manager)
    manager._scheduled_layer_futures = {}
    manager._scheduled_layer_futures_lock = threading.Lock()
    manager._pending_raw_lba_cache = {}
    manager._layers = {}
    return manager


def test_request_transition_lock_serializes_registration() -> None:
    """A new plan cannot publish while the previous plan is draining."""
    manager = _minimal_lifecycle_manager()
    drain_entered = threading.Event()
    release_drain = threading.Event()
    registered = threading.Event()

    def _slow_deactivate(
        _self: CSAAttentionKVPrefetchManager,
        timeout_s: float = 30.0,
    ) -> bool:
        del timeout_s
        drain_entered.set()
        release_drain.wait(timeout=2.0)
        return True

    def _register(
        _self: CSAAttentionKVPrefetchManager,
        _req_id: str,
        _chunks: object,
        *,
        start_profile_capture: bool,
        shared_raw_lba_cache: object,
    ) -> None:
        del start_profile_capture, shared_raw_lba_cache
        registered.set()

    manager._deactivate_request_locked = MethodType(_slow_deactivate, manager)
    manager._register_request_chunks_locked = MethodType(_register, manager)
    drain_thread = threading.Thread(target=manager.deactivate_request)
    register_thread = threading.Thread(
        target=manager.register_request_chunks,
        args=("request-b", {}),
    )
    drain_thread.start()
    assert drain_entered.wait(timeout=1.0)
    register_thread.start()
    assert not registered.wait(timeout=0.05)
    release_drain.set()
    drain_thread.join(timeout=1.0)
    register_thread.join(timeout=1.0)
    assert registered.is_set()


def test_layer_major_registration_shares_gpu_plan_and_lba_union() -> None:
    """Layers and consumers reuse immutable destination and extent tables."""

    class _Loader:
        io_stream = None

        def __init__(self) -> None:
            self.ensured: list[dict[str, list[object]]] = []

        def ensure_lba_cache(self, records: dict[str, list[object]]) -> None:
            self.ensured.append(records)

    loader = _Loader()
    manager = CSAAttentionKVPrefetchManager(
        tutti_loader=loader,  # type: ignore[arg-type]
        csa_layer_ids=[2, 4],
        compressed_block_size=1,
        token_bytes=1,
    )
    manager.register_layer(2, torch.empty((4, 1, 1), dtype=torch.uint8))
    manager.register_layer(4, torch.empty((4, 1, 1), dtype=torch.uint8))
    physical_rows = (3, 2, 1, 0)
    disk_meta = DiskCacheMetadata(path="tutti://rank0-full", size=2048)
    chunks = {
        layer_id: [
            CSAAttentionKVChunkLoc(
                first_compressed_block=0,
                n_compressed_blocks=4,
                key=SimpleNamespace(),  # type: ignore[arg-type]
                disk_meta=disk_meta,
                layer_byte_offset=layer_id * 512,
                bytes_per_block=512,
                raw_extents=((0, 100, 8),),
                physical_block_ids=physical_rows,
                read_length=2048,
                layer_major=True,
            )
        ]
        for layer_id in (2, 4)
    }
    shared_lba = build_shared_raw_lba_cache((chunks, chunks))
    assert build_shared_raw_lba_cache((chunks, chunks)) is shared_lba

    manager.register_request_chunks(
        "request-shared-plan",
        chunks,
        shared_raw_lba_cache=shared_lba,
    )

    assert loader.ensured == [shared_lba]
    assert len(shared_lba["tutti://rank0-full"]) == 1
    assert (
        manager._layers[2].layer_major_dst_rows_table
        is manager._layers[4].layer_major_dst_rows_table
    )
    first_table = manager._layers[2].layer_major_dst_rows_table

    # A new request commonly recreates an equal tuple from its slot mapping.
    # It should reuse the persistent GPU table instead of allocating and
    # uploading an identical one in the first-hit control path.
    second_rows = tuple([3, 2, 1, 0])
    assert second_rows == physical_rows
    second_chunks = {
        layer_id: [
            CSAAttentionKVChunkLoc(
                first_compressed_block=0,
                n_compressed_blocks=4,
                key=SimpleNamespace(),  # type: ignore[arg-type]
                disk_meta=disk_meta,
                layer_byte_offset=layer_id * 512,
                bytes_per_block=512,
                raw_extents=((0, 100, 8),),
                physical_block_ids=second_rows,
                read_length=2048,
                layer_major=True,
            )
        ]
        for layer_id in (2, 4)
    }
    manager.register_request_chunks(
        "request-shared-plan-2",
        second_chunks,
        shared_raw_lba_cache=shared_lba,
    )

    assert manager._layers[2].layer_major_dst_rows_table is first_table
    assert manager._layers[4].layer_major_dst_rows_table is first_table

    identity_rows = (0, 1, 2, 3)
    identity_chunks = {
        layer_id: [
            CSAAttentionKVChunkLoc(
                first_compressed_block=0,
                n_compressed_blocks=4,
                key=SimpleNamespace(),  # type: ignore[arg-type]
                disk_meta=disk_meta,
                layer_byte_offset=layer_id * 512,
                bytes_per_block=512,
                raw_extents=((0, 100, 8),),
                physical_block_ids=identity_rows,
                read_length=2048,
                layer_major=True,
            )
        ]
        for layer_id in (2, 4)
    }
    manager.register_request_chunks(
        "request-identity-plan",
        identity_chunks,
        shared_raw_lba_cache=shared_lba,
    )

    identity_table = manager._identity_rows_by_device["cpu"]
    assert manager._layers[2].layer_major_dst_rows_table.data_ptr() == (
        identity_table.data_ptr()
    )


def test_multi_generation_layer_major_plan_compiles_indexed_tables() -> None:
    """Multiple layer objects use one indexed table instead of Python ranges."""

    class _Loader:
        io_stream = None

        def ensure_lba_cache(self, _records: dict[str, list[object]]) -> None:
            return

    manager = CSAAttentionKVPrefetchManager(
        tutti_loader=_Loader(),  # type: ignore[arg-type]
        csa_layer_ids=[2],
        compressed_block_size=1,
        token_bytes=1024,
    )
    manager.register_layer(2, torch.empty((4, 1, 1024), dtype=torch.uint8))
    disk_meta = DiskCacheMetadata(path="tutti://rank0-full", size=4096)
    chunks = {
        2: [
            CSAAttentionKVChunkLoc(
                first_compressed_block=0,
                n_compressed_blocks=2,
                key=SimpleNamespace(),  # type: ignore[arg-type]
                disk_meta=disk_meta,
                layer_byte_offset=0,
                bytes_per_block=1024,
                raw_extents=((0, 100, 1), (512, 101, 3)),
                physical_block_ids=(3, 2),
                layer_major=True,
            ),
            CSAAttentionKVChunkLoc(
                first_compressed_block=2,
                n_compressed_blocks=2,
                key=SimpleNamespace(),  # type: ignore[arg-type]
                disk_meta=disk_meta,
                layer_byte_offset=2048,
                bytes_per_block=1024,
                raw_extents=((2048, 200, 1), (2560, 201, 3)),
                physical_block_ids=(1, 0),
                layer_major=True,
            ),
        ]
    }

    manager.register_request_chunks("request-two-generations", chunks)

    state = manager._layers[2]
    assert state.layer_major_dst_rows_table is not None
    assert state.layer_major_dst_rows_table.tolist() == [3, 2, 1, 0]
    assert state.indexed_slba_table is not None
    assert state.indexed_dst_rows_table is not None
    assert state.indexed_slba_table.tolist() == [100, 102, 200, 202]
    assert state.indexed_dst_rows_table.tolist() == [3, 2, 1, 0]


def test_complete_multi_generation_read_uses_full_object_fastpath() -> None:
    """A dense multi-generation restore bypasses per-block range planning."""
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    state = SimpleNamespace(
        chunks=[
            SimpleNamespace(layer_major=True, end_compressed_block=2),
            SimpleNamespace(layer_major=True, end_compressed_block=4),
        ],
        layer_major_dst_rows_table=torch.arange(4, dtype=torch.int64),
    )
    called: list[str] = []

    def _full_read(
        _self: CSAAttentionKVPrefetchManager,
        _state: object,
        *,
        io_priority: str,
    ) -> tuple[None, list[object], torch.Tensor]:
        called.append(io_priority)
        return None, [], torch.arange(4, dtype=torch.int64)

    manager._issue_full_multi_layer_major_read = MethodType(_full_read, manager)

    _event, _objects, completed = manager._issue_reads(
        state,
        torch.arange(4, dtype=torch.int64),
        io_priority="demand",
    )

    assert called == ["demand"]
    assert completed.tolist() == [0, 1, 2, 3]


def test_deactivate_timeout_retains_unresolved_future() -> None:
    """A timed-out producer remains tracked until a later drain succeeds."""

    class _Future:
        done_now = False

        def cancel(self) -> bool:
            return False

        def result(self, timeout: float) -> None:
            del timeout
            if not self.done_now:
                raise TimeoutError

        def cancelled(self) -> bool:
            return False

        def done(self) -> bool:
            return self.done_now

    manager = _minimal_lifecycle_manager()
    future = _Future()
    manager._scheduled_layer_futures[2] = future
    assert not manager.deactivate_request(timeout_s=0.0)
    assert manager._scheduled_layer_futures[2] is future
    assert not manager.request_stream_available()

    future.done_now = True
    assert manager.deactivate_request(timeout_s=0.1)
    assert not manager._scheduled_layer_futures
    assert manager.request_stream_available()


def test_stale_request_generation_cannot_submit_reads() -> None:
    """Delayed work from an older plan cannot enter a reused request id."""
    manager = _minimal_manager_with_partial_issue()
    stale_token = manager.active_request_token
    manager._request_generation += 1

    manager.fire_predicted_reads(
        2,
        torch.tensor([0, 1]),
        request_token=stale_token,
    )

    state = manager._layers[2]
    assert not bool(torch.any(state.resident_blocks_bitmap))
    assert not bool(torch.any(state.pending_reads_bitmap))


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
    _init_active_request_state(manager)
    manager._prediction_waiter = lambda _layer_id: False
    manager._miss_ids_for_topk = MethodType(
        lambda _self, _layer_id, _topk: torch.empty(0, dtype=torch.int64),
        manager,
    )
    manager.drain_for_layer = MethodType(lambda _self, _layer_id: None, manager)
    indexer = SimpleNamespace(forward=lambda: torch.tensor([1, 2]))
    manager.patch_indexer_forward(indexer, 2)

    assert indexer.forward().tolist() == [1, 2]


def test_true_indexer_waits_for_native_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The official indexer must not read a still-streaming native cache."""
    calls: list[str] = []

    class _NativeManager:
        def wait_for_native_indexer_layer(self, layer_id: int) -> bool:
            assert layer_id == 2
            calls.append("native_gate")
            return True

        def record_csa_prediction_accuracy(
            self,
            layer_id: int,
            true_topk: torch.Tensor,
        ) -> None:
            del layer_id, true_topk

        def finish_nsys_capture_for_layer(self, layer_id: int) -> None:
            del layer_id

    monkeypatch.setattr(
        "lmcache.v1.indexer_ssd_manager.get_indexer_ssd_manager",
        lambda: _NativeManager(),
    )
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._patch_lock = threading.Lock()
    manager._patched_modules = []
    _init_active_request_state(manager)
    manager._prediction_waiter = None
    manager._miss_ids_for_topk = MethodType(
        lambda _self, _layer_id, _topk: torch.empty(0, dtype=torch.int64),
        manager,
    )
    manager.drain_for_layer = MethodType(lambda _self, _layer_id: None, manager)

    def _true_indexer() -> torch.Tensor:
        calls.append("true_indexer")
        return torch.tensor([1, 2])

    indexer = SimpleNamespace(forward=_true_indexer)
    manager.patch_indexer_forward(indexer, 2)

    assert indexer.forward().tolist() == [1, 2]
    assert calls == ["native_gate", "true_indexer"]


def test_demand_only_layer_reads_true_indexer_misses() -> None:
    """A layer without prediction must demand-read only its true miss set."""
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._patch_lock = threading.Lock()
    manager._patched_modules = []
    _init_active_request_state(manager)
    manager._prediction_waiter = None
    calls: list[tuple[str, object]] = []

    def _miss_ids(
        _self: CSAAttentionKVPrefetchManager,
        layer_id: int,
        true_topk: torch.Tensor,
    ) -> torch.Tensor:
        calls.append(("miss_filter", (layer_id, true_topk.tolist())))
        return torch.tensor([3, 7], dtype=torch.int64)

    def _wait_exact(
        _self: CSAAttentionKVPrefetchManager,
        layer_id: int,
    ) -> None:
        calls.append(("wait_exact", layer_id))

    def _submit(
        _self: CSAAttentionKVPrefetchManager,
        layer_id: int,
        miss_ids: torch.Tensor,
        *,
        request_token: tuple[str, int] | None = None,
    ) -> None:
        del request_token
        calls.append(("submit", (layer_id, miss_ids.tolist())))

    def _drain(
        _self: CSAAttentionKVPrefetchManager,
        layer_id: int,
    ) -> None:
        calls.append(("drain", layer_id))

    manager._miss_ids_for_topk = MethodType(_miss_ids, manager)
    manager._wait_for_exact_topk_chunks = MethodType(_wait_exact, manager)
    manager.submit_miss_reads = MethodType(_submit, manager)
    manager.drain_for_layer = MethodType(_drain, manager)
    indexer = SimpleNamespace(forward=lambda: torch.tensor([3, 5, 7]))

    manager.patch_indexer_forward(indexer, 2)

    assert indexer.forward().tolist() == [3, 5, 7]
    assert calls == [
        ("wait_exact", 2),
        ("miss_filter", (2, [3, 5, 7])),
        ("submit", (2, [3, 7])),
        ("drain", 2),
    ]


def test_prefill_correction_reads_only_active_topk() -> None:
    """Multi-row prefill uses the compact gather's true top-K pages."""
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._patch_lock = threading.Lock()
    manager._patched_modules = []
    _init_active_request_state(manager)
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


def test_layer_major_full_read_is_split_into_bounded_segments() -> None:
    """A full-layer demand fallback must not exceed the Tutti queue."""
    captured: dict[str, object] = {}

    class _Loader:
        def ensure_lba_cache(self, _records: object) -> None:
            return

        def load_chunks_to_hbm(
            self,
            keys: list[object],
            disk_metas: list[object],
            **kwargs: object,
        ) -> list[None]:
            captured.update(keys=keys, disk_metas=disk_metas, **kwargs)
            return [None] * len(keys)

    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._tutti_loader = _Loader()
    manager._pending_raw_lba_cache = {}
    manager._scatter_stream_for = MethodType(
        lambda _self, _device: None,
        manager,
    )
    state = SimpleNamespace(
        chunks=[
            SimpleNamespace(
                key="layer-object",
                disk_meta="metadata",
                n_compressed_blocks=600,
                bytes_per_block=4096,
                read_length=600 * 4096,
                layer_byte_offset=8192,
            )
        ],
        layer_major_dst_rows_table=torch.arange(600, dtype=torch.int64),
        block_slot_scatter=False,
        k_cache_tensor=torch.zeros((600, 4096), dtype=torch.uint8),
        in_pool_bitmap=torch.zeros(600, dtype=torch.bool),
    )

    _, _, completed = manager._issue_layer_major_read(
        state,
        torch.arange(600, dtype=torch.int64),
        io_priority="demand",
    )

    assert completed.numel() == 0
    assert captured["keys"] == ["layer-object"] * 3
    assert captured["disk_metas"] == ["metadata"] * 3
    ranges = captured["read_ranges_per_key"]
    assert isinstance(ranges, list)
    assert [entry[0].length for entry in ranges] == [
        256 * 4096,
        256 * 4096,
        88 * 4096,
    ]
    assert [entry[0].offset for entry in ranges] == [
        8192,
        8192 + 256 * 4096,
        8192 + 512 * 4096,
    ]
    assert captured["max_batch_ios"] == 256
    assert captured["max_batch_bytes"] == 128 * 1024**2
