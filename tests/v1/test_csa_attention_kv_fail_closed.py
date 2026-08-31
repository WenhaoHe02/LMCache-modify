# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
import threading
from typing import Any

# Third Party
import pytest
import torch

# First Party
import lmcache.v1.csa_attention_kv_prefetch_manager as csa_manager
from lmcache.utils import DiskCacheMetadata
from lmcache.v1.csa_attention_kv_prefetch_manager import (
    CSAAttentionKVChunkLoc,
    CSAAttentionKVPrefetchManager,
    build_shared_raw_lba_cache,
)
from lmcache.v1.csa_pipeline_nvtx import csa_pipeline_nvtx


def test_dsa_native_indexer_load_has_a_distinct_timeline_kind() -> None:
    assert csa_manager._io_profile_kind("indexer_native_stream") == "dsa_indexer"
    assert csa_manager._io_profile_kind("predicted_l2") == "csa_predicted"
    assert csa_manager._io_profile_kind("hca_deterministic") == "hca_deterministic"


def _init_active_request_state(
    manager: CSAAttentionKVPrefetchManager,
) -> None:
    manager._active_request_id = "request-a"
    manager._external_kv_forward_active = True
    manager._kv_forward_phase = "prefill"
    manager._decode_step = 0
    manager._request_transition_lock = threading.RLock()
    manager._request_state = threading.Condition()
    manager._active_submissions = 0
    manager._request_generation = 1
    manager._request_cleanup_failed = False
    manager._request_lifecycle = "active"


def _minimal_manager_with_partial_issue() -> CSAAttentionKVPrefetchManager:
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    _init_active_request_state(manager)
    manager._shard_transport = None
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


@pytest.mark.parametrize("indexed_enabled", [False, True])
def test_single_layer_major_plan_stays_on_coalesced_path(
    monkeypatch: pytest.MonkeyPatch,
    indexed_enabled: bool,
) -> None:
    """Indexed single-object tables are opt-in and reused for the same map."""
    monkeypatch.setenv(
        "LMCACHE_CSA_LAYER_MAJOR_INDEXED_SPARSE", "1" if indexed_enabled else "0"
    )

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
                n_compressed_blocks=4,
                key=SimpleNamespace(),  # type: ignore[arg-type]
                disk_meta=disk_meta,
                layer_byte_offset=0,
                bytes_per_block=1024,
                raw_extents=((0, 100, 8),),
                physical_block_ids=(3, 2, 1, 0),
                layer_major=True,
            )
        ]
    }

    manager.register_request_chunks("request-one-layer-object", chunks)

    state = manager._layers[2]
    if indexed_enabled:
        assert state.indexed_slba_table.tolist() == [100, 102, 104, 106]
        assert state.indexed_dst_rows_table.tolist() == [3, 2, 1, 0]
        first_slbas = state.indexed_slba_table
        manager.register_request_chunks("request-same-map", chunks)
        assert state.indexed_slba_table is first_slbas
    else:
        assert state.indexed_slba_table is None
        assert state.indexed_dst_rows_table is None
    assert state.layer_major_dst_rows_table.tolist() == [3, 2, 1, 0]


def test_single_layer_major_plan_tracks_new_destination_rows() -> None:
    """The coalesced path follows every request's destination allocation."""

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
    manager.register_layer(2, torch.empty((2, 1, 1024), dtype=torch.uint8))
    disk_meta = DiskCacheMetadata(path="tutti://rank0-full", size=2048)

    def _chunks(rows: tuple[int, int]) -> dict[int, list[CSAAttentionKVChunkLoc]]:
        return {
            2: [
                CSAAttentionKVChunkLoc(
                    first_compressed_block=0,
                    n_compressed_blocks=2,
                    key=SimpleNamespace(),  # type: ignore[arg-type]
                    disk_meta=disk_meta,
                    layer_byte_offset=0,
                    bytes_per_block=1024,
                    raw_extents=((0, 100, 4),),
                    physical_block_ids=rows,
                    layer_major=True,
                )
            ]
        }

    manager.register_request_chunks("request-plan-first", _chunks((1, 0)))
    first_slbas = manager._layers[2].indexed_slba_table
    first_rows = manager._layers[2].indexed_dst_rows_table
    manager.register_request_chunks("request-plan-second", _chunks((0, 1)))

    assert first_slbas is None
    assert first_rows is None
    assert manager._layers[2].indexed_slba_table is None
    assert manager._layers[2].indexed_dst_rows_table is None
    assert manager._layers[2].layer_major_dst_rows_table.tolist() == [0, 1]


def test_single_layer_major_plan_crossing_extent_uses_general_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A block crossing an extent cannot use the one-command indexed op."""
    monkeypatch.setenv("LMCACHE_CSA_LAYER_MAJOR_INDEXED_SPARSE", "1")

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
    manager.register_layer(2, torch.empty((2, 1, 1024), dtype=torch.uint8))
    disk_meta = DiskCacheMetadata(path="tutti://rank0-full", size=2048)
    chunks = {
        2: [
            CSAAttentionKVChunkLoc(
                first_compressed_block=0,
                n_compressed_blocks=2,
                key=SimpleNamespace(),  # type: ignore[arg-type]
                disk_meta=disk_meta,
                layer_byte_offset=0,
                bytes_per_block=1024,
                raw_extents=((0, 100, 1), (512, 200, 3)),
                physical_block_ids=(0, 1),
                layer_major=True,
            )
        ]
    }

    manager.register_request_chunks("request-crossing-extent", chunks)

    state = manager._layers[2]
    assert state.indexed_slba_table is None
    assert state.indexed_dst_rows_table is None
    assert state.layer_major_dst_rows_table is not None


def test_single_layer_major_sparse_read_prefers_coalesced_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-object plan preserves coalescing for every I/O priority."""
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._tutti_loader = SimpleNamespace(load_indexed_chunks_to_hbm=object())
    state = SimpleNamespace(
        chunks=[SimpleNamespace(layer_major=True, end_compressed_block=4)],
        layer_major_dst_rows_table=torch.arange(4, dtype=torch.int64),
        indexed_slba_table=torch.arange(4, dtype=torch.int64),
        indexed_dst_rows_table=torch.arange(4, dtype=torch.int64),
    )
    selected = torch.tensor([1, 3], dtype=torch.int64)
    calls: list[str] = []

    def _indexed(
        _self: CSAAttentionKVPrefetchManager,
        _state: object,
        block_ids: object,
        *,
        io_priority: str,
    ) -> tuple[None, list[object], torch.Tensor]:
        assert block_ids is selected
        calls.append(f"indexed:{io_priority}")
        return None, [], selected

    def _layer_major(
        _self: CSAAttentionKVPrefetchManager,
        _state: object,
        _block_ids: object,
        *,
        io_priority: str,
    ) -> tuple[None, list[object], torch.Tensor]:
        calls.append(f"layer-major:{io_priority}")
        return None, [], selected

    manager._issue_indexed_reads = MethodType(_indexed, manager)
    manager._issue_layer_major_read = MethodType(_layer_major, manager)
    monkeypatch.setattr(
        csa_manager,
        "_csa_c_ops",
        SimpleNamespace(tutti_submit_indexed_sgl_read=object()),
    )

    _event, _objects, completed = manager._issue_reads(
        state,
        selected,
        io_priority="demand",
    )

    assert calls == ["layer-major:demand"]
    assert completed is selected

    _event, _objects, completed = manager._issue_reads(
        state,
        selected,
        io_priority="lookahead",
    )

    assert calls == ["layer-major:demand", "layer-major:lookahead"]
    assert completed is selected


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


def test_deactivate_discards_completed_unconsumed_submission() -> None:
    """A completed producer result cannot disappear during request teardown."""

    class _Future:
        def cancel(self) -> bool:
            return False

        def result(self, timeout: float) -> object:
            del timeout
            return deferred_work

        def cancelled(self) -> bool:
            return False

        def done(self) -> bool:
            return True

    manager = _minimal_lifecycle_manager()
    deferred_work = object()
    discarded: list[tuple[object, str]] = []

    def _discard(
        _self: CSAAttentionKVPrefetchManager,
        work: object,
        *,
        status: str,
    ) -> bool:
        discarded.append((work, status))
        return True

    manager.discard_deferred_shard_gather = MethodType(_discard, manager)
    manager._scheduled_layer_futures[26] = _Future()

    assert manager.deactivate_request(timeout_s=0.1)
    assert discarded == [(deferred_work, "request_deactivated")]
    assert manager._scheduled_layer_futures == {}


def test_wait_for_layer_finalizes_gate_aligned_shard_gather() -> None:
    """The consumer gate must finalize deferred work, not merely join it.

    A gate-aligned shard gather books ``pending_read_count`` when it is
    prepared and releases it only when finalized.  If ``wait_for_layer``
    joins the producer future and drops its result, the booking stays
    outstanding and the gate blocks for the full timeout on I/O that has
    already landed.
    """

    class _Future:
        def result(self, timeout: float) -> object:
            del timeout
            return deferred_work

    state = SimpleNamespace(
        layer_id=2,
        pending_reads_lock=threading.Condition(),
        pending_read_count=1,
    )
    deferred_work = csa_manager._DeferredShardGather(
        state=state,
        descriptor=SimpleNamespace(),
        selected=torch.tensor([0]),
        owned=torch.tensor([0]),
        local_ready_event=None,
        local_objects=[],
        local_complete=True,
        local_capability=True,
        pending_ids=torch.tensor([0]),
        io_range=None,  # type: ignore[arg-type]
        operation_id="prediction-1",
        request_id="request-a",
    )
    manager = _minimal_lifecycle_manager()
    manager._layers = {2: state}
    manager._scheduled_layer_futures[2] = _Future()
    finalized: list[object] = []

    def _finalize(
        _self: CSAAttentionKVPrefetchManager,
        work: object,
    ) -> bool:
        finalized.append(work)
        with state.pending_reads_lock:
            state.pending_read_count = 0
            state.pending_reads_lock.notify_all()
        return True

    manager.finalize_deferred_shard_gather = MethodType(_finalize, manager)
    manager.drain_for_layer = MethodType(lambda _self, _layer_id: None, manager)

    assert manager.wait_for_layer(2, timeout_s=0.5)
    assert finalized == [deferred_work]
    assert state.pending_read_count == 0


def test_discard_deferred_shard_work_is_idempotent(monkeypatch) -> None:
    """An unconsumed prediction releases its pending bit and staging once."""

    class _Event:
        synchronizations = 0

        def synchronize(self) -> None:
            self.synchronizations += 1

    class _Memory:
        releases = 0

        def ref_count_down(self) -> None:
            self.releases += 1

    state = SimpleNamespace(
        layer_id=26,
        pending_reads_bitmap=torch.tensor([True]),
        pending_reads_lock=threading.Condition(),
        pending_read_count=1,
    )
    event = _Event()
    memory = _Memory()
    io_range = object()
    finished: list[tuple[object, str, str | None]] = []
    monkeypatch.setattr(
        csa_pipeline_nvtx,
        "finish_io",
        lambda handle, **kwargs: finished.append(
            (handle, kwargs["status"], kwargs["request_id"])
        ),
    )
    work = csa_manager._DeferredShardGather(
        state=state,
        descriptor=SimpleNamespace(),
        selected=torch.tensor([0]),
        owned=torch.tensor([0]),
        local_ready_event=event,
        local_objects=[memory],
        local_complete=True,
        local_capability=True,
        pending_ids=torch.tensor([0]),
        io_range=io_range,  # type: ignore[arg-type]
        operation_id="prediction-1",
        request_id="request-a",
    )
    manager = object.__new__(CSAAttentionKVPrefetchManager)

    assert manager.discard_deferred_shard_gather(work)
    assert manager.discard_deferred_shard_gather(work)
    assert state.pending_read_count == 0
    assert not bool(state.pending_reads_bitmap[0])
    assert event.synchronizations == 1
    assert memory.releases == 1
    assert finished == [(io_range, "cancelled", "request-a")]


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


def test_rejected_prediction_closes_io_range_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-completion prediction cannot masquerade as in-flight I/O."""
    manager = _minimal_manager_with_partial_issue()

    def _empty_issue(
        _self: CSAAttentionKVPrefetchManager,
        _state: object,
        _ids: object,
        *,
        io_priority: str,
    ) -> tuple[None, list[object], torch.Tensor]:
        del io_priority
        return None, [], torch.empty(0, dtype=torch.int64)

    finished_statuses: list[str] = []
    manager._issue_reads = MethodType(_empty_issue, manager)
    monkeypatch.setattr(csa_pipeline_nvtx, "start_io", lambda **_kwargs: object())
    monkeypatch.setattr(
        csa_pipeline_nvtx,
        "finish_io",
        lambda _range, **kwargs: finished_statuses.append(kwargs["status"]),
    )

    manager.fire_predicted_reads(2, torch.tensor([0, 1]), prefetch_level=2)

    state = manager._layers[2]
    assert finished_statuses == ["not_submitted"]
    assert not bool(torch.any(state.resident_blocks_bitmap))
    assert not bool(torch.any(state.pending_reads_bitmap))
    assert state.pending_read_count == 0
    assert state.pending_drains == []


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


def test_shard_local_prediction_bypasses_gather_and_reads_locally() -> None:
    """Rank-local predicted reads never prepare deferred collective work."""
    manager = _minimal_manager_with_partial_issue()
    manager._shard_transport = SimpleNamespace(rank=3, world_size=8)
    manager._shard_config = SimpleNamespace(enabled=True, csa_enabled=True)
    manager._shard_prediction_path_logged = True

    def _fail_gather(
        _self: CSAAttentionKVPrefetchManager,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        raise AssertionError("rank-local prediction entered shard gather")

    manager._prepare_predicted_shard_gather = MethodType(_fail_gather, manager)

    result = manager.fire_predicted_reads(
        2,
        torch.tensor([0, 1]),
        prefetch_level=2,
        shard_local_only=True,
    )

    state = manager._layers[2]
    assert result is True
    assert state.resident_blocks_bitmap.tolist() == [True, False, False, False]
    assert not bool(torch.any(state.pending_reads_bitmap))


def test_shard_local_and_owner_partition_are_mutually_exclusive() -> None:
    """Requesting both partition overrides is a configuration error."""
    manager = _minimal_manager_with_partial_issue()

    with pytest.raises(ValueError, match="exclusive"):
        manager.fire_predicted_reads(
            2,
            torch.tensor([0]),
            prefetch_level=2,
            preserve_owner_partition=True,
            shard_local_only=True,
        )


def test_owner_gather_filters_resident_reads_and_keeps_ordering_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resident owner rows avoid SSD IDs but retain scatter-stream ordering."""
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    _init_active_request_state(manager)
    issued: list[torch.Tensor] = []
    scatter_stream = object()

    class _Transport:
        rank = 0
        world_size = 1
        healthy = True

    class _CudaTensor:
        is_cuda = True
        device = torch.device("cuda", 0)

    class _DeviceContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class _Event:
        recorded_stream: object | None = None

        def record(self, stream: object) -> None:
            self.recorded_stream = stream

    state = SimpleNamespace(
        layer_id=2,
        compressed_block_size=64,
        token_bytes=4,
        k_cache_tensor=_CudaTensor(),
        in_pool_bitmap=torch.zeros(4, dtype=torch.bool),
        chunks=[SimpleNamespace(end_compressed_block=4)],
        pending_reads_lock=threading.Condition(),
        pending_reads_bitmap=torch.zeros(4, dtype=torch.bool),
        resident_blocks_bitmap=torch.tensor([True, True, False, False]),
        pending_read_count=0,
        last_drain_event=None,
        pending_drains=[],
        block_slot_scatter=False,
        indexed_dst_rows_table=torch.arange(4, dtype=torch.int64),
        layer_major_dst_rows_table=None,
    )
    manager._layers = {2: state}
    manager._data_group = "csa"
    manager._shard_transport = _Transport()
    manager._shard_config = SimpleNamespace(
        enabled=True,
        csa_enabled=True,
        csa_replica_verified=True,
        cp_size=1,
    )
    manager._bytes_per_block = 256
    manager._shard_prediction_path_logged = False

    def _issue(
        _self: CSAAttentionKVPrefetchManager,
        _state: object,
        block_ids: torch.Tensor,
        **_kwargs: object,
    ) -> tuple[None, list[object], torch.Tensor]:
        issued.append(block_ids.clone())
        return None, [], block_ids.clone()

    manager._issue_local_reads = MethodType(_issue, manager)
    manager._scatter_stream_for = lambda _device: scatter_stream
    monkeypatch.setattr(torch.cuda, "device", lambda _device: _DeviceContext())
    monkeypatch.setattr(torch.cuda, "Event", _Event)

    work = manager.fire_predicted_reads(
        2,
        torch.tensor([0, 1]),
        request_token=("request-a", 1),
        preserve_owner_partition=True,
    )

    assert len(issued) == 1
    assert issued[0].numel() == 0
    assert work.local_complete
    assert isinstance(work.local_ready_event, _Event)
    assert work.local_ready_event.recorded_stream is scatter_stream


def test_wait_for_pending_reads_blocks_until_booking_resolves() -> None:
    """The correction barrier waits out an in-flight predicted booking."""
    manager = _minimal_manager_with_partial_issue()
    state = manager._layers[2]
    with state.pending_reads_lock:
        state.pending_reads_bitmap[1] = True
        state.pending_read_count = 1

    def _resolve_booking() -> None:
        with state.pending_reads_lock:
            state.pending_reads_bitmap[1] = False
            state.pending_read_count = 0
            state.pending_reads_lock.notify_all()

    timer = threading.Timer(0.05, _resolve_booking)
    timer.start()
    try:
        assert manager.wait_for_pending_reads(2, timeout_s=1.0)
    finally:
        timer.cancel()
    assert manager.wait_for_pending_reads(99)


def test_wait_for_pending_reads_times_out_instead_of_hanging() -> None:
    """A wedged booking produces a bounded, observable failure."""
    manager = _minimal_manager_with_partial_issue()
    state = manager._layers[2]
    with state.pending_reads_lock:
        state.pending_read_count = 1

    assert not manager.wait_for_pending_reads(2, timeout_s=0.05)


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


def test_decode_bypasses_csa_indexer_correction() -> None:
    """An inactive KV phase calls only the official indexer."""
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._patch_lock = threading.Lock()
    manager._patched_modules = []
    manager.set_external_kv_forward_active(False)
    calls: list[str] = []

    def _true_indexer() -> torch.Tensor:
        calls.append("true_indexer")
        return torch.tensor([1, 2])

    indexer = SimpleNamespace(forward=_true_indexer)
    manager.patch_indexer_forward(indexer, 2)

    assert indexer.forward().tolist() == [1, 2]
    assert calls == ["true_indexer"]


def test_decode_runs_authoritative_correction_without_prediction() -> None:
    """Decode demand-reads true top-K misses and skips prefill prediction waits."""
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._patch_lock = threading.Lock()
    manager._patched_modules = []
    _init_active_request_state(manager)
    manager.set_kv_forward_phase("decode", request_id="request-a")
    manager._prediction_waiter = lambda _layer_id: pytest.fail(
        "decode waited for a prefill prediction"
    )
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
        *,
        request_token: tuple[str, int] | None = None,
        profile_operation_id: str | None = None,
        profile_kind: str | None = None,
    ) -> None:
        calls.append(
            (
                "submit",
                (
                    layer_id,
                    miss_ids.tolist(),
                    request_token,
                    profile_operation_id,
                    profile_kind,
                ),
            )
        )

    manager._miss_ids_for_topk = MethodType(_miss_ids, manager)
    manager.submit_miss_reads = MethodType(_submit, manager)
    manager.drain_for_layer = MethodType(
        lambda _self, layer_id: calls.append(("drain", layer_id)),
        manager,
    )
    indexer = SimpleNamespace(forward=lambda: torch.tensor([3, 5, 7]))
    manager.patch_indexer_forward(indexer, 2)

    assert indexer.forward().tolist() == [3, 5, 7]
    assert calls == [
        ("miss_filter", (2, [3, 5, 7])),
        (
            "submit",
            (
                2,
                [3, 7],
                ("request-a", 1),
                "decode-1-layer-2",
                "csa_decode_miss",
            ),
        ),
        ("drain", 2),
    ]


def test_decode_phase_rejects_a_stale_request_plan() -> None:
    """Decode cannot consume another request's registered SSD plan."""
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    _init_active_request_state(manager)

    with pytest.raises(RuntimeError, match="does not match"):
        manager.set_kv_forward_phase("decode", request_id="request-b")


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

    class _DenseFuture:
        def result(self, timeout: float) -> None:
            assert timeout == 30.0
            calls.append("dense_gate")

    manager._scheduled_layer_futures = {}
    manager._scheduled_layer_futures_lock = threading.Lock()
    manager._scheduled_layer_futures[2] = _DenseFuture()

    def _true_indexer() -> torch.Tensor:
        calls.append("true_indexer")
        return torch.tensor([1, 2])

    indexer = SimpleNamespace(forward=_true_indexer)
    manager.patch_indexer_forward(indexer, 2)

    assert indexer.forward().tolist() == [1, 2]
    assert calls == ["native_gate", "dense_gate", "true_indexer"]


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


@pytest.mark.parametrize(
    "ids, expected_path",
    [
        (list(range(1, 21, 2)), "indexed"),
        (list(range(20)), "coalesced"),
        ([1, 3], "coalesced"),
    ],
)
def test_fragmented_single_object_read_keeps_exact_block_set(
    monkeypatch: pytest.MonkeyPatch,
    ids: list[int],
    expected_path: str,
) -> None:
    """Only fragmented sets use indexed submit; neither branch adds blocks."""
    monkeypatch.setenv("LMCACHE_CSA_LAYER_MAJOR_INDEXED_SPARSE", "1")
    monkeypatch.setattr(
        csa_manager,
        "_csa_c_ops",
        SimpleNamespace(tutti_submit_indexed_sgl_read=object()),
    )
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._tutti_loader = SimpleNamespace(load_indexed_chunks_to_hbm=object())
    state = SimpleNamespace(
        chunks=[SimpleNamespace(layer_major=True, end_compressed_block=32)],
        layer_major_dst_rows_table=torch.arange(32),
        indexed_slba_table=torch.arange(32),
        indexed_dst_rows_table=torch.arange(32),
    )
    selected = torch.tensor(ids, dtype=torch.int64)
    calls: list[str] = []

    def indexed(
        _self: object, _state: object, block_ids: torch.Tensor, **kwargs: Any
    ) -> tuple[None, list[object], torch.Tensor]:
        assert torch.equal(block_ids, selected)
        calls.append("indexed")
        return None, [], selected

    def coalesced(
        _self: object, _state: object, block_ids: torch.Tensor, **kwargs: Any
    ) -> tuple[None, list[object], torch.Tensor]:
        assert torch.equal(block_ids, selected)
        calls.append("coalesced")
        return None, [], selected

    manager._issue_indexed_reads = MethodType(indexed, manager)
    manager._issue_layer_major_read = MethodType(coalesced, manager)
    _event, _objects, completed = manager._issue_reads(
        state, selected, io_priority="demand"
    )
    assert calls == [expected_path]
    assert torch.equal(completed, selected)


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


@pytest.mark.parametrize("completion_order", [(0, 1, 2), (2, 0)])
@pytest.mark.parametrize("short_payload", [False, True])
@pytest.mark.parametrize("slot_scatter", [False, True])
def test_layer_major_raw_batch_preserves_rows_and_fuses_scatter(
    monkeypatch: pytest.MonkeyPatch,
    completion_order: tuple[int, ...],
    short_payload: bool,
    slot_scatter: bool,
) -> None:
    """Different run lengths share one launch, including partial completions."""
    row_bytes = 8448
    selected = torch.tensor([1, 2, 5, 6, 7, 10], dtype=torch.int64)
    logical_source = torch.arange(16, dtype=torch.uint8).repeat_interleave(row_bytes)
    staging = torch.full((131072,), 255, dtype=torch.uint8)
    offsets = [0, 32768, 65536]
    scatter_calls: list[int] = []

    def scatter(
        pointers: torch.Tensor,
        destination: torch.Tensor,
        rows: torch.Tensor,
        rows_per_object: int,
        byte_width: int,
        slots_per_block: int,
        aligned: bool,
    ) -> None:
        assert rows_per_object == 1
        assert byte_width == row_bytes and aligned
        assert slots_per_block == (4 if slot_scatter else 0)
        scatter_calls.append(int(rows.numel()))
        flat_destination = destination.reshape(16, row_bytes)
        for pointer, row in zip(pointers.tolist(), rows.tolist(), strict=True):
            offset = pointer - staging.data_ptr()
            flat_destination[row].copy_(staging[offset : offset + row_bytes])

    class Loader:
        def load_chunks_to_hbm(
            self, keys: list[Any], disk_metas: list[Any], **kwargs: Any
        ) -> list[None]:
            lengths = []
            for index, ranges in enumerate(kwargs["read_ranges_per_key"]):
                byte_range = ranges[0]
                length = byte_range.length
                lengths.append(length)
                staging[offsets[index] : offsets[index] + length].copy_(
                    logical_source[byte_range.offset : byte_range.offset + length]
                )
            completed_lengths = [lengths[index] for index in completion_order]
            if short_payload:
                completed_lengths[-1] -= 512
            kwargs["on_raw_batch_loaded"](
                0,
                list(completion_order),
                [offsets[index] for index in completion_order],
                completed_lengths,
                staging,
            )
            return [None] * len(keys)

    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        csa_manager,
        "_csa_c_ops",
        SimpleNamespace(scatter_rows_from_object_ptrs=scatter),
    )
    manager = object.__new__(CSAAttentionKVPrefetchManager)
    manager._tutti_loader = Loader()
    manager._pending_raw_lba_cache = {}
    manager._scatter_stream_for = lambda _device: SimpleNamespace(
        synchronize=lambda: None
    )
    cache = torch.full((16, row_bytes), 255, dtype=torch.uint8)
    state = SimpleNamespace(
        chunks=[
            SimpleNamespace(
                key="layer",
                disk_meta="metadata",
                n_compressed_blocks=16,
                bytes_per_block=row_bytes,
                layer_byte_offset=0,
            )
        ],
        layer_major_dst_rows_table=torch.arange(15, -1, -1, dtype=torch.int64),
        block_slot_scatter=slot_scatter,
        block_slot_size=4,
        k_cache_tensor=cache.view(4, 4, row_bytes) if slot_scatter else cache,
        in_pool_bitmap=torch.zeros(16, dtype=torch.bool),
    )
    if short_payload:
        with pytest.raises(RuntimeError, match="short payload"):
            manager._issue_layer_major_read(state, selected, io_priority="demand")
        assert not scatter_calls and not state.in_pool_bitmap.any()
        return
    _, _, completed = manager._issue_layer_major_read(
        state, selected, io_priority="demand"
    )
    selected_runs = ([1, 2], [5, 6, 7], [10])
    expected = sorted(
        block for index in completion_order for block in selected_runs[index]
    )
    assert completed.tolist() == expected
    assert scatter_calls == [len(expected)]
    assert state.in_pool_bitmap.nonzero().flatten().tolist() == expected
    for logical in range(16):
        expected_byte = logical if logical in expected else 255
        assert torch.all(cache[15 - logical] == expected_byte)


def test_layer_major_sparse_indexer_reads_are_sector_aligned() -> None:
    """Compact-indexer block runs use aligned I/O plus a payload skip."""
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
                key="indexer-layer-object",
                disk_meta="metadata",
                n_compressed_blocks=16,
                bytes_per_block=64 * 132,
                read_length=16 * 64 * 132,
                layer_byte_offset=4096,
            )
        ],
        layer_major_dst_rows_table=torch.arange(16, dtype=torch.int64),
        block_slot_scatter=False,
        k_cache_tensor=torch.zeros((16, 64 * 132), dtype=torch.uint8),
        in_pool_bitmap=torch.zeros(16, dtype=torch.bool),
    )

    manager._issue_layer_major_read(
        state,
        torch.tensor([1, 2, 5, 6, 7], dtype=torch.int64),
        io_priority="demand",
    )

    assert captured["keys"] == ["indexer-layer-object"] * 2
    ranges = captured["read_ranges_per_key"]
    assert isinstance(ranges, list)
    assert [entry[0].offset for entry in ranges] == [12288, 46080]
    assert [entry[0].length for entry in ranges] == [17408, 25600]
    assert all(entry[0].offset % 512 == 0 for entry in ranges)
    assert all(entry[0].length % 512 == 0 for entry in ranges)
