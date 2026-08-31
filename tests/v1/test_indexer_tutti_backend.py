# SPDX-License-Identifier: Apache-2.0
"""Tests for the public Tutti indexer-storage interface."""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# First Party
import lmcache.v1.indexer_ssd_manager as indexer_manager_module
from lmcache.v1.indexer_ssd_manager import (
    IndexerSSDManager,
    TuttiIndexerBlockStore,
)
from lmcache.v1.csa_prefill_cp_scorer import (
    gather_prefill_cp_local_k_rows,
    globalize_prefill_cp_topk,
    localize_prefill_cp_bounds,
    prefill_cp_contiguous_history_with_replicated_append,
    prefill_cp_contiguous_key_block_partition,
    prefill_cp_key_block_partition,
    prefill_cp_key_indices,
    prefill_cp_key_indices_with_append,
    prefill_cp_local_topk_tokens,
    prefill_cp_query_indices,
    prefill_cp_query_ranges,
    prefill_cp_reads_rank_local,
    prefill_cp_sampled_query_indices,
    prefill_cp_requires_id_exchange,
    resolve_prediction_gate_policy,
    resolve_prefill_cp_mode,
)
from lmcache.v1.indexer_tutti_backend import TuttiIndexerStorage


class FakeMemoryObj:
    """Minimal public memory object returned by the fake loader."""

    def __init__(self, payload: bytes) -> None:
        self.raw_tensor = torch.tensor(list(payload), dtype=torch.uint8)
        self.released = False

    def ref_count_down(self) -> None:
        """Record that the consumer released the staging view."""
        self.released = True


class FakeTuttiLoader:
    """Record public loader calls made by TuttiIndexerStorage."""

    def __init__(self) -> None:
        self.registered: dict[str, list[Any]] = {}
        self.calls: list[dict[str, Any]] = []
        self.results: list[Any] = []
        self._io_lock = threading.Lock()
        self.store_started = threading.Event()
        self.stored_payloads: list[bytes] = []

    def register_lba_cache(self, records: dict[str, list[Any]]) -> None:
        """Record the synthetic raw-region extent table."""
        self.registered = records

    def ensure_lba_cache(self, records: dict[str, list[Any]]) -> None:
        """Record an idempotent extent-table restore."""
        self.registered = records

    def load_chunks_to_hbm(self, *_args: Any, **kwargs: Any) -> list[Any]:
        """Record one read and return configured results."""
        self.calls.append(kwargs)
        if self.results:
            return self.results
        disk_metadatas = _args[1] if len(_args) > 1 else ()
        return [FakeMemoryObj(bytes(int(metadata.size))) for metadata in disk_metadatas]

    def store_bytes_to_raw_extents(
        self,
        payload: bytes,
        **_kwargs: Any,
    ) -> None:
        """Model the non-reentrant loader lock held by a retrieve callback."""
        self.store_started.set()
        with self._io_lock:
            self.stored_payloads.append(payload)


def make_storage(loader: FakeTuttiLoader) -> TuttiIndexerStorage:
    """Construct a one-layer raw region large enough for test reads."""
    return TuttiIndexerStorage(
        tutti_loader=loader,  # type: ignore[arg-type]
        raw_region_path="tutti://indexer-test",
        raw_region_extents=[(0, 100, 4096)],
        layer_ids=[2],
        token_bytes=4,
        max_seq_len=1024,
    )


def test_load_read_request_forwards_speculative_admission() -> None:
    """Speculative priority, cancellation, and deadline reach the loader."""
    loader = FakeTuttiLoader()
    storage = make_storage(loader)
    slot = storage.slot_for_layer(2)
    request = storage.build_read_request(slot, [0, 1])
    loader.results = [FakeMemoryObj(b"abcdefgh")]

    def keep_running() -> bool:
        return True

    results = storage.load_read_request(
        request,
        io_priority="speculative",
        should_continue=keep_running,
        deadline_monotonic=123.0,
    )

    assert results == loader.results
    assert request.disk_meta.shape == torch.Size((8,))
    assert request.disk_meta.dtype == torch.uint8
    assert loader.calls == [
        {
            "shapes_per_key": None,
            "file_offsets": [0],
            "read_ranges_per_key": [request.read_ranges],
            "io_priority": "speculative",
            "max_batch_ios": 8,
            "should_continue": keep_running,
            "deadline_monotonic": 123.0,
        }
    ]


def test_block_store_reconstructs_requested_token_order() -> None:
    """Batch reads return per-token bytes in caller order and release staging."""
    loader = FakeTuttiLoader()
    storage = make_storage(loader)
    memory_obj = FakeMemoryObj(b"aaaabbbb")
    loader.results = [memory_obj]
    store = TuttiIndexerBlockStore(storage, 2)

    result = store.read_tokens_batch([1, 0])

    assert result == [b"bbbb", b"aaaa"]
    assert memory_obj.released
    assert loader.calls[0]["io_priority"] == "demand"


def test_block_store_reports_cancelled_speculative_read() -> None:
    """An admission rejection does not fabricate indexer bytes."""
    loader = FakeTuttiLoader()
    storage = make_storage(loader)
    loader.results = [None]
    store = TuttiIndexerBlockStore(storage, 2)

    with pytest.raises(RuntimeError, match="declined"):
        store.read_tokens_batch([0], io_priority="speculative")


def test_unaligned_write_preserves_edge_sector_bytes() -> None:
    """Dense 132-byte records use RMW instead of corrupting edge sectors."""
    loader = FakeTuttiLoader()
    storage = TuttiIndexerStorage(
        tutti_loader=loader,  # type: ignore[arg-type]
        raw_region_path="tutti://indexer-test",
        raw_region_extents=[(0, 100, 4096)],
        layer_ids=[2],
        token_bytes=132,
        max_seq_len=1024,
    )
    existing = bytes([7]) * 512
    memory_obj = FakeMemoryObj(existing)
    loader.results = [memory_obj]
    slot = storage.slot_for_layer(2)
    payload = bytes([9]) * 132

    storage.write_bytes(slot, 1, payload)

    assert memory_obj.released
    assert len(loader.stored_payloads) == 1
    written = loader.stored_payloads[0]
    assert len(written) == 512
    assert written[:132] == existing[:132]
    assert written[132:264] == payload
    assert written[264:] == existing[264:]


def test_two_layer_target_runs_exactly_one_l2_prediction() -> None:
    """A two-layer target fires once and ignores a non-configured L1 call."""
    fake_csa_manager = SimpleNamespace(
        active_request_id="request-a",
        fire_predicted_reads=MagicMock(),
    )
    calls: list[int] = []
    with tempfile.TemporaryDirectory() as store_dir:
        manager = IndexerSSDManager(
            csa_layer_ids=[26],
            store_dir=store_dir,
            pool_size=8,
            token_bytes=4,
            max_seq_len=32,
            io_workers=1,
            device=torch.device("cpu"),
        )
        manager.attach_csa_attention_kv_manager(fake_csa_manager)
        manager.configure_prefetch_lookahead({26: 2})
        try:
            residual = torch.zeros((2, 8))
            positions = torch.arange(2)
            manager.fire_async_for_layer(
                26,
                residual_f=residual,
                positions=positions,
                prefetch_level=1,
            )
            fake_csa_manager.fire_predicted_reads.assert_not_called()
            with patch.object(
                manager,
                "fire_async_for_layer",
                side_effect=lambda *_args, **kwargs: calls.append(
                    int(kwargs["prefetch_level"])
                ),
            ):
                manager.fire_residual_prefetch_for_layer(
                    26,
                    residual,
                    positions,
                    lookahead=2,
                )
                manager.fire_residual_prefetch_for_layer(
                    26,
                    residual,
                    positions,
                    lookahead=1,
                )
        finally:
            manager.close()

    assert calls == [2]


def test_disabled_target_has_no_l2_or_l1_prediction() -> None:
    """Lookahead zero suppresses every speculative entry point."""
    fake_csa_manager = SimpleNamespace(active_request_id="request-a")
    with tempfile.TemporaryDirectory() as store_dir:
        manager = IndexerSSDManager(
            csa_layer_ids=[24],
            store_dir=store_dir,
            pool_size=8,
            token_bytes=4,
            max_seq_len=32,
            io_workers=1,
            device=torch.device("cpu"),
        )
        manager.attach_csa_attention_kv_manager(fake_csa_manager)
        manager.configure_prefetch_lookahead({24: 0})
        try:
            with (
                patch.object(manager, "fire_async_for_layer") as proxy_fire,
                patch.object(manager, "wait_for_seed") as wait_for_seed,
            ):
                residual = torch.zeros((2, 8))
                positions = torch.arange(2)
                manager.fire_residual_prefetch_for_layer(
                    24,
                    residual,
                    positions,
                    lookahead=2,
                )
                manager.fire_residual_prefetch_for_layer(
                    24,
                    residual,
                    positions,
                    lookahead=1,
                )
                proxy_fire.assert_not_called()
                wait_for_seed.assert_not_called()
        finally:
            manager.close()


def test_one_layer_target_runs_exactly_one_l1_prediction() -> None:
    """A one-layer target forwards one L1 fire with no L2 duplicate."""
    fake_csa_manager = SimpleNamespace(active_request_id="request-a")
    calls: list[int] = []
    with tempfile.TemporaryDirectory() as store_dir:
        manager = IndexerSSDManager(
            csa_layer_ids=[36],
            store_dir=store_dir,
            pool_size=8,
            token_bytes=4,
            max_seq_len=32,
            io_workers=1,
            device=torch.device("cpu"),
        )
        manager.attach_csa_attention_kv_manager(fake_csa_manager)
        manager.configure_prefetch_lookahead({36: 1})
        try:
            residual = torch.zeros((2, 8))
            positions = torch.arange(2)
            with patch.object(
                manager,
                "fire_async_for_layer",
                side_effect=lambda *_args, **kwargs: calls.append(
                    int(kwargs["prefetch_level"])
                ),
            ):
                manager.fire_residual_prefetch_for_layer(
                    36,
                    residual,
                    positions,
                    lookahead=1,
                )
                manager.fire_residual_prefetch_for_layer(
                    36,
                    residual,
                    positions,
                    lookahead=2,
                )
        finally:
            manager.close()

    assert calls == [1]


def test_prefill_cp_local_topk_maps_to_global_k_ids() -> None:
    """Prefill-only CP candidates map back through their selected K shard."""
    global_keys = prefill_cp_key_indices(
        96,
        rank=2,
        world_size=4,
        interleave_size=8,
        device=torch.device("cpu"),
    )
    local = torch.tensor([0, 1, 7, 8, 15, 16, -1], dtype=torch.int32)

    result = globalize_prefill_cp_topk(local, global_keys)

    assert result.dtype == local.dtype
    assert result.tolist() == [16, 17, 23, 48, 55, 80, -1]


def test_rank_local_proxy_selection_is_bounded_and_frequency_ordered() -> None:
    """Rank-local proxy output keeps only the fixed block budget."""
    block = indexer_manager_module.DEEPGEMM_PAGED_BLOCK_SIZE
    token_ids = torch.tensor(
        [
            2 * block,
            2 * block + 1,
            2 * block + 2,
            5 * block,
            5 * block + 1,
            7 * block,
            -1,
            100 * block,
        ],
        dtype=torch.int64,
    )

    selected = indexer_manager_module._select_rank_local_proxy_blocks(
        token_ids,
        cursor=8 * block,
        num_blocks=8,
        block_budget=2,
    )

    assert set(selected.tolist()) == {2, 5}
    assert selected.numel() == 2


def test_rank_local_proxy_selection_pads_empty_budget_slots() -> None:
    """A fixed-size selection uses negative IDs for unused slots."""
    selected = indexer_manager_module._select_rank_local_proxy_blocks(
        torch.tensor([1, 2], dtype=torch.int64),
        cursor=64,
        num_blocks=4,
        block_budget=3,
    )

    assert selected.numel() == 3
    assert selected.tolist().count(-1) == 2
    assert 0 in selected.tolist()


def test_proxy_consumer_uses_only_rank_local_topk_columns() -> None:
    """K-sharded consumers ignore unwritten columns in the global buffer."""
    topk = torch.full((3, 8), 777, dtype=torch.int32)
    topk[:, :2] = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)

    selected = indexer_manager_module._valid_proxy_topk_columns(
        topk,
        num_rows=2,
        runtime_info={"local_topk_tokens": 2},
    )

    assert selected.tolist() == [[1, 2], [3, 4]]


def test_weighted_predicted_block_hits_preserves_topk_frequency() -> None:
    """Weighted coverage counts repeated true entries, not only the union."""
    block = indexer_manager_module.DEEPGEMM_PAGED_BLOCK_SIZE
    entries = torch.tensor(
        [
            2 * block,
            2 * block + 1,
            2 * block + 3,
            5 * block,
            7 * block,
            -1,
            9 * block,
        ],
        dtype=torch.int64,
    )
    valid = (entries >= 0) & (entries < 8 * block)

    hits, total = indexer_manager_module._weighted_predicted_block_hits(
        entries,
        valid,
        predicted_blocks={2, 5},
        num_blocks=8,
    )

    assert hits == 4
    assert total == 5


def test_prediction_readiness_check_expires_pending_work_without_waiting() -> None:
    """The target gate drops an unfinished speculative prediction."""
    manager = object.__new__(IndexerSSDManager)
    manager._lock = threading.Lock()
    pending: Future[None] = Future()
    manager._proxy_futures = {2: [pending]}
    manager._expired_proxy_layers = set()

    assert not manager.wait_for_csa_attention_kv_prediction(2)
    assert pending.cancelled()
    assert manager._expired_proxy_layers == {2}


def test_prediction_readiness_check_joins_running_work() -> None:
    """A prediction already writing KV must finish before the target proceeds."""
    manager = object.__new__(IndexerSSDManager)
    manager._lock = threading.Lock()
    manager._prediction_gate_timeout_s = 1.0
    started = threading.Event()
    release = threading.Event()

    def _running_prediction() -> None:
        started.set()
        assert release.wait(timeout=1.0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(_running_prediction)
        assert started.wait(timeout=1.0)
        manager._proxy_futures = {2: [pending]}
        manager._expired_proxy_layers = set()
        timer = threading.Timer(0.01, release.set)
        timer.start()
        try:
            assert manager.wait_for_csa_attention_kv_prediction(2)
        finally:
            timer.cancel()

    assert pending.done()
    assert manager._expired_proxy_layers == {2}


def test_prediction_gate_drains_nested_io_and_finalizes_shard_work() -> None:
    """An aligned gate consumes I/O appended by proxy completion."""
    manager = object.__new__(IndexerSSDManager)
    manager._lock = threading.Lock()
    manager._prediction_gate_timeout_s = 1.0
    shard_work = object()
    finalized: list[object] = []
    io_future: Future[object] = Future()
    io_future.set_result(shard_work)

    proxy_future: Future[None] = Future()
    manager._proxy_futures = {2: [proxy_future]}
    manager._expired_proxy_layers = set()
    manager._last_proxy_blocks = {2: [7]}

    def _finalize(work: object | None) -> bool:
        if work is None:
            return False
        finalized.append(work)
        return True

    manager._csa_attention_kv_manager = SimpleNamespace(
        uses_gate_aligned_shard_gather=lambda: True,
        finalize_deferred_shard_gather=_finalize,
    )
    with manager._lock:
        manager._proxy_futures[2].append(io_future)
    proxy_future.set_result(None)

    assert manager.wait_for_csa_attention_kv_prediction(2)
    assert finalized == [shard_work]
    assert manager._proxy_futures[2] == []
    assert manager._expired_proxy_layers == set()


def test_gate_aligned_prediction_waits_instead_of_cancelling() -> None:
    """A running owner-shard read is joined before its gate collective."""
    manager = object.__new__(IndexerSSDManager)
    manager._lock = threading.Lock()
    manager._prediction_gate_timeout_s = 1.0
    manager._expired_proxy_layers = set()
    manager._last_proxy_blocks = {2: [7]}
    release = threading.Event()
    finalized: list[object] = []
    shard_work = object()
    manager._csa_attention_kv_manager = SimpleNamespace(
        uses_gate_aligned_shard_gather=lambda: True,
        finalize_deferred_shard_gather=lambda work: finalized.append(work),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            lambda: shard_work if release.wait(timeout=1.0) else None
        )
        manager._proxy_futures = {2: [future]}
        timer = threading.Timer(0.01, release.set)
        timer.start()
        try:
            assert manager.wait_for_csa_attention_kv_prediction(2)
        finally:
            timer.cancel()

    assert not future.cancelled()
    assert finalized == [shard_work]
    assert manager._expired_proxy_layers == set()


def test_deactivate_predictions_discards_unconsumed_shard_work() -> None:
    """Request teardown retires a completed prediction missed by its gate."""
    manager = object.__new__(IndexerSSDManager)
    manager._lock = threading.Lock()
    manager._csa_layer_ids = (2,)
    manager._proxy_futures = {2: []}
    manager._expired_proxy_layers = set()
    shard_work = object()
    future: Future[object] = Future()
    future.set_result(shard_work)
    manager._proxy_futures[2].append(future)
    discarded: list[tuple[object, str]] = []
    manager._csa_attention_kv_manager = SimpleNamespace(
        discard_deferred_shard_gather=lambda work, *, status: discarded.append(
            (work, status)
        )
    )

    assert manager.deactivate_csa_predictions(timeout_s=0.1)
    assert discarded == [(shard_work, "request_deactivated")]
    assert manager._proxy_futures[2] == []
    assert manager._expired_proxy_layers == {2}


def test_deactivate_predictions_retains_running_future_after_timeout() -> None:
    """A timed-out old prediction stays reachable for a later cleanup pass."""

    class _Future:
        done_now = False

        def cancel(self) -> bool:
            return False

        def result(self, timeout: float) -> None:
            del timeout
            if not self.done_now:
                raise TimeoutError

        def done(self) -> bool:
            return self.done_now

    manager = object.__new__(IndexerSSDManager)
    manager._lock = threading.Lock()
    manager._csa_layer_ids = (2,)
    future = _Future()
    manager._proxy_futures = {2: [future]}
    manager._expired_proxy_layers = set()
    manager._csa_attention_kv_manager = SimpleNamespace()

    assert not manager.deactivate_csa_predictions(timeout_s=0.0)
    assert manager._proxy_futures[2] == [future]

    future.done_now = True
    assert manager.deactivate_csa_predictions(timeout_s=0.1)
    assert manager._proxy_futures[2] == []


def test_gate_aligned_missing_prediction_expires_late_arrival() -> None:
    """A proxy arriving after its only TP-aligned gate cannot submit I/O."""
    manager = object.__new__(IndexerSSDManager)
    manager._lock = threading.Lock()
    manager._proxy_futures = {2: []}
    manager._expired_proxy_layers = set()
    manager._last_proxy_blocks = {2: None}
    fire_predicted = MagicMock()
    manager._csa_attention_kv_manager = SimpleNamespace(
        uses_gate_aligned_shard_gather=lambda: True,
        fire_predicted_reads=fire_predicted,
        active_request_id="request-a",
        active_request_token=("request-a", 7),
    )
    manager._csa_fired_request_id = "request-old"
    manager._csa_fired_request_generation = 6
    manager._csa_fired_levels = set()
    manager._csa_layer_ids = (2,)
    manager._expired_proxy_targets = set()
    manager._release_proxy_cpu_selection = MagicMock()
    manager._log_timing = MagicMock()
    manager._proxy_io_executor = SimpleNamespace(
        submit=MagicMock(side_effect=AssertionError("late I/O submitted"))
    )

    assert not manager.wait_for_csa_attention_kv_prediction(2)
    assert manager._expired_proxy_layers == {2}
    assert manager._expired_proxy_targets == {("request-a", 7, 2)}

    # The producer sees this request for the first time only after layer 2's
    # target gate. Its request initialization must preserve the scoped expiry.
    with manager._lock:
        manager._begin_csa_proxy_request_locked("request-a", 7)
    assert manager._expired_proxy_layers == {2}
    assert manager._expired_proxy_targets == {("request-a", 7, 2)}

    copy_done = SimpleNamespace(synchronize=lambda: None)
    manager._finish_csa_attention_kv_proxy(
        2,
        [(torch.tensor([7]), None, None, copy_done, {})],
        cursor=0,
        selected_rows=1,
        fire_start=time.perf_counter(),
        prefetch_level=2,
        request_id="request-a",
        request_token=("request-a", 7),
    )

    fire_predicted.assert_not_called()
    manager._proxy_io_executor.submit.assert_not_called()
    assert manager._last_proxy_blocks[2] is None


def test_prefill_cp_partitions_cover_k_once_and_bound_union_budget() -> None:
    """All K rows have one owner and default local quotas sum to global K."""
    partitions = [
        prefill_cp_key_indices(
            100,
            rank=rank,
            world_size=8,
            interleave_size=8,
            device=torch.device("cpu"),
        )
        for rank in range(8)
    ]
    merged = torch.cat(partitions).sort().values

    assert torch.equal(merged, torch.arange(100))
    assert prefill_cp_local_topk_tokens(2048, 8) == 256
    assert prefill_cp_local_topk_tokens(2048, 8, oversubscribe=2) == 512


def test_prefill_cp_replicated_append_shards_only_history() -> None:
    """Replicated-append K mode owns history once and append on every rank."""
    total_tokens = 38
    append_start = 26
    partitions = [
        prefill_cp_key_indices_with_append(
            total_tokens,
            append_start,
            rank,
            world_size=4,
            interleave_size=3,
            device=torch.device("cpu"),
            replicate_append=True,
        )
        for rank in range(4)
    ]

    history = torch.cat([part[part < append_start] for part in partitions])
    assert torch.equal(history.sort().values, torch.arange(append_start))
    expected_append = torch.arange(append_start, total_tokens)
    for partition in partitions:
        assert torch.equal(partition[partition >= append_start], expected_append)
        assert torch.equal(partition, partition.sort().values)


def test_prefill_cp_sharded_append_owns_every_k_row_once() -> None:
    """Fully sharded K mode partitions history and append without overlap."""
    partitions = [
        prefill_cp_key_indices_with_append(
            38,
            append_start=26,
            rank=rank,
            world_size=4,
            interleave_size=3,
            device=torch.device("cpu"),
            replicate_append=False,
        )
        for rank in range(4)
    ]

    merged = torch.cat(partitions).sort().values
    assert torch.equal(merged, torch.arange(38))


@pytest.mark.parametrize("value_dtype", [torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("selected_rows", [[0, 3, 4, 7, 8, 10], []])
@pytest.mark.parametrize("block_padding", [0, 4])
def test_prefill_cp_gathers_only_local_paged_k_rows(
    value_dtype: torch.dtype,
    selected_rows: list[int],
    block_padding: int,
) -> None:
    """Direct K gather decodes native block-planar values/scales byte-exactly."""
    storage = torch.full((3, 24 + block_padding), 255, dtype=torch.uint8)
    cache = storage[:, :24].view(3, 4, 6)
    expected_values = torch.empty((3, 4, 4), dtype=torch.uint8)
    expected_scales = torch.empty((3, 4, 2), dtype=torch.uint8)
    for physical_block in range(3):
        for block_offset in range(4):
            base = physical_block * 40 + block_offset * 6
            for byte in range(4):
                storage[physical_block, block_offset * 4 + byte] = base + byte
                expected_values[physical_block, block_offset, byte] = base + byte
            for byte in range(2):
                storage[physical_block, 16 + block_offset * 2 + byte] = base + 4 + byte
                expected_scales[physical_block, block_offset, byte] = base + 4 + byte
    block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    selected = torch.tensor(selected_rows, dtype=torch.int64)

    values, scales = gather_prefill_cp_local_k_rows(
        cache,
        block_table,
        selected,
        value_bytes=4,
        scale_bytes=2,
        value_dtype=value_dtype,
    )

    logical_blocks = selected // 4
    offsets = selected % 4
    physical_blocks = block_table[0].index_select(0, logical_blocks)
    assert values.dtype == value_dtype
    assert torch.equal(
        values.view(torch.uint8), expected_values[physical_blocks.long(), offsets]
    )
    assert torch.equal(scales, expected_scales[physical_blocks.long(), offsets])
    assert values.shape[0] == selected.numel()
    assert values.is_contiguous() and scales.is_contiguous()


def test_prefill_cp_block_partition_covers_k_once() -> None:
    """Native block gathers partition every valid K row exactly once."""
    partitions = [
        prefill_cp_key_block_partition(
            total_tokens=38,
            rank=rank,
            world_size=4,
            interleave_blocks=1,
            block_size=4,
            device=torch.device("cpu"),
        )
        for rank in range(4)
    ]

    logical_blocks = torch.cat([partition[0] for partition in partitions])
    global_rows = torch.cat([partition[1] for partition in partitions])
    assert torch.equal(logical_blocks.sort().values, torch.arange(10))
    assert torch.equal(global_rows.sort().values, torch.arange(38))


def test_prefill_cp_contiguous_block_partition_covers_k_once() -> None:
    """Contiguous K shards cover every block and token exactly once."""
    partitions = [
        prefill_cp_contiguous_key_block_partition(
            total_tokens=38,
            rank=rank,
            world_size=4,
            block_size=4,
            device=torch.device("cpu"),
        )
        for rank in range(4)
    ]

    logical_blocks = torch.cat([partition[0] for partition in partitions])
    global_rows = torch.cat([partition[1] for partition in partitions])
    assert torch.equal(logical_blocks, torch.arange(10))
    assert torch.equal(global_rows, torch.arange(38))
    for logical, _rows in partitions:
        if logical.numel() > 1:
            deltas = logical[1:] - logical[:-1]
            assert torch.equal(deltas, torch.ones_like(deltas))


def test_prefill_cp_contiguous_history_replicates_only_append() -> None:
    """History is disjoint while every rank receives all append rows."""
    partitions = [
        prefill_cp_contiguous_history_with_replicated_append(
            total_tokens=48,
            append_start=32,
            rank=rank,
            world_size=4,
            block_size=4,
            device=torch.device("cpu"),
        )
        for rank in range(4)
    ]

    history_rows = torch.cat([rows[rows < 32] for _blocks, rows in partitions])
    assert torch.equal(history_rows.sort().values, torch.arange(32))
    for _blocks, rows in partitions:
        assert torch.equal(rows[rows >= 32], torch.arange(32, 48))


def test_prefill_cp_localizes_causal_bounds_for_compact_k() -> None:
    """Global causal intervals map to the correct compact K offsets."""
    local_keys = torch.tensor([0, 2, 4, 6, 7, 8], dtype=torch.int64)
    starts, ends = localize_prefill_cp_bounds(
        local_keys,
        torch.tensor([0, 3, 7], dtype=torch.int32),
        torch.tensor([5, 7, 9], dtype=torch.int32),
    )

    assert starts.tolist() == [0, 2, 4]
    assert ends.tolist() == [3, 4, 6]
    assert starts.dtype == torch.int32
    assert ends.dtype == torch.int32


@pytest.mark.parametrize(
    ("mode", "needs_exchange"),
    [
        ("query", True),
        ("key_history_only", False),
        ("key_sharded_early", False),
        ("key_sharded_union", True),
        ("key_sharded_local", True),
        ("key_contiguous_union", True),
        ("key_contiguous_replicated_append", True),
        ("key_sharded_owner", False),
        ("key_contiguous_owner", False),
        ("key_replicated_append", False),
        ("key_sharded_append", False),
    ],
)
def test_prefill_cp_mode_controls_candidate_exchange(
    mode: str,
    needs_exchange: bool,
) -> None:
    """Only explicitly unioned modes perform a candidate collective."""
    assert resolve_prefill_cp_mode(mode) == mode
    assert prefill_cp_requires_id_exchange(mode) is needs_exchange


def test_prefill_cp_mode_rejects_unknown_value() -> None:
    """A misspelled opt-in mode fails instead of changing prediction silently."""
    with pytest.raises(ValueError, match="unsupported speculative CP mode"):
        resolve_prefill_cp_mode("key_magic")


def test_owner_candidate_budget_matches_transport_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner reads never exceed the per-rank gather slot capacity."""
    monkeypatch.setenv("LMCACHE_CSA_OWNER_BLOCKS_PER_RANK", "256")

    assert (
        indexer_manager_module._proxy_candidate_block_budget(
            2048,
            "key_sharded_owner",
        )
        == 256
    )
    assert (
        indexer_manager_module._proxy_candidate_block_budget(
            2048,
            "key_contiguous_owner",
        )
        == 256
    )
    assert indexer_manager_module._proxy_candidate_block_budget(2048, "query") == 2048


def test_prefill_cp_mode_defaults_to_existing_query_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaving the new option unset preserves the deployed query partition."""
    monkeypatch.delenv("LMCACHE_CSA_PREFETCH_CP_MODE", raising=False)

    assert resolve_prefill_cp_mode() == "query"
    assert prefill_cp_requires_id_exchange() is True


def test_rank_local_mode_splits_budget_and_defaults_to_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local mode shares one global budget and never joins at the gate."""
    monkeypatch.setenv("LMCACHE_CSA_PREFETCH_CP_MODE", "key_sharded_local")
    monkeypatch.delenv("LMCACHE_CSA_PREDICTION_GATE", raising=False)

    assert prefill_cp_reads_rank_local() is True
    assert prefill_cp_requires_id_exchange() is True
    assert resolve_prediction_gate_policy() == "fail_open"
    assert (
        indexer_manager_module._proxy_candidate_block_budget(
            2048,
            "key_sharded_local",
            8,
        )
        == 256
    )
    assert (
        indexer_manager_module._proxy_candidate_block_budget(
            2048,
            "key_sharded_local",
            1,
        )
        == 2048
    )


def test_gate_policy_defaults_to_join_and_rejects_unknown_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing partition modes keep the blocking gate unless opted in."""
    monkeypatch.delenv("LMCACHE_CSA_PREFETCH_CP_MODE", raising=False)
    monkeypatch.delenv("LMCACHE_CSA_PREDICTION_GATE", raising=False)

    assert resolve_prediction_gate_policy() == "join"
    assert resolve_prediction_gate_policy("fail_open") == "fail_open"
    with pytest.raises(ValueError, match="unsupported prediction gate policy"):
        resolve_prediction_gate_policy("open_sesame")


def test_fail_open_gate_skips_running_prediction_without_cancelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fail-open gate returns immediately and keeps running work tracked."""
    monkeypatch.setenv("LMCACHE_CSA_PREFETCH_CP_MODE", "key_sharded_local")
    monkeypatch.setenv("LMCACHE_CSA_PREDICTION_GATE", "fail_open")
    manager = object.__new__(IndexerSSDManager)
    manager._lock = threading.Lock()
    manager._prediction_gate_timeout_s = 1.0
    manager._expired_proxy_layers = set()
    manager._expired_proxy_targets = set()
    manager._last_proxy_blocks = {2: [7]}
    manager._csa_attention_kv_manager = SimpleNamespace(
        uses_gate_aligned_shard_gather=lambda: True,
        active_request_id="request-a",
        active_request_token=("request-a", 1),
    )
    manager._csa_fired_request_id = "request-a"
    manager._csa_fired_request_generation = 1
    release = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as executor:
        started = threading.Event()

        def _running_prediction() -> None:
            started.set()
            assert release.wait(timeout=2.0)

        pending = executor.submit(_running_prediction)
        assert started.wait(timeout=1.0)
        manager._proxy_futures = {2: [pending]}

        gate_start = time.perf_counter()
        assert not manager.wait_for_csa_attention_kv_prediction(2)
        gate_ms = (time.perf_counter() - gate_start) * 1000.0
        release.set()

    # The gate must not have blocked on the still-running prediction, must
    # close the submission window, and must keep the future tracked so
    # request teardown can join it.
    assert gate_ms < 500.0
    assert not pending.cancelled()
    assert manager._expired_proxy_layers == {2}
    assert manager._proxy_futures[2] == [pending]


def test_fail_open_gate_consumes_completed_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed rank-local reads are still acknowledged as ready."""
    monkeypatch.setenv("LMCACHE_CSA_PREFETCH_CP_MODE", "key_sharded_local")
    monkeypatch.setenv("LMCACHE_CSA_PREDICTION_GATE", "fail_open")
    manager = object.__new__(IndexerSSDManager)
    manager._lock = threading.Lock()
    manager._prediction_gate_timeout_s = 1.0
    manager._expired_proxy_layers = set()
    manager._expired_proxy_targets = set()
    manager._last_proxy_blocks = {2: [7]}
    manager._csa_attention_kv_manager = SimpleNamespace(
        uses_gate_aligned_shard_gather=lambda: True,
        active_request_id="request-a",
        active_request_token=("request-a", 1),
    )
    manager._csa_fired_request_id = "request-a"
    manager._csa_fired_request_generation = 1
    done: Future[bool] = Future()
    done.set_result(True)
    manager._proxy_futures = {2: [done]}

    assert manager.wait_for_csa_attention_kv_prediction(2)
    assert manager._proxy_futures[2] == []
    assert manager._expired_proxy_layers == set()


def test_prefill_cp_k_shards_share_the_same_sampled_queries() -> None:
    """K-sharded ranks retain the configured Q sampling reduction."""
    sampled = prefill_cp_sampled_query_indices(
        100,
        113,
        sample_stride=4,
        device=torch.device("cpu"),
    )

    assert sampled.tolist() == [100, 104, 108, 112]


def test_prefill_cp_history_only_never_scores_resident_append() -> None:
    """History-only K shards cover cached rows once and exclude append rows."""
    history_tokens = 26
    total_tokens = 38
    partitions = [
        prefill_cp_key_indices(
            history_tokens,
            rank=rank,
            world_size=4,
            interleave_size=3,
            device=torch.device("cpu"),
        )
        for rank in range(4)
    ]

    merged = torch.cat(partitions).sort().values
    assert torch.equal(merged, torch.arange(history_tokens))
    assert all(torch.all(partition < total_tokens - 12) for partition in partitions)


def test_prefill_cp_append_boundary_is_validated() -> None:
    """K partitioning rejects append offsets outside the logical K range."""
    with pytest.raises(ValueError, match="append_start"):
        prefill_cp_key_indices_with_append(
            8,
            append_start=9,
            rank=0,
            world_size=2,
            interleave_size=1,
            device=torch.device("cpu"),
            replicate_append=True,
        )


def test_prefill_cp_query_partitions_cover_each_row_once() -> None:
    """Query-sharded proxy scoring assigns every row to one rank."""
    partitions = [
        prefill_cp_query_indices(
            13,
            113,
            rank=rank,
            world_size=8,
            interleave_size=7,
            device=torch.device("cpu"),
        )
        for rank in range(8)
    ]

    merged = torch.cat(partitions).sort().values

    assert torch.equal(merged, torch.arange(13, 113))


def test_prefill_cp_query_sampling_is_balanced_and_disjoint() -> None:
    """Sampled query shards cover a balanced subset without duplication."""
    partitions = [
        prefill_cp_query_indices(
            0,
            1024,
            rank=rank,
            world_size=8,
            interleave_size=16,
            device=torch.device("cpu"),
            sample_stride=2,
        )
        for rank in range(8)
    ]

    merged = torch.cat(partitions).sort().values

    assert merged.numel() == 512
    assert torch.unique(merged).numel() == merged.numel()
    assert {partition.numel() for partition in partitions} == {64}


def test_prefill_cp_query_ranges_use_rank_local_logits_budget() -> None:
    """A one-eighth K shard coalesces sixteen official chunks into two."""
    official_chunk_tokens = 1082
    total_query_tokens = 16 * official_chunk_tokens
    local_k_tokens = 16384
    max_logits_bytes = 8 * official_chunk_tokens * local_k_tokens * 4

    ranges = prefill_cp_query_ranges(
        0,
        total_query_tokens,
        local_k_tokens,
        max_logits_bytes,
    )

    assert ranges == [
        (0, 8 * official_chunk_tokens),
        (8 * official_chunk_tokens, total_query_tokens),
    ]


def test_prefill_cp_query_ranges_preserve_offset_and_remainder() -> None:
    """CP query re-chunking covers a nonzero-offset range without gaps."""
    ranges = prefill_cp_query_ranges(
        7,
        18,
        local_k_tokens=10,
        max_logits_bytes=4 * 10 * 4,
    )

    assert ranges == [(7, 11), (11, 15), (15, 18)]


@pytest.mark.parametrize(
    ("local_k_tokens", "max_logits_bytes"),
    [(0, 1024), (1024, 0), (1024, 1024)],
)
def test_prefill_cp_query_ranges_reject_invalid_budget(
    local_k_tokens: int,
    max_logits_bytes: int,
) -> None:
    """Invalid or sub-row CP logits budgets fail instead of overallocating."""
    with pytest.raises(ValueError):
        prefill_cp_query_ranges(
            0,
            1,
            local_k_tokens=local_k_tokens,
            max_logits_bytes=max_logits_bytes,
        )


def test_deterministic_hca_submission_is_exposed_to_target_gate() -> None:
    """The unified manager can join HCA I/O scheduled by a preceding FFN."""
    submitted: list[tuple[int, int | None]] = []
    tracked: dict[int, Any] = {}
    fake_csa_manager = SimpleNamespace(
        active_request_id="request-a",
        active_request_token=("request-a", 1),
        fire_deterministic_layer=lambda layer_id, **kwargs: submitted.append(
            (layer_id, kwargs.get("source_layer_id"))
        ),
        track_layer_submission=lambda layer_id, future, **_kwargs: tracked.update(
            {layer_id: future}
        ),
    )
    with tempfile.TemporaryDirectory() as store_dir:
        manager = IndexerSSDManager(
            csa_layer_ids=[4],
            store_dir=store_dir,
            pool_size=8,
            token_bytes=4,
            max_seq_len=32,
            io_workers=1,
            device=torch.device("cpu"),
        )
        manager.attach_csa_attention_kv_manager(fake_csa_manager)
        try:
            manager.fire_async_for_layers([3], source_layer_id=1)
            tracked[3].result(timeout=2.0)
        finally:
            manager.close()

    assert submitted == [(3, 1)]


def test_prefire_all_hca_layers_preserves_consumer_order() -> None:
    """Request-start HCA prefire cannot let a later layer win the I/O queue."""
    submitted: list[int] = []
    tracked: dict[int, Any] = {}
    first_started = threading.Event()
    release_first = threading.Event()

    def fire(layer_id: int, **_kwargs: Any) -> None:
        if layer_id == 3:
            first_started.set()
            assert release_first.wait(timeout=2.0)
        submitted.append(layer_id)

    fake_csa_manager = SimpleNamespace(
        active_request_id="request-a",
        active_request_token=("request-a", 1),
        fire_deterministic_layer=fire,
        track_layer_submission=lambda layer_id, future, **_kwargs: tracked.update(
            {layer_id: future}
        ),
    )
    with tempfile.TemporaryDirectory() as store_dir:
        manager = IndexerSSDManager(
            csa_layer_ids=[4],
            store_dir=store_dir,
            pool_size=8,
            token_bytes=4,
            max_seq_len=32,
            io_workers=3,
            device=torch.device("cpu"),
        )
        manager.attach_csa_attention_kv_manager(fake_csa_manager)
        try:
            with patch.dict(
                indexer_manager_module.os.environ,
                {"LMCACHE_HCA_PREFIRE_ALL_LAYERS": "1"},
            ):
                manager.fire_async_for_layers([3, 5, 7], source_layer_id=-1)
            assert first_started.wait(timeout=2.0)
            time.sleep(0.05)
            assert submitted == []
            release_first.set()
            for layer_id in (3, 5, 7):
                tracked[layer_id].result(timeout=2.0)
        finally:
            release_first.set()
            manager.close()

    assert submitted == [3, 5, 7]


def test_proxy_coverage_model_switches_only_after_union_saturates() -> None:
    """The Pro cost model distinguishes sparse and near-dense appends."""
    sparse = indexer_manager_module._estimated_proxy_block_coverage(
        196_608,
        256,
        512,
        4,
        2_048,
    )
    dense = indexer_manager_module._estimated_proxy_block_coverage(
        196_608,
        8_192,
        512,
        4,
        2_048,
    )

    assert sparse < 0.8
    assert dense > 0.99


def test_adaptive_dense_prefetch_skips_redundant_proxy() -> None:
    """A saturated predicted union uses exact dense lookahead without scoring."""
    manager = object.__new__(IndexerSSDManager)
    manager._prefetch_lookahead = {2: 2}
    manager._decode_cursor = {2: 196_608}
    manager._proxy_topk_tokens_by_layer = {}
    manager._l1_proxy_topk_tokens = 2_048
    manager.wait_for_seed = MagicMock(return_value=True)
    manager.fire_dense_csa_layers = MagicMock()
    manager.fire_async_for_layer = MagicMock()
    manager._log_timing = MagicMock()
    residual = torch.empty((8_192, 1), dtype=torch.float32)
    positions = torch.arange(8_192, dtype=torch.int64)

    with patch.dict(
        indexer_manager_module.os.environ,
        {
            "LMCACHE_CSA_ADAPTIVE_DENSE_PREFETCH": "1",
            "LMCACHE_CSA_ADAPTIVE_DENSE_THRESHOLD_PERCENT": "80",
            "LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE": "4",
            "LMCACHE_CSA_PREFETCH_CP_SIZE": "8",
            "LMCACHE_CSA_PREFETCH_CP_INTERLEAVE": "64",
        },
    ):
        manager.fire_residual_prefetch_for_layer(
            2,
            residual,
            positions,
            lookahead=2,
        )

    manager.fire_dense_csa_layers.assert_called_once_with(
        (2,),
        source_layer_id=0,
    )
    manager.fire_async_for_layer.assert_not_called()


def test_adaptive_dense_prefetch_retains_sparse_proxy() -> None:
    """A short append keeps the sparse predictor and avoids dense I/O."""
    manager = object.__new__(IndexerSSDManager)
    manager._prefetch_lookahead = {2: 2}
    manager._decode_cursor = {2: 196_608}
    manager._proxy_topk_tokens_by_layer = {}
    manager._l1_proxy_topk_tokens = 2_048
    manager._csa_attention_kv_manager = SimpleNamespace(active_request_id="request-a")
    manager.wait_for_seed = MagicMock(return_value=True)
    manager.fire_dense_csa_layers = MagicMock()
    manager.fire_async_for_layer = MagicMock()
    manager._log_timing = MagicMock()
    residual = torch.empty((256, 1), dtype=torch.float32)
    positions = torch.arange(256, dtype=torch.int64)

    with patch.dict(
        indexer_manager_module.os.environ,
        {
            "LMCACHE_CSA_ADAPTIVE_DENSE_PREFETCH": "1",
            "LMCACHE_CSA_ADAPTIVE_DENSE_THRESHOLD_PERCENT": "80",
            "LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE": "4",
            "LMCACHE_CSA_PREFETCH_CP_SIZE": "8",
            "LMCACHE_CSA_PREFETCH_CP_INTERLEAVE": "64",
        },
    ):
        manager.fire_residual_prefetch_for_layer(
            2,
            residual,
            positions,
            lookahead=2,
        )

    manager.fire_dense_csa_layers.assert_not_called()
    manager.fire_async_for_layer.assert_called_once_with(
        2,
        residual_f=residual,
        positions=positions,
        prefetch_level=2,
    )


def test_native_indexer_stream_orders_and_schedules_demanded_layer() -> None:
    """A late gate queues every unscheduled layer through its target."""
    submitted: list[tuple[int, str]] = []
    waited: list[int] = []
    tracked: dict[int, Any] = {}

    def fire(
        layer_id: int,
        *,
        label: str,
        request_token: tuple[str, int] | None = None,
        source_layer_id: int | None = None,
    ) -> bool:
        del request_token, source_layer_id
        submitted.append((layer_id, label))
        return True

    fake_loader = SimpleNamespace(
        register_layer=MagicMock(),
        register_request_chunks=MagicMock(),
        deactivate_request=MagicMock(return_value=True),
        active_request_token=("request-a", 1),
        fire_deterministic_layer=fire,
        track_layer_submission=lambda layer_id, future, **_kwargs: tracked.update(
            {layer_id: future}
        ),
        wait_for_layer=lambda layer_id, timeout_s: waited.append(layer_id) or True,
        close=MagicMock(),
    )
    with tempfile.TemporaryDirectory() as store_dir:
        manager = IndexerSSDManager(
            csa_layer_ids=[2, 4, 6, 8, 10, 12],
            store_dir=store_dir,
            pool_size=8,
            token_bytes=4,
            max_seq_len=32,
            io_workers=1,
            device=torch.device("cpu"),
        )
        try:
            with patch(
                "lmcache.v1.csa_attention_kv_prefetch_manager."
                "CSAAttentionKVPrefetchManager",
                return_value=fake_loader,
            ):
                manager.attach_native_indexer_cache_loader(
                    SimpleNamespace(),
                    {
                        2: torch.empty((1, 64, 4), dtype=torch.uint8),
                        4: torch.empty((1, 64, 4), dtype=torch.uint8),
                        6: torch.empty((1, 64, 4), dtype=torch.uint8),
                        8: torch.empty((1, 64, 4), dtype=torch.uint8),
                        10: torch.empty((1, 64, 4), dtype=torch.uint8),
                        12: torch.empty((1, 64, 4), dtype=torch.uint8),
                    },
                )
                chunks = {
                    2: [SimpleNamespace(end_compressed_block=3)],
                    4: [SimpleNamespace(end_compressed_block=3)],
                    6: [SimpleNamespace(end_compressed_block=3)],
                    8: [SimpleNamespace(end_compressed_block=3)],
                    10: [SimpleNamespace(end_compressed_block=3)],
                    12: [SimpleNamespace(end_compressed_block=3)],
                }
                assert manager.register_native_indexer_stream("request-a", chunks)
                # Stage0 is queued, but registration must not wait for I/O;
                # layer 2's normal consumption gate owns the barrier.
                assert waited == []
                assert manager.wait_for_native_indexer_layer(2)
                for future in tracked.values():
                    future.result(timeout=2.0)
                assert manager.native_indexer_stream_active()
                assert manager.has_layer_rows(2, 3 * 64)
                # The initial asynchronous Stage0 plus window covers only
                # 2/4/6. A direct demand for 12 queues 8/10/12 before waiting.
                assert manager.wait_for_native_indexer_layer(12)
                for future in tracked.values():
                    future.result(timeout=2.0)
        finally:
            manager.close()

    fake_loader.register_request_chunks.assert_called_once_with(
        "request-a",
        chunks,
        start_profile_capture=False,
        shared_raw_lba_cache=None,
    )
    assert submitted == [
        (2, "indexer_native_stream"),
        (4, "indexer_native_stream"),
        (6, "indexer_native_stream"),
        (8, "indexer_native_stream"),
        (10, "indexer_native_stream"),
        (12, "indexer_native_stream"),
    ]
    assert waited == [2, 12]


def test_native_indexer_partial_registration_is_rolled_back() -> None:
    """A loader exception cannot leave a half-registered stream reusable."""
    deactivations: list[int] = []

    def deactivate_request(*, timeout_s: float) -> bool:
        del timeout_s
        deactivations.append(1)
        return True

    fake_loader = SimpleNamespace(
        deactivate_request=deactivate_request,
        register_request_chunks=MagicMock(side_effect=RuntimeError("compile failed")),
    )
    manager = object.__new__(IndexerSSDManager)
    manager._native_indexer_cache_manager = fake_loader
    manager._native_indexer_stream_active = False
    manager._native_indexer_stream_request_id = ""
    manager._native_indexer_stream_cleanup_failed = False
    manager._native_indexer_scheduled_layers = set()
    manager._csa_layer_ids = (2,)
    manager._decode_cursor = {2: 0}
    manager._lock = threading.RLock()

    with pytest.raises(RuntimeError, match="compile failed"):
        manager.register_native_indexer_stream(
            "request-a",
            {2: [SimpleNamespace(end_compressed_block=1)]},
        )

    assert len(deactivations) == 1
    assert not manager._native_indexer_stream_active
    assert manager._native_indexer_stream_request_id == ""
    assert not manager._native_indexer_stream_cleanup_failed


def test_inactive_native_indexer_stream_is_already_drained() -> None:
    """First-hit registration does not deactivate a nonexistent old request."""
    loader = MagicMock()
    manager = object.__new__(IndexerSSDManager)
    manager._native_indexer_cache_manager = loader
    manager._native_indexer_stream_active = False
    manager._native_indexer_stream_cleanup_failed = False

    assert manager.deactivate_native_indexer_stream()
    loader.deactivate_request.assert_not_called()


def test_deferred_direct_seed_publishes_hbm_before_persistence() -> None:
    """Retrieve-held loader lock does not delay direct-seed HBM readiness."""
    loader = FakeTuttiLoader()
    storage = make_storage(loader)
    manager = IndexerSSDManager(
        csa_layer_ids=[2],
        store_dir="unused-for-tutti",
        pool_size=2,
        token_bytes=4,
        max_seq_len=1024,
        io_workers=1,
        device=torch.device("cpu"),
        tutti_storage=storage,
    )
    staging = torch.arange(8, dtype=torch.uint8).view(1, 1, 2, 4)
    expected = staging.clone().reshape(-1).tolist()
    loader._io_lock.acquire()
    try:
        future = manager.submit_seed_range_from_lmcache_group(
            [2],
            staging,
            0,
            8,
            total_logical_tokens=8,
        )
        assert future is not None
        staging.fill_(255)
        assert loader.store_started.wait(timeout=2.0)
        assert future.result(timeout=2.0) == 1
        assert manager.wait_for_seed(2)
        assert manager.has_layer_rows(2, 2)
        assert loader.stored_payloads == []
    finally:
        loader._io_lock.release()

    manager.close()

    assert list(loader.stored_payloads[0][: len(expected)]) == expected


def test_deferred_seed_chain_does_not_exhaust_executor_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many retrieve chunks serialize without worker-pool self-deadlock."""
    # Windows does not expose os.pwrite; persistence itself is covered by the
    # Tutti test above. This test isolates executor dependency ordering.
    monkeypatch.setattr(
        indexer_manager_module,
        "_pwrite",
        lambda _fd, _data, _offset: None,
    )
    with tempfile.TemporaryDirectory() as store_dir:
        manager = IndexerSSDManager(
            csa_layer_ids=[2],
            store_dir=store_dir,
            pool_size=32,
            token_bytes=4,
            max_seq_len=128,
            io_workers=4,
            device=torch.device("cpu"),
        )
        futures = []
        try:
            for row in range(32):
                staging = torch.full(
                    (1, 1, 1, 4),
                    row,
                    dtype=torch.uint8,
                )
                future = manager.submit_seed_range_from_lmcache_group(
                    [2],
                    staging,
                    row * 4,
                    (row + 1) * 4,
                    total_logical_tokens=128,
                )
                assert future is not None
                futures.append(future)

            # This is the true-indexer consumer path. It must join the newest
            # future, which transitively joins every earlier chunk seed.
            manager.prepare_pool(2)
            assert futures[-1].done()
            assert manager.has_layer_rows(2, 32)
        finally:
            manager.close()


def test_deferred_seed_coalesces_chunks_into_one_layer_write() -> None:
    """A completed tail persists once instead of once per retrieve chunk."""
    loader = FakeTuttiLoader()
    storage = make_storage(loader)
    manager = IndexerSSDManager(
        csa_layer_ids=[2],
        store_dir="unused-for-tutti",
        pool_size=4,
        token_bytes=4,
        max_seq_len=1024,
        io_workers=1,
        device=torch.device("cpu"),
        tutti_storage=storage,
    )
    first = torch.arange(8, dtype=torch.uint8).view(1, 1, 2, 4)
    second = torch.arange(8, 16, dtype=torch.uint8).view(1, 1, 2, 4)
    try:
        first_future = manager.submit_seed_range_from_lmcache_group(
            [2], first, 0, 8, total_logical_tokens=16
        )
        second_future = manager.submit_seed_range_from_lmcache_group(
            [2], second, 8, 16, total_logical_tokens=16
        )
        assert first_future is not None
        assert second_future is not None
        assert second_future.result(timeout=2.0) == 1
        assert manager.wait_for_seed(2)
    finally:
        manager.close()

    assert len(loader.stored_payloads) == 1
    assert list(loader.stored_payloads[0][:16]) == list(range(16))
