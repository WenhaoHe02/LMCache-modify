# SPDX-License-Identifier: Apache-2.0
"""Tests for the public Tutti indexer-storage interface."""

# Standard
from concurrent.futures import Future
import tempfile
import threading
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
    globalize_prefill_cp_topk,
    prefill_cp_key_indices,
    prefill_cp_local_topk_tokens,
    prefill_cp_query_indices,
    prefill_cp_query_ranges,
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
    """A deep target fires exactly one L2 prediction and rejects stale L1."""
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
            # A stale native L1 hook must also be rejected at the final
            # async entry point, not merely by the source mapping.
            with pytest.raises(ValueError, match="prefetch_level must be 2"):
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
                with pytest.raises(ValueError, match="lookahead must be 2"):
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
                with pytest.raises(ValueError, match="lookahead must be 2"):
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
    submitted: list[int] = []
    tracked: dict[int, Any] = {}
    fake_csa_manager = SimpleNamespace(
        active_request_id="request-a",
        fire_deterministic_layer=lambda layer_id: submitted.append(layer_id),
        track_layer_submission=lambda layer_id, future: tracked.update(
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
            manager.fire_async_for_layers([3])
            tracked[3].result(timeout=2.0)
        finally:
            manager.close()

    assert submitted == [3]


def test_deferred_direct_seed_owns_staging_and_waits_outside_loader_lock() -> None:
    """Retrieve-held loader lock cannot deadlock deferred indexer seeding."""
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
        assert not future.done()
    finally:
        loader._io_lock.release()

    try:
        assert future.result(timeout=2.0) == 1
        assert manager.wait_for_seed(2)
        assert manager.has_layer_rows(2, 2)
    finally:
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
