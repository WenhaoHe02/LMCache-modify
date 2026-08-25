# SPDX-License-Identifier: Apache-2.0
# Standard
import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


def test_streaming_deactivation_expires_predictions_before_cache_plans() -> None:
    """Late proxy work is closed before either split-layout plan is cleared."""
    calls: list[str] = []
    csa_manager = SimpleNamespace(
        deactivate_request=lambda: calls.append("csa") or True,
    )
    indexer_manager = SimpleNamespace(
        deactivate_csa_predictions=lambda: calls.append("predictions") or True,
        deactivate_native_indexer_stream=lambda: calls.append("indexer") or True,
    )

    LMCacheEngine._dsv4_deactivate_streaming_consumers(
        csa_manager,
        indexer_manager,
    )

    assert calls == ["predictions", "indexer", "csa"]


class _RefCountedMemoryObj:
    def __init__(self) -> None:
        self.refs = 1

    def ref_count_up(self) -> None:
        self.refs += 1

    def ref_count_down(self) -> None:
        self.refs -= 1


def _snapshot_engine() -> LMCacheEngine:
    engine = object.__new__(LMCacheEngine)
    engine._dsv4_snapshot_lock = threading.Lock()
    engine._dsv4_snapshots = {}
    return engine


def test_layer_major_snapshot_is_emitted_only_for_final_batch() -> None:
    """Intermediate batches retain host objects but perform no snapshot."""
    engine = _snapshot_engine()
    first_obj = _RefCountedMemoryObj()
    final_obj = _RefCountedMemoryObj()
    first_key = Mock()
    final_key = Mock()

    assert (
        engine._stage_dsv4_layer_major_snapshot(
            "request",
            [first_key],
            [first_obj],  # type: ignore[list-item]
            256,
            is_last_prefill=False,
        )
        is None
    )
    result = engine._stage_dsv4_layer_major_snapshot(
        "request",
        [final_key],
        [final_obj],  # type: ignore[list-item]
        128,
        is_last_prefill=True,
    )

    assert result is not None
    keys, memory_objs, tokens, base_key, base_tokens = result
    assert keys == [first_key, final_key]
    assert memory_objs == [first_obj, final_obj]
    assert tokens == 384
    assert base_key is None
    assert base_tokens == 0
    assert first_obj.refs == final_obj.refs == 2
    for memory_obj in memory_objs:
        memory_obj.ref_count_down()
    assert first_obj.refs == final_obj.refs == 1


def test_interleaved_requests_keep_independent_snapshots() -> None:
    """Interleaved chunked prefills retain one snapshot per request."""
    engine = _snapshot_engine()
    first_a = _RefCountedMemoryObj()
    first_b = _RefCountedMemoryObj()
    final_a = _RefCountedMemoryObj()
    final_b = _RefCountedMemoryObj()
    keys = [Mock() for _ in range(4)]

    engine._stage_dsv4_layer_major_snapshot(
        "request-a",
        [keys[0]],
        [first_a],  # type: ignore[list-item]
        256,
        is_last_prefill=False,
    )
    engine._stage_dsv4_layer_major_snapshot(
        "request-b",
        [keys[1]],
        [first_b],  # type: ignore[list-item]
        256,
        is_last_prefill=False,
    )
    result_a = engine._stage_dsv4_layer_major_snapshot(
        "request-a",
        [keys[2]],
        [final_a],  # type: ignore[list-item]
        256,
        is_last_prefill=True,
    )
    result_b = engine._stage_dsv4_layer_major_snapshot(
        "request-b",
        [keys[3]],
        [final_b],  # type: ignore[list-item]
        256,
        is_last_prefill=True,
    )

    assert result_a is not None
    assert result_b is not None
    assert result_a[0] == [keys[0], keys[2]]
    assert result_b[0] == [keys[1], keys[3]]
    assert result_a[1] == [first_a, final_a]
    assert result_b[1] == [first_b, final_b]
    assert not engine._dsv4_snapshots
    for memory_obj in [*result_a[1], *result_b[1]]:
        memory_obj.ref_count_down()
    assert all(obj.refs == 1 for obj in [first_a, first_b, final_a, final_b])


def test_discard_releases_every_incomplete_snapshot() -> None:
    """A full clear releases retained objects for all aborted prefills."""
    engine = _snapshot_engine()
    objects = [_RefCountedMemoryObj(), _RefCountedMemoryObj()]
    for request_id, memory_obj in zip(["request-a", "request-b"], objects, strict=True):
        engine._stage_dsv4_layer_major_snapshot(
            request_id,
            [Mock()],
            [memory_obj],  # type: ignore[list-item]
            256,
            is_last_prefill=False,
        )

    engine._discard_dsv4_layer_major_snapshot()

    assert not engine._dsv4_snapshots
    assert all(memory_obj.refs == 1 for memory_obj in objects)


@pytest.mark.parametrize("admission_mode", ["deferred_cold", "deferred_hit"])
def test_deferred_admission_preserves_publish_order(admission_mode: str) -> None:
    """Deferred admission publishes sidecars before handing main ownership off."""

    class _ImmediateExecutor:
        def submit(self, callback: object) -> None:
            callback()  # type: ignore[operator]

    events: list[str] = []
    memory_obj = _RefCountedMemoryObj()
    key = Mock()
    key.to_string.return_value = "key"
    backend = SimpleNamespace(
        store_attention_layer_major_snapshot=lambda *_args, **_kwargs: (
            events.append("sidecars") or 3
        )
    )

    def _batched_put(
        _keys: object,
        memory_objs: list[_RefCountedMemoryObj],
        *,
        transfer_spec: object,
        location: object,
        on_complete_callback: object = None,
    ) -> None:
        del transfer_spec, location, on_complete_callback
        events.append("main")
        for obj in memory_objs:
            obj.ref_count_down()

    engine = object.__new__(LMCacheEngine)
    engine._dsv4_admission_lock = threading.Lock()
    engine._dsv4_admission_pending = 0
    engine._dsv4_admission_max_pending = 2
    engine._dsv4_admission_executor = _ImmediateExecutor()
    engine.metadata = SimpleNamespace(worker_id=1)
    engine.store_location = "LocalDiskBackend"
    engine.storage_manager = SimpleNamespace(
        storage_backends={"LocalDiskBackend": backend},
        batched_put=_batched_put,
    )
    engine._make_tutti_store_warmup_callback = Mock(return_value=object())

    assert engine._submit_dsv4_admission(
        [key],
        [memory_obj],  # type: ignore[list-item]
        8192,
        admission_mode=admission_mode,
        req_id="hit",
        is_last_prefill=True,
        transfer_spec=None,
    )

    assert events == ["sidecars", "main"]
    assert engine._dsv4_admission_pending == 0
    assert memory_obj.refs == 0


def test_deferred_hit_admission_is_bounded() -> None:
    """A saturated admission queue falls back instead of retaining unbounded KV."""

    class _QueuedExecutor:
        def __init__(self) -> None:
            self.callbacks: list[object] = []

        def submit(self, callback: object) -> None:
            self.callbacks.append(callback)

    executor = _QueuedExecutor()
    engine = object.__new__(LMCacheEngine)
    engine._dsv4_admission_lock = threading.Lock()
    engine._dsv4_admission_pending = 0
    engine._dsv4_admission_max_pending = 1
    engine._dsv4_admission_executor = executor
    engine.metadata = SimpleNamespace(worker_id=1)

    args = ([Mock()], [Mock()], 256)
    kwargs = {
        "req_id": "hit",
        "is_last_prefill": True,
        "transfer_spec": None,
    }
    assert engine._submit_dsv4_hit_admission(*args, **kwargs)
    assert not engine._submit_dsv4_hit_admission(*args, **kwargs)
    assert engine._dsv4_admission_pending == 1
    assert len(executor.callbacks) == 1


def test_intermediate_prefill_does_not_create_a_false_request_tail() -> None:
    """Only the final scheduler batch may retain tail-only KV groups."""
    engine = object.__new__(LMCacheEngine)
    engine.dsv4_optimized_kv = True
    engine.dsv4_optimized_tail_tokens = 256
    engine.metadata = SimpleNamespace(
        kv_layer_groups_manager=SimpleNamespace(
            kv_layer_groups=["tail-group", "ordinary-group"]
        )
    )
    engine._dsv4_group_role = Mock(
        side_effect=["swa_cache", "ordinary", "swa_cache", "ordinary"]
    )
    engine._zero_token_shape = Mock(side_effect=lambda shape: f"zero-{shape}")

    intermediate = engine._dsv4_store_shapes_for_range(
        ["tail", "ordinary"],  # type: ignore[list-item]
        [Mock(), Mock()],
        768,
        1024,
        1024,
        keep_request_tail=False,
    )
    final = engine._dsv4_store_shapes_for_range(
        ["tail", "ordinary"],  # type: ignore[list-item]
        [Mock(), Mock()],
        768,
        1024,
        1024,
        keep_request_tail=True,
    )

    assert intermediate == ["zero-tail", "ordinary"]
    assert final == ["tail", "ordinary"]


def test_streaming_compact_availability_accepts_metadata_only_main() -> None:
    """Zero-length main entries are hits without physical object records."""
    engine = object.__new__(LMCacheEngine)
    physical_record = object()
    backend = SimpleNamespace(
        get_kv_object_payload_lengths=lambda _keys, roles: [0, 0, 64],
        get_kv_object_records=lambda _keys, roles: [
            None,
            None,
            physical_record,
        ],
        kv_object_record_raw_readable=lambda record: record is physical_record,
    )
    blocks = [
        (Mock(), 0, 256),
        (Mock(), 256, 512),
        (Mock(), 512, 768),
    ]

    assert engine._dsv4_streaming_compact_retrieve_available(
        blocks,
        backend,
        "csa_hca_deferred_retrieve",
        "test compact retrieve",
    )


def test_streaming_compact_availability_rejects_missing_physical_main() -> None:
    """A positive-length main entry still requires a readable record."""
    engine = object.__new__(LMCacheEngine)
    backend = SimpleNamespace(
        get_kv_object_payload_lengths=lambda _keys, roles: [0, 64],
        get_kv_object_records=lambda _keys, roles: [None, None],
        kv_object_record_raw_readable=lambda _record: False,
    )
    blocks = [(Mock(), 0, 256), (Mock(), 256, 512)]

    assert not engine._dsv4_streaming_compact_retrieve_available(
        blocks,
        backend,
        "csa_hca_deferred_retrieve",
        "test compact retrieve",
    )


def test_lookup_filter_accepts_metadata_only_prefix_for_both_on_layouts() -> None:
    """Raw lookup preserves zero-byte main entries in CSA and CSA/HCA modes."""
    for hca_walker, expected_role in (
        ("0", "csa_deferred_retrieve"),
        ("1", "csa_hca_deferred_retrieve"),
    ):
        engine = object.__new__(LMCacheEngine)
        engine.dsv4_optimized_kv = True
        engine._tutti_config = {"slot_mb": 1, "n_slots": 1}
        physical_record = SimpleNamespace(length=64)
        engine._dsv4_streaming_runtime_consumer_ready = Mock(return_value=True)
        engine._dsv4_streaming_compact_payload_nbytes = Mock(
            side_effect=[0] * 127 + [64]
        )
        engine._tutti_lookup_read_record = Mock(return_value=physical_record)

        backend = object.__new__(LocalDiskBackend)
        backend.kv_object_store_enabled = True
        backend.kv_object_tutti_raw_enabled = True
        backend.get_kv_object_payload_lengths = Mock(return_value=[0] * 127 + [64])

        def _get_records(
            _keys: object,
            *,
            roles: object = None,
            _physical_record: object = physical_record,
        ) -> list[object | None]:
            return (
                [None] * 127 + [_physical_record] if roles is not None else [None] * 128
            )

        backend.get_kv_object_records = Mock(
            side_effect=_get_records,
        )
        backend.kv_object_record_raw_read_bytes = Mock(return_value=64)
        backend.kv_object_record_raw_readable = Mock(return_value=True)
        engine.storage_manager = SimpleNamespace(
            storage_backends={"LocalDiskBackend": backend}
        )

        keys = [Mock() for _index in range(128)]
        for index, key in enumerate(keys):
            key.to_string.return_value = f"key-{index}"
        chunks = [
            (index * 256, (index + 1) * 256, key) for index, key in enumerate(keys)
        ]
        block_mapping = {"LocalDiskBackend": list(keys)}

        with patch.dict(
            os.environ,
            {
                "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
                "LMCACHE_DSV4_HCA_WALKER": hca_walker,
            },
        ):
            assert (
                engine._filter_tutti_raw_lookup_prefix(
                    chunks,
                    128,
                    block_mapping,
                    pin=False,
                    total_tokens=32768,
                )
                == 128
            )

        backend.get_kv_object_payload_lengths.assert_called_once_with(
            keys,
            roles=[expected_role] * 128,
        )
        engine._tutti_lookup_read_record.assert_called_once()
        assert engine._dsv4_streaming_compact_payload_nbytes.call_count == 128
        assert (
            engine._tutti_lookup_read_record.call_args.kwargs["streaming_compact_role"]
            == expected_role
        )
        backend.kv_object_record_raw_read_bytes.assert_called_once_with(physical_record)
        assert block_mapping == {"LocalDiskBackend": keys}


def test_lookup_filter_ignores_stored_tail_payload_for_longer_request() -> None:
    """A former tail main becomes metadata-only after its prefix is extended."""
    engine = object.__new__(LMCacheEngine)
    engine.dsv4_optimized_kv = True
    engine._tutti_config = {"slot_mb": 1, "n_slots": 1}
    engine._dsv4_streaming_runtime_consumer_ready = Mock(return_value=True)
    engine._dsv4_streaming_compact_payload_nbytes = Mock(return_value=0)
    engine._tutti_lookup_read_record = Mock()

    physical_record = SimpleNamespace(length=64)
    backend = object.__new__(LocalDiskBackend)
    backend.kv_object_store_enabled = True
    backend.kv_object_tutti_raw_enabled = True
    backend.get_kv_object_payload_lengths = Mock(return_value=[64])
    backend.get_kv_object_records = Mock(side_effect=[[None], [physical_record]])
    engine.storage_manager = SimpleNamespace(
        storage_backends={"LocalDiskBackend": backend}
    )

    key = Mock()
    key.to_string.return_value = "former-tail"
    block_mapping = {"LocalDiskBackend": [key]}
    with patch.dict(
        os.environ,
        {
            "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
            "LMCACHE_DSV4_HCA_WALKER": "1",
        },
    ):
        hit_chunks = engine._filter_tutti_raw_lookup_prefix(
            [(0, 256, key)],
            1,
            block_mapping,
            pin=False,
            total_tokens=8448,
        )

    assert hit_chunks == 1
    assert block_mapping == {"LocalDiskBackend": [key]}
    engine._tutti_lookup_read_record.assert_not_called()


def test_lookup_filter_rejects_stale_metadata_only_shape() -> None:
    """A prior non-tail zero-byte main cannot satisfy a current tail view."""
    engine = object.__new__(LMCacheEngine)
    engine.dsv4_optimized_kv = True
    engine._tutti_config = {"slot_mb": 1, "n_slots": 1}
    engine._dsv4_streaming_runtime_consumer_ready = Mock(return_value=True)
    engine._dsv4_streaming_compact_payload_nbytes = Mock(return_value=64)

    backend = object.__new__(LocalDiskBackend)
    backend.kv_object_store_enabled = True
    backend.kv_object_tutti_raw_enabled = True
    backend.get_kv_object_payload_lengths = Mock(return_value=[0])
    backend.get_kv_object_records = Mock(return_value=[None])
    engine.storage_manager = SimpleNamespace(
        storage_backends={"LocalDiskBackend": backend}
    )

    key = Mock()
    key.to_string.return_value = "stale-non-tail"
    block_mapping = {"LocalDiskBackend": [key]}

    with patch.dict(
        os.environ,
        {
            "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
            "LMCACHE_DSV4_HCA_WALKER": "1",
        },
    ):
        assert (
            engine._filter_tutti_raw_lookup_prefix(
                [(0, 256, key)],
                1,
                block_mapping,
                pin=False,
                total_tokens=256,
            )
            == 0
        )

    assert block_mapping == {}


def test_lookup_filter_rejects_streaming_layout_without_consumer() -> None:
    """A READY split layout is not a hit when its runtime consumer is absent."""
    engine = object.__new__(LMCacheEngine)
    engine.dsv4_optimized_kv = True
    engine._tutti_config = {"slot_mb": 1, "n_slots": 1}
    engine._dsv4_streaming_runtime_consumer_ready = Mock(return_value=False)

    backend = object.__new__(LocalDiskBackend)
    backend.kv_object_store_enabled = True
    backend.kv_object_tutti_raw_enabled = True
    backend.get_kv_object_records = Mock(return_value=[None])
    engine.storage_manager = SimpleNamespace(
        storage_backends={"LocalDiskBackend": backend}
    )

    key = Mock()
    key.to_string.return_value = "consumer-inactive"
    block_mapping = {"LocalDiskBackend": [key]}

    with patch.dict(
        os.environ,
        {
            "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
            "LMCACHE_DSV4_HCA_WALKER": "0",
        },
    ):
        assert (
            engine._filter_tutti_raw_lookup_prefix(
                [(0, 256, key)],
                1,
                block_mapping,
                pin=False,
                total_tokens=256,
            )
            == 0
        )

    assert block_mapping == {}


def test_streaming_runtime_probe_requires_indexer_and_hca_consumers() -> None:
    """Lookup readiness is observational and covers every removed group."""
    engine = object.__new__(LMCacheEngine)
    csa_manager = SimpleNamespace(
        hca_layer_ids=(20,),
        request_stream_available=Mock(return_value=True),
    )
    indexer_manager = SimpleNamespace(
        native_indexer_stream_available=Mock(return_value=True)
    )

    with (
        patch(
            "lmcache.v1.csa_attention_kv_prefetch_manager."
            "get_csa_attention_kv_prefetch_manager",
            return_value=csa_manager,
        ),
        patch(
            "lmcache.v1.indexer_ssd_manager.get_indexer_ssd_manager",
            return_value=indexer_manager,
        ),
    ):
        assert engine._dsv4_streaming_runtime_consumer_ready("csa_deferred_retrieve")
        assert engine._dsv4_streaming_runtime_consumer_ready(
            "csa_hca_deferred_retrieve"
        )
        csa_manager.hca_layer_ids = ()
        assert not engine._dsv4_streaming_runtime_consumer_ready(
            "csa_hca_deferred_retrieve"
        )
        indexer_manager.native_indexer_stream_available.return_value = False
        assert not engine._dsv4_streaming_runtime_consumer_ready(
            "csa_deferred_retrieve"
        )


def test_streaming_only_hit_skips_generic_retrieve_preparation() -> None:
    """An all-zero compact main commits only the layer-major read plan."""
    engine = object.__new__(LMCacheEngine)
    engine.dsv4_optimized_kv = True
    engine.dsv4_optimized_tail_tokens = 8192
    engine._tutti_config = {"slot_mb": 1, "n_slots": 1}
    engine._dsv4_csa_attention_kv_prefetch_active = Mock(return_value=True)
    engine._dsv4_streaming_compact_payload_nbytes = Mock(return_value=0)
    engine._ensure_tutti_loader = Mock(return_value=True)
    engine._register_csa_attention_kv_chunks = Mock(return_value=(True, True, True))
    key = Mock(spec=CacheEngineKey)
    block = (key, 0, 256)
    engine.lookup_pins = {"request": {"LocalDiskBackend": [key]}}
    engine.token_database = SimpleNamespace(
        process_tokens=Mock(return_value=[(0, 256, key)])
    )
    disk_metadata = object()
    engine.storage_manager = SimpleNamespace(
        storage_backends={
            "LocalDiskBackend": SimpleNamespace(dict={key: disk_metadata})
        }
    )

    slot_mapping = torch.arange(256)
    with patch.dict(
        os.environ,
        {
            "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
            "LMCACHE_DSV4_HCA_WALKER": "1",
        },
    ):
        ret_mask = engine.prepare_dsv4_streaming_retrieve(
            torch.arange(256),
            torch.ones(256, dtype=torch.bool),
            req_id="request",
            request_total_tokens=8448,
            slot_mapping=slot_mapping,
        )

    assert ret_mask is not None
    assert ret_mask.all()
    call = engine._register_csa_attention_kv_chunks.call_args
    assert call.args == ([block], [disk_metadata], 256, "request")
    assert call.kwargs["slot_mapping"] is slot_mapping
    engine._dsv4_streaming_compact_payload_nbytes.assert_not_called()


def test_streaming_only_hit_uses_pinned_terminal_key_without_rehashing() -> None:
    """A dense aligned hit reuses the scheduler's final pinned chunk hash."""
    engine = object.__new__(LMCacheEngine)
    engine.dsv4_optimized_kv = True
    engine.dsv4_optimized_tail_tokens = 8192
    engine._tutti_config = {"slot_mb": 1, "n_slots": 1}
    engine.config = SimpleNamespace(chunk_size=256)
    engine.metadata = SimpleNamespace(
        model_name="model",
        world_size=8,
        worker_id=0,
        kv_dtype=torch.float16,
    )
    engine._dsv4_csa_attention_kv_prefetch_active = Mock(return_value=True)
    engine._dsv4_streaming_compact_payload_nbytes = Mock(return_value=0)
    engine._ensure_tutti_loader = Mock(return_value=True)
    engine._register_csa_attention_kv_chunks = Mock(return_value=(True, True, True))
    terminal_key = CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=123,
        dtype=torch.float16,
        request_configs=None,
    )
    disk_metadata = object()
    engine.lookup_pins = {"request": {"LocalDiskBackend": [Mock(), terminal_key]}}
    engine.token_database = SimpleNamespace(process_tokens=Mock())
    engine.storage_manager = SimpleNamespace(
        storage_backends={
            "LocalDiskBackend": SimpleNamespace(dict={terminal_key: disk_metadata})
        }
    )

    tokens = torch.arange(256)
    with patch.dict(
        os.environ,
        {
            "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
            "LMCACHE_DSV4_HCA_WALKER": "1",
        },
    ):
        ret_mask = engine.prepare_dsv4_streaming_retrieve(
            tokens,
            torch.ones(256, dtype=torch.bool),
            req_id="request",
            request_total_tokens=8448,
            vllm_cached_tokens=0,
            terminal_chunk_hash=123,
        )

    assert ret_mask is not None
    assert ret_mask.all()
    engine.token_database.process_tokens.assert_not_called()
    engine._register_csa_attention_kv_chunks.assert_called_once_with(
        [(terminal_key, 0, 256)],
        [disk_metadata],
        256,
        "request",
        slot_mapping=None,
    )


def test_streaming_only_hit_composes_deferred_suffix_generations() -> None:
    """A short terminal snapshot retries with the complete admitted chain."""
    engine = object.__new__(LMCacheEngine)
    engine.dsv4_optimized_kv = True
    engine.dsv4_optimized_tail_tokens = 8192
    engine._tutti_config = {"slot_mb": 1, "n_slots": 1}
    engine.config = SimpleNamespace(chunk_size=256)
    engine.metadata = SimpleNamespace(
        model_name="model",
        world_size=8,
        worker_id=0,
        kv_dtype=torch.float16,
    )
    engine._dsv4_csa_attention_kv_prefetch_active = Mock(return_value=True)
    engine._dsv4_streaming_compact_payload_nbytes = Mock(return_value=0)
    engine._ensure_tutti_loader = Mock(return_value=True)
    engine._register_csa_attention_kv_chunks = Mock(
        side_effect=[(False, False, False), (True, True, True)]
    )
    prefix_key = CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=111,
        dtype=torch.float16,
        request_configs=None,
    )
    terminal_key = CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=222,
        dtype=torch.float16,
        request_configs=None,
    )
    prefix_metadata = object()
    terminal_metadata = object()
    engine.lookup_pins = {"request": {"LocalDiskBackend": [prefix_key, terminal_key]}}
    engine.token_database = SimpleNamespace(
        process_tokens=Mock(
            return_value=[
                (0, 256, prefix_key),
                (256, 512, terminal_key),
            ]
        )
    )
    engine.storage_manager = SimpleNamespace(
        storage_backends={
            "LocalDiskBackend": SimpleNamespace(
                dict={
                    prefix_key: prefix_metadata,
                    terminal_key: terminal_metadata,
                }
            )
        }
    )

    tokens = torch.arange(512)
    with patch.dict(
        os.environ,
        {
            "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
            "LMCACHE_DSV4_HCA_WALKER": "1",
        },
    ):
        ret_mask = engine.prepare_dsv4_streaming_retrieve(
            tokens,
            torch.ones(512, dtype=torch.bool),
            req_id="request",
            request_total_tokens=512,
            vllm_cached_tokens=0,
            terminal_chunk_hash=222,
        )

    assert ret_mask is not None
    assert ret_mask.all()
    assert engine._register_csa_attention_kv_chunks.call_count == 2
    first_call, second_call = engine._register_csa_attention_kv_chunks.call_args_list
    assert first_call.args[0] == [(terminal_key, 0, 512)]
    assert second_call.args[0] == [
        (prefix_key, 0, 256),
        (terminal_key, 256, 512),
    ]
    assert second_call.args[1] == [prefix_metadata, terminal_metadata]


def test_streaming_only_hit_rejects_nonempty_compact_main() -> None:
    """Any synchronous compact-main payload retains the generic path."""
    engine = object.__new__(LMCacheEngine)
    engine.dsv4_optimized_kv = True
    engine._tutti_config = {"slot_mb": 1, "n_slots": 1}
    engine._dsv4_csa_attention_kv_prefetch_active = Mock(return_value=True)
    engine._dsv4_streaming_compact_payload_nbytes = Mock(return_value=64)
    engine._ensure_tutti_loader = Mock(return_value=True)
    engine._register_csa_attention_kv_chunks = Mock()
    key = Mock(spec=CacheEngineKey)
    engine.lookup_pins = {"request": {"LocalDiskBackend": [key]}}
    engine.token_database = SimpleNamespace(
        process_tokens=Mock(return_value=[(0, 256, key)])
    )
    engine.storage_manager = SimpleNamespace(
        storage_backends={"LocalDiskBackend": SimpleNamespace(dict={key: object()})}
    )

    with patch.dict(
        os.environ,
        {
            "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
            "LMCACHE_DSV4_HCA_WALKER": "1",
        },
    ):
        assert (
            engine.prepare_dsv4_streaming_retrieve(
                torch.arange(256),
                torch.ones(256, dtype=torch.bool),
                req_id="request",
                request_total_tokens=256,
            )
            is None
        )

    engine._ensure_tutti_loader.assert_not_called()
    engine._register_csa_attention_kv_chunks.assert_not_called()


def test_streaming_only_hit_rejects_a_changed_lookup_plan() -> None:
    """The fast path cannot synthesize a hit from a stale pinned lookup."""
    engine = object.__new__(LMCacheEngine)
    engine.dsv4_optimized_kv = True
    engine._tutti_config = {"slot_mb": 1, "n_slots": 1}
    engine._dsv4_csa_attention_kv_prefetch_active = Mock(return_value=True)
    expected_key = Mock(spec=CacheEngineKey)
    stale_key = Mock(spec=CacheEngineKey)
    engine.lookup_pins = {"request": {"LocalDiskBackend": [stale_key]}}
    engine.token_database = SimpleNamespace(
        process_tokens=Mock(return_value=[(0, 256, expected_key)])
    )
    engine.storage_manager = SimpleNamespace(storage_backends={})

    with patch.dict(
        os.environ,
        {"LMCACHE_INDEXER_ENABLE_PREFETCH": "1"},
    ):
        assert (
            engine.prepare_dsv4_streaming_retrieve(
                torch.arange(256),
                torch.ones(256, dtype=torch.bool),
                req_id="request",
            )
            is None
        )


def test_streaming_only_hit_uses_rank_local_metadata_without_lookup_pins() -> None:
    """Non-scheduler TP ranks validate their own admitted streaming plan."""
    engine = object.__new__(LMCacheEngine)
    engine.dsv4_optimized_kv = True
    engine._tutti_config = {"slot_mb": 1, "n_slots": 1}
    engine._dsv4_csa_attention_kv_prefetch_active = Mock(return_value=True)
    engine._dsv4_streaming_compact_payload_nbytes = Mock(return_value=0)
    engine._ensure_tutti_loader = Mock(return_value=True)
    engine._register_csa_attention_kv_chunks = Mock(return_value=(True, True, True))
    key = Mock(spec=CacheEngineKey)
    disk_metadata = object()
    engine.lookup_pins = {}
    engine.token_database = SimpleNamespace(
        process_tokens=Mock(return_value=[(0, 256, key)])
    )
    engine.storage_manager = SimpleNamespace(
        storage_backends={
            "LocalDiskBackend": SimpleNamespace(dict={key: disk_metadata})
        }
    )

    with patch.dict(
        os.environ,
        {
            "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
            "LMCACHE_DSV4_HCA_WALKER": "1",
        },
    ):
        ret_mask = engine.prepare_dsv4_streaming_retrieve(
            torch.arange(256),
            torch.ones(256, dtype=torch.bool),
            req_id="request",
            request_total_tokens=8448,
        )

    assert ret_mask is not None
    assert ret_mask.all()
    engine._register_csa_attention_kv_chunks.assert_called_once()


def test_layer_chunk_map_requires_dense_equal_coverage() -> None:
    """Unified CSA/HCA registration rejects gaps and unequal layer ends."""

    def chunk(start: int, end: int) -> SimpleNamespace:
        return SimpleNamespace(
            first_compressed_block=start,
            end_compressed_block=end,
        )

    complete = {
        10: [chunk(0, 2), chunk(2, 4)],
        20: [chunk(0, 4)],
    }

    assert LMCacheEngine._dsv4_layer_chunk_map_complete(
        complete,
        (10, 20),
        4,
    )
    assert not LMCacheEngine._dsv4_layer_chunk_map_complete(
        {10: [chunk(0, 2), chunk(3, 4)], 20: [chunk(0, 4)]},
        (10, 20),
        4,
    )
    assert not LMCacheEngine._dsv4_layer_chunk_map_complete(
        {10: [chunk(0, 4)], 20: [chunk(0, 3)]},
        (10, 20),
        4,
    )
    assert not LMCacheEngine._dsv4_layer_chunk_map_complete(
        complete,
        (10, 20),
        5,
    )
    assert not LMCacheEngine._dsv4_layer_chunk_map_complete(
        {**complete, 30: [chunk(0, 4)]},
        (10, 20),
        4,
    )


def test_unified_registration_reports_all_streams_ready() -> None:
    """Compact retrieve is enabled only after every split consumer is ready."""
    engine = object.__new__(LMCacheEngine)
    commit_order: list[str] = []
    disk_meta = SimpleNamespace(path="tutti://rank0-full")
    csa_chunk = SimpleNamespace(
        first_compressed_block=0,
        end_compressed_block=4,
        disk_meta=disk_meta,
        raw_extents=((0, 100, 8),),
    )
    hca_chunk = SimpleNamespace(
        first_compressed_block=0,
        end_compressed_block=8,
        disk_meta=disk_meta,
        raw_extents=((4096, 108, 8),),
    )
    indexer_chunk = SimpleNamespace(
        disk_meta=disk_meta,
        raw_extents=((8192, 116, 8),),
    )
    engine._dsv4_build_indexer_cache_chunks = Mock(return_value={30: [indexer_chunk]})
    engine._dsv4_build_csa_attention_kv_chunks = Mock(return_value={10: [csa_chunk]})
    engine._dsv4_build_hca_attention_kv_chunks = Mock(return_value={20: [hca_chunk]})
    engine._dsv4_streaming_expected_layer_coverage = Mock(side_effect=[4, 8])

    manager = SimpleNamespace(
        active_request_id="old-request",
        csa_layer_ids=(10,),
        hca_layer_ids=(20,),
        deactivate_request=Mock(
            side_effect=lambda: commit_order.append("deactivate_csa") or True
        ),
        start_full_nsys_capture_for_request=Mock(),
        register_request_chunks=Mock(
            side_effect=lambda *_args, **_kwargs: commit_order.append("manager")
        ),
    )
    indexer_manager = SimpleNamespace(
        deactivate_csa_predictions=Mock(
            side_effect=lambda: commit_order.append("deactivate_predictions")
            or True
        ),
        deactivate_native_indexer_stream=Mock(
            side_effect=lambda: commit_order.append("deactivate_indexer") or True
        ),
        register_native_indexer_stream=Mock(
            side_effect=lambda *_args, **_kwargs: commit_order.append("indexer") or True
        ),
        fire_async_for_layers=Mock(
            side_effect=lambda *_args, **_kwargs: commit_order.append("prefire_hca")
        ),
    )

    with (
        patch.dict(
            os.environ,
            {
                "LMCACHE_DSV4_HCA_WALKER": "1",
                "LMCACHE_HCA_PREFIRE_FIRST_LAYER": "1",
            },
        ),
        patch(
            "lmcache.v1.csa_attention_kv_prefetch_manager."
            "get_csa_attention_kv_prefetch_manager",
            return_value=manager,
        ),
        patch(
            "lmcache.v1.indexer_ssd_manager.get_indexer_ssd_manager",
            return_value=indexer_manager,
        ),
    ):
        readiness = engine._register_csa_attention_kv_chunks(
            [(Mock(), 0, 256)],
            [None],
            256,
            "req",
            slot_mapping=Mock(),
        )

    assert readiness == (True, True, True)
    manager_call = manager.register_request_chunks.call_args
    indexer_call = indexer_manager.register_native_indexer_stream.call_args
    assert manager_call.args == ("req", {10: [csa_chunk], 20: [hca_chunk]})
    assert indexer_call.args == ("req", {30: [indexer_chunk]})
    assert (
        manager_call.kwargs["shared_raw_lba_cache"]
        is indexer_call.kwargs["shared_raw_lba_cache"]
    )
    assert commit_order == [
        "deactivate_predictions",
        "deactivate_indexer",
        "deactivate_csa",
        "indexer",
        "manager",
        "prefire_hca",
    ]
    indexer_manager.fire_async_for_layers.assert_called_once_with(
        (20,), source_layer_id=-1
    )


def test_unified_registration_does_not_start_partial_plan() -> None:
    """An incomplete HCA plan is rejected before indexer Stage0 starts."""
    engine = object.__new__(LMCacheEngine)
    csa_chunk = SimpleNamespace(first_compressed_block=0, end_compressed_block=4)
    engine._dsv4_build_indexer_cache_chunks = Mock(return_value={30: [Mock()]})
    engine._dsv4_build_csa_attention_kv_chunks = Mock(return_value={10: [csa_chunk]})
    engine._dsv4_build_hca_attention_kv_chunks = Mock(return_value={})
    engine._dsv4_streaming_expected_layer_coverage = Mock(side_effect=[4, 8])

    manager = SimpleNamespace(
        csa_layer_ids=(10,),
        hca_layer_ids=(20,),
        deactivate_request=Mock(return_value=True),
        start_full_nsys_capture_for_request=Mock(),
        register_request_chunks=Mock(),
    )
    register_indexer = Mock(return_value=True)
    indexer_manager = SimpleNamespace(
        deactivate_native_indexer_stream=Mock(return_value=True),
        register_native_indexer_stream=register_indexer,
    )

    with (
        patch.dict(os.environ, {"LMCACHE_DSV4_HCA_WALKER": "1"}),
        patch(
            "lmcache.v1.csa_attention_kv_prefetch_manager."
            "get_csa_attention_kv_prefetch_manager",
            return_value=manager,
        ),
        patch(
            "lmcache.v1.indexer_ssd_manager.get_indexer_ssd_manager",
            return_value=indexer_manager,
        ),
    ):
        readiness = engine._register_csa_attention_kv_chunks(
            [(Mock(), 0, 256)],
            [None],
            256,
            "req",
            slot_mapping=Mock(),
        )

    assert readiness == (False, False, False)
    register_indexer.assert_not_called()
    indexer_manager.deactivate_native_indexer_stream.assert_called_once()
    manager.deactivate_request.assert_called_once()
    manager.start_full_nsys_capture_for_request.assert_not_called()
    manager.register_request_chunks.assert_not_called()


def test_unified_registration_rejects_changed_repeat_plan_before_commit() -> None:
    """A repeated request cannot reuse manager state for a different prefix."""
    engine = object.__new__(LMCacheEngine)
    disk_meta = SimpleNamespace(path="tutti://rank0-full")
    csa_chunk = SimpleNamespace(
        first_compressed_block=0,
        end_compressed_block=4,
        disk_meta=disk_meta,
        raw_extents=((0, 100, 8),),
    )
    hca_chunk = SimpleNamespace(
        first_compressed_block=0,
        end_compressed_block=8,
        disk_meta=disk_meta,
        raw_extents=((4096, 108, 8),),
    )
    engine._dsv4_build_indexer_cache_chunks = Mock(
        return_value={
            30: [
                SimpleNamespace(
                    disk_meta=disk_meta,
                    raw_extents=((8192, 116, 8),),
                )
            ]
        }
    )
    engine._dsv4_build_csa_attention_kv_chunks = Mock(return_value={10: [csa_chunk]})
    engine._dsv4_build_hca_attention_kv_chunks = Mock(return_value={20: [hca_chunk]})
    engine._dsv4_streaming_expected_layer_coverage = Mock(side_effect=[4, 8])

    manager = SimpleNamespace(
        active_request_id="req",
        csa_layer_ids=(10,),
        hca_layer_ids=(20,),
        request_chunks_match=Mock(return_value=False),
        deactivate_request=Mock(return_value=True),
        start_full_nsys_capture_for_request=Mock(),
        register_request_chunks=Mock(),
    )
    register_indexer = Mock(return_value=True)
    indexer_manager = SimpleNamespace(
        native_indexer_stream_matches=Mock(return_value=True),
        deactivate_native_indexer_stream=Mock(return_value=True),
        register_native_indexer_stream=register_indexer,
    )

    with (
        patch.dict(os.environ, {"LMCACHE_DSV4_HCA_WALKER": "1"}),
        patch(
            "lmcache.v1.csa_attention_kv_prefetch_manager."
            "get_csa_attention_kv_prefetch_manager",
            return_value=manager,
        ),
        patch(
            "lmcache.v1.indexer_ssd_manager.get_indexer_ssd_manager",
            return_value=indexer_manager,
        ),
    ):
        readiness = engine._register_csa_attention_kv_chunks(
            [(Mock(), 0, 256)],
            [None],
            256,
            "req",
            slot_mapping=Mock(),
        )

    assert readiness == (False, False, False)
    manager.request_chunks_match.assert_called_once()
    indexer_manager.native_indexer_stream_matches.assert_not_called()
    indexer_manager.deactivate_native_indexer_stream.assert_called_once()
    manager.deactivate_request.assert_called_once()
    register_indexer.assert_not_called()
    manager.register_request_chunks.assert_not_called()
