# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
import asyncio
import os
import shutil
import tempfile
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import _parse_local_disk
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo
from lmcache.v1.kv_object_store import KVObjectId, KVObjectRecord, KVObjectState
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import (
    LocalDiskBackend,
    _LayerMajorSidecar,
    _pack_layer_major_wave_on_gpu,
    _pack_layer_major_wave_with_fallback,
    _write_wave_limit_bytes,
)


def test_write_quantum_defaults_to_legacy_wave_setting() -> None:
    with patch.dict(
        os.environ,
        {"LMCACHE_DSV4_RAW_WRITE_WAVE_MB": "64"},
        clear=False,
    ):
        os.environ.pop("LMCACHE_DSV4_WRITE_QUANTUM_MB", None)
        assert _write_wave_limit_bytes(256 * 1024**2) == 64 * 1024**2


def test_write_quantum_overrides_legacy_wave_without_changing_staging() -> None:
    with patch.dict(
        os.environ,
        {
            "LMCACHE_DSV4_RAW_WRITE_WAVE_MB": "256",
            "LMCACHE_DSV4_WRITE_QUANTUM_MB": "32",
        },
    ):
        assert _write_wave_limit_bytes(256 * 1024**2) == 32 * 1024**2


def _gpu_pack_test_wave() -> tuple[list[_LayerMajorSidecar], dict[int, bytes]]:
    first_source = bytes(range(32))
    second_source = bytes(range(64, 96))
    object_id = KVObjectId(
        model_id="model",
        parallel_config_id="tp1",
        rank=0,
        layer_id=0,
        role="csa",
        block_id="first",
    )
    first_record = KVObjectRecord(
        object_id=object_id,
        pool_id="pool",
        offset=0,
        length=8,
        aligned_length=16,
        shape=(8,),
        dtype="torch.uint8",
    )
    second_record = KVObjectRecord(
        object_id=KVObjectId(
            model_id="model",
            parallel_config_id="tp1",
            rank=0,
            layer_id=1,
            role="csa",
            block_id="second",
        ),
        pool_id="pool",
        offset=16,
        length=8,
        aligned_length=16,
        shape=(8,),
        dtype="torch.uint8",
    )
    return (
        [
            _LayerMajorSidecar(
                source_role="csa_attention_kv",
                layer_id=0,
                object_role="csa",
                physical_role="csa-generation",
                payload_nbytes=8,
                segments=[
                    memoryview(first_source)[2:5],
                    memoryview(first_source)[8:13],
                ],
                segment_sources=[(100, 2), (100, 8)],
                record=first_record,
            ),
            _LayerMajorSidecar(
                source_role="csa_attention_kv",
                layer_id=1,
                object_role="csa",
                physical_role="csa-generation",
                payload_nbytes=8,
                segments=[
                    memoryview(second_source)[1:6],
                    memoryview(second_source)[10:13],
                ],
                segment_sources=[(200, 1), (200, 10)],
                record=second_record,
            ),
        ],
        {100: first_source, 200: second_source},
    )


def test_gpu_layer_major_pack_preserves_order_and_padding() -> None:
    wave, sources = _gpu_pack_test_wave()
    destination = torch.full((32,), 255, dtype=torch.uint8)

    def packer(
        source_host_ptrs: list[int],
        source_indices: list[int],
        source_offsets: list[int],
        destination_offsets: list[int],
        lengths: list[int],
        target: torch.Tensor,
    ) -> float:
        target.zero_()
        target_view = memoryview(target.numpy())
        for source_idx, source_offset, target_offset, length in zip(
            source_indices,
            source_offsets,
            destination_offsets,
            lengths,
            strict=True,
        ):
            source = sources[source_host_ptrs[source_idx]]
            target_view[target_offset : target_offset + length] = source[
                source_offset : source_offset + length
            ]
        return 0.1

    record, payload, _elapsed = _pack_layer_major_wave_on_gpu(
        wave,
        destination,
        packer,
    )
    assert record.offset == 0
    assert record.aligned_length == 32
    assert bytes(payload[:8]) == sources[100][2:5] + sources[100][8:13]
    assert bytes(payload[8:16]) == bytes(8)
    assert bytes(payload[16:24]) == sources[200][1:6] + sources[200][10:13]
    assert bytes(payload[24:32]) == bytes(8)


def test_gpu_layer_major_pack_failure_falls_back_to_cpu() -> None:
    wave, sources = _gpu_pack_test_wave()
    destination = torch.empty(32, dtype=torch.uint8)

    def failing_packer(*_args: object) -> float:
        raise RuntimeError("synthetic CUDA failure")

    _record, payload, _elapsed = _pack_layer_major_wave_with_fallback(
        wave,
        bytearray(32),
        destination,
        failing_packer,
    )
    assert bytes(payload[:8]) == sources[100][2:5] + sources[100][8:13]
    assert bytes(payload[8:16]) == bytes(8)
    assert bytes(payload[16:24]) == sources[200][1:6] + sources[200][10:13]
    assert bytes(payload[24:32]) == bytes(8)


class MockLookupServer:
    def __init__(self):
        self.removed_keys = []
        self.inserted_keys = []

    def batched_remove(self, keys):
        self.removed_keys.extend(keys)

    def batched_insert(self, keys):
        self.inserted_keys.extend(keys)


class MockLMCacheWorker:
    def __init__(self):
        self.messages = []

    def put_msg(self, msg):
        self.messages.append(msg)


def create_test_config(
    disk_path: str,
    max_disk_size: float = 1.0,
    local_disk_path_sharding: str = "by_gpu",
):
    """Create a test configuration for LocalDiskBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=disk_path,
        local_disk_path_sharding=local_disk_path_sharding,
        max_local_disk_size=max_disk_size,
        lmcache_instance_id="test_instance",
    )
    return config


def create_test_metadata():
    """Create a test metadata for LMCacheMetadata."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


def _make_shape_desc(
    *,
    kv_size: int,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    dtype: torch.dtype,
) -> Any:
    """Build a PageBufferShapeDesc for object-store indexing tests."""
    return SimpleNamespace(
        kv_size=kv_size,
        nl=nl,
        nb=nb,
        bs=bs,
        nh=nh,
        hs=hs,
        element_size=dtype.itemsize,
        block_stride_elems=0,
    )


def create_object_store_metadata() -> LMCacheMetadata:
    """Create metadata with a CSA layer group."""
    metadata = create_test_metadata()
    csa_group = KVLayerGroupInfo(
        layer_indices=[0, 1],
        shape_desc=_make_shape_desc(
            kv_size=2,
            nl=2,
            nb=1,
            bs=512,
            nh=1,
            hs=584,
            dtype=torch.uint8,
        ),
        dtype=torch.uint8,
        compress_ratio=4,
        physical_chunk_size=512,
    )
    metadata.kv_layer_groups_manager = SimpleNamespace(
        kv_layer_groups=[csa_group],
        num_groups=1,
    )
    return metadata


def create_indexer_object_store_metadata() -> LMCacheMetadata:
    """Create metadata with a compact CSA indexer-cache layer group."""
    metadata = create_test_metadata()
    indexer_group = KVLayerGroupInfo(
        layer_indices=[0, 1],
        shape_desc=_make_shape_desc(
            kv_size=1,
            nl=2,
            nb=1,
            bs=8,
            nh=1,
            hs=132,
            dtype=torch.uint8,
        ),
        dtype=torch.uint8,
        compress_ratio=4,
        physical_chunk_size=8,
    )
    metadata.kv_layer_groups_manager = SimpleNamespace(
        kv_layer_groups=[indexer_group],
        num_groups=1,
    )
    return metadata


def create_streaming_object_store_metadata() -> LMCacheMetadata:
    """Create metadata containing generic, CSA, and indexer cache groups."""
    metadata = create_test_metadata()
    generic_group = KVLayerGroupInfo(
        layer_indices=[0],
        shape_desc=_make_shape_desc(
            kv_size=2,
            nl=1,
            nb=1,
            bs=8,
            nh=1,
            hs=4,
            dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        compress_ratio=1,
        physical_chunk_size=8,
    )
    csa_group = KVLayerGroupInfo(
        layer_indices=[0],
        shape_desc=_make_shape_desc(
            kv_size=2,
            nl=1,
            nb=1,
            bs=8,
            nh=1,
            hs=584,
            dtype=torch.uint8,
        ),
        dtype=torch.uint8,
        compress_ratio=4,
        physical_chunk_size=8,
    )
    indexer_group = KVLayerGroupInfo(
        layer_indices=[0],
        shape_desc=_make_shape_desc(
            kv_size=1,
            nl=1,
            nb=1,
            bs=8,
            nh=1,
            hs=132,
            dtype=torch.uint8,
        ),
        dtype=torch.uint8,
        compress_ratio=4,
        physical_chunk_size=8,
    )
    metadata.kv_layer_groups_manager = SimpleNamespace(
        kv_layer_groups=[generic_group, csa_group, indexer_group],
        num_groups=3,
    )
    return metadata


def create_streaming_hca_object_store_metadata() -> LMCacheMetadata:
    """Create metadata for the canonical generic/CSA/HCA/indexer layout."""
    metadata = create_streaming_object_store_metadata()
    hca_group = KVLayerGroupInfo(
        layer_indices=[0],
        shape_desc=_make_shape_desc(
            kv_size=1,
            nl=1,
            nb=1,
            bs=8,
            nh=1,
            hs=584,
            dtype=torch.uint8,
        ),
        dtype=torch.uint8,
        compress_ratio=128,
        physical_chunk_size=8,
    )
    manager = metadata.kv_layer_groups_manager
    assert manager is not None
    manager.kv_layer_groups.append(hca_group)
    manager.num_groups = len(manager.kv_layer_groups)
    return metadata


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


@pytest.fixture
def temp_disk_path():
    """Create a temporary directory for disk storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def async_loop():
    """Create an asyncio event loop for testing."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ----------------------------------------------------------------------------


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    return LocalCPUBackend(config, memory_allocator=memory_allocator)


@pytest.fixture
def local_disk_backend(temp_disk_path, async_loop, local_cpu_backend):
    """Create a LocalDiskBackend for testing."""
    config = create_test_config(temp_disk_path)
    return LocalDiskBackend(
        config=config,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cuda:0",
    )


class TestLocalDiskBackend:
    """Test cases for LocalDiskBackend."""

    def test_init(self, temp_disk_path, async_loop, local_cpu_backend):
        """Test LocalDiskBackend initialization."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )

        assert backend.dst_device == "cuda:0"
        assert backend.local_cpu_backend == local_cpu_backend
        assert backend.path == temp_disk_path
        assert os.path.exists(temp_disk_path)
        assert backend.lmcache_worker is None
        assert backend.instance_id == "test_instance"
        assert backend.usage == 0
        assert len(backend.dict) == 0

        local_cpu_backend.memory_allocator.close()

    def test_init_with_lookup_server_and_worker(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Test LocalDiskBackend initialization with lookup server and worker."""
        config = create_test_config(temp_disk_path)
        lmcache_worker = MockLMCacheWorker()

        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
            lmcache_worker=lmcache_worker,
        )

        assert backend.lmcache_worker == lmcache_worker

        local_cpu_backend.memory_allocator.close()

    def test_str(self, local_disk_backend):
        """Test string representation."""
        assert str(local_disk_backend) == "LocalDiskBackend"
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_to_path(self, local_disk_backend):
        """Test key to path conversion."""
        key = create_test_key(1)
        path = local_disk_backend._key_to_path(key)

        expected_filename = key.to_string().replace("/", "-") + ".pt"
        assert path == os.path.join(local_disk_backend.path, expected_filename)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_contains_key_not_exists(self, local_disk_backend):
        """Test contains() when key doesn't exist."""
        key = create_test_key(2)
        assert not local_disk_backend.contains(key)
        assert not local_disk_backend.contains(key, pin=True)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_get_blocking_key_not_exists(self, local_disk_backend):
        """Test get_blocking() when key doesn't exist."""
        key = create_test_key(2)
        result = local_disk_backend.get_blocking(key)

        assert result is None

        local_disk_backend.local_cpu_backend.memory_allocator.close()


class TestKVObjectStoreLocalDiskBackend:
    """Tests for LocalDiskBackend KV object-store control-plane metadata."""

    def _create_object_store_backend(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> LocalDiskBackend:
        """Create a LocalDiskBackend with the KV object store enabled."""
        config = create_test_config(temp_disk_path)
        config.extra_config = {
            "kv_object_store_enable": True,
            "kv_object_store_slot_mb": 2,
            "kv_object_store_capacity": 4,
        }
        return LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
            metadata=create_object_store_metadata(),
        )

    def _allocate_object_store_memory(self) -> MemoryObj:
        """Allocate one CSA memory object."""
        allocator = AdHocMemoryAllocator(device="cpu")
        memory_obj = allocator.allocate(
            [torch.Size([2, 2, 512, 584])],
            [torch.uint8],
            fmt=MemoryFormat.KV_2LTD,
        )
        assert memory_obj is not None
        assert memory_obj.raw_tensor is not None
        memory_obj.raw_tensor.fill_(7)
        return memory_obj

    def test_raw_pool_reset_invalidates_recovered_lookup_metadata(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """Rewinding raw LBAs must make old cache and object records unreachable."""
        with patch(
            "os.statvfs",
            return_value=SimpleNamespace(f_bsize=4096),
            create=True,
        ):
            backend = self._create_object_store_backend(
                temp_disk_path,
                async_loop,
                local_cpu_backend,
            )
        key = create_test_key(91)
        backend.insert_key(
            key,
            size=4096,
            shape=torch.Size([4096]),
            dtype=torch.uint8,
            fmt=MemoryFormat.KV_2LTD,
        )
        assert backend.kv_object_pool_layout is not None
        assert backend.kv_object_metadata_store is not None
        object_id = KVObjectId(
            model_id="test_model",
            parallel_config_id="world1",
            rank=0,
            layer_id=0,
            role="full",
            block_id="recovered",
        )
        record = backend.kv_object_pool_layout.allocate(
            object_id,
            length=4096,
            shape=(4096,),
            dtype="torch.uint8",
        ).mark_ready()
        backend.kv_object_metadata_store.put(record)

        backend.reset_kv_object_pool_allocation()

        assert not backend.contains(key)
        assert backend.kv_object_metadata_store.get(object_id) is None
        assert backend.kv_object_pool_layout.next_allocation_bounds(4096)[0] == 0
        backend.local_cpu_backend.memory_allocator.close()

    def _create_streaming_object_store_backend(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> LocalDiskBackend:
        """Create a backend configured for canonical CSA streaming objects."""
        config = create_test_config(temp_disk_path)
        config.extra_config = {
            "kv_object_store_enable": True,
            "kv_object_store_slot_mb": 2,
            "kv_object_store_capacity": 16,
        }
        return LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
            metadata=create_streaming_object_store_metadata(),
        )

    @staticmethod
    def _allocate_streaming_object_store_memory() -> MemoryObj:
        """Allocate one generic/CSA/indexer memory object."""
        allocator = AdHocMemoryAllocator(device="cpu")
        memory_obj = allocator.allocate(
            [
                torch.Size([2, 1, 8, 4]),
                torch.Size([2, 1, 8, 584]),
                torch.Size([1, 1, 8, 132]),
            ],
            [torch.bfloat16, torch.uint8, torch.uint8],
            fmt=MemoryFormat.KV_2LTD,
        )
        assert memory_obj is not None
        for group_idx, value in enumerate((1, 2, 3)):
            tensor = memory_obj.get_tensor(group_idx)
            assert tensor is not None
            tensor.fill_(value)
        return memory_obj

    @staticmethod
    def _allocate_streaming_hca_object_store_memory() -> MemoryObj:
        """Allocate one generic/CSA/indexer/HCA memory object."""
        allocator = AdHocMemoryAllocator(device="cpu")
        memory_obj = allocator.allocate(
            [
                torch.Size([2, 1, 8, 4]),
                torch.Size([2, 1, 8, 584]),
                torch.Size([1, 1, 8, 132]),
                torch.Size([1, 1, 8, 584]),
            ],
            [torch.bfloat16, torch.uint8, torch.uint8, torch.uint8],
            fmt=MemoryFormat.KV_2LTD,
        )
        assert memory_obj is not None
        for group_idx, value in enumerate((1, 2, 3, 4)):
            tensor = memory_obj.get_tensor(group_idx)
            assert tensor is not None
            tensor.fill_(value)
        return memory_obj

    @staticmethod
    def _allocate_streaming_hca_memory(
        allocator: AdHocMemoryAllocator,
        *,
        ordinary_rows: int,
        streamed_rows: int,
    ) -> MemoryObj:
        """Allocate a streaming object with explicit group row counts."""
        memory_obj = allocator.allocate(
            [
                torch.Size([2, 1, ordinary_rows, 4]),
                torch.Size([2, 1, streamed_rows, 584]),
                torch.Size([1, 1, streamed_rows, 132]),
                torch.Size([1, 1, streamed_rows, 584]),
            ],
            [torch.bfloat16, torch.uint8, torch.uint8, torch.uint8],
            fmt=MemoryFormat.KV_2LTD,
        )
        assert memory_obj is not None
        for group_idx, value in enumerate((1, 2, 3, 4)):
            tensor = memory_obj.get_tensor(group_idx)
            assert tensor is not None
            tensor.fill_(value)
        return memory_obj

    def test_csa_streaming_store_publishes_compact_object_last(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """CSA admission publishes only after all required objects are ready."""
        with patch(
            "os.statvfs",
            return_value=SimpleNamespace(f_bsize=4096),
            create=True,
        ):
            backend = self._create_streaming_object_store_backend(
                temp_disk_path,
                async_loop,
                local_cpu_backend,
            )
        memory_obj = self._allocate_streaming_object_store_memory()
        key = create_test_key(499)

        with patch.dict(
            os.environ,
            {
                "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
                "LMCACHE_DSV4_HCA_WALKER": "0",
            },
        ):
            assert (
                backend.store_attention_layer_major_snapshot(
                    key,
                    [memory_obj],
                    prefix_keys=[key],
                )
                == 2
            )
            backend.async_save_bytes_to_disk(key, memory_obj)
            backend.kv_object_tutti_raw_enabled = True
            assert backend.contains(key)
            assert backend.get_kv_object_records([key])[0] is None
            compact = backend.get_kv_object_records(
                [key],
                roles=["csa_deferred_retrieve"],
            )[0]
            assert compact is not None
            assert compact.state == KVObjectState.READY
            expected_compact_bytes = 2 * 1 * 8 * 4 * 2
            assert compact.length == expected_compact_bytes
            assert backend.get_csa_layer_major_records(key, [0])[0] is not None
            assert backend.get_indexer_layer_major_records(key, [0])[0] is not None

        backend.local_cpu_backend.memory_allocator.close()

    def test_csa_streaming_store_waits_for_raw_writer_ready(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """Cold admission waits for a concurrently initialising raw writer."""
        with patch(
            "os.statvfs",
            return_value=SimpleNamespace(f_bsize=4096),
            create=True,
        ):
            backend = self._create_streaming_object_store_backend(
                temp_disk_path,
                async_loop,
                local_cpu_backend,
            )
        backend.kv_object_tutti_raw_cold_store_enabled = True
        backend.set_kv_object_tutti_raw_region_extents([(0, 1000, 32768)])
        memory_obj = self._allocate_streaming_object_store_memory()
        key = create_test_key(498)

        def install_writer() -> None:
            backend.set_kv_object_tutti_raw_writer(
                lambda record, _buffer: (
                    [
                        (
                            record.offset,
                            1000 + record.offset // 512,
                            record.aligned_length // 512,
                        )
                    ],
                    0.0,
                )
            )

        timer = threading.Timer(0.05, install_writer)
        with patch.dict(
            os.environ,
            {
                "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
                "LMCACHE_DSV4_HCA_WALKER": "0",
                "LMCACHE_KV_OBJECT_STORE_RAW_WRITER_READY_TIMEOUT_SEC": "1",
            },
        ):
            start = time.perf_counter()
            timer.start()
            try:
                assert (
                    backend.store_attention_layer_major_snapshot(
                        key,
                        [memory_obj],
                        prefix_keys=[key],
                    )
                    == 2
                )
            finally:
                timer.join()
            assert time.perf_counter() - start >= 0.04

        assert backend.wait_for_kv_object_tutti_raw_writer(0.0)
        backend.local_cpu_backend.memory_allocator.close()

    def test_csa_streaming_store_does_not_publish_partial_layout(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """A missing layer-major object makes the stored key a raw-cache miss."""
        with patch(
            "os.statvfs",
            return_value=SimpleNamespace(f_bsize=4096),
            create=True,
        ):
            backend = self._create_streaming_object_store_backend(
                temp_disk_path,
                async_loop,
                local_cpu_backend,
            )
        memory_obj = self._allocate_streaming_object_store_memory()
        key = create_test_key(500)

        with patch.dict(
            os.environ,
            {
                "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
                "LMCACHE_DSV4_HCA_WALKER": "0",
            },
        ):
            backend.async_save_bytes_to_disk(key, memory_obj)
            backend.kv_object_tutti_raw_enabled = True
            assert not backend.contains(key)
            assert not backend.contains_streaming_terminal(key, 256)

            assert (
                backend.get_kv_object_records(
                    [key],
                    roles=["csa_deferred_retrieve"],
                )[0]
                is None
            )

        backend.local_cpu_backend.memory_allocator.close()

    def test_csa_streaming_store_publishes_every_chunk_alias(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """One admission batch atomically publishes each chunk-prefix alias."""
        with patch(
            "os.statvfs",
            return_value=SimpleNamespace(f_bsize=4096),
            create=True,
        ):
            backend = self._create_streaming_object_store_backend(
                temp_disk_path,
                async_loop,
                local_cpu_backend,
            )
        first = self._allocate_streaming_object_store_memory()
        second = self._allocate_streaming_object_store_memory()
        first_key = create_test_key(501)
        second_key = create_test_key(502)

        with patch.dict(
            os.environ,
            {
                "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
                "LMCACHE_DSV4_HCA_WALKER": "0",
            },
        ):
            assert (
                backend.store_attention_layer_major_snapshot(
                    second_key,
                    [first, second],
                    prefix_keys=[first_key, second_key],
                    prefix_token_count=512,
                )
                == 2
            )
            # Retrying admission before the compact-main write must reuse the
            # immutable layer objects and reconstruct the pending manifests.
            assert (
                backend.store_attention_layer_major_snapshot(
                    second_key,
                    [first, second],
                    prefix_keys=[first_key, second_key],
                    prefix_token_count=512,
                )
                == 2
            )
            assert backend.kv_object_metadata_store is not None
            assert len(backend.kv_object_metadata_store.ready_records()) == 2
            backend.async_save_bytes_to_disk(first_key, first)
            backend.async_save_bytes_to_disk(second_key, second)
            backend.kv_object_tutti_raw_enabled = True

            assert backend.contains(first_key)
            assert backend.contains(second_key)
            assert backend.contains_streaming_terminal(first_key, 256)
            assert not backend.contains_streaming_terminal(first_key, 512)
            assert backend.contains_streaming_terminal(second_key, 512, pin=True)
            assert backend.unpin(second_key)
            first_csa = backend.get_csa_layer_major_records(first_key, [0])[0]
            second_csa = backend.get_csa_layer_major_records(second_key, [0])[0]
            assert first_csa is not None
            assert second_csa is not None
            assert second_csa.length == 2 * first_csa.length
            with patch.dict(
                os.environ,
                {"LMCACHE_DSV4_HCA_WALKER": "1"},
            ):
                assert not backend.contains(first_key)
                assert not backend.contains(second_key)
                assert not backend.contains_streaming_terminal(second_key, 512)

        backend.local_cpu_backend.memory_allocator.close()

    def test_csa_streaming_partial_hit_extends_ready_generation(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """A partial-hit suffix composes with its READY prefix generation."""
        with patch(
            "os.statvfs",
            return_value=SimpleNamespace(f_bsize=4096),
            create=True,
        ):
            backend = self._create_streaming_object_store_backend(
                temp_disk_path,
                async_loop,
                local_cpu_backend,
            )
        base = self._allocate_streaming_object_store_memory()
        suffix = self._allocate_streaming_object_store_memory()
        suffix_csa = suffix.get_tensor(1)
        suffix_indexer = suffix.get_tensor(2)
        assert suffix_csa is not None
        assert suffix_indexer is not None
        suffix_csa.fill_(12)
        suffix_indexer.fill_(13)
        base_key = create_test_key(511)
        suffix_key = create_test_key(512)

        with patch.dict(
            os.environ,
            {
                "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
                "LMCACHE_DSV4_HCA_WALKER": "0",
            },
        ):
            assert (
                backend.store_attention_layer_major_snapshot(
                    base_key,
                    [base],
                    prefix_keys=[base_key],
                    prefix_token_count=256,
                )
                == 2
            )
            backend.async_save_bytes_to_disk(base_key, base)
            assert backend.contains_streaming_terminal(base_key, 256)

            assert (
                backend.store_attention_layer_major_snapshot(
                    suffix_key,
                    [suffix],
                    prefix_keys=[suffix_key],
                    prefix_token_count=256,
                    base_prefix_key=base_key,
                    base_prefix_token_count=256,
                )
                == 2
            )
            backend.async_save_bytes_to_disk(suffix_key, suffix)

            assert backend.contains_streaming_terminal(suffix_key, 512)
            csa = backend.get_csa_layer_major_records(suffix_key, [0])[0]
            indexer = backend.get_indexer_layer_major_records(suffix_key, [0])[0]
            assert csa is not None
            assert indexer is not None
            assert len(csa.read_ranges) == 2
            assert len(indexer.read_ranges) == 2
            assert csa.aligned_length % 512 == 0
            assert indexer.aligned_length % 512 == 0
            assert backend.kv_object_pool_io is not None
            assert backend.kv_object_pool_io.read_object(csa) == bytes(
                [2] * (2 * 8 * 584) + [12] * (2 * 8 * 584)
            )
            assert backend.kv_object_pool_io.read_object(indexer) == bytes(
                [3] * (8 * 132) + [13] * (8 * 132)
            )

        backend.local_cpu_backend.memory_allocator.close()

    def test_csa_hca_streaming_hit_never_rewrites_layout(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """A READY CSA/HCA layout makes repeated admission write-free."""
        config = create_test_config(temp_disk_path)
        config.extra_config = {
            "kv_object_store_enable": True,
            "kv_object_store_slot_mb": 2,
            "kv_object_store_capacity": 16,
        }
        with patch(
            "os.statvfs",
            return_value=SimpleNamespace(f_bsize=4096),
            create=True,
        ):
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
                metadata=create_streaming_hca_object_store_metadata(),
            )
        memory_obj = self._allocate_streaming_hca_object_store_memory()
        key = create_test_key(503)

        with patch.dict(
            os.environ,
            {
                "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
                "LMCACHE_DSV4_HCA_WALKER": "1",
            },
        ):
            assert (
                backend.store_attention_layer_major_snapshot(
                    key,
                    [memory_obj],
                    prefix_keys=[key],
                )
                == 3
            )
            backend.async_save_bytes_to_disk(key, memory_obj)
            backend.kv_object_tutti_raw_enabled = True
            assert backend.contains(key)

            compact = backend.get_kv_object_records(
                [key],
                roles=["csa_hca_deferred_retrieve"],
            )[0]
            assert compact is not None
            assert compact.length == 2 * 1 * 8 * 4 * 2
            assert backend.get_csa_layer_major_records(key, [0])[0] is not None
            assert backend.get_hca_layer_major_records(key, [0])[0] is not None
            assert backend.get_indexer_layer_major_records(key, [0])[0] is not None

            retry_memory_obj = self._allocate_streaming_hca_object_store_memory()
            pool_layout = backend.kv_object_pool_layout
            pool_io = backend.kv_object_pool_io
            assert pool_layout is not None
            assert pool_io is not None
            next_offset_before = pool_layout.next_allocation_bounds(1)[0]
            with patch.object(
                pool_io,
                "write_object",
                wraps=pool_io.write_object,
            ) as write_object:
                assert (
                    backend.store_attention_layer_major_snapshot(
                        key,
                        [retry_memory_obj],
                        prefix_keys=[key],
                    )
                    == 3
                )
                backend.async_save_bytes_to_disk(key, retry_memory_obj)
                write_object.assert_not_called()
            assert pool_layout.next_allocation_bounds(1)[0] == next_offset_before

        backend.local_cpu_backend.memory_allocator.close()

    def test_csa_hca_streaming_publishes_empty_non_tail_main_chunks(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """Stream-only non-tail chunks publish metadata-only main entries."""
        config = create_test_config(temp_disk_path)
        config.extra_config = {
            "kv_object_store_enable": True,
            "kv_object_store_slot_mb": 2,
            "kv_object_store_capacity": 256,
        }
        with patch(
            "os.statvfs",
            return_value=SimpleNamespace(f_bsize=4096),
            create=True,
        ):
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
                metadata=create_streaming_hca_object_store_metadata(),
            )

        allocator = AdHocMemoryAllocator(device="cpu")
        keys = [create_test_key(600 + index) for index in range(128)]
        memory_objs = [
            self._allocate_streaming_hca_memory(
                allocator,
                ordinary_rows=0,
                streamed_rows=8,
            )
            for _key in keys[:-1]
        ]
        memory_objs.append(
            self._allocate_streaming_hca_memory(
                allocator,
                ordinary_rows=8,
                streamed_rows=8,
            )
        )

        with patch.dict(
            os.environ,
            {
                "LMCACHE_INDEXER_ENABLE_PREFETCH": "1",
                "LMCACHE_DSV4_HCA_WALKER": "1",
            },
        ):
            assert (
                backend.store_attention_layer_major_snapshot(
                    keys[-1],
                    memory_objs,
                    prefix_keys=keys,
                )
                == 3
            )
            for key, memory_obj in zip(keys, memory_objs, strict=True):
                backend.async_save_bytes_to_disk(key, memory_obj)

            backend.kv_object_tutti_raw_enabled = True
            metadata_store = backend.kv_object_metadata_store
            assert metadata_store is not None
            with patch.object(
                metadata_store,
                "get",
                wraps=metadata_store.get,
            ) as metadata_get:
                assert all(backend.contains(key) for key in keys)
                assert backend.get_kv_object_payload_lengths(
                    keys,
                    roles=["csa_hca_deferred_retrieve"] * len(keys),
                ) == [0] * 127 + [2 * 1 * 8 * 4 * 2]
                metadata_get.assert_not_called()
            compact_records = backend.get_kv_object_records(
                keys,
                roles=["csa_hca_deferred_retrieve"] * len(keys),
            )
            assert compact_records[:-1] == [None] * 127
            assert compact_records[-1] is not None
            assert compact_records[-1].length == 2 * 1 * 8 * 4 * 2

        allocator.close()
        backend.local_cpu_backend.memory_allocator.close()

    def test_write_indexes_layer_role_views(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """Writing a full object also indexes CSA per-layer views."""
        backend = self._create_object_store_backend(
            temp_disk_path,
            async_loop,
            local_cpu_backend,
        )
        key = create_test_key(201)
        memory_obj = self._allocate_object_store_memory()

        backend.async_save_bytes_to_disk(
            key,
            memory_obj,
        )

        full_record = backend.get_kv_object_records([key])[0]
        csa0, csa1 = backend.get_kv_object_records(
            [key, key],
            layer_ids=[0, 1],
            roles=[
                "csa_attention_kv",
                "csa_attention_kv",
            ],
        )
        assert full_record is not None
        assert csa0 is not None
        assert csa1 is not None
        assert csa0.length == 2 * 512 * 584
        assert csa1.length == 2 * 512 * 584
        assert [byte_range.offset for byte_range in csa0.read_ranges] == [
            full_record.offset,
            full_record.offset + 2 * 512 * 584,
        ]
        assert [byte_range.offset for byte_range in csa1.read_ranges] == [
            full_record.offset + 512 * 584,
            full_record.offset + 3 * 512 * 584,
        ]

    def test_attention_layer_major_snapshot_gathers_chunks_in_token_order(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """Layer-major sidecars concatenate each layer across input chunks."""
        with patch(
            "os.statvfs",
            return_value=SimpleNamespace(f_bsize=4096),
            create=True,
        ):
            backend = self._create_object_store_backend(
                temp_disk_path,
                async_loop,
                local_cpu_backend,
            )
        allocator = AdHocMemoryAllocator(device="cpu")
        memory_objs: list[MemoryObj] = []
        for chunk_id in range(2):
            memory_obj = allocator.allocate(
                [torch.Size([2, 2, 8, 4])],
                [torch.uint8],
                fmt=MemoryFormat.KV_2LTD,
            )
            assert memory_obj is not None
            tensor = memory_obj.get_tensor(0)
            assert tensor is not None
            for kv_id in range(2):
                for layer_id in range(2):
                    tensor[kv_id, layer_id].fill_(chunk_id * 40 + kv_id * 10 + layer_id)
            memory_objs.append(memory_obj)

        first_key = create_test_key(298)
        prefix_key = create_test_key(299)
        assert (
            backend.store_attention_layer_major_snapshot(
                prefix_key,
                memory_objs,
                prefix_keys=[first_key, prefix_key],
            )
            == 2
        )
        records = backend.get_csa_layer_major_records(prefix_key, [0, 1])
        record0, record1 = records
        assert record0 is not None
        assert record1 is not None
        assert backend.kv_object_pool_io is not None
        payload0 = backend.kv_object_pool_io.read_object(record0)
        payload1 = backend.kv_object_pool_io.read_object(record1)
        # The supplied shapes already contain physical (compressed) rows.
        # Sidecar creation must not apply the compression ratio a second time.
        assert payload0 == bytes([0] * 32 + [10] * 32 + [40] * 32 + [50] * 32)
        assert payload1 == bytes([1] * 32 + [11] * 32 + [41] * 32 + [51] * 32)

        probed = backend.get_csa_layer_major_records_for_keys(
            [first_key, prefix_key],
            layer_id=0,
        )
        assert probed[0] is not None
        assert probed[0].offset == record0.offset
        assert probed[0].length == record0.length // 2
        assert probed[1] == record0

        backend.local_cpu_backend.memory_allocator.close()

    def test_attention_layer_major_raw_store_batches_contiguous_layers(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """Contiguous layer sidecars share one bounded raw-write wave."""
        with patch(
            "os.statvfs",
            return_value=SimpleNamespace(f_bsize=4096),
            create=True,
        ):
            backend = self._create_object_store_backend(
                temp_disk_path,
                async_loop,
                local_cpu_backend,
            )
        backend.kv_object_tutti_raw_cold_store_enabled = True
        backend.kv_object_tutti_raw_staging_bytes = 1024**2
        backend.set_kv_object_tutti_raw_region_extents([(0, 1000, 2048)])
        writes: list[tuple[KVObjectRecord, bytes]] = []

        def raw_writer(
            record: KVObjectRecord,
            payload: memoryview,
        ) -> tuple[list[tuple[int, int, int]], float]:
            writes.append((record, bytes(payload)))
            return [
                (
                    record.offset,
                    1000 + record.offset // 512,
                    record.aligned_length // 512,
                )
            ], 0.0

        backend.set_kv_object_tutti_raw_writer(raw_writer)
        allocator = AdHocMemoryAllocator(device="cpu")
        memory_objs: list[MemoryObj] = []
        for chunk_id in range(2):
            memory_obj = allocator.allocate(
                [torch.Size([2, 2, 8, 4])],
                [torch.uint8],
                fmt=MemoryFormat.KV_2LTD,
            )
            assert memory_obj is not None
            tensor = memory_obj.get_tensor(0)
            assert tensor is not None
            for kv_id in range(2):
                for layer_id in range(2):
                    tensor[kv_id, layer_id].fill_(chunk_id * 40 + kv_id * 10 + layer_id)
            memory_objs.append(memory_obj)

        first_key = create_test_key(296)
        prefix_key = create_test_key(297)
        with patch.dict(os.environ, {"LMCACHE_DSV4_RAW_WRITE_WAVE_MB": "1"}):
            assert (
                backend.store_attention_layer_major_snapshot(
                    prefix_key,
                    memory_objs,
                    prefix_keys=[first_key, prefix_key],
                )
                == 2
            )

        assert len(writes) == 1
        wave_record, wave_payload = writes[0]
        assert wave_record.offset == 0
        assert wave_record.aligned_length == 8192
        assert len(wave_payload) == 8192
        assert wave_payload[:128] == bytes([0] * 32 + [10] * 32 + [40] * 32 + [50] * 32)
        assert wave_payload[128:4096] == bytes(4096 - 128)
        assert wave_payload[4096:4224] == bytes(
            [1] * 32 + [11] * 32 + [41] * 32 + [51] * 32
        )

        record0, record1 = backend.get_csa_layer_major_records(prefix_key, [0, 1])
        assert record0 is not None
        assert record1 is not None
        assert record0.raw_extents == ((0, 1000, 8),)
        assert record1.raw_extents == ((4096, 1008, 8),)
        assert backend.kv_object_record_raw_readable(record0)
        assert backend.kv_object_record_raw_readable(record1)
        backend.local_cpu_backend.memory_allocator.close()

    def test_indexer_layer_major_snapshot_keeps_all_physical_rows(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """Indexer sidecars keep every already-compressed physical row."""
        config = create_test_config(temp_disk_path)
        config.extra_config = {
            "kv_object_store_enable": True,
            "kv_object_store_slot_mb": 2,
            "kv_object_store_capacity": 4,
        }
        with patch(
            "os.statvfs",
            return_value=SimpleNamespace(f_bsize=4096),
            create=True,
        ):
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
                metadata=create_indexer_object_store_metadata(),
            )
        allocator = AdHocMemoryAllocator(device="cpu")
        memory_objs: list[MemoryObj] = []
        for chunk_id in range(2):
            memory_obj = allocator.allocate(
                [torch.Size([1, 2, 8, 132])],
                [torch.uint8],
                fmt=MemoryFormat.KV_2LTD,
            )
            assert memory_obj is not None
            tensor = memory_obj.get_tensor(0)
            assert tensor is not None
            tensor[0, 0].fill_(10 + chunk_id)
            tensor[0, 1].fill_(20 + chunk_id)
            memory_objs.append(memory_obj)

        first_key = create_test_key(398)
        prefix_key = create_test_key(399)
        assert (
            backend.store_attention_layer_major_snapshot(
                prefix_key,
                memory_objs,
                prefix_keys=[first_key, prefix_key],
            )
            == 2
        )
        record0, record1 = backend.get_indexer_layer_major_records(
            prefix_key,
            [0, 1],
        )
        assert record0 is not None
        assert record1 is not None
        assert record0.length == 2 * 8 * 132
        assert record1.length == 2 * 8 * 132
        probed = backend.get_indexer_layer_major_records_for_keys(
            [first_key, prefix_key],
            layer_id=0,
        )
        assert probed[0] is not None
        assert probed[0].length == 8 * 132
        assert probed[1] == record0

        backend.local_cpu_backend.memory_allocator.close()

    def test_raw_lba_cache_is_sliced_to_view_ranges(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """Raw Tutti LBA registration follows explicit object view ranges."""
        backend = self._create_object_store_backend(
            temp_disk_path,
            async_loop,
            local_cpu_backend,
        )
        key = create_test_key(202)
        memory_obj = self._allocate_object_store_memory()
        backend.set_kv_object_tutti_raw_writer(
            lambda record, _buffer: (
                [
                    (
                        record.offset,
                        1000 + record.offset // 512,
                        record.aligned_length // 512,
                    )
                ],
                0.0,
            )
        )
        backend.kv_object_tutti_raw_enabled = True

        backend.async_save_bytes_to_disk(
            key,
            memory_obj,
        )
        csa0 = backend.get_kv_object_records(
            [key],
            layer_ids=[0],
            roles=["csa_attention_kv"],
        )[0]
        csa1 = backend.get_kv_object_records(
            [key],
            layer_ids=[1],
            roles=["csa_attention_kv"],
        )[0]
        assert csa0 is not None
        assert csa1 is not None

        raw_lba_cache = backend.get_kv_object_raw_lba_cache([csa0, csa1])

        assert raw_lba_cache == {
            backend.kv_object_tutti_path(csa0.pool_id): [
                (0, 1000, 584),
                (598016, 2168, 584),
                (299008, 1584, 584),
                (897024, 2752, 584),
            ]
        }

        backend.local_cpu_backend.memory_allocator.close()

    def test_raw_lba_cache_pads_full_record_tail_to_sector(
        self,
        temp_disk_path: str,
        async_loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ) -> None:
        """Raw Tutti reads may cover padded sectors without changing payload length."""
        backend = self._create_object_store_backend(
            temp_disk_path,
            async_loop,
            local_cpu_backend,
        )
        record = KVObjectRecord(
            object_id=KVObjectId(
                model_id="model",
                parallel_config_id="world1",
                rank=0,
                layer_id=0,
                role="full",
                block_id="tail",
            ),
            pool_id="rank0-full",
            offset=0,
            length=1000,
            aligned_length=4096,
            shape=(1000,),
            dtype="torch.uint8",
            state=KVObjectState.READY,
            raw_extents=((0, 1000, 8),),
        )

        raw_lba_cache = backend.get_kv_object_raw_lba_cache([record])

        assert raw_lba_cache == {
            backend.kv_object_tutti_path(record.pool_id): [(0, 1000, 2)]
        }

        backend.local_cpu_backend.memory_allocator.close()

    def test_raw_region_cover_check_requires_full_range(self) -> None:
        """Raw-object writes only proceed when the reserved region fully covers them."""
        backend = LocalDiskBackend.__new__(LocalDiskBackend)
        backend.kv_object_tutti_raw_region_extents = [(0, 1000, 1)]

        assert backend.kv_object_raw_region_covers(0, 512)
        assert not backend.kv_object_raw_region_covers(0, 513)
        assert not backend.kv_object_raw_region_covers(512, 512)

    def test_raw_object_readable_requires_full_dma_coverage(self) -> None:
        """Lookup only advertises raw objects whose extents cover direct reads."""
        backend = LocalDiskBackend.__new__(LocalDiskBackend)
        object_id = KVObjectId(
            model_id="model",
            parallel_config_id="world1",
            rank=0,
            layer_id=0,
            role="full",
            block_id="readable",
        )
        record = KVObjectRecord(
            object_id=object_id,
            pool_id="rank0-full",
            offset=0,
            length=1024,
            aligned_length=1024,
            shape=(1024,),
            dtype="torch.uint8",
            state=KVObjectState.READY,
            raw_extents=((0, 1000, 1),),
        )

        assert backend.kv_object_record_raw_read_bytes(record) == 1024
        assert not backend.kv_object_record_raw_readable(record)
        assert backend.kv_object_record_raw_readable(
            record.with_raw_extents(((0, 1000, 2),))
        )


class TestMultiPathDiskBackend:
    """Test cases for multi-path (multi-device) LocalDiskBackend."""

    def test_init_multi_path(self, async_loop, local_cpu_backend):
        """Test initialisation with comma-separated paths."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

            # Path selected by device_id (0 % 2 = 0 -> dir_a)
            assert backend.path == dir_a
            # Both directories should exist
            assert os.path.isdir(dir_a)
            assert os.path.isdir(dir_b)
            # Block size is a plain int for the selected path
            assert isinstance(backend.os_disk_bs, int)
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_gpu_affinity_selects_path(self, async_loop, local_cpu_backend):
        """Different cuda devices select different paths via modulo."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)

            dirs_by_gpu = {}
            for device in ("cuda:0", "cuda:1"):
                backend = LocalDiskBackend(
                    config=config,
                    loop=async_loop,
                    local_cpu_backend=local_cpu_backend,
                    dst_device=device,
                )
                dirs_by_gpu[device] = backend.path

            assert dirs_by_gpu["cuda:0"] == dir_a
            assert dirs_by_gpu["cuda:1"] == dir_b
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_all_directories_created(self, async_loop, local_cpu_backend):
        """All paths in the list get their directories created."""
        base = tempfile.mkdtemp()
        try:
            paths = [os.path.join(base, f"nvme{i}") for i in range(3)]
            combined = ",".join(paths)
            config = create_test_config(combined)
            LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )
            for p in paths:
                assert os.path.isdir(p), f"{p} should exist"
        finally:
            shutil.rmtree(base, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_single_path_backward_compat(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """A single path (no commas) works exactly as before."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_default(self, temp_disk_path, async_loop, local_cpu_backend):
        """Default local_disk_path_sharding is 'by_gpu' (backend inits OK)."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_explicit_by_gpu(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Explicitly setting local_disk_path_sharding='by_gpu' works."""
        config = create_test_config(temp_disk_path, local_disk_path_sharding="by_gpu")
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_unsupported_raises(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Unsupported local_disk_path_sharding raises ValueError."""
        config = create_test_config(
            temp_disk_path, local_disk_path_sharding="round_robin"
        )
        with pytest.raises(ValueError, match="Unsupported path sharding strategy"):
            LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

    def test_cpu_dst_device_defaults_to_first_path(self, async_loop, local_cpu_backend):
        """dst_device='cpu' should fall back to device_id=0."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
            )
            # device_id=0 -> 0 % 2 = 0 -> dir_a
            assert backend.path == dir_a
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()


class TestParseLocalDisk:
    """Unit tests for the _parse_local_disk config parser."""

    def test_none(self):
        assert _parse_local_disk(None) is None

    def test_single_raw_path(self):
        assert _parse_local_disk("/mnt/nvme0/cache/") == "/mnt/nvme0/cache/"

    def test_single_file_uri(self):
        assert _parse_local_disk("file:///mnt/nvme0/cache/") == "/mnt/nvme0/cache/"

    def test_single_file_uri_no_trailing_slash(self):
        assert _parse_local_disk("file:///mnt/nvme0/cache") == "/mnt/nvme0/cache"

    def test_comma_separated_raw(self):
        result = _parse_local_disk("/mnt/nvme0/,/mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_comma_separated_file_uris(self):
        result = _parse_local_disk("file:///mnt/nvme0/,file:///mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_mixed_uri_and_raw(self):
        result = _parse_local_disk("file:///mnt/nvme0/,/mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_whitespace_around_paths(self):
        result = _parse_local_disk("  /mnt/nvme0/ , /mnt/nvme1/  ")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_empty_string(self):
        assert _parse_local_disk("") is None


class TestGetBlockingCachePolicyUpdate:
    """Regression tests for phantom cache hit in get_blocking() (issue #3015).

    ``get_blocking()`` must call ``cache_policy.update_on_hit()`` only when
    ``load_bytes_from_disk()`` returns a valid ``MemoryObj``.  Calling it
    before confirming load success records a phantom hit that skews future
    eviction decisions.
    """

    def _inject_key(
        self,
        backend: LocalDiskBackend,
        key: CacheEngineKey,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> None:
        """Insert a key into backend.dict without writing anything to disk."""
        meta = DiskCacheMetadata(
            path="/nonexistent/path.pt",
            size=0,
            shape=shape,
            dtype=dtype,
            cached_positions=None,
            fmt=MemoryFormat.KV_2LTD,
            pin_count=0,
        )
        with backend.disk_lock:
            backend.dict[key] = meta
            backend.cache_policy.update_on_put(key)

    def test_no_phantom_hit_when_load_fails(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """update_on_hit must NOT be called when load_bytes_from_disk returns None."""
        key = create_test_key(101)
        shape = torch.Size([28, 2, 256, 8, 128])
        self._inject_key(local_disk_backend, key, shape, torch.bfloat16)

        with patch.object(
            local_disk_backend, "load_bytes_from_disk", return_value=None
        ):
            with patch.object(
                local_disk_backend.cache_policy, "update_on_hit"
            ) as mock_update:
                result = local_disk_backend.get_blocking(key)

        assert result is None
        mock_update.assert_not_called()
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_updates_cache_policy_on_successful_load(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """update_on_hit must be called exactly once when the load succeeds."""
        key = create_test_key(102)
        shape = torch.Size([28, 2, 256, 8, 128])
        self._inject_key(local_disk_backend, key, shape, torch.bfloat16)

        fake_memory_obj = MagicMock(spec=MemoryObj)
        with patch.object(
            local_disk_backend, "load_bytes_from_disk", return_value=fake_memory_obj
        ):
            with patch.object(
                local_disk_backend.cache_policy, "update_on_hit"
            ) as mock_update:
                result = local_disk_backend.get_blocking(key)

        assert result is fake_memory_obj
        mock_update.assert_called_once_with(key, local_disk_backend.dict)
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_absent_returns_none_without_policy_update(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """get_blocking must return None immediately when the key is not cached."""
        key = create_test_key(103)

        with patch.object(
            local_disk_backend.cache_policy, "update_on_hit"
        ) as mock_update:
            result = local_disk_backend.get_blocking(key)

        assert result is None
        mock_update.assert_not_called()
        local_disk_backend.local_cpu_backend.memory_allocator.close()
