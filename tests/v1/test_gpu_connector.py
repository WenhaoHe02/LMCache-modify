# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import random
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import (
    SGLangGPUConnector,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemGPUConnectorV3,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.gpu_connector.utils import get_dtype
from lmcache.v1.memory_management import (
    GPUMemoryAllocator,
    MemoryFormat,
    PagedTensorMemoryAllocator,
    PinMemoryAllocator,
    TensorMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata

if torch.cuda.is_available():
    try:
        # First Party
        import lmcache.c_ops as lmc_ops
    except ImportError:
        lmc_ops = None
else:
    lmc_ops = None

# Mock c_ops when not available
if lmc_ops is None:

    class MockGPUKVFormat:
        NL_X_TWO_NB_BS_NH_HS = 0
        NL_X_NB_TWO_BS_NH_HS = 1
        NL_X_NB_BS_HS = 2

    class MockCOps:
        GPUKVFormat = MockGPUKVFormat

    lmc_ops = MockCOps()


# Local
from .utils import (
    check_paged_kv_cache_equal,
    check_paged_kv_cache_equal_with_mla,
    check_sglang_paged_kv_cache_equal,
    generate_kv_cache_paged_list_tensors,
    generate_sglang_kv_cache_paged_list_tensors,
    recover_gpu_connector_states,
)


def test_csa_direct_seed_uses_deferred_submission() -> None:
    """The retrieve callback never runs the lock-reentrant seed inline."""
    calls: list[tuple[object, ...]] = []
    manager = SimpleNamespace(
        submit_seed_range_from_lmcache_group=lambda *args, **kwargs: (
            calls.append((*args, kwargs)) or object()
        ),
        seed_range_from_lmcache_group=lambda *_args, **_kwargs: pytest.fail(
            "synchronous seed must not run inside the retrieve callback"
        ),
    )
    connector = SimpleNamespace(
        _dsv4_optimized_layout_is_valid=lambda: True,
        _dsv4_layer_ids_for_group=lambda _group: [2],
        _dsv4_csa_seed_fallback_logged=False,
        _dsv4_csa_seed_logged=False,
    )
    memory_tensor = torch.arange(8, dtype=torch.uint8).view(1, 1, 2, 4)

    with patch(
        "lmcache.v1.indexer_ssd_manager.get_indexer_ssd_manager",
        return_value=manager,
    ):
        VLLMPagedMemGPUConnectorV3._prepare_csa_direct_seed_for_group(
            connector,
            0,
            object(),
            memory_tensor,
            0,
            8,
            "csa_indexer_cache",
            slot_mapping=torch.arange(8),
            lmcache_cached_tokens=8,
        )

    assert len(calls) == 1
    assert calls[0][0] == [2]
    assert calls[0][1] is memory_tensor
    assert calls[0][-1] == {"total_logical_tokens": 8}


@pytest.fixture(autouse=True, scope="module")
def patch_pin_allocator():
    def fake_pin_init(self, size: int, use_paging: bool = False, **kwargs):
        """
        :param int size: The size of the pinned memory in bytes.
        """

        # self.buffer = torch.empty(size, dtype=torch.uint8)
        # ptr = self.buffer.data_ptr()
        # err = torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
        # assert err == 0, (
        #     f"cudaHostRegister failed: {torch.cuda.cudart().cudaGetErrorString(err)}"
        # )
        self._unregistered = False
        self.buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)

        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            self.allocator = TensorMemoryAllocator(self.buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    def fake_pin_close(self):
        if not self._unregistered:
            torch.cuda.synchronize()
            # torch.cuda.cudart().cudaHostUnregister(self.buffer.data_ptr())
            self._unregistered = True

    with (
        patch(
            "lmcache.v1.memory_management.PinMemoryAllocator.__init__", fake_pin_init
        ),
        patch("lmcache.v1.memory_management.PinMemoryAllocator.close", fake_pin_close),
    ):
        yield


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,  # vllm non-MLA flash infer
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
    ],  # vllm MLA
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_vllm_paged_connector_v2_with_gpu_and_mla(use_gpu, gpu_kv_format):
    use_mla = gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 1 if use_mla else 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        gpu_kv_format=gpu_kv_format,
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        gpu_kv_format=gpu_kv_format,
    )
    dtype = get_dtype(gpu_kv_src, gpu_kv_format)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        if use_mla:
            check_paged_kv_cache_equal_with_mla(
                gpu_kv_src, gpu_kv_dst, slot_mapping, head_size
            )
        else:
            check_paged_kv_cache_equal(
                gpu_kv_src,
                gpu_kv_dst,
                slot_mapping,
                num_heads,
                head_size,
                gpu_kv_format,
            )

    connector = VLLMPagedMemGPUConnectorV2(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    connector2 = VLLMPagedMemGPUConnectorV2(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape = connector.get_shape(end - start)
        memory_obj = allocator.allocate(shape, dtype)
        connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_src,
            slot_mapping=slot_mapping,
            offset=0,
        )
        recover_gpu_connector_states(connector)
        if use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        connector2.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_dst,
            slot_mapping=slot_mapping,
            offset=0,
        )
        allocator.free(memory_obj)
        assert allocator.memcheck()

    if use_mla:
        check_paged_kv_cache_equal_with_mla(
            gpu_kv_src, gpu_kv_dst, slot_mapping, head_size
        )
    else:
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size, gpu_kv_format
        )
    allocator.close()


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize("num_groups", [1, 2, 3])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,  # vllm non-MLA flash infer
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
    ],  # vllm MLA
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV3",
)
def test_vllm_paged_connector_v3_with_gpu_and_mla(use_gpu, num_groups, gpu_kv_format):
    use_mla = gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS
    head_sizes = [64, 66, 66]
    dtypes = [torch.uint8, torch.bfloat16, torch.uint8]
    num_blocks = 100
    block_size = 16
    num_heads = 1 if use_mla else 8
    device = "cuda"
    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    # generate kv cache tensors
    src_kv_groups: list[list] = []
    dst_kv_groups: list[list] = []
    src_kv_caches: dict[str, torch.Tensor] = {}
    dst_kv_caches: dict[str, torch.Tensor] = {}
    for i in range(num_groups):
        for groups, kv_caches in [
            (src_kv_groups, src_kv_caches),
            (dst_kv_groups, dst_kv_caches),
        ]:
            kv_group = generate_kv_cache_paged_list_tensors(
                num_blocks=num_blocks,
                device=device,
                block_size=block_size,
                dtype=dtypes[i],
                num_layers=8,
                head_size=head_sizes[i],
                gpu_kv_format=gpu_kv_format,
            )
            groups.append(kv_group)
            for j, layer_tensor in enumerate(kv_group):
                kv_caches[f"{i}-{j}"] = layer_tensor

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the kv group is not the same before copying
    with pytest.raises(AssertionError):
        for i in range(num_groups):
            if use_mla:
                check_paged_kv_cache_equal_with_mla(
                    src_kv_groups[i], dst_kv_groups[i], slot_mapping, head_sizes[i]
                )
            else:
                check_paged_kv_cache_equal(
                    src_kv_groups[i],
                    dst_kv_groups[i],
                    slot_mapping,
                    num_heads,
                    head_sizes[i],
                    gpu_kv_format,
                )

    # create metadata and init kv layer groups
    metadata = _create_metadata(use_mla, src_kv_caches, gpu_kv_format)
    metadata2 = _create_metadata(use_mla, dst_kv_caches, gpu_kv_format)

    # connector will copy with src_kv_groups
    connector = VLLMPagedMemGPUConnectorV3(
        metadata=metadata,
        use_gpu=use_gpu,
        device=slot_mapping.device,
    )
    # connector2 will copy with dst_kv_groups
    connector2 = VLLMPagedMemGPUConnectorV3(
        metadata=metadata2,
        use_gpu=use_gpu,
        device=slot_mapping.device,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla

    # copy from src_kv_groups to memory_obj,
    # and then copy from memory_obj to dst_kv_groups
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        memory_obj = allocator.allocate(
            metadata.get_shapes(end - start), metadata.get_dtypes()
        )
        connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=list(src_kv_caches.values()),
            slot_mapping=slot_mapping,
            offset=0,
        )
        if use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        connector2.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=list(dst_kv_caches.values()),
            slot_mapping=slot_mapping,
            offset=0,
        )
        allocator.free(memory_obj)
        assert allocator.memcheck()

    # Check the kv group is same after copying
    for i in range(num_groups):
        if use_mla:
            check_paged_kv_cache_equal_with_mla(
                src_kv_groups[i], dst_kv_groups[i], slot_mapping, head_sizes[i]
            )
        else:
            check_paged_kv_cache_equal(
                src_kv_groups[i],
                dst_kv_groups[i],
                slot_mapping,
                num_heads,
                head_sizes[i],
                gpu_kv_format,
            )
    allocator.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VLLMPagedMemGPUConnectorV3 requires CUDA",
)
def test_vllm_paged_connector_v3_batched_gpu_restore():
    """Batched restore preserves multiple full CUDA-backed chunks."""
    if not hasattr(lmc_ops, "multi_layer_block_kv_transfer_batched"):
        pytest.skip("lmcache.c_ops was built without batched pointer support")

    device = torch.device("cuda")
    num_blocks = 64
    block_size = 16
    num_layers = 4
    head_size = 128
    chunk_size = 256
    num_tokens = chunk_size * 2
    gpu_kv_format = lmc_ops.GPUKVFormat.NL_X_NB_BS_HS

    src_layers = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        dtype=torch.bfloat16,
        num_layers=num_layers,
        head_size=head_size,
        gpu_kv_format=gpu_kv_format,
    )
    dst_layers = [torch.zeros_like(layer) for layer in src_layers]
    src_caches = {
        f"model.layers.{layer_idx}.self_attn": layer
        for layer_idx, layer in enumerate(src_layers)
    }
    dst_caches = {
        f"model.layers.{layer_idx}.self_attn": layer
        for layer_idx, layer in enumerate(dst_layers)
    }
    src_metadata = _create_metadata(True, src_caches, gpu_kv_format)
    dst_metadata = _create_metadata(True, dst_caches, gpu_kv_format)
    src_connector = VLLMPagedMemGPUConnectorV3(src_metadata, device)
    dst_connector = VLLMPagedMemGPUConnectorV3(dst_metadata, device)
    allocator = GPUMemoryAllocator(64 * 1024 * 1024, device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    starts = [0, chunk_size]
    ends = [chunk_size, num_tokens]
    memory_objs = []

    for start, end in zip(starts, ends, strict=True):
        memory_obj = allocator.allocate(
            src_metadata.get_shapes(end - start),
            src_metadata.get_dtypes(),
        )
        assert memory_obj is not None
        src_connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=list(src_caches.values()),
            slot_mapping=slot_mapping,
            offset=0,
        )
        memory_objs.append(memory_obj)
    src_connector.store_stream.synchronize()

    dst_connector.batched_to_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=list(dst_caches.values()),
        slot_mapping=slot_mapping,
        block_ids_by_group=[list(range(num_tokens // block_size))],
        vllm_kv_cache_group_block_sizes=[block_size],
        offset=0,
    )

    check_paged_kv_cache_equal_with_mla(
        src_layers,
        dst_layers,
        slot_mapping,
        head_size,
    )
    for memory_obj in memory_objs:
        memory_obj.ref_count_down()
    assert allocator.memcheck()
    allocator.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VLLMPagedMemGPUConnectorV3 requires CUDA",
)
def test_vllm_paged_connector_v3_raw_tutti_restore():
    """Raw Tutti staging restores full chunks without MemoryObj wrappers."""
    if not hasattr(lmc_ops, "multi_layer_block_kv_transfer_batched"):
        pytest.skip("lmcache.c_ops was built without batched pointer support")

    device = torch.device("cuda")
    num_blocks = 64
    block_size = 16
    num_layers = 4
    head_size = 128
    chunk_size = 256
    num_tokens = chunk_size * 2
    gpu_kv_format = lmc_ops.GPUKVFormat.NL_X_NB_BS_HS
    src_layers = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        dtype=torch.bfloat16,
        num_layers=num_layers,
        head_size=head_size,
        gpu_kv_format=gpu_kv_format,
    )
    dst_layers = [torch.zeros_like(layer) for layer in src_layers]
    src_caches = {
        f"model.layers.{layer_idx}.self_attn": layer
        for layer_idx, layer in enumerate(src_layers)
    }
    dst_caches = {
        f"model.layers.{layer_idx}.self_attn": layer
        for layer_idx, layer in enumerate(dst_layers)
    }
    src_metadata = _create_metadata(True, src_caches, gpu_kv_format)
    dst_metadata = _create_metadata(True, dst_caches, gpu_kv_format)
    src_connector = VLLMPagedMemGPUConnectorV3(src_metadata, device)
    dst_connector = VLLMPagedMemGPUConnectorV3(dst_metadata, device)
    allocator = GPUMemoryAllocator(64 * 1024 * 1024, device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    starts = [0, chunk_size]
    ends = [chunk_size, num_tokens]
    memory_objs = []

    for start, end in zip(starts, ends, strict=True):
        memory_obj = allocator.allocate(
            src_metadata.get_shapes(end - start),
            src_metadata.get_dtypes(),
        )
        assert memory_obj is not None
        src_connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=list(src_caches.values()),
            slot_mapping=slot_mapping,
            offset=0,
        )
        memory_objs.append(memory_obj)
    src_connector.store_stream.synchronize()

    source_nbytes = [memory_obj.get_size() for memory_obj in memory_objs]
    source_offsets = [0, source_nbytes[0]]
    staging = torch.cat(
        [
            memory_obj.raw_tensor[:nbytes]
            for memory_obj, nbytes in zip(
                memory_objs,
                source_nbytes,
                strict=True,
            )
        ]
    )
    dst_connector.batched_raw_to_gpu(
        staging,
        source_offsets,
        source_nbytes,
        starts,
        ends,
        [memory_obj.get_shapes() for memory_obj in memory_objs],
        [memory_obj.get_dtypes() for memory_obj in memory_objs],
        kvcaches=list(dst_caches.values()),
        slot_mapping=slot_mapping,
        block_ids_by_group=[list(range(num_tokens // block_size))],
        vllm_kv_cache_group_block_sizes=[block_size],
        offset=0,
    )

    check_paged_kv_cache_equal_with_mla(
        src_layers,
        dst_layers,
        slot_mapping,
        head_size,
    )
    for memory_obj in memory_objs:
        memory_obj.ref_count_down()
    assert allocator.memcheck()
    allocator.close()


@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,  # vllm non-MLA flash infer
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemLayerwiseGPUConnector",
)
def test_layerwise_vllm_paged_connector_with_gpu(use_gpu, gpu_kv_format):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        gpu_kv_format=gpu_kv_format,
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        gpu_kv_format=gpu_kv_format,
    )
    dtype = get_dtype(gpu_kv_src, gpu_kv_format)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size, gpu_kv_format
        )

    connector = VLLMPagedMemLayerwiseGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
    )

    # from gpu to cpu
    starts = []
    ends = []
    memory_objs = []

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts.append(start)
        ends.append(end)
        memory_objs.append(memory_objs_multi_layer)

    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]

    mem_obj_generator = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
        sync=True,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator)

    # from cpu to gpu
    mem_obj_consumer = connector.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
        sync=True,
    )
    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])
    next(mem_obj_consumer)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size, gpu_kv_format
    )

    allocator.close()


@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemLayerwiseGPUConnector",
)
def test_batched_layerwise_vllm_paged_connector_with_gpu(use_gpu):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens_1 = 800
    num_tokens_2 = 500
    num_tokens_total = num_tokens_1 + num_tokens_2
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    dtype = gpu_kv_src[0][0].dtype

    slot_mapping_total = random.sample(
        range(0, num_blocks * block_size), num_tokens_total
    )
    slot_mapping_total = torch.tensor(
        slot_mapping_total, device=device, dtype=torch.int64
    )

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(gpu_kv_src, gpu_kv_dst, slot_mapping_total)

    connector = VLLMPagedMemLayerwiseGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
    )

    # from gpu to cpu
    starts_1 = []
    ends_1 = []
    memory_objs_1 = []

    for start in range(0, num_tokens_1, chunk_size):
        end = min(start + chunk_size, num_tokens_1)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts_1.append(start)
        ends_1.append(end)
        memory_objs_1.append(memory_objs_multi_layer)

    memory_objs_1 = [list(row) for row in zip(*memory_objs_1, strict=False)]

    starts_2 = []
    ends_2 = []
    memory_objs_2 = []
    for start in range(num_tokens_1, num_tokens_total, chunk_size):
        end = min(start + chunk_size, num_tokens_total)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts_2.append(start)
        ends_2.append(end)
        memory_objs_2.append(memory_objs_multi_layer)

    memory_objs_2 = [list(row) for row in zip(*memory_objs_2, strict=False)]

    mem_obj_generator_1 = connector.batched_from_gpu(
        memory_objs_1,
        starts_1,
        ends_1,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    mem_obj_generator_1 = connector.batched_from_gpu(
        memory_objs_1,
        starts_1,
        ends_1,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    mem_obj_generator_2 = connector.batched_from_gpu(
        memory_objs_2,
        starts_2,
        ends_2,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=False,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator_1)
        next(mem_obj_generator_2)

    # from cpu to gpu
    mem_obj_consumer_1 = connector.batched_to_gpu(
        starts_1,
        ends_1,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping_total,
        sync=False,
    )
    mem_obj_consumer_2 = connector.batched_to_gpu(
        starts_2,
        ends_2,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    next(mem_obj_consumer_1)
    next(mem_obj_consumer_2)
    for layer_id in range(num_layers):
        mem_obj_consumer_1.send(memory_objs_1[layer_id])
        mem_obj_consumer_2.send(memory_objs_2[layer_id])
    next(mem_obj_consumer_1)
    next(mem_obj_consumer_2)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs_1:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    for mem_obj_multi_layer in memory_objs_2:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src,
        gpu_kv_dst,
        slot_mapping_total,
        num_heads,
        head_size,
    )

    allocator.close()


@pytest.mark.skip(reason="This test is skipped due to vllm dependency")
@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMBufferLayerwiseGPUConnector",
)
def test_layerwise_vllm_buffer_connector_with_gpu(use_gpu):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    dtype = gpu_kv_src[0][0].dtype

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
        )

    connector = VLLMBufferLayerwiseGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        dtype=dtype,
        device=device,
    )

    # from gpu to cpu
    starts = []
    ends = []
    memory_objs = []

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_2TD
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts.append(start)
        ends.append(end)
        memory_objs.append(memory_objs_multi_layer)

    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]

    mem_obj_generator = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator)

    # from cpu to gpu
    mem_obj_consumer = connector.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
    )
    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])
    next(mem_obj_consumer)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
    )

    allocator.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_vllm_paged_connector_v2_to_gpu_bench(benchmark):
    """
    VLLMPagedMemGPUConnectorV2.to_gpu() micro-benchmark.

    This test is to measure the performance of
    VLLMPagedMemGPUConnectorV2.to_gpu() when both KV caches and
    memobject are on GPU.

    """
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    chunk_size = 256

    allocator = GPUMemoryAllocator(1024 * 1024 * 1024, device)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)

    slot_mapping = random.sample(range(0, num_blocks * block_size), chunk_size)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    connector = VLLMPagedMemGPUConnectorV2(hidden_dim, num_layers)
    shape = connector.get_shape(chunk_size)
    memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)
    connector.from_gpu(
        memory_obj,
        0,
        chunk_size,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
        offset=0,
    )
    recover_gpu_connector_states(connector)
    assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
    benchmark.pedantic(
        connector.to_gpu,
        args=(memory_obj, 0, chunk_size),
        kwargs={
            "kvcaches": gpu_kv_dst,
            "slot_mapping": slot_mapping,
            "offset": 0,
        },
        rounds=100,
        iterations=1000,
        warmup_rounds=10,
    )
    allocator.free(memory_obj)
    assert allocator.memcheck()

    allocator.close()


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize("use_mla", [True, False])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to SGLangGPUConnector",
)
def test_sglang_connector_with_gpu_and_mla(use_gpu, use_mla):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 1 if use_mla else 8
    head_size = 128
    device = "cuda"
    dtype = torch.bfloat16
    hidden_dim = num_heads * head_size

    num_tokens = num_blocks * block_size // 2
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=use_mla,
        device=device,
        dtype=dtype,
    )
    gpu_kv_dst = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=use_mla,
        device=device,
        dtype=dtype,
    )

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        if use_mla:
            check_paged_kv_cache_equal_with_mla(
                gpu_kv_src, gpu_kv_dst, slot_mapping, head_size
            )
        else:
            check_sglang_paged_kv_cache_equal(
                gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
            )

    connector = SGLangGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    connector2 = SGLangGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape = connector.get_shape(end - start)
        memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)
        connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_src,
            slot_mapping=slot_mapping,
            offset=0,
        )
        if use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        connector2.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_dst,
            slot_mapping=slot_mapping,
            offset=0,
        )
        allocator.free(memory_obj)
        assert allocator.memcheck()

    if use_mla:
        check_paged_kv_cache_equal_with_mla(
            gpu_kv_src, gpu_kv_dst, slot_mapping, head_size
        )
    else:
        check_sglang_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
        )

    allocator.close()


def _create_metadata(use_mla, kv_caches, gpu_kv_format):
    # First Party
    from lmcache.v1.gpu_connector.utils import get_num_blocks
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

    num_heads = 1 if use_mla else 8
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=8,
        local_world_size=8,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(32, 2, 256, num_heads, 128),
        use_mla=use_mla,
    )
    kv_list = list(kv_caches.values())
    metadata.kv_layer_groups_manager = KVLayerGroupsManager(
        kv_list,
        gpu_kv_format=gpu_kv_format,
        num_blocks=get_num_blocks(kv_list, gpu_kv_format),
    )
    return metadata
