# SPDX-License-Identifier: Apache-2.0
# Standard
import abc
import os
import re
from typing import Any, List, Optional, Sequence, Tuple, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.gpu_connector.utils import (
    assert_is_vllm_flash_attn_or_flash_infer,
    assert_is_vllm_mla_or_flash_attn_or_flash_infer,
    attempt_permute_to_contiguous_view,
    get_block_size,
    get_device,
    get_elements_per_layer,
    get_group_data_ptrs,
    get_head_size,
    get_num_blocks,
    get_num_layers,
    get_page_buffer_size,
    get_tokens_per_layer,
    normalize_kv_and_discover_format,
)

try:
    # First Party
    from lmcache.v1.gpu_connector.utils import DiscoverableKVCache, LayoutHints
except ImportError:
    # Compatibility with older container images whose helper module exposes
    # the accessors but not the annotation aliases.
    DiscoverableKVCache = Any
    LayoutHints = dict[str, Any]
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager
from lmcache.v1.memory_management import GPUMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


class GPUConnectorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Store the data in the memory object into a GPU buffer.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to be copied into GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Load the data from a GPU buffer into the memory object.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to store the data from
            GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Batched load the data from a GPU memory into the memory objects.
        Sub-classes should define the format of the kwargs.

        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data from GPU.
        :param List[int] starts: The starting indices of the data in the corresponding
            token sequence.
        :param List[int] ends: The ending indices of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Batched store the data from the memory objects to GPU kv cache.
        Sub-classes should define the format of the kwargs.

        For non-layerwise connectors:
        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data to GPU.
        :param List[int] starts: The starting indices of the data in the corresponding
            token sequence.
        :param List[int] ends: The ending indices of the data in the corresponding
            token sequence.

        For layerwise connectors (generator pattern):
        :param List[int] memory_objs: Actually the starts list
        (positional compatibility)
        :param List[int] starts: Actually the ends list
        (positional compatibility)
        Note: Layerwise connectors receive memory objects
        via generator.send()
        """
        raise NotImplementedError

    def dsv4_hca_layer_ids(self, **kwargs: object) -> tuple[int, ...]:
        """Return transformer layer ids for DSv4 HCA KV groups.

        Connectors that do not expose DSv4 heterogeneous KV groups return an
        empty tuple.  CacheEngine uses this public hook to look up per-layer
        KV-object records without reaching into connector internals.
        """
        return ()

    def dsv4_hca_layer_object_ids(self, **kwargs: object) -> tuple[tuple[int, int], ...]:
        """Return ``(manager_layer_id, object_layer_id)`` HCA mappings.

        ``manager_layer_id`` names the vLLM attention layer used by
        HCAPrefetchManager.  ``object_layer_id`` names the layer-id field used
        when LocalDiskBackend indexed per-layer KV objects.  They are equal for
        simple layouts but can differ when KV cache layer names are grouped or
        reordered.
        """
        return ()

    @abc.abstractmethod
    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens."""
        raise NotImplementedError

    def initialize_kvcaches_ptr(self, **kwargs):
        """Initialize the kvcaches pointers if not already initialized."""
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]
            # Ensure contiguity on every call.  HND tensors from vLLM have a
            # non-contiguous logical view (NHD) that must be permuted back to
            # the physical (HND) shape for correct kernel indexing.
            # attempt_permute_to_contiguous_view is a no-op when already contiguous.
            self.kvcaches = attempt_permute_to_contiguous_view(self.kvcaches)


class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        # Not sure we need a dict here. Maybe a single GPU connector always
        # works with a single device?
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}

        self.kvcaches: Optional[List[torch.Tensor]] = None

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        self.layout_hints: LayoutHints = (
            kwargs.get(  # type: ignore[assignment]
                "layout_hints"
            )
            or {}
        )
        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemGPUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemGPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            layout_hints=layout_hints,
        )

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        self.device = kv_caches[0].device
        assert self.device.type == "cuda", "The device should be CUDA."
        idx = self.device.index
        if idx in self.kv_cache_pointers_on_gpu:
            return self.kv_cache_pointers_on_gpu[idx]

        self.gpu_kv_format, kv_caches = normalize_kv_and_discover_format(
            kv_caches, EngineType.VLLM, layout_hints=self.layout_hints
        )

        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.num_layers, dtype=torch.int64, device=self.device
        )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)
        self.num_blocks = get_num_blocks(kv_caches, self.gpu_kv_format)
        self.block_size = get_block_size(kv_caches, self.gpu_kv_format)
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(kv_caches, self.gpu_kv_format)

        return self.kv_cache_pointers_on_gpu[idx]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        # avoid read/write stream race condition for shared block
        # this will only be potentially non-zero for the first
        # block lmcache is transferring back
        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))

        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.device,
            self.page_buffer_size,
            lmc_ops.TransferDirection.H2D,
            self.gpu_kv_format,
            block_size=self.block_size,
            head_size=self.head_size,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
        )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        with torch.cuda.stream(self.store_stream):
            if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                lmc_ops.multi_layer_kv_transfer(
                    memory_obj.tensor,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches[0].device,
                    self.page_buffer_size,
                    lmc_ops.TransferDirection.D2H,
                    self.gpu_kv_format,
                    block_size=self.block_size,
                    head_size=self.head_size,
                )
            else:
                # kvcaches -> gpu_buffer -> memobj
                assert self.gpu_buffer.device == self.kvcaches[0].device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                lmc_ops.multi_layer_kv_transfer(
                    tmp_gpu_buffer,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches[0].device,
                    self.page_buffer_size,
                    lmc_ops.TransferDirection.D2H,
                    self.gpu_kv_format,
                    block_size=self.block_size,
                    head_size=self.head_size,
                )
                memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])


class VLLMPagedMemGPUConnectorV3(GPUConnectorInterface):
    def __init__(
        self,
        metadata: LMCacheMetadata,
        device: torch.device,
        use_gpu: bool = False,
        layout_hints: Optional[LayoutHints] = None,
    ):
        assert device.type == "cuda", "The device should be CUDA."
        self.metadata = metadata
        self.device = device
        self.use_mla = metadata.use_mla
        self.chunk_size = metadata.chunk_size
        self.use_gpu = use_gpu
        self.layout_hints: LayoutHints = layout_hints or {}
        self.kvcaches: Optional[List[torch.Tensor]] = None

        self.init = False
        self.group_kv_cache_pointers_on_gpu: Optional[list[torch.Tensor]] = None
        self.group_tmp_buffer: Optional[list[torch.Tensor]] = None
        self.group_tmp_buffer_flat: Optional[torch.Tensor] = None
        self.group_tmp_buffer_offsets: list[int] = []
        self.group_tmp_buffer_capacities: list[int] = []
        self.dsv4_optimized_kv = self._read_config_flag(
            "dsv4_optimized_kv",
            "LMCACHE_DSV4_OPTIMIZED_KV",
        )
        self.dsv4_optimized_tail_tokens = self._read_config_int(
            "dsv4_optimized_tail_tokens",
            "LMCACHE_DSV4_OPTIMIZED_TAIL_TOKENS",
            default=self.chunk_size,
            minimum=self.chunk_size,
        )
        self.dsv4_defer_hca_to_moe = self._read_config_flag(
            "dsv4_defer_hca_to_moe",
            "LMCACHE_DSV4_DEFER_HCA_TO_MOE",
        )
        self.dsv4_defer_hca_max_tokens = self._read_config_int(
            "dsv4_defer_hca_max_tokens",
            "LMCACHE_DSV4_DEFER_HCA_MAX_TOKENS",
            default=0,
            minimum=0,
        )
        self._dsv4_layout_valid: Optional[bool] = None
        self._dsv4_policy_logged = False
        self._dsv4_hca_defer_logged = False
        self._dsv4_hca_defer_fallback_logged = False
        self._dsv4_hca_defer_gate_logged = False
        self._dsv4_csa_seed_logged = False
        self._dsv4_csa_seed_fallback_logged = False
        self._kv_cache_layer_names: tuple[str, ...] = ()
        self._layer_name_to_vllm_group: dict[str, int] = {}
        self._vllm_group_block_sizes: tuple[int, ...] = ()
        self._hma_layout_logged = False
        self._single_layer_ptr_cache: dict[tuple[int, int], torch.Tensor] = {}

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemGPUConnectorV3":
        assert device is not None
        return cls(metadata, device, use_gpu, layout_hints=layout_hints)

    def _initialize_kv_cache_pointers(self):
        """Discover KV-cache layout, build the layer-groups manager, and
        capture per-group GPU pointer tensors.

        All layout-adjacent work lives here: the connector already owns
        ``layout_hints`` and the actual ``self.kvcaches`` tensors, so the
        serving-engine adapter stays agnostic to format.
        """
        if self._kv_cache_pointers_ready():
            return
        if self.init:
            logger.warning(
                "VLLMPagedMemGPUConnectorV3 had a stale KV-cache "
                "initialization; rebuilding layout metadata"
            )
        self.init = False
        self.group_kv_cache_pointers_on_gpu = None
        self.group_tmp_buffer = None
        self.group_tmp_buffer_flat = None
        self.group_tmp_buffer_offsets = []
        self.group_tmp_buffer_capacities = []
        self._dsv4_layout_valid = None

        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is not None and not klg_manager.kv_layer_groups:
            logger.warning(
                "VLLMPagedMemGPUConnectorV3 found an empty KV layer-groups "
                "manager; rebuilding from registered vLLM KV caches"
            )
            # Do NOT set to None here: clearing to None before the rebuild
            # check (below) would cause all TP workers to see None
            # simultaneously and trigger redundant parallel rebuilds.  The
            # rebuild check already handles both None and empty-groups cases.

        self.gpu_kv_format, self.kvcaches = normalize_kv_and_discover_format(
            self.kvcaches, EngineType.VLLM, layout_hints=self.layout_hints
        )
        self.num_blocks = get_num_blocks(self.kvcaches, self.gpu_kv_format)
        self.block_size = get_block_size(self.kvcaches, self.gpu_kv_format)
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(self.kvcaches, self.gpu_kv_format)

        if (
            self.metadata.kv_layer_groups_manager is None
            or not self.metadata.kv_layer_groups_manager.kv_layer_groups
        ):
            self.metadata.kv_layer_groups_manager = KVLayerGroupsManager(
                self.kvcaches,
                self.gpu_kv_format,
                self.num_blocks,
                layout_hints=self.layout_hints,
                lmcache_logical_chunk_size=self.chunk_size,
            )
        klg_manager = self.metadata.kv_layer_groups_manager
        if not klg_manager.kv_layer_groups:
            logger.warning(
                "VLLMPagedMemGPUConnectorV3 could not discover any KV "
                "layer groups; deferring pointer initialization"
            )
            return

        if self.use_gpu:
            self.group_tmp_buffer_offsets = [0]
            for group in klg_manager.kv_layer_groups:
                shape = self._group_shape_for_tokens(group, self.chunk_size)
                byte_size = shape.numel() * group.dtype.itemsize
                self.group_tmp_buffer_offsets.append(
                    self.group_tmp_buffer_offsets[-1] + byte_size
                )
            try:
                self.group_tmp_buffer_flat = torch.empty(
                    self.group_tmp_buffer_offsets[-1],
                    dtype=torch.uint8,
                    device=self.device,
                )
                self.group_tmp_buffer_capacities = [
                    self.group_tmp_buffer_offsets[idx + 1]
                    - self.group_tmp_buffer_offsets[idx]
                    for idx in range(len(klg_manager.kv_layer_groups))
                ]
                self.group_tmp_buffer = [
                    self._group_tmp_view(group_idx)
                    for group_idx in range(len(klg_manager.kv_layer_groups))
                ]
            except torch.OutOfMemoryError:
                logger.warning(
                    "VLLMPagedMemGPUConnectorV3 could not allocate %d bytes "
                    "for optional GPU staging; continuing with direct CPU "
                    "memory transfers",
                    self.group_tmp_buffer_offsets[-1],
                )
                self.group_tmp_buffer = None
                self.group_tmp_buffer_flat = None
                self.group_tmp_buffer_offsets = []
                self.group_tmp_buffer_capacities = []

        self.group_kv_cache_pointers_on_gpu = []
        self._single_layer_ptr_cache = {}
        for group in klg_manager.kv_layer_groups:
            ptrs = get_group_data_ptrs(
                self.kvcaches, self.gpu_kv_format, group.layer_indices
            )
            cpu = torch.empty(len(ptrs), dtype=torch.int64, device="cpu")
            cpu.numpy()[:] = ptrs
            gpu = torch.empty(len(ptrs), dtype=torch.int64, device=self.device)
            gpu.copy_(cpu)
            self.group_kv_cache_pointers_on_gpu.append(gpu)

        self.init = True
        logger.info("init kv cache pointers success in VLLMPagedMemGPUConnectorV3")

    def register_kv_caches(self, kvcaches: list[torch.Tensor]) -> None:
        """Register vLLM KV cache tensors and discover heterogeneous groups."""
        self.kvcaches = kvcaches
        if not self._kv_cache_pointers_ready():
            self.init = False
        self._initialize_kv_cache_pointers()

    def initialize_kvcaches_ptr(self, **kwargs: Any) -> None:
        """Register per-request KV cache tensors and discover layer groups."""
        super().initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is not None:
            self._initialize_kv_cache_pointers()

    def _kv_cache_pointers_ready(self) -> bool:
        """Return whether cached pointer metadata matches non-empty groups."""
        if not self.init or self.kvcaches is None:
            return False
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return False
        return (
            self.group_kv_cache_pointers_on_gpu is not None
            and len(self.group_kv_cache_pointers_on_gpu)
            == len(klg_manager.kv_layer_groups)
        )

    @staticmethod
    def _read_env_flag(name: str) -> bool:
        value = os.getenv(name, "")
        return value.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _as_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return False

    def _config_value(self, key: str) -> object:
        extra = self.layout_hints.get("lmcache_extra_config")
        if isinstance(extra, dict) and key in extra:
            return extra[key]
        extra = self.metadata.kv_connector_extra_config
        if isinstance(extra, dict):
            if key in extra:
                return extra[key]
            lmcache_key = f"lmcache.extra_config.{key}"
            if lmcache_key in extra:
                return extra[lmcache_key]
        return None

    def _read_config_flag(self, key: str, env_name: str) -> bool:
        value = self._config_value(key)
        if value is not None:
            return self._as_bool(value)
        return self._read_env_flag(env_name)

    @staticmethod
    def _read_env_int(name: str, default: int, minimum: int) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        return VLLMPagedMemGPUConnectorV3._parse_int_value(
            value,
            name,
            default,
            minimum,
        )

    def _read_config_int(
        self,
        key: str,
        env_name: str,
        default: int,
        minimum: int,
    ) -> int:
        value = self._config_value(key)
        if value is not None:
            return self._parse_int_value(value, key, default, minimum)
        return self._read_env_int(env_name, default, minimum)

    @staticmethod
    def _parse_int_value(
        value: object,
        name: str,
        default: int,
        minimum: int,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            logger.warning(
                "%s=%r is not an integer; using default %d",
                name,
                value,
                default,
            )
            return default
        if parsed < minimum:
            logger.warning(
                "%s=%d is below minimum %d; using %d",
                name,
                parsed,
                minimum,
                minimum,
            )
            return minimum
        return parsed

    @staticmethod
    def _dsv4_group_role(group: KVLayerGroupInfo) -> str:
        """Classify DeepSeek V4 heterogeneous KV groups by transfer semantics."""
        hidden_dim = group.hidden_dim_size
        num_layers = group.num_layers
        compress_ratio = group.compress_ratio

        if group.dtype == torch.float32:
            return "compressor_state"

        if group.dtype != torch.uint8:
            return "unknown"

        if hidden_dim == 132:
            return "csa_indexer_cache"

        if hidden_dim != 584:
            return "unknown"

        if hidden_dim == 584 and compress_ratio == 1:
            return "swa_cache"
        if compress_ratio >= 64 or group.shape_desc.bs <= 2:
            return "hca_attention_kv"
        if compress_ratio == 4 or num_layers == 30:
            return "csa_attention_kv"
        return "unknown"

    def _dsv4_optimized_layout_is_valid(self) -> bool:
        if not self.dsv4_optimized_kv:
            return False
        if self._dsv4_layout_valid is not None:
            return self._dsv4_layout_valid
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None:
            self._dsv4_layout_valid = False
            return False
        roles = {
            self._dsv4_group_role(group)
            for group in klg_manager.kv_layer_groups
        }
        required = {
            "swa_cache",
            "hca_attention_kv",
            "csa_attention_kv",
            "csa_indexer_cache",
            "compressor_state",
        }
        self._dsv4_layout_valid = required.issubset(roles)
        if not self._dsv4_layout_valid:
            logger.warning(
                "LMCACHE_DSV4_OPTIMIZED_KV=1 but KV groups do not look like "
                "DeepSeek V4; falling back to full H2D transfer. roles=%s",
                sorted(roles),
            )
        return self._dsv4_layout_valid

    def _log_dsv4_optimized_policy_once(self) -> None:
        if self._dsv4_policy_logged or not self._dsv4_optimized_layout_is_valid():
            return
        assert self.metadata.kv_layer_groups_manager is not None
        group_summaries = []
        for idx, group in enumerate(
            self.metadata.kv_layer_groups_manager.kv_layer_groups
        ):
            role = self._dsv4_group_role(group)
            action = (
                "tail-only"
                if role in {"swa_cache", "compressor_state"}
                else "defer-to-moe"
                if role == "hca_attention_kv" and self.dsv4_defer_hca_to_moe
                else "full"
            )
            group_summaries.append(
                f"{idx}:{role}:{action}:layers={group.num_layers}:"
                f"hidden={group.hidden_dim_size}:cr={group.compress_ratio}:"
                f"dtype={group.dtype}"
            )
        logger.info(
            "LMCACHE_DSV4_OPTIMIZED_KV active: H2D transfer uses DSv4 "
            "role-aware policy; tail_tokens=%d; groups=[%s]",
            self.dsv4_optimized_tail_tokens,
            ", ".join(group_summaries),
        )
        self._dsv4_policy_logged = True

    def _dsv4_is_tail_transfer(
        self,
        start: int,
        end: int,
        **kwargs: object,
    ) -> bool:
        slot_mapping = kwargs.get("slot_mapping")
        if isinstance(slot_mapping, torch.Tensor):
            load_end = int(slot_mapping.numel())
        else:
            load_end = end
        tail_start = max(0, load_end - self.dsv4_optimized_tail_tokens)
        return end > tail_start

    def _should_transfer_group_to_gpu(
        self,
        group_idx: int,
        start: int,
        end: int,
        **kwargs: object,
    ) -> bool:
        if not self._dsv4_optimized_layout_is_valid():
            return True
        assert self.metadata.kv_layer_groups_manager is not None
        group = self.metadata.kv_layer_groups_manager.kv_layer_groups[group_idx]
        role = self._dsv4_group_role(group)
        if role in {"swa_cache", "compressor_state"}:
            return self._dsv4_is_tail_transfer(start, end, **kwargs)
        return True

    def _should_defer_hca_group(
        self,
        role: str,
        end: int,
        **kwargs: object,
    ) -> bool:
        if role != "hca_attention_kv" or not self.dsv4_defer_hca_to_moe:
            return False
        if self.dsv4_defer_hca_max_tokens <= 0:
            return True
        cached_tokens = kwargs.get("lmcache_cached_tokens")
        try:
            if cached_tokens is not None:
                logical_tokens = int(cached_tokens)
            else:
                slot_mapping = kwargs.get("slot_mapping")
                logical_tokens = (
                    int(slot_mapping.numel())
                    if isinstance(slot_mapping, torch.Tensor)
                    else end
                )
        except (TypeError, ValueError):
            logical_tokens = end
        if logical_tokens <= self.dsv4_defer_hca_max_tokens:
            return True
        if not self._dsv4_hca_defer_gate_logged:
            logger.info(
                "LMCACHE_DSV4_DEFER_HCA_TO_MOE=1 but HCA defer is gated "
                "off for %d cached tokens above max %d; using normal H2D",
                logical_tokens,
                self.dsv4_defer_hca_max_tokens,
            )
            self._dsv4_hca_defer_gate_logged = True
        return False

    @staticmethod
    def _dsv4_layer_id_from_name(layer_name: str) -> Optional[int]:
        """Extract the transformer layer id from a vLLM KV cache layer name."""
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
        if match is None:
            return None
        return int(match.group(1))

    def _dsv4_layer_ids_for_group(self, group: KVLayerGroupInfo) -> list[int]:
        """Return transformer layer ids covered by one LMCache group."""
        layer_ids: list[int] = []
        for layer_idx in group.layer_indices:
            if layer_idx >= len(self._kv_cache_layer_names):
                continue
            layer_id = self._dsv4_layer_id_from_name(
                self._kv_cache_layer_names[layer_idx]
            )
            if layer_id is not None:
                layer_ids.append(layer_id)
        return layer_ids

    def _dsv4_hca_layer_ids_for_group(self, group: KVLayerGroupInfo) -> list[int]:
        """Return transformer layer ids covered by one LMCache HCA group."""
        return self._dsv4_layer_ids_for_group(group)

    def dsv4_hca_layer_ids(self, **kwargs: object) -> tuple[int, ...]:
        """Return transformer layer ids covered by DSv4 HCA groups."""
        return tuple(
            dict.fromkeys(
                manager_layer_id
                for manager_layer_id, _object_layer_id in (
                    self.dsv4_hca_layer_object_ids(**kwargs)
                )
            )
        )

    def dsv4_hca_layer_object_ids(self, **kwargs: object) -> tuple[tuple[int, int], ...]:
        """Return HCA manager layer ids mapped to KV-object layer ids."""
        self._update_hma_metadata(**kwargs)
        if not self._dsv4_optimized_layout_is_valid():
            return ()
        assert self.metadata.kv_layer_groups_manager is not None
        layer_ids: list[tuple[int, int]] = []
        for group in self.metadata.kv_layer_groups_manager.kv_layer_groups:
            if self._dsv4_group_role(group) != "hca_attention_kv":
                continue
            resolved = self._dsv4_hca_layer_ids_for_group(group)
            if len(resolved) == len(group.layer_indices):
                layer_ids.extend(
                    (int(manager_layer_id), int(object_layer_id))
                    for manager_layer_id, object_layer_id in zip(
                        resolved,
                        group.layer_indices,
                        strict=True,
                    )
                )
            else:
                layer_ids.extend(
                    (int(layer_idx), int(layer_idx))
                    for layer_idx in group.layer_indices
                )
        return tuple(dict.fromkeys(layer_ids))


    def _prepare_csa_direct_seed_for_group(
        self,
        group_idx: int,
        group: KVLayerGroupInfo,
        memory_tensor: torch.Tensor,
        start: int,
        end: int,
        role: str,
        **kwargs: object,
    ) -> None:
        """Seed CSA indexer SSD/HBM state directly from a retrieve chunk."""
        if role != "csa_indexer_cache":
            return
        if not self._dsv4_optimized_layout_is_valid():
            return
        layer_ids = self._dsv4_layer_ids_for_group(group)
        if not layer_ids:
            if not self._dsv4_csa_seed_fallback_logged:
                logger.info(
                    "CSA direct LMCache seed skipped: layer ids could not be "
                    "resolved for group %d",
                    group_idx,
                )
                self._dsv4_csa_seed_fallback_logged = True
            return
        try:
            from lmcache.v1.indexer_ssd_manager import get_indexer_ssd_manager
        except ImportError:
            return
        manager = get_indexer_ssd_manager()
        if manager is None:
            if not self._dsv4_csa_seed_fallback_logged:
                logger.info(
                    "CSA direct LMCache seed skipped: no IndexerSSDManager "
                    "is attached"
                )
                self._dsv4_csa_seed_fallback_logged = True
            return
        seed = getattr(manager, "seed_range_from_lmcache_group", None)
        if not callable(seed):
            return
        slot_mapping = kwargs.get("slot_mapping")
        total_logical_tokens = kwargs.get("lmcache_cached_tokens", None)
        if total_logical_tokens is None:
            total_logical_tokens = (
                int(slot_mapping.numel())
                if isinstance(slot_mapping, torch.Tensor)
                else end
            )
        total_logical_tokens = int(total_logical_tokens)
        seeded = seed(
            layer_ids,
            memory_tensor,
            start,
            end,
            total_logical_tokens=total_logical_tokens,
        )
        if seeded <= 0:
            return
        if not self._dsv4_csa_seed_logged:
            logger.info(
                "CSA direct LMCache seed active: seeded IndexerSSDManager "
                "from LMCache csa_indexer_cache group during retrieve"
            )
            self._dsv4_csa_seed_logged = True

    def _prepare_hca_defer_for_group(
        self,
        group_idx: int,
        group: KVLayerGroupInfo,
        memory_tensor: torch.Tensor,
        start: int,
        end: int,
        role: str,
        **kwargs: object,
    ) -> bool:
        """Seed HCA flat store from this retrieve chunk and skip normal H2D."""
        if not self._should_defer_hca_group(role, end, **kwargs):
            return False
        self._update_hma_metadata(**kwargs)
        layer_ids = self._dsv4_hca_layer_ids_for_group(group)
        if not layer_ids:
            if not self._dsv4_hca_defer_fallback_logged:
                logger.info(
                    "LMCACHE_DSV4_DEFER_HCA_TO_MOE=1 but HCA layer ids could "
                    "not be resolved for LMCache group %d; keeping normal H2D",
                    group_idx,
                )
                self._dsv4_hca_defer_fallback_logged = True
            return False
        try:
            from lmcache.v1.hca_prefetch_manager import get_hca_prefetch_manager
        except ImportError:
            return False
        manager = get_hca_prefetch_manager()
        if manager is None:
            if not self._dsv4_hca_defer_fallback_logged:
                logger.info(
                    "LMCACHE_DSV4_DEFER_HCA_TO_MOE=1 but no HCA prefetch "
                    "manager is attached; keeping normal H2D"
                )
                self._dsv4_hca_defer_fallback_logged = True
            return False
        object_source_enabled = getattr(manager, "object_source_enabled", None)
        has_object_source = getattr(manager, "has_object_source", None)
        if (
            callable(object_source_enabled)
            and object_source_enabled()
            and callable(has_object_source)
            and all(bool(has_object_source(layer_id)) for layer_id in layer_ids)
        ):
            if not self._dsv4_hca_defer_logged:
                logger.info(
                    "LMCACHE_DSV4_DEFER_HCA_TO_MOE=1: using HCA object-source "
                    "Tutti reads and deferred normal HCA H2D"
                )
                self._dsv4_hca_defer_logged = True
            return True
        seed = getattr(manager, "seed_range_from_lmcache_group", None)
        if not callable(seed):
            return False
        seeded = seed(layer_ids, memory_tensor, start, end)
        if seeded <= 0:
            if not self._dsv4_hca_defer_fallback_logged:
                logger.info(
                    "LMCACHE_DSV4_DEFER_HCA_TO_MOE=1 but HCA LMCache chunk "
                    "could not seed the flat store; keeping normal H2D"
                )
                self._dsv4_hca_defer_fallback_logged = True
            return False
        if not self._dsv4_hca_defer_logged:
            logger.info(
                "LMCACHE_DSV4_DEFER_HCA_TO_MOE=1: seeded HCA flat store "
                "directly from LMCache retrieve and deferred normal HCA H2D"
            )
            self._dsv4_hca_defer_logged = True
        return True

    def _group_shape_for_tokens(
        self, group: KVLayerGroupInfo, num_tokens: int
    ) -> torch.Size:
        """Return this KV group's LMCache memory shape for logical tokens."""
        compress_ratio = group.compress_ratio
        if num_tokens % compress_ratio != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) is not a multiple of "
                f"compress_ratio ({compress_ratio})"
            )
        return torch.Size(
            [
                group.shape_desc.kv_size,
                group.num_layers,
                num_tokens // compress_ratio,
                group.hidden_dim_size,
            ]
        )

    def _group_tmp_view(
        self, group_idx: int, shape: Optional[torch.Size] = None
    ) -> torch.Tensor:
        """Return a typed view into the temporary flat GPU buffer."""
        assert self.group_tmp_buffer_flat is not None
        assert self.metadata.kv_layer_groups_manager is not None
        group = self.metadata.kv_layer_groups_manager.kv_layer_groups[group_idx]
        start = self.group_tmp_buffer_offsets[group_idx]
        group_capacity = self.group_tmp_buffer_offsets[group_idx + 1] - start
        if shape is None:
            shape = self._group_shape_for_tokens(group, self.chunk_size)
        byte_size = shape.numel() * group.dtype.itemsize
        if byte_size > group_capacity:
            self._ensure_group_tmp_capacity(group_idx, byte_size)
            start = self.group_tmp_buffer_offsets[group_idx]
            group_capacity = self.group_tmp_buffer_offsets[group_idx + 1] - start
            if byte_size > group_capacity:
                raise ValueError(
                    f"temporary buffer for KV group {group_idx} is too small: "
                    f"need {byte_size} bytes, capacity {group_capacity} bytes"
                )
        end = start + byte_size
        return self.group_tmp_buffer_flat[start:end].view(group.dtype).view(shape)

    def _group_tmp_can_hold(self, group_idx: int, shape: torch.Size) -> bool:
        """Return whether the fixed GPU staging buffer can hold ``shape``."""
        assert self.metadata.kv_layer_groups_manager is not None
        if self.group_tmp_buffer_flat is None:
            return False
        if not self.group_tmp_buffer_capacities:
            return False
        group = self.metadata.kv_layer_groups_manager.kv_layer_groups[group_idx]
        byte_size = shape.numel() * group.dtype.itemsize
        return byte_size <= self.group_tmp_buffer_capacities[group_idx]

    def _ensure_group_tmp_capacity(self, group_idx: int, byte_size: int) -> None:
        """Grow the flat temporary GPU buffer so one group can hold a shape."""
        assert self.group_tmp_buffer_flat is not None
        assert self.metadata.kv_layer_groups_manager is not None
        num_groups = len(self.metadata.kv_layer_groups_manager.kv_layer_groups)
        if not self.group_tmp_buffer_capacities:
            self.group_tmp_buffer_capacities = [
                self.group_tmp_buffer_offsets[idx + 1]
                - self.group_tmp_buffer_offsets[idx]
                for idx in range(num_groups)
            ]
        if byte_size <= self.group_tmp_buffer_capacities[group_idx]:
            return
        new_capacities = list(self.group_tmp_buffer_capacities)
        new_capacities[group_idx] = byte_size
        new_offsets = [0]
        for capacity in new_capacities:
            new_offsets.append(new_offsets[-1] + capacity)
        new_flat = torch.empty(
            new_offsets[-1],
            dtype=torch.uint8,
            device=self.device,
        )
        for idx in range(num_groups):
            old_start = self.group_tmp_buffer_offsets[idx]
            old_end = self.group_tmp_buffer_offsets[idx + 1]
            new_start = new_offsets[idx]
            new_flat[new_start : new_start + old_end - old_start].copy_(
                self.group_tmp_buffer_flat[old_start:old_end],
                non_blocking=True,
            )
        self.group_tmp_buffer_flat = new_flat
        self.group_tmp_buffer_offsets = new_offsets
        self.group_tmp_buffer_capacities = new_capacities
        self.group_tmp_buffer = [
            self._group_tmp_view(idx) for idx in range(num_groups)
        ]
        logger.info(
            "VLLMPagedMemGPUConnectorV3 grew temporary buffer for KV group %d "
            "to %d bytes",
            group_idx,
            byte_size,
        )

    @staticmethod
    def _normalize_hma_block_ids(block_ids: Any) -> tuple[list[int], ...]:
        """Normalize vLLM HMA block ids into ``tuple[group][block]`` form."""
        if block_ids is None:
            return ()
        if isinstance(block_ids, tuple):
            return tuple(list(group) for group in block_ids)
        if isinstance(block_ids, list):
            if block_ids and all(isinstance(group, list) for group in block_ids):
                return tuple(list(group) for group in block_ids)
            return (list(block_ids),)
        return ()

    @staticmethod
    def _normalize_name_groups(names: Any) -> tuple[tuple[str, ...], ...]:
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            return ()
        normalized: list[tuple[str, ...]] = []
        for group in names:
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                return ()
            normalized.append(tuple(str(name) for name in group))
        return tuple(normalized)

    @staticmethod
    def _normalize_int_tuple(values: Any) -> tuple[int, ...]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return ()
        try:
            return tuple(int(value) for value in values)
        except (TypeError, ValueError):
            return ()

    def _update_hma_metadata(self, **kwargs: object) -> None:
        """Refresh vLLM HMA group metadata passed by the adapter."""
        layer_names = kwargs.get("kv_cache_layer_names")
        if isinstance(layer_names, Sequence) and not isinstance(
            layer_names, (str, bytes)
        ):
            self._kv_cache_layer_names = tuple(str(name) for name in layer_names)

        group_names = self._normalize_name_groups(
            kwargs.get("vllm_kv_cache_group_layer_names")
        )
        if group_names:
            self._layer_name_to_vllm_group = {
                name: group_idx
                for group_idx, names in enumerate(group_names)
                for name in names
            }

        group_block_sizes = self._normalize_int_tuple(
            kwargs.get("vllm_kv_cache_group_block_sizes")
        )
        if group_block_sizes:
            self._vllm_group_block_sizes = group_block_sizes

        if (
            not self._hma_layout_logged
            and self._kv_cache_layer_names
            and self._layer_name_to_vllm_group
            and self._vllm_group_block_sizes
        ):
            logger.info(
                "VLLMPagedMemGPUConnectorV3 captured HMA transfer metadata: "
                "kv_cache_layers=%d, vllm_groups=%d, block_sizes=%s",
                len(self._kv_cache_layer_names),
                len(self._vllm_group_block_sizes),
                list(self._vllm_group_block_sizes),
            )
            self._hma_layout_logged = True

    def _vllm_group_for_lmcache_group(self, group_idx: int) -> Optional[int]:
        """Map an LMCache transfer group to its vLLM HMA kv_cache_group."""
        if not self._kv_cache_layer_names or not self._layer_name_to_vllm_group:
            return None
        assert self.metadata.kv_layer_groups_manager is not None
        group = self.metadata.kv_layer_groups_manager.kv_layer_groups[group_idx]
        for layer_idx in group.layer_indices:
            if layer_idx >= len(self._kv_cache_layer_names):
                continue
            layer_name = self._kv_cache_layer_names[layer_idx]
            if layer_name in self._layer_name_to_vllm_group:
                return self._layer_name_to_vllm_group[layer_name]
        return None

    def _hma_block_ids_for_group(
        self,
        group_idx: int,
        start: int,
        end: int,
        **kwargs: object,
    ) -> Optional[tuple[torch.Tensor, int]]:
        """Return engine-logical block ids from vLLM HMA metadata.

        LMCache stores one logical chunk per object. For DSv4 HMA, each
        transfer group must use the vLLM block table and block size that owns
        that layer group. Same-shaped tensors can have different HMA semantics,
        so using group 0 for every group corrupts full-hit restore.
        """
        self._update_hma_metadata(**kwargs)
        block_ids_by_group = self._normalize_hma_block_ids(
            kwargs.get("block_ids_by_group")
        )
        if not block_ids_by_group:
            return None

        vllm_group_idx = self._vllm_group_for_lmcache_group(group_idx)
        if vllm_group_idx is None:
            vllm_group_idx = 0
        if vllm_group_idx >= len(block_ids_by_group):
            raise ValueError(
                f"LMCache KV group {group_idx} maps to vLLM HMA group "
                f"{vllm_group_idx}, but only {len(block_ids_by_group)} block-id "
                "groups were provided"
            )
        if not block_ids_by_group[vllm_group_idx]:
            return None
        logical_block_size = int(
            self.layout_hints.get(
                "inference_engine_logical_block_size",
                self.block_size,
            )
        )
        if self._vllm_group_block_sizes:
            logical_block_size = self._vllm_group_block_sizes[vllm_group_idx]
        if logical_block_size <= 0:
            raise ValueError(
                f"Invalid vLLM HMA logical block size {logical_block_size}"
            )
        if start % logical_block_size != 0 or end % logical_block_size != 0:
            raise ValueError(
                f"LMCache transfer [{start}, {end}) is not aligned to vLLM "
                f"HMA logical block_size {logical_block_size}"
            )

        block_start = start // logical_block_size
        block_end = end // logical_block_size
        selected = block_ids_by_group[vllm_group_idx][block_start:block_end]
        expected = block_end - block_start
        if len(selected) != expected:
            raise ValueError(
                f"Insufficient vLLM HMA block ids for LMCache group {group_idx}: "
                f"vllm_group={vllm_group_idx}, range=[{start}, {end}), "
                f"block_size={logical_block_size}, need={expected}, "
                f"available={len(block_ids_by_group[vllm_group_idx])}"
            )
        return (
            torch.tensor(selected, dtype=torch.long, device=self.device),
            logical_block_size,
        )

    def _slot_mapping_to_block_ids(
        self,
        slot_mapping: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        token_slots = slot_mapping[start:end]
        if token_slots.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        valid = token_slots >= 0
        if not bool(valid.all().item()):
            raise ValueError(
                "VLLMPagedMemGPUConnectorV3 block transfer does not support "
                "negative slot_mapping entries inside a transferred chunk"
            )
        logical_block_size = int(
            self.layout_hints.get(
                "inference_engine_logical_block_size",
                self.block_size,
            )
        )
        if token_slots.numel() % logical_block_size != 0:
            raise ValueError(
                f"transfer token count {token_slots.numel()} is not aligned to "
                f"inference_engine_logical_block_size {logical_block_size}"
            )
        block_slots = token_slots.view(-1, logical_block_size)
        slot_offsets = block_slots % logical_block_size
        expected_offsets = torch.arange(
            logical_block_size,
            dtype=token_slots.dtype,
            device=token_slots.device,
        )
        if not bool(torch.equal(slot_offsets, expected_offsets.expand_as(slot_offsets))):
            raise ValueError(
                "VLLMPagedMemGPUConnectorV3 block transfer requires contiguous "
                "slot offsets within each inference-engine block"
            )
        block_starts = block_slots[:, 0]
        block_ids = block_starts // logical_block_size
        return block_ids.to(device=self.device, dtype=torch.long, non_blocking=True)

    def _transfer_group(
        self,
        group_idx: int,
        memory_tensor: torch.Tensor,
        block_ids: torch.Tensor,
        direction: "lmc_ops.TransferDirection",
        skip_prefix_n_blocks: int = 0,
    ) -> None:
        assert self.group_kv_cache_pointers_on_gpu is not None
        assert self.metadata.kv_layer_groups_manager is not None
        group = self.metadata.kv_layer_groups_manager.kv_layer_groups[group_idx]
        lmc_ops.multi_layer_block_kv_transfer(
            self.group_kv_cache_pointers_on_gpu[group_idx],
            [memory_tensor.data_ptr()],
            block_ids,
            self.device,
            direction,
            group.shape_desc,
            group.physical_chunk_size,
            self.gpu_kv_format,
            skip_prefix_n_blocks,
        )

    def _single_layer_shape_desc(
        self,
        group: KVLayerGroupInfo,
    ) -> "lmc_ops.PageBufferShapeDesc":
        desc = lmc_ops.PageBufferShapeDesc()
        desc.kv_size = group.shape_desc.kv_size
        desc.nl = 1
        desc.nb = group.shape_desc.nb
        desc.bs = group.shape_desc.bs
        desc.nh = group.shape_desc.nh
        desc.hs = group.shape_desc.hs
        desc.element_size = group.shape_desc.element_size
        desc.block_stride_elems = group.shape_desc.block_stride_elems
        return desc

    def _single_layer_ptrs(
        self,
        group_idx: int,
        object_layer_id: int,
    ) -> torch.Tensor:
        cache_key = (int(group_idx), int(object_layer_id))
        cached = self._single_layer_ptr_cache.get(cache_key)
        if cached is not None:
            return cached
        assert self.kvcaches is not None
        ptrs = get_group_data_ptrs(
            self.kvcaches,
            self.gpu_kv_format,
            [int(object_layer_id)],
        )
        cpu = torch.empty(len(ptrs), dtype=torch.int64, device="cpu")
        cpu.numpy()[:] = ptrs
        gpu = torch.empty(len(ptrs), dtype=torch.int64, device=self.device)
        gpu.copy_(cpu)
        self._single_layer_ptr_cache[cache_key] = gpu
        return gpu


    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.raw_tensor is not None
        assert "slot_mapping" in kwargs
        if self.use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        assert self.kvcaches[0].device == self.device
        self._initialize_kv_cache_pointers()
        assert self.group_kv_cache_pointers_on_gpu is not None
        assert self.metadata.kv_layer_groups_manager is not None

        # avoid read/write stream race condition for shared block
        # this will only be potentially non-zero for the first
        # block lmcache is transferring back
        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))

        with torch.cuda.stream(self.load_stream):
            legacy_block_ids: Optional[torch.Tensor] = None
            legacy_skip_prefix_n_blocks: Optional[int] = None
            self._update_hma_metadata(**kwargs)
            self._log_dsv4_optimized_policy_once()
            for i in range(self.metadata.kv_layer_groups_manager.num_groups):
                group = self.metadata.kv_layer_groups_manager.kv_layer_groups[i]
                role = self._dsv4_group_role(group)
                memory_obj_tensor = memory_obj.get_tensor(i)
                assert memory_obj_tensor is not None
                if memory_obj_tensor.numel() == 0:
                    continue
                if self._prepare_hca_defer_for_group(
                    i,
                    group,
                    memory_obj_tensor,
                    start,
                    end,
                    role,
                    **kwargs,
                ):
                    continue
                self._prepare_csa_direct_seed_for_group(
                    i,
                    group,
                    memory_obj_tensor,
                    start,
                    end,
                    role,
                    **kwargs,
                )
                if not self._should_transfer_group_to_gpu(
                    i,
                    start,
                    end,
                    **kwargs,
                ):
                    continue
                if self.use_gpu and self._group_tmp_can_hold(
                    i,
                    memory_obj_tensor.shape,
                ):
                    tmp_tensor = self._group_tmp_view(i, memory_obj_tensor.shape)
                    tmp_tensor.copy_(memory_obj_tensor, non_blocking=True)
                    memory_obj_tensor = tmp_tensor
                hma_block_ids = self._hma_block_ids_for_group(
                    i,
                    start,
                    end,
                    **kwargs,
                )
                if hma_block_ids is None:
                    if legacy_block_ids is None:
                        logical_block_size = int(
                            self.layout_hints.get(
                                "inference_engine_logical_block_size",
                                self.block_size,
                            )
                        )
                        if skip_prefix_n_tokens % logical_block_size != 0:
                            raise ValueError(
                                f"skip_prefix_n_tokens {skip_prefix_n_tokens} "
                                "is not aligned to "
                                "inference_engine_logical_block_size "
                                f"{logical_block_size}"
                            )
                        legacy_block_ids = self._slot_mapping_to_block_ids(
                            slot_mapping,
                            start,
                            end,
                        )
                        legacy_skip_prefix_n_blocks = (
                            skip_prefix_n_tokens // logical_block_size
                        )
                    block_ids = legacy_block_ids
                    skip_prefix_n_blocks = legacy_skip_prefix_n_blocks
                else:
                    block_ids, group_block_size = hma_block_ids
                    if skip_prefix_n_tokens % group_block_size != 0:
                        raise ValueError(
                            f"skip_prefix_n_tokens {skip_prefix_n_tokens} is not "
                            f"aligned to HMA group block_size {group_block_size}"
                        )
                    skip_prefix_n_blocks = skip_prefix_n_tokens // group_block_size
                assert skip_prefix_n_blocks is not None
                self._transfer_group(
                    i,
                    memory_obj_tensor,
                    block_ids,
                    lmc_ops.TransferDirection.H2D,
                    skip_prefix_n_blocks,
                )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.raw_tensor is not None
        assert "slot_mapping" in kwargs

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        assert self.kvcaches[0].device == self.device
        self._initialize_kv_cache_pointers()
        assert self.group_kv_cache_pointers_on_gpu is not None
        assert self.metadata.kv_layer_groups_manager is not None
        with torch.cuda.stream(self.store_stream):
            legacy_block_ids: Optional[torch.Tensor] = None
            for i in range(self.metadata.kv_layer_groups_manager.num_groups):
                dst_tensor = memory_obj.get_tensor(i)
                assert dst_tensor is not None
                if dst_tensor.numel() == 0:
                    continue
                memory_obj_tensor = dst_tensor
                staged = False
                if self.use_gpu and self._group_tmp_can_hold(i, dst_tensor.shape):
                    memory_obj_tensor = self._group_tmp_view(i, dst_tensor.shape)
                    staged = True
                hma_block_ids = self._hma_block_ids_for_group(
                    i,
                    start,
                    end,
                    **kwargs,
                )
                if hma_block_ids is None:
                    if legacy_block_ids is None:
                        legacy_block_ids = self._slot_mapping_to_block_ids(
                            slot_mapping,
                            start,
                            end,
                        )
                    block_ids = legacy_block_ids
                else:
                    block_ids, _ = hma_block_ids
                self._transfer_group(
                    i,
                    memory_obj_tensor,
                    block_ids,
                    lmc_ops.TransferDirection.D2H,
                )
                if staged:
                    dst_tensor.copy_(memory_obj_tensor, non_blocking=True)

        if not memory_obj.raw_tensor.is_cuda:
            self.store_stream.synchronize()

        if not memory_obj.raw_tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        raise NotImplementedError


class VLLMBufferLayerwiseGPUConnector(GPUConnectorInterface):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        use_double_buffer: bool = True,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.layout_hints: LayoutHints = (
            kwargs.get(  # type: ignore[assignment]
                "layout_hints"
            )
            or {}
        )

        # TODO(Jiayi): remove this hardcode
        self.cache_positions = True

        self.fused_rotary_emb = None

        assert use_gpu, "use_gpu must be true in VLLMBufferLayerwiseGPUConnector"
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        self.buffer_mapping: dict[int, MemoryObj] = {}

        # track gap positions between blended chunks
        self.current_gap_positions = None

        self.use_gpu = use_gpu
        self.gpu_buffer_allocator = None
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMBufferLayerwiseGPUConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMBufferLayerwiseGPUConnector.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            dtype=metadata.kv_dtype,
            device=device,
            layout_hints=layout_hints,
        )

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            self.gpu_kv_format, kv_caches = normalize_kv_and_discover_format(
                kv_caches, EngineType.VLLM, layout_hints=self.layout_hints
            )
            self.kvcaches = kv_caches
            assert_is_vllm_flash_attn_or_flash_infer(self.gpu_kv_format)
            self.tokens_per_layer = get_tokens_per_layer(kv_caches, self.gpu_kv_format)
            self.elements_per_layer = get_elements_per_layer(
                kv_caches, self.gpu_kv_format
            )
            logger.info(
                f"Lazily initializing GPU buffer (max tokens={self.tokens_per_layer})."
            )
            gpu_buffer_size = self.elements_per_layer * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the KV cache for the given layer ID.
        This function is used to get the KV cache from the GPU buffer.
        """
        if layer_id not in self.buffer_mapping:
            raise ValueError(f"Layer {layer_id} is not loaded into GPU buffer.")

        gpu_buffer = self.buffer_mapping[layer_id].tensor
        assert gpu_buffer is not None
        return gpu_buffer[0], gpu_buffer[1]

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to buffer GPU memory. In each iteration i, it (1) loads the KV
        cache of layer i from CPU -> GPU buffer, (2) recovers the positional
        encoding of the layer i-1's KV cache in the GPU buffer, and (3)
        moves the KV cache of layer i-2 from GPU buffer to paged GPU memory.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            # First Party
            from lmcache.integration.vllm.utils import ENGINE_NAME

            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        num_all_tokens = ends[-1] - starts[0]
        slot_mapping_full = slot_mapping[starts[0] : ends[-1]]

        # compute gap positions
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        buf_offset = starts[0]

        for start, end in zip(starts, ends, strict=False):
            gap_mask[start - buf_offset : end - buf_offset] = False

        self.current_gap_positions = torch.where(gap_mask)[0]

        buf_offset = starts[0]
        if self.cache_positions:
            new_positions_full = torch.arange(
                starts[0], ends[-1], dtype=torch.int64, device=self.kvcaches[0].device
            )

        buffer_shape = self.get_shape(num_all_tokens)
        assert self.gpu_buffer_allocator is not None
        compute_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        load_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert compute_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert load_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert compute_gpu_buffer_obj.tensor is not None
        assert load_gpu_buffer_obj.tensor is not None

        # current_stream = torch.cuda.current_stream()

        if self.cache_positions:
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=self.kvcaches[0].device
            )
        for layer_id in range(self.num_layers + 2):
            if layer_id > 1:
                lmc_ops.single_layer_kv_transfer(
                    self.buffer_mapping[layer_id - 2].tensor,
                    self.kvcaches[layer_id - 2],
                    slot_mapping_full,
                    lmc_ops.TransferDirection.H2D,
                    self.gpu_kv_format,
                    token_major=False,  # shape is [2, num_tokens, hidden_dim]
                )
                del self.buffer_mapping[layer_id - 2]

                logger.debug(f"Finished loading layer {layer_id - 2} into paged memory")

            if layer_id > 0 and layer_id <= self.num_layers:
                # NOTE: wait until both compute and load streams are done
                torch.cuda.synchronize()

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                        old_positions_full,
                        new_positions_full,
                        compute_gpu_buffer_obj.tensor[0],
                    )

                # gap zeroing after RoPE
                if self.current_gap_positions.numel():
                    compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                logger.debug(f"Finished loading layer {layer_id - 1} into buffer")

            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer
                with torch.cuda.stream(self.load_stream):
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[0], non_blocking=True)

                        load_gpu_buffer_obj.tensor[1][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[1], non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[
                                start - buf_offset : end - buf_offset
                            ] = memory_obj.metadata.cached_positions

            elif layer_id == self.num_layers:
                yield

        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(self.buffer_mapping) == 0, (
            "There are still layers in the buffer mapping after "
            "releasing the GPU buffers."
        )

        yield

    # TODO(Jiayi): Reduce repetitive operations in `batched_to_gpu`
    # and `batched_from_gpu`.
    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        buf_start = 0
        slot_mapping_chunks = []
        buf_starts_ends = []
        old_positions_chunks = []
        for start, end in zip(starts, ends, strict=False):
            buf_end = buf_start + end - start
            buf_starts_ends.append((buf_start, buf_end))
            slot_mapping_chunks.append(slot_mapping[start:end])
            buf_start = buf_end
            if self.cache_positions:
                old_positions_chunks.append(
                    torch.arange(
                        start, end, device=self.kvcaches[0].device, dtype=torch.int64
                    )
                )

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        buffer_shape = self.get_shape(num_tokens)
        assert self.gpu_buffer_allocator is not None
        tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert tmp_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[layer_id],
                    slot_mapping_full,
                    lmc_ops.TransferDirection.D2H,
                    self.gpu_kv_format,
                    token_major=False,  # shape is [2, num_tokens, hidden_dim]
                )
                for (buf_start, buf_end), memory_obj, old_positions in zip(
                    buf_starts_ends,
                    memory_objs_layer,
                    old_positions_chunks,
                    strict=False,
                ):
                    assert memory_obj.tensor is not None
                    memory_obj.tensor[0].copy_(
                        tmp_gpu_buffer_obj.tensor[0][buf_start:buf_end],
                        non_blocking=True,
                    )
                    memory_obj.tensor[1].copy_(
                        tmp_gpu_buffer_obj.tensor[1][buf_start:buf_end],
                        non_blocking=True,
                    )
                    if self.cache_positions:
                        memory_obj.metadata.cached_positions = old_positions

            yield
            self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, num_tokens, self.hidden_dim_size])


class VLLMPagedMemLayerwiseGPUConnector(GPUConnectorInterface):
    """ """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.layout_hints: LayoutHints = (
            kwargs.get(  # type: ignore[assignment]
                "layout_hints"
            )
            or {}
        )

        self.gpu_buffer_allocator = None

        assert "chunk_size" in kwargs, (
            "chunk_size should be provided to create a GPU buffer."
        )
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.kvcaches: Optional[List[torch.Tensor]] = None

        # All sizes are in bytes
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemLayerwiseGPUConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemLayerwiseGPUConnector.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            layout_hints=layout_hints,
        )

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            self.gpu_kv_format, kv_caches = normalize_kv_and_discover_format(
                kv_caches, EngineType.VLLM, layout_hints=self.layout_hints
            )
            self.kvcaches = kv_caches
            assert_is_vllm_mla_or_flash_attn_or_flash_infer(self.gpu_kv_format)
            self.tokens_per_layer = get_tokens_per_layer(kv_caches, self.gpu_kv_format)
            self.elements_per_layer = get_elements_per_layer(
                kv_caches, self.gpu_kv_format
            )
            logger.info(
                f"Lazily initializing GPU buffer (max tokens={self.tokens_per_layer})."
            )
            gpu_buffer_size = self.elements_per_layer * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        # TODO(Jiayi): Optimize away this `cat`
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        mem_fmt = MemoryFormat.KV_MLA_FMT if self.use_mla else MemoryFormat.KV_T2D

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, mem_fmt
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]
        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if sync:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            # memobj -> gpu_buffer -> kvcaches
            with torch.cuda.stream(self.load_stream):
                for start, end, memory_obj in zip(
                    starts, ends, memory_objs_layer, strict=False
                ):
                    # Validate memory format
                    if self.use_mla:
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT, (
                            f"Expected memory format {MemoryFormat.KV_MLA_FMT}, "
                            f"got {memory_obj.metadata.fmt}"
                        )
                    else:
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D, (
                            f"Expected memory format {MemoryFormat.KV_T2D}, "
                            f"got {memory_obj.metadata.fmt}"
                        )
                    if self.use_gpu:
                        tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                            memory_obj.tensor, non_blocking=True
                        )
                    else:
                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping_full,
                            lmc_ops.TransferDirection.H2D,
                            self.gpu_kv_format,
                            token_major=True,
                        )

                if self.use_gpu:
                    lmc_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        lmc_ops.TransferDirection.H2D,
                        self.gpu_kv_format,
                        token_major=True,
                    )
        yield

        # synchronize the last layer
        if sync:
            current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        if tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading all {self.num_layers} layers.")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        mem_fmt = MemoryFormat.KV_MLA_FMT if self.use_mla else MemoryFormat.KV_T2D

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, mem_fmt
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]
        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                if self.use_gpu:
                    lmc_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        lmc_ops.TransferDirection.D2H,
                        self.gpu_kv_format,
                        token_major=True,
                    )
                for start, end, memory_obj in zip(
                    starts, ends, memory_objs_layer, strict=False
                ):
                    assert memory_obj.tensor is not None
                    if self.use_gpu:
                        memory_obj.tensor.copy_(
                            tmp_gpu_buffer_obj.tensor[start - offset : end - offset],
                            non_blocking=True,
                        )
                    else:
                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            lmc_ops.TransferDirection.D2H,
                            self.gpu_kv_format,
                            token_major=True,
                        )
                    # Set metadata format
                    if self.use_mla:
                        memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

            yield
            if sync:
                self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.use_mla:
            # MLA format: [num_tokens, hidden_dim_size]
            return torch.Size([num_tokens, self.hidden_dim_size])
        else:
            # Standard format: [num_tokens, 2, hidden_dim_size]
            return torch.Size([num_tokens, 2, self.hidden_dim_size])


class SGLangGPUConnector(GPUConnectorInterface):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [page_buffer_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )

        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )
            logger.info(f"GPU buffer: {self.gpu_buffer.shape}")

    def _initialize_pointers(self, kv_caches: DiscoverableKVCache) -> torch.Tensor:
        self.gpu_kv_format, kv_caches = normalize_kv_and_discover_format(
            kv_caches, EngineType.SGLANG
        )
        num_layers = get_num_layers(kv_caches, self.gpu_kv_format)
        # SGLang registers every layer as one group; pass all indices in order.
        ptrs = get_group_data_ptrs(
            kv_caches, self.gpu_kv_format, list(range(num_layers))
        )
        assert len(ptrs) == self.num_kv_cache, (
            f"Expected {self.num_kv_cache} KV cache pointers, got {len(ptrs)}"
        )
        self.kv_cache_pointers.numpy()[:] = ptrs

        device = get_device(kv_caches)
        assert device.type == "cuda", "The device should be CUDA."
        idx = device.index
        if idx not in self.kv_cache_pointers_on_gpu:
            self.kv_cache_pointers_on_gpu[idx] = torch.empty(
                self.num_kv_cache, dtype=torch.int64, device=device
            )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        self.page_buffer_size = get_page_buffer_size(kv_caches, self.gpu_kv_format)
        return self.kv_cache_pointers_on_gpu[idx]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "partial slot mapping"
             where its length is the same as the uncached token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    f" order to be processed by {self.__class__.__name__}"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    f" order to be processed by {self.__class__.__name__}"
                )

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        offset = kwargs.get("offset", 0)

        kvcaches: DiscoverableKVCache = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)
        lmc_ops.multi_layer_kv_transfer_unilateral(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start - offset : end - offset],
            get_device(kvcaches),
            self.page_buffer_size,
            lmc_ops.TransferDirection.H2D,
            self.gpu_kv_format,
        )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "partial slot mapping"
             where its length is the same as the uncached token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: DiscoverableKVCache = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)

        if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
            lmc_ops.multi_layer_kv_transfer_unilateral(
                memory_obj.tensor,
                kv_cache_pointers,
                slot_mapping[start:end],
                get_device(kvcaches),
                self.page_buffer_size,
                lmc_ops.TransferDirection.D2H,
                self.gpu_kv_format,
            )
        else:
            # kvcaches -> gpu_buffer -> memobj
            assert self.gpu_buffer.device == get_device(kvcaches)
            tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
            lmc_ops.multi_layer_kv_transfer_unilateral(
                tmp_gpu_buffer,
                kv_cache_pointers,
                slot_mapping[start:end],
                get_device(kvcaches),
                self.page_buffer_size,
                lmc_ops.TransferDirection.D2H,
                self.gpu_kv_format,
            )
            memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            torch.cuda.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    # TODO(Yuwei): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)


# TODO: support MLA
class SGLangLayerwiseGPUConnector(GPUConnectorInterface):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [page_buffer_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        self.dtype = kwargs["dtype"]
        assert "device" in kwargs, "device should be provided to create a GPU buffer."
        self.device = kwargs["device"]

        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )
        self.use_gpu = use_gpu
        self.gpu_buffer_allocator: Optional[GPUMemoryAllocator] = None

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            self.gpu_kv_format, kv_caches = normalize_kv_and_discover_format(
                kv_caches, EngineType.SGLANG
            )
            self.tokens_per_layer = get_tokens_per_layer(kv_caches, self.gpu_kv_format)
            self.elements_per_layer = get_elements_per_layer(
                kv_caches, self.gpu_kv_format
            )
            logger.info(
                f"Lazily initializing GPU buffer (max tokens={self.tokens_per_layer})."
            )
            gpu_buffer_size = self.elements_per_layer * self.element_size
            logger.info(
                f"Lazily initializing GPU buffer (gpu buffer size={gpu_buffer_size})."
            )
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            # memobj -> gpu_buffer -> kvcaches
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                if self.use_gpu:
                    tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                        memory_obj.tensor, non_blocking=True
                    )
                else:
                    lmc_ops.single_layer_kv_transfer_sgl(
                        memory_obj.tensor,
                        self.kvcaches[0][layer_id],
                        self.kvcaches[1][layer_id],
                        slot_mapping[start:end],
                        lmc_ops.TransferDirection.H2D,
                        token_major=True,
                    )

            if self.use_gpu:
                t, h, d = self.kvcaches[0][layer_id].shape

                lmc_ops.single_layer_kv_transfer_sgl(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[0][layer_id].view(t, 1, h, d),
                    self.kvcaches[1][layer_id].view(t, 1, h, d),
                    slot_mapping_full,
                    lmc_ops.TransferDirection.H2D,
                    token_major=True,
                )

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            if self.use_gpu:
                t, h, d = self.kvcaches[0][layer_id].shape
                lmc_ops.single_layer_kv_transfer_sgl(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[0][layer_id].view(t, 1, h, d),
                    self.kvcaches[1][layer_id].view(t, 1, h, d),
                    slot_mapping_full,
                    lmc_ops.TransferDirection.D2H,
                    token_major=True,
                )

            start_idx = 0

            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.tensor is not None
                if self.use_gpu:
                    chunk_len = memory_obj.tensor.shape[0]
                    memory_obj.tensor.copy_(
                        tmp_gpu_buffer_obj.tensor[start_idx : start_idx + chunk_len],
                        non_blocking=True,
                    )
                    start_idx += chunk_len
                else:
                    lmc_ops.single_layer_kv_transfer_sgl(
                        memory_obj.tensor,
                        self.kvcaches[0][layer_id],
                        self.kvcaches[1][layer_id],
                        slot_mapping[start:end],
                        lmc_ops.TransferDirection.D2H,
                        token_major=True,
                    )

            yield
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        # TODO: support MLA
        return torch.Size([num_tokens, 2, self.hidden_dim_size])


_TRTLLM_KERNEL_BATCH_SIZE = 32


class TRTLLMGPUConnector(GPUConnectorInterface):
    """GPU connector for TRT-LLM's cross-layer KV pool.

    TRT-LLM hands LMCache a single 4-D pool tensor with shape
    ``[num_blocks, num_layers, 2, num_kv_heads * tokens_per_block * head_dim]``
    (HND layout, K and V interleaved on dim 2). On
    :meth:`register_kv_caches` the connector calls
    :func:`normalize_kv_and_discover_format` which reshapes the trailing
    dim into ``[num_kv_heads, tokens_per_block, head_dim]``, yielding the
    canonical 6-D ``[NB, NL, 2, NH, BS, HS]`` form (format
    ``NB_NL_TWO_NH_BS_HS``).

    Transfers go through :func:`lmc_ops.multi_layer_block_kv_transfer` —
    the multiprocess kernel — using the single base pointer of the pool
    tensor. Per-chunk block ids are taken from ``kwargs['block_ids']``,
    sliced as ``block_ids[i * blocks_per_chunk : (i+1) * blocks_per_chunk]``
    where ``blocks_per_chunk = chunk_size // tokens_per_block``.

    This is intentionally NOT a subclass of
    :class:`VLLMPagedMemGPUConnectorV3`. V3 drives the in-process kernel
    with ``slot_mapping`` and per-layer pointers — the wrong primitive
    for a single cross-layer base pointer.
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        hidden_dim_size: int,
        num_layers: int,
        chunk_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.device = device
        self._batch_size = _TRTLLM_KERNEL_BATCH_SIZE
        self.load_stream = torch.cuda.Stream(device=device)
        self.store_stream = torch.cuda.Stream(device=device)

        self.kv_cache_tensor: Optional[torch.Tensor] = None
        self.paged_buffer_ptrs: Optional[torch.Tensor] = None
        self.shape_desc: Optional["lmc_ops.PageBufferShapeDesc"] = None
        self._kv_format: Optional["lmc_ops.GPUKVFormat"] = None
        self.tokens_per_block: Optional[int] = None
        self.blocks_per_chunk: Optional[int] = None

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        device: torch.device,
    ) -> "TRTLLMGPUConnector":
        """Create a connector from :class:`LMCacheMetadata`.

        Args:
            metadata: Metadata carrying ``kv_shape``
                ``(num_layers, 2, chunk_size, num_kv_heads, head_size)`` and
                ``kv_dtype``.
            device: CUDA device for transfer streams and block-ids staging.
        """
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_heads = metadata.kv_shape[3]
        head_dim = metadata.kv_shape[4]
        hidden_dim_size = num_kv_heads * head_dim
        return cls(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
        )

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor) -> None:
        """Register TRT-LLM's 4-D KV pool tensor with the connector.

        Reshapes the 4-D tensor to canonical 6-D form, runs format
        discovery, and caches the base pointer / shape descriptor used
        by every subsequent transfer.

        Called once at TRT-LLM worker init — separate from
        :meth:`to_gpu` / :meth:`from_gpu`. Idempotent if called with the
        same tensor; reassigning a different tensor replaces the
        previously registered one.
        """
        if kv_cache_tensor.dim() != 4:
            raise ValueError(
                f"TRT-LLM kv_cache_tensor must be 4-D "
                f"[NB, NL, 2, flat], got shape {tuple(kv_cache_tensor.shape)}"
            )
        num_blocks, num_layers, kv_factor, flat = kv_cache_tensor.shape
        tokens_per_block = flat // (self.num_kv_heads * self.head_dim)
        if tokens_per_block * self.num_kv_heads * self.head_dim != flat:
            raise ValueError(
                f"flat dim {flat} not divisible by "
                f"num_kv_heads * head_dim ({self.num_kv_heads * self.head_dim})"
            )
        self.tokens_per_block = tokens_per_block
        self.blocks_per_chunk = self.chunk_size // tokens_per_block

        layout_hints: LayoutHints = {
            "kv_layout": "HND",
            "num_kv_heads": self.num_kv_heads,
            "tokens_per_block": tokens_per_block,
            "head_dim": self.head_dim,
        }
        kv_format, normalized = normalize_kv_and_discover_format(
            kv_cache_tensor, EngineType.TRTLLM, layout_hints=layout_hints
        )
        if not isinstance(normalized, torch.Tensor):
            raise ValueError(
                "TRT-LLM normalize must return a bare tensor; "
                f"got {type(normalized).__name__}"
            )
        self._kv_format = kv_format
        self.kv_cache_tensor = normalized

        shape_desc = lmc_ops.PageBufferShapeDesc()
        shape_desc.kv_size = kv_factor
        shape_desc.nl = num_layers
        shape_desc.nb = num_blocks
        shape_desc.bs = tokens_per_block
        shape_desc.nh = self.num_kv_heads
        shape_desc.hs = self.head_dim
        shape_desc.element_size = normalized.element_size()
        self.shape_desc = shape_desc

        self.paged_buffer_ptrs = torch.tensor(
            get_group_data_ptrs(normalized, kv_format, list(range(num_layers))),
            dtype=torch.int64,
            device=self.device,
        )

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    def _stage_block_ids(self, block_ids: List[int]) -> torch.Tensor:
        return torch.tensor(block_ids, dtype=torch.int64, device=self.device)

    def _get_chunk_block_ids(
        self, block_ids: List[int], start: int
    ) -> Optional[List[int]]:
        assert self.blocks_per_chunk is not None
        chunk_idx = start // self.chunk_size
        bs = chunk_idx * self.blocks_per_chunk
        be = bs + self.blocks_per_chunk
        if be > len(block_ids):
            return None
        return block_ids[bs:be]

    def _transfer(
        self,
        tensor_ptr: int,
        block_ids: List[int],
        direction: "lmc_ops.TransferDirection",
        stream: torch.cuda.Stream,
    ) -> None:
        with torch.cuda.stream(stream):
            block_ids_gpu = self._stage_block_ids(block_ids)
            lmc_ops.multi_layer_block_kv_transfer(
                self.paged_buffer_ptrs,
                [tensor_ptr],
                block_ids_gpu,
                self.device,
                direction,
                self.shape_desc,
                self.chunk_size,
                self._kv_format,
                0,  # skip_prefix_n_blocks
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        if self.kv_cache_tensor is None:
            raise RuntimeError("register_kv_caches must be called before to_gpu")
        chunk_blocks = self._get_chunk_block_ids(kwargs.get("block_ids", []), start)
        if chunk_blocks is None or memory_obj.tensor is None:
            return
        self._transfer(
            memory_obj.tensor.data_ptr(),
            chunk_blocks,
            lmc_ops.TransferDirection.H2D,
            self.load_stream,
        )

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        if self.kv_cache_tensor is None:
            raise RuntimeError("register_kv_caches must be called before from_gpu")
        chunk_blocks = self._get_chunk_block_ids(kwargs.get("block_ids", []), start)
        if chunk_blocks is None or memory_obj.tensor is None:
            return
        self._transfer(
            memory_obj.tensor.data_ptr(),
            chunk_blocks,
            lmc_ops.TransferDirection.D2H,
            self.store_stream,
        )

    def _batched_transfer(
        self,
        memory_objs: List[MemoryObj],
        starts: List[int],
        block_ids: List[int],
        direction: "lmc_ops.TransferDirection",
        stream: torch.cuda.Stream,
    ) -> None:
        valid: List[Tuple[MemoryObj, List[int]]] = []
        for memory_obj, start in zip(memory_objs, starts, strict=False):
            if isinstance(memory_obj, list) or memory_obj.tensor is None:
                continue
            chunk_blocks = self._get_chunk_block_ids(block_ids, start)
            if chunk_blocks is not None:
                valid.append((memory_obj, chunk_blocks))

        with torch.cuda.stream(stream):
            for i in range(0, len(valid), self._batch_size):
                batch = valid[i : i + self._batch_size]
                all_block_ids: List[int] = []
                ptrs: List[int] = []
                for mo, blocks in batch:
                    all_block_ids.extend(blocks)
                    ptrs.append(mo.tensor.data_ptr())  # type: ignore[union-attr]
                block_ids_gpu = self._stage_block_ids(all_block_ids)
                lmc_ops.multi_layer_block_kv_transfer(
                    self.paged_buffer_ptrs,
                    ptrs,
                    block_ids_gpu,
                    self.device,
                    direction,
                    self.shape_desc,
                    self.chunk_size,
                    self._kv_format,
                    0,
                )

    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ) -> None:
        if self.kv_cache_tensor is None:
            raise RuntimeError(
                "register_kv_caches must be called before batched_from_gpu"
            )
        self._batched_transfer(
            memory_objs,  # type: ignore[arg-type]
            starts,
            kwargs.get("block_ids", []),
            lmc_ops.TransferDirection.D2H,
            self.store_stream,
        )

    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        if memory_objs is None or starts is None or ends is None:
            return
        if self.kv_cache_tensor is None:
            raise RuntimeError(
                "register_kv_caches must be called before batched_to_gpu"
            )
        self._batched_transfer(
            memory_objs,  # type: ignore[arg-type]
            starts,
            kwargs.get("block_ids", []),
            lmc_ops.TransferDirection.H2D,
            self.load_stream,
        )
