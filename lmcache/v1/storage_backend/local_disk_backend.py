# SPDX-License-Identifier: Apache-2.0
# Standard
import bisect
from concurrent.futures import Future, ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import hashlib
import logging
import os
import threading
import time

# Third Party
import torch

# First Party
try:
    # First Party
    from lmcache import torch_dev, torch_device_type
except ImportError:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch_dev, torch_device_type = torch.xpu, "xpu"
    elif hasattr(torch, "hpu") and torch.hpu.is_available():
        torch_dev, torch_device_type = torch.hpu, "hpu"
    elif torch.cuda.is_available():
        torch_dev, torch_device_type = torch.cuda, "cuda"
    else:
        torch_dev, torch_device_type = torch, "cpu"
from lmcache.logging import init_logger
import lmcache.c_ops as lmc_ops
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.kv_object_store import (
    KVObjectByteRange,
    KVObjectId,
    KVObjectMetadataStore,
    KVObjectPoolIO,
    KVObjectPoolLayout,
    KVObjectRecord,
    KVObjectState,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.batched_message_sender import BatchedMessageSender
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.cpu_raw_writer import CPURawStageQueue
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.path_sharder import PathSharder

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class _TuttiProfileLogFilter(logging.Filter):
    """Suppress KV object-store profile records when profiling is disabled."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether a log record should be emitted."""
        value = os.getenv("LMCACHE_TUTTI_PROFILE", "0")
        enabled = value.lower() in {"1", "true", "yes", "on"}
        return enabled or not str(record.msg).startswith("KV_OBJECT_STORE_PROFILE")


logger.addFilter(_TuttiProfileLogFilter())

_DSV4_HCA_DEFERRED_RETRIEVE_ROLE = "hca_deferred_retrieve"
_DSV4_CSA_DEFERRED_RETRIEVE_ROLE = "csa_deferred_retrieve"
_DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE = "csa_hca_deferred_retrieve"
_DSV4_HCA_SLAB_ROLE = "hca_attention_kv_slab"
_DSV4_HCA_LAYER_MAJOR_ROLE = "hca_attention_kv_layer_major"
_DSV4_CSA_LAYER_MAJOR_ROLE = "csa_attention_kv_layer_major"
_DSV4_INDEXER_LAYER_MAJOR_ROLE = "csa_indexer_cache_layer_major"
_DSV4_CSA_STREAMING_LAYOUT_VERSION = 2


def _kv_object_prefix_view(
    record: KVObjectRecord,
    logical_length: int,
) -> Optional[KVObjectRecord]:
    """Return a prefix view of a contiguous or composed object record."""
    if logical_length <= 0 or logical_length > record.length:
        return None
    if logical_length == record.length:
        return record
    prefix_ranges: list[KVObjectByteRange] = []
    remaining = logical_length
    target_offset = 0
    for byte_range in record.read_ranges:
        if remaining <= 0:
            break
        range_length = min(byte_range.length, remaining)
        prefix_ranges.append(
            KVObjectByteRange(
                offset=byte_range.offset,
                length=range_length,
                target_offset=target_offset,
            )
        )
        target_offset += range_length
        remaining -= range_length
    if remaining:
        return None
    return record.with_byte_ranges(prefix_ranges, length=logical_length)


@dataclass(frozen=True, slots=True)
class _CSAStreamingObject:
    """One immutable object referenced by a CSA layout generation."""

    logical_role: str
    layer_id: int
    object_id: KVObjectId
    length: int


@dataclass(frozen=True, slots=True)
class _CSAStreamingLayout:
    """Atomically published set of objects for one cached chunk key."""

    generation: str
    layout_version: int
    covered_tokens: int
    required_entries: frozenset[tuple[str, int]]
    objects: tuple[_CSAStreamingObject, ...]

    def find_object(
        self,
        logical_role: str,
        layer_id: int,
    ) -> Optional[_CSAStreamingObject]:
        """Return one logical layout entry when present."""
        for entry in self.objects:
            if entry.logical_role == logical_role and entry.layer_id == layer_id:
                return entry
        return None

    def find(self, logical_role: str, layer_id: int) -> Optional[KVObjectId]:
        """Return the physical object id for one logical layout entry."""
        entry = self.find_object(logical_role, layer_id)
        return entry.object_id if entry is not None else None


@dataclass(slots=True)
class _CSAStreamingLayoutBuild:
    """Invisible generation assembled before the active-layout swap."""

    generation: str
    covered_tokens: int
    required_entries: frozenset[tuple[str, int]]
    objects: dict[tuple[str, int], _CSAStreamingObject]
    terminal: bool
    admission_started_at: float


@dataclass(slots=True)
class _CSAStagedWriteBatch:
    """Metadata ticket completed only after all staged waves reach SSD."""

    records: tuple[KVObjectRecord, ...]
    keys: tuple[CacheEngineKey, ...]
    remaining_waves: int
    failed: bool = False


@dataclass(slots=True)
class _LayerMajorSidecar:
    """One physical sidecar being prepared for a layer-major generation."""

    source_role: str
    layer_id: int
    object_role: str
    physical_role: str
    payload_nbytes: int
    segments: list[memoryview]
    segment_sources: list[Optional[tuple[int, int]]]
    record: KVObjectRecord
    ready_record: Optional[KVObjectRecord] = None


LayerSegmentLayout = tuple[
    tuple[tuple[str, int], tuple[tuple[int, int], ...]],
    ...,
]

KVObjectLayerMajorPacker = Callable[
    [
        Sequence[int],
        Sequence[int],
        Sequence[int],
        Sequence[int],
        Sequence[int],
        torch.Tensor,
    ],
    float,
]


def _layer_major_wave_record(
    wave: Sequence[_LayerMajorSidecar],
) -> tuple[KVObjectRecord, int]:
    """Return the physical record and byte length for one write wave."""
    if not wave:
        raise ValueError("layer-major wave must be non-empty")
    if len(wave) == 1:
        return wave[0].record, wave[0].payload_nbytes
    first_record = wave[0].record
    last_record = wave[-1].record
    wave_nbytes = last_record.offset + last_record.aligned_length - first_record.offset
    return (
        KVObjectRecord(
            object_id=first_record.object_id,
            pool_id=first_record.pool_id,
            offset=first_record.offset,
            length=wave_nbytes,
            aligned_length=wave_nbytes,
            shape=(wave_nbytes,),
            dtype="torch.uint8",
        ),
        wave_nbytes,
    )


def _pack_layer_major_wave(
    wave: Sequence[_LayerMajorSidecar],
    buffer: bytearray,
) -> tuple[KVObjectRecord, memoryview, float]:
    """Pack one layer-major wave with one GIL-free native copy batch.

    The source snapshot remains owned by the deferred admission until this
    function returns.  Python builds only compact pointer descriptors; the
    byte copies run in ``lmc_ops.batched_memcpy`` with the GIL released so
    background packing cannot stall vLLM's decode thread.
    """
    started = time.perf_counter()
    writer_record, wave_nbytes = _layer_major_wave_record(wave)
    if len(buffer) < wave_nbytes:
        raise ValueError("layer-major pack buffer is smaller than the wave")

    # A new bytearray is already zero-filled, which supplies every alignment
    # gap without per-sidecar Python allocation or slice assignment.
    destination_base = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    source_ptrs: list[int] = []
    destination_ptrs: list[int] = []
    copy_sizes: list[int] = []
    target_offset = 0
    for sidecar in wave:
        if len(sidecar.segment_sources) != len(sidecar.segments):
            raise ValueError("layer-major segment source descriptors disagree")
        for segment, source_descriptor in zip(
            sidecar.segments,
            sidecar.segment_sources,
            strict=True,
        ):
            source = segment.cast("B")
            source_nbytes = len(source)
            if source_nbytes <= 0:
                continue
            if source_descriptor is None:
                source_ptr = ctypes.addressof(ctypes.c_char.from_buffer(source))
            else:
                source_base_ptr, source_offset = source_descriptor
                source_ptr = source_base_ptr + source_offset
            source_ptrs.append(source_ptr)
            destination_ptrs.append(destination_base + target_offset)
            copy_sizes.append(source_nbytes)
            target_offset += source_nbytes
        if len(wave) > 1:
            padding_nbytes = sidecar.record.aligned_length - sidecar.payload_nbytes
            if padding_nbytes > 0:
                target_offset += padding_nbytes
    if target_offset != wave_nbytes:
        raise RuntimeError("layer-major packed payload does not match wave allocations")
    lmc_ops.batched_memcpy(source_ptrs, destination_ptrs, copy_sizes)
    return (
        writer_record,
        memoryview(buffer)[:wave_nbytes],
        time.perf_counter() - started,
    )


def _pack_layer_major_wave_on_gpu(
    wave: Sequence[_LayerMajorSidecar],
    destination: torch.Tensor,
    packer: KVObjectLayerMajorPacker,
) -> tuple[KVObjectRecord, memoryview, float]:
    """Pack one wave through mapped pinned memory on a CUDA stream."""
    started = time.perf_counter()
    writer_record, wave_nbytes = _layer_major_wave_record(wave)
    if destination.device.type != "cpu" or destination.dtype != torch.uint8:
        raise ValueError("GPU pack destination must be a CPU uint8 tensor")
    if destination.numel() < wave_nbytes:
        raise ValueError("GPU pack destination is smaller than the wave")

    unique_sources: list[int] = []
    source_lookup: dict[int, int] = {}
    source_indices: list[int] = []
    source_offsets: list[int] = []
    destination_offsets: list[int] = []
    lengths: list[int] = []
    target_offset = 0
    for sidecar in wave:
        if len(sidecar.segment_sources) != len(sidecar.segments):
            raise ValueError("layer-major segment source descriptors disagree")
        for segment, source in zip(
            sidecar.segments,
            sidecar.segment_sources,
            strict=True,
        ):
            if source is None:
                raise ValueError("layer-major source is not pinned CPU memory")
            source_ptr, source_offset = source
            source_idx = source_lookup.get(source_ptr)
            if source_idx is None:
                source_idx = len(unique_sources)
                source_lookup[source_ptr] = source_idx
                unique_sources.append(source_ptr)
            segment_nbytes = len(segment)
            source_indices.append(source_idx)
            source_offsets.append(source_offset)
            destination_offsets.append(target_offset)
            lengths.append(segment_nbytes)
            target_offset += segment_nbytes
        if len(wave) > 1:
            target_offset += sidecar.record.aligned_length - sidecar.payload_nbytes
    if target_offset != wave_nbytes:
        raise RuntimeError("GPU-packed layer-major wave length mismatch")

    destination_view = destination[:wave_nbytes]
    # Reused pack slots may retain bytes in alignment gaps. Clear on the CPU
    # before the mapped-host kernel overwrites every logical segment.
    destination_view.zero_()
    packer(
        unique_sources,
        source_indices,
        source_offsets,
        destination_offsets,
        lengths,
        destination_view,
    )
    return (
        writer_record,
        memoryview(destination_view.numpy()),
        time.perf_counter() - started,
    )


def _pack_layer_major_wave_with_fallback(
    wave: Sequence[_LayerMajorSidecar],
    cpu_buffer: Optional[bytearray],
    gpu_buffer: Optional[torch.Tensor],
    packer: Optional[KVObjectLayerMajorPacker],
) -> tuple[KVObjectRecord, memoryview, float]:
    """Use GPU packing when available and fail closed to the CPU packer."""
    if gpu_buffer is not None and packer is not None:
        try:
            return _pack_layer_major_wave_on_gpu(wave, gpu_buffer, packer)
        except Exception:
            logger.exception(
                "GPU layer-major pack failed; using CPU pack for this wave"
            )
    if cpu_buffer is None:
        _writer_record, wave_nbytes = _layer_major_wave_record(wave)
        cpu_buffer = bytearray(wave_nbytes)
    return _pack_layer_major_wave(wave, cpu_buffer)


KVObjectRawWriter = Callable[
    [KVObjectRecord, memoryview],
    tuple[Sequence[tuple[int, int, int]], float],
]
KVObjectPackWaiter = Callable[[], None]


def _clip_raw_extents(
    raw_extents: Sequence[tuple[int, int, int]],
    *,
    offset: int,
    aligned_length: int,
) -> tuple[tuple[int, int, int], ...]:
    """Clip one raw-write wave's extents to a sidecar allocation."""
    object_end = offset + aligned_length
    clipped: list[tuple[int, int, int]] = []
    covered = 0
    for file_offset, slba, n_sectors in raw_extents:
        extent_end = file_offset + n_sectors * 512
        write_start = max(offset, file_offset)
        write_end = min(object_end, extent_end)
        if write_start >= write_end:
            continue
        extent_skip = write_start - file_offset
        write_nbytes = write_end - write_start
        if extent_skip % 512 != 0 or write_nbytes % 512 != 0:
            raise ValueError("raw write wave produced a non-sector-aligned extent")
        clipped.append(
            (
                write_start,
                slba + extent_skip // 512,
                write_nbytes // 512,
            )
        )
        covered += write_nbytes
    if covered != aligned_length:
        raise RuntimeError(
            "raw write wave does not cover sidecar allocation "
            f"{offset}:{object_end}; covered={covered}"
        )
    return tuple(clipped)


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _env_flag_default(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _glm_dsa_layer_major_enabled() -> bool:
    """Return whether GLM uses compact main plus layer-major sidecars."""
    return _env_flag("LMCACHE_GLM_DSA_LAYER_MAJOR") or _env_flag(
        "LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH"
    )


def _env_int(name: str, default: int = 0) -> int:
    """Return an integer environment value, falling back on parse errors."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _write_wave_limit_bytes(staging_bytes: Optional[int]) -> int:
    """Return the physical write quantum without changing logical chunks.

    ``LMCACHE_DSV4_WRITE_QUANTUM_MB`` is the scheduler-facing setting.  The
    older raw-wave setting remains the fallback so existing deployments keep
    their behavior until they opt in to a smaller preemption quantum.

    Args:
        staging_bytes: Tutti staging capacity, or ``None`` when unavailable.

    Returns:
        The effective byte limit, or zero when bounded grouping is disabled.
    """
    legacy_wave_mb = _env_int("LMCACHE_DSV4_RAW_WRITE_WAVE_MB", 128)
    quantum_mb = _env_int("LMCACHE_DSV4_WRITE_QUANTUM_MB", legacy_wave_mb)
    if staging_bytes is None or staging_bytes <= 0 or quantum_mb <= 0:
        return 0
    return min(staging_bytes, quantum_mb * 1024**2)


def _select_rank_int(value: Any, rank_id: int) -> int:
    """Return an integer config value, optionally selected by rank.

    Args:
        value: Either one integer-like value or a comma-separated/list value.
        rank_id: Local rank used to select from multi-value configs.

    Returns:
        The selected integer.

    Raises:
        ValueError: If a rank-indexed value does not contain this rank.
    """
    if isinstance(value, (list, tuple)):
        values = [int(item) for item in value]
    else:
        text = str(value)
        if "," not in text:
            return int(text)
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if rank_id >= len(values):
        raise ValueError(
            f"rank {rank_id} requested from {len(values)} configured values"
        )
    return values[rank_id]


# TODO(Jiayi): handle cases where cache is repetitvely prefetched.
class LocalDiskWorker:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.put_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        self.prefetch_lock = threading.Lock()
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}

        # TODO(Jiayi): make executor and its parameters configurable
        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=4)
        self.loop = loop
        self._closed = False

    async def submit_task(
        self,
        task_type: str,
        task: Callable,
        *args,
        **kwargs,
    ) -> Any:
        if task_type == "prefetch":
            priority = 0
            # self.insert_prefetch_task(kwargs["key"], None)
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return await self.executor.submit_job(
            task,
            *args,
            priority=priority,
            **kwargs,
        )

    def remove_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            if key in self.put_tasks:
                self.put_tasks.remove(key)
            else:
                logger.warning(f"Key {key} not found in put tasks.")

    def insert_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            self.put_tasks.append(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def close(self):
        # Gracefully shut down the executor
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(wait=True)


class LocalDiskBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = torch_device_type,
        lmcache_worker: Optional["LMCacheWorker"] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ):
        if torch_dev.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()

        self.dst_device = dst_device

        self.local_cpu_backend = local_cpu_backend
        self._chunk_size = int(config.chunk_size)

        self.disk_lock = threading.Lock()

        assert config.local_disk is not None

        sharder = PathSharder(
            raw_csv=config.local_disk,
            strategy=config.local_disk_path_sharding,
            dst_device=dst_device,
            create_dirs=True,
        )
        self.path: str = sharder.selected

        logger.info(
            "Local disk cache path: %s (device %s, %d path(s) configured)",
            self.path,
            dst_device,
            len(sharder.all_paths),
        )

        self.loop = loop

        self.use_local_cpu = config.local_cpu

        # Block size (for file system I/O)
        stat = os.statvfs(self.path)
        self.os_disk_bs = stat.f_bsize
        self.use_odirect = False

        if config.extra_config is not None:
            self.use_odirect = config.extra_config.get("use_odirect", False)
        logger.info("Using O_DIRECT for disk I/O: %s", self.use_odirect)

        enable_object_store = _env_flag("LMCACHE_KV_OBJECT_STORE_ENABLE")
        if config.extra_config is not None:
            enable_object_store = bool(
                config.extra_config.get(
                    "kv_object_store_enable",
                    enable_object_store,
                )
            )
        self.kv_object_store_enabled = enable_object_store
        self.kv_object_pool_layout: Optional[KVObjectPoolLayout] = None
        self.kv_object_metadata_store: Optional[KVObjectMetadataStore] = None
        self.kv_object_pool_io: Optional[KVObjectPoolIO] = None
        self.kv_object_store_lock = threading.Lock()
        self._csa_layout_lock = threading.RLock()
        self._csa_active_layouts: dict[str, _CSAStreamingLayout] = {}
        self._csa_pending_layouts: dict[str, _CSAStreamingLayoutBuild] = {}
        self._csa_pending_layout_keys: dict[str, CacheEngineKey] = {}
        self._csa_staged_write_counts: dict[str, int] = {}
        self._csa_required_entries_by_mode: dict[
            tuple[bool, bool], frozenset[tuple[str, int]]
        ] = {}
        # Separate 1-worker executor for fire-and-forget HCA NVMe writes.
        # HCA slab/deferred writes are large and slow; submitting them here
        # keeps them off the main disk_worker pool so they cannot saturate
        # the 4-worker capacity that regular KV chunk writes need.
        self._hca_write_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lmcache-hca-disk-write"
        )
        self._layer_major_pack_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lmcache-layer-major-pack",
        )
        self._layer_segment_layout_cache: dict[
            tuple[tuple[tuple[int, ...], str], ...],
            LayerSegmentLayout,
        ] = {}
        self._diagnose_contains_misses = _env_flag("LMCACHE_DISK_CONTAINS_DIAGNOSTICS")
        self._contains_miss_log_counts: dict[str, int] = {}
        self._object_write_skip_log_counts: dict[str, int] = {}
        self.kv_object_tutti_raw_enabled = _env_flag(
            "LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE"
        )
        self.kv_object_cpu_raw_write_enabled = _env_flag_default(
            "LMCACHE_DSV4_CPU_RAW_WRITE",
            True,
        )
        self.kv_object_cpu_raw_write_mibps = float(
            os.getenv("LMCACHE_DSV4_CPU_RAW_WRITE_MIBPS", "0")
        )
        self.kv_object_cpu_raw_write_block_bytes = (
            _env_int("LMCACHE_DSV4_CPU_RAW_WRITE_BLOCK_MB", 64) * 1024**2
        )
        self.kv_object_cpu_stage_enabled = (
            self.kv_object_cpu_raw_write_enabled
            and _env_flag("LMCACHE_TUTTI_CPU_STAGE_ENABLE")
        )
        self.kv_object_cpu_stage_bytes = (
            _env_int("LMCACHE_TUTTI_CPU_STAGE_GIB", 16) * 1024**3
        )
        self._cpu_raw_stage_queue: Optional[CPURawStageQueue] = (
            CPURawStageQueue(
                self.kv_object_cpu_stage_bytes,
                thread_name_prefix="lmcache-cpu-raw-stage",
            )
            if self.kv_object_cpu_stage_enabled
            else None
        )
        self.kv_object_tutti_raw_cold_store_enabled = (
            self.kv_object_tutti_raw_enabled
            and _env_flag_default(
                "LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_COLD_STORE",
                True,
            )
        )
        rank_id = int(str(dst_device).removeprefix("cuda:") or "0")
        self.kv_object_tutti_raw_base_lba = _select_rank_int(
            os.getenv("LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_BASE_LBA", "0"),
            rank_id,
        )
        raw_region_path = os.getenv("LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_REGION_PATH")
        self.kv_object_tutti_raw_region_path: Optional[str] = (
            str(raw_region_path) if raw_region_path else None
        )
        self.kv_object_tutti_raw_region_extents: list[tuple[int, int, int]] = []
        self.kv_object_tutti_raw_writer: Optional[KVObjectRawWriter] = None
        self.kv_object_layer_major_pack_waiter: Optional[KVObjectPackWaiter] = None
        self.kv_object_gpu_layer_major_packer: Optional[KVObjectLayerMajorPacker] = None
        self._kv_object_tutti_raw_writer_ready = threading.Event()
        self._kv_object_raw_region_full_at_s = 0.0
        self.kv_object_tutti_raw_staging_bytes: Optional[int] = None
        if self.kv_object_store_enabled:
            slot_mb = int(os.getenv("LMCACHE_KV_OBJECT_STORE_SLOT_MB", "128"))
            capacity = int(os.getenv("LMCACHE_KV_OBJECT_STORE_CAPACITY", "2048"))
            raw_slot_mb = _env_int(
                "LMCACHE_ABLATION_TUTTI_SLOT_MB",
                _env_int("LMCACHE_TUTTI_SLOT_MB", 0),
            )
            raw_n_slots = _env_int(
                "LMCACHE_ABLATION_TUTTI_N_SLOTS",
                _env_int("LMCACHE_TUTTI_N_SLOTS", 0),
            )
            if config.extra_config is not None:
                slot_mb = int(
                    config.extra_config.get("kv_object_store_slot_mb", slot_mb)
                )
                capacity = int(
                    config.extra_config.get("kv_object_store_capacity", capacity)
                )
                raw_slot_mb = int(config.extra_config.get("tutti_slot_mb", raw_slot_mb))
                raw_n_slots = int(config.extra_config.get("tutti_n_slots", raw_n_slots))
                self.kv_object_tutti_raw_enabled = bool(
                    config.extra_config.get(
                        "kv_object_store_tutti_raw_enable",
                        self.kv_object_tutti_raw_enabled,
                    )
                )
                self.kv_object_cpu_raw_write_enabled = bool(
                    config.extra_config.get(
                        "kv_object_store_cpu_raw_write_enable",
                        self.kv_object_cpu_raw_write_enabled,
                    )
                )
                self.kv_object_cpu_raw_write_mibps = float(
                    config.extra_config.get(
                        "kv_object_store_cpu_raw_write_mibps",
                        self.kv_object_cpu_raw_write_mibps,
                    )
                )
                self.kv_object_cpu_raw_write_block_bytes = (
                    int(
                        config.extra_config.get(
                            "kv_object_store_cpu_raw_write_block_mb",
                            self.kv_object_cpu_raw_write_block_bytes // 1024**2,
                        )
                    )
                    * 1024**2
                )
                self.kv_object_tutti_raw_cold_store_enabled = bool(
                    config.extra_config.get(
                        "kv_object_store_tutti_raw_cold_store",
                        self.kv_object_tutti_raw_cold_store_enabled,
                    )
                )
                self.kv_object_tutti_raw_base_lba = _select_rank_int(
                    config.extra_config.get(
                        "kv_object_store_tutti_raw_base_lba",
                        self.kv_object_tutti_raw_base_lba,
                    ),
                    rank_id,
                )
                raw_region_config = config.extra_config.get(
                    "kv_object_store_tutti_raw_region_path",
                    self.kv_object_tutti_raw_region_path,
                )
                if raw_region_config:
                    if isinstance(raw_region_config, (list, tuple)):
                        if rank_id >= len(raw_region_config):
                            raise ValueError(
                                "rank requested beyond configured raw regions"
                            )
                        self.kv_object_tutti_raw_region_path = str(
                            raw_region_config[rank_id]
                        )
                    else:
                        raw_region_text = str(raw_region_config)
                        if "," in raw_region_text:
                            regions = [
                                item.strip() for item in raw_region_text.split(",")
                            ]
                            if rank_id >= len(regions):
                                raise ValueError(
                                    "rank requested beyond configured raw regions"
                                )
                            per_rank = regions[rank_id]
                            if per_rank:
                                self.kv_object_tutti_raw_region_path = per_rank
                        else:
                            self.kv_object_tutti_raw_region_path = raw_region_text
                self.kv_object_tutti_raw_cold_store_enabled = (
                    self.kv_object_tutti_raw_enabled
                    and self.kv_object_tutti_raw_cold_store_enabled
                )
            if self.kv_object_cpu_raw_write_mibps < 0:
                raise ValueError(
                    "kv_object_store_cpu_raw_write_mibps must be non-negative"
                )
            if (
                self.kv_object_cpu_raw_write_block_bytes <= 0
                or self.kv_object_cpu_raw_write_block_bytes % 512
            ):
                raise ValueError(
                    "kv_object_store_cpu_raw_write_block_mb must be positive"
                )
            self.kv_object_cpu_raw_write_enabled = (
                self.kv_object_tutti_raw_enabled
                and self.kv_object_cpu_raw_write_enabled
            )
            if raw_slot_mb > 0 and raw_n_slots > 0:
                self.kv_object_tutti_raw_staging_bytes = (
                    raw_slot_mb * 1024 * 1024 * raw_n_slots
                )
            rank_name = str(dst_device).removeprefix("cuda:")
            pool_id = f"rank{rank_name}-full"
            pool_path = Path(self.path) / "_kv_object_store" / f"{pool_id}.pool"
            if pool_path.exists():
                pool_path.unlink()
            self.kv_object_pool_layout = KVObjectPoolLayout(
                pool_id=pool_id,
                pool_path=pool_path,
                slot_bytes=slot_mb * 1024 * 1024,
                capacity=capacity,
                dense=True,
                materialize_file=not self.kv_object_tutti_raw_cold_store_enabled,
            )
            self.kv_object_metadata_store = KVObjectMetadataStore()
            self.kv_object_pool_io = KVObjectPoolIO({pool_id: pool_path})
            logger.info(
                "KV object store enabled: pool_id=%s path=%s slot_mb=%d "
                "capacity=%d tutti_raw=%s raw_cold_store=%s raw_base_lba=%d "
                "raw_staging_bytes=%s cpu_raw_write=%s cpu_raw_mibps=%.1f "
                "cpu_raw_block_mib=%.1f",
                pool_id,
                pool_path,
                slot_mb,
                capacity,
                self.kv_object_tutti_raw_enabled,
                self.kv_object_tutti_raw_cold_store_enabled,
                self.kv_object_tutti_raw_base_lba,
                self.kv_object_tutti_raw_staging_bytes,
                self.kv_object_cpu_raw_write_enabled,
                self.kv_object_cpu_raw_write_mibps,
                self.kv_object_cpu_raw_write_block_bytes / 1024**2,
            )
            if self.kv_object_tutti_raw_region_path:
                logger.info(
                    "KV object Tutti raw region path: pool_id=%s path=%s",
                    pool_id,
                    self.kv_object_tutti_raw_region_path,
                )

        self.disk_worker = LocalDiskWorker(loop)

        # TODO(Jiayi): We need a disk space allocator to avoid fragmentation
        # and hide the following details away from the backend.
        self.max_cache_size = int(config.max_local_disk_size * 1024**3)
        self.current_cache_size = 0.0

        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []

        self.lmcache_worker = lmcache_worker
        self.metadata = metadata
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

        # Batched message sender for controller communication
        self.batched_msg_sender: Optional[BatchedMessageSender] = None

        # Initialize batched message sender
        if lmcache_worker and metadata is not None:
            self.batched_msg_sender = BatchedMessageSender(
                metadata=metadata,
                config=config,
                location=str(self),
                lmcache_worker=lmcache_worker,
            )
        else:
            logger.warning("Controller message sender is not initialized")

    def __str__(self) -> str:
        return "LocalDiskBackend"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    def _key_to_object_id(
        self,
        key: CacheEngineKey,
        *,
        layer_id: int = 0,
        role: str = "full",
    ) -> KVObjectId:
        return KVObjectId(
            model_id=key.model_name,
            parallel_config_id=f"world{key.world_size}",
            rank=key.worker_id,
            layer_id=layer_id,
            role=role,
            block_id=key.chunk_hash_hex,
        )

    @staticmethod
    def _generation_object_role(logical_role: str, generation: str) -> str:
        """Return an immutable physical role for one layout generation."""
        return f"{logical_role}@g{generation}"

    @staticmethod
    def _streaming_logical_roles() -> frozenset[str]:
        """Return roles whose physical ids are resolved through a manifest."""
        return frozenset(
            {
                _DSV4_CSA_DEFERRED_RETRIEVE_ROLE,
                _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE,
                _DSV4_CSA_LAYER_MAJOR_ROLE,
                _DSV4_HCA_LAYER_MAJOR_ROLE,
                _DSV4_INDEXER_LAYER_MAJOR_ROLE,
            }
        )

    def _active_csa_layout(self, key: CacheEngineKey) -> Optional[_CSAStreamingLayout]:
        """Return the atomically published layout for a cache key."""
        with self._csa_layout_lock:
            return self._csa_active_layouts.get(key.to_string())

    def _resolve_streaming_object_id(
        self,
        key: CacheEngineKey,
        logical_role: str,
        layer_id: int,
    ) -> Optional[KVObjectId]:
        """Resolve one logical role through the key's active generation."""
        entry = self._resolve_streaming_object(key, logical_role, layer_id)
        return entry.object_id if entry is not None else None

    def _resolve_streaming_object(
        self,
        key: CacheEngineKey,
        logical_role: str,
        layer_id: int,
    ) -> Optional[_CSAStreamingObject]:
        """Resolve one logical object entry through the active generation."""
        layout = self._active_csa_layout(key)
        if layout is None or not self._csa_layout_matches_runtime(layout):
            return None
        return layout.find_object(logical_role, int(layer_id))

    def _required_csa_streaming_entries(self) -> set[tuple[str, int]]:
        """Return every logical object required by the active ON layout."""
        required: set[tuple[str, int]] = {(self._csa_streaming_compact_role(), 0)}
        if self.metadata is None:
            return required
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None:
            return required
        role_map = {
            "csa_attention_kv": _DSV4_CSA_LAYER_MAJOR_ROLE,
            "csa_indexer_cache": _DSV4_INDEXER_LAYER_MAJOR_ROLE,
        }
        if _env_flag("LMCACHE_DSV4_HCA_WALKER"):
            role_map["hca_attention_kv"] = _DSV4_HCA_LAYER_MAJOR_ROLE
        for group in klg_manager.kv_layer_groups:
            logical_role = role_map.get(self._kv_group_role(group, group.dtype))
            if logical_role is None:
                continue
            required.update(
                (logical_role, int(layer_id)) for layer_id in group.layer_indices
            )
        return required

    def _csa_streaming_generation(
        self,
        keys: Sequence[CacheEngineKey],
        covered_tokens: int,
    ) -> str:
        """Return a deterministic id for one immutable admission generation.

        Deterministic ids let a retry reuse physical objects written by an
        interrupted admission instead of leaking a fresh set of pool
        allocations.  The layout version and compact-main role are part of the
        digest so incompatible ON layouts never share physical object ids.

        Args:
            keys: Content-addressed chunk keys in admission order.
            covered_tokens: Exact logical-token coverage of this generation.

        Returns:
            A compact hexadecimal generation identifier.
        """
        digest = hashlib.sha256()
        digest.update(str(_DSV4_CSA_STREAMING_LAYOUT_VERSION).encode("ascii"))
        digest.update(b"\0")
        digest.update(self._csa_streaming_compact_role().encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(covered_tokens)).encode("ascii"))
        for key in keys:
            digest.update(b"\0")
            digest.update(key.to_string().encode("utf-8"))
        return digest.hexdigest()[:20]

    def _csa_layout_matches_runtime(self, layout: _CSAStreamingLayout) -> bool:
        """Return whether a manifest matches the currently requested ON mode."""
        mode = (
            _env_flag("LMCACHE_INDEXER_ENABLE_PREFETCH"),
            _env_flag("LMCACHE_DSV4_HCA_WALKER"),
        )
        required_entries = self._csa_required_entries_by_mode.get(mode)
        if required_entries is None:
            required_entries = frozenset(self._required_csa_streaming_entries())
            self._csa_required_entries_by_mode[mode] = required_entries
        return layout.required_entries == required_entries

    def _csa_layout_records_readable(self, layout: _CSAStreamingLayout) -> bool:
        """Validate all manifest entries and their physical record metadata."""
        if self.kv_object_metadata_store is None:
            return False
        present = {(entry.logical_role, entry.layer_id) for entry in layout.objects}
        if layout.layout_version != _DSV4_CSA_STREAMING_LAYOUT_VERSION:
            logger.warning(
                "CSA layout validation failed: generation=%s reason=layout_version",
                layout.generation,
            )
            return False
        if not layout.required_entries.issubset(present):
            logger.warning(
                "CSA layout validation failed: generation=%s "
                "reason=missing_entries missing=%s",
                layout.generation,
                sorted(layout.required_entries - present),
            )
            return False
        records = self.kv_object_metadata_store.get_many(
            [entry.object_id for entry in layout.objects],
            ready_only=False,
        )
        for entry, record in zip(layout.objects, records, strict=True):
            if entry.length == 0:
                if entry.logical_role not in {
                    _DSV4_CSA_DEFERRED_RETRIEVE_ROLE,
                    _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE,
                }:
                    logger.warning(
                        "CSA layout validation failed: generation=%s "
                        "reason=unexpected_empty role=%s layer=%d",
                        layout.generation,
                        entry.logical_role,
                        entry.layer_id,
                    )
                    return False
                continue
            read_record = (
                _kv_object_prefix_view(record, entry.length)
                if record is not None
                else None
            )
            if not self._ready_tutti_raw_record(read_record):
                logger.warning(
                    "CSA layout validation failed: generation=%s "
                    "reason=object_unreadable role=%s layer=%d expected=%d "
                    "record=%s",
                    layout.generation,
                    entry.logical_role,
                    entry.layer_id,
                    entry.length,
                    record,
                )
                return False
        return True

    def _active_csa_layout_ready(self, key: CacheEngineKey) -> bool:
        """Return whether one key has a published compatible generation.

        Publication validates every required physical record before installing
        the immutable layout in ``_csa_active_layouts``. Repeating that full
        validation for every chunk on every lookup makes long-prefix lookup
        proportional to ``chunks * streamed_objects``. Active layouts are
        invalidated when their owning cache key or object pool is removed, so a
        compatible published layout is the constant-time READY signal here.
        """
        layout = self._active_csa_layout(key)
        return bool(layout is not None and self._csa_layout_matches_runtime(layout))

    def _publish_csa_layout(
        self,
        key: CacheEngineKey,
        build: _CSAStreamingLayoutBuild,
    ) -> bool:
        """Validate a generation and atomically make it active for lookup."""
        layout = _CSAStreamingLayout(
            generation=build.generation,
            layout_version=_DSV4_CSA_STREAMING_LAYOUT_VERSION,
            covered_tokens=build.covered_tokens,
            required_entries=build.required_entries,
            objects=tuple(
                build.objects[entry_key] for entry_key in sorted(build.objects)
            ),
        )
        if not self._csa_layout_matches_runtime(layout):
            logger.warning(
                "CSA layout publication failed: key=%s generation=%s "
                "reason=runtime_mismatch covered_tokens=%d required=%d",
                key.to_string(),
                layout.generation,
                layout.covered_tokens,
                len(layout.required_entries),
            )
            return False
        if not self._csa_layout_records_readable(layout):
            return False
        with self._csa_layout_lock:
            self._csa_active_layouts[key.to_string()] = layout
            self._csa_pending_layouts.pop(key.to_string(), None)
        if build.terminal:
            logger.info(
                "CSA terminal layout published: key=%s generation=%s "
                "covered_tokens=%d objects=%d admission_total_ms=%.3f",
                key.to_string(),
                layout.generation,
                layout.covered_tokens,
                len(layout.objects),
                (time.perf_counter() - build.admission_started_at) * 1000.0,
            )
        else:
            logger.debug(
                "CSA prefix layout published: key=%s generation=%s "
                "covered_tokens=%d objects=%d",
                key.to_string(),
                layout.generation,
                layout.covered_tokens,
                len(layout.objects),
            )
        return True

    def _complete_csa_staged_wave(
        self,
        batch: _CSAStagedWriteBatch,
        error: Optional[BaseException],
    ) -> None:
        """Finalize one staged raw wave and publish only after all succeed."""
        with self.kv_object_store_lock:
            if error is not None:
                batch.failed = True
                logger.error(
                    "CSA staged raw write failed; generation remains invisible: %s",
                    error,
                )
            batch.remaining_waves -= 1
            if batch.remaining_waves > 0:
                return
            if not batch.failed and self.kv_object_metadata_store is not None:
                self.kv_object_metadata_store.extend(batch.records)

        with self._csa_layout_lock:
            for key in batch.keys:
                key_string = key.to_string()
                remaining = self._csa_staged_write_counts.get(key_string, 0) - 1
                if remaining > 0:
                    self._csa_staged_write_counts[key_string] = remaining
                    continue
                self._csa_staged_write_counts.pop(key_string, None)
                if batch.failed:
                    continue
                build = self._csa_pending_layouts.get(key_string)
                if (
                    build is not None
                    and (self._csa_streaming_compact_role(), 0) in build.objects
                ):
                    self._publish_csa_layout(key, build)

    def get_kv_object_records(
        self,
        keys: Sequence[CacheEngineKey],
        *,
        layer_ids: Optional[Sequence[int]] = None,
        roles: Optional[Sequence[str]] = None,
    ) -> list[Optional[KVObjectRecord]]:
        """Return READY KV object records for cache keys in request order.

        Args:
            keys: Cache keys to look up.
            layer_ids: Optional per-key transformer layer ids.  Omit to use
                the legacy chunk-level object id.
            roles: Optional per-key object roles, such as ``csa_attention_kv``
                or ``hca_attention_kv``.  Omit to use ``full``.

        Returns:
            One metadata record per key, with ``None`` for misses.
        """
        if not self.kv_object_store_enabled or self.kv_object_metadata_store is None:
            return [None] * len(keys)
        if layer_ids is not None and len(layer_ids) != len(keys):
            raise ValueError("layer_ids and keys must have the same length")
        if roles is not None and len(roles) != len(keys):
            raise ValueError("roles and keys must have the same length")
        object_ids: list[Optional[KVObjectId]] = []
        logical_lengths: list[Optional[int]] = []
        streaming_roles = self._streaming_logical_roles()
        for index, key in enumerate(keys):
            layer_id = layer_ids[index] if layer_ids is not None else 0
            role = roles[index] if roles is not None else "full"
            if self._csa_streaming_layout_requested() and role in streaming_roles:
                entry = self._resolve_streaming_object(key, role, int(layer_id))
                object_ids.append(entry.object_id if entry is not None else None)
                logical_lengths.append(entry.length if entry is not None else None)
            else:
                object_ids.append(
                    self._key_to_object_id(
                        key,
                        layer_id=int(layer_id),
                        role=role,
                    )
                )
                logical_lengths.append(None)
        resolved = [object_id for object_id in object_ids if object_id is not None]
        records = self.kv_object_metadata_store.get_many(resolved, ready_only=True)
        result: list[Optional[KVObjectRecord]] = []
        record_index = 0
        for object_id, logical_length in zip(
            object_ids,
            logical_lengths,
            strict=True,
        ):
            if object_id is None:
                result.append(None)
                continue
            record = records[record_index]
            record_index += 1
            if (
                record is not None
                and logical_length is not None
                and logical_length > record.length
            ):
                result.append(None)
                continue
            if (
                record is not None
                and logical_length is not None
                and 0 < logical_length < record.length
            ):
                record = _kv_object_prefix_view(record, logical_length)
                if record is None:
                    result.append(None)
                    continue
            result.append(record)
        return result

    def get_kv_object_payload_lengths(
        self,
        keys: Sequence[CacheEngineKey],
        *,
        layer_ids: Optional[Sequence[int]] = None,
        roles: Optional[Sequence[str]] = None,
    ) -> list[Optional[int]]:
        """Return logical READY payload lengths for object-store entries.

        This differs from :meth:`get_kv_object_records` because a published
        streaming-main entry may intentionally contain zero bytes. Such an
        entry is a metadata-only hit and is returned as ``0``; ``None`` means
        the entry is missing or unreadable.

        Args:
            keys: Cache keys to look up.
            layer_ids: Optional per-key transformer layer ids.
            roles: Optional per-key logical object roles.

        Returns:
            One positive, zero, or ``None`` payload length per key.
        """
        if not self.kv_object_store_enabled or self.kv_object_metadata_store is None:
            return [None] * len(keys)
        if layer_ids is not None and len(layer_ids) != len(keys):
            raise ValueError("layer_ids and keys must have the same length")
        if roles is not None and len(roles) != len(keys):
            raise ValueError("roles and keys must have the same length")

        lengths: list[Optional[int]] = []
        streaming_roles = self._streaming_logical_roles()
        for index, key in enumerate(keys):
            layer_id = layer_ids[index] if layer_ids is not None else 0
            role = roles[index] if roles is not None else "full"
            if self._csa_streaming_layout_requested() and role in streaming_roles:
                layout = self._active_csa_layout(key)
                if layout is None or not self._csa_layout_matches_runtime(layout):
                    lengths.append(None)
                    continue
                entry = layout.find_object(role, int(layer_id))
                lengths.append(entry.length if entry is not None else None)
                continue

            record = self.kv_object_metadata_store.get(
                self._key_to_object_id(
                    key,
                    layer_id=int(layer_id),
                    role=role,
                )
            )
            lengths.append(
                record.length
                if record is not None and self._ready_tutti_raw_record(record)
                else None
            )
        return lengths

    def get_hca_layer_major_records(
        self,
        key: CacheEngineKey,
        layer_ids: Sequence[int],
    ) -> list[Optional[KVObjectRecord]]:
        """Return layer-major HCA records for one complete cached prefix.

        Args:
            key: Content-addressed key of the last chunk in the prefix.
            layer_ids: KV-object layer identifiers in request order.

        Returns:
            One READY record per layer id, with ``None`` for misses.
        """
        return self.get_kv_object_records(
            [key] * len(layer_ids),
            layer_ids=layer_ids,
            roles=[_DSV4_HCA_LAYER_MAJOR_ROLE] * len(layer_ids),
        )

    def get_csa_layer_major_records(
        self,
        key: CacheEngineKey,
        layer_ids: Sequence[int],
    ) -> list[Optional[KVObjectRecord]]:
        """Return layer-major CSA records for one complete cached prefix.

        Args:
            key: Content-addressed key of the last chunk in the prefix.
            layer_ids: KV-object layer identifiers in request order.

        Returns:
            One READY record per layer id, with ``None`` for misses.
        """
        return self.get_kv_object_records(
            [key] * len(layer_ids),
            layer_ids=layer_ids,
            roles=[_DSV4_CSA_LAYER_MAJOR_ROLE] * len(layer_ids),
        )

    def get_csa_layer_major_records_for_keys(
        self,
        keys: Sequence[CacheEngineKey],
        layer_id: int,
    ) -> list[Optional[KVObjectRecord]]:
        """Return CSA layer-major records for candidate segment-end keys.

        Args:
            keys: Ordered content-addressed keys to probe.
            layer_id: KV-object layer identifier shared by all probes.

        Returns:
            One READY record per key, with ``None`` where that key does not
            terminate a stored layer-major segment.
        """
        return self.get_kv_object_records(
            list(keys),
            layer_ids=[int(layer_id)] * len(keys),
            roles=[_DSV4_CSA_LAYER_MAJOR_ROLE] * len(keys),
        )

    def get_indexer_layer_major_records(
        self,
        key: CacheEngineKey,
        layer_ids: Sequence[int],
    ) -> list[Optional[KVObjectRecord]]:
        """Return compact layer-major CSA indexer records for one prefix.

        Args:
            key: Content-addressed key terminating the stored segment.
            layer_ids: Indexer object-layer identifiers in request order.

        Returns:
            One READY record per layer id, with ``None`` for misses.
        """
        return self.get_kv_object_records(
            [key] * len(layer_ids),
            layer_ids=layer_ids,
            roles=[_DSV4_INDEXER_LAYER_MAJOR_ROLE] * len(layer_ids),
        )

    def get_indexer_layer_major_records_for_keys(
        self,
        keys: Sequence[CacheEngineKey],
        layer_id: int,
    ) -> list[Optional[KVObjectRecord]]:
        """Probe compact CSA indexer segments ending at candidate keys.

        Args:
            keys: Ordered content-addressed segment-end keys.
            layer_id: Object-layer identifier shared by all probes.

        Returns:
            One READY record per key, with ``None`` for misses.
        """
        return self.get_kv_object_records(
            list(keys),
            layer_ids=[int(layer_id)] * len(keys),
            roles=[_DSV4_INDEXER_LAYER_MAJOR_ROLE] * len(keys),
        )

    def get_hca_layer_major_records_for_keys(
        self,
        keys: Sequence[CacheEngineKey],
        layer_id: int,
    ) -> list[Optional[KVObjectRecord]]:
        """Return HCA layer-major records for candidate segment-end keys.

        Args:
            keys: Ordered content-addressed keys to probe.
            layer_id: KV-object layer identifier shared by all probes.

        Returns:
            One READY record per key, with ``None`` where that key does not
            terminate a stored layer-major prefix view.
        """
        return self.get_kv_object_records(
            list(keys),
            layer_ids=[int(layer_id)] * len(keys),
            roles=[_DSV4_HCA_LAYER_MAJOR_ROLE] * len(keys),
        )

    def store_attention_layer_major_snapshot(
        self,
        prefix_key: CacheEngineKey,
        memory_objs: Sequence[MemoryObj],
        prefix_keys: Optional[Sequence[CacheEngineKey]] = None,
        prefix_token_count: Optional[int] = None,
        base_prefix_key: Optional[CacheEngineKey] = None,
        base_prefix_token_count: int = 0,
    ) -> int:
        """Persist compact layer-major CSA/HCA/indexer sidecar objects.

        Source chunks are supplied in token order.  Their HCA layer slices
        are gathered in that same order, changing the cold-store layout from
        ``chunk -> layer`` to ``layer -> all prefix entries``.  Each resulting
        object is allocated by the shared dense object-pool allocator, so it
        cannot overlap full-KV or other sidecar objects.

        Args:
            prefix_key: Content-addressed key of the prefix's last chunk.
            memory_objs: CPU memory objects for every chunk in the prefix,
                ordered by token position.
            prefix_keys: Optional content-addressed key for every source chunk.
                When supplied, metadata-only prefix views are published for
                each key. They all reference the same physical layer-major
                object and let retrieval stop inside a long store batch without
                copying data or reading beyond the requested prefix.
            prefix_token_count: Exact logical-token count represented by all
                supplied chunks. Omit only when every chunk is full-sized.
            base_prefix_key: Terminal key of an already READY prefix that
                immediately precedes ``prefix_keys`` during partial-hit
                admission.
            base_prefix_token_count: Logical-token coverage of
                ``base_prefix_key``. Zero denotes a cold admission.

        Returns:
            Number of layer-major sidecar objects READY after the call.
            Zero means the object store or its Tutti raw writer is unavailable.
        """
        if (
            not memory_objs
            or not self.kv_object_store_enabled
            or self.kv_object_pool_layout is None
            or self.kv_object_metadata_store is None
        ):
            return 0
        raw_writer = (
            self.kv_object_tutti_raw_writer
            if self.kv_object_tutti_raw_cold_store_enabled
            else None
        )
        if self.kv_object_tutti_raw_cold_store_enabled and raw_writer is None:
            try:
                timeout_s = max(
                    0.0,
                    float(
                        os.getenv(
                            "LMCACHE_KV_OBJECT_STORE_RAW_WRITER_READY_TIMEOUT_SEC",
                            "180",
                        )
                    ),
                )
            except ValueError:
                timeout_s = 180.0
            wait_start = time.perf_counter()
            writer_ready = self.wait_for_kv_object_tutti_raw_writer(timeout_s)
            wait_ms = (time.perf_counter() - wait_start) * 1000.0
            raw_writer = self.kv_object_tutti_raw_writer if writer_ready else None
            if raw_writer is None:
                logger.error(
                    "KV_OBJECT_STORE_PROFILE op=write_attention_layer_major "
                    "key=%s status=fail reason=raw_writer_ready_timeout "
                    "timeout_s=%.3f wait_ms=%.3f",
                    prefix_key.to_string(),
                    timeout_s,
                    wait_ms,
                )
                return 0
            logger.info(
                "KV_OBJECT_STORE_PROFILE op=write_attention_layer_major "
                "key=%s status=raw_writer_ready wait_ms=%.3f",
                prefix_key.to_string(),
                wait_ms,
            )

        if prefix_keys is not None and len(prefix_keys) != len(memory_objs):
            raise ValueError("prefix_keys and memory_objs must have the same length")
        original_prefix_keys = list(prefix_keys) if prefix_keys is not None else None
        base_prefix_token_count = int(base_prefix_token_count)
        if (base_prefix_key is None) != (base_prefix_token_count == 0):
            raise ValueError(
                "base_prefix_key and base_prefix_token_count must be supplied together"
            )
        if base_prefix_token_count < 0:
            raise ValueError("base_prefix_token_count must be non-negative")
        if base_prefix_key is not None:
            base_layout = self._active_csa_layout(base_prefix_key)
            if (
                base_layout is None
                or base_layout.covered_tokens != base_prefix_token_count
                or not self._active_csa_layout_ready(base_prefix_key)
            ):
                logger.warning(
                    "KV_OBJECT_STORE_PROFILE op=write_attention_layer_major "
                    "key=%s status=skip reason=base_generation_unavailable "
                    "base_key=%s base_tokens=%d",
                    prefix_key.to_string(),
                    base_prefix_key.to_string(),
                    base_prefix_token_count,
                )
                return 0
        chunk_size = self._chunk_size
        represented_tokens = (
            len(memory_objs) * chunk_size
            if prefix_token_count is None
            else int(prefix_token_count)
        )
        minimum_tokens = (len(memory_objs) - 1) * chunk_size + 1
        maximum_tokens = len(memory_objs) * chunk_size
        if not minimum_tokens <= represented_tokens <= maximum_tokens:
            raise ValueError(
                "prefix_token_count must match the number of supplied chunks"
            )
        chunk_token_counts = [chunk_size] * len(memory_objs)
        chunk_token_counts[-1] = (
            represented_tokens - (len(memory_objs) - 1) * chunk_size
        )
        streaming_layout = self._csa_streaming_layout_requested()
        effective_memory_objs = list(memory_objs)
        effective_prefix_keys = original_prefix_keys
        effective_token_counts = chunk_token_counts
        generation_covered_tokens = base_prefix_token_count + sum(
            effective_token_counts
        )
        if streaming_layout and effective_prefix_keys is not None:
            terminal_layout = self._active_csa_layout(effective_prefix_keys[-1])
            if (
                terminal_layout is not None
                and terminal_layout.covered_tokens == generation_covered_tokens
                and self._active_csa_layout_ready(effective_prefix_keys[-1])
            ):
                return max(0, len(self._required_csa_streaming_entries()) - 1)

            # A layer-major generation is a prefix object, not a set of
            # independently composable chunk object. Cold admission therefore
            # materializes the supplied prefix. Partial-hit admission instead
            # supplies an explicit READY base generation whose immutable byte
            # ranges are composed with the newly written suffix below.
        coverage_by_key: dict[str, int] = {}
        effective_prefix_key_strings = (
            [key.to_string() for key in effective_prefix_keys]
            if effective_prefix_keys is not None
            else None
        )
        if effective_prefix_keys is not None:
            assert effective_prefix_key_strings is not None
            cumulative_tokens = base_prefix_token_count
            for key_string, token_count in zip(
                effective_prefix_key_strings,
                effective_token_counts,
                strict=True,
            ):
                cumulative_tokens += int(token_count)
                coverage_by_key[key_string] = cumulative_tokens
        else:
            coverage_by_key[prefix_key.to_string()] = generation_covered_tokens
        generation_keys = (
            list(effective_prefix_keys)
            if effective_prefix_keys is not None
            else [prefix_key]
        )
        if base_prefix_key is not None:
            generation_keys.insert(0, base_prefix_key)
        generation = (
            self._csa_streaming_generation(
                generation_keys,
                generation_covered_tokens,
            )
            if streaming_layout
            else ""
        )
        required_entries = frozenset(self._required_csa_streaming_entries())
        pending_entries: dict[
            str,
            dict[tuple[str, int], _CSAStreamingObject],
        ] = {}
        gather_started = time.perf_counter()
        layer_segments: dict[tuple[str, int], list[memoryview]] = {}
        layer_segment_sources: dict[
            tuple[str, int],
            list[Optional[tuple[int, int]]],
        ] = {}
        layer_chunk_nbytes: dict[tuple[str, int], list[int]] = {}
        layout_cache_hits = 0
        layout_cache_misses = 0
        for memory_obj in effective_memory_objs:
            buffer = memory_obj.byte_array
            raw_tensor = memory_obj.raw_tensor
            source_base_ptr: Optional[int] = None
            if (
                raw_tensor is not None
                and raw_tensor.device.type == "cpu"
                and raw_tensor.is_pinned()
            ):
                source_base_ptr = int(raw_tensor.data_ptr())
            layout, cache_hit = self._layer_segment_layout(memory_obj)
            layout_cache_hits += int(cache_hit)
            layout_cache_misses += int(not cache_hit)
            for (role, layer_id), byte_ranges in layout:
                segments = layer_segments.setdefault(
                    (role, int(layer_id)),
                    [],
                )
                segment_sources = layer_segment_sources.setdefault(
                    (role, int(layer_id)),
                    [],
                )
                chunk_nbytes = 0
                for start, length in byte_ranges:
                    # ``MemoryObj.metadata.shapes`` already contains the
                    # physical row count (logical tokens / compress_ratio;
                    # see ``LMCacheEngineMetadata.get_shapes``).  Applying
                    # the ratio again truncates CSA to 1/4 and HCA to 1/128,
                    # then aliases those bytes as later prefix blocks.  Copy
                    # the complete per-layer range exactly once.
                    if length <= 0:
                        continue
                    end = start + length
                    segments.append(buffer[start:end])
                    segment_sources.append(
                        (source_base_ptr, start)
                        if source_base_ptr is not None
                        else None
                    )
                    chunk_nbytes += length
                layer_chunk_nbytes.setdefault((role, int(layer_id)), []).append(
                    chunk_nbytes
                )
        if not layer_segments:
            return 0
        gather_time_s = time.perf_counter() - gather_started

        base_records: dict[tuple[str, int], KVObjectRecord] = {}
        if base_prefix_key is not None:
            assert self.kv_object_metadata_store is not None
            for source_role, layer_id in layer_segments:
                object_role = {
                    "csa_attention_kv": _DSV4_CSA_LAYER_MAJOR_ROLE,
                    "hca_attention_kv": _DSV4_HCA_LAYER_MAJOR_ROLE,
                    "csa_indexer_cache": _DSV4_INDEXER_LAYER_MAJOR_ROLE,
                }[source_role]
                object_id = self._resolve_streaming_object_id(
                    base_prefix_key,
                    object_role,
                    int(layer_id),
                )
                record = (
                    self.kv_object_metadata_store.get(object_id)
                    if object_id is not None
                    else None
                )
                if not self._ready_tutti_raw_record(record) or record is None:
                    logger.warning(
                        "KV_OBJECT_STORE_PROFILE op=write_attention_layer_major "
                        "key=%s status=skip reason=base_sidecar_unavailable "
                        "base_key=%s role=%s layer=%d",
                        prefix_key.to_string(),
                        base_prefix_key.to_string(),
                        source_role,
                        layer_id,
                    )
                    return 0
                base_records[(source_role, int(layer_id))] = record

        start_time = time.perf_counter()
        ready_count = 0
        total_bytes = 0
        packed_time_s = 0.0
        pack_wait_time_s = 0.0
        storage_write_time_s = 0.0
        alias_time_s = 0.0
        metadata_record_count = 0
        raw_write_wave_count = 0
        with self.kv_object_store_lock:
            sidecars: list[_LayerMajorSidecar] = []
            for (source_role, layer_id), segments in sorted(layer_segments.items()):
                payload_nbytes = sum(len(segment) for segment in segments)
                if payload_nbytes <= 0:
                    continue
                object_role = {
                    "csa_attention_kv": _DSV4_CSA_LAYER_MAJOR_ROLE,
                    "hca_attention_kv": _DSV4_HCA_LAYER_MAJOR_ROLE,
                    "csa_indexer_cache": _DSV4_INDEXER_LAYER_MAJOR_ROLE,
                }[source_role]
                physical_role = (
                    self._generation_object_role(object_role, generation)
                    if streaming_layout
                    else object_role
                )
                object_id = self._key_to_object_id(
                    prefix_key,
                    layer_id=layer_id,
                    role=physical_role,
                )
                existing = self.kv_object_metadata_store.get(object_id)
                ready_record = None
                record = existing
                if (
                    existing is not None
                    and existing.state == KVObjectState.READY
                    and existing.length == payload_nbytes
                ):
                    ready_record = existing
                else:
                    if (
                        raw_writer is not None
                        and self._cpu_raw_stage_queue is None
                        and not self.kv_object_payload_fits_tutti_staging(
                            payload_nbytes
                        )
                    ):
                        logger.warning(
                            "KV_OBJECT_STORE_PROFILE "
                            "op=write_attention_layer_major "
                            "key=%s role=%s layer=%d status=skip "
                            "reason=staging_capacity "
                            "bytes=%d staging_bytes=%s",
                            prefix_key.to_string(),
                            source_role,
                            layer_id,
                            payload_nbytes,
                            self.kv_object_tutti_raw_staging_bytes,
                        )
                        continue
                    if existing is None or existing.length != payload_nbytes:
                        if raw_writer is not None:
                            offset, _end_offset, aligned_length = (
                                self.kv_object_pool_layout.next_allocation_bounds(
                                    payload_nbytes
                                )
                            )
                            if not self.kv_object_raw_region_covers(
                                offset,
                                aligned_length,
                            ):
                                self.mark_kv_object_raw_region_full()
                                logger.warning(
                                    "KV_OBJECT_STORE_PROFILE "
                                    "op=write_attention_layer_major key=%s "
                                    "role=%s layer=%d "
                                    "status=skip reason=raw_region_full bytes=%d",
                                    prefix_key.to_string(),
                                    source_role,
                                    layer_id,
                                    payload_nbytes,
                                )
                                continue
                        record = self.kv_object_pool_layout.allocate(
                            object_id,
                            length=payload_nbytes,
                            shape=(payload_nbytes,),
                            dtype="torch.uint8",
                        )
                    else:
                        record = existing
                assert record is not None
                sidecars.append(
                    _LayerMajorSidecar(
                        source_role=source_role,
                        layer_id=int(layer_id),
                        object_role=object_role,
                        physical_role=physical_role,
                        payload_nbytes=payload_nbytes,
                        segments=segments,
                        segment_sources=layer_segment_sources[
                            (source_role, int(layer_id))
                        ],
                        record=record,
                        ready_record=ready_record,
                    )
                )

            pending_writes = [
                sidecar for sidecar in sidecars if sidecar.ready_record is None
            ]
            physical_records: list[KVObjectRecord] = []
            staged_writes: list[tuple[KVObjectRecord, bytearray]] = []
            staged_write_futures: list[
                Future[tuple[Sequence[tuple[int, int, int]], float]]
            ] = []
            staged_write_errors: list[BaseException] = []
            stage_queue = self._cpu_raw_stage_queue if raw_writer is not None else None
            if raw_writer is not None:
                staging_bytes = (
                    stage_queue.capacity_bytes
                    if stage_queue is not None
                    else self.kv_object_tutti_raw_staging_bytes
                )
                # Stage capacity bounds total queued host memory. Physical
                # pack/write waves retain their independent 64 MiB default so
                # a large stage never creates a multi-GiB Python preparation.
                wave_limit = _write_wave_limit_bytes(staging_bytes)
                raw_waves: list[list[_LayerMajorSidecar]] = []
                current_wave: list[_LayerMajorSidecar] = []
                for sidecar in pending_writes:
                    if not current_wave:
                        current_wave = [sidecar]
                        continue
                    first_record = current_wave[0].record
                    previous_record = current_wave[-1].record
                    wave_end = sidecar.record.offset + sidecar.record.aligned_length
                    contiguous = (
                        sidecar.record.pool_id == first_record.pool_id
                        and sidecar.record.offset
                        == previous_record.offset + previous_record.aligned_length
                    )
                    fits_wave = (
                        wave_limit > 0 and wave_end - first_record.offset <= wave_limit
                    )
                    if contiguous and fits_wave:
                        current_wave.append(sidecar)
                    else:
                        raw_waves.append(current_wave)
                        current_wave = [sidecar]
                if current_wave:
                    raw_waves.append(current_wave)

                wave_buffer_nbytes = [
                    (
                        wave[0].payload_nbytes
                        if len(wave) == 1
                        else wave[-1].record.offset
                        + wave[-1].record.aligned_length
                        - wave[0].record.offset
                    )
                    for wave in raw_waves
                ]
                if (
                    stage_queue is not None
                    and sum(wave_buffer_nbytes) > stage_queue.capacity_bytes
                ):
                    logger.warning(
                        "CPU raw-stage generation exceeds configured capacity; "
                        "falling back to synchronous persistence key=%s bytes=%d "
                        "capacity_bytes=%d",
                        prefix_key.to_string(),
                        sum(wave_buffer_nbytes),
                        stage_queue.capacity_bytes,
                    )
                    stage_queue = None
                if stage_queue is not None:
                    for wave, wave_nbytes in zip(
                        raw_waves,
                        wave_buffer_nbytes,
                        strict=True,
                    ):
                        pack_waiter = self.kv_object_layer_major_pack_waiter
                        if pack_waiter is not None:
                            pack_waiter()
                        # This is the generation-level host staging copy.  Do
                        # not pack into a reusable 64 MiB/O_DIRECT bounce
                        # buffer and do not clone it afterwards: the queued
                        # bytearray itself owns the full wave until SSD
                        # persistence completes.
                        stage_queue.reserve(wave_nbytes)
                        reservation_owned = True
                        try:
                            pack_started = time.perf_counter()
                            staged_payload = bytearray(wave_nbytes)
                            writer_record, payload, pack_cpu_time_s = (
                                _pack_layer_major_wave(wave, staged_payload)
                            )
                            packed_time_s += pack_cpu_time_s
                            pack_wait_time_s += time.perf_counter() - pack_started
                            staged_writes.append((writer_record, staged_payload))
                            raw_extents = self.map_kv_object_to_raw_region(
                                writer_record
                            )
                            raw_write_wave_count += 1
                            for sidecar in wave:
                                record_extents = (
                                    tuple(raw_extents)
                                    if len(wave) == 1
                                    else _clip_raw_extents(
                                        raw_extents,
                                        offset=sidecar.record.offset,
                                        aligned_length=sidecar.record.aligned_length,
                                    )
                                )
                                sidecar.ready_record = sidecar.record.with_raw_extents(
                                    record_extents
                                ).mark_ready()
                                physical_records.append(sidecar.ready_record)
                            payload.release()
                            try:
                                future = stage_queue.submit_reserved(
                                    staged_payload,
                                    lambda payload, record=writer_record: raw_writer(
                                        record,
                                        payload,
                                    ),
                                )
                                reservation_owned = False
                                staged_write_futures.append(future)
                            except Exception as error:
                                stage_queue.release_reservation(wave_nbytes)
                                reservation_owned = False
                                staged_write_errors.append(error)
                        except Exception:
                            if reservation_owned:
                                stage_queue.release_reservation(wave_nbytes)
                            raise
                gpu_pack_enabled = (
                    _env_flag("LMCACHE_DSV4_GPU_LAYER_MAJOR_PACK")
                    and stage_queue is None
                    and not self.kv_object_cpu_raw_write_enabled
                    and self.kv_object_gpu_layer_major_packer is not None
                )
                pack_buffers: list[Optional[bytearray]] = (
                    [None, None]
                    if gpu_pack_enabled
                    else [
                        bytearray(
                            max(
                                wave_buffer_nbytes[index::2],
                                default=0,
                            )
                        )
                        for index in range(2)
                    ]
                )
                gpu_pack_buffers: list[Optional[torch.Tensor]] = [None, None]
                if gpu_pack_enabled:
                    try:
                        gpu_pack_buffers = [
                            torch.empty(
                                max(
                                    wave_buffer_nbytes[index::2],
                                    default=0,
                                ),
                                dtype=torch.uint8,
                                device="cpu",
                                pin_memory=True,
                            )
                            for index in range(2)
                        ]
                    except Exception:
                        logger.exception(
                            "Unable to allocate pinned GPU-pack buffers; "
                            "using CPU layer-major pack"
                        )
                        gpu_pack_buffers = [None, None]
                pending_pack: Optional[
                    Future[tuple[KVObjectRecord, memoryview, float]]
                ] = None
                if raw_waves and stage_queue is None:
                    pending_pack = self._layer_major_pack_executor.submit(
                        _pack_layer_major_wave_with_fallback,
                        raw_waves[0],
                        pack_buffers[0],
                        gpu_pack_buffers[0],
                        self.kv_object_gpu_layer_major_packer,
                    )
                for wave_index, wave in enumerate(raw_waves):
                    if stage_queue is not None:
                        continue
                    assert pending_pack is not None
                    pack_wait_started = time.perf_counter()
                    writer_record, payload, pack_cpu_time_s = pending_pack.result()
                    pack_wait_time_s += time.perf_counter() - pack_wait_started
                    packed_time_s += pack_cpu_time_s
                    next_wave_index = wave_index + 1
                    if next_wave_index < len(raw_waves):
                        next_slot = next_wave_index % 2
                        pending_pack = self._layer_major_pack_executor.submit(
                            _pack_layer_major_wave_with_fallback,
                            raw_waves[next_wave_index],
                            pack_buffers[next_slot],
                            gpu_pack_buffers[next_slot],
                            self.kv_object_gpu_layer_major_packer,
                        )
                    write_started = time.perf_counter()
                    raw_extents, _write_ms = raw_writer(
                        writer_record,
                        payload,
                    )
                    storage_write_time_s += time.perf_counter() - write_started
                    raw_write_wave_count += 1
                    for sidecar in wave:
                        record_extents = (
                            tuple(raw_extents)
                            if len(wave) == 1
                            else _clip_raw_extents(
                                raw_extents,
                                offset=sidecar.record.offset,
                                aligned_length=sidecar.record.aligned_length,
                            )
                        )
                        sidecar.ready_record = sidecar.record.with_raw_extents(
                            record_extents
                        ).mark_ready()
                        physical_records.append(sidecar.ready_record)
                    payload.release()
            else:
                if self.kv_object_pool_io is not None:
                    for sidecar in pending_writes:
                        pack_started = time.perf_counter()
                        payload = bytearray().join(sidecar.segments)
                        packed_time_s += time.perf_counter() - pack_started
                        write_started = time.perf_counter()
                        self.kv_object_pool_io.write_object(
                            sidecar.record,
                            memoryview(payload),
                        )
                        storage_write_time_s += time.perf_counter() - write_started
                        sidecar.ready_record = sidecar.record.mark_ready()
                        physical_records.append(sidecar.ready_record)

            if stage_queue is None or not staged_writes:
                self.kv_object_metadata_store.extend(physical_records)
                metadata_record_count += len(physical_records)

            for sidecar in sidecars:
                source_role = sidecar.source_role
                layer_id = sidecar.layer_id
                object_role = sidecar.object_role
                physical_role = sidecar.physical_role
                payload_nbytes = sidecar.payload_nbytes
                ready_record = sidecar.ready_record
                if ready_record is None:
                    continue
                if effective_prefix_keys is not None:
                    assert effective_prefix_key_strings is not None
                    alias_started = time.perf_counter()
                    alias_records: list[KVObjectRecord] = []
                    prefix_nbytes = 0
                    base_record = base_records.get((source_role, int(layer_id)))
                    terminal_alias_object_id = (
                        self._key_to_object_id(
                            effective_prefix_keys[-1],
                            layer_id=layer_id,
                            role=physical_role,
                        )
                        if (
                            streaming_layout
                            and base_record is not None
                            and effective_prefix_keys
                        )
                        else None
                    )
                    alias_inputs = zip(
                        effective_prefix_keys,
                        effective_prefix_key_strings,
                        layer_chunk_nbytes[(source_role, layer_id)],
                        strict=True,
                    )
                    for alias_index, (
                        alias_key,
                        alias_key_string,
                        chunk_nbytes,
                    ) in enumerate(alias_inputs):
                        prefix_nbytes += int(chunk_nbytes)
                        if streaming_layout and base_record is None:
                            # Cold prefix views differ only in logical length.
                            # Point every manifest entry at the immutable full
                            # physical sidecar and synthesize its prefix record
                            # on lookup instead of materializing
                            # O(chunks * layers) metadata aliases here.
                            pending_entries.setdefault(
                                alias_key_string,
                                {},
                            )[(object_role, int(layer_id))] = _CSAStreamingObject(
                                logical_role=object_role,
                                layer_id=int(layer_id),
                                object_id=ready_record.object_id,
                                length=prefix_nbytes,
                            )
                            continue
                        alias_object_id = (
                            terminal_alias_object_id
                            or self._key_to_object_id(
                                alias_key,
                                layer_id=layer_id,
                                role=physical_role,
                            )
                        )
                        if base_record is None:
                            alias_record = KVObjectRecord(
                                object_id=alias_object_id,
                                pool_id=ready_record.pool_id,
                                offset=ready_record.offset,
                                length=prefix_nbytes,
                                aligned_length=ready_record.aligned_length,
                                shape=(prefix_nbytes,),
                                dtype=ready_record.dtype,
                                state=KVObjectState.READY,
                                raw_extents=ready_record.raw_extents,
                            )
                        else:
                            logical_length = base_record.length + prefix_nbytes
                            if streaming_layout and alias_index < len(
                                effective_prefix_keys
                            ) - 1:
                                assert terminal_alias_object_id is not None
                                pending_entries.setdefault(
                                    alias_key_string,
                                    {},
                                )[(object_role, int(layer_id))] = (
                                    _CSAStreamingObject(
                                        logical_role=object_role,
                                        layer_id=int(layer_id),
                                        object_id=terminal_alias_object_id,
                                        length=logical_length,
                                    )
                                )
                                continue
                            combined_ranges = [
                                KVObjectByteRange(
                                    offset=byte_range.offset,
                                    length=byte_range.length,
                                    target_offset=byte_range.target_offset,
                                )
                                for byte_range in base_record.read_ranges
                            ]
                            combined_ranges.append(
                                KVObjectByteRange(
                                    offset=ready_record.offset,
                                    length=prefix_nbytes,
                                    target_offset=base_record.length,
                                )
                            )
                            combined_extents = tuple(
                                dict.fromkeys(
                                    (
                                        *base_record.raw_extents,
                                        *ready_record.raw_extents,
                                    )
                                )
                            )
                            alias_record = KVObjectRecord(
                                object_id=alias_object_id,
                                pool_id=ready_record.pool_id,
                                offset=combined_ranges[0].offset,
                                length=logical_length,
                                # A composed record has no single physical
                                # allocation, but consumers still use this as
                                # its sector-sized logical DMA envelope.
                                aligned_length=((logical_length + 511) // 512) * 512,
                                shape=(logical_length,),
                                dtype=ready_record.dtype,
                                state=KVObjectState.READY,
                                raw_extents=combined_extents,
                                byte_ranges=tuple(combined_ranges),
                            )
                        alias_records.append(alias_record)
                        if streaming_layout:
                            pending_entries.setdefault(
                                alias_key_string,
                                {},
                            )[(object_role, int(layer_id))] = _CSAStreamingObject(
                                logical_role=object_role,
                                layer_id=int(layer_id),
                                object_id=alias_object_id,
                                length=alias_record.length,
                            )
                    if stage_queue is None or not staged_writes:
                        self.kv_object_metadata_store.extend(alias_records)
                    else:
                        physical_records.extend(alias_records)
                    metadata_record_count += len(alias_records)
                    alias_time_s += time.perf_counter() - alias_started
                elif streaming_layout:
                    pending_entries.setdefault(
                        prefix_key.to_string(),
                        {},
                    )[(object_role, int(layer_id))] = _CSAStreamingObject(
                        logical_role=object_role,
                        layer_id=int(layer_id),
                        object_id=ready_record.object_id,
                        length=ready_record.length,
                    )
                ready_count += 1
                total_bytes += payload_nbytes

        if streaming_layout:
            terminal_key_string = prefix_key.to_string()
            with self._csa_layout_lock:
                for key_string, objects in pending_entries.items():
                    self._csa_pending_layouts[key_string] = _CSAStreamingLayoutBuild(
                        generation=generation,
                        covered_tokens=coverage_by_key[key_string],
                        required_entries=required_entries,
                        objects=objects,
                        terminal=key_string == terminal_key_string,
                        admission_started_at=gather_started,
                    )
                    self._csa_pending_layout_keys[key_string] = next(
                        key
                        for key in generation_keys
                        if key.to_string() == key_string
                    )
                    if staged_writes:
                        self._csa_staged_write_counts[key_string] = (
                            self._csa_staged_write_counts.get(key_string, 0) + 1
                        )

        if stage_queue is not None and staged_writes:
            staged_keys = tuple(
                self._csa_pending_layout_keys[key_string]
                for key_string in pending_entries
            )
            batch = _CSAStagedWriteBatch(
                records=tuple(physical_records),
                keys=staged_keys,
                remaining_waves=len(staged_writes),
            )
            for future in staged_write_futures:
                def _finish_staged_write(
                    completed: Future[tuple[Sequence[tuple[int, int, int]], float]],
                    staged_batch: _CSAStagedWriteBatch = batch,
                ) -> None:
                    self._complete_csa_staged_wave(staged_batch, completed.exception())

                future.add_done_callback(_finish_staged_write)
            for error in staged_write_errors:
                self._complete_csa_staged_wave(batch, error)

        logger.info(
            "KV_OBJECT_STORE_PROFILE op=write_attention_layer_major key=%s "
            "layers=%d bytes=%d metadata_records=%d gather_ms=%.3f "
            "layout_cache_hits=%d layout_cache_misses=%d raw_write_waves=%d "
            "pack_cpu_ms=%.3f pack_wait_ms=%.3f storage_write_ms=%.3f "
            "alias_ms=%.3f total_ms=%.3f",
            prefix_key.to_string(),
            ready_count,
            total_bytes,
            metadata_record_count,
            gather_time_s * 1000.0,
            layout_cache_hits,
            layout_cache_misses,
            raw_write_wave_count,
            packed_time_s * 1000.0,
            pack_wait_time_s * 1000.0,
            storage_write_time_s * 1000.0,
            alias_time_s * 1000.0,
            (time.perf_counter() - start_time) * 1000.0,
        )
        return ready_count

    def get_kv_object_pool_paths(self) -> dict[str, Path]:
        """Return object pool paths keyed by pool id."""
        if self.kv_object_pool_layout is None:
            return {}
        return {
            self.kv_object_pool_layout.pool_id: self.kv_object_pool_layout.pool_path
        }

    def set_kv_object_tutti_raw_writer(
        self,
        writer: Optional[KVObjectRawWriter],
    ) -> None:
        """Install the Tutti raw-object writer used after snvme bind.

        Args:
            writer: Callable that writes one allocated object record to raw
                NVMe LBAs and returns raw extents plus elapsed write time.  Pass
                ``None`` to disable raw writes and use the filesystem path.
        """
        self.kv_object_tutti_raw_writer = writer
        if writer is None:
            self._kv_object_tutti_raw_writer_ready.clear()
        else:
            self._kv_object_tutti_raw_writer_ready.set()

    def set_kv_object_layer_major_pack_waiter(
        self,
        waiter: Optional[KVObjectPackWaiter],
    ) -> None:
        """Install the decode-aware admission gate for CPU sidecar packing.

        Args:
            waiter: Callback invoked before each bounded pack wave. Pass
                ``None`` to disable pack admission gating.

        Notes:
            This gate is intentionally separate from physical-write
            admission. A packed CPU staging buffer may drain to SSD during
            decode, while Python sidecar preparation must yield to decode.
        """
        self.kv_object_layer_major_pack_waiter = waiter

    def set_kv_object_gpu_layer_major_packer(
        self,
        packer: Optional[KVObjectLayerMajorPacker],
    ) -> None:
        """Install the optional CUDA-assisted pinned-host wave packer.

        Args:
            packer: Callable that packs mapped pinned CPU segments into one
                pinned CPU destination. Pass ``None`` to retain the native CPU
                pack path.
        """
        self.kv_object_gpu_layer_major_packer = packer

    def wait_for_kv_object_tutti_raw_writer(self, timeout_s: float) -> bool:
        """Wait until the Tutti raw cold-store writer is installed.

        Args:
            timeout_s: Maximum number of seconds to wait. Must be non-negative.

        Returns:
            ``True`` when raw cold-store is disabled or a writer is installed;
            otherwise ``False`` after the timeout expires.

        Raises:
            ValueError: If ``timeout_s`` is negative.
        """
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        if not self.kv_object_tutti_raw_cold_store_enabled:
            return True
        if self.kv_object_tutti_raw_writer is not None:
            return True
        if not self._kv_object_tutti_raw_writer_ready.wait(timeout_s):
            return False
        return self.kv_object_tutti_raw_writer is not None

    def has_kv_object_tutti_raw_writer(self) -> bool:
        """Return whether Tutti raw-object writes are currently available."""
        return self.kv_object_tutti_raw_writer is not None

    def reset_kv_object_pool_allocation(self) -> int:
        """Start a new raw-object generation at offset zero.

        Raw-object manifests are process-local. Recovered legacy cache entries
        cannot remain lookup-visible when the allocator is rewound because new
        writes would reuse their LBAs while stale keys still referenced the old
        contents. Atomically invalidate those lookup and object records before
        resetting allocation. Existing filesystem files are left untouched;
        they are intentionally unreachable for this raw-mode process.
        Returns:
            Number of lookup-visible keys and object metadata records
            invalidated by the generation reset.
        """
        with self._csa_layout_lock:
            self._csa_active_layouts.clear()
            self._csa_pending_layouts.clear()
            self._csa_pending_layout_keys.clear()
            self._csa_staged_write_counts.clear()
        with self.disk_lock:
            recovered_keys = len(self.dict)
            self.dict.clear()
            self.current_cache_size = 0.0
            self.usage = 0
            self.stats_monitor.update_local_storage_usage(0)
        metadata_records = 0
        if self.kv_object_metadata_store is not None:
            metadata_records = self.kv_object_metadata_store.clear()
        if self.kv_object_pool_layout is not None:
            self.kv_object_pool_layout.reset_allocation()
        self._kv_object_raw_region_full_at_s = 0.0
        if recovered_keys or metadata_records:
            logger.info(
                "KV object raw-generation reset invalidated recovered_keys=%d "
                "metadata_records=%d",
                recovered_keys,
                metadata_records,
            )
        return recovered_keys + metadata_records

    def clear(self) -> int:
        """Clear disk lookup state and reclaim the Tutti raw object region.

        Raw-object allocation is append-only within one generation. A normal
        per-arm cache clear must therefore reset both lookup metadata and the
        allocation cursor; merely dropping ordinary cache keys eventually
        exhausts the reserved raw region across benchmark arms. Callers must
        drain ordinary disk puts before clearing. The two private sidecar
        executors are fenced here so no earlier task can publish stale metadata
        after the new generation starts.

        Returns:
            Number of lookup-visible keys and object records invalidated.

        Raises:
            RuntimeError: If ordinary disk puts are still active.
        """
        with self.disk_worker.put_lock:
            pending_puts = len(self.disk_worker.put_tasks)
        if pending_puts:
            raise RuntimeError(
                f"cannot clear LocalDiskBackend with {pending_puts} active put tasks"
            )

        # Layer-pack tasks may enqueue HCA writes, so fence them first.
        self._layer_major_pack_executor.submit(lambda: None).result()
        self._hca_write_executor.submit(lambda: None).result()
        if self._cpu_raw_stage_queue is not None:
            self._cpu_raw_stage_queue.drain()

        paths: set[str] = set()
        with self.disk_lock:
            for metadata in self.dict.values():
                path = str(metadata.path)
                if os.path.isfile(path):
                    paths.add(path)

        removed = self.reset_kv_object_pool_allocation()
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        return removed

    def mark_kv_object_raw_region_full(self) -> None:
        """Record the first failed raw-region allocation in this generation."""
        if self._kv_object_raw_region_full_at_s <= 0.0:
            self._kv_object_raw_region_full_at_s = time.perf_counter()

    def kv_object_raw_region_full_since(self, since_s: float) -> bool:
        """Return whether raw allocation failed after ``since_s``.

        Args:
            since_s: Monotonic timestamp at which the publication ticket was
                queued.

        Returns:
            ``True`` when this generation observed a raw-region allocation
            failure no earlier than the supplied timestamp.
        """
        full_at_s = self._kv_object_raw_region_full_at_s
        return full_at_s > 0.0 and full_at_s >= since_s

    def set_kv_object_tutti_raw_region_extents(
        self,
        raw_extents: Sequence[tuple[int, int, int]],
    ) -> None:
        """Install FIEMAP extents for the rank-local raw region file.

        Args:
            raw_extents: Sequence of ``(file_offset, slba, n_sectors)`` tuples
                covering the reserved raw-region file.
        """
        self.kv_object_tutti_raw_region_extents = [
            (int(file_offset), int(slba), int(n_sectors))
            for file_offset, slba, n_sectors in raw_extents
        ]

    def map_kv_object_to_raw_region(
        self,
        record: KVObjectRecord,
    ) -> list[tuple[int, int, int]]:
        """Map an object record's logical pool range to raw-region extents.

        Args:
            record: Allocated object-store record.

        Returns:
            Raw extents covering ``record.offset`` through
            ``record.offset + record.aligned_length``.

        Raises:
            RuntimeError: If the configured raw region does not cover the
                record's logical byte range.
        """
        object_start = record.offset
        object_end = record.offset + record.aligned_length
        result: list[tuple[int, int, int]] = []
        covered = 0
        for file_offset, slba, n_sectors in self.kv_object_tutti_raw_region_extents:
            extent_start = file_offset
            extent_end = file_offset + n_sectors * 512
            write_start = max(object_start, extent_start)
            write_end = min(object_end, extent_end)
            if write_start >= write_end:
                continue
            extent_skip = write_start - extent_start
            write_nbytes = write_end - write_start
            result.append(
                (
                    write_start,
                    slba + extent_skip // 512,
                    write_nbytes // 512,
                )
            )
            covered += write_nbytes
        if covered != record.aligned_length:
            raise RuntimeError(
                "KV object raw region does not cover object range "
                f"{object_start}:{object_end}; covered={covered}"
            )
        return result

    def kv_object_raw_region_covers(self, offset: int, length: int) -> bool:
        """Return whether the rank-local raw region covers a byte range."""
        if length <= 0:
            return False
        object_start = offset
        object_end = offset + length
        covered = 0
        for file_offset, _slba, n_sectors in self.kv_object_tutti_raw_region_extents:
            extent_start = file_offset
            extent_end = file_offset + n_sectors * 512
            write_start = max(object_start, extent_start)
            write_end = min(object_end, extent_end)
            if write_start >= write_end:
                continue
            covered += write_end - write_start
        return covered == length

    def kv_object_payload_fits_tutti_staging(self, length: int) -> bool:
        """Return whether one raw object write fits the Tutti staging buffer."""
        if length <= 0:
            return False
        if self.kv_object_tutti_raw_staging_bytes is None:
            return True
        if self.kv_object_pool_layout is None:
            return length <= self.kv_object_tutti_raw_staging_bytes
        return (
            self.kv_object_pool_layout.align_length(length)
            <= self.kv_object_tutti_raw_staging_bytes
        )

    def kv_object_tutti_path(self, pool_id: str) -> str:
        """Return the synthetic path used for raw Tutti object extents."""
        return f"tutti://{pool_id}"

    def kv_object_data_path(self, record: KVObjectRecord) -> Optional[str]:
        """Return the path used to read one object record.

        Raw Tutti records return a synthetic ``tutti://`` path.  File-backed
        records return the dense pool file path.
        """
        if record.raw_extents:
            return self.kv_object_tutti_path(record.pool_id)
        pool_path = self.get_kv_object_pool_paths().get(record.pool_id)
        return str(pool_path) if pool_path is not None else None

    def get_kv_object_raw_lba_cache(
        self,
        records: Sequence[Optional[KVObjectRecord]],
    ) -> dict[str, list[tuple[int, int, int]]]:
        """Return raw Tutti extents grouped by synthetic object path.

        Args:
            records: Object records to expose to a Tutti loader.

        Returns:
            Mapping from synthetic path to ``(file_offset, slba, n_sectors)``
            tuples.
        """
        result: dict[str, list[tuple[int, int, int]]] = {}
        for record in records:
            if record is None or not record.raw_extents:
                continue
            path = self.kv_object_tutti_path(record.pool_id)
            result.setdefault(path, []).extend(self._raw_extents_for_record(record))
        return result

    def kv_object_record_raw_readable(self, record: KVObjectRecord) -> bool:
        """Return whether a raw Tutti object record can satisfy its read ranges.

        Args:
            record: KV object metadata record to validate.

        Returns:
            ``True`` when the record is file-backed, or when its raw extents cover
            every byte that the direct loader will request.  ``False`` means
            lookup must not advertise this object as a Tutti raw hit.
        """
        if not record.raw_extents:
            return self.kv_object_data_path(record) is not None

        expected_bytes = self.kv_object_record_raw_read_bytes(record)
        if expected_bytes <= 0:
            return False

        covered_bytes = sum(
            n_sectors * 512
            for _file_offset, _slba, n_sectors in self._raw_extents_for_record(record)
        )
        return covered_bytes == expected_bytes

    def count_tutti_raw_readable_prefix(
        self,
        keys: Sequence[CacheEngineKey],
    ) -> int:
        """Count the contiguous prefix consumable by the Tutti raw loader.

        An ordinary chunk can become visible in the disk dictionary before
        its raw-object metadata and extents are ready. Reuse barriers must use
        the same record-readability condition as the direct loader rather than
        treating the ordinary chunk lookup as sufficient.

        Args:
            keys: Request-ordered cache keys to validate.

        Returns:
            Number of leading keys with READY raw-object records whose extents
            cover every byte required by the direct loader.
        """
        readable = 0
        for record in self.get_kv_object_records(keys):
            if record is None or not self.kv_object_record_raw_readable(record):
                break
            readable += 1
        return readable

    def _ready_tutti_raw_record(
        self,
        record: Optional[KVObjectRecord],
    ) -> bool:
        """Return whether a READY record can be consumed by Tutti."""
        if record is None or record.state != KVObjectState.READY:
            return False
        return self.kv_object_record_raw_readable(record)

    def kv_object_record_raw_read_bytes(self, record: KVObjectRecord) -> int:
        """Return DMA bytes needed to read a raw Tutti object record.

        Args:
            record: KV object metadata record to inspect.

        Returns:
            The sector-aligned number of bytes Tutti will read, or ``0`` when
            any logical range violates direct-read alignment requirements.
        """
        expected_bytes = 0
        for byte_range in record.read_ranges:
            range_dma_length = ((byte_range.length + 511) // 512) * 512
            if byte_range.offset % 512 != 0:
                return 0
            expected_bytes += range_dma_length
        return expected_bytes

    def _raw_extents_for_record(
        self,
        record: KVObjectRecord,
    ) -> list[tuple[int, int, int]]:
        """Return raw extents covering the record's logical read ranges."""
        extents: list[tuple[int, int, int]] = []
        ordered_extents = sorted(record.raw_extents, key=lambda item: item[0])
        extent_starts = [
            file_offset for file_offset, _slba, _sectors in ordered_extents
        ]
        for byte_range in record.read_ranges:
            range_length = byte_range.length
            range_dma_length = ((range_length + 511) // 512) * 512
            if byte_range.offset % 512 != 0:
                logger.warning(
                    "KV object raw range is not 512-byte aligned: "
                    "object=%s offset=%d length=%d",
                    record.object_id.to_key(),
                    byte_range.offset,
                    byte_range.length,
                )
                continue
            range_start = byte_range.offset
            range_end = range_start + range_dma_length
            extent_index = bisect.bisect_right(extent_starts, range_start) - 1
            if extent_index < 0:
                extent_index = 0
            for file_offset, slba, n_sectors in ordered_extents[extent_index:]:
                extent_start = file_offset
                extent_end = file_offset + n_sectors * 512
                if extent_start >= range_end:
                    break
                read_start = max(range_start, extent_start)
                read_end = min(range_end, extent_end)
                if read_start >= read_end:
                    continue
                extent_skip = read_start - extent_start
                read_nbytes = read_end - read_start
                if extent_skip % 512 != 0 or read_nbytes % 512 != 0:
                    logger.warning(
                        "KV object raw extent overlap is not 512-byte aligned: "
                        "object=%s read_start=%d read_nbytes=%d",
                        record.object_id.to_key(),
                        read_start,
                        read_nbytes,
                    )
                    continue
                extents.append(
                    (
                        read_start,
                        slba + extent_skip // 512,
                        read_nbytes // 512,
                    )
                )
        return extents

    def _index_kv_object_layer_views(
        self,
        key: CacheEngineKey,
        full_record: KVObjectRecord,
        memory_obj: MemoryObj,
    ) -> int:
        """Register per-layer/per-role logical views for one stored chunk."""
        if self.kv_object_metadata_store is None:
            return 0
        shapes = memory_obj.metadata.shapes
        dtypes = memory_obj.metadata.dtypes
        if shapes is None or dtypes is None or len(shapes) != len(dtypes):
            return 0

        group_ranges = self._object_group_ranges(memory_obj, full_record)
        if not group_ranges:
            return 0

        group_views = self._object_group_view_specs(memory_obj, group_ranges)
        layer_views = self._object_layer_view_specs(memory_obj, group_ranges)
        indexed = 0
        for role, byte_ranges in group_views:
            view_object_id = self._key_to_object_id(key, role=role)
            view_length = sum(byte_range.length for byte_range in byte_ranges)
            view_record = KVObjectRecord(
                object_id=view_object_id,
                pool_id=full_record.pool_id,
                offset=byte_ranges[0].offset,
                length=view_length,
                aligned_length=view_length,
                shape=(view_length,),
                dtype="torch.uint8",
                state=KVObjectState.READY,
                raw_extents=full_record.raw_extents,
                byte_ranges=tuple(byte_ranges),
            )
            self.kv_object_metadata_store.put(view_record)
            indexed += 1
        for layer_id, role, byte_ranges in layer_views:
            view_object_id = self._key_to_object_id(
                key,
                layer_id=layer_id,
                role=role,
            )
            view_length = sum(byte_range.length for byte_range in byte_ranges)
            view_record = KVObjectRecord(
                object_id=view_object_id,
                pool_id=full_record.pool_id,
                offset=byte_ranges[0].offset,
                length=view_length,
                aligned_length=view_length,
                shape=(view_length,),
                dtype="torch.uint8",
                state=KVObjectState.READY,
                raw_extents=full_record.raw_extents,
                byte_ranges=tuple(byte_ranges),
            )
            self.kv_object_metadata_store.put(view_record)
            indexed += 1
        return indexed

    def _write_compact_retrieve_object(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        buffer: memoryview,
        raw_writer: Optional[KVObjectRawWriter],
        *,
        excluded_roles: frozenset[str],
        object_role: str,
        profile_op: str,
        allow_empty: bool = False,
    ) -> Optional[int]:
        """Materialize a DSv4 payload with selected KV roles removed.

        Returns:
            The logical payload length on success, including zero for an
            allowed metadata-only payload. ``None`` indicates failure.
        """
        if self.kv_object_pool_layout is None or self.kv_object_metadata_store is None:
            return None
        group_ranges = self._object_group_ranges_for_offset(memory_obj, 0)
        if not group_ranges:
            return None
        klg_manager = (
            self.metadata.kv_layer_groups_manager if self.metadata is not None else None
        )
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return None
        dtypes = memory_obj.metadata.dtypes or []

        schema_roles = {
            self._kv_group_role(
                group,
                dtypes[group_idx] if group_idx < len(dtypes) else group.dtype,
            )
            for group_idx, group in enumerate(klg_manager.kv_layer_groups)
        }
        missing_schema_roles = set(excluded_roles) - schema_roles
        if missing_schema_roles:
            logger.warning(
                "KV_OBJECT_STORE_PROFILE op=%s key=%s status=skip "
                "reason=excluded_role_not_in_schema missing=%s",
                profile_op,
                key.to_string(),
                sorted(missing_schema_roles),
            )
            return None

        selected_ranges: list[KVObjectByteRange] = []
        skipped_roles: set[str] = set()
        for group_idx, group_range in group_ranges:
            if group_idx >= len(klg_manager.kv_layer_groups):
                continue
            group = klg_manager.kv_layer_groups[group_idx]
            dtype = dtypes[group_idx] if group_idx < len(dtypes) else group.dtype
            role = self._kv_group_role(group, dtype)
            if role in excluded_roles:
                skipped_roles.add(role)
                continue
            selected_ranges.append(group_range)
        if not selected_ranges:
            if allow_empty and skipped_roles:
                logger.info(
                    "KV_OBJECT_STORE_PROFILE op=%s key=%s bytes=0 "
                    "mode=metadata_only ranges=0 direct=1 "
                    "excluded_present=%s write_ms=0.000 total_ms=0.000",
                    profile_op,
                    key.to_string(),
                    sorted(skipped_roles),
                )
                return 0
            logger.warning(
                "KV_OBJECT_STORE_PROFILE op=%s key=%s status=skip "
                "reason=empty_compact_payload excluded=%s present=%s",
                profile_op,
                key.to_string(),
                sorted(excluded_roles),
                sorted(skipped_roles),
            )
            return None

        selected_ranges.sort(key=lambda byte_range: byte_range.target_offset)
        compact_nbytes = sum(byte_range.length for byte_range in selected_ranges)
        if compact_nbytes <= 0:
            return None
        if raw_writer is not None and not self.kv_object_payload_fits_tutti_staging(
            compact_nbytes
        ):
            logger.warning(
                "KV_OBJECT_STORE_PROFILE op=%s key=%s "
                "status=skip reason=staging_capacity bytes=%d "
                "staging_bytes=%s",
                profile_op,
                key.to_string(),
                compact_nbytes,
                self.kv_object_tutti_raw_staging_bytes,
            )
            return None

        # Non-tail DSv4 MemoryObjs may already zero-shape every excluded
        # group.  In that case the input buffer is the canonical compact main
        # payload and must be registered directly; requiring every excluded
        # role to appear in ``group_ranges`` leaves all non-tail manifests
        # permanently incomplete.  Repack only when positive-size excluded
        # ranges still create holes in the source buffer.
        direct_payload = compact_nbytes == len(buffer)
        expected_source_offset = 0
        if direct_payload:
            for byte_range in selected_ranges:
                if byte_range.target_offset != expected_source_offset:
                    direct_payload = False
                    break
                expected_source_offset += byte_range.length
        compact_payload: Optional[bytearray] = None
        if direct_payload:
            payload_view = buffer
        else:
            compact_payload = bytearray(compact_nbytes)
            target_offset = 0
            for byte_range in selected_ranges:
                source_start = byte_range.target_offset
                source_end = source_start + byte_range.length
                compact_payload[target_offset : target_offset + byte_range.length] = (
                    buffer[source_start:source_end].tobytes()
                )
                target_offset += byte_range.length
            payload_view = memoryview(compact_payload)

        object_id = self._key_to_object_id(
            key,
            role=object_role,
        )
        start = time.perf_counter()
        existing = self.kv_object_metadata_store.get(object_id)
        if (
            existing is not None
            and existing.state == KVObjectState.READY
            and existing.length == compact_nbytes
        ):
            # Content-addressed key already persisted with identical length:
            # skip the redundant NVMe rewrite (fires on every hit otherwise;
            # see the matching guard in the primary object write path).
            return compact_nbytes
        if existing is None:
            if raw_writer is not None:
                offset, end_offset, aligned_length = (
                    self.kv_object_pool_layout.next_allocation_bounds(compact_nbytes)
                )
                if not self.kv_object_raw_region_covers(offset, aligned_length):
                    self.mark_kv_object_raw_region_full()
                    logger.warning(
                        "KV_OBJECT_STORE_PROFILE op=%s key=%s "
                        "status=skip reason=raw_region_full offset=%d end=%d "
                        "aligned_bytes=%d",
                        profile_op,
                        key.to_string(),
                        offset,
                        end_offset,
                        aligned_length,
                    )
                    return None
            record = self.kv_object_pool_layout.allocate(
                object_id,
                length=compact_nbytes,
                shape=(compact_nbytes,),
                dtype="torch.uint8",
            )
        else:
            record = existing
            if record.length != compact_nbytes:
                if raw_writer is not None:
                    offset, end_offset, aligned_length = (
                        self.kv_object_pool_layout.next_allocation_bounds(
                            compact_nbytes
                        )
                    )
                    if not self.kv_object_raw_region_covers(offset, aligned_length):
                        self.mark_kv_object_raw_region_full()
                        logger.warning(
                            "KV_OBJECT_STORE_PROFILE op=%s key=%s "
                            "status=skip reason=raw_region_full offset=%d end=%d "
                            "aligned_bytes=%d",
                            profile_op,
                            key.to_string(),
                            offset,
                            end_offset,
                            aligned_length,
                        )
                        return None
                logger.info(
                    "KV_OBJECT_STORE_PROFILE op=%s key=%s "
                    "status=reallocate reason=length_changed old=%d new=%d",
                    profile_op,
                    key.to_string(),
                    record.length,
                    compact_nbytes,
                )
                record = self.kv_object_pool_layout.allocate(
                    object_id,
                    length=compact_nbytes,
                    shape=(compact_nbytes,),
                    dtype="torch.uint8",
                )

        if raw_writer is not None:
            raw_extents, write_ms = raw_writer(record, payload_view)
            ready_record = record.with_raw_extents(raw_extents).mark_ready()
            mode = "tutti_raw"
        else:
            if self.kv_object_pool_io is None:
                return None
            write_ms = self.kv_object_pool_io.write_object(record, payload_view)
            ready_record = record.mark_ready()
            mode = "pool_file"
        self.kv_object_metadata_store.put(ready_record)
        logger.info(
            "KV_OBJECT_STORE_PROFILE op=%s key=%s bytes=%d "
            "mode=%s ranges=%d direct=%d excluded_present=%s "
            "write_ms=%.3f total_ms=%.3f",
            profile_op,
            key.to_string(),
            compact_nbytes,
            mode,
            len(selected_ranges),
            int(direct_payload),
            sorted(skipped_roles),
            write_ms,
            (time.perf_counter() - start) * 1000.0,
        )
        return compact_nbytes

    def _write_hca_deferred_retrieve_object(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        buffer: memoryview,
        raw_writer: Optional[KVObjectRawWriter],
    ) -> bool:
        """Materialize the non-HCA DSv4 retrieve payload as a compact object."""
        return (
            self._write_compact_retrieve_object(
                key,
                memory_obj,
                buffer,
                raw_writer,
                excluded_roles=frozenset({"hca_attention_kv"}),
                object_role=_DSV4_HCA_DEFERRED_RETRIEVE_ROLE,
                profile_op="write_hca_deferred",
            )
            is not None
        )

    def _write_csa_generation_compact_object(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        buffer: memoryview,
        raw_writer: Optional[KVObjectRawWriter],
        build: _CSAStreamingLayoutBuild,
    ) -> bool:
        """Write and publish the compact main object for one generation."""
        logical_role = self._csa_streaming_compact_role()
        physical_role = self._generation_object_role(
            logical_role,
            build.generation,
        )
        # GLM keeps its compact 21-layer shared IndexCache in the main object;
        # only the 78 sparse-attention KV layers are materialized as physical
        # layer-major sidecars.  DSv4 continues to exclude both native streams.
        excluded_roles = {"csa_attention_kv"}
        if not _glm_dsa_layer_major_enabled():
            excluded_roles.add("csa_indexer_cache")
        if _env_flag("LMCACHE_DSV4_HCA_WALKER"):
            excluded_roles.add("hca_attention_kv")
        compact_length = self._write_compact_retrieve_object(
            key,
            memory_obj,
            buffer,
            raw_writer,
            excluded_roles=frozenset(excluded_roles),
            object_role=physical_role,
            profile_op="write_csa_streaming_main",
            allow_empty=True,
        )
        if compact_length is None or self.kv_object_metadata_store is None:
            return False
        object_id = self._key_to_object_id(key, role=physical_role)
        record = self.kv_object_metadata_store.get(object_id)
        if compact_length > 0:
            if (
                not self._ready_tutti_raw_record(record)
                or record is None
                or record.length != compact_length
            ):
                return False
        elif record is not None:
            logger.warning(
                "KV_OBJECT_STORE_PROFILE op=write_csa_streaming_main key=%s "
                "status=skip reason=empty_payload_has_physical_record "
                "record_bytes=%d",
                key.to_string(),
                record.length,
            )
            return False
        build.objects[(logical_role, 0)] = _CSAStreamingObject(
            logical_role=logical_role,
            layer_id=0,
            object_id=object_id,
            length=compact_length,
        )
        with self._csa_layout_lock:
            if self._csa_staged_write_counts.get(key.to_string(), 0) > 0:
                # Sidecars are already safely owned by the host stage queue,
                # but the generation must remain lookup-invisible until the
                # last SSD completion callback publishes it.
                return True
        return self._publish_csa_layout(key, build)

    def _write_hca_slab_object(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        buffer: memoryview,
        raw_writer: Optional[KVObjectRawWriter],
    ) -> bool:
        """Materialize the whole HCA group as one slab object."""
        if self.kv_object_pool_layout is None or self.kv_object_metadata_store is None:
            return False
        group_ranges = self._object_group_ranges_for_offset(memory_obj, 0)
        if not group_ranges:
            return False
        klg_manager = (
            self.metadata.kv_layer_groups_manager if self.metadata is not None else None
        )
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return False
        dtypes = memory_obj.metadata.dtypes or []

        slab_ranges: list[KVObjectByteRange] = []
        for group_idx, group_range in group_ranges:
            if group_idx >= len(klg_manager.kv_layer_groups):
                continue
            group = klg_manager.kv_layer_groups[group_idx]
            dtype = dtypes[group_idx] if group_idx < len(dtypes) else group.dtype
            role = self._kv_group_role(group, dtype)
            if role == "hca_attention_kv":
                slab_ranges.append(group_range)
        if len(slab_ranges) != 1:
            return False

        slab_range = slab_ranges[0]
        slab_nbytes = slab_range.length
        object_id = self._key_to_object_id(key, role=_DSV4_HCA_SLAB_ROLE)
        existing = self.kv_object_metadata_store.get(object_id)
        if (
            existing is not None
            and existing.state == KVObjectState.READY
            and existing.length == slab_nbytes
        ):
            # Content-addressed key already persisted with identical length:
            # skip the redundant NVMe rewrite AND the slab payload copy
            # (fires on every hit otherwise; see the primary-path guard).
            return True
        if raw_writer is not None and not self.kv_object_payload_fits_tutti_staging(
            slab_nbytes
        ):
            logger.warning(
                "KV_OBJECT_STORE_PROFILE op=write_hca_slab key=%s status=skip "
                "reason=staging_capacity bytes=%d staging_bytes=%s",
                key.to_string(),
                slab_nbytes,
                self.kv_object_tutti_raw_staging_bytes,
            )
            return False

        source_start = slab_range.target_offset
        source_end = source_start + slab_nbytes
        slab_payload = bytearray(slab_nbytes)
        slab_payload[:] = buffer[source_start:source_end].tobytes()

        start = time.perf_counter()
        if existing is None:
            if raw_writer is not None:
                offset, end_offset, aligned_length = (
                    self.kv_object_pool_layout.next_allocation_bounds(slab_nbytes)
                )
                if not self.kv_object_raw_region_covers(offset, aligned_length):
                    self.mark_kv_object_raw_region_full()
                    logger.warning(
                        "KV_OBJECT_STORE_PROFILE op=write_hca_slab key=%s "
                        "status=skip reason=raw_region_full offset=%d end=%d "
                        "aligned_bytes=%d",
                        key.to_string(),
                        offset,
                        end_offset,
                        aligned_length,
                    )
                    return False
            record = self.kv_object_pool_layout.allocate(
                object_id,
                length=slab_nbytes,
                shape=(slab_nbytes,),
                dtype="torch.uint8",
            )
        else:
            record = existing
            if record.length != slab_nbytes:
                if raw_writer is not None:
                    offset, end_offset, aligned_length = (
                        self.kv_object_pool_layout.next_allocation_bounds(slab_nbytes)
                    )
                    if not self.kv_object_raw_region_covers(offset, aligned_length):
                        self.mark_kv_object_raw_region_full()
                        logger.warning(
                            "KV_OBJECT_STORE_PROFILE op=write_hca_slab key=%s "
                            "status=skip reason=raw_region_full offset=%d end=%d "
                            "aligned_bytes=%d",
                            key.to_string(),
                            offset,
                            end_offset,
                            aligned_length,
                        )
                        return False
                logger.info(
                    "KV_OBJECT_STORE_PROFILE op=write_hca_slab key=%s "
                    "status=reallocate reason=length_changed old=%d new=%d",
                    key.to_string(),
                    record.length,
                    slab_nbytes,
                )
                record = self.kv_object_pool_layout.allocate(
                    object_id,
                    length=slab_nbytes,
                    shape=(slab_nbytes,),
                    dtype="torch.uint8",
                )

        payload_view = memoryview(slab_payload)
        if raw_writer is not None:
            raw_extents, write_ms = raw_writer(record, payload_view)
            ready_record = record.with_raw_extents(raw_extents).mark_ready()
            mode = "tutti_raw"
        else:
            if self.kv_object_pool_io is None:
                return False
            write_ms = self.kv_object_pool_io.write_object(record, payload_view)
            ready_record = record.mark_ready()
            mode = "pool_file"
        self.kv_object_metadata_store.put(ready_record)
        logger.info(
            "KV_OBJECT_STORE_PROFILE op=write_hca_slab key=%s bytes=%d "
            "mode=%s write_ms=%.3f total_ms=%.3f",
            key.to_string(),
            slab_nbytes,
            mode,
            write_ms,
            (time.perf_counter() - start) * 1000.0,
        )
        return True

    def _object_group_ranges(
        self,
        memory_obj: MemoryObj,
        full_record: KVObjectRecord,
    ) -> list[tuple[int, KVObjectByteRange]]:
        """Derive per-group byte ranges for one stored chunk."""
        return self._object_group_ranges_for_offset(memory_obj, full_record.offset)

    def _layer_segment_layout(
        self,
        memory_obj: MemoryObj,
    ) -> tuple[LayerSegmentLayout, bool]:
        """Return cached DSv4 layer segments for one MemoryObj layout.

        Args:
            memory_obj: Chunk whose shape and dtype metadata define the byte
                layout. Returned offsets are relative to its byte buffer.

        Returns:
            An immutable role/layer segment layout and whether it came from
            the cache. Layout calculation is shape-dependent but independent
            of chunk contents, so full chunks in a long prefix share it.
        """
        shapes = memory_obj.metadata.shapes
        dtypes = memory_obj.metadata.dtypes
        if shapes is None or dtypes is None or len(shapes) != len(dtypes):
            return (), False
        signature = tuple(
            (
                tuple(int(dimension) for dimension in shape),
                str(dtype),
            )
            for shape, dtype in zip(shapes, dtypes, strict=True)
        )
        cached = self._layer_segment_layout_cache.get(signature)
        if cached is not None:
            return cached, True

        group_ranges = self._object_group_ranges_for_offset(memory_obj, 0)
        entries: list[tuple[tuple[str, int], tuple[tuple[int, int], ...]]] = []
        for layer_id, role, byte_ranges in self._object_layer_view_specs(
            memory_obj,
            group_ranges,
        ):
            if role not in {
                "csa_attention_kv",
                "hca_attention_kv",
                "csa_indexer_cache",
            }:
                continue
            entries.append(
                (
                    (role, int(layer_id)),
                    tuple(
                        (int(byte_range.offset), int(byte_range.length))
                        for byte_range in byte_ranges
                    ),
                )
            )
        layout = tuple(entries)
        # Concurrent duplicate calculations are harmless: layouts are
        # immutable and setdefault publishes one equivalent value.
        layout = self._layer_segment_layout_cache.setdefault(signature, layout)
        return layout, False

    def _object_group_ranges_for_offset(
        self,
        memory_obj: MemoryObj,
        base_offset: int,
    ) -> list[tuple[int, KVObjectByteRange]]:
        """Derive per-group byte ranges from a logical base offset."""
        shapes = memory_obj.metadata.shapes
        dtypes = memory_obj.metadata.dtypes
        if shapes is None or dtypes is None or len(shapes) != len(dtypes):
            return []
        ranges: list[tuple[int, KVObjectByteRange]] = []
        target_offset = 0
        current_offset = base_offset
        for group_idx, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True)):
            group_nbytes = shape.numel() * dtype.itemsize
            if group_nbytes > 0:
                ranges.append(
                    (
                        group_idx,
                        KVObjectByteRange(
                            offset=current_offset,
                            length=group_nbytes,
                            target_offset=target_offset,
                        ),
                    )
                )
            current_offset += group_nbytes
            target_offset += group_nbytes
        return ranges

    def _object_layer_view_specs(
        self,
        memory_obj: MemoryObj,
        group_ranges: Sequence[tuple[int, KVObjectByteRange]],
    ) -> list[tuple[int, str, list[KVObjectByteRange]]]:
        """Return ``(layer_id, role, byte_ranges)`` specs for object views."""
        klg_manager = (
            self.metadata.kv_layer_groups_manager if self.metadata is not None else None
        )
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return []
        dtypes = memory_obj.metadata.dtypes or []
        shapes = memory_obj.metadata.shapes or []
        specs: list[tuple[int, str, list[KVObjectByteRange]]] = []
        for group_idx, group_range in group_ranges:
            if group_idx >= len(klg_manager.kv_layer_groups):
                continue
            group = klg_manager.kv_layer_groups[group_idx]
            dtype = dtypes[group_idx] if group_idx < len(dtypes) else group.dtype
            shape = shapes[group_idx] if group_idx < len(shapes) else None
            role = self._kv_group_role(group, dtype)
            per_layer_ranges = self._split_group_range_by_layer(
                group,
                group_range,
                shape=shape,
            )
            for layer_id, byte_ranges in zip(
                group.layer_indices,
                per_layer_ranges,
                strict=False,
            ):
                specs.append((int(layer_id), role, byte_ranges))
        return specs

    def _object_layer_compression_ratios(
        self,
        memory_obj: MemoryObj,
    ) -> dict[tuple[str, int], int]:
        """Return compression ratios keyed by object role and layer id."""
        klg_manager = (
            self.metadata.kv_layer_groups_manager if self.metadata is not None else None
        )
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return {}
        dtypes = memory_obj.metadata.dtypes or []
        ratios: dict[tuple[str, int], int] = {}
        for group_idx, group in enumerate(klg_manager.kv_layer_groups):
            dtype = dtypes[group_idx] if group_idx < len(dtypes) else group.dtype
            role = self._kv_group_role(group, dtype)
            compression_ratio = max(1, int(group.compress_ratio))
            for layer_id in group.layer_indices:
                ratios[(role, int(layer_id))] = compression_ratio
        return ratios

    def _object_group_view_specs(
        self,
        memory_obj: MemoryObj,
        group_ranges: Sequence[tuple[int, KVObjectByteRange]],
    ) -> list[tuple[str, list[KVObjectByteRange]]]:
        """Return role-level whole-group logical views for object reads."""
        klg_manager = (
            self.metadata.kv_layer_groups_manager if self.metadata is not None else None
        )
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return []
        dtypes = memory_obj.metadata.dtypes or []
        specs: list[tuple[str, list[KVObjectByteRange]]] = []
        for group_idx, group_range in group_ranges:
            if group_idx >= len(klg_manager.kv_layer_groups):
                continue
            group = klg_manager.kv_layer_groups[group_idx]
            dtype = dtypes[group_idx] if group_idx < len(dtypes) else group.dtype
            role = self._kv_group_role(group, dtype)
            if role != "hca_attention_kv":
                continue
            specs.append(
                (
                    _DSV4_HCA_SLAB_ROLE,
                    [
                        KVObjectByteRange(
                            offset=group_range.offset,
                            length=group_range.length,
                            target_offset=0,
                        )
                    ],
                )
            )
        return specs

    def _split_group_range_by_layer(
        self,
        group: Any,
        group_range: KVObjectByteRange,
        *,
        shape: Optional[torch.Size],
    ) -> list[list[KVObjectByteRange]]:
        """Split a group byte range into per-layer logical payload ranges."""
        if shape is not None and len(shape) >= 4:
            kv_size = int(shape[0])
            num_layers = int(shape[1])
            rows = int(shape[2])
            hidden_dim = int(shape[3])
        else:
            kv_size = int(group.shape_desc.kv_size)
            num_layers = int(group.num_layers)
            rows = int(group.physical_chunk_size or group.shape_desc.bs)
            hidden_dim = int(group.hidden_dim_size)
        element_size = int(group.dtype.itemsize)
        if kv_size <= 0 or num_layers <= 0 or rows <= 0 or hidden_dim <= 0:
            return []
        layer_plane_bytes = rows * hidden_dim * element_size
        kv_plane_bytes = num_layers * layer_plane_bytes
        per_layer: list[list[KVObjectByteRange]] = []
        for layer_idx in range(num_layers):
            ranges: list[KVObjectByteRange] = []
            target_offset = 0
            for kv_idx in range(kv_size):
                offset = (
                    group_range.offset
                    + kv_idx * kv_plane_bytes
                    + layer_idx * layer_plane_bytes
                )
                ranges.append(
                    KVObjectByteRange(
                        offset=offset,
                        length=layer_plane_bytes,
                        target_offset=target_offset,
                    )
                )
                target_offset += layer_plane_bytes
            per_layer.append(ranges)
        return per_layer

    @staticmethod
    def _kv_group_role(group: Any, dtype: torch.dtype) -> str:
        """Return a role label for one KV layer group."""
        hidden_dim = int(group.hidden_dim_size)
        compress_ratio = int(group.compress_ratio)
        if dtype == torch.float32:
            return "compressor_state"
        if dtype != torch.uint8:
            return "kv"
        if _glm_dsa_layer_major_enabled():
            return "csa_attention_kv" if hidden_dim == 656 else "kv"
        if hidden_dim == 132:
            return "csa_indexer_cache"
        if hidden_dim != 584:
            return "kv"
        if compress_ratio == 1:
            return "swa_cache"
        if compress_ratio >= 64 or int(group.shape_desc.bs) <= 2:
            return "hca_attention_kv"
        if compress_ratio == 4 or int(group.num_layers) == 30:
            return "csa_attention_kv"
        return "kv"

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            if key not in self.dict:
                self._log_contains_miss_unlocked(
                    "contains",
                    key,
                    reason="dict_miss",
                    index=0,
                    hit_count=0,
                )
                return False
            if not self._has_readable_chunk_object_unlocked(key):
                self._log_contains_miss_unlocked(
                    "contains",
                    key,
                    reason="object_unreadable",
                    index=0,
                    hit_count=0,
                )
                return False
            if pin:
                self.dict[key].pin()
                # vllm lookup sets pin to True
                self.keys_in_request.append(key)
            return True

    def contains_streaming_terminal(
        self,
        key: CacheEngineKey,
        token_count: int,
        pin: bool = False,
    ) -> bool:
        """Check and optionally pin one exact terminal streaming generation.

        Args:
            key: Content-addressed key of the terminal cached chunk.
            token_count: Exact number of logical tokens required by the lookup.
            pin: Whether to pin the terminal cache entry on success.

        Returns:
            ``True`` only if the key exists and its atomically published,
            runtime-compatible manifest covers exactly ``token_count`` tokens.
        """
        if token_count <= 0:
            return False
        with self.disk_lock:
            if key not in self.dict:
                self._log_streaming_terminal_miss_unlocked(
                    key,
                    token_count,
                    reason="dict_miss",
                )
                return False
            layout = self._active_csa_layout(key)
            if layout is None:
                self._log_streaming_terminal_miss_unlocked(
                    key,
                    token_count,
                    reason="layout_missing",
                )
                return False
            if layout.covered_tokens != int(token_count):
                self._log_streaming_terminal_miss_unlocked(
                    key,
                    token_count,
                    reason=f"coverage_{layout.covered_tokens}",
                )
                return False
            if not self._csa_layout_matches_runtime(layout):
                self._log_streaming_terminal_miss_unlocked(
                    key,
                    token_count,
                    reason="runtime_mismatch",
                )
                return False
            if pin:
                self.dict[key].pin()
                self.keys_in_request.append(key)
            return True

    def _log_streaming_terminal_miss_unlocked(
        self,
        key: CacheEngineKey,
        token_count: int,
        *,
        reason: str,
    ) -> None:
        """Log a bounded exact-generation miss without affecting lookup."""
        log_key = f"streaming_terminal:{reason}"
        count = self._contains_miss_log_counts.get(log_key, 0)
        if count >= 8:
            return
        self._contains_miss_log_counts[log_key] = count + 1
        logger.info(
            "CSA streaming terminal miss: key=%s requested_tokens=%d "
            "reason=%s dict_size=%d pending=%s",
            key.to_string(),
            token_count,
            reason,
            len(self.dict),
            key.to_string() in self._csa_pending_layouts,
        )

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.disk_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.disk_worker.exists_in_put_tasks(key)

    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            else:
                return False

    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            else:
                return False

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        if force:
            self.disk_lock.acquire()

        if not (meta := self.dict.pop(key, None)):
            if force:
                self.disk_lock.release()
            return False

        with self._csa_layout_lock:
            self._csa_active_layouts.pop(key.to_string(), None)
            self._csa_pending_layouts.pop(key.to_string(), None)
            self._csa_pending_layout_keys.pop(key.to_string(), None)
            self._csa_staged_write_counts.pop(key.to_string(), None)

        path = meta.path
        size = meta.size
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # NOTE: The following code will cause deadlock
        # res = asyncio.run_coroutine_threadsafe(
        #     self.disk_worker.submit_task("delete", os.remove, path),
        #     self.loop,
        # )
        # res.result()

        os.remove(path)

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.disk_lock.release()

        # Push kv evict msg with batching
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.EVICT,
                key=key.chunk_hash,
            )

        return True

    def insert_key(
        self,
        key: CacheEngineKey,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        cached_positions: Optional[torch.Tensor] = None,
        shapes: Optional[list[torch.Size]] = None,
        dtypes: Optional[list[torch.dtype]] = None,
    ) -> None:
        path = self._key_to_path(key)

        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                # The backing file has already been overwritten by write_file.
                # Refresh the stored metadata so the next get_blocking uses
                # the new shapes and size (e.g. when a DSv4 tail chunk
                # overwrites a non-tail entry with a different shape set).
                existing = self.dict[key]
                self.dict[key] = DiskCacheMetadata(
                    path=existing.path,
                    size=size,
                    shape=shape,
                    dtype=dtype,
                    cached_positions=cached_positions,
                    fmt=fmt,
                    pin_count=existing.pin_count,
                    shapes=shapes,
                    dtypes=dtypes,
                )
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = DiskCacheMetadata(
                    path,
                    size,
                    shape,
                    dtype,
                    cached_positions,
                    fmt,
                    0,
                    shapes,
                    dtypes,
                )

        # Push kv admit msg with batching
        if self.batched_msg_sender is not None and not has_stored:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.ADMIT,
                key=key.chunk_hash,
            )

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ):
        """
        Submit a single put task to store KV cache to disk asynchronously.

        :param key: The cache key for this KV chunk.
        :param memory_obj: The memory object containing the KV data.
        :param on_complete_callback: Optional callback invoked once per key
            after the disk write completes. Callback exceptions are caught
            and logged.
        """
        if memory_obj.raw_tensor is None:
            raise ValueError("Cannot store a MemoryObj without raw tensor storage")

        # skip repeated save
        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        self.disk_worker.insert_put_task(key)

        # TODO(Jiayi): Fragmentation is not considered here.
        required_size = memory_obj.get_physical_size()
        all_evict_keys = []
        evict_success = True
        with self.disk_lock:
            while self.current_cache_size + required_size > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found. Disk space under pressure."
                    )
                    evict_success = False
                    break

                for evict_key in evict_keys:
                    self.current_cache_size -= self.dict[evict_key].size

                self.batched_remove(evict_keys, force=False)

                all_evict_keys.extend(evict_keys)
            if evict_success:
                self.current_cache_size += required_size
                self.cache_policy.update_on_put(key)

        if not evict_success:
            return None

        memory_obj.ref_count_up()

        asyncio.run_coroutine_threadsafe(
            self.disk_worker.submit_task(
                "put",
                self.async_save_bytes_to_disk,
                key=key,
                memory_obj=memory_obj,
                on_complete_callback=on_complete_callback,
            ),
            self.loop,
        )

    # TODO(Jiayi): enable real batching
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Submit batched put tasks to store KV caches to disk asynchronously.

        :param keys: The cache keys for the KV chunks.
        :param memory_objs: The memory objects containing the KV data.
        :param transfer_spec: Optional transfer specification (unused).
        :param on_complete_callback: Optional callback invoked once per key
            after that key's disk write completes (not once per batch).
            Callback exceptions are caught and logged.
        """
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(
                key, memory_obj, on_complete_callback=on_complete_callback
            )

    def _get_blocking_with_shapes(
        self,
        key: CacheEngineKey,
        shapes_override: Optional[list[torch.Size]] = None,
    ) -> Optional[MemoryObj]:
        """Internal helper: load one KV chunk with an optional shapes override.

        Factored out so that :meth:`get_blocking` and
        :meth:`batched_get_blocking` share identical lock-acquire / load /
        policy-update logic while allowing a caller-supplied
        ``shapes_override`` to control the allocation size.

        The load is performed outside the lock so the disk_lock is not held
        during a potentially slow CPU staging pool allocation + memcpy.

        Args:
            key: Cache key identifying the KV chunk.
            shapes_override: When not ``None``, overrides the stored
                ``disk_meta.shapes`` for the buffer allocation, enabling a
                partial disk read (e.g. DSv4 prefix groups only).

        Returns:
            A ``MemoryObj`` with the loaded KV data, or ``None`` if the key
            is absent or the load fails.
        """
        with self.disk_lock:
            if key not in self.dict:
                return None

            disk_meta = self.dict[key]
            path = disk_meta.path
            dtype = disk_meta.dtype
            shape = disk_meta.shape
            fmt = disk_meta.fmt
            shapes = disk_meta.shapes
            dtypes = disk_meta.dtypes
            assert dtype is not None
            assert shape is not None

        memory_obj = self.load_bytes_from_disk(
            key,
            path,
            dtype=dtype,
            shape=shape,
            fmt=fmt,
            shapes=shapes,
            dtypes=dtypes,
            shapes_override=shapes_override,
        )

        if memory_obj is not None:
            with self.disk_lock:
                if key in self.dict:
                    self.cache_policy.update_on_hit(key, self.dict)

        return memory_obj

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """Load a cached KV chunk from disk synchronously.

        The cache policy is updated only after a successful load so that a
        failed load does not record a phantom cache hit and skew future
        eviction decisions.

        Args:
            key: The cache key identifying the KV chunk.

        Returns:
            A ``MemoryObj`` containing the loaded KV data, or ``None`` if
            the key is not present or the load fails.
        """
        return self._get_blocking_with_shapes(key, shapes_override=None)

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
        shapes_per_key: Optional[List[Optional[list[torch.Size]]]] = None,
    ) -> List[Optional[MemoryObj]]:
        """Load multiple cached KV chunks from disk synchronously.

        Args:
            keys: Cache keys identifying the KV chunks.
            shapes_per_key: Optional per-key allocation shape overrides.
                When provided, ``shapes_per_key[i]`` replaces the stored
                metadata shapes for ``keys[i]``.  Pass ``None`` for an
                individual entry to fall back to stored shapes for that key.
                Used by the DSv4-optimised retrieve path to limit I/O to
                prefix groups only for non-tail chunks, reducing per-chunk
                read size from ~116 MB to ~1.4 MB.

        Returns:
            A list of ``MemoryObj`` instances (one per key), with ``None``
            entries for keys that are absent or fail to load.
        """
        mem_objs: List[Optional[MemoryObj]] = []
        for i, key in enumerate(keys):
            override = shapes_per_key[i] if shapes_per_key is not None else None
            mem_objs.append(
                self._get_blocking_with_shapes(key, shapes_override=override)
            )
        return mem_objs

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        mem_objs: list[MemoryObj] = []
        paths: list[str] = []

        logger.debug(f"lookup_id: {lookup_id}; Prefetching {len(keys)} keys from disk.")
        for key in keys:
            self.disk_lock.acquire()
            assert key in self.dict, f"Key {key} not found in disk cache after pinning"

            path = self.dict[key].path
            dtype = self.dict[key].dtype
            shape = self.dict[key].shape
            fmt = self.dict[key].fmt
            shapes = self.dict[key].shapes
            dtypes = self.dict[key].dtypes

            assert dtype is not None
            assert shape is not None

            # busy_loop=False prevents spinning on the event loop thread;
            # if staging memory is exhausted the caller will get a logged
            # error rather than a silent deadlock.
            memory_obj = self.local_cpu_backend.allocate(
                shapes if shapes is not None else shape,
                dtypes if dtypes is not None else dtype,
                fmt,
                busy_loop=False,
            )

            if memory_obj is None:
                logger.error(
                    "Memory allocation failed during async disk load for key %s. "
                    "CPU staging pool may be exhausted (unpin() not called after "
                    "a previous retrieve). Returning partial results.",
                    key,
                )
                return mem_objs

            self.dict[key].pin()

            # NOTE(Jiayi): Currently, we consider prefetch as cache hit.
            # Update cache recency
            self.cache_policy.update_on_hit(key, self.dict)

            self.disk_lock.release()
            logger.debug(f"Prefetching {key} from disk.")
            memory_obj.pin()
            mem_objs.append(memory_obj)
            paths.append(path)

        return await self.disk_worker.submit_task(
            "prefetch",
            self.batched_async_load_bytes_from_disk,
            paths=paths,
            keys=keys,
            memory_objs=mem_objs,
        )

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        with self.disk_lock:
            for index, key in enumerate(keys):
                if key not in self.dict:
                    self._log_contains_miss_unlocked(
                        "batched_async_contains",
                        key,
                        reason="dict_miss",
                        index=index,
                        hit_count=num_hit_counts,
                    )
                    return num_hit_counts
                if not self._has_readable_chunk_object_unlocked(key):
                    self._log_contains_miss_unlocked(
                        "batched_async_contains",
                        key,
                        reason="object_unreadable",
                        index=index,
                        hit_count=num_hit_counts,
                    )
                    return num_hit_counts
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        return num_hit_counts

    def _has_readable_chunk_object_unlocked(self, key: CacheEngineKey) -> bool:
        """Return whether a raw object-store chunk can be read by Tutti."""
        if not self.kv_object_store_enabled or not self.kv_object_tutti_raw_enabled:
            return True
        if self.kv_object_metadata_store is None:
            return False
        if self._csa_streaming_layout_requested():
            return self._has_readable_csa_streaming_layout_unlocked(key)
        object_id = self._key_to_object_id(key)
        record = self.kv_object_metadata_store.get(object_id)
        if self._ready_tutti_raw_record(record):
            return True
        return self._has_readable_hca_deferred_objects_unlocked(key)

    @staticmethod
    def _csa_streaming_layout_requested() -> bool:
        """Return whether cold store must publish the CSA streaming layout."""
        return _env_flag("LMCACHE_INDEXER_ENABLE_PREFETCH")

    def _csa_streaming_compact_role(self) -> str:
        """Return the canonical generic-object role for the active pipeline."""
        if _env_flag("LMCACHE_DSV4_HCA_WALKER"):
            return _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE
        return _DSV4_CSA_DEFERRED_RETRIEVE_ROLE

    def _has_readable_csa_streaming_layout_unlocked(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """Return whether the atomically admitted CSA layout is readable."""
        return self._active_csa_layout_ready(key)

    def _has_readable_hca_deferred_objects_unlocked(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """Return whether compact non-HCA and HCA slab raw objects are ready."""
        if self.kv_object_metadata_store is None:
            return False
        compact = self.kv_object_metadata_store.get(
            self._key_to_object_id(
                key,
                role=_DSV4_HCA_DEFERRED_RETRIEVE_ROLE,
            )
        )
        slab = self.kv_object_metadata_store.get(
            self._key_to_object_id(
                key,
                role=_DSV4_HCA_SLAB_ROLE,
            )
        )
        return self._ready_tutti_raw_record(compact) and self._ready_tutti_raw_record(
            slab
        )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.

        :param on_complete_callback: Optional callback invoked after the disk
            write completes for this key. Callback exceptions are caught and
            logged.
        """
        buffer = memory_obj.byte_array
        path = self._key_to_path(key)

        size = len(buffer)
        self.usage += size
        self.stats_monitor.update_local_storage_usage(self.usage)

        raw_object_attempted = self.kv_object_store_enabled
        raw_object_written = False
        if raw_object_attempted:
            raw_object_written = self._write_kv_object_store(
                key,
                buffer,
                memory_obj=memory_obj,
            )

        # TODO(Jiayi): need to add ref count in disk memory object
        if not raw_object_written:
            if not self.write_file(buffer, path):
                memory_obj.ref_count_down()
                self.disk_worker.remove_put_task(key)
                logger.warning(
                    "LocalDiskBackend store skipped metadata insert because "
                    "filesystem write failed for %s",
                    path,
                )
                return

        # ref count down here because there's a ref_count_up in
        # `submit_put_task` above.
        # Ref count down better be before `insert_key` for testing
        # purposes (e.g., testing mem_leak).
        # TODO(Jiayi): This could be problematic if the
        # freed memory object is immediately reused.
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        cached_positions = memory_obj.metadata.cached_positions
        shapes = memory_obj.metadata.shapes
        dtypes = memory_obj.metadata.dtypes
        memory_obj.ref_count_down()

        self.insert_key(
            key,
            size,
            shape,
            dtype,
            fmt,
            cached_positions=cached_positions,
            shapes=shapes,
            dtypes=dtypes,
        )

        self.disk_worker.remove_put_task(key)

        # Call the completion callback if provided
        if on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning(f"on_complete_callback failed for key {key}: {e}")

    def batched_async_load_bytes_from_disk(
        self,
        paths: list[str],
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        write_back: bool = False,
    ) -> list[MemoryObj]:
        """
        Async load bytearray from disk.
        """

        logger.debug("Executing `async_load_bytes` from disk.")
        # TODO (Jiayi): handle the case where loading fails.
        for path, key, mem_obj in zip(paths, keys, memory_objs, strict=False):
            buffer = mem_obj.byte_array
            self.read_file(key, buffer, path)

            # TODO(Jiayi): Please recover the metadata in a more
            # elegant way in the future.
            cached_positions = self.dict[key].cached_positions
            mem_obj.metadata.cached_positions = cached_positions

            self.disk_lock.acquire()
            self.dict[key].unpin()
            self.disk_lock.release()

        return memory_objs

    def load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
        shapes: Optional[list[torch.Size]] = None,
        dtypes: Optional[list[torch.dtype]] = None,
        shapes_override: Optional[list[torch.Size]] = None,
    ) -> Optional[MemoryObj]:
        """Load bytearray from disk and return as a MemoryObj.

        When ``shapes_override`` is provided it replaces ``shapes`` for the
        memory allocation, enabling partial disk reads.  The file must store
        groups in ascending order so that reading only the first N bytes
        returns exactly the overridden groups (e.g. DSv4 prefix groups).

        Args:
            key: Cache key used to recover ``cached_positions`` metadata.
            path: Absolute path of the on-disk KV file.
            dtype: Fallback element dtype when ``dtypes`` is absent.
            shape: Fallback tensor shape when ``shapes`` is absent.
            fmt: Memory layout format for the allocation.
            shapes: Per-group shapes from stored metadata; used when
                ``shapes_override`` is ``None``.
            dtypes: Per-group dtypes from stored metadata.
            shapes_override: When not ``None``, replaces ``shapes`` for the
                allocation.  The resulting buffer is sized to hold exactly the
                overridden groups, so ``read_file`` issues a partial read of
                the first ``sum(s.numel()*elem_size for s in shapes_override)``
                bytes from the file.

        Returns:
            A ``MemoryObj`` whose tensor is populated with the bytes read from
            ``path``, or ``None`` if allocation fails.
        """
        effective_shapes = shapes_override if shapes_override is not None else shapes
        memory_obj = self.local_cpu_backend.allocate(
            effective_shapes if effective_shapes is not None else shape,
            dtypes if dtypes is not None else dtype,
            fmt,
        )
        assert memory_obj is not None, "Memory allocation failed during disk load."

        buffer = memory_obj.byte_array
        self.read_file(key, buffer, path)

        # TODO(Jiayi): Please recover the metadata in a more
        # elegant way in the future.
        cached_positions = self.dict[key].cached_positions
        memory_obj.metadata.cached_positions = cached_positions

        return memory_obj

    def _write_kv_object_store(
        self,
        key: CacheEngineKey,
        buffer: memoryview,
        *,
        memory_obj: MemoryObj,
    ) -> bool:
        """Write a chunk-level object-store copy when enabled.

        Returns:
            True when the object store is the authoritative durable write and
            the caller should skip the legacy ``.pt`` filesystem write.  False
            means the caller still needs the normal filesystem write.
        """
        if (
            not self.kv_object_store_enabled
            or self.kv_object_pool_layout is None
            or self.kv_object_metadata_store is None
        ):
            return False

        start = time.perf_counter()
        object_id = self._key_to_object_id(key)
        raw_writer = (
            self.kv_object_tutti_raw_writer
            if self.kv_object_tutti_raw_cold_store_enabled
            else None
        )
        if self.kv_object_tutti_raw_cold_store_enabled and raw_writer is None:
            self._log_kv_object_write_skip(
                key,
                reason="raw_writer_missing",
                bytes_len=len(buffer),
            )
            return False
        if self._csa_streaming_layout_requested():
            # Idempotent hit-side re-save: the active immutable generation is
            # already complete, so no object is rewritten or backfilled.
            if self._active_csa_layout_ready(key):
                self._log_kv_object_write_skip(
                    key,
                    reason="csa_generation_already_ready",
                    bytes_len=len(buffer),
                )
                return True
            with self._csa_layout_lock:
                build = self._csa_pending_layouts.get(key.to_string())
            if build is None:
                self._log_kv_object_write_skip(
                    key,
                    reason="csa_generation_not_prepared",
                    bytes_len=len(buffer),
                )
                return False
            with self.kv_object_store_lock:
                return self._write_csa_generation_compact_object(
                    key,
                    memory_obj,
                    buffer,
                    raw_writer,
                    build,
                )
        if raw_writer is not None and not self.kv_object_payload_fits_tutti_staging(
            len(buffer)
        ):
            if not (
                _env_flag("LMCACHE_DSV4_DEFER_HCA_TO_MOE")
                and _env_flag("LMCACHE_HCA_ENABLE_OBJECT_SOURCE")
            ):
                self._log_kv_object_write_skip(
                    key,
                    reason="raw_staging_capacity",
                    bytes_len=len(buffer),
                )
                return False
            try:
                # Submit HCA deferred writes to a dedicated background executor
                # so the NVMe I/O (via raw_writer) does not hold kv_object_store_lock
                # on the disk-worker thread and cannot saturate the 4-worker pool.
                # Bump ref count to keep the memory alive until the task finishes.
                memory_obj.ref_count_up()

                def _do_hca_deferred_writes(
                    _key=key,
                    _memory_obj=memory_obj,
                    _buffer=buffer,
                    _raw_writer=raw_writer,
                    _start=start,
                ) -> None:
                    try:
                        with self.kv_object_store_lock:
                            compact_written = self._write_hca_deferred_retrieve_object(
                                _key,
                                _memory_obj,
                                _buffer,
                                _raw_writer,
                            )
                            slab_written = self._write_hca_slab_object(
                                _key,
                                _memory_obj,
                                _buffer,
                                _raw_writer,
                            )
                        elapsed_ms = (time.perf_counter() - _start) * 1000.0
                        logger.info(
                            "KV_OBJECT_STORE_PROFILE op=write key=%s bytes=%d "
                            "status=deferred_only hca_compact=%s slab=%s "
                            "staging_bytes=%s total_ms=%.3f",
                            _key.to_string(),
                            len(_buffer),
                            compact_written,
                            slab_written,
                            self.kv_object_tutti_raw_staging_bytes,
                            elapsed_ms,
                        )
                    except Exception as exc:
                        logger.warning(
                            "KV_OBJECT_STORE_PROFILE op=write key=%s "
                            "status=hca_deferred_failed error=%s",
                            _key.to_string(),
                            exc,
                        )
                    finally:
                        _memory_obj.ref_count_down()

                self._hca_write_executor.submit(_do_hca_deferred_writes)
                return True
            except Exception as exc:
                memory_obj.ref_count_down()
                logger.warning(
                    "KV_OBJECT_STORE_PROFILE op=write key=%s "
                    "status=hca_deferred_failed error=%s",
                    key.to_string(),
                    exc,
                )
                return False
        try:
            with self.kv_object_store_lock:
                existing = self.kv_object_metadata_store.get(object_id)
                if (
                    existing is not None
                    and existing.state == KVObjectState.READY
                    and existing.length == len(buffer)
                ):
                    # Idempotent re-store: keys are content-addressed (chunk
                    # hash), so a ready record of identical length means the
                    # same bytes are already on NVMe.  vLLM re-saves the whole
                    # prefix on every hit request; without this check each hit
                    # rewrote every chunk (measured 16k+ redundant NVMe writes
                    # in 3 minutes at 26K context), and with the loader's
                    # io_lock those writes queued ahead of retrieve reads --
                    # the direct cause of repeat-hit latency inflating from
                    # ~0.6 s to 2-6 s.
                    self._log_kv_object_write_skip(
                        key,
                        reason="already_stored",
                        bytes_len=len(buffer),
                    )
                    return True
                if existing is None:
                    if raw_writer is not None:
                        offset, end_offset, aligned_length = (
                            self.kv_object_pool_layout.next_allocation_bounds(
                                len(buffer)
                            )
                        )
                        if not self.kv_object_raw_region_covers(
                            offset,
                            aligned_length,
                        ):
                            self.mark_kv_object_raw_region_full()
                            logger.warning(
                                "KV_OBJECT_STORE_PROFILE op=write key=%s "
                                "status=skip reason=raw_region_full "
                                "offset=%d end=%d aligned_bytes=%d",
                                key.to_string(),
                                offset,
                                end_offset,
                                aligned_length,
                            )
                            return False
                    record = self.kv_object_pool_layout.allocate(
                        object_id,
                        length=len(buffer),
                        shape=(len(buffer),),
                        dtype="torch.uint8",
                    )
                else:
                    record = existing
                    if record.length != len(buffer):
                        logger.warning(
                            "KV_OBJECT_STORE_PROFILE op=write key=%s "
                            "status=skip reason=length_changed old=%d new=%d",
                            key.to_string(),
                            record.length,
                            len(buffer),
                        )
                        return False
                if raw_writer is not None:
                    raw_extents, write_ms = raw_writer(record, buffer)
                    ready_record = record.with_raw_extents(raw_extents).mark_ready()
                    self.kv_object_metadata_store.put(ready_record)
                    mode = "tutti_raw"
                else:
                    if self.kv_object_pool_io is None:
                        return False
                    write_ms = self.kv_object_pool_io.write_object(record, buffer)
                    ready_record = record.mark_ready()
                    self.kv_object_metadata_store.put(ready_record)
                    mode = "pool_file"
                indexed_views = self._index_kv_object_layer_views(
                    key,
                    ready_record,
                    memory_obj,
                )
            # Submit the supplementary HCA slab/deferred writes as a
            # fire-and-forget background task so the NVMe I/O for these
            # secondary objects does not extend the lock hold time and
            # does not occupy the disk_worker pool capacity.
            memory_obj.ref_count_up()

            def _do_hca_supplementary_writes(
                _key=key,
                _memory_obj=memory_obj,
                _buffer=buffer,
                _raw_writer=raw_writer,
            ) -> None:
                try:
                    with self.kv_object_store_lock:
                        deferred = self._write_hca_deferred_retrieve_object(
                            _key,
                            _memory_obj,
                            _buffer,
                            _raw_writer,
                        )
                        slab = self._write_hca_slab_object(
                            _key,
                            _memory_obj,
                            _buffer,
                            _raw_writer,
                        )
                    if deferred or slab:
                        logger.info(
                            "KV_OBJECT_STORE_PROFILE op=write key=%s "
                            "status=supplementary hca_deferred=%s slab=%s",
                            _key.to_string(),
                            deferred,
                            slab,
                        )
                except Exception as exc:
                    logger.warning(
                        "KV_OBJECT_STORE_PROFILE op=write key=%s "
                        "status=hca_supplementary_failed error=%s",
                        _key.to_string(),
                        exc,
                    )
                finally:
                    _memory_obj.ref_count_down()

            try:
                self._hca_write_executor.submit(_do_hca_supplementary_writes)
            except Exception:
                # submit() failed (e.g. executor shut down); balance the
                # ref_count_up we did before defining the closure.
                memory_obj.ref_count_down()
                raise
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "KV_OBJECT_STORE_PROFILE op=write key=%s bytes=%d "
                "mode=%s write_ms=%.3f views=%d total_ms=%.3f",
                key.to_string(),
                len(buffer),
                mode,
                write_ms,
                indexed_views,
                elapsed_ms,
            )
            return True
        except Exception as exc:
            logger.warning(
                "KV_OBJECT_STORE_PROFILE op=write key=%s status=failed error=%s",
                key.to_string(),
                exc,
            )
            return False

    def _log_contains_miss_unlocked(
        self,
        op: str,
        key: CacheEngineKey,
        *,
        reason: str,
        index: int,
        hit_count: int,
    ) -> None:
        """Log bounded diagnostics for disk/object-store prefix lookup misses."""
        if not self._diagnose_contains_misses:
            return
        log_key = f"{op}:{reason}"
        count = self._contains_miss_log_counts.get(log_key, 0)
        if count >= 32:
            return
        self._contains_miss_log_counts[log_key] = count + 1

        object_state = "disabled"
        raw_extents = 0
        raw_readable = False
        raw_read_bytes = 0
        if self.kv_object_store_enabled and self.kv_object_tutti_raw_enabled:
            object_state = "metadata_store_missing"
            if self.kv_object_metadata_store is not None:
                record = self.kv_object_metadata_store.get(self._key_to_object_id(key))
                if record is None:
                    object_state = "no_record"
                else:
                    object_state = getattr(record.state, "name", str(record.state))
                    raw_extents = len(record.raw_extents)
                    raw_read_bytes = self.kv_object_record_raw_read_bytes(record)
                    raw_readable = self.kv_object_record_raw_readable(record)

        logger.info(
            "DISK_CONTAINS_PROFILE op=%s status=miss reason=%s index=%d "
            "hit_count=%d key=%s dict_size=%d in_put=%s path_exists=%s "
            "raw_enabled=%s object_state=%s raw_extents=%d "
            "raw_readable=%s raw_read_bytes=%d",
            op,
            reason,
            index,
            hit_count,
            key.to_string(),
            len(self.dict),
            self.exists_in_put_tasks(key),
            os.path.exists(self._key_to_path(key)),
            self.kv_object_store_enabled and self.kv_object_tutti_raw_enabled,
            object_state,
            raw_extents,
            raw_readable,
            raw_read_bytes,
        )

    def _log_kv_object_write_skip(
        self,
        key: CacheEngineKey,
        *,
        reason: str,
        bytes_len: int,
    ) -> None:
        """Log bounded diagnostics when object-store writes fall back to files."""
        log_count = self._object_write_skip_log_counts.get(reason, 0)
        if log_count >= 32:
            return
        self._object_write_skip_log_counts[reason] = log_count + 1
        logger.info(
            "KV_OBJECT_STORE_PROFILE op=write key=%s status=skip "
            "reason=%s bytes=%d raw_enabled=%s writer_installed=%s",
            key.to_string(),
            reason,
            bytes_len,
            self.kv_object_tutti_raw_enabled,
            self.kv_object_tutti_raw_writer is not None,
        )

    def scan_existing_entries(self, metadata: LMCacheMetadata) -> int:
        """Scan the disk cache directory and register pre-existing ``.pt`` files.

        Intended for use before GPU-direct (Tutti) NVMe bind: once the drive
        is bound the filesystem goes EIO, so metadata must be available in
        ``self.dict`` before that point.

        File names follow the pattern produced by ``_key_to_path()``:
        ``<model_name_with_hyphens>@<world_size>@<worker_id>@<hash>@<dtype>.pt``

        Only files whose names start with the expected model-name prefix are
        registered; others are silently skipped.

        Args:
            metadata: Engine metadata used to recover model name, shape, and
                dtype for each registered entry.

        Returns:
            Number of entries successfully registered.
        """
        from lmcache.utils import parse_cache_key

        mangled_model = metadata.model_name.replace("/", "-")
        prefix = mangled_model + "@"
        n_recovered = 0

        try:
            filenames = os.listdir(self.path)
        except OSError as exc:
            logger.warning("scan_existing_entries: cannot list %s: %s", self.path, exc)
            return 0

        # Compute canonical per-group shapes when KV layer groups are
        # configured.  These are used for newly registered entries so that
        # get_blocking allocates the correct buffer size.  For DSv4-optimised
        # files the retrieve path further applies a shapes_per_key override
        # (see StorageManager.batched_get) to limit reads to prefix groups.
        klg_manager = metadata.kv_layer_groups_manager
        has_kv_groups = klg_manager is not None and bool(klg_manager.kv_layer_groups)
        canonical_shapes: Optional[list[torch.Size]] = None
        canonical_dtypes: Optional[list[torch.dtype]] = None
        if has_kv_groups:
            canonical_shapes = metadata.get_shapes(metadata.chunk_size)
            canonical_dtypes = metadata.get_dtypes()
        recovered_fmt = (
            MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        )

        with self.disk_lock:
            for fname in filenames:
                if not fname.endswith(".pt"):
                    continue
                if not fname.startswith(prefix):
                    continue
                key_str = metadata.model_name + "@" + fname[len(prefix) : -3]
                try:
                    key = parse_cache_key(key_str)
                except Exception:
                    continue
                if key in self.dict:
                    continue  # already registered by a live write
                fpath = os.path.join(self.path, fname)
                try:
                    fsize = os.path.getsize(fpath)
                except OSError:
                    continue
                self.dict[key] = DiskCacheMetadata(
                    path=fpath,
                    size=fsize,
                    shape=torch.Size(metadata.kv_shape),
                    dtype=metadata.kv_dtype,
                    fmt=recovered_fmt,
                    shapes=canonical_shapes,
                    dtypes=canonical_dtypes,
                )
                n_recovered += 1

        logger.info(
            "scan_existing_entries: recovered %d entries from %s",
            n_recovered,
            self.path,
        )
        return n_recovered

    def write_file(self, buffer: memoryview, path: str) -> bool:
        """Write a serialized KV chunk to a filesystem path.

        Args:
            buffer: Serialized KV bytes.
            path: Destination path.

        Returns:
            True if the write completed, False otherwise.
        """
        start_time = time.time()
        size = len(buffer)

        def _write_once() -> None:
            if size % self.os_disk_bs != 0 or not self.use_odirect:
                with open(path, "wb") as f:
                    f.write(buffer)
            else:
                fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
                try:
                    os.write(fd, buffer)
                finally:
                    os.close(fd)

        try:
            _write_once()
        except FileNotFoundError:
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            try:
                _write_once()
            except OSError as exc:
                logger.warning(
                    "write_file failed (filesystem EIO or permission error) "
                    "path=%s error=%s; write skipped (Tutti read-only mode?)",
                    path,
                    exc,
                )
                return False
        except OSError as exc:
            # Filesystem may be EIO when Tutti (snvme) has exclusive NVMe
            # control.  Log a warning and skip the write rather than crashing.
            logger.warning(
                "write_file failed (filesystem EIO or permission error) "
                "path=%s error=%s; write skipped (Tutti read-only mode?)",
                path,
                exc,
            )
            return False
        disk_write_time = time.time() - start_time
        logger.debug(
            f"Disk write size: {size} bytes, "
            f"Bandwidth: {size / disk_write_time / 1e6:.2f} MB/s"
        )
        return True

    def read_file(
        self,
        key: CacheEngineKey,
        buffer: memoryview,
        path: str,
    ) -> None:
        """Read a serialized KV chunk into an allocator-owned buffer.

        When direct I/O is enabled, the block-aligned prefix is read with
        ``O_DIRECT`` and only a possible final partial block uses buffered
        I/O. This keeps GLM KV chunks on the physical SSD path even when their
        logical byte size is not a multiple of the filesystem block size.

        Args:
            key: Cache key used to remove stale metadata on a missing file.
            buffer: Destination byte buffer.
            path: Source filesystem path.
        """
        start_time = time.time()
        size = len(buffer)
        direct_size = size - (size % self.os_disk_bs)

        try:
            if not self.use_odirect or direct_size == 0:
                with open(path, "rb") as f:
                    f.readinto(buffer)
            else:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    direct_read = fdo.readinto(buffer[:direct_size])
                if direct_read != direct_size:
                    raise OSError(
                        f"short O_DIRECT read for {path}: "
                        f"expected {direct_size}, got {direct_read}"
                    )
                if direct_size < size:
                    with open(path, "rb", buffering=0) as tail_file:
                        tail_file.seek(direct_size)
                        tail_read = tail_file.readinto(buffer[direct_size:])
                    expected_tail = size - direct_size
                    if tail_read != expected_tail:
                        raise OSError(
                            f"short buffered tail read for {path}: "
                            f"expected {expected_tail}, got {tail_read}"
                        )
        except FileNotFoundError:
            logger.warning(f"File not found on disk: {path}")
            if self.dict.get(key, None):
                self.dict.pop(key)
            return

        disk_read_time = time.time() - start_time
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )

    def get_allocator_backend(self) -> LocalCPUBackend:
        return self.local_cpu_backend

    def close(self) -> None:
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()
        if self._cpu_raw_stage_queue is not None:
            self._cpu_raw_stage_queue.close(wait=False)
        self._hca_write_executor.shutdown(wait=False)
        self._layer_major_pack_executor.shutdown(wait=False)
        self.disk_worker.close()
