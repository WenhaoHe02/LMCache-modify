# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
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
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.path_sharder import PathSharder

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

KVObjectRawWriter = Callable[
    [KVObjectRecord, memoryview],
    tuple[Sequence[tuple[int, int, int]], float],
]


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


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
        self.kv_object_tutti_raw_enabled = _env_flag(
            "LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE"
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
        if self.kv_object_store_enabled:
            slot_mb = int(os.getenv("LMCACHE_KV_OBJECT_STORE_SLOT_MB", "128"))
            capacity = int(os.getenv("LMCACHE_KV_OBJECT_STORE_CAPACITY", "2048"))
            if config.extra_config is not None:
                slot_mb = int(
                    config.extra_config.get("kv_object_store_slot_mb", slot_mb)
                )
                capacity = int(
                    config.extra_config.get("kv_object_store_capacity", capacity)
                )
                self.kv_object_tutti_raw_enabled = bool(
                    config.extra_config.get(
                        "kv_object_store_tutti_raw_enable",
                        self.kv_object_tutti_raw_enabled,
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
                                item.strip()
                                for item in raw_region_text.split(",")
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
                materialize_file=not self.kv_object_tutti_raw_enabled,
            )
            self.kv_object_metadata_store = KVObjectMetadataStore()
            self.kv_object_pool_io = KVObjectPoolIO({pool_id: pool_path})
            logger.info(
                "KV object store enabled: pool_id=%s path=%s slot_mb=%d "
                "capacity=%d tutti_raw=%s raw_base_lba=%d",
                pool_id,
                pool_path,
                slot_mb,
                capacity,
                self.kv_object_tutti_raw_enabled,
                self.kv_object_tutti_raw_base_lba,
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
        object_ids = [
            self._key_to_object_id(
                key,
                layer_id=layer_ids[index] if layer_ids is not None else 0,
                role=roles[index] if roles is not None else "full",
            )
            for index, key in enumerate(keys)
        ]
        return self.kv_object_metadata_store.get_many(object_ids, ready_only=True)

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

    def reset_kv_object_pool_allocation(self) -> None:
        """Reset logical KV object allocation offsets for future writes."""
        if self.kv_object_pool_layout is not None:
            self.kv_object_pool_layout.reset_allocation()

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
            for _file_offset, _slba, n_sectors in self._raw_extents_for_record(
                record
            )
        )
        return covered_bytes == expected_bytes

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
            is_tail_range = (
                byte_range.target_offset + byte_range.length == record.length
            )
            if byte_range.offset % 512 != 0:
                return 0
            if range_dma_length != byte_range.length and not is_tail_range:
                return 0
            expected_bytes += range_dma_length
        return expected_bytes

    def _raw_extents_for_record(
        self,
        record: KVObjectRecord,
    ) -> list[tuple[int, int, int]]:
        """Return raw extents covering the record's logical read ranges."""
        extents: list[tuple[int, int, int]] = []
        for byte_range in record.read_ranges:
            range_length = byte_range.length
            range_dma_length = ((range_length + 511) // 512) * 512
            is_tail_range = (
                byte_range.target_offset + range_length == record.length
            )
            if byte_range.offset % 512 != 0 or (
                range_dma_length != range_length and not is_tail_range
            ):
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
            for file_offset, slba, n_sectors in record.raw_extents:
                extent_start = file_offset
                extent_end = file_offset + n_sectors * 512
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

        layer_views = self._object_layer_view_specs(memory_obj, group_ranges)
        indexed = 0
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

    def _object_group_ranges(
        self,
        memory_obj: MemoryObj,
        full_record: KVObjectRecord,
    ) -> list[tuple[int, KVObjectByteRange]]:
        """Derive per-group byte ranges for one stored chunk."""
        shapes = memory_obj.metadata.shapes
        dtypes = memory_obj.metadata.dtypes
        if shapes is None or dtypes is None or len(shapes) != len(dtypes):
            return []
        ranges: list[tuple[int, KVObjectByteRange]] = []
        target_offset = 0
        current_offset = full_record.offset
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
                return False
            if not self._has_readable_chunk_object_unlocked(key):
                return False
            if pin:
                self.dict[key].pin()
                # vllm lookup sets pin to True
                self.keys_in_request.append(key)
            return True

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
            for key in keys:
                if key not in self.dict:
                    return num_hit_counts
                if not self._has_readable_chunk_object_unlocked(key):
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
        object_id = self._key_to_object_id(key)
        record = self.kv_object_metadata_store.get(object_id)
        if record is None or record.state != KVObjectState.READY:
            return False
        if not record.raw_extents:
            return False
        return self.kv_object_record_raw_readable(record)

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
            if self.kv_object_tutti_raw_enabled
            else None
        )
        if self.kv_object_tutti_raw_enabled and raw_writer is None:
            return False
        try:
            with self.kv_object_store_lock:
                existing = self.kv_object_metadata_store.get(object_id)
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
                key_str = metadata.model_name + "@" + fname[len(prefix):-3]
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

    def read_file(self, key, buffer, path):
        start_time = time.time()
        size = len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0
        if not fblock_aligned and self.use_odirect:
            logger.warning(
                "Cannot use O_DIRECT for this file, "
                "size is not aligned to disk block size."
            )

        try:
            if not fblock_aligned or not self.use_odirect:
                with open(path, "rb") as f:
                    f.readinto(buffer)
            else:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    fdo.readinto(buffer)
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
        self.disk_worker.close()
