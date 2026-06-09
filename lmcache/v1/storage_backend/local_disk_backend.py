# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
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

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            if key not in self.dict:
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
            mem_objs.append(self._get_blocking_with_shapes(key, shapes_override=override))
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
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        return num_hit_counts

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

        # TODO(Jiayi): need to add ref count in disk memory object
        self.write_file(buffer, path)

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
                    fmt=MemoryFormat.KV_2LTD,
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

    def write_file(self, buffer, path):
        start_time = time.time()
        size = len(buffer)
        try:
            if size % self.os_disk_bs != 0 or not self.use_odirect:
                with open(path, "wb") as f:
                    f.write(buffer)
            else:
                fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
                os.write(fd, buffer)
                os.close(fd)
        except OSError as exc:
            # Filesystem may be EIO when Tutti (snvme) has exclusive NVMe
            # control.  Log a warning and skip the write rather than crashing.
            logger.warning(
                "write_file failed (filesystem EIO or permission error) "
                "path=%s error=%s; write skipped (Tutti read-only mode?)",
                path,
                exc,
            )
            return
        disk_write_time = time.time() - start_time
        logger.debug(
            f"Disk write size: {size} bytes, "
            f"Bandwidth: {size / disk_write_time / 1e6:.2f} MB/s"
        )

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
