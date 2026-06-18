# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.health_monitor.base import HealthMonitor

# Standard
import asyncio
import gc
import multiprocessing
import os
import subprocess
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    _lmcache_nvtx_annotate,
    compress_slot_mapping,
    convert_tokens_to_list,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.tutti_direct_loader import TuttiDirectLoader
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.kv_object_store import KVObjectByteRange, KVObjectRecord
from lmcache.v1.memory_management import CuFileMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import (  # noqa: E501
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.system_detection import NUMADetector, NUMAMapping
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    SegmentTokenDatabase,
    TokenDatabase,
)

logger = init_logger(__name__)

# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _maybe_unmount_for_tutti(cache_path: str) -> Optional[Tuple[str, str]]:
    """Unmount a local cache filesystem before snvme binds its NVMe device.

    Tutti's SNVM_DEVICE_BIND detaches the in-tree nvme driver. Keeping ext4
    mounted while doing that causes EIO and failed snvme attach. This helper is
    called after FIEMAP has captured LBAs and before creating TuttiDirectLoader.

    Args:
        cache_path: LocalDiskBackend cache directory.

    Returns:
        ``(source, mount_point)`` if a local NVMe filesystem was unmounted,
        otherwise ``None``.
    """
    try:
        mount_result = subprocess.run(
            ["findmnt", "-nr", "-T", cache_path, "-o", "SOURCE,TARGET"],
            check=False,
            capture_output=True,
            text=True,
        )
        source, mount_point = mount_result.stdout.strip().splitlines()[0].split()
    except (IndexError, OSError) as exc:
        logger.warning("Tutti could not resolve mount for %s: %s", cache_path, exc)
        return None

    if not mount_point.startswith("/mnt/nvme"):
        logger.info(
            "Tutti leaves non-local cache mount active: path=%s mount=%s",
            cache_path,
            mount_point,
        )
        return None

    logger.info(
        "Tutti unmounting cache filesystem before snvme bind: path=%s source=%s "
        "mount=%s",
        cache_path,
        source,
        mount_point,
    )
    try:
        # Use lazy umount (-l) so that open file descriptors (e.g. async pool
        # writers still holding the file) do not cause EBUSY.  The mount point
        # is detached from the namespace immediately; the underlying block
        # device is released once all open fds to that filesystem are closed,
        # which is sufficient for snvme DEVICE_BIND to proceed.
        subprocess.run(["umount", "-l", mount_point], check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "Tutti umount -l failed for %s (errno %s); "
            "snvme bind may fail or fall back to CPU path",
            mount_point,
            exc.returncode,
        )
        return None
    return source, mount_point


def _maybe_remount_after_tutti_failure(
    mount_info: Optional[Tuple[str, str]],
) -> bool:
    """Remount a filesystem that was unmounted before a failed Tutti bind.

    Tries 'mount -t auto source mount_point' first, then falls back to
    'mount source mount_point'.  Containers running without a full fstab
    need the device path and filesystem type supplied explicitly.
    """
    if mount_info is None:
        return False
    source, mount_point = mount_info
    cmds = [
        ["mount", "-t", "auto", source, mount_point],
        ["mount", source, mount_point],
    ]
    for cmd in cmds:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(
                "Tutti remounted cache filesystem after init failure: "
                "source=%s mount=%s cmd=%s",
                source,
                mount_point,
                cmd,
            )
            return True
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning(
                "Tutti remount attempt failed: source=%s mount=%s "
                "cmd=%s error=%s",
                source,
                mount_point,
                cmd,
                exc,
            )
    return False


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return False


class CacheEngineEndSignal:
    pass


class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.token_database = token_database
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        self.dsv4_optimized_kv = (
            _as_bool(config.get_extra_config_value("dsv4_optimized_kv", False))
            or _env_flag("LMCACHE_DSV4_OPTIMIZED_KV")
        )
        self.dsv4_optimized_tail_tokens = max(
            int(
                config.get_extra_config_value(
                    "dsv4_optimized_tail_tokens",
                    os.getenv("LMCACHE_DSV4_OPTIMIZED_TAIL_TOKENS", config.chunk_size),
                )
            ),
            config.chunk_size,
        )
        # save_only_first_rank only works when use mla
        self.save_only_first_rank = (
            self.config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )

        if self.save_only_first_rank and self.gpu_connector is not None:
            self.broadcast_stream = (
                self.gpu_connector.load_stream
                if hasattr(self.gpu_connector, "load_stream")
                else torch_dev.Stream()
            )

        self.enable_controller = config.enable_controller

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        lmcache_worker_ids = config.get_lmcache_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        # lmcache_worker_ids is empty means start on all workers
        if (
            self.enable_controller
            and self.metadata.role != "scheduler"
            and (not lmcache_worker_ids or metadata.worker_id in lmcache_worker_ids)
        ):
            self.lmcache_worker = LMCacheWorker(config, metadata, self)
        else:
            self.lmcache_worker = None
            logger.info(
                "LMCacheWorker is not initialized (related configs: "
                "enable_controller: %s, role: %s, worker_id: %s, worker_ids: %s).",
                self.enable_controller,
                self.metadata.role,
                self.metadata.worker_id,
                lmcache_worker_ids,
            )

        self.async_loading = config.enable_async_loading
        self.event_manager = EventManager()

        self.use_layerwise = config.use_layerwise

        # TODO: support save_only_first_rank when use layerwise
        # if use_layerwise is True, all ranks will initialize the storage_manager
        # if save_only_first_rank is False, all ranks will initialize
        # the storage_manager
        # if save_only_first_rank is True, only the first rank and
        # lookup server workers will initialize the storage_manager
        self.storage_manager: Optional[StorageManager] = None

        # KV events
        self.kv_events_enabled = False
        self.kv_events_enabled = config.enable_kv_events
        if self.kv_events_enabled:
            self.kv_events: List[CacheStoreEvent] = []
            logger.info("KV events are enabled.")
        else:
            logger.info("KV events are disabled.")

        # HACK: remove this in the future
        # NOTE (Jiayi): This is currently used to support
        # dropping the kv cache from the buffer in PD backend
        # at decoder.
        self.remove_after_retrieve = config.enable_pd and config.pd_role == "receiver"

        # asymmetric store/retrieve location can be specified
        # this is typically used (but not limited) in PD system
        self.store_location = config.store_location
        self.retrieve_locations = config.retrieve_locations

        self.num_layers = metadata.kv_shape[0]
        self.fmt = None
        if self.use_layerwise:
            if metadata.use_mla:
                self.fmt = MemoryFormat.KV_MLA_FMT
            elif config.enable_blending:
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D
        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT

        # NOTE(ApostaC): we haven't support lookup-cache yet
        self.lookup_cache: dict[CacheEngineKey, Any] = {}

        # lookup_id -> {location -> [pinned keys]}
        self.lookup_pins: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        InitializeUsageContext(config, metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        # Initialize PinMonitor singleton with config
        PinMonitor.GetOrCreate(config)

        self.post_inited = False

        # Flag to control KVCache Check logging (can be toggled via API)
        self.kvcache_check_log_enabled = False

        gc.collect()
        if not config.py_enable_gc:
            gc.disable()

        # Health monitor reference (injected by LMCacheManager)
        self._health_monitor: Optional["HealthMonitor"] = None

        # Flag to indicate if initialization failed (irrecoverable error)
        self._init_failed = False

        # Optional GPU-direct NVMe loader (Tutti fast path for LocalDiskBackend)
        self._tutti_loader: Optional[TuttiDirectLoader] = None
        self._tutti_config: Optional[dict[str, Any]] = None
        self._tutti_init_failed = False
        self._tutti_can_cpu_fallback = True
        self._tutti_warmup_lock = threading.Lock()
        self._tutti_warmup_started = False
        self._tutti_warmup_done: Optional[threading.Event] = None
        self._tutti_store_warmup_keys: dict[str, set[CacheEngineKey]] = {}
        self._tutti_store_warmup_pending: dict[str, set[CacheEngineKey]] = {}
        self._tutti_store_warmup_last_seen: set[str] = set()

    def set_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """
        Set the health monitor reference.

        This is called by LMCacheManager after creating the HealthMonitor
        to inject the reference into the engine.

        Args:
            health_monitor: The HealthMonitor instance from LMCacheManager
        """
        self._health_monitor = health_monitor

    def is_healthy(self) -> bool:
        """
        Check if the LMCache system is healthy.

        This method returns False if:
        - Initialization failed (irrecoverable error)
        - HealthMonitor reports unhealthy

        If no health monitor is set and initialization succeeded,
        it returns True (assume healthy).

        Returns:
            bool: True if healthy, False otherwise
        """
        if self._init_failed:
            return False
        if self._health_monitor is not None:
            return self._health_monitor.is_healthy()
        return True

    def _get_req_id(self, kwargs: dict) -> str:
        """Extracts request ID from kwargs for logging."""
        return kwargs.get("req_id", "unspecified")

    def _prepare_gpu_connector_layout(self, **kwargs: Any) -> None:
        """Let the GPU connector discover KV layout before memory allocation."""
        if self.gpu_connector is None:
            return
        register = getattr(self.gpu_connector, "register_kv_caches", None)
        if callable(register) and "kvcaches" in kwargs:
            register(kwargs["kvcaches"])

    def mark_init_failed(self, reason: str = "") -> None:
        """
        Mark the engine as having failed initialization.

        This is called by LMCacheManager when an irrecoverable error occurs
        during initialization or post_init. Once marked, is_healthy() will
        always return False, causing the system to fall back to recomputation.

        Args:
            reason: Optional reason string for logging
        """
        self._init_failed = True
        if reason:
            logger.error("LMCacheEngine marked as init failed: %s", reason)
        else:
            logger.error("LMCacheEngine marked as init failed")

    def post_init(self, **kwargs) -> None:
        if not self.post_inited:
            logger.info("Post initializing LMCacheEngine")
            lookup_server_worker_ids = self.config.get_lookup_server_worker_ids(
                self.metadata.use_mla, self.metadata.world_size
            )
            if (
                self.lmcache_worker is not None
                or self.use_layerwise
                or not self.save_only_first_rank
                or self.metadata.is_first_rank()
                or len(lookup_server_worker_ids) == 0
                or self.metadata.worker_id in lookup_server_worker_ids
            ):
                logger.info(
                    f"Initialize storage manager on rank {self.metadata.worker_id}, "
                    f"use layerwise: {self.use_layerwise},"
                    f"save only first rank: {self.save_only_first_rank}"
                )
                async_lookup_server = kwargs.get("async_lookup_server", None)
                self.storage_manager = StorageManager(
                    self.config,
                    self.metadata,
                    event_manager=self.event_manager,
                    lmcache_worker=self.lmcache_worker,
                    async_lookup_server=async_lookup_server,
                )
                self._maybe_init_tutti_loader()
            self.post_inited = True

    def _maybe_init_tutti_loader(self) -> None:
        """Initialise TuttiDirectLoader if tutti_device_path is configured.

        Reads extra_config keys:
          - ``tutti_device_path``  (required, e.g. "/dev/ssnvme0")
          - ``tutti_ctrl_path``    (default "/dev/snvm_control")
          - ``tutti_pci_bdfs``     CSV of BDFs ordered by worker rank, e.g.
                                   "0000:6f:00.0,0000:10:00.0,..."
                                   Worker N uses bdfs[local_worker_id % len(bdfs)].
                                   Use "skip" for workers whose disk is not a
                                   local PCIe NVMe (e.g. NVMe-oF fabric devices).
                                   Alias ``tutti_pci_bdf`` (singular) accepted for
                                   single-drive / TP=1 setups.
          - ``tutti_n_slots``      (default 16)
          - ``tutti_slot_mb``      (default 32, per-slot MiB)
          - ``tutti_nsid``         (default 1)
        """
        device_path: Optional[str] = self.config.get_extra_config_value(
            "tutti_device_path", None
        )
        if device_path is None:
            return

        ctrl_path: str = self.config.get_extra_config_value(
            "tutti_ctrl_path", "/dev/snvm_control"
        )

        # Support both tutti_pci_bdfs (CSV for TP>1) and tutti_pci_bdf (singular).
        bdfs_csv: Optional[str] = self.config.get_extra_config_value(
            "tutti_pci_bdfs", None
        )
        if bdfs_csv is None:
            bdfs_csv = self.config.get_extra_config_value("tutti_pci_bdf", None)
        if bdfs_csv is None:
            logger.warning(
                "tutti_device_path set but tutti_pci_bdfs missing; "
                "Tutti fast path disabled"
            )
            return

        bdf_list = [b.strip() for b in bdfs_csv.split(",")]
        worker_idx: int = self.metadata.local_worker_id
        pci_bdf: str = bdf_list[worker_idx % len(bdf_list)]

        if pci_bdf.lower() in ("skip", "none", ""):
            logger.info(
                "Worker %d: Tutti disabled (BDF='%s' in tutti_pci_bdfs list)",
                worker_idx,
                pci_bdf,
            )
            return

        n_slots: int = int(
            self.config.get_extra_config_value("tutti_n_slots", 16)
        )
        slot_mb: int = int(
            self.config.get_extra_config_value("tutti_slot_mb", 32)
        )
        nsid: int = int(self.config.get_extra_config_value("tutti_nsid", 1))
        cuda_device: int = worker_idx

        self._tutti_config = {
            "device_path": device_path,
            "ctrl_path": ctrl_path,
            "pci_bdf": pci_bdf,
            "n_slots": n_slots,
            "slot_mb": slot_mb,
            "nsid": nsid,
            "cuda_device": cuda_device,
        }
        delay_s = float(
            self.config.get_extra_config_value(
                "tutti_startup_warmup_delay_sec",
                os.getenv("LMCACHE_TUTTI_STARTUP_WARMUP_DELAY_SEC", 0),
            )
        )
        logger.info(
            "TuttiDirectLoader configured: worker=%d device=%s pci=%s "
            "cuda:%d slots=%dx%dMiB startup_warmup_delay_s=%.0f",
            worker_idx,
            device_path,
            pci_bdf,
            cuda_device,
            n_slots,
            slot_mb,
            delay_s,
        )
        health_port = int(
            self.config.get_extra_config_value(
                "tutti_startup_health_port",
                os.getenv("LMCACHE_TUTTI_STARTUP_HEALTH_PORT", 8000),
            )
        )
        health_poll_interval_s = float(
            self.config.get_extra_config_value(
                "tutti_startup_health_poll_interval_sec",
                os.getenv("LMCACHE_TUTTI_STARTUP_HEALTH_POLL_INTERVAL_SEC", 10),
            )
        )
        health_poll_timeout_s = float(
            self.config.get_extra_config_value(
                "tutti_startup_health_poll_timeout_sec",
                os.getenv("LMCACHE_TUTTI_STARTUP_HEALTH_POLL_TIMEOUT_SEC", 1200),
            )
        )
        if delay_s > 0:
            # Spawn a background daemon thread that sleeps for delay_s seconds
            # before initializing Tutti.  This defers the NVMe FS unmount until
            # after the vLLM API server has finished loading the tokenizer from
            # the NVMe-backed model directory.  Without the delay, the worker
            # process would unmount /mnt/nvme0 while the API server process
            # (sharing the same Docker mount namespace) is still reading from it.
            #
            # After the fixed delay, the thread additionally polls the vLLM
            # /health endpoint (if health_poll_timeout_s > 0) so that the unmount
            # does not happen until the server is fully ready even when the warmup
            # compilation takes longer than expected.
            def _startup_warmup() -> None:
                logger.info(
                    "Tutti startup warmup thread sleeping %.1f s (worker=%d)",
                    delay_s,
                    worker_idx,
                )
                time.sleep(delay_s)
                logger.info(
                    "Tutti startup warmup thread woke up after sleep (worker=%d)",
                    worker_idx,
                )
                if health_poll_timeout_s > 0 and health_port > 0:
                    import urllib.request
                    url = f"http://127.0.0.1:{health_port}/health"
                    deadline = time.monotonic() + health_poll_timeout_s
                    ready = False
                    while time.monotonic() < deadline:
                        try:
                            urllib.request.urlopen(url, timeout=3)
                            ready = True
                            break
                        except Exception:
                            remaining = deadline - time.monotonic()
                            logger.info(
                                "Tutti startup warmup: server not ready yet, "
                                "retrying in %.0f s (%.0f s remaining, worker=%d)",
                                health_poll_interval_s,
                                remaining,
                                worker_idx,
                            )
                            time.sleep(health_poll_interval_s)
                    if ready:
                        logger.info(
                            "Tutti startup warmup: server ready, initializing (worker=%d)",
                            worker_idx,
                        )
                    else:
                        logger.warning(
                            "Tutti startup warmup: server health poll timed out "
                            "after %.0f s, proceeding anyway (worker=%d)",
                            health_poll_timeout_s,
                            worker_idx,
                        )
                else:
                    logger.info(
                        "Tutti startup warmup thread woke up, initializing (worker=%d)",
                        worker_idx,
                    )
                self._ensure_tutti_loader(wait_for_warmup=False)

            thread = threading.Thread(
                target=_startup_warmup,
                daemon=True,
                name=f"tutti-startup-warmup-{worker_idx}",
            )
            thread.start()

    def _ensure_tutti_loader(
        self,
        keys: Optional[List[CacheEngineKey]] = None,
        wait_for_warmup: bool = True,
    ) -> bool:
        """Initialise TuttiDirectLoader on the first LocalDiskBackend hit.

        DSv4 long-context cold prefill has narrow HBM headroom. Deferring
        Tutti staging allocation keeps cold/store requests on the same memory
        path as the proven non-Tutti baseline; the loader is created only when
        a disk hit can actually use it.

        Returns:
            True if the loader is available, False otherwise.
        """
        if self._tutti_loader is not None:
            return True
        if os.getenv("LMCACHE_TUTTI_FORCE_CPU_FALLBACK", "0") == "1":
            logger.info(
                "Tutti direct load disabled by LMCACHE_TUTTI_FORCE_CPU_FALLBACK"
            )
            return False
        if self._tutti_config is None or self._tutti_init_failed:
            return False

        warmup_done: Optional[threading.Event] = None
        with self._tutti_warmup_lock:
            if self._tutti_loader is not None:
                return True
            if self._tutti_config is None or self._tutti_init_failed:
                return False
            if (
                wait_for_warmup
                and self._tutti_warmup_started
                and self._tutti_warmup_done is not None
                and not self._tutti_warmup_done.is_set()
            ):
                warmup_done = self._tutti_warmup_done
            else:
                return self._ensure_tutti_loader_locked(keys)

        logger.info("Waiting for in-flight Tutti warmup to finish")
        warmup_done.wait()
        return self._tutti_loader is not None

    def _ensure_tutti_loader_locked(
        self,
        keys: Optional[List[CacheEngineKey]] = None,
    ) -> bool:
        """Initialise TuttiDirectLoader while holding _tutti_warmup_lock."""
        from lmcache.v1.gpu_connector.tutti_direct_loader import (
            FiemapHelper,
            LbaRecord,
        )
        from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

        profile_start = time.perf_counter()
        initial_lba_cache: dict = {}
        debug_expected_checksums: dict[str, tuple[int, str]] = {}
        mount_info: Optional[Tuple[str, str]] = None
        recover_ms = 0.0
        collect_paths_ms = 0.0
        fiemap_ms = 0.0
        unmount_ms = 0.0
        disk_backend = (
            self.storage_manager.storage_backends.get("LocalDiskBackend")
            if self.storage_manager is not None
            else None
        )
        if isinstance(disk_backend, LocalDiskBackend):
            recover_start = time.perf_counter()
            n_recovered = disk_backend.scan_existing_entries(self.metadata)
            recover_ms = (time.perf_counter() - recover_start) * 1000.0
            collect_start = time.perf_counter()
            with disk_backend.disk_lock:
                required_paths = [
                    disk_backend.dict[key].path
                    for key in keys or []
                    if key in disk_backend.dict
                    and disk_backend.dict[key] is not None
                ]
                if required_paths:
                    paths = list(dict.fromkeys(required_paths))
                else:
                    paths = [
                        meta.path
                        for meta in disk_backend.dict.values()
                        if meta is not None
                    ]
                object_pool_paths = [
                    str(path)
                    for path in disk_backend.get_kv_object_pool_paths().values()
                ]
                paths.extend(object_pool_paths)
                paths = list(dict.fromkeys(paths))
            collect_paths_ms = (time.perf_counter() - collect_start) * 1000.0
            fiemap_start = time.perf_counter()
            initial_lba_cache = FiemapHelper.scan_paths(paths)
            fiemap_ms = (time.perf_counter() - fiemap_start) * 1000.0
            if (
                disk_backend.kv_object_tutti_raw_enabled
                and disk_backend.kv_object_tutti_raw_region_path
            ):
                raw_region_start = time.perf_counter()
                raw_region_extents = FiemapHelper.query_extents(
                    disk_backend.kv_object_tutti_raw_region_path
                )
                disk_backend.set_kv_object_tutti_raw_region_extents(
                    [
                        (
                            extent.file_offset,
                            extent.slba,
                            extent.n_sectors,
                        )
                        for extent in raw_region_extents
                    ]
                )
                logger.info(
                    "TUTTI_PROFILE raw_region path=%s extents=%d size_mb=%.3f "
                    "scan_ms=%.3f",
                    disk_backend.kv_object_tutti_raw_region_path,
                    len(raw_region_extents),
                    sum(
                        extent.n_sectors * 512
                        for extent in raw_region_extents
                    )
                    / 1024**2,
                    (time.perf_counter() - raw_region_start) * 1000.0,
                )
            missing_required_paths = [
                path for path in required_paths if path not in initial_lba_cache
            ]
            if missing_required_paths:
                logger.warning(
                    "Tutti LBA pre-scan missed %d/%d requested files before "
                    "snvme bind; using CPU filesystem path for this request. "
                    "first_missing=%s",
                    len(missing_required_paths),
                    len(required_paths),
                    missing_required_paths[0],
                )
                return False
            if os.getenv("LMCACHE_TUTTI_DEBUG_CHECKSUM", "0") == "1":
                import hashlib

                checksum_limit = int(
                    os.getenv("LMCACHE_TUTTI_DEBUG_CHECKSUM_LIMIT", "4")
                )
                debug_start = time.perf_counter()
                for key in (keys or [])[:checksum_limit]:
                    disk_meta = disk_backend.dict.get(key)
                    if disk_meta is None:
                        continue
                    shapes = self.metadata.get_shapes(self.metadata.chunk_size)
                    dtypes = self.metadata.get_dtypes()
                    shapes_override = self._dsv4_store_shapes_for_range(
                        shapes,
                        dtypes,
                        0,
                        self.metadata.chunk_size,
                        self.metadata.chunk_size * len(keys or []),
                    )
                    nbytes = sum(
                        shape.numel() * dtype.itemsize
                        for shape, dtype in zip(shapes_override, dtypes, strict=True)
                    )
                    with open(disk_meta.path, "rb") as handle:
                        data = handle.read(nbytes)
                    debug_expected_checksums[disk_meta.path] = (
                        nbytes,
                        hashlib.sha256(data).hexdigest(),
                    )
                logger.info(
                    "TUTTI_DEBUG_CHECKSUM prepared count=%d ms=%.3f",
                    len(debug_expected_checksums),
                    (time.perf_counter() - debug_start) * 1000.0,
                )
            logger.info(
                "Tutti lazy pre-scan: %d recovered, %d files scanned, "
                "%d LBAs cached",
                n_recovered,
                len(paths),
                len(initial_lba_cache),
            )
        cfg = self._tutti_config
        try:
            if isinstance(disk_backend, LocalDiskBackend):
                unmount_start = time.perf_counter()
                mount_info = _maybe_unmount_for_tutti(disk_backend.path)
                unmount_ms = (time.perf_counter() - unmount_start) * 1000.0
            # Release PyTorch's cached GPU memory so the raw cudaMalloc inside
            # TuttiDirectLoader.create() can find contiguous free HBM pages.
            torch.cuda.empty_cache()
            create_start = time.perf_counter()
            self._tutti_loader = TuttiDirectLoader.create(
                device_path=cfg["device_path"],
                ctrl_path=cfg["ctrl_path"],
                pci_bdf=cfg["pci_bdf"],
                n_slots=cfg["n_slots"],
                slot_bytes=cfg["slot_mb"] * 1024 * 1024,
                nsid=cfg["nsid"],
                cuda_device=cfg["cuda_device"],
                initial_lba_cache=initial_lba_cache,
                debug_expected_checksums=debug_expected_checksums,
            )
            create_ms = (time.perf_counter() - create_start) * 1000.0
            if (
                isinstance(disk_backend, LocalDiskBackend)
                and disk_backend.kv_object_tutti_raw_enabled
            ):
                if (
                    disk_backend.kv_object_tutti_raw_base_lba <= 0
                    and not disk_backend.kv_object_tutti_raw_region_extents
                ):
                    logger.warning(
                        "KV object Tutti raw store requested but "
                        "kv_object_store_tutti_raw_base_lba and region_extents "
                        "both zero; raw cold-store writer not installed"
                    )
                else:

                    def _write_raw_object(
                        record: Any,
                        buffer: memoryview,
                    ) -> tuple[list[tuple[int, int, int]], float]:
                        if self._tutti_loader is None:
                            raise RuntimeError("Tutti loader is not initialised")
                        if record.offset % 512 != 0:
                            raise ValueError(
                                "KV object raw offset must be 512-byte aligned"
                            )
                        write_start = time.perf_counter()
                        if disk_backend.kv_object_tutti_raw_region_extents:
                            raw_records = self._tutti_loader.store_bytes_to_raw_extents(
                                buffer,
                                raw_extents=[
                                    LbaRecord(
                                        file_offset=file_offset,
                                        slba=slba,
                                        n_sectors=n_sectors,
                                    )
                                    for (
                                        file_offset,
                                        slba,
                                        n_sectors,
                                    ) in disk_backend.map_kv_object_to_raw_region(
                                        record
                                    )
                                ],
                                base_file_offset=record.offset,
                                logical_nbytes=record.aligned_length,
                            )
                        else:
                            raw_records = self._tutti_loader.store_bytes_to_raw_lbas(
                                buffer,
                                base_slba=(
                                    disk_backend.kv_object_tutti_raw_base_lba
                                    + record.offset // 512
                                ),
                                logical_file_offset=record.offset,
                                logical_nbytes=record.aligned_length,
                            )
                        path = disk_backend.kv_object_tutti_path(record.pool_id)
                        self._tutti_loader.register_lba_cache({path: raw_records})
                        return (
                            [
                                (
                                    raw.file_offset,
                                    raw.slba,
                                    raw.n_sectors,
                                )
                                for raw in raw_records
                            ],
                            (time.perf_counter() - write_start) * 1000.0,
                        )

                    disk_backend.reset_kv_object_pool_allocation()
                    disk_backend.set_kv_object_tutti_raw_writer(_write_raw_object)
                    logger.info(
                        "KV object Tutti raw cold-store writer installed: "
                        "pool_base_lba=%d",
                        disk_backend.kv_object_tutti_raw_base_lba,
                    )
            logger.info(
                "TuttiDirectLoader initialised: worker=%d device=%s pci=%s "
                "cuda:%d slots=%dx%dMiB lba_cache=%d",
                self.metadata.local_worker_id,
                cfg["device_path"],
                cfg["pci_bdf"],
                cfg["cuda_device"],
                cfg["n_slots"],
                cfg["slot_mb"],
                len(initial_lba_cache),
            )
            logger.info(
                "TUTTI_PROFILE ensure_loader worker=%d keys=%d recovered=%d "
                "paths=%d lba_cache=%d recover_ms=%.3f collect_paths_ms=%.3f "
                "fiemap_ms=%.3f unmount_ms=%.3f create_ms=%.3f total_ms=%.3f",
                self.metadata.local_worker_id,
                len(keys or []),
                n_recovered if isinstance(disk_backend, LocalDiskBackend) else 0,
                len(paths) if isinstance(disk_backend, LocalDiskBackend) else 0,
                len(initial_lba_cache),
                recover_ms,
                collect_paths_ms,
                fiemap_ms,
                unmount_ms,
                create_ms,
                (time.perf_counter() - profile_start) * 1000.0,
            )
            if mount_info is not None:
                self._tutti_can_cpu_fallback = False
            return True
        except Exception as exc:
            logger.warning(
                "TuttiDirectLoader init failed (worker=%d, %s); "
                "falling back to CPU path",
                self.metadata.local_worker_id,
                exc,
            )
            self._tutti_loader = None
            if mount_info is not None:
                remounted = _maybe_remount_after_tutti_failure(mount_info)
                self._tutti_can_cpu_fallback = remounted
            # cudaMalloc can fail transiently if the cold prefill request has
            # not released KV blocks yet.  Keep the loader retryable in that
            # case; permanent configuration errors will simply fail again.
            self._tutti_init_failed = not self._tutti_can_cpu_fallback
            return False

    def _tutti_batched_get(
        self,
        keys: List[CacheEngineKey],
        shapes_per_key: Optional[List[Optional[List[torch.Size]]]] = None,
    ) -> List[Optional[MemoryObj]]:
        """Fast-path disk load via GPU-direct NVMe (Tutti).

        Fetches DiskCacheMetadata from LocalDiskBackend without loading the
        tensor, then delegates to TuttiDirectLoader which DMAs directly from
        NVMe into HBM staging buffers and wraps them as TensorMemoryObj.

        Falls back to returning all-None (triggering the standard CPU path)
        on any error.

        Args:
            keys: Cache keys to load.
            shapes_per_key: Optional per-key shape overrides for DSV4
                optimised KV.  When provided, ``shapes_per_key[i]`` overrides
                the shapes used to build the MemoryObjMetadata for
                ``keys[i]``.  A ``None`` entry means "use shapes stored in
                disk metadata for that key".

        Returns:
            List of MemoryObj (GPU-resident TensorMemoryObj) or None per key.
        """
        if self.storage_manager is None:
            raise RuntimeError(
                "_tutti_batched_get called but storage_manager is None"
            )
        if self._tutti_loader is None:
            raise RuntimeError(
                "_tutti_batched_get called but _tutti_loader is None"
            )

        profile_start = time.perf_counter()
        # Retrieve DiskCacheMetadata from LocalDiskBackend without loading.
        from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
        from lmcache.v1.gpu_connector.tutti_direct_loader import LbaRecord
        from lmcache.utils import DiskCacheMetadata

        disk_backend = self.storage_manager.storage_backends.get(
            "LocalDiskBackend"
        )
        if not isinstance(disk_backend, LocalDiskBackend):
            return [None] * len(keys)

        disk_metas: List[Optional[DiskCacheMetadata]] = []
        tutti_file_offsets: Optional[List[int]] = None
        tutti_read_ranges: Optional[
            List[Optional[Tuple[KVObjectByteRange, ...]]]
        ] = None
        metadata_start = time.perf_counter()
        with disk_backend.disk_lock:
            for key in keys:
                disk_metas.append(disk_backend.dict.get(key, None))
        metadata_ms = (time.perf_counter() - metadata_start) * 1000.0

        if any(m is None for m in disk_metas):
            return [None] * len(keys)

        tutti_shapes_per_key = shapes_per_key
        kv_object_records = disk_backend.get_kv_object_records(keys)
        if all(record is not None for record in kv_object_records):
            object_metas: List[Optional[DiskCacheMetadata]] = []
            object_offsets: List[int] = []
            object_read_ranges: List[Optional[Tuple[KVObjectByteRange, ...]]] = []
            object_records_for_read: List[Optional[KVObjectRecord]] = []
            object_shapes_per_key: Optional[List[Optional[List[torch.Size]]]] = (
                [None] * len(keys) if shapes_per_key is not None else None
            )
            for index, (key, original_meta, record) in enumerate(
                zip(
                    keys,
                    disk_metas,
                    kv_object_records,
                    strict=True,
                )
            ):
                assert original_meta is not None
                assert record is not None
                object_path = disk_backend.kv_object_data_path(record)
                if object_path is None:
                    object_metas = []
                    break
                read_record = record
                shape_override = (
                    shapes_per_key[index] if shapes_per_key is not None else None
                )
                object_shapes = (
                    shape_override
                    if shape_override is not None
                    else original_meta.shapes
                )
                object_dtypes = original_meta.dtypes
                if object_shapes is not None and object_dtypes is not None:
                    requested_nbytes = sum(
                        self._shape_nbytes(shape, dtype)
                        for shape, dtype in zip(
                            object_shapes,
                            object_dtypes,
                            strict=True,
                        )
                    )
                    if requested_nbytes < record.length:
                        if requested_nbytes <= 0:
                            logger.warning(
                                "TUTTI_OBJECT_STORE_PROFILE op=shape_adjust "
                                "status=failed key=%s record_bytes=%d "
                                "requested_bytes=%d",
                                key.to_string(),
                                record.length,
                                requested_nbytes,
                            )
                            object_metas = []
                            break
                        read_record = self._kv_object_prefix_view(
                            record,
                            requested_nbytes,
                        )
                        logger.info(
                            "TUTTI_OBJECT_STORE_PROFILE op=shape_adjust "
                            "status=partial key=%s record_bytes=%d "
                            "requested_bytes=%d",
                            key.to_string(),
                            record.length,
                            requested_nbytes,
                        )
                    elif requested_nbytes > record.length:
                        fitted_shapes = self._dsv4_fit_shapes_to_payload(
                            object_shapes,
                            object_dtypes,
                            record.length,
                        )
                        if fitted_shapes is None:
                            logger.warning(
                                "TUTTI_OBJECT_STORE_PROFILE op=shape_adjust "
                                "status=failed key=%s record_bytes=%d "
                                "requested_bytes=%d",
                                key.to_string(),
                                record.length,
                                requested_nbytes,
                            )
                            object_metas = []
                            break
                        logger.info(
                            "TUTTI_OBJECT_STORE_PROFILE op=shape_adjust "
                            "status=adjusted key=%s record_bytes=%d "
                            "requested_bytes=%d",
                            key.to_string(),
                            record.length,
                            requested_nbytes,
                        )
                        object_shapes = fitted_shapes
                    if object_shapes_per_key is not None:
                        object_shapes_per_key[index] = object_shapes
                object_metas.append(
                    DiskCacheMetadata(
                        path=object_path,
                        size=read_record.length,
                        shape=original_meta.shape,
                        dtype=original_meta.dtype,
                        cached_positions=original_meta.cached_positions,
                        fmt=original_meta.fmt,
                        pin_count=original_meta.pin_count,
                        shapes=object_shapes,
                        dtypes=object_dtypes,
                    )
                )
                object_offsets.append(read_record.offset)
                object_read_ranges.append(read_record.read_ranges)
                object_records_for_read.append(read_record)
            if object_metas:
                raw_lba_cache = {
                    path: [
                        LbaRecord(
                            file_offset=file_offset,
                            slba=slba,
                            n_sectors=n_sectors,
                        )
                        for file_offset, slba, n_sectors in raw_extents
                    ]
                    for path, raw_extents in disk_backend.get_kv_object_raw_lba_cache(
                        object_records_for_read,
                    ).items()
                }
                if raw_lba_cache:
                    self._tutti_loader.register_lba_cache(raw_lba_cache)
                disk_metas = object_metas
                tutti_file_offsets = object_offsets
                tutti_read_ranges = object_read_ranges
                if object_shapes_per_key is not None:
                    tutti_shapes_per_key = object_shapes_per_key
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=select keys=%d pools=%d "
                    "raw_paths=%d ranges=%d",
                    len(keys),
                    len({meta.path for meta in object_metas if meta is not None}),
                    len(raw_lba_cache),
                    sum(
                        len(read_ranges)
                        for read_ranges in object_read_ranges
                        if read_ranges is not None
                    ),
                )

        # Recovered on-disk entries only have filename-level metadata.  Keep
        # their format aligned with the engine so the downstream GPU connector
        # accepts the GPU-resident TensorMemoryObj produced by Tutti.
        for disk_meta in disk_metas:
            if disk_meta is not None and disk_meta.fmt != self.fmt:
                logger.debug(
                    "Normalising Tutti disk metadata fmt from %s to %s for %s",
                    disk_meta.fmt,
                    self.fmt,
                    disk_meta.path,
                )
                disk_meta.fmt = self.fmt

        load_start = time.perf_counter()
        try:
            results = self._tutti_loader.load_chunks_to_hbm(
                keys,
                disk_metas,  # type: ignore[arg-type]
                shapes_per_key=tutti_shapes_per_key,
                file_offsets=tutti_file_offsets,
                read_ranges_per_key=tutti_read_ranges,
            )
        except Exception:
            logger.exception(
                "Tutti direct load failed for %d LocalDiskBackend keys; "
                "falling back to %s",
                len(keys),
                "CPU filesystem path"
                if self._tutti_can_cpu_fallback
                else "cache miss",
            )
            if self._tutti_can_cpu_fallback:
                return self.storage_manager.batched_get(
                    keys=keys,
                    location="LocalDiskBackend",
                    shapes_per_key=tutti_shapes_per_key,
                )
            return [None] * len(keys)
        load_ms = (time.perf_counter() - load_start) * 1000.0
        loaded = sum(1 for result in results if result is not None)
        total_bytes = sum(
            result.get_size() for result in results if result is not None
        )
        logger.info(
            "TUTTI_PROFILE batched_get keys=%d loaded=%d size_mb=%.3f "
            "metadata_ms=%.3f load_hbm_ms=%.3f total_ms=%.3f",
            len(keys),
            loaded,
            total_bytes / 1024**2,
            metadata_ms,
            load_ms,
            (time.perf_counter() - profile_start) * 1000.0,
        )
        return results

    def _make_tutti_store_warmup_callback(
        self,
        keys: List[CacheEngineKey],
        req_id: str,
        is_last_prefill: bool,
    ) -> Optional[Callable[[CacheEngineKey], None]]:
        """Create a callback that warms Tutti after a cold LocalDisk store.

        The callback is invoked once per key after LocalDiskBackend has written
        that key to ext4 and inserted its metadata.  Store can be called
        multiple times for one chunked-prefill request, so the callback
        aggregates keys by request and warms Tutti only after the last prefill
        store batch has completed.  That avoids unmounting the filesystem while
        earlier cold-store chunks are still being written.
        """
        if (
            self._tutti_config is None
            or self._tutti_loader is not None
            or self._tutti_init_failed
            or not (
                _env_flag("LMCACHE_TUTTI_WARMUP_AFTER_STORE")
                or _as_bool(
                    self.config.get_extra_config_value(
                        "tutti_warmup_after_store",
                        False,
                    )
                )
            )
            or (
                self.store_location is not None
                and self.store_location != "LocalDiskBackend"
            )
        ):
            return None

        req_key = req_id or "unspecified"
        with self._tutti_warmup_lock:
            known_keys = self._tutti_store_warmup_keys.setdefault(req_key, set())
            known_keys.update(keys)
            pending = self._tutti_store_warmup_pending.setdefault(req_key, set())
            pending.update(keys)
            if is_last_prefill:
                self._tutti_store_warmup_last_seen.add(req_key)

        def _warm_after_store(req_keys: List[CacheEngineKey]) -> None:
            with self._tutti_warmup_lock:
                if self._tutti_loader is not None or self._tutti_warmup_started:
                    self._clear_tutti_store_warmup_state(req_key)
                    return
                self._tutti_warmup_started = True
                self._tutti_warmup_done = threading.Event()
            logger.info(
                "Tutti warmup after final cold store started: req_id=%s chunks=%d",
                req_id,
                len(req_keys),
            )
            try:
                delay_s = float(
                    self.config.get_extra_config_value(
                        "tutti_warmup_after_store_delay_sec",
                        os.getenv("LMCACHE_TUTTI_WARMUP_AFTER_STORE_DELAY_SEC", 0),
                    )
                )
                if delay_s > 0:
                    logger.info(
                        "Tutti warmup waiting %.3f sec for request cleanup: "
                        "req_id=%s",
                        delay_s,
                        req_id,
                    )
                    time.sleep(delay_s)
                if self._ensure_tutti_loader(req_keys, wait_for_warmup=False):
                    logger.info(
                        "Tutti warmup after final cold store finished: req_id=%s "
                        "chunks=%d",
                        req_id,
                        len(req_keys),
                    )
            except Exception as exc:
                logger.warning(
                    "Tutti warmup after final cold store failed: req_id=%s error=%s",
                    req_id,
                    exc,
                )
            finally:
                if self._tutti_warmup_done is not None:
                    self._tutti_warmup_done.set()
                with self._tutti_warmup_lock:
                    self._clear_tutti_store_warmup_state(req_key)
                    self._tutti_warmup_started = False

        def _on_store_complete(key: CacheEngineKey) -> None:
            req_keys: List[CacheEngineKey] = []
            should_start = False
            with self._tutti_warmup_lock:
                req_pending = self._tutti_store_warmup_pending.get(req_key)
                if req_pending is None:
                    return
                req_pending.discard(key)
                should_start = (
                    req_key in self._tutti_store_warmup_last_seen
                    and not req_pending
                    and not self._tutti_warmup_started
                )
                if should_start:
                    req_keys = list(
                        self._tutti_store_warmup_keys.get(req_key, set())
                    )
            if should_start and req_keys:
                thread = threading.Thread(
                    target=_warm_after_store,
                    args=(req_keys,),
                    name="tutti-store-warmup",
                    daemon=True,
                )
                thread.start()

        return _on_store_complete

    def _clear_tutti_store_warmup_state(self, req_id: str) -> None:
        self._tutti_store_warmup_keys.pop(req_id, None)
        self._tutti_store_warmup_pending.pop(req_id, None)
        self._tutti_store_warmup_last_seen.discard(req_id)

    def freeze(self, enabled: bool) -> None:
        """
        Set the freeze mode for the cache engine.

        When freeze mode is enabled:
        - All store operations will be skipped (no new data stored)
        - Only local_cpu backend will be used for retrieval
        - No admit/evict messages will be generated
        This protects the local_cpu hot cache from changes.

        Args:
            enabled (bool): Whether to enable freeze mode
        """
        if self.storage_manager is not None:
            self.storage_manager.set_freeze(enabled)

    def is_frozen(self) -> bool:
        """
        Get the current freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_frozen()
        return False

    def set_hot_cache(self, enabled: bool) -> None:
        """
        Dynamically enable or disable the LocalCPUBackend hot cache.

        When disabled, the existing hot cache entries will be cleared
        and no new data will be written to the hot cache.

        Args:
            enabled (bool): Whether to enable hot cache
        """
        if self.storage_manager is not None:
            self.storage_manager.set_hot_cache(enabled)

    def is_hot_cache_enabled(self) -> bool:
        """
        Get the current hot cache status of LocalCPUBackend.

        Returns:
            bool: True if hot cache is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_hot_cache_enabled()
        return False

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store operation")
            return

        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )
        self._prepare_gpu_connector_layout(**kwargs)

        if self._is_passive():
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
            return

        assert self.storage_manager is not None

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        # Initialize num_to_store_tokens to avoid reference before assignment
        num_to_store_tokens = 0

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device=torch_device_type
            )

        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=False,
        )

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store operation for %d tokens",
                num_to_store_tokens,
            )
            return

        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts: List[int] = []
        ends: List[int] = []
        keys: List[CacheEngineKey] = []
        memory_objs: List[MemoryObj] = []

        tot_kv_size = 0
        tot_token_num = 0

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        with store_stats.profile_process_tokens():
            prev_key = 0
            for start, end, key in self.token_database.process_tokens(
                tokens,
                hashes,
                offsets,
                mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)
                # Allocate the memory object
                num_tokens = end - start
                kv_shapes = self.metadata.get_shapes(num_tokens)
                kv_dtypes = self.metadata.get_dtypes()
                kv_shapes = self._dsv4_store_shapes_for_range(
                    kv_shapes,
                    kv_dtypes,
                    start,
                    end,
                    len(tokens),
                )

                # TODO (Jiayi): should be batched in the future
                memory_obj = self.storage_manager.allocate(
                    kv_shapes,
                    kv_dtypes,
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                    fmt=self.fmt,
                )
                if memory_obj is None:
                    logger.warning(
                        "Local cpu memory under pressure so"
                        " choosing to store only "
                        f" {len(memory_objs)}"
                        " total chunks of KV cache."
                    )
                    break

                starts.append(start)
                ends.append(end)
                keys.append(key)
                memory_objs.append(memory_obj)
                tot_kv_size += memory_obj.get_size()
                tot_token_num += num_tokens

                # Create KV event
                if self.kv_events_enabled:
                    stored_event = CacheStoreEvent(
                        block_hashes=[key.chunk_hash],
                        parent_block_hash=None if start == 0 else prev_key,
                        token_ids=[],
                        block_size=num_tokens,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    if tokens is not None:
                        stored_event.token_ids = convert_tokens_to_list(
                            tokens,
                            start,
                            end,
                        )
                        if isinstance(tokens, torch.Tensor):
                            stored_event.medium = tokens.device
                    elif hashes is not None:
                        stored_event.token_ids = hashes[start : end + 1]
                    logger.debug(
                        (
                            "Added kv cache event '%s' to kv cache events queue"
                            % stored_event
                        )
                    )
                    self.kv_events.append(stored_event)
                    prev_key = key.chunk_hash

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return

        with store_stats.profile_from_gpu():
            self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)

        with store_stats.profile_put():
            transfer_spec = kwargs.get("transfer_spec", None)
            is_last_prefill = bool(
                kwargs.get(
                    "is_last_prefill",
                    getattr(transfer_spec, "is_last_prefill", False),
                )
            )
            tutti_warmup_callback = self._make_tutti_store_warmup_callback(
                list(keys),
                req_id,
                is_last_prefill,
            )
            # TODO: we implicitly rely on batched_put to call ref_count_down
            # this management should be done in a cleaner way
            self.storage_manager.batched_put(
                keys,
                memory_objs,
                transfer_spec=transfer_spec,
                location=self.store_location,
                on_complete_callback=tutti_warmup_callback,
            )

        self.stats_monitor.on_store_finished(
            store_stats,
            tot_token_num,
        )
        tot_time = store_stats.time_to_store()

        logger.info(
            "[req_id=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms",
            req_id,
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
            store_stats.put_time * 1000,
        )

    @staticmethod
    def _shape_nbytes(shape: torch.Size, dtype: torch.dtype) -> int:
        return int(shape.numel() * dtype.itemsize)

    @staticmethod
    def _zero_token_shape(shape: torch.Size) -> torch.Size:
        if len(shape) >= 3:
            return torch.Size([*shape[:2], 0, *shape[3:]])
        return torch.Size([0])

    @staticmethod
    def _kv_object_prefix_view(
        record: KVObjectRecord,
        prefix_nbytes: int,
    ) -> KVObjectRecord:
        """Return a byte-range view for the logical prefix of a KV object."""
        prefix_ranges: list[KVObjectByteRange] = []
        remaining = prefix_nbytes
        next_target_offset = 0
        for byte_range in sorted(
            record.read_ranges,
            key=lambda item: item.target_offset,
        ):
            if remaining <= 0:
                break
            range_length = min(byte_range.length, remaining)
            prefix_ranges.append(
                KVObjectByteRange(
                    offset=byte_range.offset,
                    length=range_length,
                    target_offset=next_target_offset,
                )
            )
            next_target_offset += range_length
            remaining -= range_length
        if remaining != 0:
            raise ValueError(
                "KV object prefix length exceeds record read-range coverage"
            )
        return record.with_byte_ranges(prefix_ranges, length=prefix_nbytes)

    def _dsv4_fit_shapes_to_payload(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        payload_nbytes: int,
    ) -> Optional[list[torch.Size]]:
        fitted: list[torch.Size] = []
        remaining = payload_nbytes
        for shape, dtype in zip(shapes, dtypes, strict=True):
            group_nbytes = self._shape_nbytes(shape, dtype)
            if group_nbytes == 0:
                fitted.append(shape)
                continue
            if remaining >= group_nbytes:
                fitted.append(shape)
                remaining -= group_nbytes
                continue
            if remaining == 0:
                fitted.append(self._zero_token_shape(shape))
                continue
            if len(shape) < 3 or int(shape[2]) <= 0:
                return None
            token_count = int(shape[2])
            if group_nbytes % token_count != 0:
                return None
            bytes_per_token = group_nbytes // token_count
            if bytes_per_token <= 0 or remaining % bytes_per_token != 0:
                return None
            fitted.append(
                torch.Size([*shape[:2], remaining // bytes_per_token, *shape[3:]])
            )
            remaining = 0
        if remaining != 0:
            return None
        return fitted

    def _dsv4_store_shapes_for_range(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        start: int,
        end: int,
        total_tokens: int,
    ) -> list[torch.Size]:
        if not self.dsv4_optimized_kv:
            return shapes
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return shapes
        tail_start = max(0, total_tokens - self.dsv4_optimized_tail_tokens)
        keep_tail_groups = end > tail_start
        optimized: list[torch.Size] = []
        for shape, dtype, group in zip(
            shapes,
            dtypes,
            klg_manager.kv_layer_groups,
            strict=True,
        ):
            role = self._dsv4_group_role(group, dtype)
            if role in {"swa_cache", "compressor_state"} and not keep_tail_groups:
                optimized.append(self._zero_token_shape(shape))
            else:
                optimized.append(shape)
        return optimized

    @staticmethod
    def _dsv4_group_role(group: Any, dtype: torch.dtype) -> str:
        hidden_dim = group.hidden_dim_size
        compress_ratio = group.compress_ratio
        num_layers = group.num_layers
        if dtype == torch.float32:
            return "compressor_state"
        if dtype != torch.uint8:
            return "unknown"
        if hidden_dim == 132:
            return "csa_indexer_cache"
        if hidden_dim != 584:
            return "unknown"
        if compress_ratio == 1:
            return "swa_cache"
        if compress_ratio >= 64 or group.shape_desc.bs <= 2:
            return "hca_attention_kv"
        if compress_ratio == 4 or num_layers == 30:
            return "csa_attention_kv"
        return "unknown"

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store_layer operation")
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Layerwise store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=True,
        )

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store_layer for %d tokens",
                num_to_store_tokens,
            )
            # Still need to yield to avoid StopIteration
            for layer_id in range(self.num_layers):
                yield
            return

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        prev_key = 0
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)
            # Only check the first layer
            if self.storage_manager.contains(
                keys_multi_layer[0], self.retrieve_locations
            ):
                continue

            # Allocate the memory object
            num_tokens = end - start
            kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=self.fmt,
                busy_loop=self.config.get_extra_config_value("force_store_wait", False),
            )

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            tot_token_num += num_tokens

            # Create KV event
            if self.kv_events_enabled and tokens is not None:
                stored_event = CacheStoreEvent(
                    block_hashes=[key.chunk_hash],
                    parent_block_hash=None if start == 0 else prev_key,
                    token_ids=[],
                    block_size=num_tokens,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                )
                if tokens is not None:
                    stored_event.token_ids = convert_tokens_to_list(
                        tokens,
                        start,
                        end,
                    )
                    if isinstance(tokens, torch.Tensor):
                        stored_event.medium = tokens.device
                logger.debug(
                    f"Added kv cache event '{stored_event}' to kv cache events queue"
                )
                self.kv_events.append(stored_event)
                prev_key = key.chunk_hash

        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]

            # Calculate total KV size for logging
            tot_kv_size = sum(
                mo.get_size() for layer_objs in memory_objs for mo in layer_objs
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

            t_start = time.perf_counter()
            mem_obj_generator = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )

            next(mem_obj_generator)

            for layer_id in range(self.num_layers):
                yield
                next(mem_obj_generator)
                self.storage_manager.batched_put(
                    keys[layer_id], memory_objs[layer_id], location=self.store_location
                )

            tot_time = time.perf_counter() - t_start
            logger.info(
                "[req_id=%s] Stored %d out of total %d tokens. "
                "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s",
                req_id,
                tot_token_num,
                len(tokens),
                tot_kv_size / 1024**3,
                tot_time * 1000,
                tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            )
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        yield

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve operation")
            return torch.zeros(len(tokens), dtype=torch.bool)

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )
        self._prepare_gpu_connector_layout(**kwargs)

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        tot_kv_size = 0

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="retrieve",
            kwargs=kwargs,
            token_count=num_required_tokens,
            require_req_id=True,
        )

        retrieve_stats = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[ProcessedChunk] = []
        retrieve_profile_start = time.perf_counter()
        process_tokens_ms = 0.0
        broadcast_ms = 0.0
        to_gpu_ms = 0.0
        cleanup_ms = 0.0
        if not self._is_passive():
            process_tokens_start = time.perf_counter()
            with retrieve_stats.profile_process_tokens():
                if self.async_loading:
                    reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )
                else:
                    reordered_chunks, tot_kv_size = self._process_tokens_internal(
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )
            process_tokens_ms = (
                time.perf_counter() - process_tokens_start
            ) * 1000.0

        if self.save_only_first_rank:
            broadcast_start = time.perf_counter()
            with retrieve_stats.profile_broadcast():
                with torch_dev.stream(self.broadcast_stream):
                    self._broadcast_or_receive_memory_objs(
                        reordered_chunks,
                        ret_mask,
                    )

                # if self.gpu_connector has load_stream, self.broadcast_stream is equals
                # to self.gpu_connector.load_stream, the broadcast and to_gpu operation
                # will execute sequentially within the stream.
                # if self.gpu_connector does not have load_stream, self.broadcast_stream
                # is created by torch_dev.Stream(), we need to synchronize broadcast
                # operation, and then process to_cpu operation.
                if not hasattr(self.gpu_connector, "load_stream"):
                    self.broadcast_stream.synchronize()
            broadcast_ms = (time.perf_counter() - broadcast_start) * 1000.0

        # NOTE(Jiayi): memory_obj doesn't have to be a pinned
        # cpu tensor for the sake of performance.
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        if len(reordered_chunks) > 0:
            to_gpu_start = time.perf_counter()
            with retrieve_stats.profile_to_gpu():
                _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
                self.gpu_connector.batched_to_gpu(
                    list(memory_objs), list(starts), list(ends), **kwargs
                )
            to_gpu_ms = (time.perf_counter() - to_gpu_start) * 1000.0

        # TODO(Jiayi): Remove the following for loop with batched operations
        # TODO(Jiayi): Need to refactor the `remove_after_retrieve` logic.
        cleanup_start = time.perf_counter()
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                assert self.storage_manager is not None
                self.storage_manager.remove(key, self.retrieve_locations)
                # Sync PDBackend.remove() does NOT call ref_count_down() internally
                # (unlike async PD and other backends), so we must call it manually.
                # See pd_backend.py line 605 TODO comment.
                if self._is_sync_pd_backend():
                    memory_obj.ref_count_down()
            elif not self.async_loading:
                memory_obj.ref_count_down()
        cleanup_ms = (time.perf_counter() - cleanup_start) * 1000.0

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(
            retrieve_stats,
            retrieved_tokens,
        )
        onload_time = retrieve_stats.time_to_retrieve()
        # The retrieved may be larger than the need_to_load
        # Example (page_size=16, chunk_size=256):
        #
        # chunks:  [0..255]                [256..511]
        # pages:   [0..15]...[240..255]    [256..271][272..287] ...
        #
        # num_computed_tokens = 288 => vLLM already has [0..287] (18 pages)
        # LMCache hit_prefix_tokens = 512 => cache covers [0..511] (2 chunks)
        #
        # Skip chunk 1, retrieve chunk 2, overwrite [256..287] (32-token overlap)
        # need_to_load: 512 - 288 = 224 tokens
        # retrieved: 256 tokens
        if not self._is_passive():
            profile_total_ms = (
                time.perf_counter() - retrieve_profile_start
            ) * 1000.0
            logger.info(
                "[req_id=%s] Retrieved %d out of %d required tokens "
                "(from %d total tokens). size: %.4f gb, "
                "cost %.4f ms, throughput: %.4f GB/s;",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
                tot_kv_size / 1024**3,
                onload_time * 1000,
                tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
            )
            logger.info(
                "LMCACHE_RETRIEVE_PROFILE req_id=%s chunks=%d retrieved=%d "
                "required=%d size_mb=%.3f process_tokens_ms=%.3f "
                "broadcast_ms=%.3f to_gpu_ms=%.3f cleanup_ms=%.3f "
                "total_ms=%.3f stats_total_ms=%.3f",
                req_id,
                len(reordered_chunks),
                int(retrieved_tokens),
                int(num_required_tokens),
                tot_kv_size / 1024**2,
                process_tokens_ms,
                broadcast_ms,
                to_gpu_ms,
                cleanup_ms,
                profile_total_ms,
                onload_time * 1000,
            )
        return ret_mask

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve_layer operation")
            yield torch.zeros(len(tokens), dtype=torch.bool)
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        location = None
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)

            # NOTE: Only check the first layer
            if current_location := self.storage_manager.contains(
                keys_multi_layer[0], self.retrieve_locations
            ):
                if location is None:
                    location = current_location
                else:
                    # TODO(Jiayi): Support multi-location retrieval in the future
                    assert location == current_location, (
                        "All retrieved keys should be from the same location "
                        "when use layerwise retrieval."
                        "Please support multi-location retrieval in the future."
                    )
            else:
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)

            ret_mask[start:end] = True

        if keys:
            # Transpose the keys into layer major format
            keys_layer_major = [list(row) for row in zip(*keys, strict=False)]

            get_generator = self.storage_manager.layerwise_batched_get(
                keys_layer_major,
                location=location,
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

            mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
            next(mem_obj_consumer)

            to_count_down = []
            for layer_id in range(self.num_layers):
                task = next(get_generator)

                assert task is not None

                if layer_id == 0:
                    # NOTE(Yuwei): For sglang integration we need to provide retrieved
                    # tokens number in the first layer loading since there is no lookup
                    yield torch.sum(ret_mask)
                else:
                    yield None

                mem_objs_layer = task.result()
                mem_obj_consumer.send(mem_objs_layer)
                to_count_down.extend(mem_objs_layer)

            for mem_obj in to_count_down:
                mem_obj.ref_count_down()
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        yield None

        # synchronize the last layer
        next(mem_obj_consumer)

        # Unpin any disk-loaded staging objects now that the device-side sync
        # has been enqueued (mem_obj_consumer advanced past its sync point).
        # Without this, pin_count stays at 1 forever and the CPU staging pool
        # fills up, causing the next retrieve to deadlock inside allocate().
        for mem_obj in to_count_down:
            if mem_obj.is_pinned:
                mem_obj.unpin()

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d out of total %d tokens",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
            )

        yield ret_mask

    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: the input tokens,
        with shape [seq_len]

        :param Optional[List[int]] hashes: the input hashes, with length [num_chunks]
        :param Optional[List[int]] offsets: the offsets of each chunk,
        with length [num_chunks]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param Optional[str] lookup_id: The lookup ID to
            associate with the lookup. When pin is true, this argument is
            required to be not None.

        :param bool pin: If True, pin the KV cache in the storage.

        :param Optional[dict] request_configs: the configs of the request.

        :return: An int indicating how many prefix tokens exist inside LMCache.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping lookup operation")
            return 0

        assert self.storage_manager is not None

        if tokens is not None:
            lookup_stats = self.stats_monitor.on_lookup_request(len(tokens))
        else:
            assert offsets is not None
            assert hashes is not None
            lookup_stats = self.stats_monitor.on_lookup_request(sum(offsets))

        if search_range is None:
            search_range = self.retrieve_locations

        res = 0
        try:
            chunk_info_iterator = self.token_database.process_tokens(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
            )

            # TODO: support batched_contains when layerwise is enabled
            if self.use_layerwise:
                for start, end, key in chunk_info_iterator:
                    assert isinstance(key, CacheEngineKey)

                    # TODO(Jiayi): Optimize by checking only the existence of the key
                    # of one layer
                    key_all_layers = key.split_layers(self.num_layers)

                    hit_chunks, block_mapping = self.storage_manager.batched_contains(
                        key_all_layers,  # type: ignore
                        search_range,
                        pin,
                    )
                    # Only all layers are hit and hit in one location,
                    # we consider this key as a hit
                    if hit_chunks == self.num_layers and len(block_mapping) == 1:
                        if pin:
                            assert lookup_id is not None, (
                                "lookup_id is required when pin is True"
                            )
                            location = next(iter(block_mapping.keys()))
                            self.lookup_pins[lookup_id][location].extend(key_all_layers)
                        res = end
                        continue
                    return res
            else:
                chunk_info_list = []
                keys = []
                for chunk_info in chunk_info_iterator:
                    assert isinstance(chunk_info[2], CacheEngineKey)
                    start, end, _ = chunk_info
                    chunk_info_list.append(chunk_info)
                    # chunk_info contains (start, end, key)
                    # chunk_info[2] is the key
                    keys.append(chunk_info[2])
                # hit chunks by prefix matching
                hit_chunks, block_mapping = self.storage_manager.batched_contains(
                    keys, search_range, pin
                )
                if pin and block_mapping:
                    assert lookup_id is not None, (
                        "lookup_id is required when pin is True"
                    )
                    self.lookup_pins[lookup_id] = block_mapping
                for idx, (start, end, key) in enumerate(chunk_info_list):
                    if idx < hit_chunks:
                        res = end
                        continue
                    return res

            # all tokens where found, return the maximal end
            return res
        finally:
            self.stats_monitor.on_lookup_finished(lookup_stats, res)
            # vllm lookup sets pin to True
            if pin:
                # touch_cache is tightly coupled with batched_contains
                self.storage_manager.touch_cache()

    @_lmcache_nvtx_annotate
    def move(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
    ) -> int:
        """
        Perform cross-node move of the KV cache.
        """
        assert self.storage_manager is not None

        num_tokens = self.lookup(
            tokens,
            search_range=[old_position],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[old_position]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=old_position,
        )
        assert None not in memory_objs, "Failed to get memory objects to move"
        logger.debug(
            f"Trying to send {len(memory_objs)} memory objects to {new_position}"
        )

        # TODO: reduce loops
        token_dim = memory_objs[0].meta.fmt.token_dim()  # type: ignore
        offsets = [m.meta.shape[token_dim] for m in memory_objs]  # type: ignore

        transfer_spec = {
            "target_peer_init_url": new_position[0],
            "offsets": offsets,
        }

        logger.info(self.storage_manager.storage_backends)
        p2p_backend = self.storage_manager.storage_backends["P2PBackend"]

        future = asyncio.run_coroutine_threadsafe(
            p2p_backend.async_batched_submit_put_task(
                keys,
                memory_objs,  # type: ignore
                transfer_spec=transfer_spec,
            ),
            self.storage_manager.loop,
        )

        future.result()

        if not do_copy:
            self.storage_manager.batched_remove(keys, locations=[old_position])

        logger.debug(f"Moving {num_tokens} token from {old_position} to {new_position}")
        return num_tokens

    # TODO(Jiayi): Add layerwise support.
    @_lmcache_nvtx_annotate
    def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> None:
        """
        An async version of lookup + prefetch.

        There are three categories of backends:
        (1) sync lookup + sync retrieval (e.g., cpu)
        (2) sync lookup + async retrieval (e.g., disk)
        (3) async lookup + async retrieval (e.g., p2p)
        """
        assert self.storage_manager is not None

        keys: list[CacheEngineKey] = []
        cum_chunk_lengths = [0]

        if search_range is None:
            search_range = self.retrieve_locations

        # TODO(Jiayi): make token database able to return list.
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            keys.append(key)
            cum_chunk_lengths.append(end)

        asyncio.run_coroutine_threadsafe(
            self.storage_manager.async_lookup_and_prefetch(
                lookup_id, keys, cum_chunk_lengths, search_range, pin
            ),
            self.storage_manager.loop,
        )

    def cleanup_memory_objs(self, lookup_id: str) -> None:
        """
        Cleanup memory objects allocated during prefetch for an aborted lookup.

        Called by the scheduler when it determines that an aborted lookup
        has finished its prefetch tasks.
        """
        try:
            # Get the completed future from event_manager
            if (
                self.event_manager.get_event_status(EventType.LOADING, lookup_id)
                != EventStatus.DONE
            ):
                logger.debug(
                    "No completed event found for lookup_id=%s to clean up.", lookup_id
                )
                return
            future = self.event_manager.pop_event(EventType.LOADING, lookup_id)

            # Get memory objects from the future result
            memory_objs = future.result()
            # Flatten nested lists (each backend returns a list of chunks)
            memory_objs_flat = [mm for m in memory_objs for mm in m]

            # Release each memory object
            for key, memory_obj in memory_objs_flat:
                try:
                    logger.debug("Releasing memory object for lookup_id=%s", lookup_id)
                    memory_obj.unpin()
                    memory_obj.ref_count_down()
                except Exception as e:
                    logger.error(f"Error releasing memory object: {e}")
        except Exception as e:
            logger.error(
                f"Error during cleanup_memory_objs for lookup_id={lookup_id}: {e}"
            )

    # TODO(Jiayi): Need to handle the case where `tokens=None`.
    # In this case, we compress all tokens.
    # TODO(Jiayi): support other compression methods.
    @_lmcache_nvtx_annotate
    def compress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported compression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        serializer, _ = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )
        assert None not in memory_objs, (
            "LMCacheEngine.compress: Failed to get memory objects to compress"
        )

        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def decompress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported decompression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        _, deserializer = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("there are no tokens to decompress.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )

        assert None not in compressed_memory_objs, (
            "LMCacheEngine.compress: Failed to get compressed "
            "memory objects to decompress"
        )

        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def lookup_unpin(self, lookup_id: str) -> None:
        if lookup_id in self.lookup_pins:
            assert self.storage_manager is not None
            for location, keys in self.lookup_pins.pop(lookup_id).items():
                self.storage_manager.batched_unpin(keys, [location])

        elif (
            self.async_loading is not None
            and self.event_manager.get_event_status(EventType.LOADING, lookup_id)
            != EventStatus.NOT_FOUND
        ):
            self.cleanup_memory_objs(lookup_id)

    @_lmcache_nvtx_annotate
    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        # TODO: need to clear by request_configs
        if self.save_only_first_rank:
            if self.metadata.is_first_rank():
                num_removed = self._clear(tokens, locations, request_configs)
                return num_removed
            else:
                return 0
        return self._clear(tokens, locations, request_configs)

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.kv_events_enabled and (events := self.kv_events):
            self.kv_events = []
            return events
        return []

    def _clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        assert self.storage_manager is not None
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    @_lmcache_nvtx_annotate
    def health(
        self,
    ) -> int:
        """
        Check the health of the cache engine.
        return: 0 if healthy, otherwise the error code
        """
        assert self.storage_manager is not None
        return 0 if self.storage_manager.memcheck() else -1

    def close(self) -> None:
        """Close the cache engine and free all the resources"""
        logger.info("Closing LMCacheEngine...")

        if self.lmcache_worker is not None:
            try:
                logger.info("Closing lmcache_worker...")
                self.lmcache_worker.close()
                logger.info("lmcache_worker closed successfully")
            except Exception as e:
                logger.error(f"Error closing lmcache_worker: {e}")

        try:
            logger.info("Closing storage_manager...")
            if self.storage_manager is not None:
                self.storage_manager.close()
            logger.info("storage_manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing storage_manager: {e}")

        if self._tutti_loader is not None:
            try:
                self._tutti_loader.close()
            except Exception as exc:
                logger.error("Error closing TuttiDirectLoader: %s", exc)
            self._tutti_loader = None

        logger.info("LMCacheEngine closed.")

    def _async_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """
        This function is used to get the memory objects from the event manager.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert "req_id" in kwargs, "req_id is required for async loading"
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        tot_kv_size = 0
        chunks: List[ProcessedChunk] = []
        future = self.event_manager.get_event_future(
            EventType.LOADING, kwargs["req_id"]
        )
        # As mentioned in async_lookup_and_prefetch(), the future.result()
        # is key data pair for each chunk in each tier. So extract the key
        # and memory object pairs to memory_obj_map
        try:
            keyed_memory_objs = future.result()
            memory_obj_map: dict[CacheEngineKey, MemoryObj] = {}
        except Exception as e:
            logger.error(f"Error popping event for request {kwargs['req_id']}: {e}")
            return [], 0

        for backend_results in keyed_memory_objs:
            for key, memory_obj in backend_results:
                memory_obj_map[key] = memory_obj

        # TODO(Jiayi): hashing inside `process_tokens` can be skipped.
        used_keys: set[CacheEngineKey] = set()
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            memory_obj = memory_obj_map.get(key)
            if memory_obj is None:
                # returned chunks are expected to be contiguous.
                # break at the first missing chunk.
                break
            chunks.append((key, memory_obj, start, end))
            tot_kv_size += memory_obj.get_size()
            ret_mask[start:end] = True
            used_keys.add(key)

        # NOTE: free the memory objects that are not hit.
        for key, mem_obj in memory_obj_map.items():
            if key not in used_keys:
                mem_obj.ref_count_down()

        return chunks, tot_kv_size

    def _process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """Process tokens and populate the reordered lists.

        This function is used to process tokens and populate the reordered lists.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert self.storage_manager is not None

        tot_kv_size = 0
        reordered_chunks: List[ProcessedChunk] = []
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        chunk_infos = []
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            chunk_infos.append((key, start, end))

        # block_mapping: location -> [(CacheEngineKey, start, end)]
        if (
            "req_id" in kwargs
            and kwargs["req_id"] in self.lookup_pins
            and len(self.lookup_pins[kwargs["req_id"]]) == 1
        ):
            location = next(iter(self.lookup_pins[kwargs["req_id"]].keys()))
            block_mapping = {location: chunk_infos}
        else:
            block_mapping = self.storage_manager.get_block_mapping(chunk_infos)

        total_tokens = len(tokens)

        last_failed_block_start = None
        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]

            # For DSv4-optimised KV layouts, compute per-chunk shape overrides
            # so that non-tail chunks only read prefix groups (groups 0-2,
            # ~1.4 MB) instead of all 8 groups (~116 MB).  This mirrors the
            # shape masking already applied in the store path and prevents
            # vLLM RPC timeouts caused by reading ~110 GB sequentially for
            # large prompts.
            if self.dsv4_optimized_kv:
                shapes_per_key: Optional[List[Optional[List[torch.Size]]]] = [
                    self._dsv4_store_shapes_for_range(
                        self.metadata.get_shapes(end - start),
                        self.metadata.get_dtypes(),
                        start,
                        end,
                        total_tokens,
                    )
                    for _, start, end in blocks
                ]
            else:
                shapes_per_key = None

            if location == "LocalDiskBackend" and self._tutti_config is not None:
                if self._ensure_tutti_loader(keys):
                    memory_objs = self._tutti_batched_get(
                        keys,
                        shapes_per_key=shapes_per_key,
                    )
                elif self._tutti_can_cpu_fallback:
                    memory_objs = self.storage_manager.batched_get(
                        keys=keys,
                        location=location,
                        shapes_per_key=shapes_per_key,
                    )
                else:
                    logger.warning(
                        "Tutti is configured for LocalDiskBackend but unavailable; "
                        "treating %d disk blocks as misses to avoid CPU filesystem "
                        "fallback after snvme bind",
                        len(keys),
                    )
                    memory_objs = [None] * len(keys)
            else:
                memory_objs = self.storage_manager.batched_get(
                    keys=keys,
                    location=location,
                    shapes_per_key=shapes_per_key,
                )

            used_keys: set[CacheEngineKey] = set()
            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "The cache block is in the storage, but it can't be retrieved"
                    )
                    if (
                        last_failed_block_start is None
                        # The minimum value should be taken here to ensure that
                        # the prefix keys are all consecutive successful.
                        or last_failed_block_start > start
                    ):
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()
                ret_mask[start:end] = True
                used_keys.add(key)

            for (key, _, _), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is not None and key not in used_keys:
                    logger.debug(
                        "ref_count_down for %s of %s as the previous key failed",
                        key,
                        location,
                    )
                    memory_obj.ref_count_down()

        if last_failed_block_start is not None:
            ret_mask[last_failed_block_start:] = False

            kept_chunks: List[ProcessedChunk] = []
            for key, memory_obj, start, end in reordered_chunks:
                if end <= last_failed_block_start:
                    kept_chunks.append((key, memory_obj, start, end))
                else:
                    tot_kv_size -= memory_obj.get_size()
                    # This chunk will not be used. If the engine is configured
                    # to remove-after-retrieve, the caller would normally call
                    # remove (which frees the block), but since we are dropping
                    # these chunks here, we must free them ourselves to avoid
                    # leaking PD buffer pool memory.
                    if self.remove_after_retrieve:
                        assert self.storage_manager is not None
                        self.storage_manager.remove(key, self.retrieve_locations)
                        # Sync PDBackend.remove() does NOT call ref_count_down()
                        # internally (unlike async PD and other backends), so we
                        # must call it manually. See pd_backend.py line 605.
                        if self._is_sync_pd_backend():
                            memory_obj.ref_count_down()
                    else:
                        memory_obj.ref_count_down()
            reordered_chunks = kept_chunks
        return reordered_chunks, tot_kv_size

    def _broadcast_or_receive_memory_objs(
        self,
        reordered_chunks,
        ret_mask,
    ):
        """
        Handles broadcasting or receiving memory objects in a distributed environment.

        This function implements the communication logic where:
        - The first rank (coordinator) broadcasts memory objects and metadata to others
        - Other ranks receive and reconstruct the memory objects

        Parameters:
        reordered_chunks: List of tuples containing [key, memory object, start, end]
        ret_mask: Boolean mask indicating which positions have been processed

        Side Effects:
        - On first rank:
          * Broadcasts chunk count and each chunk's combined metadata
          * Broadcasts tensor data
        - On other ranks:
          * Receives chunk data and populates reordered_chunks
          * Updates ret_mask to mark received positions as True
        """
        if self.metadata.is_first_rank():
            # Broadcast total chunk count
            chunk_count = len(reordered_chunks)
            self.broadcast_object_fn(chunk_count, self.metadata.first_rank)

            # Broadcast each chunk's data
            for key, memory_obj, start, end in reordered_chunks:
                # Combine (start, end) and metadata into single broadcast
                metadata_dict = memory_obj.metadata.to_dict()
                combined_metadata = (start, end, metadata_dict)
                self.broadcast_object_fn(combined_metadata, self.metadata.first_rank)

                # Broadcast tensor data
                raw_tensor = memory_obj.raw_tensor
                assert raw_tensor is not None
                tensor_to_broadcast = raw_tensor.to(
                    f"{torch_device_type}:{self.metadata.worker_id}"
                )
                self.broadcast_fn(tensor_to_broadcast, self.metadata.first_rank)
        else:
            # Receive total chunk count
            chunk_count = self.broadcast_object_fn(None, self.metadata.first_rank)
            if chunk_count is None:
                logger.warning(
                    f"rank={self.metadata.worker_id} received None chunk_count"
                )
                return

            # Fill reordered_chunks with received data
            for _ in range(chunk_count):
                # Receive combined metadata (start, end, metadata_dict)
                combined_metadata = self.broadcast_object_fn(
                    None, self.metadata.first_rank
                )
                if combined_metadata is None:
                    logger.warning(
                        f"rank={self.metadata.worker_id} "
                        "received None combined_metadata"
                    )
                    break
                start, end, metadata_dict = combined_metadata
                ret_mask[start:end] = True

                # Create tensor and receive data
                metadata = MemoryObjMetadata.from_dict(metadata_dict)
                local_rank = self.metadata.worker_id % torch_dev.device_count()
                raw_tensor = torch.empty(
                    torch.Size([metadata.get_size()]),
                    dtype=torch.uint8,
                    device=f"{torch_device_type}:{local_rank}",
                )
                self.broadcast_fn(raw_tensor, self.metadata.first_rank)

                # Create temporary memory object (key not needed for other ranks)
                memory_obj = TensorMemoryObj(
                    raw_data=raw_tensor, metadata=metadata, parent_allocator=None
                )
                reordered_chunks.append((None, memory_obj, start, end))

    def _is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()

    def _is_sync_pd_backend(self) -> bool:
        """Check if the PD backend is the sync variant.

        :return: True when PD is enabled and ``pd_backend_mode`` is ``"sync"``.
        :rtype: bool
        """
        return self.config.enable_pd and self.config.pd_backend_mode == "sync"

    def _get_slot_mapping_list(
        self,
        slot_mapping: Optional[Union[torch.Tensor, List[int]]],
    ) -> Optional[List[int]]:
        """
        Convert slot_mapping to list if it's a tensor, otherwise return as is.

        :param slot_mapping: The slot_mapping to convert,
            can be a torch.Tensor or List[int], or None
        :type slot_mapping: Optional[Union[torch.Tensor, List[int]]]
        :return: The slot_mapping as a List[int], or None if input is None
        :rtype: Optional[List[int]]
        """
        if slot_mapping is None:
            return None
        if isinstance(slot_mapping, torch.Tensor):
            return slot_mapping.tolist()
        # At this point, slot_mapping must be List[int]
        return slot_mapping

    def _log_kvcache_for_check(
        self,
        operation: str,
        kwargs: dict,
        token_count: int,
        require_req_id: bool = False,
    ) -> None:
        """
        Helper method to log KVCache Check information.

        This method centralizes the KVCache Check logging logic that was
        duplicated in multiple methods.

        Args:
            operation: The operation being performed (e.g., "Store", "retrieve")
            kwargs: The keyword arguments containing slot_mapping and req_id
            token_count: The number of tokens involved in the operation
            require_req_id: Whether req_id must be present (default: False)
        """
        if not self.kvcache_check_log_enabled:
            return

        slot_mapping = kwargs.get("slot_mapping")
        if slot_mapping is None:
            return

        if require_req_id:
            req_id = kwargs.get("req_id")
            if req_id is None:
                return
        else:
            req_id = kwargs.get("req_id", "unspecified")

        # Convert slot_mapping to list if it's a tensor
        slot_mapping_list = self._get_slot_mapping_list(slot_mapping)
        # slot_mapping_list should not be None when slot_mapping is not None
        assert slot_mapping_list is not None

        logger.info(
            "[KVCache Check] %s request %s, tokens=%d, slot_mapping: %s",
            operation,
            req_id,
            token_count,
            compress_slot_mapping(slot_mapping_list),
        )


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    # TODO(Jiayi): Please remove this helper function in the future.
    # Currently, it's only used for testing.
    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> MemoryAllocatorInterface:
        # NOTE: should remove this function after fixing the unit tests:
        # raise RuntimeError("_Create_memory_allocator is deprecated!")
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )

        if enable_nixl_storage:
            # TODO(Jiayi): weird to import from transfer utils.
            # First Party
            from lmcache.v1.transfer_channel.transfer_utils import (
                get_correct_device,
            )

            corrected_device = get_correct_device(
                config.nixl_buffer_device,
                metadata.worker_id,
            )

            buffer = torch.empty(
                config.nixl_buffer_size,
                dtype=torch.uint8,
                device=corrected_device,
            )

            if corrected_device == "cpu":
                # Not all backends support cudart() for host memory pinning
                if not hasattr(torch_dev, "cudart"):
                    raise RuntimeError(
                        f"Backend '{torch_device_type}' does not support "
                        "cudart(). NIXL storage CPU buffer requires "
                        "pinned memory via cudaHostRegister, which is "
                        "not available on this backend."
                    )
                else:
                    torch_dev.cudart().cudaHostRegister(
                        buffer.data_ptr(), config.nixl_buffer_size, 0
                    )
            else:
                logger.info(f"Setting device to {corrected_device} ")
                torch_dev.set_device(corrected_device)

            return PagedTensorMemoryAllocator(
                buffer,
                [torch.Size(metadata.kv_shape)],
                [metadata.kv_dtype],
                MemoryFormat.KV_2LTD,
            )

        if config.gds_path is not None:
            assert config.gds_buffer_size is not None
            return CuFileMemoryAllocator(config.gds_buffer_size * 1024**2)

        max_local_cpu_size = config.max_local_cpu_size
        # save_only_first_rank only works when use mla
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank and metadata.is_first_rank():
            # Only the first rank will save the cache,
            # so we need to set it lager than other ranks
            first_rank_max_local_cpu_size = (
                config.extra_config.get(
                    "first_rank_max_local_cpu_size", max_local_cpu_size
                )
                if config.extra_config
                else max_local_cpu_size
            )
            return MixedMemoryAllocator(
                int(first_rank_max_local_cpu_size * 1024**3),
                numa_mapping=numa_mapping,
            )
        return MixedMemoryAllocator(
            int(max_local_cpu_size * 1024**3),
            numa_mapping=numa_mapping,
        )

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> TokenDatabase:
        if config.enable_blending:
            return SegmentTokenDatabase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            numa_mapping = NUMADetector.get_numa_mapping(config)
            logger.info(f"NUMA mapping for instance {instance_id}: {numa_mapping}")
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(
                metadata,
                log_interval=10,
                config=config,
            )

            engine = LMCacheEngine(
                config,
                metadata,
                token_database,
                gpu_connector,
                broadcast_fn,
                broadcast_object_fn,
            )

            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (
                cls._cfgs[instance_id] != config
                or cls._metadatas[instance_id] != metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata."
                )
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        logger.info(f"Destroying LMCacheEngine instance: {instance_id}")

        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            try:
                logger.info("Shutting down stats logger...")
                stat_logger.shutdown()
                logger.info("Stats logger shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down stats logger: {e}")

            engine = cls._instances[instance_id]
            try:
                logger.info("Closing cache engine...")
                engine.close()
                logger.info("Cache engine closed successfully")
            except Exception as e:
                logger.error(f"Error closing cache engine: {e}")

            try:
                logger.info("Cleaning up instance dictionaries...")
                cls._instances.pop(instance_id, None)
                cls._cfgs.pop(instance_id, None)
                cls._metadatas.pop(instance_id, None)
                cls._stat_loggers.pop(instance_id, None)
                logger.info("Instance dictionaries cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up instances: {e}")

            try:
                logger.info("Destroying stats monitor...")
                LMCStatsMonitor.DestroyInstance()
                logger.info("Stats monitor destroyed successfully")
            except Exception as e:
                logger.error(f"Error destroying stats monitor: {e}")

            logger.info(f"LMCacheEngine instance {instance_id} destroyed")
        else:
            logger.warning(f"Instance {instance_id} not found for destruction")
