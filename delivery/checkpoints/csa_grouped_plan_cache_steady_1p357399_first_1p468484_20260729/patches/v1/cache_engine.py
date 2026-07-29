# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
import inspect
import logging
import multiprocessing
import os
import subprocess
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
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    DiskCacheMetadata,
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


class _TuttiProfileLogFilter(logging.Filter):
    """Suppress Tutti profile-only records when profiling is disabled."""

    _PREFIXES = (
        "TUTTI_OBJECT_STORE_PROFILE",
        "LMCACHE_RETRIEVE_PROFILE",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether a log record should be emitted."""
        value = os.getenv("LMCACHE_TUTTI_PROFILE", "0")
        enabled = value.lower() in {"1", "true", "yes", "on"}
        return enabled or not str(record.msg).startswith(self._PREFIXES)


logger.addFilter(_TuttiProfileLogFilter())

_DSV4_HCA_DEFERRED_RETRIEVE_ROLE = "hca_deferred_retrieve"
_DSV4_CSA_DEFERRED_RETRIEVE_ROLE = "csa_deferred_retrieve"
_DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE = "csa_hca_deferred_retrieve"

# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]


@dataclass(frozen=True, slots=True)
class _DSV4StreamingPlanCacheEntry:
    """Immutable source and destination plan for one safe request binding."""

    slot_mapping: torch.Tensor
    indexer_chunks: tuple[tuple[int, tuple[Any, ...]], ...]
    chunks_by_layer: tuple[tuple[int, tuple[Any, ...]], ...]
    shared_raw_lba_cache: dict[str, list[Any]]
    csa_ready: bool
    hca_ready: bool


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d", name, value, default)
        return default


def _configure_dsv4_admission_worker() -> None:
    """Keep deferred admission below latency-sensitive worker CPU work.

    Linux schedules nice values per thread.  The sidecar gather is intentionally
    best-effort background work, so it must yield to the vLLM worker thread that
    is still preparing or offloading another tensor-parallel rank.  Failure to
    adjust priority is harmless and leaves the platform default unchanged.
    """
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
    except (AttributeError, OSError):
        logger.debug("Could not lower DSv4 admission worker priority", exc_info=True)


def _raw_ranges_sector_readable(
    ranges: Tuple[KVObjectByteRange, ...],
    logical_length: int,
) -> bool:
    """Return whether Tutti can read ranges, using ordered waves if needed."""
    _ = logical_length
    for byte_range in ranges:
        if byte_range.offset % 512 != 0:
            return False
    return True


def _dsv4_layer_major_record_segments(
    record: KVObjectRecord,
    *,
    block_nbytes: int,
    total_blocks: int,
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Map one logical layer-major record to physical DMA segments.

    Each returned tuple is ``(first_block, n_blocks, aligned_offset,
    payload_skip, read_length)``.  Explicit object ranges are preserved so a
    composed base+suffix generation is never mistaken for one contiguous pool
    allocation.
    """
    if block_nbytes <= 0 or total_blocks <= 0:
        return ()
    expected_payload = block_nbytes * total_blocks
    if int(record.length) != expected_payload or not record.raw_extents:
        return ()
    expected_target = 0
    segments: list[tuple[int, int, int, int, int]] = []
    for byte_range in record.read_ranges:
        source_offset = int(byte_range.offset)
        target_offset = int(byte_range.target_offset)
        payload_length = int(byte_range.length)
        if (
            target_offset != expected_target
            or target_offset % block_nbytes
            or payload_length <= 0
            or payload_length % block_nbytes
        ):
            return ()
        aligned_offset = source_offset & ~511
        payload_skip = source_offset - aligned_offset
        read_length = ((payload_skip + payload_length + 511) // 512) * 512
        segments.append(
            (
                target_offset // block_nbytes,
                payload_length // block_nbytes,
                aligned_offset,
                payload_skip,
                read_length,
            )
        )
        expected_target += payload_length
    if expected_target != expected_payload:
        return ()
    return tuple(segments)


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
        # The Docker bind mount lives in a separate mount namespace. Even an
        # rshared bind does not propagate an unmount of the bind root back to
        # the host, so a successful container-local umount can still leave
        # host ext4 active. With --pid=host, PID 1 exposes the host mount
        # namespace. Release that reference first, then the container bind.
        host_handoff = os.getenv(
            "LMCACHE_TUTTI_HOST_MOUNT_HANDOFF", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if host_handoff:
            host_umount = subprocess.run(
                [
                    "nsenter",
                    "--mount=/proc/1/ns/mnt",
                    "--",
                    "umount",
                    mount_point,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            host_still_mounted = subprocess.run(
                [
                    "nsenter",
                    "--mount=/proc/1/ns/mnt",
                    "--",
                    "findmnt",
                    "-nr",
                    "-M",
                    mount_point,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if host_umount.returncode != 0 or host_still_mounted.stdout.strip():
                detail = host_umount.stderr.strip() or "mount remains visible"
                raise RuntimeError(
                    "Tutti host mount handoff failed before snvme bind: "
                    f"mount={mount_point} detail={detail}"
                )
            logger.info(
                "Tutti synchronously unmounted host cache filesystem: "
                "source=%s mount=%s",
                source,
                mount_point,
            )

        # SNVM_DEVICE_BIND detaches the controller from the in-tree NVMe
        # driver.  A lazy unmount is not sufficient here: it returns while
        # ext4 can still own the superblock and flush its journal.  Binding
        # snvme during that interval produces lost page writes and can leave
        # the GPU-visible queue/BAR mappings invalid.  Require a synchronous
        # unmount so successful return is the hand-off boundary.
        local_mount = subprocess.run(
            ["findmnt", "-nr", "-M", mount_point],
            check=False,
            capture_output=True,
            text=True,
        )
        if local_mount.stdout.strip():
            subprocess.run(["umount", mount_point], check=True)
        local_still_mounted = subprocess.run(
            ["findmnt", "-nr", "-M", mount_point],
            check=False,
            capture_output=True,
            text=True,
        )
        if local_still_mounted.stdout.strip():
            raise RuntimeError(
                "Tutti container mount remains active before snvme bind: "
                f"mount={mount_point}"
            )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Tutti refused to bind snvme while the cache filesystem is "
            f"still mounted: mount={mount_point} errno={exc.returncode}"
        ) from exc
    logger.info(
        "Tutti synchronously unmounted cache filesystem: source=%s mount=%s",
        source,
        mount_point,
    )
    _wait_for_tutti_mount_handoff_barrier(mount_point)
    return source, mount_point


def _wait_for_tutti_mount_handoff_barrier(mount_point: str) -> None:
    """Wait until every rank and the host release ext4 before any bind."""
    handoff_id = os.getenv("LMCACHE_TUTTI_HANDOFF_ID", "").strip()
    if not handoff_id:
        return

    expected = int(os.getenv("LMCACHE_TUTTI_HANDOFF_RANKS", "8"))
    timeout_s = float(os.getenv("LMCACHE_TUTTI_HANDOFF_TIMEOUT_SEC", "30"))
    barrier_dir = f"/tmp/lmcache_tutti_mount_handoff_{handoff_id}"
    os.makedirs(barrier_dir, exist_ok=True)
    marker_name = mount_point.strip("/").replace("/", "_") + ".ready"
    marker_path = os.path.join(barrier_dir, marker_name)
    marker_fd = os.open(marker_path, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(marker_fd)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        names = os.listdir(barrier_dir)
        error_path = os.path.join(barrier_dir, "host.error")
        if "host.error" in names:
            try:
                with open(error_path, encoding="utf-8") as error_file:
                    detail = error_file.read().strip()
            except OSError:
                detail = "host mount coordinator failed"
            raise RuntimeError(
                "Tutti host mount coordinator failed before snvme bind: "
                f"id={handoff_id} detail={detail}"
            )
        ready = sum(
            name.startswith("mnt_nvme") and name.endswith(".ready") for name in names
        )
        if ready >= expected and "host.ready" in names:
            logger.info(
                "Tutti mount handoff barrier complete: "
                "id=%s rank_ready=%d/%d host_ready=1",
                handoff_id,
                ready,
                expected,
            )
            return
        time.sleep(0.05)
    raise RuntimeError(
        "Tutti mount handoff barrier timed out before snvme bind: "
        f"id={handoff_id} mount={mount_point} expected={expected}"
    )


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
                "Tutti remount attempt failed: source=%s mount=%s cmd=%s error=%s",
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
        self.dsv4_optimized_kv = _as_bool(
            config.get_extra_config_value("dsv4_optimized_kv", False)
        ) or _env_flag("LMCACHE_DSV4_OPTIMIZED_KV")
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
        # A complete DSv4 streaming plan contains immutable source extents and
        # request-specific destination rows.  Rebuilding all 62 layer plans on
        # every repeated hit is pure CPU overhead.  Cache only an exact binding
        # (published-layout revision + cache-key ranges + exact slot-map), so
        # an eviction/re-admission or a different vLLM block allocation cannot
        # reuse stale LBAs or write into stale GPU rows.
        self._dsv4_streaming_plan_cache: OrderedDict[
            tuple[Any, ...], _DSV4StreamingPlanCacheEntry
        ] = OrderedDict()
        self._dsv4_streaming_plan_cache_lock = threading.Lock()
        self._dsv4_streaming_plan_cache_capacity = max(
            0,
            _env_int("LMCACHE_DSV4_STREAMING_PLAN_CACHE_CAPACITY", 8),
        )
        self._dsv4_streaming_plan_cache_hits = 0
        self._dsv4_streaming_plan_cache_misses = 0
        # Long prefills reach ``store`` in several scheduler batches. Keep an
        # extra reference to each CPU snapshot until the final batch, then
        # build one request-wide layer-major generation. Writing one complete
        # generation per intermediate batch multiplies admission traffic by
        # O(number_of_batches) and can keep the cache unready for minutes.
        self._dsv4_snapshot_lock = threading.Lock()
        self._dsv4_snapshot_req_id: Optional[str] = None
        self._dsv4_snapshot_keys: list[CacheEngineKey] = []
        self._dsv4_snapshot_memory_objs: list[MemoryObj] = []
        self._dsv4_snapshot_tokens = 0
        self._dsv4_snapshot_base_key: Optional[CacheEngineKey] = None
        self._dsv4_snapshot_base_tokens = 0
        self._dsv4_completed_snapshot_base_key: Optional[CacheEngineKey] = None
        self._dsv4_completed_snapshot_base_tokens = 0
        # A cache-hit suffix is new admission work, but publishing its next
        # generation must not delay the current request's first token.  Keep
        # the exact synchronous cold-admission ordering on one bounded worker:
        # sidecars first, compact main second, manifest last.
        self._dsv4_admission_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lmcache-dsv4-admission",
            initializer=_configure_dsv4_admission_worker,
        )
        self._dsv4_admission_lock = threading.Lock()
        self._dsv4_admission_pending = 0
        self._dsv4_admission_max_pending = 2
        # Callbacks invoked once the Tutti loader is first created (from the
        # warmup daemon or the first retrieve/store path that succeeds). Used by
        # the vLLM adapter to re-attach the CSA attention-KV prefetch manager,
        # which cannot attach at worker startup because the loader is created
        # lazily. Callbacks MUST NOT touch CUDA -- they may run on a daemon
        # thread; defer any CUDA work to a main-thread callsite.
        self._tutti_loader_ready_callbacks: List[
            Callable[["TuttiDirectLoader"], None]
        ] = []

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
        initialize_kvcaches_ptr = getattr(
            self.gpu_connector,
            "initialize_kvcaches_ptr",
            None,
        )
        if callable(initialize_kvcaches_ptr):
            initialize_kvcaches_ptr(**kwargs)
        self._sync_connector_kv_layer_groups()

    def _sync_connector_kv_layer_groups(self) -> None:
        """Mirror discovered connector KV groups into engine metadata."""
        if self.gpu_connector is None:
            return
        connector_metadata = getattr(self.gpu_connector, "metadata", None)
        connector_manager = getattr(
            connector_metadata,
            "kv_layer_groups_manager",
            None,
        )
        if connector_manager is None or not connector_manager.kv_layer_groups:
            return
        if self.metadata.kv_layer_groups_manager is connector_manager:
            return
        self.metadata.kv_layer_groups_manager = connector_manager
        logger.info(
            "LMCacheEngine synced KV layer groups from GPU connector: groups=%d",
            connector_manager.num_groups,
        )

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

        n_slots: int = int(self.config.get_extra_config_value("tutti_n_slots", 16))
        slot_mb: int = int(self.config.get_extra_config_value("tutti_slot_mb", 32))
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
                            "Tutti startup warmup: server ready, initializing "
                            "(worker=%d)",
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

    def register_tutti_loader_ready_callback(
        self,
        callback: "Callable[[TuttiDirectLoader], None]",
    ) -> None:
        """Register a callback fired once the Tutti loader becomes available.

        If the loader already exists, the callback fires immediately on the
        caller's thread. Otherwise it is stored and invoked from
        :meth:`_ensure_tutti_loader_locked` on whichever thread first succeeds
        at creating the loader (the startup warmup daemon or the first
        retrieve/store path that hits ``LocalDiskBackend``).

        The callback MUST NOT allocate CUDA tensors or launch CUDA kernels: it
        may run on a background daemon thread while the main thread is mid
        forward. Use it only to record a reference or set a flag, and defer any
        CUDA work to a main-thread callsite.

        Args:
            callback: Callable receiving the ready ``TuttiDirectLoader``.
        """
        with self._tutti_warmup_lock:
            if self._tutti_loader is not None:
                loader = self._tutti_loader
            else:
                self._tutti_loader_ready_callbacks.append(callback)
                return
        try:
            callback(loader)
        except Exception:
            logger.exception(
                "Tutti loader ready callback (immediate) failed (worker=%d)",
                self.metadata.local_worker_id,
            )

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
                    if key in disk_backend.dict and disk_backend.dict[key] is not None
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
                and getattr(
                    disk_backend,
                    "kv_object_tutti_raw_cold_store_enabled",
                    disk_backend.kv_object_tutti_raw_enabled,
                )
                and disk_backend.kv_object_tutti_raw_region_path
            ):
                raw_region_start = time.perf_counter()
                raw_region_extents = FiemapHelper.query_extents(
                    disk_backend.kv_object_tutti_raw_region_path
                )
                # Preserve the filesystem path as an alias in the loader's
                # pre-bind cache. Lazy indexer attachment happens after snvme
                # unmounts the filesystem and must not issue FIEMAP again.
                initial_lba_cache[disk_backend.kv_object_tutti_raw_region_path] = list(
                    raw_region_extents
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
                    sum(extent.n_sectors * 512 for extent in raw_region_extents)
                    / 1024**2,
                    (time.perf_counter() - raw_region_start) * 1000.0,
                )
            indexer_region_paths = [
                path.strip()
                for path in os.getenv(
                    "LMCACHE_INDEXER_TUTTI_RAW_REGION_PATH", ""
                ).split(",")
                if path.strip()
            ]
            if indexer_region_paths:
                indexer_region_path = indexer_region_paths[
                    self.metadata.local_worker_id % len(indexer_region_paths)
                ]
                initial_lba_cache[indexer_region_path] = FiemapHelper.query_extents(
                    indexer_region_path
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
                "Tutti lazy pre-scan: %d recovered, %d files scanned, %d LBAs cached",
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
            # Release cached memory on this rank's CUDA device before the raw
            # cudaMalloc inside TuttiDirectLoader.create().
            create_start = time.perf_counter()
            with torch.cuda.device(cfg["cuda_device"]):
                torch.cuda.empty_cache()
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
                and getattr(
                    disk_backend,
                    "kv_object_tutti_raw_cold_store_enabled",
                    disk_backend.kv_object_tutti_raw_enabled,
                )
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
            # Notify registered callbacks that the loader is now available.
            # The caller holds _tutti_warmup_lock; callbacks must not touch CUDA
            # (they may run on the warmup daemon thread) and must not re-enter
            # loader init. See register_tutti_loader_ready_callback.
            for _cb in list(self._tutti_loader_ready_callbacks):
                try:
                    _cb(self._tutti_loader)
                except Exception:
                    logger.exception(
                        "Tutti loader ready callback failed (worker=%d)",
                        self.metadata.local_worker_id,
                    )
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
        read_ranges_per_key: Optional[
            List[Optional[Tuple[KVObjectByteRange, ...]]]
        ] = None,
        kv_object_roles: Optional[List[str]] = None,
        kv_object_layer_ids: Optional[List[int]] = None,
        on_batch_loaded: Optional[
            Callable[[int, List[Optional[MemoryObj]]], None]
        ] = None,
        on_raw_batch_loaded: Optional[
            Callable[
                [
                    int,
                    List[int],
                    List[int],
                    List[int],
                    torch.Tensor,
                    List[List[torch.Size]],
                    List[List[torch.dtype]],
                ],
                None,
            ]
        ] = None,
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
            read_ranges_per_key: Optional per-key compact read ranges relative
                to the selected object record.  This is used by HCA-deferred
                DSv4 retrieval to read non-HCA groups without requiring the
                retained groups to form a prefix of the full object payload.
            kv_object_roles: Optional per-key object-store roles to read
                instead of the default chunk-level ``full`` object.
            kv_object_layer_ids: Optional per-key object-store layer ids paired
                with ``kv_object_roles``.
            on_batch_loaded: Optional callback invoked after each Tutti staging
                batch is read. When supplied, memory objects are staging views
                that must be consumed before the callback returns.
            on_raw_batch_loaded: Optional callback invoked directly from the
                Tutti completion path. It receives staging offsets plus the
                effective shape/dtype layout and avoids constructing per-key
                ``TensorMemoryObj`` wrappers. Mutually exclusive with
                ``on_batch_loaded``.

        Returns:
            List of MemoryObj (GPU-resident TensorMemoryObj) or None per key.
        """
        if self.storage_manager is None:
            raise RuntimeError("_tutti_batched_get called but storage_manager is None")
        if self._tutti_loader is None:
            raise RuntimeError("_tutti_batched_get called but _tutti_loader is None")
        if on_batch_loaded is not None and on_raw_batch_loaded is not None:
            raise ValueError(
                "on_batch_loaded and on_raw_batch_loaded are mutually exclusive"
            )

        profile_start = time.perf_counter()
        # Retrieve DiskCacheMetadata from LocalDiskBackend without loading.
        from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
        from lmcache.v1.gpu_connector.tutti_direct_loader import LbaRecord
        from lmcache.utils import DiskCacheMetadata

        disk_backend = self.storage_manager.storage_backends.get("LocalDiskBackend")
        if not isinstance(disk_backend, LocalDiskBackend):
            return [None] * len(keys)

        disk_metas: List[Optional[DiskCacheMetadata]] = []
        tutti_file_offsets: Optional[List[int]] = None
        tutti_read_ranges: Optional[List[Optional[Tuple[KVObjectByteRange, ...]]]] = (
            None
        )
        metadata_start = time.perf_counter()
        with disk_backend.disk_lock:
            for key in keys:
                disk_metas.append(disk_backend.dict.get(key, None))
        metadata_ms = (time.perf_counter() - metadata_start) * 1000.0

        if any(m is None for m in disk_metas):
            return [None] * len(keys)

        original_key_count = len(keys)
        tutti_shapes_per_key = shapes_per_key
        kv_object_records = disk_backend.get_kv_object_records(
            keys,
            layer_ids=kv_object_layer_ids,
            roles=kv_object_roles,
        )
        pad_missing_objects = 0
        if (
            disk_backend.kv_object_store_enabled
            and disk_backend.kv_object_tutti_raw_enabled
        ):
            first_unreadable = len(kv_object_records)
            for index, record in enumerate(kv_object_records):
                if record is None or not disk_backend.kv_object_record_raw_readable(
                    record
                ):
                    first_unreadable = index
                    break
            if first_unreadable < len(keys):
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=select status=trimmed "
                    "keys=%d readable=%d missing=%d",
                    len(keys),
                    first_unreadable,
                    len(keys) - first_unreadable,
                )
                if first_unreadable == 0:
                    return [None] * len(keys)
                pad_missing_objects = len(keys) - first_unreadable
                keys = keys[:first_unreadable]
                disk_metas = disk_metas[:first_unreadable]
                kv_object_records = kv_object_records[:first_unreadable]
                if shapes_per_key is not None:
                    shapes_per_key = shapes_per_key[:first_unreadable]
                if read_ranges_per_key is not None:
                    read_ranges_per_key = read_ranges_per_key[:first_unreadable]
                tutti_shapes_per_key = shapes_per_key
            first_unreadable = len(kv_object_records)
            for index, (original_meta, record) in enumerate(
                zip(disk_metas, kv_object_records, strict=True)
            ):
                if record is None or original_meta is None:
                    first_unreadable = index
                    break
                read_record = record
                external_ranges = (
                    read_ranges_per_key[index]
                    if read_ranges_per_key is not None
                    else None
                )
                if external_ranges is not None:
                    read_record = record.with_byte_ranges(
                        tuple(
                            KVObjectByteRange(
                                offset=record.offset + byte_range.offset,
                                length=byte_range.length,
                                target_offset=byte_range.target_offset,
                            )
                            for byte_range in external_ranges
                        )
                    )
                object_shapes = (
                    shapes_per_key[index] if shapes_per_key is not None else None
                )
                if object_shapes is None:
                    object_shapes = original_meta.shapes
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
                    if requested_nbytes <= 0:
                        first_unreadable = index
                        break
                    if requested_nbytes < read_record.length:
                        if external_ranges is not None:
                            first_unreadable = index
                            break
                        try:
                            read_record = self._kv_object_prefix_view(
                                record,
                                requested_nbytes,
                            )
                        except ValueError:
                            first_unreadable = index
                            break
                    elif requested_nbytes > read_record.length:
                        fitted_shapes = self._dsv4_fit_shapes_to_payload(
                            object_shapes,
                            object_dtypes,
                            read_record.length,
                        )
                        if fitted_shapes is None:
                            first_unreadable = index
                            break
                if not disk_backend.kv_object_record_raw_readable(read_record):
                    first_unreadable = index
                    break
            if first_unreadable < len(keys):
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=select status=trimmed "
                    "keys=%d readable=%d missing=%d reason=read_ranges",
                    len(keys),
                    first_unreadable,
                    len(keys) - first_unreadable,
                )
                if first_unreadable == 0:
                    return [None] * original_key_count
                pad_missing_objects += len(keys) - first_unreadable
                keys = keys[:first_unreadable]
                disk_metas = disk_metas[:first_unreadable]
                kv_object_records = kv_object_records[:first_unreadable]
                if shapes_per_key is not None:
                    shapes_per_key = shapes_per_key[:first_unreadable]
                if read_ranges_per_key is not None:
                    read_ranges_per_key = read_ranges_per_key[:first_unreadable]
                tutti_shapes_per_key = shapes_per_key
        if all(record is not None for record in kv_object_records):
            object_metas: List[Optional[DiskCacheMetadata]] = []
            object_offsets: List[int] = []
            object_read_ranges: List[Optional[Tuple[KVObjectByteRange, ...]]] = []
            object_records_for_read: List[Optional[KVObjectRecord]] = []
            object_shapes_per_key: Optional[List[Optional[List[torch.Size]]]] = (
                [None] * len(keys) if shapes_per_key is not None else None
            )
            shape_adjust_partial_count = 0
            shape_adjust_adjusted_count = 0
            first_shape_adjust_key: Optional[str] = None
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
                external_ranges = (
                    read_ranges_per_key[index]
                    if read_ranges_per_key is not None
                    else None
                )
                if external_ranges is not None:
                    absolute_ranges = tuple(
                        KVObjectByteRange(
                            offset=record.offset + byte_range.offset,
                            length=byte_range.length,
                            target_offset=byte_range.target_offset,
                        )
                        for byte_range in external_ranges
                    )
                    read_record = record.with_byte_ranges(absolute_ranges)
                if object_shapes is not None and object_dtypes is not None:
                    requested_nbytes = sum(
                        self._shape_nbytes(shape, dtype)
                        for shape, dtype in zip(
                            object_shapes,
                            object_dtypes,
                            strict=True,
                        )
                    )
                    available_nbytes = read_record.length
                    if requested_nbytes < available_nbytes:
                        if requested_nbytes <= 0:
                            logger.warning(
                                "TUTTI_OBJECT_STORE_PROFILE op=shape_adjust "
                                "status=failed key=%s record_bytes=%d "
                                "requested_bytes=%d",
                                key.to_string(),
                                available_nbytes,
                                requested_nbytes,
                            )
                            object_metas = []
                            break
                        if external_ranges is not None:
                            logger.warning(
                                "TUTTI_OBJECT_STORE_PROFILE op=shape_adjust "
                                "status=failed key=%s view_bytes=%d "
                                "requested_bytes=%d reason=external_ranges",
                                key.to_string(),
                                available_nbytes,
                                requested_nbytes,
                            )
                            object_metas = []
                            break
                        read_record = self._kv_object_prefix_view(
                            record,
                            requested_nbytes,
                        )
                        shape_adjust_partial_count += 1
                        if first_shape_adjust_key is None:
                            first_shape_adjust_key = key.to_string()
                    elif requested_nbytes > available_nbytes:
                        fitted_shapes = self._dsv4_fit_shapes_to_payload(
                            object_shapes,
                            object_dtypes,
                            available_nbytes,
                        )
                        if fitted_shapes is None:
                            logger.warning(
                                "TUTTI_OBJECT_STORE_PROFILE op=shape_adjust "
                                "status=failed key=%s record_bytes=%d "
                                "requested_bytes=%d",
                                key.to_string(),
                                available_nbytes,
                                requested_nbytes,
                            )
                            object_metas = []
                            break
                        shape_adjust_adjusted_count += 1
                        if first_shape_adjust_key is None:
                            first_shape_adjust_key = key.to_string()
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
            if object_metas and (
                shape_adjust_partial_count or shape_adjust_adjusted_count
            ):
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=shape_adjust status=summary "
                    "keys=%d partial=%d adjusted=%d first_key=%s",
                    len(object_metas),
                    shape_adjust_partial_count,
                    shape_adjust_adjusted_count,
                    first_shape_adjust_key,
                )
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
            loader_raw_callback = None
            if on_raw_batch_loaded is not None:

                def _forward_raw_batch(
                    batch_start: int,
                    completed_indices: List[int],
                    completed_offsets: List[int],
                    completed_nbytes: List[int],
                    staging: torch.Tensor,
                ) -> None:
                    completed_shapes: List[List[torch.Size]] = []
                    completed_dtypes: List[List[torch.dtype]] = []
                    for local_index in completed_indices:
                        key_index = batch_start + local_index
                        disk_meta = disk_metas[key_index]
                        if disk_meta is None:
                            raise RuntimeError(
                                "Tutti raw callback completed a key without "
                                "disk metadata"
                            )
                        shapes = (
                            tutti_shapes_per_key[key_index]
                            if tutti_shapes_per_key is not None
                            else None
                        )
                        if shapes is None:
                            shapes = disk_meta.shapes
                        if shapes is None:
                            if disk_meta.shape is None:
                                raise RuntimeError(
                                    "Tutti raw callback is missing object shapes"
                                )
                            shapes = [disk_meta.shape]
                        dtypes = disk_meta.dtypes
                        if dtypes is None:
                            if disk_meta.dtype is None:
                                raise RuntimeError(
                                    "Tutti raw callback is missing object dtypes"
                                )
                            dtypes = [disk_meta.dtype] * len(shapes)
                        if len(shapes) != len(dtypes):
                            raise RuntimeError(
                                "Tutti raw callback shape/dtype counts differ"
                            )
                        completed_shapes.append(list(shapes))
                        completed_dtypes.append(list(dtypes))
                    on_raw_batch_loaded(
                        batch_start,
                        completed_indices,
                        completed_offsets,
                        completed_nbytes,
                        staging,
                        completed_shapes,
                        completed_dtypes,
                    )

                loader_raw_callback = _forward_raw_batch

            results = self._tutti_loader.load_chunks_to_hbm(
                keys,
                disk_metas,  # type: ignore[arg-type]
                shapes_per_key=tutti_shapes_per_key,
                file_offsets=tutti_file_offsets,
                read_ranges_per_key=tutti_read_ranges,
                on_batch_loaded=on_batch_loaded,
                on_raw_batch_loaded=loader_raw_callback,
                # Keep foreground retrieve as one queue owner. Per-batch
                # locking fragments sequential NVMe throughput; speculative
                # staged reads are admitted by the loader's priority gate.
            )
        except Exception:
            logger.exception(
                "Tutti direct load failed for %d LocalDiskBackend keys; "
                "falling back to %s",
                len(keys),
                "CPU filesystem path" if self._tutti_can_cpu_fallback else "cache miss",
            )
            # A streaming callback may already have materialized an earlier
            # batch into the final KV cache. Its caller tracks that prefix and
            # cannot consume a second list of CPU fallback objects safely.
            streaming_callback = (
                on_batch_loaded is not None or on_raw_batch_loaded is not None
            )
            if self._tutti_can_cpu_fallback and not streaming_callback:
                fallback_results = self.storage_manager.batched_get(
                    keys=keys,
                    location="LocalDiskBackend",
                    shapes_per_key=tutti_shapes_per_key,
                )
                if pad_missing_objects:
                    fallback_results.extend([None] * pad_missing_objects)
                return fallback_results
            return [None] * original_key_count
        load_ms = (time.perf_counter() - load_start) * 1000.0
        streaming = on_batch_loaded is not None or on_raw_batch_loaded is not None
        loaded = (
            len(keys)
            if streaming
            else sum(1 for result in results if result is not None)
        )
        total_bytes = (
            sum(
                self._shape_nbytes(shape, dtype)
                for meta in disk_metas
                if meta is not None
                for shape, dtype in zip(
                    meta.shapes or [meta.shape],
                    meta.dtypes or [meta.dtype],
                    strict=True,
                )
            )
            if streaming
            else sum(result.get_size() for result in results if result is not None)
        )
        logger.info(
            "TUTTI_PROFILE batched_get keys=%d loaded=%d size_mb=%.3f "
            "metadata_ms=%.3f load_hbm_ms=%.3f total_ms=%.3f streaming=%s",
            len(keys),
            loaded,
            total_bytes / 1024**2,
            metadata_ms,
            load_ms,
            (time.perf_counter() - profile_start) * 1000.0,
            streaming,
        )
        if pad_missing_objects:
            results.extend([None] * pad_missing_objects)
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
                        "Tutti warmup waiting %.3f sec for request cleanup: req_id=%s",
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
                    req_keys = list(self._tutti_store_warmup_keys.get(req_key, set()))
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
        d2h_timing_enabled = _env_flag("LMCACHE_D2H_TIMING")
        allocation_time_s = 0.0
        allocation_max_s = 0.0
        allocation_count = 0
        prepare_thread_cpu_s = 0.0

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        transfer_spec = kwargs.get("transfer_spec", None)
        is_last_prefill = bool(
            kwargs.get(
                "is_last_prefill",
                getattr(transfer_spec, "is_last_prefill", False),
            )
        )
        try:
            lmcache_cached_tokens = int(kwargs.get("lmcache_cached_tokens", 0))
        except (TypeError, ValueError):
            lmcache_cached_tokens = 0
        base_prefix_key: Optional[CacheEngineKey] = None
        terminal_chunk_hash = kwargs.get("lmcache_terminal_chunk_hash")
        if lmcache_cached_tokens > 0 and isinstance(terminal_chunk_hash, int):
            base_prefix_key = CacheEngineKey(
                model_name=self.metadata.model_name,
                world_size=int(self.metadata.world_size),
                worker_id=int(self.metadata.worker_id),
                chunk_hash=int(terminal_chunk_hash),
                dtype=self.metadata.kv_dtype,
                request_configs=request_configs,
            )
        elif (
            lmcache_cached_tokens > 0
            and tokens is not None
            and len(tokens) >= lmcache_cached_tokens
        ):
            # Some vLLM releases do not propagate the scheduler's terminal
            # chunk hash into worker metadata. Recompute only the cached-prefix
            # terminal key in that compatibility case. This happens once on
            # the first store batch of a partial-hit request and preserves the
            # fast scheduler-provided path when the hash is available.
            for _start, _end, key in self.token_database.process_tokens(
                tokens=tokens[:lmcache_cached_tokens],
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)
                base_prefix_key = key
        effective_cached_tokens = (
            lmcache_cached_tokens if base_prefix_key is not None else 0
        )
        if lmcache_cached_tokens > 0 and base_prefix_key is None:
            logger.warning(
                "Partial-hit admission cannot resolve its base key; "
                "skipping suffix generation composition req_id=%s tokens=%d",
                req_id,
                lmcache_cached_tokens,
            )

        prepare_thread_cpu_started = time.thread_time() if d2h_timing_enabled else 0.0
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
                    keep_request_tail=is_last_prefill,
                )

                # TODO (Jiayi): should be batched in the future
                allocation_started = time.perf_counter() if d2h_timing_enabled else 0.0
                memory_obj = self.storage_manager.allocate(
                    kv_shapes,
                    kv_dtypes,
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                    fmt=self.fmt,
                )
                if d2h_timing_enabled:
                    allocation_elapsed = time.perf_counter() - allocation_started
                    allocation_time_s += allocation_elapsed
                    allocation_max_s = max(allocation_max_s, allocation_elapsed)
                    allocation_count += 1
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
        if d2h_timing_enabled:
            prepare_thread_cpu_s = time.thread_time() - prepare_thread_cpu_started

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return

        with store_stats.profile_from_gpu():
            self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)

        put_keys = keys
        put_memory_objs = memory_objs

        # Accumulate CPU snapshots across scheduler store batches and persist
        # exactly one final layer-major generation. Intermediate batches keep
        # only ref-counted host objects; they perform no sidecar GPU/NVMe I/O.
        if (
            (
                _env_flag("LMCACHE_DSV4_HCA_WALKER")
                or _env_flag("LMCACHE_INDEXER_ENABLE_PREFETCH")
            )
            and bool(starts)
            and bool(ends)
        ):
            pending_snapshot = self._stage_dsv4_layer_major_snapshot(
                req_id,
                keys,
                memory_objs,
                tot_token_num,
                is_last_prefill=is_last_prefill,
                base_prefix_key=base_prefix_key,
                base_prefix_token_count=effective_cached_tokens,
            )
            # The retained snapshot reference becomes the ownership passed to
            # ``batched_put`` at final admission. Release this call's original
            # allocation reference now, for both intermediate and final
            # batches. Intermediate objects must not be put yet: their compact
            # main generation is created together with the final sidecars.
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()
            if pending_snapshot is None:
                put_keys = []
                put_memory_objs = []
            else:
                snapshot_keys, snapshot_memory_objs, snapshot_tokens = pending_snapshot
                snapshot_base_key = self._dsv4_completed_snapshot_base_key
                snapshot_base_tokens = self._dsv4_completed_snapshot_base_tokens
                put_keys = snapshot_keys
                put_memory_objs = snapshot_memory_objs
                if snapshot_base_tokens > 0 and self._submit_dsv4_hit_admission(
                    snapshot_keys,
                    snapshot_memory_objs,
                    snapshot_tokens,
                    req_id=req_id,
                    is_last_prefill=is_last_prefill,
                    transfer_spec=transfer_spec,
                    base_prefix_key=snapshot_base_key,
                    base_prefix_token_count=snapshot_base_tokens,
                ):
                    # The bounded admission worker now owns the retained
                    # MemoryObj references and performs sidecar -> main ->
                    # manifest publication in that order.
                    put_keys = []
                    put_memory_objs = []
                else:
                    self._store_dsv4_layer_major_snapshot(
                        snapshot_keys,
                        snapshot_memory_objs,
                        snapshot_tokens,
                        mode="final",
                        base_prefix_key=snapshot_base_key,
                        base_prefix_token_count=snapshot_base_tokens,
                    )

        with store_stats.profile_put():
            tutti_warmup_callback = self._make_tutti_store_warmup_callback(
                list(put_keys),
                req_id,
                is_last_prefill,
            )
            # TODO: we implicitly rely on batched_put to call ref_count_down
            # this management should be done in a cleaner way
            put_kwargs: dict[str, Any] = {
                "transfer_spec": transfer_spec,
                "location": self.store_location,
            }
            if tutti_warmup_callback is not None:
                try:
                    put_signature = inspect.signature(self.storage_manager.batched_put)
                except (TypeError, ValueError):
                    put_signature = None
                supports_store_callback = (
                    put_signature is not None
                    and "on_complete_callback" in put_signature.parameters
                )
                if not supports_store_callback:
                    raise RuntimeError(
                        "StorageManager.batched_put must support "
                        "on_complete_callback for Tutti store warmup. "
                        "Sync lmcache/v1/storage_backend/storage_manager.py "
                        "into the runtime patch bundle."
                    )
                put_kwargs["on_complete_callback"] = tutti_warmup_callback
            if put_memory_objs:
                self.storage_manager.batched_put(
                    put_keys,
                    put_memory_objs,
                    **put_kwargs,
                )

        self.stats_monitor.on_store_finished(
            store_stats,
            tot_token_num,
        )
        tot_time = store_stats.time_to_store()

        logger.info(
            "[req_id=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, prepare_time: %.4f ms, "
            "d2h_time: %.4f ms, put_time: %.4f ms",
            req_id,
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
            store_stats.process_tokens_time * 1000,
            store_stats.from_gpu_time * 1000,
            store_stats.put_time * 1000,
        )
        if d2h_timing_enabled:
            logger.info(
                "LMCACHE_STORE_PREP_TIMING req_id=%s rank=%d chunks=%d "
                "allocation_sum_ms=%.3f allocation_max_ms=%.3f "
                "prepare_other_ms=%.3f prepare_thread_cpu_ms=%.3f "
                "prepare_descheduled_ms=%.3f",
                req_id,
                int(self.metadata.worker_id),
                allocation_count,
                allocation_time_s * 1000.0,
                allocation_max_s * 1000.0,
                max(
                    0.0,
                    (store_stats.process_tokens_time - allocation_time_s) * 1000.0,
                ),
                prepare_thread_cpu_s * 1000.0,
                max(
                    0.0,
                    (store_stats.process_tokens_time - prepare_thread_cpu_s) * 1000.0,
                ),
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

    def _filter_tutti_raw_lookup_prefix(
        self,
        chunk_info_list: list[tuple[int, int, CacheEngineKey]],
        hit_chunks: int,
        block_mapping: dict[str, list[CacheEngineKey]],
        *,
        pin: bool,
        total_tokens: int,
    ) -> int:
        """Trim LocalDisk hits that the Tutti raw object path cannot read.

        The legacy disk ``contains`` check only verifies chunk-level metadata.
        When KV object raw mode is active, retrieve will instead use object
        records and explicit byte ranges.  Lookup must advertise only the prefix
        that the object path can actually DMA; otherwise vLLM treats a later
        miss as an invalid-block update and can fail the request.
        """
        if hit_chunks <= 0 or self.storage_manager is None:
            return hit_chunks
        if self._tutti_config is None or "LocalDiskBackend" not in block_mapping:
            return hit_chunks

        from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

        disk_backend = self.storage_manager.storage_backends.get("LocalDiskBackend")
        if not isinstance(disk_backend, LocalDiskBackend):
            return hit_chunks
        if (
            not disk_backend.kv_object_store_enabled
            or not disk_backend.kv_object_tutti_raw_enabled
        ):
            return hit_chunks

        locations = list(block_mapping)
        disk_location_index = locations.index("LocalDiskBackend")
        preceding_hit_chunks = sum(
            len(block_mapping[location]) for location in locations[:disk_location_index]
        )
        disk_keys = block_mapping["LocalDiskBackend"]
        records = disk_backend.get_kv_object_records(disk_keys)
        logical_lengths: Optional[list[Optional[int]]] = None
        streaming_compact_role: Optional[str] = None
        streaming_consumer_inactive = False
        use_hca_deferred_compact = False
        hca_deferred_readable_limit: Optional[int] = None
        streaming_layout_requested = self.dsv4_optimized_kv and _env_flag(
            "LMCACHE_INDEXER_ENABLE_PREFETCH"
        )
        if streaming_layout_requested:
            streaming_compact_role = (
                _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE
                if _env_flag("LMCACHE_DSV4_HCA_WALKER")
                else _DSV4_CSA_DEFERRED_RETRIEVE_ROLE
            )
            if not self._dsv4_streaming_runtime_consumer_ready(streaming_compact_role):
                # The manifest may be READY, but without its runtime consumer
                # retrieve cannot safely omit the independently stored groups.
                streaming_consumer_inactive = True
            else:
                get_lengths = getattr(
                    disk_backend,
                    "get_kv_object_payload_lengths",
                    None,
                )
                if not callable(get_lengths):
                    logical_lengths = [None] * len(disk_keys)
                else:
                    logical_lengths = get_lengths(
                        disk_keys,
                        roles=[streaming_compact_role] * len(disk_keys),
                    )
                records = disk_backend.get_kv_object_records(
                    disk_keys,
                    roles=[streaming_compact_role] * len(disk_keys),
                )
        elif self._dsv4_hca_defer_requested(total_tokens):
            compact_records = disk_backend.get_kv_object_records(
                disk_keys,
                roles=[_DSV4_HCA_DEFERRED_RETRIEVE_ROLE] * len(disk_keys),
            )
            disk_blocks = chunk_info_list[
                preceding_hit_chunks : preceding_hit_chunks + len(disk_keys)
            ]
            hca_deferred_readable_limit = self._dsv4_hca_deferred_prefix_len(
                disk_blocks,
                disk_backend,
                manager=self._dsv4_hca_object_source_manager(),
            )
            records = compact_records
            use_hca_deferred_compact = True
        readable = 0
        staging_bytes = (
            int(self._tutti_config["slot_mb"])
            * 1024
            * 1024
            * int(self._tutti_config["n_slots"])
        )
        metadata_only = 0
        for index, (key, record) in enumerate(zip(disk_keys, records, strict=True)):
            if streaming_consumer_inactive:
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter status=miss "
                    "index=%d key=%s reason=streaming_consumer_inactive",
                    index,
                    key.to_string(),
                )
                break
            if (
                hca_deferred_readable_limit is not None
                and index >= hca_deferred_readable_limit
            ):
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter status=miss "
                    "index=%d key=%s reason=hca_deferred_objects",
                    index,
                    key.to_string(),
                )
                break
            if logical_lengths is not None:
                logical_length = (
                    logical_lengths[index] if index < len(logical_lengths) else None
                )
                if logical_length is None:
                    logger.info(
                        "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter "
                        "status=miss index=%d key=%s "
                        "reason=no_logical_payload role=%s",
                        index,
                        key.to_string(),
                        streaming_compact_role,
                    )
                    break
                expected_bytes = self._dsv4_streaming_compact_payload_nbytes(
                    chunk_info_list[preceding_hit_chunks + index],
                    streaming_compact_role,
                    total_tokens,
                )
                # A chunk that was the tail of the stored request can own a
                # physical compact-main payload.  When the same prefix is used
                # by a longer request, that chunk is no longer in the active
                # tail window and its main payload is intentionally ignored.
                # Treat it exactly like a metadata-only entry; the streamed
                # CSA/HCA/indexer objects remain authoritative.
                if expected_bytes == 0:
                    readable += 1
                    metadata_only += 1
                    continue
                if logical_length == 0:
                    logger.info(
                        "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter "
                        "status=miss index=%d key=%s "
                        "reason=metadata_only_shape_mismatch "
                        "expected_bytes=%d role=%s",
                        index,
                        key.to_string(),
                        expected_bytes,
                        streaming_compact_role,
                    )
                    break
            if record is None:
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter status=miss "
                    "index=%d key=%s reason=no_record",
                    index,
                    key.to_string(),
                )
                break
            if logical_lengths is not None and record.length != logical_length:
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter status=miss "
                    "index=%d key=%s reason=length_mismatch "
                    "logical_bytes=%d record_bytes=%d",
                    index,
                    key.to_string(),
                    logical_length,
                    record.length,
                )
                break
            read_record = self._tutti_lookup_read_record(
                chunk_info_list[preceding_hit_chunks + index],
                record,
                total_tokens,
                hca_deferred_compact=use_hca_deferred_compact,
                streaming_compact_role=streaming_compact_role,
            )
            if read_record is None:
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter status=miss "
                    "index=%d key=%s reason=shape_mismatch",
                    index,
                    key.to_string(),
                )
                break
            read_bytes = disk_backend.kv_object_record_raw_read_bytes(read_record)
            if read_bytes <= 0:
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter status=miss "
                    "index=%d key=%s reason=alignment",
                    index,
                    key.to_string(),
                )
                break
            if read_bytes > staging_bytes:
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter status=miss "
                    "index=%d key=%s reason=staging_capacity read_bytes=%d "
                    "staging_bytes=%d",
                    index,
                    key.to_string(),
                    read_bytes,
                    staging_bytes,
                )
                break
            if not disk_backend.kv_object_record_raw_readable(read_record):
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter status=miss "
                    "index=%d key=%s reason=raw_extents",
                    index,
                    key.to_string(),
                )
                break
            readable += 1

        if readable == len(disk_keys):
            if metadata_only:
                logger.info(
                    "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter "
                    "status=ready hit_chunks=%d readable=%d "
                    "metadata_only=%d role=%s",
                    hit_chunks,
                    preceding_hit_chunks + readable,
                    metadata_only,
                    streaming_compact_role,
                )
            return hit_chunks

        dropped_keys = disk_keys[readable:]
        following_locations = locations[disk_location_index + 1 :]
        if pin:
            if dropped_keys:
                self.storage_manager.batched_unpin(
                    dropped_keys,
                    ["LocalDiskBackend"],
                )
            for location in following_locations:
                self.storage_manager.batched_unpin(
                    block_mapping[location],
                    [location],
                )
        if readable == 0:
            del block_mapping["LocalDiskBackend"]
        else:
            block_mapping["LocalDiskBackend"] = disk_keys[:readable]
        for location in following_locations:
            del block_mapping[location]
        trimmed_hit_chunks = preceding_hit_chunks + readable
        logger.info(
            "TUTTI_OBJECT_STORE_PROFILE op=lookup_filter status=trimmed "
            "hit_chunks=%d readable=%d dropped=%d metadata_only=%d",
            hit_chunks,
            trimmed_hit_chunks,
            hit_chunks - trimmed_hit_chunks,
            metadata_only,
        )
        return trimmed_hit_chunks

    def _dsv4_streaming_runtime_consumer_ready(self, role: str) -> bool:
        """Return whether split-layout consumers are already attached.

        Lookup runs on a background server thread, so this probe must stay
        observational: manager construction and model patching belong to the
        adapter's main-thread lifecycle.
        """
        try:
            from lmcache.v1.csa_attention_kv_prefetch_manager import (
                get_csa_attention_kv_prefetch_manager,
            )
        except ImportError:
            return False
        csa_manager = get_csa_attention_kv_prefetch_manager()
        if csa_manager is None:
            return False
        stream_available = getattr(csa_manager, "request_stream_available", None)
        if not callable(stream_available) or not stream_available():
            return False
        try:
            from lmcache.v1.indexer_ssd_manager import get_indexer_ssd_manager
        except ImportError:
            return False
        indexer_manager = get_indexer_ssd_manager()
        indexer_available = getattr(
            indexer_manager,
            "native_indexer_stream_available",
            None,
        )
        if not callable(indexer_available) or not indexer_available():
            return False
        if role == _DSV4_CSA_DEFERRED_RETRIEVE_ROLE:
            return True
        if role != _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE or not getattr(
            csa_manager,
            "hca_layer_ids",
            (),
        ):
            return False
        return True

    def _dsv4_streaming_compact_payload_nbytes(
        self,
        chunk_info: tuple[int, int, CacheEngineKey],
        role: Optional[str],
        total_tokens: int,
    ) -> int:
        """Return bytes required by the current request's compact main view."""
        start, end, _key = chunk_info
        dtypes = self.metadata.get_dtypes()
        base_shapes = self.metadata.get_shapes(end - start)
        if role == _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE:
            shapes = self._dsv4_csa_hca_compact_retrieve_shapes_for_range(
                base_shapes,
                dtypes,
                start,
                end,
                total_tokens,
            )
        elif role == _DSV4_CSA_DEFERRED_RETRIEVE_ROLE:
            shapes = self._dsv4_csa_compact_retrieve_shapes_for_range(
                base_shapes,
                dtypes,
                start,
                end,
                total_tokens,
            )
        else:
            return -1
        return sum(
            self._shape_nbytes(shape, dtype)
            for shape, dtype in zip(shapes, dtypes, strict=True)
        )

    def _tutti_lookup_read_record(
        self,
        chunk_info: tuple[int, int, CacheEngineKey],
        record: KVObjectRecord,
        total_tokens: int,
        *,
        hca_deferred_compact: bool = False,
        streaming_compact_role: Optional[str] = None,
    ) -> Optional[KVObjectRecord]:
        """Return the object record shape-adjusted the same way retrieve will read."""
        start, end, _key = chunk_info
        if not self.dsv4_optimized_kv:
            return record
        dtypes = self.metadata.get_dtypes()
        base_shapes = self.metadata.get_shapes(end - start)
        if streaming_compact_role == _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE:
            shapes = self._dsv4_csa_hca_compact_retrieve_shapes_for_range(
                base_shapes,
                dtypes,
                start,
                end,
                total_tokens,
            )
            read_ranges = None
        elif streaming_compact_role == _DSV4_CSA_DEFERRED_RETRIEVE_ROLE:
            shapes = self._dsv4_csa_compact_retrieve_shapes_for_range(
                base_shapes,
                dtypes,
                start,
                end,
                total_tokens,
            )
            read_ranges = None
        else:
            shapes, read_ranges = self._dsv4_retrieve_view_for_range(
                base_shapes,
                dtypes,
                start,
                end,
                total_tokens,
                require_sector_readable=not hca_deferred_compact,
            )
        read_record = record
        if read_ranges is not None and not hca_deferred_compact:
            try:
                read_record = record.with_byte_ranges(
                    tuple(
                        KVObjectByteRange(
                            offset=record.offset + byte_range.offset,
                            length=byte_range.length,
                            target_offset=byte_range.target_offset,
                        )
                        for byte_range in read_ranges
                    )
                )
            except ValueError:
                return None
        requested_nbytes = sum(
            self._shape_nbytes(shape, dtype)
            for shape, dtype in zip(shapes, dtypes, strict=True)
        )
        if requested_nbytes <= 0:
            return None
        if requested_nbytes < read_record.length:
            if read_ranges is not None and not hca_deferred_compact:
                return None
            return self._kv_object_prefix_view(read_record, requested_nbytes)
        if requested_nbytes > read_record.length:
            fitted_shapes = self._dsv4_fit_shapes_to_payload(
                shapes,
                dtypes,
                read_record.length,
            )
            if fitted_shapes is None:
                return None
        return read_record

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

    def _dsv4_hca_defer_requested(self, total_tokens: int) -> bool:
        """Return whether this request should use HCA-deferred retrieval."""
        if not _env_flag("LMCACHE_DSV4_DEFER_HCA_TO_MOE"):
            return False
        if not _env_flag("LMCACHE_HCA_ENABLE_OBJECT_SOURCE"):
            return False
        max_tokens = _env_int("LMCACHE_DSV4_DEFER_HCA_MAX_TOKENS", 0)
        return max_tokens <= 0 or total_tokens <= max_tokens

    def _dsv4_csa_attention_kv_prefetch_active(self) -> bool:
        """Return whether the CSA attention KV prefetch manager is attached.

        When attached, the on-demand prefetcher owns the load lifecycle of
        ``csa_attention_kv`` group bytes via Tutti GPU-direct reads timed to
        the FFN/MoE overlap window.  In that case the synchronous retrieve
        path must NOT scatter those bytes to vLLM's KV cache, and must
        register the per-request chunk locations with the prefetcher so it
        can issue range reads when the Lightning Indexer outputs top-K.

        Lazily attaches the manager when the Tutti loader has become
        available since post_init.  ``_attach_csa_attention_kv_prefetch``
        skips if its env gate is off, so this call is a no-op when the
        operator has not opted into the full-overlap pipeline.
        """
        try:
            from lmcache.v1.csa_attention_kv_prefetch_manager import (
                get_csa_attention_kv_prefetch_manager,
            )
        except ImportError:
            return False
        manager = get_csa_attention_kv_prefetch_manager()
        if manager is not None:
            return True
        tutti_loader = getattr(self, "_tutti_loader", None)
        if tutti_loader is None:
            ensure_tutti_loader = getattr(self, "_ensure_tutti_loader", None)
            if callable(ensure_tutti_loader):
                try:
                    ensure_tutti_loader(wait_for_warmup=False)
                except Exception:
                    logger.exception(
                        "Failed to initialize Tutti loader for CSA attention "
                        "KV prefetch attach"
                    )
            tutti_loader = getattr(self, "_tutti_loader", None)
        if tutti_loader is None:
            return False
        try:
            from lmcache.integration.vllm.vllm_v1_adapter import (
                _ensure_csa_attention_kv_prefetch_attached,
            )
        except ImportError:
            return False
        manager = _ensure_csa_attention_kv_prefetch_attached(tutti_loader)
        return manager is not None

    def _dsv4_retrieve_shapes_for_range(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        start: int,
        end: int,
        total_tokens: int,
    ) -> list[torch.Size]:
        """Return per-group shapes for a retrieve chunk, applying both the
        DSv4-optimized tail-only masking from
        :meth:`_dsv4_store_shapes_for_range` and the additional
        ``csa_attention_kv`` zero-shape when the canonical CSA attention KV
        prefetcher is attached. Attachment and retrieve filtering are one
        atomic mode; a half-enabled pipeline is not supported.

        Args:
            shapes: Original per-group shapes from
                :meth:`KVCacheMetadata.get_shapes` for this chunk's token
                count.
            dtypes: Per-group dtypes from
                :meth:`KVCacheMetadata.get_dtypes`.
            start: Logical token start of this chunk inside the request.
            end: Logical token end (exclusive).
            total_tokens: Total logical tokens in the request.

        Returns:
            Per-group shapes with all relevant zero-shape masks applied.
        """
        self._dsv4_log_retrieve_group_table_once(shapes, dtypes)
        base = self._dsv4_store_shapes_for_range(
            shapes,
            dtypes,
            start,
            end,
            total_tokens,
        )
        if not self._dsv4_csa_attention_kv_prefetch_active():
            return base
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return base
        # V28: with the HCA walker enabled the hca_attention_kv group is
        # ALSO left on NVMe for the layer-major walker (it accounts for 84%
        # of the post-CSA-filter sync retrieve bytes).  Gated separately so
        # the proven CSA-only behavior (V27) stays the default.  SAFETY
        # INTERLOCK: only skip the sync read when the manager actually has
        # HCA layers registered — otherwise nobody would load those bytes
        # and attention would consume stale K rows.
        hca_walker = _env_flag("LMCACHE_DSV4_HCA_WALKER")
        native_indexer_stream = False
        try:
            from lmcache.v1.indexer_ssd_manager import get_indexer_ssd_manager

            indexer_manager = get_indexer_ssd_manager()
            active = getattr(
                indexer_manager,
                "native_indexer_stream_active",
                None,
            )
            native_indexer_stream = bool(callable(active) and active())
        except ImportError:
            native_indexer_stream = False
        if hca_walker:
            try:
                from lmcache.v1.csa_attention_kv_prefetch_manager import (
                    get_csa_attention_kv_prefetch_manager,
                )

                manager = get_csa_attention_kv_prefetch_manager()
                hca_walker = bool(
                    manager is not None and getattr(manager, "hca_layer_ids", ())
                )
            except ImportError:
                hca_walker = False
            if not hca_walker and not getattr(
                self, "_hca_walker_interlock_logged", False
            ):
                self._hca_walker_interlock_logged = True
                logger.warning(
                    "LMCACHE_DSV4_HCA_WALKER=1 but no HCA layers are "
                    "registered with the prefetch manager; keeping the "
                    "synchronous HCA retrieve (no zero-shape)"
                )
        filtered: list[torch.Size] = []
        for shape, dtype, group in zip(
            base,
            dtypes,
            klg_manager.kv_layer_groups,
            strict=True,
        ):
            role = self._dsv4_group_role(group, dtype)
            if (
                role == "csa_attention_kv"
                or (hca_walker and role == "hca_attention_kv")
                or (native_indexer_stream and role == "csa_indexer_cache")
            ):
                filtered.append(self._zero_token_shape(shape))
            else:
                filtered.append(shape)
        return filtered

    def _dsv4_csa_compact_retrieve_shapes_for_range(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        start: int,
        end: int,
        total_tokens: int,
    ) -> list[torch.Size]:
        """Return shapes matching the CSA streaming compact main object.

        CSA attention KV and native Indexer K are independently layer-major,
        so neither is duplicated in the main object. HCA remains here unless
        the HCA walker is enabled.
        """
        stored_shapes = self._dsv4_store_shapes_for_range(
            shapes,
            dtypes,
            start,
            end,
            total_tokens,
        )
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return stored_shapes
        return [
            self._zero_token_shape(shape)
            if self._dsv4_group_role(group, dtype)
            in {"csa_attention_kv", "csa_indexer_cache"}
            else shape
            for shape, dtype, group in zip(
                stored_shapes,
                dtypes,
                klg_manager.kv_layer_groups,
                strict=True,
            )
        ]

    def _dsv4_csa_hca_compact_retrieve_shapes_for_range(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        start: int,
        end: int,
        total_tokens: int,
    ) -> list[torch.Size]:
        """Return shapes matching the CSA/HCA streaming compact main object."""
        stored_shapes = self._dsv4_store_shapes_for_range(
            shapes,
            dtypes,
            start,
            end,
            total_tokens,
        )
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return stored_shapes
        streamed_roles = {
            "csa_attention_kv",
            "hca_attention_kv",
            "csa_indexer_cache",
        }
        return [
            self._zero_token_shape(shape)
            if self._dsv4_group_role(group, dtype) in streamed_roles
            else shape
            for shape, dtype, group in zip(
                stored_shapes,
                dtypes,
                klg_manager.kv_layer_groups,
                strict=True,
            )
        ]

    def _dsv4_log_retrieve_group_table_once(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
    ) -> None:
        """Log every KV group's role/shape/bytes on the first retrieve.

        The connector's ``_log_dsv4_optimized_policy_once`` only fires on
        the legacy ``to_gpu`` path; the Tutti streaming retrieve never
        reaches it, leaving the group->role mapping invisible in production
        logs.  This one-time table is the ground truth for choosing V28
        zero-shape targets (per-chunk byte sizes at full chunk_size).

        Args:
            shapes: Per-group shapes for one full retrieve chunk.
            dtypes: Per-group dtypes aligned with ``shapes``.
        """
        if getattr(self, "_dsv4_group_table_logged", False):
            return
        self._dsv4_group_table_logged = True
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return
        try:
            rows = []
            for idx, (shape, dtype, group) in enumerate(
                zip(shapes, dtypes, klg_manager.kv_layer_groups, strict=True)
            ):
                role = self._dsv4_group_role(group, dtype)
                nbytes = int(shape.numel()) * int(dtype.itemsize)
                rows.append(
                    f"{idx}:{role}:layers={group.num_layers}:"
                    f"hidden={group.hidden_dim_size}:cr={group.compress_ratio}:"
                    f"shape={tuple(shape)}:bytes={nbytes}"
                )
            logger.info(
                "DSV4_GROUP_TABLE (first retrieve chunk): [%s]",
                ", ".join(rows),
            )
        except Exception:
            logger.exception("DSV4_GROUP_TABLE logging failed")

    @staticmethod
    def _dsv4_layer_chunk_map_complete(
        chunks_by_layer: dict[int, list[Any]],
        layer_ids: Iterable[int],
        expected_end: Optional[int],
    ) -> bool:
        """Return whether every expected layer exactly covers the request."""
        expected_layers = frozenset(int(layer_id) for layer_id in layer_ids)
        if (
            not expected_layers
            or expected_end is None
            or expected_end <= 0
            or set(chunks_by_layer) != expected_layers
        ):
            return False
        common_end: Optional[int] = None
        for layer_id in expected_layers:
            chunks = sorted(
                chunks_by_layer.get(layer_id, ()),
                key=lambda chunk: int(chunk.first_compressed_block),
            )
            cursor = 0
            for chunk in chunks:
                start = int(chunk.first_compressed_block)
                end = int(chunk.end_compressed_block)
                if start != cursor or end <= start:
                    return False
                cursor = end
            if cursor <= 0:
                return False
            if common_end is None:
                common_end = cursor
            elif cursor != common_end:
                return False
        return common_end == expected_end

    def _dsv4_streaming_expected_layer_coverage(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        group_role: str,
    ) -> Optional[int]:
        """Return the descriptor coverage required for one streamed group."""
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return None
        dtypes = self.metadata.get_dtypes()
        compression_ratio: Optional[int] = None
        for group, dtype in zip(
            klg_manager.kv_layer_groups,
            dtypes,
            strict=True,
        ):
            if self._dsv4_group_role(group, dtype) == group_role:
                compression_ratio = int(group.compress_ratio)
                break
        if compression_ratio is None or compression_ratio <= 0:
            return None
        descriptor_tokens = compression_ratio * (
            64 if group_role == "csa_attention_kv" else 1
        )
        coverage = 0
        for _key, start, end in blocks:
            chunk_tokens = int(end) - int(start)
            if chunk_tokens <= 0 or chunk_tokens % descriptor_tokens:
                return None
            coverage += chunk_tokens // descriptor_tokens
        return coverage if coverage > 0 else None

    @staticmethod
    def _dsv4_deactivate_streaming_consumers(
        csa_manager: Any,
        indexer_manager: Any,
    ) -> None:
        """Ensure no previous split-layout plan can write the next request."""
        deactivate_indexer = getattr(
            indexer_manager,
            "deactivate_native_indexer_stream",
            None,
        )
        if callable(deactivate_indexer) and not deactivate_indexer():
            raise RuntimeError("native indexer stream did not deactivate")
        deactivate_csa = getattr(csa_manager, "deactivate_request", None)
        if callable(deactivate_csa) and not deactivate_csa():
            raise RuntimeError("CSA/HCA stream did not deactivate")

    def _dsv4_streaming_plan_cache_key(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        slot_mapping: Optional[torch.Tensor],
    ) -> Optional[tuple[Any, ...]]:
        """Build a correctness-safe identity for one bound streaming plan.

        The published generation alone identifies source bytes but not the
        physical GPU rows assigned by vLLM.  This source key includes the
        process-local publication revision and every logical cache-key range.
        Cache lookup separately compares the exact slot-map tensor, avoiding
        both unsafe source-only reuse and the cost of hashing 480K entries.

        Args:
            blocks: Ordered cache-key ranges used by the streaming plan.
            slot_mapping: CPU int64 destination mapping for the request.

        Returns:
            A hashable plan identity, or ``None`` when safe reuse cannot be
            proven.
        """
        if (
            self._dsv4_streaming_plan_cache_capacity <= 0
            or not blocks
            or not isinstance(slot_mapping, torch.Tensor)
            or slot_mapping.device.type != "cpu"
            or self.storage_manager is None
        ):
            return None
        try:
            from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

            disk_backend = self.storage_manager.storage_backends.get("LocalDiskBackend")
            if not isinstance(disk_backend, LocalDiskBackend):
                return None
            get_plan_token = getattr(
                disk_backend,
                "get_csa_streaming_plan_token",
                None,
            )
            if not callable(get_plan_token):
                return None
            layout_token = get_plan_token(blocks[-1][0])
            if layout_token is None:
                return None

            first_start = int(blocks[0][1])
            final_end = int(blocks[-1][2])
            if (
                first_start < 0
                or final_end <= first_start
                or final_end > int(slot_mapping.numel())
                or int(layout_token[2]) != final_end - first_start
            ):
                return None
            expected_start = first_start
            block_signature: list[tuple[str, int, int]] = []
            for key, start, end in blocks:
                start_int = int(start)
                end_int = int(end)
                if start_int != expected_start or end_int <= start_int:
                    return None
                block_signature.append((key.to_string(), start_int, end_int))
                expected_start = end_int

            return (
                tuple(layout_token),
                tuple(block_signature),
                final_end - first_start,
            )
        except Exception:
            logger.exception("Failed to construct DSv4 streaming plan cache key")
            return None

    def _dsv4_streaming_plan_cache_get(
        self,
        cache_key: Optional[tuple[Any, ...]],
        slot_mapping: Optional[torch.Tensor],
    ) -> Optional[_DSV4StreamingPlanCacheEntry]:
        """Return and refresh an exact cached plan binding, if present."""
        if cache_key is None or not isinstance(slot_mapping, torch.Tensor):
            return None
        block_signature = cache_key[1]
        mapping_start = int(block_signature[0][1])
        mapping_end = int(block_signature[-1][2])
        relevant_mapping = slot_mapping[mapping_start:mapping_end]
        with self._dsv4_streaming_plan_cache_lock:
            entry = self._dsv4_streaming_plan_cache.get(cache_key)
            mapping_matches = bool(
                entry is not None
                and entry.slot_mapping.shape == relevant_mapping.shape
                and entry.slot_mapping.dtype == relevant_mapping.dtype
                and torch.equal(entry.slot_mapping, relevant_mapping)
            )
            if not mapping_matches:
                self._dsv4_streaming_plan_cache_misses += 1
                return None
            assert entry is not None
            self._dsv4_streaming_plan_cache.move_to_end(cache_key)
            self._dsv4_streaming_plan_cache_hits += 1
            return entry

    def _dsv4_streaming_plan_cache_put(
        self,
        cache_key: Optional[tuple[Any, ...]],
        indexer_chunks: dict[int, list[Any]],
        chunks_by_layer: dict[int, list[Any]],
        shared_raw_lba_cache: dict[str, list[Any]],
        slot_mapping: Optional[torch.Tensor],
        *,
        csa_ready: bool,
        hca_ready: bool,
    ) -> None:
        """Store one fully preflighted immutable streaming plan binding."""
        if (
            cache_key is None
            or not isinstance(slot_mapping, torch.Tensor)
            or self._dsv4_streaming_plan_cache_capacity <= 0
        ):
            return
        block_signature = cache_key[1]
        mapping_start = int(block_signature[0][1])
        mapping_end = int(block_signature[-1][2])
        entry = _DSV4StreamingPlanCacheEntry(
            slot_mapping=slot_mapping[mapping_start:mapping_end].detach().clone(),
            indexer_chunks=tuple(
                (int(layer_id), tuple(chunks))
                for layer_id, chunks in sorted(indexer_chunks.items())
            ),
            chunks_by_layer=tuple(
                (int(layer_id), tuple(chunks))
                for layer_id, chunks in sorted(chunks_by_layer.items())
            ),
            # Preserve these exact list objects.  Tutti's LBA registration has
            # an identity fast path, so copying them would throw away much of
            # the cache hit's benefit.
            shared_raw_lba_cache=shared_raw_lba_cache,
            csa_ready=bool(csa_ready),
            hca_ready=bool(hca_ready),
        )
        with self._dsv4_streaming_plan_cache_lock:
            self._dsv4_streaming_plan_cache[cache_key] = entry
            self._dsv4_streaming_plan_cache.move_to_end(cache_key)
            while (
                len(self._dsv4_streaming_plan_cache)
                > self._dsv4_streaming_plan_cache_capacity
            ):
                self._dsv4_streaming_plan_cache.popitem(last=False)

    @staticmethod
    def _dsv4_materialize_streaming_plan_cache_entry(
        entry: _DSV4StreamingPlanCacheEntry,
    ) -> tuple[dict[int, list[Any]], dict[int, list[Any]]]:
        """Create private mutable outer containers for manager registration."""
        indexer_chunks = {
            layer_id: list(chunks) for layer_id, chunks in entry.indexer_chunks
        }
        chunks_by_layer = {
            layer_id: list(chunks) for layer_id, chunks in entry.chunks_by_layer
        }
        return indexer_chunks, chunks_by_layer

    def _register_csa_attention_kv_chunks(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        disk_metas: list[Optional[DiskCacheMetadata]],
        total_tokens: int,
        req_id: str,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> tuple[bool, bool, bool]:
        """Build and register the CSA attention KV chunk map with the manager.

        All CSA, optional HCA, and native-indexer plans are preflighted before
        any I/O starts. An incomplete plan leaves every consumer uncommitted.

        Args:
            blocks: Ordered ``(key, start_token, end_token)`` triples.
            disk_metas: Per-block ``DiskCacheMetadata``; ``None`` is allowed.
            total_tokens: Total logical tokens in the request.
            req_id: Request identifier for the prefetcher's internal logging.
            slot_mapping: vLLM physical destination mapping for this request.

        Returns:
            ``(csa_ready, hca_ready, indexer_ready)`` for the unified
            layer-major consumers. HCA readiness is false when the HCA walker
            is disabled.

        Raises:
            RuntimeError: If the fully preflighted plan cannot be committed to
                the CSA/HCA manager.
        """
        register_profile_start = time.perf_counter()
        indexer_build_ms = 0.0
        csa_build_ms = 0.0
        hca_build_ms = 0.0
        lba_build_ms = 0.0
        indexer_commit_ms = 0.0
        manager_commit_ms = 0.0
        slot_mapping_ms = 0.0
        plan_cache_key_ms = 0.0
        plan_cache_hit = False
        try:
            from lmcache.v1.csa_attention_kv_prefetch_manager import (
                build_shared_raw_lba_cache,
                get_csa_attention_kv_prefetch_manager,
            )
        except ImportError:
            return False, False, False
        manager = get_csa_attention_kv_prefetch_manager()
        if manager is None:
            # Tutti loader may have been bootstrapped after post_init; retry
            # the attach lazily now that we're inside a retrieve and the
            # loader has had a chance to warm up.
            try:
                from lmcache.integration.vllm.vllm_v1_adapter import (
                    _ensure_csa_attention_kv_prefetch_attached,
                )
            except ImportError:
                return False, False, False
            manager = _ensure_csa_attention_kv_prefetch_attached(
                getattr(self, "_tutti_loader", None)
            )
            if manager is None:
                return False, False, False
        # Construct all three plans before starting any I/O. Compact-main
        # selection is an atomic contract: partial consumer activation cannot
        # safely fall back to restoring the removed groups from a full object.
        try:
            from lmcache.v1.indexer_ssd_manager import get_indexer_ssd_manager

            indexer_manager = get_indexer_ssd_manager()
        except ImportError:
            indexer_manager = None
        indexer_chunks: dict[int, list[Any]] = {}
        register_indexer: Optional[Callable[..., Any]] = None
        prepared_slot_mapping = slot_mapping
        if isinstance(slot_mapping, torch.Tensor):
            phase_start = time.perf_counter()
            try:
                # All three layer-major builders consume the same mapping.
                # Materialize it once instead of performing three blocking
                # device-to-host copies before the model forward starts.
                prepared_slot_mapping = (
                    slot_mapping.detach()
                    .to(device="cpu", dtype=torch.int64)
                    .reshape(-1)
                )
            except Exception:
                logger.warning(
                    "DSv4 unified layer-major slot mapping materialization "
                    "failed request=%s",
                    req_id,
                )
                prepared_slot_mapping = None
            slot_mapping_ms = (time.perf_counter() - phase_start) * 1000.0
        hca_requested = _env_flag("LMCACHE_DSV4_HCA_WALKER")
        if indexer_manager is not None:
            register_indexer = getattr(
                indexer_manager,
                "register_native_indexer_stream",
                None,
            )

        phase_start = time.perf_counter()
        plan_cache_key = self._dsv4_streaming_plan_cache_key(
            blocks,
            prepared_slot_mapping,
        )
        if plan_cache_key is not None:
            plan_cache_key = (
                *plan_cache_key,
                bool(hca_requested),
                tuple(int(value) for value in getattr(manager, "csa_layer_ids", ())),
                tuple(int(value) for value in getattr(manager, "hca_layer_ids", ())),
            )
        plan_cache_entry = self._dsv4_streaming_plan_cache_get(
            plan_cache_key,
            prepared_slot_mapping,
        )
        plan_cache_key_ms = (time.perf_counter() - phase_start) * 1000.0
        plan_cache_hit = plan_cache_entry is not None

        if plan_cache_entry is not None:
            indexer_chunks, chunks_by_layer = (
                self._dsv4_materialize_streaming_plan_cache_entry(plan_cache_entry)
            )
            shared_raw_lba_cache = plan_cache_entry.shared_raw_lba_cache
            csa_ready = plan_cache_entry.csa_ready
            hca_ready = plan_cache_entry.hca_ready
        else:
            if indexer_manager is not None:
                phase_start = time.perf_counter()
                indexer_chunks = self._dsv4_build_indexer_cache_chunks(
                    blocks,
                    total_tokens,
                    slot_mapping=prepared_slot_mapping,
                )
                indexer_build_ms = (time.perf_counter() - phase_start) * 1000.0

            phase_start = time.perf_counter()
            chunks_by_layer = self._dsv4_build_csa_attention_kv_chunks(
                blocks,
                disk_metas,
                total_tokens,
                slot_mapping=prepared_slot_mapping,
            )
            csa_build_ms = (time.perf_counter() - phase_start) * 1000.0
            csa_expected_end = self._dsv4_streaming_expected_layer_coverage(
                blocks,
                "csa_attention_kv",
            )
            csa_ready = self._dsv4_layer_chunk_map_complete(
                chunks_by_layer,
                getattr(manager, "csa_layer_ids", ()),
                csa_expected_end,
            )
            hca_ready = False
            if hca_requested:
                phase_start = time.perf_counter()
                hca_chunks = self._dsv4_build_hca_attention_kv_chunks(
                    blocks,
                    total_tokens,
                    slot_mapping=prepared_slot_mapping,
                )
                hca_build_ms = (time.perf_counter() - phase_start) * 1000.0
                overlapping_layers = set(chunks_by_layer) & set(hca_chunks)
                hca_expected_end = self._dsv4_streaming_expected_layer_coverage(
                    blocks,
                    "hca_attention_kv",
                )
                hca_ready = self._dsv4_layer_chunk_map_complete(
                    hca_chunks,
                    getattr(manager, "hca_layer_ids", ()),
                    hca_expected_end,
                )
                if overlapping_layers:
                    logger.warning(
                        "DSv4 unified layer-major layer conflict layers=%s",
                        sorted(overlapping_layers),
                    )
                    hca_ready = False
                else:
                    chunks_by_layer.update(hca_chunks)

            # All three plans address objects in the same rank-local raw pool.
            # Register their union once and retain the exact list objects in
            # the cache for Tutti's identity fast path.
            phase_start = time.perf_counter()
            shared_raw_lba_cache = build_shared_raw_lba_cache(
                (indexer_chunks, chunks_by_layer)
            )
            lba_build_ms = (time.perf_counter() - phase_start) * 1000.0

        preflight_ready = bool(
            csa_ready
            and (not hca_requested or hca_ready)
            and indexer_chunks
            and callable(register_indexer)
        )
        if not preflight_ready:
            logger.warning(
                "DSv4 unified layer-major preflight incomplete "
                "request=%s csa_ready=%s hca_ready=%s indexer_plan=%s",
                req_id,
                csa_ready,
                hca_ready,
                bool(indexer_chunks),
            )
            self._dsv4_deactivate_streaming_consumers(
                manager,
                indexer_manager,
            )
            return False, False, False

        if not shared_raw_lba_cache:
            logger.warning(
                "DSv4 unified layer-major extent plan is empty request=%s",
                req_id,
            )
            self._dsv4_deactivate_streaming_consumers(
                manager,
                indexer_manager,
            )
            return False, False, False

        if plan_cache_entry is None:
            self._dsv4_streaming_plan_cache_put(
                plan_cache_key,
                indexer_chunks,
                chunks_by_layer,
                shared_raw_lba_cache,
                prepared_slot_mapping,
                csa_ready=csa_ready,
                hca_ready=hca_ready,
            )

        if str(getattr(manager, "active_request_id", "")) == str(req_id):
            request_chunks_match = getattr(manager, "request_chunks_match", None)
            indexer_chunks_match = getattr(
                indexer_manager,
                "native_indexer_stream_matches",
                None,
            )
            repeated_plan_matches = bool(
                callable(request_chunks_match)
                and request_chunks_match(req_id, chunks_by_layer)
                and callable(indexer_chunks_match)
                and indexer_chunks_match(req_id, indexer_chunks)
            )
            if not repeated_plan_matches:
                logger.warning(
                    "DSv4 unified layer-major repeated request plan changed "
                    "request=%s; refusing stale registration",
                    req_id,
                )
                self._dsv4_deactivate_streaming_consumers(
                    manager,
                    indexer_manager,
                )
                return False, False, False

        # The request-level capture owner is the canonical CSA/HCA manager,
        # but compact indexer Stage0 starts first to maximize its overlap with
        # request setup and layers 0-1. Destination-row tables are prewarmed at
        # manager attachment, so this early submission no longer races a
        # first-hit CUDA allocation/upload in the CSA/HCA registration below.
        manager.start_full_nsys_capture_for_request(str(req_id))
        assert callable(register_indexer)
        try:
            phase_start = time.perf_counter()
            indexer_ready = bool(
                register_indexer(
                    req_id,
                    indexer_chunks,
                    shared_raw_lba_cache=shared_raw_lba_cache,
                )
            )
            indexer_commit_ms = (time.perf_counter() - phase_start) * 1000.0
        except Exception as exc:
            self._dsv4_deactivate_streaming_consumers(
                manager,
                indexer_manager,
            )
            raise RuntimeError("native indexer read-plan registration failed") from exc
        if not indexer_ready:
            logger.warning(
                "DSv4 unified layer-major indexer commit failed request=%s",
                req_id,
            )
            self._dsv4_deactivate_streaming_consumers(
                manager,
                indexer_manager,
            )
            return False, False, False
        try:
            phase_start = time.perf_counter()
            manager.register_request_chunks(
                req_id,
                chunks_by_layer,
                shared_raw_lba_cache=shared_raw_lba_cache,
            )
            manager_commit_ms = (time.perf_counter() - phase_start) * 1000.0
        except Exception as exc:
            self._dsv4_deactivate_streaming_consumers(
                manager,
                indexer_manager,
            )
            logger.exception(
                "Failed to register CSA attention KV chunks for request %s; "
                "filtered attention KV cannot be consumed safely",
                req_id,
            )
            raise RuntimeError(
                "CSA attention KV read-plan registration failed"
            ) from exc
        if _env_flag("LMCACHE_TUTTI_PROFILE"):
            logger.info(
                "TUTTI_PROFILE streaming_register request=%s "
                "plan_cache_hit=%s plan_cache_key_ms=%.3f "
                "slot_mapping_ms=%.3f indexer_build_ms=%.3f "
                "csa_build_ms=%.3f hca_build_ms=%.3f "
                "lba_build_ms=%.3f indexer_commit_ms=%.3f "
                "manager_commit_ms=%.3f total_ms=%.3f "
                "cache_hits=%d cache_misses=%d",
                req_id,
                plan_cache_hit,
                plan_cache_key_ms,
                slot_mapping_ms,
                indexer_build_ms,
                csa_build_ms,
                hca_build_ms,
                lba_build_ms,
                indexer_commit_ms,
                manager_commit_ms,
                (time.perf_counter() - register_profile_start) * 1000.0,
                self._dsv4_streaming_plan_cache_hits,
                self._dsv4_streaming_plan_cache_misses,
            )
        return csa_ready, hca_ready, indexer_ready

    def _dsv4_build_indexer_cache_chunks(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        total_tokens: int,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> dict[int, list[Any]]:
        """Build a complete compact layer-major native-indexer read plan.

        Unlike the legacy fallback in the CSA attention planner, this method
        accepts only compact sidecars with dense segment coverage. Returning
        an empty mapping is the safety signal that keeps the ordinary padded
        LMCache indexer group in the synchronous retrieve.

        Args:
            blocks: Ordered cached-prefix chunks for the active request.
            total_tokens: Total logical tokens in the request.
            slot_mapping: vLLM physical slot mapping for the request.

        Returns:
            Per-transformer-layer compact read descriptors, or an empty
            mapping when any layer, segment, extent, or physical mapping is
            incomplete.
        """
        del total_tokens  # Sidecars describe only the supplied cached prefix.
        try:
            from lmcache.v1.csa_attention_kv_prefetch_manager import (
                CSAAttentionKVChunkLoc,
            )
        except ImportError:
            return {}
        if not blocks or self.gpu_connector is None or self.storage_manager is None:
            return {}
        disk_backend = self.storage_manager.storage_backends.get("LocalDiskBackend")
        if disk_backend is None or not getattr(
            disk_backend,
            "kv_object_store_enabled",
            False,
        ):
            return {}
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return {}
        dtypes = self.metadata.get_dtypes()
        indexer_group: Optional[Any] = None
        indexer_dtype: Optional[torch.dtype] = None
        for group, dtype in zip(
            klg_manager.kv_layer_groups,
            dtypes,
            strict=True,
        ):
            if self._dsv4_group_role(group, dtype) == "csa_indexer_cache":
                indexer_group = group
                indexer_dtype = dtype
                break
        if indexer_group is None or indexer_dtype is None:
            return {}
        manager_layer_ids = self.gpu_connector._dsv4_layer_ids_for_group(  # noqa: SLF001
            indexer_group
        )
        object_layer_ids = [int(value) for value in indexer_group.layer_indices]
        if not manager_layer_ids or len(manager_layer_ids) != len(object_layer_ids):
            return {}
        if slot_mapping is None:
            return {}
        try:
            if isinstance(slot_mapping, torch.Tensor):
                slots_cpu = (
                    slot_mapping.detach()
                    .to(
                        device="cpu",
                        dtype=torch.int64,
                    )
                    .reshape(-1)
                )
            else:
                slots_cpu = torch.as_tensor(
                    slot_mapping,
                    dtype=torch.int64,
                ).reshape(-1)
        except Exception:
            return {}

        compression_ratio = int(indexer_group.compress_ratio)
        block_size = 64
        kv_size = int(indexer_group.shape_desc.kv_size)
        token_bytes = (
            kv_size * int(indexer_group.hidden_dim_size) * int(indexer_dtype.itemsize)
        )
        bytes_per_block = block_size * token_bytes
        tokens_per_block = compression_ratio * block_size
        if compression_ratio <= 0 or kv_size != 1 or bytes_per_block <= 0:
            return {}

        expected_start = int(blocks[0][1])
        block_ends: list[int] = []
        block_cursor = 0
        for _key, start, end in blocks:
            chunk_tokens = int(end) - int(start)
            if (
                int(start) != expected_start
                or chunk_tokens <= 0
                or chunk_tokens % tokens_per_block
            ):
                return {}
            block_cursor += chunk_tokens // tokens_per_block
            block_ends.append(block_cursor)
            expected_start = int(end)
        total_blocks = block_cursor
        positions = torch.arange(
            total_blocks, dtype=torch.int64
        ) * tokens_per_block + int(blocks[0][1])
        if positions.numel() == 0 or int(positions[-1]) >= int(slots_cpu.numel()):
            return {}
        selected_slots = slots_cpu.index_select(0, positions)
        if not bool(torch.all(selected_slots >= 0)):
            return {}
        physical_blocks = torch.div(
            selected_slots,
            tokens_per_block,
            rounding_mode="floor",
        )

        probe = getattr(
            disk_backend,
            "get_indexer_layer_major_records_for_keys",
            None,
        )
        get_layers = getattr(
            disk_backend,
            "get_indexer_layer_major_records",
            None,
        )
        if not callable(probe) or not callable(get_layers):
            return {}
        candidate_keys = [key for key, _start, _end in blocks]
        last_records = probe([candidate_keys[-1]], object_layer_ids[0])
        last_record = last_records[0] if last_records else None
        last_segments = (
            _dsv4_layer_major_record_segments(
                last_record,
                block_nbytes=bytes_per_block,
                total_blocks=total_blocks,
            )
            if last_record is not None
            else ()
        )
        if len(last_segments) > 1:
            layer_records = get_layers(candidate_keys[-1], object_layer_ids)
            result: dict[int, list[Any]] = {
                int(layer_id): [] for layer_id in manager_layer_ids
            }
            logical_geometry = tuple(
                (first_block, n_blocks)
                for first_block, n_blocks, _offset, _skip, _length in last_segments
            )
            if len(layer_records) != len(manager_layer_ids):
                return {}
            for manager_layer_id, record in zip(
                manager_layer_ids,
                layer_records,
                strict=True,
            ):
                if record is None:
                    return {}
                physical_segments = _dsv4_layer_major_record_segments(
                    record,
                    block_nbytes=bytes_per_block,
                    total_blocks=total_blocks,
                )
                if (
                    tuple(
                        (first_block, n_blocks)
                        for first_block, n_blocks, _offset, _skip, _length in (
                            physical_segments
                        )
                    )
                    != logical_geometry
                ):
                    return {}
                disk_meta = DiskCacheMetadata(
                    path=disk_backend.kv_object_tutti_path(record.pool_id),
                    size=int(record.aligned_length),
                    fmt=MemoryFormat.BINARY_BUFFER,
                    shape=torch.Size((int(record.aligned_length),)),
                    dtype=torch.uint8,
                )
                physical_blocks_tuple = tuple(int(value) for value in physical_blocks)
                for (
                    first_block,
                    n_blocks,
                    aligned_offset,
                    payload_skip,
                    read_length,
                ) in physical_segments:
                    result[int(manager_layer_id)].append(
                        CSAAttentionKVChunkLoc(
                            first_compressed_block=first_block,
                            n_compressed_blocks=n_blocks,
                            key=candidate_keys[-1],
                            disk_meta=disk_meta,
                            layer_byte_offset=aligned_offset,
                            bytes_per_block=bytes_per_block,
                            raw_extents=record.raw_extents,
                            physical_block_ids=physical_blocks_tuple[
                                first_block : first_block + n_blocks
                            ],
                            payload_skip=payload_skip,
                            read_length=read_length,
                            layer_major=True,
                        )
                    )
            logger.info(
                "DSv4 compact native indexer composed read plan "
                "layers=%d ranges=%d blocks=%d",
                len(result),
                len(last_segments),
                total_blocks,
            )
            return result
        if (
            last_record is not None
            and last_record.raw_extents
            and int(last_record.length) == total_blocks * bytes_per_block
            and len(last_record.read_ranges) == 1
        ):
            segments = [(0, total_blocks, candidate_keys[-1])]
        else:
            candidate_records = probe(candidate_keys, object_layer_ids[0])
            segment_candidates: dict[int, tuple[int, CacheEngineKey]] = {}
            for block_index, record in enumerate(candidate_records):
                if (
                    record is None
                    or not record.raw_extents
                    or len(record.read_ranges) != 1
                    or int(record.length) % bytes_per_block
                ):
                    continue
                segment_blocks = int(record.length) // bytes_per_block
                segment_end = block_ends[block_index]
                segment_start = segment_end - segment_blocks
                if segment_start < 0 or segment_end > total_blocks:
                    continue
                current = segment_candidates.get(segment_start)
                if current is None or segment_end > current[0]:
                    segment_candidates[segment_start] = (
                        segment_end,
                        candidate_keys[block_index],
                    )

            segments = []
            cursor = 0
            while cursor < total_blocks:
                candidate = segment_candidates.get(cursor)
                if candidate is None:
                    logger.warning(
                        "DSv4 compact indexer coverage failed covered=%d/%d "
                        "ready=%d keys=%d",
                        cursor,
                        total_blocks,
                        sum(record is not None for record in candidate_records),
                        len(candidate_keys),
                    )
                    return {}
                segment_end, segment_key = candidate
                segments.append((cursor, segment_end, segment_key))
                cursor = segment_end

        result: dict[int, list[Any]] = {
            int(layer_id): [] for layer_id in manager_layer_ids
        }
        for segment_start, segment_end, segment_key in segments:
            layer_records = get_layers(segment_key, object_layer_ids)
            expected_nbytes = (segment_end - segment_start) * bytes_per_block
            if len(layer_records) != len(manager_layer_ids):
                return {}
            physical_segment = tuple(
                int(value)
                for value in physical_blocks[segment_start:segment_end].tolist()
            )
            for manager_layer_id, record in zip(
                manager_layer_ids,
                layer_records,
                strict=True,
            ):
                if (
                    record is None
                    or not record.raw_extents
                    or len(record.read_ranges) != 1
                    or int(record.length) != expected_nbytes
                ):
                    return {}
                disk_meta = DiskCacheMetadata(
                    path=disk_backend.kv_object_tutti_path(record.pool_id),
                    size=int(record.aligned_length),
                    fmt=MemoryFormat.BINARY_BUFFER,
                    shape=torch.Size((int(record.aligned_length),)),
                    dtype=torch.uint8,
                )
                result[int(manager_layer_id)].append(
                    CSAAttentionKVChunkLoc(
                        first_compressed_block=segment_start,
                        n_compressed_blocks=segment_end - segment_start,
                        key=segment_key,
                        disk_meta=disk_meta,
                        layer_byte_offset=int(record.offset),
                        bytes_per_block=bytes_per_block,
                        raw_extents=record.raw_extents,
                        physical_block_ids=physical_segment,
                        read_length=int(record.aligned_length),
                        layer_major=True,
                    )
                )
        if (
            _env_flag("LMCACHE_TUTTI_PROFILE")
            or int(getattr(getattr(self, "metadata", None), "worker_id", 0)) == 0
        ):
            logger.info(
                "DSv4 compact native indexer read plan layers=%d segments=%d "
                "blocks=%d bytes_per_layer=%d",
                len(result),
                len(segments),
                total_blocks,
                total_blocks * bytes_per_block,
            )
        return result

    def _dsv4_build_csa_attention_kv_chunks(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        disk_metas: list[Optional[DiskCacheMetadata]],
        total_tokens: int,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> dict[int, list[Any]]:
        """Build the per-CSA-layer chunk map for the active request.

        The returned mapping is suitable for passing into
        :meth:`CSAAttentionKVPrefetchManager.register_request_chunks`.  Each
        entry locates the LMCache chunks that contain a given CSA layer's
        compressed attention KV bytes, plus the byte offset within each
        chunk where that layer's slab begins.

        Reads must work post-snvme-bind when filesystem paths are no longer
        mountable, so the chunk map is built against the
        :class:`KVObjectStore` raw NVMe path (synthetic ``tutti://...``
        identifier + pre-FIEMAP'd raw LBA extents).  The per-chunk byte
        offset is expressed in pool-absolute units so the prefetcher can
        pass it through ``load_chunks_to_hbm``'s ``read_ranges_per_key``
        unchanged.

        Args:
            blocks: Ordered ``(key, start_token, end_token)`` triples for the
                chunks covering the request's prefix.
            disk_metas: ``DiskCacheMetadata`` entries aligned with
                ``blocks``.  Currently unused (the kv_object_store records
                are looked up directly) but kept in the signature for API
                stability.
            total_tokens: Total logical tokens in the request.

        Returns:
            ``{transformer_layer_id: [CSAAttentionKVChunkLoc, ...]}``.
            Returns an empty dict when the layout is invalid, when no
            ``csa_attention_kv`` group exists, when the GPU connector is
            unavailable, or when the LocalDiskBackend is not using
            kv_object_store raw mode.
        """
        del disk_metas  # see docstring
        try:
            from lmcache.v1.csa_attention_kv_prefetch_manager import (
                CSAAttentionKVChunkLoc,
            )
        except ImportError:
            return {}
        if not self.dsv4_optimized_kv:
            return {}
        if self.gpu_connector is None or self.storage_manager is None:
            return {}
        disk_backend = self.storage_manager.storage_backends.get("LocalDiskBackend")
        if disk_backend is None or not getattr(
            disk_backend, "kv_object_store_enabled", False
        ):
            # The CSA attention KV prefetcher requires the kv_object_store
            # raw path because filesystem paths are unmounted after snvme
            # bind; bail out cleanly so the retrieve falls back to the
            # synchronous scatter.
            return {}
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return {}
        dtypes = self.metadata.get_dtypes()
        csa_group_idx: Optional[int] = None
        csa_group: Optional[Any] = None
        for idx, group in enumerate(klg_manager.kv_layer_groups):
            if self._dsv4_group_role(group, dtypes[idx]) == "csa_attention_kv":
                csa_group_idx = idx
                csa_group = group
                break
        if csa_group_idx is None or csa_group is None:
            return {}
        layer_ids_for_group = self.gpu_connector._dsv4_layer_ids_for_group(  # noqa: SLF001
            csa_group
        )
        if not layer_ids_for_group:
            return {}

        # Pre-materialise slot_mapping CPU view; the per-chunk physical
        # block id lookup is vectorized for layer-major records.
        slot_mapping_cpu: Optional[torch.Tensor] = None
        if slot_mapping is not None:
            try:
                if isinstance(slot_mapping, torch.Tensor):
                    slot_mapping_cpu = (
                        slot_mapping.detach()
                        .to(
                            device="cpu",
                            dtype=torch.int64,
                        )
                        .reshape(-1)
                    )
                else:
                    slot_mapping_cpu = torch.as_tensor(
                        slot_mapping,
                        dtype=torch.int64,
                    ).reshape(-1)
            except Exception:
                logger.warning(
                    "DSv4 CSA chunk builder: failed to materialise "
                    "slot_mapping; physical_block_ids will be empty"
                )
                slot_mapping_cpu = None

        compress_ratio = int(csa_group.compress_ratio)
        compressed_block_size = 64
        kv_size = int(csa_group.shape_desc.kv_size)
        csa_dtype = dtypes[csa_group_idx]
        bytes_per_token = (
            kv_size * int(csa_group.hidden_dim_size) * int(csa_dtype.itemsize)
        )
        contiguous_prefix = bool(blocks) and kv_size == 1
        segment_compatible = contiguous_prefix
        expected_start = int(blocks[0][1]) if blocks else 0
        total_rows = 0
        compressed_block_ends: list[int] = []
        compressed_block_cursor = 0
        for _key, start, end in blocks:
            if int(start) != expected_start or int(end) < int(start):
                contiguous_prefix = False
                break
            chunk_tokens = int(end) - int(start)
            if chunk_tokens % compress_ratio:
                segment_compatible = False
            chunk_rows = chunk_tokens // compress_ratio
            if chunk_rows % compressed_block_size:
                segment_compatible = False
            total_rows += chunk_rows
            compressed_block_cursor += chunk_rows // compressed_block_size
            compressed_block_ends.append(compressed_block_cursor)
            expected_start = int(end)
        total_compressed_blocks = total_rows // compressed_block_size
        tokens_per_compressed_block = compress_ratio * compressed_block_size
        object_layer_ids = [int(v) for v in csa_group.layer_indices]
        if (
            contiguous_prefix
            and segment_compatible
            and slot_mapping_cpu is not None
            and total_rows > 0
            and total_rows % compressed_block_size == 0
            and len(layer_ids_for_group) == len(object_layer_ids)
        ):
            positions = torch.arange(
                total_compressed_blocks, dtype=torch.int64
            ) * tokens_per_compressed_block + int(blocks[0][1])
            if positions.numel() > 0 and int(positions[-1]) < int(
                slot_mapping_cpu.numel()
            ):
                slots = slot_mapping_cpu.index_select(0, positions)
                if bool(torch.all(slots >= 0)):
                    physical_rows = torch.div(
                        slots,
                        tokens_per_compressed_block,
                        rounding_mode="floor",
                    )
                    probe_segments = getattr(
                        disk_backend,
                        "get_csa_layer_major_records_for_keys",
                        None,
                    )
                    if callable(probe_segments):
                        candidate_keys = [key for key, _start, _end in blocks]
                        bytes_per_compressed_block = (
                            compressed_block_size * bytes_per_token
                        )
                        last_records = probe_segments(
                            [candidate_keys[-1]],
                            object_layer_ids[0],
                        )
                        last_record = last_records[0] if last_records else None
                        last_segments = (
                            _dsv4_layer_major_record_segments(
                                last_record,
                                block_nbytes=bytes_per_compressed_block,
                                total_blocks=total_compressed_blocks,
                            )
                            if last_record is not None
                            else ()
                        )
                        if len(last_segments) > 1:
                            layer_records = disk_backend.get_csa_layer_major_records(
                                candidate_keys[-1],
                                object_layer_ids,
                            )
                            result = {
                                int(layer_id): [] for layer_id in layer_ids_for_group
                            }
                            logical_geometry = tuple(
                                (first_block, n_blocks)
                                for first_block, n_blocks, _offset, _skip, _length in (
                                    last_segments
                                )
                            )
                            valid_composed = len(layer_records) == len(
                                layer_ids_for_group
                            )
                            physical_rows_tuple = tuple(
                                int(value) for value in physical_rows
                            )
                            if valid_composed:
                                for manager_layer_id, record in zip(
                                    layer_ids_for_group,
                                    layer_records,
                                    strict=True,
                                ):
                                    if record is None:
                                        valid_composed = False
                                        break
                                    physical_segments = (
                                        _dsv4_layer_major_record_segments(
                                            record,
                                            block_nbytes=bytes_per_compressed_block,
                                            total_blocks=total_compressed_blocks,
                                        )
                                    )
                                    if (
                                        tuple(
                                            segment[:2] for segment in physical_segments
                                        )
                                        != logical_geometry
                                    ):
                                        valid_composed = False
                                        break
                                    disk_meta = DiskCacheMetadata(
                                        path=disk_backend.kv_object_tutti_path(
                                            record.pool_id
                                        ),
                                        size=int(record.aligned_length),
                                        fmt=MemoryFormat.BINARY_BUFFER,
                                        shape=torch.Size((int(record.aligned_length),)),
                                        dtype=torch.uint8,
                                    )
                                    for (
                                        first_block,
                                        n_blocks,
                                        aligned_offset,
                                        payload_skip,
                                        read_length,
                                    ) in physical_segments:
                                        result[int(manager_layer_id)].append(
                                            CSAAttentionKVChunkLoc(
                                                first_compressed_block=first_block,
                                                n_compressed_blocks=n_blocks,
                                                key=candidate_keys[-1],
                                                disk_meta=disk_meta,
                                                layer_byte_offset=aligned_offset,
                                                bytes_per_block=(
                                                    bytes_per_compressed_block
                                                ),
                                                raw_extents=record.raw_extents,
                                                physical_block_ids=(
                                                    physical_rows_tuple[
                                                        first_block : first_block
                                                        + n_blocks
                                                    ]
                                                ),
                                                payload_skip=payload_skip,
                                                read_length=read_length,
                                                layer_major=True,
                                            )
                                        )
                            if valid_composed:
                                logger.info(
                                    "DSv4 CSA composed layer-major read plan "
                                    "layers=%d ranges=%d blocks=%d",
                                    len(result),
                                    len(last_segments),
                                    total_compressed_blocks,
                                )
                                return result
                        if (
                            last_record is not None
                            and last_record.raw_extents
                            and int(last_record.length)
                            == total_compressed_blocks * bytes_per_compressed_block
                            and len(last_record.read_ranges) == 1
                        ):
                            segments = [
                                (0, total_compressed_blocks, candidate_keys[-1])
                            ]
                            expected_segment_start = total_compressed_blocks
                        else:
                            candidate_records = probe_segments(
                                candidate_keys,
                                object_layer_ids[0],
                            )
                            segment_candidates: dict[
                                int, tuple[int, CacheEngineKey]
                            ] = {}
                            for block_index, record in enumerate(candidate_records):
                                if (
                                    record is None
                                    or not record.raw_extents
                                    or len(record.read_ranges) != 1
                                ):
                                    continue
                                if int(record.length) % bytes_per_compressed_block:
                                    continue
                                segment_blocks = (
                                    int(record.length) // bytes_per_compressed_block
                                )
                                segment_end = compressed_block_ends[block_index]
                                segment_start = segment_end - segment_blocks
                                if (
                                    segment_start < 0
                                    or segment_end > total_compressed_blocks
                                ):
                                    continue
                                current = segment_candidates.get(segment_start)
                                if current is None or segment_end > current[0]:
                                    segment_candidates[segment_start] = (
                                        segment_end,
                                        candidate_keys[block_index],
                                    )

                            segments = []
                            expected_segment_start = 0
                            while expected_segment_start < total_compressed_blocks:
                                candidate = segment_candidates.get(
                                    expected_segment_start
                                )
                                if candidate is None:
                                    segments = []
                                    break
                                segment_end, segment_key = candidate
                                segments.append(
                                    (
                                        expected_segment_start,
                                        segment_end,
                                        segment_key,
                                    )
                                )
                                expected_segment_start = segment_end

                            if not segments:
                                logger.warning(
                                    "DSv4 CSA layer-major segment coverage failed "
                                    "keys=%d ready=%d candidates=%d covered=%d/%d",
                                    len(candidate_keys),
                                    sum(
                                        record is not None
                                        for record in candidate_records
                                    ),
                                    len(segment_candidates),
                                    expected_segment_start,
                                    total_compressed_blocks,
                                )

                        if (
                            segments
                            and expected_segment_start == total_compressed_blocks
                        ):
                            result = {
                                int(layer_id): [] for layer_id in layer_ids_for_group
                            }
                            valid_segments = True
                            for segment_start, segment_end, segment_key in segments:
                                layer_records = (
                                    disk_backend.get_csa_layer_major_records(
                                        segment_key,
                                        object_layer_ids,
                                    )
                                )
                                expected_nbytes = (
                                    segment_end - segment_start
                                ) * bytes_per_compressed_block
                                if len(layer_records) != len(
                                    layer_ids_for_group
                                ) or not all(
                                    record is not None
                                    and bool(record.raw_extents)
                                    and len(record.read_ranges) == 1
                                    and int(record.length) == expected_nbytes
                                    for record in layer_records
                                ):
                                    valid_segments = False
                                    break
                                segment_rows = tuple(
                                    int(v)
                                    for v in physical_rows[
                                        segment_start:segment_end
                                    ].tolist()
                                )
                                for manager_layer_id, record in zip(
                                    layer_ids_for_group,
                                    layer_records,
                                    strict=True,
                                ):
                                    assert record is not None
                                    disk_meta = DiskCacheMetadata(
                                        path=(
                                            disk_backend.kv_object_tutti_path(
                                                record.pool_id
                                            )
                                        ),
                                        size=int(record.aligned_length),
                                        fmt=MemoryFormat.BINARY_BUFFER,
                                        shape=torch.Size((int(record.aligned_length),)),
                                        dtype=torch.uint8,
                                    )
                                    result[int(manager_layer_id)].append(
                                        CSAAttentionKVChunkLoc(
                                            first_compressed_block=(segment_start),
                                            n_compressed_blocks=(
                                                segment_end - segment_start
                                            ),
                                            key=segment_key,
                                            disk_meta=disk_meta,
                                            layer_byte_offset=int(record.offset),
                                            bytes_per_block=(
                                                bytes_per_compressed_block
                                            ),
                                            raw_extents=record.raw_extents,
                                            physical_block_ids=segment_rows,
                                            read_length=int(record.aligned_length),
                                            layer_major=True,
                                        )
                                    )
                            if valid_segments:
                                if (
                                    _env_flag("LMCACHE_TUTTI_PROFILE")
                                    or int(
                                        getattr(
                                            getattr(self, "metadata", None),
                                            "worker_id",
                                            0,
                                        )
                                    )
                                    == 0
                                ):
                                    logger.info(
                                        "DSv4 CSA segmented layer-major read plan "
                                        "layers=%d segments=%d blocks=%d",
                                        len(result),
                                        len(segments),
                                        total_compressed_blocks,
                                    )
                                return result

        keys = [key for key, _, _ in blocks]
        object_records = disk_backend.get_kv_object_records(keys, roles=None)

        chunks_by_layer: dict[int, list[Any]] = {
            int(layer_id): [] for layer_id in layer_ids_for_group
        }
        compressed_blocks_cursor = 0
        for (key, start, end), record in zip(blocks, object_records, strict=True):
            if record is None:
                continue
            chunk_tokens = end - start
            if chunk_tokens <= 0:
                continue
            store_shapes = self._dsv4_store_shapes_for_range(
                self.metadata.get_shapes(chunk_tokens),
                dtypes,
                start,
                end,
                total_tokens,
            )
            # Byte offset of csa_attention_kv group inside the chunk's
            # serialised payload: sum of prior groups' byte sizes using the
            # store shapes (not the retrieve shapes, since the chunk was
            # written with the store layout).
            group_byte_offset = 0
            for prior_idx in range(csa_group_idx):
                prior_shape = store_shapes[prior_idx]
                prior_dtype = dtypes[prior_idx]
                group_byte_offset += int(prior_shape.numel()) * int(
                    prior_dtype.itemsize
                )
            csa_shape = store_shapes[csa_group_idx]
            if csa_shape.numel() <= 0:
                # This chunk has zero-shape csa_attention_kv (e.g., a non-tail
                # variant that the store path masked out).  Skip — there is
                # no source data to prefetch.
                continue
            # csa_attention_kv shape is [kv_size, num_layers_in_group, rows,
            # hidden_dim].  Compressed rows per chunk = chunk_tokens //
            # compress_ratio.
            compress_ratio = int(csa_group.compress_ratio)
            if compress_ratio <= 0:
                continue
            rows_per_layer = chunk_tokens // compress_ratio
            if rows_per_layer <= 0:
                continue
            csa_dtype = dtypes[csa_group_idx]
            hidden_dim = int(csa_shape[-1])
            bytes_per_token = hidden_dim * int(csa_dtype.itemsize)
            bytes_per_layer_in_chunk = rows_per_layer * bytes_per_token
            num_layers_in_group = int(csa_shape[1])
            # vLLM packs the K cache as [num_blocks, compressed_block_size,
            # token_bytes] with compressed_block_size == DEEPGEMM_PAGED_BLOCK_SIZE
            # (64).  Mirror that constant here so the indexer's top-K block ids
            # can index into our chunk locations.
            compressed_block_size = 64
            if rows_per_layer % compressed_block_size != 0:
                logger.warning(
                    "DSv4 CSA chunk %s has %d compressed rows per layer, not a "
                    "multiple of compressed_block_size=%d; skipping",
                    key.to_string() if hasattr(key, "to_string") else repr(key),
                    rows_per_layer,
                    compressed_block_size,
                )
                continue
            blocks_per_layer_in_chunk = rows_per_layer // compressed_block_size
            bytes_per_compressed_block = compressed_block_size * bytes_per_token

            # Compute the physical K-cache row(s) this chunk maps to.  Each
            # row of the CSA K cache holds ``compress_ratio *
            # compressed_block_size`` source tokens (= 256 for DSv4
            # csa_attention_kv).  ``slot_mapping[t] // 256`` gives the
            # physical block id assigned by vLLM's block allocator.  When
            # ``slot_mapping`` is unavailable we fall back to the
            # sequence-position index in the prefetcher's _issue_reads,
            # which only happens to be correct for fresh per-request CSA
            # caches without a block table.
            tokens_per_compressed_block = compress_ratio * compressed_block_size
            chunk_physical_block_ids: tuple[int, ...] = ()
            if slot_mapping_cpu is not None:
                ids: list[int] = []
                base = int(start)
                for block_local in range(blocks_per_layer_in_chunk):
                    pos = base + block_local * tokens_per_compressed_block
                    if 0 <= pos < len(slot_mapping_cpu):
                        slot = int(slot_mapping_cpu[pos])
                        if slot >= 0:
                            ids.append(slot // tokens_per_compressed_block)
                if len(ids) == blocks_per_layer_in_chunk:
                    chunk_physical_block_ids = tuple(ids)

            # Locate the chunk's kv_object_store record so reads can resolve
            # post-snvme-bind via the synthetic ``tutti://...`` path that
            # was registered in TuttiDirectLoader's _lba_cache.
            #
            # Byte semantics: ``_logical_read_ranges`` in the Tutti loader
            # returns ``read_ranges_per_key`` verbatim and IGNORES
            # ``file_offsets`` when ranges are present, so the offset we put
            # in :class:`KVObjectByteRange` must already be pool-absolute
            # (the registered LBA extents' ``file_offset`` is pool-absolute
            # too).  ``_tutti_batched_get`` adds ``record.offset`` for its
            # own read_ranges; the prefetcher calls
            # ``load_chunks_to_hbm`` directly so we add it here ourselves.
            try:
                synth_path = disk_backend.kv_object_tutti_path(record.pool_id)
            except Exception:
                continue
            if not synth_path or not record.raw_extents:
                continue
            pool_byte_offset = int(record.offset) + group_byte_offset
            synth_disk_meta = DiskCacheMetadata(
                path=synth_path,
                size=int(record.aligned_length),
                fmt=MemoryFormat.BINARY_BUFFER,
                shape=torch.Size((int(record.aligned_length),)),
                dtype=torch.uint8,
            )
            record_end = int(record.offset) + int(record.aligned_length)
            for layer_slot_idx, transformer_layer_id in enumerate(layer_ids_for_group):
                if layer_slot_idx >= num_layers_in_group:
                    break
                layer_byte_offset_in_pool = (
                    pool_byte_offset + layer_slot_idx * bytes_per_layer_in_chunk
                )
                layer_byte_end_in_pool = (
                    layer_byte_offset_in_pool
                    + blocks_per_layer_in_chunk * bytes_per_compressed_block
                )
                # Skip layers whose final byte falls past the chunk's
                # registered aligned_length.  Tutti aborts the whole batch
                # when ANY byte_range is not fully covered by the
                # registered LBA extents, so admitting an out-of-range
                # layer here would silently destroy the entire layer-60
                # miss-correction batch.
                if layer_byte_end_in_pool > record_end:
                    logger.warning(
                        "DSv4 CSA chunk %s layer slot %d byte range "
                        "[%d, %d) past record aligned_length end %d; "
                        "skipping",
                        key.to_string() if hasattr(key, "to_string") else repr(key),
                        layer_slot_idx,
                        layer_byte_offset_in_pool,
                        layer_byte_end_in_pool,
                        record_end,
                    )
                    continue
                chunks_by_layer[int(transformer_layer_id)].append(
                    CSAAttentionKVChunkLoc(
                        first_compressed_block=compressed_blocks_cursor,
                        n_compressed_blocks=blocks_per_layer_in_chunk,
                        key=key,
                        disk_meta=synth_disk_meta,
                        layer_byte_offset=layer_byte_offset_in_pool,
                        bytes_per_block=bytes_per_compressed_block,
                        raw_extents=record.raw_extents,
                        physical_block_ids=chunk_physical_block_ids,
                    )
                )
            compressed_blocks_cursor += blocks_per_layer_in_chunk
        return chunks_by_layer

    def _dsv4_build_hca_attention_kv_chunks(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        total_tokens: int,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> dict[int, list[Any]]:
        """Build per-HCA-layer chunk locations for the V28 walker.

        Mirrors :meth:`_dsv4_build_csa_attention_kv_chunks` for the
        ``hca_attention_kv`` group with two structural differences:

        * **Granularity**: HCA compresses 128:1, so a 256-token chunk holds
          only 2 compressed entries and vLLM packs the HCA K cache as
          ``[num_blocks, 8, token_bytes]`` — four chunks share one physical
          block.  The prefetcher therefore addresses COMPRESSED-ENTRY rows:
          ``n_compressed_blocks`` counts entries, ``bytes_per_block`` is one
          entry (584 bytes), and ``physical_block_ids`` carries flattened
          row ids ``slot_block_id * 8 + entry_slot`` for a
          ``k_cache.view(num_blocks * 8, token_bytes)`` scatter target.
        * **Alignment**: the per-layer slab stride (2 * 584 = 1168 bytes)
          is never 512B-aligned and Tutti rejects unaligned reads, so each
          chunk's read window is rounded down to a 512B boundary with the
          true payload offset carried in ``payload_skip`` and the rounded
          length in ``read_length``.

        Args:
            blocks: Ordered ``(key, start_token, end_token)`` triples for
                the chunks covering the request's prefix.
            total_tokens: Total logical tokens in the request.
            slot_mapping: vLLM slot mapping for the request; required for
                physical row resolution (entries without it are skipped
                because HCA has no safe sequence-position fallback).

        Returns:
            ``{transformer_layer_id: [CSAAttentionKVChunkLoc, ...]}`` for
            the HCA layers, or an empty dict when the layout, backend, or
            slot mapping is unavailable.
        """
        try:
            from lmcache.v1.csa_attention_kv_prefetch_manager import (
                CSAAttentionKVChunkLoc,
            )
        except ImportError:
            return {}
        if not self.dsv4_optimized_kv:
            return {}
        if self.gpu_connector is None or self.storage_manager is None:
            return {}
        disk_backend = self.storage_manager.storage_backends.get("LocalDiskBackend")
        if disk_backend is None or not getattr(
            disk_backend, "kv_object_store_enabled", False
        ):
            return {}
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return {}
        dtypes = self.metadata.get_dtypes()
        hca_group_idx: Optional[int] = None
        hca_group: Optional[Any] = None
        for idx, group in enumerate(klg_manager.kv_layer_groups):
            if self._dsv4_group_role(group, dtypes[idx]) == "hca_attention_kv":
                hca_group_idx = idx
                hca_group = group
                break
        if hca_group_idx is None or hca_group is None:
            return {}
        layer_pairs = tuple(self.gpu_connector.dsv4_hca_layer_object_ids())
        if layer_pairs:
            layer_ids_for_group = [int(pair[0]) for pair in layer_pairs]
            object_layer_ids = [int(pair[1]) for pair in layer_pairs]
        else:
            layer_ids_for_group = self.gpu_connector._dsv4_layer_ids_for_group(  # noqa: SLF001
                hca_group
            )
            object_layer_ids = [int(v) for v in hca_group.layer_indices]
        if not layer_ids_for_group:
            return {}
        if slot_mapping is None:
            logger.warning(
                "DSv4 HCA chunk builder: slot_mapping unavailable; HCA "
                "walker registration skipped (no positional fallback)"
            )
            return {}
        try:
            if isinstance(slot_mapping, torch.Tensor):
                slot_mapping_cpu = (
                    slot_mapping.detach()
                    .to(
                        device="cpu",
                        dtype=torch.int64,
                    )
                    .reshape(-1)
                )
            else:
                slot_mapping_cpu = torch.as_tensor(
                    slot_mapping,
                    dtype=torch.int64,
                ).reshape(-1)
        except Exception:
            logger.warning("DSv4 HCA chunk builder: failed to materialise slot_mapping")
            return {}

        keys = [key for key, _, _ in blocks]

        compress_ratio = int(hca_group.compress_ratio)
        if compress_ratio <= 0:
            return {}
        # vLLM packs the HCA K cache as [num_blocks, hca_block_size,
        # token_bytes]; HMA layout probing measured hca_block_size == 8
        # entries (1024 tokens per physical block at cr=128).
        hca_block_size = int(hca_group.shape_desc.bs)
        if hca_block_size <= 0:
            return {}
        tokens_per_entry = compress_ratio
        tokens_per_physical_block = compress_ratio * hca_block_size

        # Preferred V28 layout: one content-addressed object per HCA layer
        # containing all compressed entries for the cached prefix.  Resolve
        # every destination row in native tensor operations, then describe
        # the layer with one logical NVMe range.  No 1,874-chunk Python plan
        # is built on the hit path.
        contiguous_prefix = bool(blocks)
        expected_start = int(blocks[0][1]) if blocks else 0
        total_entries = 0
        for _key, start, end in blocks:
            if int(start) != expected_start or int(end) < int(start):
                contiguous_prefix = False
                break
            chunk_entries = (int(end) - int(start)) // compress_ratio
            total_entries += chunk_entries
            expected_start = int(end)
        kv_size = int(hca_group.shape_desc.kv_size)
        token_bytes = (
            kv_size
            * int(hca_group.hidden_dim_size)
            * int(dtypes[hca_group_idx].itemsize)
        )
        if (
            contiguous_prefix
            and total_entries > 0
            and len(layer_ids_for_group) == len(object_layer_ids)
            and int(blocks[0][1]) + (total_entries - 1) * compress_ratio
            < int(slot_mapping_cpu.numel())
        ):
            positions = torch.arange(
                total_entries, dtype=torch.int64
            ) * compress_ratio + int(blocks[0][1])
            slots = slot_mapping_cpu.index_select(0, positions)
            if bool(torch.all(slots >= 0)):
                entry_ids = torch.arange(total_entries, dtype=torch.int64)
                physical_rows = torch.div(
                    slots,
                    tokens_per_physical_block,
                    rounding_mode="floor",
                ) * hca_block_size + torch.remainder(entry_ids, hca_block_size)
                probe_segments = getattr(
                    disk_backend,
                    "get_hca_layer_major_records_for_keys",
                    None,
                )
                if callable(probe_segments):
                    candidate_ends: list[int] = []
                    entry_cursor = 0
                    for _key, start, end in blocks:
                        entry_cursor += (int(end) - int(start)) // compress_ratio
                        candidate_ends.append(entry_cursor)
                    last_records = probe_segments(
                        [keys[-1]],
                        object_layer_ids[0],
                    )
                    last_record = last_records[0] if last_records else None
                    if (
                        last_record is not None
                        and last_record.raw_extents
                        and int(last_record.length) == total_entries * token_bytes
                        and len(last_record.read_ranges) > 1
                    ):
                        # A composed generation is logically contiguous but
                        # its base and suffix live at independent pool
                        # offsets.  Treating ``raw_extents`` as one continuous
                        # file range works until the first HCA layer whose
                        # rounded read crosses the base/suffix boundary, then
                        # Tutti cannot resolve any bytes.  Preserve each
                        # explicit source range as a layer-major chunk and let
                        # the existing multi-generation scatter concatenate
                        # them in logical target order.
                        composed_records = disk_backend.get_hca_layer_major_records(
                            keys[-1],
                            object_layer_ids,
                        )
                        composed_chunks: dict[int, list[Any]] = {
                            int(layer_id): [] for layer_id in layer_ids_for_group
                        }
                        composed_valid = len(composed_records) == len(
                            layer_ids_for_group
                        )
                        expected_target = 0
                        reference_ranges = tuple(last_record.read_ranges)
                        for byte_range in reference_ranges:
                            if (
                                int(byte_range.target_offset) != expected_target
                                or int(byte_range.target_offset) % token_bytes
                                or int(byte_range.length) % token_bytes
                            ):
                                composed_valid = False
                                break
                            expected_target += int(byte_range.length)
                        composed_valid = bool(
                            composed_valid
                            and expected_target == total_entries * token_bytes
                        )
                        if composed_valid:
                            physical_rows_tuple = tuple(
                                int(value) for value in physical_rows.tolist()
                            )
                            for manager_layer_id, record in zip(
                                layer_ids_for_group,
                                composed_records,
                                strict=True,
                            ):
                                if (
                                    record is None
                                    or not record.raw_extents
                                    or int(record.length) != total_entries * token_bytes
                                    or tuple(
                                        (
                                            int(item.target_offset),
                                            int(item.length),
                                        )
                                        for item in record.read_ranges
                                    )
                                    != tuple(
                                        (
                                            int(item.target_offset),
                                            int(item.length),
                                        )
                                        for item in reference_ranges
                                    )
                                ):
                                    composed_valid = False
                                    break
                                synth_disk_meta = DiskCacheMetadata(
                                    path=disk_backend.kv_object_tutti_path(
                                        record.pool_id
                                    ),
                                    size=int(record.aligned_length),
                                    fmt=MemoryFormat.BINARY_BUFFER,
                                    shape=torch.Size((int(record.aligned_length),)),
                                    dtype=torch.uint8,
                                )
                                for byte_range in record.read_ranges:
                                    first_entry = (
                                        int(byte_range.target_offset) // token_bytes
                                    )
                                    n_entries = int(byte_range.length) // token_bytes
                                    source_offset = int(byte_range.offset)
                                    aligned_offset = source_offset & ~511
                                    payload_skip = source_offset - aligned_offset
                                    read_length = (
                                        (payload_skip + int(byte_range.length) + 511)
                                        // 512
                                    ) * 512
                                    composed_chunks[int(manager_layer_id)].append(
                                        CSAAttentionKVChunkLoc(
                                            first_compressed_block=first_entry,
                                            n_compressed_blocks=n_entries,
                                            key=keys[-1],
                                            disk_meta=synth_disk_meta,
                                            layer_byte_offset=aligned_offset,
                                            bytes_per_block=token_bytes,
                                            raw_extents=record.raw_extents,
                                            physical_block_ids=(
                                                physical_rows_tuple[
                                                    first_entry : first_entry
                                                    + n_entries
                                                ]
                                            ),
                                            payload_skip=payload_skip,
                                            read_length=read_length,
                                            layer_major=True,
                                        )
                                    )
                        if composed_valid:
                            logger.info(
                                "DSv4 HCA composed layer-major read plan "
                                "layers=%d ranges=%d entries=%d",
                                len(composed_chunks),
                                len(reference_ranges),
                                total_entries,
                            )
                            return composed_chunks
                    if (
                        last_record is not None
                        and last_record.raw_extents
                        and int(last_record.length) == total_entries * token_bytes
                        and len(last_record.read_ranges) == 1
                    ):
                        segments = [(0, total_entries, keys[-1])]
                        expected_start = total_entries
                    else:
                        candidate_records = probe_segments(
                            keys,
                            object_layer_ids[0],
                        )
                        segment_candidates: dict[int, tuple[int, CacheEngineKey]] = {}
                        for index, record in enumerate(candidate_records):
                            if (
                                record is None
                                or not record.raw_extents
                                or len(record.read_ranges) != 1
                            ):
                                continue
                            if int(record.length) % token_bytes:
                                continue
                            segment_entries = int(record.length) // token_bytes
                            segment_end = candidate_ends[index]
                            segment_start = segment_end - segment_entries
                            if segment_start < 0 or segment_end > total_entries:
                                continue
                            current = segment_candidates.get(segment_start)
                            if current is None or segment_end > current[0]:
                                segment_candidates[segment_start] = (
                                    segment_end,
                                    keys[index],
                                )

                        segments = []
                        expected_start = 0
                        while expected_start < total_entries:
                            candidate = segment_candidates.get(expected_start)
                            if candidate is None:
                                segments = []
                                break
                            segment_end, segment_key = candidate
                            segments.append((expected_start, segment_end, segment_key))
                            expected_start = segment_end

                        if not segments:
                            logger.warning(
                                "DSv4 HCA layer-major segment coverage failed "
                                "keys=%d ready=%d candidates=%d covered=%d/%d",
                                len(keys),
                                sum(record is not None for record in candidate_records),
                                len(segment_candidates),
                                expected_start,
                                total_entries,
                            )

                    if segments and expected_start == total_entries:
                        segmented_chunks: dict[int, list[Any]] = {
                            int(layer_id): [] for layer_id in layer_ids_for_group
                        }
                        valid_segments = True
                        for segment_start, segment_end, segment_key in segments:
                            layer_records = disk_backend.get_hca_layer_major_records(
                                segment_key,
                                object_layer_ids,
                            )
                            expected_nbytes = (
                                segment_end - segment_start
                            ) * token_bytes
                            if len(layer_records) != len(
                                layer_ids_for_group
                            ) or not all(
                                record is not None
                                and bool(record.raw_extents)
                                and len(record.read_ranges) == 1
                                and int(record.length) == expected_nbytes
                                for record in layer_records
                            ):
                                valid_segments = False
                                break
                            segment_rows = tuple(
                                int(value)
                                for value in physical_rows[
                                    segment_start:segment_end
                                ].tolist()
                            )
                            for manager_layer_id, record in zip(
                                layer_ids_for_group,
                                layer_records,
                                strict=True,
                            ):
                                assert record is not None
                                segmented_chunks[int(manager_layer_id)].append(
                                    CSAAttentionKVChunkLoc(
                                        first_compressed_block=segment_start,
                                        n_compressed_blocks=(
                                            segment_end - segment_start
                                        ),
                                        key=segment_key,
                                        disk_meta=DiskCacheMetadata(
                                            path=(
                                                disk_backend.kv_object_tutti_path(
                                                    record.pool_id
                                                )
                                            ),
                                            size=int(record.aligned_length),
                                            fmt=MemoryFormat.BINARY_BUFFER,
                                            shape=torch.Size(
                                                (int(record.aligned_length),)
                                            ),
                                            dtype=torch.uint8,
                                        ),
                                        layer_byte_offset=int(record.offset),
                                        bytes_per_block=token_bytes,
                                        raw_extents=record.raw_extents,
                                        physical_block_ids=segment_rows,
                                        read_length=int(record.aligned_length),
                                        layer_major=True,
                                    )
                                )
                        if valid_segments:
                            if (
                                _env_flag("LMCACHE_TUTTI_PROFILE")
                                or int(
                                    getattr(
                                        getattr(self, "metadata", None),
                                        "worker_id",
                                        0,
                                    )
                                )
                                == 0
                            ):
                                logger.info(
                                    "DSv4 HCA segmented layer-major read plan "
                                    "layers=%d segments=%d entries=%d",
                                    len(segmented_chunks),
                                    len(segments),
                                    total_entries,
                                )
                            return segmented_chunks
                prefix_key = blocks[-1][0]
                layer_records = disk_backend.get_hca_layer_major_records(
                    prefix_key,
                    object_layer_ids,
                )
                expected_nbytes = total_entries * token_bytes
                if len(layer_records) == len(layer_ids_for_group) and all(
                    record is not None
                    and bool(record.raw_extents)
                    and len(record.read_ranges) == 1
                    and int(record.length) == expected_nbytes
                    for record in layer_records
                ):
                    physical_block_ids = tuple(int(v) for v in physical_rows.tolist())
                    layer_major_chunks: dict[int, list[Any]] = {}
                    for manager_layer_id, record in zip(
                        layer_ids_for_group,
                        layer_records,
                        strict=True,
                    ):
                        assert record is not None
                        synth_path = disk_backend.kv_object_tutti_path(record.pool_id)
                        synth_disk_meta = DiskCacheMetadata(
                            path=synth_path,
                            size=int(record.aligned_length),
                            fmt=MemoryFormat.BINARY_BUFFER,
                            shape=torch.Size((int(record.aligned_length),)),
                            dtype=torch.uint8,
                        )
                        layer_major_chunks[int(manager_layer_id)] = [
                            CSAAttentionKVChunkLoc(
                                first_compressed_block=0,
                                n_compressed_blocks=total_entries,
                                key=prefix_key,
                                disk_meta=synth_disk_meta,
                                layer_byte_offset=int(record.offset),
                                bytes_per_block=token_bytes,
                                raw_extents=record.raw_extents,
                                physical_block_ids=physical_block_ids,
                                read_length=int(record.aligned_length),
                                layer_major=True,
                            )
                        ]
                    logger.info(
                        "DSv4 HCA layer-major read plan key=%s layers=%d "
                        "entries=%d bytes_per_layer=%d",
                        prefix_key.to_string(),
                        len(layer_major_chunks),
                        total_entries,
                        expected_nbytes,
                    )
                    return layer_major_chunks
                logger.warning(
                    "DSv4 HCA layer-major snapshot unavailable for key=%s; "
                    "falling back to aligned chunk-major records",
                    prefix_key.to_string(),
                )

        object_records = disk_backend.get_kv_object_records(keys, roles=None)
        chunks_by_layer: dict[int, list[Any]] = {
            int(layer_id): [] for layer_id in layer_ids_for_group
        }
        entries_cursor = 0
        for (key, start, end), record in zip(blocks, object_records, strict=True):
            if record is None:
                continue
            chunk_tokens = end - start
            if chunk_tokens <= 0:
                continue
            store_shapes = self._dsv4_store_shapes_for_range(
                self.metadata.get_shapes(chunk_tokens),
                dtypes,
                start,
                end,
                total_tokens,
            )
            group_byte_offset = 0
            for prior_idx in range(hca_group_idx):
                prior_shape = store_shapes[prior_idx]
                prior_dtype = dtypes[prior_idx]
                group_byte_offset += int(prior_shape.numel()) * int(
                    prior_dtype.itemsize
                )
            hca_shape = store_shapes[hca_group_idx]
            if hca_shape.numel() <= 0:
                continue
            entries_per_layer = chunk_tokens // compress_ratio
            if entries_per_layer <= 0:
                continue
            hca_dtype = dtypes[hca_group_idx]
            token_bytes = int(hca_shape[-1]) * int(hca_dtype.itemsize)
            bytes_per_layer_in_chunk = entries_per_layer * token_bytes
            num_layers_in_group = int(hca_shape[1])

            # Physical flattened row ids: one per compressed entry.  Entry
            # slot within its physical block is the GLOBAL entry index
            # modulo the block size (allocation order), and the block id
            # comes from slot_mapping at the entry's first source token.
            entry_row_ids: list[int] = []
            base = int(start)
            for entry_local in range(entries_per_layer):
                global_entry = entries_cursor + entry_local
                pos = base + entry_local * tokens_per_entry
                if not 0 <= pos < len(slot_mapping_cpu):
                    break
                slot = int(slot_mapping_cpu[pos])
                if slot < 0:
                    break
                block_id = slot // tokens_per_physical_block
                entry_slot = global_entry % hca_block_size
                entry_row_ids.append(block_id * hca_block_size + entry_slot)
            if len(entry_row_ids) != entries_per_layer:
                entries_cursor += entries_per_layer
                continue

            try:
                synth_path = disk_backend.kv_object_tutti_path(record.pool_id)
            except Exception:
                entries_cursor += entries_per_layer
                continue
            if not synth_path or not record.raw_extents:
                entries_cursor += entries_per_layer
                continue
            pool_byte_offset = int(record.offset) + group_byte_offset
            synth_disk_meta = DiskCacheMetadata(
                path=synth_path,
                size=int(record.aligned_length),
                fmt=MemoryFormat.BINARY_BUFFER,
                shape=torch.Size((int(record.aligned_length),)),
                dtype=torch.uint8,
            )
            record_end = int(record.offset) + int(record.aligned_length)
            raw_extents = tuple(
                (int(fo), int(slba), int(n_sectors))
                for fo, slba, n_sectors in record.raw_extents
            )
            for layer_slot_idx, transformer_layer_id in enumerate(layer_ids_for_group):
                if layer_slot_idx >= num_layers_in_group:
                    break
                true_offset = (
                    pool_byte_offset + layer_slot_idx * bytes_per_layer_in_chunk
                )
                aligned_offset = true_offset & ~511
                payload_skip = true_offset - aligned_offset
                payload_len = entries_per_layer * token_bytes
                read_length = ((payload_skip + payload_len + 511) // 512) * 512
                if aligned_offset + read_length > record_end:
                    # Clamp the rounded-up window to the record end; the
                    # loader treats tail ranges leniently but overrunning
                    # the registered extents aborts the whole batch.
                    read_length = record_end - aligned_offset
                    if read_length < payload_skip + payload_len:
                        continue
                chunks_by_layer[int(transformer_layer_id)].append(
                    CSAAttentionKVChunkLoc(
                        first_compressed_block=entries_cursor,
                        n_compressed_blocks=entries_per_layer,
                        key=key,
                        disk_meta=synth_disk_meta,
                        layer_byte_offset=aligned_offset,
                        bytes_per_block=token_bytes,
                        raw_extents=raw_extents,
                        physical_block_ids=tuple(entry_row_ids),
                        payload_skip=payload_skip,
                        read_length=read_length,
                    )
                )
            entries_cursor += entries_per_layer
        return chunks_by_layer

    def _dsv4_store_shapes_for_range(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        start: int,
        end: int,
        total_tokens: int,
        *,
        keep_request_tail: bool = True,
    ) -> list[torch.Size]:
        if not self.dsv4_optimized_kv:
            return shapes
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return shapes
        tail_start = max(0, total_tokens - self.dsv4_optimized_tail_tokens)
        # Chunked prefill presents a growing request prefix on every scheduler
        # step.  Only the final step contains the real request tail; treating
        # every intermediate batch boundary as the tail creates one oversized
        # compact-main object per scheduler batch and breaks chunk shapes on
        # lookup.
        keep_tail_groups = keep_request_tail and end > tail_start
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

    def _dsv4_hca_defer_retrieve_enabled(self, total_tokens: int) -> bool:
        """Return whether retrieve should leave HCA payloads for overlap."""
        if not self._dsv4_hca_defer_requested(total_tokens):
            return False
        manager = self._dsv4_hca_object_source_manager()
        return manager is not None

    def _dsv4_hca_object_source_manager(self) -> Any | None:
        """Return the active HCA manager when object-source reads are enabled."""
        try:
            from lmcache.v1.hca_prefetch_manager import get_hca_prefetch_manager
        except ImportError:
            return None
        manager = get_hca_prefetch_manager()
        if manager is None:
            return None
        object_source_enabled = getattr(manager, "object_source_enabled", None)
        if callable(object_source_enabled) and not object_source_enabled():
            return None
        set_source = getattr(manager, "set_layer_object_source", None)
        if not callable(set_source):
            return None
        return manager

    def _dsv4_hca_object_source_layer_pairs(
        self,
        manager: Any,
        **kwargs: Any,
    ) -> tuple[tuple[int, int], ...]:
        """Return ``(manager_layer_id, object_layer_id)`` HCA mappings."""
        connector_pairs: tuple[tuple[int, int], ...] = ()
        if self.gpu_connector is not None:
            get_layer_pairs = getattr(
                self.gpu_connector,
                "dsv4_hca_layer_object_ids",
                None,
            )
            if callable(get_layer_pairs):
                try:
                    connector_pairs = tuple(
                        (int(manager_layer_id), int(object_layer_id))
                        for manager_layer_id, object_layer_id in (
                            get_layer_pairs(**kwargs)
                        )
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to resolve DSv4 HCA layer object ids: %s",
                        exc,
                    )
                    connector_pairs = ()
        registered_layer_ids = getattr(manager, "registered_layer_ids", None)
        registered = (
            tuple(int(v) for v in registered_layer_ids())
            if callable(registered_layer_ids)
            else ()
        )
        if connector_pairs and registered:
            registered_set = set(registered)
            filtered = tuple(
                pair for pair in connector_pairs if pair[0] in registered_set
            )
            return filtered or connector_pairs
        if connector_pairs:
            return connector_pairs
        return tuple((layer_id, layer_id) for layer_id in registered)

    def _dsv4_hca_slab_layer_index_map(
        self,
        layer_pairs: tuple[tuple[int, int], ...],
    ) -> dict[int, tuple[int, int]]:
        """Return manager layer id to ``(slab_index, slab_layers)`` mapping."""
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return {}
        dtypes = self.metadata.get_dtypes()
        result: dict[int, tuple[int, int]] = {}
        for group, dtype in zip(
            klg_manager.kv_layer_groups,
            dtypes,
            strict=False,
        ):
            if self._dsv4_group_role(group, dtype) != "hca_attention_kv":
                continue
            group_layer_ids = tuple(int(layer_id) for layer_id in group.layer_indices)
            layer_index_by_id = {
                object_layer_id: index
                for index, object_layer_id in enumerate(group_layer_ids)
            }
            num_layers = len(group_layer_ids)
            for manager_layer_id, object_layer_id in layer_pairs:
                slab_index = layer_index_by_id.get(int(object_layer_id))
                if slab_index is None:
                    continue
                result[int(manager_layer_id)] = (slab_index, num_layers)
        return result

    def _dsv4_hca_object_source_available(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        manager: Any,
        disk_backend: Any,
        **kwargs: Any,
    ) -> bool:
        """Return whether the current LocalDisk hit has HCA object views."""
        layer_pairs = self._dsv4_hca_object_source_layer_pairs(manager, **kwargs)
        if not blocks or not layer_pairs:
            return False
        keys = [key for key, _start, _end in blocks]
        slab_records = disk_backend.get_kv_object_records(
            keys,
            roles=["hca_attention_kv_slab"] * len(keys),
        )
        slab_map = self._dsv4_hca_slab_layer_index_map(layer_pairs)
        if (
            slab_map
            and len(slab_map) == len(layer_pairs)
            and slab_records
            and all(record is not None for record in slab_records)
        ):
            for record in slab_records:
                if not self._dsv4_kv_object_record_readable(disk_backend, record):
                    break
            else:
                return True
        for _manager_layer_id, object_layer_id in layer_pairs:
            records = disk_backend.get_kv_object_records(
                keys,
                layer_ids=[object_layer_id] * len(keys),
                roles=["hca_attention_kv"] * len(keys),
            )
            if not records or any(record is None for record in records):
                logger.info(
                    "HCAPrefetchManager: HCA object-source unavailable "
                    "object_layer=%d blocks=%d missing=%d",
                    object_layer_id,
                    len(blocks),
                    len(blocks)
                    if not records
                    else sum(1 for record in records if record is None),
                )
                return False
            if any(
                not self._dsv4_kv_object_record_readable(disk_backend, record)
                for record in records
            ):
                logger.info(
                    "HCAPrefetchManager: HCA object-source unavailable "
                    "object_layer=%d blocks=%d reason=not_tutti_readable",
                    object_layer_id,
                    len(blocks),
                )
                return False
        return True

    @staticmethod
    def _dsv4_kv_object_record_readable(
        disk_backend: Any,
        record: Optional[KVObjectRecord],
    ) -> bool:
        """Return whether ``record`` is usable by the active object read path."""
        if record is None:
            return False
        is_readable = getattr(disk_backend, "kv_object_record_raw_readable", None)
        if callable(is_readable):
            return bool(is_readable(record))
        return True

    def _dsv4_hca_deferred_retrieve_available(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        disk_backend: Any,
    ) -> bool:
        """Return whether compact non-HCA retrieve objects are ready."""
        if not blocks:
            return False
        keys = [key for key, _start, _end in blocks]
        records = disk_backend.get_kv_object_records(
            keys,
            roles=[_DSV4_HCA_DEFERRED_RETRIEVE_ROLE] * len(keys),
        )
        if not records or any(record is None for record in records):
            logger.info(
                "HCAPrefetchManager: compact HCA-deferred retrieve unavailable "
                "blocks=%d missing=%d",
                len(blocks),
                len(blocks)
                if not records
                else sum(1 for record in records if record is None),
            )
            return False
        for record in records:
            if not self._dsv4_kv_object_record_readable(disk_backend, record):
                logger.info(
                    "HCAPrefetchManager: compact HCA-deferred retrieve "
                    "unavailable reason=not_tutti_readable blocks=%d",
                    len(blocks),
                )
                return False
        return True

    def _dsv4_streaming_compact_retrieve_available(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        disk_backend: Any,
        role: str,
        description: str,
    ) -> bool:
        """Return whether physical and metadata-only main entries are ready."""
        if not blocks:
            return False
        keys = [key for key, _start, _end in blocks]
        get_lengths = getattr(disk_backend, "get_kv_object_payload_lengths", None)
        if not callable(get_lengths):
            return False
        lengths = get_lengths(keys, roles=[role] * len(keys))
        records = disk_backend.get_kv_object_records(
            keys,
            roles=[role] * len(keys),
        )
        if len(lengths) != len(keys) or len(records) != len(keys):
            return False
        missing = sum(length is None for length in lengths)
        if missing:
            logger.info(
                "%s unavailable blocks=%d missing=%d empty=%d",
                description,
                len(blocks),
                missing,
                sum(length == 0 for length in lengths),
            )
            return False
        for length, record in zip(lengths, records, strict=True):
            if length == 0:
                continue
            if length is None or not self._dsv4_kv_object_record_readable(
                disk_backend,
                record,
            ):
                logger.info(
                    "%s unavailable reason=not_tutti_readable blocks=%d",
                    description,
                    len(blocks),
                )
                return False
        return True

    def _dsv4_csa_deferred_retrieve_available(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        disk_backend: Any,
    ) -> bool:
        """Return whether compact non-CSA retrieve objects are ready."""
        return self._dsv4_streaming_compact_retrieve_available(
            blocks,
            disk_backend,
            _DSV4_CSA_DEFERRED_RETRIEVE_ROLE,
            "CSAAttentionKVPrefetchManager: compact non-CSA retrieve",
        )

    def _dsv4_csa_hca_deferred_retrieve_available(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        disk_backend: Any,
    ) -> bool:
        """Return whether compact non-CSA/non-HCA objects are ready."""
        return self._dsv4_streaming_compact_retrieve_available(
            blocks,
            disk_backend,
            _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE,
            "CSA/HCA prefetch: compact non-CSA/non-HCA retrieve",
        )

    def _dsv4_hca_deferred_prefix_len(
        self,
        blocks: list[tuple[int, int, CacheEngineKey]],
        disk_backend: Any,
        *,
        manager: Any | None = None,
        **kwargs: Any,
    ) -> int:
        """Return the contiguous prefix serviceable by HCA-deferred objects."""
        if not blocks:
            return 0
        keys = [key for _start, _end, key in blocks]
        compact_records = disk_backend.get_kv_object_records(
            keys,
            roles=[_DSV4_HCA_DEFERRED_RETRIEVE_ROLE] * len(keys),
        )
        slab_records = disk_backend.get_kv_object_records(
            keys,
            roles=["hca_attention_kv_slab"] * len(keys),
        )
        layer_pairs = (
            self._dsv4_hca_object_source_layer_pairs(manager, **kwargs)
            if manager is not None
            else ()
        )
        per_layer_records = [
            disk_backend.get_kv_object_records(
                keys,
                layer_ids=[object_layer_id] * len(keys),
                roles=["hca_attention_kv"] * len(keys),
            )
            for _manager_layer_id, object_layer_id in layer_pairs
        ]
        readable = 0
        for index, _block in enumerate(blocks):
            compact = compact_records[index] if index < len(compact_records) else None
            if not self._dsv4_kv_object_record_readable(disk_backend, compact):
                break
            slab = slab_records[index] if index < len(slab_records) else None
            if self._dsv4_kv_object_record_readable(disk_backend, slab):
                readable += 1
                continue
            if not per_layer_records:
                break
            if all(
                index < len(records)
                and self._dsv4_kv_object_record_readable(
                    disk_backend,
                    records[index],
                )
                for records in per_layer_records
            ):
                readable += 1
                continue
            break
        return readable

    def _dsv4_register_hca_object_sources(
        self,
        blocks: list[tuple[CacheEngineKey, int, int]],
        manager: Any,
        disk_backend: Any,
        **kwargs: Any,
    ) -> int:
        """Register per-layer object-source chunks for HCA overlap."""
        if self._tutti_loader is None or not blocks:
            return 0
        layer_pairs = self._dsv4_hca_object_source_layer_pairs(manager, **kwargs)
        if not layer_pairs:
            return 0
        try:
            from lmcache.v1.gpu_connector.tutti_direct_loader import LbaRecord
            from lmcache.v1.hca_prefetch_manager import HCAObjectChunk
        except ImportError:
            return 0

        keys = [key for key, _start, _end in blocks]
        object_source_entries = []
        combined_raw_lba_cache: dict[str, list[LbaRecord]] = {}
        slab_records = disk_backend.get_kv_object_records(
            keys,
            roles=["hca_attention_kv_slab"] * len(keys),
        )
        slab_map = self._dsv4_hca_slab_layer_index_map(layer_pairs)
        use_slab_records = (
            slab_map
            and len(slab_map) == len(layer_pairs)
            and slab_records
            and all(record is not None for record in slab_records)
            and all(
                self._dsv4_kv_object_record_readable(disk_backend, record)
                for record in slab_records
            )
        )
        if use_slab_records:
            slab_chunks_by_layer: dict[int, list[HCAObjectChunk]] = {}
            raw_lba_cache: dict[str, list[LbaRecord]] = {}
            for manager_layer_id, _object_layer_id in layer_pairs:
                slab_index, slab_layers = slab_map[int(manager_layer_id)]
                source_chunks = []
                for (key, start, end), record in zip(
                    blocks,
                    slab_records,
                    strict=True,
                ):
                    if not self._dsv4_kv_object_record_readable(
                        disk_backend,
                        record,
                    ):
                        source_chunks = []
                        break
                    path = disk_backend.kv_object_data_path(record)
                    if path is None:
                        source_chunks = []
                        break
                    source_chunks.append(
                        HCAObjectChunk(
                            start_row_id=start // 128,
                            rows=max(0, (end - start) // 128),
                            key=key,
                            path=path,
                            record=record,
                            slab_layer_index=slab_index,
                            slab_num_layers=slab_layers,
                        )
                    )
                    if record.raw_extents:
                        raw_lba_cache.setdefault(path, []).extend(
                            LbaRecord(
                                file_offset=file_offset,
                                slba=slba,
                                n_sectors=n_sectors,
                            )
                            for file_offset, slba, n_sectors in record.raw_extents
                        )
                if source_chunks:
                    slab_chunks_by_layer[int(manager_layer_id)] = source_chunks
            if len(slab_chunks_by_layer) == len(layer_pairs):
                for path, records_for_path in raw_lba_cache.items():
                    combined_raw_lba_cache.setdefault(path, []).extend(records_for_path)
                for manager_layer_id, source_chunks in slab_chunks_by_layer.items():
                    object_source_entries.append(
                        (
                            manager_layer_id,
                            source_chunks,
                            self._tutti_loader,
                            self._tutti_warmup_lock,
                        )
                    )
                logger.info(
                    "HCAPrefetchManager: registered HCA slab object-source "
                    "chunks layers=%d blocks=%d",
                    len(object_source_entries),
                    len(blocks),
                )

        if not object_source_entries:
            for manager_layer_id, object_layer_id in layer_pairs:
                records = disk_backend.get_kv_object_records(
                    keys,
                    layer_ids=[object_layer_id] * len(keys),
                    roles=["hca_attention_kv"] * len(keys),
                )
                source_chunks = []
                raw_lba_cache: dict[str, list[LbaRecord]] = {}
                for (key, start, end), record in zip(blocks, records, strict=True):
                    if not self._dsv4_kv_object_record_readable(
                        disk_backend,
                        record,
                    ):
                        source_chunks = []
                        break
                    path = disk_backend.kv_object_data_path(record)
                    if path is None:
                        source_chunks = []
                        break
                    source_chunks.append(
                        HCAObjectChunk(
                            start_row_id=start // 128,
                            rows=max(0, (end - start) // 128),
                            key=key,
                            path=path,
                            record=record,
                        )
                    )
                    if record.raw_extents:
                        raw_lba_cache.setdefault(path, []).extend(
                            LbaRecord(
                                file_offset=file_offset,
                                slba=slba,
                                n_sectors=n_sectors,
                            )
                            for file_offset, slba, n_sectors in record.raw_extents
                        )
                if not source_chunks:
                    continue
                for path, records_for_path in raw_lba_cache.items():
                    combined_raw_lba_cache.setdefault(path, []).extend(records_for_path)
                object_source_entries.append(
                    (
                        manager_layer_id,
                        source_chunks,
                        self._tutti_loader,
                        self._tutti_warmup_lock,
                    )
                )
        registered_layers = 0
        replace_sources = getattr(manager, "replace_object_sources", None)
        if object_source_entries and callable(replace_sources):
            if combined_raw_lba_cache:
                self._tutti_loader.register_lba_cache(combined_raw_lba_cache)
            registered_layers = int(replace_sources(object_source_entries))
        elif object_source_entries:
            clear_sources = getattr(manager, "clear_object_sources", None)
            if callable(clear_sources):
                clear_sources()
            if combined_raw_lba_cache:
                self._tutti_loader.register_lba_cache(combined_raw_lba_cache)
            for (
                manager_layer_id,
                source_chunks,
                loader,
                loader_lock,
            ) in object_source_entries:
                manager.set_layer_object_source(
                    manager_layer_id,
                    source_chunks,
                    loader,
                    loader_lock,
                )
                registered_layers += 1
        if registered_layers:
            logger.info(
                "HCAPrefetchManager: registered object-source chunks for "
                "%d HCA layers blocks=%d",
                registered_layers,
                len(blocks),
            )
        else:
            logger.info(
                "HCAPrefetchManager: no HCA object-source layers registered "
                "blocks=%d layer_pairs=%d",
                len(blocks),
                len(layer_pairs),
            )
        return registered_layers

    def _dsv4_retrieve_view_for_range(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        start: int,
        end: int,
        total_tokens: int,
        *,
        require_sector_readable: bool = True,
    ) -> tuple[list[torch.Size], Optional[Tuple[KVObjectByteRange, ...]]]:
        """Return retrieve shapes and compact object ranges for one chunk.

        The store layout may already omit DSv4 tail-only groups for non-tail
        chunks.  HCA overlap needs a second view: keep the stored non-HCA
        groups compacted in the returned MemoryObj while skipping HCA bytes so
        those per-layer objects can be fetched independently by the overlap
        path.
        """
        stored_shapes = self._dsv4_store_shapes_for_range(
            shapes,
            dtypes,
            start,
            end,
            total_tokens,
        )
        if not self._dsv4_hca_defer_retrieve_enabled(total_tokens):
            return stored_shapes, None
        klg_manager = self.metadata.kv_layer_groups_manager
        if klg_manager is None or not klg_manager.kv_layer_groups:
            return stored_shapes, None

        retrieve_shapes: list[torch.Size] = []
        read_ranges: list[KVObjectByteRange] = []
        source_offset = 0
        target_offset = 0
        skipped_hca = False
        for shape, dtype, group in zip(
            stored_shapes,
            dtypes,
            klg_manager.kv_layer_groups,
            strict=True,
        ):
            group_nbytes = self._shape_nbytes(shape, dtype)
            role = self._dsv4_group_role(group, dtype)
            if role == "hca_attention_kv" and group_nbytes > 0:
                retrieve_shapes.append(self._zero_token_shape(shape))
                skipped_hca = True
            else:
                retrieve_shapes.append(shape)
                if group_nbytes > 0:
                    read_ranges.append(
                        KVObjectByteRange(
                            offset=source_offset,
                            length=group_nbytes,
                            target_offset=target_offset,
                        )
                    )
                    target_offset += group_nbytes
            source_offset += group_nbytes

        if not skipped_hca:
            return stored_shapes, None
        if not read_ranges:
            return stored_shapes, None
        compact_ranges = tuple(read_ranges)
        if require_sector_readable and not _raw_ranges_sector_readable(
            compact_ranges,
            target_offset,
        ):
            logger.info(
                "HCAPrefetchManager: skip compact HCA-deferred retrieve "
                "start=%d end=%d compact_bytes=%d reason=unaligned_ranges",
                start,
                end,
                target_offset,
            )
            return stored_shapes, None
        return retrieve_shapes, compact_ranges

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
    def prepare_dsv4_streaming_retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Prepare a metadata-only DSv4 hit without generic KV retrieval.

        This fast path reuses the block map pinned by lookup and registers the
        CSA, HCA, and indexer layer-major read plans. It is admitted only when
        the compact-main view for every cached chunk is empty for the current
        request. Consequently, every byte needed by model execution is owned
        by a layer-major consumer and calling :meth:`retrieve` would perform
        only redundant synchronous bookkeeping.

        Args:
            tokens: Tokens in the externally cached prefix.
            mask: Tokens that LMCache, rather than vLLM, must restore.
            **kwargs: Request metadata. ``req_id``, ``request_total_tokens``,
                ``request_configs``, and ``slot_mapping`` are used here.

        Returns:
            A CPU boolean mask when the streaming-only plan was committed, or
            ``None`` when the caller must use normal :meth:`retrieve`.
        """
        profile_start = time.perf_counter()
        if (
            not self.dsv4_optimized_kv
            or not _env_flag("LMCACHE_INDEXER_ENABLE_PREFETCH")
            or self.storage_manager is None
            or self._tutti_config is None
            or not self._dsv4_csa_attention_kv_prefetch_active()
        ):
            return None

        req_id = str(kwargs.get("req_id", ""))
        pinned_mapping = self.lookup_pins.get(req_id)
        request_configs = kwargs.get("request_configs")
        build_blocks_start = time.perf_counter()
        terminal_chunk_hash = kwargs.get("terminal_chunk_hash")
        mask_is_dense = mask is None or bool(torch.all(mask).item())
        use_terminal_key = bool(
            isinstance(terminal_chunk_hash, int)
            and int(kwargs.get("vllm_cached_tokens", 0)) == 0
            and mask_is_dense
            and len(tokens) > 0
            and len(tokens) % int(self.config.chunk_size) == 0
        )
        expected_blocks: list[tuple[CacheEngineKey, int, int]] = []
        if use_terminal_key:
            terminal_key = CacheEngineKey(
                model_name=self.metadata.model_name,
                world_size=int(self.metadata.world_size),
                worker_id=int(self.metadata.worker_id),
                chunk_hash=int(terminal_chunk_hash),
                dtype=self.metadata.kv_dtype,
                request_configs=request_configs,
            )
            expected_blocks.append((terminal_key, 0, len(tokens)))
        else:
            for start, end, key in self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)
                expected_blocks.append((key, start, end))
        if not expected_blocks:
            return None
        blocks = expected_blocks
        build_blocks_ms = (time.perf_counter() - build_blocks_start) * 1000.0

        validate_start = time.perf_counter()
        disk_backend = self.storage_manager.storage_backends.get("LocalDiskBackend")
        if disk_backend is None:
            return None
        expected_keys = [key for key, _start, _end in blocks]
        if pinned_mapping is not None:
            if (
                set(pinned_mapping) != {"LocalDiskBackend"}
                or not pinned_mapping["LocalDiskBackend"]
                or (
                    use_terminal_key
                    and pinned_mapping["LocalDiskBackend"][-1] != expected_keys[-1]
                )
                or (
                    not use_terminal_key
                    and list(pinned_mapping["LocalDiskBackend"]) != expected_keys
                )
            ):
                return None
        elif any(key not in disk_backend.dict for key in expected_keys):
            # Only the scheduler-facing rank performs lookup and owns pins.
            # Other TP ranks may use their atomically admitted rank-local
            # metadata; plan registration below validates every sidecar before
            # committing any streaming consumer.
            return None

        role = (
            _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE
            if _env_flag("LMCACHE_DSV4_HCA_WALKER")
            else _DSV4_CSA_DEFERRED_RETRIEVE_ROLE
        )
        request_total_tokens = max(
            len(tokens),
            int(kwargs.get("request_total_tokens", len(tokens))),
        )
        tail_tokens = getattr(self, "dsv4_optimized_tail_tokens", None)
        tail_start = (
            max(0, request_total_tokens - int(tail_tokens))
            if isinstance(tail_tokens, int) and tail_tokens >= 0
            else None
        )
        # The streaming compact main contains only request-tail groups after
        # CSA, HCA, and indexer roles have been split into layer-major objects.
        # Any cached chunk ending at or before the active tail boundary is
        # therefore zero-byte by construction. Lookup already validated the
        # immutable generation; expand schemas only for chunks that overlap
        # the current request tail instead of repeating it for the full prefix.
        compact_candidates = (
            ((key, start, end) for key, start, end in blocks if end > tail_start)
            if tail_start is not None
            else ((key, start, end) for key, start, end in blocks)
        )
        if any(
            self._dsv4_streaming_compact_payload_nbytes(
                (start, end, key),
                role,
                request_total_tokens,
            )
            != 0
            for key, start, end in compact_candidates
        ):
            return None
        validate_ms = (time.perf_counter() - validate_start) * 1000.0

        ensure_loader_start = time.perf_counter()
        if not self._ensure_tutti_loader(expected_keys):
            return None
        ensure_loader_ms = (time.perf_counter() - ensure_loader_start) * 1000.0
        register_start = time.perf_counter()
        csa_ready, hca_ready, indexer_ready = self._register_csa_attention_kv_chunks(
            blocks,
            [disk_backend.dict.get(key) for key, _start, _end in blocks],
            len(tokens),
            req_id,
            slot_mapping=kwargs.get("slot_mapping"),
        )
        # A terminal sidecar normally snapshots the complete admitted prefix,
        # which makes the one-key path ideal for the common first hit.  A
        # deferred hit admission, however, snapshots only its newly computed
        # suffix under the new terminal key.  When that suffix is extended by
        # another request, the one-key preflight correctly rejects its short
        # record.  Rebuild the conventional chunk list here so the layer-major
        # planners can greedily compose the earlier full-prefix generation and
        # the suffix generation, instead of falling through to generic main-
        # object retrieval.
        if use_terminal_key and not (
            csa_ready
            and indexer_ready
            and (not _env_flag("LMCACHE_DSV4_HCA_WALKER") or hca_ready)
        ):
            expanded_blocks: list[tuple[CacheEngineKey, int, int]] = []
            for start, end, key in self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)
                expanded_blocks.append((key, start, end))
            if len(expanded_blocks) > 1 and all(
                key in disk_backend.dict for key, _start, _end in expanded_blocks
            ):
                blocks = expanded_blocks
                csa_ready, hca_ready, indexer_ready = (
                    self._register_csa_attention_kv_chunks(
                        blocks,
                        [disk_backend.dict.get(key) for key, _start, _end in blocks],
                        len(tokens),
                        req_id,
                        slot_mapping=kwargs.get("slot_mapping"),
                    )
                )
                if (
                    csa_ready
                    and indexer_ready
                    and (not _env_flag("LMCACHE_DSV4_HCA_WALKER") or hca_ready)
                ):
                    use_terminal_key = False
                    logger.info(
                        "DSv4 streaming-only terminal snapshot composed from "
                        "%d admitted chunks request=%s",
                        len(blocks),
                        req_id,
                    )
        if not (
            csa_ready
            and indexer_ready
            and (not _env_flag("LMCACHE_DSV4_HCA_WALKER") or hca_ready)
        ):
            return None
        register_ms = (time.perf_counter() - register_start) * 1000.0

        mask_start = time.perf_counter()
        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
        # ``ChunkedTokenDatabase.process_tokens`` yields a dense suffix after
        # the optional vLLM-cached prefix. The unified sidecar preflight above
        # has already rejected incomplete coverage, so one slice is exactly
        # equivalent to 1,875 per-chunk assignments for a 480K hit.
        ret_mask[blocks[0][1] : blocks[-1][2]] = True
        if (
            _env_flag("LMCACHE_TUTTI_PROFILE")
            or int(getattr(getattr(self, "metadata", None), "worker_id", 0)) == 0
        ):
            logger.info(
                "DSv4 streaming-only hit: skipped generic LMCacheEngine.retrieve "
                "request=%s blocks=%d tokens=%d",
                req_id,
                len(blocks),
                int(blocks[-1][2]) - int(blocks[0][1]),
            )
        if _env_flag("LMCACHE_TUTTI_PROFILE"):
            logger.info(
                "TUTTI_PROFILE streaming_retrieve request=%s blocks=%d "
                "terminal_key_fastpath=%d "
                "build_blocks_ms=%.3f validate_ms=%.3f "
                "ensure_loader_ms=%.3f register_ms=%.3f mask_ms=%.3f "
                "total_ms=%.3f",
                req_id,
                len(blocks),
                int(use_terminal_key),
                build_blocks_ms,
                validate_ms,
                ensure_loader_ms,
                register_ms,
                (time.perf_counter() - mask_start) * 1000.0,
                (time.perf_counter() - profile_start) * 1000.0,
            )
        return ret_mask

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
            process_tokens_ms = (time.perf_counter() - process_tokens_start) * 1000.0

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
            profile_total_ms = (time.perf_counter() - retrieve_profile_start) * 1000.0
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
    def lookup_streaming_terminal(
        self,
        terminal_hash: int,
        token_count: int,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        """Look up one exact atomically published streaming generation.

        Args:
            terminal_hash: Content hash of the final cached chunk.
            token_count: Exact logical-token coverage required from the
                generation.
            lookup_id: Request identifier used to own a successful pin.
            pin: Whether to pin the terminal key until ``lookup_unpin``.
            request_configs: Optional request-specific key configuration.

        Returns:
            ``token_count`` on an exact LocalDiskBackend generation hit, or
            zero when exact coverage cannot be proven.

        Notes:
            This method does not reinterpret the terminal key as a synthetic
            ``[0, token_count)`` ordinary chunk. The manifest itself carries
            coverage, so short deferred-admission sidecars fail closed.
        """
        if (
            token_count <= 0
            or token_count % int(self.config.chunk_size) != 0
            or not self.is_healthy()
            or not self.dsv4_optimized_kv
            or not _env_flag("LMCACHE_INDEXER_ENABLE_PREFETCH")
        ):
            return 0
        if pin and lookup_id is None:
            raise ValueError("lookup_id is required when pin is True")
        assert self.storage_manager is not None
        if (
            self.retrieve_locations is not None
            and "LocalDiskBackend" not in self.retrieve_locations
        ):
            return 0

        lookup_stats = self.stats_monitor.on_lookup_request(token_count)
        result = 0
        try:
            key = CacheEngineKey(
                model_name=self.metadata.model_name,
                world_size=int(self.metadata.world_size),
                worker_id=int(self.metadata.worker_id),
                chunk_hash=int(terminal_hash),
                dtype=self.metadata.kv_dtype,
                request_configs=request_configs,
            )
            location = self.storage_manager.contains_streaming_terminal(
                key,
                token_count,
                search_range=["LocalDiskBackend"],
                pin=pin,
            )
            if location is None:
                return 0
            if pin:
                assert lookup_id is not None
                self.lookup_pins[lookup_id] = {location: [key]}
            result = token_count
            return result
        finally:
            self.stats_monitor.on_lookup_finished(lookup_stats, result)
            if pin:
                self.storage_manager.touch_cache()

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
                if _env_flag("LMCACHE_DISK_CONTAINS_DIAGNOSTICS"):
                    logger.info(
                        "LMCache lookup prefix diagnostic stage=before_filter "
                        "hit_chunks=%d total_chunks=%d first_key=%s miss_key=%s",
                        hit_chunks,
                        len(keys),
                        keys[0].to_string() if keys else "none",
                        (
                            keys[hit_chunks].to_string()
                            if hit_chunks < len(keys)
                            else "none"
                        ),
                    )
                hit_chunks = self._filter_tutti_raw_lookup_prefix(
                    chunk_info_list,
                    hit_chunks,
                    block_mapping,
                    pin=pin,
                    total_tokens=len(tokens)
                    if tokens is not None
                    else sum(offsets or []),
                )
                if _env_flag("LMCACHE_DISK_CONTAINS_DIAGNOSTICS"):
                    logger.info(
                        "LMCache lookup prefix diagnostic stage=after_filter "
                        "hit_chunks=%d total_chunks=%d",
                        hit_chunks,
                        len(keys),
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
        self._discard_dsv4_layer_major_snapshot()

        # Deferred hit admissions access both the object store and the Tutti
        # writer, so drain them before either backend is closed.
        self._dsv4_admission_executor.shutdown(wait=True, cancel_futures=False)

        try:
            from lmcache.integration.vllm.vllm_v1_adapter import (
                close_vllm_prefetch_managers,
            )

            close_vllm_prefetch_managers()
        except ImportError:
            pass

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

    def _store_dsv4_layer_major_snapshot(
        self,
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        token_count: int,
        *,
        mode: str,
        base_prefix_key: Optional[CacheEngineKey] = None,
        base_prefix_token_count: int = 0,
    ) -> None:
        """Build sidecars for one complete request snapshot."""
        disk_backend = self.storage_manager.storage_backends.get("LocalDiskBackend")
        store_snapshot = getattr(
            disk_backend,
            "store_attention_layer_major_snapshot",
            None,
        )
        try:
            if callable(store_snapshot):
                stored_layers = int(
                    store_snapshot(
                        keys[-1],
                        memory_objs,
                        prefix_keys=keys,
                        prefix_token_count=token_count,
                        base_prefix_key=base_prefix_key,
                        base_prefix_token_count=base_prefix_token_count,
                    )
                )
                logger.info(
                    "DSv4 attention layer-major snapshot key=%s "
                    "layers=%d chunks=%d tokens=%d base_tokens=%d mode=%s",
                    keys[-1].to_string(),
                    stored_layers,
                    len(memory_objs),
                    token_count,
                    base_prefix_token_count,
                    mode,
                )
        except Exception:
            logger.exception(
                "DSv4 attention layer-major snapshot store failed for %s",
                keys[-1].to_string(),
            )

    def _submit_dsv4_hit_admission(
        self,
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        token_count: int,
        *,
        req_id: str,
        is_last_prefill: bool,
        transfer_spec: Any,
        base_prefix_key: Optional[CacheEngineKey] = None,
        base_prefix_token_count: int = 0,
    ) -> bool:
        """Queue one cache-hit suffix admission outside first-token latency."""
        with self._dsv4_admission_lock:
            if self._dsv4_admission_pending >= self._dsv4_admission_max_pending:
                return False
            self._dsv4_admission_pending += 1

        def _run() -> None:
            try:
                self._commit_dsv4_hit_admission(
                    keys,
                    memory_objs,
                    token_count,
                    req_id=req_id,
                    is_last_prefill=is_last_prefill,
                    transfer_spec=transfer_spec,
                    base_prefix_key=base_prefix_key,
                    base_prefix_token_count=base_prefix_token_count,
                )
            finally:
                with self._dsv4_admission_lock:
                    self._dsv4_admission_pending -= 1

        try:
            self._dsv4_admission_executor.submit(_run)
        except RuntimeError:
            with self._dsv4_admission_lock:
                self._dsv4_admission_pending -= 1
            return False
        if int(getattr(self.metadata, "worker_id", 0)) == 0:
            logger.info(
                "DSv4 deferred hit admission queued request=%s chunks=%d "
                "tokens=%d base_tokens=%d terminal=%s",
                req_id,
                len(memory_objs),
                token_count,
                base_prefix_token_count,
                keys[-1].to_string(),
            )
        return True

    def _commit_dsv4_hit_admission(
        self,
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        token_count: int,
        *,
        req_id: str,
        is_last_prefill: bool,
        transfer_spec: Any,
        base_prefix_key: Optional[CacheEngineKey] = None,
        base_prefix_token_count: int = 0,
    ) -> None:
        """Publish deferred sidecars and compact main as one generation."""
        put_started = False
        try:
            self._store_dsv4_layer_major_snapshot(
                keys,
                memory_objs,
                token_count,
                mode="deferred_hit",
                base_prefix_key=base_prefix_key,
                base_prefix_token_count=base_prefix_token_count,
            )
            tutti_warmup_callback = self._make_tutti_store_warmup_callback(
                list(keys),
                req_id,
                is_last_prefill,
            )
            put_kwargs: dict[str, Any] = {
                "transfer_spec": transfer_spec,
                "location": self.store_location,
            }
            if tutti_warmup_callback is not None:
                put_signature = inspect.signature(self.storage_manager.batched_put)
                if "on_complete_callback" not in put_signature.parameters:
                    raise RuntimeError(
                        "StorageManager.batched_put must support "
                        "on_complete_callback for deferred Tutti admission"
                    )
                put_kwargs["on_complete_callback"] = tutti_warmup_callback
            put_started = True
            self.storage_manager.batched_put(keys, memory_objs, **put_kwargs)
            if int(getattr(self.metadata, "worker_id", 0)) == 0:
                logger.info(
                    "DSv4 deferred hit admission submitted request=%s "
                    "chunks=%d tokens=%d base_tokens=%d terminal=%s",
                    req_id,
                    len(memory_objs),
                    token_count,
                    base_prefix_token_count,
                    keys[-1].to_string(),
                )
        except Exception:
            logger.exception(
                "DSv4 deferred hit admission failed request=%s chunks=%d",
                req_id,
                len(memory_objs),
            )
            if not put_started:
                for memory_obj in memory_objs:
                    memory_obj.ref_count_down()

    def _stage_dsv4_layer_major_snapshot(
        self,
        req_id: str,
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        token_count: int,
        *,
        is_last_prefill: bool,
        base_prefix_key: Optional[CacheEngineKey] = None,
        base_prefix_token_count: int = 0,
    ) -> Optional[tuple[list[CacheEngineKey], list[MemoryObj], int]]:
        """Retain one store batch and return the complete final snapshot."""
        stale_memory_objs: list[MemoryObj] = []
        result: Optional[tuple[list[CacheEngineKey], list[MemoryObj], int]] = None
        with self._dsv4_snapshot_lock:
            if (
                self._dsv4_snapshot_req_id is not None
                and self._dsv4_snapshot_req_id != req_id
            ):
                stale_memory_objs = self._dsv4_snapshot_memory_objs
                self._dsv4_snapshot_keys = []
                self._dsv4_snapshot_memory_objs = []
                self._dsv4_snapshot_tokens = 0
                self._dsv4_snapshot_base_key = None
                self._dsv4_snapshot_base_tokens = 0
            self._dsv4_snapshot_req_id = req_id

            if base_prefix_token_count > 0:
                if base_prefix_key is None:
                    raise ValueError(
                        "partial-hit snapshot requires its terminal prefix key"
                    )
                snapshot_base_tokens = int(
                    getattr(self, "_dsv4_snapshot_base_tokens", 0)
                )
                if snapshot_base_tokens not in {
                    0,
                    int(base_prefix_token_count),
                }:
                    raise ValueError("partial-hit snapshot base changed within request")
                self._dsv4_snapshot_base_key = base_prefix_key
                self._dsv4_snapshot_base_tokens = int(base_prefix_token_count)

            retained: list[MemoryObj] = []
            try:
                for memory_obj in memory_objs:
                    memory_obj.ref_count_up()
                    retained.append(memory_obj)
            except Exception:
                for memory_obj in retained:
                    memory_obj.ref_count_down()
                raise
            self._dsv4_snapshot_keys.extend(keys)
            self._dsv4_snapshot_memory_objs.extend(memory_objs)
            self._dsv4_snapshot_tokens += int(token_count)

            if is_last_prefill:
                self._dsv4_completed_snapshot_base_key = getattr(
                    self,
                    "_dsv4_snapshot_base_key",
                    None,
                )
                self._dsv4_completed_snapshot_base_tokens = int(
                    getattr(self, "_dsv4_snapshot_base_tokens", 0)
                )
                result = (
                    self._dsv4_snapshot_keys,
                    self._dsv4_snapshot_memory_objs,
                    self._dsv4_snapshot_tokens,
                )
                self._dsv4_snapshot_req_id = None
                self._dsv4_snapshot_keys = []
                self._dsv4_snapshot_memory_objs = []
                self._dsv4_snapshot_tokens = 0
                self._dsv4_snapshot_base_key = None
                self._dsv4_snapshot_base_tokens = 0

        for memory_obj in stale_memory_objs:
            memory_obj.ref_count_down()
        return result

    def _discard_dsv4_layer_major_snapshot(self) -> None:
        """Release retained host snapshots for an incomplete request."""
        with self._dsv4_snapshot_lock:
            memory_objs = self._dsv4_snapshot_memory_objs
            self._dsv4_snapshot_req_id = None
            self._dsv4_snapshot_keys = []
            self._dsv4_snapshot_memory_objs = []
            self._dsv4_snapshot_tokens = 0
            self._dsv4_snapshot_base_key = None
            self._dsv4_snapshot_base_tokens = 0
        for memory_obj in memory_objs:
            memory_obj.ref_count_down()

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
            block_mapping = {}
            # Only the scheduler-facing TP rank owns the lookup pin.  The
            # remaining ranks must not prefix-scan the individual streaming
            # chunks here: non-terminal chunks intentionally have a zero-byte
            # compact main and therefore fail ``batched_contains`` by design.
            # Validate the exact self-contained terminal generation instead,
            # then consume the same chunk geometry used by the pinned rank.
            # This preserves the rule that a metadata-only intermediate chunk
            # can never be treated as an independently restorable prefix.
            if self.dsv4_optimized_kv and chunk_infos:
                terminal_key = chunk_infos[-1][0]
                terminal_location = self.storage_manager.contains_streaming_terminal(
                    terminal_key,
                    len(tokens),
                    search_range=["LocalDiskBackend"],
                    pin=False,
                )
                if terminal_location is not None:
                    block_mapping = {terminal_location: chunk_infos}
                    if _env_flag("LMCACHE_TUTTI_PROFILE"):
                        logger.info(
                            "DSv4 worker terminal manifest admitted retrieve "
                            "request=%s worker=%d blocks=%d tokens=%d",
                            kwargs.get("req_id", "unknown"),
                            int(getattr(self.metadata, "worker_id", 0)),
                            len(chunk_infos),
                            len(tokens),
                        )
            if not block_mapping:
                block_mapping = self.storage_manager.get_block_mapping(chunk_infos)

        total_tokens = len(tokens)
        # vLLM truncates ``tokens`` to the cacheable load prefix and normally
        # retains at least one block for recomputation.  Tail-only DSv4 groups
        # must therefore be classified against the full prompt, not against
        # this shorter retrieve slice.
        request_total_tokens = max(
            total_tokens,
            int(kwargs.get("request_total_tokens", total_tokens)),
        )

        last_failed_block_start = None
        for location, mapped_blocks in block_mapping.items():
            # Storage-manager mappings may be shared with lookup bookkeeping.
            # Freeze the request view before planners and streaming callbacks
            # retain it; otherwise a late append changes callback indices after
            # ``keys`` and shape lists have already been materialised.
            blocks = list(mapped_blocks)
            keys = [key for key, _, _ in blocks]

            # For DSv4-optimised KV layouts, compute per-chunk shape overrides
            # so that non-tail chunks only read prefix groups (groups 0-2,
            # ~1.4 MB) instead of all 8 groups (~116 MB).  This mirrors the
            # shape masking already applied in the store path and prevents
            # vLLM RPC timeouts caused by reading ~110 GB sequentially for
            # large prompts.  ``_dsv4_retrieve_shapes_for_range`` additionally
            # zero-shapes ``csa_attention_kv`` when the CSA attention KV
            # prefetcher is attached, leaving those ~100 MiB for the
            # prefetcher to load lazily via Tutti during the FFN/MoE overlap
            # window.
            csa_prefetch_active = False
            csa_stream_ready = False
            hca_stream_ready = False
            indexer_stream_ready = False
            if self.dsv4_optimized_kv:
                csa_prefetch_active = self._dsv4_csa_attention_kv_prefetch_active()
                if csa_prefetch_active:
                    csa_disk_backend = self.storage_manager.storage_backends.get(
                        "LocalDiskBackend"
                    )
                    if csa_disk_backend is not None:
                        (
                            csa_stream_ready,
                            hca_stream_ready,
                            indexer_stream_ready,
                        ) = self._register_csa_attention_kv_chunks(
                            blocks,
                            [csa_disk_backend.dict.get(key) for key, _, _ in blocks],
                            total_tokens,
                            kwargs.get("req_id", "unknown"),
                            slot_mapping=kwargs.get("slot_mapping"),
                        )
                store_shapes_per_key: Optional[List[Optional[List[torch.Size]]]] = [
                    self._dsv4_retrieve_shapes_for_range(
                        self.metadata.get_shapes(end - start),
                        self.metadata.get_dtypes(),
                        start,
                        end,
                        request_total_tokens,
                    )
                    for _, start, end in blocks
                ]
            else:
                store_shapes_per_key = None

            # Read-plan construction must be observational. Re-materialise the
            # key list from the authoritative block map so a planner-side list
            # transformation can never make the streaming callback one block
            # shorter than the retrieve mask.
            keys = [key for key, _, _ in blocks]

            shapes_per_key = store_shapes_per_key
            read_ranges_per_key: Optional[
                List[Optional[Tuple[KVObjectByteRange, ...]]]
            ] = None
            kv_object_roles: Optional[List[str]] = None

            if location == "LocalDiskBackend" and self._tutti_config is not None:
                if self._ensure_tutti_loader(keys):
                    if self.dsv4_optimized_kv:
                        manager = self._dsv4_hca_object_source_manager()
                        disk_backend = self.storage_manager.storage_backends.get(
                            "LocalDiskBackend"
                        )
                        use_csa_deferred_compact = bool(
                            disk_backend is not None
                            and csa_prefetch_active
                            and csa_stream_ready
                            and indexer_stream_ready
                            and not _env_flag("LMCACHE_DSV4_HCA_WALKER")
                            and self._dsv4_csa_deferred_retrieve_available(
                                blocks,
                                disk_backend,
                            )
                        )
                        if use_csa_deferred_compact:
                            shapes_per_key = [
                                self._dsv4_csa_compact_retrieve_shapes_for_range(
                                    self.metadata.get_shapes(end - start),
                                    self.metadata.get_dtypes(),
                                    start,
                                    end,
                                    request_total_tokens,
                                )
                                for _, start, end in blocks
                            ]
                            kv_object_roles = [_DSV4_CSA_DEFERRED_RETRIEVE_ROLE] * len(
                                blocks
                            )
                            read_ranges_per_key = None
                            logger.info(
                                "CSAAttentionKVPrefetchManager: using compact "
                                "non-CSA retrieve objects blocks=%d",
                                len(blocks),
                            )
                        elif (
                            disk_backend is not None
                            and csa_prefetch_active
                            and csa_stream_ready
                            and hca_stream_ready
                            and indexer_stream_ready
                            and _env_flag("LMCACHE_DSV4_HCA_WALKER")
                            and self._dsv4_csa_hca_deferred_retrieve_available(
                                blocks,
                                disk_backend,
                            )
                        ):
                            shapes_per_key = [
                                self._dsv4_csa_hca_compact_retrieve_shapes_for_range(
                                    self.metadata.get_shapes(end - start),
                                    self.metadata.get_dtypes(),
                                    start,
                                    end,
                                    request_total_tokens,
                                )
                                for _, start, end in blocks
                            ]
                            kv_object_roles = [
                                _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE
                            ] * len(blocks)
                            read_ranges_per_key = None
                            logger.info(
                                "CSA/HCA prefetch: using unified layer-major "
                                "compact non-CSA/non-HCA retrieve objects "
                                "blocks=%d",
                                len(blocks),
                            )
                        elif (
                            not _env_flag("LMCACHE_INDEXER_ENABLE_PREFETCH")
                            and manager is not None
                            and disk_backend is not None
                            and _env_flag("LMCACHE_DSV4_HCA_WALKER")
                            and self._dsv4_hca_object_source_available(
                                blocks,
                                manager,
                                disk_backend,
                                **kwargs,
                            )
                        ):
                            registered_hca_layers = (
                                self._dsv4_register_hca_object_sources(
                                    blocks,
                                    manager,
                                    disk_backend,
                                    **kwargs,
                                )
                            )
                            if registered_hca_layers > 0:
                                use_csa_hca_deferred_compact = bool(
                                    self._dsv4_csa_attention_kv_prefetch_active()
                                    and self._dsv4_csa_hca_deferred_retrieve_available(
                                        blocks,
                                        disk_backend,
                                    )
                                )
                                use_hca_deferred_compact = (
                                    not use_csa_hca_deferred_compact
                                    and self._dsv4_hca_deferred_retrieve_available(
                                        blocks,
                                        disk_backend,
                                    )
                                )
                                if use_csa_hca_deferred_compact:
                                    shapes_per_key = [
                                        self._dsv4_csa_hca_compact_retrieve_shapes_for_range(
                                            self.metadata.get_shapes(end - start),
                                            self.metadata.get_dtypes(),
                                            start,
                                            end,
                                            request_total_tokens,
                                        )
                                        for _, start, end in blocks
                                    ]
                                    kv_object_roles = [
                                        _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE
                                    ] * len(blocks)
                                    read_ranges_per_key = None
                                    logger.info(
                                        "CSA/HCA prefetch: using compact "
                                        "non-CSA/non-HCA retrieve objects "
                                        "blocks=%d registered_hca_layers=%d",
                                        len(blocks),
                                        registered_hca_layers,
                                    )
                                else:
                                    retrieve_views = [
                                        self._dsv4_retrieve_view_for_range(
                                            self.metadata.get_shapes(end - start),
                                            self.metadata.get_dtypes(),
                                            start,
                                            end,
                                            request_total_tokens,
                                            require_sector_readable=not (
                                                use_hca_deferred_compact
                                            ),
                                        )
                                        for _, start, end in blocks
                                    ]
                                    shapes_per_key = [
                                        shapes
                                        for shapes, _read_ranges in retrieve_views
                                    ]
                                if use_hca_deferred_compact:
                                    kv_object_roles = [
                                        _DSV4_HCA_DEFERRED_RETRIEVE_ROLE
                                    ] * len(blocks)
                                    read_ranges_per_key = None
                                    logger.info(
                                        "HCAPrefetchManager: using compact "
                                        "HCA-deferred retrieve objects blocks=%d "
                                        "registered_layers=%d",
                                        len(blocks),
                                        registered_hca_layers,
                                    )
                                elif not use_csa_hca_deferred_compact:
                                    read_ranges_per_key = [
                                        read_ranges
                                        for _shapes, read_ranges in retrieve_views
                                    ]
                        elif csa_prefetch_active:
                            # A zero-shaped group in the middle of a raw object
                            # cannot be represented by a shorter prefix read.
                            # Until the compact non-CSA object is READY, retain
                            # every stored group so later bytes never shift into
                            # the wrong tensor. This costs one fallback full CSA
                            # read but preserves the connector's hit contract.
                            shapes_per_key = [
                                self._dsv4_store_shapes_for_range(
                                    self.metadata.get_shapes(end - start),
                                    self.metadata.get_dtypes(),
                                    start,
                                    end,
                                    request_total_tokens,
                                )
                                for _, start, end in blocks
                            ]
                            read_ranges_per_key = None
                            kv_object_roles = None
                            logger.warning(
                                "CSAAttentionKVPrefetchManager: compact non-CSA "
                                "objects unavailable; retaining generic CSA "
                                "retrieve for correctness blocks=%d",
                                len(blocks),
                            )
                    stream_tutti_retrieve = (
                        self.gpu_connector is not None and not self.save_only_first_rank
                    )
                    raw_tutti_retrieve = stream_tutti_retrieve and callable(
                        getattr(
                            self.gpu_connector,
                            "batched_raw_to_gpu",
                            None,
                        )
                    )
                    tutti_blocks = blocks
                    tutti_keys = [key for key, _, _ in blocks]
                    tutti_shapes = shapes_per_key
                    tutti_read_ranges = read_ranges_per_key
                    tutti_roles = kv_object_roles
                    if stream_tutti_retrieve and shapes_per_key is not None:
                        io_indices = [
                            index
                            for index, shape_list in enumerate(shapes_per_key)
                            if shape_list is None
                            or any(shape.numel() > 0 for shape in shape_list)
                        ]
                        io_index_set = set(io_indices)
                        metadata_only_indices = [
                            index
                            for index in range(len(blocks))
                            if index not in io_index_set
                        ]
                        for index in metadata_only_indices:
                            _key, start, end = blocks[index]
                            ret_mask[start:end] = True
                        if metadata_only_indices:
                            logger.info(
                                "TUTTI_PROFILE metadata_only_hits blocks=%d/%d",
                                len(metadata_only_indices),
                                len(blocks),
                            )
                        if not io_indices:
                            continue
                        tutti_blocks = [blocks[index] for index in io_indices]
                        tutti_keys = [tutti_keys[index] for index in io_indices]
                        tutti_shapes = [shapes_per_key[index] for index in io_indices]
                        if read_ranges_per_key is not None:
                            tutti_read_ranges = [
                                read_ranges_per_key[index] for index in io_indices
                            ]
                        if kv_object_roles is not None:
                            tutti_roles = [
                                kv_object_roles[index] for index in io_indices
                            ]
                    streaming_consumed = False
                    streaming_failed = False
                    streamed_blocks = 0

                    def _consume_tutti_batch(
                        batch_start: int,
                        batch_results: List[Optional[MemoryObj]],
                        retrieve_blocks: List[
                            Tuple[CacheEngineKey, int, int]
                        ] = tutti_blocks,
                    ) -> None:
                        nonlocal last_failed_block_start
                        nonlocal streaming_consumed, streaming_failed
                        nonlocal streamed_blocks, tot_kv_size

                        streaming_consumed = True
                        batch_memory_objs: List[MemoryObj] = []
                        batch_starts: List[int] = []
                        batch_ends: List[int] = []
                        batch_sizes: List[int] = []
                        for offset, memory_obj in enumerate(batch_results):
                            block_index = batch_start + offset
                            if block_index >= len(retrieve_blocks):
                                if memory_obj is not None:
                                    memory_obj.ref_count_down()
                                continue
                            _key, start, end = retrieve_blocks[block_index]
                            if streaming_failed or memory_obj is None:
                                if memory_obj is not None:
                                    memory_obj.ref_count_down()
                                if not streaming_failed:
                                    logger.warning(
                                        "The cache block is in the storage, "
                                        "but it can't be retrieved"
                                    )
                                    if (
                                        last_failed_block_start is None
                                        or last_failed_block_start > start
                                    ):
                                        last_failed_block_start = start
                                    streaming_failed = True
                                continue
                            batch_memory_objs.append(memory_obj)
                            batch_starts.append(start)
                            batch_ends.append(end)
                            batch_sizes.append(memory_obj.get_size())

                        if not batch_memory_objs:
                            return
                        try:
                            self.gpu_connector.batched_to_gpu(
                                batch_memory_objs,
                                batch_starts,
                                batch_ends,
                                **kwargs,
                            )
                        finally:
                            for memory_obj in batch_memory_objs:
                                memory_obj.ref_count_down()
                        for start, end, size in zip(
                            batch_starts,
                            batch_ends,
                            batch_sizes,
                            strict=True,
                        ):
                            ret_mask[start:end] = True
                            tot_kv_size += size
                            streamed_blocks += 1

                    def _consume_tutti_raw_batch(
                        batch_start: int,
                        completed_indices: List[int],
                        completed_offsets: List[int],
                        completed_nbytes: List[int],
                        staging: torch.Tensor,
                        completed_shapes: List[List[torch.Size]],
                        completed_dtypes: List[List[torch.dtype]],
                        retrieve_blocks: List[
                            Tuple[CacheEngineKey, int, int]
                        ] = tutti_blocks,
                    ) -> None:
                        nonlocal streaming_consumed, streaming_failed
                        nonlocal streamed_blocks, tot_kv_size
                        nonlocal last_failed_block_start

                        streaming_consumed = True
                        restore_offsets: List[int] = []
                        restore_nbytes: List[int] = []
                        restore_starts: List[int] = []
                        restore_ends: List[int] = []
                        restore_shapes: List[List[torch.Size]] = []
                        restore_dtypes: List[List[torch.dtype]] = []
                        restore_sizes: List[int] = []
                        for completed_pos, local_index in enumerate(completed_indices):
                            block_index = batch_start + local_index
                            if block_index >= len(retrieve_blocks):
                                continue
                            _key, start, end = retrieve_blocks[block_index]
                            restore_offsets.append(completed_offsets[completed_pos])
                            restore_nbytes.append(completed_nbytes[completed_pos])
                            restore_starts.append(start)
                            restore_ends.append(end)
                            restore_shapes.append(completed_shapes[completed_pos])
                            restore_dtypes.append(completed_dtypes[completed_pos])
                            restore_sizes.append(completed_nbytes[completed_pos])

                        if not restore_offsets:
                            return
                        raw_restore = getattr(
                            self.gpu_connector,
                            "batched_raw_to_gpu",
                            None,
                        )
                        if not callable(raw_restore):
                            raise RuntimeError(
                                "raw Tutti retrieve selected without a GPU "
                                "connector raw restore method"
                            )
                        try:
                            raw_restore(
                                staging,
                                restore_offsets,
                                restore_nbytes,
                                restore_starts,
                                restore_ends,
                                restore_shapes,
                                restore_dtypes,
                                **kwargs,
                            )
                        except Exception:
                            streaming_failed = True
                            if restore_starts:
                                failed_start = restore_starts[0]
                                if (
                                    last_failed_block_start is None
                                    or last_failed_block_start > failed_start
                                ):
                                    last_failed_block_start = failed_start
                            raise
                        for start, end, size in zip(
                            restore_starts,
                            restore_ends,
                            restore_sizes,
                            strict=True,
                        ):
                            ret_mask[start:end] = True
                            tot_kv_size += size
                            streamed_blocks += 1

                    authoritative_keys = tutti_keys
                    memory_objs = self._tutti_batched_get(
                        authoritative_keys,
                        shapes_per_key=tutti_shapes,
                        read_ranges_per_key=tutti_read_ranges,
                        kv_object_roles=tutti_roles,
                        on_batch_loaded=(
                            _consume_tutti_batch
                            if stream_tutti_retrieve and not raw_tutti_retrieve
                            else None
                        ),
                        on_raw_batch_loaded=(
                            _consume_tutti_raw_batch if raw_tutti_retrieve else None
                        ),
                    )
                    if streaming_consumed:
                        if not streaming_failed and streamed_blocks < len(tutti_blocks):
                            missing_start = tutti_blocks[streamed_blocks][1]
                            if (
                                last_failed_block_start is None
                                or last_failed_block_start > missing_start
                            ):
                                last_failed_block_start = missing_start
                        logger.info(
                            "TUTTI_PROFILE streaming_retrieve blocks=%d/%d failed=%s",
                            streamed_blocks,
                            len(tutti_blocks),
                            streaming_failed,
                        )
                        continue
                elif self._tutti_can_cpu_fallback:
                    memory_objs = self.storage_manager.batched_get(
                        keys=keys,
                        location=location,
                        shapes_per_key=store_shapes_per_key,
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
                    shapes_per_key=store_shapes_per_key,
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
