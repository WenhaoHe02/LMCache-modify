# SPDX-License-Identifier: Apache-2.0
"""GPU-direct NVMe KV loader using Tutti's snvme kernel module.

Replaces the LocalDiskBackend → CPU pinned buffer → H2D scatter path
with NVMe DMA → HBM staging → G2G scatter (no CPU involvement after setup).

Architecture
------------
The loading path with this module active:

    NVMe controller
         │  (snvme GPU-direct DMA, no CPU staging)
         ▼
    HBM staging slot    ← TuttiDirectLoader manages this pool
         │  (G2G scatter via existing multi_layer_block_kv_transfer)
         ▼
    vLLM KV cache slots

Profiled savings on DSv4 rank:
  * Eliminates 0.38 s SSD → CPU read
  * Eliminates 0.95 s H2D scatter (PCIe bottleneck)
  * Replaces with ~0.38 s SSD → HBM + ~0.02 s G2G scatter

Requirements (Linux x86_64 only)
---------------------------------
  * snvme-core.ko + snvme.ko loaded
  * NVIDIA GPU with P2P / nvidia_p2p_get_pages support
  * Root or CAP_SYS_ADMIN (PCI bind requires it)
  * lmcache.c_ops built with csrc/tutti_kv_ops.cu

Usage
-----
    loader = TuttiDirectLoader.create(
        device_path="/dev/ssnvme0",
        ctrl_path="/dev/snvm_control",
        pci_bdf="0000:08:00.0",
    )
    gpu_objs = loader.load_chunks_to_hbm(keys, disk_metadatas)
    # gpu_objs[i].raw_tensor.is_cuda == True
    # existing gpu_connector.to_gpu() handles it via G2G copy
    loader.close()
"""

import bisect
import ctypes
import hashlib
import mmap
import os
import struct as _struct
import sys
import threading
import time
import types
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import torch

# ``fcntl`` is POSIX-only.  Import-time fallback keeps the heavily mocked unit
# tests importable on Windows; real snvme paths still fail loudly if called.
try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = types.SimpleNamespace(ioctl=None)  # type: ignore[assignment]
    sys.modules.setdefault("fcntl", fcntl)

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.kv_object_store import KVObjectByteRange
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata

# TensorMemoryObj lives here; import carefully to avoid circular deps
from lmcache.v1.memory_management import TensorMemoryObj

logger = init_logger(__name__)

# ── conditional import of CUDA ops ──────────────────────────────────────────
try:
    import lmcache.c_ops as _c_ops  # type: ignore[import]

    _HAS_C_OPS: bool = hasattr(_c_ops, "tutti_submit_batch_sgl_read")
    _HAS_WRITE_C_OPS: bool = hasattr(_c_ops, "tutti_submit_batch_sgl_write")
except ImportError:
    _c_ops = None  # type: ignore[assignment]
    _HAS_C_OPS = False
    _HAS_WRITE_C_OPS = False

# ── constants ────────────────────────────────────────────────────────────────

# GPU page size enforced by snvme (map.c:GPU_PAGE_SHIFT=16).
_GPU_PAGE_SIZE: int = 1 << 16  # 64 KiB

# NVMe logical block size (512-byte sectors).
_NVME_LBS: int = 512

# Default per-CQE spin budget (~100 ms at ~10 ns/iter on a GPU).
# Reduced from 500_000_000 (5s) so NVMe read failures time out quickly instead
# of stalling the entire forward pass.  _submit_reads catches the resulting
# RuntimeError and skips the prefetch gracefully.
_DEFAULT_MAX_ITERS: int = 10_000_000

# cudaHostRegisterIoMemory flag (CUDA runtime header).
_CUDA_HOST_REGISTER_IO_MEMORY: int = 0x04

RawBatchLoadedCallback = Callable[
    [int, list[int], list[int], list[int], torch.Tensor], None
]
IndexedBatchLoadedCallback = Callable[[int, torch.Tensor, int, int, torch.Tensor], None]
_LocalRawBatchLoadedCallback = Callable[
    [list[int], list[int], list[int], torch.Tensor], None
]


def _raw_write_window_ready(
    readers_waiting: int,
    idle_for_s: float,
    waited_s: float,
    write_slack_s: float,
    write_max_delay_s: float,
) -> bool:
    """Return whether a background raw write may acquire the I/O queue."""
    if readers_waiting > 0:
        return False
    return idle_for_s >= write_slack_s or waited_s >= write_max_delay_s


def _align_up(x: int, align: int) -> int:
    """Round x up to the next multiple of align."""
    return ((x + align - 1) // align) * align


def _elapsed_ms(start: float) -> float:
    """Return elapsed wall-clock milliseconds since start."""
    return (time.perf_counter() - start) * 1000.0


def _tutti_profile_enabled() -> bool:
    """Return whether Tutti hot-path profiling is enabled."""
    value = os.environ.get("LMCACHE_TUTTI_PROFILE")
    if value is None:
        value = os.environ.get("LMCACHE_CSA_ATTENTION_KV_TIMING", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _effective_shapes(
    disk_meta: DiskCacheMetadata,
    shapes_override: Optional[list[torch.Size]],
) -> Optional[list[torch.Size]]:
    """Return the logical group shapes to expose for one cached chunk."""
    return shapes_override if shapes_override is not None else disk_meta.shapes


def _effective_dtypes(
    disk_meta: DiskCacheMetadata,
    effective_shapes: Optional[list[torch.Size]],
) -> Optional[list[torch.dtype]]:
    """Return per-group dtypes matching ``effective_shapes``."""
    if effective_shapes is None:
        return disk_meta.dtypes
    if disk_meta.dtypes is not None:
        if len(disk_meta.dtypes) != len(effective_shapes):
            raise ValueError(
                f"DiskCacheMetadata for {disk_meta.path} has "
                f"{len(disk_meta.dtypes)} dtypes but {len(effective_shapes)} shapes"
            )
        return disk_meta.dtypes
    if disk_meta.dtype is None:
        raise ValueError(
            f"DiskCacheMetadata for {disk_meta.path} is missing dtype information"
        )
    return [disk_meta.dtype] * len(effective_shapes)


def _effective_nbytes(
    disk_meta: DiskCacheMetadata,
    shapes_override: Optional[list[torch.Size]],
) -> int:
    """Return the byte length Tutti must read for one cached chunk."""
    effective_shapes = _effective_shapes(disk_meta, shapes_override)
    if effective_shapes is None:
        return int(disk_meta.size)
    effective_dtypes = _effective_dtypes(disk_meta, effective_shapes)
    if effective_dtypes is None:
        return int(disk_meta.size)
    return int(
        sum(
            shape.numel() * dtype.itemsize
            for shape, dtype in zip(effective_shapes, effective_dtypes, strict=True)
        )
    )


def _logical_read_ranges(
    disk_meta: DiskCacheMetadata,
    shapes_override: Optional[list[torch.Size]],
    *,
    file_offset: int = 0,
    read_ranges: Optional[Sequence[KVObjectByteRange]] = None,
) -> tuple[KVObjectByteRange, ...]:
    """Return source ranges that compose one logical HBM payload.

    Args:
        disk_meta: Metadata for the backing file or object pool.
        shapes_override: Optional per-key logical shape override.
        file_offset: Contiguous source offset used when explicit ranges are
            not supplied.
        read_ranges: Optional object-store byte ranges.  When provided,
            ``target_offset`` controls where each range lands in staging.

    Returns:
        Source ranges covering the logical payload.
    """
    if read_ranges is not None:
        return tuple(read_ranges)
    return (
        KVObjectByteRange(
            offset=file_offset,
            length=_effective_nbytes(disk_meta, shapes_override),
            target_offset=0,
        ),
    )


def _logical_read_nbytes(
    disk_meta: DiskCacheMetadata,
    shapes_override: Optional[list[torch.Size]],
    *,
    file_offset: int = 0,
    read_ranges: Optional[Sequence[KVObjectByteRange]] = None,
) -> int:
    """Return logical bytes produced by one direct-read request."""
    return sum(
        byte_range.length
        for byte_range in _logical_read_ranges(
            disk_meta,
            shapes_override,
            file_offset=file_offset,
            read_ranges=read_ranges,
        )
    )


# ── ctypes helpers ───────────────────────────────────────────────────────────

_libcudart: Optional[ctypes.CDLL] = None


def _get_cudart() -> ctypes.CDLL:
    global _libcudart
    if _libcudart is None:
        for name in ("libcudart.so.12", "libcudart.so.11", "libcudart.so"):
            try:
                _libcudart = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if _libcudart is None:
            raise RuntimeError(
                "Cannot find libcudart.so – is CUDA installed and LD_LIBRARY_PATH set?"
            )
    return _libcudart


def _cuda_host_register(cpu_ptr: int, size: int) -> None:
    """Register CPU-mapped I/O memory (BAR0) with CUDA for GPU access."""
    ret = _get_cudart().cudaHostRegister(
        ctypes.c_void_p(cpu_ptr),
        ctypes.c_size_t(size),
        ctypes.c_uint(_CUDA_HOST_REGISTER_IO_MEMORY),
    )
    if ret != 0:
        _get_cudart().cudaGetLastError()
        raise RuntimeError(f"cudaHostRegister failed (error {ret})")


def _cuda_host_unregister(cpu_ptr: int) -> None:
    """Unregister memory previously registered with cudaHostRegister."""
    if cpu_ptr == 0:
        return
    ret = _get_cudart().cudaHostUnregister(ctypes.c_void_p(cpu_ptr))
    if ret != 0:
        _get_cudart().cudaGetLastError()
        logger.warning("cudaHostUnregister failed (error %d)", ret)


def _cuda_host_get_device_pointer(cpu_ptr: int) -> int:
    """Return the GPU VA for a CPU-side pointer registered with cudaHostRegister."""
    dev_ptr = ctypes.c_void_p(0)
    ret = _get_cudart().cudaHostGetDevicePointer(
        ctypes.byref(dev_ptr),
        ctypes.c_void_p(cpu_ptr),
        ctypes.c_uint(0),
    )
    if ret != 0:
        raise RuntimeError(f"cudaHostGetDevicePointer failed (error {ret})")
    result = dev_ptr.value
    if result is None:
        raise RuntimeError("cudaHostGetDevicePointer returned NULL")
    return result


def _cuda_malloc_managed(size: int) -> int:
    """Allocate CUDA managed (unified) memory accessible from both CPU and GPU."""
    ptr = ctypes.c_void_p(0)
    # cudaMemAttachGlobal = 1
    ret = _get_cudart().cudaMallocManaged(
        ctypes.byref(ptr),
        ctypes.c_size_t(size),
        ctypes.c_uint(1),
    )
    if ret != 0:
        _get_cudart().cudaGetLastError()
        raise RuntimeError(f"cudaMallocManaged failed (error {ret})")
    result = ptr.value
    if result is None:
        raise RuntimeError("cudaMallocManaged returned NULL")
    return result


def _cuda_free(ptr: int) -> None:
    if ptr == 0:
        return
    _get_cudart().cudaFree(ctypes.c_void_p(ptr))


def _cuda_malloc_device(size: int, device_id: int) -> int:
    """Allocate device memory on device_id via cudaMalloc (not PyTorch allocator).

    PyTorch 2.2+ on Hopper (CC 9.0) defaults to expandable_segments, which
    uses cuMemCreate/cuMemMap (Virtual Memory Management).  VMM allocations
    are invisible to nvidia_p2p_get_pages_persistent's RM VA-space scan and
    cause it to return EINVAL.  Using cudaMalloc directly avoids this path.

    Args:
        size:      Number of bytes to allocate.
        device_id: CUDA device index.

    Returns:
        Integer GPU virtual address of the allocated buffer.

    Raises:
        RuntimeError: If cudaSetDevice or cudaMalloc fails.
    """
    rt = _get_cudart()
    ret = rt.cudaSetDevice(ctypes.c_int(device_id))
    if ret != 0:
        raise RuntimeError(f"cudaSetDevice({device_id}) failed (error {ret})")
    ptr = ctypes.c_void_p(0)
    ret = rt.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    if ret != 0:
        # Consume the CUDA error from the runtime's thread-local state so it
        # does not surface later as a spurious "CUDA error [ASYNC]" in PyTorch
        # when control returns to Python torch operations.
        rt.cudaGetLastError()
        raise RuntimeError(
            f"cudaMalloc({size} bytes, device {device_id}) failed (error {ret})"
        )
    result = ptr.value
    if result is None:
        raise RuntimeError("cudaMalloc returned NULL")
    return result


class _ExternalCudaBuffer:
    """Exposes a raw CUDA pointer via the CUDA Array Interface.

    ``torch.as_tensor()`` reads ``__cuda_array_interface__`` and creates a
    non-owning CUDA tensor that shares the underlying memory without copying
    or taking ownership.  The caller is responsible for calling
    ``_cuda_free()`` after all tensors derived from this buffer are done.

    Attributes:
        __cuda_array_interface__: CUDA Array Interface v3 descriptor dict.
    """

    __cuda_array_interface__: dict

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "strides": None,
            "version": 3,
        }


# ── ioctl number helpers (Linux/x86_64 ABI) ──────────────────────────────────

_IOC_WRITE: int = 1
_IOC_READ: int = 2


def _ioc(dir_: int, type_: int, nr: int, size: int) -> int:
    return (dir_ << 30) | ((size & 0x3FFF) << 16) | ((type_ & 0xFF) << 8) | (nr & 0xFF)


def _IOW(type_: int, nr: int, struct_type: type) -> int:  # noqa: N802
    return _ioc(_IOC_WRITE, type_, nr, ctypes.sizeof(struct_type))


def _IOR(type_: int, nr: int, struct_type: type) -> int:  # noqa: N802
    return _ioc(_IOC_READ, type_, nr, ctypes.sizeof(struct_type))


def _IOWR(type_: int, nr: int, struct_type: type) -> int:  # noqa: N802
    return _ioc(_IOC_READ | _IOC_WRITE, type_, nr, ctypes.sizeof(struct_type))


def _IOW_uint32(type_: int, nr: int) -> int:  # noqa: N802
    return _ioc(_IOC_WRITE, type_, nr, ctypes.sizeof(ctypes.c_uint32))


def _IOW_uint64(type_: int, nr: int) -> int:  # noqa: N802
    return _ioc(_IOC_WRITE, type_, nr, ctypes.sizeof(ctypes.c_uint64))


# ── ctypes struct definitions (must match ioctl.h exactly on x86_64) ─────────


class _PciDeviceAddr(ctypes.Structure):
    _fields_ = [
        ("domain", ctypes.c_int),
        ("bus", ctypes.c_int),
        ("slot", ctypes.c_int),
        ("func", ctypes.c_int),
    ]


class _NvmIoctlMap(ctypes.Structure):
    _fields_ = [
        ("vaddr_start", ctypes.c_uint64),
        ("n_pages", ctypes.c_size_t),
        ("ioaddrs", ctypes.POINTER(ctypes.c_uint64)),
        ("ioq_idx", ctypes.c_int),
        ("is_cq", ctypes.c_int),
        ("group_id", ctypes.c_uint32),
        ("map_kind", ctypes.c_uint8),
        ("reserved0", ctypes.c_uint8 * 3),
    ]


class _NvmIoctlDev(ctypes.Structure):
    _fields_ = [
        ("nr_user_q", ctypes.c_uint32),
        ("start_cq_idx", ctypes.c_uint32),
        ("dstrd", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8 * 7),  # align size_t to 8-byte boundary
        ("max_data_size", ctypes.c_size_t),
        ("block_size", ctypes.c_size_t),
        ("disk_name", ctypes.c_char * 32),
        ("q_depth", ctypes.c_uint16),
        ("reserved0", ctypes.c_uint16),
        ("bar0_size", ctypes.c_uint32),
        ("max_user_qid", ctypes.c_uint32),
        ("max_queues_per_group", ctypes.c_uint32),
        ("sgl_supported", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32 * 5),
    ]


class _NvmIoctlQueueGroup(ctypes.Structure):
    _fields_ = [
        ("group_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("max_queues", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 5),
    ]


class _NvmUserQueuePairIn(ctypes.Structure):
    _fields_ = [
        ("sq_vaddr", ctypes.c_uint64),
        ("cq_vaddr", ctypes.c_uint64),
    ]


class _NvmUserQueuePairOut(ctypes.Structure):
    _fields_ = [
        ("sq_doorbell_offset", ctypes.c_uint32),
        ("cq_doorbell_offset", ctypes.c_uint32),
        ("qid", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


_NVM_MAX_QUEUES_PER_GROUP: int = 16


class _NvmIoctlAddUserQueue(ctypes.Structure):
    _fields_ = [
        ("group_id", ctypes.c_uint32),
        ("nr_pairs", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 5),
        ("pairs", _NvmUserQueuePairIn * _NVM_MAX_QUEUES_PER_GROUP),
        ("out_pairs", _NvmUserQueuePairOut * _NVM_MAX_QUEUES_PER_GROUP),
    ]


# ── FIEMAP structs (linux/fiemap.h) ──────────────────────────────────────────


class _FiemapExtent(ctypes.Structure):
    _fields_ = [
        ("fe_logical", ctypes.c_uint64),
        ("fe_physical", ctypes.c_uint64),
        ("fe_length", ctypes.c_uint64),
        ("fe_reserved64", ctypes.c_uint64 * 2),
        ("fe_flags", ctypes.c_uint32),
        ("fe_reserved", ctypes.c_uint32 * 3),
    ]


class _FiemapHeader(ctypes.Structure):
    _fields_ = [
        ("fm_start", ctypes.c_uint64),
        ("fm_length", ctypes.c_uint64),
        ("fm_flags", ctypes.c_uint32),
        ("fm_mapped_extents", ctypes.c_uint32),
        ("fm_extent_count", ctypes.c_uint32),
        ("fm_reserved", ctypes.c_uint32),
    ]


# ── ioctl numbers (computed at import time) ───────────────────────────────────

_NVM_IOCTL_TYPE: int = 0x80
_CTRL_IOCTL_TYPE: int = 0x90

_NVM_MAP_DEVICE_MEMORY: int = _IOW(_NVM_IOCTL_TYPE, 2, _NvmIoctlMap)
_NVM_UNMAP_DEVICE_MEMORY: int = _IOW_uint64(_NVM_IOCTL_TYPE, 5)
_NVM_GET_DEV_INFO: int = _IOR(_NVM_IOCTL_TYPE, 9, _NvmIoctlDev)
_NVM_CREATE_QUEUE_GROUP: int = _IOWR(_NVM_IOCTL_TYPE, 12, _NvmIoctlQueueGroup)
_NVM_DESTROY_QUEUE_GROUP: int = _IOW_uint32(_NVM_IOCTL_TYPE, 13)
_NVM_ADD_USER_QUEUE: int = _IOWR(_NVM_IOCTL_TYPE, 14, _NvmIoctlAddUserQueue)
_NVM_SET_KERNEL_IOQ_CAP: int = _IOW_uint32(_NVM_IOCTL_TYPE, 15)

_SNVM_DEVICE_BIND: int = _IOW(_CTRL_IOCTL_TYPE, 1, _PciDeviceAddr)
_SNVM_CHRDEV_CREATE: int = _IOWR(_CTRL_IOCTL_TYPE, 3, _PciDeviceAddr)

# FS_IOC_FIEMAP = _IOWR('f', 11, struct fiemap_header)
_FS_IOC_FIEMAP: int = _IOWR(ord("f"), 11, _FiemapHeader)
_FIEMAP_EXTENT_LAST: int = 0x00000001

# snvme NVM_MAP_KIND_* enum values (ioctl.h)
_NVM_MAP_KIND_DATA: int = 3
_NVM_MAP_KIND_RING_SQ: int = 1
_NVM_MAP_KIND_RING_CQ: int = 2


# ── low-level ioctl helper ────────────────────────────────────────────────────


def _ioctl(fd: int, request: int, arg: ctypes.Structure) -> None:
    """Call ioctl(fd, request, &arg), updating arg fields in-place."""
    if fcntl.ioctl is None:
        raise OSError("fcntl is unavailable on this platform")
    size = ctypes.sizeof(arg)
    buf = ctypes.create_string_buffer(bytes(arg), size)
    fcntl.ioctl(fd, request, buf, True)
    ctypes.memmove(ctypes.addressof(arg), buf, size)


def _ioctl_u64(fd: int, request: int, value: int) -> None:
    """Call ioctl(fd, request, &uint64_t(value))."""
    if fcntl.ioctl is None:
        raise OSError("fcntl is unavailable on this platform")
    fcntl.ioctl(fd, request, _struct.pack("Q", value))


def _clear_driver_override(pci_bdf: str) -> None:
    """Clear Linux PCI driver_override before snvme attaches a controller."""
    override_path = f"/sys/bus/pci/devices/{pci_bdf}/driver_override"
    try:
        with open(override_path, "w", encoding="utf-8") as override_file:
            override_file.write("\n")
    except OSError as exc:
        logger.warning(
            "Could not clear driver_override for %s before snvme bind: %s",
            pci_bdf,
            exc,
        )


# ── FIEMAP helper ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LbaRecord:
    """Physical location of one KV chunk on an NVMe device."""

    slba: int  # starting logical block address (512-byte sectors)
    n_sectors: int  # number of 512-byte sectors
    file_offset: int = 0  # logical byte offset inside the file


class FiemapHelper:
    """Query physical LBA extents for a file using the Linux FIEMAP ioctl.

    Works on ext4, xfs, btrfs and most Linux block-device-backed filesystems.
    The filesystem must not have inline data or extent encryption for the
    physical offsets to be usable for direct NVMe I/O.
    """

    _MAX_EXTENTS: int = 256

    @staticmethod
    def query_extents(file_path: str) -> list[LbaRecord]:
        """Return a list of LbaRecords for every extent of file_path.

        Args:
            file_path: Absolute path to the file.

        Returns:
            Ordered list of LbaRecord covering the file's physical layout.

        Raises:
            FileNotFoundError: If file_path does not exist.
            OSError: If the FIEMAP ioctl fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if fcntl.ioctl is None:
            raise OSError("fcntl is unavailable on this platform")

        records: list[LbaRecord] = []
        n = FiemapHelper._MAX_EXTENTS
        hdr_size = ctypes.sizeof(_FiemapHeader)
        ext_size = ctypes.sizeof(_FiemapExtent)
        file_size = os.path.getsize(file_path)
        start = 0

        with open(file_path, "rb") as f:
            while start < file_size:
                buf = ctypes.create_string_buffer(hdr_size + n * ext_size)
                hdr = _FiemapHeader.from_buffer(buf)
                hdr.fm_start = start
                hdr.fm_length = 0xFFFF_FFFF_FFFF_FFFF
                hdr.fm_flags = 0
                hdr.fm_extent_count = n
                fcntl.ioctl(f.fileno(), _FS_IOC_FIEMAP, buf, True)

                if hdr.fm_mapped_extents == 0:
                    break

                last_end = start
                saw_last = False
                for i in range(hdr.fm_mapped_extents):
                    ext = _FiemapExtent.from_buffer(buf, hdr_size + i * ext_size)
                    if ext.fe_length <= 0:
                        continue
                    records.append(
                        LbaRecord(
                            slba=ext.fe_physical // _NVME_LBS,
                            n_sectors=ext.fe_length // _NVME_LBS,
                            file_offset=ext.fe_logical,
                        )
                    )
                    last_end = max(last_end, ext.fe_logical + ext.fe_length)
                    saw_last = saw_last or bool(ext.fe_flags & _FIEMAP_EXTENT_LAST)
                if saw_last or hdr.fm_mapped_extents < n or last_end <= start:
                    break
                start = last_end
        return sorted(records, key=lambda record: record.file_offset)

    @staticmethod
    def single_contiguous(file_path: str) -> LbaRecord:
        """Return the LbaRecord for a file that occupies a single extent.

        Args:
            file_path: Absolute path to the file.

        Returns:
            LbaRecord for the file's single physical extent.

        Raises:
            ValueError: If the file is fragmented (more than one extent).
            FileNotFoundError: If file_path does not exist.
        """
        extents = FiemapHelper.query_extents(file_path)
        if len(extents) == 0:
            raise ValueError(f"No extents found for {file_path}")
        if len(extents) > 1:
            raise ValueError(
                f"{file_path} has {len(extents)} extents; "
                "defragment the file or handle multi-extent loading"
            )
        return extents[0]

    @staticmethod
    def scan_paths(
        file_paths: list[str],
    ) -> dict[str, list[LbaRecord]]:
        """Pre-compute LBA records for a list of files.

        Must be called BEFORE snvme DEVICE_BIND (while the filesystem is
        still accessible).  Returns a dict suitable for passing as
        ``initial_lba_cache`` to ``TuttiDirectLoader.create()``.

        Inaccessible files are skipped and will not appear in the returned
        dict. Fragmented files are kept as multiple records so the loader can
        issue multiple NVMe reads into one HBM staging slot.

        Args:
            file_paths: List of absolute paths to scan.

        Returns:
            Dict mapping file path → LbaRecord for successfully scanned files.
        """
        result: dict[str, list[LbaRecord]] = {}
        for path in file_paths:
            try:
                extents = FiemapHelper.query_extents(path)
            except Exception as exc:
                logger.warning("Tutti FIEMAP pre-scan failed for %s: %s", path, exc)
                continue
            if extents:
                result[path] = extents
            else:
                logger.warning("Tutti FIEMAP pre-scan found no extents for %s", path)
        return result


# ── SnvmeSession: device lifecycle ───────────────────────────────────────────


@dataclass
class _QueueResources:
    """GPU-resident SQ/CQ rings for one NVMe user I/O queue."""

    sq_tensor: torch.Tensor
    cq_tensor: torch.Tensor
    sq_db_offset: int
    cq_db_offset: int
    qid: int
    group_id: int


class SnvmeSession:
    """Manages one snvme device session.

    Performs the full setup sequence documented in ioctl.h:
    1. Open /dev/snvm_control → SNVM_CHRDEV_CREATE → open /dev/ssnvme<N>
    2. NVM_SET_KERNEL_IOQ_CAP → SNVM_DEVICE_BIND → NVM_GET_DEV_INFO
    3. mmap(BAR0) → cudaHostRegister → cudaHostGetDevicePointer
    4. Register staging DATA buffer via NVM_MAP_DEVICE_MEMORY(group_id=0)
    5. NVM_CREATE_QUEUE_GROUP → allocate GPU rings → NVM_MAP_DEVICE_MEMORY
       (RING_SQ/RING_CQ) → NVM_ADD_USER_QUEUE

    Linux x86_64 only; requires CAP_SYS_ADMIN for SNVM_DEVICE_BIND.
    """

    def __init__(
        self,
        device_path: str,
        ctrl_path: str,
        pci_bdf: str,
        staging_tensor: torch.Tensor,
        nsid: int = 1,
        kernel_ioq_cap: int = 36,
        cuda_device: int = 0,
    ) -> None:
        """
        Args:
            device_path:    Path to the snvme char-dev (e.g. /dev/ssnvme0).
            ctrl_path:      Path to the snvme control device (/dev/snvm_control).
            pci_bdf:        PCI BDF string "DDDD:BB:SS.F".
            staging_tensor: Pre-allocated GPU uint8 staging pool.
                            Must be contiguous and GPU_PAGE_SIZE-aligned.
            nsid:           NVMe namespace ID to use for I/O commands.
            kernel_ioq_cap: Cap on kernel-side IOQ count (default 36).
            cuda_device:    CUDA device index for GPU memory allocations.
                            Must be NUMA-local to the NVMe controller (same
                            PCIe root complex) for nvidia_p2p_get_pages to
                            succeed.
        """
        if not staging_tensor.is_cuda:
            raise ValueError("staging_tensor must be a CUDA tensor")
        if not staging_tensor.is_contiguous():
            raise ValueError("staging_tensor must be contiguous")

        self._device_path = device_path
        self._nsid = nsid
        self._cuda_device = cuda_device
        self._fd_dev: int = -1
        self._fd_ctrl: int = -1
        self._bar0_mmap: Optional[mmap.mmap] = None
        self._bar0_arr: Optional[ctypes.Array] = None
        self._bar0_cpu_ptr: int = 0
        self._bar0_gpu_ptr: int = 0
        self._info: Optional[_NvmIoctlDev] = None
        self._queue: Optional[_QueueResources] = None
        self._staging_iovas: list[int] = []
        self._staging_data_mapped: bool = False
        self._staging_tensor = staging_tensor

        try:
            self._setup(ctrl_path, pci_bdf, kernel_ioq_cap)
        except Exception:
            self.close()
            raise

    def _parse_bdf(self, bdf: str) -> _PciDeviceAddr:
        parts = bdf.replace(":", " ").replace(".", " ").split()
        if len(parts) != 4:
            raise ValueError(f"Invalid PCI BDF: {bdf!r}; expected format DDDD:BB:SS.F")
        return _PciDeviceAddr(
            domain=int(parts[0], 16),
            bus=int(parts[1], 16),
            slot=int(parts[2], 16),
            func=int(parts[3], 16),
        )

    def _map_device_memory(
        self,
        dev_ptr: int,
        n_pages: int,
        group_id: int,
        map_kind: int,
    ) -> list[int]:
        """Register n_pages of GPU memory and return their NVMe IOVAs."""
        align = dev_ptr % _GPU_PAGE_SIZE
        logger.debug(
            "NVM_MAP_DEVICE_MEMORY: ptr=0x%x align=%d n_pages=%d kind=%d",
            dev_ptr,
            align,
            n_pages,
            map_kind,
        )
        if align != 0:
            raise ValueError(
                f"GPU pointer 0x{dev_ptr:x} is not {_GPU_PAGE_SIZE}-byte "
                f"(64 KiB) aligned (misalignment={align}); "
                "nvidia_p2p_get_pages requires GPU_PAGE_SIZE alignment"
            )
        ioaddrs_buf = (ctypes.c_uint64 * n_pages)()
        req = _NvmIoctlMap(
            vaddr_start=dev_ptr,
            n_pages=n_pages,
            ioaddrs=ioaddrs_buf,
            ioq_idx=-1,
            is_cq=-1,
            group_id=group_id,
            map_kind=map_kind,
        )
        _ioctl(self._fd_dev, _NVM_MAP_DEVICE_MEMORY, req)
        return [ioaddrs_buf[i] for i in range(n_pages)]

    @staticmethod
    def _ensure_dev_node(ctrl_path: str, dev_path: str, minor: int) -> None:
        """Create the ssnvme<N> char-device node if it does not exist.

        Inside a Docker container the kernel creates the node on the host but
        not inside the container's /dev.  We derive the major from
        /proc/devices ("libsnvm helper") and call os.mknod().
        """
        if os.path.exists(dev_path):
            return
        major = 0
        try:
            with open("/proc/devices") as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2 and parts[1] == "libsnvm helper":
                        major = int(parts[0])
                        break
        except OSError:
            pass
        if major == 0:
            major = os.major(os.stat(ctrl_path).st_rdev)
        import stat as _stat

        os.mknod(dev_path, 0o660 | _stat.S_IFCHR, os.makedev(major, minor))
        logger.debug("Created device node %s (%d:%d)", dev_path, major, minor)

    def _setup(
        self,
        ctrl_path: str,
        pci_bdf: str,
        kernel_ioq_cap: int,
    ) -> None:
        bdf = self._parse_bdf(pci_bdf)

        # 1. Open control device and create per-controller char-dev.
        #    SNVM_CHRDEV_CREATE writes the minor number back into
        #    addr.domain (the kernel modifies the struct in-place) and
        #    returns 0 on success.  Do NOT read the minor from the ioctl
        #    return value (that would always be 0).
        self._fd_ctrl = os.open(ctrl_path, os.O_RDWR)
        bdf_create = self._parse_bdf(pci_bdf)
        size = ctypes.sizeof(bdf_create)
        buf = ctypes.create_string_buffer(bytes(bdf_create), size)
        fcntl.ioctl(self._fd_ctrl, _SNVM_CHRDEV_CREATE, buf, True)
        bdf_result = _PciDeviceAddr.from_buffer_copy(buf)
        minor = bdf_result.domain
        logger.debug("SNVM_CHRDEV_CREATE: minor=%d", minor)
        # Derive the authoritative device path from the minor returned by the kernel.
        # The caller-supplied device_path may point to a stale or wrong ssnvme node
        # when re-using an existing /dev/ssnvme0 for a different BDF.
        ctrl_dev_dir = os.path.dirname(ctrl_path)
        self._device_path = os.path.join(ctrl_dev_dir, f"ssnvme{minor}")
        self._ensure_dev_node(ctrl_path, self._device_path, minor)

        # 2. Open the per-controller device, set cap (per-dev ioctl), bind (ctrl ioctl).
        self._fd_dev = os.open(self._device_path, os.O_RDWR)
        cap_buf = _struct.pack("I", kernel_ioq_cap)
        fcntl.ioctl(self._fd_dev, _NVM_SET_KERNEL_IOQ_CAP, cap_buf)
        _clear_driver_override(pci_bdf)
        _ioctl(self._fd_ctrl, _SNVM_DEVICE_BIND, bdf)

        # 3. Query device info.
        info = _NvmIoctlDev()
        _ioctl(self._fd_dev, _NVM_GET_DEV_INFO, info)
        self._info = info

        logger.info(
            "NVM_GET_DEV_INFO: max_data_size=%d MiB  block_size=%d  "
            "sgl_supported=0x%08x  q_depth=%d  bar0_size=%d  disk=%s",
            info.max_data_size >> 20,
            info.block_size,
            info.sgl_supported,
            info.q_depth,
            info.bar0_size,
            info.disk_name.decode(errors="replace"),
        )

        if not (info.sgl_supported & 0x3):
            raise RuntimeError(
                f"NVMe controller does not support SGL "
                f"(sgl_supported=0x{info.sgl_supported:08x})"
            )

        # 4. Map BAR0 for GPU doorbell writes.
        logger.info("BAR0 mmap: fd_dev=%d bar0_size=%d", self._fd_dev, info.bar0_size)
        self._bar0_mmap = mmap.mmap(
            self._fd_dev,
            info.bar0_size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        arr_type = ctypes.c_char * info.bar0_size
        self._bar0_arr = arr_type.from_buffer(self._bar0_mmap)
        bar0_cpu = ctypes.addressof(self._bar0_arr)
        self._bar0_cpu_ptr = bar0_cpu
        logger.info(
            "cudaHostRegister BAR0: ptr=0x%x size=%d device=%d",
            bar0_cpu,
            info.bar0_size,
            self._cuda_device,
        )
        _cuda_host_register(bar0_cpu, info.bar0_size)
        self._bar0_gpu_ptr = _cuda_host_get_device_pointer(bar0_cpu)

        # 5. Register staging buffer as persistent DATA pool (survives group destroy).
        staging_pages = len(self._staging_tensor) // _GPU_PAGE_SIZE
        if staging_pages == 0:
            raise ValueError("staging_tensor must be at least one GPU page (64 KiB)")
        self._staging_iovas = self._map_device_memory(
            dev_ptr=self._staging_tensor.data_ptr(),
            n_pages=staging_pages,
            group_id=0,
            map_kind=_NVM_MAP_KIND_DATA,
        )
        self._staging_data_mapped = True
        # Validate physical contiguity (required for single-SGL-descriptor per slot).
        for i in range(1, len(self._staging_iovas)):
            if self._staging_iovas[i] != self._staging_iovas[i - 1] + _GPU_PAGE_SIZE:
                raise RuntimeError(
                    f"GPU pages are not physically contiguous at index {i}; "
                    "cudaMalloc should always give contiguous HBM pages"
                )

        # 6. Create queue group and one user I/O queue.
        self._queue = self._create_queue(info)

    def _create_queue(self, info: _NvmIoctlDev) -> _QueueResources:
        grp = _NvmIoctlQueueGroup()
        _ioctl(self._fd_dev, _NVM_CREATE_QUEUE_GROUP, grp)
        group_id = grp.group_id

        # Allocate one GPU page per ring, aligned to GPU_PAGE_SIZE (64 KiB).
        # Allocate 2× headroom and slice to ensure 64 KiB alignment for P2P.
        # Must use the same CUDA device as the staging buffer so that
        # nvidia_p2p_get_pages sees a consistent PCIe topology.
        _dev = f"cuda:{self._cuda_device}"

        def _alloc_aligned_page() -> torch.Tensor:
            raw = torch.zeros(2 * _GPU_PAGE_SIZE, dtype=torch.uint8, device=_dev)
            off = (-raw.data_ptr()) % _GPU_PAGE_SIZE
            return raw[off : off + _GPU_PAGE_SIZE] if off else raw[:_GPU_PAGE_SIZE]

        sq_tensor = _alloc_aligned_page()
        cq_tensor = _alloc_aligned_page()

        self._map_device_memory(
            sq_tensor.data_ptr(), 1, group_id, _NVM_MAP_KIND_RING_SQ
        )
        self._map_device_memory(
            cq_tensor.data_ptr(), 1, group_id, _NVM_MAP_KIND_RING_CQ
        )

        add_req = _NvmIoctlAddUserQueue()
        add_req.group_id = group_id
        add_req.nr_pairs = 1
        add_req.pairs[0].sq_vaddr = sq_tensor.data_ptr()
        add_req.pairs[0].cq_vaddr = cq_tensor.data_ptr()
        _ioctl(self._fd_dev, _NVM_ADD_USER_QUEUE, add_req)

        return _QueueResources(
            sq_tensor=sq_tensor,
            cq_tensor=cq_tensor,
            sq_db_offset=add_req.out_pairs[0].sq_doorbell_offset,
            cq_db_offset=add_req.out_pairs[0].cq_doorbell_offset,
            qid=add_req.out_pairs[0].qid,
            group_id=group_id,
        )

    @property
    def info(self) -> _NvmIoctlDev:
        if self._info is None:
            raise RuntimeError("SnvmeSession not initialised")
        return self._info

    @property
    def queue(self) -> _QueueResources:
        if self._queue is None:
            raise RuntimeError("SnvmeSession not initialised")
        return self._queue

    @property
    def staging_iovas(self) -> list[int]:
        """NVMe IOVAs of the staging pool's GPU pages (one per 64 KiB)."""
        return self._staging_iovas

    @property
    def nsid(self) -> int:
        return self._nsid

    def db_gpu_ptr(self, bar0_offset: int) -> int:
        """GPU pointer to a BAR0 doorbell register at byte offset bar0_offset."""
        return self._bar0_gpu_ptr + bar0_offset

    def close(self) -> None:
        """Tear down the session, releasing all NVMe and GPU resources."""
        if self._queue is not None and self._fd_dev >= 0:
            grp_id_buf = _struct.pack("I", self._queue.group_id)
            try:
                fcntl.ioctl(self._fd_dev, _NVM_DESTROY_QUEUE_GROUP, grp_id_buf)
            except OSError as exc:
                logger.warning("NVM_DESTROY_QUEUE_GROUP failed: %s", exc)
            self._queue = None

        if self._staging_data_mapped and self._fd_dev >= 0:
            try:
                _ioctl_u64(
                    self._fd_dev,
                    _NVM_UNMAP_DEVICE_MEMORY,
                    self._staging_tensor.data_ptr(),
                )
            except OSError as exc:
                logger.warning("NVM_UNMAP_DEVICE_MEMORY(DATA) failed: %s", exc)
            self._staging_data_mapped = False
            self._staging_iovas = []
        self._staging_tensor = torch.empty(0, dtype=torch.uint8)

        if self._bar0_cpu_ptr != 0:
            _cuda_host_unregister(self._bar0_cpu_ptr)
            self._bar0_gpu_ptr = 0
            self._bar0_cpu_ptr = 0

        if self._bar0_mmap is not None:
            self._bar0_arr = None  # release ctypes ref before closing mmap
            self._bar0_mmap.close()
            self._bar0_mmap = None

        if self._fd_dev >= 0:
            os.close(self._fd_dev)
            self._fd_dev = -1
        if self._fd_ctrl >= 0:
            os.close(self._fd_ctrl)
            self._fd_ctrl = -1

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "SnvmeSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ── TuttiDirectLoader ─────────────────────────────────────────────────────────


def _make_memory_obj_metadata(
    disk_meta: DiskCacheMetadata,
    shapes_override: Optional[list[torch.Size]] = None,
) -> MemoryObjMetadata:
    """Build a MemoryObjMetadata from DiskCacheMetadata.

    Handles both the single-tensor case (shape/dtype set) and the
    multi-tensor case (only shapes/dtypes set, as in layerwise storage).

    Args:
        disk_meta:      Metadata retrieved from the local disk backend.
        shapes_override: When provided, overrides ``disk_meta.shapes`` for
            the constructed metadata.  Used by the DSV4-optimised KV path
            where non-tail chunks store only prefix groups; the override
            carries the correct partial-group shapes so that the returned
            TensorMemoryObj's metadata is consistent with its byte layout.

    Returns:
        MemoryObjMetadata suitable for wrapping the staged GPU tensor.

    Raises:
        ValueError: If neither disk_meta nor the override can supply a valid
            shape/dtype.
    """
    effective_shapes = _effective_shapes(disk_meta, shapes_override)
    effective_dtypes = _effective_dtypes(disk_meta, effective_shapes)
    effective_size = _effective_nbytes(disk_meta, shapes_override)
    shape = disk_meta.shape
    dtype = disk_meta.dtype
    if shape is None and effective_shapes:
        shape = effective_shapes[0]
    if dtype is None and effective_dtypes:
        dtype = effective_dtypes[0]
    if shape is None or dtype is None:
        raise ValueError(
            f"DiskCacheMetadata for {disk_meta.path} is missing shape/dtype"
        )
    return MemoryObjMetadata(
        shape=shape,
        dtype=dtype,
        address=0,
        phy_size=effective_size,
        ref_count=1,
        pin_count=0,
        fmt=disk_meta.fmt or MemoryFormat.KV_2LTD,
        shapes=effective_shapes,
        dtypes=effective_dtypes,
    )


class TuttiDirectLoader:
    """GPU-direct NVMe KV cache loader.

    Replaces LocalDiskBackend → CPU staging → H2D scatter with
    NVMe DMA → HBM staging → G2G scatter.

    The caller integrates at the ``_process_tokens_internal`` level in
    ``CacheEngine``: for chunks known to be on local disk, call
    ``load_chunks_to_hbm`` instead of ``storage_manager.batched_get``.
    The returned ``TensorMemoryObj`` instances have ``raw_tensor`` on GPU;
    the existing ``gpu_connector.to_gpu(use_gpu=True)`` path handles them
    correctly (``copy_`` becomes G2G, which is ~50× faster than H2D).
    """

    # Maximum iterations to spin in k_poll_batch per CQE.
    POLL_MAX_ITERS: int = _DEFAULT_MAX_ITERS

    def __init__(
        self,
        session: SnvmeSession,
        staging_tensor: torch.Tensor,
        staging_raw_ptr: int,
        n_slots: int,
        slot_gpu_pages: int,
        sq_tail_ptr: int,
        cq_head_ptr: int,
        cq_phase_ptr: int,
        timed_out_ptr: int,
        status_buf: torch.Tensor,
        cuda_device: int = 0,
        initial_lba_cache: Optional[dict[str, list[LbaRecord]]] = None,
        debug_expected_checksums: Optional[dict[str, tuple[int, str]]] = None,
    ) -> None:
        self._session = session
        self._staging = staging_tensor
        self._staging_raw_ptr = staging_raw_ptr
        self._n_slots = n_slots
        self._slot_gpu_pages = slot_gpu_pages
        self._slot_bytes = slot_gpu_pages * _GPU_PAGE_SIZE
        self._cuda_device = cuda_device
        # Serialises every NVMe submit/poll cycle (loads AND raw stores).
        # The SQ/CQ ring, doorbells, staging pool, and status buffer are all
        # single-instance per loader; the CSA prefetch path calls
        # load_chunks_to_hbm from proxy executor threads while the main
        # thread may be inside another load or a raw store.  Unsynchronised
        # concurrent submit/poll interleaves ring updates and staging reuse,
        # which manifests as NVMe poll timeouts, kernel-side
        # cuda-EvtHandlr/cache_mgr_main blocking, and CUDA illegal memory
        # access (see docs/design/v1/csa_prefetch_illegal_access_bug.md).
        self._io_lock = threading.Lock()
        # High-priority (synchronous retrieve) readers currently inside
        # load_chunks_to_hbm.  Bulk prefetch batches wait until this drops
        # to zero before grabbing _io_lock for their next batch.
        self._hp_readers = 0
        # Read-priority gate over _io_lock.  Deferred HCA store writes
        # (background, latency-insensitive, ~27 ms p90 each and thousands
        # per request) share the single NVMe queue with prefetch/retrieve
        # reads (critical path, ~3 ms each).  With a bare lock, a read
        # arriving while a write burst holds the lock queues behind the
        # WHOLE burst -- measured as multi-second correction-wait tails
        # (p99 spikes up to 13 s) even though median reads are fully
        # overlapped.  Readers announce on a SEPARATE mutex (so the
        # announcement is never blocked by an in-flight write) and writers
        # re-check between stores, yielding until no reader is waiting.
        self._readers_waiting = 0
        # Writers waiting for an idle window are advisory only. They must not
        # cancel speculative reads during an active request: doing so made a
        # deferred LMCache seed writer suppress every CSA prediction for as
        # long as the writer was parked. ``_active_writers`` is set only after
        # the idle-window policy admits one write; speculative reads yield to
        # that bounded operation, while parked writers continue to wait.
        self._writers_waiting = 0
        self._active_writers = 0
        self._reader_gate = threading.Condition(threading.Lock())
        # Slack-only write scheduling (Tutti design: reads own the queue
        # while a request is active; writes go into idle windows).  A write
        # may proceed only when no read has touched the queue for
        # ``_write_slack_s`` seconds -- during prefill, prefetch reads
        # arrive every few ms, so writers stay parked until the
        # inter-request gap and then drain at full speed.
        # ``_write_max_delay_s`` lets an overdue writer bypass the idle-slack
        # test after announced readers drain. It never bypasses an active
        # reader: otherwise thousands of overdue cold-store tasks can barge
        # ahead of a new request and create multi-second hit-1 tails.
        self._last_read_end = 0.0
        self._write_slack_s = float(
            os.environ.get("LMCACHE_TUTTI_WRITE_SLACK_SEC", "0.05")
        )
        self._write_max_delay_s = float(
            os.environ.get("LMCACHE_TUTTI_WRITE_MAX_DELAY_SEC", "2.0")
        )
        speculative_rate_mbps = float(
            os.environ.get("LMCACHE_TUTTI_SPECULATIVE_RATE_MBPS", "256")
        )
        speculative_burst_mb = float(
            os.environ.get("LMCACHE_TUTTI_SPECULATIVE_BURST_MB", "8")
        )
        if speculative_rate_mbps < 0:
            raise ValueError("LMCACHE_TUTTI_SPECULATIVE_RATE_MBPS must be non-negative")
        if speculative_burst_mb <= 0:
            raise ValueError("LMCACHE_TUTTI_SPECULATIVE_BURST_MB must be positive")
        self._speculative_rate_bytes_per_s = speculative_rate_mbps * 1024**2
        self._speculative_burst_bytes = speculative_burst_mb * 1024**2
        self._speculative_tokens = self._speculative_burst_bytes
        self._speculative_last_refill = time.perf_counter()
        # Dedicated stream for the submit/poll spin kernels.  On the default
        # stream a polling kernel hard-serialises with model forward kernels
        # (prefetch reads stop overlapping compute entirely) and a slow NVMe
        # completion turns the spin into a global freeze: forward and NCCL
        # kernels queue behind it on every rank that fired a read.
        self._io_stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available():
            self._io_stream = torch.cuda.Stream(device=cuda_device)

        # Managed-memory control scalars (writable from both CPU and GPU).
        self._sq_tail_ptr = sq_tail_ptr
        self._cq_head_ptr = cq_head_ptr
        self._cq_phase_ptr = cq_phase_ptr
        self._timed_out_ptr = timed_out_ptr

        # Reusable GPU tensor for per-CQE status codes.
        self._status_buf = status_buf

        # Persistent lookup used by the indexed CSA submit kernel. snvme
        # exposes one IOVA per 64-KiB GPU page; keeping the table on-device
        # avoids rebuilding a staging-IOVA list for every layer.
        self._staging_page_iovas_gpu: Optional[torch.Tensor] = None
        if torch.cuda.is_available():
            with torch.cuda.device(cuda_device):
                self._staging_page_iovas_gpu = torch.tensor(
                    session.staging_iovas,
                    dtype=torch.int64,
                    device=f"cuda:{cuda_device}",
                )

        # LBA cache: file path to LbaRecords.  Seeded from initial_lba_cache
        # (pre-computed before SNVM_DEVICE_BIND while the filesystem is still
        # accessible); additional entries are added lazily via FIEMAP if the
        # filesystem remains accessible (i.e. drive not yet bound).
        self._lba_cache: dict[str, list[LbaRecord]] = dict(initial_lba_cache or {})
        # Sorted-extent index per path for range queries; invalidated by
        # identity check against the _lba_cache entry (register_lba_cache
        # replaces the list object).
        self._extent_index: dict[
            str, tuple[list[LbaRecord], list[LbaRecord], list[int]]
        ] = {}
        # Cache physical I/O templates per logical byte range. CSA requests
        # may select an arbitrary subset of chunks, so caching a whole layer
        # would be both brittle and wasteful. A range template is independent
        # of its staging destination and can be reused by any subset that
        # touches the same object bytes.
        self._lba_cache_versions: dict[str, int] = {path: 0 for path in self._lba_cache}
        self._resolved_range_cache: OrderedDict[
            tuple[str, int, int, int, int],
            Optional[tuple[tuple[int, int, int], ...]],
        ] = OrderedDict()
        self._resolved_range_cache_capacity = max(
            1,
            int(os.environ.get("LMCACHE_TUTTI_RANGE_CACHE_CAPACITY", "131072")),
        )
        self._resolved_range_cache_hits = 0
        self._resolved_range_cache_misses = 0
        self._debug_expected_checksums = debug_expected_checksums or {}

    @property
    def cuda_device(self) -> int:
        """Return the CUDA device index bound to this loader."""
        return self._cuda_device

    @property
    def io_stream(self) -> Optional[torch.cuda.Stream]:
        """Return the CUDA stream that orders Tutti I/O and staging reuse.

        Callers may enqueue staging consumers on this stream so a following
        NVMe batch cannot overwrite the staging rows before those consumers
        finish. ``None`` means that the loader uses the current CUDA stream.
        """
        return self._io_stream

    @staticmethod
    def create(
        device_path: str = "/dev/ssnvme0",
        ctrl_path: str = "/dev/snvm_control",
        pci_bdf: str = "",
        n_slots: int = 16,
        slot_bytes: int = 32 * 1024 * 1024,  # 32 MiB per slot
        nsid: int = 1,
        kernel_ioq_cap: int = 36,
        cuda_device: int = 0,
        initial_lba_cache: Optional[dict[str, list[LbaRecord]]] = None,
        debug_expected_checksums: Optional[dict[str, tuple[int, str]]] = None,
    ) -> "TuttiDirectLoader":
        """Create and initialise a TuttiDirectLoader.

        Args:
            device_path:   Path to /dev/ssnvme<N>.
            ctrl_path:     Path to /dev/snvm_control.
            pci_bdf:       PCI BDF of the target NVMe controller.
            n_slots:       Number of HBM staging slots (concurrent I/Os).
            slot_bytes:    Bytes per slot (must be a multiple of GPU_PAGE_SIZE).
            nsid:          NVMe namespace ID.
            kernel_ioq_cap: Kernel IOQ cap (passed to NVM_SET_KERNEL_IOQ_CAP).
            cuda_device:   CUDA device index to use for staging buffer and ring
                           allocations.  Must be on the same PCIe root complex
                           (NUMA node) as the NVMe controller at pci_bdf,
                           otherwise nvidia_p2p_get_pages will return -EINVAL.

        Returns:
            An initialised TuttiDirectLoader ready for ``load_chunks_to_hbm``.

        Raises:
            RuntimeError: If the CUDA ops extension is not available.
            RuntimeError: If the staging pool is too small for the queue depth.
        """
        if not _HAS_C_OPS:
            raise RuntimeError(
                "lmcache.c_ops.tutti_submit_batch_sgl_read not found; "
                "rebuild lmcache with csrc/tutti_kv_ops.cu"
            )
        if slot_bytes % _GPU_PAGE_SIZE != 0:
            raise ValueError(
                f"slot_bytes ({slot_bytes}) must be a multiple of "
                f"GPU_PAGE_SIZE ({_GPU_PAGE_SIZE})"
            )

        cuda_dev_str = f"cuda:{cuda_device}"
        slot_gpu_pages = slot_bytes // _GPU_PAGE_SIZE
        total_bytes = n_slots * slot_bytes
        profile_start = time.perf_counter()
        session: Optional[SnvmeSession] = None
        staging_raw_ptr = 0
        sq_tail_ptr = 0
        cq_head_ptr = 0
        cq_phase_ptr = 0
        timed_out_ptr = 0

        # Allocate the staging pool via raw cudaMalloc (not PyTorch's caching
        # allocator) so that nvidia_p2p_get_pages_persistent can find the VA
        # via the RM's GPU VA-space scan.  PyTorch 2.2+ on Hopper defaults to
        # expandable_segments (cuMemCreate / cuMemMap), which is invisible to
        # the legacy P2P RM lookup and causes EINVAL.
        alloc_bytes = total_bytes + _GPU_PAGE_SIZE  # headroom for 64 KiB alignment
        cuda_malloc_start = time.perf_counter()
        try:
            staging_raw_ptr = _cuda_malloc_device(alloc_bytes, cuda_device)
            _get_cudart().cudaMemset(
                ctypes.c_void_p(staging_raw_ptr),
                ctypes.c_int(0),
                ctypes.c_size_t(alloc_bytes),
            )
            torch.cuda.synchronize(device=cuda_dev_str)
            cuda_malloc_ms = _elapsed_ms(cuda_malloc_start)
            align_offset = (-staging_raw_ptr) % _GPU_PAGE_SIZE
            with torch.cuda.device(cuda_device):
                # Wrap cudaMalloc pointer as a non-owning PyTorch tensor via CAI.
                _buf_obj = _ExternalCudaBuffer(staging_raw_ptr, alloc_bytes)
                staging_full = torch.as_tensor(_buf_obj)
                if align_offset > 0:
                    staging = staging_full[align_offset : align_offset + total_bytes]
                else:
                    staging = staging_full[:total_bytes]
                if staging.data_ptr() % _GPU_PAGE_SIZE != 0:
                    raise RuntimeError(
                        f"staging ptr 0x{staging.data_ptr():x} not 64 KiB aligned"
                    )
                logger.debug(
                    "Staging buffer: ptr=0x%x device=%s total_bytes=%d "
                    "(cudaMalloc direct)",
                    staging.data_ptr(),
                    cuda_dev_str,
                    total_bytes,
                )

                session_start = time.perf_counter()
                session = SnvmeSession(
                    device_path=device_path,
                    ctrl_path=ctrl_path,
                    pci_bdf=pci_bdf,
                    staging_tensor=staging,
                    nsid=nsid,
                    kernel_ioq_cap=kernel_ioq_cap,
                    cuda_device=cuda_device,
                )
                session_ms = _elapsed_ms(session_start)

                # Allocate managed-memory scalars for SQ/CQ state.
                aux_alloc_start = time.perf_counter()
                sq_tail_ptr = _cuda_malloc_managed(ctypes.sizeof(ctypes.c_uint16))
                cq_head_ptr = _cuda_malloc_managed(ctypes.sizeof(ctypes.c_uint16))
                cq_phase_ptr = _cuda_malloc_managed(ctypes.sizeof(ctypes.c_uint8))
                timed_out_ptr = _cuda_malloc_managed(ctypes.sizeof(ctypes.c_int32))

                # Initialise queue state. NVMe completion phase starts at 1.
                ctypes.c_uint16.from_address(sq_tail_ptr).value = 0
                ctypes.c_uint16.from_address(cq_head_ptr).value = 0
                ctypes.c_uint8.from_address(cq_phase_ptr).value = 1
                ctypes.c_int32.from_address(timed_out_ptr).value = 0

                q_depth = int(session.info.q_depth)
                status_buf = torch.zeros(
                    q_depth,
                    dtype=torch.int32,
                    device=cuda_dev_str,
                )
            aux_alloc_ms = _elapsed_ms(aux_alloc_start)

            if n_slots > q_depth:
                raise RuntimeError(
                    f"n_slots ({n_slots}) > q_depth ({q_depth}); "
                    "reduce n_slots or increase the NVMe queue depth"
                )

            logger.info(
                "TUTTI_PROFILE create cuda_device=%d pci=%s slots=%d slot_mb=%.1f "
                "total_mb=%.1f cuda_malloc_ms=%.3f session_bind_map_ms=%.3f "
                "aux_alloc_ms=%.3f total_ms=%.3f q_depth=%d",
                cuda_device,
                pci_bdf,
                n_slots,
                slot_bytes / 1024**2,
                total_bytes / 1024**2,
                cuda_malloc_ms,
                session_ms,
                aux_alloc_ms,
                _elapsed_ms(profile_start),
                q_depth,
            )

            return TuttiDirectLoader(
                session=session,
                staging_tensor=staging,
                staging_raw_ptr=staging_raw_ptr,
                n_slots=n_slots,
                slot_gpu_pages=slot_gpu_pages,
                sq_tail_ptr=sq_tail_ptr,
                cq_head_ptr=cq_head_ptr,
                cq_phase_ptr=cq_phase_ptr,
                timed_out_ptr=timed_out_ptr,
                status_buf=status_buf,
                cuda_device=cuda_device,
                initial_lba_cache=initial_lba_cache,
                debug_expected_checksums=debug_expected_checksums,
            )
        except Exception:
            if session is not None:
                session.close()
            for ptr in (sq_tail_ptr, cq_head_ptr, cq_phase_ptr, timed_out_ptr):
                _cuda_free(ptr)
            _cuda_free(staging_raw_ptr)
            raise

    # ── internal helpers ────────────────────────────────────────────────────

    def _get_extents(self, file_path: str) -> list[LbaRecord]:
        """Look up or populate the LBA records for file_path."""
        if file_path not in self._lba_cache:
            self._lba_cache[file_path] = FiemapHelper.query_extents(file_path)
            self._lba_cache_versions[file_path] = (
                self._lba_cache_versions.get(file_path, 0) + 1
            )
        return self._lba_cache[file_path]

    def _extents_overlapping(
        self,
        file_path: str,
        start: int,
        end: int,
    ) -> list[LbaRecord]:
        """Return only the extents intersecting ``[start, end)``.

        KV object pools register ONE synthetic path (e.g.
        ``tutti://rank0-full``) whose extent list holds every object's
        extents — tens of thousands of records.  The read loops used to scan
        that whole list per byte-range per key (O(keys x extents) Python
        iterations ~= 13M per batch, measured 1.3 s/batch of pure CPU on the
        ON retrieve path).  A bisect over the sorted extent starts reduces
        each lookup to O(log n + hits).
        """
        extents = self._get_extents(file_path)
        index = self._extent_index.get(file_path)
        if index is None or index[0] is not extents:
            self._build_extent_index(file_path)
            index = self._extent_index.get(file_path)
            if index is None:
                return []
        _, ordered, starts = index
        # First extent whose END could exceed `start`: extents are
        # non-overlapping, but the one preceding the insertion point may
        # span past `start`, so step back one.
        lo = bisect.bisect_right(starts, start) - 1
        if lo < 0:
            lo = 0
        hits: list[LbaRecord] = []
        for record in ordered[lo:]:
            if record.file_offset >= end:
                break
            if record.file_offset + record.n_sectors * _NVME_LBS > start:
                hits.append(record)
        return hits

    def register_lba_cache(self, records_by_path: dict[str, list[LbaRecord]]) -> None:
        """Register known LBA extents that do not require FIEMAP.

        Args:
            records_by_path: Mapping from logical path to raw LBA records.  The
                KV object-store raw path uses synthetic paths such as
                ``tutti://rank0-full`` because no filesystem file exists after
                snvme bind.
        """
        for path, records in records_by_path.items():
            if records:
                self._lba_cache[path] = list(records)
                self._lba_cache_versions[path] = (
                    self._lba_cache_versions.get(path, 0) + 1
                )
                # Build the sorted range-query index NOW, outside any read
                # path: pool paths carry tens of thousands of extents and
                # sorting them inside the first _load_batch (under _io_lock)
                # stalls the whole loader.
                self._build_extent_index(path)

    def get_lba_records(self, file_path: str) -> list[LbaRecord]:
        """Return cached extents for a path without touching the filesystem.

        Args:
            file_path: Original file path used during the pre-bind FIEMAP scan.

        Returns:
            A copy of the cached extent list, or an empty list when the path
            was not scanned before snvme detached the filesystem.
        """
        return list(self._lba_cache.get(file_path, ()))

    def ensure_lba_cache(self, records_by_path: dict[str, list[LbaRecord]]) -> None:
        """Idempotently ensure ``records_by_path`` is the active extent table.

        Stores the caller's list objects by reference so a later call can
        detect "still ours" via identity and skip the O(n log n) index
        rebuild entirely.  Only when another path (e.g. a retrieve that
        registered CSA-filtered extents) has overwritten an entry do we
        restore it and re-sort. Identity short-circuiting avoids
        rebuilding a large pool extent table for repeated staged reads.
        """
        for path, records in records_by_path.items():
            if not records:
                continue
            if self._lba_cache.get(path) is records:
                continue
            self._lba_cache[path] = records
            self._lba_cache_versions[path] = self._lba_cache_versions.get(path, 0) + 1
            self._build_extent_index(path)

    def _build_extent_index(self, file_path: str) -> None:
        """(Re)build the sorted extent index for one path."""
        extents = self._lba_cache.get(file_path)
        if not extents:
            self._extent_index.pop(file_path, None)
            return
        ordered = sorted(extents, key=lambda record: record.file_offset)
        starts = [record.file_offset for record in ordered]
        self._extent_index[file_path] = (extents, ordered, starts)

    def _staging_slice(self, slot_idx: int, nbytes: int) -> torch.Tensor:
        """Return the uint8 view into staging slot slot_idx, trimmed to nbytes."""
        start = slot_idx * self._slot_bytes
        return self._staging[start : start + nbytes]

    def _staging_slice_at(self, offset: int, nbytes: int) -> torch.Tensor:
        """Return the uint8 view into staging at an arbitrary byte offset."""
        return self._staging[offset : offset + nbytes]

    def _slot_iova(self, slot_idx: int) -> int:
        """IOVA of the first GPU page in staging slot slot_idx."""
        page_offset = slot_idx * self._slot_gpu_pages
        return self._session.staging_iovas[page_offset]

    def _staging_iova_at(self, offset: int) -> int:
        """IOVA inside the staging pool at an arbitrary byte offset."""
        page_offset = offset // _GPU_PAGE_SIZE
        return self._session.staging_iovas[page_offset] + offset % _GPU_PAGE_SIZE

    def _slot_iova_with_offset(self, slot_idx: int, offset: int) -> int:
        """IOVA inside a staging slot at an arbitrary byte offset."""
        page_offset = slot_idx * self._slot_gpu_pages + offset // _GPU_PAGE_SIZE
        return self._session.staging_iovas[page_offset] + offset % _GPU_PAGE_SIZE

    def _q_depth(self) -> int:
        return int(self._session.info.q_depth)

    def _staging_capacity_bytes(self) -> int:
        """Total bytes available in the HBM staging pool."""
        return self._n_slots * self._slot_bytes

    def _check_nvme_status(
        self,
        *,
        op_name: str,
        n_ios: int,
        paths: Optional[Sequence[str]] = None,
        path_for_io: Optional[Callable[[int], str]] = None,
        gpu_has_error: Optional[torch.Tensor] = None,
    ) -> None:
        """Raise if the most recent NVMe command batch reported an error."""
        if ctypes.c_int32.from_address(self._timed_out_ptr).value != 0:
            raise RuntimeError(
                f"Tutti NVMe {op_name} poll timed out; "
                "check snvme module and NVMe controller health"
            )

        # Normal completions are overwhelmingly common. A one-element GPU
        # reduction avoids copying and iterating over up to q_depth status
        # words on every batch. Preserve the full status buffer for the rare
        # error path so diagnostics keep the exact command index and code.
        if gpu_has_error is not None and not bool(gpu_has_error.item()):
            return

        status_cpu = self._status_buf[:n_ios].cpu()
        for j in range(n_ios):
            raw = int(status_cpu[j])
            nvme_status = (raw >> 1) & 0x7FFF
            if nvme_status != 0:
                if path_for_io is not None:
                    path = path_for_io(j)
                elif paths is not None and j < len(paths):
                    path = paths[j]
                else:
                    path = "<unknown>"
                raise RuntimeError(
                    f"NVMe {op_name} failed for io {j} (path {path}): "
                    f"raw status 0x{raw:04x} "
                    f"(SC=0x{nvme_status & 0xFF:02x} "
                    f"SCT=0x{(nvme_status >> 8) & 0x7:x})"
                )

    def _enqueue_nvme_status_reduction(
        self,
        n_ios: int,
        io_stream: Optional[torch.cuda.Stream],
    ) -> Optional[torch.Tensor]:
        """Return no device-side CQ status reduction.

        CQ status words are written by the GPU polling kernel, but launching
        an eager PyTorch reduction over that buffer from inside model forward
        is not safe on the production Tutti path.  In particular, the first
        layer-2 demand read can run while vLLM owns other CUDA streams; the
        reduction was the first operation to report ``cudaErrorIllegalAddress``
        even with ``CUDA_LAUNCH_BLOCKING=1``.  The caller already synchronizes
        the I/O stream and :meth:`_check_nvme_status` handles ``None`` by
        copying only the valid status words to the CPU.  Keep that small,
        deterministic check instead of injecting another GPU kernel into the
        model-forward critical section.

        Args:
            n_ios: Number of valid status entries.
            io_stream: Stream on which ``tutti_poll_batch`` was launched.

        Returns:
            Always ``None`` so the caller performs the post-sync CPU check.
        """
        del n_ios, io_stream
        return None

    def _estimate_chunk_ios(
        self,
        meta: DiskCacheMetadata,
        nbytes: int,
        file_offset: int = 0,
    ) -> int:
        """Estimate how many NVMe READ commands are needed for one chunk."""
        resolved = self._resolve_range_ios(meta.path, file_offset, nbytes)
        if resolved is None:
            return self._q_depth()
        return len(resolved)

    def _resolve_range_ios(
        self,
        file_path: str,
        file_offset: int,
        nbytes: int,
    ) -> Optional[tuple[tuple[int, int, int], ...]]:
        """Resolve one logical range into reusable physical I/O segments.

        The returned tuples are ``(slba, byte_length, target_offset)``. The
        target offset is relative to the beginning of this logical range, so
        callers can bind the same cached template to any staging location.

        Args:
            file_path: Logical file or raw-pool path.
            file_offset: Byte offset of the range in ``file_path``.
            nbytes: DMA byte length, already aligned to the NVMe LBA size.

        Returns:
            Physical I/O segments, or ``None`` when registered extents do not
            cover the complete range.
        """
        if nbytes <= 0 or file_offset % _NVME_LBS or nbytes % _NVME_LBS:
            return None
        self._get_extents(file_path)
        version = self._lba_cache_versions.get(file_path, 0)
        max_io = int(self._session.info.max_data_size)
        cache_key = (file_path, version, file_offset, nbytes, max_io)
        cached = self._resolved_range_cache.get(cache_key)
        if cached is not None or cache_key in self._resolved_range_cache:
            self._resolved_range_cache_hits += 1
            self._resolved_range_cache.move_to_end(cache_key)
            return cached
        self._resolved_range_cache_misses += 1

        resolved: list[tuple[int, int, int]] = []
        covered = 0
        range_end = file_offset + nbytes
        for extent in self._extents_overlapping(file_path, file_offset, range_end):
            extent_start = extent.file_offset
            extent_end = extent_start + extent.n_sectors * _NVME_LBS
            read_start = max(file_offset, extent_start)
            read_end = min(range_end, extent_end)
            if read_start >= read_end:
                continue
            read_nbytes = read_end - read_start
            covered += read_nbytes
            extent_skip = read_start - extent_start
            target_skip = read_start - file_offset
            cursor = 0
            while cursor < read_nbytes:
                segment_nbytes = read_nbytes - cursor
                if max_io > 0:
                    segment_nbytes = min(segment_nbytes, max_io)
                segment_nbytes = (segment_nbytes // _NVME_LBS) * _NVME_LBS
                if segment_nbytes <= 0:
                    resolved = []
                    covered = -1
                    break
                resolved.append(
                    (
                        extent.slba + (extent_skip + cursor) // _NVME_LBS,
                        segment_nbytes,
                        target_skip + cursor,
                    )
                )
                cursor += segment_nbytes
            if covered < 0:
                break

        result = tuple(resolved) if covered == nbytes else None
        self._resolved_range_cache[cache_key] = result
        self._resolved_range_cache.move_to_end(cache_key)
        while len(self._resolved_range_cache) > self._resolved_range_cache_capacity:
            self._resolved_range_cache.popitem(last=False)
        return result

    def _debug_verify_direct_read(
        self,
        meta: DiskCacheMetadata,
        gpu_raw: torch.Tensor,
    ) -> None:
        """Compare one direct-read HBM slice with a CPU pread checksum."""
        expected = self._debug_expected_checksums.get(meta.path)
        if expected is None:
            return
        expected_nbytes, expected_sha = expected
        if gpu_raw.numel() != expected_nbytes:
            logger.error(
                "TUTTI_DEBUG_CHECKSUM size_mismatch path=%s expected=%d actual=%d",
                meta.path,
                expected_nbytes,
                gpu_raw.numel(),
            )
            return
        actual = bytes(gpu_raw.cpu().numpy().tobytes())
        actual_sha = hashlib.sha256(actual).hexdigest()
        if actual_sha == expected_sha:
            logger.info(
                "TUTTI_DEBUG_CHECKSUM ok path=%s bytes=%d sha=%s",
                meta.path,
                expected_nbytes,
                actual_sha[:16],
            )
        else:
            logger.error(
                "TUTTI_DEBUG_CHECKSUM mismatch path=%s bytes=%d expected=%s actual=%s",
                meta.path,
                expected_nbytes,
                expected_sha[:16],
                actual_sha[:16],
            )

    # ── public API ──────────────────────────────────────────────────────────

    def load_indexed_chunks_to_hbm(
        self,
        selected_ids: torch.Tensor,
        slba_table: torch.Tensor,
        io_nbytes: int,
        on_batch_loaded: IndexedBatchLoadedCallback,
        io_priority: str = "demand",
        profile_layer_id: Optional[int] = None,
        input_ready_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Load fixed-size, arbitrarily selected chunks with GPU planning.

        ``selected_ids`` indexes the GPU-resident ``slba_table``. The CUDA
        submit kernel resolves the SLBA and packed staging IOVA while writing
        each NVMe SQE, avoiding Python descriptor lists and their H2D copies.
        Selection may be sparse and non-contiguous; only each individual
        chunk must occupy one physical NVMe range.

        Args:
            selected_ids: Contiguous CUDA int64 tensor of logical chunk ids.
            slba_table: Contiguous CUDA int64 lookup from chunk id to SLBA.
            io_nbytes: Fixed bytes per chunk, aligned to 512 bytes.
            on_batch_loaded: Callback invoked after every queue-depth-bounded
                batch. It receives selection offset, GPU id slice, staging
                stride, logical length, and staging tensor. It must consume
                the referenced bytes or enqueue their consumption on
                :attr:`io_stream` before returning.
            io_priority: ``"demand"`` for L1/miss reads or ``"speculative"``
                for early CP L2 reads. Speculative indexed reads are rejected
                before queue acquisition when demand work or a writer is
                already waiting; once admitted, their queue ownership is
                bounded by the indexed request (at most two batches for the
                1,874-block V28 geometry).
            profile_layer_id: Optional transformer layer id included in
                per-layer profiling logs. CUDA phase events are collected
                only when ``LMCACHE_CSA_ATTENTION_KV_TIMING=1``.
            input_ready_event: Optional CUDA event proving that selected ids
                and lookup tables are ready. When omitted, the loader orders
                its I/O stream after the caller's current stream for backward
                compatibility. The staged CSA path records this event directly
                on the I/O stream and therefore never waits for model compute.

        Raises:
            ValueError: If tensor layout or transfer geometry is invalid.
            RuntimeError: If the indexed CUDA op is unavailable or NVMe fails.
        """
        if _c_ops is None or not hasattr(_c_ops, "tutti_submit_indexed_sgl_read"):
            raise RuntimeError(
                "lmcache.c_ops.tutti_submit_indexed_sgl_read not found; "
                "rebuild lmcache with the indexed Tutti CUDA op"
            )
        if io_priority not in {"demand", "speculative"}:
            raise ValueError(f"unsupported Tutti I/O priority: {io_priority}")
        if selected_ids.dtype != torch.int64 or not selected_ids.is_cuda:
            raise ValueError("selected_ids must be a CUDA int64 tensor")
        if slba_table.dtype != torch.int64 or not slba_table.is_cuda:
            raise ValueError("slba_table must be a CUDA int64 tensor")
        if selected_ids.device != slba_table.device:
            raise ValueError("selected_ids and slba_table must share a device")
        if not selected_ids.is_contiguous() or not slba_table.is_contiguous():
            raise ValueError("indexed read tensors must be contiguous")
        if selected_ids.device.index != self._cuda_device:
            raise ValueError(
                f"indexed tensors are on {selected_ids.device}, expected "
                f"cuda:{self._cuda_device}"
            )
        if io_nbytes <= 0 or io_nbytes % _NVME_LBS:
            raise ValueError("io_nbytes must be a positive multiple of 512")
        if io_nbytes > int(self._session.info.max_data_size):
            raise ValueError("io_nbytes exceeds the NVMe maximum transfer size")
        if self._staging_page_iovas_gpu is None:
            raise RuntimeError("indexed staging IOVA table is unavailable")
        if selected_ids.numel() == 0:
            return

        is_demand = io_priority == "demand"
        with self._reader_gate:
            if not is_demand and (self._hp_readers > 0 or self._active_writers > 0):
                raise RuntimeError(
                    "speculative indexed Tutti read was not admitted because "
                    "demand I/O or an active writer owns the queue"
                )
            # Every admitted read, including speculative prefetch, prevents a
            # parked background writer from claiming the shared SQ/CQ ring.
            self._readers_waiting += 1
            if is_demand:
                self._hp_readers += 1
            self._reader_gate.notify_all()
        try:
            profile_enabled = _tutti_profile_enabled()
            lock_wait_start = time.perf_counter() if profile_enabled else 0.0
            with self._io_lock:
                lock_wait_ms = _elapsed_ms(lock_wait_start) if profile_enabled else 0.0
                self._load_indexed_chunks_to_hbm_locked(
                    selected_ids,
                    slba_table,
                    io_nbytes,
                    on_batch_loaded,
                    profile_layer_id=profile_layer_id,
                    lock_wait_ms=lock_wait_ms,
                    input_ready_event=input_ready_event,
                )
        finally:
            with self._reader_gate:
                self._readers_waiting -= 1
                if is_demand:
                    self._hp_readers -= 1
                self._last_read_end = time.perf_counter()
                self._reader_gate.notify_all()

    def load_chunks_to_hbm(
        self,
        keys: list[CacheEngineKey],
        disk_metadatas: list[Optional[DiskCacheMetadata]],
        shapes_per_key: Optional[list[Optional[list[torch.Size]]]] = None,
        file_offsets: Optional[list[int]] = None,
        read_ranges_per_key: Optional[
            list[Optional[Sequence[KVObjectByteRange]]]
        ] = None,
        on_batch_loaded: Optional[
            Callable[[int, list[Optional[MemoryObj]]], None]
        ] = None,
        on_raw_batch_loaded: Optional[RawBatchLoadedCallback] = None,
        lock_per_batch: bool = False,
        before_batch: Optional[Callable[[], None]] = None,
        io_priority: Optional[str] = None,
        max_batch_bytes: Optional[int] = None,
        max_batch_ios: Optional[int] = None,
        should_continue: Optional[Callable[[], bool]] = None,
        deadline_monotonic: Optional[float] = None,
        throttle_speculative: bool = True,
    ) -> list[Optional[MemoryObj]]:
        """Thread-safe wrapper around the GPU-direct NVMe read path.

        Serialises the submit/poll/consume cycle on ``_io_lock``: the
        CSA prefetch path issues reads from proxy executor threads while the
        retrieve path may be mid-load on the main thread, and the loader has
        a single SQ/CQ ring and staging pool.  The lock also protects
        callback consumers, whose staging views become invalid the moment
        another batch overwrites the pool. Reads announce themselves via
        ``_readers_waiting`` so raw stores (background writes) yield the queue;
        see ``_reader_gate``. See ``_load_chunks_to_hbm_locked`` for the full
        contract.

        Args:
            lock_per_batch: When True, ``_io_lock`` is acquired around each
                internal staging batch instead of the whole call, letting a
                concurrent load (e.g. the synchronous retrieve path) interleave
                its batches with this one. Requires ``on_batch_loaded`` or
                ``on_raw_batch_loaded``: each batch's staging bytes are fully
                consumed inside the lock, so releasing it between batches is
                safe. Long bulk prefetch walks use this so they never stall
                the retrieve path for more than one batch.
            on_raw_batch_loaded: Optional zero-wrapper callback receiving the
                global batch start, batch-local completed key indices, staging
                byte offsets, logical byte lengths, and the staging tensor.
                It must consume all referenced bytes before returning. This is
                mutually exclusive with ``on_batch_loaded``.
            before_batch: Optional callback invoked under the batch lock just
                before each batch's extents are resolved.  Bulk prefetch uses
                it to re-register its full-record LBA cache: an interleaved
                retrieve overwrites the loader's extent table with
                CSA-filtered ranges between batches, which the walker's byte
                ranges cannot resolve against.  For whole-call locking, the
                callback runs once immediately after acquiring ``_io_lock``;
                no other loader operation can replace the table until the
                call completes.
            io_priority: ``"demand"`` for foreground retrieval or
                ``"speculative"`` for predicted reads. When omitted,
                whole-call loads are demand and per-batch loads are
                speculative. Speculative batches yield before submission if
                a demand read or an already-active store writer owns the queue.
            max_batch_bytes: Optional byte cap for one NVMe batch.
            max_batch_ios: Optional command cap for one NVMe batch.
            should_continue: Optional cancellation predicate checked between
                batches. Returning False leaves unsubmitted results as None.
            deadline_monotonic: Optional absolute ``time.perf_counter()``
                deadline. Speculative batches are not submitted after it.
            throttle_speculative: Apply the shared speculative token bucket.
                Set this to ``False`` for a bounded lookahead read that must
                finish inside a known compute window.
        """
        if on_batch_loaded is not None and on_raw_batch_loaded is not None:
            raise ValueError(
                "on_batch_loaded and on_raw_batch_loaded are mutually exclusive"
            )
        if io_priority is None:
            io_priority = "speculative" if lock_per_batch else "demand"
        if io_priority not in {"demand", "speculative"}:
            raise ValueError(f"unsupported Tutti I/O priority: {io_priority}")
        is_demand = io_priority == "demand"
        if io_priority == "speculative":
            if max_batch_bytes is None:
                max_batch_bytes = (
                    int(
                        os.environ.get(
                            "LMCACHE_TUTTI_SPECULATIVE_BATCH_MB",
                            "8",
                        )
                    )
                    * 1024**2
                )
            if max_batch_ios is None:
                max_batch_ios = int(
                    os.environ.get(
                        "LMCACHE_TUTTI_SPECULATIVE_BATCH_IOS",
                        "8",
                    )
                )
        if max_batch_bytes is not None and max_batch_bytes <= 0:
            raise ValueError("max_batch_bytes must be positive")
        if max_batch_ios is not None and max_batch_ios <= 0:
            raise ValueError("max_batch_ios must be positive")
        with self._reader_gate:
            # Speculation yields only to an operation that is already active,
            # not to a writer merely parked until the request becomes idle.
            if not is_demand and (self._hp_readers > 0 or self._active_writers > 0):
                return [None] * len(keys)
            self._readers_waiting += 1
            if is_demand:
                # Synchronous retrieve = high-priority reader.  Bulk prefetch
                # batches yield to it (see the per-batch wait below) so the
                # foreground TTFT path always owns the NVMe queue; the bulk
                # walker then runs alone during the compute phase, which is
                # exactly the window it is meant to hide in.
                self._hp_readers += 1
            self._reader_gate.notify_all()
        try:
            if lock_per_batch:
                return self._load_chunks_to_hbm_locked(
                    keys,
                    disk_metadatas,
                    shapes_per_key=shapes_per_key,
                    file_offsets=file_offsets,
                    read_ranges_per_key=read_ranges_per_key,
                    on_batch_loaded=on_batch_loaded,
                    on_raw_batch_loaded=on_raw_batch_loaded,
                    batch_lock=self._io_lock,
                    before_batch=before_batch,
                    io_priority=io_priority,
                    max_batch_bytes=max_batch_bytes,
                    max_batch_ios=max_batch_ios,
                    should_continue=should_continue,
                    deadline_monotonic=deadline_monotonic,
                    throttle_speculative=throttle_speculative,
                )
            with self._io_lock:
                if before_batch is not None:
                    before_batch()
                return self._load_chunks_to_hbm_locked(
                    keys,
                    disk_metadatas,
                    shapes_per_key=shapes_per_key,
                    file_offsets=file_offsets,
                    read_ranges_per_key=read_ranges_per_key,
                    on_batch_loaded=on_batch_loaded,
                    on_raw_batch_loaded=on_raw_batch_loaded,
                    io_priority=io_priority,
                    max_batch_bytes=max_batch_bytes,
                    max_batch_ios=max_batch_ios,
                    should_continue=should_continue,
                    deadline_monotonic=deadline_monotonic,
                    throttle_speculative=throttle_speculative,
                )
        finally:
            with self._reader_gate:
                self._readers_waiting -= 1
                if is_demand:
                    self._hp_readers -= 1
                self._last_read_end = time.perf_counter()
                self._reader_gate.notify_all()

    def _load_indexed_chunks_to_hbm_locked(
        self,
        selected_ids: torch.Tensor,
        slba_table: torch.Tensor,
        io_nbytes: int,
        on_batch_loaded: IndexedBatchLoadedCallback,
        *,
        profile_layer_id: Optional[int] = None,
        lock_wait_ms: float = 0.0,
        input_ready_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Execute a validated indexed read while holding the I/O lock."""
        profile_enabled = _tutti_profile_enabled()
        profile_start = time.perf_counter() if profile_enabled else 0.0
        cuda_phase_profile = profile_enabled
        staging_stride = _align_up(io_nbytes, _GPU_PAGE_SIZE)
        # A single batched SQ tail update cannot represent a completely full
        # circular queue because tail == head denotes empty. Keep one slot
        # free and submit the 1,874-block CSA layer as 1,023 + 851 commands.
        usable_queue_depth = self._q_depth() - 1
        if usable_queue_depth <= 0:
            raise RuntimeError("Tutti indexed reads require queue depth >= 2")
        batch_limit = min(
            usable_queue_depth,
            self._staging_capacity_bytes() // staging_stride,
        )
        if batch_limit <= 0:
            raise RuntimeError("Tutti staging pool cannot hold one indexed chunk")

        q = self._session.queue
        sq_dev_ptr = q.sq_tensor.data_ptr()
        cq_dev_ptr = q.cq_tensor.data_ptr()
        sq_db_ptr = self._session.db_gpu_ptr(q.sq_db_offset)
        cq_db_ptr = self._session.db_gpu_ptr(q.cq_db_offset)
        io_stream = self._io_stream
        io_stream_ptr = io_stream.cuda_stream if io_stream is not None else 0
        if io_stream is not None:
            if input_ready_event is not None:
                io_stream.wait_event(input_ready_event)
            else:
                io_stream.wait_stream(torch.cuda.current_stream())
        total = int(selected_ids.numel())
        batch_start = 0
        batch_count = 0
        submit_cpu_ms = 0.0
        submit_gpu_ms = 0.0
        poll_launch_cpu_ms = 0.0
        nvme_poll_gpu_ms = 0.0
        status_gpu_ms = 0.0
        status_cpu_ms = 0.0
        callback_ms = 0.0
        while batch_start < total:
            batch_end = min(total, batch_start + batch_limit)
            batch_ids = selected_ids[batch_start:batch_end]
            n_ios = int(batch_ids.numel())
            batch_profile_start = time.perf_counter() if profile_enabled else 0.0
            with torch.cuda.device(self._cuda_device):
                phase_events = (
                    tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))
                    if cuda_phase_profile
                    else None
                )
                if phase_events is not None:
                    phase_events[0].record(io_stream)
                submit_start = time.perf_counter() if profile_enabled else 0.0
                _c_ops.tutti_submit_indexed_sgl_read(
                    sq_dev_ptr=sq_dev_ptr,
                    cq_dev_ptr=cq_dev_ptr,
                    sq_db_ptr=sq_db_ptr,
                    cq_db_ptr=cq_db_ptr,
                    sq_tail_ptr=self._sq_tail_ptr,
                    q_depth=self._q_depth(),
                    qid=q.qid,
                    nsid=self._session.nsid,
                    staging_page_iovas=self._staging_page_iovas_gpu,
                    staging_stride=staging_stride,
                    slba_table=slba_table,
                    selected_ids=batch_ids,
                    byte_len=io_nbytes,
                    stream_ptr=io_stream_ptr,
                )
                if profile_enabled:
                    submit_cpu_ms += _elapsed_ms(submit_start)
                if phase_events is not None:
                    phase_events[1].record(io_stream)
                poll_launch_start = time.perf_counter() if profile_enabled else 0.0
                _c_ops.tutti_poll_batch(
                    sq_dev_ptr=sq_dev_ptr,
                    cq_dev_ptr=cq_dev_ptr,
                    sq_db_ptr=sq_db_ptr,
                    cq_db_ptr=cq_db_ptr,
                    cq_head_ptr=self._cq_head_ptr,
                    cq_phase_ptr=self._cq_phase_ptr,
                    q_depth=self._q_depth(),
                    n_ios=n_ios,
                    status_out=self._status_buf,
                    timed_out_ptr=self._timed_out_ptr,
                    max_iters=self.POLL_MAX_ITERS,
                    stream_ptr=io_stream_ptr,
                )
                if profile_enabled:
                    poll_launch_cpu_ms += _elapsed_ms(poll_launch_start)
                if phase_events is not None:
                    phase_events[2].record(io_stream)
                status_has_error = self._enqueue_nvme_status_reduction(
                    n_ios,
                    io_stream,
                )
                if phase_events is not None:
                    phase_events[3].record(io_stream)
                if io_stream is not None:
                    io_stream.synchronize()
                else:
                    torch.cuda.synchronize(device=self._cuda_device)
                if phase_events is not None:
                    submit_gpu_ms += float(
                        phase_events[0].elapsed_time(phase_events[1])
                    )
                    nvme_poll_gpu_ms += float(
                        phase_events[1].elapsed_time(phase_events[2])
                    )
                    status_gpu_ms += float(
                        phase_events[2].elapsed_time(phase_events[3])
                    )

            status_start = time.perf_counter() if profile_enabled else 0.0
            self._check_nvme_status(
                op_name="INDEXED READ",
                n_ios=n_ios,
                path_for_io=lambda _index: "<indexed-slba-table>",
                gpu_has_error=status_has_error,
            )
            if profile_enabled:
                status_cpu_ms += _elapsed_ms(status_start)
            callback_start = time.perf_counter() if profile_enabled else 0.0
            on_batch_loaded(
                batch_start,
                batch_ids,
                staging_stride,
                io_nbytes,
                self._staging,
            )
            if profile_enabled:
                callback_ms += _elapsed_ms(callback_start)
            batch_count += 1
            if profile_enabled:
                logger.info(
                    "TUTTI_PROFILE indexed_batch batch=%d selection_start=%d "
                    "ios=%d bytes_mb=%.3f total_ms=%.3f",
                    batch_count,
                    batch_start,
                    n_ios,
                    n_ios * io_nbytes / 1024**2,
                    _elapsed_ms(batch_profile_start),
                )
            batch_start = batch_end
        if profile_enabled:
            logger.info(
                "TUTTI_PROFILE indexed_total chunks=%d batches=%d total_ms=%.3f",
                total,
                batch_count,
                _elapsed_ms(profile_start),
            )
            logger.info(
                "TUTTI_LAYER_PROFILE device=%d layer=%s chunks=%d batches=%d "
                "bytes_mib=%.3f lock_wait_ms=%.3f submit_cpu_ms=%.3f "
                "submit_gpu_ms=%.3f poll_launch_cpu_ms=%.3f "
                "nvme_poll_gpu_ms=%.3f status_gpu_ms=%.3f status_cpu_ms=%.3f "
                "g2g_callback_ms=%.3f total_ms=%.3f",
                self._cuda_device,
                str(profile_layer_id) if profile_layer_id is not None else "none",
                total,
                batch_count,
                total * io_nbytes / 1024**2,
                lock_wait_ms,
                submit_cpu_ms,
                submit_gpu_ms,
                poll_launch_cpu_ms,
                nvme_poll_gpu_ms,
                status_gpu_ms,
                status_cpu_ms,
                callback_ms,
                _elapsed_ms(profile_start),
            )

    def _load_chunks_to_hbm_locked(
        self,
        keys: list[CacheEngineKey],
        disk_metadatas: list[Optional[DiskCacheMetadata]],
        shapes_per_key: Optional[list[Optional[list[torch.Size]]]] = None,
        file_offsets: Optional[list[int]] = None,
        read_ranges_per_key: Optional[
            list[Optional[Sequence[KVObjectByteRange]]]
        ] = None,
        on_batch_loaded: Optional[
            Callable[[int, list[Optional[MemoryObj]]], None]
        ] = None,
        on_raw_batch_loaded: Optional[RawBatchLoadedCallback] = None,
        batch_lock: Optional[threading.Lock] = None,
        before_batch: Optional[Callable[[], None]] = None,
        io_priority: str = "demand",
        max_batch_bytes: Optional[int] = None,
        max_batch_ios: Optional[int] = None,
        should_continue: Optional[Callable[[], bool]] = None,
        deadline_monotonic: Optional[float] = None,
        throttle_speculative: bool = True,
    ) -> list[Optional[MemoryObj]]:
        """Load KV chunks directly from NVMe into HBM staging.

        For each key whose metadata is available, issues a GPU-direct NVMe
        READ into an HBM staging slot and returns a GPU-resident TensorMemoryObj.
        Keys with missing metadata (None) yield None in the output list.

        Large batches are processed in sub-batches bounded by NVMe queue depth
        and total staging pool capacity.  The staging pool is packed by actual
        chunk size, so small chunks do not waste a full ``slot_bytes`` region.

        Args:
            keys:           Cache engine keys identifying the chunks.
            disk_metadatas: Per-key DiskCacheMetadata from the local disk
                            backend's internal dict.  None means the key is
                            not cached on disk; those entries yield None.
            shapes_per_key: Optional per-key shape overrides for DSV4
                            optimised KV.  When provided, ``shapes_per_key[i]``
                            overrides the shapes used for building the
                            MemoryObjMetadata for ``keys[i]``.  A ``None``
                            entry means "use shapes from disk metadata".
            file_offsets: Optional per-key byte offset inside the metadata
                          path. This is used by the KV object-store path,
                          where many objects live inside one pool file.
            read_ranges_per_key: Optional per-key explicit source ranges.
                This is the object-view path: each range is read from the
                source path and placed at its ``target_offset`` in staging.
            on_batch_loaded: Optional callback invoked after each internal
                staging batch is read. When supplied, returned memory objects
                are staging views and must be consumed before the callback
                returns.
            on_raw_batch_loaded: Optional callback invoked before any
                ``TensorMemoryObj`` is built. It receives completed batch-local
                key indices, byte offsets/lengths, and the shared staging
                tensor. The callback must consume the bytes before returning.

        Returns:
            List parallel to keys. Each element is either a GPU-resident
            TensorMemoryObj (raw_tensor.is_cuda == True) or None. Raw callback
            mode returns an all-None list because ownership never leaves the
            callback.

        Raises:
            RuntimeError: If any NVMe command times out.
        """
        n = len(keys)
        if n == 0:
            return []
        if read_ranges_per_key is not None and len(read_ranges_per_key) != n:
            raise ValueError("read_ranges_per_key and keys must have the same length")
        if on_batch_loaded is not None and on_raw_batch_loaded is not None:
            raise ValueError(
                "on_batch_loaded and on_raw_batch_loaded are mutually exclusive"
            )

        profile_start = time.perf_counter()
        range_cache_hits_start = self._resolved_range_cache_hits
        range_cache_misses_start = self._resolved_range_cache_misses
        results: list[Optional[MemoryObj]] = [None] * n

        q_depth = self._q_depth()
        usable_queue_depth = q_depth - 1
        if usable_queue_depth <= 0:
            raise RuntimeError("Tutti NVMe queue must have at least two slots")
        staging_capacity = self._staging_capacity_bytes()
        batch_start = 0
        n_batches = 0
        n_loaded = 0
        while batch_start < n:
            if should_continue is not None and not should_continue():
                break
            if (
                deadline_monotonic is not None
                and time.perf_counter() >= deadline_monotonic
            ):
                break
            if io_priority == "speculative":
                with self._reader_gate:
                    if self._hp_readers > 0 or self._active_writers > 0:
                        break
            pack_start = time.perf_counter()
            batch_end = batch_start
            batch_ios = 0
            batch_bytes = 0
            batch_io_limit = min(
                usable_queue_depth,
                max_batch_ios or usable_queue_depth,
            )
            batch_byte_limit = min(
                staging_capacity,
                max_batch_bytes or staging_capacity,
            )
            while batch_end < n:
                meta = disk_metadatas[batch_end]
                if meta is None:
                    if batch_end == batch_start or (batch_end - batch_start) < q_depth:
                        batch_end += 1
                        continue
                    break

                key_shapes_override = (
                    shapes_per_key[batch_end] if shapes_per_key is not None else None
                )
                chunk_read_ranges = (
                    read_ranges_per_key[batch_end]
                    if read_ranges_per_key is not None
                    else None
                )
                chunk_logical_nbytes = _logical_read_nbytes(
                    meta,
                    key_shapes_override,
                    file_offset=(
                        file_offsets[batch_end] if file_offsets is not None else 0
                    ),
                    read_ranges=chunk_read_ranges,
                )
                chunk_logical_dma_nbytes = _align_up(
                    chunk_logical_nbytes,
                    _NVME_LBS,
                )
                chunk_bytes = _align_up(chunk_logical_dma_nbytes, _GPU_PAGE_SIZE)
                chunk_file_offset = (
                    file_offsets[batch_end] if file_offsets is not None else 0
                )
                try:
                    chunk_ios = sum(
                        self._estimate_chunk_ios(
                            meta,
                            _align_up(byte_range.length, _NVME_LBS),
                            file_offset=byte_range.offset,
                        )
                        for byte_range in _logical_read_ranges(
                            meta,
                            key_shapes_override,
                            file_offset=chunk_file_offset,
                            read_ranges=chunk_read_ranges,
                        )
                    )
                except (FileNotFoundError, ValueError, OSError):
                    chunk_ios = 1

                if (
                    io_priority == "speculative"
                    and batch_end == batch_start
                    and (chunk_ios > batch_io_limit or chunk_bytes > batch_byte_limit)
                ):
                    logger.info(
                        "TUTTI_PROFILE speculative_admission status=stopped "
                        "key_start=%d chunk_ios=%d io_limit=%d "
                        "chunk_mb=%.3f byte_limit_mb=%.3f",
                        batch_start,
                        chunk_ios,
                        batch_io_limit,
                        chunk_bytes / 1024**2,
                        batch_byte_limit / 1024**2,
                    )
                    return results

                if batch_end > batch_start and (
                    batch_ios + chunk_ios > batch_io_limit
                    or batch_bytes + chunk_bytes > batch_byte_limit
                ):
                    break

                batch_ios += chunk_ios
                batch_bytes += chunk_bytes
                batch_end += 1

                # A single oversized chunk still needs to reach _load_batch so
                # it follows the existing loud failure path.
                if batch_ios > batch_io_limit or batch_bytes > batch_byte_limit:
                    break

            if io_priority == "speculative":
                admitted = False
                while not admitted:
                    if should_continue is not None and not should_continue():
                        return results
                    with self._reader_gate:
                        now = time.perf_counter()
                        if self._hp_readers > 0 or self._active_writers > 0:
                            return results
                        if deadline_monotonic is not None and now >= deadline_monotonic:
                            return results
                        rate = (
                            self._speculative_rate_bytes_per_s
                            if throttle_speculative
                            else 0.0
                        )
                        if rate <= 0:
                            admitted = True
                            continue
                        elapsed = max(0.0, now - self._speculative_last_refill)
                        self._speculative_tokens = min(
                            self._speculative_burst_bytes,
                            self._speculative_tokens + elapsed * rate,
                        )
                        self._speculative_last_refill = now
                        if batch_bytes <= self._speculative_tokens:
                            self._speculative_tokens -= batch_bytes
                            admitted = True
                            continue
                        wait_s = (batch_bytes - self._speculative_tokens) / rate
                        if deadline_monotonic is not None:
                            wait_s = min(wait_s, deadline_monotonic - now)
                        self._reader_gate.wait(timeout=max(0.0, min(wait_s, 0.05)))

            batch_keys = keys[batch_start:batch_end]
            batch_metas = disk_metadatas[batch_start:batch_end]
            batch_shapes = (
                shapes_per_key[batch_start:batch_end]
                if shapes_per_key is not None
                else None
            )

            pack_ms = _elapsed_ms(pack_start)
            batch_profile_start = time.perf_counter()
            raw_batch_loaded = 0

            def _consume_raw_batch(
                completed_indices: list[int],
                completed_offsets: list[int],
                completed_nbytes: list[int],
                staging: torch.Tensor,
                batch_start: int = batch_start,
            ) -> None:
                nonlocal raw_batch_loaded
                raw_batch_loaded = len(completed_indices)
                if on_raw_batch_loaded is not None:
                    on_raw_batch_loaded(
                        batch_start,
                        completed_indices,
                        completed_offsets,
                        completed_nbytes,
                        staging,
                    )

            local_raw_callback = (
                _consume_raw_batch if on_raw_batch_loaded is not None else None
            )
            if batch_lock is not None:
                # Per-batch locking (bulk prefetch): submit/poll/consume of
                # THIS batch happens under the lock, then it is released so
                # a concurrent retrieve can interleave its own batches.
                # NOTE: no priority yield here.  An earlier version waited
                # for _hp_readers == 0 before each batch; with multiple
                # back-to-back requests (each triggering a fresh walk and a
                # fresh retrieve) the walker starved for 10s+ while the
                # forward's miss path waited on the walker's pending marks —
                # a livelock that tripped vLLM's RPC watchdog.  Plain
                # per-batch interleave bounds retrieve delay to one batch
                # (~150 ms) and keeps the walker finishing fast.
                with batch_lock:
                    if before_batch is not None:
                        try:
                            before_batch()
                        except Exception:
                            logger.exception(
                                "Tutti before_batch callback failed; "
                                "continuing with current LBA cache"
                            )
                    batch_results = self._load_batch(
                        batch_keys,
                        batch_metas,
                        shapes_per_key=batch_shapes,
                        file_offsets=(
                            file_offsets[batch_start:batch_end]
                            if file_offsets is not None
                            else None
                        ),
                        read_ranges_per_key=(
                            read_ranges_per_key[batch_start:batch_end]
                            if read_ranges_per_key is not None
                            else None
                        ),
                        clone_results=(
                            on_batch_loaded is None and on_raw_batch_loaded is None
                        ),
                        on_raw_batch_loaded=local_raw_callback,
                    )
                    n_batches += 1
                    batch_loaded = (
                        raw_batch_loaded
                        if on_raw_batch_loaded is not None
                        else sum(1 for res in batch_results if res is not None)
                    )
                    n_loaded += batch_loaded
                    if on_raw_batch_loaded is None:
                        if on_batch_loaded is not None:
                            on_batch_loaded(batch_start, batch_results)
                        else:
                            # clone_results=True: the results are owned copies,
                            # safe to hand out after the lock is released.
                            for i, res in enumerate(batch_results):
                                results[batch_start + i] = res
                batch_start = batch_end
                # A bare threading.Lock has no FIFO ordering. Yield only when
                # another class has actually announced work; unconditional
                # 1 ms sleeps made a 30-batch prediction spend ~30 ms asleep.
                with self._reader_gate:
                    should_yield = self._hp_readers > 0 or self._active_writers > 0
                if should_yield:
                    time.sleep(0.001)
                continue
            batch_results = self._load_batch(
                batch_keys,
                batch_metas,
                shapes_per_key=batch_shapes,
                file_offsets=(
                    file_offsets[batch_start:batch_end]
                    if file_offsets is not None
                    else None
                ),
                read_ranges_per_key=(
                    read_ranges_per_key[batch_start:batch_end]
                    if read_ranges_per_key is not None
                    else None
                ),
                clone_results=(on_batch_loaded is None and on_raw_batch_loaded is None),
                on_raw_batch_loaded=local_raw_callback,
            )
            n_batches += 1
            batch_loaded = (
                raw_batch_loaded
                if on_raw_batch_loaded is not None
                else sum(1 for res in batch_results if res is not None)
            )
            n_loaded += batch_loaded
            logger.info(
                "TUTTI_PROFILE load_batch batch=%d key_start=%d keys=%d "
                "loaded=%d estimated_ios=%d estimated_mb=%.3f pack_ms=%.3f "
                "batch_total_ms=%.3f",
                n_batches,
                batch_start,
                len(batch_keys),
                batch_loaded,
                batch_ios,
                batch_bytes / 1024**2,
                pack_ms,
                _elapsed_ms(batch_profile_start),
            )
            if on_raw_batch_loaded is None:
                if on_batch_loaded is None:
                    for i, res in enumerate(batch_results):
                        results[batch_start + i] = res
                else:
                    on_batch_loaded(batch_start, batch_results)
            batch_start = batch_end

        logger.info(
            "TUTTI_PROFILE load_total keys=%d loaded=%d batches=%d "
            "range_cache_hits=%d range_cache_misses=%d total_ms=%.3f",
            n,
            n_loaded,
            n_batches,
            self._resolved_range_cache_hits - range_cache_hits_start,
            self._resolved_range_cache_misses - range_cache_misses_start,
            _elapsed_ms(profile_start),
        )
        return results

    def store_bytes_to_raw_lbas(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        base_slba: int,
        logical_file_offset: int,
        logical_nbytes: Optional[int] = None,
    ) -> list[LbaRecord]:
        """Store one KV object into raw NVMe LBAs through Tutti.

        This is the cold-store counterpart of ``load_chunks_to_hbm`` for the
        KV object-store path.  The current MVP copies the CPU payload into the
        Tutti HBM staging pool, then submits NVMe WRITE commands from GPU code.
        The returned extents can be registered in ``_lba_cache`` and reused by
        the normal direct-read path without FIEMAP.

        Args:
            payload: Source bytes to persist.
            base_slba: Starting logical block address for the object.
            logical_file_offset: Logical byte offset represented by the first
                returned extent.  For dense object pools this is the object's
                pool offset.
            logical_nbytes: Optional logical object reservation size.  When
                larger than ``len(payload)``, the tail is zero padded before
                writing so reads can cover the full reserved extent.

        Returns:
            A list containing the raw extent for this object.

        Raises:
            RuntimeError: If the extension does not expose the WRITE op, the
                transfer exceeds staging capacity, or NVMe reports an error.
            ValueError: If sizing or LBA arguments are invalid.
        """
        if base_slba < 0:
            raise ValueError("base_slba must be non-negative")
        if logical_file_offset < 0:
            raise ValueError("logical_file_offset must be non-negative")
        payload_nbytes = len(memoryview(payload).cast("B"))
        if payload_nbytes <= 0:
            raise ValueError("payload must be non-empty")
        if logical_nbytes is None:
            logical_nbytes = payload_nbytes
        dma_nbytes = _align_up(logical_nbytes, _NVME_LBS)
        return self.store_bytes_to_raw_extents(
            payload,
            raw_extents=[
                LbaRecord(
                    slba=base_slba,
                    n_sectors=dma_nbytes // _NVME_LBS,
                    file_offset=logical_file_offset,
                )
            ],
            base_file_offset=logical_file_offset,
            logical_nbytes=logical_nbytes,
        )

    def store_bytes_to_raw_extents(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        raw_extents: list[LbaRecord],
        base_file_offset: int,
        logical_nbytes: Optional[int] = None,
    ) -> list[LbaRecord]:
        """Store one KV object through Tutti into known raw LBA extents.

        Args:
            payload: Source bytes to persist.
            raw_extents: Physical destination extents covering the object's
                logical byte range.
            base_file_offset: Logical byte offset corresponding to
                ``payload[0]``.
            logical_nbytes: Optional logical reservation size.  Bytes after the
                payload are zero padded before WRITE.

        Returns:
            The normalized raw extents that cover this object.

        Raises:
            RuntimeError: If WRITE support is unavailable, extents do not cover
                the object, or NVMe reports an error.
            ValueError: If arguments are invalid.
        """
        if not _HAS_WRITE_C_OPS:
            raise RuntimeError(
                "lmcache.c_ops.tutti_submit_batch_sgl_write not found; "
                "rebuild lmcache with csrc/tutti_kv_ops.cu"
            )
        if base_file_offset < 0:
            raise ValueError("base_file_offset must be non-negative")
        if not raw_extents:
            raise ValueError("raw_extents must be non-empty")

        payload_view = memoryview(payload).cast("B")
        payload_nbytes = len(payload_view)
        if payload_nbytes <= 0:
            raise ValueError("payload must be non-empty")
        if logical_nbytes is None:
            logical_nbytes = payload_nbytes
        if logical_nbytes < payload_nbytes:
            raise ValueError("logical_nbytes cannot be smaller than payload")

        dma_nbytes = _align_up(logical_nbytes, _NVME_LBS)
        aligned_nbytes = _align_up(dma_nbytes, _GPU_PAGE_SIZE)
        if aligned_nbytes > self._staging_capacity_bytes():
            raise RuntimeError(
                f"Tutti raw store needs {aligned_nbytes} bytes of staging, "
                f"but only {self._staging_capacity_bytes()} bytes are available"
            )

        profile_start = time.perf_counter()
        _dev = f"cuda:{self._cuda_device}"
        # Same single SQ/CQ ring and staging pool as the read path: a raw
        # store racing a concurrent load corrupts both. Serialise on _io_lock
        # from the staging copy through the last poll/status batch.
        # Slack-only scheduling (Tutti design): writes wait for an idle
        # window -- no reader waiting AND no read finished within
        # _write_slack_s.  During an active request, prefetch reads arrive
        # every few ms, so the queue never looks idle and writes park until
        # the inter-request gap, where they drain at full bandwidth.
        # _write_max_delay_s may bypass only the idle-slack requirement; an
        # announced reader always keeps priority over overdue writes.
        park_start = time.perf_counter()
        with self._reader_gate:
            self._writers_waiting += 1
            self._reader_gate.notify_all()
        writer_active = False
        try:
            while True:
                with self._reader_gate:
                    now = time.perf_counter()
                    if _raw_write_window_ready(
                        readers_waiting=self._readers_waiting,
                        idle_for_s=now - self._last_read_end,
                        waited_s=now - park_start,
                        write_slack_s=self._write_slack_s,
                        write_max_delay_s=self._write_max_delay_s,
                    ):
                        break
                    self._reader_gate.wait(timeout=self._write_slack_s)
            with self._reader_gate:
                self._active_writers += 1
                writer_active = True
                self._reader_gate.notify_all()
            with self._io_lock:
                return self._store_bytes_to_raw_extents_locked(
                    payload_view=payload_view,
                    payload_nbytes=payload_nbytes,
                    dma_nbytes=dma_nbytes,
                    aligned_nbytes=aligned_nbytes,
                    raw_extents=raw_extents,
                    base_file_offset=base_file_offset,
                    profile_start=profile_start,
                    dev=_dev,
                )
        finally:
            with self._reader_gate:
                if writer_active:
                    self._active_writers -= 1
                self._writers_waiting -= 1
                self._reader_gate.notify_all()

    def _store_bytes_to_raw_extents_locked(
        self,
        *,
        payload_view: memoryview,
        payload_nbytes: int,
        dma_nbytes: int,
        aligned_nbytes: int,
        raw_extents: list[LbaRecord],
        base_file_offset: int,
        profile_start: float,
        dev: str,
    ) -> list[LbaRecord]:
        """Body of :meth:`store_bytes_to_raw_extents`; caller holds _io_lock."""
        _dev = dev
        with torch.cuda.device(self._cuda_device):
            copy_start = time.perf_counter()
            staging = self._staging_slice_at(0, aligned_nbytes)
            staging.zero_()
            cpu_tensor = torch.frombuffer(payload_view, dtype=torch.uint8)
            staging[:payload_nbytes].copy_(cpu_tensor, non_blocking=False)
            torch.cuda.synchronize(device=self._cuda_device)
            h2d_ms = _elapsed_ms(copy_start)

        max_io = self._session.info.max_data_size
        q_depth = self._q_depth()
        usable_queue_depth = q_depth - 1
        if usable_queue_depth <= 0:
            raise RuntimeError("Tutti NVMe queue must have at least two slots")
        object_end = base_file_offset + dma_nbytes
        normalized_extents: list[LbaRecord] = []
        io_specs: list[tuple[int, int, int]] = []
        covered_nbytes = 0
        for extent in sorted(raw_extents, key=lambda item: item.file_offset):
            extent_start = extent.file_offset
            extent_end = extent.file_offset + extent.n_sectors * _NVME_LBS
            write_start = max(base_file_offset, extent_start)
            write_end = min(object_end, extent_end)
            if write_start >= write_end:
                continue
            write_nbytes = write_end - write_start
            if write_nbytes % _NVME_LBS != 0:
                raise ValueError("raw extent write size must be 512-byte aligned")
            object_skip = write_start - base_file_offset
            extent_skip = write_start - extent_start
            normalized_extents.append(
                LbaRecord(
                    slba=extent.slba + extent_skip // _NVME_LBS,
                    n_sectors=write_nbytes // _NVME_LBS,
                    file_offset=write_start,
                )
            )
            cursor = 0
            while cursor < write_nbytes:
                io_nbytes = write_nbytes - cursor
                if max_io > 0:
                    io_nbytes = min(io_nbytes, max_io)
                io_nbytes = (io_nbytes // _NVME_LBS) * _NVME_LBS
                if io_nbytes <= 0:
                    raise ValueError("Tutti raw store tail is smaller than one LBA")
                io_specs.append(
                    (
                        object_skip + cursor,
                        extent.slba + (extent_skip + cursor) // _NVME_LBS,
                        io_nbytes,
                    )
                )
                cursor += io_nbytes
            covered_nbytes += write_nbytes
        if covered_nbytes != dma_nbytes:
            raise RuntimeError(
                f"Tutti raw extents cover {covered_nbytes}/{dma_nbytes} bytes"
            )

        cursor = 0
        n_ios = 0
        submit_ms = 0.0
        poll_sync_ms = 0.0
        status_ms = 0.0
        while cursor < len(io_specs):
            batch_iovas: list[int] = []
            batch_slbas: list[int] = []
            batch_lens: list[int] = []
            batch_paths: list[str] = []
            while cursor < len(io_specs) and len(batch_iovas) < usable_queue_depth:
                staging_offset, slba, io_nbytes = io_specs[cursor]
                batch_iovas.append(self._staging_iova_at(staging_offset))
                batch_slbas.append(slba)
                batch_lens.append(io_nbytes)
                batch_paths.append(f"raw://slba/{slba}")
                cursor += 1

            arg_start = time.perf_counter()
            staging_iovas_t = torch.tensor(
                batch_iovas,
                dtype=torch.int64,
                device=_dev,
            )
            slbas_t = torch.tensor(batch_slbas, dtype=torch.int64, device=_dev)
            byte_lens_t = torch.tensor(batch_lens, dtype=torch.int32, device=_dev)
            arg_ms = _elapsed_ms(arg_start)

            q = self._session.queue
            sq_dev_ptr = q.sq_tensor.data_ptr()
            cq_dev_ptr = q.cq_tensor.data_ptr()
            sq_db_ptr = self._session.db_gpu_ptr(q.sq_db_offset)
            cq_db_ptr = self._session.db_gpu_ptr(q.cq_db_offset)

            # Same dedicated-stream rationale as the read path: keep the
            # spinning poll kernel off the shared default stream so raw
            # stores cannot serialise against forward/NCCL kernels.
            io_stream = self._io_stream
            io_stream_ptr = io_stream.cuda_stream if io_stream is not None else 0
            with torch.cuda.device(self._cuda_device):
                # Order the io_stream after the argument-tensor H2D copies
                # issued on the current stream (same race as the read path).
                if io_stream is not None:
                    io_stream.wait_stream(torch.cuda.current_stream())
                submit_start = time.perf_counter()
                _c_ops.tutti_submit_batch_sgl_write(
                    sq_dev_ptr=sq_dev_ptr,
                    cq_dev_ptr=cq_dev_ptr,
                    sq_db_ptr=sq_db_ptr,
                    cq_db_ptr=cq_db_ptr,
                    sq_tail_ptr=self._sq_tail_ptr,
                    q_depth=q_depth,
                    qid=q.qid,
                    nsid=self._session.nsid,
                    staging_iovas=staging_iovas_t,
                    slbas=slbas_t,
                    byte_lens=byte_lens_t,
                    stream_ptr=io_stream_ptr,
                )
                submit_ms += _elapsed_ms(submit_start)

                poll_start = time.perf_counter()
                _c_ops.tutti_poll_batch(
                    sq_dev_ptr=sq_dev_ptr,
                    cq_dev_ptr=cq_dev_ptr,
                    sq_db_ptr=sq_db_ptr,
                    cq_db_ptr=cq_db_ptr,
                    cq_head_ptr=self._cq_head_ptr,
                    cq_phase_ptr=self._cq_phase_ptr,
                    q_depth=q_depth,
                    n_ios=len(batch_iovas),
                    status_out=self._status_buf,
                    timed_out_ptr=self._timed_out_ptr,
                    max_iters=self.POLL_MAX_ITERS,
                    stream_ptr=io_stream_ptr,
                )
                status_has_error = self._enqueue_nvme_status_reduction(
                    len(batch_iovas),
                    io_stream,
                )
                if io_stream is not None:
                    io_stream.synchronize()
                else:
                    torch.cuda.synchronize(device=self._cuda_device)
                poll_sync_ms += _elapsed_ms(poll_start)

            status_start = time.perf_counter()
            self._check_nvme_status(
                op_name="WRITE",
                n_ios=len(batch_iovas),
                paths=batch_paths,
                gpu_has_error=status_has_error,
            )
            status_ms += _elapsed_ms(status_start)
            n_ios += len(batch_iovas)
            logger.debug(
                "TUTTI_PROFILE store_raw_batch ios=%d bytes_mb=%.3f arg_ms=%.3f",
                len(batch_iovas),
                sum(batch_lens) / 1024**2,
                arg_ms,
            )

        logger.info(
            "TUTTI_PROFILE store_raw bytes=%d dma_bytes=%d extents=%d ios=%d "
            "h2d_ms=%.3f submit_launch_ms=%.3f poll_sync_ms=%.3f "
            "status_ms=%.3f total_ms=%.3f",
            payload_nbytes,
            dma_nbytes,
            len(normalized_extents),
            n_ios,
            h2d_ms,
            submit_ms,
            poll_sync_ms,
            status_ms,
            _elapsed_ms(profile_start),
        )
        return normalized_extents

    def _load_batch(
        self,
        keys: list[CacheEngineKey],
        metas: list[Optional[DiskCacheMetadata]],
        shapes_per_key: Optional[list[Optional[list[torch.Size]]]] = None,
        file_offsets: Optional[list[int]] = None,
        read_ranges_per_key: Optional[
            list[Optional[Sequence[KVObjectByteRange]]]
        ] = None,
        clone_results: bool = True,
        on_raw_batch_loaded: Optional[_LocalRawBatchLoadedCallback] = None,
    ) -> list[Optional[MemoryObj]]:
        """Load one queue/staging-capacity-bounded batch into HBM staging."""

        profile_start = time.perf_counter()
        # Build per-I/O parameters. A single KV file can occupy multiple
        # filesystem extents, so one logical chunk may expand to multiple NVMe
        # READs into different offsets of the same contiguous staging pool.
        completed_indices: list[int] = []
        completed_offsets: list[int] = []
        completed_nbytes: list[int] = []
        io_to_key_index: list[int] = []
        staging_iovas_list: list[int] = []
        slbas_list: list[int] = []
        byte_lens_list: list[int] = []
        next_staging_offset = 0
        extents_ms = 0.0

        for i, (key, meta) in enumerate(zip(keys, metas, strict=True)):
            if meta is None:
                logger.debug("Tutti: no metadata for key %s, skipping", key)
                continue
            try:
                extents_start = time.perf_counter()
                # Populate the extent cache (FIEMAP on first touch); range
                # queries below use the sorted index, not this list.
                self._get_extents(meta.path)
                extents_ms += _elapsed_ms(extents_start)
            except (FileNotFoundError, ValueError, OSError) as exc:
                logger.warning("Tutti FIEMAP failed for %s: %s", meta.path, exc)
                continue

            key_shapes_override = (
                shapes_per_key[i] if shapes_per_key is not None else None
            )
            base_file_offset = file_offsets[i] if file_offsets is not None else 0
            read_ranges = (
                read_ranges_per_key[i] if read_ranges_per_key is not None else None
            )
            logical_read_ranges = _logical_read_ranges(
                meta,
                key_shapes_override,
                file_offset=base_file_offset,
                read_ranges=read_ranges,
            )
            nbytes = sum(byte_range.length for byte_range in logical_read_ranges)
            dma_nbytes = _align_up(nbytes, _NVME_LBS)
            aligned_nbytes = _align_up(dma_nbytes, _GPU_PAGE_SIZE)
            if next_staging_offset + aligned_nbytes > self._staging_capacity_bytes():
                logger.warning(
                    "Tutti batch staging capacity exceeded for %s: need %d bytes, "
                    "remaining %d bytes; skipping",
                    key,
                    aligned_nbytes,
                    self._staging_capacity_bytes() - next_staging_offset,
                )
                continue
            chunk_offset = next_staging_offset
            io_start = len(io_to_key_index)
            file_ios = 0
            skip_file = False
            for byte_range in logical_read_ranges:
                range_dma_nbytes = _align_up(byte_range.length, _NVME_LBS)
                is_tail_range = byte_range.target_offset + byte_range.length == nbytes
                if byte_range.offset % _NVME_LBS != 0:
                    logger.warning(
                        "Chunk %s read offset %d is not 512B aligned; skipping",
                        key,
                        byte_range.offset,
                    )
                    skip_file = True
                    break
                if range_dma_nbytes != byte_range.length and not is_tail_range:
                    logger.warning(
                        "Chunk %s non-tail read length %d is not 512B aligned; "
                        "skipping",
                        key,
                        byte_range.length,
                    )
                    skip_file = True
                    break
                resolved = self._resolve_range_ios(
                    meta.path,
                    byte_range.offset,
                    range_dma_nbytes,
                )
                if resolved is None:
                    logger.warning(
                        "Tutti extents for %s cover range %d/%d bytes; skipping",
                        meta.path,
                        0,
                        range_dma_nbytes,
                    )
                    skip_file = True
                    break
                usable_queue_depth = self._q_depth() - 1
                if len(io_to_key_index) + len(resolved) > usable_queue_depth:
                    logger.warning(
                        "Tutti batch for %s would exceed usable queue depth %d; "
                        "skipping",
                        key,
                        usable_queue_depth,
                    )
                    skip_file = True
                    break
                for lba_slba, chunk_nbytes, relative_target in resolved:
                    io_target_offset = byte_range.target_offset + relative_target
                    staging_iovas_list.append(
                        self._staging_iova_at(chunk_offset + io_target_offset)
                    )
                    slbas_list.append(lba_slba)
                    byte_lens_list.append(chunk_nbytes)
                    io_to_key_index.append(i)
                    file_ios += 1
                    logger.debug(
                        "Tutti extent read: key=%s staging_offset=%d "
                        "slba=%d offset=%d bytes=%d",
                        key,
                        chunk_offset,
                        lba_slba,
                        io_target_offset,
                        chunk_nbytes,
                    )

            total_file_bytes = sum(byte_lens_list[io_start:]) if file_ios else 0
            if skip_file or total_file_bytes != dma_nbytes:
                del staging_iovas_list[io_start:]
                del slbas_list[io_start:]
                del byte_lens_list[io_start:]
                del io_to_key_index[io_start:]
                logger.warning(
                    "Tutti extents for %s cover %d/%d bytes; skipping",
                    meta.path,
                    total_file_bytes,
                    dma_nbytes,
                )
                continue
            completed_indices.append(i)
            completed_offsets.append(chunk_offset)
            completed_nbytes.append(nbytes)
            next_staging_offset += aligned_nbytes

        n_ios = len(io_to_key_index)
        build_ms = _elapsed_ms(profile_start)
        if n_ios == 0:
            if any(meta is not None for meta in metas):
                raise RuntimeError("Tutti direct load found no readable KV extents")
            return [None] * len(keys)

        # Build GPU tensors for kernel arguments (same device as staging).
        # Upload ON the io_stream itself: these pointers are dereferenced by
        # the submit kernel on io_stream, and uploading on the calling
        # thread's current stream + wait_stream() only builds the dependency
        # for THAT thread's stream. With concurrent retrieve and speculative
        # reader threads, the H2D copies and consuming kernel can end up on
        # unrelated streams -> submit kernel read half-written argument
        # tensors -> DMA to garbage IOVAs -> illegal memory access family.
        arg_start = time.perf_counter()
        _dev = f"cuda:{self._cuda_device}"
        _arg_stream = self._io_stream
        with torch.cuda.device(self._cuda_device):
            if _arg_stream is not None:
                with torch.cuda.stream(_arg_stream):
                    staging_iovas_t = torch.tensor(
                        staging_iovas_list,
                        dtype=torch.int64,
                    ).to(_dev, non_blocking=True)
                    slbas_t = torch.tensor(slbas_list, dtype=torch.int64).to(
                        _dev, non_blocking=True
                    )
                    byte_lens_t = torch.tensor(byte_lens_list, dtype=torch.int32).to(
                        _dev, non_blocking=True
                    )
            else:
                staging_iovas_t = torch.tensor(
                    staging_iovas_list,
                    dtype=torch.int64,
                    device=_dev,
                )
                slbas_t = torch.tensor(slbas_list, dtype=torch.int64, device=_dev)
                byte_lens_t = torch.tensor(
                    byte_lens_list, dtype=torch.int32, device=_dev
                )
        arg_ms = _elapsed_ms(arg_start)

        q = self._session.queue
        sq_dev_ptr = q.sq_tensor.data_ptr()
        cq_dev_ptr = q.cq_tensor.data_ptr()
        sq_db_ptr = self._session.db_gpu_ptr(q.sq_db_offset)
        cq_db_ptr = self._session.db_gpu_ptr(q.cq_db_offset)

        # All kernel launches and the sync must run on the device that owns the
        # staging buffer and ring tensors.  Without an explicit device guard,
        # multi-GPU callers would launch on whatever device happens to be active,
        # causing cudaErrorIllegalAddress when the kernel dereferences pointers
        # that live on a different GPU.
        #
        # The submit/poll pair runs on the loader's dedicated stream, NOT the
        # shared default stream: tutti_poll_batch is a spinning kernel, and on
        # the legacy default stream it interleaves with model-forward and NCCL
        # kernels enqueued concurrently by the main thread.  Ranks enqueue the
        # spin and their collectives in different orders, which produces a
        # cross-rank circular wait (all-rank freeze, sample_tokens RPC
        # timeout).  On a private stream the spin coexists with compute and
        # only this host thread blocks on it.
        io_stream = self._io_stream
        io_stream_ptr = io_stream.cuda_stream if io_stream is not None else 0
        with torch.cuda.device(self._cuda_device):
            # The argument tensors above (staging_iovas_t/slbas_t/byte_lens_t)
            # were H2D-copied on the CURRENT stream; the submit/poll kernels
            # below run on the private io_stream and dereference those device
            # pointers.  Without an explicit dependency the io_stream kernel
            # can race ahead of the copies -> CUDA illegal memory access
            # (observed at io_stream.synchronize() on warm repeats, where the
            # launch path is fast enough to expose the window).
            if io_stream is not None:
                io_stream.wait_stream(torch.cuda.current_stream())
            # Submit all reads in one kernel launch.
            submit_start = time.perf_counter()
            _c_ops.tutti_submit_batch_sgl_read(
                sq_dev_ptr=sq_dev_ptr,
                cq_dev_ptr=cq_dev_ptr,
                sq_db_ptr=sq_db_ptr,
                cq_db_ptr=cq_db_ptr,
                sq_tail_ptr=self._sq_tail_ptr,
                q_depth=self._q_depth(),
                qid=q.qid,
                nsid=self._session.nsid,
                staging_iovas=staging_iovas_t,
                slbas=slbas_t,
                byte_lens=byte_lens_t,
                stream_ptr=io_stream_ptr,
            )
            submit_launch_ms = _elapsed_ms(submit_start)

            # Poll completions on the same dedicated stream (submit
            # happens-before poll within the stream without an explicit sync).
            poll_start = time.perf_counter()
            _c_ops.tutti_poll_batch(
                sq_dev_ptr=sq_dev_ptr,
                cq_dev_ptr=cq_dev_ptr,
                sq_db_ptr=sq_db_ptr,
                cq_db_ptr=cq_db_ptr,
                cq_head_ptr=self._cq_head_ptr,
                cq_phase_ptr=self._cq_phase_ptr,
                q_depth=self._q_depth(),
                n_ios=n_ios,
                status_out=self._status_buf,
                timed_out_ptr=self._timed_out_ptr,
                max_iters=self.POLL_MAX_ITERS,
                stream_ptr=io_stream_ptr,
            )
            status_has_error = self._enqueue_nvme_status_reduction(
                n_ios,
                io_stream,
            )

            # Sync the I/O stream before reading back status / building
            # MemoryObjs.  NVMe DMA writes are visible to all streams once the
            # poll kernel has confirmed the CQEs, so a host-side sync of just
            # this stream orders the staging reads that follow.
            if io_stream is not None:
                io_stream.synchronize()
            else:
                torch.cuda.synchronize(device=self._cuda_device)
            poll_sync_ms = _elapsed_ms(poll_start)

        if ctypes.c_int32.from_address(self._timed_out_ptr).value != 0:
            cq_head = ctypes.c_uint16.from_address(self._cq_head_ptr).value
            cq_phase = ctypes.c_uint8.from_address(self._cq_phase_ptr).value
            raw_statuses = self._status_buf[:n_ios].cpu().tolist()
            missing_cq_slots: list[int] = []
            for i, raw_status in enumerate(raw_statuses):
                wrapped = int(cq_head + i >= self._q_depth())
                if (int(raw_status) & 0x1) != (cq_phase ^ wrapped):
                    missing_cq_slots.append(i)
            logger.error(
                "TUTTI_PROFILE poll_timeout keys=%d completed=%d ios=%d "
                "bytes_mb=%.3f build_ms=%.3f extents_ms=%.3f arg_ms=%.3f "
                "submit_launch_ms=%.3f poll_sync_ms=%.3f cq_head=%d "
                "cq_phase=%d missing_cqes=%d first_missing=%s",
                len(keys),
                len(completed_indices),
                n_ios,
                sum(byte_lens_list) / 1024**2,
                build_ms,
                extents_ms,
                arg_ms,
                submit_launch_ms,
                poll_sync_ms,
                cq_head,
                cq_phase,
                len(missing_cq_slots),
                missing_cq_slots[:8],
            )
            raise RuntimeError(
                "Tutti NVMe poll timed out; "
                "check snvme module and NVMe controller health"
            )

        # Check per-CQE status.
        status_start = time.perf_counter()

        def _read_path_for_io(io_index: int) -> str:
            meta = metas[io_to_key_index[io_index]]
            return meta.path if meta is not None else "<unknown>"

        self._check_nvme_status(
            op_name="READ",
            n_ios=n_ios,
            path_for_io=_read_path_for_io,
            gpu_has_error=status_has_error,
        )
        status_ms = _elapsed_ms(status_start)

        if on_raw_batch_loaded is not None:
            expected_completed = sum(meta is not None for meta in metas)
            if len(completed_indices) != expected_completed:
                completed = set(completed_indices)
                missing_index = next(
                    i
                    for i, meta in enumerate(metas)
                    if meta is not None and i not in completed
                )
                missing_meta = metas[missing_index]
                assert missing_meta is not None
                raise RuntimeError(
                    "Tutti direct load incomplete for key index "
                    f"{missing_index}, path {missing_meta.path}"
                )

            # Checksum verification is a debug-only facility. Keep it in raw
            # mode without paying the per-object staging-view cost in normal
            # production runs.
            if self._debug_expected_checksums:
                for chunk_offset, nbytes, i_orig in zip(
                    completed_offsets,
                    completed_nbytes,
                    completed_indices,
                    strict=True,
                ):
                    meta = metas[i_orig]
                    assert meta is not None
                    self._debug_verify_direct_read(
                        meta,
                        self._staging_slice_at(chunk_offset, nbytes),
                    )

            consume_start = time.perf_counter()
            on_raw_batch_loaded(
                completed_indices,
                completed_offsets,
                completed_nbytes,
                self._staging,
            )
            consume_ms = _elapsed_ms(consume_start)
            logger.info(
                "TUTTI_PROFILE batch_detail keys=%d completed=%d ios=%d "
                "bytes_mb=%.3f build_ms=%.3f extents_ms=%.3f arg_ms=%.3f "
                "submit_launch_ms=%.3f poll_sync_ms=%.3f status_ms=%.3f "
                "persist_ms=0.000 wrap_ms=0.000 total_ms=%.3f",
                len(keys),
                len(completed_indices),
                n_ios,
                sum(byte_lens_list) / 1024**2,
                build_ms,
                extents_ms,
                arg_ms,
                submit_launch_ms,
                poll_sync_ms,
                status_ms,
                _elapsed_ms(profile_start),
            )
            logger.info(
                "TUTTI_PROFILE raw_batch_consume completed=%d consume_ms=%.3f",
                len(completed_indices),
                consume_ms,
            )
            return [None] * len(keys)

        # Build GPU-resident MemoryObj for each completed I/O.
        wrap_start = time.perf_counter()
        persist_ms = 0.0
        results: list[Optional[MemoryObj]] = [None] * len(keys)
        for chunk_offset, nbytes, i_orig in zip(
            completed_offsets,
            completed_nbytes,
            completed_indices,
            strict=True,
        ):
            meta = metas[i_orig]
            if meta is None:
                raise RuntimeError(
                    f"Internal error: completed_indices contains {i_orig} but "
                    "metas[i_orig] is None; this should never happen"
                )
            # Apply per-key shape override when provided (DSV4 optimised KV:
            # non-tail chunks carry only prefix groups, so their shapes differ
            # from the canonical multi-group shapes stored in disk metadata).
            key_shapes_override = (
                shapes_per_key[i_orig] if shapes_per_key is not None else None
            )
            gpu_raw = self._staging_slice_at(chunk_offset, nbytes)
            self._debug_verify_direct_read(meta, gpu_raw)
            persist_start = time.perf_counter()
            if clone_results:
                # Clone on the private io_stream, NOT the caller's current
                # (default) stream: prefetch fires run on background threads,
                # and a default-stream clone there deadlocks against the main
                # forward's collectives (fire waits for default stream ->
                # default stream runs all_reduce waiting for other ranks ->
                # other ranks' forwards wait for their own fires).  Captured
                # by py-spy under CUDA_LAUNCH_BLOCKING=1: fire thread in
                # clone(), main thread in symm_mem all_reduce, all ranks.
                if io_stream is not None:
                    with torch.cuda.stream(io_stream):
                        owned_raw = gpu_raw.clone()
                    owned_raw.record_stream(io_stream)
                else:
                    owned_raw = gpu_raw.clone()
            else:
                owned_raw = gpu_raw
            persist_ms += _elapsed_ms(persist_start)

            # Wrap the staging slice as a TensorMemoryObj with GPU raw_data.
            # parent_allocator=None means Python GC manages the tensor lifetime.
            obj_meta = _make_memory_obj_metadata(
                meta,
                shapes_override=key_shapes_override,
            )
            results[i_orig] = TensorMemoryObj(
                metadata=obj_meta,
                raw_data=owned_raw,
                parent_allocator=None,
            )
        wrap_ms = _elapsed_ms(wrap_start)

        for i, (meta, result) in enumerate(zip(metas, results, strict=True)):
            if meta is not None and result is None:
                raise RuntimeError(
                    f"Tutti direct load incomplete for key index {i}, path {meta.path}"
                )

        logger.info(
            "TUTTI_PROFILE batch_detail keys=%d completed=%d ios=%d bytes_mb=%.3f "
            "build_ms=%.3f extents_ms=%.3f arg_ms=%.3f submit_launch_ms=%.3f "
            "poll_sync_ms=%.3f status_ms=%.3f persist_ms=%.3f wrap_ms=%.3f "
            "total_ms=%.3f",
            len(keys),
            len(completed_indices),
            n_ios,
            sum(byte_lens_list) / 1024**2,
            build_ms,
            extents_ms,
            arg_ms,
            submit_launch_ms,
            poll_sync_ms,
            status_ms,
            persist_ms,
            wrap_ms,
            _elapsed_ms(profile_start),
        )
        return results

    def close(self) -> None:
        """Release all GPU and NVMe resources."""
        self._session.close()
        _cuda_free(self._sq_tail_ptr)
        self._sq_tail_ptr = 0
        _cuda_free(self._cq_head_ptr)
        self._cq_head_ptr = 0
        _cuda_free(self._cq_phase_ptr)
        self._cq_phase_ptr = 0
        _cuda_free(self._timed_out_ptr)
        self._timed_out_ptr = 0
        self._status_buf = torch.empty(0, dtype=torch.int32)
        self._staging = torch.empty(0, dtype=torch.uint8)
        _cuda_free(self._staging_raw_ptr)
        self._staging_raw_ptr = 0

    def __enter__(self) -> "TuttiDirectLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
