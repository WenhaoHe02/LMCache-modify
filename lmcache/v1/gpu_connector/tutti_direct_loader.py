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

import ctypes
import hashlib
import mmap
import os
import struct as _struct
import sys
import time
import types
from collections.abc import Sequence
from dataclasses import dataclass, field
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

# Default per-CQE spin budget (~5 s at ~10 ns/iter on a GPU).
_DEFAULT_MAX_ITERS: int = 500_000_000

# cudaHostRegisterIoMemory flag (CUDA runtime header).
_CUDA_HOST_REGISTER_IO_MEMORY: int = 0x04


def _align_up(x: int, align: int) -> int:
    """Round x up to the next multiple of align."""
    return ((x + align - 1) // align) * align


def _elapsed_ms(start: float) -> float:
    """Return elapsed wall-clock milliseconds since start."""
    return (time.perf_counter() - start) * 1000.0


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

    slba: int       # starting logical block address (512-byte sectors)
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

        n = FiemapHelper._MAX_EXTENTS
        hdr_size = ctypes.sizeof(_FiemapHeader)
        ext_size = ctypes.sizeof(_FiemapExtent)
        buf = ctypes.create_string_buffer(hdr_size + n * ext_size)

        hdr = _FiemapHeader.from_buffer(buf)
        hdr.fm_start = 0
        hdr.fm_length = 0xFFFF_FFFF_FFFF_FFFF
        hdr.fm_flags = 0
        hdr.fm_extent_count = n

        with open(file_path, "rb") as f:
            fcntl.ioctl(f.fileno(), _FS_IOC_FIEMAP, buf, True)

        records: list[LbaRecord] = []
        for i in range(hdr.fm_mapped_extents):
            ext = _FiemapExtent.from_buffer(buf, hdr_size + i * ext_size)
            records.append(
                LbaRecord(
                    slba=ext.fe_physical // _NVME_LBS,
                    n_sectors=ext.fe_length // _NVME_LBS,
                    file_offset=ext.fe_logical,
                )
            )
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
            raise ValueError(
                f"Invalid PCI BDF: {bdf!r}; expected format DDDD:BB:SS.F"
            )
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
            dev_ptr, align, n_pages, map_kind,
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
            info.max_data_size >> 20, info.block_size,
            info.sgl_supported, info.q_depth,
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
        logger.info("cudaHostRegister BAR0: ptr=0x%x size=%d device=%d", bar0_cpu, info.bar0_size, self._cuda_device)
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

        # Managed-memory control scalars (writable from both CPU and GPU).
        self._sq_tail_ptr = sq_tail_ptr
        self._cq_head_ptr = cq_head_ptr
        self._cq_phase_ptr = cq_phase_ptr
        self._timed_out_ptr = timed_out_ptr

        # Reusable GPU tensor for per-CQE status codes.
        self._status_buf = status_buf

        # LBA cache: file path to LbaRecords.  Seeded from initial_lba_cache
        # (pre-computed before SNVM_DEVICE_BIND while the filesystem is still
        # accessible); additional entries are added lazily via FIEMAP if the
        # filesystem remains accessible (i.e. drive not yet bound).
        self._lba_cache: dict[str, list[LbaRecord]] = dict(initial_lba_cache or {})
        self._debug_expected_checksums = debug_expected_checksums or {}

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
                staging.data_ptr(), cuda_dev_str, total_bytes,
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

            # Initialise: sq_tail=0, cq_head=0, cq_phase=1 (NVMe starts with phase=1).
            ctypes.c_uint16.from_address(sq_tail_ptr).value = 0
            ctypes.c_uint16.from_address(cq_head_ptr).value = 0
            ctypes.c_uint8.from_address(cq_phase_ptr).value = 1
            ctypes.c_int32.from_address(timed_out_ptr).value = 0

            q_depth = int(session.info.q_depth)
            status_buf = torch.zeros(q_depth, dtype=torch.int32, device=cuda_dev_str)
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
        return self._lba_cache[file_path]

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
        paths: list[str],
    ) -> None:
        """Raise if the most recent NVMe command batch reported an error."""
        if ctypes.c_int32.from_address(self._timed_out_ptr).value != 0:
            raise RuntimeError(
                f"Tutti NVMe {op_name} poll timed out; "
                "check snvme module and NVMe controller health"
            )

        status_cpu = self._status_buf[:n_ios].cpu()
        for j in range(n_ios):
            raw = int(status_cpu[j])
            nvme_status = (raw >> 1) & 0x7FFF
            if nvme_status != 0:
                path = paths[j] if j < len(paths) else "<unknown>"
                raise RuntimeError(
                    f"NVMe {op_name} failed for io {j} (path {path}): "
                    f"raw status 0x{raw:04x} "
                    f"(SC=0x{nvme_status & 0xFF:02x} "
                    f"SCT=0x{(nvme_status >> 8) & 0x7:x})"
                )

    def _estimate_chunk_ios(
        self,
        meta: DiskCacheMetadata,
        nbytes: int,
        file_offset: int = 0,
    ) -> int:
        """Estimate how many NVMe READ commands are needed for one chunk."""
        max_io = self._session.info.max_data_size
        n_ios = 0
        covered = 0
        for extent in self._get_extents(meta.path):
            extent_start = extent.file_offset
            extent_end = extent_start + extent.n_sectors * _NVME_LBS
            read_start = max(file_offset, extent_start)
            read_end = min(file_offset + nbytes, extent_end)
            if read_start >= read_end:
                continue
            read_nbytes = read_end - read_start
            covered += read_nbytes
            if max_io > 0:
                n_ios += (read_nbytes + max_io - 1) // max_io
            else:
                n_ios += 1
        if covered != nbytes:
            return self._q_depth() + 1
        return n_ios

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
                "TUTTI_DEBUG_CHECKSUM mismatch path=%s bytes=%d expected=%s "
                "actual=%s",
                meta.path,
                expected_nbytes,
                expected_sha[:16],
                actual_sha[:16],
            )

    # ── public API ──────────────────────────────────────────────────────────

    def load_chunks_to_hbm(
        self,
        keys: list[CacheEngineKey],
        disk_metadatas: list[Optional[DiskCacheMetadata]],
        shapes_per_key: Optional[list[Optional[list[torch.Size]]]] = None,
        file_offsets: Optional[list[int]] = None,
        read_ranges_per_key: Optional[
            list[Optional[Sequence[KVObjectByteRange]]]
        ] = None,
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

        Returns:
            List parallel to keys.  Each element is either a GPU-resident
            TensorMemoryObj (raw_tensor.is_cuda == True) or None.

        Raises:
            RuntimeError: If any NVMe command times out.
        """
        n = len(keys)
        if n == 0:
            return []
        if read_ranges_per_key is not None and len(read_ranges_per_key) != n:
            raise ValueError("read_ranges_per_key and keys must have the same length")

        profile_start = time.perf_counter()
        results: list[Optional[MemoryObj]] = [None] * n

        q_depth = self._q_depth()
        staging_capacity = self._staging_capacity_bytes()
        batch_start = 0
        n_batches = 0
        n_loaded = 0
        while batch_start < n:
            pack_start = time.perf_counter()
            batch_end = batch_start
            batch_ios = 0
            batch_bytes = 0
            while batch_end < n:
                meta = disk_metadatas[batch_end]
                if meta is None:
                    if batch_end == batch_start or (batch_end - batch_start) < q_depth:
                        batch_end += 1
                        continue
                    break

                key_shapes_override = (
                    shapes_per_key[batch_end]
                    if shapes_per_key is not None
                    else None
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
                    batch_end > batch_start
                    and (
                        batch_ios + chunk_ios > q_depth
                        or batch_bytes + chunk_bytes > staging_capacity
                    )
                ):
                    break

                batch_ios += chunk_ios
                batch_bytes += chunk_bytes
                batch_end += 1

                # A single oversized chunk still needs to reach _load_batch so
                # it follows the existing loud failure path.
                if batch_ios > q_depth or batch_bytes > staging_capacity:
                    break

            batch_keys = keys[batch_start:batch_end]
            batch_metas = disk_metadatas[batch_start:batch_end]
            batch_shapes = (
                shapes_per_key[batch_start:batch_end]
                if shapes_per_key is not None
                else None
            )

            pack_ms = _elapsed_ms(pack_start)
            batch_profile_start = time.perf_counter()
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
            )
            n_batches += 1
            batch_loaded = sum(1 for res in batch_results if res is not None)
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
            for i, res in enumerate(batch_results):
                results[batch_start + i] = res
            batch_start = batch_end

        logger.info(
            "TUTTI_PROFILE load_total keys=%d loaded=%d batches=%d "
            "total_ms=%.3f",
            n,
            n_loaded,
            n_batches,
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
            while cursor < len(io_specs) and len(batch_iovas) < q_depth:
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

            with torch.cuda.device(self._cuda_device):
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
                    stream_ptr=0,
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
                    stream_ptr=0,
                )
                torch.cuda.synchronize(device=self._cuda_device)
                poll_sync_ms += _elapsed_ms(poll_start)

            status_start = time.perf_counter()
            self._check_nvme_status(
                op_name="WRITE",
                n_ios=len(batch_iovas),
                paths=batch_paths,
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

        for i, (key, meta) in enumerate(zip(keys, metas)):
            if meta is None:
                logger.debug("Tutti: no metadata for key %s, skipping", key)
                continue
            try:
                extents_start = time.perf_counter()
                extents = self._get_extents(meta.path)
                extents_ms += _elapsed_ms(extents_start)
            except (FileNotFoundError, ValueError, OSError) as exc:
                logger.warning("Tutti FIEMAP failed for %s: %s", meta.path, exc)
                continue

            key_shapes_override = (
                shapes_per_key[i]
                if shapes_per_key is not None
                else None
            )
            base_file_offset = file_offsets[i] if file_offsets is not None else 0
            read_ranges = (
                read_ranges_per_key[i]
                if read_ranges_per_key is not None
                else None
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
            max_io = self._session.info.max_data_size
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
                range_ios_start = len(io_to_key_index)
                range_file_bytes = 0
                range_end_offset = byte_range.offset + range_dma_nbytes
                for extent in extents:
                    extent_start = extent.file_offset
                    extent_end = extent_start + extent.n_sectors * _NVME_LBS
                    read_start = max(byte_range.offset, extent_start)
                    read_end = min(range_end_offset, extent_end)
                    if read_start >= read_end:
                        continue
                    read_nbytes = read_end - read_start
                    extent_skip = read_start - extent_start
                    target_skip = read_start - byte_range.offset
                    if read_start % _GPU_PAGE_SIZE != 0:
                        logger.debug(
                            "Chunk %s extent offset %d is not 64 KiB aligned",
                            key,
                            read_start,
                        )
                    if len(io_to_key_index) >= self._q_depth():
                        logger.warning(
                            "Tutti batch for %s would exceed queue depth %d; "
                            "skipping",
                            key,
                            self._q_depth(),
                        )
                        skip_file = True
                        break
                    if read_nbytes % _NVME_LBS != 0:
                        logger.warning(
                            "Chunk %s extent read size %d is not 512B aligned; "
                            "skipping",
                            key,
                            read_nbytes,
                        )
                        skip_file = True
                        break
                    cursor = 0
                    while cursor < read_nbytes:
                        chunk_nbytes = read_nbytes - cursor
                        if max_io > 0:
                            chunk_nbytes = min(chunk_nbytes, max_io)
                        chunk_nbytes = (chunk_nbytes // _NVME_LBS) * _NVME_LBS
                        if chunk_nbytes == 0:
                            logger.warning(
                                "Chunk %s extent tail is smaller than 512B; "
                                "skipping",
                                key,
                            )
                            skip_file = True
                            break
                        if len(io_to_key_index) >= self._q_depth():
                            logger.warning(
                                "Tutti batch for %s would exceed queue depth %d; "
                                "skipping",
                                key,
                                self._q_depth(),
                            )
                            skip_file = True
                            break
                        io_target_offset = (
                            byte_range.target_offset + target_skip + cursor
                        )
                        lba_slba = (
                            extent.slba + (extent_skip + cursor) // _NVME_LBS
                        )
                        staging_iovas_list.append(
                            self._staging_iova_at(chunk_offset + io_target_offset)
                        )
                        slbas_list.append(lba_slba)
                        byte_lens_list.append(chunk_nbytes)
                        io_to_key_index.append(i)
                        file_ios += 1
                        range_file_bytes += chunk_nbytes
                        logger.debug(
                            "Tutti extent read: key=%s staging_offset=%d "
                            "slba=%d offset=%d bytes=%d",
                            key,
                            chunk_offset,
                            lba_slba,
                            io_target_offset,
                            chunk_nbytes,
                        )
                        cursor += chunk_nbytes
                    if skip_file:
                        break
                if skip_file:
                    break
                if range_file_bytes != range_dma_nbytes:
                    range_ios_added = len(io_to_key_index) - range_ios_start
                    del staging_iovas_list[range_ios_start:]
                    del slbas_list[range_ios_start:]
                    del byte_lens_list[range_ios_start:]
                    del io_to_key_index[range_ios_start:]
                    file_ios -= range_ios_added
                    logger.warning(
                        "Tutti extents for %s cover range %d/%d bytes; skipping",
                        meta.path,
                        range_file_bytes,
                        range_dma_nbytes,
                    )
                    skip_file = True
                    break

            total_file_bytes = sum(
                byte_lens_list[io_start:]
            ) if file_ios else 0
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
        arg_start = time.perf_counter()
        _dev = f"cuda:{self._cuda_device}"
        staging_iovas_t = torch.tensor(
            staging_iovas_list,
            dtype=torch.int64,
            device=_dev,
        )
        slbas_t = torch.tensor(slbas_list, dtype=torch.int64, device=_dev)
        byte_lens_t = torch.tensor(byte_lens_list, dtype=torch.int32, device=_dev)
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
        with torch.cuda.device(self._cuda_device):
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
                stream_ptr=0,
            )
            submit_launch_ms = _elapsed_ms(submit_start)

            # Poll completions synchronously (default CUDA stream, so submit
            # happens-before poll without an explicit sync).
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
                stream_ptr=0,
            )

            # Sync before reading back status / building MemoryObjs.
            torch.cuda.synchronize(device=self._cuda_device)
            poll_sync_ms = _elapsed_ms(poll_start)

        if ctypes.c_int32.from_address(self._timed_out_ptr).value != 0:
            raise RuntimeError(
                "Tutti NVMe poll timed out; "
                "check snvme module and NVMe controller health"
            )

        # Check per-CQE status.
        status_start = time.perf_counter()
        status_cpu = self._status_buf[:n_ios].cpu()
        for j in range(n_ios):
            raw = int(status_cpu[j])
            # NVMe status field: bit 0 = phase, bits[15:1] = SC/SCT/DNR/More.
            nvme_status = (raw >> 1) & 0x7FFF
            if nvme_status != 0:
                i_orig = io_to_key_index[j]
                path = metas[i_orig].path if metas[i_orig] else "<unknown>"
                raise RuntimeError(
                    f"NVMe READ failed for io {j} (key index {i_orig}, "
                    f"path {path}): raw status 0x{raw:04x} "
                    f"(SC=0x{nvme_status & 0xFF:02x} "
                    f"SCT=0x{(nvme_status >> 8) & 0x7:x})"
                )
        status_ms = _elapsed_ms(status_start)

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
                shapes_per_key[i_orig]
                if shapes_per_key is not None
                else None
            )
            gpu_raw = self._staging_slice_at(chunk_offset, nbytes)
            self._debug_verify_direct_read(meta, gpu_raw)
            persist_start = time.perf_counter()
            owned_raw = gpu_raw.clone()
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

        for i, (meta, result) in enumerate(zip(metas, results)):
            if meta is not None and result is None:
                raise RuntimeError(
                    "Tutti direct load incomplete for key index "
                    f"{i}, path {meta.path}"
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
