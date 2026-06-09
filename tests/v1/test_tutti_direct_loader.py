# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TuttiDirectLoader and supporting helpers.

All hardware dependencies (ioctl, CUDA managed memory, c_ops kernels) are
mocked so the tests run on any machine — no snvme module or NVMe device needed.
"""

# Standard
import ctypes
import os
import sys
import tempfile
import types
from typing import Optional
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# ── helpers ──────────────────────────────────────────────────────────────────

# The lmcache.v1.gpu_connector package __init__.py imports LMCacheEngineConfig
# which has a heavy dependency chain (yaml, requests, …).  Pre-register a
# stub module with __path__ pointing at the real directory so Python finds
# submodules (tutti_direct_loader.py) without running __init__.py.
if "lmcache.v1.gpu_connector" not in sys.modules:
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, "..", ".."))
    _gc_dir = os.path.join(_root, "lmcache", "v1", "gpu_connector")

    _gc_stub = types.ModuleType("lmcache.v1.gpu_connector")
    _gc_stub.__path__ = [_gc_dir]       # makes it act as a package
    _gc_stub.__package__ = "lmcache.v1.gpu_connector"
    sys.modules["lmcache.v1.gpu_connector"] = _gc_stub

import lmcache.v1.gpu_connector.tutti_direct_loader as _tdl  # noqa: E402
from lmcache.v1.gpu_connector.tutti_direct_loader import (  # noqa: E402
    FiemapHelper,
    LbaRecord,
    TuttiDirectLoader,
    _cuda_host_get_device_pointer,
    _cuda_host_register,
    _cuda_malloc_managed,
    _ioc,
    _IOW,
    _IOR,
    _IOWR,
    _make_memory_obj_metadata,
)
from lmcache.utils import CacheEngineKey, DiskCacheMetadata  # noqa: E402
from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj  # noqa: E402


# ── ioctl number helpers ─────────────────────────────────────────────────────


class TestIocHelpers:
    """Sanity-check the _ioc / _IOW / _IOR / _IOWR wrappers."""

    def test_ioc_encoding_direction(self) -> None:
        # _IOC_WRITE = 1, direction lives at bits [31:30]
        v = _ioc(1, 0x80, 2, 4)
        assert (v >> 30) == 1

    def test_ioc_encoding_type(self) -> None:
        v = _ioc(0, 0xAB, 0, 0)
        assert ((v >> 8) & 0xFF) == 0xAB

    def test_ioc_encoding_nr(self) -> None:
        v = _ioc(0, 0, 0x0F, 0)
        assert (v & 0xFF) == 0x0F

    def test_ioc_encoding_size(self) -> None:
        v = _ioc(0, 0, 0, 64)
        assert ((v >> 16) & 0x3FFF) == 64

    def test_IOWR_has_both_dirs(self) -> None:
        # _IOWR sets both _IOC_READ (2) and _IOC_WRITE (1) → direction = 3
        v = _IOWR(0x80, 12, ctypes.c_uint32)
        assert (v >> 30) == 3


# ── LbaRecord ────────────────────────────────────────────────────────────────


class TestLbaRecord:
    def test_frozen(self) -> None:
        rec = LbaRecord(slba=1024, n_sectors=256)
        with pytest.raises(Exception):
            rec.slba = 999  # type: ignore[misc]

    def test_fields(self) -> None:
        rec = LbaRecord(slba=512, n_sectors=32)
        assert rec.slba == 512
        assert rec.n_sectors == 32
        assert rec.file_offset == 0


# ── FiemapHelper ─────────────────────────────────────────────────────────────


class TestFiemapHelper:
    """Tests for FiemapHelper with the ioctl call mocked out."""

    def _make_fake_fiemap_buf(
        self,
        extents: list[tuple[int, int, int]],  # (logical, physical, length)
    ) -> bytes:
        """Build the raw bytes that the kernel would write back via FIEMAP ioctl."""
        import ctypes as ct

        # Reproduce the struct layout from the module.
        hdr_size = ctypes.sizeof(_tdl._FiemapHeader)
        ext_size = ctypes.sizeof(_tdl._FiemapExtent)
        n = len(extents)
        buf = (ctypes.c_uint8 * (hdr_size + n * ext_size))()

        hdr = _tdl._FiemapHeader.from_buffer(buf)
        hdr.fm_mapped_extents = n

        for i, (logical, physical, length) in enumerate(extents):
            ext = _tdl._FiemapExtent.from_buffer(buf, hdr_size + i * ext_size)
            ext.fe_logical = logical
            ext.fe_physical = physical
            ext.fe_length = length
            ext.fe_flags = 0

        return bytes(buf)

    def test_query_extents_single(self) -> None:
        # One extent: physical=512*100, length=512*8 → slba=100, n_sectors=8
        fake_bytes = self._make_fake_fiemap_buf([(0, 512 * 100, 512 * 8)])

        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp = f.name

        try:
            with patch("fcntl.ioctl") as mock_ioctl:

                def side_effect(fd, req, buf, mutate):
                    buf[: len(fake_bytes)] = fake_bytes
                    return 0

                mock_ioctl.side_effect = side_effect
                records = FiemapHelper.query_extents(tmp)

            assert len(records) == 1
            assert records[0].slba == 100
            assert records[0].n_sectors == 8
            assert records[0].file_offset == 0
        finally:
            os.unlink(tmp)

    def test_query_extents_two(self) -> None:
        fake_bytes = self._make_fake_fiemap_buf(
            [
                (0, 512 * 10, 512 * 5),
                (512 * 5, 512 * 20, 512 * 3),
            ]
        )
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp = f.name
        try:
            with patch("fcntl.ioctl") as mock_ioctl:

                def side_effect(fd, req, buf, mutate):
                    buf[: len(fake_bytes)] = fake_bytes
                    return 0

                mock_ioctl.side_effect = side_effect
                records = FiemapHelper.query_extents(tmp)

            assert len(records) == 2
            assert records[0].slba == 10
            assert records[0].file_offset == 0
            assert records[1].slba == 20
            assert records[1].file_offset == 512 * 5
        finally:
            os.unlink(tmp)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            FiemapHelper.query_extents("/nonexistent/path/to/file.kv")

    def test_single_contiguous_ok(self) -> None:
        fake_bytes = self._make_fake_fiemap_buf([(0, 512 * 42, 512 * 16)])
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp = f.name
        try:
            with patch("fcntl.ioctl") as mock_ioctl:

                def side_effect(fd, req, buf, mutate):
                    buf[: len(fake_bytes)] = fake_bytes
                    return 0

                mock_ioctl.side_effect = side_effect
                rec = FiemapHelper.single_contiguous(tmp)

            assert rec.slba == 42
        finally:
            os.unlink(tmp)

    def test_single_contiguous_fragmented_raises(self) -> None:
        fake_bytes = self._make_fake_fiemap_buf(
            [(0, 512 * 1, 512 * 4), (512 * 4, 512 * 9, 512 * 4)]
        )
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp = f.name
        try:
            with patch("fcntl.ioctl") as mock_ioctl:

                def side_effect(fd, req, buf, mutate):
                    buf[: len(fake_bytes)] = fake_bytes
                    return 0

                mock_ioctl.side_effect = side_effect
                with pytest.raises(ValueError, match="2 extents"):
                    FiemapHelper.single_contiguous(tmp)
        finally:
            os.unlink(tmp)


# ── _make_memory_obj_metadata ─────────────────────────────────────────────────


class TestMakeMemoryObjMetadata:
    def _disk_meta(
        self,
        shapes: Optional[list] = None,
        dtypes: Optional[list] = None,
        fmt: Optional[MemoryFormat] = None,
    ) -> DiskCacheMetadata:
        s = shapes or [torch.Size([2, 4, 8])]
        d = dtypes or [torch.float16]
        return DiskCacheMetadata(
            path="/tmp/fake.kv",
            size=sum(
                t.numel() * t.element_size()
                for t in [
                    torch.empty(shape, dtype=dtype)
                    for shape, dtype in zip(s, d)
                ]
            ),
            shape=s[0] if len(s) == 1 else None,
            dtype=d[0] if len(d) == 1 else None,
            fmt=fmt or MemoryFormat.KV_2LTD,
            shapes=s,
            dtypes=d,
        )

    def test_returns_metadata(self) -> None:
        meta = _make_memory_obj_metadata(self._disk_meta())
        assert isinstance(meta, MemoryObjMetadata)

    def test_shapes_dtypes_propagated(self) -> None:
        shapes = [torch.Size([2, 4, 8]), torch.Size([2, 4, 8])]
        dtypes = [torch.float16, torch.bfloat16]
        meta = _make_memory_obj_metadata(self._disk_meta(shapes=shapes, dtypes=dtypes))
        assert meta.shapes == shapes
        assert meta.dtypes == dtypes

    def test_fmt_default(self) -> None:
        meta = _make_memory_obj_metadata(self._disk_meta(fmt=None))
        assert meta.fmt == MemoryFormat.KV_2LTD


# ── TuttiDirectLoader (mocked hardware) ──────────────────────────────────────


def _make_mock_session(q_depth: int = 32) -> MagicMock:
    """Return a mock SnvmeSession with the minimal attributes used by _load_batch."""
    session = MagicMock()
    session.nsid = 1

    # staging_iovas: list of fake IOVAs, one per GPU page
    _GPU_PAGE_SIZE = 1 << 16
    n_slots = 16
    slot_gpu_pages = 32 * 1024 * 1024 // _GPU_PAGE_SIZE  # 32 MiB / 64 KiB = 512
    total_pages = n_slots * slot_gpu_pages
    session.staging_iovas = list(range(total_pages))  # fake IOVAs 0..total_pages-1

    # info.q_depth
    info = MagicMock()
    info.q_depth = q_depth
    info.max_data_size = 4 * 1024 * 1024
    session.info = info

    # queue resources
    q = MagicMock()
    q.sq_tensor = torch.zeros(64 * 64, dtype=torch.uint8, device="cpu")  # fake SQ
    q.cq_tensor = torch.zeros(64 * 16, dtype=torch.uint8, device="cpu")  # fake CQ
    q.sq_db_offset = 0
    q.cq_db_offset = 4
    q.qid = 1
    session.queue = q

    def db_gpu_ptr(offset):
        return offset + 0xDEAD_0000  # fake GPU VA

    session.db_gpu_ptr.side_effect = db_gpu_ptr

    return session


def _make_loader(n_slots: int = 4, slot_mb: int = 32, q_depth: Optional[int] = None):
    """Build a (loader, ctrl_arrays) tuple — keep ctrl_arrays alive!

    Returns (TuttiDirectLoader, dict[str, ctypes array]) so callers can read
    back scalars and prevent GC of the arrays used as managed-memory surrogates.
    """
    _GPU_PAGE_SIZE = 1 << 16
    slot_bytes = slot_mb * 1024 * 1024
    slot_gpu_pages = slot_bytes // _GPU_PAGE_SIZE
    total_bytes = n_slots * slot_bytes

    staging = torch.zeros(total_bytes, dtype=torch.uint8)
    session = _make_mock_session(q_depth=q_depth or n_slots)

    # Keep ctypes arrays alive by storing them in a dict returned to the caller.
    sq_tail_arr = (ctypes.c_uint16 * 1)(0)
    cq_head_arr = (ctypes.c_uint16 * 1)(0)
    cq_phase_arr = (ctypes.c_uint8 * 1)(1)
    timed_out_arr = (ctypes.c_int32 * 1)(0)
    ctrl = {
        "sq_tail": sq_tail_arr,
        "cq_head": cq_head_arr,
        "cq_phase": cq_phase_arr,
        "timed_out": timed_out_arr,
    }

    loader = TuttiDirectLoader(
        session=session,
        staging_tensor=staging,
        n_slots=n_slots,
        slot_gpu_pages=slot_gpu_pages,
        sq_tail_ptr=ctypes.addressof(sq_tail_arr),
        cq_head_ptr=ctypes.addressof(cq_head_arr),
        cq_phase_ptr=ctypes.addressof(cq_phase_arr),
        timed_out_ptr=ctypes.addressof(timed_out_arr),
        status_buf=torch.zeros(q_depth or n_slots, dtype=torch.int32),
    )
    return loader, ctrl


def _disk_meta_for(size_bytes: int, path: str = "/tmp/fake.kv") -> DiskCacheMetadata:
    return DiskCacheMetadata(
        path=path,
        size=size_bytes,
        shape=torch.Size([1, size_bytes]),
        dtype=torch.uint8,
        fmt=MemoryFormat.KV_2LTD,
        shapes=[torch.Size([1, size_bytes])],
        dtypes=[torch.uint8],
    )


def _fake_key(i: int = 0) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test",
        world_size=1,
        worker_id=0,
        chunk_hash=i,
        dtype=torch.float16,
    )


class TestTuttiDirectLoaderLoadBatch:
    """Test _load_batch logic via heavy mocking of c_ops and FIEMAP."""

    def _run_load(
        self,
        loader: TuttiDirectLoader,
        keys: list[CacheEngineKey],
        metas: list[Optional[DiskCacheMetadata]],
        fiemap_lba: int = 0,
        nvme_status: int = 0,  # 0 = success, non-zero = error after phase strip
    ) -> list:
        """
        Run _load_batch with mocked FIEMAP and c_ops.

        nvme_status: the raw u16 STATUS word in the CQE (phase bit already
        stripped, so 0 = success). This value is left-shifted by 1 into the
        int32 status_out tensor to simulate CQE layout.
        """
        _GPU_PAGE_SIZE = 1 << 16

        meta_by_path = {meta.path: meta for meta in metas if meta is not None}

        def fake_fiemap(path):
            meta = meta_by_path.get(path)
            n_sectors = meta.size // 512 if meta is not None else 1
            return [LbaRecord(slba=fiemap_lba, n_sectors=n_sectors)]

        # Patch FIEMAP so no real ioctl is issued.
        with patch.object(
            _tdl.FiemapHelper, "query_extents", side_effect=fake_fiemap
        ):
            # Patch c_ops submit/poll to be no-ops but set status_buf correctly.
            with patch.object(_tdl, "_c_ops") as mock_c:

                def fake_poll(**kwargs):
                    n = kwargs.get("n_ios", len(keys))
                    loader._status_buf[:n] = nvme_status * 2  # phase=0 → raw=status<<1

                mock_c.tutti_submit_batch_sgl_read = MagicMock()
                mock_c.tutti_poll_batch.side_effect = fake_poll

                # Patch synchronize so tests run without a GPU.
                with patch("torch.cuda.synchronize"):
                    return loader._load_batch(keys, metas)

    def test_all_valid_returns_memory_objs(self) -> None:
        loader, _ctrl = _make_loader(n_slots=4)
        # Use size that's a multiple of 512
        size = 512 * 64  # 32 KiB
        keys = [_fake_key(i) for i in range(2)]
        metas = [_disk_meta_for(size) for _ in range(2)]

        results = self._run_load(loader, keys, metas)

        assert len(results) == 2
        for obj in results:
            assert obj is not None
            assert isinstance(obj, TensorMemoryObj)

    def test_none_meta_yields_none(self) -> None:
        loader, _ctrl = _make_loader(n_slots=4)
        size = 512 * 64
        keys = [_fake_key(0), _fake_key(1)]
        metas = [None, _disk_meta_for(size)]

        results = self._run_load(loader, keys, metas)

        # key 0 had None meta → None; key 1 should have a result
        assert results[0] is None
        assert results[1] is not None

    def test_all_none_metas(self) -> None:
        loader, _ctrl = _make_loader(n_slots=4)
        keys = [_fake_key(i) for i in range(3)]
        metas = [None, None, None]

        with patch.object(_tdl.FiemapHelper, "query_extents"):
            results = loader._load_batch(keys, metas)

        assert all(r is None for r in results)

    def test_chunk_too_large_raises(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, slot_mb=1)  # 1 MiB slots
        # 2 MiB chunk > 1 MiB slot → should be skipped
        size = 2 * 1024 * 1024
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]

        with patch.object(_tdl.FiemapHelper, "query_extents"):
            with pytest.raises(RuntimeError, match="no readable KV extents"):
                loader._load_batch(keys, metas)

    def test_non_sector_aligned_size_raises(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2)
        size = 1000  # not a multiple of 512
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]

        with patch.object(_tdl.FiemapHelper, "query_extents"):
            with pytest.raises(RuntimeError, match="no readable KV extents"):
                loader._load_batch(keys, metas)

    def test_nvme_error_raises(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2)
        size = 512 * 4
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]

        # nvme_status=1 means the NVMe READ failed.
        with pytest.raises(RuntimeError, match="NVMe READ failed"):
            self._run_load(loader, keys, metas, nvme_status=1)

    def test_multi_extent_file_submits_multiple_reads(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, q_depth=8)
        size = 64 * 1024
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]
        extents = [
            LbaRecord(slba=100, n_sectors=(32 * 1024) // 512, file_offset=0),
            LbaRecord(
                slba=200,
                n_sectors=(32 * 1024) // 512,
                file_offset=32 * 1024,
            ),
        ]

        with patch.object(
            _tdl.FiemapHelper, "query_extents", return_value=extents
        ):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs):
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll

                with patch("torch.cuda.synchronize"):
                    results = loader._load_batch(keys, metas)

        assert results[0] is not None
        submit_kwargs = mock_c.tutti_submit_batch_sgl_read.call_args.kwargs
        assert submit_kwargs["byte_lens"].cpu().tolist() == [32 * 1024, 32 * 1024]
        assert submit_kwargs["slbas"].cpu().tolist() == [100, 200]

    def test_timeout_raises(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2)
        size = 512 * 4
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]

        def fake_fiemap(path):
            return [LbaRecord(slba=0, n_sectors=size // 512)]

        with patch.object(
            _tdl.FiemapHelper, "query_extents", side_effect=fake_fiemap
        ):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()
                mock_c.tutti_poll_batch = MagicMock()

                with patch("torch.cuda.synchronize"):
                    # Manually set timed_out_ptr = 1 to simulate timeout.
                    ctypes.c_int32.from_address(loader._timed_out_ptr).value = 1

                    with pytest.raises(RuntimeError, match="timed out"):
                        loader._load_batch(keys, metas)


class TestTuttiDirectLoaderLoadChunksToHbm:
    """Test the chunking logic in load_chunks_to_hbm."""

    def test_empty_input(self) -> None:
        loader, _ctrl = _make_loader()
        assert loader.load_chunks_to_hbm([], []) == []

    def test_batch_chunking(self) -> None:
        """More keys than n_slots → multiple sub-batches processed."""
        n_slots = 2
        loader, _ctrl = _make_loader(n_slots=n_slots)
        size = 512 * 4
        n_keys = 5  # > n_slots, forces two batches
        keys = [_fake_key(i) for i in range(n_keys)]
        metas = [_disk_meta_for(size) for _ in range(n_keys)]

        def fake_fiemap(path):
            return [LbaRecord(slba=0, n_sectors=size // 512)]

        with patch.object(
            _tdl.FiemapHelper, "query_extents", side_effect=fake_fiemap
        ):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs):
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll

                with patch("torch.cuda.synchronize"):
                    results = loader.load_chunks_to_hbm(keys, metas)

        assert len(results) == n_keys
        # All five should succeed
        for r in results:
            assert r is not None

    def test_small_chunks_pack_beyond_slot_count(self) -> None:
        """Small chunks use queue/staging capacity instead of fixed slot count."""
        loader, _ctrl = _make_loader(n_slots=2, slot_mb=1, q_depth=8)
        size = 512 * 4
        n_keys = 4
        keys = [_fake_key(i) for i in range(n_keys)]
        metas = [_disk_meta_for(size, path=f"/tmp/fake_{i}.kv") for i in range(n_keys)]

        def fake_fiemap(path):
            return [LbaRecord(slba=0, n_sectors=size // 512)]

        with patch.object(
            _tdl.FiemapHelper, "query_extents", side_effect=fake_fiemap
        ):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs):
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll

                with patch("torch.cuda.synchronize"):
                    results = loader.load_chunks_to_hbm(keys, metas)

        assert all(r is not None for r in results)
        assert mock_c.tutti_submit_batch_sgl_read.call_count == 1
        submit_kwargs = mock_c.tutti_submit_batch_sgl_read.call_args.kwargs
        assert submit_kwargs["staging_iovas"].numel() == n_keys

    def test_partial_none_in_batch(self) -> None:
        """Mix of valid and None metas within a single batch."""
        loader, _ctrl = _make_loader(n_slots=4)
        size = 512 * 4
        keys = [_fake_key(i) for i in range(3)]
        metas = [_disk_meta_for(size), None, _disk_meta_for(size)]

        def fake_fiemap(path):
            return [LbaRecord(slba=0, n_sectors=size // 512)]

        with patch.object(
            _tdl.FiemapHelper, "query_extents", side_effect=fake_fiemap
        ):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs):
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll

                with patch("torch.cuda.synchronize"):
                    results = loader.load_chunks_to_hbm(keys, metas)

        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None


class TestTuttiDirectLoaderCreate:
    """Test TuttiDirectLoader.create() factory (all hardware mocked)."""

    def _mock_create_patches(self):
        """Return a list of patch objects for all hardware calls in .create()."""
        return [
            patch.object(_tdl, "_HAS_C_OPS", True),
            patch.object(
                _tdl,
                "SnvmeSession",
                return_value=_make_mock_session(q_depth=16),
            ),
            patch.object(_tdl, "_cuda_malloc_managed", side_effect=lambda sz: id(sz)),
            patch.object(_tdl, "_cuda_free"),
            patch("ctypes.c_uint16.from_address", return_value=MagicMock()),
            patch("ctypes.c_uint8.from_address", return_value=MagicMock()),
            patch("ctypes.c_int32.from_address", return_value=MagicMock()),
        ]

    def test_create_no_c_ops_raises(self) -> None:
        with patch.object(_tdl, "_HAS_C_OPS", False):
            with pytest.raises(RuntimeError, match="tutti_submit_batch_sgl_read"):
                TuttiDirectLoader.create(
                    device_path="/dev/ssnvme0",
                    ctrl_path="/dev/snvm_control",
                    pci_bdf="0000:08:00.0",
                )

    def test_create_bad_slot_bytes_raises(self) -> None:
        with patch.object(_tdl, "_HAS_C_OPS", True):
            with pytest.raises(ValueError, match="multiple of GPU_PAGE_SIZE"):
                TuttiDirectLoader.create(
                    device_path="/dev/ssnvme0",
                    ctrl_path="/dev/snvm_control",
                    pci_bdf="0000:08:00.0",
                    slot_bytes=1000,  # not multiple of 64 KiB
                )

    def test_create_n_slots_gt_q_depth_raises(self) -> None:
        mock_session = _make_mock_session(q_depth=8)

        with patch.object(_tdl, "_HAS_C_OPS", True), patch.object(
            _tdl, "SnvmeSession", return_value=mock_session
        ), patch.object(
            _tdl, "_cuda_malloc_managed", return_value=id(b"x")
        ), patch(
            "ctypes.c_uint16.from_address", return_value=MagicMock()
        ), patch(
            "ctypes.c_uint8.from_address", return_value=MagicMock()
        ), patch(
            "ctypes.c_int32.from_address", return_value=MagicMock()
        ):
            with pytest.raises(RuntimeError, match="n_slots"):
                TuttiDirectLoader.create(
                    device_path="/dev/ssnvme0",
                    ctrl_path="/dev/snvm_control",
                    pci_bdf="0000:08:00.0",
                    n_slots=16,  # > q_depth=8
                    slot_bytes=64 * 1024,
                )


# ── TuttiDirectLoader.close ───────────────────────────────────────────────────


class TestTuttiDirectLoaderClose:
    def test_close_frees_managed_ptrs(self) -> None:
        loader, _ctrl = _make_loader()
        freed = []

        original_free = _tdl._cuda_free

        def spy_free(ptr):
            freed.append(ptr)

        with patch.object(_tdl, "_cuda_free", side_effect=spy_free):
            loader.close()

        assert len(freed) == 4  # sq_tail, cq_head, cq_phase, timed_out
        loader._session.close.assert_called_once()

    def test_context_manager(self) -> None:
        loader, _ctrl = _make_loader()
        with patch.object(_tdl, "_cuda_free"):
            with loader:
                pass
        loader._session.close.assert_called_once()


# ── staging slice / slot iova helpers ────────────────────────────────────────


class TestStagingHelpers:
    def test_staging_slice_correct_offset(self) -> None:
        loader, _ctrl = _make_loader(n_slots=4)
        # Slot 0: bytes [0, slot_bytes)
        # Slot 1: bytes [slot_bytes, 2*slot_bytes)
        slot_bytes = loader._slot_bytes
        s0 = loader._staging_slice(0, 512)
        s1 = loader._staging_slice(1, 512)
        data_ptr0 = loader._staging.data_ptr()
        assert s0.data_ptr() == data_ptr0
        assert s1.data_ptr() == data_ptr0 + slot_bytes

    def test_staging_slice_length(self) -> None:
        loader, _ctrl = _make_loader(n_slots=4)
        nbytes = 1024
        sl = loader._staging_slice(0, nbytes)
        assert sl.numel() == nbytes

    def test_slot_iova_offset(self) -> None:
        loader, _ctrl = _make_loader(n_slots=4)
        # staging_iovas is [0, 1, 2, ...]; slot_gpu_pages pages per slot
        slot_gpu_pages = loader._slot_gpu_pages
        assert loader._slot_iova(0) == 0
        assert loader._slot_iova(1) == slot_gpu_pages
