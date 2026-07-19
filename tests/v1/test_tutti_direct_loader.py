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
import threading
import time
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from typing import Any, Optional
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
    _gc_stub.__path__ = [_gc_dir]  # makes it act as a package
    _gc_stub.__package__ = "lmcache.v1.gpu_connector"
    sys.modules["lmcache.v1.gpu_connector"] = _gc_stub

import lmcache.v1.gpu_connector.tutti_direct_loader as _tdl  # noqa: E402
from lmcache.v1.gpu_connector.tutti_direct_loader import (  # noqa: E402
    FiemapHelper,
    LbaRecord,
    TuttiDirectLoader,
    _ioc,
    _IOWR,
    _make_memory_obj_metadata,
    _raw_write_window_ready,
    _tutti_profile_enabled,
)
from lmcache.v1.kv_object_store import KVObjectByteRange  # noqa: E402
from lmcache.utils import CacheEngineKey, DiskCacheMetadata  # noqa: E402
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)  # noqa: E402


@contextmanager
def _patch_cuda_runtime_for_cpu_tests() -> Iterator[None]:
    """Run mocked Tutti loader paths on CPU-only PyTorch builds."""
    original_tensor = torch.tensor
    original_as_tensor = torch.as_tensor
    original_zeros = torch.zeros

    def tensor_on_cpu(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    def zeros_on_cpu(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    def as_tensor_on_cpu(data, *args, **kwargs):
        if hasattr(data, "__cuda_array_interface__"):
            shape = data.__cuda_array_interface__["shape"]
            return torch.zeros(shape, dtype=torch.uint8)
        return original_as_tensor(data, *args, **kwargs)

    with (
        patch("torch.tensor", side_effect=tensor_on_cpu),
        patch("torch.as_tensor", side_effect=as_tensor_on_cpu),
        patch("torch.zeros", side_effect=zeros_on_cpu),
        patch("torch.cuda.synchronize"),
        patch("torch.cuda.device"),
    ):
        yield


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


class TestRawWriteWindow:
    """Verify that overdue background writes never bypass active readers."""

    def test_idle_writer_can_run_after_slack(self) -> None:
        assert _raw_write_window_ready(0, 0.051, 0.1, 0.05, 2.0)

    def test_overdue_writer_can_bypass_slack_without_reader(self) -> None:
        assert _raw_write_window_ready(0, 0.0, 2.1, 0.05, 2.0)

    def test_overdue_writer_cannot_bypass_waiting_reader(self) -> None:
        assert not _raw_write_window_ready(1, 10.0, 100.0, 0.05, 2.0)


class TestTuttiProfileGate:
    """Verify that hot-path profiling is disabled unless explicitly enabled."""

    def test_explicit_profile_off_overrides_csa_timing(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LMCACHE_TUTTI_PROFILE": "0",
                "LMCACHE_CSA_ATTENTION_KV_TIMING": "1",
            },
        ):
            assert not _tutti_profile_enabled()

    def test_falls_back_to_csa_timing(self) -> None:
        with patch.dict(
            os.environ,
            {"LMCACHE_CSA_ATTENTION_KV_TIMING": "true"},
            clear=True,
        ):
            assert _tutti_profile_enabled()


# ── LbaRecord ────────────────────────────────────────────────────────────────


class TestLbaRecord:
    def test_frozen(self) -> None:
        rec = LbaRecord(slba=1024, n_sectors=256)
        with pytest.raises(FrozenInstanceError):
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
                    for shape, dtype in zip(s, d, strict=True)
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
        staging_raw_ptr=staging.data_ptr(),
        n_slots=n_slots,
        slot_gpu_pages=slot_gpu_pages,
        sq_tail_ptr=ctypes.addressof(sq_tail_arr),
        cq_head_ptr=ctypes.addressof(cq_head_arr),
        cq_phase_ptr=ctypes.addressof(cq_phase_arr),
        timed_out_ptr=ctypes.addressof(timed_out_arr),
        status_buf=torch.zeros(q_depth or n_slots, dtype=torch.int32),
    )
    return loader, ctrl


class _FakeCudaInt64Tensor:
    """Small CUDA-tensor protocol stub for indexed-loader CPU tests."""

    dtype = torch.int64
    is_cuda = True
    device = torch.device("cuda:0")

    def __init__(self, values: list[int]) -> None:
        self.values = values

    def is_contiguous(self) -> bool:
        """Match the tensor contiguity API used by the public loader."""
        return True

    def numel(self) -> int:
        """Return the logical element count."""
        return len(self.values)

    def __getitem__(self, index: slice) -> "_FakeCudaInt64Tensor":
        """Return a sliced protocol stub."""
        return _FakeCudaInt64Tensor(self.values[index])


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

    def test_indexed_read_keeps_one_queue_slot_free_and_uses_ready_event(
        self,
    ) -> None:
        """Indexed batches use q_depth-1 and avoid the caller-stream wait."""
        loader, _ctrl = _make_loader(n_slots=2, slot_mb=1, q_depth=4)
        loader._staging_page_iovas_gpu = torch.ones(64, dtype=torch.int64)
        loader._io_stream = MagicMock()
        loader._io_stream.cuda_stream = 123
        ready_event = MagicMock()
        selected_ids = _FakeCudaInt64Tensor(list(range(7)))
        slba_table = _FakeCudaInt64Tensor(list(range(7)))
        batch_sizes: list[int] = []

        def consume(
            _batch_start: int,
            batch_ids: _FakeCudaInt64Tensor,
            _staging_stride: int,
            _logical_nbytes: int,
            _staging: torch.Tensor,
        ) -> None:
            batch_sizes.append(batch_ids.numel())

        with patch.object(_tdl, "_c_ops") as mock_c:
            mock_c.tutti_submit_indexed_sgl_read = MagicMock()

            def fake_poll(**kwargs: Any) -> None:
                n_ios = int(kwargs["n_ios"])
                loader._status_buf[:n_ios] = 0

            mock_c.tutti_poll_batch.side_effect = fake_poll
            with (
                patch("torch.cuda.device"),
                patch("torch.cuda.current_stream") as current_stream,
            ):
                loader.load_indexed_chunks_to_hbm(
                    selected_ids,  # type: ignore[arg-type]
                    slba_table,  # type: ignore[arg-type]
                    512,
                    consume,  # type: ignore[arg-type]
                    input_ready_event=ready_event,
                )

        assert batch_sizes == [3, 3, 1]
        loader._io_stream.wait_event.assert_called_once_with(ready_event)
        current_stream.assert_not_called()

    def _run_load(
        self,
        loader: TuttiDirectLoader,
        keys: list[CacheEngineKey],
        metas: list[Optional[DiskCacheMetadata]],
        fiemap_lba: int = 0,
        nvme_status: int = 0,  # 0 = success, non-zero = error after phase strip
        **load_kwargs: Any,
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
            n_sectors = (meta.size + 511) // 512 if meta is not None else 1
            return [LbaRecord(slba=fiemap_lba, n_sectors=n_sectors)]

        # Patch FIEMAP so no real ioctl is issued.
        with patch.object(_tdl.FiemapHelper, "query_extents", side_effect=fake_fiemap):
            # Patch c_ops submit/poll to be no-ops but set status_buf correctly.
            with patch.object(_tdl, "_c_ops") as mock_c:

                def fake_poll(**kwargs):
                    n = kwargs.get("n_ios", len(keys))
                    loader._status_buf[:n] = nvme_status * 2  # phase=0 → raw=status<<1

                mock_c.tutti_submit_batch_sgl_read = MagicMock()
                mock_c.tutti_poll_batch.side_effect = fake_poll

                # Patch CUDA helpers so tests run without a GPU.
                with _patch_cuda_runtime_for_cpu_tests():
                    return loader._load_batch(keys, metas, **load_kwargs)

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

    def test_raw_callback_skips_memory_object_wrapping(self) -> None:
        loader, _ctrl = _make_loader(n_slots=4)
        size = 512 * 4
        keys = [_fake_key(0), _fake_key(1)]
        metas = [
            _disk_meta_for(size, path="/tmp/raw_0.kv"),
            _disk_meta_for(size, path="/tmp/raw_1.kv"),
        ]
        callbacks: list[tuple[list[int], list[int], list[int], torch.Tensor]] = []

        def consume_raw(
            indices: list[int],
            offsets: list[int],
            nbytes: list[int],
            staging: torch.Tensor,
        ) -> None:
            callbacks.append((indices, offsets, nbytes, staging))

        with patch.object(_tdl, "_make_memory_obj_metadata") as make_metadata:
            results = self._run_load(
                loader,
                keys,
                metas,
                on_raw_batch_loaded=consume_raw,
            )

        assert results == [None, None]
        assert len(callbacks) == 1
        indices, offsets, nbytes, staging = callbacks[0]
        assert indices == [0, 1]
        assert offsets == [0, 1 << 16]
        assert nbytes == [size, size]
        assert staging.dtype == torch.uint8
        assert staging.numel() >= offsets[-1] + nbytes[-1]
        make_metadata.assert_not_called()

    def test_all_none_metas(self) -> None:
        loader, _ctrl = _make_loader(n_slots=4)
        keys = [_fake_key(i) for i in range(3)]
        metas = [None, None, None]

        with patch.object(_tdl.FiemapHelper, "query_extents"):
            results = loader._load_batch(keys, metas)

        assert all(r is None for r in results)

    def test_chunk_larger_than_slot_can_span_staging_pool(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, slot_mb=1)  # 1 MiB slots
        # 2 MiB chunk > 1 MiB slot but fits the 2 MiB staging pool.
        size = 2 * 1024 * 1024
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]

        results = self._run_load(loader, keys, metas)

        assert results[0] is not None
        assert results[0].get_size() == size

    def test_chunk_larger_than_staging_pool_raises(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, slot_mb=1)  # 2 MiB total
        size = 2 * 1024 * 1024 + 512
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]

        with patch.object(_tdl.FiemapHelper, "query_extents"):
            with pytest.raises(RuntimeError, match="no readable KV extents"):
                loader._load_batch(keys, metas)

    def test_non_sector_aligned_size_reads_padded_tail(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2)
        size = 1000  # not a multiple of 512
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]

        with patch.object(
            _tdl.FiemapHelper,
            "query_extents",
            return_value=[LbaRecord(slba=10, n_sectors=2)],
        ):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs):
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll

                with _patch_cuda_runtime_for_cpu_tests():
                    results = loader._load_batch(keys, metas)

        assert results[0] is not None
        assert results[0].get_size() == size
        assert results[0].raw_data.numel() == size
        submit_kwargs = mock_c.tutti_submit_batch_sgl_read.call_args.kwargs
        assert submit_kwargs["byte_lens"].cpu().tolist() == [1024]

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

        with patch.object(_tdl.FiemapHelper, "query_extents", return_value=extents):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs):
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll

                with _patch_cuda_runtime_for_cpu_tests():
                    results = loader._load_batch(keys, metas)

        assert results[0] is not None
        submit_kwargs = mock_c.tutti_submit_batch_sgl_read.call_args.kwargs
        assert submit_kwargs["byte_lens"].cpu().tolist() == [32 * 1024, 32 * 1024]
        assert submit_kwargs["slbas"].cpu().tolist() == [100, 200]

    def test_layer_major_read_splits_at_controller_max_data_size(self) -> None:
        loader, _ctrl = _make_loader(n_slots=3, slot_mb=32, q_depth=32)
        size = 70_042_624
        max_io = 4 * 1024 * 1024
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]

        with patch.object(
            _tdl.FiemapHelper,
            "query_extents",
            return_value=[LbaRecord(slba=100, n_sectors=size // 512)],
        ):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs: Any) -> None:
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll
                with _patch_cuda_runtime_for_cpu_tests():
                    results = loader._load_batch(keys, metas)

        assert results[0] is not None
        submit_kwargs = mock_c.tutti_submit_batch_sgl_read.call_args.kwargs
        byte_lens = submit_kwargs["byte_lens"].cpu().tolist()
        slbas = submit_kwargs["slbas"].cpu().tolist()
        assert len(byte_lens) == 17
        assert byte_lens == [max_io] * 16 + [size - 16 * max_io]
        assert sum(byte_lens) == size
        assert slbas == [100 + i * (max_io // 512) for i in range(17)]

    def test_sparse_range_resolution_is_cached_and_versioned(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, q_depth=8)
        path = "tutti://range-cache"
        nbytes = 37_376
        first_records = [
            LbaRecord(
                slba=100,
                n_sectors=nbytes // 512,
                file_offset=0,
            )
        ]
        loader.register_lba_cache({path: first_records})

        with patch.object(
            loader,
            "_extents_overlapping",
            wraps=loader._extents_overlapping,
        ) as overlapping:
            first = loader._resolve_range_ios(path, 0, nbytes)
            second = loader._resolve_range_ios(path, 0, nbytes)
            assert first == second == ((100, nbytes, 0),)
            assert overlapping.call_count == 1

            # Re-registering the path represents a changed physical layout.
            # The versioned key must not return the stale SLBA template.
            loader.register_lba_cache(
                {
                    path: [
                        LbaRecord(
                            slba=900,
                            n_sectors=nbytes // 512,
                            file_offset=0,
                        )
                    ]
                }
            )
            refreshed = loader._resolve_range_ios(path, 0, nbytes)
            assert refreshed == ((900, nbytes, 0),)
            assert overlapping.call_count == 2
            assert loader._resolved_range_cache_hits == 1
            assert loader._resolved_range_cache_misses == 2

    def test_status_fast_path_preserves_error_fallback(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, q_depth=8)
        loader._status_buf[0] = 0
        loader._check_nvme_status(
            op_name="READ",
            n_ios=1,
            paths=["ok"],
            gpu_has_error=torch.tensor(False),
        )

        loader._status_buf[0] = 2
        with pytest.raises(RuntimeError, match="NVMe READ failed"):
            loader._check_nvme_status(
                op_name="READ",
                n_ios=1,
                paths=["bad"],
                gpu_has_error=torch.tensor(True),
            )

    def test_timeout_raises(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2)
        size = 512 * 4
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]

        def fake_fiemap(path):
            return [LbaRecord(slba=0, n_sectors=size // 512)]

        with patch.object(_tdl.FiemapHelper, "query_extents", side_effect=fake_fiemap):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()
                mock_c.tutti_poll_batch = MagicMock()

                with _patch_cuda_runtime_for_cpu_tests():
                    # Manually set timed_out_ptr = 1 to simulate timeout.
                    ctypes.c_int32.from_address(loader._timed_out_ptr).value = 1

                    with pytest.raises(RuntimeError, match="timed out"):
                        loader._load_batch(keys, metas)


class TestTuttiDirectLoaderLoadChunksToHbm:
    """Test the chunking logic in load_chunks_to_hbm."""

    def test_empty_input(self) -> None:
        loader, _ctrl = _make_loader()
        assert loader.load_chunks_to_hbm([], []) == []

    def test_callbacks_are_mutually_exclusive(self) -> None:
        loader, _ctrl = _make_loader()

        def consume_raw(*_args: Any) -> None:
            return None

        with pytest.raises(ValueError, match="mutually exclusive"):
            loader.load_chunks_to_hbm(
                [],
                [],
                on_batch_loaded=lambda _start, _results: None,
                on_raw_batch_loaded=consume_raw,
            )

    def test_before_batch_runs_under_whole_call_lock(self) -> None:
        loader, _ctrl = _make_loader()
        callback_lock_states: list[bool] = []

        def before_batch() -> None:
            callback_lock_states.append(loader._io_lock.locked())

        with patch.object(
            loader,
            "_load_chunks_to_hbm_locked",
            return_value=[None],
        ) as load_locked:
            results = loader.load_chunks_to_hbm(
                [_fake_key(0)],
                [_disk_meta_for(512 * 4)],
                before_batch=before_batch,
            )

        assert results == [None]
        assert callback_lock_states == [True]
        load_locked.assert_called_once()

    def test_raw_callback_reports_global_batch_start(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, q_depth=2)
        size = 512 * 4
        keys = [_fake_key(i) for i in range(3)]
        metas = [_disk_meta_for(size, path=f"/tmp/raw_batch_{i}.kv") for i in range(3)]
        callbacks: list[tuple[int, list[int], list[int]]] = []

        def fake_fiemap(_path: str) -> list[LbaRecord]:
            return [LbaRecord(slba=0, n_sectors=size // 512)]

        def consume_raw(
            batch_start: int,
            indices: list[int],
            offsets: list[int],
            _nbytes: list[int],
            _staging: torch.Tensor,
        ) -> None:
            callbacks.append((batch_start, indices, offsets))

        with patch.object(
            _tdl.FiemapHelper,
            "query_extents",
            side_effect=fake_fiemap,
        ):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs: Any) -> None:
                    n_ios = kwargs.get("n_ios", 0)
                    kwargs["status_out"][:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll
                with _patch_cuda_runtime_for_cpu_tests():
                    results = loader.load_chunks_to_hbm(
                        keys,
                        metas,
                        on_raw_batch_loaded=consume_raw,
                    )

        assert results == [None, None, None]
        assert callbacks == [
            (0, [0], [0]),
            (1, [0], [0]),
            (2, [0], [0]),
        ]

    def test_speculative_read_yields_to_demand_reader(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, q_depth=2)
        size = 512 * 4
        demand_started = threading.Event()
        release_demand = threading.Event()
        demand_done: list[list[Optional[MemoryObj]]] = []
        demand_error: list[BaseException] = []

        def fake_fiemap(_path: str) -> list[LbaRecord]:
            return [LbaRecord(slba=0, n_sectors=size // 512)]

        def fake_poll(**kwargs: Any) -> None:
            demand_started.set()
            assert release_demand.wait(timeout=5.0)
            n_ios = kwargs.get("n_ios", 0)
            kwargs["status_out"][:n_ios] = 0

        def run_demand() -> None:
            try:
                demand_done.append(
                    loader.load_chunks_to_hbm(
                        [_fake_key(0)],
                        [_disk_meta_for(size, path="/tmp/demand.kv")],
                        on_raw_batch_loaded=lambda *_args: None,
                        io_priority="demand",
                    )
                )
            except BaseException as exc:
                demand_error.append(exc)

        with patch.object(
            _tdl.FiemapHelper,
            "query_extents",
            side_effect=fake_fiemap,
        ):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()
                mock_c.tutti_poll_batch.side_effect = fake_poll
                with _patch_cuda_runtime_for_cpu_tests():
                    demand_thread = threading.Thread(target=run_demand)
                    demand_thread.start()
                    assert demand_started.wait(timeout=5.0)
                    results = loader.load_chunks_to_hbm(
                        [_fake_key(1)],
                        [_disk_meta_for(size, path="/tmp/speculative.kv")],
                        lock_per_batch=True,
                        on_raw_batch_loaded=lambda *_args: None,
                        io_priority="speculative",
                    )
                    release_demand.set()
                    demand_thread.join(timeout=5.0)

        assert not demand_thread.is_alive()
        assert demand_error == []
        assert demand_done == [[None]]
        assert results == [None]

    def test_speculative_read_yields_to_store_writer(self, monkeypatch) -> None:
        monkeypatch.setenv("LMCACHE_TUTTI_WRITE_SLACK_SEC", "60")
        monkeypatch.setenv("LMCACHE_TUTTI_WRITE_MAX_DELAY_SEC", "60")
        loader, _ctrl = _make_loader(n_slots=2, q_depth=2)
        store_started = threading.Event()
        release_store = threading.Event()
        store_error: list[BaseException] = []

        def fake_store(**_kwargs: Any) -> list[LbaRecord]:
            store_started.set()
            assert release_store.wait(timeout=5.0)
            return [LbaRecord(slba=0, n_sectors=1)]

        def run_store() -> None:
            try:
                loader.store_bytes_to_raw_extents(
                    b"x" * 512,
                    raw_extents=[LbaRecord(slba=0, n_sectors=1)],
                    base_file_offset=0,
                )
            except BaseException as exc:
                store_error.append(exc)

        with patch.object(_tdl, "_HAS_WRITE_C_OPS", True):
            with patch.object(
                loader,
                "_store_bytes_to_raw_extents_locked",
                side_effect=fake_store,
            ):
                store_thread = threading.Thread(target=run_store)
                store_thread.start()
                assert store_started.wait(timeout=5.0)
                results = loader.load_chunks_to_hbm(
                    [_fake_key(0)],
                    [_disk_meta_for(512 * 4)],
                    lock_per_batch=True,
                    on_raw_batch_loaded=lambda *_args: None,
                    io_priority="speculative",
                )
                release_store.set()
                store_thread.join(timeout=5.0)

        assert not store_thread.is_alive()
        assert store_error == []
        assert results == [None]

    def test_parked_store_writer_does_not_cancel_speculative_read(self) -> None:
        """A writer waiting for idle time cannot suppress CSA prefetch."""
        loader, _ctrl = _make_loader(n_slots=2, q_depth=2)
        with loader._reader_gate:
            loader._writers_waiting = 1

        with patch.object(loader, "_load_batch", return_value=[None]) as load_batch:
            results = loader.load_chunks_to_hbm(
                [_fake_key(0)],
                [_disk_meta_for(512 * 4)],
                lock_per_batch=True,
                on_raw_batch_loaded=lambda *_args: None,
                io_priority="speculative",
                throttle_speculative=False,
            )

        load_batch.assert_called_once()
        assert results == [None]

    def test_speculative_io_cap_splits_batches(self) -> None:
        loader, _ctrl = _make_loader(n_slots=4, q_depth=4)
        size = 512 * 4
        keys = [_fake_key(i) for i in range(3)]
        metas = [
            _disk_meta_for(size, path=f"/tmp/speculative_{i}.kv") for i in range(3)
        ]
        callbacks: list[int] = []

        with patch.object(
            _tdl.FiemapHelper,
            "query_extents",
            return_value=[LbaRecord(slba=0, n_sectors=size // 512)],
        ):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs: Any) -> None:
                    n_ios = kwargs.get("n_ios", 0)
                    kwargs["status_out"][:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll
                with _patch_cuda_runtime_for_cpu_tests():
                    loader.load_chunks_to_hbm(
                        keys,
                        metas,
                        lock_per_batch=True,
                        on_raw_batch_loaded=(
                            lambda _start, indices, *_args: callbacks.append(
                                len(indices)
                            )
                        ),
                        io_priority="speculative",
                        max_batch_ios=1,
                    )

        assert callbacks == [1, 1, 1]

    def test_speculative_oversized_object_is_not_submitted(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, q_depth=4)
        key = _fake_key(0)
        meta = _disk_meta_for(512 * 4)

        with patch.object(loader, "_estimate_chunk_ios", return_value=2):
            with patch.object(loader, "_load_batch") as load_batch:
                results = loader.load_chunks_to_hbm(
                    [key],
                    [meta],
                    lock_per_batch=True,
                    on_raw_batch_loaded=lambda *_args: None,
                    io_priority="speculative",
                    max_batch_ios=1,
                )

        assert results == [None]
        load_batch.assert_not_called()

    def test_speculative_cancellation_stops_before_submission(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, q_depth=2)
        key = _fake_key(0)
        meta = _disk_meta_for(512 * 4)

        with patch.object(loader, "_load_batch") as load_batch:
            results = loader.load_chunks_to_hbm(
                [key],
                [meta],
                lock_per_batch=True,
                on_raw_batch_loaded=lambda *_args: None,
                io_priority="speculative",
                should_continue=lambda: False,
            )

        assert results == [None]
        load_batch.assert_not_called()

    def test_speculative_deadline_stops_before_submission(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, q_depth=2)
        key = _fake_key(0)
        meta = _disk_meta_for(512 * 4)

        with patch.object(loader, "_load_batch") as load_batch:
            results = loader.load_chunks_to_hbm(
                [key],
                [meta],
                lock_per_batch=True,
                on_raw_batch_loaded=lambda *_args: None,
                io_priority="speculative",
                deadline_monotonic=time.perf_counter() - 1.0,
            )

        assert results == [None]
        load_batch.assert_not_called()

    def test_speculative_budget_never_delays_announced_demand(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, q_depth=2)
        key = _fake_key(0)
        meta = _disk_meta_for(512 * 4)
        loader._speculative_tokens = 0
        loader._speculative_rate_bytes_per_s = 1
        loader._hp_readers = 1

        with patch.object(loader, "_load_batch") as load_batch:
            results = loader.load_chunks_to_hbm(
                [key],
                [meta],
                lock_per_batch=True,
                on_raw_batch_loaded=lambda *_args: None,
                io_priority="speculative",
                deadline_monotonic=time.perf_counter() + 1.0,
            )

        assert results == [None]
        load_batch.assert_not_called()

    def test_bounded_speculative_read_can_bypass_token_bucket(self) -> None:
        """A compute-window read may opt out of the background rate limit."""
        loader, _ctrl = _make_loader(n_slots=2, q_depth=2)
        loader._speculative_tokens = 0
        loader._speculative_rate_bytes_per_s = 1

        with patch.object(loader, "_load_batch", return_value=[None]) as load_batch:
            loader.load_chunks_to_hbm(
                [_fake_key(0)],
                [_disk_meta_for(512 * 4)],
                lock_per_batch=True,
                on_batch_loaded=lambda *_args: None,
                io_priority="speculative",
                deadline_monotonic=time.perf_counter() + 0.01,
                throttle_speculative=False,
            )

        load_batch.assert_called_once()

    def test_speculative_batches_sleep_only_for_announced_waiters(self) -> None:
        """Back-to-back lookahead batches do not pay an unconditional sleep."""
        loader, _ctrl = _make_loader(n_slots=2, q_depth=2)
        keys = [_fake_key(0), _fake_key(1)]
        metas = [_disk_meta_for(512 * 4), _disk_meta_for(512 * 4)]

        with patch.object(loader, "_load_batch", return_value=[None]):
            with patch.object(_tdl.time, "sleep") as sleep:
                loader.load_chunks_to_hbm(
                    keys,
                    metas,
                    lock_per_batch=True,
                    on_batch_loaded=lambda *_args: None,
                    io_priority="speculative",
                    max_batch_ios=1,
                    throttle_speculative=False,
                )

        sleep.assert_not_called()

    @pytest.mark.parametrize("argument", ["max_batch_bytes", "max_batch_ios"])
    def test_batch_limits_must_be_positive(self, argument: str) -> None:
        loader, _ctrl = _make_loader()

        with pytest.raises(ValueError, match=f"{argument} must be positive"):
            loader.load_chunks_to_hbm([], [], **{argument: 0})

    def test_batch_chunking(self) -> None:
        """More keys than n_slots → multiple sub-batches processed."""
        n_slots = 2
        loader, _ctrl = _make_loader(n_slots=n_slots)
        size = 512 * 4
        n_keys = 5  # > n_slots, forces two batches
        keys = [_fake_key(i) for i in range(n_keys)]
        metas = [_disk_meta_for(size) for _ in range(n_keys)]
        submitted_batch_sizes: list[int] = []

        def fake_fiemap(path):
            return [LbaRecord(slba=0, n_sectors=size // 512)]

        with patch.object(_tdl.FiemapHelper, "query_extents", side_effect=fake_fiemap):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read.side_effect = lambda **kwargs: (
                    submitted_batch_sizes.append(int(kwargs["staging_iovas"].numel()))
                )

                def fake_poll(**kwargs):
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll

                with _patch_cuda_runtime_for_cpu_tests():
                    results = loader.load_chunks_to_hbm(keys, metas)

        assert len(results) == n_keys
        # All five should succeed
        for r in results:
            assert r is not None
        assert submitted_batch_sizes == [1, 1, 1, 1, 1]
        assert max(submitted_batch_sizes) < loader._q_depth()

    def test_small_chunks_pack_beyond_slot_count(self) -> None:
        """Small chunks use queue/staging capacity instead of fixed slot count."""
        loader, _ctrl = _make_loader(n_slots=2, slot_mb=1, q_depth=8)
        size = 512 * 4
        n_keys = 4
        keys = [_fake_key(i) for i in range(n_keys)]
        metas = [_disk_meta_for(size, path=f"/tmp/fake_{i}.kv") for i in range(n_keys)]

        def fake_fiemap(path):
            return [LbaRecord(slba=0, n_sectors=size // 512)]

        with patch.object(_tdl.FiemapHelper, "query_extents", side_effect=fake_fiemap):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs):
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll

                with _patch_cuda_runtime_for_cpu_tests():
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

        with patch.object(_tdl.FiemapHelper, "query_extents", side_effect=fake_fiemap):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs):
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll

                with _patch_cuda_runtime_for_cpu_tests():
                    results = loader.load_chunks_to_hbm(keys, metas)

        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None

    def test_explicit_read_ranges_submit_multiple_reads(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, q_depth=8)
        size = 1024
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]
        read_ranges = [
            [
                KVObjectByteRange(offset=0, length=512, target_offset=0),
                KVObjectByteRange(offset=1024, length=512, target_offset=512),
            ]
        ]
        extents = [
            LbaRecord(slba=100, n_sectors=1, file_offset=0),
            LbaRecord(slba=200, n_sectors=1, file_offset=1024),
        ]

        with patch.object(_tdl.FiemapHelper, "query_extents", return_value=extents):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs):
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll

                with _patch_cuda_runtime_for_cpu_tests():
                    results = loader.load_chunks_to_hbm(
                        keys,
                        metas,
                        read_ranges_per_key=read_ranges,
                    )

        assert results[0] is not None
        submit_kwargs = mock_c.tutti_submit_batch_sgl_read.call_args.kwargs
        assert submit_kwargs["byte_lens"].cpu().tolist() == [512, 512]
        assert submit_kwargs["slbas"].cpu().tolist() == [100, 200]
        assert submit_kwargs["staging_iovas"].cpu().tolist() == [0, 512]

    def test_explicit_single_read_range_reads_padded_tail(self) -> None:
        loader, _ctrl = _make_loader(n_slots=2, q_depth=8)
        size = 1000
        keys = [_fake_key(0)]
        metas = [_disk_meta_for(size)]
        read_ranges = [[KVObjectByteRange(offset=0, length=size, target_offset=0)]]

        with patch.object(
            _tdl.FiemapHelper,
            "query_extents",
            return_value=[LbaRecord(slba=100, n_sectors=2, file_offset=0)],
        ):
            with patch.object(_tdl, "_c_ops") as mock_c:
                mock_c.tutti_submit_batch_sgl_read = MagicMock()

                def fake_poll(**kwargs):
                    n_ios = kwargs.get("n_ios", 0)
                    loader._status_buf[:n_ios] = 0

                mock_c.tutti_poll_batch.side_effect = fake_poll

                with _patch_cuda_runtime_for_cpu_tests():
                    results = loader.load_chunks_to_hbm(
                        keys,
                        metas,
                        read_ranges_per_key=read_ranges,
                    )

        assert results[0] is not None
        assert results[0].get_size() == size
        submit_kwargs = mock_c.tutti_submit_batch_sgl_read.call_args.kwargs
        assert submit_kwargs["byte_lens"].cpu().tolist() == [1024]
        assert submit_kwargs["staging_iovas"].cpu().tolist() == [0]


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

        with (
            patch.object(_tdl, "_HAS_C_OPS", True),
            patch.object(_tdl, "SnvmeSession", return_value=mock_session),
            patch.object(_tdl, "_cuda_malloc_device", return_value=1 << 20),
            patch.object(_tdl, "_get_cudart", return_value=MagicMock()),
            patch.object(_tdl, "_cuda_malloc_managed", return_value=id(b"x")),
            patch("ctypes.c_uint16.from_address", return_value=MagicMock()),
            patch("ctypes.c_uint8.from_address", return_value=MagicMock()),
            patch("ctypes.c_int32.from_address", return_value=MagicMock()),
        ):
            with _patch_cuda_runtime_for_cpu_tests():
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

        def spy_free(ptr):
            freed.append(ptr)

        with patch.object(_tdl, "_cuda_free", side_effect=spy_free):
            loader.close()

        assert len(freed) == 5  # control scalars plus raw staging allocation
        loader._session.close.assert_called_once()

    def test_context_manager(self) -> None:
        loader, _ctrl = _make_loader()
        with patch.object(_tdl, "_cuda_free"):
            with loader:
                pass
        loader._session.close.assert_called_once()


# ── staging slice / slot iova helpers ────────────────────────────────────────


class TestStagingHelpers:
    def test_cuda_device_is_public(self) -> None:
        loader, _ctrl = _make_loader()
        assert loader.cuda_device == 0

    def test_io_stream_is_public(self) -> None:
        loader, _ctrl = _make_loader()
        assert loader.io_stream is None

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
