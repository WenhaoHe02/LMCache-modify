# SPDX-License-Identifier: Apache-2.0
"""Benchmark one DSv4-Flash CSA layer through Tutti and GPU scatter.

This benchmark intentionally isolates the geometry seen in the V28 480K
prefix run.  It measures two independent stages:

* Tutti reads 1,874 CSA-attention chunks (one 64 x 584-byte slice per chunk).
* GPU-to-GPU materialization compares the V28 one-launch-per-chunk path with
  one dynamic-pointer batch for the whole layer.

The Tutti mode binds an NVMe controller to snvme.  Run it only on a dedicated
test host and restore the native NVMe driver after the process exits.
"""

# Standard
import argparse
import inspect
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

# Third Party
import torch


@dataclass(frozen=True)
class DSv4FlashCSALayerGeometry:
    """V28 geometry for one CSA attention layer on one TP rank."""

    base_tokens: int = 480_000
    recompute_tokens: int = 16_000
    chunk_tokens: int = 256
    num_chunks: int = 1_874
    csa_layers: int = 21
    rows_per_chunk: int = 64
    attention_row_bytes: int = 584
    indexer_row_bytes: int = 132

    @property
    def cached_tokens(self) -> int:
        """Return the full 256-token chunks present in the V28 prefix hit."""
        return self.num_chunks * self.chunk_tokens

    @property
    def attention_chunk_bytes(self) -> int:
        """Return bytes read for one CSA-attention layer in one chunk."""
        return self.rows_per_chunk * self.attention_row_bytes

    @property
    def attention_layer_bytes(self) -> int:
        """Return bytes read for one CSA-attention layer across the prefix."""
        return self.num_chunks * self.attention_chunk_bytes

    @property
    def indexer_chunk_bytes(self) -> int:
        """Return bytes in one CSA-indexer tensor slice per chunk."""
        return self.rows_per_chunk * self.indexer_row_bytes

    @property
    def indexer_layer_bytes(self) -> int:
        """Return bytes in one CSA-indexer layer across the prefix."""
        return self.num_chunks * self.indexer_chunk_bytes

    def validate(self) -> None:
        """Reject accidental use of the older 30-layer DSv4 layout."""
        if self.csa_layers != 21:
            raise ValueError("DSv4-Flash must use 21 CSA layers")
        if self.num_chunks != 1_874:
            raise ValueError("V28 480K prefix must contain 1,874 full chunks")
        if self.rows_per_chunk * 4 != self.chunk_tokens:
            raise ValueError("CSA cache must use compression ratio 4")


def _summary(values: list[float]) -> dict[str, float]:
    """Return compact millisecond statistics for benchmark samples."""
    if not values:
        return {}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "min_ms": min(values),
        "p50_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": ordered[p95_index],
        "max_ms": max(values),
    }


def geometry_report(geometry: DSv4FlashCSALayerGeometry) -> dict[str, Any]:
    """Build the JSON-serializable shape and byte report."""
    geometry.validate()
    return {
        **asdict(geometry),
        "cached_tokens": geometry.cached_tokens,
        "attention_chunk_shape": [
            geometry.rows_per_chunk,
            1,
            geometry.attention_row_bytes,
        ],
        "attention_chunk_bytes": geometry.attention_chunk_bytes,
        "attention_layer_bytes": geometry.attention_layer_bytes,
        "attention_layer_mib": geometry.attention_layer_bytes / 2**20,
        "indexer_chunk_shape": [geometry.rows_per_chunk, 1, geometry.indexer_row_bytes],
        "indexer_chunk_bytes": geometry.indexer_chunk_bytes,
        "indexer_layer_bytes": geometry.indexer_layer_bytes,
        "indexer_layer_mib": geometry.indexer_layer_bytes / 2**20,
    }


def _cuda_timing(
    operation: Callable[[], None],
    repetitions: int,
) -> tuple[list[float], list[float]]:
    """Measure synchronized wall and CUDA-stream elapsed time."""
    wall_samples: list[float] = []
    cuda_samples: list[float] = []
    for _ in range(repetitions):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start_event.record()
        operation()
        end_event.record()
        end_event.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1_000)
        cuda_samples.append(float(start_event.elapsed_time(end_event)))
    return wall_samples, cuda_samples


def benchmark_g2g(
    geometry: DSv4FlashCSALayerGeometry,
    cuda_device: int,
    repetitions: int,
) -> dict[str, Any]:
    """Compare 1,874 legacy launches with one pointer-batched G2G launch."""
    import lmcache.c_ops as ops

    if not hasattr(ops, "multi_layer_block_kv_transfer_batched"):
        raise RuntimeError("lmcache.c_ops lacks multi_layer_block_kv_transfer_batched")

    geometry.validate()
    device = torch.device(f"cuda:{cuda_device}")
    num_chunks = geometry.num_chunks
    rows = geometry.rows_per_chunk
    row_bytes = geometry.attention_row_bytes

    with torch.cuda.device(device), torch.inference_mode():
        # [object, kv, layer, compressed-token, byte]
        sources = torch.empty(
            num_chunks,
            1,
            1,
            rows,
            row_bytes,
            dtype=torch.uint8,
            device=device,
        )
        for chunk_index in range(num_chunks):
            sources[chunk_index].fill_(chunk_index % 251 + 1)
        destination = torch.zeros(
            num_chunks * rows,
            1,
            row_bytes,
            dtype=torch.uint8,
            device=device,
        )

        pointer_build_start = time.perf_counter()
        source_ptrs_list = [sources[index].data_ptr() for index in range(num_chunks)]
        source_ptrs = torch.tensor(source_ptrs_list, dtype=torch.int64, device=device)
        destination_ptrs = torch.tensor(
            [destination.data_ptr()], dtype=torch.int64, device=device
        )
        block_ids = torch.arange(num_chunks, dtype=torch.int64, device=device)
        torch.cuda.synchronize()
        descriptor_build_ms = (time.perf_counter() - pointer_build_start) * 1_000

        shape_desc = ops.PageBufferShapeDesc()
        shape_desc.kv_size = 1
        shape_desc.nl = 1
        shape_desc.nb = num_chunks
        shape_desc.bs = rows
        shape_desc.nh = 1
        shape_desc.hs = row_bytes
        shape_desc.element_size = 1
        gpu_format = ops.GPUKVFormat.NL_X_NBBS_ONE_HS

        def legacy_operation() -> None:
            for chunk_index, source_ptr in enumerate(source_ptrs_list):
                ops.multi_layer_block_kv_transfer(
                    destination_ptrs,
                    [source_ptr],
                    block_ids[chunk_index : chunk_index + 1],
                    device,
                    ops.TransferDirection.H2D,
                    shape_desc,
                    rows,
                    gpu_format,
                    0,
                )

        def batched_operation() -> None:
            ops.multi_layer_block_kv_transfer_batched(
                destination_ptrs,
                source_ptrs,
                block_ids,
                device,
                ops.TransferDirection.H2D,
                shape_desc,
                rows,
                gpu_format,
                0,
            )

        legacy_operation()
        batched_operation()
        torch.cuda.synchronize()
        legacy_wall, legacy_cuda = _cuda_timing(legacy_operation, repetitions)
        batched_wall, batched_cuda = _cuda_timing(batched_operation, repetitions)

        for chunk_index in (0, num_chunks // 2, num_chunks - 1):
            expected = chunk_index % 251 + 1
            actual = int(destination[chunk_index * rows, 0, 0].item())
            if actual != expected:
                raise AssertionError(
                    f"G2G mismatch at chunk {chunk_index}: {actual} != {expected}"
                )

    legacy_p50 = statistics.median(legacy_wall)
    batched_p50 = statistics.median(batched_wall)
    return {
        "num_chunks": num_chunks,
        "bytes": geometry.attention_layer_bytes,
        "legacy_launches": num_chunks,
        "batched_launches": 1,
        "descriptor_build_ms": descriptor_build_ms,
        "legacy_wall": _summary(legacy_wall),
        "legacy_cuda": _summary(legacy_cuda),
        "batched_wall": _summary(batched_wall),
        "batched_cuda": _summary(batched_cuda),
        "wall_speedup": legacy_p50 / batched_p50,
    }


def _prepare_layer_file(path: Path, nbytes: int) -> None:
    """Allocate the synthetic layer file before snvme takes the drive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        file_obj.truncate(nbytes)
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(file_obj.fileno(), 0, nbytes)
        file_obj.flush()
        os.fsync(file_obj.fileno())


def benchmark_tutti_load(
    geometry: DSv4FlashCSALayerGeometry,
    *,
    file_path: Path,
    pci_bdf: str,
    cuda_device: int,
    repetitions: int,
    device_path: str,
    ctrl_path: str,
) -> dict[str, Any]:
    """Read one synthetic CSA layer using the real Tutti direct-load API."""
    from lmcache.utils import CacheEngineKey, DiskCacheMetadata
    from lmcache.v1.gpu_connector.tutti_direct_loader import (
        FiemapHelper,
        TuttiDirectLoader,
    )
    from lmcache.v1.memory_management import MemoryFormat

    geometry.validate()
    _prepare_layer_file(file_path, geometry.attention_layer_bytes)
    extents = FiemapHelper.query_extents(str(file_path))

    keys = [
        CacheEngineKey(
            model_name="dsv4_flash_csa_layer_bench",
            world_size=1,
            worker_id=cuda_device,
            chunk_hash=chunk_index,
            dtype=torch.uint8,
        )
        for chunk_index in range(geometry.num_chunks)
    ]
    chunk_shape = torch.Size([geometry.rows_per_chunk, 1, geometry.attention_row_bytes])
    metadata = DiskCacheMetadata(
        path=str(file_path),
        size=geometry.attention_chunk_bytes,
        shape=chunk_shape,
        dtype=torch.uint8,
        fmt=MemoryFormat.KV_2LTD,
        shapes=[chunk_shape],
        dtypes=[torch.uint8],
    )
    disk_metadatas = [metadata] * geometry.num_chunks
    file_offsets = [
        chunk_index * geometry.attention_chunk_bytes
        for chunk_index in range(geometry.num_chunks)
    ]

    init_start = time.perf_counter()
    loader = TuttiDirectLoader.create(
        device_path=device_path,
        ctrl_path=ctrl_path,
        pci_bdf=pci_bdf,
        n_slots=4,
        slot_bytes=128 * 1024**2,
        nsid=1,
        cuda_device=cuda_device,
        initial_lba_cache={str(file_path): extents},
    )
    init_ms = (time.perf_counter() - init_start) * 1_000
    samples: list[float] = []
    indexed_samples: list[float] = []
    batches_per_run: list[int] = []
    indexed_batches_per_run: list[int] = []
    bytes_per_run: list[int] = []
    load_parameters = inspect.signature(loader.load_chunks_to_hbm).parameters
    supports_raw_callback = "on_raw_batch_loaded" in load_parameters
    try:
        indexed_supported = hasattr(loader, "load_indexed_chunks_to_hbm")
        indexed_supported = indexed_supported and hasattr(
            __import__("lmcache.c_ops", fromlist=["c_ops"]),
            "tutti_submit_indexed_sgl_read",
        )
        selected_ids = torch.arange(
            geometry.num_chunks,
            dtype=torch.int64,
            device=f"cuda:{cuda_device}",
        )
        slbas: list[int] = []
        for file_offset in file_offsets:
            extent = next(
                (
                    record
                    for record in extents
                    if record.file_offset <= file_offset
                    and file_offset + geometry.attention_chunk_bytes
                    <= record.file_offset + record.n_sectors * 512
                ),
                None,
            )
            if extent is None:
                indexed_supported = False
                break
            slbas.append(
                extent.slba + (file_offset - extent.file_offset) // 512
            )
        slba_table = torch.tensor(
            slbas,
            dtype=torch.int64,
            device=f"cuda:{cuda_device}",
        )

        for _ in range(repetitions):
            batch_count = 0
            completed_bytes = 0

            def consume_raw(
                _batch_start: int,
                completed_indices: list[int],
                _completed_offsets: list[int],
                completed_nbytes: list[int],
                _staging: torch.Tensor,
            ) -> None:
                nonlocal batch_count, completed_bytes
                batch_count += 1
                if len(completed_indices) != len(completed_nbytes):
                    raise AssertionError("Tutti completion descriptors disagree")
                completed_bytes += sum(completed_nbytes)

            def consume_wrapped(
                _batch_start: int,
                batch_results: list[Any],
            ) -> None:
                nonlocal batch_count, completed_bytes
                loaded = sum(result is not None for result in batch_results)
                batch_count += 1
                completed_bytes += loaded * geometry.attention_chunk_bytes

            start = time.perf_counter()
            load_kwargs: dict[str, Any] = {"file_offsets": file_offsets}
            if "io_priority" in load_parameters:
                load_kwargs["io_priority"] = "demand"
            if supports_raw_callback:
                load_kwargs["on_raw_batch_loaded"] = consume_raw
            else:
                load_kwargs["on_batch_loaded"] = consume_wrapped
            loader.load_chunks_to_hbm(keys, disk_metadatas, **load_kwargs)
            samples.append((time.perf_counter() - start) * 1_000)
            batches_per_run.append(batch_count)
            bytes_per_run.append(completed_bytes)
            if completed_bytes != geometry.attention_layer_bytes:
                raise AssertionError(
                    f"short Tutti layer read: {completed_bytes} / "
                    f"{geometry.attention_layer_bytes}"
                )

        if indexed_supported:
            for _ in range(repetitions):
                batch_count = 0
                completed_bytes = 0

                def consume_indexed(
                    _batch_start: int,
                    batch_ids: torch.Tensor,
                    _staging_stride: int,
                    logical_nbytes: int,
                    _staging: torch.Tensor,
                ) -> None:
                    nonlocal batch_count, completed_bytes
                    batch_count += 1
                    completed_bytes += int(batch_ids.numel()) * logical_nbytes

                start = time.perf_counter()
                loader.load_indexed_chunks_to_hbm(
                    selected_ids,
                    slba_table,
                    geometry.attention_chunk_bytes,
                    consume_indexed,
                )
                indexed_samples.append((time.perf_counter() - start) * 1_000)
                indexed_batches_per_run.append(batch_count)
                if completed_bytes != geometry.attention_layer_bytes:
                    raise AssertionError(
                        f"short indexed Tutti layer read: {completed_bytes} / "
                        f"{geometry.attention_layer_bytes}"
                    )
    finally:
        loader.close()

    p50_ms = statistics.median(samples)
    report = {
        "file": str(file_path),
        "extents": len(extents),
        "callback_mode": "raw" if supports_raw_callback else "wrapped_v28",
        "init_ms": init_ms,
        "load_wall": _summary(samples),
        "batches_per_run": batches_per_run,
        "bytes_per_run": bytes_per_run,
        "effective_gib_per_s": geometry.attention_layer_bytes
        / 2**30
        / (p50_ms / 1_000),
    }
    if indexed_samples:
        indexed_p50_ms = statistics.median(indexed_samples)
        report.update(
            {
                "indexed_load_wall": _summary(indexed_samples),
                "indexed_batches_per_run": indexed_batches_per_run,
                "indexed_effective_gib_per_s": geometry.attention_layer_bytes
                / 2**30
                / (indexed_p50_ms / 1_000),
                "indexed_wall_speedup": p50_ms / indexed_p50_ms,
            }
        )
    return report


def main() -> None:
    """Run the selected one-layer benchmark stages and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--skip-g2g", action="store_true")
    parser.add_argument("--tutti", action="store_true")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("/mnt/nvme0/tutti_csa_layer_bench.bin"),
    )
    parser.add_argument("--pci-bdf", default="0000:6f:00.0")
    parser.add_argument("--device-path", default="/dev/ssnvme0")
    parser.add_argument("--ctrl-path", default="/dev/snvm_control")
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    geometry = DSv4FlashCSALayerGeometry()
    report: dict[str, Any] = {"geometry": geometry_report(geometry)}
    if not args.skip_g2g:
        report["g2g"] = benchmark_g2g(
            geometry,
            args.cuda_device,
            args.repetitions,
        )
    if args.tutti:
        report["tutti_load"] = benchmark_tutti_load(
            geometry,
            file_path=args.file,
            pci_bdf=args.pci_bdf,
            cuda_device=args.cuda_device,
            repetitions=args.repetitions,
            device_path=args.device_path,
            ctrl_path=args.ctrl_path,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
