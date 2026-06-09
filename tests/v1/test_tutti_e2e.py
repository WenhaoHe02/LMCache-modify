# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU-direct NVMe test for TuttiDirectLoader.

Requires on the test host:
  - snvme-core.ko + snvme.ko loaded
  - /dev/snvm_control accessible
  - At least one NVMe drive with a mounted filesystem
  - lmcache.c_ops built with tutti_kv_ops.cu

Usage (single drive):
  python test_tutti_e2e.py --bdf 0000:10:00.0 --mount /mnt/nvme2

Usage (8 drives):
  python test_tutti_e2e.py --multi \
    0000:10:00.0:/mnt/nvme2 \
    0000:1c:00.0:/mnt/nvme3 ...
"""

import argparse
import os
import sys
import time
import tempfile
import concurrent.futures

import torch

# ── bootstrap: stub out heavy lmcache package imports ──────────────────────
import types
if "lmcache.v1.gpu_connector" not in sys.modules:
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, "..", ".."))
    _gc_dir = os.path.join(_root, "lmcache", "v1", "gpu_connector")
    _gc_stub = types.ModuleType("lmcache.v1.gpu_connector")
    _gc_stub.__path__ = [_gc_dir]
    _gc_stub.__package__ = "lmcache.v1.gpu_connector"
    sys.modules["lmcache.v1.gpu_connector"] = _gc_stub

from lmcache.v1.gpu_connector.tutti_direct_loader import TuttiDirectLoader, FiemapHelper  # noqa
from lmcache.utils import CacheEngineKey, DiskCacheMetadata  # noqa
from lmcache.v1.memory_management import MemoryFormat  # noqa

_CTRL = "/dev/snvm_control"
_DEV  = "/dev/ssnvme0"
_NSID = 1
_SLOT_MB = 8
_N_SLOTS = 8
_CHUNK_BYTES = 4 * 1024 * 1024   # 4 MiB test chunk (≤ drive MDTS of 4 MiB)


def _write_test_file(mount: str) -> str:
    """Write a known-pattern file and return its path."""
    import subprocess
    path = os.path.join(mount, "tutti_e2e_test.bin")
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    # fallocate first to guarantee a single contiguous extent on ext4.
    subprocess.run(["fallocate", "-l", str(_CHUNK_BYTES), path], check=True)
    data = bytes(range(256)) * (_CHUNK_BYTES // 256)
    with open(path, "r+b") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return path


def _verify(tensor: torch.Tensor, size: int) -> bool:
    """Check that GPU tensor matches the known pattern."""
    cpu = tensor[:size].cpu()
    expected = torch.tensor(
        list(bytes(range(256)) * (size // 256)), dtype=torch.uint8
    )
    return torch.equal(cpu, expected)


def run_single(bdf: str, mount: str, cuda_device: int = 0) -> None:
    print(f"\n=== Single-drive test: {bdf} -> {mount} (cuda:{cuda_device}) ===")

    if not os.path.exists(_CTRL):
        print(f"[SKIP] {_CTRL} not found; is snvme loaded?")
        return

    path = _write_test_file(mount)
    print(f"  Wrote {_CHUNK_BYTES // (1024*1024)} MiB test file: {path}")

    # Pre-compute FIEMAP before SNVM_DEVICE_BIND — bind takes exclusive NVMe control,
    # making the filesystem EIO.  Calling FIEMAP after bind may fail on ext4 errors.
    lba_record = FiemapHelper.single_contiguous(path)
    print(f"  FIEMAP: slba={lba_record.slba} n_sectors={lba_record.n_sectors}")

    try:
        t0 = time.perf_counter()
        loader = TuttiDirectLoader.create(
            device_path=_DEV,
            ctrl_path=_CTRL,
            pci_bdf=bdf,
            n_slots=_N_SLOTS,
            slot_bytes=_SLOT_MB * 1024 * 1024,
            nsid=_NSID,
            cuda_device=cuda_device,
        )
        t_init = time.perf_counter() - t0
        print(f"  Session init: {t_init*1000:.1f} ms")

        # Inject pre-computed LBA so load_chunks_to_hbm skips FIEMAP
        # (the filesystem is EIO after SNVM_DEVICE_BIND).
        loader._lba_cache[path] = lba_record

        key = CacheEngineKey(
            model_name="e2e_test", world_size=1, worker_id=0,
            chunk_hash=0, dtype=torch.float16,
        )
        meta = DiskCacheMetadata(
            path=path,
            size=_CHUNK_BYTES,
            shape=torch.Size([1, _CHUNK_BYTES]),
            dtype=torch.uint8,
            fmt=MemoryFormat.KV_2LTD,
            shapes=[torch.Size([1, _CHUNK_BYTES])],
            dtypes=[torch.uint8],
        )

        t0 = time.perf_counter()
        results = loader.load_chunks_to_hbm([key], [meta])
        t_load = time.perf_counter() - t0

        if results[0] is None:
            print("  [FAIL] load_chunks_to_hbm returned None")
        else:
            obj = results[0]
            tensor = obj.raw_data if hasattr(obj, "raw_data") else obj._tensor
            ok = _verify(tensor, _CHUNK_BYTES)
            bw_gbs = _CHUNK_BYTES / t_load / 1e9
            print(f"  Load time : {t_load*1000:.1f} ms")
            print(f"  Bandwidth : {bw_gbs:.2f} GB/s")
            print(f"  Data check: {'PASS' if ok else 'FAIL'}")

        loader.close()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass  # filesystem EIO after bind; file will be removed on next mount


def run_multi(bdf_mounts: list[tuple[str, str, int]]) -> None:
    print(f"\n=== Multi-drive test: {len(bdf_mounts)} drives ===")

    if not os.path.exists(_CTRL):
        print(f"[SKIP] {_CTRL} not found")
        return

    loaders = []
    paths = []
    lba_records = []
    try:
        for i, (bdf, mount, _cuda_dev) in enumerate(bdf_mounts):
            path = _write_test_file(mount)
            paths.append(path)
            # Pre-compute FIEMAP before any bind (filesystem goes EIO after bind).
            lba_records.append(FiemapHelper.single_contiguous(path))
            print(f"  [{i}] {bdf} -> {mount}  slba={lba_records[-1].slba}")

        for i, (bdf, mount, cuda_dev) in enumerate(bdf_mounts):
            loader = TuttiDirectLoader.create(
                device_path="/dev/ssnvme0",
                ctrl_path=_CTRL,
                pci_bdf=bdf,
                n_slots=_N_SLOTS,
                slot_bytes=_SLOT_MB * 1024 * 1024,
                nsid=_NSID,
                cuda_device=cuda_dev,
            )
            loader._lba_cache[paths[i]] = lba_records[i]
            loaders.append(loader)

        print(f"  All {len(loaders)} sessions initialised")

        def _load_one(args):
            i, loader, path = args
            key = CacheEngineKey(
                model_name="e2e_multi", world_size=1, worker_id=i,
                chunk_hash=i, dtype=torch.float16,
            )
            meta = DiskCacheMetadata(
                path=path,
                size=_CHUNK_BYTES,
                shape=torch.Size([1, _CHUNK_BYTES]),
                dtype=torch.uint8,
                fmt=MemoryFormat.KV_2LTD,
                shapes=[torch.Size([1, _CHUNK_BYTES])],
                dtypes=[torch.uint8],
            )
            return loader.load_chunks_to_hbm([key], [meta])

        # Fire all drives in parallel threads
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(loaders)) as pool:
            all_results = list(pool.map(
                _load_one,
                [(i, loader, path) for i, (loader, path) in enumerate(zip(loaders, paths))],
            ))
        t_total = time.perf_counter() - t0

        pass_count = sum(1 for r in all_results if r[0] is not None)
        total_bytes = _CHUNK_BYTES * len(bdf_mounts)
        agg_bw = total_bytes / t_total / 1e9
        print(f"  Pass: {pass_count}/{len(bdf_mounts)}")
        print(f"  Total time   : {t_total*1000:.1f} ms")
        print(f"  Aggregate BW : {agg_bw:.2f} GB/s")

    finally:
        for loader in loaders:
            try:
                loader.close()
            except Exception:
                pass
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdf", default="0000:e4:00.0")
    parser.add_argument("--mount", default="/mnt/nvme5")
    parser.add_argument(
        "--cuda-device", type=int, default=7,
        help=(
            "CUDA device index for staging buffer (must be NUMA-local to the NVMe). "
            "DGX H200: NUMA0 GPUs 0-3 (buses 03,29,59,63), "
            "NUMA1 GPUs 4-7 (buses 7B,A3,D3,E5). "
            "NVMe NUMA0: 10,1c,4b,6f; NUMA1: 88,a2,cc,e4. "
            "Default 7 matches default BDF 0000:e4:00.0 (NUMA1)."
        ),
    )
    parser.add_argument("--multi", nargs="*",
                        help="bdf:mount:cuda_device triples, e.g. 0000:10:00.0:/mnt/nvme2:1")
    args = parser.parse_args()

    if args.multi is not None:
        triples = []
        for item in args.multi:
            # Expected format: "DDDD:BB:SS.F:/mnt/path:cuda_device"
            # rsplit on ":", 1 separates cuda_device from the rest.
            parts = item.rsplit(":", 1)
            if len(parts) == 2 and parts[1].isdigit():
                bdf_mount, cuda_dev_str = parts
                # bdf_mount = "DDDD:BB:SS.F:/mnt/path" — 4 colon-separated fields
                bm_parts = bdf_mount.split(":")
                if len(bm_parts) == 4:
                    # Standard BDF with domain: DDDD:BB:SS.F:/mnt/path
                    bdf = ":".join(bm_parts[:3])
                    mount = bm_parts[3]
                elif len(bm_parts) == 3:
                    # Short BDF without domain: BB:SS.F:/mnt/path
                    bdf = ":".join(bm_parts[:2])
                    mount = bm_parts[2]
                else:
                    # Fallback
                    bdf = ":".join(bm_parts[:-1])
                    mount = bm_parts[-1]
                triples.append((bdf, mount, int(cuda_dev_str)))
            else:
                # Legacy "bdf:mount" format — derive cuda_device from BDF bus
                bdf, mount = item.split(":", 1)
                triples.append((bdf, mount, args.cuda_device))
        run_multi(triples)
    else:
        run_single(args.bdf, args.mount, cuda_device=args.cuda_device)
