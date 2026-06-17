"""
bench_csa_prefetch.py — Benchmark speculative block prefetcher for DSV4 CSA.

Measures the I/O latency reduction achievable by prefetching only the
delta blocks (new blocks not seen in the previous decode step) instead of
naive full-topk reads each step.

The benchmark does NOT require the actual DSV4 model to be present: it works
in two modes:

  --mode fake
      Writes synthetic random blocks to a temp directory, then runs a simulated
      decode loop using pre-recorded topk traces (or random topk with realistic
      delta pattern) and CSAPrefetcher.  Measures:
        (a) Baseline:     sequential pread for all 1 024 blocks/step/layer.
        (b) Speculative:  async pread for only delta blocks (~92/step/layer).

  --mode real
      Loads the actual DSV4 model, patches it, runs decode, then reports
      real per-step delta sizes, overlap rates, and I/O timing.

Run (fake mode):
  python bench_csa_prefetch.py --mode fake \\
      --n-layers 30 --decode-steps 50 --block-size 4096 \\
      --n-blocks 8192 --io-workers 8

Run (real model, TP=8):
  torchrun --nproc_per_node=8 bench_csa_prefetch.py --mode real \\
      --model-path /mnt/nvme0/models/DeepSeek-V4-Pro \\
      --store-dir /mnt/nvme0/csa_blocks \\
      --prompt-len 32768 --decode-steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import as_completed
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="fake", choices=["fake", "real"],
                    help="fake = synthetic IO benchmark; real = model+IO")

# Fake mode
parser.add_argument("--n-layers",     type=int, default=30)
parser.add_argument("--decode-steps", type=int, default=50)
parser.add_argument("--block-size",   type=int, default=4096,
                    help="Bytes per block (compress_ratio * kv_lora_rank * 2)")
parser.add_argument("--n-blocks",     type=int, default=8192,
                    help="Total compressed blocks per layer "
                         "(= max_seq_len // compress_ratio)")
parser.add_argument("--index-topk",   type=int, default=1024)
parser.add_argument("--delta-rate",   type=float, default=0.09,
                    help="Fraction of topk that changes each step (~9%%)")
parser.add_argument("--io-workers",   type=int, default=8)
parser.add_argument("--store-dir",    default="",
                    help="Directory for block files (default: tmp dir)")
parser.add_argument("--speculative",  action="store_true",
                    help="Use HC-proxy speculative mode (CSASpecPrefetcher)")

# Real mode
parser.add_argument("--model-path",  default="")
parser.add_argument("--prompt-len",  type=int, default=32768)
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--compress-ratio", type=int, default=4)
parser.add_argument("--kv-lora-rank",   type=int, default=512)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Import prefetcher
# ---------------------------------------------------------------------------

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from lmcache.v1.csa_prefetcher import (
    CSABlockStore,
    CSABlockStoreConfig,
    CSAPrefetcher,
    CSASpecPrefetcher,
)

# ---------------------------------------------------------------------------
# Helper: write synthetic block data
# ---------------------------------------------------------------------------


def write_fake_blocks(store: CSABlockStore, layer_ids: List[int],
                      n_blocks: int, block_size: int) -> None:
    """Fill every layer file with random bytes."""
    total_mb = len(layer_ids) * n_blocks * block_size / 1024 / 1024
    print(f"Writing {len(layer_ids)} layer files × {n_blocks} blocks × "
          f"{block_size} B = {total_mb:.1f} MB ...")
    rng = np.random.default_rng(0)
    for lid in layer_ids:
        raw = rng.integers(0, 256, n_blocks * block_size,
                           dtype=np.uint8).tobytes()
        path = os.path.join(store._cfg.store_dir, f"csa_layer_{lid}.bin")
        with open(path, "wb") as f:
            f.write(raw)
    print("  done.")


# ---------------------------------------------------------------------------
# Helper: simulate realistic topk trace
# ---------------------------------------------------------------------------


def make_topk_trace(n_steps: int, index_topk: int, n_blocks: int,
                    delta_rate: float,
                    seed: int = 42) -> List[List[Set[int]]]:
    """Generate a realistic per-layer topk trace.

    Returns:
        trace[step][layer] = set of selected block ids
    """
    rng = random.Random(seed)
    n_layers = 1  # caller uses same trace for all layers

    # Initial topk: random
    current = set(rng.sample(range(n_blocks), index_topk))
    trace = []
    for step in range(n_steps):
        trace.append(current)
        # Replace delta_rate fraction with fresh random draws
        n_replace = max(1, int(index_topk * delta_rate))
        to_remove = set(rng.sample(sorted(current), n_replace))
        available = list(set(range(n_blocks)) - current)
        to_add = set(rng.sample(available, n_replace))
        current = (current - to_remove) | to_add

    return trace


# ---------------------------------------------------------------------------
# Fake mode benchmark
# ---------------------------------------------------------------------------


def bench_fake() -> None:
    store_dir = args.store_dir or tempfile.mkdtemp(prefix="csa_bench_")
    layer_ids = list(range(args.n_layers))

    cfg = CSABlockStoreConfig(
        store_dir=store_dir,
        n_blocks=args.n_blocks,
        block_size_bytes=args.block_size,
        io_workers=args.io_workers,
    )
    store = CSABlockStore(cfg)

    write_fake_blocks(store, layer_ids, args.n_blocks, args.block_size)

    prefetcher = CSAPrefetcher(store, csa_layer_ids=layer_ids,
                               index_topk=args.index_topk)

    # Generate per-layer topk traces
    print(f"Generating topk traces: {args.n_layers} layers × "
          f"{args.decode_steps} steps, delta_rate={args.delta_rate:.0%} ...")
    traces: Dict[int, List[Set[int]]] = {}
    for lid in layer_ids:
        traces[lid] = make_topk_trace(
            args.decode_steps, args.index_topk, args.n_blocks,
            args.delta_rate, seed=lid)

    # ----------------------------------------------------------------
    # Baseline: synchronous full-topk reads each step
    # ----------------------------------------------------------------
    print("\n--- Baseline: sync read all topk blocks each step ---")
    baseline_step_ms: List[float] = []
    for step in range(args.decode_steps):
        t0 = time.perf_counter()
        futs = {}
        for lid in layer_ids:
            blk_set = traces[lid][step]
            futs.update(store.read_blocks_async(lid, blk_set))
        # block until all done
        for f in as_completed(futs.values()):
            f.result()
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        baseline_step_ms.append(elapsed_ms)
        if step % 10 == 0:
            print(f"  step {step:3d}: {elapsed_ms:.2f} ms")

    # ----------------------------------------------------------------
    # Speculative: async delta-only reads each step
    # ----------------------------------------------------------------
    print("\n--- Speculative: async delta reads (concurrent with FFN) ---")
    prefetcher.reset()
    spec_step_ms: List[float] = []

    # Simulate the decode loop
    for step in range(args.decode_steps):
        t0 = time.perf_counter()

        # Simulate each CSA layer running sequentially (like the real model)
        for lid in layer_ids:
            topk_set = traces[lid][step]
            # This is what _on_real_topk does: compute delta + submit reads
            prefetcher._on_real_topk(lid, _set_to_tensor(topk_set), step + 1)

        # Simulate FFN_window = 3300 µs of "other compute" per layer
        # In reality the reads are concurrent with this; here we just wait
        # for whatever is still pending after the loop to measure overlap.
        t_after_submit = time.perf_counter()

        # Drain pending reads (should be near-zero wait if IO < FFN window)
        n_resolved = prefetcher.wait_all(timeout=30.0)

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1e3
        submit_ms = (t_after_submit - t0) * 1e3
        spec_step_ms.append(elapsed_ms)

        if step % 10 == 0:
            total_delta = sum(
                len(traces[lid][step] - (traces[lid][step - 1]
                    if step > 0 else set()))
                for lid in layer_ids
            )
            print(f"  step {step:3d}: submit={submit_ms:.2f} ms  "
                  f"total(+drain)={elapsed_ms:.2f} ms  "
                  f"delta_blocks={total_delta}")

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    prefetcher.print_stats()

    bl_mean = float(np.mean(baseline_step_ms))
    bl_p99  = float(np.percentile(baseline_step_ms, 99))
    sp_mean = float(np.mean(spec_step_ms))
    sp_p99  = float(np.percentile(spec_step_ms, 99))
    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"  Baseline  (sync full topk):  mean={bl_mean:.2f} ms  "
          f"p99={bl_p99:.2f} ms")
    print(f"  Speculative (async delta):   mean={sp_mean:.2f} ms  "
          f"p99={sp_p99:.2f} ms")
    print(f"  Speedup: {bl_mean / sp_mean:.1f}×  "
          f"(I/O overlap if FFN≥{bl_mean:.1f} ms → near-zero visible latency)")
    print()
    print(f"  n_layers={args.n_layers}  topk={args.index_topk}  "
          f"delta_rate={args.delta_rate:.0%}")
    print(f"  block_size={args.block_size} B  "
          f"delta_bytes_per_step="
          f"{int(args.n_layers * args.index_topk * args.delta_rate * args.block_size / 1024)} KB")
    print(f"  store: {store_dir}")

    store.close()


def _set_to_tensor(s: Set[int]) -> torch.Tensor:
    return torch.tensor(sorted(s), dtype=torch.int32).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Real model mode
# ---------------------------------------------------------------------------


def bench_real() -> None:
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed)

    # ── Load model ────────────────────────────────────────────────────
    inference_dir = str(Path(args.model_path) / "inference")
    sys.path.insert(0, inference_dir)

    import model as model_module
    model_module.world_size = world_size
    model_module.rank = rank

    from model import ModelArgs, Transformer
    from safetensors.torch import load_model

    config_path = Path(args.model_path) / "inference" / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    cfg["max_batch_size"] = 1
    cfg["max_seq_len"] = args.prompt_len + args.decode_steps + 64

    if rank == 0:
        print(f"Loading {args.model_path} ...")
    model_args  = ModelArgs(**cfg)
    transformer = Transformer(model_args)
    ckpt = Path(args.model_path) / f"model{rank}-mp{world_size}.safetensors"
    load_model(transformer, str(ckpt), strict=False)
    transformer.eval()
    if rank == 0:
        print("Model loaded.")

    # ── Set up prefetcher ─────────────────────────────────────────────
    from lmcache.v1.csa_prefetcher import build_from_model

    store_dir = args.store_dir or tempfile.mkdtemp(prefix="csa_real_bench_")
    prefetcher = build_from_model(
        transformer,
        store_dir=store_dir,
        compress_ratio=args.compress_ratio,
        kv_lora_rank=args.kv_lora_rank,
        max_seq_len=cfg["max_seq_len"],
        io_workers=args.io_workers,
        speculative=args.speculative,
    )

    if rank == 0:
        csa_lids = sorted(prefetcher._csa_layer_ids)
        print(f"CSA layers: {len(csa_lids)} — {csa_lids[:8]}...")
        block_bytes = args.compress_ratio * args.kv_lora_rank * 2
        print(f"block_size={block_bytes} B  "
              f"topk={args.index_topk}  "
              f"store={store_dir}")

    # Write mock block data (rank 0 only; others skip disk init)
    if rank == 0:
        print("Initializing block store with mock data ...")
        write_fake_blocks(
            prefetcher._store,
            sorted(prefetcher._csa_layer_ids),
            cfg["max_seq_len"] // args.compress_ratio,
            args.compress_ratio * args.kv_lora_rank * 2,
        )

    # Barrier: wait for rank 0 to finish writing
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    prefetcher.patch_transformer(transformer)
    prefetcher.reset()

    # ── Prefill ───────────────────────────────────────────────────────
    L = args.prompt_len
    prompt_ids = torch.randint(10, model_args.vocab_size - 10,
                               (1, L), device=f"cuda:{local_rank}")
    if rank == 0:
        print(f"\nPrefill {L} tokens ...")
    with torch.no_grad():
        logits = transformer.forward(prompt_ids, start_pos=0)
    next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    # Reset delta cache AFTER prefill so step 0 starts clean
    prefetcher.reset()

    if rank == 0:
        print(f"Decoding {args.decode_steps} steps ...")

    step_ms: List[float] = []
    for step in range(args.decode_steps):
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = transformer.forward(next_token, start_pos=L + step)
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        # Drain all pending reads before measuring (simulates worst-case)
        prefetcher.wait_all(timeout=30.0)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        step_ms.append(elapsed_ms)

    prefetcher.unpatch_transformer()

    if rank == 0:
        prefetcher.print_stats()
        print("\n--- Step latency ---")
        arr = np.array(step_ms)
        print(f"  mean={arr.mean():.2f} ms  p50={np.median(arr):.2f} ms  "
              f"p99={np.percentile(arr, 99):.2f} ms")

    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()

    prefetcher._store.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if args.mode == "fake":
    bench_fake()
else:
    bench_real()
