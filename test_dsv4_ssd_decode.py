"""
test_dsv4_ssd_decode.py — SSD naive vs SSD spec-prefetch latency.

Measures both incremental prefill and decode phases.  Both regimes use
SSD-offloaded kv_cache and the same CSASSDPoolManager pool-scoring path.
The only difference is I/O scheduling:

  Naive  (use_spec=False):  blocking pread of missing blocks inside
                            pool_score_fn — I/O stalls the forward pass.

  Spec   (use_spec=True):   same blocks, but async-read during the previous
                            layer's MoE FFN window (~3300 µs) so I/O is
                            overlapped with compute.

The overlap condition is start_pos > 0 (prev_topk available), which covers:
  - Incremental prefill (chunk_size > 1, longer FFN window → more overlap)
  - Decode (chunk_size = 1)
  - Full-reuse prefill

Run (TP=8):
    torchrun --nproc_per_node=8 test_dsv4_ssd_decode.py \\
        --model-path /mnt/nvme0/models/DeepSeek-V4-Pro \\
        --store-dir  /mnt/nvme0/csa_ssd_pool_bench \\
        --prompt-len 32768 --chunk-size 128 --incr-chunks 16 \\
        --decode-steps 50 --io-workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model-path",     required=True)
parser.add_argument("--store-dir",      required=True)
parser.add_argument("--prompt-len",     type=int, default=32768)
parser.add_argument("--chunk-size",     type=int, default=128,
                    help="Tokens per incremental-prefill chunk (> 1 to test "
                         "prefill overlap; use 1 to skip prefill phase)")
parser.add_argument("--incr-chunks",    type=int, default=16,
                    help="Number of incremental-prefill chunks to benchmark")
parser.add_argument("--decode-steps",   type=int, default=50)
parser.add_argument("--io-workers",     type=int, default=8)
parser.add_argument("--pool-size",      type=int, default=2048)
parser.add_argument("--compress-ratio", type=int, default=4)
parser.add_argument("--kv-lora-rank",   type=int, default=512)
parser.add_argument("--seed",           type=int, default=42)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

rank       = int(os.environ.get("RANK",       0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

if world_size > 1:
    import torch.distributed as dist
    dist.init_process_group("nccl")

torch.cuda.set_device(local_rank)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(f"cuda:{local_rank}")
torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

inference_dir = str(Path(args.model_path) / "inference")
sys.path.insert(0, inference_dir)
import model as model_module
model_module.world_size = world_size
model_module.rank = rank

from model import ModelArgs, Transformer
from safetensors.torch import load_model

with open(Path(args.model_path) / "inference" / "config.json") as f:
    cfg = json.load(f)

incr_len = args.chunk_size * args.incr_chunks
cfg["max_batch_size"] = 1
cfg["max_seq_len"] = args.prompt_len + incr_len * 2 + args.decode_steps * 2 + 64

if rank == 0:
    print(f"[rank 0] Loading {args.model_path} ...")

model_args  = ModelArgs(**cfg)
transformer = Transformer(model_args)
load_model(transformer, str(
    Path(args.model_path) / f"model{rank}-mp{world_size}.safetensors"), strict=False)
transformer.eval()
if rank == 0:
    print("[rank 0] Model loaded.")

# ---------------------------------------------------------------------------
# Import pool manager
# ---------------------------------------------------------------------------

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from lmcache.v1.csa_ssd_pool import CSASSDPoolManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def warmup() -> None:
    ids = torch.randint(10, model_args.vocab_size - 10,
                        (1, 16), device=f"cuda:{local_rank}")
    with torch.no_grad():
        transformer.forward(ids, start_pos=0)
    torch.cuda.synchronize()


def incr_prefill_loop(
    n_chunks: int,
    chunk_size: int,
    start_pos: int,
) -> Tuple[List[float], int]:
    """Run incremental prefill and return per-chunk latencies (ms) and end_pos."""
    chunk_ms: List[float] = []
    pos = start_pos
    for _ in range(n_chunks):
        ids = torch.randint(10, model_args.vocab_size - 10,
                            (1, chunk_size), device=f"cuda:{local_rank}")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            transformer.forward(ids, start_pos=pos)
        torch.cuda.synchronize()
        chunk_ms.append((time.perf_counter() - t0) * 1e3)
        pos += chunk_size
    return chunk_ms, pos


def decode_loop(
    n_steps: int,
    start_pos: int,
    first_token: torch.Tensor,
) -> Tuple[List[float], torch.Tensor]:
    token = first_token
    step_ms: List[float] = []
    for s in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = transformer.forward(token, start_pos=start_pos + s)
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1e3)
        token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    return step_ms, token


# ---------------------------------------------------------------------------
# Phase 1: Initial prefill
# ---------------------------------------------------------------------------

L  = args.prompt_len
C  = args.chunk_size
NC = args.incr_chunks
D  = args.decode_steps

if rank == 0:
    print(f"\n[rank 0] Phase 1 — Initial prefill {L} tokens ...")

warmup()
prompt_ids = torch.randint(10, model_args.vocab_size - 10,
                            (1, L), device=f"cuda:{local_rank}")
with torch.no_grad():
    logits = transformer.forward(prompt_ids, start_pos=0)
torch.cuda.synchronize()
first_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
if rank == 0:
    print("[rank 0] Initial prefill done.")

# ---------------------------------------------------------------------------
# Evict kv_cache to SSD (shared for both modes)
# ---------------------------------------------------------------------------

if world_size > 1:
    dist.barrier()

if rank == 0:
    print(f"\n[rank 0] Evicting kv_cache to SSD ...")

pool_mgr_spec = CSASSDPoolManager.build(
    transformer,
    store_dir=args.store_dir,
    compress_ratio=args.compress_ratio,
    kv_lora_rank=args.kv_lora_rank,
    max_seq_len=cfg["max_seq_len"],
    io_workers=args.io_workers,
    pool_size=args.pool_size,
    use_spec=True,
)
pool_mgr_spec.evict_after_prefill(transformer)

if world_size > 1:
    dist.barrier()

if rank == 0:
    n_csa = len(pool_mgr_spec._csa_layer_ids)
    print(f"  {n_csa} CSA layers evicted. store={args.store_dir}")

# ---------------------------------------------------------------------------
# Phase 2: Naive — blocking pread (use_spec=False)
# ---------------------------------------------------------------------------

if rank == 0:
    print(f"\n[rank 0] Phase 2 — NAIVE  "
          f"incr-prefill {NC}×{C} tok + decode {D} steps ...")

pool_mgr_naive = CSASSDPoolManager.build(
    transformer,
    store_dir=args.store_dir,
    compress_ratio=args.compress_ratio,
    kv_lora_rank=args.kv_lora_rank,
    max_seq_len=cfg["max_seq_len"],
    io_workers=args.io_workers,
    pool_size=args.pool_size,
    use_spec=False,
)
pool_mgr_naive.evict_after_prefill(transformer)
pool_mgr_naive.patch_transformer(transformer)
pool_mgr_naive.reset()

naive_prefill_ms, pos_after_naive_prefill = incr_prefill_loop(NC, C, L)
naive_decode_ms, naive_last_token = decode_loop(D, pos_after_naive_prefill,
                                                 first_token)

pool_mgr_naive.wait_all(timeout=5.0)
pool_mgr_naive.unpatch_transformer()

if rank == 0:
    np_ = np.array(naive_prefill_ms)
    nd  = np.array(naive_decode_ms)
    print(f"  prefill chunk: mean={np_.mean():.2f}  p50={np.median(np_):.2f}  "
          f"p95={np.percentile(np_,95):.2f}  (ms/{C}-tok chunk)")
    print(f"  decode step:   mean={nd.mean():.2f}  p50={np.median(nd):.2f}  "
          f"p95={np.percentile(nd,95):.2f}  (ms/step)")
    pool_mgr_naive.print_stats()

# ---------------------------------------------------------------------------
# Phase 3: Spec — async reads overlapped with MoE FFN (use_spec=True)
# ---------------------------------------------------------------------------

if rank == 0:
    print(f"\n[rank 0] Phase 3 — SPEC   "
          f"incr-prefill {NC}×{C} tok + decode {D} steps ...")

pos_spec_start = pos_after_naive_prefill + D
pool_mgr_spec.patch_transformer(transformer)
pool_mgr_spec.reset()

spec_prefill_ms, pos_after_spec_prefill = incr_prefill_loop(
    NC, C, pos_spec_start)
spec_decode_ms, _ = decode_loop(D, pos_after_spec_prefill, naive_last_token)

pool_mgr_spec.wait_all(timeout=30.0)
pool_mgr_spec.unpatch_transformer()

if rank == 0:
    sp_ = np.array(spec_prefill_ms)
    sd  = np.array(spec_decode_ms)
    print(f"  prefill chunk: mean={sp_.mean():.2f}  p50={np.median(sp_):.2f}  "
          f"p95={np.percentile(sp_,95):.2f}  (ms/{C}-tok chunk)")
    print(f"  decode step:   mean={sd.mean():.2f}  p50={np.median(sd):.2f}  "
          f"p95={np.percentile(sd,95):.2f}  (ms/step)")
    pool_mgr_spec.print_stats()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

if rank == 0:
    print("\n" + "=" * 68)
    print("Summary  (both: SSD-offloaded kv_cache + CSASSDPoolManager)")
    print("=" * 68)
    print(f"  prompt_len={L}  chunk_size={C}  incr_chunks={NC}  "
          f"decode_steps={D}  pool_size={args.pool_size}")
    print()
    print(f"  {'':30s}  {'naive':>10s}  {'spec':>10s}  {'speedup':>8s}")
    print(f"  {'-'*62}")

    pf_speedup = np_.mean() / sp_.mean()
    dc_speedup = nd.mean()  / sd.mean()
    print(f"  {'incr-prefill (ms/chunk)':30s}  "
          f"{np_.mean():>10.2f}  {sp_.mean():>10.2f}  {pf_speedup:>7.2f}x")
    print(f"  {'decode (ms/step)':30s}  "
          f"{nd.mean():>10.2f}  {sd.mean():>10.2f}  {dc_speedup:>7.2f}x")
    print(f"\n  store: {args.store_dir}")

pool_mgr_spec._store.close()
if world_size > 1:
    dist.destroy_process_group()
