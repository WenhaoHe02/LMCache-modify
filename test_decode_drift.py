"""
Measure how CSA Indexer selections drift over decode steps.

Two metrics per step T:
  adj_overlap[T]  = |sel[T] ∩ sel[T-1]| / |sel[T-1]|   (adjacent, already in stability test)
  long_overlap[T] = |sel[T] ∩ sel[0]|   / |sel[0]|      (drift from initial decode state)

Run once per prompt length (separate torchrun invocations for different lengths,
since running multiple lengths in one process causes OOM from hc_post intermediates).

Run:
  torchrun --nproc_per_node=8 test_decode_drift.py \
      --model-path /mnt/nvme0/models/DeepSeek-V4-Pro \
      --prompt-len 32768 --decode-steps 200
"""

import argparse
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--model-path",   required=True)
parser.add_argument("--prompt-len",   type=int, default=32768)
parser.add_argument("--decode-steps", type=int, default=200)
parser.add_argument("--seed",         type=int, default=42)
args = parser.parse_args()

rank       = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

if world_size > 1:
    dist.init_process_group("nccl")

torch.cuda.set_device(local_rank)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(f"cuda:{local_rank}")
torch.manual_seed(args.seed)

inference_dir = str(Path(args.model_path) / "inference")
sys.path.insert(0, inference_dir)

import model as model_module
model_module.world_size = world_size
model_module.rank       = rank

from model import ModelArgs, Transformer, Indexer

# ── Patch ─────────────────────────────────────────────────────────────────────
# selections[layer_id] = list of sets, one per decode step (in order)
selections = defaultdict(list)
_orig_indexer_forward = Indexer.forward

def _patched_indexer_forward(self, x, qr, start_pos, offset):
    topk = _orig_indexer_forward(self, x, qr, start_pos, offset)
    if start_pos > 0:
        layer_id = getattr(self, "_layer_id", -1)
        selections[layer_id].append(set(topk[0, 0].cpu().tolist()))
    return topk

Indexer.forward = _patched_indexer_forward

# ── Load model ────────────────────────────────────────────────────────────────
config_path = Path(args.model_path) / "inference" / "config.json"
with open(config_path) as f:
    cfg = json.load(f)

cfg["max_batch_size"] = 1
cfg["max_seq_len"]    = args.prompt_len + args.decode_steps + 64

if rank == 0:
    print(f"Loading model from {args.model_path} ...")

model_args  = ModelArgs(**cfg)
transformer = Transformer(model_args)
ckpt_file   = Path(args.model_path) / f"model{rank}-mp{world_size}.safetensors"
if not ckpt_file.exists():
    raise FileNotFoundError(str(ckpt_file))
load_model(transformer, str(ckpt_file), strict=False)
transformer.eval()

for name, module in transformer.named_modules():
    if isinstance(module, Indexer):
        parts = name.split(".")
        try:
            layer_id = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            layer_id = -1
        module._layer_id = layer_id

if rank == 0:
    print("Model loaded.")

# ── Prefill ───────────────────────────────────────────────────────────────────
prompt_len = args.prompt_len

if rank == 0:
    print(f"Prefilling {prompt_len} tokens ...")

prompt_ids = torch.randint(10, model_args.vocab_size - 10,
                           (1, prompt_len), device=f"cuda:{local_rank}")

with torch.no_grad():
    logits = transformer.forward(prompt_ids, start_pos=0)

next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
selections.clear()   # discard prefill-phase selections (seqlen>1)

# ── Decode ────────────────────────────────────────────────────────────────────
if rank == 0:
    print(f"Decoding {args.decode_steps} steps ...")

for step in range(args.decode_steps):
    with torch.no_grad():
        logits = transformer.forward(next_token, start_pos=prompt_len + step)
    next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    if rank == 0 and step % 50 == 49:
        print(f"  step {step+1}/{args.decode_steps}")

# ── Compute drift metrics ─────────────────────────────────────────────────────
if rank == 0:
    n_steps = args.decode_steps
    adj_per_step  = [[] for _ in range(n_steps)]
    long_per_step = [[] for _ in range(n_steps)]

    for layer_id, steps in selections.items():
        if len(steps) < 2:
            continue
        sel0 = steps[0]
        for t, sel_t in enumerate(steps):
            if t == 0:
                long_per_step[t].append(1.0)
                continue
            long = len(sel_t & sel0) / len(sel0) if sel0 else 0.0
            long_per_step[t].append(long)
            sel_prev = steps[t - 1]
            adj = len(sel_t & sel_prev) / len(sel_prev) if sel_prev else 0.0
            adj_per_step[t].append(adj)

    results = {
        "adj":  [float(np.mean(v)) if v else float("nan") for v in adj_per_step],
        "long": [float(np.mean(v)) if v else float("nan") for v in long_per_step],
    }

# ── Report ────────────────────────────────────────────────────────────────────
if rank == 0:
    print("\n" + "=" * 72)
    print("Decode selection drift")
    print(f"Model: {args.model_path}")
    print(f"prompt_len={prompt_len}  index_topk={cfg.get('index_topk', 1024)}  decode_steps={args.decode_steps}")
    print("=" * 72)

    horizons = [1, 5, 10, 25, 50, 100, 200]
    horizons = [h for h in horizons if h <= args.decode_steps]

    print(f"\n  {'Step':>6}  {'adj_overlap':>12}  {'long_overlap':>14}")
    print("  " + "-" * 38)
    for h in horizons:
        adj  = results["adj"][h]  if h < len(results["adj"])  else float("nan")
        long = results["long"][h] if h < len(results["long"]) else float("nan")
        print(f"  {h:6d}  {adj:12.3f}  {long:14.3f}")

    print()
    print("adj_overlap:  |sel[T] ∩ sel[T-1]| / |sel[T-1]|  (step-to-step stability)")
    print("long_overlap: |sel[T] ∩ sel[0]|   / |sel[0]|    (drift from initial decode state)")

if world_size > 1:
    dist.destroy_process_group()
