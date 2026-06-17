"""
Measure CSA Indexer selection similarity across different prefill lengths.

Models the partial-KV-cache scenario in LMCache: user has first K tokens cached
(KV loaded from NVMe), then decodes immediately.  We compare decode-step-0
selections for K-token context against the full-L-token-context baseline.

NOTE: The model's Indexer does NOT support chunked prefill (start_pos>0,
seqlen>1), so we cannot simulate "prefill the remaining L-K tokens after
loading K tokens' KV."  Instead we measure the simpler but more useful
question: how similar is the selection when only K tokens of context are
available vs. the full L-token context?

For each K in [10%, 25%, 50%, 75%, 100%] of L:
  - forward(tokens[0:K], start_pos=0)
  - decode at start_pos=K, step 0 → selection S_K
Baseline S_full = S_K for K=L.
Report overlap(S_K, S_full) per layer (mean).
Also measure adj-step stability during decode for each K.

Run:
  torchrun --nproc_per_node=8 test_incremental_prefill.py \
      --model-path /mnt/nvme0/models/DeepSeek-V4-Pro \
      --prompt-len 32768 \
      --decode-steps 50
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
parser.add_argument("--decode-steps", type=int, default=50)
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
selections = defaultdict(list)
_orig_indexer_forward = Indexer.forward

def _patched_indexer_forward(self, x, qr, start_pos, offset):
    topk = _orig_indexer_forward(self, x, qr, start_pos, offset)
    if start_pos > 0:   # decode phase only (seqlen=1)
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

# ── Prompt (fixed across all runs) ───────────────────────────────────────────
L = args.prompt_len
prompt_ids = torch.randint(10, model_args.vocab_size - 10,
                           (1, L), device=f"cuda:{local_rank}")

# ── Helper ────────────────────────────────────────────────────────────────────
def run_case(K: int):
    """Prefill first K tokens, decode for decode_steps, return selections."""
    selections.clear()

    if K > 0:
        with torch.no_grad():
            logits = transformer.forward(prompt_ids[:, :K], start_pos=0)
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    else:
        # K=0: no context, start from a random token
        next_token = prompt_ids[:, :1]

    for step in range(args.decode_steps):
        with torch.no_grad():
            logits = transformer.forward(next_token, start_pos=K + step)
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    return {lid: list(sels) for lid, sels in selections.items()}

# ── Run each fraction ─────────────────────────────────────────────────────────
fractions = [1.0, 0.75, 0.5, 0.25, 0.1]
splits    = [L, int(0.75*L), L//2, L//4, int(0.1*L)]

case_sels = {}
for frac, K in zip(fractions, splits):
    if rank == 0:
        print(f"\nK={K} ({frac:.0%} of prompt) ...")
    case_sels[frac] = run_case(K)
    if rank == 0:
        print("  done.")

# ── Analysis (rank 0 only) ────────────────────────────────────────────────────
if rank != 0:
    if world_size > 1:
        dist.destroy_process_group()
    sys.exit(0)

baseline = case_sels[1.0]

print("\n" + "=" * 72)
print("Partial KV cache hit: decode selection similarity vs full-context baseline")
print(f"Model: {args.model_path}")
print(f"Full prompt={L}  decode_steps={args.decode_steps}  topk={cfg.get('index_topk',1024)}")
print("=" * 72)
print()
print(f"  {'K/L':>6}  {'K tokens':>10}  {'step0_sim':>12}  {'adj_stab':>12}")
print("  " + "-" * 48)

for frac, K in zip(fractions, splits):
    sels = case_sels[frac]
    step0_sims, adj_stabs = [], []

    for layer_id in sorted(baseline.keys()):
        b_steps = baseline.get(layer_id, [])
        c_steps = sels.get(layer_id, [])
        if not b_steps or not c_steps:
            continue
        b0, c0 = b_steps[0], c_steps[0]
        if b0:
            step0_sims.append(len(b0 & c0) / len(b0))
        for t in range(1, len(c_steps)):
            a, b = c_steps[t-1], c_steps[t]
            if a:
                adj_stabs.append(len(a & b) / len(a))

    m_sim = float(np.mean(step0_sims)) if step0_sims else float("nan")
    m_adj = float(np.mean(adj_stabs))  if adj_stabs  else float("nan")
    note  = "← baseline" if frac == 1.0 else ""
    print(f"  {frac:6.2f}  {K:10d}  {m_sim:12.3f}  {m_adj:12.3f}  {note}")

print()
print("  step0_sim: decode-step-0 selection overlap with full-context baseline")
print("  adj_stab:  mean adjacent-step overlap during decode for this context length")
print()
print("Per-layer step0_sim:")
print(f"  {'Layer':>6}" + "".join(f"  {f:>7.0%}" for f in fractions))
print("  " + "-" * (8 + 9*len(fractions)))
for layer_id in sorted(baseline.keys()):
    vals = []
    for frac in fractions:
        b0 = baseline.get(layer_id, [None])[0]
        c0 = case_sels[frac].get(layer_id, [None])[0]
        if b0 and c0:
            vals.append(len(b0 & c0) / len(b0))
        else:
            vals.append(float("nan"))
    print(f"  {layer_id:6d}" + "".join(f"  {v:7.3f}" for v in vals))

if world_size > 1:
    dist.destroy_process_group()
