"""
Test speculative prefetch accuracy for CSA layers.

The question: if we use h_attn (hidden state AFTER attention, BEFORE FFN)
to speculatively compute q_L and run Lightning Indexer, how many of the
selected blocks match the blocks selected using the TRUE q_L (after FFN)?

High accuracy (~90%+) => speculative prefetch is viable: issue NVMe reads
during the previous layer's A2A window, only fallback-fetch the ~10% misses.

Run:
  torchrun --nproc_per_node=8 test_speculative_prefetch_accuracy.py \
      --model-path /mnt/nvme0/models/DeepSeek-V4-flash \
      --prompt-len 32768 --decode-steps 20
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
import torch.nn as nn
from safetensors.torch import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--prompt-len", type=int, default=32768)
parser.add_argument("--decode-steps", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ── Distributed init ─────────────────────────────────────────────────────────
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

from model import ModelArgs, Transformer, Indexer, Attention

# ── Patch: intercept true selection AND run speculative selection ─────────────
#
# Measures per-query-position speculative prefetch accuracy during PREFILL.
# h_approx = Attention OUTPUT of the previous layer (after L-1's attention,
# before L-1's MoE/FFN) — shape [b, seqlen, d].
#
# Execution order:
#   Attention L-1 runs → Indexer L-1 runs (sees _last_attn_out from L-2)
#     → Attention L-1 ends → _last_attn_out = L-1's attention output
#   Attention L   runs → Indexer L   runs (sees _last_attn_out = L-1's output) ✓
#
# Per-position accuracy: for each query position q,
#   acc[q] = |spec_topk[q] ∩ true_topk[q]| / index_topk
# then average over all positions.

# layer_id -> list of per-position-mean accuracy floats (one per prefill call)
per_layer_acc = defaultdict(list)

_last_attn_out = [None]   # attention OUTPUT of previous layer: [b, s, d]

_orig_indexer_forward = Indexer.forward
_orig_attn_forward    = Attention.forward

def _per_position_accuracy(true_topk, spec_topk):
    """Compute mean per-query-position top-k hit rate on GPU, return as float.

    true_topk / spec_topk: [seqlen, topk] int tensors.
    For each position q: |spec[q] ∩ true[q]| / topk.
    Uses searchsorted to avoid scatter_ index constraints.
    """
    _, topk = true_topk.shape
    # Sort true set per position for binary search
    true_sorted = true_topk.long().sort(dim=-1).values   # [seqlen, topk]
    spec_long   = spec_topk.long()                        # [seqlen, topk]
    # For each spec element, find its insertion point in the sorted true set
    pos = torch.searchsorted(true_sorted.contiguous(), spec_long.contiguous())
    pos = pos.clamp(0, topk - 1)
    # A spec element is a hit if true_sorted[q, pos[q,k]] == spec[q,k]
    matches = (true_sorted.gather(1, pos) == spec_long).float()  # [seqlen, topk]
    return matches.mean().item()

def _patched_indexer_forward(self, x, qr, start_pos, offset):
    true_topk = _orig_indexer_forward(self, x, qr, start_pos, offset)

    if start_pos > 0:  # decode only: seqlen=1, one token per step
        layer_id = getattr(self, "_layer_id", -1)
        h_approx = _last_attn_out[0]
        if h_approx is not None:
            # Save only the slot the compressor will write (start_pos // compress_ratio)
            comp = self.compressor
            ratio   = self.compress_ratio
            slot    = start_pos // ratio
            saved_kv_slot = comp.kv_cache[:, slot].clone() if comp.kv_cache is not None else None
            saved_kv_st   = comp.kv_state.clone()    if hasattr(comp, "kv_state")    else None
            saved_sc_st   = comp.score_state.clone() if hasattr(comp, "score_state")  else None
            with torch.no_grad():
                spec_topk = _orig_indexer_forward(self, h_approx, qr, start_pos, offset)
            if saved_kv_slot is not None: comp.kv_cache[:, slot].copy_(saved_kv_slot)
            if saved_kv_st   is not None: comp.kv_state.copy_(saved_kv_st)
            if saved_sc_st   is not None: comp.score_state.copy_(saved_sc_st)
            # seqlen=1 in decode → true_topk/spec_topk: [1, 1, topk]
            true_set = set(true_topk[0, 0].cpu().tolist())
            spec_set = set(spec_topk[0, 0].cpu().tolist())
            acc = len(true_set & spec_set) / len(true_set) if true_set else 0.0
            per_layer_acc[layer_id].append(acc)

    return true_topk

def _patched_attn_forward(self, x, start_pos):
    out = _orig_attn_forward(self, x, start_pos)
    # Save attention OUTPUT after the call: next layer's Indexer sees this as h_approx.
    _last_attn_out[0] = out.detach()
    return out

Indexer.forward   = _patched_indexer_forward
Attention.forward = _patched_attn_forward


# ── Load model ───────────────────────────────────────────────────────────────
config_path = Path(args.model_path) / "inference" / "config.json"
with open(config_path) as f:
    cfg = json.load(f)

cfg["max_batch_size"] = 1
cfg["max_seq_len"] = max(cfg.get("max_seq_len", 163840),
                         args.prompt_len + args.decode_steps + 64)

if rank == 0:
    print(f"Loading model from {args.model_path} ...")

model_args = ModelArgs(**cfg)
transformer = Transformer(model_args)

ckpt_file = Path(args.model_path) / f"model{rank}-mp{world_size}.safetensors"
if not ckpt_file.exists():
    raise FileNotFoundError(
        f"{ckpt_file} not found. Run convert.py first:\n"
        f"  python inference/convert.py --hf-ckpt-path {args.model_path} "
        f"--save-path {args.model_path} --n-experts {cfg['n_routed_experts']} "
        f"--model-parallel {world_size}"
    )

load_model(transformer, str(ckpt_file), strict=False)
transformer.eval()

# Tag Indexers with their layer id
for name, module in transformer.named_modules():
    if isinstance(module, Indexer):
        parts = name.split(".")
        try:
            layer_id = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            layer_id = -1
        module._layer_id = layer_id

if rank == 0:
    print("Model loaded and patched.")

# ── Synthetic prompt ──────────────────────────────────────────────────────────
prompt_ids = torch.randint(10, model_args.vocab_size - 10,
                           (1, args.prompt_len), device=f"cuda:{local_rank}")

if rank == 0:
    print(f"Prefilling {args.prompt_len} tokens ...")

with torch.no_grad():
    logits = transformer.forward(prompt_ids, start_pos=0)

next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

if rank == 0:
    print(f"Decoding {args.decode_steps} steps ...")

for step in range(args.decode_steps):
    with torch.no_grad():
        logits = transformer.forward(next_token,
                                     start_pos=args.prompt_len + step)
    next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    if rank == 0 and step % 10 == 9:
        print(f"  step {step+1}/{args.decode_steps}")

# ── Analysis ─────────────────────────────────────────────────────────────────
if rank != 0:
    if world_size > 1:
        dist.destroy_process_group()
    sys.exit(0)

print("\n" + "="*60)
print("Speculative Prefetch Accuracy  (per query position)")
print(f"Prompt: {args.prompt_len} tokens | h_approx: prev layer attn output (pre-MoE)")
print(f"Compressed block pool: {args.prompt_len // 4}")
print("="*60)
print("(accuracy = mean per-position top-k hit rate, averaged over all query positions)")
print()

if not per_layer_acc:
    print("[ERROR] No accuracy recorded — check that prefill ran and Indexer was patched.")
    sys.exit(1)

all_accs = []

for layer_id in sorted(per_layer_acc.keys()):
    accs = per_layer_acc[layer_id]
    m = float(np.mean(accs))
    all_accs.append(m)
    print(f"  Layer {layer_id:3d} (CSA): spec_accuracy={m:.3f}")

if all_accs:
    print("-"*60)
    overall = float(np.mean(all_accs))
    print(f"  ALL CSA: mean={overall:.3f}  min={min(all_accs):.3f}  max={max(all_accs):.3f}")

    index_topk = cfg.get("index_topk", 512)
    mean_miss = 1 - overall
    print()
    print(f"  Expected fallback reads per CSA layer per token: "
          f"~{mean_miss * index_topk:.0f} / {index_topk} blocks ({mean_miss*100:.1f}%)")
    print()
    print("Interpretation:")
    print("  >90% → speculative prefetch viable, fallback overhead small")
    print("  70-90% → viable but ~10-30% blocks need fallback reads")
    print("  <70% → speculation too inaccurate, wait for true q_L")

if world_size > 1:
    dist.destroy_process_group()
