"""
Measure the total window available for NVMe prefetch.

From "attention L-1 output" to "Indexer L runs", the main stream executes:
  HC_post_attn(L-1) + HC_pre_ffn(L-1) + ffn_norm + local_experts + A2A
    + shared_experts + HC_post_ffn(L-1) + HC_pre_attn(L) + attn_norm(L)

If we run proxy on a SIDE STREAM starting right after attention L-1:
  proxy_time = HC_pre_attn(L) + attn_norm(L) = ~124µs
  NVMe read  = ~200-400µs (depends on block size and disk)
  Total side: ~320-520µs

If total_window > total_side, the read can complete before Indexer L.

Run:
  torchrun --nproc_per_node=8 bench_ffn_window.py \\
      --model-path /mnt/nvme0/models/DeepSeek-V4-flash
"""

import argparse
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.distributed as dist
from safetensors.torch import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--warmup",     type=int, default=10)
parser.add_argument("--iters",      type=int, default=50)
args = parser.parse_args()

rank       = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

if world_size > 1:
    dist.init_process_group("nccl")

torch.cuda.set_device(local_rank)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(f"cuda:{local_rank}")

inference_dir = str(Path(args.model_path) / "inference")
sys.path.insert(0, inference_dir)

import model as model_module
model_module.world_size = world_size
model_module.rank       = rank

from model import ModelArgs, Transformer, Block, Indexer, Attention

config_path = Path(args.model_path) / "inference" / "config.json"
with open(config_path) as f:
    cfg = json.load(f)

cfg["max_batch_size"] = 1
cfg["max_seq_len"]    = 512

if rank == 0:
    print("Loading model ...")

model_args  = ModelArgs(**cfg)
transformer = Transformer(model_args)
ckpt_file   = Path(args.model_path) / f"model{rank}-mp{world_size}.safetensors"
load_model(transformer, str(ckpt_file), strict=False)
transformer.eval()

dim     = cfg["dim"]
hc_mult = cfg["hc_mult"]

# ── Measure per-layer timing via event injection ──────────────────────────────
# We time the actual Block.forward with a real decode input token.
# Events are recorded around key points inside the patched forward.

_events = defaultdict(list)   # key -> list of (start_event, end_event)

_orig_block_forward = Block.forward
_orig_attn_forward  = Attention.forward


def _patched_block_forward(self, x, start_pos, input_ids):
    lid = self.layer_id
    # Time entire block
    s0 = torch.cuda.Event(enable_timing=True); e0 = torch.cuda.Event(enable_timing=True)
    s0.record()

    # Attention path
    residual = x
    x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)

    s_attn = torch.cuda.Event(enable_timing=True); e_attn = torch.cuda.Event(enable_timing=True)
    s_attn.record()
    x = self.attn_norm(x)
    x = self.attn(x, start_pos)
    e_attn.record()

    x = self.hc_post(x, residual, post, comb)

    # FFN path
    residual = x
    x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)

    s_ffn = torch.cuda.Event(enable_timing=True); e_ffn = torch.cuda.Event(enable_timing=True)
    s_ffn.record()
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    e_ffn.record()

    x = self.hc_post(x, residual, post, comb)
    e0.record()

    _events[("block", lid)].append((s0, e0))
    _events[("attn",  lid)].append((s_attn, e_attn))
    _events[("ffn",   lid)].append((s_ffn,  e_ffn))
    return x


Block.forward = _patched_block_forward

# ── Run decode steps ──────────────────────────────────────────────────────────
prompt_ids = torch.randint(10, model_args.vocab_size - 10,
                           (1, 32), device=f"cuda:{local_rank}")

with torch.no_grad():
    logits = transformer.forward(prompt_ids, start_pos=0)

next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

total_steps = args.warmup + args.iters
for step in range(total_steps):
    if step == args.warmup:
        _events.clear()   # discard warmup
    with torch.no_grad():
        logits = transformer.forward(next_token, start_pos=32 + step)
    next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

torch.cuda.synchronize()

if rank != 0:
    if world_size > 1:
        dist.destroy_process_group()
    sys.exit(0)

# ── Compute mean timings ──────────────────────────────────────────────────────
import numpy as np

def mean_ms(key):
    evs = _events.get(key, [])
    if not evs:
        return float("nan")
    return np.mean([s.elapsed_time(e) for s, e in evs])

n_layers = len(transformer.layers)

print("\n" + "="*70)
print(f"Per-layer timing  (TP={world_size}, decode seqlen=1, iters={args.iters})")
print("="*70)
print(f"  {'Layer':>6}  {'block_ms':>10}  {'attn_ms':>10}  {'ffn_ms':>10}  {'window_µs':>12}")
print("  " + "-"*58)

all_blocks = []; all_attn = []; all_ffn = []
for lid in range(n_layers):
    b = mean_ms(("block", lid)) * 1000
    a = mean_ms(("attn",  lid)) * 1000
    f = mean_ms(("ffn",   lid)) * 1000
    if not any(k == ("block", lid) for k in _events):
        continue
    # window = from end-of-attn to start-of-next-layer's-attn
    # approximated as: (block - attn) + next_layer_hc_pre_attn_norm
    # We'll just show block, attn, ffn and compute the window separately
    all_blocks.append(b); all_attn.append(a); all_ffn.append(f)
    # window = FFN of this layer (HC_post_attn + HC_pre_ffn + ffn_norm + ffn + HC_post_ffn)
    #        + HC_pre_attn of NEXT layer + attn_norm of NEXT layer
    # approximate: (block - attn) + hc_pre_attn of next ~ block - attn + 0.12ms
    win = (b - a + 0.124) * 1000  # µs; 124µs is HC_pre+attn_norm from bench_proxy_timing
    print(f"  Layer {lid:3d}  {b/1000:10.3f}  {a/1000:10.3f}  {f/1000:10.3f}  {win:12.1f}")

print("  " + "-"*58)
mb = np.mean(all_blocks); ma = np.mean(all_attn); mf = np.mean(all_ffn)
mwin = (mb - ma + 0.124) * 1000
print(f"  {'MEAN':>6}  {mb/1000:10.3f}  {ma/1000:10.3f}  {mf/1000:10.3f}  {mwin:12.1f}")

print()
proxy_us  = 123.8   # HC_pre + attn_norm from bench_proxy_timing
nvme_us   = 300.0   # typical NVMe read latency (conservative)
total_side = proxy_us + nvme_us
print(f"  Side-stream pipeline: proxy({proxy_us:.0f}µs) + NVMe({nvme_us:.0f}µs) = {total_side:.0f}µs")
print(f"  Available window (mean): {mwin:.1f}µs")
if mwin > total_side:
    print(f"  → FEASIBLE: window({mwin:.1f}µs) > side({total_side:.0f}µs)")
else:
    print(f"  → TOO TIGHT: window({mwin:.1f}µs) < side({total_side:.0f}µs)  "
          f"(need {total_side - mwin:.0f}µs more)")

if world_size > 1:
    dist.destroy_process_group()
