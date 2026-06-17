"""
Measure attn+residual proxy accuracy across different prefill lengths and
decode steps.

For each K in [10%, 25%, 50%, 75%, 100%] of prompt_len:
  - Prefill first K tokens
  - During decode steps 1..decode_steps:
      proxy_x  = attn_norm_L( HC_pre_L( residual_f_{L-1} ) )
      spec_topk = Indexer(proxy_x)   [speculative, state restored]
      true_topk = Indexer(x)         [real]
      accuracy  = |spec_topk ∩ true_topk| / |true_topk|
  - Report: mean accuracy per K, per decode-step horizon

Also tracks accuracy vs decode step index to see if proxy degrades over time.

Run:
  torchrun --nproc_per_node=8 test_proxy_accuracy_vs_prefill.py \\
      --model-path /mnt/nvme0/models/DeepSeek-V4-Pro \\
      --prompt-len 32768 --decode-steps 50
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

from model import ModelArgs, Transformer, Indexer, Block

# ── Global proxy state (set by Block L-1, read at Block L) ───────────────────
_residual_f = [None]   # HC state right after attention: HC_post(attn_out, residual_a, ...)

# Per-layer per-step accuracy accumulator (only populated during active run)
# layer_id -> list of per-step accuracy floats
_step_acc = defaultdict(list)
_current_step = [0]

_orig_block_forward   = Block.forward
_orig_indexer_forward = Indexer.forward


def _run_spec(indexer, proxy, qr, start_pos, offset):
    """Run Indexer speculatively; restore compressor state afterwards."""
    comp  = indexer.compressor
    ratio = indexer.compress_ratio
    slot  = start_pos // ratio
    saved_kv = comp.kv_cache[:, slot].clone() if comp.kv_cache is not None else None
    saved_ks = comp.kv_state.clone()    if hasattr(comp, "kv_state")    else None
    saved_ss = comp.score_state.clone() if hasattr(comp, "score_state")  else None
    with torch.no_grad():
        topk = _orig_indexer_forward(indexer, proxy, qr, start_pos, offset)
    if saved_kv is not None: comp.kv_cache[:, slot].copy_(saved_kv)
    if saved_ks is not None: comp.kv_state.copy_(saved_ks)
    if saved_ss is not None: comp.score_state.copy_(saved_ss)
    return topk


def _patched_indexer_forward(self, x, qr, start_pos, offset):
    true_topk = _orig_indexer_forward(self, x, qr, start_pos, offset)

    if start_pos > 0 and _residual_f[0] is not None:
        layer_id = getattr(self, "_layer_id", -1)
        proxy = _block_proxy[layer_id]
        if proxy is not None:
            spec = _run_spec(self, proxy, qr, start_pos, offset)
            true_set = set(true_topk[0, 0].cpu().tolist())
            spec_set = set(spec[0, 0].cpu().tolist())
            acc = len(true_set & spec_set) / len(true_set) if true_set else 0.0
            _step_acc[layer_id].append((_current_step[0], acc))

    return true_topk


# Per-layer proxy tensors (computed in Block.forward before Indexer runs)
_block_proxy = defaultdict(lambda: None)


def _patched_block_forward(self, x, start_pos, input_ids):
    lid = self.layer_id

    # ── Attention path ────────────────────────────────────────────────────────
    residual_a = x
    x_a, post_a, comb_a = self.hc_pre(x, self.hc_attn_fn,
                                       self.hc_attn_scale, self.hc_attn_base)

    # Compute proxy for THIS layer from prev layer's residual_f
    prev_rf = _residual_f[0]
    if prev_rf is not None and start_pos > 0:
        with torch.no_grad():
            p, _, _ = self.hc_pre(prev_rf, self.hc_attn_fn,
                                   self.hc_attn_scale, self.hc_attn_base)
            _block_proxy[lid] = self.attn_norm(p)
    else:
        _block_proxy[lid] = None

    x_a = self.attn_norm(x_a)
    x_a = self.attn(x_a, start_pos)          # patched Indexer fires here
    x   = self.hc_post(x_a, residual_a, post_a, comb_a)

    # ── FFN path ──────────────────────────────────────────────────────────────
    residual_f = x
    x_f, post_f, comb_f = self.hc_pre(x, self.hc_ffn_fn,
                                       self.hc_ffn_scale, self.hc_ffn_base)
    x_f = self.ffn_norm(x_f)

    # Store residual_f for next layer's proxy (decode only to save memory)
    if x_f.size(1) == 1:
        _residual_f[0] = residual_f.detach()
    else:
        _residual_f[0] = None

    x = self.ffn(x_f, input_ids)
    x = self.hc_post(x, residual_f, post_f, comb_f)
    return x


Block.forward   = _patched_block_forward
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
            lid = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            lid = -1
        module._layer_id = lid

if rank == 0:
    print("Model loaded.")

# ── Prompt ────────────────────────────────────────────────────────────────────
L = args.prompt_len
prompt_ids = torch.randint(10, model_args.vocab_size - 10,
                           (1, L), device=f"cuda:{local_rank}")

# ── Cases ─────────────────────────────────────────────────────────────────────
fractions = [1.0, 0.75, 0.5, 0.25, 0.1]
splits    = [L, int(0.75*L), L//2, L//4, int(0.1*L)]

# results[frac] = {layer_id: [(step, acc), ...]}
results = {}

for frac, K in zip(fractions, splits):
    if rank == 0:
        print(f"\nK={K} ({frac:.0%} of prompt) ...")

    _step_acc.clear()
    _residual_f[0] = None

    # Prefill
    if K > 0:
        with torch.no_grad():
            logits = transformer.forward(prompt_ids[:, :K], start_pos=0)
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    else:
        next_token = prompt_ids[:, :1]

    # Decode
    for step in range(args.decode_steps):
        _current_step[0] = step
        with torch.no_grad():
            logits = transformer.forward(next_token, start_pos=K + step)
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    results[frac] = {lid: list(v) for lid, v in _step_acc.items()}
    if rank == 0:
        print("  done.")

# ── Analysis (rank 0 only) ────────────────────────────────────────────────────
if rank != 0:
    if world_size > 1:
        dist.destroy_process_group()
    sys.exit(0)

topk = cfg.get("index_topk", 1024)

print("\n" + "=" * 72)
print("attn+residual proxy accuracy vs partial KV cache hit ratio")
print(f"Model: {args.model_path}")
print(f"prompt_len={L}  decode_steps={args.decode_steps}  index_topk={topk}")
print("=" * 72)

# ── Summary: mean accuracy per fraction ──────────────────────────────────────
print()
print(f"  {'K/L':>6}  {'K tokens':>10}  {'mean_acc':>10}  {'~correct':>10}  {'adj_note'}")
print("  " + "-" * 58)
for frac, K in zip(fractions, splits):
    all_acc = [acc for pairs in results[frac].values() for _, acc in pairs]
    m = float(np.mean(all_acc)) if all_acc else float("nan")
    note = "← baseline" if frac == 1.0 else ""
    print(f"  {frac:6.2f}  {K:10d}  {m:10.3f}  {round(topk*m):>10d}  {note}")

# ── Accuracy vs decode step horizon ──────────────────────────────────────────
horizons = [0, 4, 9, 24, 49]
horizons = [h for h in horizons if h < args.decode_steps]

print()
print("Mean proxy accuracy by decode step and K/L fraction:")
header = f"  {'Step':>6}" + "".join(f"  {f:>7.0%}" for f in fractions)
print(header)
print("  " + "-" * (8 + 9*len(fractions)))

for h in horizons:
    row = f"  {h+1:6d}"
    for frac in fractions:
        accs = [acc for step, acc in results[frac].get(
                    next(iter(results[frac]), -1), []) if step == h]
        # Average across all layers at this step
        all_at_h = [acc for lid_pairs in results[frac].values()
                    for step, acc in lid_pairs if step == h]
        m = float(np.mean(all_at_h)) if all_at_h else float("nan")
        row += f"  {m:7.3f}"
    print(row)

# ── Per-layer mean accuracy at K/L=1.0 vs K/L=0.5 ───────────────────────────
print()
print("Per-layer mean accuracy (K/L=100% vs 75% vs 50%):")
print(f"  {'Layer':>6}  {'100%':>8}  {'75%':>8}  {'50%':>8}")
print("  " + "-" * 36)
all_lids = sorted(set(results[1.0].keys()) & set(results[0.75].keys()) & set(results[0.5].keys()))
for lid in all_lids:
    def m(frac):
        v = [acc for _, acc in results[frac].get(lid, [])]
        return float(np.mean(v)) if v else float("nan")
    print(f"  {lid:6d}  {m(1.0):8.3f}  {m(0.75):8.3f}  {m(0.5):8.3f}")

if world_size > 1:
    dist.destroy_process_group()
