"""
Speculative prefetch: score-stratified precision analysis (decode phase).

torch.topk returns indices in descending score order, so spec_topk[0,0,:n]
is already the top-n highest-confidence speculative blocks.

Question: does precision (hit rate vs. true top-k) increase for higher-ranked
speculative blocks?  E.g., top-10% of 512 blocks may hit at 60-80% while
the full set hits at only ~27%.

Run:
  torchrun --nproc_per_node=8 test_speculative_score_precision.py \\
      --model-path /mnt/nvme0/models/DeepSeek-V4-flash \\
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
parser.add_argument("--model-path", required=True)
parser.add_argument("--prompt-len",   type=int, default=32768)
parser.add_argument("--decode-steps", type=int, default=50)
parser.add_argument("--seed",         type=int, default=42)
args = parser.parse_args()

# ── Distributed init ──────────────────────────────────────────────────────────
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

# ── Percentile thresholds to evaluate ────────────────────────────────────────
PCTS = [0.05, 0.10, 0.20, 0.25, 0.33, 0.50, 0.75, 1.00]

# layer_id -> list of (pct -> precision) dicts, one per decode step
# precision@pct = |spec_topk[:n] ∩ true_set| / n  where n = round(topk * pct)
per_layer_prec = defaultdict(list)   # layer_id -> list of dicts

_last_attn_out      = [None]
_orig_indexer_forward = Indexer.forward
_orig_attn_forward    = Attention.forward


def _patched_indexer_forward(self, x, qr, start_pos, offset):
    true_topk = _orig_indexer_forward(self, x, qr, start_pos, offset)

    if start_pos > 0:
        layer_id = getattr(self, "_layer_id", -1)
        h_approx = _last_attn_out[0]
        if h_approx is not None:
            comp  = self.compressor
            ratio = self.compress_ratio
            slot  = start_pos // ratio

            # Save compressor state mutated by the speculative call
            saved_kv_slot = comp.kv_cache[:, slot].clone() if comp.kv_cache is not None else None
            saved_kv_st   = comp.kv_state.clone()    if hasattr(comp, "kv_state")    else None
            saved_sc_st   = comp.score_state.clone() if hasattr(comp, "score_state")  else None

            with torch.no_grad():
                # Returns topk indices in DESCENDING score order (torch.topk default).
                spec_topk = _orig_indexer_forward(self, h_approx, qr, start_pos, offset)

            # Restore compressor state
            if saved_kv_slot is not None: comp.kv_cache[:, slot].copy_(saved_kv_slot)
            if saved_kv_st   is not None: comp.kv_state.copy_(saved_kv_st)
            if saved_sc_st   is not None: comp.score_state.copy_(saved_sc_st)

            # spec_topk / true_topk: [1, 1, topk] — decode seqlen=1
            topk     = spec_topk.shape[-1]
            true_set = set(true_topk[0, 0].cpu().tolist())
            spec_list = spec_topk[0, 0].cpu().tolist()   # already sorted by score desc

            step_prec = {}
            for pct in PCTS:
                n    = max(1, round(topk * pct))
                hits = sum(1 for idx in spec_list[:n] if idx in true_set)
                step_prec[pct] = hits / n
            per_layer_prec[layer_id].append(step_prec)

    return true_topk


def _patched_attn_forward(self, x, start_pos):
    out = _orig_attn_forward(self, x, start_pos)
    _last_attn_out[0] = out.detach()
    return out


Indexer.forward   = _patched_indexer_forward
Attention.forward = _patched_attn_forward

# ── Load model ────────────────────────────────────────────────────────────────
config_path = Path(args.model_path) / "inference" / "config.json"
with open(config_path) as f:
    cfg = json.load(f)

cfg["max_batch_size"] = 1
cfg["max_seq_len"]    = max(cfg.get("max_seq_len", 163840),
                            args.prompt_len + args.decode_steps + 64)

if rank == 0:
    print(f"Loading model from {args.model_path} ...")

model_args  = ModelArgs(**cfg)
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

# ── Prefill ───────────────────────────────────────────────────────────────────
prompt_ids = torch.randint(10, model_args.vocab_size - 10,
                           (1, args.prompt_len), device=f"cuda:{local_rank}")

if rank == 0:
    print(f"Prefilling {args.prompt_len} tokens ...")

with torch.no_grad():
    logits = transformer.forward(prompt_ids, start_pos=0)

next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

# ── Decode ────────────────────────────────────────────────────────────────────
if rank == 0:
    print(f"Decoding {args.decode_steps} steps ...")

for step in range(args.decode_steps):
    with torch.no_grad():
        logits = transformer.forward(next_token,
                                     start_pos=args.prompt_len + step)
    next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    if rank == 0 and step % 10 == 9:
        print(f"  step {step+1}/{args.decode_steps}")

# ── Analysis (rank 0 only) ────────────────────────────────────────────────────
if rank != 0:
    if world_size > 1:
        dist.destroy_process_group()
    sys.exit(0)

print("\n" + "="*72)
print("Speculative Prefetch  Score-Stratified Precision")
print(f"Prompt: {args.prompt_len} tokens | Decode steps: {args.decode_steps}")
print(f"index_topk={cfg.get('index_topk', 512)}  "
      f"h_approx=prev-layer attn output (pre-MoE)")
print("="*72)

if not per_layer_prec:
    print("[ERROR] No data — check Indexer was patched and decode ran.")
    sys.exit(1)

# Aggregate over all layers × all steps
agg = {pct: [] for pct in PCTS}
for layer_id in sorted(per_layer_prec.keys()):
    steps = per_layer_prec[layer_id]
    layer_means = {pct: float(np.mean([s[pct] for s in steps])) for pct in PCTS}
    row = "  ".join(f"{layer_means[p]:.3f}" for p in PCTS)
    pct_labels = "  ".join(f"top{int(p*100):3d}%" for p in PCTS)
    if layer_id == sorted(per_layer_prec.keys())[0]:
        print(f"\n  {'Layer':>6}  {pct_labels}")
        print("  " + "-"*70)
    print(f"  Layer {layer_id:3d}  {row}")
    for pct in PCTS:
        agg[pct].extend([s[pct] for s in steps])

print("\n  " + "-"*70)
row = "  ".join(f"{float(np.mean(agg[p])):6.3f}" for p in PCTS)
print(f"  {'ALL':>6}  {row}")

print("\n")
index_topk = cfg.get("index_topk", 512)
print(f"  Precision@top-k% of {index_topk} spec blocks:")
for pct in PCTS:
    n_blocks  = max(1, round(index_topk * pct))
    mean_prec = float(np.mean(agg[pct]))
    n_hits    = round(n_blocks * mean_prec)
    print(f"    top {int(pct*100):3d}%  ({n_blocks:4d} blocks) → "
          f"precision={mean_prec:.3f}  expected correct blocks ~{n_hits}")

print()
print("Interpretation:")
print("  If precision rises steeply toward top-5/10%, the spec signal is reliable")
print("  for the highest-confidence blocks — prefetch those, skip the rest.")
print("  If precision is flat (~27% everywhere), score rank carries no information.")

if world_size > 1:
    dist.destroy_process_group()
