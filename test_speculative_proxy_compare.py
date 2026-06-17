"""
Compare three proxy signals for speculative prefetch accuracy (decode phase).

For each CSA layer L, measures how well three proxies from layer L-1
predict layer L's Indexer block selection:

  h_attn:   attention OUTPUT of layer L-1 (before MoE)      — already tested ~26%
  h_shared: shared-expert-only output of layer L-1           — available before/during A2A
  h_moe:    full MoE output of layer L-1 (allreduce+shared)  — available after A2A

Execution order inside Block L-1:
  ... → Attention L-1 → h_attn saved
        → gate + local_experts → all_reduce → shared_expert → MoE done → h_moe saved
             [A2A window here]  only h_attn + h_shared available
  → HC_post → Layer L-1 complete

Run:
  torchrun --nproc_per_node=8 test_speculative_proxy_compare.py \\
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

from model import ModelArgs, Transformer, Indexer, Attention, MoE

# ── Per-layer accuracy storage ────────────────────────────────────────────────
# layer_id -> list of dicts {"attn": float, "shared": float, "moe": float}
per_layer_acc = defaultdict(list)

# Proxy state: hold previous layer's signals
_last_attn_out   = [None]   # [b, s, d]  attention output of previous layer
_last_shared_out = [None]   # [b, s, d]  shared-expert-only of previous layer
_last_moe_out    = [None]   # [b, s, d]  full MoE output of previous layer

_orig_indexer_forward = Indexer.forward
_orig_attn_forward    = Attention.forward
_orig_moe_forward     = MoE.forward


def _run_spec_and_restore(indexer, proxy, qr, start_pos, offset):
    """Run Indexer speculatively with proxy input; restore compressor state."""
    comp  = indexer.compressor
    ratio = indexer.compress_ratio
    slot  = start_pos // ratio

    saved_kv_slot = comp.kv_cache[:, slot].clone() if comp.kv_cache is not None else None
    saved_kv_st   = comp.kv_state.clone()    if hasattr(comp, "kv_state")    else None
    saved_sc_st   = comp.score_state.clone() if hasattr(comp, "score_state")  else None

    with torch.no_grad():
        topk = _orig_indexer_forward(indexer, proxy, qr, start_pos, offset)

    if saved_kv_slot is not None: comp.kv_cache[:, slot].copy_(saved_kv_slot)
    if saved_kv_st   is not None: comp.kv_state.copy_(saved_kv_st)
    if saved_sc_st   is not None: comp.score_state.copy_(saved_sc_st)
    return topk


def _patched_indexer_forward(self, x, qr, start_pos, offset):
    true_topk = _orig_indexer_forward(self, x, qr, start_pos, offset)

    if start_pos > 0:
        layer_id  = getattr(self, "_layer_id", -1)
        true_set  = set(true_topk[0, 0].cpu().tolist())

        step_acc = {}
        for name, proxy in [("attn",   _last_attn_out[0]),
                             ("shared", _last_shared_out[0]),
                             ("moe",    _last_moe_out[0])]:
            if proxy is None:
                continue
            spec = _run_spec_and_restore(self, proxy, qr, start_pos, offset)
            spec_set = set(spec[0, 0].cpu().tolist())
            step_acc[name] = len(true_set & spec_set) / len(true_set) if true_set else 0.0

        if step_acc:
            per_layer_acc[layer_id].append(step_acc)

    return true_topk


def _patched_attn_forward(self, x, start_pos):
    out = _orig_attn_forward(self, x, start_pos)
    _last_attn_out[0] = out.detach()
    return out


def _patched_moe_forward(self, x, input_ids):
    # Capture shared-expert-only output (runs on un-reduced x, independent of routing)
    with torch.no_grad():
        sh = self.shared_experts(x.view(-1, self.dim)).view(x.shape)
    _last_shared_out[0] = sh.detach()

    result = _orig_moe_forward(self, x, input_ids)
    _last_moe_out[0] = result.detach()
    return result


Indexer.forward   = _patched_indexer_forward
Attention.forward = _patched_attn_forward
MoE.forward       = _patched_moe_forward

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
        f"{ckpt_file} not found.\n"
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

# ── Analysis ──────────────────────────────────────────────────────────────────
if rank != 0:
    if world_size > 1:
        dist.destroy_process_group()
    sys.exit(0)

print("\n" + "="*72)
print("Speculative Proxy Comparison  (decode phase, seqlen=1)")
print(f"Prompt: {args.prompt_len} tokens  |  Decode steps: {args.decode_steps}")
print(f"index_topk={cfg.get('index_topk', 512)}  "
      f"pool={args.prompt_len // 4} compressed blocks")
print("="*72)
print(f"  {'Layer':>6}  {'h_attn':>8}  {'h_shared':>10}  {'h_moe':>8}")
print("  " + "-"*44)

if not per_layer_acc:
    print("[ERROR] No data recorded.")
    sys.exit(1)

agg = defaultdict(list)
for layer_id in sorted(per_layer_acc.keys()):
    steps = per_layer_acc[layer_id]
    means = {k: float(np.mean([s[k] for s in steps if k in s]))
             for k in ("attn", "shared", "moe")}
    for k, v in means.items():
        agg[k].append(v)
    print(f"  Layer {layer_id:3d}  {means['attn']:8.3f}  {means.get('shared', float('nan')):10.3f}  {means.get('moe', float('nan')):8.3f}")

print("  " + "-"*44)
for k in ("attn", "shared", "moe"):
    vals = agg[k]
    if vals:
        print(f"  {'ALL':>6}  [{k:>6}]  mean={np.mean(vals):.3f}  "
              f"min={np.min(vals):.3f}  max={np.max(vals):.3f}")

print()
print("Timing availability:")
print("  h_attn  : after Attention, BEFORE A2A  → can prefetch DURING A2A (short window ~1ms)")
print("  h_shared: shared expert runs AFTER A2A → available at block end, too late for same-A2A")
print("  h_moe   : full MoE output, after A2A   → available at block end")
print()
print("If h_moe >> h_attn: MoE transformation carries critical routing signal")
print("If h_attn ≈ h_moe:  attention already captures most of the useful signal")

if world_size > 1:
    dist.destroy_process_group()
