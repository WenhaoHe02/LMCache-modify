"""
Check whether per-layer proxy accuracy is content-dependent or structural.
Runs K/L=1.0 with multiple random seeds and dumps per-layer model attributes.

The key question: are low-accuracy layers (2, 8, 24, 32, 54, 56) always low
regardless of prompt content, or is this an artifact of seed=42?

Also inspects per-layer HC attributes to diagnose root cause:
  - hc_attn_fn type (does it change between adjacent CSA layers?)
  - hc_attn_scale, hc_attn_base
  - Whether adjacent layers have mismatched HC function types
    (proxy uses layer L's hc_attn_fn on layer L-1's residual — mismatch = error)

Run:
  torchrun --nproc_per_node=8 test_layer_proxy_consistency.py \\
      --model-path /mnt/nvme0/models/DeepSeek-V4-Pro \\
      --prompt-len 32768 --decode-steps 30 \\
      --seeds 42,0,1,123,456
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
parser.add_argument("--decode-steps", type=int, default=30)
parser.add_argument("--seeds",        default="42,0,1,123,456")
args = parser.parse_args()
seeds = [int(s) for s in args.seeds.split(",")]

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

from model import ModelArgs, Transformer, Indexer, Block

# ── Patching (same as test_proxy_accuracy_vs_prefill.py) ─────────────────────
_residual_f  = [None]
_block_proxy = defaultdict(lambda: None)
_step_acc    = defaultdict(list)
_orig_block_forward   = Block.forward
_orig_indexer_forward = Indexer.forward


def _run_spec(indexer, proxy, qr, start_pos, offset):
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
            _step_acc[layer_id].append(acc)
    return true_topk


def _patched_block_forward(self, x, start_pos, input_ids):
    lid = self.layer_id
    residual_a = x
    x_a, post_a, comb_a = self.hc_pre(x, self.hc_attn_fn,
                                       self.hc_attn_scale, self.hc_attn_base)
    prev_rf = _residual_f[0]
    if prev_rf is not None and start_pos > 0:
        with torch.no_grad():
            p, _, _ = self.hc_pre(prev_rf, self.hc_attn_fn,
                                   self.hc_attn_scale, self.hc_attn_base)
            _block_proxy[lid] = self.attn_norm(p)
    else:
        _block_proxy[lid] = None
    x_a = self.attn_norm(x_a)
    x_a = self.attn(x_a, start_pos)
    x   = self.hc_post(x_a, residual_a, post_a, comb_a)
    residual_f = x
    x_f, post_f, comb_f = self.hc_pre(x, self.hc_ffn_fn,
                                       self.hc_ffn_scale, self.hc_ffn_base)
    x_f = self.ffn_norm(x_f)
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

# ── Dump per-layer attributes (rank 0 only) ───────────────────────────────────
if rank == 0:
    # Find all CSA layer IDs (those that have an Indexer sub-module)
    csa_lids = set()
    for name, m in transformer.named_modules():
        if isinstance(m, Indexer):
            parts = name.split(".")
            try:
                csa_lids.add(int(parts[parts.index("layers") + 1]))
            except (ValueError, IndexError):
                pass

    # Map layer_id -> Block
    block_by_lid = {}
    for blk in transformer.layers:
        if hasattr(blk, "layer_id"):
            block_by_lid[blk.layer_id] = blk

    print("\n--- Per-CSA-layer model attributes ---")
    print("Hypothesis: proxy uses layer L's hc_attn_fn on layer L-1's residual.")
    print("If fn type changes between L-1 and L, the proxy applies wrong transform → low accuracy.\n")
    print(f"  {'Layer':>6}  {'hc_attn_fn':>24}  {'scale':>8}  {'base':>8}  {'fn_chg':>7}")
    print("  " + "-"*62)

    prev_fn = None
    layer_attrs = {}   # lid -> (fn_type, scale, base, fn_changed)
    for lid in sorted(csa_lids):
        blk = block_by_lid.get(lid)
        if blk is None:
            continue
        fn_name = type(getattr(blk, "hc_attn_fn", None)).__name__
        scale   = getattr(blk, "hc_attn_scale", "?")
        base    = getattr(blk, "hc_attn_base",  "?")
        changed = (prev_fn is not None and fn_name != prev_fn)
        layer_attrs[lid] = (fn_name, scale, base, changed)
        flag = "  YES" if changed else ""
        print(f"  {lid:6d}  {fn_name:>24}  {str(scale):>8}  {str(base):>8}{flag}")
        prev_fn = fn_name

    # Also check hc_ffn_fn for each CSA layer (the fn used on the FFN path of L-1)
    print()
    print("--- Layer L-1 hc_ffn_fn vs Layer L hc_attn_fn (cross-path mismatch) ---")
    print(f"  {'L-1':>6}  {'L-1 hc_ffn_fn':>24}    {'L':>6}  {'L hc_attn_fn':>24}  {'match':>6}")
    print("  " + "-"*70)
    csa_sorted = sorted(csa_lids)
    for i, lid in enumerate(csa_sorted):
        blk = block_by_lid.get(lid)
        if blk is None:
            continue
        attn_fn = type(getattr(blk, "hc_attn_fn", None)).__name__
        # Find previous block (any layer, not just CSA)
        prev_lid = lid - 1
        prev_blk = block_by_lid.get(prev_lid)
        if prev_blk is None:
            print(f"  {'N/A':>6}  {'N/A':>24}    {lid:>6}  {attn_fn:>24}")
            continue
        ffn_fn = type(getattr(prev_blk, "hc_ffn_fn", None)).__name__
        match  = "OK" if ffn_fn == attn_fn else "MISMATCH"
        print(f"  {prev_lid:>6}  {ffn_fn:>24}    {lid:>6}  {attn_fn:>24}  {match:>6}")

# ── Multiple seed runs ────────────────────────────────────────────────────────
L = args.prompt_len
seed_results = {}

for seed in seeds:
    torch.manual_seed(seed)
    _step_acc.clear()
    _residual_f[0] = None

    if rank == 0:
        print(f"\nSeed={seed} — prefilling {L} tokens ...")

    prompt_ids = torch.randint(10, model_args.vocab_size - 10,
                               (1, L), device=f"cuda:{local_rank}")
    with torch.no_grad():
        logits = transformer.forward(prompt_ids, start_pos=0)
    next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    for step in range(args.decode_steps):
        with torch.no_grad():
            logits = transformer.forward(next_token, start_pos=L + step)
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    seed_results[seed] = {
        lid: float(np.mean(v)) if v else float("nan")
        for lid, v in _step_acc.items()
    }
    if rank == 0:
        print(f"  done.")

# ── Report ────────────────────────────────────────────────────────────────────
if rank != 0:
    if world_size > 1:
        dist.destroy_process_group()
    sys.exit(0)

all_lids = sorted(seed_results[seeds[0]].keys()) if seeds else []

print("\n" + "=" * 80)
print("Per-layer proxy accuracy across random seeds (K/L=100%)")
print(f"Model: {args.model_path}")
print(f"prompt_len={L}  decode_steps={args.decode_steps}")
print("=" * 80)
print(f"  {'Layer':>6}" + "".join(f"  {'s='+str(s):>8}" for s in seeds)
      + f"  {'mean':>8}  {'std':>7}  note")
print("  " + "-" * (8 + 10 * len(seeds) + 24))

for lid in all_lids:
    vals = [seed_results[s].get(lid, float("nan")) for s in seeds]
    m  = float(np.nanmean(vals))
    sd = float(np.nanstd(vals))
    if m < 0.5:
        note = "STRUCTURAL-LOW" if sd < 0.05 else "VARIABLE-LOW"
    elif m < 0.75:
        note = "structural-med" if sd < 0.05 else "variable-med"
    elif m < 0.88:
        note = ""
    else:
        note = ""
    row = (f"  {lid:6d}"
           + "".join(f"  {v:8.3f}" for v in vals)
           + f"  {m:8.3f}  {sd:7.4f}  {note}")
    print(row)

print()
print("Interpretation:")
print("  std < 0.02 → structural: same architecture every run, not prompt-specific")
print("  std > 0.05 → content-dependent: varies with input")
print()
print("If a low-accuracy layer also shows MISMATCH in the cross-path table above,")
print("root cause is: proxy applies layer L's hc_attn_fn to layer L-1's residual,")
print("but L-1's residual was built with a different hc_ffn_fn type.")

if world_size > 1:
    dist.destroy_process_group()
