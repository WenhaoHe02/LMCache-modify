"""
Speculative prefetch accuracy using HC_post(shared_out) + HC_pre + attn_norm as proxy.

Pipeline (during layer L-1's A2A window):
  1. shared_out = shared_experts(ffn_norm_input)  ← before A2A
  2. hc_approx  = HC_post(shared_out, residual_f, post_f, comb_f)  ← during A2A
  3. proxy_x    = attn_norm_L( HC_pre_L(hc_approx) )               ← during A2A
  4. Issue NVMe prefetch using Indexer(proxy_x)                     ← during A2A

Compare against h_shared baseline (using shared_out directly, no HC transforms).

Run:
  torchrun --nproc_per_node=8 test_speculative_hcpost_proxy.py \\
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

from model import ModelArgs, Transformer, Indexer, Block

# ── Global proxy state ────────────────────────────────────────────────────────
# All set by Block L-1, read at the start of Block L's attention path.
_hc_approx_ffn  = [None]   # [b,s,hc,d]  HC_post(shared_out, residual_f, post_f, comb_f)
_hc_approx_attn = [None]   # [b,s,hc,d]  residual_f  (HC state after attn, before FFN)
_last_shared    = [None]   # [b,s,d]      raw shared_out (baseline)

# Computed at the start of Block L's attn path from the HC state proxies above.
_block_proxy_ffn  = [None]  # [b,s,d]  attn_norm( HC_pre( hc_approx_ffn  ) )
_block_proxy_attn = [None]  # [b,s,d]  attn_norm( HC_pre( hc_approx_attn ) )
_block_proxy_raw  = [None]  # [b,s,d]  attn_norm( raw_shared )

per_layer_acc = defaultdict(list)   # layer_id -> [{"ffn": f, "attn_res": f, "raw": f}, ...]

_orig_indexer_forward = Indexer.forward
_orig_block_forward   = Block.forward


def _run_spec(indexer, proxy, qr, start_pos, offset):
    """Run Indexer speculatively; restore compressor state."""
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

    if start_pos > 0:
        layer_id = getattr(self, "_layer_id", -1)
        true_set = set(true_topk[0, 0].cpu().tolist())
        step_acc = {}
        for name, proxy in [("ffn",      _block_proxy_ffn[0]),
                             ("attn_res", _block_proxy_attn[0]),
                             ("raw",      _block_proxy_raw[0])]:
            if proxy is None:
                continue
            spec = _run_spec(self, proxy, qr, start_pos, offset)
            s = set(spec[0, 0].cpu().tolist())
            step_acc[name] = len(true_set & s) / len(true_set) if true_set else 0.0
        if step_acc:
            per_layer_acc[layer_id].append(step_acc)

    return true_topk


def _patched_block_forward(self, x, start_pos, input_ids):
    # ── ATTENTION PATH ────────────────────────────────────────────────────────
    residual_a = x
    x_a, post_a, comb_a = self.hc_pre(x, self.hc_attn_fn,
                                       self.hc_attn_scale, self.hc_attn_base)

    # Compute proxies BEFORE attn runs (and before Indexer inside it).
    hc_ffn_prev  = _hc_approx_ffn[0]
    hc_attn_prev = _hc_approx_attn[0]
    raw_prev     = _last_shared[0]

    with torch.no_grad():
        if hc_ffn_prev is not None:
            p, _, _ = self.hc_pre(hc_ffn_prev, self.hc_attn_fn,
                                   self.hc_attn_scale, self.hc_attn_base)
            _block_proxy_ffn[0] = self.attn_norm(p)
        else:
            _block_proxy_ffn[0] = None

        if hc_attn_prev is not None:
            p, _, _ = self.hc_pre(hc_attn_prev, self.hc_attn_fn,
                                   self.hc_attn_scale, self.hc_attn_base)
            _block_proxy_attn[0] = self.attn_norm(p)
        else:
            _block_proxy_attn[0] = None

        if raw_prev is not None:
            _block_proxy_raw[0] = self.attn_norm(raw_prev)
        else:
            _block_proxy_raw[0] = None

    x_a = self.attn_norm(x_a)
    x_a = self.attn(x_a, start_pos)          # Indexer runs here
    x   = self.hc_post(x_a, residual_a, post_a, comb_a)

    # ── FFN PATH ──────────────────────────────────────────────────────────────
    residual_f = x
    x_f, post_f, comb_f = self.hc_pre(x, self.hc_ffn_fn,
                                       self.hc_ffn_scale, self.hc_ffn_base)
    x_f = self.ffn_norm(x_f)

    # Skip proxy computation during prefill (seqlen>1) to save memory.
    # Proxies are only meaningful and used during decode (seqlen==1).
    if x_f.size(1) == 1:
        with torch.no_grad():
            sh = self.ffn.shared_experts(
                x_f.view(-1, self.ffn.dim)
            ).view(x_f.shape).detach()
        _hc_approx_attn[0] = residual_f.detach()
        with torch.no_grad():
            _hc_approx_ffn[0] = self.hc_post(sh, residual_f, post_f, comb_f).detach()
            _last_shared[0]   = sh
    else:
        _hc_approx_attn[0] = None
        _hc_approx_ffn[0]  = None
        _last_shared[0]    = None

    # Real FFN + HC_post
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

ckpt_file = Path(args.model_path) / f"model{rank}-mp{world_size}.safetensors"
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
print("HC_post proxy vs raw-shared baseline  (decode phase, seqlen=1)")
print(f"Prompt: {args.prompt_len} tokens  |  Decode steps: {args.decode_steps}")
print(f"index_topk={cfg.get('index_topk', 512)}")
print("="*72)
print(f"  {'Layer':>6}  {'hc_post(shared)':>17}  {'attn+residual':>15}  {'raw_shared':>12}")
print("  " + "-"*56)

if not per_layer_acc:
    print("[ERROR] No data.")
    sys.exit(1)

agg = defaultdict(list)
for layer_id in sorted(per_layer_acc.keys()):
    steps = per_layer_acc[layer_id]
    ffn_m  = float(np.mean([s["ffn"]      for s in steps if "ffn"      in s]))
    attn_m = float(np.mean([s["attn_res"] for s in steps if "attn_res" in s]))
    raw_m  = float(np.mean([s["raw"]      for s in steps if "raw"      in s]))
    agg["ffn"].append(ffn_m)
    agg["attn"].append(attn_m)
    agg["raw"].append(raw_m)
    print(f"  Layer {layer_id:3d}  {ffn_m:17.3f}  {attn_m:15.3f}  {raw_m:12.3f}")

print("  " + "-"*56)
ffn_all  = float(np.mean(agg["ffn"]))
attn_all = float(np.mean(agg["attn"]))
raw_all  = float(np.mean(agg["raw"]))
print(f"  {'ALL':>6}  {ffn_all:17.3f}  {attn_all:15.3f}  {raw_all:12.3f}")

print()
topk = cfg.get("index_topk", 512)
print(f"  hc_post(shared): {ffn_all:.1%}  → ~{round(topk*ffn_all)}/{topk} correct  [available after A2A starts]")
print(f"  attn+residual  : {attn_all:.1%}  → ~{round(topk*attn_all)}/{topk} correct  [available right after attn]")
print(f"  raw_shared     : {raw_all:.1%}  → ~{round(topk*raw_all)}/{topk} correct")
print(f"  delta-selection: ~90.5%  (from previous test)")
print()
print("Timing:")
print("  attn+residual  = residual_f (HC_post after attn) → available during ENTIRE FFN")
print("  hc_post(shared)= HC_post(shared_out, ...)        → available during A2A only")

if world_size > 1:
    dist.destroy_process_group()
