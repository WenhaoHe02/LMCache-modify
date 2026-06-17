"""
Measure wall-clock time of operations relevant to speculative prefetch.

Measures (all on real model tensors, decode shape [1,1,d]):
  - dist.all_reduce (A2A window duration)
  - shared_experts forward
  - HC_pre + attn_norm  (proxy computation for attn+residual approach)
  - HC_post             (needed for hc_post(shared) approach)
  - HC_post + HC_pre + attn_norm  (full hc_post(shared) proxy pipeline)

Run:
  torchrun --nproc_per_node=8 bench_proxy_timing.py \\
      --model-path /mnt/nvme0/models/DeepSeek-V4-flash
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--warmup",     type=int, default=20)
parser.add_argument("--iters",      type=int, default=200)
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

from model import ModelArgs, Transformer, Block, MoE

config_path = Path(args.model_path) / "inference" / "config.json"
with open(config_path) as f:
    cfg = json.load(f)

cfg["max_batch_size"] = 1
cfg["max_seq_len"]    = 512

if rank == 0:
    print(f"Loading model ...")

model_args  = ModelArgs(**cfg)
transformer = Transformer(model_args)

ckpt_file = Path(args.model_path) / f"model{rank}-mp{world_size}.safetensors"
load_model(transformer, str(ckpt_file), strict=False)
transformer.eval()

# Pick a representative CSA block (middle of the network)
n_layers = len(transformer.layers)
mid = n_layers // 2
block: Block = transformer.layers[mid]
moe: MoE    = block.ffn
dim         = cfg["dim"]
hc_mult     = cfg["hc_mult"]

if rank == 0:
    print(f"Using layer {mid} for timing (dim={dim}, hc_mult={hc_mult})")

# ── Synthetic tensors (decode shape) ─────────────────────────────────────────
# HC state: [1, 1, hc_mult, dim]
x_hc = torch.randn(1, 1, hc_mult, dim, device=f"cuda:{local_rank}")
# FFN input (after hc_pre + ffn_norm): [1, 1, dim]
x_ffn = torch.randn(1, 1, dim, device=f"cuda:{local_rank}")
# all_reduce target: [1, dim]  (same shape as local expert accumulator)
x_ar  = torch.randn(1, dim, device=f"cuda:{local_rank}")


def timed(fn, warmup=args.warmup, iters=args.iters):
    """Return mean latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # ms → µs


with torch.no_grad():

    # 1. dist.all_reduce  (the A2A window)
    def fn_allreduce():
        y = x_ar.clone()
        dist.all_reduce(y)
    t_ar = timed(fn_allreduce)

    # 2. shared_experts forward  [1,1,dim] → [1,1,dim]
    def fn_shared():
        moe.shared_experts(x_ffn.view(-1, dim))
    t_shared = timed(fn_shared)

    # 3. HC_pre + attn_norm  (proxy computation for attn+residual approach)
    def fn_hcpre_norm():
        p, _, _ = block.hc_pre(x_hc, block.hc_attn_fn,
                                block.hc_attn_scale, block.hc_attn_base)
        block.attn_norm(p)
    t_hcpre_norm = timed(fn_hcpre_norm)

    # 4. HC_post alone
    residual_f = x_hc.clone()
    post_f = torch.randn(1, 1, hc_mult, device=f"cuda:{local_rank}")
    comb_f = torch.randn(1, 1, hc_mult, hc_mult, device=f"cuda:{local_rank}")
    sh_out = x_ffn.clone()
    def fn_hcpost():
        block.hc_post(sh_out, residual_f, post_f, comb_f)
    t_hcpost = timed(fn_hcpost)

    # 5. Full hc_post(shared) proxy pipeline:
    #    shared_experts + HC_post + HC_pre + attn_norm
    def fn_full_pipeline():
        sh = moe.shared_experts(x_ffn.view(-1, dim)).view(x_ffn.shape)
        hc = block.hc_post(sh, residual_f, post_f, comb_f)
        p, _, _ = block.hc_pre(hc, block.hc_attn_fn,
                                block.hc_attn_scale, block.hc_attn_base)
        block.attn_norm(p)
    t_full = timed(fn_full_pipeline)

    # 6. attn+residual proxy: HC_pre + attn_norm on existing HC state (no shared)
    def fn_attn_res_proxy():
        p, _, _ = block.hc_pre(x_hc, block.hc_attn_fn,
                                block.hc_attn_scale, block.hc_attn_base)
        block.attn_norm(p)
    t_attn_res = timed(fn_attn_res_proxy)

if rank != 0:
    if world_size > 1:
        dist.destroy_process_group()
    sys.exit(0)

print("\n" + "="*62)
print(f"Proxy timing benchmark  (TP={world_size}, decode seqlen=1)")
print(f"Layer {mid}, dim={dim}, hc_mult={hc_mult}, iters={args.iters}")
print("="*62)
print(f"  {'Operation':<40}  {'µs':>8}")
print("  " + "-"*52)
print(f"  {'dist.all_reduce (A2A window)':<40}  {t_ar:8.1f}")
print(f"  {'shared_experts forward':<40}  {t_shared:8.1f}")
print(f"  {'HC_pre + attn_norm':<40}  {t_hcpre_norm:8.1f}")
print(f"  {'HC_post':<40}  {t_hcpost:8.1f}")
print(f"  {'shared + HC_post + HC_pre + attn_norm':<40}  {t_full:8.1f}")
print(f"  {'attn+residual proxy (HC_pre + attn_norm)':<40}  {t_attn_res:8.1f}")
print("  " + "-"*52)

def fits(t_op, t_window, label):
    ratio = t_op / t_window
    verdict = "FITS" if ratio < 0.8 else ("TIGHT" if ratio < 1.0 else "TOO SLOW")
    print(f"  {label}: {t_op:.1f}µs / {t_window:.1f}µs A2A = {ratio:.2f}x  [{verdict}]")

print()
print("Feasibility (proxy must fit inside A2A window):")
fits(t_full,    t_ar, "hc_post(shared) full pipeline")
fits(t_attn_res, t_ar, "attn+residual proxy          ")

print()
print("Note: 'FITS' = <80% of A2A budget, leaving margin for NVMe issue overhead.")

if world_size > 1:
    dist.destroy_process_group()
