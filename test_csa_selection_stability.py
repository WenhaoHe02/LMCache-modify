# SPDX-License-Identifier: Apache-2.0
"""Measure DeepSeek-V4 CSA Indexer selection reuse.

Run with torchrun against converted TP shards:

    torchrun --nproc_per_node=8 test_csa_selection_stability.py \
        --model-path /mnt/nvme0/models/DeepSeek-V4-flash \
        --prompt-len 32768 --decode-steps 50 \
        --json-summary /tmp/csa_selection_summary.json \
        --dump-selection-jsonl /tmp/csa_selection_blocks.jsonl

The script measures two reuse axes for CSA layers:

1. Cross-step reuse: how much layer L's selected block set at decode step t
   overlaps with the same layer at step t-1.
2. Cross-layer reuse: how much layer L's selected block set overlaps with
   layer L+1 at the same decode step.

High cross-step reuse supports delta-selection across decode steps. High
cross-layer numeric block-id reuse can inform a unified HBM residency manager,
but it does not imply that layer-specific KV payloads are physically shared.
"""

from __future__ import annotations

# Standard
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Third Party
import numpy as np


SelectionMap = dict[int, list[set[int]]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CSA selection experiment.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-len", type=int, default=32768)
    parser.add_argument("--decode-steps", type=int, default=300)
    parser.add_argument("--segment-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json-summary",
        type=Path,
        default=None,
        help="Optional path for a machine-readable metrics summary.",
    )
    parser.add_argument(
        "--dump-selection-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional path for raw per-step, per-layer selected block IDs. "
            "This can be large: decode_steps * CSA_layers JSONL records."
        ),
    )
    return parser.parse_args()


def overlap_ratio(a: set[int], b: set[int]) -> float:
    """Return ``|a intersection b| / |a|``.

    Args:
        a: Denominator block set.
        b: Compared block set.

    Returns:
        Overlap ratio. Returns 0.0 when ``a`` is empty.
    """
    if not a:
        return 0.0
    return len(a & b) / len(a)


def jaccard_ratio(a: set[int], b: set[int]) -> float:
    """Return Jaccard similarity for two block sets.

    Args:
        a: First block set.
        b: Second block set.

    Returns:
        ``|a intersection b| / |a union b|``. Returns 0.0 for two empty sets.
    """
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def summarize_values(values: list[float]) -> dict[str, float]:
    """Summarize a list of float values.

    Args:
        values: Numeric samples.

    Returns:
        Dictionary with count, mean, std, min, p50, p90, and max.
    """
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def topk_to_block_set(topk_idxs: Any) -> set[int]:
    """Convert an Indexer top-k tensor into a Python set of block IDs.

    Args:
        topk_idxs: Raw tensor returned by ``Indexer.forward``.

    Returns:
        Selected compressed-block IDs for the first batch item/query.
    """
    if topk_idxs.ndim >= 3:
        flat = topk_idxs[0, 0].reshape(-1)
    else:
        flat = topk_idxs.reshape(-1)
    return set(flat.detach().cpu().tolist())


def compute_cross_step(selections: SelectionMap) -> tuple[dict[int, dict[str, float]], list[float]]:
    """Compute same-layer overlap between adjacent decode steps.

    Args:
        selections: Mapping from CSA layer ID to per-step selected block sets.

    Returns:
        Pair of per-layer summaries and all overlap samples.
    """
    by_layer: dict[int, dict[str, float]] = {}
    all_overlaps: list[float] = []
    for layer_id in sorted(selections):
        steps = selections[layer_id]
        overlaps = [
            overlap_ratio(steps[t], steps[t + 1])
            for t in range(len(steps) - 1)
            if steps[t]
        ]
        if not overlaps:
            continue
        by_layer[layer_id] = summarize_values(overlaps)
        all_overlaps.extend(overlaps)
    return by_layer, all_overlaps


def compute_cross_layer(
    selections: SelectionMap,
) -> tuple[dict[str, dict[str, float]], list[float], list[float]]:
    """Compute adjacent-layer overlap for each decode step.

    Args:
        selections: Mapping from CSA layer ID to per-step selected block sets.

    Returns:
        Tuple of per-pair summaries, all directed overlap samples, and all
        Jaccard samples.
    """
    layer_ids = sorted(selections)
    by_pair: dict[str, dict[str, float]] = {}
    all_directed: list[float] = []
    all_jaccard: list[float] = []

    for left, right in zip(layer_ids, layer_ids[1:]):
        n_steps = min(len(selections[left]), len(selections[right]))
        directed = [
            overlap_ratio(selections[left][step], selections[right][step])
            for step in range(n_steps)
        ]
        jaccards = [
            jaccard_ratio(selections[left][step], selections[right][step])
            for step in range(n_steps)
        ]
        by_pair[f"{left}->{right}"] = {
            **{f"overlap_{k}": v for k, v in summarize_values(directed).items()},
            **{f"jaccard_{k}": v for k, v in summarize_values(jaccards).items()},
        }
        all_directed.extend(directed)
        all_jaccard.extend(jaccards)

    return by_pair, all_directed, all_jaccard


def compute_unique_pool(selections: SelectionMap) -> dict[str, float]:
    """Estimate all-CSA-layer HBM pool reuse at each decode step.

    Args:
        selections: Mapping from CSA layer ID to per-step selected block sets.

    Returns:
        Summary metrics for unique block counts and naive/unique reuse factor.
    """
    if not selections:
        return {}
    layer_ids = sorted(selections)
    n_steps = min(len(selections[layer_id]) for layer_id in layer_ids)
    unique_counts: list[float] = []
    naive_counts: list[float] = []
    reuse_factors: list[float] = []
    unique_ratios: list[float] = []

    for step in range(n_steps):
        step_sets = [selections[layer_id][step] for layer_id in layer_ids]
        naive = sum(len(blocks) for blocks in step_sets)
        unique = len(set().union(*step_sets)) if step_sets else 0
        unique_counts.append(float(unique))
        naive_counts.append(float(naive))
        if unique:
            reuse_factors.append(naive / unique)
        if naive:
            unique_ratios.append(unique / naive)

    return {
        "steps": float(n_steps),
        "layers": float(len(layer_ids)),
        "naive_blocks_mean": summarize_values(naive_counts)["mean"],
        "unique_blocks_mean": summarize_values(unique_counts)["mean"],
        "unique_blocks_min": summarize_values(unique_counts)["min"],
        "unique_blocks_max": summarize_values(unique_counts)["max"],
        "reuse_factor_mean": summarize_values(reuse_factors)["mean"],
        "unique_ratio_mean": summarize_values(unique_ratios)["mean"],
    }


def compute_random_baseline(
    n_blocks: int,
    topk_per_layer: int,
    n_layers: int,
) -> dict[str, float]:
    """Estimate reuse metrics for independent random top-k selections.

    Args:
        n_blocks: Number of compressed block IDs in the prompt pool.
        topk_per_layer: Number of block IDs selected by each layer.
        n_layers: Number of CSA layers.

    Returns:
        Approximate expected overlap, Jaccard, unique count, and reuse factor.
    """
    if n_blocks <= 0 or topk_per_layer <= 0 or n_layers <= 0:
        return {}
    overlap = topk_per_layer / n_blocks
    jaccard = overlap / (2.0 - overlap)
    unique = n_blocks * (1.0 - (1.0 - overlap) ** n_layers)
    naive = topk_per_layer * n_layers
    return {
        "adjacent_overlap_mean": overlap,
        "adjacent_jaccard_mean": jaccard,
        "naive_blocks_mean": float(naive),
        "unique_blocks_mean": unique,
        "reuse_factor_mean": naive / unique if unique else 0.0,
        "unique_ratio_mean": unique / naive if naive else 0.0,
    }


def write_selection_jsonl(path: Path, selections: SelectionMap, rank: int) -> None:
    """Write raw selected block IDs as JSONL.

    Args:
        path: Destination JSONL path.
        selections: Mapping from CSA layer ID to per-step selected block sets.
        rank: Distributed rank that produced the records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for layer_id in sorted(selections):
            for step, block_ids in enumerate(selections[layer_id]):
                handle.write(
                    json.dumps(
                        {
                            "rank": rank,
                            "step": step,
                            "layer_id": layer_id,
                            "block_ids": sorted(block_ids),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    """Write the experiment summary JSON.

    Args:
        path: Destination JSON path.
        summary: JSON-serializable summary dictionary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def print_distribution(values: list[float]) -> None:
    """Print a compact bucket distribution for overlap values.

    Args:
        values: Overlap samples in the range [0, 1].
    """
    buckets = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.01]
    labels = ["<50%", "50-80%", "80-90%", "90-95%", "95-99%", "99-100%"]
    counts = [0] * len(labels)
    for value in values:
        for idx, high in enumerate(buckets[1:]):
            if value < high:
                counts[idx] += 1
                break
    max_count = max(counts) if counts else 0
    for label, count in zip(labels, counts):
        bar = "#" * int(count / max_count * 30) if max_count > 0 else ""
        print(f"    {label:10s}  {count:4d}  {bar}")


def main() -> None:
    """Run the CSA selection stability experiment."""
    # Third Party
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_model

    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
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
    model_module.rank = rank

    from model import Indexer, ModelArgs, Transformer

    selections: SelectionMap = defaultdict(list)
    orig_indexer_forward = Indexer.forward

    def patched_indexer_forward(self: Any, x: Any, qr: Any, start_pos: int, offset: Any) -> Any:
        topk_idxs = orig_indexer_forward(self, x, qr, start_pos, offset)
        if start_pos > 0:
            layer_id = getattr(self, "_layer_id", -1)
            selections[layer_id].append(topk_to_block_set(topk_idxs))
        return topk_idxs

    Indexer.forward = patched_indexer_forward

    config_path = Path(args.model_path) / "inference" / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)

    cfg["max_batch_size"] = 1
    cfg["max_seq_len"] = args.prompt_len + args.decode_steps + 64

    if rank == 0:
        print(f"Loading model (rank {rank}/{world_size}) from {args.model_path} ...")

    model_args = ModelArgs(**cfg)
    model = Transformer(model_args)

    ckpt_file = Path(args.model_path) / f"model{rank}-mp{world_size}.safetensors"
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"{ckpt_file} not found. Run convert.py first:\n"
            f"  python inference/convert.py --hf-ckpt-path {args.model_path} "
            f"--save-path {args.model_path} --n-experts {cfg['n_routed_experts']} "
            f"--model-parallel {world_size} --expert-dtype "
            f"{cfg.get('expert_dtype', 'fp8')}"
        )

    load_model(model, str(ckpt_file), strict=False)
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, Indexer):
            parts = name.split(".")
            try:
                layer_id = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                layer_id = -1
            setattr(module, "_layer_id", layer_id)

    if rank == 0:
        print("Model loaded.")
        print(f"Building prompt of {args.prompt_len} tokens ...")

    prompt_ids = torch.randint(
        10,
        model_args.vocab_size - 10,
        (1, args.prompt_len),
        device=f"cuda:{local_rank}",
    )

    if rank == 0:
        print("Running prefill ...")

    with torch.no_grad():
        logits = model.forward(prompt_ids, start_pos=0)

    next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)

    if rank == 0:
        print("Running decode ...")

    for step in range(args.decode_steps):
        with torch.no_grad():
            logits = model.forward(next_token, start_pos=args.prompt_len + step)
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        if rank == 0 and step % 10 == 9:
            print(f"  step {step + 1}/{args.decode_steps}")

    if rank != 0:
        if world_size > 1:
            dist.destroy_process_group()
        return

    print("\n" + "=" * 72)
    print("CSA Selection Reuse Analysis")
    print(f"Prompt length : {args.prompt_len} tokens")
    print(f"Decode steps  : {args.decode_steps}")
    print(f"Compressed blocks in pool: {args.prompt_len // 4}")
    print("=" * 72)

    if not selections:
        print("[ERROR] No Indexer selections recorded. All layers may use HCA.")
        if world_size > 1:
            dist.destroy_process_group()
        sys.exit(1)

    cross_step_by_layer, cross_step_all = compute_cross_step(selections)
    cross_layer_by_pair, cross_layer_all, cross_layer_jaccard_all = compute_cross_layer(
        selections
    )
    unique_pool = compute_unique_pool(selections)
    random_baseline = compute_random_baseline(
        args.prompt_len // 4,
        1024,
        len(selections),
    )

    print("\nCross-step overlap: same layer, adjacent decode steps")
    for layer_id in sorted(cross_step_by_layer):
        stats = cross_step_by_layer[layer_id]
        print(
            f"  Layer {layer_id:3d}: mean={stats['mean']:.3f}  "
            f"std={stats['std']:.3f}  min={stats['min']:.3f}  "
            f"p50={stats['p50']:.3f}  p90={stats['p90']:.3f}"
        )
    if cross_step_all:
        stats = summarize_values(cross_step_all)
        print("-" * 72)
        print(
            f"  ALL CSA layers: mean={stats['mean']:.3f}  "
            f"std={stats['std']:.3f}  min={stats['min']:.3f}  "
            f"p50={stats['p50']:.3f}  p90={stats['p90']:.3f}"
        )
        print("\n  Cross-step overlap distribution:")
        print_distribution(cross_step_all)

    print("\nCross-layer overlap: adjacent CSA layers at the same decode step")
    for pair in sorted(cross_layer_by_pair):
        stats = cross_layer_by_pair[pair]
        print(
            f"  Layer {pair:>7s}: overlap_mean={stats['overlap_mean']:.3f}  "
            f"jaccard_mean={stats['jaccard_mean']:.3f}  "
            f"overlap_min={stats['overlap_min']:.3f}"
        )
    if cross_layer_all:
        directed = summarize_values(cross_layer_all)
        jaccard = summarize_values(cross_layer_jaccard_all)
        print("-" * 72)
        print(
            f"  Adjacent-layer ALL: overlap_mean={directed['mean']:.3f}  "
            f"overlap_p50={directed['p50']:.3f}  "
            f"jaccard_mean={jaccard['mean']:.3f}"
        )

    if unique_pool:
        print("\nUnified HBM numeric-id pool estimate: all CSA layers per step")
        print(
            f"  layers={unique_pool['layers']:.0f}  "
            f"steps={unique_pool['steps']:.0f}  "
            f"naive_blocks_mean={unique_pool['naive_blocks_mean']:.1f}  "
            f"unique_blocks_mean={unique_pool['unique_blocks_mean']:.1f}"
        )
        print(
            f"  reuse_factor_mean={unique_pool['reuse_factor_mean']:.3f}x  "
            f"unique_ratio_mean={unique_pool['unique_ratio_mean']:.3f}"
        )
        if random_baseline:
            print(
                f"  random_baseline: adjacent_overlap="
                f"{random_baseline['adjacent_overlap_mean']:.3f}  "
                f"unique_blocks_mean="
                f"{random_baseline['unique_blocks_mean']:.1f}  "
                f"reuse_factor_mean="
                f"{random_baseline['reuse_factor_mean']:.3f}x"
            )

    print("\nInterpretation:")
    print("  Cross-step >90%: delta-selection across decode steps is effective.")
    print("  Cross-layer high overlap: unified residency scheduling may help.")
    print("  Low cross-layer overlap: each CSA layer likely needs separate hot residency.")
    print("  Same numeric block id across layers is not shared KV payload.")

    summary: dict[str, Any] = {
        "prompt_len": args.prompt_len,
        "decode_steps": args.decode_steps,
        "compressed_blocks": args.prompt_len // 4,
        "rank": rank,
        "world_size": world_size,
        "layers": sorted(selections),
        "cross_step_by_layer": cross_step_by_layer,
        "cross_step_all": summarize_values(cross_step_all),
        "cross_layer_adjacent_by_pair": cross_layer_by_pair,
        "cross_layer_adjacent_all": summarize_values(cross_layer_all),
        "cross_layer_adjacent_jaccard_all": summarize_values(
            cross_layer_jaccard_all
        ),
        "unified_hbm_pool": unique_pool,
        "random_independent_baseline": random_baseline,
    }

    if args.json_summary is not None:
        write_summary_json(args.json_summary, summary)
        print(f"\nWrote JSON summary: {args.json_summary}")

    if args.dump_selection_jsonl is not None:
        write_selection_jsonl(args.dump_selection_jsonl, selections, rank)
        print(f"Wrote raw selection JSONL: {args.dump_selection_jsonl}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
