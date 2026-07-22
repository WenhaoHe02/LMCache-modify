#!/usr/bin/env bash
set -euo pipefail

root=/home/zbuser02/csa_cp8_ab_20260717
exec 9>"${root}/run_csa_all_l2_accuracy.lock"
if ! flock -n 9; then
  echo 'another all-L2 accuracy harness is already running' >&2
  exit 75
fi

base_tokens=480000
recompute_tokens=8192
tag="csa_all_l2_accuracy_${base_tokens}p${recompute_tokens}_$(date -u +%Y%m%d_%H%M%S)"
result_dir="${root}/results/${tag}"
container=dsv4-csa-cp8-on
mkdir -p "${result_dir}"
printf '%s\n' "${tag}" > "${root}/current_all_l2_accuracy_tag.txt"
printf 'starting\n' > "${result_dir}/status"

cleanup() {
  local rc=$?
  sudo docker logs "${container}" > "${result_dir}/server.log" 2>&1 || true
  sudo docker inspect "${container}" \
    > "${result_dir}/container_inspect.json" 2>&1 || true
  sudo docker stop -t 90 "${container}" >/dev/null 2>&1 || true
  sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  if [[ ${rc} -ne 0 ]]; then
    printf 'failed rc=%d\n' "${rc}" > "${result_dir}/status"
  fi
}
trap cleanup EXIT

sudo docker rm -f "${container}" >/dev/null 2>&1 || true
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  2>/dev/null | grep -Eq '[0-9]'; then
  echo 'GPU is not idle' >&2
  exit 3
fi

unset LMCACHE_EXEC_PREFIX
export LMCACHE_NSYS_CAPTURE=0
export LMCACHE_NSYS_CAPTURE_SKIP_REQUESTS=0
export LMCACHE_NSYS_FULL_CAPTURE=0
export LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS=0
export LMCACHE_ABLATION_PATCH_DIR="${root}/patches"
export LMCACHE_ABLATION_STARTUP_SCRIPT="${root}/startup_cp8_ab.sh"
export LMCACHE_ABLATION_MAX_MODEL_LEN=530000
export LMCACHE_ABLATION_MAX_BATCHED_TOKENS=65536
export LMCACHE_ABLATION_GPU_UTIL=0.55
export LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB=8
export LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY=48000
export LMCACHE_ABLATION_TUTTI_N_SLOTS=4
export LMCACHE_ABLATION_TUTTI_SLOT_MB=128
export LMCACHE_ABLATION_TUTTI_STARTUP_DELAY=120
export LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY=10
export LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER=default:2
export LMCACHE_CSA_PREFETCH_CP_SIZE=8
export LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=64
export LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE=1
export LMCACHE_CSA_PREFETCH_BLOCK_BUDGET=256
export LMCACHE_DSV4_HCA_WALKER=1
export LMCACHE_INDEXER_PROFILE_ACCURACY=1
export LMCACHE_CSA_PIPELINE_NVTX=0
export LMCACHE_CSA_ATTENTION_KV_TIMING=0
export LMCACHE_TUTTI_PROFILE=0
export LMCACHE_INDEXER_TIMING=0
export LMCACHE_TTFT_STAGE_PROFILE=0
export LMCACHE_HCA_TIMING=0
export CUDA_LAUNCH_BLOCKING=0

bash "${root}/run_container_cp8_ab.sh" on \
  > "${result_dir}/launch.log" 2>&1

container_pid=$(sudo docker inspect -f '{{.State.Pid}}' "${container}")
sudo sh -c "tr '\000' '\n' </proc/${container_pid}/environ" \
  > "${result_dir}/process_env.txt"
grep -qx 'LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER=default:2' \
  "${result_dir}/process_env.txt"
grep -qx 'LMCACHE_INDEXER_PROFILE_ACCURACY=1' \
  "${result_dir}/process_env.txt"

printf 'running workload\n' > "${result_dir}/status"
sleep 20
ENABLE_TORCH_PROFILE=0 DISTINCT_HIT_PROMPTS=1 \
  NUM_WARMUP_HITS=1 NUM_HITS=4 HIT_WAIT_S=5 \
  BASE_TOKENS="${base_tokens}" RECOMPUTE_TOKENS="${recompute_tokens}" \
  python3 "${root}/run_hermes_trial2_480k200.py" 60 \
  > "${result_dir}/workload.jsonl" \
  2> "${result_dir}/workload.err"

sudo docker logs "${container}" > "${result_dir}/server.log" 2>&1 || true
python3 - "${result_dir}" <<'PY'
import json
from pathlib import Path
import re
import sys

result_dir = Path(sys.argv[1])
log = (result_dir / "server.log").read_text(errors="replace")
for marker in (
    "CUDA error",
    "illegal memory access",
    "out of memory",
    "Traceback (most recent call last)",
    "Retrieved 488192 out of 488192 required tokens",
):
    if marker in log:
        raise SystemExit(f"invalid server marker: {marker}")

pattern = re.compile(
    r"attention_true_topk_profile layer (?P<layer>\d+) sample=(?P<sample>\d+) "
    r"true_tokens=(?P<true_tokens>\d+) true_blocks=(?P<true_blocks>\d+) "
    r"predicted_blocks=(?P<predicted_blocks>\d+) block_hits=(?P<hits>\d+) "
    r"block_misses=(?P<misses>\d+).*?"
    r"weighted_block_hits=(?P<weighted_hits>\d+) "
    r"weighted_block_total=(?P<weighted_total>\d+)"
)
by_layer: dict[int, list[dict[str, int]]] = {}
for match in pattern.finditer(log):
    row = {key: int(value) for key, value in match.groupdict().items()}
    by_layer.setdefault(row["layer"], []).append(row)

gate_pattern = re.compile(
    r"prediction_target_gate layer (?P<layer>\d+) submitted=(?P<submitted>[01]) "
    r"was_ready=(?P<was_ready>[01]) cancelled=(?P<cancelled>[01]) "
    r"running=(?P<running>\d+) wait_ms=(?P<wait_ms>[0-9.]+)"
)
gates_by_layer: dict[int, list[dict[str, float]]] = {}
for match in gate_pattern.finditer(log):
    row = {
        "layer": int(match.group("layer")),
        "submitted": int(match.group("submitted")),
        "was_ready": int(match.group("was_ready")),
        "cancelled": int(match.group("cancelled")),
        "running": int(match.group("running")),
        "wait_ms": float(match.group("wait_ms")),
    }
    gates_by_layer.setdefault(int(row["layer"]), []).append(row)

expected = list(range(2, 43, 2))
missing = [layer for layer in expected if layer not in by_layer]
if missing:
    raise SystemExit(f"missing accuracy records for layers {missing}")

layers = []
for layer in expected:
    rows = by_layer[layer]
    gate_rows = gates_by_layer.get(layer, [])
    if not gate_rows:
        raise SystemExit(f"missing target-gate records for layer {layer}")
    ready = [row for row in rows if row["predicted_blocks"] > 0]
    true_all = sum(row["true_blocks"] for row in rows)
    hits_all = sum(row["hits"] for row in rows)
    pred_all = sum(row["predicted_blocks"] for row in rows)
    true_ready = sum(row["true_blocks"] for row in ready)
    hits_ready = sum(row["hits"] for row in ready)
    pred_ready = sum(row["predicted_blocks"] for row in ready)
    weighted_all = sum(row["weighted_total"] for row in rows)
    weighted_hits_all = sum(row["weighted_hits"] for row in rows)
    weighted_ready = sum(row["weighted_total"] for row in ready)
    weighted_hits_ready = sum(row["weighted_hits"] for row in ready)
    layers.append(
        {
            "layer": layer,
            "records": len(rows),
            "prediction_ready_records": len(ready),
            "prediction_ready_fraction": len(ready) / len(rows),
            "target_gate_ready_fraction": (
                sum(row["was_ready"] for row in gate_rows) / len(gate_rows)
            ),
            "target_gate_mean_wait_ms": (
                sum(row["wait_ms"] for row in gate_rows) / len(gate_rows)
            ),
            "target_gate_max_wait_ms": max(row["wait_ms"] for row in gate_rows),
            "end_to_end_recall": hits_all / true_all if true_all else 0.0,
            "ready_only_recall": hits_ready / true_ready if true_ready else 0.0,
            "ready_only_precision": hits_ready / pred_ready if pred_ready else 0.0,
            "end_to_end_weighted_recall": (
                weighted_hits_all / weighted_all if weighted_all else 0.0
            ),
            "ready_only_weighted_recall": (
                weighted_hits_ready / weighted_ready if weighted_ready else 0.0
            ),
            "mean_predicted_blocks_when_ready": pred_ready / len(ready) if ready else 0.0,
        }
    )

candidates = [
    row["layer"]
    for row in layers
    if row["layer"] < 26
    and row["target_gate_ready_fraction"] >= 0.75
    and row["ready_only_recall"] >= 0.80
    and row["ready_only_precision"] >= 0.80
    and row["ready_only_weighted_recall"] >= 0.80
]
summary = {
    "policy": "default:2",
    "base_tokens": 480000,
    "recompute_tokens": 8192,
    "candidate_rule": (
        "early layer, target-gate ready fraction >= 0.75, ready-only unique "
        "block recall and precision >= 0.80, and ready-only weighted block "
        "recall >= 0.80"
    ),
    "candidate_early_layers": candidates,
    "layers": layers,
}
(result_dir / "layer_accuracy.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True)
)
with (result_dir / "layer_accuracy.tsv").open("w") as output:
    output.write(
        "layer\trecords\tproxy_ready_fraction\tgate_ready_fraction\t"
        "gate_mean_wait_ms\tgate_max_wait_ms\te2e_recall\tready_recall\t"
        "e2e_weighted_recall\tready_weighted_recall\tready_precision\t"
        "mean_predicted_blocks\n"
    )
    for row in layers:
        output.write(
            f'{row["layer"]}\t{row["records"]}\t'
            f'{row["prediction_ready_fraction"]:.4f}\t'
            f'{row["target_gate_ready_fraction"]:.4f}\t'
            f'{row["target_gate_mean_wait_ms"]:.3f}\t'
            f'{row["target_gate_max_wait_ms"]:.3f}\t'
            f'{row["end_to_end_recall"]:.4f}\t'
            f'{row["ready_only_recall"]:.4f}\t'
            f'{row["end_to_end_weighted_recall"]:.4f}\t'
            f'{row["ready_only_weighted_recall"]:.4f}\t'
            f'{row["ready_only_precision"]:.4f}\t'
            f'{row["mean_predicted_blocks_when_ready"]:.2f}\n'
        )
print(json.dumps(summary, sort_keys=True))
PY

printf 'complete\n' > "${result_dir}/status"
printf '%s\n' "${tag}" > "${root}/completed_all_l2_accuracy_tag.txt"
trap - EXIT
cleanup
