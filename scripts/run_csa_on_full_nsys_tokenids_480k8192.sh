#!/usr/bin/env bash
set -euo pipefail

root=/home/zbuser02/csa_cp8_ab_20260717
exec 9>"${root}/run_csa_on_full_nsys_tokenids.lock"
if ! flock -n 9; then
  echo 'another CSA on full-Nsys harness is already running' >&2
  exit 75
fi

base_tokens=480000
recompute_tokens=8192
tag="csa_on_full_nsys_tokenids_${base_tokens}p${recompute_tokens}_$(date -u +%Y%m%d_%H%M%S)"
result_dir="${root}/results/${tag}"
trace_dir="/tmp/${tag}"
trace_name=csa_on_tokenids_480k8192_full
container=dsv4-csa-cp8-on
nsys=/opt/nvidia/nsight-systems/2025.3.2/bin/nsys
mkdir -p "${result_dir}" "${trace_dir}"
printf '%s\n' "${tag}" > "${root}/current_on_full_nsys_tokenids_tag.txt"
printf 'starting\n' > "${result_dir}/status"

cleanup() {
  local rc=$?
  if sudo docker inspect "${container}" >/dev/null 2>&1; then
    sudo docker logs "${container}" > "${result_dir}/server.log" 2>&1 || true
    sudo docker inspect "${container}" > "${result_dir}/container_inspect.json" 2>&1 || true
    sudo docker stop -t 90 "${container}" >/dev/null 2>&1 || true
    sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  fi
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

nsys_prefix="${nsys} profile --trace=cuda,nvtx,osrt --sample=none \
--cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
--force-overwrite=true --output=${trace_dir}/${trace_name}"

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
export LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER=profile80
export LMCACHE_CSA_PREFETCH_CP_SIZE=8
export LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=64
export LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE=1
export LMCACHE_CSA_PREFETCH_BLOCK_BUDGET=256
export LMCACHE_DSV4_HCA_WALKER=1
export LMCACHE_INDEXER_PROFILE_ACCURACY=1
export LMCACHE_CSA_PIPELINE_NVTX=1
export LMCACHE_CSA_ATTENTION_KV_TIMING=0
export LMCACHE_TUTTI_PROFILE=0
export LMCACHE_INDEXER_TIMING=0
export LMCACHE_TTFT_STAGE_PROFILE=0
export LMCACHE_HCA_TIMING=0
export LMCACHE_NSYS_CAPTURE=0
export LMCACHE_NSYS_CAPTURE_SKIP_REQUESTS=0
export LMCACHE_NSYS_FULL_CAPTURE=1
export LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS=1
export LMCACHE_NSYS_FULL_CAPTURE_SCOPE=decoder
export LMCACHE_EXEC_PREFIX="${nsys_prefix}"
export CUDA_LAUNCH_BLOCKING=0

bash "${root}/run_container_cp8_ab.sh" on \
  > "${result_dir}/launch.log" 2>&1

container_pid=$(sudo docker inspect -f '{{.State.Pid}}' "${container}")
sudo sh -c "tr '\000' '\n' </proc/${container_pid}/environ" \
  > "${result_dir}/process_env.txt"
grep -qx 'LMCACHE_NSYS_FULL_CAPTURE=1' "${result_dir}/process_env.txt"
grep -qx 'LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS=1' "${result_dir}/process_env.txt"
grep -qx 'LMCACHE_NSYS_FULL_CAPTURE_SCOPE=decoder' "${result_dir}/process_env.txt"
grep -qx 'LMCACHE_DSV4_HCA_WALKER=1' "${result_dir}/process_env.txt"
grep -qx 'LMCACHE_INDEXER_PROFILE_ACCURACY=1' "${result_dir}/process_env.txt"

{
  date -u '+utc=%FT%TZ'
  hostname
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,power.limit \
    --format=csv,noheader
  "${nsys}" --version
  sha256sum \
    "${root}/patches/v1/csa_attention_kv_prefetch_manager.py" \
    "${root}/patches/v1/gpu_connector/tutti_direct_loader.py"
} > "${result_dir}/environment_manifest.txt" 2>&1

printf 'running workload\n' > "${result_dir}/status"
sleep 20
ENABLE_TORCH_PROFILE=0 DISTINCT_HIT_PROMPTS=1 \
  NUM_WARMUP_HITS=1 NUM_HITS=1 HIT_WAIT_S=5 \
  BASE_TOKENS="${base_tokens}" RECOMPUTE_TOKENS="${recompute_tokens}" \
  python3 "${root}/run_hermes_trial2_480k200.py" 60 \
  > "${result_dir}/workload.jsonl" \
  2> "${result_dir}/workload.err"

sleep 10
sudo docker logs "${container}" > "${result_dir}/server.log" 2>&1 || true
sudo docker inspect "${container}" > "${result_dir}/container_inspect.json" 2>&1 || true

python3 - "${result_dir}" <<'PY'
import json
from pathlib import Path
import sys

result_dir = Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in (result_dir / "workload.jsonl").read_text().splitlines()
    if line.strip()
]
ready = next(row for row in rows if row.get("event") == "prompt_ready")
hashes = ready.get("continuation_hashes", [])
if ready.get("distinct_hit_prompts") is not True or len(set(hashes)) != 2:
    raise SystemExit("workload continuations are not distinct")
hits = [
    row
    for row in rows
    if str(row.get("label", "")).startswith("hit_trial2_8192_")
]
warmups = [
    row
    for row in rows
    if str(row.get("label", "")).startswith("warmup_trial2_8192_")
]
if len(warmups) != 1 or len(hits) != 1:
    raise SystemExit("wrong warmup/hit count")
if any(row.get("status") != 200 for row in warmups + hits):
    raise SystemExit("request failed")
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
if "NSYS_FULL_CAPTURE start" not in log or "NSYS_FULL_CAPTURE stop" not in log:
    raise SystemExit("full Nsys capture did not start and stop")
accuracy_records = log.count("attention_true_topk_profile")
if accuracy_records < 1:
    raise SystemExit("accuracy profiling produced no records")
summary = {
    "cold_s": next(
        float(row["elapsed_s"])
        for row in rows
        if row.get("label") == "cold_store"
    ),
    "warmup_s": float(warmups[0]["elapsed_s"]),
    "captured_hit_s": float(hits[0]["elapsed_s"]),
    "accuracy_records": accuracy_records,
    "capture_scope": "decoder",
    "full_retrieval_seen": False,
}
(result_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True)
)
print(json.dumps(summary, sort_keys=True))
PY

sudo docker stop -t 90 "${container}" >/dev/null 2>&1 || true
sudo docker rm -f "${container}" >/dev/null 2>&1 || true

for _ in $(seq 1 60); do
  [[ -s "${trace_dir}/${trace_name}.nsys-rep" ]] && break
  sleep 5
done
test -s "${trace_dir}/${trace_name}.nsys-rep"
cp "${trace_dir}/${trace_name}.nsys-rep" "${result_dir}/"

"${nsys}" stats --report cuda_gpu_kern_sum \
  "${result_dir}/${trace_name}.nsys-rep" > "${result_dir}/cuda_gpu_kern_sum.txt" 2>&1 || true
"${nsys}" stats --report cuda_api_sum \
  "${result_dir}/${trace_name}.nsys-rep" > "${result_dir}/cuda_api_sum.txt" 2>&1 || true
"${nsys}" stats --report nvtx_sum \
  "${result_dir}/${trace_name}.nsys-rep" > "${result_dir}/nvtx_sum.txt" 2>&1 || true
"${nsys}" stats --report osrt_sum \
  "${result_dir}/${trace_name}.nsys-rep" > "${result_dir}/osrt_sum.txt" 2>&1 || true
NSYS_SQLITE_ANALYZER="${root}/analyze_nsys_sqlite.py" \
  bash "${root}/audit_nsys_report.sh" \
  "${result_dir}/${trace_name}.nsys-rep" \
  "${result_dir}/audit" > "${result_dir}/audit.log" 2>&1 || true
sha256sum "${result_dir}/${trace_name}.nsys-rep" \
  > "${result_dir}/SHA256SUMS"

printf 'complete\n' > "${result_dir}/status"
printf '%s\n' "${tag}" > "${root}/completed_on_full_nsys_tokenids_tag.txt"
trap - EXIT
cleanup
