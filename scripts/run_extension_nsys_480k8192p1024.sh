#!/usr/bin/env bash
set -euo pipefail

root=/home/zbuser02/csa_cp8_ab_20260717
exec 9>"${root}/run_extension_nsys.lock"
flock -n 9 || { echo 'another extension nsys run is active' >&2; exit 75; }

tag="on_extension_nsys_default_0_480000p8192p1024_$(date -u +%Y%m%d_%H%M%S)"
result_dir="${root}/results/${tag}"
trace_dir="/tmp/${tag}"
trace_name=on_extension_480k8192p1024
container=dsv4-csa-cp8-on
nsys=/opt/nvidia/nsight-systems/2025.3.2/bin/nsys
mkdir -p "${result_dir}" "${trace_dir}"
printf '%s\n' "${tag}" > "${root}/current_extension_nsys_tag.txt"
printf 'starting\n' > "${result_dir}/status"

cleanup() {
  local rc=$?
  if sudo docker inspect "${container}" >/dev/null 2>&1; then
    sudo docker logs "${container}" > "${result_dir}/server.log" 2>&1 || true
    sudo docker inspect "${container}" \
      > "${result_dir}/container_inspect.json" 2>&1 || true
    sudo docker stop -t 90 "${container}" >/dev/null 2>&1 || true
    sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  fi
  [[ ${rc} -eq 0 ]] || printf 'failed rc=%d\n' "${rc}" > "${result_dir}/status"
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

export LMCACHE_ABLATION_PATCH_DIR=/home/zbuser02/codex_sync_overlap_fix/patches_native_compact_perf_20260721
export LMCACHE_ABLATION_STARTUP_SCRIPT=/home/zbuser02/codex_sync_overlap_fix/startup_native_compact_perf_20260721.sh
export LMCACHE_ABLATION_MAX_MODEL_LEN=530000
export LMCACHE_ABLATION_MAX_BATCHED_TOKENS=65536
export LMCACHE_ABLATION_GPU_UTIL=0.55
export LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB=8
export LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY=48000
export LMCACHE_ABLATION_TUTTI_N_SLOTS=4
export LMCACHE_ABLATION_TUTTI_SLOT_MB=128
export LMCACHE_ABLATION_TUTTI_STARTUP_DELAY=1
export LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY=10
export LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER=default:0
export LMCACHE_CSA_PREFETCH_CP_SIZE=8
export LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=64
export LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE=1
export LMCACHE_CSA_PREFETCH_BLOCK_BUDGET=256
export LMCACHE_CSA_EXACT_CHUNK_PREFETCH=0
export LMCACHE_DSV4_HCA_WALKER=1
export LMCACHE_INDEXER_PROFILE_ACCURACY=0
export LMCACHE_CSA_PIPELINE_NVTX=1
export LMCACHE_NSYS_CAPTURE=0
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

printf 'running workload\n' > "${result_dir}/status"
sleep 10
MODE_LABEL=on BASE_TOKENS=480000 FIRST_SUFFIX_TOKENS=8192 \
  EXTENSION_TOKENS=1024 MAX_TOKENS=96 STORE_WAIT_S=20 ADMISSION_WAIT_S=15 \
  python3 "${root}/run_extension_admission_correctness_480k8192.py" \
  > "${result_dir}/workload.jsonl" 2> "${result_dir}/workload.err"

sleep 10
sudo docker logs "${container}" > "${result_dir}/server.log" 2>&1
grep -q 'LMCache hit tokens: 488192' "${result_dir}/server.log"
grep -q 'NSYS_FULL_CAPTURE start' "${result_dir}/server.log"
grep -q 'NSYS_FULL_CAPTURE stop' "${result_dir}/server.log"
if grep -Eq 'Traceback \(most recent call last\)|CUDA error|object_unreadable' \
  "${result_dir}/server.log"; then
  echo 'extension nsys log contains a fatal marker' >&2
  exit 4
fi

sudo docker stop -t 90 "${container}" >/dev/null 2>&1 || true
sudo docker rm -f "${container}" >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
  [[ -s "${trace_dir}/${trace_name}.nsys-rep" ]] && break
  sleep 5
done
test -s "${trace_dir}/${trace_name}.nsys-rep"
cp "${trace_dir}/${trace_name}.nsys-rep" "${result_dir}/"
"${nsys}" stats --report cuda_gpu_kern_sum \
  "${result_dir}/${trace_name}.nsys-rep" \
  > "${result_dir}/cuda_gpu_kern_sum.txt" 2>&1 || true
"${nsys}" stats --report cuda_api_sum \
  "${result_dir}/${trace_name}.nsys-rep" \
  > "${result_dir}/cuda_api_sum.txt" 2>&1 || true
"${nsys}" stats --report nvtx_sum \
  "${result_dir}/${trace_name}.nsys-rep" \
  > "${result_dir}/nvtx_sum.txt" 2>&1 || true
NSYS_SQLITE_ANALYZER="${root}/analyze_nsys_sqlite.py" \
  bash "${root}/audit_nsys_report.sh" \
  "${result_dir}/${trace_name}.nsys-rep" \
  "${result_dir}/audit" > "${result_dir}/audit.log" 2>&1 || true
sha256sum "${result_dir}/${trace_name}.nsys-rep" \
  > "${result_dir}/SHA256SUMS"
printf 'complete\n' > "${result_dir}/status"
printf '%s\n' "${tag}" > "${root}/completed_extension_nsys_tag.txt"
trap - EXIT
cleanup
