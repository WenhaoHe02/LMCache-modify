#!/usr/bin/env bash
set -euo pipefail

root=/home/zbuser02/csa_cp8_ab_20260717
base_tokens=480000
recompute_tokens=8192
tag="csa_on_no_nsys_distinct_${base_tokens}p${recompute_tokens}_$(date -u +%Y%m%d_%H%M%S)"
result_dir="${root}/results/${tag}"
container=dsv4-csa-cp8-on
mkdir -p "${result_dir}"
printf '%s\n' "${tag}" > "${root}/current_no_nsys_distinct_tag.txt"
printf 'starting\n' > "${result_dir}/status"

cleanup() {
  local rc=$?
  sudo docker logs "${container}" > "${result_dir}/server.log" 2>&1 || true
  sudo docker stop -t 90 "${container}" >/dev/null 2>&1 || true
  sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  if [[ ${rc} -ne 0 ]]; then
    printf 'failed rc=%d\n' "${rc}" > "${result_dir}/status"
  fi
}
trap cleanup EXIT

(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  2>/dev/null | grep -E '[0-9]' | xargs -r sudo kill -9) || true
sudo docker rm -f "${container}" >/dev/null 2>&1 || true

unset LMCACHE_EXEC_PREFIX
unset LMCACHE_NSYS_CAPTURE
unset LMCACHE_NSYS_FULL_CAPTURE
unset LMCACHE_NSYS_FULL_CAPTURE_SCOPE
unset LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS
export LMCACHE_ABLATION_PATCH_DIR="${root}/patches"
export LMCACHE_ABLATION_STARTUP_SCRIPT="${root}/startup_cp8_ab.sh"
export LMCACHE_ABLATION_MAX_MODEL_LEN=530000
export LMCACHE_ABLATION_MAX_BATCHED_TOKENS=65536
export LMCACHE_ABLATION_GPU_UTIL=0.60
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
export LMCACHE_CSA_PIPELINE_NVTX=0
export LMCACHE_CSA_ATTENTION_KV_TIMING=0
export LMCACHE_TUTTI_PROFILE=0
export LMCACHE_INDEXER_TIMING=0
export LMCACHE_TTFT_STAGE_PROFILE=0
export LMCACHE_HCA_TIMING=0

bash "${root}/run_container_cp8_ab.sh" on \
  > "${result_dir}/launch.log" 2>&1

container_pid=$(sudo docker inspect -f '{{.State.Pid}}' "${container}")
sudo sh -c "tr '\000' '\n' </proc/${container_pid}/environ" \
  > "${result_dir}/process_env.txt"
if grep -Eq \
  '^(LMCACHE_EXEC_PREFIX=.+|LMCACHE_NSYS_(CAPTURE|FULL_CAPTURE)=(1|true|TRUE|on|ON))$' \
  "${result_dir}/process_env.txt"; then
  echo 'profiler environment leaked into no-nsys run' >&2
  exit 8
fi

printf 'running workload\n' > "${result_dir}/status"
BASE_TOKENS="${base_tokens}" RECOMPUTE_TOKENS="${recompute_tokens}" \
  STORE_WAIT_S=60 WARMUP_WAIT_S=60 \
  python3 "${root}/run_distinct_string_480k8192.py" \
  > "${result_dir}/workload.jsonl" \
  2> "${result_dir}/workload.err"

sudo docker logs "${container}" > "${result_dir}/server.log" 2>&1 || true
grep -q '"label": "target_distinct_8192", "status": 200' \
  "${result_dir}/workload.jsonl"
grep -q '"all_corresponding_continuation_blocks_differ": true' \
  "${result_dir}/workload.jsonl"
retrieved_480k=$(grep -Ec \
  'Retrieved 480000 out of (480000|488192) required tokens' \
  "${result_dir}/server.log" || true)
retrieved_full=$(grep -Ec \
  'Retrieved 488192 out of 488192 required tokens' \
  "${result_dir}/server.log" || true)
if [[ ${retrieved_480k} -lt 16 || ${retrieved_full} -ne 0 ]]; then
  echo "invalid recompute workload: prefix_hits=${retrieved_480k} full_hits=${retrieved_full}" >&2
  exit 7
fi

printf 'complete\n' > "${result_dir}/status"
printf '%s\n' "${tag}" > "${root}/completed_no_nsys_distinct_tag.txt"
trap - EXIT
cleanup
