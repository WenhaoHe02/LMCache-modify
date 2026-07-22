#!/usr/bin/env bash
set -euo pipefail

root=/home/zbuser02/csa_cp8_ab_20260717
tag="csa_attention_only_first_token_$(date -u +%Y%m%d_%H%M%S)"
result_dir="${root}/results/${tag}"
container=dsv4-csa-cp8-on
mkdir -p "${result_dir}"
printf '%s\n' "${tag}" > "${root}/current_attention_only_tag.txt"
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

export LMCACHE_ABLATION_PATCH_DIR="${root}/patches"
export LMCACHE_ABLATION_STARTUP_SCRIPT="${root}/startup_cp8_diag.sh"
export LMCACHE_ABLATION_MAX_MODEL_LEN=530000
export LMCACHE_ABLATION_MAX_BATCHED_TOKENS=8192
export LMCACHE_ABLATION_GPU_UTIL=0.55
export LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB=1
export LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY=56000
export LMCACHE_ABLATION_TUTTI_N_SLOTS=4
export LMCACHE_ABLATION_TUTTI_SLOT_MB=128
export LMCACHE_ABLATION_TUTTI_STARTUP_DELAY=1
export LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY=10
export LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER=default:0
export LMCACHE_CSA_PREFETCH_CP_SIZE=8
export LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=64
export LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE=1
export LMCACHE_CSA_PREFETCH_BLOCK_BUDGET=256
export LMCACHE_DSV4_HCA_WALKER=0
export LMCACHE_INDEXER_PROFILE_ACCURACY=0
export LMCACHE_NSYS_CAPTURE=0
export LMCACHE_NSYS_FULL_CAPTURE=0
export LMCACHE_CSA_PIPELINE_NVTX=0
export CUDA_LAUNCH_BLOCKING=0
unset LMCACHE_EXEC_PREFIX

sudo docker rm -f "${container}" >/dev/null 2>&1 || true
bash "${root}/run_container_cp8_ab.sh" on > "${result_dir}/launch.log" 2>&1
pid=$(sudo docker inspect -f '{{.State.Pid}}' "${container}")
sudo sh -c "tr '\000' '\n' </proc/${pid}/environ" \
  > "${result_dir}/process_env.txt"
grep -qx 'LMCACHE_INDEXER_TUTTI_BACKEND=0' "${result_dir}/process_env.txt"
sleep 15
printf 'running\n' > "${result_dir}/status"
MODE_LABEL=attention_only BASE_TOKENS=480000 RECOMPUTE_TOKENS=8192 \
  MAX_TOKENS=64 MIN_TOKENS=64 STORE_WAIT_S=60 \
  python3 "${root}/run_meaningful_correctness_480k8192.py" \
  > "${result_dir}/workload.jsonl" 2> "${result_dir}/workload.err"
printf 'complete\n' > "${result_dir}/status"
trap - EXIT
cleanup
