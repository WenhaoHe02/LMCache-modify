#!/usr/bin/env bash
set -euo pipefail

root=/home/zbuser02/csa_cp8_ab_20260717
lookahead_policy="${LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER:-default:0}"
policy_tag="$(printf '%s' "${lookahead_policy}" | tr -c '[:alnum:]' '_')"
tag="on_extension_correctness_${policy_tag}_480000p8192p1024_$(date -u +%Y%m%d_%H%M%S)"
result_dir="${root}/results/${tag}"
container=dsv4-csa-cp8-on
mkdir -p "${result_dir}"
printf '%s\n' "${tag}" > "${root}/current_extension_correctness_tag.txt"

cleanup() {
  local rc=$?
  sudo docker logs "${container}" > "${result_dir}/server.log" 2>&1 || true
  sudo docker inspect "${container}" \
    > "${result_dir}/container_inspect.json" 2>&1 || true
  sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  [[ ${rc} -eq 0 ]] || printf 'failed rc=%d\n' "${rc}" > "${result_dir}/status"
}
trap cleanup EXIT

export LMCACHE_ABLATION_PATCH_DIR="/home/zbuser02/codex_sync_overlap_fix/patches_native_compact_perf_20260721"
export LMCACHE_ABLATION_STARTUP_SCRIPT="/home/zbuser02/codex_sync_overlap_fix/startup_native_compact_perf_20260721.sh"
export LMCACHE_ABLATION_MAX_MODEL_LEN=530000
export LMCACHE_ABLATION_MAX_BATCHED_TOKENS=65536
export LMCACHE_ABLATION_GPU_UTIL=0.55
export LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB=8
export LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY=48000
export LMCACHE_ABLATION_TUTTI_N_SLOTS=4
export LMCACHE_ABLATION_TUTTI_SLOT_MB=128
export LMCACHE_ABLATION_TUTTI_STARTUP_DELAY=1
export LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY=10
export LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER="${lookahead_policy}"
export LMCACHE_CSA_PREFETCH_CP_SIZE=8
export LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=64
export LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE=1
export LMCACHE_CSA_PREFETCH_BLOCK_BUDGET=256
export LMCACHE_DSV4_HCA_WALKER=1
export LMCACHE_INDEXER_PROFILE_ACCURACY=0
export LMCACHE_NSYS_CAPTURE=0
export LMCACHE_NSYS_FULL_CAPTURE=0
export LMCACHE_CSA_PIPELINE_NVTX=0
export LMCACHE_CSA_ATTENTION_KV_TIMING=0
export LMCACHE_TUTTI_PROFILE=0
export LMCACHE_INDEXER_TIMING=0
export LMCACHE_TTFT_STAGE_PROFILE=0
export LMCACHE_HCA_TIMING=0
export CUDA_LAUNCH_BLOCKING=0
unset LMCACHE_EXEC_PREFIX

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  2>/dev/null | grep -Eq '[0-9]'; then
  echo 'GPU is not idle' >&2
  exit 3
fi

printf 'launching\n' > "${result_dir}/status"
sudo docker rm -f "${container}" >/dev/null 2>&1 || true
bash "${root}/run_container_cp8_ab.sh" on > "${result_dir}/launch.log" 2>&1
sleep 10
printf 'running\n' > "${result_dir}/status"
MODE_LABEL=on BASE_TOKENS=480000 FIRST_SUFFIX_TOKENS=8192 \
  EXTENSION_TOKENS=1024 MAX_TOKENS=96 STORE_WAIT_S=20 ADMISSION_WAIT_S=15 \
  python3 "${root}/run_extension_admission_correctness_480k8192.py" \
  > "${result_dir}/workload.jsonl" 2> "${result_dir}/workload.err"

log="${result_dir}/server.log"
sudo docker logs "${container}" > "${log}" 2>&1
grep -q 'LMCache hit tokens: 488192' "${log}"
grep -q 'mode=deferred_hit' "${log}"
if grep -Eq 'Traceback \(most recent call last\)|CUDA error|object_unreadable' "${log}"; then
  echo 'extension correctness log contains a fatal marker' >&2
  exit 4
fi
printf 'complete\n' > "${result_dir}/status"
trap - EXIT
cleanup
