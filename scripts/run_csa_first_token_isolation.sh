#!/usr/bin/env bash
set -euo pipefail

root=/home/zbuser02/csa_cp8_ab_20260717
exec 9>"${root}/run_csa_first_token_isolation.lock"
if ! flock -n 9; then
  echo 'another first-token isolation run is active' >&2
  exit 75
fi

tag="csa_first_token_isolation_$(date -u +%Y%m%d_%H%M%S)"
result_dir="${root}/results/${tag}"
container=dsv4-csa-cp8-on
mkdir -p "${result_dir}"
printf '%s\n' "${tag}" > "${root}/current_first_token_isolation_tag.txt"
printf 'starting\n' > "${result_dir}/status"
active=0

cleanup() {
  local rc=$?
  if [[ ${active} -eq 1 ]]; then
    sudo docker logs "${container}" > "${result_dir}/cleanup.server.log" 2>&1 || true
    sudo docker stop -t 90 "${container}" >/dev/null 2>&1 || true
    sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  fi
  if [[ ${rc} -ne 0 ]]; then
    printf 'failed rc=%d\n' "${rc}" > "${result_dir}/status"
  fi
}
trap cleanup EXIT

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
export LMCACHE_CSA_PREFETCH_CP_SIZE=8
export LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=64
export LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE=1
export LMCACHE_CSA_PREFETCH_BLOCK_BUDGET=256
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

run_case() {
  local name=$1
  local lookahead=$2
  local hca=$3
  local case_dir="${result_dir}/${name}"
  mkdir -p "${case_dir}"
  printf 'launching %s\n' "${name}" > "${result_dir}/status"
  sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  export LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER="${lookahead}"
  export LMCACHE_DSV4_HCA_WALKER="${hca}"
  active=1
  bash "${root}/run_container_cp8_ab.sh" on \
    > "${case_dir}/launch.log" 2>&1
  local container_pid
  container_pid=$(sudo docker inspect -f '{{.State.Pid}}' "${container}")
  sudo sh -c "tr '\000' '\n' </proc/${container_pid}/environ" \
    > "${case_dir}/process_env.txt"
  sleep 15
  printf 'running %s\n' "${name}" > "${result_dir}/status"
  MODE_LABEL="${name}" BASE_TOKENS=480000 RECOMPUTE_TOKENS=8192 \
    MAX_TOKENS=1 MIN_TOKENS=1 STORE_WAIT_S=15 \
    python3 "${root}/run_meaningful_correctness_480k8192.py" \
    > "${case_dir}/workload.jsonl" \
    2> "${case_dir}/workload.err"
  sudo docker logs "${container}" > "${case_dir}/server.log" 2>&1 || true
  sudo docker stop -t 90 "${container}" >/dev/null 2>&1 || true
  sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  active=0
}

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  2>/dev/null | grep -Eq '[0-9]'; then
  echo 'GPU is not idle' >&2
  exit 3
fi

# "on_base" retains the ON storage/filter machinery but disables both
# speculative L2 prediction and the HCA walker.
run_case on_base default:0 0
run_case csa_prediction profile80 0
run_case hca_walker default:0 1

python3 - "${result_dir}" <<'PY'
import json
from pathlib import Path
import sys

result_dir = Path(sys.argv[1])
summary = {}
prompt_hashes = set()
for name in ("on_base", "csa_prediction", "hca_walker"):
    rows = [
        json.loads(line)
        for line in (result_dir / name / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ready = next(row for row in rows if row.get("event") == "prompt_ready")
    hit = next(row for row in rows if row.get("label") == "meaningful_hit_480k8192")
    prompt_hashes.add(ready["hit_sha256"])
    summary[name] = {
        "elapsed_s": hit["elapsed_s"],
        "output_text": hit["output_text"],
        "output_token_ids": hit["output_token_ids"],
        "matches_known_off_first_token": hit["output_token_ids"] == [128822],
    }
if len(prompt_hashes) != 1:
    raise SystemExit("isolation cases used different prompts")
summary["prompt_sha256"] = prompt_hashes.pop()
(result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, sort_keys=True))
PY

printf 'complete\n' > "${result_dir}/status"
printf '%s\n' "${tag}" > "${root}/completed_first_token_isolation_tag.txt"
trap - EXIT
cleanup
