#!/usr/bin/env bash
set -euo pipefail

root=/home/zbuser02/csa_cp8_ab_20260717
exec 9>"${root}/run_off_repeat_determinism.lock"
flock -n 9 || exit 75

tag="off_repeat_determinism_480000p8192_$(date -u +%Y%m%d_%H%M%S)"
result_dir="${root}/results/${tag}"
container=dsv4-csa-cp8-off
mkdir -p "${result_dir}"
printf '%s\n' "${tag}" > "${root}/current_off_repeat_determinism_tag.txt"
active=0

cleanup() {
  local rc=$?
  if [[ ${active} -eq 1 ]]; then
    sudo docker logs "${container}" > "${result_dir}/cleanup.server.log" 2>&1 || true
    sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  fi
  [[ ${rc} -eq 0 ]] || printf 'failed rc=%d\n' "${rc}" > "${result_dir}/status"
}
trap cleanup EXIT

export LMCACHE_ABLATION_PATCH_DIR="${LMCACHE_ABLATION_PATCH_DIR:-/home/zbuser02/codex_sync_overlap_fix/patches_native_compact_perf_20260721}"
export LMCACHE_ABLATION_STARTUP_SCRIPT="${LMCACHE_ABLATION_STARTUP_SCRIPT:-/home/zbuser02/codex_sync_overlap_fix/startup_native_compact_perf_20260721.sh}"
export LMCACHE_ABLATION_MAX_MODEL_LEN=530000
export LMCACHE_ABLATION_MAX_BATCHED_TOKENS=65536
export LMCACHE_ABLATION_GPU_UTIL=0.55
export LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB=8
export LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY=48000
export LMCACHE_ABLATION_TUTTI_N_SLOTS=4
export LMCACHE_ABLATION_TUTTI_SLOT_MB=128
export LMCACHE_ABLATION_TUTTI_STARTUP_DELAY=1
export LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY=10
export LMCACHE_NSYS_CAPTURE=0
export LMCACHE_NSYS_FULL_CAPTURE=0
export LMCACHE_CSA_PIPELINE_NVTX=0
export LMCACHE_CSA_ATTENTION_KV_TIMING=0
export LMCACHE_TUTTI_PROFILE=0
export LMCACHE_INDEXER_TIMING=0
export LMCACHE_TTFT_STAGE_PROFILE=0
export LMCACHE_HCA_TIMING=0
export LMCACHE_INDEXER_PROFILE_ACCURACY=0
export CUDA_LAUNCH_BLOCKING=0
unset LMCACHE_EXEC_PREFIX

run_once() {
  local label=$1
  local case_dir="${result_dir}/${label}"
  mkdir -p "${case_dir}"
  printf 'launching %s\n' "${label}" > "${result_dir}/status"
  sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  active=1
  bash "${root}/run_container_cp8_ab.sh" off > "${case_dir}/launch.log" 2>&1
  sleep 10
  printf 'running %s\n' "${label}" > "${result_dir}/status"
  MODE_LABEL="${label}" BASE_TOKENS=480000 RECOMPUTE_TOKENS=8192 \
    MAX_TOKENS=64 MIN_TOKENS=64 LOGPROBS=20 STORE_WAIT_S=20 \
    python3 "${root}/run_meaningful_correctness_480k8192.py" \
    > "${case_dir}/workload.jsonl" 2> "${case_dir}/workload.err"
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

run_once off_a
run_once off_b

python3 - "${result_dir}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
hits = {}
prompt_hashes = set()
for label in ("off_a", "off_b"):
    rows = [
        json.loads(line)
        for line in (root / label / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ready = next(row for row in rows if row.get("event") == "prompt_ready")
    hit = next(
        row
        for row in rows
        if row.get("label") == "meaningful_hit_480k8192_trial1"
    )
    prompt_hashes.add(ready["hit_sha256"])
    hits[label] = hit
if len(prompt_hashes) != 1:
    raise SystemExit("OFF repeats used different prompts")
summary = {
    "prompt_sha256": prompt_hashes.pop(),
    "exact_text_match": hits["off_a"]["output_text"] == hits["off_b"]["output_text"],
    "exact_token_match": hits["off_a"]["output_token_ids"]
    == hits["off_b"]["output_token_ids"],
    "off_a": hits["off_a"],
    "off_b": hits["off_b"],
}
(root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
)
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY

printf 'complete\n' > "${result_dir}/status"
trap - EXIT
cleanup
