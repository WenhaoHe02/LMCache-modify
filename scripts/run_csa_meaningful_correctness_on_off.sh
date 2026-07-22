#!/usr/bin/env bash
set -euo pipefail

root=/home/zbuser02/csa_cp8_ab_20260717
exec 9>"${root}/run_csa_meaningful_correctness.lock"
if ! flock -n 9; then
  echo 'another meaningful correctness harness is already running' >&2
  exit 75
fi

tag="csa_meaningful_correctness_480000p8192_$(date -u +%Y%m%d_%H%M%S)"
result_dir="${root}/results/${tag}"
mkdir -p "${result_dir}"
printf '%s\n' "${tag}" > "${root}/current_meaningful_correctness_tag.txt"
printf 'starting\n' > "${result_dir}/status"
active_container=""

cleanup() {
  local rc=$?
  if [[ -n "${active_container}" ]] \
    && sudo docker inspect "${active_container}" >/dev/null 2>&1; then
    sudo docker logs "${active_container}" \
      > "${result_dir}/${active_container}.cleanup.log" 2>&1 || true
    sudo docker stop -t 90 "${active_container}" >/dev/null 2>&1 || true
    sudo docker rm -f "${active_container}" >/dev/null 2>&1 || true
  fi
  if [[ ${rc} -ne 0 ]]; then
    printf 'failed rc=%d\n' "${rc}" > "${result_dir}/status"
  fi
}
trap cleanup EXIT

export LMCACHE_ABLATION_PATCH_DIR="${LMCACHE_ABLATION_PATCH_DIR:-${root}/patches}"
export LMCACHE_ABLATION_STARTUP_SCRIPT="${LMCACHE_ABLATION_STARTUP_SCRIPT:-${root}/startup_cp8_ab.sh}"
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
export LMCACHE_NSYS_CAPTURE=0
export LMCACHE_NSYS_CAPTURE_SKIP_REQUESTS=0
export LMCACHE_NSYS_FULL_CAPTURE=0
export LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS=0
export LMCACHE_CSA_PIPELINE_NVTX=0
export LMCACHE_CSA_ATTENTION_KV_TIMING=0
export LMCACHE_TUTTI_PROFILE=0
export LMCACHE_INDEXER_TIMING=0
export LMCACHE_TTFT_STAGE_PROFILE=0
export LMCACHE_HCA_TIMING=0
export CUDA_LAUNCH_BLOCKING=0
unset LMCACHE_EXEC_PREFIX

run_mode() {
  local mode=$1
  local mode_dir="${result_dir}/${mode}"
  local container="dsv4-csa-cp8-${mode}"
  mkdir -p "${mode_dir}"
  active_container="${container}"
  sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU is not idle before ${mode}" >&2
    return 3
  fi
  if [[ "${mode}" == "on" ]]; then
    export LMCACHE_DSV4_HCA_WALKER=1
    export LMCACHE_INDEXER_PROFILE_ACCURACY=1
  else
    export LMCACHE_DSV4_HCA_WALKER=0
    export LMCACHE_INDEXER_PROFILE_ACCURACY=0
  fi
  printf 'launching %s\n' "${mode}" > "${result_dir}/status"
  bash "${root}/run_container_cp8_ab.sh" "${mode}" \
    > "${mode_dir}/launch.log" 2>&1
  local container_pid
  container_pid=$(sudo docker inspect -f '{{.State.Pid}}' "${container}")
  sudo sh -c "tr '\000' '\n' </proc/${container_pid}/environ" \
    > "${mode_dir}/process_env.txt"
  sleep 20
  printf 'running %s\n' "${mode}" > "${result_dir}/status"
  MODE_LABEL="${mode}" BASE_TOKENS=480000 RECOMPUTE_TOKENS=8192 \
    MAX_TOKENS=768 MIN_TOKENS=256 STORE_WAIT_S=60 \
    python3 "${root}/run_meaningful_correctness_480k8192.py" \
    > "${mode_dir}/workload.jsonl" \
    2> "${mode_dir}/workload.err"
  sudo docker logs "${container}" > "${mode_dir}/server.log" 2>&1 || true
  sudo docker inspect "${container}" > "${mode_dir}/container_inspect.json" 2>&1 || true
  for marker in 'CUDA error' 'illegal memory access' 'out of memory' \
    'Traceback (most recent call last)' \
    'Retrieved 488192 out of 488192 required tokens'; do
    if grep -Fq "${marker}" "${mode_dir}/server.log"; then
      echo "invalid ${mode} server marker: ${marker}" >&2
      return 7
    fi
  done
  sudo docker stop -t 90 "${container}" >/dev/null 2>&1 || true
  sudo docker rm -f "${container}" >/dev/null 2>&1 || true
  active_container=""
}

run_mode off
run_mode on

python3 - "${result_dir}" <<'PY'
import json
from pathlib import Path
import sys

result_dir = Path(sys.argv[1])


def load(mode: str) -> tuple[dict, dict]:
    rows = [
        json.loads(line)
        for line in (result_dir / mode / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ready = next(row for row in rows if row.get("event") == "prompt_ready")
    hit = next(
        row
        for row in rows
    if row.get("label") == "meaningful_hit_480k8192_trial1"
    )
    return ready, hit


off_ready, off_hit = load("off")
on_ready, on_hit = load("on")
if off_ready["hit_sha256"] != on_ready["hit_sha256"]:
    raise SystemExit("ON and OFF did not use the same prompt")
if off_ready["hit_tokens"] != 488192 or on_ready["hit_tokens"] != 488192:
    raise SystemExit("wrong hit prompt length")
for mode, hit in (("off", off_hit), ("on", on_hit)):
    usage = hit.get("usage") or {}
    if hit.get("status") != 200:
        raise SystemExit(f"{mode} request failed")
    if int(usage.get("prompt_tokens", 0)) != 488192:
        raise SystemExit(f"{mode} reported the wrong prompt length")
    if int(usage.get("completion_tokens", 0)) < 128:
        raise SystemExit(f"{mode} decoded fewer than 128 tokens")
    if "1073" not in hit.get("output_text", ""):
        raise SystemExit(f"{mode} answer does not contain the arithmetic result")
exact_text = off_hit["output_text"] == on_hit["output_text"]
exact_tokens = off_hit["output_token_ids"] == on_hit["output_token_ids"]
summary = {
    "prompt_sha256": off_ready["hit_sha256"],
    "off_ttft_s": off_hit["ttft_s"],
    "on_ttft_s": on_hit["ttft_s"],
    "off_elapsed_s": off_hit["elapsed_s"],
    "on_elapsed_s": on_hit["elapsed_s"],
    "off_completion_tokens": off_hit["usage"]["completion_tokens"],
    "on_completion_tokens": on_hit["usage"]["completion_tokens"],
    "off_output_sha256": off_hit["output_sha256"],
    "on_output_sha256": on_hit["output_sha256"],
    "exact_text_match": exact_text,
    "exact_token_match": exact_tokens,
    "off_output_text": off_hit["output_text"],
    "on_output_text": on_hit["output_text"],
}
(result_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
)
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
if not exact_text or not exact_tokens:
    raise SystemExit("ON and OFF outputs differ")
PY

printf 'complete\n' > "${result_dir}/status"
printf '%s\n' "${tag}" > "${root}/completed_meaningful_correctness_tag.txt"
trap - EXIT
cleanup
