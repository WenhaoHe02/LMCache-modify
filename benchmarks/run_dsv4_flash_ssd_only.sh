#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Launch the GPU002 V4 Flash baseline on the traditional LMCache SSD path.

set -euo pipefail

source_launcher=${SOURCE_LAUNCHER:-\
/home/zbuser02/codex_sync_overlap_fix/run_container_off_raw_20260711.sh}
temporary_launcher=$(mktemp /tmp/run_container_ssd_only.XXXXXX.sh)
trap 'rm -f "$temporary_launcher"' EXIT
ssd_startup=${SSD_STARTUP:-\
/home/zbuser02/startup_dsv4_flash_ssd_only_wrapper.sh}

# Mount the original startup as input to the SSD-only wrapper, then mount the
# wrapper as the container entrypoint. The wrapper removes Tutti and object
# store configuration while preserving the model/vLLM settings.
sed \
  -e 's#"$base/startup_csa_prefill_tutti.sh:/startup.sh:ro"#"$base/startup_csa_prefill_tutti.sh:/startup_tutti_original.sh:ro" \\\n  -v "'"$ssd_startup"':/startup.sh:ro"#' \
  "$source_launcher" > "$temporary_launcher"

if ! grep -q 'startup_tutti_original.sh' "$temporary_launcher"; then
  echo "failed to inject the SSD-only startup wrapper" >&2
  exit 1
fi
if [[ ${DRY_RUN:-0} == 1 ]]; then
  cat "$temporary_launcher"
  exit 0
fi

export LMCACHE_ABLATION_MODEL_HOST=${LMCACHE_ABLATION_MODEL_HOST:-\
/mnt/nvme0/models/DeepSeek-V4-flash}
export LMCACHE_ABLATION_MAX_MODEL_LEN=${LMCACHE_ABLATION_MAX_MODEL_LEN:-131072}
export LMCACHE_ABLATION_MAX_BATCHED_TOKENS=${\
LMCACHE_ABLATION_MAX_BATCHED_TOKENS:-2048}
export LMCACHE_ABLATION_GPU_UTIL=${LMCACHE_ABLATION_GPU_UTIL:-0.85}
export LMCACHE_LOG_MOE_TIMING=${LMCACHE_LOG_MOE_TIMING:-0}

bash "$temporary_launcher" off
