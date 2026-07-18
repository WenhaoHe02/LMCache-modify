#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Launch the GPU002 V4 Flash baseline on LMCache's cuFile GDS backend.

set -euo pipefail

source_launcher=${SOURCE_LAUNCHER:-\
/home/zbuser02/codex_sync_overlap_fix/run_container_off_raw_20260711.sh}
gds_startup=${GDS_STARTUP:-\
/home/zbuser02/startup_dsv4_flash_gds_wrapper.sh}
host_cufile_lib=${HOST_CUFILE_LIB:-\
/usr/local/cuda-13.2/targets/x86_64-linux/lib}
temporary_launcher=$(mktemp /tmp/run_container_gds.XXXXXX.sh)
trap 'rm -f "$temporary_launcher"' EXIT

# Give this baseline its own container and inject only the GDS-specific mounts.
sed \
  -e 's#name="dsv4-csa-prefill-${case_name}"#name="${LMCACHE_ABLATION_CONTAINER_NAME:-dsv4-lmcache-gds}"#' \
  -e 's#"$base/startup_csa_prefill_tutti.sh:/startup.sh:ro"#"$base/startup_csa_prefill_tutti.sh:/startup_tutti_original.sh:ro" \\\n  -v "'"$gds_startup"':/startup.sh:ro" \\\n  -v "'"$host_cufile_lib"':/opt/cuda-gds/lib:ro" \\\n  -v "/etc/cufile.json:/etc/cufile.json:ro"#' \
  -e 's#  -e MODEL_PATH=/pro_model \\#  -e LMCACHE_GDS_RUN_ID="${LMCACHE_GDS_RUN_ID:-agent_trace_gds_20260715}" \\\n  -e LMCACHE_GDS_BUFFER_MB="${LMCACHE_GDS_BUFFER_MB:-8192}" \\\n  -e LMCACHE_GDS_IO_THREADS="${LMCACHE_GDS_IO_THREADS:-4}" \\\n  -e MODEL_PATH=/pro_model \\#' \
  "$source_launcher" > "$temporary_launcher"

if ! grep -q 'startup_tutti_original.sh' "$temporary_launcher"; then
  echo "failed to inject the GDS startup wrapper" >&2
  exit 1
fi
if ! grep -q '/opt/cuda-gds/lib:ro' "$temporary_launcher"; then
  echo "failed to mount the host cuFile library" >&2
  exit 1
fi
if ! grep -q 'name="${LMCACHE_ABLATION_CONTAINER_NAME:-dsv4-lmcache-gds}"' \
  "$temporary_launcher"; then
  echo "failed to isolate the GDS container name" >&2
  exit 1
fi
if [[ ${DRY_RUN:-0} == 1 ]]; then
  cat "$temporary_launcher"
  exit 0
fi

if [[ ! -f $host_cufile_lib/libcufile.so ]]; then
  echo "missing host cuFile library: $host_cufile_lib/libcufile.so" >&2
  exit 2
fi
if [[ ! -c /dev/nvidia-fs0 ]]; then
  echo "missing nvidia-fs device; GDS is not ready" >&2
  exit 2
fi

# Never disturb another benchmark. The generated launcher only removes its own
# uniquely named container, and these guards refuse to start on a busy host.
if ss -ltnH 'sport = :8000' | grep -q .; then
  echo "port 8000 is already in use; refusing to start GDS benchmark" >&2
  exit 3
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "GPU compute processes are active; refusing to start GDS benchmark" >&2
  exit 3
fi

export LMCACHE_ABLATION_MODEL_HOST=${LMCACHE_ABLATION_MODEL_HOST:-\
/mnt/nvme0/models/DeepSeek-V4-flash}
export LMCACHE_ABLATION_MAX_MODEL_LEN=${LMCACHE_ABLATION_MAX_MODEL_LEN:-131072}
export LMCACHE_ABLATION_MAX_BATCHED_TOKENS=${\
LMCACHE_ABLATION_MAX_BATCHED_TOKENS:-2048}
export LMCACHE_ABLATION_GPU_UTIL=${LMCACHE_ABLATION_GPU_UTIL:-0.85}
export LMCACHE_LOG_MOE_TIMING=${LMCACHE_LOG_MOE_TIMING:-0}
export LMCACHE_GDS_RUN_ID=${LMCACHE_GDS_RUN_ID:-agent_trace_gds_20260715}
export LMCACHE_GDS_BUFFER_MB=${LMCACHE_GDS_BUFFER_MB:-8192}
export LMCACHE_GDS_IO_THREADS=${LMCACHE_GDS_IO_THREADS:-4}

bash "$temporary_launcher" off
