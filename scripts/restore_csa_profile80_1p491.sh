#!/usr/bin/env bash
set -euo pipefail

safe_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
protected_image="lmcache/csa-prefetch:protected-1p491-20260727"
runner_src="$safe_root/scripts/run_container_cp8_ab.sh"
runner_tmp=$(mktemp /tmp/run_container_1p491.XXXXXX.sh)
trap 'rm -f "$runner_tmp"' EXIT

if ! docker image inspect "$protected_image" >/dev/null 2>&1; then
  sha256sum -c "$safe_root/protected_image.tar.zst.sha256"
  zstd -dc "$safe_root/protected_image.tar.zst" | docker load
fi

for path in /dev/snvm_control /dev/ssnvme{0..7}; do
  if [[ ! -e "$path" ]]; then
    echo "missing required Tutti device: $path" >&2
    exit 2
  fi
done

if [[ ! -f /mnt/nvme0/models/DeepSeek-V4-flash/config.json ]]; then
  echo "missing model: /mnt/nvme0/models/DeepSeek-V4-flash" >&2
  exit 2
fi

sed \
  's#^image=.*#image="lmcache/csa-prefetch:protected-1p491-20260727"#' \
  "$runner_src" > "$runner_tmp"
chmod +x "$runner_tmp"

LMCACHE_ABLATION_PATCH_DIR="$safe_root/patches" \
LMCACHE_ABLATION_STARTUP_SCRIPT="$safe_root/scripts/startup_cp8_ab.sh" \
LMCACHE_ABLATION_MAX_MODEL_LEN=530000 \
LMCACHE_ABLATION_MAX_BATCHED_TOKENS=65536 \
LMCACHE_ABLATION_GPU_UTIL=0.55 \
LMCACHE_ABLATION_TUTTI_STARTUP_DELAY=120 \
LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY=10 \
LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER=profile80_hybrid \
LMCACHE_CSA_PREFETCH_CP_SIZE=8 \
LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=64 \
LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE=1 \
LMCACHE_CSA_PREFETCH_CP_EXCHANGE_IDS=1 \
LMCACHE_CSA_PREFETCH_BLOCK_BUDGET=2048 \
LMCACHE_CSA_L1_PROXY_TOPK_TOKENS=2048 \
LMCACHE_CSA_PROXY_TOPK_TOKENS_BY_LAYER=28:2048 \
LMCACHE_INDEXER_PROFILE_ACCURACY=1 \
LMCACHE_CSA_PREDICTION_GATE_TIMEOUT_SEC=5 \
LMCACHE_NATIVE_INDEXER_STAGE0_LAYERS=1 \
LMCACHE_NATIVE_INDEXER_WINDOW_LAYERS=21 \
LMCACHE_NSYS_CAPTURE=0 \
LMCACHE_NSYS_FULL_CAPTURE=0 \
LMCACHE_TUTTI_PROFILE=0 \
LMCACHE_TTFT_STAGE_PROFILE=0 \
bash "$runner_tmp" on
