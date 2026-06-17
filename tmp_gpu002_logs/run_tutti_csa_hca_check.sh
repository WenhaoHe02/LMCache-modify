#!/usr/bin/env bash
set -euo pipefail

bash /dev/shm/run_tutti_output_check_no_restart.sh

echo "==== ENV ===="
sudo docker exec dsv4-256k-measure-tutti env \
  | grep -E 'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL' \
  | sort || true

echo "==== FEATURE_LOGS ===="
sudo docker logs --since 30m dsv4-256k-measure-tutti 2>&1 \
  | grep -E \
    'IndexerSSDManager|reuse prefetch|residual_proxy|HCAPrefetch|DEFER_HCA|enabled pinned|prefill rows|PREFETCH_PREFILL|DECODE_PREFETCH|TUTTI_PROFILE|LMCACHE_RETRIEVE_PROFILE|Retrieved [0-9]' \
  | tail -n 260 || true
