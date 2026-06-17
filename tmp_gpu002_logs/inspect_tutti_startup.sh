#!/usr/bin/env bash
set -euo pipefail

echo "==== /tmp/startup_256k_tutti.sh env/config ===="
grep -nE \
  'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL|extra_config|tutti_|max_model_len|gpu_memory' \
  /tmp/startup_256k_tutti.sh || true

echo "==== container env if running ===="
sudo docker exec dsv4-256k-measure-tutti env 2>/dev/null \
  | grep -E 'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL' \
  | sort || true

echo "==== recent feature logs ===="
sudo docker logs --tail 300 dsv4-256k-measure-tutti 2>&1 \
  | grep -E \
    'IndexerSSDManager|reuse prefetch|residual_proxy|HCAPrefetch|DEFER_HCA|enabled pinned|PREFETCH_PREFILL|DECODE_PREFETCH|LMCACHE_INDEXER|LMCACHE_HCA' \
  | tail -n 120 || true
