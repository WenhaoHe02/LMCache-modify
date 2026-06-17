#!/usr/bin/env bash
set -euo pipefail

container="${1:-dsv4-256k-measure-tutti}"

echo "==== host restart startup mount ===="
grep -n "startup_256k" /dev/shm/restart_tutti_container_csa_hca.sh || true

echo "==== docker inspect mounts ===="
sudo docker inspect "$container" \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' \
  | grep -E 'startup|patch|lmcache|nvme' || true

echo "==== docker inspect env ===="
sudo docker inspect "$container" \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E 'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL' \
  | sort || true

echo "==== container startup env lines ===="
sudo docker exec "$container" bash -lc \
  "grep -nE 'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL' /tmp/startup_256k_tutti.sh | sort" \
  || true

echo "==== container env command ===="
sudo docker exec "$container" bash -lc \
  "env | grep -E 'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL' | sort" \
  || true

echo "==== vllm process env ===="
pid="$(sudo docker exec "$container" bash -lc "pgrep -f 'vllm|api_server' | head -n 1" || true)"
if [[ -n "$pid" ]]; then
  sudo docker exec "$container" bash -lc \
    "tr '\0' '\n' </proc/$pid/environ | grep -E 'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL' | sort" \
    || true
else
  echo "no vllm/api_server pid found"
fi

echo "==== recent startup/log feature lines ===="
sudo docker logs --since 20m "$container" 2>&1 \
  | grep -E 'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL|IndexerSSDManager|HCAPrefetch|DEFER_HCA|residual_proxy' \
  | tail -n 180 || true
