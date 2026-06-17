#!/usr/bin/env bash
set -euo pipefail

container=dsv4-256k-measure-tutti

sudo docker rm -f "$container" >/dev/null 2>&1 || true
sleep 3
bash /dev/shm/restart_tutti_container_csa_hca_off.sh

for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "ready_after=${i}"
    break
  fi
  if ! sudo docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    echo "container exited"
    sudo docker logs --tail 240 "$container" || true
    exit 1
  fi
  sleep 5
done

echo "==== vllm env ===="
pid="$(sudo docker exec "$container" bash -lc "pgrep -f 'vllm|api_server' | head -n 1")"
sudo docker exec "$container" bash -lc \
  "tr '\0' '\n' </proc/$pid/environ | grep -E 'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL' | sort"
