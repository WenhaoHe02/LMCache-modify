#!/usr/bin/env bash
set -euo pipefail

CONTAINER=dsv4-256k-measure-tutti

for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "ready_after=${i}s"
    break
  fi
  if ! sudo docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "container_exited"
    sudo docker logs "${CONTAINER}" --tail 200 || true
    exit 1
  fi
  if (( i % 20 == 0 )); then
    echo "waiting ${i}s"
    sudo docker logs "${CONTAINER}" --tail 8 || true
  fi
  sleep 1
done

curl -fsS http://127.0.0.1:8000/v1/models | head -c 400
echo
echo "== container patch markers =="
sudo docker exec "${CONTAINER}" bash -lc \
  "grep -n 'def _estimate_chunk_ios\\|while batch_start < n\\|_staging_iova_at' /opt/venv/lib/python3.12/site-packages/lmcache/v1/gpu_connector/tutti_direct_loader.py"
