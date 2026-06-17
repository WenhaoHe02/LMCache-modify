#!/usr/bin/env bash
set -euo pipefail

echo "== tar candidates =="
ls -lh /dev/shm/*tutti*tar.gz /tmp/*tutti*tar.gz /dev/shm/*full*tar.gz /tmp/*full*tar.gz 2>/dev/null || true

echo "== patch tree =="
find /tmp/lmcache_patch -maxdepth 5 -type f 2>/dev/null | sort | head -n 160 || true

echo "== container key files =="
sudo docker exec dsv4-256k-measure-tutti /bin/bash -lc '
for p in \
  /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_v1_adapter.py \
  /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_service_factory.py \
  /opt/venv/lib/python3.12/site-packages/lmcache/v1/gpu_connector/tutti_direct_loader.py; do
  echo "-- $p"
  ls -l "$p"
done
'
