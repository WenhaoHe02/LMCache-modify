#!/usr/bin/env bash
set -euo pipefail

echo "== gpu002 =="
hostname

PATCH_ROOT=/tmp/lmcache_patch
CONTAINER=dsv4-256k-measure-tutti

echo "== container =="
sudo docker ps --format '{{.Names}}	{{.Status}}' | grep -E "${CONTAINER}|dsv4" || true

echo "== patch root =="
if [[ ! -d "${PATCH_ROOT}" ]]; then
  sudo mkdir -p "${PATCH_ROOT}"
fi
sudo rm -rf "${PATCH_ROOT}/v1/gpu_connector"
sudo mkdir -p "${PATCH_ROOT}/v1/gpu_connector" "${PATCH_ROOT}/tests/v1"

echo "ready for scp payload"
