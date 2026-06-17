#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT=/tmp/lmcache_patch
TAR=/dev/shm/tutti_lazy_fix_full_structured.tar.gz

sudo rm -rf "${PATCH_ROOT}"
sudo mkdir -p "${PATCH_ROOT}"
sudo tar -xzf "${TAR}" -C "${PATCH_ROOT}"
sudo cp /dev/shm/tutti_direct_loader.py \
  "${PATCH_ROOT}/v1/gpu_connector/tutti_direct_loader.py"
sudo mkdir -p "${PATCH_ROOT}/tests/v1"
sudo cp /dev/shm/test_tutti_direct_loader.py \
  "${PATCH_ROOT}/tests/v1/test_tutti_direct_loader.py"
sudo mkdir -p "${PATCH_ROOT}/docs/design/v1"
sudo cp /dev/shm/tutti_codebase_analysis.md \
  "${PATCH_ROOT}/docs/design/v1/tutti_codebase_analysis.md"
sudo chown -R root:root "${PATCH_ROOT}"
sudo chmod -R a+rX "${PATCH_ROOT}"

echo "== repaired patch root =="
grep -n "def _estimate_chunk_ios" \
  "${PATCH_ROOT}/v1/gpu_connector/tutti_direct_loader.py"
grep -n "while batch_start < n" \
  "${PATCH_ROOT}/v1/gpu_connector/tutti_direct_loader.py"
