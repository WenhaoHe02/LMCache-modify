#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT=/tmp/lmcache_patch

sudo mkdir -p \
  "${PATCH_ROOT}/v1/gpu_connector" \
  "${PATCH_ROOT}/tests/v1" \
  "${PATCH_ROOT}/docs/design/v1"

sudo cp /dev/shm/tutti_direct_loader.py \
  "${PATCH_ROOT}/v1/gpu_connector/tutti_direct_loader.py"
sudo cp /dev/shm/test_tutti_direct_loader.py \
  "${PATCH_ROOT}/tests/v1/test_tutti_direct_loader.py"
sudo cp /dev/shm/tutti_codebase_analysis.md \
  "${PATCH_ROOT}/docs/design/v1/tutti_codebase_analysis.md"

echo "== deployed markers =="
grep -n "def _estimate_chunk_ios" \
  "${PATCH_ROOT}/v1/gpu_connector/tutti_direct_loader.py"
grep -n "while batch_start < n" \
  "${PATCH_ROOT}/v1/gpu_connector/tutti_direct_loader.py"
grep -n "_staging_iova_at" \
  "${PATCH_ROOT}/v1/gpu_connector/tutti_direct_loader.py"
