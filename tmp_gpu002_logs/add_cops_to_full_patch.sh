#!/usr/bin/env bash
set -euo pipefail

BASE=/dev/shm/full_patch_with_cops
rm -rf "$BASE"
mkdir -p "$BASE"
tar -xzf /dev/shm/tutti_lazy_fix_full_structured.tar.gz -C "$BASE"

cp /tmp/lmcache_build/lmcache/c_ops.cpython-312-x86_64-linux-gnu.so \
  "$BASE/c_ops.cpython-312-x86_64-linux-gnu.so"

tar -czf /dev/shm/tutti_lazy_fix_full_structured.tar.gz -C "$BASE" .

echo "== rebuilt marker =="
tar -tzf /dev/shm/tutti_lazy_fix_full_structured.tar.gz | grep -E \
  'c_ops.cpython-312-x86_64-linux-gnu.so$|vllm_v1_adapter.py$|v1/gpu_connector/tutti_direct_loader.py$' \
  | sort
ls -lh "$BASE/c_ops.cpython-312-x86_64-linux-gnu.so"
