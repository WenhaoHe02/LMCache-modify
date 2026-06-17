#!/usr/bin/env bash
set -euo pipefail

BASE=/dev/shm/lmcache_full_patch_rebuild
rm -rf "$BASE"
mkdir -p "$BASE"

# Start from the current full patch tree if available.  It contains the
# vLLM/integration patches needed for DSv4 long-context store.
if [[ -d /tmp/lmcache_patch ]]; then
  cp -a /tmp/lmcache_patch/. "$BASE"/
fi

# Overlay the newly uploaded multi-extent Tutti files.  Keep both layouts:
# startup copies /patches/v1, while some ad-hoc scripts inspect /patches/lmcache.
mkdir -p "$BASE/v1/gpu_connector" "$BASE/v1/storage_backend"
cp /tmp/lmcache_patch/v1/cache_engine.py "$BASE/v1/cache_engine.py"
cp /tmp/lmcache_patch/v1/gpu_connector/tutti_direct_loader.py \
  "$BASE/v1/gpu_connector/tutti_direct_loader.py"
cp /tmp/lmcache_patch/v1/storage_backend/local_disk_backend.py \
  "$BASE/v1/storage_backend/local_disk_backend.py"

mkdir -p "$BASE/lmcache/v1/gpu_connector" "$BASE/lmcache/v1/storage_backend"
cp "$BASE/v1/cache_engine.py" "$BASE/lmcache/v1/cache_engine.py"
cp "$BASE/v1/gpu_connector/tutti_direct_loader.py" \
  "$BASE/lmcache/v1/gpu_connector/tutti_direct_loader.py"
cp "$BASE/v1/storage_backend/local_disk_backend.py" \
  "$BASE/lmcache/v1/storage_backend/local_disk_backend.py"

tar -czf /dev/shm/tutti_lazy_fix_full_structured.tar.gz -C "$BASE" .
echo "rebuilt /dev/shm/tutti_lazy_fix_full_structured.tar.gz"
tar -tzf /dev/shm/tutti_lazy_fix_full_structured.tar.gz | sed -n '1,80p'
