#!/usr/bin/env bash
set -euo pipefail

NEW=/tmp/lmcache_patch
BASE=/dev/shm/lmcache_full_patch_rebuild

choose_base() {
  for tarball in \
    /dev/shm/tutti_lazy_fix.tar.gz \
    /dev/shm/tutti_fix_full_interface.tar.gz \
    /tmp/tutti_patch.tar.gz \
    /dev/shm/tutti_fix_patch.tar.gz; do
    [[ -f "$tarball" ]] || continue
    if tar -tzf "$tarball" | grep -Eq '(^|/)integration/vllm/vllm_v1_adapter.py$|^vllm_v1_adapter.py$'; then
      echo "$tarball"
      return 0
    fi
  done
  return 1
}

OLD_TAR=$(choose_base)
echo "old_base=$OLD_TAR"

rm -rf "$BASE"
mkdir -p "$BASE"
tar -xzf "$OLD_TAR" -C "$BASE"

echo "== base before overlay =="
find "$BASE" -maxdepth 5 -type f | sort | head -n 120

# Overlay the new multi-extent Tutti/cache-engine/local-disk files.
mkdir -p "$BASE/v1/gpu_connector" "$BASE/v1/storage_backend"
cp "$NEW/v1/cache_engine.py" "$BASE/v1/cache_engine.py"
cp "$NEW/v1/gpu_connector/tutti_direct_loader.py" \
  "$BASE/v1/gpu_connector/tutti_direct_loader.py"
cp "$NEW/v1/storage_backend/local_disk_backend.py" \
  "$BASE/v1/storage_backend/local_disk_backend.py"

# Keep lmcache/ layout too, harmless for startup and useful for inspection.
mkdir -p "$BASE/lmcache/v1/gpu_connector" "$BASE/lmcache/v1/storage_backend"
cp "$BASE/v1/cache_engine.py" "$BASE/lmcache/v1/cache_engine.py"
cp "$BASE/v1/gpu_connector/tutti_direct_loader.py" \
  "$BASE/lmcache/v1/gpu_connector/tutti_direct_loader.py"
cp "$BASE/v1/storage_backend/local_disk_backend.py" \
  "$BASE/lmcache/v1/storage_backend/local_disk_backend.py"

tar -czf /dev/shm/tutti_lazy_fix_full_structured.tar.gz -C "$BASE" .

echo "== rebuilt contents =="
tar -tzf /dev/shm/tutti_lazy_fix_full_structured.tar.gz | grep -E \
  'vllm_v1_adapter.py$|vllm_service_factory.py$|v1/cache_engine.py$|v1/gpu_connector/tutti_direct_loader.py$|v1/storage_backend/local_disk_backend.py$' \
  | sort

echo "== marker in rebuilt tutti =="
grep -n "_MAX_EXTENTS\|def _slot_iova_with_offset\|query_extents(path)\|single_contiguous(path)" \
  "$BASE/v1/gpu_connector/tutti_direct_loader.py" | head -n 30
