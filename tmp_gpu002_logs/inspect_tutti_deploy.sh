#!/usr/bin/env bash
set -euo pipefail

echo "== files =="
ls -l /dev/shm/restart_tutti_container.sh /tmp/startup_256k_tutti.sh \
  /dev/shm/tutti_lazy_fix_full_structured.tar.gz || true

echo "== restart script =="
sed -n '1,240p' /dev/shm/restart_tutti_container.sh || true

echo "== startup script =="
sed -n '1,220p' /tmp/startup_256k_tutti.sh || true

echo "== tar contents =="
tar -tzf /dev/shm/tutti_lazy_fix_full_structured.tar.gz | sed -n '1,40p' || true

echo "== /tmp/lmcache_patch =="
find /tmp/lmcache_patch -maxdepth 6 -type f 2>/dev/null | sed -n '1,80p' || true

echo "== container source markers =="
sudo docker exec dsv4-256k-measure-tutti /bin/bash -lc '
set -e
for p in \
  /opt/venv/lib/python3.12/site-packages/lmcache/v1/gpu_connector/tutti_direct_loader.py \
  /tmp/lmcache_patch/lmcache/v1/gpu_connector/tutti_direct_loader.py; do
  echo "-- $p"
  if [[ -f "$p" ]]; then
    grep -n "_MAX_EXTENTS\|def _slot_iova_with_offset\|scan_paths\|single_contiguous(path)\|query_extents(path)" "$p" | head -n 30
  else
    echo missing
  fi
done
'
