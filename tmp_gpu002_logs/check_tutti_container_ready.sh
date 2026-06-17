#!/usr/bin/env bash
set -euo pipefail

for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "ready_after=${i}s"
    sudo docker exec dsv4-256k-measure-tutti python - <<'PY'
from pathlib import Path
p = Path('/opt/venv/lib/python3.12/site-packages/lmcache/v1/gpu_connector/tutti_direct_loader.py')
s = p.read_text()
print('multi_extent_marker', 'def _slot_iova_with_offset' in s, '_MAX_EXTENTS: int = 256' in s)
PY
    exit 0
  fi
  sleep 1
done

echo "not_ready"
sudo docker logs --tail 80 dsv4-256k-measure-tutti || true
exit 1
