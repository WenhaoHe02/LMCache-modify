#!/usr/bin/env bash
set -euo pipefail

for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "ready_after=${i}s"
    break
  fi
  sleep 1
done

sudo docker exec dsv4-256k-measure-tutti python - <<'PY'
from pathlib import Path
import lmcache.c_ops as c_ops
p = Path('/opt/venv/lib/python3.12/site-packages/lmcache/v1/gpu_connector/tutti_direct_loader.py')
s = p.read_text()
print('multi_extent', '_MAX_EXTENTS: int = 256' in s, 'def _slot_iova_with_offset' in s)
print('c_ops_tutti_submit', hasattr(c_ops, 'tutti_submit_batch_sgl_read'))
print('c_ops_tutti_poll', hasattr(c_ops, 'tutti_poll_batch'))
PY
