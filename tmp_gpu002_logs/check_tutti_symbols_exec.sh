#!/usr/bin/env bash
set -euo pipefail

cat >/dev/shm/check_tutti_symbols.py <<'PY'
from pathlib import Path
import lmcache.c_ops as c_ops

p = Path('/opt/venv/lib/python3.12/site-packages/lmcache/v1/gpu_connector/tutti_direct_loader.py')
s = p.read_text()
print('multi_extent', '_MAX_EXTENTS: int = 256' in s, 'def _slot_iova_with_offset' in s, flush=True)
print('c_ops_tutti_submit', hasattr(c_ops, 'tutti_submit_batch_sgl_read'), flush=True)
print('c_ops_tutti_poll', hasattr(c_ops, 'tutti_poll_batch'), flush=True)
PY

sudo docker cp /dev/shm/check_tutti_symbols.py dsv4-256k-measure-tutti:/tmp/check_tutti_symbols.py
sudo docker exec dsv4-256k-measure-tutti python /tmp/check_tutti_symbols.py
