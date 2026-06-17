#!/usr/bin/env bash
set -euo pipefail

echo "== marker =="
sudo docker exec dsv4-256k-measure-tutti python - <<'PY'
from pathlib import Path
p = Path('/opt/venv/lib/python3.12/site-packages/lmcache/v1/gpu_connector/tutti_direct_loader.py')
s = p.read_text()
print('multi_extent_marker', 'def _slot_iova_with_offset' in s, '_MAX_EXTENTS: int = 256' in s)
PY

cat >/dev/shm/tutti_44k_direct_check.py <<'PY'
import json
import time
import urllib.request

url = "http://127.0.0.1:8000/v1/completions"
prefix = "TUTTI_MULTI_EXTENT_44K_DIRECT_20260609_UNIQUE_PREFIX"
payload = {
    "model": "deepseek-v4-pro",
    "prompt": prefix + "\n" + ("0123456789abcdef " * 24000),
    "max_tokens": 1,
    "temperature": 0,
    "stream": False,
}

def post(label):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read()
            elapsed = time.time() - start
            obj = json.loads(body)
            usage = obj.get("usage", {})
            print(label, "status", resp.status, "sec", round(elapsed, 3), "usage", usage)
    except Exception as exc:
        elapsed = time.time() - start
        print(label, "ERROR", type(exc).__name__, str(exc), "sec", round(elapsed, 3))
        raise

post("cold")
time.sleep(5)
post("hit")
PY

sudo docker cp /dev/shm/tutti_44k_direct_check.py \
  dsv4-256k-measure-tutti:/tmp/tutti_44k_direct_check.py
sudo docker exec dsv4-256k-measure-tutti python /tmp/tutti_44k_direct_check.py

echo "== relevant logs =="
sudo docker logs --since 20m dsv4-256k-measure-tutti 2>&1 \
  | grep -E "Tutti|LMCache hit tokens|need to load|Retrieved|LBA pre-scan|FIEMAP|NVMe READ|direct load" \
  | tail -n 220 || true
