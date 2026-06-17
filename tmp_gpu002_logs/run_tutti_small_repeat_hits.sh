#!/usr/bin/env bash
set -euo pipefail

cat >/dev/shm/tutti_small_repeat_hits.py <<'PY'
import json
import time
import urllib.request

url = "http://127.0.0.1:8000/v1/completions"
prefix = "TUTTI_MULTI_EXTENT_SMALL_DIRECT_20260609_UNIQUE_PREFIX"
payload = {
    "model": "deepseek-v4-pro",
    "prompt": prefix + "\n" + ("0123456789abcdef " * 7000),
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
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read()
    elapsed = time.time() - start
    obj = json.loads(body)
    print(label, "sec", round(elapsed, 3), "usage", obj.get("usage", {}))

for i in range(3):
    post(f"repeat_hit_{i+1}")
    time.sleep(5)
PY

sudo docker cp /dev/shm/tutti_small_repeat_hits.py \
  dsv4-256k-measure-tutti:/tmp/tutti_small_repeat_hits.py
sudo docker exec dsv4-256k-measure-tutti python /tmp/tutti_small_repeat_hits.py

echo "== relevant logs =="
sudo docker logs --since 5m dsv4-256k-measure-tutti 2>&1 \
  | grep -E "LMCache hit tokens|need to load|Retrieved|Tutti lazy pre-scan|TuttiDirectLoader initialised|LBA pre-scan|NVMe READ|direct load" \
  | tail -n 180 || true
