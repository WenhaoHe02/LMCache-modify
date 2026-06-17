#!/usr/bin/env bash
set -euo pipefail

cat >/dev/shm/tutti_warmup_check.py <<'PY'
import json
import time
import urllib.request

url = "http://127.0.0.1:8000/v1/completions"
prefix = "TUTTI_COLD_STORE_WARMUP_20260610_UNIQUE_PREFIX"
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
    with urllib.request.urlopen(req, timeout=900) as resp:
        body = resp.read()
    elapsed = time.time() - start
    obj = json.loads(body)
    print(label, "sec", round(elapsed, 3), "usage", obj.get("usage", {}), flush=True)

post("cold")
time.sleep(35)
post("first_hit_after_warmup")
PY

sudo docker cp /dev/shm/tutti_warmup_check.py \
  dsv4-256k-measure-tutti:/tmp/tutti_warmup_check.py
sudo docker exec dsv4-256k-measure-tutti python /tmp/tutti_warmup_check.py

echo "== relevant logs =="
sudo docker logs --since 8m dsv4-256k-measure-tutti 2>&1 \
  | grep -E "Tutti warmup|Waiting for in-flight|LMCache hit tokens|need to load|Retrieved|Tutti lazy pre-scan|TuttiDirectLoader initialised|unmounting|Stored" \
  | tail -n 260 || true
