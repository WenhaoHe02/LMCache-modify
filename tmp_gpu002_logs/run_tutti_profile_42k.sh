#!/usr/bin/env bash
set -euo pipefail
cat >/dev/shm/tutti_profile_42k.py <<'PY'
import json
import time
import urllib.request

url = "http://127.0.0.1:8000/v1/completions"
prefix = "TUTTI_PROFILE_42K_20260610_UNIQUE_PREFIX"
payload = {
    "model": "deepseek-v4-pro",
    "prompt": prefix + "\n" + ("0123456789abcdef " * 7000),
    "max_tokens": 1,
    "temperature": 0,
    "stream": False,
}

def post(label):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        obj = json.loads(resp.read())
    print(label, round(time.time() - start, 3), obj.get("usage", {}), flush=True)

post("cold")
time.sleep(35)
post("hit1_lazy")
time.sleep(10)
post("hit2_steady")
PY
sudo docker cp /dev/shm/tutti_profile_42k.py dsv4-256k-measure-tutti:/tmp/tutti_profile_42k.py
sudo docker exec dsv4-256k-measure-tutti python /tmp/tutti_profile_42k.py
sudo docker logs --since 8m dsv4-256k-measure-tutti 2>&1 \
  | grep -E "TUTTI_PROFILE|LMCACHE_RETRIEVE_PROFILE|Reqid:|Retrieved [0-9]" \
  > /dev/shm/tutti_profile_42k.log || true
cat /dev/shm/tutti_profile_42k.log
