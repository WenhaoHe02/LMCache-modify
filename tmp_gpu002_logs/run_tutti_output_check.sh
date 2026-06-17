#!/usr/bin/env bash
set -euo pipefail

CONTAINER=dsv4-256k-measure-tutti

bash /dev/shm/restart_tutti_container_packing.sh

for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "ready_after=${i}"
    break
  fi
  if [ "$i" = "180" ]; then
    echo "service_not_ready" >&2
    sudo docker logs --tail 120 "${CONTAINER}" >&2 || true
    exit 1
  fi
  sleep 5
done

cat >/dev/shm/tutti_output_check.py <<'PY'
import hashlib
import json
import time
import urllib.request

url = "http://127.0.0.1:8000/v1/completions"
prefix = "TUTTI_OUTPUT_CHECK_20260610_UNIQUE_PREFIX"
prompt = (
    prefix
    + "\n"
    + "We are checking whether an SSD KV-cache full hit returns the same text. "
    + "Continue with one concise deterministic sentence.\n"
    + ("0123456789abcdef " * 7000)
)
payload = {
    "model": "deepseek-v4-pro",
    "prompt": prompt,
    "max_tokens": 64,
    "temperature": 0,
    "top_p": 1,
    "stream": False,
}


def post(label: str) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=1200) as resp:
        obj = json.loads(resp.read())
    elapsed = time.time() - start
    text = obj["choices"][0].get("text", "")
    usage = obj.get("usage", {})
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(
        f"=== {label} elapsed={elapsed:.3f}s sha256={digest} usage={usage}",
        flush=True,
    )
    print(text.encode("unicode_escape").decode("ascii"), flush=True)
    return text


cold = post("cold")
time.sleep(35)
hit1 = post("hit1_lazy")
time.sleep(10)
hit2 = post("hit2_steady")

print("=== compare", flush=True)
print(f"cold_vs_hit1_exact={cold == hit1}", flush=True)
print(f"cold_vs_hit2_exact={cold == hit2}", flush=True)
print(f"hit1_vs_hit2_exact={hit1 == hit2}", flush=True)
PY

sudo docker cp /dev/shm/tutti_output_check.py "${CONTAINER}":/tmp/tutti_output_check.py
sudo docker exec "${CONTAINER}" python /tmp/tutti_output_check.py \
  | tee /dev/shm/tutti_output_check_client.txt

sudo docker logs --since 20m "${CONTAINER}" 2>&1 \
  | grep -E "Reqid:|Retrieved [0-9]|LMCACHE_RETRIEVE_PROFILE|TUTTI_PROFILE batched_get" \
  > /dev/shm/tutti_output_check_profile.log || true

echo "=== profile_tail"
tail -n 80 /dev/shm/tutti_output_check_profile.log || true
