#!/usr/bin/env bash
set -euo pipefail

container=dsv4-256k-measure-tutti

for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "ready_after=${i}"
    break
  fi
  sleep 5
done

cat >/dev/shm/tutti_csa_hca_unique_long.py <<'PY'
import hashlib
import json
import time
import urllib.error
import urllib.request

url = "http://127.0.0.1:8000/v1/completions"
run_id = f"CSA_HCA_TUTTI_UNIQUE_{int(time.time())}"


def build_prompt(n: int) -> str:
    body = " ".join(f"x{i:06x}" for i in range(n))
    return (
        run_id
        + "\nLong unique prompt for Tutti CSA/HCA full-hit correctness.\n"
        + body
    )


def post(label: str, prompt: str) -> tuple[str, dict, float]:
    payload = {
        "model": "deepseek-v4-pro",
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
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
    return text, usage, elapsed


prompt = ""
last_error = None
for n in [42000, 36000, 30000, 24000, 18000, 12000]:
    candidate = build_prompt(n)
    try:
        cold, usage, _ = post(f"cold_n{n}", candidate)
        prompt = candidate
        break
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:800]
        last_error = f"n={n} http={exc.code} body={body}"
        print(f"=== cold_failed {last_error}", flush=True)
        time.sleep(3)

if not prompt:
    raise RuntimeError(f"all cold sizes failed: {last_error}")

time.sleep(40)
hit1, usage1, _ = post("hit1_lazy", prompt)
time.sleep(12)
hit2, usage2, _ = post("hit2_steady", prompt)

print("=== compare", flush=True)
print(f"cold_vs_hit1_exact={cold == hit1}", flush=True)
print(f"cold_vs_hit2_exact={cold == hit2}", flush=True)
print(f"hit1_vs_hit2_exact={hit1 == hit2}", flush=True)
print(f"hit1_prompt_tokens={usage1.get('prompt_tokens')}", flush=True)
print(f"hit2_prompt_tokens={usage2.get('prompt_tokens')}", flush=True)
PY

sudo docker cp /dev/shm/tutti_csa_hca_unique_long.py "$container":/tmp/tutti_csa_hca_unique_long.py
sudo docker exec "$container" python /tmp/tutti_csa_hca_unique_long.py \
  | tee /dev/shm/tutti_csa_hca_unique_long_client.txt

echo "==== summary ===="
sudo docker logs --since 45m "$container" 2>&1 \
  | grep -E 'Tutti direct load failed|Tutti direct load found no readable|Reqid:|Retrieved [0-9]|LMCACHE_RETRIEVE_PROFILE|reuse_prefetch_seed failed|reuse_prefetch_seed complete|reuse prefetch seeded|reuse prefetch prepared|residual_proxy prefill|event=prefill_fire_async |HCAPrefetchManager: (seeded|fire|drain)' \
  | tail -n 320 || true
