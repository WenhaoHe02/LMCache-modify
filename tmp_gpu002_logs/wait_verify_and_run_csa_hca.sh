#!/usr/bin/env bash
set -euo pipefail

container=dsv4-256k-measure-tutti

echo "==== wait server ===="
for i in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/tmp/tutti_models.json 2>/dev/null; then
    cat /tmp/tutti_models.json
    echo
    break
  fi
  if ! sudo docker ps --format '{{.Names}}' | grep -Fxq "$container"; then
    echo "container exited"
    sudo docker logs --tail 220 "$container" || true
    exit 1
  fi
  sleep 10
done

echo "==== patched indexer _open ===="
sudo docker exec "$container" bash -lc "python - <<'PY'
import inspect
import lmcache.v1.indexer_ssd_manager as m
print(m.__file__)
print(inspect.getsource(m.IndexerBlockStore._open))
print(inspect.getsource(m.IndexerBlockStore._ensure_file))
PY"

echo "==== vllm env ===="
pid="$(sudo docker exec "$container" bash -lc "pgrep -f 'vllm|api_server' | head -n 1")"
sudo docker exec "$container" bash -lc \
  "tr '\0' '\n' </proc/$pid/environ | grep -E 'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL' | sort"

echo "==== run output check ===="
bash /dev/shm/run_tutti_output_check_no_restart.sh

echo "==== feature logs ===="
sudo docker logs --since 30m "$container" 2>&1 \
  | grep -E 'reuse_prefetch_seed failed|FileNotFoundError|previous prefill eviction failed|reuse prefetch seeded|reuse prefetch prepared|residual_proxy prefill|prefill_fire_async|HCAPrefetchManager: (seeded|fire|drain)|TUTTI_PROFILE|LMCACHE_RETRIEVE_PROFILE|Retrieved [0-9]' \
  | tail -n 320 || true
