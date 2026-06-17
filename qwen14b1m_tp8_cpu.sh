#!/usr/bin/env bash
set -euo pipefail

docker rm -f qwen14b1m-tp8 2>/dev/null || true

docker run -d --name qwen14b1m-tp8 \
  --gpus all \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e PYTHONHASHSEED=0 \
  -e LMCACHE_USE_EXPERIMENTAL=True \
  -e LMCACHE_CONFIG_FILE=/config/lmcache.yaml \
  -e LMCACHE_LOG_LEVEL=INFO \
  -v /mnt/nvme0/models/Qwen2.5-14B-Instruct-1M-vllm-upstream:/model:ro \
  -v /mnt/nvme0/lmcache_cpu.yaml:/config/lmcache.yaml:ro \
  -v /root/lmcache-glp/lmcache:/opt/venv/lib/python3.12/site-packages/lmcache \
  --entrypoint vllm lmcache/vllm-openai:latest-nightly \
  serve /model \
  --served-model-name qwen14b1m \
  --tensor-parallel-size 8 \
  --max-model-len 500000 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 500000 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1Dynamic","kv_role":"kv_both","kv_connector_module_path":"lmcache.integration.vllm.lmcache_connector_v1"}'

echo "Container started. Waiting for ready (up to 300s)..."
for i in $(seq 1 150); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "READY after ${i}x2s"
    exit 0
  fi
  sleep 2
done

echo NOT_READY
docker logs --tail 50 qwen14b1m-tp8
exit 1
