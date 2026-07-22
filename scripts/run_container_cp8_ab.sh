#!/usr/bin/env bash
set -euo pipefail

case_name=${1:?usage: run_container.sh off|on}
case "$case_name" in
  off)
    filter=0
    hca_walker=0
    profile_accuracy=0
    ;;
  on)
    filter=1
    hca_walker="${LMCACHE_DSV4_HCA_WALKER:-1}"
    profile_accuracy="${LMCACHE_INDEXER_PROFILE_ACCURACY:-1}"
    ;;
  *) echo "unknown case: $case_name" >&2; exit 2 ;;
esac

base=/home/zbuser02/csa_cp8_ab_20260717
patches_host=${LMCACHE_ABLATION_PATCH_DIR:-$base/patches}
startup_host=${LMCACHE_ABLATION_STARTUP_SCRIPT:-$base/startup_cp8_ab.sh}
name="dsv4-csa-cp8-${case_name}"
image="lmcache/vllm-openai:indexer-ssd-hca-prefetch-decodegate-20260528_0630"
model_host=/mnt/nvme0/models/DeepSeek-V4-flash

sudo mount /mnt/nvme0 >/dev/null 2>&1 || true
if [ ! -f "$model_host/config.json" ]; then
  echo "missing model config: $model_host/config.json" >&2
  exit 2
fi

sudo docker rm -f "$name" >/dev/null 2>&1 || true

tutti_device_args=()
if [ -e /dev/snvm_control ]; then
  tutti_device_args+=(--device /dev/snvm_control:/dev/snvm_control)
else
  echo "missing Tutti control device: /dev/snvm_control" >&2
  echo "Both A/B cases require the real Tutti data path." >&2
  exit 2
fi

for dev in /dev/ssnvme*; do
  [ -e "$dev" ] || continue
  tutti_device_args+=(--device "$dev:$dev")
done

csa_prefetch_env_args=()
if [ "$filter" = "1" ]; then
  csa_prefetch_env_args+=(
    -e LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER="${LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER:-profile80}"
    -e LMCACHE_CSA_PREFETCH_CP_SIZE="${LMCACHE_CSA_PREFETCH_CP_SIZE:-8}"
    -e LMCACHE_CSA_PREFETCH_CP_INTERLEAVE="${LMCACHE_CSA_PREFETCH_CP_INTERLEAVE:-64}"
    -e LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE="${LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE:-1}"
    -e LMCACHE_CSA_PREFETCH_BLOCK_BUDGET="${LMCACHE_CSA_PREFETCH_BLOCK_BUDGET:-256}"
    -e LMCACHE_CSA_EXACT_CHUNK_PREFETCH="${LMCACHE_CSA_EXACT_CHUNK_PREFETCH:-0}"
    -e LMCACHE_CSA_EXACT_CHUNK_PREFETCH_MAX_CHUNKS="${LMCACHE_CSA_EXACT_CHUNK_PREFETCH_MAX_CHUNKS:-1}"
  )
fi

sudo docker run -d \
  --name "$name" \
  --gpus all \
  --network host \
  --pid host \
  --ipc host \
  --cap-add SYS_ADMIN \
  --cap-add SYS_RAWIO \
  --privileged \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${tutti_device_args[@]}" \
  "${csa_prefetch_env_args[@]}" \
  -v /sys:/sys \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  -v /mnt:/mnt \
  -v /tmp:/tmp \
  -v "$model_host:/pro_model:ro" \
  -v "$patches_host:/patches:ro" \
  -v "$startup_host:/startup.sh:ro" \
  -e MODEL_PATH=/pro_model \
  -e LMCACHE_ABLATION_CSA_ATTENTION_KV_FILTER="$filter" \
  -e LMCACHE_ABLATION_MAX_MODEL_LEN="${LMCACHE_ABLATION_MAX_MODEL_LEN:-32768}" \
  -e LMCACHE_ABLATION_MAX_BATCHED_TOKENS="${LMCACHE_ABLATION_MAX_BATCHED_TOKENS:-1024}" \
  -e LMCACHE_ABLATION_GPU_UTIL="${LMCACHE_ABLATION_GPU_UTIL:-0.75}" \
  -e LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB="${LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB:-8}" \
  -e LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY="${LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY:-48000}" \
  -e LMCACHE_ABLATION_TUTTI_N_SLOTS="${LMCACHE_ABLATION_TUTTI_N_SLOTS:-4}" \
  -e LMCACHE_ABLATION_TUTTI_SLOT_MB="${LMCACHE_ABLATION_TUTTI_SLOT_MB:-128}" \
  -e LMCACHE_ABLATION_TUTTI_STARTUP_DELAY="${LMCACHE_ABLATION_TUTTI_STARTUP_DELAY:-120}" \
  -e LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY="${LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY:-10}" \
  -e LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES="${LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES:-}" \
  -e LMCACHE_DSV4_HCA_WALKER="$hca_walker" \
  -e LMCACHE_CSA_VALIDATE_AGAINST_GENERIC="${LMCACHE_CSA_VALIDATE_AGAINST_GENERIC:-0}" \
  -e LMCACHE_CSA_DEBUG_TOPK="${LMCACHE_CSA_DEBUG_TOPK:-0}" \
  -e LMCACHE_CSA_VALIDATE_BYTES="${LMCACHE_CSA_VALIDATE_BYTES:-0}" \
  -e LMCACHE_INDEXER_CROSS_LAYER_PREFETCH=0 \
  -e LMCACHE_INDEXER_REUSE_RESIDUAL_TOPK=0 \
  -e LMCACHE_INDEXER_EXPERIMENTAL_RESIDUAL_LOOKAHEAD=0 \
  -e LMCACHE_NATIVE_INDEXER_STAGE0_LAYERS="${LMCACHE_NATIVE_INDEXER_STAGE0_LAYERS:-1}" \
  -e LMCACHE_NATIVE_INDEXER_WINDOW_LAYERS="${LMCACHE_NATIVE_INDEXER_WINDOW_LAYERS:-21}" \
  -e LMCACHE_CSA_PIPELINE_NVTX="${LMCACHE_CSA_PIPELINE_NVTX:-0}" \
  -e LMCACHE_INDEXER_TIMING="${LMCACHE_INDEXER_TIMING:-0}" \
  -e LMCACHE_INDEXER_PROFILE_ACCURACY="$profile_accuracy" \
  -e LMCACHE_CSA_PREDICTION_GATE_TIMEOUT_SEC="${LMCACHE_CSA_PREDICTION_GATE_TIMEOUT_SEC:-5}" \
  -e LMCACHE_NSYS_CAPTURE="${LMCACHE_NSYS_CAPTURE:-0}" \
  -e LMCACHE_NSYS_CAPTURE_SKIP_REQUESTS="${LMCACHE_NSYS_CAPTURE_SKIP_REQUESTS:-0}" \
  -e LMCACHE_NSYS_FULL_CAPTURE="${LMCACHE_NSYS_FULL_CAPTURE:-0}" \
  -e LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS="${LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS:-0}" \
  -e LMCACHE_NSYS_FULL_CAPTURE_SCOPE="${LMCACHE_NSYS_FULL_CAPTURE_SCOPE:-decoder}" \
  -e LMCACHE_EXEC_PREFIX="${LMCACHE_EXEC_PREFIX:-}" \
  -e LMCACHE_CSA_ATTENTION_KV_TIMING="${LMCACHE_CSA_ATTENTION_KV_TIMING:-0}" \
  -e LMCACHE_TUTTI_PROFILE="${LMCACHE_TUTTI_PROFILE:-0}" \
  -e LMCACHE_TTFT_STAGE_PROFILE="${LMCACHE_TTFT_STAGE_PROFILE:-0}" \
  -e LMCACHE_HCA_TIMING="${LMCACHE_HCA_TIMING:-0}" \
  -e LMCACHE_HCA_ENABLE_DECODE_HOOK=0 \
  -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  --entrypoint /bin/bash \
  "$image" /startup.sh

for i in $(seq 1 360); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "ready_after_seconds=$((i * 2))"
    curl -sf http://127.0.0.1:8000/v1/models
    exit 0
  fi
  sleep 2
done

echo "not ready" >&2
sudo docker logs --tail 240 "$name" >&2 || true
exit 1
