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
image="${LMCACHE_ABLATION_IMAGE:-lmcache/vllm-openai:indexer-ssd-hca-prefetch-decodegate-20260528_0630}"
model_host=${LMCACHE_ABLATION_MODEL_HOST:-/mnt/dockerdisk/models/DeepSeek-V4-flash}
model_container=/pro_model
runtime_cache_host=${LMCACHE_RUNTIME_CACHE_HOST:-/home/zbuser02/lmcache_v026_20260806_run/runtime_cache/vllm_v0260}
mkdir -p "$runtime_cache_host/vllm" "$runtime_cache_host/flashinfer"
# This legacy recovery service remounts every fstab entry every 20 seconds.
# It must stay inactive while snvme owns the cache controllers.
sudo systemctl stop lmcache-remount.service 2>/dev/null || true

if [ ! -f "$model_host/config.json" ]; then
  echo "missing model config: $model_host/config.json" >&2
  exit 2
fi

# The worker synchronously unmounts each cache filesystem before snvme takes
# ownership. Propagate those eight unmounts into the host mount namespace;
# otherwise ext4 remains live while SNVM_DEVICE_BIND detaches its controller.
# Share only the cache mountpoints: /mnt/dockerdisk contains Docker's data root
# and must never participate in the Tutti mount handoff.
cache_mounts=(
  /mnt/nvme0
  /mnt/nvme2
  /mnt/nvme3
  /mnt/nvme4
  /mnt/nvme5
  /mnt/nvme6
  /mnt/nvme8
  /mnt/nvme9
)
cache_mount_args=()
for cache_mount in "${cache_mounts[@]}"; do
  sudo mount "$cache_mount" >/dev/null 2>&1 || true
done
for cache_mount in "${cache_mounts[@]}"; do
  if ! mountpoint -q "$cache_mount"; then
    echo "cache filesystem is not mounted: $cache_mount" >&2
    exit 2
  fi
  propagation=$(findmnt -n -o PROPAGATION --target "$cache_mount" | head -n 1)
  if [[ "$propagation" != *shared* ]]; then
    sudo mount --bind "$cache_mount" "$cache_mount"
    sudo mount --make-shared "$cache_mount"
  fi
  cache_mount_args+=(
    --mount
    "type=bind,source=$cache_mount,target=$cache_mount,bind-propagation=rshared"
  )
done

# Let vLLM/LMCache run Tutti teardown before removing the stopped container.
sudo docker stop -t "${LMCACHE_DOCKER_STOP_TIMEOUT_SEC:-60}" "$name" \
  >/dev/null 2>&1 || true
sudo docker rm "$name" >/dev/null 2>&1 || true

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

csa_prefetch_env_args=(
  -e LMCACHE_SSD_TP_SHARDED_PREFETCH="${LMCACHE_SSD_TP_SHARDED_PREFETCH:-0}"
  -e LMCACHE_SSD_TP_SHARD_INDEXER="${LMCACHE_SSD_TP_SHARD_INDEXER:-0}"
  -e LMCACHE_SSD_TP_SHARD_CSA="${LMCACHE_SSD_TP_SHARD_CSA:-1}"
  -e LMCACHE_SSD_TP_CSA_REPLICA_VERIFIED="${LMCACHE_SSD_TP_CSA_REPLICA_VERIFIED:-0}"
  -e LMCACHE_SSD_TP_INDEXER_CP_VERIFIED="${LMCACHE_SSD_TP_INDEXER_CP_VERIFIED:-0}"
  -e LMCACHE_SSD_TP_CP_SIZE="${LMCACHE_SSD_TP_CP_SIZE:-8}"
  -e LMCACHE_SSD_TP_CP_INTERLEAVE="${LMCACHE_SSD_TP_CP_INTERLEAVE:-64}"
  -e LMCACHE_SSD_TP_DENSE_LAYERS="${LMCACHE_SSD_TP_DENSE_LAYERS-}"
  -e LMCACHE_SSD_TP_DENSE_EAGER_GROUP_SIZE="${LMCACHE_SSD_TP_DENSE_EAGER_GROUP_SIZE:-1}"
  -e LMCACHE_SSD_TP_MIN_UNION_BLOCKS="${LMCACHE_SSD_TP_MIN_UNION_BLOCKS:-128}"
  -e LMCACHE_SSD_TP_LOCAL_DIRECT_COVERAGE_RATIO="${LMCACHE_SSD_TP_LOCAL_DIRECT_COVERAGE_RATIO:-0}"
  -e LMCACHE_SSD_TP_EARLY_COLLECTIVE="${LMCACHE_SSD_TP_EARLY_COLLECTIVE:-0}"
  -e LMCACHE_SSD_TP_DENSE_EARLY_COLLECTIVE="${LMCACHE_SSD_TP_DENSE_EARLY_COLLECTIVE:-0}"
  -e LMCACHE_SSD_TP_COLLECTIVE_STREAM_PRIORITY="${LMCACHE_SSD_TP_COLLECTIVE_STREAM_PRIORITY:-0}"
  -e LMCACHE_SSD_TP_STAGING_SLOT_BYTES="${LMCACHE_SSD_TP_STAGING_SLOT_BYTES:-134217728}"
  -e LMCACHE_SSD_TP_STAGING_SLOTS="${LMCACHE_SSD_TP_STAGING_SLOTS:-2}"
  -e LMCACHE_SSD_TP_EARLY_LOOKAHEAD="${LMCACHE_SSD_TP_EARLY_LOOKAHEAD:-2}"
  -e LMCACHE_SSD_TP_DEBUG_VERIFY="${LMCACHE_SSD_TP_DEBUG_VERIFY:-0}"
)
case "${LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH:-0}" in
  1|true|TRUE|yes|YES|on|ON) glm_dsa_enabled=1 ;;
  *) glm_dsa_enabled=0 ;;
esac
case "${LMCACHE_GLM_DSA_LAYER_MAJOR:-0}" in
  1|true|TRUE|yes|YES|on|ON) glm_dsa_layer_major_enabled=1 ;;
  *) glm_dsa_layer_major_enabled=$glm_dsa_enabled ;;
esac
tutti_handoff_id=${LMCACHE_TUTTI_HANDOFF_ID:-$(date +%s%N)}
tutti_handoff_dir="/tmp/lmcache_tutti_mount_handoff_${tutti_handoff_id}"
if [[ -e "$tutti_handoff_dir" ]]; then
  echo "Tutti handoff directory already exists: $tutti_handoff_dir" >&2
  exit 2
fi
mkdir -p "$tutti_handoff_dir"

# Mount namespace reassociation from a multi-threaded worker can fail with
# EINVAL. Keep the host handoff in this single host-side process instead: all
# eight ranks first release their container mounts and publish rank markers;
# only then are the host ext4 mounts synchronously released and host.ready is
# published. The low-level SNVM_DEVICE_BIND path independently requires that
# final marker.
(
  deadline=$((SECONDS + ${LMCACHE_TUTTI_HOST_HANDOFF_TIMEOUT_SEC:-1200}))
  while true; do
    rank_ready=$(find "$tutti_handoff_dir" -maxdepth 1 \
      -name 'mnt_nvme*.ready' -type f 2>/dev/null | wc -l)
    if [ "$rank_ready" -ge 8 ]; then
      break
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      printf 'timed out waiting for rank mount markers: ready=%s/8\n' \
        "$rank_ready" > "$tutti_handoff_dir/host.error"
      exit 1
    fi
    sleep 0.05
  done
  for cache_mount in "${cache_mounts[@]}"; do
    if mountpoint -q "$cache_mount"; then
      if ! sudo umount "$cache_mount"; then
        printf 'synchronous host umount failed: %s\n' "$cache_mount" \
          > "$tutti_handoff_dir/host.error"
        exit 1
      fi
    fi
  done
  for cache_mount in "${cache_mounts[@]}"; do
    if mountpoint -q "$cache_mount"; then
      printf 'host mount remains active: %s\n' "$cache_mount" \
        > "$tutti_handoff_dir/host.error"
      exit 1
    fi
  done
  touch "$tutti_handoff_dir/host.ready"
) >"$tutti_handoff_dir/host.log" 2>&1 &
if [ "$filter" = "1" ] || [ "$glm_dsa_layer_major_enabled" = "1" ]; then
  csa_prefetch_env_args+=(
    -e LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER="${LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER:-profile80_hybrid}"
    -e LMCACHE_CSA_PREFETCH_CP_SIZE="${LMCACHE_CSA_PREFETCH_CP_SIZE:-8}"
    -e LMCACHE_CSA_PREFETCH_CP_INTERLEAVE="${LMCACHE_CSA_PREFETCH_CP_INTERLEAVE:-64}"
    -e LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE="${LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE:-1}"
    -e LMCACHE_CSA_PREFETCH_CP_MODE="${LMCACHE_CSA_PREFETCH_CP_MODE:-query}"
    -e LMCACHE_CSA_PREDICTION_GATE="${LMCACHE_CSA_PREDICTION_GATE:-}"
    -e LMCACHE_CSA_PROXY_WORKSPACE_SLOTS="${LMCACHE_CSA_PROXY_WORKSPACE_SLOTS:-4}"
    -e LMCACHE_CSA_OWNER_BLOCKS_PER_RANK="${LMCACHE_CSA_OWNER_BLOCKS_PER_RANK:-64}"
    -e LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE="${LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE:-1}"
    -e LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE_BY_LAYER="${LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE_BY_LAYER:-}"
    -e LMCACHE_CSA_PROXY_HC_PREWARM_ROWS="${LMCACHE_CSA_PROXY_HC_PREWARM_ROWS:-}"
    -e LMCACHE_CSA_PREFETCH_CP_EXCHANGE_IDS="${LMCACHE_CSA_PREFETCH_CP_EXCHANGE_IDS:-1}"
    -e LMCACHE_CSA_PREFETCH_CP_GLOBAL_BLOCK_COUNTS="${LMCACHE_CSA_PREFETCH_CP_GLOBAL_BLOCK_COUNTS:-0}"
    -e LMCACHE_CSA_PREFETCH_CP_GLOBAL_BLOCK_BITMAP="${LMCACHE_CSA_PREFETCH_CP_GLOBAL_BLOCK_BITMAP:-0}"
    -e LMCACHE_CSA_PREFETCH_BLOCK_BUDGET="${LMCACHE_CSA_PREFETCH_BLOCK_BUDGET:-2048}"
    -e LMCACHE_CSA_L1_PROXY_TOPK_TOKENS="${LMCACHE_CSA_L1_PROXY_TOPK_TOKENS:-2048}"
    -e LMCACHE_CSA_PROXY_TOPK_TOKENS_BY_LAYER="${LMCACHE_CSA_PROXY_TOPK_TOKENS_BY_LAYER:-28:2048}"
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
  "${cache_mount_args[@]}" \
  -v /sys:/sys \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  -v /tmp:/tmp \
  -v "$runtime_cache_host/vllm:/root/.cache/vllm" \
  -v "$runtime_cache_host/flashinfer:/root/.cache/flashinfer" \
  -v "$model_host:$model_container:ro" \
  -v "$patches_host:/patches:ro" \
  -v "$startup_host:/startup.sh:ro" \
  -e MODEL_PATH="$model_container" \
  -e LMCACHE_STORAGE_MODE="${LMCACHE_STORAGE_MODE:-tutti}" \
  -e LMCACHE_SSD_LOCAL_CPU_GB="${LMCACHE_SSD_LOCAL_CPU_GB:-5.0}" \
  -e LMCACHE_SSD_INTERNAL_API_ENABLED="${LMCACHE_SSD_INTERNAL_API_ENABLED:-false}" \
  -e LMCACHE_SSD_DSV4_OPTIMIZED_KV="${LMCACHE_SSD_DSV4_OPTIMIZED_KV:-true}" \
  -e LMCACHE_SSD_DSV4_OPTIMIZED_TAIL_TOKENS="${LMCACHE_SSD_DSV4_OPTIMIZED_TAIL_TOKENS:-256}" \
  -e LMCACHE_SSD_DSV4_DEFER_HCA_TO_MOE="${LMCACHE_SSD_DSV4_DEFER_HCA_TO_MOE:-false}" \
  -e LMCACHE_ABLATION_CSA_ATTENTION_KV_FILTER="$filter" \
  -e LMCACHE_ABLATION_MAX_MODEL_LEN="${LMCACHE_ABLATION_MAX_MODEL_LEN:-32768}" \
  -e LMCACHE_ABLATION_MAX_BATCHED_TOKENS="${LMCACHE_ABLATION_MAX_BATCHED_TOKENS:-1024}" \
  -e LMCACHE_ABLATION_GPU_UTIL="${LMCACHE_ABLATION_GPU_UTIL:-0.75}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-300}" \
  -e VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-relax}" \
  -e LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB="${LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB:-8}" \
  -e LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY="${LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY:-48000}" \
  -e LMCACHE_TUTTI_RAW_REGION_BYTES="${LMCACHE_TUTTI_RAW_REGION_BYTES:-25769803776}" \
  -e LMCACHE_INDEXER_RAW_REGION_BYTES="${LMCACHE_INDEXER_RAW_REGION_BYTES:-536870912}" \
  -e LMCACHE_ABLATION_TUTTI_N_SLOTS="${LMCACHE_ABLATION_TUTTI_N_SLOTS:-4}" \
  -e LMCACHE_ABLATION_TUTTI_SLOT_MB="${LMCACHE_ABLATION_TUTTI_SLOT_MB:-128}" \
  -e LMCACHE_ABLATION_TUTTI_STARTUP_DELAY="${LMCACHE_ABLATION_TUTTI_STARTUP_DELAY:-120}" \
  -e LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY="${LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY:-10}" \
  -e LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES="${LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES:-}" \
  -e LMCACHE_DSV4_HCA_WALKER="$hca_walker" \
  -e LMCACHE_HCA_PREFIRE_FIRST_LAYER="${LMCACHE_HCA_PREFIRE_FIRST_LAYER:-0}" \
  -e LMCACHE_HCA_PREFIRE_ALL_LAYERS="${LMCACHE_HCA_PREFIRE_ALL_LAYERS:-0}" \
  -e LMCACHE_CSA_ADAPTIVE_DENSE_PREFETCH="${LMCACHE_CSA_ADAPTIVE_DENSE_PREFETCH:-0}" \
  -e LMCACHE_CSA_ADAPTIVE_DENSE_THRESHOLD_PERCENT="${LMCACHE_CSA_ADAPTIVE_DENSE_THRESHOLD_PERCENT:-80}" \
  -e LMCACHE_CSA_VALIDATE_AGAINST_GENERIC="${LMCACHE_CSA_VALIDATE_AGAINST_GENERIC:-0}" \
  -e LMCACHE_CSA_DEBUG_TOPK="${LMCACHE_CSA_DEBUG_TOPK:-0}" \
  -e LMCACHE_CSA_VALIDATE_BYTES="${LMCACHE_CSA_VALIDATE_BYTES:-0}" \
  -e LMCACHE_INDEXER_CROSS_LAYER_PREFETCH=0 \
  -e LMCACHE_GLM_DSA_LAYER_MAJOR="${LMCACHE_GLM_DSA_LAYER_MAJOR:-0}" \
  -e LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH="${LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH:-0}" \
  -e LMCACHE_GLM_DSA_BOOTSTRAP_SOURCE_LAYER="${LMCACHE_GLM_DSA_BOOTSTRAP_SOURCE_LAYER:-0}" \
  -e LMCACHE_GLM_DSA_FULL_LAYER_LOOKAHEAD="${LMCACHE_GLM_DSA_FULL_LAYER_LOOKAHEAD:-1}" \
  -e LMCACHE_GLM_DSA_PHYSICAL_PREDICTION="${LMCACHE_GLM_DSA_PHYSICAL_PREDICTION:-1}" \
  -e LMCACHE_GLM_DSA_ACCURACY_ROWS="${LMCACHE_GLM_DSA_ACCURACY_ROWS:-32}" \
  -e LMCACHE_GLM_DSA_IO_WORKERS="${LMCACHE_GLM_DSA_IO_WORKERS:-4}" \
  -e LMCACHE_GLM_DSA_PREFETCH_BLOCK_BUDGET="${LMCACHE_GLM_DSA_PREFETCH_BLOCK_BUDGET:-2048}" \
  -e LMCACHE_GLM_DSA_PREDICT_SHARED_CONSUMERS="${LMCACHE_GLM_DSA_PREDICT_SHARED_CONSUMERS:-1}" \
  -e LMCACHE_GLM_DSA_GATE_TIMEOUT_SEC="${LMCACHE_GLM_DSA_GATE_TIMEOUT_SEC:-30}" \
  -e LMCACHE_GLM_DSA_OWNER_PARTITION="${LMCACHE_GLM_DSA_OWNER_PARTITION:-0}" \
  -e LMCACHE_GLM_DSA_ASYNC_PREDICTION="${LMCACHE_GLM_DSA_ASYNC_PREDICTION:-0}" \
  -e LMCACHE_GLM_DSA_SHARED_CORRECTION_AT_CONSUMER="${LMCACHE_GLM_DSA_SHARED_CORRECTION_AT_CONSUMER:-0}" \
  -e LMCACHE_GLM_DSA_SHARE_TOPK_SELECTION="${LMCACHE_GLM_DSA_SHARE_TOPK_SELECTION:-1}" \
  -e LMCACHE_GLM_DSA_PREFIRE_OWNER_GATHER="${LMCACHE_GLM_DSA_PREFIRE_OWNER_GATHER:-0}" \
  -e LMCACHE_GLM_DSA_OWNER_READ_WORKERS="${LMCACHE_GLM_DSA_OWNER_READ_WORKERS:-1}" \
  -e LMCACHE_GLM_DSA_CPU_K_BOUNDS="${LMCACHE_GLM_DSA_CPU_K_BOUNDS:-1}" \
  -e LMCACHE_GLM_DSA_ASYNC_SHARED_CORRECTION="${LMCACHE_GLM_DSA_ASYNC_SHARED_CORRECTION:-0}" \
  -e LMCACHE_CSA_OWNER_GPU_METADATA="${LMCACHE_CSA_OWNER_GPU_METADATA:-0}" \
  -e LMCACHE_CSA_OWNER_APPEND_RESERVE_BLOCKS="${LMCACHE_CSA_OWNER_APPEND_RESERVE_BLOCKS:-0}" \
  -e LMCACHE_CSA_LAYER_MAJOR_INDEXED_SPARSE="${LMCACHE_CSA_LAYER_MAJOR_INDEXED_SPARSE:-0}" \
  -e LMCACHE_INDEXER_REUSE_RESIDUAL_TOPK=0 \
  -e LMCACHE_INDEXER_EXPERIMENTAL_RESIDUAL_LOOKAHEAD=0 \
  -e LMCACHE_NATIVE_INDEXER_STAGE0_LAYERS="${LMCACHE_NATIVE_INDEXER_STAGE0_LAYERS:-1}" \
  -e LMCACHE_NATIVE_INDEXER_WINDOW_LAYERS="${LMCACHE_NATIVE_INDEXER_WINDOW_LAYERS:-21}" \
  -e LMCACHE_CSA_PIPELINE_NVTX="${LMCACHE_CSA_PIPELINE_NVTX:-0}" \
  -e LMCACHE_INDEXER_TIMING="${LMCACHE_INDEXER_TIMING:-0}" \
  -e LMCACHE_INDEXER_TIMING_LIMIT="${LMCACHE_INDEXER_TIMING_LIMIT:-20000}" \
  -e LMCACHE_INDEXER_PROFILE_ACCURACY="$profile_accuracy" \
  -e LMCACHE_TRUE_INDEXER_CP="${LMCACHE_TRUE_INDEXER_CP:-0}" \
  -e LMCACHE_TRUE_INDEXER_CP_SIZE="${LMCACHE_TRUE_INDEXER_CP_SIZE:-8}" \
  -e LMCACHE_TRUE_INDEXER_CP_MIN_ROWS="${LMCACHE_TRUE_INDEXER_CP_MIN_ROWS:-1024}" \
  -e LMCACHE_PROXY_INDEXER_CP="${LMCACHE_PROXY_INDEXER_CP:-0}" \
  -e LMCACHE_PROXY_INDEXER_CP_SIZE="${LMCACHE_PROXY_INDEXER_CP_SIZE:-8}" \
  -e LMCACHE_PROXY_INDEXER_CP_MIN_ROWS="${LMCACHE_PROXY_INDEXER_CP_MIN_ROWS:-1}" \
  -e LMCACHE_PROXY_INDEXER_CP_EXCHANGE="${LMCACHE_PROXY_INDEXER_CP_EXCHANGE:-0}" \
  -e LMCACHE_PROXY_INDEXER_K_CP="${LMCACHE_PROXY_INDEXER_K_CP:-0}" \
  -e LMCACHE_PROXY_INDEXER_K_CP_SIZE="${LMCACHE_PROXY_INDEXER_K_CP_SIZE:-8}" \
  -e LMCACHE_CSA_PREDICTION_GATE_TIMEOUT_SEC="${LMCACHE_CSA_PREDICTION_GATE_TIMEOUT_SEC:-5}" \
  -e LMCACHE_NSYS_CAPTURE="${LMCACHE_NSYS_CAPTURE:-0}" \
  -e LMCACHE_NSYS_CAPTURE_SKIP_REQUESTS="${LMCACHE_NSYS_CAPTURE_SKIP_REQUESTS:-0}" \
  -e LMCACHE_NSYS_FULL_CAPTURE="${LMCACHE_NSYS_FULL_CAPTURE:-0}" \
  -e LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS="${LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS:-0}" \
  -e LMCACHE_NSYS_FULL_CAPTURE_SCOPE="${LMCACHE_NSYS_FULL_CAPTURE_SCOPE:-decoder}" \
  -e LMCACHE_EXEC_PREFIX="${LMCACHE_EXEC_PREFIX:-}" \
  -e LMCACHE_CSA_ATTENTION_KV_TIMING="${LMCACHE_CSA_ATTENTION_KV_TIMING:-0}" \
  -e LMCACHE_TUTTI_PROFILE="${LMCACHE_TUTTI_PROFILE:-0}" \
  -e LMCACHE_TUTTI_PAUSE_WRITES_DURING_DECODE="${LMCACHE_TUTTI_PAUSE_WRITES_DURING_DECODE:-0}" \
  -e LMCACHE_TUTTI_UNLIMITED_WRITES_DURING_DECODE="${LMCACHE_TUTTI_UNLIMITED_WRITES_DURING_DECODE:-0}" \
  -e LMCACHE_TUTTI_DECODE_WRITE_MIBPS="${LMCACHE_TUTTI_DECODE_WRITE_MIBPS:-0}" \
  -e LMCACHE_TUTTI_DECODE_WRITE_GUARD_S="${LMCACHE_TUTTI_DECODE_WRITE_GUARD_S:-2}" \
  -e LMCACHE_TUTTI_BACKGROUND_WRITE_MIBPS="${LMCACHE_TUTTI_BACKGROUND_WRITE_MIBPS:-0}" \
  -e LMCACHE_TUTTI_BACKGROUND_WRITE_BURST_MIB="${LMCACHE_TUTTI_BACKGROUND_WRITE_BURST_MIB:-8}" \
  -e LMCACHE_DSV4_STREAMING_PLAN_CACHE_CAPACITY="${LMCACHE_DSV4_STREAMING_PLAN_CACHE_CAPACITY:-0}" \
  -e LMCACHE_DSV4_STREAMING_PLAN_PREBUILD="${LMCACHE_DSV4_STREAMING_PLAN_PREBUILD:-0}" \
  -e LMCACHE_DSV4_STATIC_IO_GROUP_MAX="${LMCACHE_DSV4_STATIC_IO_GROUP_MAX:-1}" \
  -e LMCACHE_DSV4_STATIC_IO_GROUP_TARGET_MIB="${LMCACHE_DSV4_STATIC_IO_GROUP_TARGET_MIB:-32}" \
  -e LMCACHE_TTFT_STAGE_PROFILE="${LMCACHE_TTFT_STAGE_PROFILE:-0}" \
  -e LMCACHE_LOOKUP_TOKEN_DIAGNOSTICS="${LMCACHE_LOOKUP_TOKEN_DIAGNOSTICS:-0}" \
  -e LMCACHE_PY_ENABLE_GC="${LMCACHE_PY_ENABLE_GC:-true}" \
  -e LMCACHE_HCA_TIMING="${LMCACHE_HCA_TIMING:-0}" \
  -e LMCACHE_HCA_PREFETCH_LOOKAHEAD_LAYERS="${LMCACHE_HCA_PREFETCH_LOOKAHEAD_LAYERS:-1}" \
  -e LMCACHE_D2H_TIMING="${LMCACHE_D2H_TIMING:-0}" \
  -e LMCACHE_DSV4_GPU_LAYER_MAJOR_PACK="${LMCACHE_DSV4_GPU_LAYER_MAJOR_PACK:-0}" \
  -e LMCACHE_DSV4_CPU_RAW_WRITE="${LMCACHE_DSV4_CPU_RAW_WRITE:-1}" \
  -e LMCACHE_DSV4_CPU_RAW_WRITE_MIBPS="${LMCACHE_DSV4_CPU_RAW_WRITE_MIBPS:-0}" \
  -e LMCACHE_DSV4_CPU_RAW_WRITE_BLOCK_MB="${LMCACHE_DSV4_CPU_RAW_WRITE_BLOCK_MB:-64}" \
  -e LMCACHE_DSV4_WRITE_QUANTUM_MB="${LMCACHE_DSV4_WRITE_QUANTUM_MB:-64}" \
  -e LMCACHE_TUTTI_CPU_STAGE_ENABLE="${LMCACHE_TUTTI_CPU_STAGE_ENABLE:-0}" \
  -e LMCACHE_TUTTI_CPU_STAGE_GIB="${LMCACHE_TUTTI_CPU_STAGE_GIB:-16}" \
  -e LMCACHE_TUTTI_HOST_MOUNT_HANDOFF=0 \
  -e LMCACHE_TUTTI_HANDOFF_ID="$tutti_handoff_id" \
  -e LMCACHE_TUTTI_HANDOFF_RANKS=8 \
  -e LMCACHE_TUTTI_HANDOFF_TIMEOUT_SEC=30 \
  -e LMCACHE_HCA_ENABLE_DECODE_HOOK=0 \
  -e LMCACHE_ALLOW_OVERSIZED_KV_CACHE="${LMCACHE_ALLOW_OVERSIZED_KV_CACHE:-0}" \
  -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  --entrypoint /bin/bash \
  "$image" /startup.sh

pin_tp_workers_by_numa() {
  if [ "${LMCACHE_PIN_TP_NUMA:-0}" != "1" ]; then
    return 0
  fi
  local rank pid cpus
  for rank in $(seq 0 7); do
    pid=$(sudo docker top "$name" -eo pid,args 2>/dev/null \
      | awk -v marker="VLLM::Worker_TP${rank}_EP${rank}" \
          'index($0, marker) && !found { print $1; found = 1 }')
    if [ -z "$pid" ]; then
      echo "warning: unable to locate TP worker rank ${rank} for NUMA pinning" >&2
      continue
    fi
    if [ "$rank" -lt 4 ]; then
      cpus="0-63,128-191"
    else
      cpus="64-127,192-238"
    fi
    sudo taskset -pc "$cpus" "$pid" >/dev/null
    echo "tp_numa_pin rank=${rank} pid=${pid} cpus=${cpus}"
  done
}

wait_for_tutti_raw_writers() {
  local timeout_s=${LMCACHE_TUTTI_WRITER_READY_TIMEOUT_SEC:-180}
  local deadline=$((SECONDS + timeout_s))
  local ready_ranks
  while [ "$SECONDS" -lt "$deadline" ]; do
    ready_ranks=$(sudo docker logs "$name" 2>&1 \
      | sed -n '/KV object raw cold-store writer installed/ s/.*Worker_TP\([0-7]\)_EP.*/\1/p' \
      | sort -u | wc -l)
    if [ "$ready_ranks" -ge 8 ]; then
      echo "tutti_raw_writers_ready=8/8"
      return 0
    fi
    sleep 1
  done
  echo "timed out waiting for Tutti raw writers: ready=${ready_ranks:-0}/8" >&2
  sudo docker logs --tail 240 "$name" >&2 || true
  return 1
}

for i in $(seq 1 360); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    wait_for_tutti_raw_writers
    pin_tp_workers_by_numa
    echo "ready_after_seconds=$((i * 2))"
    curl -sf http://127.0.0.1:8000/v1/models
    exit 0
  fi
  sleep 2
done

echo "not ready" >&2
sudo docker logs --tail 240 "$name" >&2 || true
exit 1
