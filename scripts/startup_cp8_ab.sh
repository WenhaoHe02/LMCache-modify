#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/pro_model}
LMCACHE_STORAGE_MODE=${LMCACHE_STORAGE_MODE:-tutti}
CSA_FILTER=${LMCACHE_ABLATION_CSA_ATTENTION_KV_FILTER:-0}
if [ "$LMCACHE_STORAGE_MODE" = "cpu" ] || \
   [ "$LMCACHE_STORAGE_MODE" = "ssd" ]; then
  CSA_FILTER=0
fi
if [ "$CSA_FILTER" = "1" ]; then
  CSA_ON=1
else
  CSA_ON=0
fi
CROSS_LAYER_VALUE=${LMCACHE_INDEXER_CROSS_LAYER_PREFETCH:-0}
case "${CROSS_LAYER_VALUE,,}" in
  1|true|yes|on) CROSS_LAYER_ON=1 ;;
  *) CROSS_LAYER_ON=0 ;;
esac
GLM_DSA_VALUE=${LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH:-0}
case "${GLM_DSA_VALUE,,}" in
  1|true|yes|on) GLM_DSA_ON=1 ;;
  *) GLM_DSA_ON=0 ;;
esac
GLM_DSA_LAYER_MAJOR_VALUE=${LMCACHE_GLM_DSA_LAYER_MAJOR:-0}
case "${GLM_DSA_LAYER_MAJOR_VALUE,,}" in
  1|true|yes|on) GLM_DSA_LAYER_MAJOR_ON=1 ;;
  *) GLM_DSA_LAYER_MAJOR_ON=$GLM_DSA_ON ;;
esac
if [ "$CSA_ON" = "1" ] || [ "$CROSS_LAYER_ON" = "1" ] || [ "$GLM_DSA_LAYER_MAJOR_ON" = "1" ]; then
  INDEXER_ON=1
else
  INDEXER_ON=0
fi

echo "=== dsv4 csa prefill startup storage=${LMCACHE_STORAGE_MODE} filter=${CSA_FILTER} model=${MODEL_PATH} ===" >&2

# Engine startup loads the tokenizer again after weight/DeepGEMM warmup. The
# cache filesystem is deliberately unmounted before snvme takes ownership, so
# keep the small tokenizer metadata on the container root filesystem instead
# of pinning the multi-terabyte model mount for the lifetime of the service.
TOKENIZER_PATH=/tmp/dsv4_tokenizer_snapshot
rm -rf "$TOKENIZER_PATH"
mkdir -p "$TOKENIZER_PATH"
for tokenizer_file in \
  tokenizer.json \
  tokenizer_config.json \
  special_tokens_map.json \
  added_tokens.json \
  chat_template.json \
  config.json; do
  if [ -f "$MODEL_PATH/$tokenizer_file" ]; then
    cp -a "$MODEL_PATH/$tokenizer_file" "$TOKENIZER_PATH/"
  fi
done
if [ ! -f "$TOKENIZER_PATH/tokenizer.json" ]; then
  echo "missing tokenizer snapshot: $TOKENIZER_PATH/tokenizer.json" >&2
  exit 2
fi

LMCACHE_SITE=$(python3 -c 'import pathlib, lmcache; print(pathlib.Path(lmcache.__file__).resolve().parent)')
VLLM_SITE=$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')
VLLM_RUNTIME_VERSION=$(python3 -c 'from vllm.version import __version__; print(__version__)')
case "$VLLM_RUNTIME_VERSION" in
  0.26.0|0.26.0+*) VLLM_IS_0260=1 ;;
  *) VLLM_IS_0260=0 ;;
esac

if [ "$VLLM_IS_0260" = "1" ]; then
  VLLM_OVERLAY_SCRIPT=/patches/apply_vllm_0260_lmcache.py
  if [ ! -f "$VLLM_OVERLAY_SCRIPT" ]; then
    VLLM_OVERLAY_SCRIPT=/tmp/lmcache_src/scripts/apply_vllm_0260_lmcache.py
  fi
  if [ ! -f "$VLLM_OVERLAY_SCRIPT" ]; then
    echo "missing vLLM 0.26.0 LMCache overlay tool" >&2
    exit 3
  fi
  python3 "$VLLM_OVERLAY_SCRIPT" --vllm-root "$VLLM_SITE"
fi

if [ -f /tmp/lmcache_src/build/lib.linux-x86_64-cpython-312/lmcache/c_ops.cpython-312-x86_64-linux-gnu.so ]; then
  cp /tmp/lmcache_src/build/lib.linux-x86_64-cpython-312/lmcache/c_ops.cpython-312-x86_64-linux-gnu.so \
    "$LMCACHE_SITE/c_ops.cpython-312-x86_64-linux-gnu.so"
fi

python3 - <<'PY'
from pathlib import Path
import vllm

p = (
    Path(vllm.__file__).resolve().parent
    / "distributed/kv_transfer/kv_connector/v1/lmcache_connector.py"
)
s = p.read_text()
changed = False
if "SupportsHMA" not in s:
    s = s.replace(
        "    KVConnectorRole,\n)",
        "    KVConnectorRole,\n    SupportsHMA,\n)",
    )
    changed = True
if "class LMCacheConnectorV1(KVConnectorBase_V1, SupportsHMA):" not in s:
    s = s.replace(
        "class LMCacheConnectorV1(KVConnectorBase_V1):",
        "class LMCacheConnectorV1(KVConnectorBase_V1, SupportsHMA):",
    )
    changed = True
if "def request_finished_all_groups(" not in s:
    marker = "    def request_finished(\n"
    insert = '''    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Notify LMCache using the primary HMA KV group."""
        primary_block_ids = block_ids[0] if block_ids else []
        return self.request_finished(request, primary_block_ids)

'''
    if marker not in s:
        raise RuntimeError("LMCacheConnectorV1.request_finished marker not found")
    s = s.replace(marker, insert + marker, 1)
    changed = True
if changed:
    p.write_text(s)
    print("lmcache_connector.py: patched SupportsHMA/request_finished_all_groups")
else:
    print("lmcache_connector.py: SupportsHMA already present, skipping")
PY

python3 - <<'PY'
from pathlib import Path
import vllm

p = (
    Path(vllm.__file__).resolve().parent
    / "entrypoints/openai/api_server.py"
)
s = p.read_text()
hook = '''from lmcache.integration.vllm.tool_slack_hook import (
    install_vllm_tool_slack_hook,
)

install_vllm_tool_slack_hook()

'''
if "install_vllm_tool_slack_hook" not in s:
    marker = 'if __name__ == "__main__":\n'
    if marker not in s:
        raise RuntimeError("vLLM api_server __main__ marker not found")
    p.write_text(s.replace(marker, hook + marker, 1))
    print("api_server.py: installed LMCache tool-slack hook bootstrap")
else:
    print("api_server.py: LMCache tool-slack hook already present, skipping")
PY

python3 - <<'PY'
from pathlib import Path
import lmcache

p = Path(lmcache.__file__).resolve()
s = p.read_text()
if 'torch_dev' not in s:
    p.write_text(s + '''

def _detect_device():
    try:
        import torch
    except ImportError:
        return None, "cpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu, "xpu"
    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return torch.hpu, "hpu"
    if torch.cuda.is_available():
        return torch.cuda, "cuda"
    return None, "cpu"

torch_dev, torch_device_type = _detect_device()
__all__ = list(__all__) + ['torch_dev', 'torch_device_type']
''')
PY

if [ -d /patches/lmcache ]; then
  cp -a /patches/lmcache/. "$LMCACHE_SITE/"
elif [ -f /patches/vllm_v1_adapter.py ]; then
  mkdir -p "$LMCACHE_SITE/integration/vllm"
  cp /patches/vllm_v1_adapter.py "$LMCACHE_SITE/integration/vllm/vllm_v1_adapter.py"
fi
if [ ! -d /patches/lmcache ] && [ -d /patches/integration ]; then
  mkdir -p "$LMCACHE_SITE/integration"
  cp -a /patches/integration/. "$LMCACHE_SITE/integration/"
fi
if [ ! -d /patches/lmcache ] && [ -d /patches/v1 ]; then
  cp -a /patches/v1/. "$LMCACHE_SITE/v1/"
fi
if [ ! -d /patches/lmcache ] && [ -f /patches/utils.py ]; then
  cp /patches/utils.py "$LMCACHE_SITE/utils.py"
fi
if [ -f /patches/c_ops.cpython-312-x86_64-linux-gnu.so ]; then
  cp /patches/c_ops.cpython-312-x86_64-linux-gnu.so "$LMCACHE_SITE/"
fi
if [ "$VLLM_IS_0260" != "1" ]; then
  if [ -f /patches/sparse_attn_indexer.py ]; then
    cp /patches/sparse_attn_indexer.py "$VLLM_SITE/model_executor/layers/sparse_attn_indexer.py"
  fi
  if [ -f /patches/deepseek_v4.py ]; then
    cp /patches/deepseek_v4.py "$VLLM_SITE/model_executor/models/deepseek_v4.py" || true
  fi
  if [ "$CSA_ON" = "1" ] && [ -f /patches/deepseek_v4_attention.py ]; then
    cp /patches/deepseek_v4_attention.py \
      "$VLLM_SITE/model_executor/layers/deepseek_v4_attention.py"
  fi
fi

python3 -m compileall \
  "$LMCACHE_SITE/integration/vllm/lmcache_connector_v1.py" \
  "$LMCACHE_SITE/integration/vllm/vllm_v1_adapter.py" \
  "$LMCACHE_SITE/integration/vllm/glm_dsa_vllm_0202.py" \
  "$LMCACHE_SITE/integration/vllm/glm_dsa_vllm_0260.py" \
  "$LMCACHE_SITE/integration/vllm/tool_slack_hook.py" \
  "$LMCACHE_SITE/v1/config.py" \
  "$LMCACHE_SITE/v1/indexer_ssd_manager.py" \
  "$LMCACHE_SITE/v1/csa_attention_kv_prefetch_manager.py" \
  "$LMCACHE_SITE/v1/ssd_tp_sharded_prefetch.py" \
  "$LMCACHE_SITE/v1/write_planner.py" \
  "$LMCACHE_SITE/v1/write_slack_client.py" \
  "$LMCACHE_SITE/v1/internal_api_server/vllm/write_slack_api.py" \
  "$LMCACHE_SITE/v1/cache_engine.py" \
  "$LMCACHE_SITE/v1/csa_pipeline_nvtx.py" \
  "$LMCACHE_SITE/v1/csa_prefetch_policy.py" \
  "$LMCACHE_SITE/v1/indexer_tutti_backend.py" \
  "$LMCACHE_SITE/v1/glm_dsa_predictive_prefetch.py" \
  "$LMCACHE_SITE/v1/storage_backend/local_disk_backend.py" \
  "$LMCACHE_SITE/v1/gpu_connector/tutti_direct_loader.py" \
  "$LMCACHE_SITE/v1/kv_object_store"

python3 - <<'PY'
from pathlib import Path
import lmcache

site = Path(lmcache.__file__).resolve().parent
files = (
    "integration/vllm/vllm_v1_adapter.py",
    "integration/vllm/tool_slack_hook.py",
    "v1/config.py",
    "v1/cache_engine.py",
    "v1/csa_attention_kv_prefetch_manager.py",
    "v1/indexer_ssd_manager.py",
    "v1/ssd_tp_sharded_prefetch.py",
    "v1/write_planner.py",
    "v1/write_slack_client.py",
    "v1/internal_api_server/vllm/write_slack_api.py",
    "v1/indexer_tutti_backend.py",
    "v1/csa_pipeline_nvtx.py",
    "v1/csa_prefetch_policy.py",
    "v1/gpu_connector/gpu_connectors.py",
    "v1/gpu_connector/tutti_direct_loader.py",
    "v1/storage_backend/local_disk_backend.py",
    "c_ops.cpython-312-x86_64-linux-gnu.so",
)
for relative in files:
    path = site / relative
    info = path.stat()
    print(f"DEPLOY_FILE bytes={info.st_size} mtime_ns={info.st_mtime_ns} {relative}")
PY

if [ "$CSA_ON" = "1" ]; then
  python3 - <<'PY'
import lmcache.c_ops as c_ops

missing = [
    name
    for name in (
        "tutti_submit_batch_sgl_read",
        "tutti_submit_indexed_sgl_read",
        "tutti_poll_batch",
        "scatter_rows_from_object_ptrs",
    )
    if not hasattr(c_ops, name)
]
if missing:
    raise SystemExit(
        "CSA attention-KV FILTER=1 requires Tutti c_ops symbols; "
        f"missing={missing} c_ops={getattr(c_ops, '__file__', None)}"
    )
print(
    "CSA attention-KV Tutti c_ops symbols verified: "
    f"{getattr(c_ops, '__file__', None)}"
)
PY
fi

if [ "$LMCACHE_STORAGE_MODE" = "cpu" ]; then
cat > /tmp/lmcache_ssd_tutti_kvobj.yaml <<YAML
chunk_size: 256
py_enable_gc: ${LMCACHE_PY_ENABLE_GC:-true}
local_cpu: true
max_local_cpu_size: ${LMCACHE_CPU_PER_RANK_GB:-160.0}
reserve_local_cpu_size: ${LMCACHE_CPU_RESERVE_GB:-128.0}
use_gpu_connector_v3: true
use_layerwise: false
numa_mode: ${LMCACHE_CPU_NUMA_MODE:-null}
store_location: "LocalCPUBackend"
retrieve_locations: ["LocalCPUBackend"]
extra_config:
  save_only_first_rank: ${LMCACHE_CPU_SAVE_ONLY_FIRST_RANK:-false}
  first_rank_max_local_cpu_size: ${LMCACHE_CPU_FIRST_RANK_GB:-160.0}
  dsv4_optimized_kv: false
YAML
elif [ "$LMCACHE_STORAGE_MODE" = "ssd" ]; then
cat > /tmp/lmcache_ssd_tutti_kvobj.yaml <<YAML
chunk_size: 256
py_enable_gc: ${LMCACHE_PY_ENABLE_GC:-true}
local_cpu: false
# LocalDiskBackend requires enough CPU staging for one per-rank 65,280-token
# GLM KV transfer (about 3.53 GB). Keep the allocator, but disable its hot
# cache so every SSD-only retrieval performs physical disk I/O.
max_local_cpu_size: ${LMCACHE_SSD_LOCAL_CPU_GB:-5.0}
local_disk: "/mnt/nvme0/lmcache_dsv4_cache/,/mnt/nvme2/lmcache_dsv4_cache/,/mnt/nvme3/lmcache_dsv4_cache/,/mnt/nvme4/lmcache_dsv4_cache/,/mnt/nvme5/lmcache_dsv4_cache/,/mnt/nvme6/lmcache_dsv4_cache/,/mnt/nvme8/lmcache_dsv4_cache/,/mnt/nvme9/lmcache_dsv4_cache/"
local_disk_path_sharding: "by_gpu"
max_local_disk_size: 4096.0
use_gpu_connector_v3: true
use_layerwise: false
layer_group_size: 1
internal_api_server_enabled: ${LMCACHE_SSD_INTERNAL_API_ENABLED:-false}
internal_api_server_host: ${LMCACHE_WRITE_SLACK_API_HOST:-127.0.0.1}
internal_api_server_port_start: ${LMCACHE_WRITE_SLACK_API_PORT_START:-6999}
store_location: "LocalDiskBackend"
retrieve_locations: ["LocalDiskBackend"]
extra_config:
  # Bypass the Linux page cache so the SSD-only baseline measures physical I/O.
  use_odirect: true
  save_only_first_rank: false
  dsv4_optimized_kv: ${LMCACHE_SSD_DSV4_OPTIMIZED_KV:-true}
  dsv4_optimized_tail_tokens: ${LMCACHE_SSD_DSV4_OPTIMIZED_TAIL_TOKENS:-256}
  dsv4_defer_hca_to_moe: ${LMCACHE_SSD_DSV4_DEFER_HCA_TO_MOE:-false}
  kv_object_store_enable: false
  kv_object_store_tutti_raw_enable: false
YAML
else
cat > /tmp/lmcache_ssd_tutti_kvobj.yaml <<YAML
chunk_size: 256
py_enable_gc: ${LMCACHE_PY_ENABLE_GC:-true}
local_cpu: false
max_local_cpu_size: 256.0
local_disk: "/mnt/nvme0/lmcache_dsv4_cache/,/mnt/nvme2/lmcache_dsv4_cache/,/mnt/nvme3/lmcache_dsv4_cache/,/mnt/nvme4/lmcache_dsv4_cache/,/mnt/nvme5/lmcache_dsv4_cache/,/mnt/nvme6/lmcache_dsv4_cache/,/mnt/nvme8/lmcache_dsv4_cache/,/mnt/nvme9/lmcache_dsv4_cache/"
local_disk_path_sharding: "by_gpu"
max_local_disk_size: 4096.0
use_gpu_connector_v3: true
use_layerwise: false
layer_group_size: 1
internal_api_server_enabled: ${LMCACHE_WRITE_SLACK_API_ENABLED:-true}
internal_api_server_host: ${LMCACHE_WRITE_SLACK_API_HOST:-127.0.0.1}
internal_api_server_port_start: ${LMCACHE_WRITE_SLACK_API_PORT_START:-6999}
extra_config:
  use_odirect: false
  save_only_first_rank: false
  dsv4_optimized_kv: true
  dsv4_optimized_tail_tokens: 256
  dsv4_defer_hca_to_moe: false
  kv_object_store_enable: true
  kv_object_store_slot_mb: ${LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB:-128}
  kv_object_store_capacity: ${LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY:-48000}
  kv_object_store_tutti_raw_enable: true
  kv_object_store_tutti_raw_region_path: "/mnt/nvme0/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme2/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme3/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme4/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme5/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme6/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme8/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme9/tutti_raw_reserve/rank_raw_region_3g.bin"
  tutti_device_path: "/dev/ssnvme0"
  tutti_ctrl_path: "/dev/snvm_control"
  tutti_pci_bdfs: "0000:6f:00.0,0000:10:00.0,0000:1c:00.0,0000:4b:00.0,0000:e4:00.0,0000:88:00.0,0000:cc:00.0,0000:a2:00.0"
  tutti_n_slots: ${LMCACHE_ABLATION_TUTTI_N_SLOTS:-4}
  tutti_slot_mb: ${LMCACHE_ABLATION_TUTTI_SLOT_MB:-128}
  tutti_nsid: 1
YAML
fi

if [ "$LMCACHE_STORAGE_MODE" = "ssd" ]; then
  python3 - <<'PY'
from pathlib import Path

import yaml

config = yaml.safe_load(Path("/tmp/lmcache_ssd_tutti_kvobj.yaml").read_text())
expected = {
    "local_cpu": False,
    "store_location": "LocalDiskBackend",
    "retrieve_locations": ["LocalDiskBackend"],
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(
            f"invalid SSD-only config: {key}={config.get(key)!r}, expected={value!r}"
        )
if config.get("extra_config", {}).get("use_odirect") is not True:
    raise SystemExit("invalid SSD-only config: extra_config.use_odirect must be true")
print(
    "SSD_ONLY_CONFIG_VERIFIED "
    "local_cpu=false retrieve=LocalDiskBackend use_odirect=true"
)
PY
fi

RAW_REGION_PATH="/mnt/nvme0/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme2/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme3/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme4/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme5/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme6/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme8/tutti_raw_reserve/rank_raw_region_3g.bin,/mnt/nvme9/tutti_raw_reserve/rank_raw_region_3g.bin"
INDEXER_RAW_REGION_PATH="/mnt/nvme0/tutti_raw_reserve/indexer_raw_region_512m.bin,/mnt/nvme2/tutti_raw_reserve/indexer_raw_region_512m.bin,/mnt/nvme3/tutti_raw_reserve/indexer_raw_region_512m.bin,/mnt/nvme4/tutti_raw_reserve/indexer_raw_region_512m.bin,/mnt/nvme5/tutti_raw_reserve/indexer_raw_region_512m.bin,/mnt/nvme6/tutti_raw_reserve/indexer_raw_region_512m.bin,/mnt/nvme8/tutti_raw_reserve/indexer_raw_region_512m.bin,/mnt/nvme9/tutti_raw_reserve/indexer_raw_region_512m.bin"

export LMCACHE_CONFIG_FILE=/tmp/lmcache_ssd_tutti_kvobj.yaml
if grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"glm_moe_dsa"' \
  "${MODEL_PATH}/config.json"; then
  model_logical_block_size=64
else
  model_logical_block_size=256
fi
export LMCACHE_ENGINE_LOGICAL_BLOCK_SIZE=${LMCACHE_ENGINE_LOGICAL_BLOCK_SIZE:-${model_logical_block_size}}
echo "LMCACHE_ENGINE_LOGICAL_BLOCK_SIZE=${LMCACHE_ENGINE_LOGICAL_BLOCK_SIZE}"
export LMCACHE_DSV4_OPTIMIZED_KV=1
export LMCACHE_DSV4_OPTIMIZED_TAIL_TOKENS=256
export LMCACHE_DSV4_CSA_ATTENTION_KV_FILTER="$CSA_FILTER"
export LMCACHE_KV_OBJECT_STORE_ENABLE=1
export LMCACHE_KV_OBJECT_STORE_SLOT_MB=${LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB:-128}
export LMCACHE_KV_OBJECT_STORE_CAPACITY=${LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY:-48000}
export LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE=1
export LMCACHE_INDEXER_ENABLE_PREFETCH="$INDEXER_ON"
export LMCACHE_INDEXER_FULL_OVERLAP="$CSA_ON"
export LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY="$CSA_ON"
export LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH="$CSA_ON"
export LMCACHE_INDEXER_TUTTI_BACKEND="$INDEXER_ON"
export LMCACHE_INDEXER_TUTTI_RAW_REGION_PATH="$INDEXER_RAW_REGION_PATH"
export LMCACHE_REUSE_PREFETCH_ASYNC=0
export LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0
export LMCACHE_INDEXER_ENABLE_PREFILL_RESIDUAL_PROXY=0
export LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION=0
export LMCACHE_INDEXER_ENABLE_POOL_SCORING=0
if [ "$LMCACHE_STORAGE_MODE" = "cpu" ] || \
   [ "$LMCACHE_STORAGE_MODE" = "ssd" ]; then
  export LMCACHE_DSV4_OPTIMIZED_KV=0
  export LMCACHE_DSV4_CSA_ATTENTION_KV_FILTER=0
  export LMCACHE_KV_OBJECT_STORE_ENABLE=0
  export LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE=0
  export LMCACHE_INDEXER_ENABLE_PREFETCH=0
  export LMCACHE_INDEXER_FULL_OVERLAP=0
  export LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=0
  export LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=0
  export LMCACHE_INDEXER_TUTTI_BACKEND=0
  export LMCACHE_DSV4_HCA_WALKER=0
fi
if [ "$CSA_ON" = "1" ]; then
  export LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER=${LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER:-profile80_hybrid}
  export LMCACHE_CSA_PREFETCH_CP_SIZE=${LMCACHE_CSA_PREFETCH_CP_SIZE:-8}
  export LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=${LMCACHE_CSA_PREFETCH_CP_INTERLEAVE:-64}
  export LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE=${LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE:-1}
  export LMCACHE_CSA_PREFETCH_BLOCK_BUDGET=${LMCACHE_CSA_PREFETCH_BLOCK_BUDGET:-2048}
  CP_CONFIG="cp_size=${LMCACHE_CSA_PREFETCH_CP_SIZE} cp_interleave=${LMCACHE_CSA_PREFETCH_CP_INTERLEAVE} cp_oversubscribe=${LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE} block_budget=${LMCACHE_CSA_PREFETCH_BLOCK_BUDGET}"
else
  unset LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER
  unset LMCACHE_CSA_PREFETCH_CP_SIZE
  unset LMCACHE_CSA_PREFETCH_CP_INTERLEAVE
  unset LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE
  unset LMCACHE_CSA_PREFETCH_BLOCK_BUDGET
  CP_CONFIG="cp_size=off cp_interleave=off cp_oversubscribe=off"
fi
export LMCACHE_DSV4_HCA_WALKER=${LMCACHE_DSV4_HCA_WALKER:-0}
export LMCACHE_INDEXER_REUSE_RESIDUAL_TOPK=0
export LMCACHE_INDEXER_EXPERIMENTAL_RESIDUAL_LOOKAHEAD=0
export LMCACHE_CSA_PIPELINE_NVTX=${LMCACHE_CSA_PIPELINE_NVTX:-0}
export LMCACHE_INDEXER_SSD_DIR=${LMCACHE_ABLATION_INDEXER_SSD_DIR:-/mnt/nvme0/lmcache_csa,/mnt/nvme2/lmcache_csa,/mnt/nvme3/lmcache_csa,/mnt/nvme4/lmcache_csa,/mnt/nvme5/lmcache_csa,/mnt/nvme6/lmcache_csa,/mnt/nvme8/lmcache_csa,/mnt/nvme9/lmcache_csa}
export LMCACHE_INDEXER_IO_WORKERS=${LMCACHE_INDEXER_IO_WORKERS:-8}
export LMCACHE_INDEXER_MAX_SEQ_LEN=${LMCACHE_ABLATION_INDEXER_MAX_SEQ_LEN:-131072}
export LMCACHE_CSA_ATTENTION_KV_PROXY_MICROBATCH_ROWS=${LMCACHE_CSA_ATTENTION_KV_PROXY_MICROBATCH_ROWS:-64}
export LMCACHE_INDEXER_TIMING=0
export LMCACHE_INDEXER_TIMING_LIMIT=${LMCACHE_INDEXER_TIMING_LIMIT:-20000}
export LMCACHE_INDEXER_PROFILE_ACCURACY=${LMCACHE_INDEXER_PROFILE_ACCURACY:-$CSA_ON}
export LMCACHE_NSYS_CAPTURE=${LMCACHE_NSYS_CAPTURE:-0}
export LMCACHE_NSYS_CAPTURE_SKIP_REQUESTS=${LMCACHE_NSYS_CAPTURE_SKIP_REQUESTS:-0}
export LMCACHE_NSYS_FULL_CAPTURE=${LMCACHE_NSYS_FULL_CAPTURE:-0}
export LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS=${LMCACHE_NSYS_FULL_CAPTURE_SKIP_REQUESTS:-0}
export LMCACHE_NSYS_FULL_CAPTURE_SCOPE=${LMCACHE_NSYS_FULL_CAPTURE_SCOPE:-decoder}
export LMCACHE_CSA_ATTENTION_KV_TIMING=${LMCACHE_CSA_ATTENTION_KV_TIMING:-0}
export LMCACHE_TUTTI_PROFILE=${LMCACHE_TUTTI_PROFILE:-0}
export LMCACHE_TOOL_SLACK_HOOK=${LMCACHE_TOOL_SLACK_HOOK:-1}
if [ "$LMCACHE_STORAGE_MODE" = "ssd" ]; then
  export LMCACHE_TOOL_SLACK_HOOK=0
fi
export LMCACHE_WRITE_SLACK_WORKER_COUNT=${LMCACHE_WRITE_SLACK_WORKER_COUNT:-8}
export LMCACHE_WRITE_SLACK_FIRST_WORKER_PORT=${LMCACHE_WRITE_SLACK_FIRST_WORKER_PORT:-7000}
export LMCACHE_TOOL_SLACK_DEFAULT_SEC=${LMCACHE_TOOL_SLACK_DEFAULT_SEC:-30}
export LMCACHE_TUTTI_BACKGROUND_WRITE_MIBPS=${LMCACHE_TUTTI_BACKGROUND_WRITE_MIBPS:-0}
export LMCACHE_TUTTI_BACKGROUND_WRITE_BURST_MIB=${LMCACHE_TUTTI_BACKGROUND_WRITE_BURST_MIB:-8}
export LMCACHE_TUTTI_PAUSE_WRITES_DURING_DECODE=${LMCACHE_TUTTI_PAUSE_WRITES_DURING_DECODE:-0}
export LMCACHE_TUTTI_UNLIMITED_WRITES_DURING_DECODE=${LMCACHE_TUTTI_UNLIMITED_WRITES_DURING_DECODE:-0}
export LMCACHE_TUTTI_DECODE_WRITE_MIBPS=${LMCACHE_TUTTI_DECODE_WRITE_MIBPS:-0}
export LMCACHE_TUTTI_DECODE_WRITE_GUARD_S=${LMCACHE_TUTTI_DECODE_WRITE_GUARD_S:-2}
export LMCACHE_TTFT_STAGE_PROFILE=${LMCACHE_TTFT_STAGE_PROFILE:-0}
export LMCACHE_HCA_ENABLE_PREFETCH=0
export LMCACHE_HCA_ENABLE_PINNED_BOUNCE=0
export LMCACHE_DSV4_DEFER_HCA_TO_MOE=0
export LMCACHE_HCA_ENABLE_DECODE_HOOK=0
export LMCACHE_HCA_TIMING=0
export LMCACHE_D2H_TIMING=${LMCACHE_D2H_TIMING:-0}
export LMCACHE_TUTTI_WARMUP_AFTER_STORE_DELAY_SEC=${LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY:-10}
export LMCACHE_TUTTI_STARTUP_WARMUP_DELAY_SEC=${LMCACHE_ABLATION_TUTTI_STARTUP_DELAY:-120}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

printf '%s\n' \
  "EXPERIMENT_CONFIG csa_filter=${CSA_FILTER} indexer_prefetch=${INDEXER_ON} glm_layer_major=${GLM_DSA_LAYER_MAJOR_ON} glm_prediction=${GLM_DSA_ON} ${CP_CONFIG} hca_walker=${LMCACHE_DSV4_HCA_WALKER} accuracy=${LMCACHE_INDEXER_PROFILE_ACCURACY} indexer_timing=${LMCACHE_INDEXER_TIMING} csa_timing=${LMCACHE_CSA_ATTENTION_KV_TIMING} ttft_profile=${LMCACHE_TTFT_STAGE_PROFILE} d2h_timing=${LMCACHE_D2H_TIMING} nvtx=${LMCACHE_CSA_PIPELINE_NVTX}" >&2

IFS=',' read -ra _CSA_DIRS <<< "$LMCACHE_INDEXER_SSD_DIR"
for d in "${_CSA_DIRS[@]}"; do
  for r in 0 1 2 3 4 5 6 7; do
    mkdir -p "${d}/rank_${r}" 2>/dev/null || true
  done
done

for drv in nvme0 nvme2 nvme3 nvme4 nvme5 nvme6 nvme8 nvme9; do
  [ -d "/mnt/$drv" ] || continue
  # Never create cache dirs / fallocate 30G on an UNMOUNTED mountpoint:
  # writes land on the root fs (hidden once the drive mounts) and fill /.
  mountpoint -q "/mnt/$drv" || { echo "skip /mnt/$drv (not mounted)"; continue; }
  mkdir -p "/mnt/$drv/tutti_raw_reserve"
  f="/mnt/$drv/tutti_raw_reserve/rank_raw_region_3g.bin"
  raw_region_bytes=${LMCACHE_TUTTI_RAW_REGION_BYTES:-25769803776}
  # 480K uses ~15 GiB for token-major objects; segmented CSA/HCA sidecars
  # need additional headroom or later segments silently fall back.
  if [ ! -f "$f" ] || [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" -lt "$raw_region_bytes" ]; then
    fallocate -l "$raw_region_bytes" "$f"
  fi
  [ "$(stat -c %s "$f")" -ge "$raw_region_bytes" ] || {
    echo "raw region is smaller than requested: $f" >&2
    exit 1
  }
  indexer_f="/mnt/$drv/tutti_raw_reserve/indexer_raw_region_512m.bin"
  indexer_raw_region_bytes=${LMCACHE_INDEXER_RAW_REGION_BYTES:-536870912}
  if [ ! -f "$indexer_f" ] || [ "$(stat -c %s "$indexer_f" 2>/dev/null || echo 0)" -lt "$indexer_raw_region_bytes" ]; then
    fallocate -l "$indexer_raw_region_bytes" "$indexer_f"
  fi
  [ "$(stat -c %s "$indexer_f")" -ge "$indexer_raw_region_bytes" ] || {
    echo "indexer raw region is smaller than requested: $indexer_f" >&2
    exit 1
  }
done

KV_CACHE_ARGS=()
if [ -n "${LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES:-}" ]; then
  KV_CACHE_ARGS=(
    --kv-cache-memory-bytes
    "${LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES}"
  )
fi

SCHEDULER_ARGS=()
if [ "$VLLM_IS_0260" = "1" ]; then
  # The SSD/HMA path deliberately admits one chunk at a time; requiring the
  # entire 530K input to reside in GPU KV would deadlock before chunk zero.
  SCHEDULER_ARGS=(--no-scheduler-reserve-full-isl)
fi

exec ${LMCACHE_EXEC_PREFIX:-} python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tokenizer "$TOKENIZER_PATH" \
  --generation-config vllm \
  --served-model-name deepseek-v4-pro \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --all2all-backend allgather_reducescatter \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --max-model-len ${LMCACHE_ABLATION_MAX_MODEL_LEN:-32768} \
  --max-num-seqs 1 \
  --max-num-batched-tokens ${LMCACHE_ABLATION_MAX_BATCHED_TOKENS:-1024} \
  --enable-chunked-prefill \
  --gpu-memory-utilization ${LMCACHE_ABLATION_GPU_UTIL:-0.75} \
  "${KV_CACHE_ARGS[@]}" \
  "${SCHEDULER_ARGS[@]}" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --no-disable-hybrid-kv-cache-manager \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --port 8000
