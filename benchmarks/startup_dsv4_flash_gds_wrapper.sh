#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Convert the Tutti V4 Flash startup into a clean LMCache GDS startup.

set -euo pipefail

source_startup=${SOURCE_STARTUP:-/startup_tutti_original.sh}
clean_startup=$(mktemp /tmp/startup_lmcache_gds.XXXXXX.sh)
trap 'rm -f "$clean_startup"' EXIT

if [[ ${LMCACHE_ABLATION_CSA_ATTENTION_KV_FILTER:-0} != 0 ]]; then
  echo "GDS baseline requires CSA/prefetch to be disabled" >&2
  exit 2
fi

gds_run_id=${LMCACHE_GDS_RUN_ID:-agent_trace_gds_20260715}
if [[ ! $gds_run_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid LMCACHE_GDS_RUN_ID: $gds_run_id" >&2
  exit 2
fi

gds_paths=()
for drv in nvme0 nvme2 nvme3 nvme4 nvme5 nvme6 nvme8 nvme9; do
  mountpoint="/mnt/$drv"
  if [[ ${DRY_RUN:-0} != 1 ]] && ! mountpoint -q "$mountpoint"; then
    echo "GDS mount is missing: $mountpoint" >&2
    exit 2
  fi
  gds_paths+=("$mountpoint/lmcache_dsv4_gds_cache/$gds_run_id")
done
LMCACHE_GDS_PATHS=$(IFS=,; echo "${gds_paths[*]}")
export LMCACHE_GDS_PATHS

# Use the host's cuFile userspace library so it matches the loaded nvidia-fs
# driver. The image's bundled cuFile 1.14 library is intentionally bypassed.
export LD_LIBRARY_PATH="/opt/cuda-gds/lib:${LD_LIBRARY_PATH:-}"
if [[ ${DRY_RUN:-0} != 1 ]]; then
  mkdir -p /tmp/lmcache-cufile-logs
  python3 - <<'PY'
import ctypes
import json
import os
from pathlib import Path

source = Path("/etc/cufile.json")
target = Path("/tmp/lmcache-cufile.json")
config = json.loads(source.read_text())
config.setdefault("logging", {})["dir"] = "/tmp/lmcache-cufile-logs"
config["logging"]["level"] = os.environ.get("LMCACHE_CUFILE_LOG_LEVEL", "ERROR")
config.setdefault("properties", {})["allow_compat_mode"] = False
config["properties"]["force_compat_mode"] = False
target.write_text(json.dumps(config, indent=2) + "\n")

ctypes.CDLL("libcufile.so")
loaded = {
    line.split()[-1]
    for line in Path("/proc/self/maps").read_text().splitlines()
    if "libcufile.so" in line
}
if not loaded or not all(path.startswith("/opt/cuda-gds/lib/") for path in loaded):
    raise RuntimeError(f"unexpected libcufile mapping: {sorted(loaded)}")
print(f"verified host cuFile library: {sorted(loaded)}")
PY
  export CUFILE_ENV_PATH_JSON=/tmp/lmcache-cufile.json
fi

python3 - "$source_startup" "$clean_startup" <<'PY'
from pathlib import Path
import re
import sys

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
text = source_path.read_text()

yaml_pattern = re.compile(
    r"cat > /tmp/lmcache_ssd_tutti_kvobj\.yaml <<YAML\n.*?\nYAML",
    re.DOTALL,
)
gds_yaml = r'''cat > /tmp/lmcache_gds.yaml <<YAML
chunk_size: 256
local_cpu: false
max_local_cpu_size: 0
local_disk: null
max_local_disk_size: 0
gds_path: "${LMCACHE_GDS_PATHS}"
gds_path_sharding: "by_gpu"
gds_buffer_size: ${LMCACHE_GDS_BUFFER_MB:-8192}
use_gds: true
gds_backend: "cufile"
store_location: "GdsBackend"
retrieve_locations:
  - "GdsBackend"
use_gpu_connector_v3: true
use_layerwise: false
layer_group_size: 1
extra_config:
  use_direct_io: true
  gds_io_threads: ${LMCACHE_GDS_IO_THREADS:-4}
  save_only_first_rank: false
  dsv4_optimized_kv: true
  dsv4_optimized_tail_tokens: 256
  dsv4_defer_hca_to_moe: false
  kv_object_store_enable: false
  kv_object_store_tutti_raw_enable: false
YAML'''
text, replacements = yaml_pattern.subn(gds_yaml, text, count=1)
if replacements != 1:
    raise RuntimeError("could not replace the Tutti LMCache YAML block")

text = text.replace(
    "export LMCACHE_CONFIG_FILE=/tmp/lmcache_ssd_tutti_kvobj.yaml",
    "export LMCACHE_CONFIG_FILE=/tmp/lmcache_gds.yaml",
)
text = text.replace(
    "export LMCACHE_KV_OBJECT_STORE_ENABLE=1",
    "export LMCACHE_KV_OBJECT_STORE_ENABLE=0",
)
text = text.replace(
    "export LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE=1",
    "export LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE=0",
)
text = text.replace(
    'export LMCACHE_INDEXER_TUTTI_BACKEND="$CSA_ON"',
    "export LMCACHE_INDEXER_TUTTI_BACKEND=0",
)
text = text.replace(
    'export LMCACHE_INDEXER_TUTTI_RAW_REGION_PATH="$RAW_REGION_PATH"',
    "export LMCACHE_INDEXER_TUTTI_RAW_REGION_PATH=",
)
text = re.sub(r'\nRAW_REGION_PATH="[^"]*"\n', "\n", text, count=1)
text, removed_loops = re.subn(
    r"\nfor drv in nvme0 nvme2 nvme3 nvme4 nvme5 nvme6 nvme8 nvme9; do\n"
    r".*?\ndone\n",
    "\n",
    text,
    count=1,
    flags=re.DOTALL,
)
if removed_loops != 1:
    raise RuntimeError("could not remove the Tutti raw-reserve loop")

# LMCache currently registers the large GDS buffer before explicitly opening
# the cuFile driver. CUDA 13.2/cuFile 1.17 requires deterministic driver-first
# initialization in this image, so retain the singleton before registration.
driver_patch = r'''python3 - <<'PY_GDS_DRIVER'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/site-packages/lmcache/v1/memory_management.py")
source = path.read_text()
old = """        from cufile.bindings import cuFileBufDeregister, cuFileBufRegister

        self.cuFileBufDeregister = cuFileBufDeregister
"""
new = """        import cufile
        from cufile.bindings import cuFileBufDeregister, cuFileBufRegister

        self._gds_driver = cufile.CuFileDriver()
        self.cuFileBufDeregister = cuFileBufDeregister
"""
if old not in source:
    raise RuntimeError("CuFileMemoryAllocator initialization marker not found")
path.write_text(source.replace(old, new, 1))
print("patched CuFileMemoryAllocator to open the cuFile driver first")

# The image's StorageManager assumes LocalCPUBackend always exists, despite
# GdsBackend implementing AllocatorBackendInterface and the documented GDS
# configuration disabling local CPU. Select GdsBackend when it is the only
# allocator backend so the engine does not silently degrade to recompute.
path = Path(
    "/usr/local/lib/python3.12/site-packages/lmcache/v1/"
    "storage_backend/storage_manager.py"
)
source = path.read_text()
old = """        else:
            allocator_backend = self.storage_backends["LocalCPUBackend"]
        assert isinstance(allocator_backend, AllocatorBackendInterface)
"""
new = """        elif "LocalCPUBackend" in self.storage_backends:
            allocator_backend = self.storage_backends["LocalCPUBackend"]
        elif "GdsBackend" in self.storage_backends:
            allocator_backend = self.storage_backends["GdsBackend"]
        else:
            raise RuntimeError("No allocator-capable storage backend is configured")
        assert isinstance(allocator_backend, AllocatorBackendInterface)
"""
if old not in source:
    raise RuntimeError("StorageManager allocator selection marker not found")
path.write_text(source.replace(old, new, 1))
print("patched StorageManager to select GdsBackend as allocator")
PY_GDS_DRIVER

'''
compile_marker = "python3 -m compileall \\\n"
if compile_marker not in text:
    raise RuntimeError("compileall insertion marker not found")
text = text.replace(compile_marker, driver_patch + compile_marker, 1)

target_path.write_text(text)
PY

if grep -qE 'tutti_device_path:|kv_object_store_enable: true' "$clean_startup"; then
  echo "failed to remove active Tutti/object-store configuration" >&2
  exit 1
fi
if ! grep -q '^gds_backend: "cufile"$' "$clean_startup"; then
  echo "failed to enable the cuFile GDS backend" >&2
  exit 1
fi
if ! grep -q '^  use_direct_io: true$' "$clean_startup"; then
  echo "failed to require direct I/O" >&2
  exit 1
fi
if [[ ${DRY_RUN:-0} == 1 ]]; then
  cat "$clean_startup"
  exit 0
fi

exec bash "$clean_startup"
