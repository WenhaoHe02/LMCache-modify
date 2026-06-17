#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path

startup_src = Path("/tmp/startup_256k_tutti.sh")
startup_dst = Path("/tmp/startup_256k_tutti_csa_hca.sh")
text = startup_src.read_text()
repls = {
    "export LMCACHE_INDEXER_ENABLE_PREFETCH=0": "export LMCACHE_INDEXER_ENABLE_PREFETCH=1",
    "export LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=0": "export LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=1",
    "export LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=0": "export LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=1",
    "export LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=1": "export LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0",
    "export LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION=1": "export LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION=0",
    "export LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS=0": "export LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS=1",
    "export LMCACHE_HCA_ENABLE_PREFETCH=0": "export LMCACHE_HCA_ENABLE_PREFETCH=1",
    "export LMCACHE_HCA_ENABLE_PINNED_BOUNCE=0": "export LMCACHE_HCA_ENABLE_PINNED_BOUNCE=1",
    "export LMCACHE_DSV4_DEFER_HCA_TO_MOE=0": "export LMCACHE_DSV4_DEFER_HCA_TO_MOE=1",
}
for old, new in repls.items():
    text = text.replace(old, new)
if "LMCACHE_HCA_SSD_DIR" not in text:
    text = text.replace(
        "export LMCACHE_INDEXER_SSD_DIR=/mnt/nvme0/lmcache_csa\n",
        "export LMCACHE_INDEXER_SSD_DIR=/mnt/nvme0/lmcache_csa\n"
        "export LMCACHE_HCA_SSD_DIR=/mnt/nvme0/lmcache_csa\n",
    )
if "LMCACHE_INDEXER_POOL_SIZE" not in text:
    text = text.replace(
        "export LMCACHE_INDEXER_IO_WORKERS=8\n",
        "export LMCACHE_INDEXER_IO_WORKERS=8\n"
        "export LMCACHE_INDEXER_POOL_SIZE=4096\n",
    )
startup_dst.write_text(text)
startup_dst.chmod(0o755)

restart_src = Path("/dev/shm/restart_tutti_container_packing.sh")
restart_dst = Path("/dev/shm/restart_tutti_container_csa_hca.sh")
restart = restart_src.read_text()
restart = restart.replace(
    "-v /tmp/startup_256k_tutti.sh:/tmp/startup_256k_tutti.sh:ro \\",
    "-v /tmp/startup_256k_tutti_csa_hca.sh:/tmp/startup_256k_tutti.sh:ro \\",
)
restart_dst.write_text(restart)
restart_dst.chmod(0o755)
PY

echo "==== csa/hca startup env ===="
grep -nE \
  'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL' \
  /tmp/startup_256k_tutti_csa_hca.sh
