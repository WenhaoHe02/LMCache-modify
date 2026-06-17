#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path

startup_src = Path("/tmp/startup_256k_tutti.sh")
startup_dst = Path("/tmp/startup_256k_tutti_csa_hca_off.sh")
text = startup_src.read_text()

settings = {
    "LMCACHE_DSV4_OPTIMIZED_KV": "1",
    "LMCACHE_DSV4_OPTIMIZED_TAIL_TOKENS": "256",
    "LMCACHE_INDEXER_ENABLE_PREFETCH": "0",
    "LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH": "0",
    "LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY": "0",
    "LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH": "0",
    "LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION": "0",
    "LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS": "0",
    "LMCACHE_INDEXER_ENABLE_POOL_SCORING": "0",
    "LMCACHE_HCA_ENABLE_PREFETCH": "0",
    "LMCACHE_HCA_ENABLE_PINNED_BOUNCE": "0",
    "LMCACHE_DSV4_DEFER_HCA_TO_MOE": "0",
    "LMCACHE_INDEXER_IO_WORKERS": "8",
    "LMCACHE_INDEXER_POOL_SIZE": "4096",
    "LMCACHE_INDEXER_MAX_SEQ_LEN": "131072",
}

lines = []
seen = set()
for line in text.splitlines():
    stripped = line.strip()
    matched = False
    if stripped.startswith("export "):
        name = stripped[len("export ") :].split("=", 1)[0]
        if name in settings:
            lines.append(f"export {name}={settings[name]}")
            seen.add(name)
            matched = True
    if not matched:
        lines.append(line)

insert_at = 0
for i, line in enumerate(lines):
    if line.strip().startswith("export LMCACHE_"):
        insert_at = i + 1
missing = [f"export {name}={value}" for name, value in settings.items() if name not in seen]
if missing:
    lines[insert_at:insert_at] = missing

startup_dst.write_text("\n".join(lines) + "\n")
startup_dst.chmod(0o755)

restart_src = Path("/dev/shm/restart_tutti_container_csa_hca.sh")
restart_dst = Path("/dev/shm/restart_tutti_container_csa_hca_off.sh")
restart = restart_src.read_text()
restart = restart.replace(
    "-v /tmp/startup_256k_tutti_csa_hca.sh:/tmp/startup_256k_tutti.sh:ro \\",
    "-v /tmp/startup_256k_tutti_csa_hca_off.sh:/tmp/startup_256k_tutti.sh:ro \\",
)
restart_dst.write_text(restart)
restart_dst.chmod(0o755)
PY

echo "==== off startup env ===="
grep -nE \
  'LMCACHE_INDEXER|LMCACHE_HCA|LMCACHE_DSV4|DECODE_PREFETCH|PREFETCH_PREFILL' \
  /tmp/startup_256k_tutti_csa_hca_off.sh

echo "==== off restart startup mount ===="
grep -n "startup_256k" /dev/shm/restart_tutti_container_csa_hca_off.sh
