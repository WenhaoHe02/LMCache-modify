#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path

src = Path("/dev/shm/restart_tutti_container_packing.sh")
dst = Path("/dev/shm/restart_tutti_container_debug_checksum.sh")
text = src.read_text()
needle = "  --privileged \\\n"
insert = (
    needle
    + "  -e LMCACHE_TUTTI_DEBUG_CHECKSUM=1 \\\n"
    + "  -e LMCACHE_TUTTI_DEBUG_CHECKSUM_LIMIT=4 \\\n"
)
if "LMCACHE_TUTTI_DEBUG_CHECKSUM" not in text:
    text = text.replace(needle, insert)
dst.write_text(text)
dst.chmod(0o755)
PY
