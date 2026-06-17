#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path

src = Path("/dev/shm/restart_tutti_container_packing.sh")
dst = Path("/dev/shm/restart_tutti_container_cpu_fallback.sh")
text = src.read_text()
needle = "  --privileged \\\n"
insert = needle + "  -e LMCACHE_TUTTI_FORCE_CPU_FALLBACK=1 \\\n"
if "LMCACHE_TUTTI_FORCE_CPU_FALLBACK" not in text:
    text = text.replace(needle, insert)
dst.write_text(text)
dst.chmod(0o755)
PY
