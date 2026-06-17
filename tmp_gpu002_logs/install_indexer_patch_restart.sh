#!/usr/bin/env bash
set -euo pipefail

restart="/dev/shm/restart_tutti_container_csa_hca.sh"
src="/dev/shm/indexer_ssd_manager.py"

if [[ ! -f "$src" ]]; then
  echo "missing $src" >&2
  exit 1
fi

python3 - <<'PY'
from pathlib import Path

path = Path("/dev/shm/restart_tutti_container_csa_hca.sh")
text = path.read_text()
marker = (
    'sudo cp /dev/shm/cache_engine.py \\\n'
    '  "${PATCH_ROOT}/v1/cache_engine.py"\n'
)
insert = (
    'sudo cp /dev/shm/indexer_ssd_manager.py \\\n'
    '  "${PATCH_ROOT}/v1/indexer_ssd_manager.py"\n'
)
if insert not in text:
    if marker not in text:
        raise SystemExit("restart marker not found")
    text = text.replace(marker, marker + insert, 1)
    path.write_text(text)
PY

echo "==== restart patch markers ===="
grep -n "indexer_ssd_manager.py" "$restart"

bash "$restart"
