#!/usr/bin/env bash
set -euo pipefail

echo "== code markers =="
sudo docker exec dsv4-256k-measure-tutti /bin/bash -lc '
set -e
P=/opt/venv/lib/python3.12/site-packages/lmcache/v1/gpu_connector/tutti_direct_loader.py
grep -n "_MAX_EXTENTS\|def _slot_iova_with_offset\|query_extents(path)\|single_contiguous(path)" "$P" | head -n 20
'

echo "== missing file stat/filefrag =="
paths=(
  "/mnt/nvme2/lmcache_dsv4_cache/-mnt-nvme0-models-DeepSeek-V4-Pro@8@1@635519a74cd21acf@uint8.pt"
  "/mnt/nvme4/lmcache_dsv4_cache/-mnt-nvme0-models-DeepSeek-V4-Pro@8@3@635519a74cd21acf@uint8.pt"
  "/mnt/nvme8/lmcache_dsv4_cache/-mnt-nvme0-models-DeepSeek-V4-Pro@8@6@635519a74cd21acf@uint8.pt"
  "/mnt/nvme9/lmcache_dsv4_cache/-mnt-nvme0-models-DeepSeek-V4-Pro@8@7@71954e947b815eda@uint8.pt"
)

for p in "${paths[@]}"; do
  echo "-- $p"
  if [[ -e "$p" ]]; then
    stat -c 'size=%s blocks=%b mode=%a' "$p" || true
    filefrag -v "$p" 2>&1 | sed -n '1,12p' || true
  else
    echo "missing on host"
    dir=$(dirname "$p")
    base=$(basename "$p")
    echo "nearby:"
    ls "$dir" | grep "${base##*@}" | head -n 10 || true
  fi
done

echo "== fiemap via container helper =="
sudo docker exec dsv4-256k-measure-tutti python - <<'PY'
from lmcache.v1.gpu_connector.tutti_direct_loader import FiemapHelper
paths = [
  "/mnt/nvme2/lmcache_dsv4_cache/-mnt-nvme0-models-DeepSeek-V4-Pro@8@1@635519a74cd21acf@uint8.pt",
  "/mnt/nvme4/lmcache_dsv4_cache/-mnt-nvme0-models-DeepSeek-V4-Pro@8@3@635519a74cd21acf@uint8.pt",
  "/mnt/nvme8/lmcache_dsv4_cache/-mnt-nvme0-models-DeepSeek-V4-Pro@8@6@635519a74cd21acf@uint8.pt",
  "/mnt/nvme9/lmcache_dsv4_cache/-mnt-nvme0-models-DeepSeek-V4-Pro@8@7@71954e947b815eda@uint8.pt",
]
for path in paths:
    print("--", path)
    try:
        extents = FiemapHelper.query_extents(path)
        print("n_extents", len(extents), "first", extents[:3])
        print("scan_has", path in FiemapHelper.scan_paths([path]))
    except Exception as exc:
        print(type(exc).__name__, exc)
PY

echo "== recent FIEMAP warnings =="
sudo docker logs --since 20m dsv4-256k-measure-tutti 2>&1 \
  | grep -E "FIEMAP|LBA pre-scan|first_missing" \
  | tail -n 120 || true
