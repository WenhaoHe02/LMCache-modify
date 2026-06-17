#!/usr/bin/env bash
set -euo pipefail

echo "==== patch files ===="
find /tmp/lmcache_patch -maxdepth 6 -type f | sort | sed -n '1,160p' || true

echo "==== patch v1 dir ===="
ls -la /tmp/lmcache_patch/lmcache/v1 2>/dev/null || true

echo "==== restart script ===="
sed -n '1,150p' /dev/shm/restart_tutti_container_csa_hca.sh

echo "==== container patch connector ===="
sudo docker exec dsv4-256k-measure-tutti bash -lc \
  "sed -n '1,220p' /scripts/patch_connector.py; echo ====; python - <<'PY'
import inspect
import lmcache.v1.indexer_ssd_manager as m
print(m.__file__)
print(inspect.getsource(m.IndexerBlockStore._open))
PY" || true
