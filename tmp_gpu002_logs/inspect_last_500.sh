#!/usr/bin/env bash
set -euo pipefail

container="${1:-dsv4-256k-measure-tutti}"

echo "==== recent errors ===="
sudo docker logs --since 10m "$container" 2>&1 \
  | grep -A80 -B20 -E 'ERROR|Exception|Traceback|HTTP|Internal Server Error|ValueError|RuntimeError|CUDA error|AssertionError' \
  | tail -n 360 || true

echo "==== tail ===="
sudo docker logs --tail 260 "$container" 2>&1 || true
