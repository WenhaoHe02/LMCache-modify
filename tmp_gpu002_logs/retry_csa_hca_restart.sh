#!/usr/bin/env bash
set -euo pipefail

container=dsv4-256k-measure-tutti

echo "==== before ===="
sudo docker ps -a --filter "name=${container}" --no-trunc || true

for attempt in 1 2 3 4; do
  echo "==== rm attempt ${attempt} ===="
  sudo docker rm -f "${container}" || true
  sleep 3
  if ! sudo docker ps -a --format '{{.Names}}' | grep -Fxq "${container}"; then
    break
  fi
done

echo "==== after rm ===="
sudo docker ps -a --filter "name=${container}" --no-trunc || true

bash /dev/shm/restart_tutti_container_csa_hca.sh

echo "==== after restart ===="
sudo docker ps -a --filter "name=${container}" --no-trunc || true
