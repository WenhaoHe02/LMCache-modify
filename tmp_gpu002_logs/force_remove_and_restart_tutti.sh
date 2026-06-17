#!/usr/bin/env bash
set -euo pipefail

sudo docker rm -f dsv4-256k-measure-tutti || true
sleep 2
sudo docker ps -a --filter name=dsv4-256k-measure-tutti --no-trunc || true
bash /dev/shm/restart_tutti_container_packing.sh
