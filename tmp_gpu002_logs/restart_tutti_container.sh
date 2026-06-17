# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
set -euo pipefail

CONTAINER=dsv4-256k-measure-tutti
IMAGE=lmcache/vllm-openai:indexer-ssd-hca-prefetch-decodegate-20260528_0630
PATCH_TAR=/dev/shm/tutti_lazy_fix_full_structured.tar.gz

sudo docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

for bdf in 0000:10:00.0 0000:1c:00.0 0000:4b:00.0 0000:e4:00.0 0000:cc:00.0 0000:a2:00.0; do
  if [ -e "/sys/bus/pci/drivers/snvme/${bdf}" ]; then
    echo "${bdf}" | sudo tee /sys/bus/pci/drivers/snvme/unbind >/dev/null || true
  fi
  if [ -e "/sys/bus/pci/devices/${bdf}" ]; then
    echo nvme | sudo tee "/sys/bus/pci/devices/${bdf}/driver_override" >/dev/null || true
    echo "${bdf}" | sudo tee /sys/bus/pci/drivers/nvme/bind >/dev/null || true
    printf "\n" | sudo tee "/sys/bus/pci/devices/${bdf}/driver_override" >/dev/null || true
  fi
done

sleep 3

sudo mkdir -p /mnt/nvme2 /mnt/nvme3 /mnt/nvme4 /mnt/nvme5 /mnt/nvme8 /mnt/nvme9
findmnt /mnt/nvme2 >/dev/null || sudo mount /dev/nvme2n1 /mnt/nvme2
findmnt /mnt/nvme3 >/dev/null || sudo mount /dev/nvme3n1 /mnt/nvme3
findmnt /mnt/nvme4 >/dev/null || sudo mount /dev/nvme5n1 /mnt/nvme4
findmnt /mnt/nvme5 >/dev/null || sudo mount /dev/nvme10n1 /mnt/nvme5
findmnt /mnt/nvme8 >/dev/null || sudo mount /dev/nvme9n1 /mnt/nvme8
findmnt /mnt/nvme9 >/dev/null || sudo mount /dev/nvme8n1 /mnt/nvme9

mkdir -p /tmp/lmcache_patch
tar -xzf "${PATCH_TAR}" -C /tmp/lmcache_patch

sudo docker run -d \
  --name "${CONTAINER}" \
  --entrypoint /bin/bash \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /mnt:/mnt \
  -v /tmp/startup_256k_tutti.sh:/tmp/startup_256k_tutti.sh:ro \
  -v /tmp/lmcache_patch:/patches:ro \
  -v /tmp/patch_connector.py:/scripts/patch_connector.py:ro \
  "${IMAGE}" \
  /tmp/startup_256k_tutti.sh

echo "container_started"
sudo docker ps --filter "name=${CONTAINER}" --no-trunc
df -h / /mnt/nvme0 /mnt/nvme2 /mnt/nvme3 /mnt/nvme4 /mnt/nvme5 /mnt/nvme6 /mnt/nvme8 /mnt/nvme9
