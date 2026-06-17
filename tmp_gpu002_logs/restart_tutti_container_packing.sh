#!/usr/bin/env bash
set -euo pipefail

CONTAINER=dsv4-256k-measure-tutti
IMAGE=lmcache/vllm-openai:indexer-ssd-hca-prefetch-decodegate-20260528_0630
PATCH_TAR=/dev/shm/tutti_lazy_fix_full_structured.tar.gz
PATCH_ROOT=/tmp/lmcache_patch

sudo docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

for bdf in 0000:6f:00.0 0000:10:00.0 0000:1c:00.0 0000:4b:00.0 0000:e4:00.0 0000:88:00.0 0000:cc:00.0 0000:a2:00.0; do
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

sudo mkdir -p /mnt/nvme0 /mnt/nvme2 /mnt/nvme3 /mnt/nvme4 /mnt/nvme5 /mnt/nvme6 /mnt/nvme8 /mnt/nvme9

mount_by_bdf() {
  local bdf="$1"
  local target="$2"
  findmnt "${target}" >/dev/null && return 0
  local ctrl ns
  ctrl=$(for c in /sys/class/nvme/nvme*; do
    [ -e "$c" ] || continue
    if readlink -f "$c" | grep -q "${bdf}"; then
      basename "$c"
      break
    fi
  done)
  if [ -z "${ctrl}" ]; then
    echo "No nvme controller found for ${bdf} -> ${target}" >&2
    return 1
  fi
  ns="${ctrl}n1"
  if [ ! -b "/dev/${ns}" ]; then
    ns=$(ls "/dev/${ctrl}"n* 2>/dev/null | sed -E 's#/dev/##' | head -n 1 || true)
  fi
  if [ -z "${ns}" ] || [ ! -b "/dev/${ns}" ]; then
    echo "No nvme namespace block device found for ${bdf} -> ${target}" >&2
    return 1
  fi
  echo "mounting /dev/${ns} -> ${target} (${bdf})"
  sudo mount "/dev/${ns}" "${target}"
}

mount_by_bdf 0000:6f:00.0 /mnt/nvme0
mount_by_bdf 0000:10:00.0 /mnt/nvme2
mount_by_bdf 0000:1c:00.0 /mnt/nvme3
mount_by_bdf 0000:4b:00.0 /mnt/nvme4
mount_by_bdf 0000:e4:00.0 /mnt/nvme5
mount_by_bdf 0000:88:00.0 /mnt/nvme6
mount_by_bdf 0000:cc:00.0 /mnt/nvme8
mount_by_bdf 0000:a2:00.0 /mnt/nvme9

sudo rm -rf "${PATCH_ROOT}"
sudo mkdir -p "${PATCH_ROOT}"
sudo tar -xzf "${PATCH_TAR}" -C "${PATCH_ROOT}"
sudo cp /dev/shm/tutti_direct_loader.py \
  "${PATCH_ROOT}/v1/gpu_connector/tutti_direct_loader.py"
sudo cp /dev/shm/cache_engine.py \
  "${PATCH_ROOT}/v1/cache_engine.py"
sudo cp /dev/shm/storage_manager.py \
  "${PATCH_ROOT}/v1/storage_backend/storage_manager.py"
sudo mkdir -p "${PATCH_ROOT}/integration/vllm"
sudo cp /dev/shm/vllm_v1_adapter.py \
  "${PATCH_ROOT}/integration/vllm/vllm_v1_adapter.py"
sudo mkdir -p "${PATCH_ROOT}/tests/v1" "${PATCH_ROOT}/docs/design/v1"
sudo cp /dev/shm/test_tutti_direct_loader.py \
  "${PATCH_ROOT}/tests/v1/test_tutti_direct_loader.py"
sudo cp /dev/shm/tutti_codebase_analysis.md \
  "${PATCH_ROOT}/docs/design/v1/tutti_codebase_analysis.md"
sudo chmod -R a+rX "${PATCH_ROOT}"

echo "== patch markers =="
grep -n "def _estimate_chunk_ios" \
  "${PATCH_ROOT}/v1/gpu_connector/tutti_direct_loader.py"
grep -n "while batch_start < n" \
  "${PATCH_ROOT}/v1/gpu_connector/tutti_direct_loader.py"
grep -n "Tutti warmup after final cold store" \
  "${PATCH_ROOT}/v1/cache_engine.py"

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
  -v "${PATCH_ROOT}":/patches:ro \
  -v /tmp/patch_connector.py:/scripts/patch_connector.py:ro \
  "${IMAGE}" \
  /tmp/startup_256k_tutti.sh

echo "container_started"
sudo docker ps --filter "name=${CONTAINER}" --no-trunc
