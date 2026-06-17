#!/usr/bin/env bash
set -euo pipefail

for bdf in 0000:6f:00.0 0000:10:00.0 0000:1c:00.0 0000:4b:00.0 0000:e4:00.0 0000:88:00.0 0000:cc:00.0 0000:a2:00.0; do
  echo "== ${bdf} =="
  if [ ! -e "/sys/bus/pci/devices/${bdf}" ]; then
    echo "missing pci device"
    continue
  fi
  driver=$(basename "$(readlink -f "/sys/bus/pci/devices/${bdf}/driver" 2>/dev/null || echo none)")
  echo "driver=${driver}"
  find "/sys/bus/pci/devices/${bdf}" -maxdepth 5 -type d \
    \( -name 'nvme*n*' -o -name 'snvme*n*' -o -name 'nvme*' \) \
    | sed 's#^#  #'
done

echo "== docker =="
sudo docker ps -a --filter name=dsv4-256k-measure-tutti --no-trunc || true
