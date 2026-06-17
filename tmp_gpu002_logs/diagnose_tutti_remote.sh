# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
set -euo pipefail

echo "== devices =="
ls -l /dev/snvm_control /dev/ssnvme* 2>/dev/null || true

echo "== target bdf drivers =="
for bdf in 0000:10:00.0 0000:1c:00.0 0000:4b:00.0 0000:e4:00.0 0000:cc:00.0 0000:a2:00.0 0000:6f:00.0 0000:88:00.0 0000:bb:00.0; do
  printf "%s " "${bdf}"
  if [ -e "/sys/bus/pci/devices/${bdf}/driver" ]; then
    readlink "/sys/bus/pci/devices/${bdf}/driver" | sed 's#.*/##'
  else
    echo "MISSING"
  fi
done

echo "== mounted nvme block devices =="
for d in /sys/block/nvme*n*; do
  dev=$(basename "${d}")
  bdf=$(readlink -f "${d}/device" | sed -n 's#.*\(0000:[0-9a-f:]*\.[0-9]\).*#\1#p')
  target=$(findmnt -nr -S "/dev/${dev}" -o TARGET 2>/dev/null || true)
  echo "${dev} ${bdf} ${target}"
done | sort

echo "== snvme/nvme modules =="
lsmod | grep -E '^(snvme|snvme_core|nvme)\s' || true

echo "== recent kernel snvme messages =="
sudo dmesg | tail -120 | grep -Ei 'snvme|ssnvme|nvme|No such|ENODEV|bind' || true
