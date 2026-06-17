# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
set -euo pipefail

for bdf in 0000:10:00.0 0000:1c:00.0 0000:4b:00.0 0000:e4:00.0 0000:cc:00.0 0000:a2:00.0; do
  printf "%s driver=" "${bdf}"
  readlink "/sys/bus/pci/devices/${bdf}/driver" | sed 's#.*/##'
  printf "%s override=" "${bdf}"
  cat "/sys/bus/pci/devices/${bdf}/driver_override"
done
