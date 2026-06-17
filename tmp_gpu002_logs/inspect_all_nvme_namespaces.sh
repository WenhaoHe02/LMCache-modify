#!/usr/bin/env bash
set -euo pipefail

echo "== mounted nvme namespaces =="
findmnt -rn -o TARGET,SOURCE | grep '/mnt/nvme' || true

echo
echo "== all nvme namespace pci mapping =="
for dev in /sys/block/nvme*n*; do
  [ -e "$dev" ] || continue
  name=$(basename "$dev")
  real=$(readlink -f "$dev")
  pci=$(echo "$real" | grep -oE '0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | tail -n 1 || true)
  mount=$(findmnt -rn -S "/dev/${name}" -o TARGET 2>/dev/null || true)
  serial=$(cat "$dev/device/serial" 2>/dev/null || true)
  model=$(cat "$dev/device/model" 2>/dev/null || true)
  echo "$name pci=${pci:-unknown} mount=${mount:-none} serial=${serial:-unknown} model=${model:-unknown}"
done

echo
echo "== pci nvme class via sysfs =="
for ctrl in /sys/class/nvme/nvme*; do
  [ -e "$ctrl" ] || continue
  name=$(basename "$ctrl")
  real=$(readlink -f "$ctrl")
  pci=$(echo "$real" | grep -oE '0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | tail -n 1 || true)
  serial=$(cat "$ctrl/serial" 2>/dev/null || true)
  model=$(cat "$ctrl/model" 2>/dev/null || true)
  echo "$name pci=${pci:-unknown} serial=${serial:-unknown} model=${model:-unknown}"
done
