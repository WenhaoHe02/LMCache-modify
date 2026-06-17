#!/usr/bin/env bash
set -euo pipefail

echo "== startup tutti config =="
grep -nE "tutti_|local_disk|path_sharding|max_model_len|gpu-memory-utilization" \
  /tmp/startup_256k_tutti.sh || true

echo
echo "== container runtime config snippets =="
sudo docker exec dsv4-256k-measure-tutti bash -lc \
  "grep -R -nE 'tutti_|local_disk|path_sharding' /tmp /opt/venv 2>/dev/null | head -n 120" \
  || true

echo
echo "== mounts =="
findmnt -rn -o TARGET,SOURCE,FSTYPE,OPTIONS | grep -E '/mnt/nvme|TARGET' || true

echo
echo "== nvme block devices =="
lsblk -o NAME,TYPE,SIZE,MOUNTPOINT,MODEL,SERIAL,TRAN | grep -E 'nvme|NAME' || true

echo
echo "== nvme sysfs pci mapping =="
for dev in /sys/block/nvme*n1; do
  [ -e "$dev" ] || continue
  name=$(basename "$dev")
  real=$(readlink -f "$dev")
  pci=$(echo "$real" | grep -oE '0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | tail -n 1 || true)
  mount=$(findmnt -rn -S "/dev/${name}" -o TARGET 2>/dev/null || true)
  echo "$name pci=${pci:-unknown} mount=${mount:-none} path=$real"
done

echo
echo "== nvme pci controllers =="
lspci -D | grep -i 'Non-Volatile memory\\|NVMe' || true
