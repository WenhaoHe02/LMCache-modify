#!/usr/bin/env bash
set -euo pipefail

echo "== container =="
sudo docker ps -a --filter name=dsv4-256k-measure-tutti

echo "== concise tutti timeline =="
sudo docker logs --since 15m dsv4-256k-measure-tutti 2>&1 \
  | grep -E "Reqid:|Tutti lazy|TuttiDirectLoader|snvme|SNVM|unmounting|NVMe READ|Retrieved|LMCache hit tokens|need to load|timed out|EngineCore encountered|RPC call|ERROR|WARNING" \
  | grep -v "SchedulerOutput" \
  | tail -n 240 || true

echo "== mounts after attempt =="
findmnt /mnt/nvme2 /mnt/nvme3 /mnt/nvme4 /mnt/nvme5 /mnt/nvme8 /mnt/nvme9 || true

echo "== driver state =="
for bdf in 0000:10:00.0 0000:1c:00.0 0000:4b:00.0 0000:e4:00.0 0000:cc:00.0 0000:a2:00.0; do
  printf "%s " "$bdf"
  readlink "/sys/bus/pci/devices/$bdf/driver" 2>/dev/null | xargs basename || true
done
