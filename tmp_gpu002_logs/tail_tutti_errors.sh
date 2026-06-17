#!/usr/bin/env bash
set -euo pipefail

sudo docker logs --since 8m dsv4-256k-measure-tutti 2>&1 \
  | grep -E "Tutti|LMCache hit tokens|need to load|Retrieved|LBA pre-scan|FIEMAP|NVMe READ|direct load|unmounting|RuntimeError|Traceback|ERROR|ValueError|Exception" \
  | tail -n 320 || true
