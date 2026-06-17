#!/usr/bin/env bash
set -euo pipefail
sudo docker logs --since 15m dsv4-256k-measure-tutti 2>&1 \
  | grep -E "Tutti warmup|Tutti lazy|TuttiDirectLoader|unmounting cache|LBA pre-scan|configured lazily|configured for LocalDiskBackend but unavailable|Retrieved 0|Retrieved 41984|Stored 1024|Stored 8192|Worker_TP[0-9]" \
  | sed -E 's/\x1b\[[0-9;]*m//g' \
  > /dev/shm/tutti_failure_grep.log || true
wc -l /dev/shm/tutti_failure_grep.log
tail -240 /dev/shm/tutti_failure_grep.log
