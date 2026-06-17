#!/usr/bin/env bash
set -euo pipefail

sudo docker logs --since 5m dsv4-256k-measure-tutti 2>&1 \
  | sed -E 's/\x1b\[[0-9;]*m//g' \
  | grep -E "Tutti warmup|Waiting for in-flight|Traceback|ERROR|Exception|RuntimeError|LMCache hit tokens|Retrieved|Stored|Tutti lazy pre-scan|TuttiDirectLoader initialised|unmounting|FIEMAP|LBA" \
  | tail -n 260 || true
