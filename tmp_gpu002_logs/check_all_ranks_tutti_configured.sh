#!/usr/bin/env bash
set -euo pipefail

echo "== bdfs =="
grep -n "tutti_pci_bdfs" /tmp/startup_256k_tutti.sh

echo "== configured ranks =="
sudo docker logs --since 5m dsv4-256k-measure-tutti 2>&1 \
  | grep -E "TuttiDirectLoader configured lazily|Tutti disabled" \
  | sed -E 's/\x1b\[[0-9;]*m//g' \
  | tail -n 40 || true
