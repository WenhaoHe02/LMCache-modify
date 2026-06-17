#!/usr/bin/env bash
set -euo pipefail

STARTUP=/tmp/startup_256k_tutti.sh
ALL_BDFS="0000:6f:00.0,0000:10:00.0,0000:1c:00.0,0000:4b:00.0,0000:e4:00.0,0000:88:00.0,0000:cc:00.0,0000:a2:00.0"

sudo sed -i \
  "s#tutti_pci_bdfs: \".*\"#tutti_pci_bdfs: \"${ALL_BDFS}\"#" \
  "${STARTUP}"

echo "== updated startup =="
grep -n "tutti_pci_bdfs" "${STARTUP}"
