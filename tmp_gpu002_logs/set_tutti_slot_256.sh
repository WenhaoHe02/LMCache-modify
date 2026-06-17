#!/usr/bin/env bash
set -euo pipefail

sudo sed -i 's/tutti_slot_mb: 4/tutti_slot_mb: 256/' /tmp/startup_256k_tutti.sh
grep -n 'tutti_slot_mb' /tmp/startup_256k_tutti.sh
