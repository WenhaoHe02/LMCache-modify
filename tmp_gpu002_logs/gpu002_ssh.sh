#!/usr/bin/env bash
set -euo pipefail
sshpass -p 'Pass2025' ssh -tt -o StrictHostKeyChecking=no zbuser02@172.16.8.32 "$@"
