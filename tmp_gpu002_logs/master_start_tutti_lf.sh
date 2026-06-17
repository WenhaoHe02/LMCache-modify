#!/usr/bin/env bash
set -euo pipefail
sshpass -p 'Pass2025' ssh -o StrictHostKeyChecking=no zbuser02@172.16.8.32 'bash /dev/shm/restart_tutti_container_packing.sh'
