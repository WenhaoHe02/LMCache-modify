#!/usr/bin/env bash
set -euo pipefail
sshpass -p 'Pass2025' ssh -o StrictHostKeyChecking=no zbuser02@172.16.8.32 'bash /dev/shm/run_tutti_output_check_no_restart.sh'
