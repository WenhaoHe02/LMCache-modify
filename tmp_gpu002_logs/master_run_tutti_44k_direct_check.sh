#!/usr/bin/env bash
set -euo pipefail

GPU_HOST=zbuser02@172.16.8.32
SSHPASS="sshpass -p 'Pass2025'"
SSH_OPTS="-o StrictHostKeyChecking=no"

eval "${SSHPASS} scp ${SSH_OPTS} /tmp/run_tutti_44k_direct_check.sh ${GPU_HOST}:/dev/shm/"
eval "${SSHPASS} ssh ${SSH_OPTS} ${GPU_HOST} 'bash /dev/shm/run_tutti_44k_direct_check.sh'"
