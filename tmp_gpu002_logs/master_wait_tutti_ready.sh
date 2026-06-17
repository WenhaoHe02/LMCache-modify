#!/usr/bin/env bash
set -euo pipefail

GPU_HOST=zbuser02@172.16.8.32
SSHPASS="sshpass -p 'Pass2025'"
SSH_OPTS="-o StrictHostKeyChecking=no"

eval "${SSHPASS} scp ${SSH_OPTS} /tmp/wait_tutti_ready_and_verify.sh ${GPU_HOST}:/dev/shm/"
eval "${SSHPASS} ssh ${SSH_OPTS} ${GPU_HOST} 'bash /dev/shm/wait_tutti_ready_and_verify.sh'"
