#!/usr/bin/env bash
set -euo pipefail

GPU_HOST=zbuser02@172.16.8.32
SSHPASS="sshpass -p 'Pass2025'"
SSH_OPTS="-o StrictHostKeyChecking=no"

eval "${SSHPASS} scp ${SSH_OPTS} /tmp/repair_tutti_patch_root.sh ${GPU_HOST}:/dev/shm/"
eval "${SSHPASS} ssh ${SSH_OPTS} ${GPU_HOST} 'grep -n tests/v1 /dev/shm/repair_tutti_patch_root.sh; bash /dev/shm/repair_tutti_patch_root.sh'"
eval "${SSHPASS} ssh ${SSH_OPTS} ${GPU_HOST} 'sed -n \"1,220p\" /dev/shm/restart_tutti_container.sh'"
