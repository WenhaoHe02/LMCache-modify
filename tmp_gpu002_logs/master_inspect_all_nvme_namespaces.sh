#!/usr/bin/env bash
set -euo pipefail

GPU_HOST=zbuser02@172.16.8.32
SSHPASS="sshpass -p 'Pass2025'"
SSH_OPTS="-o StrictHostKeyChecking=no"

eval "${SSHPASS} scp ${SSH_OPTS} /tmp/inspect_all_nvme_namespaces.sh ${GPU_HOST}:/dev/shm/"
eval "${SSHPASS} ssh ${SSH_OPTS} ${GPU_HOST} 'bash /dev/shm/inspect_all_nvme_namespaces.sh'"
