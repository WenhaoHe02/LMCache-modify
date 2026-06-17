#!/usr/bin/env bash
set -euo pipefail

GPU_HOST=zbuser02@172.16.8.32
SSHPASS="sshpass -p 'Pass2025'"
SSH_OPTS="-o StrictHostKeyChecking=no"

eval "${SSHPASS} scp ${SSH_OPTS} /tmp/set_tutti_all_ranks_bdfs.sh /tmp/restart_tutti_container_packing.sh /tmp/tutti_direct_loader.py /tmp/cache_engine.py /tmp/storage_manager.py /tmp/vllm_v1_adapter.py /tmp/test_tutti_direct_loader.py /tmp/tutti_codebase_analysis.md ${GPU_HOST}:/dev/shm/"
eval "${SSHPASS} ssh ${SSH_OPTS} ${GPU_HOST} 'bash /dev/shm/set_tutti_all_ranks_bdfs.sh; bash /dev/shm/restart_tutti_container_packing.sh'"
