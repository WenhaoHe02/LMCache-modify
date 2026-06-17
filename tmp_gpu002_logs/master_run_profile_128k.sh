#!/usr/bin/env bash
set -euo pipefail
sshpass -p 'Pass2025' ssh -o StrictHostKeyChecking=no zbuser02@172.16.8.32 "tr -d '\r' < /dev/shm/run_tutti_profile_128k.sh > /dev/shm/run_tutti_profile_128k.unix.sh && mv /dev/shm/run_tutti_profile_128k.unix.sh /dev/shm/run_tutti_profile_128k.sh && bash /dev/shm/run_tutti_profile_128k.sh"
