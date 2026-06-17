#!/usr/bin/env bash
set -euo pipefail

sudo sed -i \
  -e 's/tutti_slot_mb: [0-9][0-9]*/tutti_slot_mb: 128/' \
  -e 's/--max-model-len 262144/--max-model-len 131072/' \
  -e 's/--gpu-memory-utilization 0\.[0-9][0-9]*/--gpu-memory-utilization 0.88/' \
  -e 's/export LMCACHE_INDEXER_MAX_SEQ_LEN=.*/export LMCACHE_INDEXER_MAX_SEQ_LEN=131072/' \
  /tmp/startup_256k_tutti.sh

grep -nE 'tutti_slot_mb|max-model-len|gpu-memory-utilization|LMCACHE_INDEXER_MAX_SEQ_LEN' \
  /tmp/startup_256k_tutti.sh
