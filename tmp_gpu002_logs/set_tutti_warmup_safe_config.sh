#!/usr/bin/env bash
set -euo pipefail

sudo sed -i \
  -e 's/tutti_n_slots: [0-9][0-9]*/tutti_n_slots: 2/' \
  -e 's/tutti_slot_mb: [0-9][0-9]*/tutti_slot_mb: 128/' \
  -e 's/--max-model-len 262144/--max-model-len 131072/' \
  -e 's/--gpu-memory-utilization 0\.[0-9][0-9]*/--gpu-memory-utilization 0.84/' \
  -e 's/export LMCACHE_INDEXER_MAX_SEQ_LEN=.*/export LMCACHE_INDEXER_MAX_SEQ_LEN=131072/' \
  /tmp/startup_256k_tutti.sh

if grep -q 'LMCACHE_TUTTI_WARMUP_AFTER_STORE_DELAY_SEC' /tmp/startup_256k_tutti.sh; then
  sudo sed -i \
    's/export LMCACHE_TUTTI_WARMUP_AFTER_STORE_DELAY_SEC=.*/export LMCACHE_TUTTI_WARMUP_AFTER_STORE_DELAY_SEC=15/' \
    /tmp/startup_256k_tutti.sh
else
  sudo sed -i \
    '/export LMCACHE_INDEXER_MAX_SEQ_LEN=/a export LMCACHE_TUTTI_WARMUP_AFTER_STORE_DELAY_SEC=15' \
    /tmp/startup_256k_tutti.sh
fi

grep -nE 'tutti_n_slots|tutti_slot_mb|max-model-len|gpu-memory-utilization|LMCACHE_INDEXER_MAX_SEQ_LEN|LMCACHE_TUTTI_WARMUP_AFTER_STORE_DELAY_SEC' \
  /tmp/startup_256k_tutti.sh
