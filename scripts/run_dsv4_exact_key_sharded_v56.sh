#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

base_launcher=${LMCACHE_BASE_LAUNCHER:?set LMCACHE_BASE_LAUNCHER}

# Exact speculative Indexer-K sharding: each TP rank scores one contiguous
# P/8 key slice and keeps k/8 candidates. Candidate IDs are not globally
# merged; each owner reads its local KV rows before gate-aligned KV AllGather.
export LMCACHE_CSA_PREFETCH_CP_MODE=key_contiguous_owner
export LMCACHE_CSA_PREFETCH_CP_SIZE=8
export LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=1
export LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE=1
export LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE=1
export LMCACHE_CSA_PREFETCH_CP_EXCHANGE_IDS=1
export LMCACHE_CSA_PREFETCH_CP_GLOBAL_BLOCK_COUNTS=0
export LMCACHE_CSA_PREFETCH_CP_GLOBAL_BLOCK_BITMAP=0
export LMCACHE_CSA_L1_PROXY_TOPK_TOKENS=512
export LMCACHE_CSA_PROXY_TOPK_TOKENS_BY_LAYER=
export LMCACHE_CSA_PREDICTION_GATE=join
export LMCACHE_CSA_OWNER_GPU_METADATA=1

# The independent HCA walker delayed the exact K-sharded CSA path by about
# 70 ms at 98K+256. HCA remains demand-loaded; only the harmful lookahead
# walker is disabled.
export LMCACHE_DSV4_HCA_WALKER=0
export LMCACHE_HCA_PREFIRE_FIRST_LAYER=0
export LMCACHE_HCA_PREFIRE_ALL_LAYERS=0

exec bash "${base_launcher}" "$@"
