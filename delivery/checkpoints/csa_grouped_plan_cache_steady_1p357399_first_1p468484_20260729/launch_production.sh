#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Correctness profiling performs an additional checksum reduction and
# collective after every real CSA data gather. Keep it available through
# launch_full.sh, but exclude it and verbose timing logs from production TTFT.
export LMCACHE_SSD_TP_DEBUG_VERIFY=0
export LMCACHE_INDEXER_PROFILE_ACCURACY=0
export LMCACHE_INDEXER_TIMING=0
export LMCACHE_CSA_ATTENTION_KV_TIMING=0
export LMCACHE_TTFT_STAGE_PROFILE=0
export LMCACHE_LOOKUP_TOKEN_DIAGNOSTICS=0
# Reuse an exact generation + slot-binding plan on repeated hits. The cache
# fails closed and rebuilds when either the published LBA revision or vLLM's
# physical destination mapping changes.
export LMCACHE_DSV4_STREAMING_PLAN_CACHE_CAPACITY="${LMCACHE_DSV4_STREAMING_PLAN_CACHE_CAPACITY:-8}"
# Batch only deterministic Indexer/HCA full-layer reads. Predicted CSA and
# true-topK correction remain layer-local. The byte target makes long contexts
# automatically fall back to group size one.
export LMCACHE_DSV4_STATIC_IO_GROUP_MAX="${LMCACHE_DSV4_STATIC_IO_GROUP_MAX:-8}"
export LMCACHE_DSV4_STATIC_IO_GROUP_TARGET_MIB="${LMCACHE_DSV4_STATIC_IO_GROUP_TARGET_MIB:-32}"
# Reference counting still releases ordinary hot-path objects immediately.
# Disable cyclic-GC scans, which otherwise introduce periodic rank-local
# 0.7-0.8 second request-preparation stalls under composed-prefix traffic.
export LMCACHE_PY_ENABLE_GC=false

exec bash "$base/launch_full.sh" "${1:-on}"
