#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Prediction only: keep every append query row, partition its complete causal
# K interval contiguously across TP ranks, and return rank-local top-k/8 IDs.
# The authoritative indexer remains unpartitioned.
export LMCACHE_TRUE_INDEXER_CP=0
export LMCACHE_PROXY_INDEXER_CP=0
export LMCACHE_PROXY_INDEXER_CP_EXCHANGE=0
export LMCACHE_PROXY_INDEXER_K_CP=1
export LMCACHE_PROXY_INDEXER_K_CP_SIZE=8
export LMCACHE_GLM_DSA_ASYNC_PREDICTION=1
export LMCACHE_GLM_DSA_SHARED_CORRECTION_AT_CONSUMER=1

# GLM's official top-k is 2048, hence up to 256 candidates per rank and query.
# Keep headroom for the row union; compact AllGather still transmits only the
# actual width rather than this capacity.
export LMCACHE_CSA_OWNER_BLOCKS_PER_RANK="${LMCACHE_CSA_OWNER_BLOCKS_PER_RANK:-512}"
# GLM K-cache rows are much wider than DSv4 rows.  The former 128 MiB slot
# could warm fewer rows than the 512-row owner bound and made the first gather
# fail before compact-width transmission was reached.  Two slots at 1 GiB cap
# still allocate only the geometry requested by warm(), while safely covering
# 512 rows/rank for this model.
export LMCACHE_SSD_TP_STAGING_SLOT_BYTES="${LMCACHE_SSD_TP_STAGING_SLOT_BYTES:-1073741824}"

exec bash "${script_dir}/launch_glm52_owner_io_v60.sh"
