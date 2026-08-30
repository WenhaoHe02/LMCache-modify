#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Owner-path multilen A/B matrix across DSv4-flash, GLM-5.2, and DSv4-Pro.
#
# Algorithm under test (semantics pinned, implementation-only changes):
#   K split block-cyclically across 8 ranks (interleave 64), each rank takes
#   top-k/8 on its local K shard with no global top-K merge, each rank reads
#   only its own candidates' KV from its SSD replica, and the consumer gate
#   AllGathers rows to restore the complete KV.
#
# Implementation changes being validated (vs the v44/v59 baseline):
#   1. resident-block filter: owner SSD reads skip blocks already in the
#      K cache (multi-chunk appends re-select heavily overlapping sets);
#   2. effective-width AllGather: the KV collective shrinks from the warmed
#      OWNER_BLOCKS_PER_RANK capacity to the actual max advertised count
#      (fixes the GLM rejection: 512-block fixed width = 576 MiB/layer);
#   3. gate publishes the gather event through the layer drain instead of
#      synchronizing inside the collective lock;
#   4. no per-layer send-slot zero-fill;
#   5. receive-position fix for duplicate candidates (correctness).
#
# Usage on the serving host (one model at a time):
#   MODEL=flash|glm|pro SUFFIXES="256 2048 8192" \
#     bash scripts/run_owner_multilen_ab.sh
#
# Histories swept per model (raw tokens):
#   flash: 32k 131k 262k 491k 786k 1m
#   glm:   131k 262k          (GLM corpus bound)
#   pro:   131k 262k 393k 524k
#
# Each point runs: no-prefetch baseline, then owner path, 3 formal reps,
# steady state = median of hit reps. Compare per-layer telemetry:
#   CSA_SHARD_GATHER ... gather_total_bytes   (must shrink with change 2)
#   IO_LOADER_CALL blocks=                    (must shrink with change 1)
#   correction ... wait_ms                    (must shrink with change 3)
set -euo pipefail

MODEL="${MODEL:-flash}"
SUFFIXES="${SUFFIXES:-256}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/owner_multilen_ab_$(date +%Y%m%d)}"

case "$MODEL" in
  flash)
    LAUNCH=".codex_remote_work/v40_launch_dsv4_flash.sh"
    HISTORIES="${HISTORIES:-32768 131072 262144 491520 786432 1048320}"
    ;;
  glm)
    LAUNCH="scripts/launch_glm52_final.sh"
    HISTORIES="${HISTORIES:-131072 262144}"
    ;;
  pro)
    LAUNCH="scripts/launch_dsv4_pro_final.sh"
    HISTORIES="${HISTORIES:-131072 262144 393216 524288}"
    ;;
  *)
    echo "MODEL must be flash, glm, or pro" >&2
    exit 1
    ;;
esac

mkdir -p "$RESULT_ROOT"

common_env=(
  LMCACHE_CSA_PREFETCH_CP_MODE=key_sharded_owner
  LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=64
  LMCACHE_CSA_OWNER_BLOCKS_PER_RANK=512
  LMCACHE_SSD_TP_SHARDED_PREFETCH=1
  LMCACHE_SSD_TP_SHARD_CSA=1
  LMCACHE_SSD_TP_STAGING_SLOT_BYTES=268435456
  LMCACHE_CSA_PREDICTION_DENSIFY_PERCENT=0
  LMCACHE_CSA_PROPAGATE_TOPK_LAYERS=0
  LMCACHE_CSA_PREFETCH_MIN_HISTORY_BLOCKS=0
)

for history in $HISTORIES; do
  for suffix in $SUFFIXES; do
    for arm in nopref owner; do
      tag="${MODEL}_${arm}_${history}_${suffix}"
      echo "=== $tag"
      if [ "$arm" = "nopref" ]; then
        arm_env=(LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER=default:0)
      else
        arm_env=("${common_env[@]}")
      fi
      env "${arm_env[@]}" bash "$LAUNCH" \
        --history-tokens "$history" \
        --suffix-tokens "$suffix" \
        --formal-repetitions 3 \
        --result-json "$RESULT_ROOT/$tag.json" \
        2>&1 | tee "$RESULT_ROOT/$tag.log"
    done
  done
done

echo "results in $RESULT_ROOT"
