#!/usr/bin/env bash
set -euo pipefail

container="${1:-dsv4-256k-measure-tutti}"

echo "==== latest reuse_prefetch_seed traceback/context ===="
sudo docker logs --since 45m "$container" 2>&1 \
  | grep -A35 -B8 -E 'reuse_prefetch_seed failed|IndexError|RuntimeError|ValueError|Traceback' \
  | tail -n 260 || true

echo "==== latest CSA/HCA fire/drain/profile lines ===="
sudo docker logs --since 45m "$container" 2>&1 \
  | grep -E 'prefill_fire_async|residual_proxy prefill|correct_true_topk|prefill_correct_true_topk|HCAPrefetchManager: (seeded|fire|drain)|TUTTI_PROFILE|LMCACHE_RETRIEVE_PROFILE' \
  | tail -n 240 || true
