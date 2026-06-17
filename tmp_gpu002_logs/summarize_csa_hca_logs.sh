#!/usr/bin/env bash
set -euo pipefail

container="${1:-dsv4-256k-measure-tutti}"
since="${2:-40m}"
log="/tmp/csa_hca_${container}.log"

sudo docker logs --since "$since" "$container" >"$log" 2>&1 || true

echo "==== counts since $since ===="
for pat in \
  "FileNotFoundError" \
  "reuse_prefetch_seed failed" \
  "previous prefill eviction failed" \
  "reuse_prefetch_seed complete" \
  "reuse prefetch seeded" \
  "reuse prefetch prepared" \
  "residual_proxy prefill" \
  "event=prefill_fire_async " \
  "event=prefill_fire_async_skip" \
  "prefill_correct_true_topk" \
  "event=correct_true_topk" \
  "HCAPrefetchManager: seeded" \
  "HCAPrefetchManager: fire" \
  "HCAPrefetchManager: drain" \
  "Retrieved 41984 out of 41984"; do
  printf '%-42s %s\n' "$pat" "$(grep -cF "$pat" "$log" || true)"
done

echo "==== first errors ===="
grep -E 'FileNotFoundError|reuse_prefetch_seed failed|previous prefill eviction failed|Traceback' "$log" | head -n 40 || true

echo "==== CSA seed complete samples ===="
grep -F "reuse_prefetch_seed complete" "$log" | head -n 20 || true

echo "==== CSA prefill samples ===="
grep -E 'residual_proxy prefill|event=prefill_fire_async |prefill_correct_true_topk' "$log" | head -n 80 || true

echo "==== HCA samples ===="
grep -E 'HCAPrefetchManager: (enabled|reuse prefetch prepared|seeded|fire|drain)' "$log" | head -n 120 || true

echo "==== retrieve samples ===="
grep -E 'Retrieved 41984 out of 41984|LMCACHE_RETRIEVE_PROFILE req_id=.*retrieved=41984' "$log" | tail -n 60 || true
