#!/usr/bin/env bash
set -euo pipefail

container="${1:-dsv4-256k-measure-tutti}"
since="${2:-20m}"
log="/tmp/recent_csa_hca_status.log"

sudo docker logs --since "$since" "$container" >"$log" 2>&1 || true

count() {
  local pattern="$1"
  grep -cF "$pattern" "$log" || true
}

echo "FileNotFoundError=$(count 'FileNotFoundError')"
echo "reuse_prefetch_seed_failed=$(count 'reuse_prefetch_seed failed')"
echo "previous_prefill_eviction_failed=$(count 'previous prefill eviction failed')"
echo "Tutti_direct_load_failed=$(count 'Tutti direct load failed')"
echo "Retrieved_119040=$(count 'Retrieved 119040 out of 119040')"
echo "CSA_seeded=$(count 'reuse prefetch seeded')"
echo "HCA_prepared=$(count 'reuse prefetch prepared')"
echo "HCA_drain=$(count 'HCAPrefetchManager: drain')"
echo "CSA_residual_proxy_prefill=$(count 'residual_proxy prefill')"
