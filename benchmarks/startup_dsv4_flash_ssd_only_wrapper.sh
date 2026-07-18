#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Convert the Tutti startup script into a traditional LMCache SSD startup.

set -euo pipefail

source_startup=${SOURCE_STARTUP:-/startup_tutti_original.sh}
clean_startup=$(mktemp /tmp/startup_lmcache_ssd_only.XXXXXX.sh)
trap 'rm -f "$clean_startup"' EXIT

sed \
  -e 's/kv_object_store_enable: true/kv_object_store_enable: false/' \
  -e 's/kv_object_store_tutti_raw_enable: true/kv_object_store_tutti_raw_enable: false/' \
  -e '/kv_object_store_tutti_raw_region_path:/d' \
  -e '/^  tutti_device_path:/,/^  tutti_nsid:/d' \
  -e 's/export LMCACHE_KV_OBJECT_STORE_ENABLE=1/export LMCACHE_KV_OBJECT_STORE_ENABLE=0/' \
  -e 's/export LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE=1/export LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE=0/' \
  "$source_startup" > "$clean_startup"

if grep -q '^  tutti_device_path:' "$clean_startup"; then
  echo "failed to remove Tutti loader configuration" >&2
  exit 1
fi
if ! grep -q '^  kv_object_store_enable: false$' "$clean_startup"; then
  echo "failed to disable the KV object store" >&2
  exit 1
fi
if [[ ${DRY_RUN:-0} == 1 ]]; then
  cat "$clean_startup"
  exit 0
fi

exec bash "$clean_startup"
