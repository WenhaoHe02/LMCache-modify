#!/usr/bin/env bash
set -euo pipefail

echo "== tar c_ops candidates =="
for tarball in /dev/shm/*.tar.gz /tmp/*.tar.gz; do
  [[ -f "$tarball" ]] || continue
  if tar -tzf "$tarball" 2>/dev/null | grep -q 'c_ops.cpython-312-x86_64-linux-gnu.so'; then
    echo "-- $tarball"
    tar -tzf "$tarball" | grep 'c_ops.cpython-312-x86_64-linux-gnu.so'
  fi
done

echo "== filesystem c_ops candidates =="
find /dev/shm /tmp /opt -name 'c_ops.cpython-312-x86_64-linux-gnu.so' -type f 2>/dev/null \
  -exec ls -lh {} \; | head -n 40 || true
