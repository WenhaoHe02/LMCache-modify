# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
set -euo pipefail

for _ in {1..120}; do
  if curl -sS --max-time 2 http://127.0.0.1:8000/v1/models >/tmp/tutti_ready.out 2>/tmp/tutti_ready.err; then
    cat /tmp/tutti_ready.out
    exit 0
  fi
  sleep 5
done

cat /tmp/tutti_ready.err
exit 1
