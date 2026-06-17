#!/usr/bin/env bash
set -euo pipefail
sudo docker logs --since 10m dsv4-256k-measure-tutti 2>&1 \
  | grep -E "Tutti warmup|Tutti lazy|TuttiDirectLoader|remounted|cudaMalloc|LMCache hit tokens|need to load|Retrieved|Stored|unmounting|fallback|Internal Server|EngineCore|ERROR|Waiting for in-flight" \
  | tail -320 || true
