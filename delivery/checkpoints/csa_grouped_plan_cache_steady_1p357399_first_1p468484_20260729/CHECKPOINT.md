# CSA grouped-plan-cache checkpoint

Status: **qualified performance checkpoint, not the final first-hit design**.

This snapshot preserves the exact-plan cache and adaptive deterministic-layer
I/O grouping implementation measured on GPU002 on 2026-07-29. It is based on
the protected `1.457558 s` CSA version and does not replace that baseline.

## Validated configuration

- Model: DeepSeek V4 Flash, CP8 / TP8.
- Prefix: 480,000 tokens.
- Recompute suffix: 8,192 tokens.
- `LMCACHE_DSV4_STREAMING_PLAN_CACHE_CAPACITY=8`.
- `LMCACHE_DSV4_STATIC_IO_GROUP_MAX=8`.
- `LMCACHE_DSV4_STATIC_IO_GROUP_TARGET_MIB=32`.
- All timed requests hit exactly 480,000 cached tokens.
- Each valid hit reported 168 CSA shard gathers, 21 on every rank, and no
  fatal marker.

## Measured result

| Metric | Time |
|---|---:|
| First actual hit after fresh cold admission | 1.468484 s |
| Following three-hit median | 1.357399 s |
| Stable seven-hit median | 1.347259 s |
| Stable seven-hit range | 1.344298-1.351067 s |

The first-hit penalty against the same-run three-hit median is 111.085 ms, or
8.18%. Steady state is 110.159 ms (7.56%) faster than the protected
`1.457558 s` baseline, but the first hit is 19.518 ms slower than that
baseline's first hit. Therefore this version is a useful recovery point, but
it does **not** meet the target that the first hit should be close to later
hits.

The next optimization must move exact-plan construction out of the first-hit
critical path. The safe direction is to precompile after the cold generation
is fully READY, then reuse only after exact generation/revision and physical
slot-mapping validation. A mismatch must remain a normal cache miss and rebuild.

## Evidence files

- `candidate_480k8192_1.jsonl`: primary fresh-admission run; all validations
  passed and `matrix_end.failures=0`.
- `candidate_480k8192_repeat7.jsonl`: steady-state samples. Its initial
  reused-cache `cold_store` validation is intentionally invalid for that
  runner mode, so do not treat its `matrix_end.failures=1` as a timed-hit
  failure; all seven timed hits are individually valid.
- `correctness_candidate_1.jsonl`: logical-output probe.
- `plan8_group1_480k8192.jsonl`: plan-cache-on, grouping-off comparison.

## Restore

Use the unchanged protected 1.457558 s image/overlay as the base, then overlay
the four files under `patches/v1/` and use the copied launch scripts. The full
local frozen copy is at:

`F:\LMCache\protected_versions\csa_grouped_plan_cache_steady_1p357399_first_1p468484_20260729`

