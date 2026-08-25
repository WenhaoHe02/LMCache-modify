# DeepSeek-V4-Pro adaptive predictive prefetch V17

V17 preserves sparse CSA prediction for small appends and avoids running a
proxy whose predicted block union is already effectively dense. It also
orders request-start HCA reads by consumer layer so later HCA layers cannot
win the single Tutti queue ahead of earlier layers.

## Cost model

For each CSA target, the source-layer hook estimates the expected historical
block coverage from:

- cached history length;
- append/recompute rows;
- CP query-sampling stride; and
- the target's proxy top-K width.

The model uses occupancy coverage
`1 - exp(-sampled_queries * topk_blocks / history_blocks)`. When estimated
coverage reaches `LMCACHE_CSA_ADAPTIVE_DENSE_THRESHOLD_PERCENT` (80 by
default), the hook submits the exact full CSA layer through the existing
lookahead path and skips proxy scoring. Below the threshold, the prior sparse
prediction path is unchanged. Both behaviors preserve the official indexer's
true top-K as the correctness source.

Enable the production path with:

```bash
export LMCACHE_HCA_PREFIRE_ALL_LAYERS=1
export LMCACHE_CSA_ADAPTIVE_DENSE_PREFETCH=1
export LMCACHE_CSA_ADAPTIVE_DENSE_THRESHOLD_PERCENT=80
```

All three variables default to disabled or conservative behavior outside the
versioned Pro launcher.

## Validated key point

On gpu002 with a 196,608-token exact hit and 8,192 recompute tokens, profile
and accuracy instrumentation disabled:

- Tutti no-prefetch: `1.496083 / 1.501303 / 1.496119 s` TTFT.
- V17: `1.329654 / 1.330925 / 1.330233 s` TTFT.
- Six additional exact-hit requests: `1.333356 / 1.318514 / 1.319786 /
  1.311709 / 1.332086 / 1.331791 s` TTFT.

The frozen gpu002 runtime is
`20260825_dsv4_pro_adaptive_dense_v17`; its launcher keeps profile, accuracy,
timing, and nsys capture disabled.
