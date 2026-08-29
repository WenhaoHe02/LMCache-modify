# Key-sharded rank-local prediction with a fail-open target gate

Status: experimental, opt-in. Applies to the GLM/DSv4 CSA speculative
prefetch path (`lmcache/v1/indexer_ssd_manager.py`,
`lmcache/v1/csa_prefill_cp_scorer.py`,
`lmcache/v1/csa_attention_kv_prefetch_manager.py`).

## Problem

The measured `key_*` speculative CP modes (v68/v69 A/B runs, 128k steady
state) regressed TTFT below the no-prefetch baseline (~0.9 s vs ~0.58 s).
Profiling attributed the regression to synchronization added on the model's
critical path, not to prediction quality:

1. **Model-stream ordering after proxy scoring.** The patched indexer forward
   called `order_true_indexer_after_proxy()` before every official indexer,
   and all targets shared one proxy CUDA stream (forced by a single shared K
   workspace). The model waited for *all* queued speculative scoring at every
   CSA layer.
2. **Blocking target gate.** `wait_for_csa_attention_kv_prediction` joined the
   proxy scoring future and the prediction I/O future per layer
   (`wait_ms` ≈ 10–22 ms × 30 layers ≈ 0.7 s of critical-path waiting per
   rank), while the misses it avoided were only ~0–50 blocks per layer.
3. **Gate-side collectives.** `key_sharded_owner` ran a KV-row AllGather under
   the shard collective lock inside the gate.

## Design

New partition mode `key_sharded_local`
(`LMCACHE_CSA_PREFETCH_CP_MODE=key_sharded_local`) plus a new gate policy
(`LMCACHE_CSA_PREDICTION_GATE=join|fail_open`, defaulting to `fail_open` for
this mode and `join` for all others):

- **Sharded scoring, replicated reads.** Each rank scores the shared sampled
  query rows against its own block-cyclic K shard
  (`prefill_cp_key_block_partition`) and emits one `1/world_size` share of
  the global candidate budget (`_proxy_candidate_block_budget`). The bounded
  IDs are exchanged with the existing in-hook GPU AllGather (the same tiny
  collective the `query` mode uses, enqueued on the model stream inside the
  source decoder hook — not at the consumer gate). Every rank then reads the
  exchanged union from its own SSD replica: `fire_predicted_reads(...,
  shard_local_only=True)` forces the preserved rank-local read path and never
  creates deferred shard-gather work. Requires
  `LMCACHE_SSD_TP_CSA_REPLICA_VERIFIED`-grade replicas.
- **Fail-open target gate.** Under `fail_open`, the gate closes the target's
  submission window (so a late proxy result is dropped by the existing expiry
  check), harvests already-completed futures, and *skips* running ones —
  re-tracking them for request teardown. It never blocks on speculative work
  and never cancels work that may own storage-side resources.
- **Correction-side barrier.** Miss filtering dedupes against pending
  bookings, so with a fail-open gate the correction path must not trust the
  drain event until bookings resolve. `wait_for_pending_reads(layer_id)` was
  added between miss submission and `drain_for_layer` in the patched indexer
  forward. It waits only for I/O already submitted to the storage layer (the
  same reads the old gate would have joined much earlier), preserving the
  invariant that attention never consumes a block whose scatter is not
  ordered on the synchronized drain event.
- **No model-stream ordering.** `order_true_indexer_after_proxy` is removed.
  Per-target proxy CUDA streams are restored; the scorer's private K
  workspace is keyed by a per-target slot
  (`workspace_slot`, `LMCACHE_CSA_PROXY_WORKSPACE_SLOTS`, default 4) so
  concurrent targets cannot overwrite each other's gather scratch.

`fail_open` is honoured only when the prediction path cannot have produced
deferred collective work (transport not gate-aligned, or rank-local reads).
Otherwise the gate silently falls back to `join`, because skipping a
deferred KV-row collective on a subset of ranks would deadlock the rest.

## Correctness argument

- The resident bitmap is only ever set for blocks whose read completed and
  whose scatter is ordered on the layer's drain event (`_submit_reads_active`
  publishes under `pending_reads_lock`).
- Blocks the fail-open gate did not wait for fall into two classes:
  - *Booked (pending)*: miss filtering skips them, and the new
    `wait_for_pending_reads` barrier blocks until their submission resolves
    and publishes a drain event that `drain_for_layer` then synchronizes.
  - *Not yet booked*: the gate closed the submission window first (under the
    producer's lock), so the late producer observes the expiry and drops the
    work; the block is simply a miss and is demand-read authoritatively.
- Rank-local reads never publish another rank's bytes, so replica divergence
  is the only cross-rank correctness assumption, identical to the existing
  `LMCACHE_SSD_TP_CSA_REPLICA_VERIFIED` attestation.

## Rollout / A/B

`scripts/run_container_cp8_ab.sh` forwards `LMCACHE_CSA_PREFETCH_CP_MODE`,
`LMCACHE_CSA_PREDICTION_GATE`, and `LMCACHE_CSA_PROXY_WORKSPACE_SLOTS`.
Recommended A/B against no-prefetch and `query`:

```bash
LMCACHE_CSA_PREFETCH_CP_MODE=key_sharded_local \
LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE=4 \
scripts/run_container_cp8_ab.sh
```

Success criterion: steady-state TTFT at or below the no-prefetch baseline
with `prediction_target_gate ... fail_open_skipped>0` appearing rarely (a
high skip rate means prediction I/O is not finishing inside the two-layer
lookahead window and the budget or stride should be reduced).
