# CSA Attention-KV Prefetch: Overlap Analysis and TTFT Model

Data: 2026-07-04, gpu002, DSv4-Pro TP=8, `max_model_len=13200`,
`max_num_batched_tokens=1024`, `gpu_util=0.82`, clean-GPU window (fresh
reboot), crash-isolated protocol: per-point fresh server + cold caches,
2 reps (rep1 = cold store+hit cycle, rep2 = warm), sleep 90/5.
Raw JSONL: gpu002 `/tmp/rerun_20260704/iso_{on,off}/`.

## Grid results (steady state = median of hit-3/hit-5)

| point | base (cached) | extra (new) | OFF rep2 | ON rep2 | ON tax | OFF repeat | ON repeat |
|---|---|---|---|---|---|---|---|
| A | 11000 | 1000 | 0.67 | 1.26 | +0.59 | 0.25 | 0.83 |
| B | 8000 | 4000 | 1.66 | 2.20 | +0.54 | 0.26 | 0.83 |
| C | 5000 | 8000 | 3.18 | 4.36 | +1.18* | 0.27 | 0.84 |
| D | 2000 | 10000 | 4.05 | 4.81 | +0.76 | 0.43 | 1.07 |

\* C had the highest ON variance (3.90-4.82); its tax estimate is noisy.

Cold (rep1) steady: ON 4.7-6.3 vs OFF 4.6-5.5 — statistically tied in this
window (an earlier window showed ON winning cold 3.25 vs 5.16; cold numbers
are the most environment-sensitive, treat with care).

## TTFT model

```
TTFT_OFF(base, extra) ~= t0 + c*extra + r_hit(base+extra)
TTFT_ON (base, extra) ~= TTFT_OFF + F * n_chunks(base+extra)
```

- `c*extra` (compute) dominates OFF's trend: 0.67 -> 4.05 as extra goes
  1000 -> 10000 at near-constant total tokens (~12000). Fitting the OFF
  column: c ~= 0.37 s / 1000 extra tokens.
- `r_hit`: full-reuse read+scatter floor is the repeat row: ~0.25 s at 12000
  tokens for OFF (Tutti GPU-direct ~20 GB/s + scatter).
- **`F * n_chunks` is the entire ON-OFF gap.** n_chunks = total/1024 ~= 12.
  Per-chunk fire cost F ~= 0.05-0.06 s (30 CSA layers x ~1.5-2 ms
  fire+dispatch+correction). Measured tax 0.54-0.76 s across A/B/D matches
  12 x F almost exactly. The repeat row isolates it perfectly:
  ON 0.83 - OFF 0.26 = 0.57 s of pure fire tax with zero useful reads.

Cross-check with direct instrumentation (7/3 run16 logs): fire finish
median 1.48 ms x 30 layers x 12 chunks ~= 0.53 s; correction 0.48 ms x 360
~= 0.17 s. Sum ~= 0.7 s. Model, telemetry, and end-to-end all agree.

## Why ON does not win yet (at <=13K contexts)

At these lengths the OFF read path (~0.25 s full-reuse) is already so fast
that there is nothing for prefetch to hide: the MoE window per chunk
(~2 ms/layer) can hide reads, but the reads only cost ~2.8 ms each anyway.
Prefetch's upside is bounded by r_hit (~0.25 s) while its cost is
F*n_chunks (~0.6 s). **At 12K tokens the best possible ON is ~0.35 s slower
than OFF unless F drops.**

ON's regime advantage: (a) cold paths (one window showed -37%), and
(b) longer contexts, where r_hit grows linearly with base while
F*n_chunks grows with total/chunk_size — prefetch wins when
`r_hit_saved > F * n_chunks`, i.e., large base AND large chunks.

## Optimization space (ranked by measured leverage)

1. **Bigger chunks (batched 1024 -> 4096): measured ON steady 1.51 s at B**
   — beats OFF's 1.66! n_chunks drops 12 -> 3, tax 0.6 -> 0.15 s, and the
   MoE window per chunk grows 4x (more room to hide reads). BLOCKED by a
   concurrency bug: second rep crashes with CUDA illegal access in
   `_load_batch`/`stream.synchronize` (high fire concurrency); a
   forward-side lock made it worse (stalls DeepGEMM metadata), a
   proxy-side lock was insufficient. Needs structural fix: proxy-private
   indexer_op state (no swap of shared attrs) + staging buffer refcount
   audit under concurrent loads.
2. **Cheaper fires**: 1.5 ms median is mostly Python + stream setup +
   event bookkeeping per (layer, chunk). Batch all 30 layers' fires per
   chunk into one submission (amortise setup 30x); or C++/graph-capture
   the proxy scoring call.
3. **Fire only when useful**: skip fires when the request's chunks were
   fully resident at registration (pure-repeat detection). Would cut the
   repeat row tax to ~0 but not help first-hit; low risk, moderate win.
4. **Longer contexts** (needs max_model_len back to 16000+, kv_cache_bytes
   tuning at util=0.82): grows OFF's r_hit (the thing prefetch can
   actually save) relative to the tax. The 120K profile (memory:
   project_tutti_profile_120k) shows retrieve ~0.66 s/rank at 120K —
   at such lengths a working overlap can hide hundreds of ms.

## Long-context probe: base=24000, extra=10000 (34K, max_model_len=35000, util=0.92, kv_cache_bytes=21GiB)

| | rep1 steady | rep2 steady | rep3 steady | health |
|---|---|---|---|---|
| OFF | ~14.2 | **2.4-3.5** | 6.8 | clean |
| ON | ~13.5-14.9 | **dead (OOM hang)** | — | 8-12 OOM events |

Deep-profile findings (both ON runs consistent):

1. **Fire cost scales with base, not chunk**: per-fire median 38-68 ms at
   24K base vs 1.5 ms at 8K base — the proxy scores the ENTIRE prefix
   K-cache on every (layer, chunk) fire. Total fire wall 103-124 s per run;
   correction wall 110-119 s. The `F` in the TTFT model is **F(base)**, not
   a constant. Long contexts make the tax worse, not better, until fires
   are made incremental.
2. **Overlap is dead at this length**: MoE window median 1.7 ms vs 38-68 ms
   per fire — 22-40x overflow. Fires run serially on the critical path.
3. **ON rep2 always OOMs**: proxy microbatch intermediates
   (1024x4x7168 activations per layer) plus deferred-store buffers exhaust
   the pool on the second request. An explicit 21 GiB KV budget did not
   help; the leak-like growth is in fire intermediates.
4. **Warm variance is system-wide, not CSA-only**: OFF rep2=2.4 s vs
   rep3=6.8 s with zero errors. Store wall is ~250-263 s in BOTH modes —
   deferred HCA writes during hits contend with reads. This is the prime
   suspect for all rep-to-rep warm variance and needs its own fix
   (backpressure or hit-phase deferral).

### Required fixes for long-context ON (in order)

1. **Incremental fires**: score only the new chunk's tokens (O(chunk))
   instead of the full prefix (O(base)); reuse prior chunks' predictions —
   the prefix does not change within a request.
2. **Preallocated per-layer proxy buffers**: stop allocating ~59 MB of
   intermediates per fire; reuse one buffer set per layer.
3. **Store backpressure**: bound deferred-write queue during active
   prefill; drain between requests.

## Environment invariants (violating any of these invalidates measurements)

- GPU UVM leak accumulates ~1.5 GiB per hard crash; at ~12 GiB residual,
  rep1 slows ~2x and rep2 OOM-hangs (EngineDead 300 s). Benchmark only in
  clean windows; count crashes; reboot before final numbers.
- Tutti rebind unmounts all cache drives on every container lifecycle;
  launch guard in run_container.sh enforces mount -a + 8-mount check
  (regex must match both `nvme*` and `snvme*` device names).
- `IndexerBlockStore` opens fds eagerly (fix 2026-07-04) so post-bind
  lazy opens cannot abort retrieves; seed failures degrade gracefully
  (OSError guard in gpu_connectors).
