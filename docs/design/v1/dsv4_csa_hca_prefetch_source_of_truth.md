# DSv4 CSA/HCA Prefetch Source Of Truth

This document is the compact handoff for the DSv4-Pro CSA attention-KV prefetch
work. It intentionally removes old experiment logs and rejected prototypes so a
future reader does not reload stale context.

> **2026-07-08 SUPERSEDED NOTICE (read first).** The per-layer HC-proxy
> predicted-read pipeline described below is NO LONGER the default path.
> Measurement at 384K+ contexts showed the prefill top-K union saturates
> the whole prefix (predicted == everything), so per-layer predicted fires
> degenerate into ~21 full-prefix reads (~11x NVMe amplification).  The
> production path since V24/V27 is:
>
> 1. **Bulk layer-major walker** (`LMCACHE_CSA_BULK_PREDICTED=1`, default
>    on): armed at chunk registration, started at the first CSA gate
>    (compute phase, NVMe idle), reads each layer's whole slab once.
>    `fire_predicted_reads` intentionally no-ops in this mode.
> 2. **V27 resident-chunk relocation** (`LMCACHE_CSA_WALKER_RESIDENT_SKIP=1`):
>    repeat prefixes skip the NVMe read entirely; rows are relocated
>    GPU-internally at registration time.
>
> Healthy logs in bulk mode show `bulk read-ahead req=... chunks=... total_ms=...`
> and `resident-chunk skip ... matched=...`, NOT `_submit_reads label=predicted`
> (that label only appears when bulk mode is explicitly disabled).
> Current results and the V28 plan (pipelining the main `mla_latent_kv`
> group): see `csa_prefetch_v24_v25_findings.md` and
> `csa_prefetch_gpu_centric_plan.md`.  The HC-proxy math below is retained
> for the record and for short-context regimes where prediction has
> selectivity.

## Goal

For prefill cache-hit reuse with:

```text
LMCACHE_DSV4_CSA_ATTENTION_KV_FILTER=1
```

LMCache should not synchronously scatter all `csa_attention_kv` during
`retrieve()`. Instead, the previous layer's compute window should trigger a
target-layer HC-proxy prediction, submit Tutti GPU-direct NVMe reads for the
predicted CSA attention-KV blocks, and hide the I/O under the previous layer's
MoE/FFN work.

The target result is:

```text
FILTER=1 hit-N <= FILTER=0 hit-N
```

not the earlier `+160 ms` regression.

## Correct Math

For a target CSA layer `m + 1`, the predictor must run after layer `m` attention
HC-post has produced the residual state and before layer `m` MoE/FFN starts.
This gives the layer `m` MoE window to overlap the Tutti reads for layer
`m + 1`.

Formula-level contract:

```text
R_m = residual after layer m attention HC_post
P_{m+1} = IndexerProjection_{m+1}(
            AttnNorm_{m+1}(
              HCPre_{m+1}(R_m)))
T_{m+1,pred} = topK(P_{m+1}) over every valid token row
B_{m+1,pred} = unique floor(T_{m+1,pred} / compressed_block_size)
```

Use `B_{m+1,pred}` to call
`CSAAttentionKVPrefetchManager.fire_predicted_reads(layer=m+1, block_ids=...)`.

Do not use:

```text
T_{m+1,pred} = T_{m,true}
```

That was the rejected `dispatch_csa_kv_overlap_unconditional` shortcut. It is
not the high-recall method and should not be revived.

Do not use tail-row-only prediction. The high-recall path uses all valid token
rows in the current prefill chunk. Microbatching is only a memory-control
mechanism; it must not change the selected token set.

## Rows And Blocks

There are two different "row" concepts:

| Term | Meaning |
|---|---|
| proxy row | one token row in the current prefill chunk, shape-compatible with DSv4 HC input |
| K-cache physical row | one allocated row in vLLM's global paged MLA K-cache pool |

CSA proxy prediction iterates token rows. K-cache writes must use physical rows
computed from vLLM `slot_mapping`, not request-local block ids.

For the fixed DSv4-Pro path used here:

```text
compressed_block_size = 64
compress_ratio = 4
physical_block_id = slot_mapping[chunk_start_token] /
                    (compress_ratio * compressed_block_size)
                  = slot_mapping[chunk_start_token] / 256
```

The `/256` denominator is fixed by this MLA compressed-block layout. Re-check it
only if the model's compression ratio or block size changes.

## Correctness Fixes Already Needed

These are required for correct bytes:

1. `DiskCacheMetadata` synthesized for CSA attention-KV must include placeholder
   byte shape/dtype, for example `shape=Size((aligned_length,))` and
   `dtype=uint8`.
2. Before each CSA attention-KV Tutti read, restore the full-record LBA cache.
   The normal retrieve path may register extents that exclude filtered
   `csa_attention_kv` bytes.
3. CSA attention-KV writes must target K-cache physical rows derived from
   `slot_mapping`, not request-local `compressed_block_id`.

## Runtime Path

Expected prefill cache-hit sequence:

```text
LMCache retrieve:
  loads all non-filtered KV
  registers CSA attention-KV chunk locations and full raw LBA records

DeepSeekV4 layer m:
  attention HC_post returns R_m
  hook fires target layer m+1 CSA proxy
  target layer m+1 HC_pre + attn_norm + indexer projection runs over all rows
  predicted CSA block ids are submitted to Tutti
  layer m MoE/FFN runs while Tutti reads are in flight

DeepSeekV4 layer m+1:
  official SparseAttnIndexer produces true topK
  CSA manager drains predicted reads
  CSA manager submits only missing true blocks as correction
  attention consumes resident K-cache physical rows
```

`prefill_proxy_enabled()` does not mean "fall to decode". If the prefill hook
does not fire, the request remains a chunked prefill request and CSA reads happen
later as synchronous miss correction inside attention.

## Side Effects To Avoid

Do not compute proxy topK by calling the official `DeepseekV4Indexer.forward()`.
That path runs the compressor first and can allocate compressor/indexer
workspace or mutate compressor state/indexer K-cache rows.

The current implementation computes proxy topK through the read-only part of
the target indexer path:

```text
target.wq_b(qr)
target.fused_indexer_q_rope_quant(...)
target.SparseAttnIndexer(proxy_hidden, q_quant, k=None, weights)
```

The target `SparseAttnIndexer` instance is forced to
`skip_k_cache_insert=True` for the proxy call and uses a private topK buffer.
When CSA attention-KV has patched `indexer_op.forward` for true-topK
correction, proxy scoring calls the saved original forward
(`_lmcache_csa_attention_kv_original_forward`) directly so it cannot recurse
into `drain_for_layer()` or `submit_miss_reads()`. If the runtime op does not
expose a skip-insert attribute or kwarg, proxy scoring must fail closed and log
`indexer_op_read_only_unsupported`; it must not fall back to a normal
state-mutating op call.

The proxy call therefore gathers/scans the already-loaded target indexer
K-cache, but does not compute `indexer_kv_score`, does not run the compressor,
does not write request KV state, and does not trigger CSA attention-KV
correction side effects.

This matters because prediction runs before the target layer attention. Any
state mutation there can corrupt the official true-topK/correction path.

## Required Logs

LMCache summary fields such as:

```text
lmcache_hit_tokens_tail=[0,0,0,0]
retrieve_count=0
```

are not enough to judge CSA attention-KV prefetch. They only describe the
ordinary LMCache retrieve profile. Always inspect CSA/Tutti logs too.

Healthy evidence:

```text
LMCACHE_TTFT_STAGE event=csa_overlap_hook_fire source=hc_post
IndexerSSDTiming: event=dispatch_csa_attention_kv_predicted
CSAAttentionKVPrefetchManager: _submit_reads label=predicted
TUTTI_PROFILE load_total keys=... loaded=...
CSAAttentionKVPrefetchManager: correction ... miss_blocks=0
```

Bad evidence:

```text
_submit_reads label=predicted count = 0
_submit_reads label=miss is high
correction ... miss_blocks>0 with large total_ms
failed to issue miss reads
Tutti extents cover 0/... bytes
Tutti direct load found no readable KV extents
CUDA illegal memory access during proxy microbatch
```

The 2026-06-30 layer-24 failure showed Tutti reads themselves can be fast
(`load_total keys=31 loaded=31 total_ms~=1.3 ms`) while the CSA path is still
wrong because miss correction remains and some ranks hit illegal memory access.

## Current Status

Known good baseline correctness:

```text
FILTER=0 hit-3 ~= 0.408 s, output " Modern"
FILTER=1 hit-3 ~= 0.566 s, output " Modern"
```

Known bad performance symptom:

```text
predicted reads = 0 or too late
miss correction reads = high
hit-N slower by about 160 ms
```

The fix direction is the HC-proxy prefill hook described above, not a
cross-layer true-topK shortcut and not a 30-layer batch prefetch.

## Deprecated Material

Older sections about tail-row prefill proxy, `PREFETCH_PREFILL_ROWS`,
`dispatch_csa_kv_overlap_unconditional`, profile-only candidate reads, and
per-hit Linux page-cache dropping were removed from this source-of-truth. Use
git history only if an old experiment needs forensic review.
