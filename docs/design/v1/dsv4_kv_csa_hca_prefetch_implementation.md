# DSv4 KV, CSA Prefetch, And HCA Overlap Implementation

This document describes the implementation contract. Historical experiments are
kept out of this file to keep handoff context small.

> **2026-07-08 update.** The default production path is the BULK
> layer-major walker (`LMCACHE_CSA_BULK_PREDICTED=1`) + V27 resident-chunk
> relocation (`LMCACHE_CSA_WALKER_RESIDENT_SKIP=1`), not the per-layer
> HC-proxy predicted reads described in parts of this file.  In bulk mode
> `fire_predicted_reads` no-ops and `_submit_reads label=predicted` never
> appears; the walker is armed at chunk registration and started at the
> first CSA gate.  See `csa_prefetch_v24_v25_findings.md` (results),
> `csa_prefetch_gpu_centric_plan.md` (V28 plan), and the superseded notice
> in `dsv4_csa_hca_prefetch_source_of_truth.md`.  The HC-proxy contract
> below remains valid for short-context regimes where top-K has
> selectivity.

## Scope

The current work targets prefill cache-hit reuse for DSv4-Pro with CSA
attention-KV filtering:

```text
LMCACHE_DSV4_CSA_ATTENTION_KV_FILTER=1
```

Normal KV retrieve skips `csa_attention_kv`; predicted CSA attention-KV blocks
are read later through Tutti and inserted into vLLM's MLA K-cache physical rows
before the target CSA attention consumes them.

## Main Files

| File | Responsibility |
|---|---|
| `lmcache/v1/cache_engine.py` | retrieve metadata, filtered CSA chunk locations, slot mapping to physical rows |
| `lmcache/v1/csa_attention_kv_prefetch_manager.py` | predicted/miss reads, Tutti load, K-cache scatter |
| `lmcache/v1/indexer_ssd_manager.py` | HC-proxy prediction and block-id dispatch |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | DSv4 prefill hook after attention HC-post |
| `lmcache/v1/gpu_connector/tutti_direct_loader.py` | GPU-direct NVMe read execution and profiling |

## Correct Hook Point

For target CSA layer `m + 1`, fire from layer `m` after attention HC-post and
before MoE/FFN. This is the timing needed to hide target-layer Tutti reads under
the previous layer MoE/FFN window.

Expected order:

```text
layer m attention / HC-post
  -> csa_overlap_hook_fire(source=hc_post, target=m+1)
  -> proxy prediction for target layer m+1
  -> Tutti predicted read submit
layer m MoE/FFN
layer m+1 official SparseAttnIndexer true topK
  -> drain predicted reads
  -> miss correction only for uncovered true blocks
```

`DeepseekV4DecoderLayer.hc_post` in this patched path is the attention HC-post
point used by the overlap hook. It must not be treated as "after official MoE".

## Prediction Contract

For residual `R_m` from layer `m` attention HC-post:

```text
scores = target_layer_{m+1}.indexer_projection(
           target_layer_{m+1}.attn_norm(
             target_layer_{m+1}.hc_pre(R_m)))
topk = topK(scores) for every valid token row
block_ids = unique(topk // compressed_block_size)
```

Then call:

```text
CSAAttentionKVPrefetchManager.fire_predicted_reads(
    layer_id=m+1,
    block_ids=block_ids,
)
```

The proxy must be side-effect free. The current implementation does not call
the official `DeepseekV4Indexer.forward()` because that path runs the target
compressor before scoring. Instead it runs only the read-only scoring half:

```text
target.wq_b(qr)
target.fused_indexer_q_rope_quant(...)
target.SparseAttnIndexer(proxy_hidden, q_quant, k=None, weights)
```

The proxy call forces the target `SparseAttnIndexer` instance to
`skip_k_cache_insert=True` and uses a private topK buffer. It reads the target
indexer K-cache that was populated by LMCache reuse. If CSA attention-KV has
patched the op for true-topK correction, the proxy calls the saved original
forward (`_lmcache_csa_attention_kv_original_forward`) instead of the patched
wrapper. That keeps proxy scoring out of `drain_for_layer()` and
`submit_miss_reads()`.

The read-only contract is strict: if the runtime op does not expose either a
`skip_k_cache_insert` style attribute or a skip-insert kwarg, proxy scoring
logs `indexer_op_read_only_unsupported` and skips the prediction. It must never
fall back to a normal state-mutating op call. In the valid path it does not
compute `indexer_kv_score`, does not run the compressor, and does not write
compressor state or indexer K-cache rows.

## All-Token Rows

High recall requires all valid token rows in the current prefill chunk. Do not
select only the tail row.

The current GPU002 path computes the full prefill chunk in one proxy call. We
tried slicing rows, but DeepGEMM's prefill metadata is built for the current
forward chunk, so slicing made `seq_len` disagree with `cu_seq_len_k_start`.
The `microbatch_rows` field in logs is therefore historical; it must not be
interpreted as tail-only prediction.

```text
topK(row_i) for every valid row_i in the current chunk
```

If a future implementation reintroduces real microbatches to bound activation
memory, it must preserve the union of all row topK results:

```text
union_rows(topK(row_i)) for i in valid_rows
```

not:

```text
topK(last_row_only)
```

## K-Cache Placement

The predicted read payload is CSA attention-KV bytes. When it is copied into
vLLM MLA K-cache, the destination row is the global physical row allocated by
vLLM, not the request-local compressed-block index.

For current DSv4-Pro:

```text
physical_block_id = slot_mapping[chunk_start_token] // 256
```

where:

```text
256 = compress_ratio(4) * compressed_block_size(64)
```

Store `physical_block_ids` on `CSAAttentionKVChunkLoc` during retrieve and use
those ids during scatter.

## LBA Cache Requirement

`FILTER=1` means ordinary retrieve may re-register LBA extents that exclude
filtered `csa_attention_kv` byte ranges. The CSA manager must restore the full
raw-record LBA cache before each Tutti read.

If this is broken, logs contain:

```text
Tutti extents cover 0/... bytes
Tutti direct load found no readable KV extents
```

## Runtime Switches

Recommended CSA-only prefill overlap path:

```text
LMCACHE_DSV4_CSA_ATTENTION_KV_FILTER=1
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0
LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION=0
LMCACHE_INDEXER_ENABLE_PREFILL_RESIDUAL_PROXY=0
LMCACHE_CSA_ATTENTION_KV_PROXY_MICROBATCH_ROWS=64
```

`LMCACHE_INDEXER_ENABLE_PREFILL_RESIDUAL_PROXY=0` disables the old
indexer-cache prefill correction path. It must not prevent the CSA
attention-KV prefill hook from computing native target-layer proxy topK and
submitting Tutti reads.

## Log Invariants

Healthy run:

```text
csa_overlap_hook_fire source=hc_post count ~= 30 layers * TP ranks * hit requests
dispatch_csa_attention_kv_predicted > 0
_submit_reads label=predicted > 0
TUTTI_PROFILE load_total loaded=keys
correction miss_blocks close to 0
```

Unhealthy run:

```text
dispatch_csa_attention_kv_predicted = 0
_submit_reads label=miss dominates
correction total_ms is large
failed to issue miss reads
CUDA illegal memory access during proxy microbatch
```

If `lmcache_hit_tokens_tail` or `retrieve_count` looks empty, still inspect the
CSA/Tutti logs. Those summary fields do not prove CSA attention-KV prefetch did
or did not run.

## Rejected Paths

Do not reintroduce:

```text
dispatch_csa_kv_overlap_unconditional
T_next_predicted = T_prev_csa_true
LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS
tail-row-only CSA proxy
batch prefetch of all 30 CSA layers at retrieve end
per-hit drop_caches for this benchmark
```

These either produce the wrong prediction semantics or measure the wrong thing.
