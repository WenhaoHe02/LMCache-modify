# DSv4 exact-plan cache and adaptive static I/O grouping

This directory is an experimental successor to
`suffix_generation_fix_20260729_0016`. The protected 1.457558 s baseline is not
modified.

## What changed

### Exact streaming-plan cache

`patches/v1/cache_engine.py` caches the fully compiled Indexer/CSA/HCA source
plan and its Tutti LBA table. Reuse requires all of the following to match:

1. the terminal generation, layout version, and covered token count;
2. a process-local publication revision that changes after every re-admission;
3. every composed cache key and logical token range;
4. an exact `torch.equal` comparison of the prefix portion of vLLM's current
   physical `slot_mapping`.

A different GPU page allocation, eviction/re-admission, suffix composition, or
raw-pool reset produces a cache miss and rebuild. The cache never uses a hash as
a correctness test.

`patches/v1/storage_backend/local_disk_backend.py` exposes the public
`get_csa_streaming_plan_token()` API and maintains the publication revision.

### Adaptive cross-layer submit

`patches/v1/csa_attention_kv_prefetch_manager.py` and
`patches/v1/indexer_ssd_manager.py` combine adjacent deterministic Indexer-K or
HCA full-layer reads into one Tutti call. Group size is computed from actual
bytes per layer:

```text
group_size = clamp(target_bytes // layer_bytes, 1, max_group)
```

Defaults are a 32 MiB group target and at most eight layers. Therefore short
contexts amortize prepare/submit/poll overhead, while large contexts retain the
existing fine-grained pipeline automatically.

Predicted CSA reads, true-topK correction reads, and their per-layer timing are
unchanged. Early dense CSA is also unchanged because its SSD phase is followed
by an ordered model-thread NCCL gather; combining it safely requires a separate
batched-collective design.

## Environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `LMCACHE_DSV4_STREAMING_PLAN_CACHE_CAPACITY` | `8` | Exact bound-plan LRU entries per rank; use `0` to disable. |
| `LMCACHE_DSV4_STATIC_IO_GROUP_MAX` | `8` | Maximum adjacent deterministic Indexer/HCA layers per Tutti call; use `1` to disable grouping. |
| `LMCACHE_DSV4_STATIC_IO_GROUP_TARGET_MIB` | `32` | Target total payload for one grouped call. |

`launch_production.sh` exports these defaults while allowing caller overrides.

## Run

The copied scripts use the same protected image, CP8 configuration, SNvme
devices, model path, and patch overlay as the baseline:

```bash
cd /path/to/grouped_plan_cache_20260729_1
bash launch_production.sh on
```

Disable one optimization at a time for an A/B:

```bash
LMCACHE_DSV4_STREAMING_PLAN_CACHE_CAPACITY=0 \
LMCACHE_DSV4_STATIC_IO_GROUP_MAX=1 \
bash launch_production.sh on
```

## Required GPU acceptance order

This candidate was intentionally not deployed to GPU002. Before using its TTFT
numbers:

1. Run a short logical-output probe and compare with the protected baseline.
2. Run a same-prompt repeated hit and confirm `plan_cache_hit=True` after the
   first compiled binding.
3. Run a different allocation/prefix and confirm the first request reports
   `plan_cache_hit=False`.
4. With grouping enabled, confirm `TUTTI_PROFILE static_layer_group` appears
   only for Indexer/HCA and never for predicted or miss-correction CSA I/O.
5. Compare 32K, 100K, 128K, and 480K TTFT with grouping `1` versus `8`.
6. Only after correctness passes, tune `TARGET_MIB` among `16`, `32`, and `64`.

## Local verification completed

- `py_compile`: passed for all four modified Python files.
- `ruff check`: passed.
- `ruff format --check`: passed.
- Existing Indexer/Tutti tests: 27 passed.
- CSA streaming layout regressions: 6 passed.
- Exact binding checks covered same mapping reuse, prefix mapping rejection,
  suffix-only mapping tolerance, and publication revision invalidation.
- Mock scheduling checks verified one shared Future is registered at every
  layer gate in a grouped Indexer/HCA wave.
