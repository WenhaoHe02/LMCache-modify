# KV Object Store MVP

This module defines the lower storage namespace for KV objects. The current
implementation has a real framework integration in `LocalDiskBackend`: when
`LMCACHE_KV_OBJECT_STORE_ENABLE=1` or `extra_config.kv_object_store_enable` is
set, disk store writes both the legacy `.pt` file and a chunk-level KV object
pool entry. The intended fast retrieve path is Tutti: `CacheEngine` selects
READY object records and passes the pool file plus per-object byte offset to
`TuttiDirectLoader`, which issues NVMe reads directly into HBM staging.

CSA/HCA overlap managers still own fire/drain timing. The object store is the
lower read/write substrate that those managers can target.

## Boundary

The object store owns:

- Stable object identity:
  `(model_id, parallel_config_id, rank, layer_id, role, block_id)`.
- Metadata:
  `pool_id`, byte `offset`, `length`, tensor `shape`, `dtype`, and lifecycle
  state.
- Pool file layout:
  the current Tutti path uses dense aligned objects in a pool file so FIEMAP
  can cover the written byte ranges without sparse holes.
- Batch lookup:
  callers can ask for many objects and get request-ordered records or misses.
- Optional LocalDiskBackend integration:
  chunk-level `role=full` object writes, plus READY record lookup for Tutti.

The object store does not own:

- CSA/HCA prediction.
- Prefill/forward overlap windows.
- HBM residency policy.
- CSA/HCA/Tutti read scheduling.

## Current Framework Integration

```text
LocalDiskBackend.async_save_bytes_to_disk()
  -> write legacy <CacheEngineKey>.pt
  -> if object store enabled:
       allocate or reuse KVObjectRecord(role="full", block_id=chunk_hash_hex)
       pwritev() the chunk bytes into the dense aligned object pool
       mark record READY

CacheEngine._tutti_batched_get()
  -> lookup READY KVObjectRecord values for requested keys
  -> build DiskCacheMetadata(path=<pool file>) and file_offsets=<record.offset>
  -> TuttiDirectLoader.load_chunks_to_hbm(..., file_offsets=...)
  -> snvme reads pool extents directly into HBM staging
  -> GPU-resident TensorMemoryObj goes to the existing vLLM connector
```

There is no framework CPU read path for the object pool. If Tutti is not
available, `LocalDiskBackend` falls back to the legacy `.pt` files rather than
reading object-pool bytes through CPU staging.

## MVP Flow

```text
DSv4 overlap manager selects layer/block objects
  -> build KVObjectId values
  -> KVObjectMetadataStore.get_many(..., ready_only=True)
  -> storage layer groups records by pool_id and offset
  -> read engine submits contiguous batches
  -> connector copies or aliases results into vLLM KV slots
```

## Why Dense Pool First

Dense aligned objects keep the pool compact and make FIEMAP useful for Tutti:
each stored chunk has real extents around `record.offset` instead of a 128 MiB
sparse hole per object. Fixed slots are still available as a layout option for
tests and future allocator experiments, but the framework fast path uses dense
pool files.

## Validation

gpu002, 2026-06-10, DSv4 128K, TP=8, `--no-enable-prefix-caching`,
`LMCACHE_KV_OBJECT_STORE_ENABLE=1`, Tutti enabled:

- Prompt: 121,800 tokens, `max_tokens=1`.
- Cold/store: 17.091 s.
- First full-hit after cold: 30.711 s. This includes first-time Tutti
  bind/session setup; rank 0 spent 29.47 s in `session_bind_map_ms`.
- Second full-hit: 1.020 s end-to-end.
- LMCache logs reported full hit on all ranks:
  `Retrieved 121600 out of 121600 required tokens`.
- Tutti object-store path was used on all ranks:
  `TUTTI_OBJECT_STORE_PROFILE op=select keys=475 pools=1`.
- Steady Tutti direct read per rank loaded 475 chunks, about 753.8 MiB/rank,
  with `TUTTI_PROFILE batched_get` around 79-84 ms.
- Framework CPU object-pool reads were absent:
  `KV_OBJECT_STORE_PROFILE op=read` count was 0.
