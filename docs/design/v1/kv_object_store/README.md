# KV Object Store MVP

This module defines the lower storage namespace for KV objects. The current
implementation has a real framework integration in `LocalDiskBackend`: when
`LMCACHE_KV_OBJECT_STORE_ENABLE=1` or `extra_config.kv_object_store_enable` is
set, disk store writes both the legacy `.pt` file and a chunk-level KV object
pool entry; disk retrieve tries the object pool first and falls back to the
legacy file when the object is absent or incompatible.

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
  the MVP uses fixed-size aligned slots in a sparse file.
- Batch lookup:
  callers can ask for many objects and get request-ordered records or misses.
- Optional LocalDiskBackend integration:
  chunk-level `role=full` object writes and reads, with
  `KV_OBJECT_STORE_PROFILE` logs proving whether the framework path used the
  object store.

The object store does not own:

- CSA/HCA prediction.
- Prefill/forward overlap windows.
- HBM residency policy.
- Tutti vs normal LMCache read selection.

## Current Framework Integration

```text
LocalDiskBackend.async_save_bytes_to_disk()
  -> write legacy <CacheEngineKey>.pt
  -> if object store enabled:
       allocate or reuse KVObjectRecord(role="full", block_id=chunk_hash_hex)
       pwritev() the chunk bytes into the fixed-slot pool
       mark record READY

LocalDiskBackend.load_bytes_from_disk()
  -> allocate the normal MemoryObj
  -> if object store has a READY record with matching byte length:
       preadv() directly into MemoryObj.byte_array
       log KV_OBJECT_STORE_PROFILE op=read status=hit
  -> else:
       fall back to read_file() from legacy .pt
```

This is intentionally chunk-level first. It proves that the vLLM/LMCache
framework path can really store and retrieve through the object namespace. The
next step is to split the object identity from `role=full` chunks into native
`layer_id x role x block_id` objects for CSA/HCA/SWA.

## MVP Flow

```text
DSv4 overlap manager selects layer/block objects
  -> build KVObjectId values
  -> KVObjectMetadataStore.get_many(..., ready_only=True)
  -> storage layer groups records by pool_id and offset
  -> read engine submits contiguous batches
  -> connector copies or aliases results into vLLM KV slots
```

## Why Fixed Slots First

Fixed slots make the first integration deterministic: object offsets do not move,
pool size is known up front, and FIEMAP/Tutti batching can be reasoned about
without a free-list allocator. A future allocator can replace the layout while
keeping the `KVObjectId` and `KVObjectRecord` contract stable.
