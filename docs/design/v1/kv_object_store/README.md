# KV Object Store MVP

This module defines the lower storage namespace for layer/block-addressed KV
objects. It does not decide when a CSA/HCA/SWA object should be read; the
existing DSv4 overlap managers still own fire/drain timing.

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

The object store does not own:

- CSA/HCA prediction.
- Prefill/forward overlap windows.
- HBM residency policy.
- Tutti vs normal LMCache read selection.

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
