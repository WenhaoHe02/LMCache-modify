# KV Object Store Design Note

The KV object store is the metadata + pool-file layer behind
`LocalDiskBackend`. It gives LMCache a stable object namespace, a pool layout,
and a read-range abstraction that Tutti can lower into NVMe DMA.

## Current Shape

```text
LocalDiskBackend.async_save_bytes_to_disk()
  -> try object-store write first
  -> on success, skip legacy .pt write
  -> on miss/failure, fall back to legacy .pt file

CacheEngine._tutti_batched_get()
  -> get READY KVObjectRecord values
  -> build DiskCacheMetadata(path=<pool file>)
  -> pass file_offsets=[record.offset] to TuttiDirectLoader
  -> Tutti reads NVMe -> HBM staging -> GPU tensor
```

## Object Identity

`KVObjectId` is the stable object key:

`(model_id, parallel_config_id, rank, layer_id, role, block_id, schema_version)`

Current `LocalDiskBackend` writes chunk-level objects as
`layer_id=0`, `role="full"`, `block_id=chunk_hash_hex`. The schema already
allows per-layer / per-role logical objects, and the backend can index those
views when KV layer-group metadata is present.

## Metadata

`KVObjectRecord` stores:

- `object_id`
- `pool_id`
- `offset`
- `length`
- `aligned_length`
- `shape`
- `dtype`
- `state` (`ALLOCATED`, `READY`, `EVICTED`)
- `raw_extents` as `(file_offset, slba, n_sectors)`
- `byte_ranges` as logical read ranges inside the object payload

`read_ranges` is the public read view. If `byte_ranges` is empty, the record
behaves like one contiguous range at `offset`. If `byte_ranges` is present, the
record is a logical view that can be reassembled by copying each range to its
`target_offset`.

`KVObjectMetadataStore` is an in-memory, thread-safe index keyed by
`KVObjectId.to_key()`. `get_many(..., ready_only=True)` returns request-ordered
READY records or `None` for misses.

## Pool Layout

`KVObjectPoolLayout` currently uses a dense Tutti-friendly pool file layout:

- `pool_id = f"rank{rank}-full"`
- `offset` grows by `aligned_length`
- `aligned_length = ceil(length / alignment) * alignment`
- `materialize_file=False` in raw Tutti mode

Dense layout avoids sparse holes, keeps FIEMAP output compact, and makes a
shared pool file usable as the physical backing for many objects. The older
fixed-slot mode still exists, but it is not the Tutti fast path.

`KVObjectPoolIO` is the blocking POSIX fallback. It writes and reads by record
offset, and it already honors `read_ranges` when a record is a logical view.

## Tutti Alignment

Tutti's SPI uses two parallel batches:

- `IORequestBatch`
- `BufferDescriptorBatch`

The contract is simple: same count, same order, and
`IORequest[i].descriptor` must point at `BufferDescriptor[i]`. Both batches
carry a `MemoryRegion*` that names where the arrays live, so the backend can
distinguish host, pinned, managed, or device residency.

For LMCache object-store reads, the intended lowering is:

`KVObjectRecord.read_ranges[] -> IORequest[] + BufferDescriptor[]`

with one request/descriptor pair per logical read range. For contiguous
records, the current `file_offsets=[record.offset]` path is enough. For
split-range records, the current Tutti path does not yet emit a full
request/descriptor batch, so the logical view is not carried end-to-end.

## Current Gaps

- No framework CPU read path for object-pool bytes; fallback is still legacy
  `.pt` files when Tutti is unavailable.
- Tutti retrieval still assumes one contiguous logical object per record and
  does not yet consume `byte_ranges` directly.
- `KVObjectMetadataStore` is process-local only; there is no first-class durable
  metadata replay path wired into engine startup.
- Raw mode still depends on successful Tutti warmup/bind before it becomes the
  authoritative write path.
