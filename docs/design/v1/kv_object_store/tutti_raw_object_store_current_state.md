# KV Object Store and Tutti Raw Path Current State

This note describes the current LMCache KV object-store path as implemented
around `LocalDiskBackend`, `LMCacheEngine`, `TuttiDirectLoader`, and the DSv4
GPU connector seed hooks. It is intentionally descriptive: it records what the
code does today, where role-aware metadata exists, and where the runtime still
uses chunk-level `role="full"` objects.

## Key Files

Object-store control plane:

- `lmcache/v1/kv_object_store/object_id.py`
- `lmcache/v1/kv_object_store/record.py`
- `lmcache/v1/kv_object_store/metadata_store.py`
- `lmcache/v1/kv_object_store/pool_layout.py`
- `lmcache/v1/kv_object_store/pool_io.py`

Object-store integration and Tutti orchestration:

- `lmcache/v1/storage_backend/local_disk_backend.py`
- `lmcache/v1/cache_engine.py`
- `lmcache/v1/gpu_connector/tutti_direct_loader.py`
- `lmcache/v1/gpu_connector/gpu_connectors.py`

CSA/HCA seed and overlap managers:

- `lmcache/v1/indexer_ssd_manager.py`
- `lmcache/v1/hca_prefetch_manager.py`

Related but separate raw-block backend:

- `lmcache/v1/storage_backend/raw_block/`
- `lmcache/v1/storage_backend/plugins/rust_raw_block_backend.py`

## CPU Object Control Plane

`KVObjectId` defines the logical namespace:

```text
(model_id, parallel_config_id, rank, layer_id, role, block_id, schema_version)
```

Example for the current full-chunk path:

```text
model_id=<CacheEngineKey.model_name>
parallel_config_id=world<CacheEngineKey.world_size>
rank=<CacheEngineKey.worker_id>
layer_id=0
role=full
block_id=<CacheEngineKey.chunk_hash_hex>
```

The schema can name per-layer and per-role objects such as
`role="csa_indexer_cache"` or `role="hca_attention_kv"`, but the authoritative
write in `LocalDiskBackend._write_kv_object_store()` is still one chunk-level
full object. The default lookup in `LMCacheEngine._tutti_batched_get()` also
uses `LocalDiskBackend.get_kv_object_records(keys)` without `layer_ids` or
`roles`, so normal retrieval asks for `layer_id=0, role="full"`.

The metadata record, `KVObjectRecord`, is the CPU-side mapping layer:

```text
object_id       logical KVObjectId
pool_id         usually rank<rank>-full
offset          byte offset in pool/raw namespace
length          logical payload bytes
aligned_length  reserved/persisted bytes after alignment
shape,dtype     byte-view tensor metadata
state           allocated | ready | evicted
raw_extents     optional (file_offset, slba, n_sectors) tuples
byte_ranges     optional logical source ranges for object views
```

`KVObjectMetadataStore` is an in-memory, process-local index keyed by
`KVObjectId.to_key()`. It can dump/load JSONL, but the current
`LocalDiskBackend` object-store path is not a durable metadata database:
startup creates a fresh metadata store and, for the dense pool path, removes
the rank-local pool file if it exists.

## Pool And View Organization

When object store is enabled, `LocalDiskBackend.__init__()` creates one dense
rank-local pool:

```text
pool_id   = rank<rank>-full
pool_path = <local_disk>/_kv_object_store/<pool_id>.pool
layout    = dense append by aligned object length
```

Dense allocation is used because Tutti/FIEMAP prefers compact physical ranges:
large sparse slots would create holes and a larger extent map. If
`kv_object_store_tutti_raw_enable` is true, the pool layout is still used as a
logical byte-address allocator, but `materialize_file=False` and bytes are
written to raw LBA extents by the Tutti raw writer.

After writing the full object, `LocalDiskBackend._index_kv_object_layer_views()`
adds optional logical views when per-group metadata is available. The flow is:

```text
MemoryObj.metadata.shapes/dtypes
  -> group byte ranges in the full chunk payload
  -> KVLayerGroupsManager group metadata
  -> role labels by dtype/hidden_dim/compress_ratio
  -> per-layer KVObjectRecord views with byte_ranges
```

Current role classification is heuristic and DSv4-shaped:

```text
float32                  -> compressor_state
uint8 hidden_dim == 132  -> csa_indexer_cache
uint8 hidden_dim == 584, compress_ratio == 1
                         -> swa_cache
uint8 hidden_dim == 584, compress_ratio >= 64 or bs <= 2
                         -> hca_attention_kv
uint8 hidden_dim == 584, compress_ratio == 4 or num_layers == 30
                         -> csa_attention_kv
else                     -> kv
```

These per-layer/per-role records are metadata views over the same full chunk
payload. They do not write separate layer files or raw regions. Their purpose
is to make the object namespace capable of role-aware retrieval later, and to
let a reader reassemble a logical object from `byte_ranges`.

## Store Flow

The normal store path is still chunk oriented:

```text
LMCacheEngine.store()
  -> token_database.process_tokens()
  -> allocate MemoryObj with per-group shapes/dtypes
  -> gpu_connector.batched_from_gpu()
  -> StorageManager submit put
  -> LocalDiskBackend.async_save_bytes_to_disk()
  -> LocalDiskBackend._write_kv_object_store()
  -> LocalDiskBackend.insert_key()
```

`LMCacheEngine._dsv4_store_shapes_for_range()` may zero the non-tail SWA and
compressor-state groups when DSv4 optimized KV is enabled. That changes the
payload shape stored for the chunk, but it does not change the object id:
the write is still the default full chunk object for that cache key.

Inside `_write_kv_object_store()`:

1. Build the default `KVObjectId` with `layer_id=0, role="full"`.
2. Allocate a dense pool range or reuse the existing record.
3. If the Tutti raw writer is installed, write bytes to raw extents and attach
   `record.raw_extents`.
4. Otherwise, write bytes to the pool file through `KVObjectPoolIO`.
5. Mark the full record `READY`.
6. Add per-layer/per-role logical view records when group metadata allows it.

If the object-store write is not authoritative, the backend falls back to the
legacy `.pt` file write. The legacy `DiskCacheMetadata` entry is still inserted
because cache lookup and scheduling continue to use the existing
`LocalDiskBackend.dict` control plane.

## Retrieve Flow

The direct-load path is selected in `LMCacheEngine._process_tokens_internal()`:

```text
storage_manager.get_block_mapping()
  -> LocalDiskBackend hit
  -> _ensure_tutti_loader(keys)
  -> _tutti_batched_get(keys, shapes_per_key)
  -> gpu_connector.batched_to_gpu()
```

For DSv4 optimized KV, `shapes_per_key` mirrors the store-side shape masking so
non-tail chunks do not ask Tutti to read the groups that were not stored.

`_tutti_batched_get()` first reads legacy disk metadata from
`LocalDiskBackend.dict`. It then asks the object metadata store for the default
full object records. If every requested key has a ready object record, it
builds the Tutti read inputs:

```text
KVObjectRecord
  -> path:
       raw_extents present -> tutti://<pool_id>
       otherwise           -> dense pool file path
  -> file_offsets:
       read_record.offset
  -> read_ranges_per_key:
       read_record.read_ranges
  -> DiskCacheMetadata:
       size=read_record.length, shapes=effective shapes
```

When a shape override requests fewer bytes than `record.length`,
`LMCacheEngine._kv_object_prefix_view()` creates a byte-range prefix view. When
raw extents are present, `LocalDiskBackend.get_kv_object_raw_lba_cache()` maps
the record's logical read ranges to LBA extents and registers them under the
synthetic `tutti://<pool_id>` path.

The direct path trims the readable prefix on misses or alignment failures. If
Tutti was configured but cannot be used after snvme bind, LMCache treats those
blocks as misses instead of reading through a broken filesystem path.

## Tutti Direct Load

`LMCacheEngine._ensure_tutti_loader_locked()` prepares the loader:

```text
scan LocalDiskBackend entries and object pool paths
  -> FIEMAP file paths
  -> FIEMAP raw-region path when configured
  -> unmount local disk path for snvme bind
  -> TuttiDirectLoader.create()
  -> install LocalDiskBackend raw writer when raw object mode is enabled
```

`TuttiDirectLoader.create()` allocates an HBM staging pool with raw
`cudaMalloc`, maps it through the snvme driver, creates a user queue, and seeds
an LBA cache from the FIEMAP results. The cache key is either a normal file path
or a synthetic raw-object path such as `tutti://rank0-full`.

`TuttiDirectLoader.load_chunks_to_hbm()` lowers object reads into NVMe I/O:

```text
logical read ranges
  -> sector-aligned source byte ranges
  -> FIEMAP/LBA extent lookup
  -> one or more NVMe SGL reads per logical chunk
  -> DMA into packed HBM staging offsets
  -> clone staging slice into an owned GPU tensor
  -> TensorMemoryObj(raw_tensor.is_cuda == True)
```

The current loader supports explicit `read_ranges_per_key`, so a logical
object view can be reassembled by placing each source range at its
`target_offset`. For raw objects, `LocalDiskBackend.kv_object_record_raw_readable()`
enforces 512-byte source alignment. A non-sector-aligned final tail can be read
only when it is the logical tail range.

## Tutti Raw Cold Store

Raw cold store is installed only after the Tutti loader exists. The writer
installed in `LMCacheEngine._ensure_tutti_loader_locked()` uses either:

```text
record.offset + configured base LBA
```

or:

```text
record.offset mapped through a rank-local raw-region file's FIEMAP extents
```

The write path is:

```text
LocalDiskBackend._write_kv_object_store()
  -> raw_writer(record, buffer)
  -> TuttiDirectLoader.store_bytes_to_raw_lbas()
     or store_bytes_to_raw_extents()
  -> register LBA cache for tutti://<pool_id>
  -> record.with_raw_extents(...).mark_ready()
```

This is still a chunk-level full-object write. It copies the CPU payload bytes
into HBM staging before issuing Tutti NVMe writes, so it is not a zero-copy
GPU-source store.

## GPU Connector Direct Seed

`TuttiDirectLoader` only returns `TensorMemoryObj` instances. The role-aware
seed behavior happens later in `VLLMPagedMemGPUConnectorV3.to_gpu()`, after the
retrieved chunk has been materialized as per-group tensors.

For each LMCache group, the connector:

1. Classifies the group role with `_dsv4_group_role()`.
2. Optionally seeds an overlap manager from the group tensor.
3. Decides whether to run the normal H2D group transfer.
4. Uses the correct vLLM HMA block table for that group when transferring.

CSA indexer seed:

```text
role == csa_indexer_cache
  -> get_indexer_ssd_manager()
  -> manager.seed_range_from_lmcache_group(layer_ids, memory_tensor, start, end)
  -> continue normal group H2D transfer unless other policy skips it
```

HCA defer seed:

```text
role == hca_attention_kv
and LMCACHE_DSV4_DEFER_HCA_TO_MOE=1
  -> get_hca_prefetch_manager()
  -> manager.seed_range_from_lmcache_group(layer_ids, memory_tensor, start, end)
  -> skip normal HCA H2D transfer for this group
```

This seed path consumes the retrieved LMCache group tensor. It does not query
the per-layer/per-role `KVObjectRecord` views directly today.

## CSA/HCA Overlap Boundary

CSA and HCA overlap managers have their own runtime stores and scheduling
state. The LMCache object store is currently a source of chunk bytes for
seeding those managers, not the unified backing store for their decode-time
row/block I/O.

CSA `IndexerSSDManager`:

- Keeps one flat SSD file per CSA indexer layer.
- Keeps an HBM pool per CSA layer.
- `seed_range_from_lmcache_group()` writes rows from the LMCache
  `csa_indexer_cache` group into the manager's flat store/HBM state.
- `fire_async_for_layer()` and `prepare_pool()` implement overlap and
  correction around the true Lightning Indexer path.

HCA `HCAPrefetchManager`:

- Keeps one flat SSD file per HCA layer's compressed rows.
- `seed_range_from_lmcache_group()` writes rows from the LMCache
  `hca_attention_kv` group into that flat store.
- `fire_async_for_layer()` schedules deterministic compressed-row reads.
- `drain_for_layer()` copies ready rows into the target vLLM HCA KV cache.

The important boundary is correctness: CSA speculative prefetch may predict
which rows to read early, but true Lightning Indexer output remains the source
of attention slots. HCA is deterministic and may skip normal H2D only when the
prefetch manager is seeded and active. If a manager is missing or seed fails,
the connector keeps the normal group transfer path.

Current overlap data movement is not the same as Tutti object direct load:

```text
LMCache full/object chunk
  -> Tutti or CPU retrieve
  -> TensorMemoryObj group tensor
  -> GPU connector seed hook
  -> CSA/HCA manager flat store and HBM/runtime state
  -> later overlap fire/drain near attention
```

There is no end-to-end path today that asks the object metadata store for
`role="hca_attention_kv"` or `role="csa_indexer_cache"` records and directly
drives CSA/HCA overlap I/O from those records.

## Raw-Block Backend Is Separate

`lmcache/v1/storage_backend/raw_block/` and
`RustRawBlockBackend` are a separate raw-device storage plugin. They store
legacy cache keys or distributed `ObjectKey` values in fixed raw-device slots
with headers, an in-memory index, and metadata checkpoints. Reads populate
caller-provided `MemoryObj` buffers through the raw-block core.

That path is not the `LocalDiskBackend` KV object-store path described above:

```text
RustRawBlockBackend:
  raw device slots -> CPU/local allocator MemoryObj -> normal GPU connector

LocalDisk + KV object store + Tutti:
  dense pool/logical raw region -> Tutti NVMe DMA -> HBM TensorMemoryObj
```

The naming overlap matters because both paths use "raw", but only the
LocalDisk object-store raw mode produces `KVObjectRecord.raw_extents` and
synthetic `tutti://<pool_id>` paths for `TuttiDirectLoader`.

## Current Boundaries And Gaps

- The authoritative object write/read path remains chunk-level
  `layer_id=0, role="full"`.
- Per-layer/per-role object records exist as logical metadata views over the
  full chunk payload, but the main direct-load path does not yet request them
  by role.
- Raw object mode is authoritative only after the Tutti loader installs the raw
  writer. Before that point, writes can fall back to the legacy `.pt` path.
- `KVObjectMetadataStore` and the Tutti LBA cache are process-local runtime
  state, not a durable cross-process metadata service.
- The Tutti read path supports `byte_ranges`, but raw reads are constrained by
  512-byte alignment and staging capacity. The engine trims to the first
  unreadable chunk on failure.
- Direct load still clones the HBM staging slice into an owned GPU tensor.
- Raw cold store still copies CPU payload bytes into HBM staging before the
  Tutti write.
- CSA/HCA overlap managers use their own flat stores and HBM/runtime state;
  LMCache object-store bytes seed them through GPU connector hooks rather than
  replacing their internal stores.

## Why This Shape Exists

The current design keeps the proven LMCache chunk lookup contract intact while
adding a lower-level object metadata layer that Tutti can consume. That lets
the fast path avoid CPU disk reads for full-hit LocalDisk chunks without
requiring the whole engine to become role-aware at once.

The per-layer/per-role views are useful because they preserve a future route to
role-specific object retrieval. The implementation can index DSv4 CSA/HCA/SWA
regions by logical role today while still storing one contiguous chunk payload
that matches the existing cache key and hit-count semantics.

The CSA/HCA direct seed hooks are deliberately placed in the GPU connector
because only that layer has both pieces of information: the LMCache group
tensor returned by retrieval and the vLLM HMA/group mapping needed to seed or
transfer the right runtime state.
