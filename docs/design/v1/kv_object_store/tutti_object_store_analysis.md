# LMCache KV Object Store / TUTTI Direct Loader 梳理

## 1. 代码入口

- `lmcache/v1/kv_object_store/object_id.py`
  - `KVObjectId`
- `lmcache/v1/kv_object_store/record.py`
  - `KVObjectRecord`
  - `KVObjectState`
- `lmcache/v1/kv_object_store/pool_layout.py`
  - `KVObjectPoolLayout`
- `lmcache/v1/kv_object_store/pool_io.py`
  - `KVObjectPoolIO`
  - `KVObjectReadBatch`
- `lmcache/v1/kv_object_store/metadata_store.py`
  - `KVObjectMetadataStore`
- `lmcache/v1/storage_backend/local_disk_backend.py`
  - `_write_kv_object_store`
  - `get_kv_object_records`
  - `kv_object_data_path`
  - `get_kv_object_raw_lba_cache`
- `lmcache/v1/cache_engine.py`
  - `_maybe_init_tutti_loader`
  - `_ensure_tutti_loader`
  - `_tutti_batched_get`
- `lmcache/v1/gpu_connector/tutti_direct_loader.py`
  - `TuttiDirectLoader.create`
  - `TuttiDirectLoader.load_chunks_to_hbm`
  - `TuttiDirectLoader._load_batch`
  - `store_bytes_to_raw_lbas`
  - `store_bytes_to_raw_extents`
  - `FiemapHelper.query_extents`
  - `FiemapHelper.scan_paths`
- `lmcache/integration/vllm/vllm_v1_adapter.py`
  - `wait_for_save`
  - `save_kv_layer`
  - `wait_for_layer_load`

## 2. object store 的 key / chunk / layer / tensor 组织

### 逻辑对象 ID

`KVObjectId` 结构：

```text
(model_id, parallel_config_id, rank, layer_id, role, block_id, schema_version)
```

当前 `LocalDiskBackend` 的映射是 chunk 级别的 full object：

```python
KVObjectId(
    model_id=key.model_name,
    parallel_config_id=f"world{key.world_size}",
    rank=key.worker_id,
    layer_id=0,
    role="full",
    block_id=key.chunk_hash_hex,
)
```

### 记录内容

`KVObjectRecord` 记录：

```text
object_id
pool_id
offset
length
aligned_length
shape
dtype
state
raw_extents
```

解释：

- `offset`：pool 文件里的逻辑字节偏移
- `length`：真实 payload 长度
- `aligned_length`：按 alignment 向上对齐后的长度
- `shape` / `dtype`：当前 object store 里通常是 byte payload 视角，常见是 `(len(buffer),)` + `torch.uint8`
- `raw_extents`：TUTTI raw 模式下的 `(file_offset, slba, n_sectors)` 列表

### pool layout

`KVObjectPoolLayout` 当前在 LMCache 集成里用 `dense=True`：

```python
pool_id = f"rank{rank}-full"
pool_path = Path(self.path) / "_kv_object_store" / f"{pool_id}.pool"
```

这意味着：

- object 按 `aligned_length` 紧密追加
- 目的是让 FIEMAP / raw LBA 读取更直接
- 不是“每个对象一个大 sparse slot”的旧式布局

### 当前数据格式

这里需要区分“object id 的表达能力”和“当前 LocalDiskBackend 已接入的实际格式”：

- `KVObjectId` 的 schema 有 `layer_id` 和 `role`，因此可以表达 per-layer / per-role object。
- 但当前 `LocalDiskBackend._key_to_object_id()` 实际固定为 `layer_id=0`、`role="full"`，所以现有 framework fast path 存的是 LMCache chunk 级 full object。
- full object 的 payload 是 `MemoryObj.byte_array` 对应的原始 bytes，不再拆成单独的 layer tensor object。
- tensor 的 layout、shape、dtype、multi-group 信息仍来自原始 `DiskCacheMetadata`，retrieve 时再用这些 metadata 把 bytes 包装回 `MemoryObj` / `TensorMemoryObj`。

因此，当前代码状态下 object store 是“支持 layer/role 命名空间的 object-store 框架”，但 `LocalDiskBackend` 集成只使用了 chunk-level full-object 格式。未来如果 CSA/HCA 路径直接写 per-layer object，可以复用同一套 `KVObjectId` / `KVObjectRecord` 结构，但那不是本文描述的当前主路径。

## 3. 写入路径：prefill / store 阶段如何落到 NVMe / object store

### vLLM -> LMCache

入口在 `lmcache/integration/vllm/vllm_v1_adapter.py`：

- `wait_for_save()`
- `save_kv_layer()`

这些方法会把 `token_ids`、`slot_mapping`、`kvcaches` 交给 `LMCacheEngine.store()` 或 `store_layer()`。

### LMCacheEngine -> LocalDiskBackend

主链路：

```text
vLLM connector
  -> LMCacheEngine.store / store_layer
  -> storage_manager.submit_put_task / batched_submit_put_task
  -> LocalDiskBackend.async_save_bytes_to_disk
  -> _write_kv_object_store
```

### object store 写法

`LocalDiskBackend._write_kv_object_store()`：

1. 通过 `_key_to_object_id()` 生成 object id
2. `KVObjectMetadataStore.get(object_id)` 查是否已有 record
3. 没有则 `KVObjectPoolLayout.allocate(...)`
4. 有 raw writer 时，走 TUTTI raw store
5. 否则走 `KVObjectPoolIO.write_object(...)`
6. 把 record 标成 `READY`

pool_file 模式是：

```text
pwritev(payload + padding) -> pool file
```

日志：

```text
KV_OBJECT_STORE_PROFILE op=write ... mode=pool_file
```

### raw 模式

当启用：

- `LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE=1`

并且 `CacheEngine._ensure_tutti_loader()` 成功安装 raw writer 后：

```text
raw_writer(record, buffer)
  -> TuttiDirectLoader.store_bytes_to_raw_lbas()
     或 store_bytes_to_raw_extents()
  -> 返回 raw_extents
  -> record.with_raw_extents(...).mark_ready()
```

这时 object 的物理持久位置不再依赖普通文件系统路径，而是依赖 raw LBA 区间。

## 4. 读取路径：命中后如何 batched_get / direct load 到 HBM

### retrieve 入口

`LMCacheEngine` 在 LocalDiskBackend 命中时，会优先走 TUTTI fast path：

```text
_ensure_tutti_loader(keys)
  -> _tutti_batched_get(keys, shapes_per_key)
```

如果 TUTTI 不可用，则 fallback 到普通 `storage_manager.batched_get(...)`。

### object store 命中

`_tutti_batched_get()` 会先从 `LocalDiskBackend` 取 `KVObjectRecord`：

```python
kv_object_records = disk_backend.get_kv_object_records(keys)
```

如果全部是 READY record：

- `kv_object_data_path(record)` 选择 pool file path 或 `tutti://pool_id`
- `file_offsets` 使用 `record.offset`
- 如果有 `raw_extents`，则注册到 `loader.register_lba_cache(...)`

然后调用：

```python
TuttiDirectLoader.load_chunks_to_hbm(
    keys,
    disk_metas,
    shapes_per_key=shapes_per_key,
    file_offsets=tutti_file_offsets,
)
```

### direct load 内部

`load_chunks_to_hbm()` 会：

- 按 queue depth / staging capacity 分 batch
- 每个 chunk 通过 `_load_batch()` 读取
- `_load_batch()`：
  - `_get_extents(meta.path)` 获取 FIEMAP 结果或缓存 extents
  - 计算每个 chunk 的 LBA / byte lens
  - `tutti_submit_batch_sgl_read(...)`
  - `tutti_poll_batch(...)`
  - `cuda synchronize`
  - 把 staging slice clone 成独立 GPU tensor
  - 包成 `TensorMemoryObj`

所以它不是 CPU read + H2D，而是：

```text
NVMe DMA -> HBM staging -> GPU tensor -> vLLM KV slots
```

### TUTTI_PROFILE 字段怎么理解

常见字段：

- `TUTTI_PROFILE ensure_loader`
  - `recover_ms`：恢复已存在 disk metadata
  - `fiemap_ms`：FIEMAP 扫 path
  - `unmount_ms`：umount 本地 cache fs
  - `create_ms`：TuttiDirectLoader 初始化耗时
- `TUTTI_PROFILE batched_get`
  - `metadata_ms`：查 LocalDiskBackend metadata
  - `load_hbm_ms`：`load_chunks_to_hbm()` 总耗时
- `TUTTI_PROFILE load_batch`
  - 一次 sub-batch 的 packing + load 统计
- `TUTTI_PROFILE batch_detail`
  - `extents_ms`：取 extents 耗时
  - `submit_launch_ms`：kernel submit 开销
  - `poll_sync_ms`：poll + sync 的主要 I/O 等待
  - `persist_ms`：staging slice clone 成独立 GPU tensor 的开销
- `TUTTI_PROFILE store_raw`
  - raw 写路径的 HBM copy、submit、poll、status 分段统计

## 5. 与 vLLM connector / LMCache cache_engine 的交互

### vLLM connector

`LMCacheConnectorV1Impl` 负责把 vLLM 的请求状态翻译成 LMCache 输入：

- `token_ids`
- `slot_mapping`
- `kvcaches`
- `skip_leading_tokens`
- `is_last_prefill`

store 侧：

```text
wait_for_save() / save_kv_layer()
  -> lmcache_engine.store / store_layer
```

retrieve 侧：

```text
wait_for_layer_load()
  -> 等待每层 load 完成
```

### cache_engine

`LMCacheEngine` 是真正的编排层：

- 决定是否启用 TUTTI
- 决定 LocalDiskBackend 命中时走 direct load 还是 CPU fallback
- 在 raw 模式下安装 TUTTI raw writer
- 负责把 load 出来的 `TensorMemoryObj` 交回 GPU connector

也就是说：

```text
vLLM connector 负责请求语义
LMCacheEngine 负责缓存编排
LocalDiskBackend / TuttiDirectLoader 负责落盘和直读
```

## 6. 现有限制与性能瓶颈

1. 当前 object store 仍主要是 chunk-level full object，不是 per-layer object store。
2. `KVObjectMetadataStore` 主要是进程内索引，虽然支持 JSONL dump/load，但 LMCache 当前流程没有把它当成完整持久恢复链路。
3. `LocalDiskBackend.__init__` 会重建 / 清空当前 pool path 相关状态，适合进程生命周期内 fast path，不像长期数据库。
4. 首次 TUTTI 命中有明显初始化开销：
   - 盘路径扫描
   - FIEMAP
   - umount
   - snvme bind
   - staging / queue 初始化
5. batch 受 queue depth、staging capacity、`max_data_size` 限制，碎片化 extent 会拉高 sub-batch 数。
6. `_load_batch()` 里会把 staging slice `clone()` 成独立 GPU tensor，有额外 HBM copy。
7. raw store 仍有 CPU payload -> HBM staging 的 copy，不是纯零拷贝写入。
8. filesystem 和 snvme bind 互斥，bind 后不能继续指望 ext4 路径正常读写。

## 参考测试

- `tests/v1/test_kv_object_store.py`
- `tests/v1/test_tutti_direct_loader.py`
- `tests/v1/test_tutti_e2e.py`

## 备注

这份草稿按当前代码状态整理，重点偏向：

- object store 的“字节容器 + 元数据索引”定位
- TUTTI direct load 的实际 HBM 读路径
- vLLM / LMCache / LocalDiskBackend 的交接面

如果后续要补成正式设计文档，可以再加一张时序图，把：

`vLLM connector -> LMCacheEngine -> LocalDiskBackend -> TuttiDirectLoader -> GPUConnector`

串成一条完整链路。
