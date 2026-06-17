# KV Object Store — Tutti 路径实现详解

> 文档对应代码分支：`codex/tutti-lazy-multiextent`  
> 最后更新：2026-06-17

---

## 1. 背景与目标

传统 LMCache 磁盘路径：

```
NVMe → (内核/文件系统) → CPU 内存 → H2D scatter → vLLM KV cache
```

代价：约 0.38 s SSD→CPU + 约 0.95 s H2D PCIe 传输，合计 ~1.33 s。

**KV Object Store + Tutti GPU-direct 路径**：

```
NVMe → (snvme DMA, 绕过 CPU) → HBM staging → G2G scatter → vLLM KV cache
```

代价：~0.38 s SSD→HBM + ~0.02 s G2G scatter，合计 ~0.40 s。

Object Store 的另一个目标是把 chunk-level 的 KV 数据从"一个 key 一个文件"
转移到**合并的 pool 文件**里，降低文件系统 metadata 开销，并为 Tutti raw LBA
访问（绕过文件系统）提供地址基础。

---

## 2. 模块结构

```
lmcache/v1/
├── kv_object_store/
│   ├── object_id.py          # KVObjectId — 稳定的 layer/block 标识符
│   ├── record.py             # KVObjectRecord — 一个对象的完整位置+状态
│   ├── metadata_store.py     # KVObjectMetadataStore — 内存索引（线程安全）
│   ├── pool_layout.py        # KVObjectPoolLayout — pool 文件的分配器
│   └── pool_io.py            # KVObjectPoolIO — 同步文件 I/O 实现
│
├── gpu_connector/
│   └── tutti_direct_loader.py  # TuttiDirectLoader / SnvmeSession / FiemapHelper
│
├── storage_backend/
│   └── local_disk_backend.py   # LocalDiskBackend — 集成入口
│
└── cache_engine.py             # _tutti_batched_get / _ensure_tutti_loader
```

---

## 3. 数据模型

### 3.1 KVObjectId

[object_id.py](lmcache/v1/kv_object_store/object_id.py) 是一个 frozen dataclass，
字段如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `model_id` | str | 模型名或 cache 命名空间 |
| `parallel_config_id` | str | TP 配置标识，如 `"world8"` |
| `rank` | int | 本地 rank |
| `layer_id` | int | Transformer 层 index |
| `role` | str | KV 角色：`csa` / `hca` / `swa` / `full` |
| `block_id` | str | vLLM/LMCache block 的哈希 hex |
| `schema_version` | int | 标识符 schema 版本（目前 = 1）|

`to_key()` 返回 sort_keys=True 的 JSON 字符串，用作 in-memory 索引键。  
`LocalDiskBackend._key_to_object_id()` 把 `CacheEngineKey` 映射到 `KVObjectId`，
当前实现将 `layer_id=0` + `role="full"` + `block_id=chunk_hash_hex`。

### 3.2 KVObjectRecord

[record.py](lmcache/v1/kv_object_store/record.py) 存放一个对象的**物理位置 + 生命周期状态**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `object_id` | KVObjectId | 标识符 |
| `pool_id` | str | 所属 pool，如 `"rank0-full"` |
| `offset` | int | pool 文件内的字节偏移 |
| `length` | int | 实际数据字节数 |
| `aligned_length` | int | 对齐后的保留字节数（用于 pool 分配） |
| `shape` | tuple[int, ...] | tensor shape |
| `dtype` | str | dtype 字符串，如 `"torch.bfloat16"` |
| `state` | KVObjectState | `ALLOCATED` / `READY` / `EVICTED` |
| `raw_extents` | tuple[tuple[int,int,int], ...] | Tutti raw LBA：`(file_offset, slba, n_sectors)` |

状态转换：
```
allocate() → ALLOCATED
  ↓ 写入完成
mark_ready() → READY
  ↓ 淘汰
mark_evicted() → EVICTED
```

`with_raw_extents()` 返回附加了 raw LBA 信息的副本，用于 Tutti raw 路径。  
该 dataclass 是 **frozen + immutable**，所有"修改"都返回新对象。

### 3.3 KVObjectState

```python
class KVObjectState(str, Enum):
    ALLOCATED = "allocated"   # 已分配但数据未落盘
    READY     = "ready"       # 数据已落盘，可读
    EVICTED   = "evicted"     # 已淘汰
```

`get_many(..., ready_only=True)` 只返回 READY 状态的记录，ALLOCATED/EVICTED 记作 miss。

---

## 4. KVObjectMetadataStore

[metadata_store.py](lmcache/v1/kv_object_store/metadata_store.py)

- 内部用 `dict[str, KVObjectRecord]` 保存索引，key 来自 `KVObjectId.to_key()`。
- 用 `threading.RLock` 保证线程安全（put/get/delete/records 都加锁）。
- 不关心底层存储引擎，只负责内存中的查找。

主要接口：

| 方法 | 说明 |
|---|---|
| `put(record)` | 插入或覆盖 |
| `get(object_id)` | 单条查找 |
| `get_many(ids, ready_only=True)` | 批量查找，返回列表（miss 为 None）|
| `delete(object_id)` | 删除并返回旧 record |
| `dump_jsonl(path)` | 序列化到 JSONL |
| `load_jsonl(path)` | 从 JSONL 反序列化 |

---

## 5. KVObjectPoolLayout

[pool_layout.py](lmcache/v1/kv_object_store/pool_layout.py)

管理 pool 文件的**逻辑地址分配**，有两种模式：

### 5.1 Fixed-slot 模式（默认）

每个对象占一个固定大小的 slot：

```
offset = slot_index × aligned_slot_bytes
```

文件大小 = `aligned_slot_bytes × capacity`（预分配稀疏文件）。

### 5.2 Dense 模式（Tutti 路径使用）

对象紧密排列，偏移量按实际对齐后大小累加：

```
offset = 上一个对象的 offset + 上一个对象的 aligned_length
```

文件随写入增长（`truncate` 到当前 `_next_offset`）。  
好处：避免稀疏文件的空洞，FIEMAP 结果更干净。

### 5.3 materialize_file=False

Tutti raw 模式下，`pool_path` 对应的文件并不实际存在（驱动器已 bind 给 snvme）。
此时 `allocate()` 只分配逻辑偏移，不创建文件。

### 5.4 reset_allocation()

Tutti 绑定前的 warmup 阶段可能有失败的写入，这些写入没有实际消耗 raw LBA 空间，
所以 bind 完成后调用 `reset_allocation()` 把 `_next_slot` 和 `_next_offset` 清零，
确保 cold-store 写入从 LBA region 的起点开始。

---

## 6. KVObjectPoolIO

[pool_io.py](lmcache/v1/kv_object_store/pool_io.py)

pool 文件的**同步 POSIX I/O 实现**，作为非 Tutti 路径的后备。

| 方法 | 说明 |
|---|---|
| `write_object(record, payload)` | 单对象写入（委托 write_many）|
| `write_many(records, payloads)` | 批量 pwritev，含尾部零填充到 aligned_length |
| `read_object(record)` | 单对象读取（返回 bytes）|
| `read_many(records)` | 批量 pread，返回 KVObjectReadBatch |
| `read_into_many(records, buffers)` | 批量 preadv，写入调用方提供的 buffer |

底层使用 `os.pwritev` / `os.pread` / `os.preadv`（零拷贝写入多段 iovec）。  
每次读写都独立打开/关闭 fd（无长期 fd 持有）。

---

## 7. LocalDiskBackend 集成

[local_disk_backend.py](lmcache/v1/storage_backend/local_disk_backend.py)

### 7.1 初始化（构造函数）

由 `LMCACHE_KV_OBJECT_STORE_ENABLE=1` 或 `extra_config["kv_object_store_enable"]`
触发。创建：

```python
pool_id  = f"rank{rank_id}-full"           # 例如 "rank0-full"
pool_path = Path(disk_path) / "_kv_object_store" / f"{pool_id}.pool"
kv_object_pool_layout  = KVObjectPoolLayout(dense=True, materialize_file=not tutti_raw)
kv_object_metadata_store = KVObjectMetadataStore()
kv_object_pool_io        = KVObjectPoolIO({pool_id: pool_path})
```

相关环境变量/配置：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LMCACHE_KV_OBJECT_STORE_ENABLE` | 0 | 启用 object store |
| `LMCACHE_KV_OBJECT_STORE_SLOT_MB` | 128 | 每个 slot 的 MB 上限 |
| `LMCACHE_KV_OBJECT_STORE_CAPACITY` | 2048 | pool 最大对象数 |
| `LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE` | 0 | 启用 raw LBA 写入 |
| `LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_BASE_LBA` | 0 | raw 区域起始 LBA（可逗号分隔 per-rank）|
| `LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_REGION_PATH` | - | pre-allocated raw 区域文件路径 |

### 7.2 写路径 `_write_kv_object_store`

`async_save_bytes_to_disk` 在正常文件写之前先尝试 object store 写：

```
1. _key_to_object_id(key) → KVObjectId
2. 检查 metadata store 是否已有该记录
3. 若无 → pool_layout.allocate(object_id, length=len(buffer))
4. 若有且 length 未变 → 复用旧 record
5a. Tutti raw 路径：raw_writer(record, buffer) → raw_extents
      → metadata_store.put(record.with_raw_extents(extents).mark_ready())
5b. 普通 pool 文件路径：pool_io.write_object(record, buffer)
      → metadata_store.put(record.mark_ready())
6. 返回 True → 调用方跳过普通 .pt 文件写入
```

当 `kv_object_tutti_raw_enabled=True` 但 `raw_writer=None`（Tutti 尚未初始化）时，
返回 False，回退到普通文件写。

### 7.3 读接口

```python
get_kv_object_records(keys)       # → list[Optional[KVObjectRecord]]，READY only
get_kv_object_pool_paths()        # → dict[pool_id, Path]
kv_object_data_path(record)       # → pool 文件路径 或 "tutti://rank0-full"（raw）
get_kv_object_raw_lba_cache(records)  # → {path: [(file_offset, slba, n_sectors)]}
```

### 7.4 Raw 区域映射 `map_kv_object_to_raw_region`

当使用 raw region 文件（`kv_object_tutti_raw_region_extents` 已设置）时，
把对象的逻辑 pool 偏移区间 `[record.offset, record.offset + aligned_length]`
映射到原始 LBA extents，供 Tutti 写使用。

---

## 8. Tutti 集成（cache_engine.py）

### 8.1 _ensure_tutti_loader：初始化序列

```
1. disk_backend.scan_existing_entries()         # 扫描现有 .pt 文件，注册到 dict
2. 收集所有文件路径，包含 object pool 文件路径
3. FiemapHelper.scan_paths(paths)               # 预取全部文件的 LBA extents（文件系统挂载时）
4. 若有 raw region → FiemapHelper.query_extents(raw_region_path)
5. _maybe_unmount_for_tutti()                   # 可选卸载文件系统
6. TuttiDirectLoader.create(initial_lba_cache)  # 绑定 snvme 驱动
7. 若 tutti_raw_enabled → 安装 _write_raw_object 闭包为 raw_writer
8. disk_backend.reset_kv_object_pool_allocation()
9. disk_backend.set_kv_object_tutti_raw_writer(writer)
```

### 8.2 _tutti_batched_get：读路径（含 object store 分支）

```python
# 1. 从 disk_backend.dict 取 DiskCacheMetadata（不读文件）
disk_metas = [disk_backend.dict.get(key) for key in keys]

# 2. 查询 object store
kv_object_records = disk_backend.get_kv_object_records(keys)

# 3. 若全部命中 → 切换到 object store 路径
if all(record is not None for record in kv_object_records):
    for original_meta, record in zip(disk_metas, kv_object_records):
        object_path = disk_backend.kv_object_data_path(record)
        # raw extents → 注册到 Tutti LBA cache
        if record.raw_extents:
            raw_lba_cache[object_path].extend(LbaRecord(...))
        object_metas.append(DiskCacheMetadata(path=object_path, size=record.length, ...))
        object_offsets.append(record.offset)
    # 更新 disk_metas 和 tutti_file_offsets
    disk_metas = object_metas
    tutti_file_offsets = object_offsets

# 4. 调用 Tutti DMA
results = _tutti_loader.load_chunks_to_hbm(
    keys, disk_metas,
    shapes_per_key=shapes_per_key,
    file_offsets=tutti_file_offsets,   # ← object store 偏移
)
```

`file_offsets` 是 object store 路径新增的参数，告诉 Tutti 在 pool 文件中的起始字节，
使多个对象可以共享同一个 pool 文件（密集 layout）。

---

## 9. TuttiDirectLoader：GPU-direct NVMe 路径

[tutti_direct_loader.py](lmcache/v1/gpu_connector/tutti_direct_loader.py)

### 9.1 读路径 load_chunks_to_hbm

```
for each sub-batch (bounded by q_depth & staging capacity):
    for each key in sub-batch:
        1. _get_extents(meta.path)    # FIEMAP 或 LBA cache
        2. 计算 staging 偏移（密集 pack by 实际 chunk 大小）
        3. 构造 NVMe READ SGL 描述符
    → _c_ops.tutti_submit_batch_sgl_read(...)  # GPU kernel 提交
    → _c_ops.tutti_poll_batch(...)             # GPU kernel 轮询 CQE
    → _check_nvme_status()
    → 为每个结果构造 TensorMemoryObj（GPU resident）
```

`file_offsets` 参数新增后，staging 偏移计算时会把 `extent.file_offset` 和
`file_offsets[i]` 一起考虑，正确读取 pool 文件内的子范围。

### 9.2 写路径 store_bytes_to_raw_extents / store_bytes_to_raw_lbas

```
payload (CPU bytes)
   → copy 到 HBM staging（zero_ + copy_）
   → 构造 NVMe WRITE SGL 描述符
   → _c_ops.tutti_submit_batch_sgl_write(...)
   → _c_ops.tutti_poll_batch(...)
   → _check_nvme_status()
   → 返回 list[LbaRecord]
```

返回的 `LbaRecord` 被存入 `KVObjectRecord.raw_extents`，持久化到 metadata store，
下次读取时可直接从 LBA 发起 NVMe READ，不需要 FIEMAP。

### 9.3 register_lba_cache

```python
loader.register_lba_cache({path: [LbaRecord(...)]})
```

支持 synthetic path（如 `"tutti://rank0-full"`），因为 raw 路径下文件系统
已不可访问，没有真实文件可 FIEMAP，只能靠预先存入 metadata 的 extents。

---

## 10. 两种存储模式对比

| | Pool 文件模式 | Tutti Raw LBA 模式 |
|---|---|---|
| 触发条件 | `kv_object_store_enable=True` | 同上 + `tutti_raw_enable=True` |
| 写入方式 | `KVObjectPoolIO.write_object` | `TuttiDirectLoader.store_bytes_to_raw_extents` |
| 文件 | 有实际 `.pool` 文件 | `materialize_file=False`，无文件 |
| 读取路径 | Tutti 通过 pool 文件路径 + offset 发起读 | Tutti 通过 `tutti://` synthetic path + raw LBA extents 发起读 |
| FIEMAP | 初始化时扫描 pool 文件 | 写时记录 extents，跳过 FIEMAP |
| 适用场景 | 文件系统可访问（热重启） | bind snvme 后文件系统 EIO |

---

## 11. 完整数据流图

### 写路径（含 raw）

```
vLLM KV blocks
      │ from_gpu()
      ▼
MemoryObj (CPU staging)
      │ async_save_bytes_to_disk()
      ▼
_write_kv_object_store()
 ├─ pool_layout.allocate()  ──────────────── KVObjectRecord (ALLOCATED)
 │                                               │
 ├─ [pool file mode]                             │
 │   pool_io.write_object()                      │
 │   metadata_store.put(record.mark_ready())     │
 │                                               │
 └─ [raw mode]                                   │
     raw_writer(record, buffer)                  │
      = store_bytes_to_raw_extents()             │
          CPU → HBM staging (H2D)                │
          NVMe WRITE DMA                         │
          → LbaRecord list                       │
     metadata_store.put(                         │
       record.with_raw_extents(...)              │
             .mark_ready())  ◄───────────────────┘
```

### 读路径

```
_process_tokens_internal()
      │ _tutti_batched_get(keys)
      ▼
disk_backend.get_kv_object_records(keys)  → list[KVObjectRecord|None]
      │ all READY?
      ├─ YES → 切换到 object store 路径
      │         disk_metas[i].path = kv_object_data_path(record)
      │         tutti_file_offsets[i] = record.offset
      │         raw_lba_cache → register_lba_cache()
      │
      └─ NO  → 回退到原始 disk_metas（per-key .pt 文件路径）
      │
      ▼
TuttiDirectLoader.load_chunks_to_hbm(
    keys, disk_metas, file_offsets=tutti_file_offsets)
      │
      ▼
[per sub-batch]
  _get_extents(path)   ← FIEMAP 或 raw_extents 缓存
  staging pack by chunk size
  NVMe READ DMA → HBM staging
  TensorMemoryObj (raw_tensor on GPU)
      │
      ▼
gpu_connector.to_gpu() (G2G copy)
      │
      ▼
vLLM KV cache slots
```

---

## 12. 关键设计决策

1. **对象不可变（frozen dataclass）**：`KVObjectRecord` 所有"修改"返回新对象，
   避免并发写元数据时的 race condition。

2. **metadata store 与 I/O 分离**：`KVObjectMetadataStore` 只管内存索引，
   `KVObjectPoolIO` / `TuttiDirectLoader` 分别处理文件 I/O 和 GPU-direct I/O，
   两者可独立替换。

3. **dense layout**：pool 文件使用密集布局（无空洞），保证 FIEMAP 返回连续或少量
   extent，减少每次读取时的 NVMe I/O 数量。

4. **raw extents 持久化**：cold-store 写入时把 LBA extents 写入 metadata，
   warm-hit 读取时直接用 extents 跳过 FIEMAP，避免文件系统 EIO 路径上的失败。

5. **graceful fallback**：
   - object store 未全部命中 → 回退到 per-key `.pt` 文件路径
   - Tutti raw_writer 未安装（warmup 期间）→ 返回 False → 普通文件写
   - Tutti init 失败 → CPU 路径

6. **pool 文件路径集成到 FIEMAP 预扫描**：初始化时把 pool 文件路径也加入
   `FiemapHelper.scan_paths`，确保在 snvme bind 前就缓存了 pool 文件的 LBA。
