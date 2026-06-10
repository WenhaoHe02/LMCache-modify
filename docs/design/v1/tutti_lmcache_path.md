# LMCache 基于 Tutti 的 KV 读取路径说明

本文只描述 LMCache 当前集成 Tutti 后的功能路径：KV 如何从 vLLM 请求、LMCache
本地 SSD 命中，进入 Tutti GPU-direct NVMe 读取，再作为 GPU resident
`MemoryObj` 回填给 vLLM。CSA/HCA 预取是另一路功能，不在本文展开。

## 目标

Tutti 路径解决的是 LMCache full-hit 时的本地 SSD KV 读取方式。

传统路径是：

```text
LocalDiskBackend file read
  -> CPU 内存 buffer
  -> LMCache MemoryObj
  -> GPU connector H2D copy
  -> vLLM paged KV cache
```

Tutti 路径改成：

```text
LocalDiskBackend metadata hit
  -> Tutti 根据文件 LBA/extent 发 NVMe read
  -> NVMe DMA 到 GPU HBM staging
  -> 包装成 GPU resident MemoryObj
  -> vLLM GPU connector 回填 paged KV cache
```

所以当前实现的功能重点不是改变 LMCache 的 cache key、lookup 或 hit 语义，而是替换
`LocalDiskBackend` 命中后的数据搬运路径。

## 主要组件

| 组件 | 职责 |
|---|---|
| `LMCacheEngine` | 决定是否启用 Tutti、创建 loader、在 retrieve 时选择 Tutti 或 fallback |
| `StorageManager` | 仍然负责 cache backend 管理和普通 `batched_get` fallback |
| `LocalDiskBackend` | 仍然负责写 KV 文件、保存 `DiskCacheMetadata`、扫描已有文件 |
| `TuttiDirectLoader` | 根据 disk metadata 和 FIEMAP/LBA 信息，把 KV chunk 从 NVMe DMA 到 HBM |
| `lmcache.c_ops` | 提供 GPU kernel：提交 NVMe read SQE、轮询 CQE |
| snvme/libnvm | 把 NVMe controller、SQ/CQ ring、doorbell、HBM staging 注册成 GPU 可访问资源 |

## 冷请求 store 阶段

冷请求不直接使用 Tutti 读取，因为此时 KV 还不存在于本地 SSD。vLLM 完成 prefill
后，LMCache 仍走原来的 store 路径：

```text
vLLM prefill computes KV
  -> LMCacheEngine.store()
  -> LocalDiskBackend.async_save_bytes_to_disk()
  -> disk file + DiskCacheMetadata
```

当前实现可以在 cold store 完成后做 Tutti warmup。这个 warmup 的作用是把首次
full-hit 的 Tutti 初始化成本尽量挪出用户可见请求路径：

```text
最后一个 cold-store chunk 完成
  -> _make_tutti_store_warmup_callback()
  -> 后台线程等待本请求所有 keys 写完
  -> _ensure_tutti_loader_locked(keys)
       -> scan_existing_entries()
       -> FIEMAP pre-scan: file path -> LBA extents
       -> unmount ext4 cache filesystem
       -> SNVM_DEVICE_BIND / session create
       -> map SQ/CQ ring, doorbell, HBM staging
```

FIEMAP 必须在 snvme bind/unmount 之前完成。bind 后文件系统路径可能不再可读，
Tutti 只能依赖已经缓存好的 LBA extent 信息发 NVMe read。

## full-hit retrieve 阶段

请求再次到来时，vLLM 通过 LMCache connector 触发 `LMCacheEngine.retrieve()`。
lookup 命中后，LMCache 得到需要加载的 chunk key 列表：

```text
request arrives
  -> LMCache lookup by token hash
  -> StorageManager block mapping
  -> LocalDiskBackend metadata hit
  -> LMCacheEngine._tutti_batched_get()
```

`_tutti_batched_get()` 不立即读文件内容，只从 `LocalDiskBackend.dict` 取
`DiskCacheMetadata`，然后交给 `TuttiDirectLoader.load_chunks_to_hbm()`。

### DSv4 optimized KV 的读取长度

DSv4 optimized mode 下，每个 chunk 的实际读取长度不能简单相信磁盘 metadata 的
原始 shape。LMCache 会根据当前 chunk 在请求里的 `[start, end)` 范围计算
`shapes_per_key`：

```text
non-tail chunk -> 只读本轮需要的 DSv4 groups
tail chunk     -> 读 tail-specific groups
```

Tutti loader 使用 `_effective_nbytes(meta, shapes_per_key[i])` 计算实际 DMA 长度。
这样一个物理文件可以保存较完整的 KV，但 full-hit retrieve 只把当前 DSv4
执行需要的 bytes DMA 到 HBM。

## TuttiDirectLoader 内部流程

每次 `load_chunks_to_hbm()` 会把多个 cache chunks 合成若干 batch。batch 同时受两类限制：

1. NVMe queue depth：本批展开后的 IO 数不能超过 queue depth。
2. HBM staging pool：本批实际 DMA bytes 不能超过 `tutti_n_slots * tutti_slot_mb`。

单个 chunk 的处理流程：

```text
DiskCacheMetadata(path, shapes, dtypes)
  -> _get_extents(path)
  -> multi-extent LBA list
  -> effective_nbytes
  -> align to 512B NVMe LBA
  -> split by controller max_data_size
  -> build arrays:
       staging_iova[]
       slba[]
       byte_len[]
       io_to_key_index[]
```

提交阶段：

```text
c_ops.tutti_submit_batch_sgl_read(...)
  -> GPU kernel writes NVMe SQE
  -> GPU rings SQ doorbell

c_ops.tutti_poll_batch(...)
  -> GPU kernel polls CQE
  -> GPU rings CQ doorbell
```

完成后，loader 从 HBM staging 里按 key 切出对应 bytes，包装成
`TensorMemoryObj`。当前实现会把 staging 内容 clone 成独立 GPU tensor，再交给
LMCache；否则 staging pool 在下一批读时会被复用，导致前一批 `MemoryObj` 内容被覆盖。

## 回填 vLLM KV cache

Tutti 返回的是 GPU resident `MemoryObj`，所以后续仍走 LMCache 原有的 GPU connector
接口，只是输入已经在 HBM：

```text
Tutti TensorMemoryObj
  -> LMCache retrieve reordered_chunks
  -> VLLMPagedMemGPUConnectorV3.to_gpu()
  -> 按 vLLM slot_mapping / HMA block metadata 回填 paged KV cache
  -> vLLM prefill/decode 使用官方 KV cache
```

也就是说，Tutti 不直接改 vLLM attention kernel 的读取地址。它只是把 LMCache
SSD chunk 更快地变成 GPU resident memory object，最终仍由 connector 按 vLLM
metadata 写入官方 KV cache。

## 错误处理与 fallback

当前实现有三层保护：

1. 初始化失败：如果 snvme bind、HBM staging 分配或 session 创建失败，LMCache 会尝试
   remount 原 ext4 文件系统，允许 CPU filesystem fallback。
2. 运行期 extent 缺失：如果 Tutti direct load 找不到可读 KV extent，`_tutti_batched_get()`
   不再把 vLLM worker 打成 500；能 CPU fallback 就调用 `StorageManager.batched_get()`，
   否则把这些 blocks 当 miss，让请求正常计算。
3. debug checksum：`LMCACHE_TUTTI_DEBUG_CHECKSUM=1` 时，初始化阶段会为部分文件计算
   CPU 侧 checksum，loader 完成后可对比 GPU direct read 内容。

这些保护只影响异常路径；正常 full-hit 时仍优先走 Tutti direct path。

## 配置开关

| 配置 | 含义 |
|---|---|
| `tutti_device_path` | snvme 字符设备，如 `/dev/ssnvme0` |
| `tutti_ctrl_path` | snvme control device，通常是 `/dev/snvm_control` |
| `tutti_pci_bdfs` | 每个 local rank 对应的 NVMe PCI BDF；可用 `skip` 跳过某些 rank |
| `tutti_n_slots` | HBM staging slot 数，也是 batch 并发容量的重要上限 |
| `tutti_slot_mb` | 每个 staging slot 大小；当前 DSv4 大 chunk 需要约 128 MiB |
| `tutti_nsid` | NVMe namespace id |
| `tutti_warmup_after_store` | cold store 完成后后台初始化 Tutti |
| `tutti_warmup_after_store_delay_sec` | warmup 前等待时间，避免 cold prefill 刚结束时 HBM 太紧 |
| `LMCACHE_TUTTI_FORCE_CPU_FALLBACK=1` | 强制关闭 Tutti direct path |
| `LMCACHE_TUTTI_DEBUG_CHECKSUM=1` | 打开 direct read checksum 诊断 |

## 当前功能边界

1. 一个 LMCache chunk 必须能放进一个 Tutti staging slot。当前还不支持一个 chunk
   跨多个 staging slot。
2. 支持一个文件有多个 filesystem extents；不能要求 single-contiguous extent。
3. LBA pre-scan 是当前实现的关键前提。snvme bind/unmount 后新写入的文件如果没有更新
   LBA cache，不能直接被 Tutti 读取。
4. Tutti 是 KV chunk retrieve 加速路径，不负责 CSA/HCA 预取策略。
5. 当前路径已经验证 correctness，但性能仍受 `process_tokens_ms`、batch submit/poll、
   MDTS 分片、queue depth、rank 间长尾等影响。

## 已验证行为

在 gpu002 的 `dsv4-256k-measure-tutti` 容器里，DSv4 128K max length、TP=8、
Tutti direct KV retrieve 路径已跑通：

| 场景 | 结果 |
|---|---|
| 42K full-hit | cold/hit1/hit2 输出完全一致 |
| 119K full-hit | cold/hit1/hit2 输出完全一致 |
| 119K hit tokens | `Retrieved 119040 out of 119040 required tokens` |
| steady retrieve profile | 每 rank 约 740 MiB，`to_gpu_ms` 约 425-434 ms |

这些结果说明：当前 LMCache + Tutti 路径已经能把本地 SSD KV chunk 通过 GPU-direct
NVMe 读入 HBM，并正确回填 vLLM KV cache。下一步优化应集中在吞吐和长尾，而不是
correctness 主链路。
