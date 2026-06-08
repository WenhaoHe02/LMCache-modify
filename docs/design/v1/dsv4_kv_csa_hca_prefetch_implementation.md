# DSv4 KV 管理、CSA 推测预取与 HCA overlap 实现说明

本文说明当前 LMCache + vLLM DeepSeek V4 路线里，我们实现和验证过的三块能力：

1. DSv4 异构 KV cache 的管理与传输。
2. CSA residual-proxy speculative prefetch。
3. HCA deterministic prefetch 与 MoE/FFN 窗口 overlap。

本文只描述实现机制和正确性约束；实验命令、容器状态和一次性结果记录放在
`docs/design/v1/dsv4_csa_hca_prefetch_runbook.md` 和
`docs/design/v1/dsv4_csa_hca_prefetch_source_of_truth.md`。

## 目标

DeepSeek V4 的 KV 状态不是普通 dense transformer 的单一 KV cache。vLLM runtime 里
存在多类不同语义的状态：

| 角色 | 语义 | 压缩比 | GPU 策略 |
|---|---|---:|---|
| `swa_cache` | 最近窗口的精确 SWA 状态 | 1 | 只保留 tail/window |
| `hca_attention_kv` | HCA compressed attention KV | 128 | deterministic read，可提前提交 |
| `csa_attention_kv` | CSA compressed attention KV | 4 | true LI sparse 选择，proxy 只负责预取 |
| `csa_indexer_cache` | Lightning Indexer scoring 所需 K cache | 4 | hot/resident pool + SSD fallback |
| `compressor_state` | 继续压缩所需边界状态 | 1 | 只保留 tail/window |

优化目标不是简单减少 SSD 上保存的数据，而是降低 full-hit 时进入 GPU/HBM 的
critical-path payload，并把 SSD/NVMe read 尽量藏到 prefill 或 FFN/MoE 计算窗口里。

正确性约束：

- 官方 true Lightning Indexer 永远是 CSA attention top-K 的正确性来源。
- CSA proxy 只决定提前读哪些 block，不能替代 true LI 的结果。
- HCA 和 CSA 必须分开调度：HCA 是 deterministic read，CSA 是 learned-indexer speculative read。
- CPU pinned memory 只能是 transient I/O bounce buffer，不能作为 cache hit、resident set 或长期 KV cache。
- `LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH` 是 per-token decode 原型开关，不是 prefill proxy 开关；性能/稳定性实验默认不打开。

## 代码入口

当前实现分布在四个主要位置：

| 文件 | 责任 |
|---|---|
| `lmcache/v1/gpu_connector/gpu_connectors.py` | DSv4 HMA group 识别、role-aware H2D、HCA defer |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | vLLM adapter hook、LMCache hit 后 seed CSA/HCA 状态 |
| `lmcache/v1/indexer_ssd_manager.py` | CSA SSD store、HBM pool、residual proxy、true-topK correction |
| `lmcache/v1/hca_prefetch_manager.py` | HCA flat SSD store、deterministic fire/drain、pinned transient buffer |

vLLM 侧需要 DeepSeek decoder layer 暴露以下 hook：

```text
attach_indexer_prefetch(manager, next_csa_layer_id)
attach_hca_prefetch(manager, current_hca_layer_id, next_hca_layer_id)
```

这些 hook 的语义是让 LMCache manager 在 decoder forward 内拿到 FFN/MoE 前后的调度窗口：

```text
previous/current layer attention done
  -> CSA/HCA fire_async(...)
FFN / MoE / A2A compute window
  -> SSD/NVMe read runs in background
target CSA/HCA attention before use
  -> drain/correction
  -> official attention consumes true slots
```

## DSv4 KV 管理与传输

### HMA group 识别

`VLLMPagedMemGPUConnectorV3` 在注册 vLLM KV cache 时调用
`KVLayerGroupsManager`，把 DSv4 的异构 cache 分成多个 LMCache transfer group。

`_dsv4_group_role()` 用 dtype、hidden size、compress ratio 和 layer 数识别 group：

```text
float32                         -> compressor_state
uint8 hidden=132                -> csa_indexer_cache
uint8 hidden=584 compress=1     -> swa_cache
uint8 hidden=584 compress>=64   -> hca_attention_kv
uint8 hidden=584 compress=4     -> csa_attention_kv
```

这个识别是 role-aware transfer 的基础。同样 shape 的 tensor 不能只按位置硬猜；
必须结合 HMA metadata，把 LMCache group 映射回 vLLM 的 KV cache group。

### block id 与 slot mapping

HMA 下每个 vLLM group 可以有不同 logical block size。`to_gpu()`/`from_gpu()`
优先走 `_hma_block_ids_for_group()`：

```text
LMCache group
  -> layer name
  -> vLLM HMA group id
  -> block_ids_by_group[group_id]
  -> group-specific logical block size
  -> multi_layer_block_kv_transfer()
```

只有拿不到 HMA metadata 时才 fallback 到 legacy `slot_mapping -> block_ids`。
这样避免把所有 group 都错误地当作 group 0 的 block table 来 scatter。

### role-aware H2D policy

`LMCACHE_DSV4_OPTIMIZED_KV=1` 时，`to_gpu()` 会启用 DSv4 role-aware policy：

```text
swa_cache          -> tail-only H2D
compressor_state   -> tail-only H2D
hca_attention_kv   -> full H2D 或 defer-to-moe
csa_attention_kv   -> full H2D，后续目标是 selective/hot load
csa_indexer_cache  -> full H2D，后续由 CSA HBM pool 管理 hot entries
```

tail 长度由 `LMCACHE_DSV4_OPTIMIZED_TAIL_TOKENS` 或
`extra_config.dsv4_optimized_tail_tokens` 控制，最小为 LMCache chunk size。

如果 `LMCACHE_DSV4_DEFER_HCA_TO_MOE=1`，HCA group 不走普通 H2D scatter：

```text
LMCache retrieve memory_tensor
  -> HCAPrefetchManager.seed_range_from_lmcache_group()
  -> write HCA flat store
  -> skip normal HCA H2D
  -> later fire/drain into target vLLM HCA KV cache
```

这一步把 HCA attention KV 从“retrieve 后立即回填 GPU”改成“先变成 HCA
deterministic prefetch 的数据源”。当前实现仍然是 Python/pinned-transient 原型；
最终性能路径应该替换成 GDS/cuFile、BAT/gio_uring 或 GPU-visible staging。

### 当前 payload 口径

248K full-hit 实测 LMCache log 里：

```text
size = 4.7469 GB/rank
tokens = 248,320
payload ~= 18.7 KiB/token/rank
TP8 total ~= 149 KiB/token
```

只算 DSv4 prefix-scaled compressed attention + indexer cache 的理论量级：

```text
HCA attention KV   = 31 * 584 / 128 ~= 141 B/token/rank
CSA attention KV   = 30 * 584 / 4   ~= 4380 B/token/rank
CSA indexer cache  = 30 * 132 / 4   ~= 990 B/token/rank
total              ~= 5.4 KiB/token/rank
```

因此当前 full-hit load payload 仍高于理论 prefix-load 下限。后续要继续把
`csa_attention_kv` 和 `csa_indexer_cache` 从 full H2D 变成 selective/hot HBM residency，
并为每个 group 打出 transferred bytes，定位额外 payload 来自哪个 group。

## CSA speculative prefetch

### 数据结构

CSA 路径由 `IndexerSSDManager` 管理，核心状态如下：

```text
per CSA layer:
  IndexerBlockStore
    - one flat SSD file
    - key: logical compressed token id
    - value: vLLM packed indexer-cache row bytes

  CSAHBMPool-like pool
    - fixed-size HBM pool
    - token_id -> physical pool slot
    - resident seed LRU + ordinary LRU

  pending reads
    - token_id -> Future[bytes]
    - drained before true attention consumes the data
```

SSD 文件保存的是 indexer cache 的 K vectors。HBM pool 只保存当前可能会被
Lightning Indexer scoring 用到的 hot rows。pool 命中不是 LMCache hit；它只是
当前请求内部的 HBM residency。

### LMCache hit 后的 reuse seed

当 LMCache full-hit 成功后，`LMCacheConnectorV1Impl` 调用
`_maybe_seed_indexer_reuse_prefetch()`：

```text
LMCache load success
  -> scan registered DeepSeek decoder layers
  -> find CSA indexer op
  -> read k_cache.kv_cache from current vLLM HBM
  -> compute compressed slot mapping from logical slot_mapping
  -> seed tail compressed token ids into IndexerSSDManager
```

这里不再限制 `local_worker_id == 0`，TP8 每个 rank 都要 seed 自己 rank 的
indexer cache。seed 只准备 CSA SSD/HBM pool 状态，不改变官方 attention 结果。

### prefill proxy

prefill 阶段的正确开关是：

```text
LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=1
LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS=1
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0
```

`LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS=1` 表示允许在 prefill chunk 内用 tail row
做 CSA proxy prefetch。`DECODE_PREFETCH=0` 不会禁止 prefill proxy；它只禁止
per-token decode 原型。

当前实现里，prefill proxy 的关键点是：

```text
proxy_state rows > 1
  -> classify as prefill proxy
  -> run target CSA layer's residual proxy path
  -> only use last row's top-K to submit SSD reads
```

曾经尝试只把 proxy tensor 切成 `[1, ...]` 再跑 indexer，但 vLLM/SparseAttnIndexer
的 prefill metadata 仍按整个 chunk 构造，直接切 token 维会破坏 CUDA metadata shape。
所以当前工作实现保留完整 prefill compute shape，只在 top-K buffer 输出侧取 tail row。
这保证语义正确，但 proxy compute 仍然偏重，是后续需要优化的原型成本。

### decode proxy

decode 阶段只有显式打开 `LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=1` 才会执行。
当前它是 Python prototype，历史实验显示会把 decode 拉慢；默认性能实验不打开。

decode proxy 的语义：

```text
before target CSA layer:
  residual_f / HC-post state
    -> target layer HC_pre + attention norm + indexer projection
    -> predicted top-K logical token ids
    -> filter already resident ids
    -> submit SSD reads for missing predicted ids

after official true LI:
  true top-K logical token ids
    -> compare resident/predicted ids
    -> read missing true ids
    -> insert into HBM pool
    -> sparse attention consumes true slots
```

如果 proxy 不可用，manager 暂停投机预取，等 true top-K 返回后再发 miss read。

### true-topK correction

`record_attention_topk_slots()` 和 `correct_true_topk()` 负责把官方 true LI 结果
反馈给 manager：

```text
official SparseAttnIndexer.forward_cuda()
  -> true_topk token ids
  -> IndexerSSDManager.correct_true_topk()
  -> _drain(layer)
  -> read missing true ids synchronously/as fallback
  -> pool.insert()
  -> build true token id -> HBM pool slot mapping
```

prefill proxy 下，`_correct_prefill_true_topk()` 会单独记录：

- `prefill_correct_true_topk.total_ms`
- `drain_ms`
- `read_ms`
- `insert_ms`
- true/predicted/spec-hit/missing 数量

这些指标用来判断 proxy accuracy 和 exposed wait，而不是替代端到端 TTFT。

### 证明 SSD read 是否被隐藏

不能只看 `prefill_correct_true_topk.read_ms`。更可靠的口径是：

```text
IndexerSSDTiming drain:
  wait_ms mean/max
  pending count
  fallback count

IndexerSSDTiming prefill_correct_true_topk:
  drain_ms
  read_ms
  insert_ms
  missing count
```

如果 drain 的 `wait_ms mean/max` 很小且 `fallback=0`，说明异步 SSD read 在
prefill/compute 窗口内基本完成。若 `missing` 很大，则说明 proxy/residency 命中率
不足，后续要优化 prediction 和 pool policy；这和“SSD read 是否已经完成”是两个问题。

## HCA deterministic prefetch 与 overlap

### HCA 为什么可以 deterministic

HCA 层使用 `compress_ratio=128`。对一个 reused prefix，目标 HCA layer 需要读取的
compressed rows 只由 prefix 长度、compress ratio 和当前请求的 HMA slot mapping 决定，
不依赖 MoE expert routing，也不依赖 CSA true LI。

因此 HCA 可以在目标 attention 之前提前提交：

```text
seq_len S
  -> rows = ceil(S / 128)
  -> logical compressed row ids known
  -> current request slot mapping known
  -> submit SSD/NVMe read
  -> target HCA attention before use drain
```

### HCA manager 状态

`HCAPrefetchManager` 为每个 HCA layer 维护：

```text
HCALayerState:
  layer_id
  compress_ratio = 128
  block_size
  row_bytes
  kv_cache pointer
  HCACompressedStore flat SSD file
  initialized_rows

runtime:
  active compressed row -> current vLLM physical slot mapping
  pending HCAPendingRead list
  resident_hbm metadata for current forward
  transient pinned I/O slab
```

`resident_hbm` 只表示当前 forward 已经 drain 到目标 vLLM KV cache 的 compressed rows；
不能把 flat SSD store 或 CPU pinned buffer 当成 HBM resident。

### seed 来源

HCA seed 有两条路径：

1. `submit_seed_after_reuse()`：LMCache hit 后，从已经回填到 HBM 的 HCA KV cache
   读出 rows，写入 HCA flat SSD store。这是早期诊断路径。
2. `seed_range_from_lmcache_group()`：`to_gpu()` 看到 `hca_attention_kv` group 后，
   直接用 LMCache retrieve 出来的 group tensor seed HCA flat store，然后跳过普通 H2D。

第二条路径是当前更接近正确设计的路径，因为它避免了“先 full H2D 再反拷 seed”的
自相矛盾流程。

### fire/drain

HCA fire/drain 的目标时序：

```text
current decoder layer attention done
  -> next_hca_layer_id known
  -> HCAPrefetchManager.fire_async_for_layer(next_hca, seq_len)
  -> submit missing compressed rows to background executor

FFN / MoE / A2A window
  -> SSD/NVMe read into transient pinned buffer

target HCA attention before use
  -> HCAPrefetchManager.drain_for_layer(current_hca)
  -> copy rows into target vLLM HCA kv_cache slots
  -> release pinned buffer
```

`LMCACHE_HCA_BLOCKING_DRAIN=0` 时，fire 提交异步读，drain 只在目标 attention
前等待未完成部分。profile 里需要看：

- `HCAPrefetchTiming event=fire`
- `HCAPrefetchTiming event=drain`
- rows/missing/written
- wait/copy/write 相关耗时

### pinned transient 不是 cache

当前 Python 原型需要：

```text
LMCACHE_HCA_ENABLE_PREFETCH=1
LMCACHE_HCA_ENABLE_PINNED_BOUNCE=1
```

这条路径是：

```text
SSD/NVMe -> CPU pinned transient slab -> target vLLM HCA KV cache
```

它只用于验证状态机和 overlap 时序。最终性能路径应该是：

```text
SSD/NVMe -> GPU-visible staging / HBM -> target vLLM HCA KV cache
```

所以文档和实验结论不能把 pinned buffer 称为 cache，也不能把 HCA Python
pinned-transient 原型当成最终性能实现。

## full-hit 请求时序

下面是当前 full-hit + prefill proxy + HCA overlap 的完整逻辑时序：

```text
request arrives
  -> vLLM builds HMA block metadata and slot_mapping
  -> LMCache lookup by token hash
  -> retrieve full-hit chunks from SSD

VLLMPagedMemGPUConnectorV3.to_gpu()
  -> classify DSv4 groups
  -> SWA/compressor_state: tail-only H2D
  -> HCA group:
       if defer enabled:
         seed HCA flat store from LMCache group tensor
         skip normal H2D
       else:
         normal H2D
  -> CSA groups:
       current path still mostly normal H2D
       future path should become selective/hot load

LMCacheConnectorV1Impl load success
  -> _maybe_seed_indexer_reuse_prefetch()
       seed CSA SSD/HBM pool from reused prefix
  -> _maybe_seed_hca_reuse_prefetch()
       set active HCA slot mapping
       fire deterministic HCA rows if flat store is ready

prefill forward
  -> decoder layer attention/HC-post state available
  -> IndexerSSDManager.fire_async_for_layer(next CSA)
       if PREFETCH_PREFILL_ROWS=1:
         run prefill proxy
         submit predicted tail-row CSA reads
  -> HCAPrefetchManager.fire_async_for_layer(next HCA)
       submit deterministic HCA reads
  -> FFN/MoE/A2A runs
       SSD reads overlap compute
  -> target CSA/HCA before use
       drain pending reads
       true LI correction for CSA
       HCA rows copied into target KV cache
  -> official attention consumes true slots
```

## 环境开关建议

### corrected prefill-only CSA proxy

```text
LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=1
LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS=1
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0
LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION=0
LMCACHE_HCA_ENABLE_PREFETCH=0
```

用途：验证 CSA prefill proxy 能否把 SSD read wait 藏进 prefill compute 窗口。

### HCA deterministic overlap diagnostic

```text
LMCACHE_HCA_ENABLE_PREFETCH=1
LMCACHE_HCA_ENABLE_PINNED_BOUNCE=1
LMCACHE_DSV4_DEFER_HCA_TO_MOE=1
LMCACHE_HCA_BLOCKING_DRAIN=0
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0
```

用途：验证 HCA seed/fire/drain 状态机。当前不是最终性能默认开关。

### 不建议默认打开

```text
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION=1
LMCACHE_HCA_ENABLE_DECODE_HOOK=1
```

这些属于旧诊断或 decode Python prototype，容易污染 TTFT 结论。

## profile 指标

CSA prefill proxy 主要看：

```text
IndexerSSDTiming: event=reuse_prefetch_seed
IndexerSSDTiming: event=prefill_fire_async
IndexerSSDTiming: event=drain
IndexerSSDTiming: event=prefill_correct_true_topk
IndexerSSDTiming: event=attention_true_topk
```

解释口径：

- `prefill_fire_async.proxy_ms`：proxy compute 成本。
- `prefill_fire_async.submit_ms`：向 executor 提交 SSD read 的成本。
- `drain.wait_ms mean/max`：真正暴露在 critical path 上的等待。
- `drain.fallback`：异步读没有覆盖住时的 fallback 次数。
- `prefill_correct_true_topk.read_ms`：true miss correction 的同步读成本。
- `attention_true_topk.spec_hits / true_misses`：proxy/pool 命中质量。

HCA overlap 主要看：

```text
HCAPrefetchTiming: event=seed
HCAPrefetchTiming: event=fire
HCAPrefetchTiming: event=drain
```

解释口径：

- `fire missing` 表示需要从 HCA flat store 读的 compressed rows。
- `drain written` 表示最终写入目标 HCA KV cache 的 rows。
- `drain wait` 表示 HCA SSD read 暴露在目标 attention 前的等待。
- 如果启用 defer，日志必须出现 HCA group seeded and deferred normal H2D。

## 已知限制

1. CSA prefill proxy 当前保留完整 prefill compute shape，只取 tail row top-K；
   语义正确但 proxy compute 成本偏高。
2. CSA pool insertion 和 future submit/drain 仍是 Python prototype，profile 里会出现
   明显 overhead。
3. 当前 CSA true-topK recall 仍可能不足，missing correction 多时会证明 prediction/pool
   需要优化，但不等价于 SSD read 没有被隐藏。
4. HCA pinned-transient 只用于状态机诊断，不是最终性能路径。
5. 当前 LMCache SSD persistent payload 和 GPU critical-path load payload 还没有完全解耦。
6. 后续必须补 per-group bytes profile，明确每个 DSv4 group 的 SSD read bytes、H2D bytes
   和 effective GPU resident bytes。

## 后续实现方向

短期：

- 给 `to_gpu()` 加 per-group profile：role、shape、tokens、SSD bytes、H2D bytes、是否 tail-only、是否 defer。
- 给 CSA manager 加 per-layer/prefill-row overlap profile：predicted ids、true ids、Jaccard/recall、missing reads。
- 把 CSA prefill proxy 的 compute 从“全 chunk compute + tail top-K”优化成和 vLLM metadata 兼容的 tail-only runner。

中期：

- `csa_attention_kv` 和 `csa_indexer_cache` 改成 selective/hot load，不再 full H2D。
- HCA 从 Python pinned transient 改成 GPU-visible staging。
- CSA/HCA 共用 HBM residency manager，统一 eviction、slot ownership 和 profile。

长期：

- SSD persistent format 与 GPU load format 解耦。
- 对 DSv4 role 建立独立 object namespace，而不是把所有 vLLM groups 都包装成普通 LMCache full-prefix chunk。
- 对 CSA 跨 step、跨 layer block overlap 做统计；若 overlap 高，推动统一 HBM residency 和跨层共享策略。
