# 面向原生稀疏注意力的 Index-to-I/O 编译器

> 本文件是 `index_to_io_compiler.md` 的中文翻译版。

## 状态与范围

本文档规定了一个分阶段的编译器与运行时，用于把稀疏注意力决策转化为
SSD 读取和最终的 KV-cache 放置。它首先面向 DSv4 的 CSA/HCA/SWA，但可复用
的接口不得硬编码仅适用于 DSv4 的布局假设。

该设计建立在当前分阶段的 HCA-to-CSA 流水线之上。它不会把残差预测器
（residual predictor）当作权威来源，默认情况下也不会削弱目标层注意力的
语义。

实现目标是：

1. 当稀疏决策的正确性标签允许时，跨层或跨请求复用该决策；
2. 把同一个决策下译（lower）为 indexer-K 与 attention-KV 的 SSD 范围；
3. 基于实测代价而非固定的环境变量开关，在稀疏、区间（range）、整体
   （bulk）I/O 之间做选择；
4. 把 NVMe 完成与 GPU 物化（materialization）分离，使每种资源都能按
   目标层的截止时间（deadline）独立调度；
5. 在预测失败、取消、并发请求、行复用、部分 I/O 与迟到完成等情况下
   保持正确性。

第一版实现仍保持进程本地（process-local）与 rank 本地（rank-local）。
在本地正确性与收益得到验证之前，计划的持久化或跨 rank 协调不在范围内。

## 核心论点

稀疏的计算索引并不会自动产生稀疏的存储 I/O。在一次长 prefill 中，多个
per-query top-K 结果的并集可能覆盖前缀的大部分。此时，即使 SSD 数据面能
维持 9--11 GB/s，逐对象的 host 回调与立即 scatter 也可能成为主导开销。

所提出的系统把一个真实的或带守护（guarded）的稀疏索引视为一个不可变的
计划（plan）：

```text
 true anchor index / exact decision replay / speculative prediction
                              |
                              v
                    correctness-tagged Plan IR
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
     indexer-K physical ranges         attention-KV physical ranges
             |                                 |
             +---------- adaptive lowering ----+
                              |
                              v
        NVMe admission -> CQ_READY -> GPU materialization -> HMA_READY
```

因此，一个计划可以同时：消除冗余的 indexer 计算、省去对应的 indexer-K
字节、并预取 attention-KV 字节。这三种收益只有在该计划的正确性等级允许
时才可以被声称。

## 正确性分类学

每个计划恰好属于一个正确性等级。

| 等级 | 来源 | 可预取 KV | 可替代逐行 top-K | 可跳过 indexer |
|---|---|---:|---:|---:|
| `EXACT_TRUE` | 当前官方 indexer | 是 | 是 | 已经执行过 |
| `EXACT_REPLAY` | 精确 decision-key 命中且缓存了逐行 top-K | 是 | 是 | 是 |
| `CERTIFIED_EXACT` | 未来的证明/证书机制 | 是 | 是 | 是 |
| `SPECULATIVE` | anchor 复用、残差预测、heavy hitters | 是 | 否 | 否 |
| `QUALITY_GUARDED` | 经校准的近似复用 | 是 | 需显式开启 | 需显式开启 |

有两条不可协商的边界：

- 块并集位图（block-union bitmap）是一个 I/O 计划，不是逐行的注意力
  决策。它永远不能被当作目标 indexer 的 top-K 返回。
- 真实 top-K 的 miss 校正只修复 KV 驻留（residency），不修复近似的注意力
  索引。投机（speculative）的目标层仍必须恰好执行一次官方 indexer。

`QUALITY_GUARDED` 是一种显式的近似模式。校准、影子执行（shadow
execution）与漂移检测可以限制其风险，但无法使单个请求变得精确。默认的
生产路径只使用 `SPECULATIVE` 预取与精确的目标层计算。

## 架构

系统包含四个逻辑组件。

```text
  PlanRecorder              PlanCompiler               PlanRuntime
  ------------              ------------               -----------
  true top-K scan  ---> immutable logical IR ---> bind request layout
  query/content key         codec and delta       choose physical strategy
  calibration stats         cost estimation       reserve I/O and staging
                                                          |
                                                          v
                                                   TuttiDirectLoader
                                                   submit / CQ completion
                                                          |
                                                          v
  ResidencyTracker <--- generation-checked materialization / CUDA event
```

`PlanRecorder` 观察真实 indexer 已经在执行的工作，它不得对完整的
per-query top-K 张量增加第二次扫描。`PlanCompiler` 拥有逻辑计划，并把
它们转化为不可变的、绑定到请求的计划。`PlanRuntime` 负责执行、调度、
共享与取消。`ResidencyTracker` 拥有供 miss 校正使用的公开
resident/in-flight 视图。

## Plan IR（计划中间表示）

### 枚举

```python
class Correctness(Enum):
    EXACT_TRUE = auto()
    EXACT_REPLAY = auto()
    CERTIFIED_EXACT = auto()
    SPECULATIVE = auto()
    QUALITY_GUARDED = auto()


class LayerRole(Enum):
    INDEXER_K = auto()
    CSA_KV = auto()
    HCA_KV = auto()
    SWA_KV = auto()


class PlanRole(Enum):
    ANCHOR = auto()
    SHARED = auto()
    DELTA = auto()
    CORRECTION = auto()


class LogicalCodec(Enum):
    LIST = auto()
    BITMAP = auto()
    RANGES = auto()
    BULK = auto()
    XOR_DELTA = auto()


class PlacementKind(Enum):
    VLLM_ROWS = auto()
    BLOCK_SLOT = auto()
    INDEXER_POOL = auto()
    STAGING_ONLY = auto()
```

### 可复用的逻辑计划

```python
@dataclass(frozen=True, slots=True)
class ContentKey:
    schema: int
    model_fingerprint: bytes
    tp_world_size: int
    rank: int
    prefix_root: bytes
    prefix_blocks: int


@dataclass(frozen=True, slots=True)
class DecisionKey:
    content: ContentKey
    target_layer: int
    query_root: bytes
    query_start: int
    query_count: int
    indexer_config_hash: bytes


@dataclass(frozen=True, slots=True)
class Coverage:
    logical_limit: int
    selected_count: int
    density: float
    query_start: int
    query_count: int
    observed_queries: int
    recall_lcb: float | None
    read_amplification_bound: float


@dataclass(frozen=True, slots=True)
class Confidence:
    correctness: Correctness
    calibration_id: str | None
    sample_count: int
    row_recall_lcb: float | None
    union_recall_lcb: float | None
    quality_bound: float | None


@dataclass(frozen=True, slots=True)
class LogicalSelection:
    codec: LogicalCodec
    block_limit: int
    payload: bytes
    base_plan_id: bytes | None


@dataclass(frozen=True, slots=True)
class LayerPlan:
    plan_id: bytes
    decision_key: DecisionKey
    target_layer: int
    anchor_layer: int
    layer_role: LayerRole
    plan_role: PlanRole
    selection: LogicalSelection
    coverage: Coverage
    confidence: Confidence
    per_row_topk_ref: bytes | None
    deadline_offset_us: int
    max_read_bytes: int
    max_hbm_bytes: int
```

`plan_id` 是所有语义字段的规范序列化（canonical serialization）的哈希。
可复用计划中不得包含 vLLM 物理行、SSD 偏移、CUDA 事件、请求 ID 或可变的
驻留状态。

`per_row_topk_ref` 对 `EXACT_REPLAY` 是必需的。它指向一个单独进行校验和
（checksum）保护、并带形状描述的 top-K 载荷。仅含并集（union-only）的
计划必须保持该字段为空。

### 绑定到请求的物理计划

```python
@dataclass(frozen=True, slots=True)
class PhysicalRange:
    source_path: str
    source_offset: int
    dma_bytes: int
    payload_skip: int
    logical_bytes: int
    logical_blocks: tuple[int, ...]
    target_rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BoundPlan:
    logical: LayerPlan
    request_id: str
    source_layout_id: bytes
    layout_generation: int
    ranges: tuple[PhysicalRange, ...]
    physical_strategy: LogicalCodec
    deadline_ns: int
    placement: PlacementKind
    estimated_io_us: int
    estimated_materialize_us: int
```

编译器从不可变的 `RequestLayoutSnapshot` 创建 `BoundPlan`。
`layout_generation` 防止迟到的 scatter 写入 vLLM 已经重新分配给更新请求
的行。

### 示例

假设 query 行 0--1023 的真实 top-K 在第 18 层选中了压缩块
`{0, 1, 2, 5, 6}`。缓存表示可以是一个位图。对当前活跃请求而言，块 0--2
已经驻留，块 5--6 占据一段连续的源 extent，其目标行为 91 和 104。

```text
LogicalSelection: BITMAP {0,1,2,5,6}
Resident filter:  {5,6}
Physical strategy: RANGE
DMA: [source(block 5), source(end block 6))
Materialize: staging blocks [0,1] -> HMA rows [91,104]
```

可复用位图保持不变。只有绑定后的计划包含当前的物理行。

## 内容键与决策键

### Merkle 构造

每个有序的源 chunk 贡献一个叶节点：

```text
leaf = H(
    schema,
    CacheEngineKey logical content identity,
    logical_start,
    logical_end,
    bytes_per_block,
    layer-role layout schema
)
```

物理行、LBA 地址、池偏移和请求 ID 被排除在外。有序的叶节点构成
`prefix_root`。保留分段根（segment root），使 append 能以 `O(log n)`
更新，且无需重新哈希每个 chunk 即可找到最长公共前缀。

`model_fingerprint` 覆盖模型权重、稀疏 indexer 权重、布局版本、TP 大小
和压缩参数。`indexer_config_hash` 覆盖 top-K、RoPE/位置语义、压缩器配置
和算子版本。

### 源复用与决策复用的区别

两个缓存被有意分开：

```text
ContentKey  -> SourceMap / extents / 可复用的源执行
DecisionKey -> 逻辑选择以及可选的精确逐行 top-K
```

仅仅共享前缀并不能使新后缀的 top-K 相同。新后缀改变了 query 行，因而
改变了 `query_root`。这样的请求可以复用源 extent、在途（in-flight）
DMA，或者一个投机的 heavy-hitter 计划，但不能使用 `EXACT_REPLAY`。

精确决策命中要求相同的前缀、query 内容、query 位置、模型指纹、目标层与
indexer 配置。

### Delta 计划

Delta 计划相对于一个不可变的基（base）计划存储 XOR 位图或增/删范围。
编码器在以下几种表示中选择更小者：

```text
完整位图
排序后的新增与移除块 ID
XOR 位图，可选做游程编码（RLE）
```

每个存活的 delta 都强引用其基计划。在淘汰一个基计划之前，缓存必须先把
依赖它的 delta 物化，或将它们一并淘汰。Delta 编码是元数据与下译层面的
优化；仅凭一个 235 字节的位图不足以支撑一项系统性贡献的声明。

## Anchor 与共享层策略

### 校准表

离线校准以下列键索引：

```text
(模型指纹, anchor 层, 目标层,
 上下文长度桶, query 深度桶, 层角色)
```

每条记录包括样本数、逐行 recall 分布、并集 recall、Jaccard 相似度、
增/删块密度、质量实验 ID，以及置信下界（而非仅有均值）。

DSv4 的历史测量支持深度敏感的策略：浅层的 age-3 recall 降到 57--81%，
而更深的层保持 86--95%。因此，固定的全模型统一跨度（stride）是无效的。

### 运行时选择

```python
def choose_index_policy(ctx, anchor_layer, target_layer):
    if exact_decision_cache.contains(ctx.decision_key(target_layer)):
        return EXACT_REPLAY

    calibration = calibration_table.lookup(ctx, anchor_layer, target_layer)
    if calibration is None or calibration.samples < min_samples:
        return DISABLED
    if calibration.union_recall_lcb < prefetch_min_recall:
        return DISABLED

    if approximate_skip_enabled:
        if calibration.row_recall_lcb >= skip_min_recall:
            if quality_registry.accepts(calibration.quality_experiment_id):
                return QUALITY_GUARDED

    return SPECULATIVE
```

初始实现只支持 `DISABLED`、`SPECULATIVE` 与 `EXACT_REPLAY`。
`QUALITY_GUARDED` 只有在显式的质量评估之后才会加入，且默认关闭。

在投机模式下，anchor 提供候选读取，目标层的官方 indexer 保持权威。在
精确重放（exact replay）模式下，缓存的逐行 top-K 在校验其 checksum、
形状、query 跨度与键之后才会被返回。

近似模式需要周期性的影子真实 indexer 执行与漂移熔断器（drift circuit
breaker）。校准缺失、新的模型指纹、不支持的形状或超出误差阈值，都会
立即恢复官方 indexer。

## 自适应下译（Adaptive lowering）

### 候选策略

缓存的编码（codec）与物理策略是两回事。下译之前，编译器计算：

```text
missing = selected - resident - in_flight
```

它评估以下候选：

- `LIST`：极稀疏计划使用排序后的逻辑块 ID；
- `BITMAP`：GPU 位图直接压实（compact）成 ID，避免大规模 CPU 传输；
- `RANGES`：把相邻的源块合并为对齐的字节区间；
- `BULK`：完整的已注册 slab 或完整 chunk；
- `XOR_DELTA`：相对其基计划解码后，再用上述某种方式下译。

源侧连续的块即使目标行分散，也可以使用一次 DMA；随后由单个 GPU
materializer 完成放置。当源和目标都连续时，使用直接的连续拷贝。区间
绝不跨路径或跨不兼容的层角色盲目合并。

每次源读取都向 512 字节的 NVMe 边界取整。`payload_skip` 与
`logical_bytes` 保留精确的载荷。GPU staging 分配使用自己的页对齐。

### 代价模型

对候选 `p`：

```text
T_lower(p) = decode(codec)
           + compact(bitmap_blocks)
           + map(selected_blocks)
           + resolve(extents)

T_io(p) = NVMe_queue_wait
        + alpha_submit
        + beta_iocb * number_of_iocbs
        + dma_bytes / measured_NVMe_bandwidth

T_materialize(p) = SM_queue_wait
                 + alpha_launch
                 + beta_row * target_rows
                 + copied_bytes / measured_G2G_bandwidth
                 + phase_interference

T_finish(p) = now + T_lower(p) + T_io(p) + T_materialize(p)

Score(p) = T_finish(p)
         + lambda_waste * (dma_bytes - expected_consumed_bytes)
         + lambda_hbm * peak_staging_bytes
         + lambda_deadline * deadline_miss_probability
```

各常数由 rank 本地的指数加权测量更新。初始值来自实测的稀疏读关系
`T_load(N) = 0.42 + 0.021*N ms`、约 3.3 微秒/块的连续 I/O 开销，以及
9--11 GB/s 的聚合数据面带宽。

被选中的候选必须满足 deadline、staging、HBM、批量字节和 IOCB 预算。
10--15% 的滞回（hysteresis）防止在稀疏/整体读之间反复切换。

如果没有任何完整的投机计划能在截止时间前完成，编译器按"每 DMA 字节
预期避免的 gate 等待时间"对可取消的微批（microbatch）排序，只提交该
排序中负担得起的前缀。剩余的块留给精确的按需（demand）校正；投机绝不
为了自身完成而推迟 gate。

## 双资源调度

### 状态机

```text
                         cancel
                           |
                           v
CACHED -> BOUND -> ADMISSION_WAIT -> SUBMITTED -> CQ_READY
  |         |             |              |           |
  |         +-> CANCELLED <-+              +-> ORPHANED
  |                                                  |
  +-> EVICTED                              MATERIALIZE_WAIT
                                                     |
                                          +----------+---------+
                                          |                    |
                                          v                    v
                                      HMA_READY              FAILED
                                          |
                                          v
                                       CONSUMED
```

`CQ_READY` 表示 SSD 数据已到达 staging，并不表示目标 HMA 行已有效。
只有完成且通过 generation 校验的物化事件，才会把该请求的子任务转移到
`HMA_READY`。

### NVMe 准入（admission）

优先级顺序为：

```text
前台按需读取 > 冷数据落盘（cold store） > 投机读取
```

只有在没有更高优先级工作被宣告、令牌桶（token bucket）有余额，且
staging/IOCB 预留可用时，投机才提交下一个微批。初始上限保持为每个投机
批次 8 个 I/O 与 8 MiB。SQ 已提交的请求不可取消；只能移除尚未提交的
批次。

### GPU 物化

NVMe 与 GPU 调度使用分离的队列和额度（credits）。一个 CQ-ready 的结果
可以留在已注册的 staging 中，同时 GPU 正在运行 MoE、EP/NCCL、官方
indexer 或更紧急的 materializer。

物化按最早截止时间优先（EDF）顺序执行，并带一个守护量（guard）：

```text
guard = p99(物化耗时)
      + p99(materializer 队列延迟)
      + 时钟与 launch 裕量
```

在到达 guard 之前，请求可以在已知冲突的模型阶段推迟 scatter。到达
guard 时，它要么物化，要么上报 not-ready，让前台按需路径接管。调度器
使用已知的层/NVTX 阶段信息；它不得引入同步的在线 SM 利用率查询。

运行时暴露不透明的 `PlanTicket`/`PlanLease` 对象，提供公开的
`query_io_ready`、`query_hma_ready`、`wait_hma`、`promote` 与 `cancel`
操作。调用方不得窥探运行时队列。

## 并发、引用计数与失败语义

### 所有权层级

```text
不可变 LayerPlan
    -> 请求本地的 BoundPlan
        -> 源 DMA 共享的 PlanExecution
            -> 每个目标布局 generation 一个物化子任务
                -> 订阅者本地的 PlanLease
```

执行去重键为 `(plan_id, source_layout_id, rank)`。两个请求可以共享一次
写入 staging 的 DMA，但每个不同的目标行 generation 需要自己的物化
子任务。

以下对象都有显式引用计数：

- 缓存的计划及其 base/delta 依赖；
- 执行的订阅者；
- staging 缓冲区；
- 目标物化子任务；
- HMA 行驻留 epoch。

### 取消

- `BOUND` 或 `ADMISSION_WAIT`：移除该工作并释放预留。
- `SUBMITTED`：把该订阅者标记为孤儿（orphaned）。若无订阅者剩余，则在
  CQ 完成后立即释放 staging 结果。
- `CQ_READY` 或 `MATERIALIZE_WAIT`：只取消该目标子任务；共享的源执行
  为其他订阅者继续。
- `HMA_READY`：释放租约（lease）。行驻留仍由其 epoch 和行引用管理。

按需请求可以原子地提升（promote）一个已有的投机租约并继承已完成的
I/O。提升绝不能对同一逻辑块提交两次。

### 部分失败

- 键、schema、checksum、解码或下译失败：丢弃该计划，回退到官方
  indexer 与按需路径。
- 部分 NVMe 失败：只把成功完成的范围标记为可物化。失败的块保持
  missing，由按需路径重试。
- scatter 或 CUDA 事件失败：整个子任务保持非驻留状态。
- 超时：取消尚未提交的投机并回退；不会杀死共享的 worker 线程。
- 精确 top-K 缓存损坏：失败关闭（fail closed），改为运行官方 indexer。

### 必须保持的不变量

1. 一个块只有在其物化事件完成、且布局 generation 仍匹配时才算驻留。
2. `SPECULATIVE` 计划绝不改变官方 top-K 的返回值。
3. 只有 `EXACT_REPLAY` 或 `CERTIFIED_EXACT` 可以跳过官方 indexer。
4. 一个物理 `(generation, layer, block)` 至多有一个物化 owner；其他
   使用者订阅它的事件。
5. staging 在所有依赖它的 CUDA 事件完成之前保持被引用。
6. 已取消请求的迟到完成不能更新更新请求的行或驻留位图。
7. 错过 deadline 可以降低性能，但不能降低正确性。
8. 绝不在持有锁的情况下跨越 CUDA 同步、`Future.result` 或 I/O。

全局加锁顺序为：plan cache、execution table、request layout、layer
residency。

## 公开 API 草图

```python
class PlanCompiler:
    def record_true_topk(
        self,
        context: RequestPlanContext,
        layer_id: int,
        query_span: tuple[int, int],
        topk: torch.Tensor,
    ) -> LayerPlan: ...

    def lookup(self, key: DecisionKey) -> LayerPlan | None: ...

    def bind(
        self,
        plan: LayerPlan,
        layout: RequestLayoutSnapshot,
    ) -> BoundPlan: ...

    def lower(
        self,
        plan: BoundPlan,
        residency: ResidencySnapshot,
    ) -> LoweredPlan: ...


class PlanRuntime:
    def submit(self, plan: LoweredPlan, priority: Priority) -> PlanLease: ...
    def promote(self, lease: PlanLease, priority: Priority) -> None: ...
    def wait_hma(self, lease: PlanLease, timeout_s: float) -> Completion: ...
    def cancel(self, lease: PlanLease) -> None: ...
```

所有公开方法都有带类型标注的输入与返回值。任何模块都不得访问其他类的
下划线前缀字段。

## 与当前实现的映射

### `csa_block_plan_cache.py`

`CSABlockPlan` 变为 `LogicalSelection(BITMAP)` 的兼容包装。
`CSABlockPlanCache` 演化为不可变的、带 checksum 的 `DecisionKey` LRU，
并包含 base/delta 依赖记账。

当前的 `record` 方法把所有观察结果在同一个内容签名下做 OR。新缓存把
query 跨度/根哈希纳入决策键，或者显式记录一个聚合计划。否则，不相关的
query 观察会悄悄地夸大并集覆盖率。

新缓存还需要一个不刷新 LRU 的 `peek` 操作。"先检查、再查找"不能把 LRU
顺序更新两次。

### `CSAAttentionKVPrefetchManager`

`register_request_chunks` 创建一个不可变的 `RequestLayoutSnapshot`，
包含 Merkle 内容键与单调递增的布局 generation。后台工作绑定到该快照，
而不是读取可变的 `state.chunks`。

`_miss_ids_for_topk` 中现有的 GPU `seen` 位图就是 recorder 的输入。它
已经避免了对超大 top-K 张量的第二次扫描。编译器先记录 query 跨度与
决策键，然后正常的精确 miss-mask 计算继续进行。

`fire_cached_block_plan` 变为 `lookup -> bind -> lower -> submit`。
`_submit_reads` 与 `_issue_reads` 初期保持为适配器，随后移入
`PlanRuntime` 之后。HCA 的 block-slot scatter 变成
`PlacementKind.BLOCK_SLOT`，而不是编译器层面的特例。

当前的驻留位图仍是过渡性的底层存储，最终只通过公开的
`ResidencyTracker` API 暴露。

### `IndexerSSDManager`

`fire_stage0_for_layer` 调用编译器。Stage1 的 HCA 残差预测产生一个
`SPECULATIVE` 的 delta 候选。

现有的真实 top-K 记录/校正点发布 anchor 校准遥测与精确决策条目。第一版
实现绝不跳过目标 indexer。之后公开的 `select_index_policy` 返回
`DISABLED`、`SPECULATIVE`、`EXACT_REPLAY` 或 `QUALITY_GUARDED`。

Indexer K 使用与 attention KV 相同的 IR 与下译路径。这是"一个共享索引
同时消除 indexer 计算和 indexer-K SSD 字节"的必要条件。当前 Tutti 的
indexer 字节 API 包含一次 GPU 到 CPU 的中转（bounce）；原型阶段可以
保留它，但用于评估的路径必须直接消费 GPU 张量。

### `TuttiIndexerStorage` 与 `TuttiDirectLoader`

当前 `load_chunks_to_hbm` 的优先级、deadline、批量字节、批量 I/O 与
raw 回调参数，足以支撑到自适应下译原型阶段。

把 CQ 就绪与物化分离，需要一个非阻塞 API：

```python
def submit_ranges(
    ranges: Sequence[PhysicalRange],
    reservation: IoReservation,
    priority: Priority,
) -> IoTicket: ...

def wait_cq(ticket: IoTicket) -> CqResult: ...

def release(ticket: IoTicket) -> None: ...
```

`CqResult` 保留 staging 视图、逐范围状态和一个完成事件。materializer
不在 `on_raw_batch_loaded` 内部运行。完成与 scatter 不得持有唯一的
SQ/CQ I/O 锁。

## 实现阶段

每个阶段都可以独立度量、独立回退。

### P0：仅遥测

记录真实 top-K 的 query 跨度并集、层对（layer-pair）重叠、extent 数量、
密度、驻留状态与层 deadline。不增加任何新的读取或同步。

### P1：IR 与影子编译器

加入规范键、Merkle 构造、编码、plan cache 和 dry-run 下译。从现有的
真实 top-K 扫描中记录计划。旧的 Stage0 路径继续执行。

### P2：编译器等价性

把精确的内容计划预取路由到编译器。对每一层，其逻辑块集合必须与旧的
`fire_cached_block_plan` 结果相等。不启用自适应下译或索引跳过。

### P3：自适应下译

启用基于实测代价的 list/range/bulk 选择。继续使用当前阻塞式的 Tutti
回调与 scatter 实现。对比固定稀疏、固定整体读与自适应三种策略。

### P4：拆分运行时

引入 `IoTicket`、显式的 `CQ_READY`、EDF materializer、generation 校验、
租约、引用计数与取消。先在单请求上验证，再启用源执行共享。

### P5：共享 anchor 的投机预取

加载校准表，仅将选定的 anchor 层用于投机 I/O。目标层的官方 indexer
保持不变。

### P6：精确决策重放

先对小形状缓存逐行 top-K。只有在完整的决策键命中时才跳过官方
indexer。quality-guarded 复用保持在一个单独的、默认关闭的实验开关之后。

### P7：跨请求执行共享

加入 source-map 复用、共享在途 DMA、分段 delta 与面向特定目标的物化
子任务。

### P8：精确 query-tile DAG（研究分支）

流水线化官方真实 indexer 的 tile `i+1`、tile `i` 的 I/O 与 tile `i-1`
的注意力计算。它在不依赖预测的前提下解决整个 prefill 的并集饱和问题，
但会改变 vLLM/算子边界，并与先前的 microtask-DAG 工作强烈重叠。在离线
模拟器达到阈值之前，它保持为研究分支。

## 测试矩阵

### 单元与性质（property）测试

- 规范序列化与 `plan_id` 的稳定性；
- Merkle append、分段根、最长公共前缀与模型失效（invalidation）；
- list、bitmap、ranges、bulk 与 XOR-delta 的往返（round trip）；
- 存在存活 delta 时的 base 淘汰；
- 随机化的逻辑集合到物理区间的等价性；
- 512 字节对齐、payload skip 与 HCA block-slot 放置；
- 在模型、query、位置、形状或 checksum 不匹配时拒绝精确重放；
- 投机计划绝不改变返回的 top-K；
- 代价模型 oracle 与稀疏/整体读滞回；
- 假时钟（fake-clock）下的 EDF、优先级、令牌桶与 deadline guard；
- 从每个状态发起的取消以及按需提升（demand promotion）；
- 部分 CQ 失败与回退；
- 布局 generation 的 ABA 问题与迟到完成的拒绝；
- 共享执行的订阅者竞态；
- staging 保留到最后一个 CUDA 事件完成。

测试面向公开接口，不读取其他类的私有字段。

### 集成测试

| 场景 | 要求的结果 |
|---|---|
| 精确 Stage0 等价性 | 与旧路径相同的块集合和原始 top-K |
| 计划未命中 | 官方 indexer 与按需校正成功 |
| 投机部分完成 | 缺失的块由按需路径取回 |
| 相同前缀、不同后缀 | 允许源复用，拒绝精确决策复用 |
| 完全相同的请求 | 仅在存在逐行载荷时才允许精确重放 |
| 两个订阅者、一个取消 | 剩余订阅者正常完成 |
| I/O 途中行 generation 变化 | 丢弃迟到的 scatter |
| 投机期间到达按需请求 | 按需请求提升或抢占未提交的工作 |
| materializer 错过 deadline | gate 回退且不产生错误驻留 |
| 精确载荷损坏 | 运行官方 indexer |

## 离线证伪（falsification）工作流

### 输入

使用 Parquet 或 NPZ，包含：

- run、请求、模型、rank、层、上下文长度、query 起点与 query 数量；
- 逐行真实 top-K 的压缩块 ID；
- 可用时的 anchor/残差预测 top-K；
- chunk 逻辑范围、每块字节数、源路径/extent 与目标角色；
- resident 与 in-flight 时间线；
- indexer、注意力、MoE/NCCL、I/O 提交/CQ、scatter 与 gate 的 NVTX
  计时；
- 用于近似跳过实验的质量运行（quality-run）ID。

### 输出

```text
pair_calibration.json
    每个层对/桶：recall 置信下界、Jaccard、delta 密度、样本数

lowering_oracle.parquet
    每个候选：字节数、IOCB 数、下译/IO/物化耗时、浪费量、完成时间

tile_sim.parquet
    tile 大小 x HBM 预算：并集、重读字节、预测关键路径

scheduler_replay.json
    eager 与 EDF 对比：gate p50/p99、kernel 拉长、staging 峰值

reuse_report.json
    精确决策、source-map、在途共享的命中率与 delta 大小
```

### 成功阈值

- 跨层预取：覆盖至少 50% 的目标 indexer 调用，并集 recall 下界至少
  90%，读放大不超过 1.25。
- 近似索引跳过：单独的端到端质量评估，无统计显著的回归；仅有 recall
  是不够的。
- 自适应下译：预测完成时间比最优固定策略至少低 10%，或者 IOCB 至少减少
  2 倍且字节数不超过 1.10 倍。
- 调度器：gate p99 至少降低 20%，MoE、NCCL 或 indexer kernel 拉长不
  超过 3%，且 staging 在显式预算之内。
- 精确重放：在目标负载中决策键命中率至少 20%，才可作为主要结果。
- Query-tile DAG：在 HBM 预算内字节数不超过 bulk 的 1.25 倍，且预测
  关键路径至少缩短 20%。

阈值不达标会收窄论文声明。例如，跨层 recall 不佳时，保留自适应物理
下译，但从论文声明中移除通用的索引复用。

## 评估矩阵

至少评估两个原生稀疏注意力模型。DSv4 必须包含其 21 个真实 CSA indexer
与异构的 HCA/SWA 角色；另一个模型必须使用其真实的层映射，而非合成的
全 CSA 假设。

必需的负载包括：短的健全性（sanity）用例、480K 加 16K/32K/48K 的长
prefill、若干新前缀、精确重复、延迟的同前缀命中，以及并发的不同前缀
请求。报告 p50、p95、p99、样本数、离散度与失败情况。

指标包括 TTFT、goodput、gate 停顿、indexer 调用次数与 GPU 时间、逻辑与
物理 SSD 字节、IOCB、读放大、计划/下译的 CPU 时间、NVMe 时间、物化
时间、SM 干扰、staging/HBM 峰值，以及浪费的投机字节。仅有 warm repeat
不能作为首次命中（first-hit）收益的证据。

## OSDI 论文声明与相关工作边界

### 拟议声明

> 一个带正确性标签的稀疏索引可以只编译一次，同时消除冗余的索引计算，
> 并把同一决策下译为受 deadline 约束的 SSD 区间与异构的最终 KV 放置。
> 把 NVMe 完成与 SM 感知的物化分离，可以在不使投机预测成为权威的前提
> 下，把存储稀疏性转化为首次命中的 TTFT 收益。

论文应呈现一条链，而不是五个互不相关的特性：

```text
真实的或带守护的共享索引
       -> Plan IR
       -> 自适应物理下译
       -> 双资源 deadline 执行
```

Plan cache、delta 编码、materializer 与调度器是这条链的必要组成部分，
而不是各自独立的主打贡献。

### 相关工作风险

- **IndexCache** 已经覆盖跨层 top-K 复用。区分点必须是：同一个索引还
  消除了 indexer-K/attention-KV 的 SSD 字节，并被下译为物理放置与
  deadline。
- **ECHO** 公开声称无损预测与融合的 indexer/召回。本工作不应声称首创
  预测机制。其正式发表后需要重新做相关工作审计；存储物理计划的声明
  必须经得起该对比。
- **Strata** 覆盖层级缓存与 delay-hit 调度。区分点是共享稀疏决策与
  物理执行，并改善新前缀的首次命中率，而非只合并并发加载。
- **SolidAttention** 包含历史预测、工作集控制与 microtask DAG。P8 的
  重叠风险最高。要与之区分，需要精确的原生学习型 indexer 执行、异构
  布局与长上下文 SSD 证据。
- **Swarm** 用共激活（co-activation）做放置。放置本身不是本文的主要
  贡献；它是由共享稀疏计划驱动的一个可选下译决策。
- **Tutti** 已经提供 GPU 原生对象 I/O 与 slack 感知机制。仅仅把当前的
  indexer 后端接到 Tutti 上是必要的工程工作，不构成论文的新颖性。

### 主要风险

1. 深层 recall 可能足以支持预取，但不足以支持索引跳过。
2. 整个 prefill 的并集饱和可能使 oracle 几乎处处选择 bulk，缩小算法
   贡献。
3. Host 侧物化开销可能掩盖编译器改进，直到 CQ 与 HMA 状态被分离。
4. 后缀多样时，精确决策键命中率可能偏低。
5. 机制清单过宽可能掩盖唯一的"计算到存储编译器"这一核心贡献。

以上风险正是 P0/P1 离线证伪要先于侵入式运行时或算子改动的原因。
