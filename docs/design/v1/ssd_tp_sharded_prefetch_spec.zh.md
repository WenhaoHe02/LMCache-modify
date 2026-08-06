# SSD 分片读取与 NVLink 汇聚需求规约

状态：Draft v0.1
日期：2026-07-28
适用范围：当前 8-rank CSA/HCA SSD layer-major 恢复与 incremental
prefill 路径

## 1. 背景与结论

当前 SSD 路径的主要问题不是 SSD 带宽不可预测，而是多个 rank 会从各自的
本地 SSD 重复读取可以按 context block 分工的数据。在 480K+8K 稳态请求中，
每个 rank 实际读取约 1.6749 GiB，8 rank 合计约 13.40 GiB；按当前约
11.8 GB/s 的 GPUDirect SSD 带宽计算，对应约 149--152 ms/rank 的累计 SSD
服务时间。

本设计只优化 SSD 路径，实施以下两项改动：

1. Indexer-K 只从 SSD 读取当前 CP rank 负责计算的 1/8 context，不做
   NVLink 汇聚。
2. CSA attention KV 的全局 block union 在 8 个 rank 间确定性分片。每个
   rank 从本地 SSD 读取约 1/8 blocks，再通过 NVLink all-gather 恢复每个
   rank 消费 attention 所需的完整 KV。

早层缺少可靠预测时使用 dense shard-gather；深层继续使用预测结果，但对
predicted union 做分片读取。小规模读取保留当前 layer-wise 本地直接读取，
避免 collective 的固定延迟。

第一版不修改 DRAM 路径、不修改模型数学、不修改 SSD 对象格式、不分片 HCA，
也不捆绑 vLLM 0.26 升级。旧的本地读取路径必须完整保留为 fallback 和运行时
kill switch。

## 2. 目标与非目标

### 2.1 目标

- 消除 Indexer-K 和可复制 CSA KV 的 8-rank 重复 SSD 读取。
- 让 2--24 层不再依赖低精度 proxy，也能提前读取完整所需 KV。
- 在 overlap 窗口不足甚至为零时，对大 union 仍然比完整本地 SSD 读取更快。
- 根据每种算子的实测时间、资源占用、SSD 队列和 rank skew 自动选择读取模式。
- 减少 SSD 带宽占用，提高多请求和多租户下的吞吐容量。
- 保持已有模型结果、block union、缓存布局和命中语义不变。
- 将通信和 vLLM 生命周期集成隔离在稳定接口后，避免阻碍后续迁移到
  vLLM 0.26。

### 2.2 非目标

- 不把本次优化扩展为 DRAM/CPU H2D 路径重构。
- 不更改 Indexer-K 的 132 B/entry 表示格式。
- 不对 HCA KV 做语义压缩或窗口化持久化。
- 不改变 top-k、CSA/HCA 选择算法或模型精度。
- 第一版不尝试让预取 all-gather 与模型 TP AllReduce 并发。
- 第一版不重写已有 layer-major SSD 对象。

## 3. 当前 SSD 基线

基线来自当前 480K+8K HCA-L2 稳态 trace。此前把 Indexer-K 误按
584 B/entry 计算、并把 HCA 解释为两个完整 context 子流的统计是错误的：

- Indexer-K 实际为 132 B/entry。
- CSA KV 为 584 B/entry。
- HCA 的 `shape=(2, 584)` 表示一个 256-token source block 内包含两个
  C128 compressed entries，不是两个完整 context 数据流。

### 3.1 当前读取量

| 数据组 | 当前每 rank | 依据 | 备注 |
|---|---:|---|---|
| Indexer-K | 332,640,000 B = 0.3098 GiB | 21 层 × 1875 blocks × 8448 B | 完整 context 重复读取 |
| CSA KV | 1,406,571,008 B = 1.3100 GiB | 21 层共 37,633 个实际 union blocks × 37,376 B | 相比完整读取只少约 4.4% |
| HCA KV | 43,800,000 B = 0.0408 GiB | 20 层 × 3750 entries × 584 B | 每层约 2.19 MB |
| tail/state | 15,420,416 B = 0.0144 GiB | 当前固定附加数据 | decoder capture 外加载 |
| **合计** | **1,798,431,424 B = 1.6749 GiB** | | |

8 rank 的系统总读取量为：

\[
8\times1.6749\ \mathrm{GiB}=13.40\ \mathrm{GiB/request}.
\]

当前 SSD 服务时间与字节量一致：

\[
\frac{1.798\ \mathrm{GB}}{11.8\ \mathrm{GB/s}}
\approx 152\ \mathrm{ms/rank}.
\]

trace 中约 149 ms/rank 的累计读取时间因此是合理的。TTFT 不能直接减去
149 ms，因为其中一部分已经与计算重叠；当前审计估计约 82 ms/rank 真正暴露
在关键路径上。

### 3.2 新方案目标读取量

| 数据组 | 当前每 rank | 第一版每 rank | 处理方式 |
|---|---:|---:|---|
| Indexer-K | 0.3098 GiB | 0.0387 GiB | CP-local 1/8 SSD 读取 |
| CSA KV | 1.3100 GiB | 0.1638 GiB | 1/8 SSD 读取 + NVLink gather |
| HCA KV | 0.0408 GiB | 0.0408 GiB | 保持本地读取 |
| tail/state | 0.0144 GiB | 0.0144 GiB | 保持本地读取 |
| **合计** | **1.6749 GiB** | **0.2576 GiB** | **减少 84.6%** |

第一版实施 Indexer 和 CSA 分片、HCA 保持本地读取后：

- 每 rank SSD 读取约 0.2576 GiB。
- 全机 SSD 读取约 2.06 GiB/request。
- SSD 总字节量减少约 6.5 倍。

如果后续证明 HCA 多层批量汇聚有收益，理论上还可降至约
0.2219 GiB/rank、1.78 GiB/request；这不属于第一版目标。

## 4. 功能需求

### 4.1 Indexer-K：CP-local SSD 读取

当前单层 Indexer-K 数据量为：

\[
1875\times(64\times132)=15.84\ \mathrm{MB/rank}.
\]

CP8 分片后，每个 rank 每层只需读取约 1.98 MB。

实现必须满足：

1. 根据实际 CP interleave 和 ownership 映射计算本 rank 拥有的 context rows。
2. 从现有 layer-major 对象生成对应 SGL，第一版不修改磁盘格式。
3. 合并物理相邻的读取范围，避免 1/8 数据产生大量小 I/O。
4. 本路径不进行 NVLink gather；每个 rank 的 `mqa_logits` 只消费自己负责的
   rows。
5. 如果运行配置、kernel 行为或 layout 不满足 CP-local 消费条件，必须在提交
   SSD I/O 前回退到完整本地读取。
6. 必须记录计划读取 rows、实际读取 bytes 和 kernel 访问范围，能够发现越界或
   漏读。

### 4.2 CSA attention KV：SSD 分片读取和 NVLink 汇聚

CSA 处理流程如下：

```text
rank-local top-k
    -> 现有 ID all-gather
    -> 全局去重、确定性排序的 block union
    -> 按 union 顺序分成 P 份
    -> 每个 rank 从本地 SSD 读取自己的 1/P blocks
    -> NVLink all-gather
    -> 根据 owner/block mapping scatter 到本地 HMA
    -> 目标 attention layer 消费
```

在 480K context 下，一个完整 CSA 层包含：

- 1875 blocks。
- 每个 block 为 `64 × 584 = 37,376 B`。
- 完整层为 70.08 MB。
- 8-way 分片后每 rank 最多 235 blocks，即约 8.78 MB。
- ring all-gather 中每 rank 需要从其他 rank 接收的有效数据约 61.3 MB。

实现必须满足：

1. 所有 rank 对 union 使用完全相同的排序、padding 和 owner 映射。
2. 默认 owner 由 union 中的稳定序号决定，不使用 Python hash 或进程局部状态。
3. 各 rank 的 block 数量差不得超过 1。
4. collective 使用固定大小或显式 padding 的 buffer；实际 block count 作为独立
   metadata 传递。
5. rank-major gather 结果必须能通过相同 owner mapping 恢复到 union 顺序。
6. 每层 gather 后的完整数据不超过现有 128 MiB staging slot。
7. 至少使用双缓冲，使第 L 层消费时能够准备第 L+1 或 L+2 层。
8. 热路径不得动态分配大块 GPU buffer。
9. predicted blocks 和后续 miss correction 共享同一 loaded bitmap，禁止重复读取。
10. 较小 residual miss 默认使用本地直接读取，避免第二次 collective；只有成本
    模型证明 residual 足够大时才可再次 shard-gather。

### 4.3 早层 dense shard-gather

当前 2--24 层没有可靠的预测来源，旧方案会在层边界同步读取接近完整的 KV，
形成约 88 ms 的早层暴露 I/O。第一版对这部分使用
`SHARD_GATHER_DENSE`：

1. 不执行额外 proxy 预测。
2. 直接把完整 1875-block 层按 8 rank 分片。
3. 每 rank 从 SSD 读取约 1/8 层，再通过 NVLink 汇聚。
4. 从目标层之前 1--2 层的可用 hook 发起，实际提前距离由成本模型决定。
5. 目标层只等待 layer-ready CUDA event，不等待生命周期型 host range。

### 4.4 深层 predicted shard-gather

对 26--42 层保留现有预测来源，但把 predicted union 的数据面读取改为
`SHARD_GATHER_PREDICTED`：

1. rank-local top-k 仍按当前方式计算。
2. ID exchange 后得到全 rank 一致的去重 union。
3. 对 union 做 byte-balanced 分片读取和 gather。
4. 真正消费时只修正 predicted union 未覆盖的 blocks。
5. 当 query 足够大、union 已接近完整层时，成本模型可以直接选择 dense
   shard-gather，避免维护稀疏计划却几乎不节省字节。

### 4.5 小 union 和正常 layer-wise fallback

当 union 很小时，SSD 完整读取时间可能低于 collective 固定延迟。因此必须保留：

- `LOCAL_DIRECT`：当前 rank 直接读取本地所需 blocks。
- `CP_LOCAL_INDEXER`：Indexer 的 CP-local 读取。
- `SHARD_GATHER_DENSE`：完整层分片汇聚。
- `SHARD_GATHER_PREDICTED`：预测 union 分片汇聚。

请求热路径通过预先生成的 decision table 选择模式，不进行复杂求解。

### 4.6 HCA 第一版策略

HCA 第一版保持当前本地确定性读取：

- 每层只有约 2.19 MB。
- 8-way 分片后每 rank 仅约 0.27 MB。
- 节省的 SSD 时间可能小于 collective launch、rank 同步和 gather 固定开销。

只有在多层 HCA 合并成一次 gather 后，实测 p90 仍稳定快于本地读取，才允许在
后续版本启用 HCA 分片。

## 5. 硬性正确性前提

任何代码切换到分片路径前，都必须验证以下条件。

### 5.1 CSA block 可复制性

对于代表性的请求、layer 和 block，比较冷写入后以及重新加载后的 8-rank
字节内容。同一个 CSA 逻辑 block 必须在所有候选 source rank 上完全相同。

如果数据实际是 attention-head/rank-specific 表示，则其他 rank 不能从某个
owner rank 的 SSD 数据恢复自己的 KV，该 group 必须继续使用本地读取。

### 5.2 Indexer CP-local 消费范围

必须通过 instrumentation 或 kernel 输入范围证明：每个 rank 的
`mqa_logits` 只访问本 CP rank 拥有的 context rows。仅仅证明计算总量约为
1/8 不足以证明能够少读，必须验证具体 row mapping。

### 5.3 全 rank union 一致性

完成 ID exchange 和去重后，每个 rank 必须得到相同的：

- union 长度；
- 有序 block IDs；
- union hash；
- 读取模式；
- padding 大小；
- collective sequence number。

任何不一致必须在 collective 提交前触发一致的 fallback 或请求失败，禁止部分
rank 进入 gather、部分 rank 进入本地读取。

### 5.4 按 group 独立验证

Indexer、CSA 和 HCA 的复制与消费性质必须分别验证。一个 group 验证成功不能
作为其他 group 的正确性依据。

## 6. 按算子和资源建模

### 6.1 建模原则

单一的“从发起到消费的墙钟窗口”不足以判断 overlap。一个 I/O 可以在消费前
完成，但期间 GPU 完全空闲；这种情况满足 deadline，却没有降低 TTFT。

成本模型必须分别追踪：

- CPU prepare；
- SSD submit、queue 和 poll；
- NVLink/NCCL；
- materialize/scatter 使用的 SM 和 HBM；
- 模型 compute stream；
- rank 间最慢值和 skew。

对于目标 layer L、提前距离 k 和资源 r，定义：

\[
W_r(L,k)=
\sum_{o\in\mathcal{O}(L,k)}
\alpha_{o,r}C_o(s,q)
-Q_r-\delta_{\mathrm{rank\ skew}}.
\]

其中：

- \(C_o(s,q)\) 是算子 o 在 context 长度 s、incremental query 长度 q 下的
  实测时间。
- \(\alpha_{o,r}\) 表示算子 o 的执行时间有多少能够遮住资源 r 的工作。
- \(Q_r\) 是 SSD 或通信资源上的已有排队时间。
- \(\delta_{\mathrm{rank\ skew}}\) 是根据 max-rank 而不是平均 rank 计算的
  安全余量。

### 6.2 第一版资源兼容矩阵

| 当前算子 | SSD 读取 | NVLink gather | materialize/scatter |
|---|---:|---:|---:|
| `mqa_logits` | 可重叠 | 可重叠 | 默认不重叠 |
| `sparse_attn` | 可重叠 | 测量后决定 | 默认不重叠 |
| MoE/Marlin | 可重叠 | 可重叠 | 测量后决定 |
| TopK | 可重叠 | 可重叠 | 默认不重叠 |
| 模型 NCCL | 可重叠 SSD | 第一版禁止同时 gather | 禁止 |
| CPU prepare | 单独 CPU 时间线 | 不适用 | 不适用 |

这里的“可重叠”只是调度许可，不代表免费重叠。任何造成算子 p95 明显上升的资源
竞争都必须反映为 `interference` 项。

### 6.3 SSD 时间模型

稳态 SSD 时间采用带队列深度的线性模型：

\[
T_{\mathrm{SSD}}(m,p,QD)=
t_{\mathrm{submit}}+t_{\mathrm{poll}}
+\left\lceil\frac{m}{p}\right\rceil
\frac{b}{B_{\mathrm{SSD}}(QD)}.
\]

其中：

- m 是 union blocks 数量。
- p 是参与分片的 rank 数，当前为 8。
- b 是每个 block 的实际字节数。
- `B_SSD(QD)` 必须按当前并发 batch/SQ 深度校准。

当前整层约 1800 blocks 的 poll 时间 p10--max 约 4.99--5.75 ms，变异系数
约 5.8%，因此稳态 I/O 是可预测的。模型使用 p90 参数并额外增加 10% 安全
余量。首次 session bind、FIEMAP、allocator/sidecar readiness 等冷启动事件属于
另一分布，不能混入稳态决策模型。

按当前近似参数：

\[
T_{\mathrm{IO}}(m)\approx t_0+m\tau_b,
\quad t_0\approx0.5\ \mathrm{ms},
\quad \tau_b\approx3.2\ \mu\mathrm{s/block}.
\]

完整 CSA 层约为 5.9--7 ms；235-block shard 预计约为 1.25 ms。

### 6.4 NVLink all-gather 模型

\[
T_{\mathrm{AG}}(m,p)=
t_{\mathrm{ag0}}+
\frac{p-1}{p}\frac{mb}{B_{\mathrm{NVLink,eff}}}
+T_{\mathrm{sync}}.
\]

完整 CSA 层的初始预算为约 0.3--0.8 ms，但这只是待验证估计。正式决策必须
使用当前拓扑、独立 communicator、目标 stream 和真实模型负载下测得的 p90。

第一版把模型 NCCL 区间设置为 `alpha=0`，即不假设 all-gather 能与模型
AllReduce 免费并发。只有 trace 证明不会显著增加 AllReduce 或 gather p95 后，
后续版本才可引入部分重叠系数。

### 6.5 本地读取与分片读取比较

本地读取：

\[
T_{\mathrm{local}}=
T_{\mathrm{prepare}}+T_{\mathrm{SSD}}(m,1)
+T_{\mathrm{materialize}}.
\]

分片读取：

\[
T_{\mathrm{shard}}=
T_{\mathrm{prepare}}+T_{\mathrm{SSD}}(m,p)
+T_{\mathrm{AG}}(m,p)+T_{\mathrm{materialize}}
+T_{\mathrm{interference}}.
\]

最基本的分片启用条件为：

\[
T_{\mathrm{shard,p90}}+\mathrm{margin}
<T_{\mathrm{local,p90}}.
\]

等价地，节省的 7/8 SSD 传输必须大于 collective 固定成本、NVLink 传输、
rank 同步和额外干扰：

\[
\left(1-\frac1p\right)\frac{mb}{B_{\mathrm{SSD}}}
>
t_{\mathrm{ag0}}+
\frac{p-1}{p}\frac{mb}{B_{\mathrm{NVLink,eff}}}
+T_{\mathrm{sync}}+T_{\mathrm{interference}}.
\]

最终需要比较进入 TTFT 的暴露时间：

\[
E_{\mathrm{mode}}=
\max(0,T_{\mathrm{ready,mode}}-W_{\mathrm{effective}})
+T_{\mathrm{interference}}.
\]

实现应使用按阶段的 resource timeline 模拟，而不是假设 prepare、SSD、gather 和
scatter 全部能够被同一个标量窗口遮住。

### 6.6 决策表和在线校准

- 初始化时按 group、layer、context bucket、query bucket 和 union bucket 生成
  decision table。
- 请求热路径只执行查表和边界检查。
- 稳态样本通过低开销 EWMA/分位数统计更新 p50、p90 和 max-rank skew。
- 参数更新必须有上下限和迟滞，禁止单个异常请求使全部层模式震荡。
- 冷启动样本单独记录，不参与稳态表更新。
- 选择模式时使用 p90 加 10% margin，不使用平均 rank 或平均耗时。

## 7. 无法 overlap 时的行为

对于大 union，分片读取的收益不依赖 overlap 才成立。

以完整 CSA 层为例：

- 完整本地 SSD 读取约 5.9--7 ms。
- 分片 SSD 读取约 1.25 ms。
- NVLink gather 初始预算约 0.3--0.8 ms。
- materialize 是两种路径大体共有的成本。

即使 gather 完全没有被模型计算遮住，分片路径仍预计比完整本地读取快约
3.5--5 ms/层。因此，新的大 union 路径既减少服务时间，也增加可 overlap 的
余量。

可能反而变慢的情况包括：

- union 很小，原本本地 SSD 读取低于约 1 ms；
- collective 固定延迟高于节省的 SSD 时间；
- 一个 rank 的 SSD 长尾通过 collective 传染给所有 rank；
- gather 与模型 AllReduce 争用 NVLink/NCCL；
- CP interleave 被编译成大量无法合并的小 SGL；
- block 数据并非跨 rank 可复制；
- staging、prepare 或 scatter 引入新的串行依赖。

这些情况必须由正确性 gate、p90 成本模型和 `LOCAL_DIRECT` fallback 处理。

## 8. 理论性能提升

### 8.1 SSD 服务时间

当前：

- 每 rank 约 1.6749 GiB。
- 原始传输和服务时间约 149--152 ms/rank。

第一版：

- 每 rank 约 0.2576 GiB。
- 单纯按字节计算约 23--24 ms/rank。
- 计入每层 submit/poll、HCA 本地读取、gather、scatter 和固定延迟后，预计
  约 40--55 ms/rank。

因此预计减少约 95--110 ms/rank 的累计服务时间，并减少 84.6% 的 SSD
带宽需求。

### 8.2 TTFT 上限

当前实测稳态 TTFT 约为 1.474 s，审计得到的真实暴露 SSD 关键路径约为
82 ms/rank。只有暴露部分能够直接转化为 TTFT 改善，因此本优化的绝对理论
上限为：

\[
1.474\ \mathrm{s}-0.082\ \mathrm{s}
=1.392\ \mathrm{s}.
\]

考虑 collective 同步、max-rank SSD 尾部、materialize 和模型 NCCL 调度，第一版
工程目标为：

| 指标 | 目标或预测 |
|---|---:|
| SSD 总字节 | 减少 84.6% |
| SSD 累计服务时间 | 149--152 ms → 40--55 ms |
| TTFT 绝对理论上限 | 最多回收约 82 ms |
| 理想 TTFT | 约 1.392 s |
| 第一版工程目标 | 1.40--1.43 s |
| 预计 TTFT 收益 | 40--70 ms |

不能宣称 149 ms 都能从 TTFT 中删除，因为其中已有一部分被计算遮住。即使
TTFT 收益小于字节收益，6.5 倍 SSD 流量降低仍会直接改善多请求并发和多租户
带宽容量。

## 9. 通信、流和内存规约

### 9.1 Communicator

- 使用独立于模型 TP group 的 NCCL communicator。
- communicator 在 engine 启动阶段创建和预热，禁止请求热路径初始化。
- 所有 rank 必须以相同顺序提交 `(request_generation, layer_id, phase)`。
- 第一版不得与模型 TP AllReduce 并发；通过已有 layer hook 和 CUDA event 安排
  到 compute-only 窗口。
- 独立 communicator 不能被当作没有 fabric 竞争；仍需测量模型 NCCL p95。

### 9.2 CUDA stream 和事件

- SSD、gather 和 materialize 使用职责明确的非默认 stream。
- 阶段间只通过 CUDA event 建立依赖。
- 禁止在热路径调用 `cudaDeviceSynchronize()`。
- 目标 layer 只等待对应 `layer_ready_event`。
- lifecycle range 不能被用作 I/O 时长或 ready 时间。

### 9.3 Staging

- 一个完整 CSA layer 的 gathered data 约 70.08 MB，必须适配现有 128 MiB
  slot。
- 使用至少两个独立 slot 组成有界 ring，不在单个 slot 内堆叠双层数据。
- 明确最大 inflight layer 数、最大 send bytes 和最大 gathered bytes。
- slot 只有在 SSD、gather、scatter 和消费者引用全部结束后才能复用。
- 不允许为了提前更多层而无界增加 staging。

## 10. 错误处理与 fallback

### 10.1 Collective 提交前

以下情况必须在所有 rank 上一致地切换到 `LOCAL_DIRECT`：

- replica/capability gate 未通过；
- union hash、长度、mode 或 sequence 不一致；
- union 小于成本模型阈值；
- staging capacity 不足；
- 当前 communicator 已标记 unhealthy；
- 请求或 layer 不支持当前 layout。

如需 consensus，必须在进入数据 all-gather 前完成，不能由 rank-local 条件直接
跳过 collective。

### 10.2 Collective 提交后

collective 一旦提交，不允许部分 rank 取消并本地重读。NCCL 错误可能使
communicator 失效，因此：

- 当前请求按现有分布式错误语义失败。
- 独立 prefetch communicator 标记 unhealthy。
- 后续请求关闭 shard-gather，使用旧本地路径。
- 是否能够在不重启进程的情况下重建 communicator 必须单独验证，不能作为第一版
  正确性前提。

### 10.3 Kill switch

必须提供：

- 全局关闭 SSD 分片读取；
- 只关闭 Indexer 分片；
- 只关闭 CSA shard-gather；
- 按 group 或 layer range 关闭；
- debug checksum/union verification 开关。

配置应集中在 LMCache 配置对象中，环境变量只能作为入口，禁止在多个 manager
中散落独立开关。

## 11. 可观测性规约

每个 request、layer 和 rank 必须记录：

- `mode`；
- `union_blocks`；
- `owned_blocks`；
- `ssd_bytes`；
- `gather_send_bytes` 和 `gather_total_bytes`；
- CPU prepare 时间；
- SSD submit、queue、poll 和完成时间；
- gather 排队与执行时间；
- scatter/materialize 时间；
- issue 到 ready 时间；
- consumer deadline；
- deadline slack；
- residual miss blocks 和 bytes；
- decision table 的预测值和实际误差。

报表必须同时给出 per-rank 分布和 max-rank，不能只报告均值。

Overlap 分析器必须从所有实际 SSD submit/poll 和 compute-stream 空闲区间出发，
不能再以 `predicted_l1/predicted_l2` NVTX range 为唯一采样入口。没有 prediction
的 layer 也必须被统计。`io_in_flight` 等 lifecycle range 不能求和当作 I/O
服务时间。

## 12. 代码边界

预计修改范围如下，最终以现有公共接口审计为准：

- `lmcache/v1/indexer_ssd_manager.py`
  - CP ownership 到 SSD SGL 的编译；
  - Indexer 模式选择和统计。
- `lmcache/v1/csa_attention_kv_prefetch_manager.py`
  - union 分片计划；
  - gather/scatter pipeline；
  - ready event、loaded bitmap 和 residual correction。
- `lmcache/v1/cache_engine.py`
  - 暴露按逻辑 block/range 读取的公共接口；
  - capability 和配置生命周期。
- `lmcache/v1/gpu_connector/tutti_direct_loader.py`
  - 有界 SGL batch 提交；
  - 实际 SSD byte 和阶段时间统计。
- `lmcache/integration/vllm/vllm_v1_adapter.py`
  - 初始化独立 communicator/stream；
  - 在稳定 layer 生命周期 hook 上发起和等待；
  - 避免依赖 vLLM 私有字段。
- `scripts/analyze_csa_nsys_overlap.py`
  - 修复无 prediction layer 的 I/O 统计盲区；
  - 报告真实 compute/SSD/NVLink 区间交集。

新增公共 API 必须最小化并带完整类型标注和 docstring。不同 manager 之间不得访问
对方的私有成员。vLLM 相关实现应隐藏在 adapter/transport 接口后，后续迁移到
vLLM 0.26 时只替换生命周期接入层，不重写 SSD 分片算法。

## 13. 测试与验收标准

### 13.1 单元测试

- deterministic partition：空 union、1 block、少于 rank 数、非整除、完整
  1875 blocks。
- duplicate IDs 去重和稳定排序。
- rank ownership 和 SGL coalescing。
- padding、rank-major gather layout 和 inverse mapping。
- loaded bitmap 与 residual miss 去重。
- decision table 的边界和迟滞。
- capability gate、fallback 和 unhealthy communicator 状态。

测试公共接口和文档契约，避免依赖其他类的私有成员。

### 13.2 正确性集成测试

- 固定 seed 下生成输出与旧路径一致。
- top-k、去重 union 和最终 loaded block set 与旧路径一致。
- debug 模式下 gathered destination checksum 一致。
- 验证没有未初始化读取和 slot 提前复用。
- 覆盖 32K、128K、256K、480K context。
- 覆盖 2K 和 8K suffix。
- 覆盖 rank 数 1、2、4、8。
- 覆盖 cold store、首次 hit、稳态 hit 和 residual miss。
- 连续至少 1000 次 hit，无 GPU Xid、NCCL timeout、deadlock 或数据错误。

### 13.3 性能验收

在与当前 1.474 s 基线相同的 480K+8K 稳态配置下：

- 每 rank SSD bytes 不超过 0.28 GiB。
- 8 rank SSD bytes 不超过 2.2 GiB/request。
- rank 间 owned block 数差异不超过 1。
- 被 decision table 选择的 shard-gather 层，其端到端 p90 必须比
  `LOCAL_DIRECT` p90 至少快 1 ms。
- 模型 AllReduce、`mqa_logits` 和 `sparse_attn` p95 不得恶化超过 2%。
- 第一版 TTFT 首要目标不高于 1.43 s。
- 进阶目标不高于 1.41 s，理论极限约 1.392 s。
- TTFT p95 不得劣于当前稳定基线。
- 小 context、decode 和小 union fallback 的性能恶化不得超过 2%。
- 实际 SSD bytes 与 block 几何模型的偏差不得超过 5%。

### 13.4 Overlap 验收

“在消费前完成”不能作为完全 overlap 的判据。必须同时报告：

- SSD/NVLink 服务区间与独立 GPU compute 区间的真实交集；
- consumer 前的 ready slack；
- compute stream 在 I/O 生命周期内的空闲时间；
- 每层和每 rank 的 exposed SSD/NVLink wall time；
- 8 rank 同步后的 max-rank critical path。

只有 exposed wall time 接近零且模型算子没有等量回退，才能称为完全 overlap。

## 14. 实施顺序

### Phase 0：证明前提并修复测量

1. 增加 CSA block 跨 rank checksum 验证。
2. 增加 Indexer kernel row-touch/ownership 验证。
3. 增加 union hash 和 collective sequence 验证。
4. 修复 overlap 分析器，使没有 prediction 的 layer 也进入统计。
5. 建立当前路径的 per-rank bytes、p50/p90 和 exposed wall 基线。

### Phase 1：Indexer-K CP-local 读取

1. 实现 CP ownership 到 SGL 的编译。
2. 每 rank 只读约 1/8 Indexer-K。
3. 验证 `mqa_logits` 结果和时间不回退。
4. 独立部署和 A/B，确认无误后作为后续 CSA 优化的基础。

### Phase 2：CSA shard-gather 基础路径

1. 实现 deterministic union partition。
2. 初始化独立 communicator、stream 和固定 staging ring。
3. 先通过手动 layer 开关运行完整 dense layer。
4. 验证 checksum、collective 顺序、SSD bytes 和模型 NCCL 干扰。

### Phase 3：覆盖早层暴露 I/O

1. 对 2--24 层启用 dense shard-gather。
2. 从前 1--2 层发起。
3. 以真实 exposed wall 和 TTFT 判断提前距离，不使用 lifecycle deadline 代替。

### Phase 4：深层预测路径和自动决策

1. 将 26--42 层 predicted union 接入 shard-gather。
2. 启用 residual local correction 和 loaded bitmap。
3. 接入按算子/资源的 decision table。
4. 对 context、query、union size 和 queue depth 做完整矩阵。

### Phase 5：可选优化

- 评估多层 HCA 批量 shard-gather。
- 评估在 trace 证明安全后与模型 NCCL 部分并发。
- 评估 SSD 对象布局重写是否能进一步减少 Indexer SGL 数量。
- 在当前版本稳定后单独迁移到 vLLM 0.26。

## 15. 第一版批准范围

第一版建议批准以下范围：

1. Indexer-K CP-local 1/8 SSD 读取。
2. CSA dense/predicted union 的 8-way SSD 分片与 NVLink all-gather。
3. 早层 dense shard-gather、深层 predicted shard-gather。
4. 基于 p90 和具体算子资源窗口的模式决策。
5. 独立 NCCL communicator 和 CUDA stream。
6. HCA 保持当前本地读取。
7. 现有 SSD 格式不变。
8. 当前本地路径保留为完整 fallback 和 kill switch。
9. 不同时升级 vLLM 0.26。

该范围预计把全机 SSD 数据量从约 13.40 GiB/request 降至约
2.06 GiB/request，将累计 SSD 服务时间从约 149--152 ms/rank 降至约
40--55 ms/rank，并以回收当前约 82 ms/rank 暴露 I/O 为理论上限，把
480K+8K 稳态 TTFT 从约 1.474 s 优化到约 1.40--1.43 s。
