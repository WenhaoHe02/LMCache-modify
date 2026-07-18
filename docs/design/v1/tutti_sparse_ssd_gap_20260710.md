# Tutti 接入差距、首命中税与 sparse-attention SSD 论文方向

日期：2026-07-10

## 1. 先说结论

当前 LMCache 接入的是 Tutti 的 **snvme/GPU-direct 读取前半段**，不是论文中的
完整 Tutti runtime。SSD 数据面已经能达到约 9--11 GB/s；当前失败点主要在
SSD 之后：host-driven 分批、staging 到最终 KV 的 scatter、按请求构造描述符，
以及 sparse index 本身仍走文件后端。

V30 的 B/C 两个全新前缀已经证明 raw-batch materializer 能把 ON 路径的 hit-1
降到稳态附近；这回答了工程实现问题，但**不能挽救 CSA prefetch 本身的论文
假设**。V30 只保留为 materializer ablation。论文主线和后续基线统一使用
`CSA prefetch OFF` 的普通 LMCache SSD store/retrieve 路径。

## 2. 已测事实

### 2.1 SSD 不是 0.6 GB/s

- `1.35 GB / 2.52 s ~= 0.6 GB/s` 是 retrieve **端到端有效吞吐**，包含 Python
  callback、对象/范围描述符构造和 scatter，不能称为 SSD 带宽。
- 同一链路中，NVMe mega-batch 合计约 146 ms，数据面约 9--11 GB/s。

### 2.2 V29 scatter 修改

修改包括：

1. 任意数量 GPU source pointer 可由一次 CUDA launch 完成 scatter；不再每四个
   object 切一刀。
2. HMA block-id 描述符由每 chunk 重建改为每 group、每 batch 只构造一次。
3. walker 实验分支把同一 staging batch 内的几十次 `index_copy_`/`index_put_`
   合并为每层一次，并允许多个相邻层共用一次 Tutti 调用。

H200 隔离微基准（DSv4 indexer 实际几何，1.239 GiB）：

| 路径 | launches | GPU p50 | wall p50 |
|---|---:|---:|---:|
| 旧 4-pointer 分片 | 469 | 21.377 ms | 21.388 ms |
| 新动态 pointer batch | 1 | 0.860 ms | 0.870 ms |

这证明四指针限制是真开销，但绝对值只约 20.5 ms，**不能解释秒级首命中税**。

### 2.3 480K+16K 完整链路

V28 flags，`VECTORIZED_CONSUME=1`，无 profiler：

| 轮次 | cold | hit-1 | hit-2..5（n=4） | repeat-5b |
|---|---:|---:|---|---:|
| 进程内第一个前缀 | 48.327 s | 17.221 s | p50 4.849 s，mean 4.874 s，sample std 0.074 s，min/max 4.818/4.979 s | 3.344 s |
| 第二个全新前缀 | 42.323 s | 7.194 s | p50 4.720 s，mean 4.742 s，sample std 0.387 s，min/max 4.405/5.122 s | 2.841 s |

两轮全部 HTTP 200，未发现 illegal memory access。旧 V28 profiler 轮的
`hit-2..5` p50 为 6.465 s，但 profiler 条件不同，不能当作严格 A/B；需要在同一
干净启动下用 `VECTORIZED_CONSUME=0/1` 重跑。

按 8 rank 的 `req_id` 拆解：

- 第一个前缀 hit-1：walker rank 中位 13.816 s。
- 第二个前缀 hit-1：walker rank 中位 3.847 s。
- 第二个前缀后续 hit：8/8 rank 均命中 resident signature，walker 消失。
- 第二轮主 KV retrieve 的 rank 中位约 1.18--2.11 s；首命中差值不是 SSD
  主 retrieve 单独造成的。

因此，当前代码优化了稳态，但没有满足任意新前缀的首命中 SLO。

### 2.4 多层 walker 合并后的复测（同机、干净 reboot）

在上述 V29 基础上再启用两项实验优化：

1. 41 个 CSA/HCA 层不再各自调用一次 Tutti，而是保持层顺序、在一个调用中
   流水提交；早层落地后立即通知 gate，不等待整个 walker。
2. 同一 staging batch 内的 object 不再逐 object `index_copy_`/`index_put_`，
   而是按目标层合并 source/row 后每层 scatter 一次。

结果：

| 轮次 | cold | hit-1 | hit-2..5（n=4） | repeat-5b |
|---|---:|---:|---|---:|
| 进程内第一个前缀 | 49.848 s | 16.860 s | p50 4.649 s，mean 4.782 s，sample std 0.498 s，min/max 4.377/5.453 s | 3.315 s |
| 第二个全新前缀 | 43.144 s | **6.449 s** | p50 4.468 s，mean 4.810 s，sample std 0.703 s，min/max 4.439/5.864 s | 2.889 s |

与修改前的第二个新前缀相比：

- hit-1：7.194 -> 6.449 s，减少 0.745 s（10.35%）。
- walker rank p50：3.847 -> 2.819 s，减少 1.028 s（26.71%）。
- 第二个新前缀仍比自身稳态 p50 多 1.981 s，所以依然不满足 first-hit SLO。
- 两轮全部 HTTP 200，未发现 illegal memory access。

第二个新前缀的 walker（8 rank，均为 76,834 objects、76 batches、
1,462.2 MiB）进一步分解如下：

| 阶段 | rank p50 |
|---|---:|
| NVMe submit kernel launch | 2.3 ms |
| NVMe poll/sync（真实数据等待） | 369.6 ms |
| Python I/O plan/build | 257.2 ms |
| CQ status readback/check | 43.4 ms |
| 76,834 个 staging view + `TensorMemoryObj` wrap | **1,035.8 ms** |
| raw load 内 batch total 合计 | 1,726.2 ms |
| callback/outer（含 scatter、Python consume、每批公平让锁） | **875.3 ms** |
| `load_chunks_to_hbm` total | 2,594.0 ms |
| walker total | 2,819.4 ms |

因此剩余瓶颈已经不是“SSD 只有 0.6 GB/s”。该 walker 的 NVMe poll 仅约
0.37 s；约 2 s 消耗在 76,834 个 Python/Tensor 对象和 post-DMA consume。
这正是论文 Tutti 的 GPU-native object/completion path 在当前 LMCache 中缺失的
部分，也直接决定下一步应先做 raw-batch/final-layout consume，而不是继续调整
SSD queue 参数。

### 2.5 raw-batch heterogeneous materializer（V30）

V30 保持 V29 的 480K+16K、41 个 CSA/HCA 层合并、
`VECTORIZED_CONSUME=1`，新增两部分：

注意：该实验在 CSA prefetch ON walker 上完成，只证明 raw completion 和最终布局
materializer 的机制价值；不能作为主方案配置或 prefetch 有效性的证据。

1. Tutti CQ 校验完成后直接回调 `(staging, offsets, nbytes)`，不再为 76,834
   个对象构造 staging view、metadata 和 `TensorMemoryObj`。
2. 一个 CUDA row materializer 直接读取 object pointer array，并按最终 HMA
   layout 写入 flat CSA rows 或非连续 HCA `(block, slot)`；按真实指针对齐情况
   自动选择 64-bit 或 byte copy。

同一干净 reboot 内顺序运行三个不同前缀，三轮均执行完整的 8-rank、76,834-key
raw walk；全部请求 HTTP 200，日志中 illegal access/CUDA error/Traceback 为 0。

| 前缀 | cold | hit-1 | hit-2..5（n=4） | hit-1 - 稳态 p50 |
|---|---:|---:|---|---:|
| A（进程内第一次） | 49.322 s | 17.509 s | mean 4.688 s，sample std 0.349 s，p50 4.664 s，min/max 4.383/5.041 s | +12.846 s |
| B（第二个新前缀） | 45.493 s | **4.658 s** | mean 4.991 s，sample std 0.299 s，p50 5.117 s，min/max 4.546/5.183 s | **-0.459 s** |
| C（第三个新前缀） | 40.164 s | **4.898 s** | mean 4.938 s，sample std 0.493 s，p50 4.941 s，min/max 4.482/5.387 s | **-0.043 s** |

后两个新前缀合并口径：hit-1 `n=2`，mean/p50 4.778 s，sample std
0.169 s，min/max 4.658/4.898 s；对应 hit-2..5 `n=8`，mean 4.964 s，
sample std 0.379 s，p50 5.117 s，min/max 4.482/5.387 s。因此后续新前缀
的 hit-1 已不再比自身稳态长。

三个 full walk 的 loader rank p50 分别为 1,141.5、1,200.0、1,209.5 ms
（每轮 `n=8`）；后两轮 walker rank 统计为：

| 轮次 | n | mean | sample std | p50 | min/max |
|---|---:|---:|---:|---:|---:|
| B | 8 | 1,427.3 ms | 15.2 ms | 1,428.5 ms | 1,403.3/1,446.5 ms |
| C | 8 | 1,457.7 ms | 33.6 ms | 1,459.8 ms | 1,404.1/1,502.7 ms |

raw materializer callback 的 per-batch 口径为 `n=1824`，mean 2.230 ms，
sample std 3.117 ms，p50 1.054 ms，min/max 0.123/14.320 ms；`wrap_ms`
在所有 raw batch 中均为 0。与 V29 第二新前缀相比：

- `load_chunks_to_hbm`：2,594.0 -> 约 1,200.0 ms，减少约 53.7%。
- walker rank p50：2,819.4 -> 1,428.5 ms，减少约 49.3%。
- hit-1：6.449 -> 4.658 s，减少 1.791 s（27.8%），且低于自身稳态 p50。

这验证了 H4 的核心判断：V29 的主要损失确实来自 post-DMA object lifecycle，
不是 SSD 带宽。它仍不是完整论文版 Tutti：每轮还有 rank p50 约 257 ms 的
Python plan build、约 44--45 ms CQ status 检查和 76 次 host-side batch 循环。

第一次全局 hit-1 的异常需单列：rank 0 walker 为 1.868 s，而其余 7 rank
为 13.816--14.214 s；但这 8 个 rank 真正进入 loader 后都只运行
1.032--1.213 s。根因是 cold-store writer 的 `max_delay=2s` 原先会无条件绕过
`readers_waiting`，大量 overdue writer 对裸锁连续 barging。代码已改为“超时只
绕过 idle slack，绝不绕过已声明读者”，并有 4 个单元测试通过；**该 gate 修复
未包含在本次 V30 端到端数据中，不能把 A 的 17.509 s 宣称为已经修复。**

## 3. 当前 LMCache 与论文版 Tutti 的差距

| 论文版 Tutti | 当前 LMCache 接入 |
|---|---|
| 最终对象/KV 内存预注册并建立 P2P/SGL 映射 | DMA 先到 4 GiB HBM staging，再 G2G scatter 到 vLLM HMA |
| 一个 IOCB 包含多个 IOCTX，由 GPU 批量提交/完成 | Python 为 key/range 建描述符，GPU kernel 仍以单线程生成 SQE/轮询为主 |
| 异步 event 驱动，I/O、计算、完成处理流水化 | 多处 `stream.synchronize()`、CPU status readback 和 Python callback |
| green context/slack-aware 调度，限制 I/O 对前向 SM 的干扰 | scatter 与 TP8 MoE/EP、NCCL、sparse attention 竞争同一 GPU |
| 对象生命周期与队列 teardown 是 runtime 的一部分 | 跑后常残留 8--19 GB UVM；容器 stop 可能卡住，只能 reboot |

还有一个直接的接入 bug：启动日志显示
`LMCACHE_INDEXER_TUTTI_BACKEND=1`，但 21 个 CSA indexer 注册时 Tutti loader
尚未异步创建，于是永久退回 file backend。loader 后续就绪只会延迟挂接
CSA-attention-KV manager，不会升级 indexer manager。

即使修复这个时序，现有 `TuttiIndexerBlockStore` 仍会把 HBM tensor `.cpu()` 成
Python bytes，再交回旧 `IndexerBlockStore` API；写路径还要求 512 B 对齐，而
132 B/token 的任意起点天然不对齐。因此当前 indexer Tutti 代码是 scaffold，
不是可用于生产结论的 GPU-native backend。

## 4. 为什么论文成功、这里失败

| 工作 | 真正做了什么 | 为什么不能直接类比当前失败 |
|---|---|---|
| [Tutti](https://arxiv.org/pdf/2605.03375) | GPU 发起 NVMe I/O、预注册最终对象内存、批 IOCB、异步完成和 GPU 调度 | 论文评测主要是 H100 上较小模型/普通 prefix cache；没有 DSv4 TP8 MoE/EP、五种 HMA layout、480K+16K 和两套 sparse index/KV |
| [SolidAttention](https://www.usenix.org/system/files/fast26-zheng.pdf) | 近似 block-sparse、K/V interleave、历史预测、DAG microtask，主动限制 working set | 其 batch=1、3B--8B、128K；我们的 16K prefill 对 top-k 取 union 后覆盖趋近全前缀，稀疏性先消失了 |
| [ECHO](https://www.usenix.org/conference/osdi26/presentation/liu-guangda) | 公开摘要声称 GPU graph cache manager、lossless prediction、fused indexer/recall | 截至本记录只有公开摘要可核，不能据此宣称它在同等 SSD/DSv4 负载成功 |
| [Strata](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang) | GPU-assisted 大块 I/O、cache-aware scheduling、delay-hit 合并与互补工作重叠 | 它说明并发同前缀请求必须作为调度问题处理；我们当前只测串行并保留上一个请求的 signature |
| [SparseServe](https://arxiv.org/pdf/2509.24626) | HBM--DRAM 分层、FlashH2D、working-set 控制、分层 prefill | 它不是 SSD 系统，避开了 NVMe 对象与 LBA 生命周期问题 |
| [SPIN](https://arxiv.org/abs/2604.26837) | 统一 sparse partition/page substrate、按请求 HBM budget、working-set metadata | 已覆盖“多种 sparse attention + hierarchical memory”的宽故事；本文必须靠 cross-layer index-to-I/O 与 first-hit SSD 证据区分 |
| [Swarm](https://arxiv.org/html/2603.17803) | 按 co-activation 放置，DRAM medoid/hot cluster，多 SSD 并行 | 依赖约 10% 稀疏工作集和大 DRAM；我们 union saturation 后接近全读 |
| [IndexCache](https://arxiv.org/pdf/2603.12201) | 在 F/S 层复用 top-k，减少 50%--75% indexer 计算 | 它本身不是 SSD 工作；价值在于让跨层共享 index 同时删除 indexer FLOPs 和对应 SSD bytes |

根因按优先级是：

1. **prefill union explosion**：单 query 稀疏不等于 16K query union 稀疏。
2. **每前缀首命中 walker**：只缓存上一个请求的 HBM signature，新前缀必须全读
   41 个 CSA/HCA 层。
3. **host-driven post-DMA path**：SSD 很快，CPU 描述符、分批 callback 和 scatter
   把端到端时间拉长。
4. **indexer 没走 Tutti且有 CPU bounce**：配置与真实执行路径不一致。
5. **资源竞争**：TP8/EP MoE、NCCL、indexer、walker scatter 同时占 GPU。
6. **生命周期 bug**：illegal access、UVM 残留、队列 teardown 卡死会破坏实验可重复性。

Strata 所说的 delay hit 也需要单独区分：同一前缀的第一个 SSD load 尚未完成时，
后来的请求应订阅同一个 in-flight object/index plan，而不是重复发起 retrieve 和
walker。这能解决并发放大，但不能替代“任意新前缀自身的 first hit 也必须快”；
后者仍要求减少/隐藏首次 walker。

## 5. 更像论文的创新点

不建议把论文主线写成“把 LMCache 的 Tutti 接好”。更强的系统抽象是
**Index-to-I/O：把跨层共享的 sparse index 直接编译成 SSD 读取计划**。

1. GLM 的 IndexShare/F-S 层、DSv4 的 CSA 深度分组复用 top-k；没有重新打分的层
   也不再读取自己的 indexer K。
2. 同一共享 index 同时决定 CSA/HCA/SWA 的 byte ranges、跨层合并方式和
   sparse-vs-bulk 切换；以 union density 为阈值，而不是固定打开 bulk。
3. I/O 直接 DMA 到最终 HMA rows，或至少由一个 persistent completion kernel
   完成 descriptor consume + scatter，删除 Python callback。
4. 用 deadline/SM slack 调度 scatter；目标是让每个 sparse layer 在 gate 前完成，
   而不是追求脱离前向计算的峰值 GB/s。
5. 以任意新前缀的 first-hit p50/p99、读放大、indexer FLOPs、SSD bytes 和
   GPU interference 为核心指标。稳态重复同一前缀只能作为次要 ablation。

这个方向同时组合了 IndexCache 的“跨层不重复索引”、Tutti 的 GPU-initiated
storage，以及 DSv4/GLM 特有的异构 sparse-attention layout；创新点比单独的
prefetch policy 或 SSD 接线更完整。

### 5.1 建议的 paper thesis

> **Sparse attention 的计算稀疏性不会自动变成存储 I/O 稀疏性。** 长 prefill
> 中，成千上万 query 的 top-k union 会迅速覆盖整个前缀；如果 storage runtime
> 仍按层、按 object、按 Python callback 恢复 KV，那么 SSD 已经跑满也无法获得
> 低 TTFT。系统必须把跨层 sparse index 编译为受 working-set 约束、直接落到
> 最终异构 KV layout 的 I/O plan。

该 thesis 包含一个算法/系统共同问题，而不是“把 Tutti 移植到 LMCache”：

- 算法侧决定哪些层真正需要重新 index、每个 prefill segment 的 union 有多密、
  何时从 sparse read 切换到 bulk read。
- 存储侧把同一个共享 index 变成跨层 byte-range/placement plan，删除不必要的
  indexer bytes 和 KV bytes。
- runtime 侧在 layer deadline 前直接 materialize 到 CSA/HCA/SWA 的最终 HMA
  rows，避免 per-object Python lifecycle 和无节制的 scatter/SM 竞争。

### 5.2 主创新：Cross-layer Index-to-I/O Compiler

输入是少数 anchor/full 层产生的 top-k、当前 prefill segment、各层 attention
role 和 HMA layout；输出是一个跨层物理计划：

```text
shared top-k / score summary
        + layer role (CSA/HCA/SWA/DSA)
        + HMA rows and SSD extents
        + union-density / deadline budget
                          |
                          v
       {skip indexer, sparse ranges, bulk extent, resident reuse}
                          |
                          v
             GPU-consumable cross-layer I/O plan
```

关键机制：

1. **Index reuse 同时删计算和 I/O。** Shared 层不跑 mqa/topK，也不读自己的
   indexer K；anchor index 直接驱动该层 KV ranges。
2. **Union-bounded prefill。** 不以整个 16K/32K/48K 增量做一次 union；动态选择
   segment，使 union density 留在 SSD/HBM budget 内。超过阈值时切 bulk，避免
   “名义 sparse、实际全读”。
3. **Heterogeneous lowering。** 同一逻辑 index 分别 lowering 到 CSA 的压缩条目、
   HCA 的 `(block, slot)`、SWA 的确定窗口，以及 GLM DSA 的共享层布局。
4. **Cross-layer coalescing。** 按消费顺序把相邻层 ranges 合并为少量 IOCB，而不
   是 21/41 次独立 host call。

与已有工作的边界：

| 已有工作 | 已覆盖 | 本文仍新增什么 |
|---|---|---|
| IndexCache | 跨层复用 top-k，删除 indexer 计算 | 把复用结果继续 lowering 为 SSD bytes/placement/deadline，联合删除 I/O |
| Tutti | GPU-native object I/O 与 slack-aware runtime | 不决定 sparse layer 读哪些 bytes，也不处理跨层 index reuse/union saturation |
| SolidAttention | block 化、历史预测、SSD prefetch DAG | 不处理 DSv4/GLM 原生异构 attention 与跨层 index-to-I/O lowering |
| ECHO | lossless intra/inter-query prefetch、fused indexer/recall（公开摘要） | 重点转向跨层复用、SSD 物理计划及 prefill union 的 sparse/bulk 决策 |
| Swarm | co-activation placement、DRAM hot clusters、多 SSD | placement 信号来自 token co-activation；本文用共享 index 同时控制计算与读取计划 |
| Strata | hierarchical cache scheduling、delay-hit 合并 | 本文合并的是 sparse index/I/O plan，并解决新前缀 first hit，不只并发同前缀 delay hit |

### 5.3 辅创新：GPU-native Heterogeneous Materializer

这是主创新的必要 runtime，但不应单独冒充比 Tutti 更新的 idea：

- 启动时预注册最终 HMA rows，或让 persistent completion kernel 消费 raw-batch
  layout 后直接 scatter；不创建 76,834 个 `TensorMemoryObj`。
- 一个 GPU plan 描述 source IOVA、目标 HMA row、row stride、attention role 和
  layer deadline；完成队列直接唤醒对应 gate。
- 按 slack/deadline 调度 copy，避免 walker 与 TP8 MoE/EP、NCCL、indexer 同时
  抢 SM。目标不是离线峰值 GB/s，而是 first-hit TTFT/p99。
- CSA/HCA/SWA/DSA 共用计划格式，但由不同 lowering 规则产生目标地址。

当前测量给它的必要性提供了直接证据：1.462 GiB walker 中 NVMe 等待约
0.37 s，而 object wrap + callback/consume 约 1.9 s。

### 5.4 辅创新：Sparse Delay-hit 与计划共享

- 同一前缀已有 in-flight index/I/O plan 时，后续请求订阅其 completion，不重复
  retrieve/walker。
- 不同请求若共享长 prefix，只为差异 segment 新建 plan；公共 anchor index 和
  materialized ranges 引用计数共享。
- 该机制与 Strata 的 delay-hit scheduler 相容，但共享单位从“cache load”细化为
  “跨层 sparse plan + layer completion”。
- 它解决并发放大；任意新前缀自身的 first hit 仍必须靠 5.2/5.3 降低和隐藏。

### 5.5 哪些不能作为主创新

- “把 0.6 GB/s 优化到 11 GB/s”：0.6 本来就是错误的端到端/SSD 混合口径。
- “把四指针改为任意指针”：实测只省约 20.5 ms，是工程优化。
- “把当前 indexer file backend 换成 Tutti”：必要修复，但论文 Tutti 已覆盖
  GPU-native object I/O；单独不新。
- “只做 resident signature”：只对紧邻重复请求有效，且当前没有并发 row epoch
  安全性，无法支撑 production/paper 主结论。
- “固定 bulk walker”：16K union saturation 时合理，但没有解释何时 sparse、何时
  bulk；缺少算法决策就只是特定 workload tuning。

### 5.6 可证伪的研究假设

1. **H1（union）**：prefill query 数增加时，per-layer top-k union density 存在
   快速相变；固定 segment 会在 480K+16K 接近全覆盖。
2. **H2（cross-layer）**：相邻/共享层 index 足够相似，anchor 层可同时删除至少
   一半 indexer 调用和对应 indexer SSD bytes，而质量下降可忽略。
3. **H3（I/O plan）**：union-bounded segment + cross-layer coalescing 能显著降低
   读取放大和 IOCB/object 数；收益不是来自更高裸 SSD GB/s。
4. **H4（materializer）**：删除 Python object/wrap/callback 后，新前缀 first-hit
   walker 可被 16K compute window 大部分或全部隐藏。
5. **H5（scheduler）**：deadline/slack 调度比“尽快 scatter”有更低 TTFT p99，
   即使其离线 copy 带宽较低。

任一假设失败都应收缩 claim。例如 H2 若在 DSv4 上质量/recall 不成立，则论文
不能宣称通用跨层 index reuse，只能保留 union-aware I/O compiler。

### 5.7 Anchor-every-N-layer 的受约束提前读取

“每隔 N 层预测一次”不应实现成固定 N 的后台 walker。更可检验的机制是让少数
anchor 层产生真实 top-k/score summary，并为未来若干层编译候选 I/O plan；N 和
lookahead depth 由运行时共同决定：

1. **准确度约束**：只有 layer-pair top-k overlap、recall 和 union-density 仍在
   离线标定阈值内时才复用 anchor；否则在下一层重新 index。
2. **deadline 约束**：候选层的最晚完成时间是该层 attention gate。可使用 anchor
   后的 attention、MoE/EP、NCCL 以及其他层计算窗口，不限于单个 MoE 窗口。
3. **I/O admission 约束**：demand retrieve 最高优先级，cold store 次之，预测
   read 最低。预测只在前两者均未等待且 token-bucket 有余额时提交下一 microbatch；
   新 demand 到达后不再提交后续批。已经进入 NVMe 的一个 microbatch 不可抢占，
   因此必须限制其 bytes/I/O 数和实测服务时间。
4. **GPU admission 约束**：NVMe 完成不等于立即 scatter。若 MoE/EP、NCCL 或前台
   materializer 正在占用 SM，完成数据停在预注册 staging，由独立 deadline queue
   在最晚时刻前落到最终 HMA rows。
5. **浪费约束**：每个计划记录 predicted/read/consumed bytes。错预测、union 扩张
   或前台 I/O 到达时可取消未提交 extents；浪费超过预算就缩短 lookahead 或回退
   demand load。

这与原 CSA prefetch ON 的差别是：anchor 来自模型真实 sparse index，研究对象是
跨层 index reuse 和受 SLO 约束的 I/O 编译；OFF 基线不依赖 predicted walker。
它也比 Tutti 多了 sparse-layer deadline/accuracy/union 决策，比 IndexCache 多了
SSD byte-range lowering 和多租户 I/O 干扰约束。

### 5.8 论文实验矩阵

**模型/架构**：DSv4（21 个实际 CSA indexer + HCA/SWA 异构组）、GLM-5/5.2
DSA。GLM 的确切 F/S/IndexShare 映射必须从实际模型配置核验，不能假定每层都
是 DSA/CSA。

**形状**：64K+2K（机制 sanity）、480K+16K/32K/48K（决胜形状），并加入多个
全新 prefix，而不只重复同一 prefix。

**并发**：串行 first hit、同 prefix delay hits、不同 prefix 并发；报告 p50/p95/
p99，而非只报最好一次。

**基线/ablation**：

1. vLLM 无 SSD reuse；LMCache file/GDS；当前 Tutti staging V28/V29。
2. transport-only（修好 Tutti，不做 index reuse）。
3. IndexCache-only（删 indexer compute，不改 SSD plan）。
4. union-aware-only、cross-layer-only、materializer-only、scheduler-only。
5. 完整 Index-to-I/O。

**指标**：TTFT/TPOT/goodput、first-hit p50/p99、per-layer gate stall、indexer
`mqa_logits/topK` 次数和 GPU ms、SSD logical/physical bytes、read amplification、
IOCB/object 数、NVMe poll、host descriptor/wrap/callback、scatter GPU ms、MoE/NCCL
干扰、临时 HBM/UVM 和失败率。

### 5.9 推荐的论文叙事与标题

推荐贡献顺序：

1. 首次揭示 native sparse-attention 在长 prefill 下的 **index sparsity 与 I/O
   sparsity 脱钩**，并用 union/first-hit 分解量化。

2. 提出 Cross-layer Index-to-I/O compiler，把 shared index 联合 lowering 为
   compute skip、SSD ranges、layout placement 与 sparse/bulk 决策。
3. 提出 GPU-native heterogeneous materializer + deadline scheduler，使计划在
   DSv4/GLM 异构 KV layout 上可执行。
4. 在 first-hit、delay-hit 和并发 workload 上验证，而非只展示 warm repeat。

暂定标题：

- **From Index Sparsity to I/O Sparsity: SSD Acceleration for Native Sparse-Attention LLMs**
- **Index-to-I/O: Cross-Layer Sparse KV Materialization for Long-Context LLM Serving**
- **Beyond Fast SSDs: First-Hit-Efficient Storage for Native Sparse Attention**

## 6. 下一步实验

1. 停止追加 CSA prefetch ON/V28 证明实验。V30 B/C 只归档为 materializer
   ablation；V31 不进入主结论。
2. 以 `CSA prefetch OFF` 启动普通 LMCache SSD hit，A/B 原 streaming
   `MemoryObj` consume 与 OFF raw-completion/final-layout consume；至少 3 个新 prefix，
   报 hit-1/steady p50、sample std、范围和所有 rank 的阶段分解。
3. 在 OFF 路径测前台 demand read + cold store + speculative read 三类并发。预测批
   从 1/2/4/8 MiB 扫描，要求 demand TTFT p99 和 store drain time 都有显式上限；
   不能只报告预测 load 被隐藏的比例。
4. 把当前每轮约 257 ms Python plan、44--45 ms CQ status readback 和 host batch
   loop 继续 lower 成 GPU-consumable descriptor/completion path；同时测 deadline-aware
   scatter 对 MoE/NCCL 的干扰，而非只测离线 copy GB/s。
5. 单独实现 GPU-native indexer backend：启动前保存 rank-specific FIEMAP，loader
   ready 后延迟绑定；read API 返回 GPU tensor，不再 `.cpu().numpy().tobytes()`。
6. 先采集 DSv4/GLM 的 layer-pair top-k overlap、recall、union-density 与层间计算
   slack，再决定 anchor N/lookahead；避免先写死“每 4 层共享”。
7. 做 IndexCache-style 两阶段 ablation：先只删除重复 mqa/topK，再让同一个
   shared index 删除对应 indexer SSD bytes 与 KV ranges。
8. ncu 分析 `mqa_logits`/`topKPerRowPrefill`，nsys 验证 materializer 的 NVMe、
   completion/scatter 和 MoE overlap；torch profiler 的 NCCL 时间不用于最终数字。

### 6.1 OFF 跨层 indexer 预取接口

当前实现先覆盖不需要 attention-KV filter 的安全子集：anchor CSA 层产生真实
top-k 后，只把这些 compressed-token id 用于后续层的 **Indexer cache** SSD
预取。它不跳过后续层的真实 indexer，不写 attention KV，也不改变 true-topK
correctness path，因此可在 CSA prefetch OFF 下单独启用：

```bash
export LMCACHE_INDEXER_CROSS_LAYER_PREFETCH=1
export LMCACHE_INDEXER_CROSS_LAYER_ANCHOR_STRIDE=4
export LMCACHE_INDEXER_CROSS_LAYER_LOOKAHEAD=3
export LMCACHE_INDEXER_CROSS_LAYER_MAX_TOKENS=1024
export LMCACHE_INDEXER_CROSS_LAYER_DEADLINE_MS=50
```

`ANCHOR_STRIDE=4` 表示第 0、4、8... 个 CSA 层是 anchor；一个 anchor 最多覆盖
其后的 3 个 CSA 层，且绝不越过下一个 anchor。这个顺序按实际注册的 CSA layer
列表计算，不假定模型每个 transformer layer 都是 CSA。

Tutti indexer backend 将候选拆成默认最多 8 token 的 speculative microbatch。
loader 在每批提交前执行以下 admission：

1. demand reader 或 store writer 已宣告等待时，取消所有未提交预测批；
2. 超过绝对 layer deadline 时停止；
3. 使用 `LMCACHE_TUTTI_SPECULATIVE_RATE_MBPS`（默认 256 MiB/s）和
   `LMCACHE_TUTTI_SPECULATIVE_BURST_MB`（默认 8 MiB）token bucket；
4. 已进入 NVMe 的一个批不可抢占，因此仍受 8 I/O/8 MiB 单批上限约束。

被 admission 取消的 indexer 预测不会自动升级为 demand read。后续层真实 top-k
若确实需要这些 token，原 miss-correction 路径才按 demand 读取，保证取消预测不会
破坏正确性，也不会在后台偷偷绕过前台优先级。

这还不是最终的 Index-to-I/O：attention-KV range defer、基于实测 layer-pair recall
的动态 stride/lookahead，以及 deadline-aware final-layout scatter 仍需后续实现。

## 7. 一手资料

- Tutti: <https://arxiv.org/abs/2605.03375>
- IndexCache: <https://arxiv.org/abs/2603.12201>
- SolidAttention (FAST '26):
  <https://www.usenix.org/conference/fast26/presentation/zheng>
- ECHO (OSDI '26 public page):
  <https://www.usenix.org/conference/osdi26/presentation/liu-guangda>
- Strata (OSDI '26 public page):
  <https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang>
- SparseServe: <https://arxiv.org/abs/2509.24626>
- SPIN: <https://arxiv.org/abs/2604.26837>
- Swarm: <https://arxiv.org/abs/2603.17803>

ECHO/Strata 在 2026-07-10 可核的是 USENIX 公开页面/摘要；没有拿到正文支持的
细节不写成既定事实。所有“为什么他们成功”的比较都区分论文事实、当前测量和
本文推断，不用不同模型/硬件/上下文的加速比直接对打。
