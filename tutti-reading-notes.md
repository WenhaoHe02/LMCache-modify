# Tutti 论文深度阅读笔记

> **论文**：Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving
> **arXiv**：2605.03375v1 (2026-05-05)
> **作者**：Shi Qiu, Yifan Hu, Xintao Wang, Wenhao Zhu, Jianqin Yan, Hao Chen, Kaiqiang Xu, Kai Chen, Yiming Zhang (厦大 / 上交 / 港科大)
> **对标系统**：LMCache（with/without GDS）
>
> 笔记说明：本文是阅读笔记，不是全文翻译。每节给出核心思想、图表解读、关键数字、我的点评。图自己对照原文 PDF 看——页码已标注。

---

## 总论：这篇论文到底在解决什么问题

**一句话**：LMCache 把 KV cache 放到 SSD 上本来就能做，但 GPU 会白等 70–80% 的时间，因为 I/O 路径上 CPU 还在当"指挥官"。Tutti 把 CPU 从这个关键路径上踢出去，让 GPU 自己发起 I/O，端到端 TTFT 降 78.3%，吞吐翻倍，成本降 27%。

**Tutti 的三板斧**：

1. **GPU-centric object store**（§3.1）：把 KV cache block 当 object 管，GPU 直接寻址 NVMe
2. **GPU io_uring / gio_uring**（§3.2）：GPU 侧模拟 Linux io_uring 做异步 I/O
3. **Slack-aware I/O scheduler**（§3.3）：离线 profile 每层的"闲暇窗口"，把读写 I/O 精确塞进去

---

## §1 Intro：问题的来龙去脉

### 1.1 背景逻辑链

- Prefix caching 是现代 LLM serving 的命门：命中后能省掉 90% per-token 成本
- 但上下文越来越长（百万 token）→ KV cache 爆炸 → HBM 不够 → DRAM 也不够（2TB DRAM 只能存 5 分钟的 KV）→ 必须上 SSD
- 单机可装 100TB NVMe，理论上能存一小时以上的 KV cache

### 1.2 为什么现有方案不行

HBM-DRAM-SSD 三层结构**速度太慢**，瓶颈不在 SSD 裸带宽，而在：

1. **分页 KV 布局**（vLLM/SGLang 的 paged attention）把逻辑上连续的 KV 切成一堆小块 → SSD 上变成海量随机小 I/O
2. **CPU 成为瓶颈**：每个 I/O 都要 CPU 发起，即使用了 GDS（GPU Direct Storage）也一样——GDS 只去掉了 DRAM 中转，没去掉 CPU 作为 I/O 控制路径
3. 结果：**GPU 70–80% 时间在等数据**，KV 重用比重算还慢

### 1.3 Tutti 的核心 insight

CPU 可以在"控制面"管元数据、分配、索引——但**不能卡在数据面的关键路径上**。把 CPU 的角色从"每次 I/O 都调度"（O(layer × blocks)）降到"每层加载一次 kernel"（O(layer)）。

**对照图 Fig. 1**（page 2）：
- 左：CPU-centric LMCache，不管有没有 GDS，CPU 都发起每次 memcopy / I/O
- 右：GPU-centric Tutti，CPU 只负责 "Load I/O Kernels"，然后 `Batch_Retrieval` 由 GPU 自驱动

---

## §2 Background：为什么 SSD 卡住了

### 2.1 LLM 推理基础（这节可略）
- Prefill（首 token）vs Decode（后续 token）
- KV cache = 用内存换计算
- Paged 管理：block 形状 `[Block, h, d]`，每个 block 16–32 token

### 2.2 SSD 层为什么慢

**关键数据（Fig. 2，page 4）** — vLLM + LMCache，Llama3-8B，64K sequence，75% hit rate：

两个 vLLM 版本对比（v0.12.0 / v0.17.0）下 GPU bubble 时间占比：

| 层级 | v0.12.0 bubble% | v0.17.0 bubble% |
|------|----------------|------------------|
| HBM | 9.4% | 1.7% |
| DRAM | 30.5% | 14.2% |
| SSD | 76.9% | 72.8% |
| SSD-LW (layerwise) | **84.0%** | **78.9%** |
| SSD-GDS | 73.0% | 72.3% |

**结论**：
- DRAM 层能和 HBM 基本持平（小开销）
- SSD 无论加不加 GDS，GPU 都在"鼓包"70%+
- 随着 vLLM 计算优化（v0.12 → v0.17），**重算反而比 SSD 重用更快了**（v0.17 的 dashed line 比 SSD 低 33.9%）
- 这是个很狠的说法：现有 SSD 方案本质已经"没用"了

**量化拆解**：Qwen3-32B（64 层，block=64），重建一个 128K 的 prefix 需要 **~256K 个 80KB 对象**的零散 I/O。LMCache 用 256-token chunk 合并，也要 1000+ 个 chunk；开 pipelining 后要上万次。

### 2.3 GPU-centric storage 的已有工作
- **BaM**（raw block 级）
- **GeminiFS**（文件级，本文作者之前的工作，被复用）
- **GoFS**（另一个 GPU 文件系统）

共同思路：GPU 线程直接操作 NVMe 提交队列（SQ）和完成队列（CQ），不经过 CPU。

### 2.4 把 GPU-centric storage 套到 KV cache 的三个真正挑战

**1. 抽象不匹配**：vLLM/SGLang 要动态分配 GPU 内存 block + 哈希索引，GPU-centric storage 只暴露 block/file 接口。把哈希下推到 GPU 做？**Fig. 3（page 5）狠打脸**：

| Sequence Length | CPU Insert | GPU Insert | CPU Lookup | GPU Lookup |
|-----------------|-----------|-----------|-----------|-----------|
| 128K | 12.0 ms | 108.5 ms | 4.2 ms | 107.6 ms |
| 256K | 13.4 ms | 216.2 ms | 4.3 ms | 215.1 ms |
| 512K | 20.7 ms | 431.6 ms | 13.1 ms | 430.0 ms |
| 1M | 35.3 ms | 862.0 ms | 27.7 ms | 860.0 ms |

GPU 哈希比 CPU 慢 9.0–24.2×（insert）、25.6–50.0×（lookup）。原因：哈希是串行 pointer-chasing，与 SIMT 模型不配。**结论：索引必须留在 CPU。**

**2. 粒度鸿沟**：
- GPU NVMe driver 为 4KB 小 I/O 优化（饱和 IOPS 但只用 16% 写带宽、80% 读带宽）
- KV 传输需要 ~100KB 大块
- 大块要用 PRP List pages，地址翻译必须走特权 CPU 代码 → 又回到了 CPU

**3. 资源争抢**：
- **SM 竞争**：GPU 调度非抢占，I/O kernel 长时间占 SM 会阻塞计算 kernel
- **带宽竞争**：Prefix caching 同时产生读（下一层）和写（上一层）流量，争抢 NVMe 内部 cache

---

## §3 Design：Tutti 三大件怎么实现

### 3.1 GPU-Centric Object Store（§3.1）

**核心抉择**：索引留 CPU，数据路径给 GPU。

**Fig. 4（page 5）讲解**：三层结构
- `GPU KV Cache Pool`：vLLM 原有的内存块池，每块 `[Layer × (K,V)]`
- `GPU File Pool`：每个 GPU file = `2 × L objects`（L = 层数，K 和 V 各一）；通过 P2P memory mapping table 映射到 PCI 地址
- `NVMe File Pool`：由 GeminiFS 管理，用 **Tensor-Stripe** 布局（按 tensor 粒度切片，不是传统的块级 striping），多盘 round-robin

**P2P mapping 的关键细节**：
- 朴素 PRP 方案：60GB KV cache on 80GB HBM → 需要 983,040 个 PRP list pages → 浪费 3.75GB HBM
- Tutti 改用 **SGL (Scatter Gather List)**：16 字节描述一大块连续内存（8B 物理地址 + 4B 长度 + 4B ID）
- 开销降到 **15MB**（250× 压缩）

运行时：engine 只做 block lookup + P2P table lookup，生成轻量 `GPU I/O contexts` 批量下发。CPU 开销从 `O(layer × blocks)` 降到 `O(layer)`。

**我的点评**：这一节的算法增量不大（SGL 是 NVMe 规范里现成的），但工程上是实打实的瓶颈——`O(layer × blocks)` → `O(layer)` 这个降阶是 Tutti 所有性能提升的源头。

### 3.2 GPU io_uring / gio_uring（§3.2）

**模型**："CPU-prepared, GPU-executed"：CPU 提前把 IOCB（I/O control block）准备好排队，GPU 自己触发执行。

**Fig. 5（page 6）讲解**：四个接口 + SQ/CQ
1. `init_queue(depth)`：创建 SQ 和 CQ，包含 depth 个 IOCB
2. `get_iocb(nums, event)`：取空闲 IOCB，CPU 填虚拟地址和 `num_ioctx`；插入 CUDA event 保证依赖顺序
3. `issue_io(IOCB_ids, SMs)`：enqueue GPU I/O kernel，指定 SM 分配；kernel 内部生成并下发 SSD 命令
4. `wait_cqe()`：GPU 原子写 CQ，CPU 轮询特定 IOCB index

**三个关键设计**：

**Zero-Copy Ring Buffers**：SQ/CQ 放在 GPU HBM，但通过 non-cached `mmap` 映射给 CPU；每个 SQ entry 是一个 IOCB，每个 IOCB 含 **2048 个 IOCTX**（对应 H100 的最小调度单元 2 SMs × 64 warps × 32 threads = 4096 threads 除以 2）。IOCTX 记 {SGL 地址, GPU file 偏移, 长度}。这个"批处理队列"是和传统 CPU io_uring 的关键区别。

**SM Partitioning（NVIDIA green context）**：硬件级把 SM 划成"Compute Domain"和"I/O Control Domain"，避免长时间 I/O kernel 阻塞关键计算 kernel。提供确定性 QoS。

**Async 处理**：I/O 命令生成和下发完全在 GPU 内完成，CPU 不用参与每个 I/O。

**我的点评**：把 io_uring 的设计搬到 GPU 是个很自然的类比，但细节（2048 IOCTX 的打包、green context 分区）需要对 H100 硬件调度机制有深理解才能踩到点上。这是论文最硬核的工程贡献。

### 3.3 Slack-Aware I/O Scheduler（§3.3）

**两个干扰源**：
1. 读写同时打 SSD → **带宽骤降 60%**（Fig. 6，page 7；作者用 FIO 单读+单写 256MB 粒度复现了这个现象）
2. I/O kernel 和 compute kernel 争 SM（embedding/normalization/GEMM 能吃 90% GPU）

**解法**：离线 profile 每层的 slack window，表驱动调度（Fig. 7，page 7 的时间线图）。

**离线 profiling**：
- 为什么 prefill 时间和 prefix 长度有关：attention 开销随 prefix 线性增长，但 Linear Projection / Normalization 不随其变
- 每层打表：索引 = `(L_input, L_prefix)`，值 = `(slack duration, available SM budget)`
- Step size 对齐单 warp 的 token 长度，profile 成本可控

**读写解耦调度**：
- **不**用朴素 layerwise pipeline（读写混着发，带宽崩）
- Prefill 阶段：**读优先**，因为在 reuse 关键路径上。查表找下一层 slack，能塞多少 IOCB 就塞多少；没 slack 就立即发读避免 stall
- 写请求**延后**：如果当前层还有 slack 就发，没有就 defer 到 decode 阶段或下一个请求的 slack 中
- Decode slack 短而不可预测，best-effort 塞写

**我的点评**：这是 Tutti 性能领先 LMCache-GDS 的第二个重要武器（第一个是去 CPU 化）。本质是把"读写时序"当成一个离线优化问题预先求解，运行时只查表。好处是决策零开销；代价是每个"model × hardware config"都要 profile 一次（作者承认这点，但说"只做一次就能复用"）。

### 3.4 Implementation

- **规模**：~8000 行 C++ + ~1500 行 Python（vLLM KVConnector 集成，已适配多个 vLLM 版本）
- **接口**：暴露给 vLLM 的是 `retrieve_layer` 和 `store_layer`（层级粒度），对应 slack scheduler 的读/写路径
- **多 GPU 支持**：TP 时每个 GPU 跑一个 Tutti 实例，local daemon 给每 GPU 独立分配 NVMe 队列（Solidigm D7-PS1010 支持 256 个 I/O queues，8 GPU 每 GPU 32 个够用）
- **跨节点扩展**：复用 **Mooncake** 做分布式控制平面（空间分配、副本元数据、location lookup）；Tutti 是每节点的快速数据平面。当前 remote 读走 RDMA（经 CPU 中转），未来计划做 GPU-initiated RDMA

**⚠️ 重要更正**：Mooncake **有**被提到，但不是 baseline 对比对象——它是 Tutti 的**上层协作者**（作者承认 Tutti 只解决单机快路径，跨节点靠 Mooncake）。我前面说"没对比 Mooncake"方向对，但表述不精确：应该说"Tutti 和 Mooncake 是互补的，不是竞争的"。

---

## §4 Evaluation：实验设置与结果

### 硬件环境
- 64-core Intel Xeon 6530 + 512GB RAM
- **2× H100 80GB GPU**
- **4× Solidigm D7-PS1010 7.68TB 企业级 NVMe**（PCIe 5.0）
- 每 GPU 分配 256GB pinned DRAM + 14TB SSD volume

### 模型
- **Llama3-8B**（单 GPU 实验）
- **GLM-4-9B-Chat-1M**（1M context 窗口，2-GPU TP）

### Workload
- **LEval**：20 子任务，输入 3k–200k token
- **LooGLE**：4 子任务，很多样本 >100k token
- Poisson 到达分布模拟多用户并发

### Baseline（4 个）
1. **HBM**（纯 vLLM）
2. **LMCache-DRAM-LW**（DRAM + layer-wise pipelining）
3. **LMCache-SSD**（SSD + memcopy + 标准 async I/O）
4. **LMCache-GDS**（SOTA，用 GDS 绕过 CPU bounce buffer）

### Table 1（page 9）：不同层 cache hit rate

| 存储 | LEval | LooGLE |
|------|-------|--------|
| HBM | 8% | 4% |
| DRAM | 53% | 24% |
| **SSD** | **84%** | **86%** |

HBM 严重不够用；DRAM 在 LooGLE（极长上下文）上掉到 24%；只有 SSD 层能撑起 80%+ 的命中率——这是论文立足的根本前提。

### 4.1 端到端性能（Fig. 8，page 9）

**关键数字汇总**：

**LEval + vLLM 0.17.0（新版，compute 更快，I/O 更显眼）**：
- TTFT：Tutti 比 DRAM 降 **69.1%**，比 GDS 降 **78.3%**
- 在 1s TTFT SLO 下：Tutti 相对 DRAM 的可支撑 RPS 提升 **50%**，相对 GDS 提升 **100%**
- ITL @ 1.5 RPS：Tutti 比 DRAM 降 22.0%，比 GDS 降 24.4%

**LooGLE + vLLM 0.17.0**：
- @ 0.6 RPS：GDS 的 TTFT 仍是 Tutti 的 **2.63×**
- Tutti 比 DRAM 降 **93.2%**，比 GDS 降 62.0%
- ITL：LooGLE 上优势收窄（只比 GDS 好 10.2–18.3%），作者解释为 decode 变成 compute-bound

### 4.2 Ablations

#### 4.2.1 原始带宽（Fig. 9，page 10）

**Retrieve bandwidth**（随 context 从 1K 到 128K）：
- LMCache-DRAM 不稳定，16K 时因内存碎片掉到 8.5 GB/s
- **Tutti 平滑线性增长，最高 25.9 GB/s**
- LMCache-GDS **饱和在 11.9 GB/s**（即使双盘）
- Tutti 对 GDS 在 retrieve 上达到 **2.08× 带宽**

**Store bandwidth**：
- LMCache-DRAM 18.4 GB/s（不持久，容量受限）
- Tutti 稳定 ~10 GB/s（128K 时 9.8 GB/s），受限于单盘峰值
- LMCache-GDS ~7 GB/s（同样双盘配置）

#### 4.2.2 PRP vs SGL（Fig. 10，page 11）

单 GPU 线程 500MB 读写 microbenchmark：

| 方案 | Read | Write |
|------|------|-------|
| PRP | 0.287 GB/s | 0.032 GB/s |
| **SGL** | **8.891 GB/s** | **2.922 GB/s** |
| 提升 | **31.0×** | **91.3×** |

这是 §3.1 SGL 设计选择的实证。

#### 4.2.3 TTFT vs Prefix 长度（Fig. 11，page 11）

固定总输入 128K，变化 cached prefix 16K→128K：
- LMCache-SSD 在 112K prefix 时 TTFT 飙到 **7.84s**
- Tutti 在同点 **3.43s**（2.28× 更快）
- 相对 GDS：32K 快 5.8%，128K 快 **61.4%**
- 16K–96K 区间 Tutti 甚至**超过 DRAM 13.4%**（I/O–compute 重叠抵消了 DRAM 的延迟优势）
- 仅在 >96K 极端纯 retrieval-bound 时 Tutti 落后 DRAM 最多 20.6%

#### 4.2.4 分布式可扩展性（Fig. 12，page 11）

GLM-4-9B-1M，2 GPU × 2 SSD：
- 128K prefix：Tutti 155.743s，GDS 207.12s（Tutti 省 ~25%）
- **512K / 640K prefix：GDS 直接 OOM 崩了**（因为 cufile 的 staging buffer 吃光 GPU 内存）
- Tutti 在 **640K 下 TTFT 仅 1.2s**

作者论点："高性能 I/O 必须与计算深度集成，不能作为第三方插件"——这是对 GDS 路线的直接批评。

#### 4.2.5 Layer-wise Async Pipelining（Fig. 13，page 12）

固定 32K prompt，扫 cache hit rate（50%→100%），分解 compute time vs bubble time：

- **LMCache-SSD-LW**：bubble 持续很大，无法被计算掩盖
- **LMCache-DRAM-LW**：crossover（bubble 开始超过 compute）在 **97.9%** hit rate
- **Tutti**：crossover 推到 **98.3%**，平均 bubble 仅 **25ms**，93.75% hit 时低至 **6ms**

"Zero-bubble zone"被推到了物理极限。**这是整篇论文最亮的单张图**——它证明 SSD-backed Tutti 的计算-I/O 重叠能力已经逼近 DRAM。

### 4.3 Inference Cost（Fig. 14，page 12）

成本公式：
```
Cost_1M = (P_GPU · N_GPU + P_mem · S_mem + P_ssd · S_ssd) / Throughput(tok/hr) × 10^6
```

**云价格取值**：
- H100: $5/hour/GPU
- DRAM: $0.0088/GB/hour
- NVMe SSD: $0.000082/GB/hour（**比 DRAM 便宜 ~100×**）

**关键结果（LooGLE @ 0.5 QPS）**：
- Tutti 比 LMCache-SSD 省 **66.2%**
- Tutti 比 LMCache-GDS 省 **~27%**

省钱根源不是 SSD 便宜（那所有 SSD 方案都会便宜），而是 Tutti **吃饱了 GPU**——没有 bubble → 更高 throughput → 单位 token 摊到的 GPU-hour 少。

---

## §5 Conclusion

作者的收尾：Tutti 首次证明 SSD-backed KV cache 可以做到 DRAM-like 的效率；未来工作是 GPU-initiated RDMA（分布式快路径）。

---

## 📌 图表速查清单（按 PDF 页码）

| 图 | 页 | 讲什么 |
|----|-----|--------|
| Fig. 1 | 2 | CPU-centric LMCache vs GPU-centric Tutti 的架构对比 |
| Fig. 2 | 4 | vLLM+LMCache 在各层的 GPU bubble 占比（两个 vLLM 版本） |
| Fig. 3 | 5 | CPU vs GPU 哈希性能（为什么索引不能下推） |
| Fig. 4 | 5 | Tutti GPU-centric KV cache store 三层布局 |
| Fig. 5 | 6 | gio_uring 架构与 I/O 流程（4 个接口） |
| Fig. 6 | 7 | 并发读写 vs 解耦读写的带宽利用率（60% drop） |
| Fig. 7 | 7 | Slack-aware scheduler 时间线图 |
| Fig. 8 | 9 | **端到端 TTFT / ITL**（LEval/LooGLE × 2 vLLM 版本） |
| Fig. 9 | 10 | Retrieve / Store 原始带宽 vs context 长度 |
| Fig. 10 | 11 | PRP vs SGL 带宽对比 |
| Fig. 11 | 11 | TTFT vs prefix 长度（单 GPU） |
| Fig. 12 | 11 | 分布式 scalability（GDS 在 512K+ OOM） |
| Fig. 13 | 12 | **Compute/Bubble 分解 vs hit rate，crossover 98.3%** |
| Fig. 14 | 12 | 每 1M token 推理成本 |
| Table 1 | 9 | 各层 cache hit rate |

---

## 🔥 我的批判性点评

### 真正的贡献（不是包装）

1. **把 CPU 从 I/O 控制路径上干掉**，用 GPU io_uring 模型替代，这是 LMCache-GDS 做不到的——GDS 只去了 data path 的 CPU，没去 control path 的 CPU
2. **Slack-aware scheduling + 读写解耦**，利用离线 profile 把 read/write 精确塞进 slack，绕开 SSD 带宽争抢这个没被现有工作重视的点
3. **SGL 替代 PRP**，工程细节但是大头（31×/91× 带宽提升）
4. **Fig. 13 的 crossover 98.3%** 是一个很强的 framing——说明 SSD 方案不再是"性价比替代"而是"性能平替 DRAM"

### 真实的局限

1. **硬件依赖强**：Solidigm D7-PS1010 是企业级 PCIe 5.0 盘（每盘 10 GB/s 峰值），消费级 NVMe 能不能复现这个结论，论文没答
2. **模型偏小**：Llama3-8B 和 GLM-4-9B 都是 10B 以下，70B/405B/MoE 这种真正吃 KV cache 的场景没测
3. **Slack profiling 的可迁移性**：每个 model × hardware 组合都要重新 profile 一次；生产环境模型经常微调/切换/量化，profile 失效代价多大，没量化
4. **读优先 + 写延后**的策略在持续高负载下会不会出现 write starvation？作者没讨论
5. **Mooncake 协作只停在 design**：remote path 走"CPU → RDMA"的临时方案，跨节点真实性能数据缺席
6. **SSD 寿命 / 写放大**：企业 SSD 贵，大量反复 store 会不会啃穿 TBW？成本公式里没算进去
7. **Fig. 2 的"重算比 SSD 快"论断**是一把双刃剑——它证明现有 SSD 方案已死，但也暗示如果 vLLM 继续优化，Tutti 的加速比也会被压缩

### 最真诚的一句
如果你是 LMCache 的贡献者，这篇论文**值得仔细研究**：它提供了一份清晰的"LMCache 现在差在哪、怎么补"的路线图——
- 核心改造点：`O(layer × blocks)` → `O(layer)` 的 CPU 开销降阶
- 工程抓手：SGL + GPU-managed NVMe queue pair
- 调度抓手：离线 slack profile + 读写解耦
