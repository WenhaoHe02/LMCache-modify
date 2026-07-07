# V24/V25 关键数据(2026-07-08 追加)

## V27:HBM 行搬迁 —— 稳态追平 OFF(突破!)

V26 诊断日志给出精确根因:**chunk key 命中 100%(41979/41979),但
physical rows 不匹配 100%**——vLLM 块分配器每个请求给同一前缀分配不同
物理行。同行跳过永远打不中。

V27 修法([csa_attention_kv_prefetch_manager.py](../../../lmcache/v1/csa_attention_kv_prefetch_manager.py)
`_match_resident_chunks`):key 命中 + slab 一致 + 行不同 → **GPU 内
gather(旧行)→ scatter(新行)搬迁**,在注册时(retrieve 阶段,SM 空闲
等 NVMe)执行,~123ms/rank 搬迁全部 40677 chunk。NVMe 重读和 compute
期 scatter 双双消失。gather 先于 scatter(经临时张量)使任意新旧行
别名安全;上传在消费流上(V17 教训)。

**V27-16K 结果(480K+16K,干净环境)**:

| | r1 | r2 | OFF 基线 |
|---|---|---|---|
| hit-2..5 | 6.40/6.03/6.04/6.65 | **6.06/6.06/6.03/6.72** | 6.04-6.68 |
| hit-1(签名未建立) | 17.3(冷) | 7.04 | — |
| repeat | 5.44/4.46 | **4.49/4.43** | 5.51/4.59 |
| illegal | 0 | 0 | — |

**稳态 ON = OFF 线(6.03-6.72 vs 6.04-6.68),repeat 反超 OFF**
(4.4 vs 4.6)。V25 的 +0.7-1.0s 稳态税被搬迁完全消除——SM 争用模型
的最终验证:把 scatter 从 compute 期挪到 retrieve 期(或者干脆不做),
差距就消失。

**V27-32K 确认跑(480K+32K,干净 reboot)**:
- r1 hits 7.70/8.36/7.74/8.44,r2 hits 8.53/7.78/7.77/8.59(hit-1 7.94)
- repeat 4.70/4.70(V25 是 6.4;OFF-16K repeat 4.6)
- relocation matched=41979/41979,147ms/rank,illegal=0

**同夜新鲜 OFF-32K 基线(含 repeat,公平对照)**:
- r1 hits 7.80/8.33/7.71/7.80/7.74,repeat 10.17/4.82
- r2 hits 7.76/7.83/7.75/8.39/8.52,repeat 4.73/4.78

**结论:32K 上 ON 与 OFF 分布重合**(ON 7.70-8.59 vs OFF 7.71-8.52,
均值 ~8.1 vs ~8.0;两者都有 8.3-8.5 波动点——之前"OFF 紧凑 7.74-7.82"
是彼时 boot 的运气,今晚同环境 OFF 同样抖)。repeat ON 4.70/4.70 略优
于 OFF 4.73-4.82。**V27 在两个形状上都实现统计意义上的 ON=OFF,
repeat 均反超**。方差源(rank-skew retrieve)是共享的,不是 ON 独有。

**32K 首启注意**:reboot 后第一次 32K cold_store 曾把 engine 打死
(shm broadcast TimeoutError,worker 无 traceback无 OOM——EP rank 间
cold_store 433s 分歧超时)。重启容器后同脚本正常。这与 CSA 代码无关
(seeding 阶段 walker 尚未参与),记录为 32K cold_store 的已知脆弱点。

剩余差距:hit-1(首次命中,签名尚未建立)仍付全额 walker 税;真正
反超 OFF 还需 csrc zero-copy(读路径首跑就免 scatter)。

## V26(已证伪,诊断价值):resident-chunk 同行跳过

SM 争用模型的直接推论:稳态 hit(同一 480K 前缀反复命中)每次都重复
1.5GB 读 + scatter,但 **前缀 chunk 的 (key, physical rows) 两次完全相同
时,K-cache 行里的字节还在**——retrieve 在 filter 下不写 csa_attention_kv
组,decode 只写新分配行,串行负载下无人弄脏这些行。

实现([csa_attention_kv_prefetch_manager.py](../../../lmcache/v1/csa_attention_kv_prefetch_manager.py)):
- walker 完整落地一个 chunk 后记录签名 `(layer, key) -> (first_block,
  n_blocks, byte_offset, physical_rows)`;
- 下次注册时逐 chunk 比对签名,命中者:bitmap 置位、arm/walk 全跳过
  (不读不 scatter)、全命中层直接 fully_resident;
- 未匹配的旧签名立即丢弃(残留声明活不过一个可能弄脏行的请求);
- 行退化路径(无 slot_mapping)永不记录;并发生产负载用
  `LMCACHE_CSA_WALKER_RESIDENT_SKIP=0` 关闭。

预期:hit-2+ 的 walker 读+scatter 全免 → ON 稳态应落到 OFF 线以下。
若 vLLM 分配器跨请求行号不稳定则 matched=0,回退 V25 行为(无损)。

**V26 首跑(未重启环境)作废**:hit-1 12.2s / hit-2+ 13.9s 且
resident_lines=0,walker 线程在 `torch.cuda.set_stream` 处抛
`CUDA error: out of memory`(UVM 残留 ~19GB/卡 挤压到连 stream 切换的
内部分配都失败)。这是 UVM 泄漏症状的新面貌:之前表现为 Triton OOM in
forward,这次直接打死 walker 线程(gate 全部走 1s miss grace + 自读,
所以 hit 反而比 V5 慢 5s)。判据不变:跑实验前 `memory.used` 必须
< 10GB/卡,否则先 reboot。

**V26-16K 重启后干净跑**:r1 hits 7.01/7.05/7.05/7.27,r2 hits
7.09/7.08/6.23/7.05/7.10 —— 与 V25 持平(无回归无 crash),但
**resident-skip 未触发(matched=0)**。当时部署的 build 对 matched=0
静默;已加诊断日志(prev_sigs/key_hits/row_mismatches + 首个不匹配样本)
再跑 32K,区分两种可能:(a) chunk key 每次命中都变(不应该——内容哈希);
(b) vLLM 块分配器给同一前缀分配了不同物理行(slot_mapping 变化)。
若是 (b),下一步是 HBM 内行到行拷贝(GPU-GPU copy 替代 NVMe 重读,
仍省 NVMe 但 scatter 成本回来)或研究分配器行为。

## V25 最终判决与 SM 争用模型

V25(fast-gate 跳过 miss 扫描)结果:correction 从 46ms→0.001ms/层,
**但墙钟没动**(32K 稳态 8.5-9.0s)——那 1s 扫描原本就与其他等待重叠。

**时间轴分解揭示的最终物理**(TTFT_STAGE + MoE timing):
- retrieve(rank 偏斜 2.64-3.28s,EP 等最慢)
- MoE 总量仅 0.52s/步;forward 大头在 attention/indexer
- **ON 的 forward 比 OFF 慢 ~0.8s = walker scatter(1.5GB index_copy_)
  在 compute 阶段与计算 kernel 抢 SM**
- OFF 的同样 scatter 在 retrieve 阶段做(NVMe 等待间隙,SM 空闲)= 免费

**结论**:ON 省同步读时间(-0.4s)但付 compute 期 SM 争用(+0.8s),净亏。
"读藏进计算"藏得掉 NVMe 时间,藏不掉 scatter 的 GPU 时间。

里程碑:V25 r2 hit-1 = **7.763s < OFF 7.78s**(首个跑赢数据点,
warm 条件下可赢);但方差大(7.76-9.01 vs OFF 紧凑 7.74-7.82)。

## 数据矩阵(V25 架构,全部同代码同环境)

| 形状 | ON 稳态 | OFF 新鲜基线 | gap |
|---|---|---|---|
| 480K+32K | 8.5-9.0(best 7.76) | 7.74-7.82 | +0.8-1.1 |
| 480K+16K | 6.80-7.48(best 6.12) | 6.04-6.68 | +0.6-0.9 |

两形状 gap 都 ≈ 0.7-1.0s(scatter 体积相同 1.5GB)——SM 争用模型自洽。
warm 最优点(ON 7.76@32K / 6.12@16K)都能打到 OFF 线,但方差是 ON 的
固有属性(walker/scatter 时点抖动 + rank 偏斜放大)。

## 通往真赢的最后一扇门(超出 Python 层)

scatter 的 SM 成本只有一种消法:**NVMe DMA 直写 K-cache 目标物理行**
(zero-copy,不经 staging + index_copy_)。Tutti SGL(sgl_supported=0x000f0001)
按行散射正是为此;需要 csrc 改造(GPU-centric 计划阶段 3)。
Python 层已到收敛点:每一毫秒的 gap 都有名有姓。


## 公平对照(同代码同环境,32K 增量,480K base)

| | ON V24 | OFF(新鲜基线) |
|---|---|---|
| hit 稳态 | 8.57-8.9s | **7.74-7.82s** |
| retrieve | 6.5GB @2.43GB/s = 2.7s | 12.0GB @3.85GB/s = 3.1s |
| walker | 2.07s 全藏(层 44-108ms,gate 零阻塞) | — |
| store | ~135ms | ~133ms |

**悖论**:ON 同步读更少更快 + walker 全藏,但 hit 反而慢 0.8s。
差距 = ON 独有的 per-hit 固定税:
1. correction miss_filter:46ms × 21 层 ≈ **1.0s**(33M 条目 top-K 扫描,
   bulk 全量落地后纯属浪费)← **V25 修:layer_fully_resident 快速门**
2. 注册/arm:1874 chunks × 21 层 Python 构建 + 4 万块 pending 标记
3. prefix-view 读 2.43GB/s vs 整 chunk 读 3.85GB/s(filter 的 IO 效率损失)

## V24 架构(已固化,勿回退)

- walker **armed at register, started at first gate**(compute 阶段队列空闲)
- 单层单 Tutti 调用(任何多层混合 scatter 都会复活 illegal-access 竞态,
  V10/V22 两次验证)
- retrieve whole-call 锁(逐批锁会碎片化到 2GB/s)
- ensure_lba_cache 身份短路(每批重排 3 万 extent 曾吃掉 2.8s)

## 时间线真相(V19-V23 五个版本的教训)

hit 请求 = [retrieve 全部步] → [compute 增量]。NVMe 队列在 compute 阶段
完全空闲。walker 在 register(retrieve 开始)启动必然打架;
在 first gate(compute 开始)启动天然无争抢。
