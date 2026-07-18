# CSA Prefetch 20260706 通宵会战:形状阶梯、根因链与三方对照

**日期**: 2026-07-06/07 通宵
**环境**: GPU002, DeepSeek V4 **Flash** (43 层, 21 CSA 层), TP=8/EP=8, util=0.72,
默认 KV profiling 关闭 (手动 slot=8MB, capacity=48000, region 1500G/盘)。
**协议**: base 冷写 → 90s → hit×5 (共享 base, 各自新增量) → repeat×2, 每遍间 30s。

## 一、最终对照表 (480K base, hit 稳态中位数)

| 增量 | OFF | ON-bulk (提前读) | ON-proxy (逐层预测) |
|---|---|---|---|
| 8K  | **6.35s** | 7.1s (V11, GREEN×2) | 未测 |
| 16K | **6.1s**  | 崩 (illegal access) | 13.1s (rep1 GREEN; rep2 rank0 OOM) |
| 32K | **7.8-8.5s** | 11.8-16.1s (V14, rep1 完成 rep2 崩) | 18.9-21.1s (rep1, 0 错误) |

ON-proxy 在 32K 比 16K 更差 (proxy 读放大随增量步数增长: 每个计算步每层
重发全前缀 predicted 读, fire-once 只挡同请求同层, 不挡跨步重 fire)。

512K 全长 (128K+384K 形状, cap 修复后): OFF hit 36.5s / repeat 6.2s;
ON-bulk hit 37.9s / repeat 8.0s (V11 架构, GREEN)。

## 二、今晚修掉的根因链 (全部有数据实锤)

1. **容量 env 断链**: `LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY` 不在
   run_container.sh 的 `-e` 列表 → 容器内永远 12000 槽 → 一遍实验写爆
   (poolfull 43984-159920) → lookup 截断 (Retrieved 225K/384K)、repeat 全量重算
   33-39s。修复后 repeat 6.2s、512K 全量认领。**教训: 任何新 env 必须验证
   容器内 printenv。** (同款问题再犯一次: BULK_PREDICTED, 同修复)
2. **池三重放大**: 32MB 定长槽装 6.84MB 对象 (浪费 78%) × 每 chunk 3 对象
   (主 KV + hca_deferred + slab) × 槽位永不回收 (`_next_slot` 单调)。
   slot=8MB 后浪费降至 15%。**槽回收仍未实现** (设计缺陷, 排期)。
3. **extent 线性扫描 O(keys×extents)**: 合成池路径单路径挂 3 万+ extent,
   每 byte_range 全表扫 → ON retrieve build_ms=190ms/批 vs OFF 10ms,
   吞吐 1.34GB/s (盘明明能 2.9)。**bisect 排序索引** (注册时构建) 后
   build_ms=0.02-0.17ms, ON retrieve 3.35s < OFF 4.2s (CSA filter 少读一半
   字节的收益首次兑现)。这正是 Tutti 论文 "CPU O(layer) not O(layer×blocks)"
   原则; 我们的集成层此前违反。
4. **proxy 预测在 prefill 无选择性**: 64K query × top-512 的并集覆盖全前缀
   (candidates=1499/1499)。每层各读一遍全前缀 → 11 倍读放大 (963s 累积读)。
   合成 pangram 语料把选择性抹平, 真实语料未测。→ 催生 bulk 提前读架构。
5. **miss 等待 vs chunk-major walker 完成序死锁**: walker 按 chunk 序走,
   layer L 的最后块在 walk 结束才落地; miss 路径等 30s → forward 钉死 →
   EP 发散 → watchdog 杀引擎。改 0.15s 宽限 + 自读。
6. **重复读放大 (V13 引入)**: 自读绕过 bitmap 过滤传全量 candidates →
   每层重读全 slab (new=1875×21)。V15b 修正: 全量传入但依赖 bitmap 过滤,
   只有真在途块会重复。

## 三、未解问题 (阻塞 bulk 模式)

**illegal access 家族 (bulk walker 独有)**: 在 16K/32K 增量 (hit 有 2 个计算步)
时高概率触发, 8K 单步不触发。已排除: 流排序 (V14 同流上传/V15 io_stream 复用
均未根治)、torch.cat 分配 (V11 已移除)。现场特征: store_raw H2D 与 walker
scatter 并发期间, NCCL watchdog/MoE sync 处报 async illegal access。
**下一步取证**: CUDA_LAUNCH_BLOCKING=1 + 逐 kernel 校验 dst_rows 边界;
重点怀疑跨层大读的 `base` 偏移跨 record 边界时 staging 布局与 scatter 计划
不一致 (usable 裁剪路径)。

**rank0 OOM (proxy 路径, 496-512K ctx)**: proxy fire 的 GPU 打分在 480K 前缀上
分配 4GiB pad → rank0 爆。**系统性: P16/P32 的 rep2 均在同点死** (rep1 存活) —
fire 打分缓冲跨请求累积不释放, 第二个请求叠加时越界。修复方向: fire 打分
microbatch 显存上限, 或 bulk 模式下直接禁用 proxy 打分 (无用计算)。

## 四、理论模型 (实测校准)

TTFT_OFF = 读(同步) + 算;TTFT_ON 下界 = max(读, 算) + 首层延迟。
**赢面 = min(读, 算) − ON 开销**。
- 实测校准: Tutti 聚合读 2.9GB/s (共享会话上限), Flash 计算 ~0.22ms/K token/层。
- 8K 增量: 算 1.8s < walker 3.3s → 藏不满, ON 理论下界 ~7s (实测 7.1 ✓)
- vLLM chunked prefill 自带块间读算流水 → 增量 >64K 时 OFF 已部分重叠,
  ON 的净赢面被上游吃掉大半 (W 形状实测: OFF 17.7 vs ON 19.1)。
- **ON 的真主场**: 增量 16K-64K (单/双步, 上游流水失效, 算窗 ≥ walker) —
  理论赢 10-25%。但正是该区间触发 bulk 崩溃 → 修掉 illegal access 才能兑现。

## 五、下一步 (优先级)

1. CUDA_LAUNCH_BLOCKING 取证修 bulk illegal access (赢面兑现的唯一阻塞)
2. proxy fire 的 GPU 打分显存约束 (microbatch 上限或直接砍掉 proxy: bulk
   模式下无用)
3. 槽位回收 (free-list)
4. 真实语料重测 predicted 选择性 (决定 proxy 路径去留)
5. async store (store_raw 仍在 TTFT 内的场景)
