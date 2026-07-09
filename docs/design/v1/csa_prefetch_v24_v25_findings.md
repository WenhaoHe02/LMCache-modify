# V24/V25 关键数据(2026-07-08 追加)

## 会议任务 2 最终结论(2026-07-10,生产栈 torch.profiler trace)

**对象**:vLLM+LMCache 完整服务栈,预测臂,fused kernel,62 层。
钩子:CSA gate 内 torch.profiler 一次性捕获一个 chunked-prefill
microbatch 的完整 forward(LMCACHE_TORCH_PROFILE,SKIP=3)。两个形状
各一份 chrome trace(rank0),已下载到本地 `Desktop\vllm_traces\`。
注意:8 个 worker 的 RANK 均为 0,trace 文件名需 pid 后缀否则互相覆盖
(已修)。

**采集窗口口径(重要,勿混淆)**:
- **64K trace = 真稳态 hit**:SKIP=3 落在 hit-2 的第一个 chunk
  (65536 token 全量,hit-2 墙钟 1.32s,run 全绿:cold 8.9s/hit-1
  21.9s/hit-2..5 1.3-3.3s)。
- **480K trace = cold_store 预填充的第 4 个 chunk**(65536 token,
  KV 深度 ~197K→262K),**不是 hit**:480K cold_store 有 8 个 chunk,
  SKIP=3 时捕获窗口尚在 seeding。该 run 在 trace 导出后 ~1 分钟死于
  illegal access(store 写路径,BULK=0 + profiler 导出后首见),
  cold_store 返回 500,hits 全 exception。因此 480K 表中的 k_poll
  自旋属于 store 写路径,不是 hit 读路径;indexer 打分占比在真
  480K hit(深度 48 万)只会**更高**——21.9% 是深度 ~26 万时的值。

**64K+2K forward(GPU kernel 总时间 810ms,10255 kernels)**:
| kernel | ms | 占比 |
|---|---|---|
| multimem_all_reduce | 435 | **53.7%** |
| k_poll_batch(Tutti NVMe 轮询自旋) | 148 | **18.3%** |
| index_elementwise(KV scatter 类) | 55 | 6.8% |
| Marlin MoE(fused experts) | 27 | 3.4% |
| deep_gemm mqa_logits(indexer 打分) | 23 | 2.8% |
| sparse_attn | 15 | 1.9% |

**480K+2K forward(GPU kernel 总时间 5193ms,6129 kernels)**:
| kernel | ms | 占比 |
|---|---|---|
| nccl all_reduce | 1270 | **24.5%** |
| **deep_gemm mqa_logits(indexer 打分,672 次)** | **1136** | **21.9%** |
| sparse_attn | 764 | 14.7% |
| Marlin MoE | 385 | 7.4% |
| k_poll_batch(Tutti 轮询) | 330 | 6.4% |
| topKPerRowPrefill(672 次) | 186 | 3.6% |
| dequantize_gather_k | 129 | 2.5% |

**生产结论**:
1. **indexer 打分(mqa_logits)在 480K 是第二大 GPU 消耗**(1136ms
   =21.9%,672 次调用 = 21 层×32 microbatch)——**indexer 成本随上下文
   线性增长,长上下文下它本身就是重负载**,"launch 开销"是次要项
   (topk+rope+quant 合计 ~250ms);graph 化收益远小于减少打分量
   (跨层共享直接砍打分次数,一石二鸟)。
2. **all_reduce 双形状都是第一**(53.7%/24.5%)——TP=8 的通信税;
   indexer 的 index_score all_reduce 是其中一部分。
   **⚠ 此条被 nsys 交叉验证推翻,见下节:53.7% 是 torch.profiler
   八进程互相拖慢造成的假象,真实 all_reduce 在 64K 稳态只占 ~15%
   busy。**
3. **Tutti poll 自旋占 GPU 6-18%**——NVMe 等待烧在 GPU kernel 上
   (64K hit 里是读路径的等待,480K cold_store 里是写路径的等待);
   此前的 host 口径没看到这块。GPU-driven poll(csrc 计划阶段 3)
   或减少 IO 次数都能砍它。
4. 参考实现(官方 model.py)的 profile 因无 fused kernel/无 LMCache
   被判定不可用于生产结论(用户指出),仅保留 indexer 算子链结构
   参考(~161 launches/call)。

## 会议任务 2 补充:nsys 交叉验证(2026-07-10,同一 gate 状态机)

**方法**:manager 的 profile 钩子加 `LMCACHE_NSYS_CAPTURE=1` 分支
(cudaProfilerStart/Stop,与 torch 口径完全相同的 skip/gate 状态机),
宿主机 nsight-systems 2025.3.2 挂载进容器,`nsys profile
--capture-range=cudaProfilerApi -s none` 包 api_server。**一份 capture
同时覆盖全部 8 张 GPU**(torch trace 只能看单进程)。64K+2K,干净
reboot 后全绿 run(cold 10.5s/hit-2 1.25s/hit-5 1.15s,全部 200)。
报告:`Desktop\vllm_traces\prod64k.nsys-rep`(gpu002 镜像
`vprof_archive_20260710/`)。

**capture 窗口结构**(dev0,gate→gate 语义):cluster0 = hit-2 的
完整 forward(span 614ms,busy 326ms,占用率 53%);中间 14.9s 空闲
= 请求间 sleep;cluster1 = hit-3 的 retrieve 段+前两层(span 615ms,
busy 108ms:k_poll 2 次 85ms 大读 + multi_layer_block_transfer 500 次
scatter 10ms)。

**逐 kernel 对照(dev0,一个 forward)**:

| kernel | torch trace | nsys | 判定 |
|---|---|---|---|
| k_poll_batch | 148ms | 150.3ms | ✓ 吻合 |
| index_elementwise(scatter 家族) | 55ms | 55.1ms | ✓ 吻合 |
| Marlin MoE | 27ms | 28.1ms | ✓ 吻合 |
| mqa_logits(indexer 打分) | 23ms | 23.1ms | ✓ 吻合 |
| sparse_attn | 15ms | 15.5ms | ✓ 吻合 |
| **multimem_all_reduce** | **435ms** | **53.9ms** | **✗ torch 虚高 8×** |
| GPU busy 合计 | 810ms | 433.5ms | 差额≈all_reduce 虚增部分 |

**解读**:multimem all_reduce kernel 是自旋等 peer 的——8 个 worker
同时开 torch.profiler,CUPTI 开销让各 rank 步调错开,rank0 的
all_reduce kernel 把等待时间全记在自己头上(0.6ms/次 → 5ms/次)。
nsys 用 `-s none` + 全局一份 session,开销极小,54ms(82 次,
0.6ms/次)才是真值。**结论修正:64K 稳态 all_reduce 只占 busy 的
~15%(不是 53.7%),第一大 GPU 消耗实为 Tutti k_poll 自旋;其余
所有 torch 数字(indexer/MoE/attention/scatter/poll)被 nsys 双口径
确认可信。** 480K torch 表的 nccl 1270ms 同理存疑偏高,mqa_logits
的真实占比只会比 21.9% 更高。

**dev0 forward 内部构成(nsys,busy 326ms)**:k_poll 65ms(预测读
等待)、all_reduce 49ms、scatter 家族 ~49ms、Marlin 27ms、mqa 22.5ms、
sparse_attn 15ms、dequant_gather 14ms。占用率仅 53%——forward 期
仍有近半 SM 空闲,散块预测读的时延(不是带宽)是瓶颈的另一证据。

## 会议任务 3 最终结论(2026-07-09,官方推理代码离线验证)

**方法**:官方 `inference/model.py`(43 层,topk=512)一次 7K prefill,
捕获每层 hc_post 残差 + 每 CSA 层真 top-K(后 192 行);事后用不同
年龄的残差过目标层自己的 proxy 链(hc_pre→attn_norm→索引打分,对着
真 forward 的 indexer kv_cache),**逐行 recall**(每 query token 单独
比,避开并集饱和)。口径与生产时序一致:预测在前层 MoE 窗口内发,
可用状态是前层 hc_post(未过该层 MoE)。

**校验锚点**:age0(本层真输入)= 99.8% 证明评分器正确;age1(L-1
残差,现行方案)= 92.7% 与历史生产数据 93% 吻合。

| 残差年龄 | 语义 | recall mean | min(最差层) |
|---|---|---|---|
| age1 | 现行方案(L-1) | **92.7%** | 83.7%(L2) |
| age3 | 隔 1 个 CSA 层 | 82.2% | 57.2%(L4) |
| age5 | 隔 2 个 | 77.0% | 64.3% |
| age7 | 隔 3 个 | 71.9% | 57.5% |

**深浅分层结构(核心发现)**:浅层(L2-16)age3 即掉到 57-81%,
跨层共享不可行;**深层(L26-42)age3 仍 86-95%、age7 仍 75-87%,
隔 2-4 层共享可行**。→ GLM 式全模型统一共享在 DSv4 不成立,
**深度分段共享成立**:浅层逐层预测(92.7% 够用,瓶颈在窗口),
深层(~14 个 CSA 层)隔 2-4 层共享,窗口预算 ×2-4 →结合任务 1
预算线,深层预取在 480K 进入可行域。

数据:gpu002 `/tmp/offxl/`(per-layer JSON);脚本 `offline_xlayer.py`。
无效号数勘误:此前"miss 口径 100% 准确率"与"块并集口径 99.7% 跨层
覆盖"两个数字均为测量假象(池覆盖≠预测命中;prefill 并集饱和),
以本节为准。

## 会议任务 1 最终结论(2026-07-09,全实测口径)

**两条实测线**(修复计时污染 + 重启后干净矩阵):
- 碎块读:`T_load(N) = 0.42 + 0.021·N` ms(N=块数;21496 点,分桶
  p50 残差 <1.5ms;**每块 21µs,碎读有效带宽 ~1.7GB/s,大连续 IO 的
  11GB/s 不适用于 per-层散块读**——此前用 11GB/s 外推是错误,已当面
  被用户纠正)
- MoE 窗口(全读组无干扰,fused 路径,stream-local sync):
  `T_MoE_p50(Q) = 1.80 + 0.219·Q_k` ms(6 桶 n=1720/桶,最大残差
  0.41ms,适用 Q≤8K;p90≈1.35×p50;>10K 外推低估 ~30%)

**预算判定**:`可藏 ⟺ N ≤ 66 + 10.4·Q_k(块/层)`

| 场景 | 每层块数 | 结论 |
|---|---|---|
| 64K 全读/预测(实测饱和 s≈100%) | 250 | 超预算 3.8×,需 Q≥17.7K |
| 480K 预测(实测 s=70%) | 1350 | 超预算 20×,不可行 |
| 任意上下文,选中 ≤66 块 | ≤66 | Q=0 恒可藏 |

**端到端互洽**:阈值矩阵 20 个点预测组全部不敌全读组(64K 差
0.1-0.7s,480K 差 1.2-3.4s;gate wait p90=25-27ms、miss p90 高)。
**当前实现下预测方案的有效 Q 阈值不存在**;活路=①固定开销
2.3-3.85ms→<1ms(graph 化)、②窗口×W(跨层共享 indexer,W 待任务 3
实测)、③读排布连续化(碎读 21µs/块 → 大 IO 3.3µs/块,6.5× 空间)。

数据归档:gpu002 `postreboot_threshold_20260708/`(四格矩阵+README)、
`/tmp/moemeas`(MoE 干净窗口)、`/tmp/load_pairs.txt`(载入拟合点)。

## 最终三形状全矩阵(V28,2026-07-08 收官)

| 增量 | OFF(新鲜) | V27 | V28 | V28 repeat vs OFF |
|---|---|---|---|---|
| 16K | 6.04-6.76 | 6.03-6.72(平) | **5.48-6.20(赢)** | **4.04** vs 4.53 |
| 32K | 7.71-8.52 | 7.70-8.59(平) | **7.10-7.98(赢)** | **4.11** vs 4.73 |
| 48K | ~9.6 | ~9.6(平) | 8.9-9.8(平) | **4.35** vs 4.86 |

**V28-32K(winchain,同 boot):r1 hits 8.11/7.10/7.75/7.18,r2
8.59/7.17/7.98/7.19/7.75 —— 中位 ~7.5 vs OFF 中位 ~7.9,32K 也赢
~0.4-0.8s;repeat 4.11/4.14 双双历史级。illegal=0。**

规律(论文叙事):稳态赢的绝对值 ≈ OFF 多读 11.5GB 的时间差,与
compute 无关 → 16K/32K 明确赢,48K 被 compute 稀释;repeat 三形状
全赢(零重算+零读,只剩 ~250ms 搬迁)。

## V28(HCA walker):16K 首次明确跑赢 OFF

V28 = 把 CSA walker/gate/V27 搬迁扩展到 HCA 组(剩余同步读的 84%,
5.6GB)。开关 `LMCACHE_DSV4_HCA_WALKER=1`。三个硬工程问题的解:
- **512B 对齐**(HCA 层 stride 1168B 永不对齐,Tutti 拒读)→ 读窗口
  向下取整 + `payload_skip` 字段,scatter 时跳过;
- **混合粒度**(cr=128,4 chunk 共享 1 个 bs=8 物理块)→ 按压缩条目
  行寻址,physical id = 块×8+槽位;
- **非连续张量**(vLLM HCA K cache 是大 buffer 的切片,view 抛异常)
  → 层状态保持 3-D,全部四条写路径用 (block, slot) 双索引
  `index_put_`;
- **安全联锁**:manager 无 HCA 层注册时自动保留同步读(第一跑注册数
  =0 时靠它避免了脏 KV)。

**V28-48K**(机制验证,compute 主导形状):HCA 20 层全注册,同步
retrieve 3.07→**2.06s(-33%)**,walker HCA 层 7-18ms,repeat **4.35**
(历史最佳),稳态 8.9-9.8 ≈ 基线(compute 稀释);illegal=0,gate 超时=0。

**V28-16K(决定性,retrieve 占比高的形状)**:

| | r1 | r2 | 基线(V27 / OFF) |
|---|---|---|---|
| hits | **5.48**/6.04/6.74/6.01 | **5.59**/6.20/6.17/**5.59** | 6.03-6.72 / 6.04-6.68 |
| repeat | 5.59 | 4.86/**4.04** | 4.43-4.49 / 4.59-5.51 |
| retrieve | 1.93s | — | ~3.0s |

**r2 稳态 5.59-6.20,多数 hit 落在 OFF 下界(6.04)之下;best 5.48;
repeat 4.04 历史最佳。这是全部 28 个版本以来第一次在稳态 hit 上
明确跑赢 OFF。** illegal=0,gate 超时=0。

**同 boot 新鲜 OFF-16K 确认(winchain,无 boot 运气)**:
- OFF r1 hits 6.13/6.08/6.76/6.22/6.10,r2 6.14/6.19/6.07/6.20/6.14
  (repeat 4.53-5.41)
- **V28 十个 hit 中位 ~5.9 vs OFF 二十个 hit 中位 ~6.14;V28 best 5.48
  vs OFF floor 6.07;repeat 4.04 vs 4.53** —— 同机同 boot 同协议,
  胜利成立。

**稳态 hit-5 的逐时间戳 profile(r2, 5.586s,全部测量值)**:
- pre-retrieve(HTTP+tokenize 496K+调度+lookup):**2.84s(51%)**
- retrieve(只剩 indexer 组 1.35GB):**2.52s(45%)**——0.6GB/s,
  1874 个 0.71MB 碎读,比 walker mega-batch 的 11GB/s 慢 18 倍
- 搬迁 262-275ms(retrieve 窗口内,免费);compute+21 gate:0.23s
  (每 gate <0.1ms)
- **下一个最大杠杆:indexer 组合并大读或进 walker,可再省 ~2s,
  稳态可望 ~3.5-4s**。pre-retrieve 的 2.84s 属 API/tokenize 层,
  与 KV 系统无关。

**为什么 ON 能赢(机制)**:OFF 每个 hit 无条件重读 12.8GB;V28 稳态
hit 利用跨请求 HBM 残留(内容没变,只是 vLLM 换了行)——250ms GPU 内
搬迁替代 11.5GB 读,同步路径只剩 1.35GB。前 27 个版本只做到"少读",
但被挪走的字节回来的方式都收税(gate 阻塞/SM 争用);V27/V28 让字节
根本不用回来。边界:依赖前缀复用;冷 miss 退化为 walker 藏读(=OFF
打平);赢的绝对值与 compute 无关(48K 被稀释成 parity 的原因)。

叙事闭环:增量越小(读占比越高)V28 稳态优势越大;增量越大 repeat
优势越大(48K repeat 4.35 vs OFF 4.86)。scatter SM 争用仍是残余
tax(48K 稳态被 compute 稀释),彻底消除 = csrc zero-copy(future)。

## 48K 增量实验(用户问:重算变大 ON 会不会拉开?答:不会,原因结构性)

480K base + 48K 增量(528K 总长,贴 530K 上限),同 boot ON(V27)→
新鲜 OFF 链式:

| | hits(r1) | hits(r2) | repeat |
|---|---|---|---|
| ON V27 | 9.49/9.54/9.54/10.05 | 9.48/9.46/10.66/9.52 | 4.90/4.78 |
| OFF | 9.49/10.12/9.55/9.50/9.53 | 9.54/9.48/10.29/9.58/9.46 | 4.86/4.89 |

**仍是 parity(均值 ~9.6 vs ~9.6),没有随重算变大而拉开。** 搬迁
scaling 正常(43302 chunks,稳态 ~115ms,首个 hit 875ms),illegal=0。

**为什么重算变大不帮 ON**:两臂的执行都是 [同步 retrieve 全部] →
[compute 增量] 串行。CSA filter 只把 csa_attention_kv(1.5GB / 12.2GB
≈ 12%)挪出同步 retrieve(实测 retrieve ON 2.83s vs OFF 3.07s,差
0.24s);剩余 10.7GB 主 KV 在两臂都同步读完才开始 compute。重算加大
只是给两臂同加一个常数。**ON 的理论优势 = 被挪出的字节数,与 compute
大小无关**——想利用大 compute,必须把主 KV retrieve 也藏进 compute
(layerwise/流水 retrieve),那是另一个数量级的改造。

推论(论文角度):当前架构下 CSA 路径的可赢空间被 filter 覆盖率
(12%)封顶为 ~0.3-0.4s/hit;V27 已把这部分从"倒亏 0.9s"做到"打平
+repeat 反超"。下一个杠杆按收益排序:(1) 主 KV retrieve 与 compute
重叠(结构大改,收益 ~3s);(2) csrc zero-copy 免首跑 scatter
(~0.1-0.2s);(3) HCA 组同样 filter+prefetch(~0.2s)。

## V27 正确性疑点(已结案:V27 无罪)

V27-32K r1 的 hit-1..5 首 token 不同曾引发搬迁损坏怀疑。两点排除:

1. **基准的 hit-1..5 prompt 本就各不相同**(run_incremental_v2.py 给每
   个 hit 加 `Variant {i} nonce` 头)——跨 hit 输出不同是合法的,当初
   的"翻转"是误读。r2 全 '178' 反而是填充文本主导输出的巧合。
2. **同 boot A/B(abtext.sh,env 透传修复后干净跑)**:
   - cycle-A(RESIDENT_SKIP=0,V25 每 hit 全量 walker):
     hits 6.94/7.08/7.32/7.37,texts ' Now'/'314'/'314'/'314'/' Now',
     repeat '5'/'5'
   - cycle-B(=1,V27 搬迁):hits 6.40/6.00/6.00/6.61,
     texts '202'/'2'/'583'/' Now'/'202',repeat '5'/'5'
   - **两臂行为模式一致**:hit-5 与 repeat(同 prompt)都不同色、
     repeat 两次都稳定 '5'。搬迁开或关没有引入任何新的输出变化模式。
     (注:两 cycle nonce 不同,绝对文本跨 cycle 不可比。)

**A/B 附带产出——同 boot 最干净的性能对照**:唯一变量是搬迁开关,
skip0 稳态 6.94-7.37(V25 水平)vs skip1 稳态 6.00-6.61(OFF 线),
**V27 净赚 ~0.9s/hit,与历史跨 boot 数据完全吻合**。

遗留:严格的 vs-OFF 输出对照做不了(OFF 容器不回传 text);要做需给
OFF 环境加 text 回传或用 logprobs 比对,列为后续项。

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

**hit-1 解剖(V27-32K r1 hit-1=19.3s 的去向)**:retrieve 2.6s 正常;
layer-2 gate correction 在 7/8 rank 上 total=10.76s,但所有被测组件
(wait 0.006ms / drain 0.017ms / miss_filter 32ms / 二次 drain 0.007ms)
都可忽略——**10.6s 在 orig_forward(真 Lightning Indexer op)内部**,
仅每次 boot 后第一个 hit 出现(r2 hit-1 = 7.9s 无此税;TP0 rank 无此税
119ms)。walker 同窗口的 Tutti 读实际只 70ms(group_ms=10754 里 99% 是
等这个 orig_forward 完成后 io_lock 才轮到它)。结论:**每 boot 一次性
的 kernel warmup/JIT 税,与 CSA 读路径无关**,不值得修;基准协议应
丢弃每 boot 的第一个 hit(既有协议已经这么做——r1 hit-1 从不计入稳态)。

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
