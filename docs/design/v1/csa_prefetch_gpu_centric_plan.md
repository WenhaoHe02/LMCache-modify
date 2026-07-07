# CSA Prefetch: GPU-Centric 重构目标与做法

> 与 `01-idea-and-problems.md` 配套。本文档定义"我们要做什么、为什么",
> 不再记录实验流水账。

## 目标(不可妥协)

**让 CSA attention-KV 的 NVMe 读路径成为 GPU-centric 的,对齐 Tutti 论文
([arXiv:2605.03375](https://arxiv.org/abs/2605.03375))的设计原则。**

判据:GPU→SSD 聚合读带宽达到**单盘 Tutti 裸测水平 ×盘数**(单盘曾测
1.49 GB/s,8 盘聚合应 ≥10 GB/s,当前仅 2.9 GB/s——差 3.4 倍)。在这个
带宽下,480K base 的 12.2GB retrieve < 1.5s,ON 才有空间把读藏进计算。

## Tutti 论文的原则 vs 当前实现的差距

| 论文原则 | 当前实现 | 差距 |
|---|---|---|
| **CPU-prepared, GPU-executed**:CPU 每层一次异步加载 I/O kernel,SQE 生成/提交全在 GPU | host 每批 Python 构建 keys/metas/ranges → 上传参数张量 → submit kernel → poll kernel → sync → wrap MemoryObj,host 介入 ~6 次/批 | host-driven 批处理,GPU 不自驱动 |
| **IOCB 批量化**:一个 SQE 打包 2048 IOCTX | 每个 NVMe 命令一个 SQE(单段 SGL) | 命令粒度细,SQE 利用率低 |
| **CPU O(layer)** | bulk walker O(chunks) 遍历,retrieve O(keys) | 远超 O(layer) |
| **P2P 映射表预计算复用** | staging cudaMalloc + IOVA 已有 ✓ | 已对齐 |
| **SGL not PRP** | 用了单段 slba+byte_len,不是真多段 SGL | 未用满 SGL |
| **索引 off data path** | LBA cache 每次 bisect 查询在 data path | 在 data path |

## 带宽低的根因(实测拆解)

retrieve 12.2GB / 4.2s = 2.9 GB/s。每批 `batch_detail`:
- `poll_sync_ms ≈ 36ms`(NVMe 真实 DMA,有用)
- `build_ms + wrap_ms + persist_ms ≈ 150ms`(host 构建/包装,纯开销)
- **NVMe 75% 时间空转**(等 host 准备下一批)

staging pool = 4×128MB = 512MB,12.2GB 要 ~24 批 × 150ms host gap = **3.6s 浪费在 host**。
单盘 1.49 GB/s 时 host 介入极少(一批读完),所以快。

## 做法(分阶段,每阶段都要带宽提升)

### 阶段 1:host 介入最小化(立即做,不改 csrc)

1. **staging pool 4→16 槽 × 256MB = 4GB**:一批能装 ~5GB 数据,12.2GB
   只要 3 批,host 介入次数降 8 倍。显存代价 4GB/卡(util 已留余量)。
2. **层序完成 walker(V19)**:walker 按 layer 走(layer 2 全落地 → layer 4
   → ...),每层把所有 chunk 的该层 slab 合成**一批大读**(~90MB)。
   - 每层一次 `load_chunks_to_hbm` 调用 = host 介入一次,21 层 = 21 次
     (不是 1906 次 chunk 遍历)
   - layer L 落地后通知 gate,forward 永不等整个 walk
   - 这把 host 介入从 O(chunks) 降到 O(layers),对齐论文
3. **resident-chunk 签名跳过(V26,取代原 repeat 跳过)**:walker 完整落
   地一个 chunk 后记录 `(layer, key) → (blocks, offset, physical_rows)`
   签名;下一请求注册时签名完全相同的 chunk 直接标记 resident——不读、
   不 scatter(SM 争用税 ~0.8s 全免)。filter 下 retrieve 不写
   csa_attention_kv 组、decode 只写新行,串行负载下行内容不变。未匹配
   签名立即丢弃;`LMCACHE_CSA_WALKER_RESIDENT_SKIP=0` 可关。这同时覆盖
   了 repeat 全命中场景(全部 chunk 均 resident → walker 空转秒退)。

预期:retrieve 4.2s → ~2.0s(带宽 ~6 GB/s),walker 21 次调用每次 ~50ms
host gap,总 ~1s,藏进计算。ON hit 目标 < OFF。

### 阶段 2:真正的 GPU-centric 提交(改 loader,不改 csrc)

4. **整请求一次提交**:retrieve/walker 不再"批"调用,而是把整个请求的
   所有 (slba, byte_len, staging_offset) 三元组一次性上传成一个 GPU 张量,
   一次 `tutti_submit_batch_sgl_read` 提交全部,一次 poll。host 只介入 2 次
   (上传参数 + sync),不是 24 次。
5. **staging 复用环形缓冲**:批与批之间不重新分配,用环形游标,减少
   cudaMalloc/cache 抖动。

预期:retrieve 2.0s → ~1.2s(带宽 ~10 GB/s,逼近单盘×8 上限)。

### 阶段 3:对齐论文 IOCB(改 csrc,长期)

6. **IOCB 批量化**:csrc 的 `k_submit_batch_sgl_read` 改成每个 SQE 打包
   多个 IOCTX(论文 2048 个),减少 SQE 数量。
7. **GPU 自驱动 poll**:poll kernel 不由 host 发起,而是常驻 GPU 上一
   个轻量 spin,host 只 wait event。

这是论文的终态,但工程量大,放最后。

## 不做的事(明确排除)

- **不再在 host 锁/流/延迟上打补丁**:V12-V18 的锁粒度、流选择、延迟
  起跑都是治标。根因是 host 介入次数,不是调度。
- **不再做 proxy 逐层预测**:prefill 选择性在合成语料饱和,真实语料未
  测;在 GPU-centric 读带宽拉起来之前,proxy 的"省字节"收益(最多 50%)
  抵不过 host 开销。bulk 全读 + GPU-centric 高带宽是更稳的路径。
- **不改 csrc 除非阶段 3**:阶段 1-2 在 Python 层就能拿到 3-5 倍带宽。

## 验证

每个阶段的判据:
- `TUTTI_PROFILE load_total total_ms` 下降(直接看 retrieve 墙钟)
- `batch_detail build_ms + wrap_ms` 占比下降(host 开销占比)
- 聚合带宽 = bytes / total_ms 趋近单盘×8
- ON hit < OFF hit(最终判据)

实验形状:480K base + 16K/32K 增量(主场)。
