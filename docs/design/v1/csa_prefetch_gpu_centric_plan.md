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

### 阶段 2.5(V28,修订版):HCA 组 walker 化(真实最大杠杆)

> **2026-07-08 修订**:早先版本此节写"10.7GB 主 KV"——**错误**。
> DSV4_GROUP_TABLE 实测(每 256-token chunk 的 8 个组):

| 组 | role | 层数 | hidden | bytes/chunk | 存储 |
|---|---|---|---|---|---|
| 0 | csa_indexer_cache | 21 | 132 | 0.71MB | 每 chunk |
| 1 | csa_attention_kv | 21 | 584 | 3.14MB | 每 chunk(walker 已接管) |
| 2 | hca_attention_kv | 20 | 584 | 2.99MB | 每 chunk |
| 3+4 | swa_cache | 43 | 584 | 6.4MB | 仅尾 chunk |
| 5-7 | compressor_state(fp32) | 62 | — | 76MB | 仅尾 chunk |

> DSv4-Flash 62 层 = SWA 43 + CSA 21 + HCA 20 混合注意力,**没有
> "MLA 主 latent KV"组**。非尾 chunk 存储载荷 = 0.71+3.14+2.99 =
> 6.84MB(与 store 日志逐字节吻合);ON 的剩余同步 retrieve 6.9GB =
> **HCA 5.6GB(84%)+ indexer 1.3GB(16%)**。

**V28 真实做法 = 把已收敛的 CSA walker/gate/搬迁架构扩展到 HCA 组**:
1. `_dsv4_retrieve_shapes_for_range` 在 flag 下对 hca_attention_kv
   也 zero-shape(indexer cache 1.3GB 暂留同步——它是 gate 之前就要
   的元数据,处理顺序敏感,第二阶段再动);
2. `_dsv4_build_csa_attention_kv_chunks` 参数化 role:同一记录里 HCA
   组的 layer_byte_offset = 前缀组字节和(indexer+csa)+ 层内 stride,
   结构与 CSA 完全同源;HCA 20 层加入 walker 的层序循环(按 transformer
   层号与 CSA 层交错排序,单层单调用不变);
3. HCA 层的 gate:HCA attention 的 kv_cache 张量(compress 128:1)
   注册进 manager,层入口用与 CSA 相同的 pending/notify;HCA 层没有
   indexer forward 可 patch → 用 vLLM 每层 `wait_for_layer_load`
    钩子(可行性已验证,见下);
4. V27 搬迁签名自然覆盖 HCA chunk(同一签名机制,零新代码);
5. 预期:ON 同步 retrieve 6.9→1.3GB(-80%),读时间 ~1.7s→~0.4s;
   扣 walker scatter SM 税,48K 形状净赢 **~1.0-1.3s/hit**;fp32
   compressor(尾 chunk 76MB)与 swa 不动。
6. 注意与既有 `LMCACHE_DSV4_DEFER_HCA_TO_MOE`/HCAPrefetchManager
   (独立未验证子系统,配置全关)划清边界:V28 不启用它,走 CSA
   walker 的同一条代码路径。

环境开关:`LMCACHE_DSV4_HCA_WALKER=1`(默认 0 = V27 行为)。

**V28 实现地图**(修订):
- **HCA 粒度差异(实现前必读)**:CSA cr=4 → 256-token chunk = 64 压缩
  行 = 恰好 1 个 vLLM 块(bs=64),chunk↔块 1:1,scatter 可整块
  `index_copy_`。**HCA cr=128 → 256-token chunk 只有 2 压缩行**,而
  vLLM HCA 组 bs=8(HMA g4=20L/bs=8),即 **4 个 chunk 共享 1 个物理
  块**。因此:
  - scatter 必须按"压缩行"粒度:`kv_cache.view(num_blocks*8, 584)`,
    行 id = 物理块 id×8 + 块内槽位(槽位 = 压缩条目序号 % 8);
  - `physical_block_ids` 语义换成 per-compressed-entry 的行 id
    (slot_mapping[pos]//(128*8) 给物理块,pos 按 128-token 步进);
  - manager 的 `register_layer` 校验 shape[1]==64 需放宽为 per-layer
    `compressed_block_size`(state 字段已存在,只是构造校验写死);
  - 读侧不变:HCA slab 在记录内同样是 per-layer 连续(前缀和寻址)。
- [cache_engine.py](../../../lmcache/v1/cache_engine.py):
  `_dsv4_retrieve_shapes_for_range` 加 HCA zero-shape 分支;
  `_dsv4_build_csa_attention_kv_chunks` 抽出按 role 的通用版本,
  给 HCA 组产 chunks_by_layer(注意 HCA cr=128 → tokens_per_block
  = 128*64,physical row 换算随 cr 变);
- [csa_attention_kv_prefetch_manager.py](../../../lmcache/v1/csa_attention_kv_prefetch_manager.py):
  `register_layer` 已按 layer_id 泛化,HCA 层直接注册(token_bytes
  同 584);walker 层序循环把 HCA 层并入;gate 对无 indexer 的 HCA
  层暴露 `wait_for_layer(layer_id)` 公开方法;
- [vllm_v1_adapter.py](../../../lmcache/integration/vllm/vllm_v1_adapter.py):
  attach 时收集 HCA 层的 attention.kv_cache;`wait_for_layer_load`
  (layerwise_retrievers 为空时直通)接 manager.wait_for_layer;
  **可行性已验证(部署镜像 20260528_0630)**:`attention.py:753` 与
  `mla_attention.py:1046` 均被 `@maybe_transfer_kv_layer` 包裹,每层
  入口调 `connector.wait_for_layer_load`——gate 不需要改 vLLM。

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
