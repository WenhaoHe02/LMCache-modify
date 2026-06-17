# Bench Handoff: CSA Speculative Block Prefetch

## 目标

验证 CSA 投机预取效果：

- **Baseline**: 每个 decode step，每个 CSA 层同步读取全部 1 024 个 KV block
- **Speculative**: 只异步读取 delta block（~92 块/step/层，~9% 变化率）
- 预期加速：约 11× I/O 减少；delta reads 与后续层 FFN（~3 300 µs）并行 → 几乎零可见延迟

## 文件

| 文件 | 说明 |
|------|------|
| `lmcache/v1/csa_prefetcher.py` | 核心实现：CSABlockStore + CSAPrefetcher |
| `bench_csa_prefetch.py` | benchmark 脚本（fake mode + real mode）|

## Fake Mode（不需要模型，只测 I/O）

```bash
# 快速验证：4 layers, 50 steps, 4KB blocks
python bench_csa_prefetch.py --mode fake \
    --n-layers 30 --decode-steps 50 \
    --block-size 4096 --n-blocks 8192 \
    --index-topk 1024 --delta-rate 0.09 \
    --io-workers 8 \
    --store-dir /mnt/nvme0/csa_bench_fake
```

预期输出：
- Baseline：每步 ~30 × 1024 × 4KB = 120MB 同步读 → 几百 ms
- Speculative：每步 ~30 × 92 × 4KB = 11MB 异步提交 → 提交本身 <1ms

## Real Model Mode（gpu002，TP=8）

```bash
cd /path/to/LMCache
torchrun --nproc_per_node=8 bench_csa_prefetch.py \
    --mode real \
    --model-path /mnt/nvme0/models/DeepSeek-V4-Pro \
    --store-dir /mnt/nvme0/csa_blocks \
    --prompt-len 32768 --decode-steps 50 \
    --compress-ratio 4 --kv-lora-rank 512 \
    --io-workers 8
```

注意：`--store-dir` 会写入 mock 随机数据（约 30 层 × 8192 blocks × 4KB = ~1GB），
运行前确保 NVMe 有足够空间。

## 集成到真实推理框架

在模型加载后插入：

```python
from lmcache.v1.csa_prefetcher import build_from_model

prefetcher = build_from_model(
    transformer,
    store_dir="/mnt/nvme0/csa_kv",
    compress_ratio=4,
    kv_lora_rank=512,
    max_seq_len=cfg["max_seq_len"],
    io_workers=8,
)
prefetcher.patch_transformer(transformer)  # 猴子补丁 Indexer.forward

# 在每次 decode run 前：
prefetcher.reset()

# 在 decode 结束后查看统计：
prefetcher.print_stats()
```

## 关键数字（预期）

| 指标 | 值 |
|------|----|
| block_size | 4096 B（4×512×2） |
| blocks/layer | 8192（32768//4） |
| topk/layer/step | 1024 |
| delta/layer/step | ~92（9%） |
| 全量 I/O/step | 1024×4KB×30层 = 120 MB |
| delta I/O/step | 92×4KB×30层 = 11 MB |
| FFN window/layer | ~3 300 µs |
| NVMe 顺序带宽 | ~3 GB/s（gpu002） |
| delta I/O 时间 | 11MB / 3GBps = ~3.6 ms ≈ FFN window |

## HC-proxy 投机模式（CSASpecPrefetcher）

在 `real` 模式下加 `--speculative` 启用完整的 HC-proxy 投机：

```bash
torchrun --nproc_per_node=8 bench_csa_prefetch.py \
    --mode real \
    --model-path /mnt/nvme0/models/DeepSeek-V4-Pro \
    --store-dir /mnt/nvme0/csa_blocks \
    --prompt-len 32768 --decode-steps 50 \
    --io-workers 8 \
    --speculative
```

投机流程：
1. Block L forward 开始时：`proxy = attn_norm_L(HC_pre_L(residual_f_{L-1}))`
2. 用 proxy 跑 spec Indexer → `spec_topk`，立即提交 ~1024 个 NVMe 异步读
3. 真实 Indexer 运行后得 `true_topk`，计算 miss = `true_topk − spec_topk`
4. 仅对 miss blocks 补发 fallback reads（~165 blocks，V4-Pro 均值精度 83.8%）

期望输出（新增 accuracy table）：

```
--- CSASpecPrefetcher: HC-proxy accuracy ---
  Layer   Steps   HitRate  FallbackMean
  -----------------------------------------
      2      50     0.567         443.0   ← 已知结构性差层
      4      50     0.929          72.8
     ...
     54      50     0.166         857.3   ← 低精度层需额外预算
  Overall: hit_rate=0.839  fallback/step=165.0
```

低精度层（8、54、56）可配置 `fallback_budget` 放大其 spec 窗口，或在
`CSABlockStore` 里为这几层单独维护更大的 HBM 常驻集。

## 下一步

- [x] 实现 CSASpecPrefetcher（HC-proxy + fallback reads）
- [ ] gpu002 上跑 `--speculative` 模式，确认 hit_rate ≈ 0.84，fallback ≈ 165
- [ ] 验证 per-layer hit_rate 与 test_proxy_accuracy_vs_prefill.py 结果一致
- [ ] 低精度层（8、54、56）特殊处理：扩大 spec window 或增加 HBM 常驻
- [ ] 端到端 e2e 延迟：decode step latency with / without spec prefetch（需 NVMe 真实 I/O）
- [ ] 写 KV eviction + fetch 的完整读路径（prefill 后 evict kv_cache → SSD，1M context 必须）
