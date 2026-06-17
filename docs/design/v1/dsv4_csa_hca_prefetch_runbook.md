# DSv4 CSA/HCA 预取运行手册与必记事项

本文存放 compact 后最容易忘、但做实验必须知道的信息。它不是设计论文；设计语义见
`F:\LMCache\docs\design\v1\dsv4_csa_hca_prefetch_source_of_truth.md`。

## 外部资料入口

- 飞书 wiki：
  `https://icnkc0g3t6cv.feishu.cn/wiki/WZ7Cwg6ZLidtHpkOKiyc3eoFnJf?from=from_copylink`
- 飞书页面需要登录，自动代理通常只能看到登录页。
- 若飞书更新，人工同步到 source-of-truth 文档。

## 当前机器和 IP

当前 DSv4 测试目标：

| 名称 | 地址/路径 |
|---|---|
| 远端 GPU 机 | `gpu002` |
| 内网地址 | `172.16.8.32` |
| 模型路径 | `/mnt/nvme0/models/DeepSeek-V4-Pro` |
| SSD pool | `/mnt/nvme0/csa_ssd_pool/` |
| vLLM endpoint | `http://127.0.0.1:8000` |
| 主要容器 | `dsv4-indexer-ssd` |

聊天截图里的 `10.8.8.3`、`10.8.8.5` 不是当前本地文档确认过的 DSv4 测试路径。
本项目当前使用 `172.16.8.32` 访问 gpu002。

## SSH

本机 `C:\Users\30141\.ssh\config` 已有 `master` 和 `gpu002` alias。

PowerShell 中最稳定的远程执行方式：

```powershell
ssh -o StrictHostKeyChecking=no master "sshpass -p '<gpu002-password>' ssh -o StrictHostKeyChecking=no zbuser02@172.16.8.32 'COMMAND'"
```

敏感口令不要写入仓库文档。若需要密码，查本机私有 memory：
`C:\Users\30141\.claude\projects\f--LMCache\memory\reference_gpu002_ssh.md`。

多行命令用 base64 包一层，避免引号炸掉：

```powershell
$script = @'
set -euo pipefail
sudo docker ps
'@
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($script))
ssh -o StrictHostKeyChecking=no master "sshpass -p '<gpu002-password>' ssh -o StrictHostKeyChecking=no zbuser02@172.16.8.32 'echo $b64 | base64 -d | bash -s'"
```

注意：

- 从 Windows PowerShell 走 `ssh master` 可用。
- `sshpass` 在 master 上可用，不在 Windows 侧用。
- 直接依赖 Windows OpenSSH 的多层 ProxyJump 曾经不稳定。

## 容器

当前活跃容器：

```text
name:     dsv4-256k                      # 256K 上下文验证（2026-05-29）
image:    lmcache/vllm-openai:indexer-ssd-hca-prefetch-decodegate-20260528_0630
          # vLLM: 0.20.2+cu129  LMCache: 0.4.4  (confirmed 2026-06-13)
          # 镜像在 gpu002 /var/lib/docker/ 完好，无需重建；删 tar 不影响本体
entrypoint: /bin/bash（需显式 --entrypoint /bin/bash）
model id: /mnt/nvme0/models/DeepSeek-V4-Pro
endpoint: http://127.0.0.1:8000
关键 vLLM flags:
  --tensor-parallel-size 8
  --enable-expert-parallel          # EP=8，MoE expert 分片到 8 张卡
  --enable-ep-weight-filter
  --all2all-backend allgather_reducescatter
  --max-model-len 262144
  --gpu-memory-utilization 0.92
  --max-num-batched-tokens 8192
  --no-disable-hybrid-kv-cache-manager   ⚠️ 必须！否则 HMA 禁用，KV cache 按全量 7 group 分配（116GB），256K 无法启动
  LMCACHE_HCA_MAX_SEQ_LEN=262144
  LMCACHE_INDEXER_MAX_SEQ_LEN=262144
  LMCACHE_HCA_ENABLE_PREFETCH=1
runtime patches（⚠️ 镜像不含 SupportsHMA / request_finished_all_groups / token-clamping 修复）：
  启动时通过 --entrypoint /bin/bash 进入 shell，-lc 内先执行两段 inline python3 patch，
  再 exec vllm server：
  patch 1 (lmcache_connector.py): 加 SupportsHMA 继承 + request_finished_all_groups()
    原因: use_layerwise:false 下 vLLM scheduler 看到多个 KV group，无 SupportsHMA 则
    assert len(kv_cache_groups) == 1 失败。
  patch 2 (vllm_v1_adapter.py): _build_req_meta() 加 token_ids 截断
    原因: len(token_ids) > num_blocks*block_size 时 slot_mapping 比 token_ids 短，
    导致 assert len(slot_mapping) == len(token_ids) 失败。
当前测试状态（2026-05-29）：
  ≤10K tokens：通过
  ~52K tokens：FAIL — VLLMPagedMemGPUConnectorV3 contiguous slot offsets
  根因：_build_req_meta() 取 block_ids[0] (HMA 第一个 KV group，非 full-block-size group)
  彻底修复：需新镜像含 _select_primary_block_ids(block_size=256)（本地代码已有）

name:     dsv4-ep8tp8-128k-hma          # 128K HMA 验证（2026-05-29，stopped）
image:    lmcache/vllm-openai:indexer-ssd-hca-prefetch-decodegate-20260528_0630
entrypoint: /bin/bash
关键 vLLM flags: 同上，但 --max-model-len 131072

name:     dsv4-hca-overlap              # HCA overlap 验证（2026-05-29，stopped）
image:    lmcache/vllm-openai:indexer-ssd-hca-prefetch-decodegate-20260528_0630
特点:     LMCACHE_HCA_ENABLE_PREFETCH=1，max-model-len=32768，TP8
          未加 --no-disable-hybrid-kv-cache-manager（32K 不需要 HMA）

name:     dsv4-indexer-ssd              # CSA/HCA prefetch 基础容器（stopped）
image:    lmcache/vllm-openai:indexer-ssd-hca-prefetch-decodegate-20260528_0630
特点:     LMCACHE_HCA_ENABLE_PREFETCH=0（HCA disabled），max-model-len=32768
```

注意：所有容器均使用 all 8 GPUs（TP=8），不能同时运行。切换时先 `docker stop` 再 `docker start`。

**⚠️ HMA 关键说明**：`--no-disable-hybrid-kv-cache-manager` 是让 vLLM 启用 `HybridKVCacheManager`
的开关。DSv4 有 7 个 KV group，各 group 的 block_size 不同（SWA=64、Full MLA=256、C4=4、C128=8），
默认路径会因 `UniformTypeKVCacheSpecs.is_uniform_type()` 返回 False 而退化为全量分配，
导致 KV 预算按 ~461 KiB/token/rank 计算。加了这个 flag 后，HMA 正确识别各 group 的压缩比，
实际 KV 驻留降至 ~6 KiB/token/rank，256K 才能启动。

**HCA SSD 路径注意**：宿主机 `/mnt/nvme0/csa_ssd_pool/hca` 在容器内映射为
`/lmcache/nvme0/csa_ssd_pool/hca`。`LMCACHE_HCA_SSD_DIR` 环境变量必须使用容器内路径。

常用命令：

```bash
sudo docker ps --filter name=dsv4-hca-overlap
sudo docker logs --tail 200 dsv4-hca-overlap
sudo docker restart dsv4-hca-overlap
sudo docker exec -i dsv4-hca-overlap python3 -u -
# CSA container
sudo docker ps --filter name=dsv4-indexer-ssd
sudo docker logs --tail 200 dsv4-indexer-ssd
sudo docker restart dsv4-indexer-ssd
sudo docker exec -it dsv4-indexer-ssd bash
```

跑 heredoc Python 请求时必须加 `-i`：

```bash
sudo docker exec -i dsv4-indexer-ssd python3 -u - <<'PY'
print("stdin works")
PY
```

没有 `-i` 时 Python 会收到空 stdin，可能“成功退出”但实际什么都没跑。

## LMCache 配置

目标是 normal vLLM + `LMCacheConnectorV1`，不是 MP mode。
MP mode 目前不适合作为这个 DSv4 多盘 SSD-only 实验的主路径。

注意：`local_disk` 是 POSIX 路径，用来做 correctness 和普通 SSD-only baseline。它可以配合
大块 CPU pinned slab 做临时 I/O bounce buffer，但 pinned memory 不能成为 cache：不能参与
hit，不能驻留 KV，不能作为 resident set。一次 prefetch/drain 完成后必须立刻把 payload
送入 HBM staging 或目标 KV cache，并释放/复用 pinned slab。

若机器支持 GPU direct，优先用 `gds_path` + `gds_buffer_size` + `use_gds: true`，让数据直接进
GPU-visible buffer。若用 `local_disk`，必须在日志里把它标成 pinned transient bounce 路径，
不要声称它是纯 GDS。

容器配置文件：

```text
host:      /mnt/nvme0/lmcache_ssd_only.yaml
container: /etc/lmcache/lmcache_ssd_only.yaml
```

关键 YAML：

```yaml
chunk_size: 256
local_cpu: false
max_local_cpu_size: 256.0
local_disk: "/mnt/nvme0/lmcache_dsv4_cache/,/mnt/nvme2/lmcache_dsv4_cache/,/mnt/nvme3/lmcache_dsv4_cache/,/mnt/nvme4/lmcache_dsv4_cache/,/mnt/nvme5/lmcache_dsv4_cache/,/mnt/nvme6/lmcache_dsv4_cache/,/mnt/nvme8/lmcache_dsv4_cache/,/mnt/nvme9/lmcache_dsv4_cache/"
local_disk_path_sharding: "by_gpu"
max_local_disk_size: 4096.0
use_gpu_connector_v3: true
use_layerwise: false
extra_config:
  use_odirect: false
  save_only_first_rank: false
  dsv4_optimized_kv: true
  dsv4_optimized_tail_tokens: 256
  dsv4_defer_hca_to_moe: true
```

不要把 8 盘路径写成 `/lmcache/nvme0/,...,/lmcache/nvme7/`，除非确认这些目录本身就是
8 块真实 NVMe 的 mount point。2026-05-30 复查发现 `/lmcache/nvme*` 只是 root 盘
上的目录，文件虽然分到 8 个目录，但没有用到 8 块数据盘。当前有效配置必须使用真实
宿主机挂载点 `/mnt/nvme0,2,3,4,5,6,8,9`，容器也必须显式 `-v` 挂载这些路径。

GPU-direct/GDS 版本应使用类似配置：

```yaml
chunk_size: 256
local_cpu: false
local_disk: null
max_local_disk_size: 0
gds_path: "/lmcache/nvme0/,/lmcache/nvme2/,/lmcache/nvme3/,/lmcache/nvme4/,/lmcache/nvme5/,/lmcache/nvme6/,/lmcache/nvme8/,/lmcache/nvme9/"
gds_path_sharding: "by_gpu"
gds_buffer_size: 8192
use_gds: true
gds_backend: "cufile"
use_gpu_connector_v3: true
use_layerwise: false
```

若 GDS 不可用，允许退到 CPU pinned transient bounce，但不能退到 Python bytes/长期 CPU cache。

TP8 + SSD-only + `local_disk_path_sharding: by_gpu` 的性能/稳定性测试需要覆盖：

```text
LMCACHE_EXTRA_CONFIG={"use_odirect":true,"save_only_first_rank":false}
```

原因：`save_only_first_rank=true` 会让 local worker 0 从 SSD 读入，再通过 GPU tensor
broadcast 给其他 TP ranks。30K 以上 full-hit 时每 rank 可能需要 100 MiB 级临时 CUDA
buffer，在 H200 141 GiB 也会因为 vLLM KV 预分配后剩余显存太少而 OOM。
`save_only_first_rank=false` 让每个 TP rank 按 `by_gpu` 路径从自己的 NVMe 读，既符合多盘
SSD-only 实验目标，也避免这条广播临时显存路径。

必须有：

```text
use_gpu_connector_v3: true
```

否则 DSv4 243 个 vLLM KV tensors 会撞到 legacy 61 pointer slots，典型错误是：

```text
could not broadcast input array from shape (243,) into shape (61,)
```

不要用“过滤成 61 个 cache”作为最终修复；那会破坏 DSv4 HCA/CSA 异构 group 发现。

## vLLM 启动关键项

必须关闭 vLLM 自带 prefix caching，避免它掩盖 LMCache 外部命中：

```text
--no-enable-prefix-caching
--kv-transfer-config {"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}
```

常见 serve 参数摘要：

```text
/mnt/nvme0/models/DeepSeek-V4-Pro
--tensor-parallel-size 8
--enforce-eager
--kv-cache-dtype fp8
--max-model-len 32768
--kv-transfer-config {"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}
--port 8000
--trust-remote-code
--no-enable-prefix-caching
```

`--enforce-eager` 会影响性能，不代表最终系统速度。

## 环境变量

LMCache 基础：

```text
LMCACHE_CONFIG_FILE=/etc/lmcache/lmcache_ssd_only.yaml
```

DSv4 optimized KV GPU 回填实验开关：

```text
LMCACHE_DSV4_OPTIMIZED_KV=1
LMCACHE_DSV4_OPTIMIZED_TAIL_TOKENS=256
```

这个开关只改变 LMCache retrieve 命中后 `MemoryObj -> vLLM GPU KV cache` 的 H2D
回填策略，不改变 SSD 上保存的 chunk，也不改变 LMCache hit 计数。启用后，V3 GPU connector
按 DSv4 7 个 heterogeneous KV groups 做 role-aware transfer：

```text
full transfer:
  - HCA compressed attention KV
  - CSA compressed attention KV
  - CSA indexer cache

tail-only transfer:
  - SWA cache
  - HCA/CSA/indexer compressor state
```

`tail-only` 的窗口由 `LMCACHE_DSV4_OPTIMIZED_TAIL_TOKENS` 控制，默认一个 LMCache
chunk。这样先解决 full-hit retrieve 把 full-prefix SWA 和 float32 compressor state 全部打回
GPU 的问题；SSD 里仍可保留完整 chunk 作为 correctness/debug 数据源。

CSA prefetch 主路径：

```text
LMCACHE_INDEXER_ENABLE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=1
LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0
LMCACHE_INDEXER_REUSE_PREFETCH_MAX_TAIL_TOKENS=4096
LMCACHE_INDEXER_SSD_DIR=/mnt/nvme0/csa_ssd_pool/indexer
LMCACHE_INDEXER_POOL_SIZE=4096
LMCACHE_INDEXER_IO_WORKERS=8
LMCACHE_INDEXER_MAX_SEQ_LEN=131072
```

`LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=1` 的含义是：LMCache retrieve 成功后，
用已经加载回 HBM 的 CSA compressed IndexerCache 初始化 prototype SSD/pool 状态，
让后续 decode 的 residual proxy 能真正查 pool/发读。它不是 pool-only scoring，也不会把
预测 top-K 当成 correctness source。

`LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0` 是当前性能安全默认值：允许 full-hit prefill
后的 CSA pool seed，但不在每个 decode token、每个 CSA 层执行 Python residual proxy
读、decode-token insert、drain 和 true-miss fallback。只有要测 CSA residual proxy
accuracy/state-machine 时才设为 `1`。

HCA deterministic prefetch 主路径：

```text
LMCACHE_HCA_ENABLE_PREFETCH=1
LMCACHE_HCA_ENABLE_PINNED_BOUNCE=1
LMCACHE_DSV4_DEFER_HCA_TO_MOE=1
LMCACHE_HCA_SSD_DIR=/mnt/nvme0/csa_ssd_pool/hca
LMCACHE_HCA_ENABLE_DECODE_HOOK=0
LMCACHE_HCA_PREFETCH_WINDOW_TOKENS=0
LMCACHE_HCA_PREFETCH_MAX_BLOCKS=0
LMCACHE_HCA_RESIDENT_BUDGET_BLOCKS=0
LMCACHE_HCA_IO_WORKERS=8
LMCACHE_HCA_PINNED_BUFFER_MB=64
LMCACHE_HCA_MAX_SEQ_LEN=131072
LMCACHE_HCA_BLOCKING_DRAIN=0
```

HCA 开关独立于 CSA。正确目标是 prefill hit 复用阶段的 deterministic overlap：
LMCache 从 SSD 把 prefix KV 加载回 HBM 时，HCA 层的读集合由 `S/128` 直接确定，
应该按层提前加载并与当前层计算/通信重叠。decode 阶段开始后，同一请求的 prefix
HCA KV 已经在 HBM，不需要再做 HCA prefetch。

当前 pinned-transient prototype 的语义（2026-05-30 已在 `dsv4-256k` smoke 验证）：

- 第一次 full-hit 仍保守执行正常 HCA group 回填，并从 HBM HCA KV seed flat HCA store。
- 当所有 HCA layer flat store ready 且 `LMCACHE_DSV4_DEFER_HCA_TO_MOE=1` 时，后续 full-hit
  retrieve 会跳过普通 `hca_attention_kv` group 回填。
- HCA manager 用当前请求的 compressed slot mapping 把 deterministic rows 读入 CPU pinned
  transient buffer，再在目标 HCA attention 前写入 HBM KV cache。
- `resident_hbm` 只在写入 HBM 后标记；CPU pinned buffer 不参与 hit、不驻留、不当 cache。
- 第一 HCA 层没有上一层 FFN 窗口，adapter 会在 retrieve 后预先 fire；后续 HCA 层由
  decoder FFN 前 hook fire，HCA attention 前 drain。

验证过的短 smoke：

```text
容器: dsv4-256k
prompt_tokens=3806, completion_tokens=8, same prompt repeated 4 times
RUN 1: 1.636s  cold/store
RUN 2: 1.244s  full-hit, normal HCA H2D + seed flat HCA store
RUN 3: 1.136s  full-hit, store becoming ready
RUN 4: 0.808s  full-hit, hca_attention_kv H2D deferred

关键日志:
LMCACHE_DSV4_DEFER_HCA_TO_MOE=1 but HCA flat store is not ready; keeping normal HCA H2D transfer
HCAPrefetchManager: reuse prefetch seeded 31 HCA layers ... compressed_tokens~=28
LMCACHE_DSV4_DEFER_HCA_TO_MOE=1: deferring hca_attention_kv H2D to HCA pinned-transient prefetch
HCAPrefetchManager: fire layer=... rows=28 missing=28 mode=pinned_transient
HCAPrefetchManager: drain layer=... written=28 mode=pinned_transient
```

这个结果只说明 HCA defer 状态机和正确性 smoke 成立，不是最终性能结论。性能版还要把
first-hit seed flat store 替换为直接读取 LMCache/HCA SSD payload 或 GDS/cuFile staging。

历史代码里还有一个 HCA decode fire/drain 原型：full-hit retrieve 后从已加载到 HBM 的
HCA KV 反拷到 flat store，decode 时按 `positions` fire/drain。这个只能作为 hook smoke
test，不能作为 HCA overlap 性能结论；默认不要开启，除非显式做诊断。
诊断时必须同时设置：

```text
LMCACHE_HCA_ENABLE_PREFETCH=1
LMCACHE_HCA_ENABLE_DECODE_HOOK=1
```

默认安全关闭：

```text
LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION=0
LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS=0
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0
```

诊断：

```text
LMCACHE_INDEXER_PROFILE_ACCURACY=1
LMCACHE_INDEXER_TIMING=1
```

也有文件 flag 形式用于临时调试：

```text
/tmp/lmcache_indexer_profile_accuracy
/tmp/lmcache_indexer_timing
/tmp/lmcache_indexer_enable_prefill_eviction
```

注意：`/tmp/lmcache_indexer_enable_prefill_eviction` 是实验 flag。若只想验证官方 HBM path
和 LMCache 基础复用，应删除它。
`/tmp/lmcache_indexer_timing` 会覆盖 `LMCACHE_INDEXER_TIMING=0` 并继续输出
`IndexerSSDTiming`；正式性能测试前也要删除。

## 代码路径

本地 LMCache 代码：

```text
F:\LMCache\lmcache\integration\vllm\vllm_v1_adapter.py
F:\LMCache\lmcache\v1\indexer_ssd_manager.py
F:\LMCache\lmcache\v1\gpu_connector\gpu_connectors.py
F:\LMCache\lmcache\v1\kv_layer_groups.py
F:\LMCache\lmcache\v1\metadata.py
F:\LMCache\lmcache\v1\storage_backend\local_disk_backend.py
```

本地 vLLM 新 layout 参考代码：

```text
F:\vllm_dev\vllm\models\deepseek_v4\nvidia\model.py
F:\vllm_dev\vllm\models\deepseek_v4\amd\model.py
F:\vllm_dev\vllm\model_executor\layers\sparse_attn_indexer.py
F:\vllm_dev\vllm\v1\attention\backends\mla\flashmla_sparse.py
```

容器里的实际 vLLM 可能是旧 layout，部署前必须看 import 的 site-packages：

```text
/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v4.py
/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py
/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py
/opt/venv/lib/python3.12/site-packages/lmcache/v1/indexer_ssd_manager.py
/opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_v1_adapter.py
```

不要只改 `F:\vllm_dev` 然后以为容器生效。要确认容器实际导入路径。
当前 `dsv4-indexer-ssd` 没有 `vllm.models.deepseek_v4...` 新 layout；HCA hook 需要同步到
`/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v4.py`。
本地 patch 拷贝是 `F:\vllm_patch\vllm\model_executor\models\deepseek_v4.py`。

## 运行时版本检查

DSv4 heterogeneous KV 需要 LMCache Python 包和 C extension 都不是旧的。

容器内检查：

```bash
/opt/venv/bin/python - <<'PY'
import lmcache
import lmcache.c_ops as lmc_ops
print(lmcache.__version__, lmcache.__file__)
desc = lmc_ops.PageBufferShapeDesc()
print("block_stride_elems:", hasattr(desc, "block_stride_elems"))
PY
```

必须看到：

```text
block_stride_elems: True
```

如果缺失，单纯复制 Python 文件不够，需要 rebuild/install LMCache extension。

## 必看日志

LMCache 基础复用：

```text
LMCache hit tokens: ...
need to load: ...
Stored ... tokens
Retrieved ... tokens
External prefix cache hit rate: ...
```

vLLM 自带 prefix hit 不用改，也不能作为 LMCache 命中证据。看 `External prefix cache hit rate`
和 LMCache 自己的 `Retrieved/need to load`。

### `size: ... GB` 的真实含义

LMCache 日志里的：

```text
Retrieved ... size: 13.0322 gb
Stored ... size: 3.6263 GB
```

来自 `LMCacheEngine.retrieve/store()` 对每个 retrieved/stored chunk 的
`memory_obj.get_size()` 求和。它不是 vLLM prefix hit 计数，也不是论文里只计算
HCA/CSA attention compressed KV 的理论值。

当前 DSv4 注册到 LMCache 的是 243 个 vLLM KV-like tensors，V3 connector 会把它们分成
7 个 heterogeneous KV groups。实际日志示例：

```text
Group 0: layers=61, bs=256, hs=584,  uint8,   compress_ratio=1    # SWA cache
Group 1: layers=31, bs=2,   hs=584,  uint8,   compress_ratio=128  # HCA attention KV
Group 2: layers=31, bs=256, hs=1024, float32, compress_ratio=1    # HCA compressor state
Group 3: layers=30, bs=64,  hs=132,  uint8,   compress_ratio=4    # CSA indexer cache
Group 4: layers=30, bs=256, hs=512,  float32, compress_ratio=1    # CSA indexer compressor state
Group 5: layers=30, bs=64,  hs=584,  uint8,   compress_ratio=4    # CSA attention KV
Group 6: layers=30, bs=256, hs=2048, float32, compress_ratio=1    # CSA attention compressor state
```

所以当前 LMCache full-hit load/store 体量约为：

```text
8192 tokens  -> 3.6263 GiB/rank
29440 tokens -> 13.0322 GiB/rank
per token    -> about 464 KiB/rank
```

而只算目标论文路径里的 HCA/CSA attention compressed KV，理论值约为：

```text
article simplified 512B entry:
CSA attention KV = 30 * (S / 4)   * 512 B ~= 3840 * S bytes
HCA attention KV = 31 * (S / 128) * 512 B ~=  124 * S bytes
total attention compressed KV ~= 3.9 KiB/token

vLLM current 584B entry:
CSA attention KV = 30 * (S / 4)   * 584 B ~= 4380 * S bytes
HCA attention KV = 31 * (S / 128) * 584 B ~=  141 * S bytes
total attention compressed KV ~= 4.4 KiB/token
```

若再加 CSA indexer cache：

```text
CSA indexer cache = 30 * (S / 4) * 132 B ~= 990 * S bytes
attention KV + indexer cache ~= 5.25 KiB/token
```

这解释了为什么 LMCache 显示远大于“DSv4 KV 应该很小”的直觉：当前保存的是官方 vLLM
correctness 所需的 SWA cache 和多个 float32 compressor state，而不是只保存最终系统希望
用 NVMe 管理的 HCA/CSA compressed attention KV。更重要的是，当前 `retrieve` 命中后会把这些
memory objects 重新 scatter 到 GPU/HBM；这才是长上下文下的显存和加载瓶颈。

容量规划时必须分清：

- `current LMCache payload`：约 `464 KiB/token/rank`，决定当前容器能不能 full-hit load。
- `target prefetch payload`：约 `4.4-5.3 KiB/token/rank`，决定论文设计里的 NVMe/HBM pool 规模。
- `SSD persistent payload`：可以比 GPU 驻留更大，不是首要限制。
- `GPU resident/load payload`：必须按 DSv4 group 语义裁剪，不能 full-hit 时全量回填 7 个 group。

完整文章里的 Together 数据是系统级 optimized KV footprint 口径：未优化 SWA 时约
`3.8 KiB/token`，做 SWA eviction 后单节点 B200 容量从约 `1.2M` 到 `3.7M` tokens。
这和当前 LMCache `464 KiB/token/rank` 不矛盾：当前 LMCache 为了官方 correctness 保存了
SWA cache 和多个 float32 compressor state；文章口径强调的是 DSv4 optimized serving
不应长期保存完整 SWA。

因此，之后实现 `LMCACHE_DSV4_OPTIMIZED_KV=1` 时，判断标准不是“SSD 里是否还有冗余数据”，
而是：

```text
LMCache hit 后进入 GPU 的数据量是否只包含本轮必要的 DSv4 role-specific subset。
```

SSD 里可以完整保存 CSA compressed attention KV、indexer cache 和必要 metadata；GPU 里不能
因为 full-hit retrieve 自动变成完整 SWA + 全量 compressor state + 全量 CSA/HCA KV 回填。

prefetch 挂接：

```text
IndexerSSDManager: enabled CSA prefetch ...
attached ... CSA indexers
HCAPrefetchManager: enabled deterministic HCA prefetch ...
```

CSA 诊断：

```text
profile_full_flat
profile_accuracy
residual_proxy_accuracy
record_attention_topk_slots
correct_true_topk
fire_async_for_layer
```

HCA 诊断：

```text
HCAPrefetchManager: reuse prefetch seeded ... HCA layers
HCAPrefetchManager: seeded layer ...
HCAPrefetchManager: fire layer=...
HCAPrefetchManager: drain layer=...
HCAPrefetchTiming: event=fire/drain/seed ...
```

错误检索：

```bash
grep -E 'Traceback|RuntimeError|ValueError|CRITICAL|ERROR|illegal memory|async read failed|fallback sync read' LOG
```

## 2026-05-30 HMA full-hit 修复记录

这次修复的是 DSv4 + LMCache full-hit retrieve 后输出异常、以及长上下文 HMA
block table 选错的问题。核心结论：

1. `vllm_config.cache_config.block_size` 在 DSv4 HMA 下可能是 `4`，这是最小压缩
   group 的 block size，不能作为 LMCache token-addressed logical block size。
   LMCache metadata 和 request meta 必须使用 vLLM HMA group0 的 logical block size，
   当前容器里为 `256`，也可以用 `LMCACHE_ENGINE_LOGICAL_BLOCK_SIZE=256` 显式覆盖。
2. vLLM HMA 的不同 KV group 有自己的 block table 和 block size。不能把
   `block_ids[0]` 或 group0 block size 用到所有 LMCache groups。
3. 同样物理 shape 的 tensor 也可能属于不同 HMA 语义。DSv4 里 SWA 和 CSA attention
   都可能是 `uint8, hidden=584, block_size=64`，仅按 shape 分组会把不同语义合并，
   full-hit 回填会写错目标 block。
4. `LMCACHE_DSV4_OPTIMIZED_KV=1` 的 role-aware H2D policy 必须在 HMA 分组正确后才可信。
   正确分组后，SWA 被识别为 `swa_cache`，compressor state 被识别为
   `compressor_state`，HCA/CSA compressed KV 和 CSA indexer cache 做 full transfer。

对应代码：

```text
F:\LMCache\lmcache\integration\vllm\vllm_service_factory.py
  _engine_logical_block_size()
  layout_hints["inference_engine_logical_block_size"] = 256

F:\LMCache\lmcache\integration\vllm\vllm_v1_adapter.py
  _capture_vllm_hma_layout()
  layout_hints["vllm_kv_cache_group_ids"]
  layout_hints["vllm_kv_cache_layer_block_sizes"]

F:\LMCache\lmcache\v1\kv_layer_groups.py
  LayerGroupIdentity 加入 vLLM HMA group id
  每个 LMCache group 用该 group 自己的 vLLM logical block size 推导压缩语义

F:\LMCache\lmcache\v1\gpu_connector\gpu_connectors.py
  _hma_block_ids_for_group()
  _dsv4_group_role()
```

修复后日志应看到类似 8 个 LMCache groups，而不是把同 shape 的 DSv4 tensors 合并：

```text
group0: hca_attention_kv, compress_ratio=128, full transfer
group1: csa_indexer_cache, compress_ratio=4, full transfer
group2: csa_attention_kv, compress_ratio=4, full transfer
group3: swa_cache, compress_ratio=1, tail-only
group4: swa_cache, compress_ratio=1, tail-only
group5: compressor_state, tail-only
group6: compressor_state, tail-only
group7: compressor_state, tail-only
```

验证结果：

| 测试 | 结果 |
|---|---|
| 16K prompt / 64 output / 同 prompt 两次 | 第二次 full-hit 输出正常可读，无乱码崩坏 |
| 16.6K prompt / 512 output / 同 prompt 两次 | 第二次 full-hit 输出正常可读，完成约 507 tokens |
| LMCache 日志 | `LMCache hit tokens` 与 `need to load` 匹配，`Retrieved ... out of ... required tokens` 全命中 |
| 8 盘映射 | 旧配置错误地使用 `/lmcache/nvme*` 目录；2026-05-30 已改为真实 `/mnt/nvme0,2,3,4,5,6,8,9` |

accuracy 当前按用户要求直接看输出文本：full-hit 后输出应是正常语言、能持续生成 512 token
级别内容；不再为了这项 sanity check 额外加 accuracy 日志。

剩余性能问题：

- 当前 `local_disk` 仍是 Python/file read + CPU pinned transient bounce 路径；若不
  `drop_caches`，Linux page cache 会掩盖真实 NVMe 读带宽。
- 当前 role-aware policy 主要减少 retrieve 后 H2D/GPU 回填体量；SSD 上的持久化 payload
  仍沿用 LMCache MemoryObj/chunk 路径，不是最终 BAT/gio_uring/staging-slot 版。

## 2026-05-30 真实 8 盘路径修正与带宽验证

问题：之前把 `local_disk` 写成 `/lmcache/nvme0/,...,/lmcache/nvme7/`。容器里
`/lmcache` 是单独挂载目录，底层落在 root 盘；这只是 8 个目录，不是 8 块 NVMe。

修正后使用 8 个真实数据盘路径：

```text
/mnt/nvme0/lmcache_dsv4_cache/
/mnt/nvme2/lmcache_dsv4_cache/
/mnt/nvme3/lmcache_dsv4_cache/
/mnt/nvme4/lmcache_dsv4_cache/
/mnt/nvme5/lmcache_dsv4_cache/
/mnt/nvme6/lmcache_dsv4_cache/
/mnt/nvme8/lmcache_dsv4_cache/
/mnt/nvme9/lmcache_dsv4_cache/
```

启动日志必须看到 8 个 rank 分别选择 8 块真实盘：

```text
cuda:0 -> /mnt/nvme0/lmcache_dsv4_cache/
cuda:1 -> /mnt/nvme2/lmcache_dsv4_cache/
cuda:2 -> /mnt/nvme3/lmcache_dsv4_cache/
cuda:3 -> /mnt/nvme4/lmcache_dsv4_cache/
cuda:4 -> /mnt/nvme5/lmcache_dsv4_cache/
cuda:5 -> /mnt/nvme6/lmcache_dsv4_cache/
cuda:6 -> /mnt/nvme8/lmcache_dsv4_cache/
cuda:7 -> /mnt/nvme9/lmcache_dsv4_cache/
```

验证命令：

```bash
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
iostat -dxm 1 30 > /tmp/dsv4_8disk_dropcache_iostat.log &
# 立即发送同一个 full-hit prompt
```

238K full-hit 结果：

| 测项 | 结果 |
|---|---:|
| prompt tokens | 238,245 |
| LMCache hit tokens | 238,080 |
| hot page-cache hit | 3.228 s |
| drop-cache hit | 3.878 s |
| retrieve payload | 4.5823 GB/rank |
| hot retrieve | 1.24-1.29 s/rank, 3.56-3.69 GB/s/rank |
| drop-cache retrieve | 1.90-1.95 s/rank, 2.35-2.41 GB/s/rank |
| iostat peak aggregate read | 27.66 GB/s |
| iostat per-disk peak | 3.43-3.50 GB/s on all 8 data disks |

结论：真实 8 盘已经同时读起来了，但单盘仍只有约 3.5 GB/s，说明下一步瓶颈不再是
路径没分布，而是每 rank 单盘单线程文件读、page-cache 路径、CPU bounce/H2D pipeline。
要进一步吃满 8 盘，需要改 storage backend：每 rank 内并发读/更大连续块、GDS/cuFile
或最终 BAT + gio_uring + HBM staging slot 路径。

## 2026-05-30 prefill full-hit CSA/HCA 开关对比

本实验只测 full-hit prefill/retrieve，不测 decode prefetch。请求统一使用 `max_tokens=1`，
并保持：

```text
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0
LMCACHE_HCA_ENABLE_DECODE_HOOK=0
```

运行位置：

```text
gpu002: /tmp/dsv4_prefill_compare
summary: /tmp/dsv4_prefill_compare/summary.jsonl
container: dsv4-256k
endpoint: http://127.0.0.1:8000
```

三组语义：

| 组 | CSA prefetch | HCA overlap | 用途 |
|---|---:|---:|---|
| A | off | off | full-hit 基线 |
| B | on | off | 只看 CSA prefill reuse seed |
| C | on | on | 在 B 基础上打开 HCA pinned-transient overlap |

结果：

| 组 | prompt tokens | cold | hit1 | hit2 | LMCache load | payload/rank | retrieve/rank | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 253,845 | 38.460 s | 25.439 s | 25.501 s | 253,696 | 112.303 GB | 20.28-24.68 s | `Answer`, `Pref` |
| B | 248,644 | 38.142 s | 22.121 s | 22.301 s | 248,576 | 110.037 GB | 19.55-21.21 s | `Answer` |
| C | 243,443 | 37.095 s | 24.810 s | 24.268 s | 243,200 | 107.657 GB | 19.44-22.21 s | `Answer` |

8 盘读带宽确认：

| 组 | hit1 iostat peak | hit2 iostat peak | 说明 |
|---|---:|---:|---|
| A | 46.81 GB/s | 46.60 GB/s | 8 块真实 NVMe 全部活跃 |
| B | 46.30 GB/s | 46.28 GB/s | 8 块真实 NVMe 全部活跃 |
| C | 46.26 GB/s | 46.46 GB/s | 8 块真实 NVMe 全部活跃 |

关键日志：

```text
# B
IndexerSSDManager: reuse prefetch seeded 30 CSA layers ...

# C
HCAPrefetchManager: reuse prefetch seeded 31 HCA layers ...
HCAPrefetchManager: fire layer=... rows=1900 missing=1900 mode=pinned_transient
HCAPrefetchManager: drain layer=... written=1900 mode=pinned_transient
LMCACHE_DSV4_DEFER_HCA_TO_MOE=1 but HCA flat store is not ready; keeping normal HCA H2D transfer
```

解释：

1. B 组说明 CSA prefill reuse seed 路径触发，并且 full-hit TTFT 相比 A 有工程趋势收益。
2. C 组说明 HCA manager/fire/drain 路径触发，但 `HCA flat store is not ready` 大量出现，
   所以当前 HCA overlap 没有真正掩盖 HCA H2D 主路径，不能声称 C 已加速成功。
3. 本轮 prompt 里带 group label，导致三组 token 数不同。最终对比必须用完全相同 prompt
   重跑，或者先写入同一个稳定 cache key，再只切换 runtime 开关做 hit-only 测试。
4. 本轮 LMCache `Retrieved ... size:` 是 107-112 GB/rank。不要把它解释成
   `LMCACHE_DSV4_OPTIMIZED_KV` 被关掉了：日志已经确认 `LMCACHE_DSV4_OPTIMIZED_KV active`
   和 `KV layer groups` 正确加载。当前优化发生在 retrieve 之后的 role-aware H2D/GPU
   回填阶段，SSD 侧仍按 LMCache MemoryObj 读完整磁盘对象。
5. 之前记录的 4.582 GB/rank 与本轮 107-112 GB/rank 口径冲突，必须复核它到底是
   SSD read bytes、H2D copied bytes，还是经过 role-aware selection 后的有效 GPU payload。
   在复核前不要把 4.582 GB/rank 当作同一类 `Retrieved size` 对比。

### 2026-05-30 HCA defer 修复记录

已修复 `HCA flat store is not ready` 导致 C 组 fallback 的直接原因：

1. `HCAPrefetchManager.seed_range_from_lmcache_group()` 会从 LMCache retrieve 出来的
   `hca_attention_kv` group tensor 直接按 chunk offset 写 flat HCA store。
2. `VLLMPagedMemGPUConnectorV3.to_gpu()` 在 `LMCACHE_DSV4_DEFER_HCA_TO_MOE=1` 时，
   对 HCA group 先 seed flat store，再跳过普通 HCA H2D。
3. `_maybe_seed_hca_reuse_prefetch()` 检测到 flat store 已经有足够 rows 时，只设置当前
   request 的 slot mapping，不再从可能未回填的 `kv_cache` 反拷覆盖。

复测时要看两类日志：

```text
LMCACHE_DSV4_DEFER_HCA_TO_MOE=1: seeded HCA flat store directly from LMCache retrieve and deferred normal HCA H2D
HCAPrefetchManager: fire layer=... rows=... missing=... mode=pinned_transient
HCAPrefetchManager: drain layer=... written=... mode=pinned_transient
```

如果还看到大量：

```text
HCA flat store is not ready; keeping normal HCA H2D transfer
```

说明容器没有加载新补丁，或 manager 没挂上。

注意：这个修复只去掉 HCA group 的普通 H2D fallback。现在 256K full-hit 仍慢的主因是
SSD backend 读取完整 MemoryObj。复测 optimized store/retrieve 体量必须清空旧 cache path
或换新 cache 目录；否则会继续命中之前写入的 full-object，`Retrieved size` 还是 100GB/rank
级别。

### 2026-05-30 runtimefix smoke 结果

容器 `dsv4-256k` 已同步完整 LMCache runtime patch tree，并修改 startup 让 `/patches/v1/`
和 `/patches/integration/` 下的文件都覆盖 site-packages。之前 startup 只覆盖
`vllm_v1_adapter.py`、`gpu_connectors.py`、`hca_prefetch_manager.py`、`kv_layer_groups.py`，
没有覆盖 `cache_engine.py`，所以 optimized store shape 可能没有真正进入容器。

为避免命中旧 full-object cache，local disk 改成新的 8 盘目录：

```text
/mnt/nvme{0,2,3,4,5,6,8,9}/lmcache_dsv4_cache_runtimefix/
```

启动后容器内确认：

```text
has_seed_range True
has_old_ready_msg False
has_store_shapes True
```

60K/256K smoke（`max_tokens=1`，第二轮前 drop cache）：

| prompt tokens | cold | drop-cache full-hit | LMCache hit/load | Retrieved size/rank | retrieve cost/rank |
|---:|---:|---:|---:|---:|---:|
| 60,001 | 7.583 s | 1.201 s | 59,904 | 1.2036 GB | 0.50-0.53 s |
| 252,001 | 33.168 s | 3.052 s | 251,904 | 4.7653 GB | 2.07-2.12 s |

对比旧 64K 曲线：旧 `Retrieved size/rank` 是 27.878 GB，full-hit TTFT 6.201 s。
runtimefix 后同量级 prompt 的磁盘对象已经降到 1.2 GB/rank，full-hit 降到 1.2 s。
对比旧 256K 曲线：旧 `Retrieved size/rank` 是 111.283 GB，full-hit TTFT 22.342 s；
runtimefix 后是 4.7653 GB/rank 和 3.052 s。这证明真正慢点不是 8 盘带宽，而是容器
没有加载完整 optimized store 补丁/复用了旧 full-object。

重跑时的容器清理要用强制循环，避免 Docker name conflict：

```bash
sudo docker rm -f dsv4-256k || true
sleep 3
for i in $(seq 1 12); do
  if ! sudo docker ps -a --format '{{.Names}}' | grep -qx dsv4-256k; then
    break
  fi
  sudo docker rm -f dsv4-256k || true
  sleep 2
done
```

## 2026-05-30 CSA on 多长度 full-hit 曲线

固定只开 CSA prefill reuse seed，关闭 HCA overlap 和 decode prefetch：

```text
LMCACHE_INDEXER_ENABLE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0
LMCACHE_HCA_ENABLE_PREFETCH=0
LMCACHE_HCA_ENABLE_DECODE_HOOK=0
max_tokens=1
```

结果文件：

```text
/tmp/dsv4_csa_on_curve/summary.jsonl
/tmp/dsv4_csa_on_curve/docker_20260530_140102.log
```

结果：

| case | prompt tokens | cold TTFT | drop-cache full-hit TTFT | LMCache hit/load | Retrieved size/rank | retrieve/rank | 输出 |
|---|---:|---:|---:|---:|---:|---:|---|
| 16K | 15,547 | 4.995 s | 2.433 s | 15,360 | 6.799 GB | 1.23-1.33 s | `Answer` |
| 32K | 31,527 | 3.568 s | 3.618 s | 31,488 | 13.939 GB | 2.49-2.66 s | `Answer` |
| 64K | 63,017 | 7.373 s | 6.201 s | 62,976 | 27.878 GB | 4.99-5.28 s | `Answer` |
| 128K | 125,997 | 15.769 s | 11.455 s | 125,952 | 55.755 GB | 10.00-10.58 s | `Answer` |
| 192K | 188,977 | 31.418 s | 16.896 s | 188,928 | 83.632 GB | 15.11-15.86 s | `Answer` |
| 256K | 251,487 | 34.748 s | 22.342 s | 251,392 | 111.283 GB | 20.17-21.47 s | `Answer` |

日志检查：

```text
LMCache hit tokens == need to load
IndexerSSDManager: reuse prefetch seeded 30 CSA layers
```

解释：

1. 这组数据比单点 256K 更可信：full-hit TTFT 随长度基本线性增长。
2. 每 rank retrieve throughput 稳定在约 5.2-5.6 GB/s。
3. `Retrieved size/rank` 是 SSD 完整 MemoryObj 读取口径，不是 role-aware GPU 回填 payload。
4. 16K/32K cold TTFT 有 warmup/page-cache 噪声，不能单独解读；full-hit 曲线更稳定。

### 同 prompt CSA on/off 对比

上面的 CSA on 曲线本身不是对比。后续用同一批 prompt 文本补跑 CSA off。因为 LMCache
配置使用 `pre_caching_hash_algorithm: builtin`，跨容器重启后同 prompt 的 hash key 不稳定；
OFF 第一轮日志是 `LMCache hit tokens: 0`，不能用。OFF 第二轮才是真正 full-hit。

有效对比：

| case | tokens | CSA on full-hit | CSA off full-hit | on - off |
|---|---:|---:|---:|---:|
| 16K | 15,547 | 2.433 s | 1.726 s | +0.707 s |
| 32K | 31,527 | 3.618 s | 3.060 s | +0.558 s |
| 64K | 63,017 | 6.201 s | 5.620 s | +0.581 s |
| 128K | 125,997 | 11.455 s | 11.665 s | -0.210 s |
| 192K | 188,977 | 16.896 s | 17.779 s | -0.883 s |
| 256K | 251,487 | 22.342 s | 24.101 s | -1.759 s |

结论：

1. 当前 CSA on 的端到端 TTFT 收益只在长上下文出现，192K/256K 约 5-7%；短上下文因为
   seed 开销反而变慢。
2. 时间明显太长是事实：256K full-hit 仍要 22-24 s，因为 SSD backend 仍在读完整
   LMCache MemoryObj，单 rank `Retrieved size` 达 111.283 GB。
3. 当前 role-aware 优化只减少 GPU 回填，不减少 SSD read。要拿到论文里想要的 TTFT，
   下一步必须改 SSD I/O 层的对象布局/range read/BAT+gio_uring，不能只靠 H2D 少写。
4. 以后做跨容器 hit-only 对比，不要用 `builtin` hash 直接假设 cache key 一致；需要同容器
   store 后测第二轮，或改成稳定 hash。

## 2026-05-30 58K ON/OFF 对比重测

Claude 之前的对比数据不要继续引用。本次用修复后的 `dsv4-256k` 容器重新测，固定条件：

```text
LMCACHE_DSV4_OPTIMIZED_KV=1
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0
LMCACHE_HCA_ENABLE_PREFETCH=0
max_tokens=1
served model: deepseek-v4-pro
```

OFF 条件：

```text
LMCACHE_INDEXER_ENABLE_PREFETCH=0
LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=0
```

ON 条件：

```text
LMCACHE_INDEXER_ENABLE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=1
```

结果：

| 条件 | prompt tokens | cold elapsed | full-hit elapsed | LMCache retrieve |
|---|---:|---:|---:|---|
| OFF | 58,801 | 7.922 s | 3.262 s | 58,624 tokens, 25.951 GB/rank, 2.388-2.599 s/rank |
| ON | 57,401 | 7.399 s | 3.646 s | 57,344 tokens, 25.384 GB/rank, 2.210-3.227 s/rank |

解释：

- 当前 `decode prefetch` 是关闭的，所以 ON 只多做 full-hit 后的
  `reuse prefetch seeded 30 CSA layers`，不会在 `max_tokens=1` 场景体现 decode 期收益。
- 这次 58K 口径下 ON 不比 OFF 快，反而慢约 0.38 s；这部分主要是 reuse seed 的额外开销。
- 因此当前可信结论是：修复后的 optimized KV full-hit 基础复用可跑，8 rank retrieve
  约 80 GB/s 级别；CSA reuse seed 目前是状态初始化/后续 decode 预取准备，不应声称它已经
  在 gen=1 TTFT 对比中加速。
- 若要证明 CSA prefetch 加速，必须打开真正 decode 阶段的 prefetch 或实现最终
  BAT/gio_uring/staging-slot 路径，再测长 decode 的 first-token/steady-token latency。

## 测试门槛

不同问题用不同请求长度：

| 目标 | 请求 |
|---|---|
| first-decode proxy accuracy | `max_tokens=1` 或 `2` 就够 |
| 长 decode 稳定性 | output 至少 512 token |
| 长上下文 sanity | prompt 至少 32K tokens |
| LMCache 基础复用 | 同 prompt 连跑两次，看第二次 load/retrieve |
| 性能 | 拆 TTFT、load、first decode、steady decode，不只看总 elapsed |

用户要求“说明没有跑坏”时，不要只跑很短输出；至少跑 `max_tokens>=256`，最好 `>=512`。

## OpenAI-compatible 请求模板

容器内短 probe：

```bash
sudo docker exec -i dsv4-indexer-ssd python3 -u - <<'PY'
import json, time, urllib.request

payload = {
    "model": "/mnt/nvme0/models/DeepSeek-V4-Pro",
    "prompt": "CSA prefetch verification prompt. " * 1800,
    "max_tokens": 32,
    "temperature": 0,
    "ignore_eos": True,
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as r:
    obj = json.loads(r.read().decode())
print("HTTP", r.status, "elapsed", round(time.time() - t0, 3))
print("usage", obj["usage"])
print("finish", obj["choices"][0]["finish_reason"])
PY
```

32K / 512 长验证：

```bash
sudo docker exec -i dsv4-indexer-ssd python3 -u - <<'PY'
import json, time, urllib.request

payload = {
    "model": "/mnt/nvme0/models/DeepSeek-V4-Pro",
    "prompt": "LMCache DeepSeek V4 thirty two thousand token validation paragraph. " * 2144,
    "max_tokens": 512,
    "temperature": 0,
    "ignore_eos": True,
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as r:
    obj = json.loads(r.read().decode())
print("HTTP", r.status, "elapsed", round(time.time() - t0, 3))
print("usage", obj["usage"])
print("finish", obj["choices"][0]["finish_reason"])
PY
```

历史 HCA 3 阶段 decode-hook 验证模板（容器 `dsv4-hca-overlap`，仅诊断 hook，不代表主路径）：

```python
# 确认模型 ID
import requests, json, time, sys
url = "http://127.0.0.1:8000/v1/completions"
model_id = requests.get("http://127.0.0.1:8000/v1/models").json()["data"][0]["id"]

prompt = "HCA overlap validation prompt. " * 512  # ~5K tokens

# Phase 1: initial store
resp = requests.post(url, json={"model": model_id, "prompt": prompt,
                                "max_tokens": 4, "temperature": 0}, timeout=300)
print("Phase 1 store:", resp.json()["usage"])

time.sleep(3)  # wait for SSD write

# Phase 2: full-hit -> seed HCA store
# 日志应出现: HCAPrefetchManager: reuse prefetch seeded 31 HCA layers
resp = requests.post(url, json={"model": model_id, "prompt": prompt,
                                "max_tokens": 4, "temperature": 0}, timeout=300)
print("Phase 2 hit+seed:", resp.json()["usage"])

# Phase 3: decode -> fire/drain (legacy diagnostic only)
# 日志应出现: HCAPrefetchTiming: event=fire layer=... rows=40 missing=0
resp = requests.post(url, json={"model": model_id, "prompt": prompt,
                                "max_tokens": 64, "temperature": 0}, timeout=300)
print("Phase 3 decode:", resp.json()["usage"])
print("output:", resp.json()["choices"][0]["text"][:80])
```

## 2026-05-29 256K 容器 token 梯度测试

容器 `dsv4-256k`，runtime patch 注入后，用 `test_256k_slot.py` 依次发送不同长度请求
（served model name `deepseek-v4-pro`）：

| prompt tokens | 结果 |
|---:|---|
| ~65 | OK |
| ~260 | OK |
| ~2,600 | OK |
| ~10,400 | OK |
| ~52,000 | FAIL → EngineCore 崩溃 |

FAIL 错误信息：

```
ValueError: VLLMPagedMemGPUConnectorV3 block transfer requires contiguous slot offsets
within each inference-engine block.
```

根因：`_build_req_meta()` 的 `allocated_block_ids = block_ids[0]` 取的是 HMA 第一个
KV group 的 block IDs，该 group 并非 full-block-size（256-token）group；大长度下 block
分配不连续时 `_slot_mapping_to_block_ids()` 的 offset 检查失败。

短期 workaround：token clamping patch（patch 2）截断过长 token_ids，使 ≤10K 请求通过。
长期修复：需新镜像包含 `_select_primary_block_ids(block_size=256)`（本地
`feature/dsv4-compat` 分支已有）。

## 2026-05-29 HCA overlap smoke 结果

容器 `dsv4-hca-overlap`（image `indexer-ssd-hca-prefetch-decodegate-20260528_0630`，
TP8，max-model-len 32768，`LMCACHE_HCA_ENABLE_PREFETCH=1`）。

| Phase | 耗时 | 关键日志 |
|---|---:|---|
| Phase 1 store | 1.1 s | `prompt_tokens=5121` |
| Phase 2 hit+seed | 0.8 s | `reuse prefetch seeded 31 HCA layers, compressed_tokens~=40` |
| Phase 3 decode 64 | 7.7 s | `fire layer rows=40 missing=0`, `drain pending=0 written=0` |

启动日志：`HCAPrefetchManager: enabled ... 61 decoder layers ... 31 HCA caches`

## 2026-05-27 smoke 结果

当前已在容器 `dsv4-indexer-ssd` 上跑通 normal LMCache SSD-only + CSA reuse prefetch 原型。

| 请求 | prompt tokens | completion tokens | elapsed | 关键证据 |
|---|---:|---:|---:|---|
| 同 prompt 第一次 | 17,875 | 1 | 2.600 s | store path 成功 |
| 同 prompt 第二次 | 17,875 | 1 | 1.418 s | `Retrieved 16384 out of 16384 required tokens` |
| LMCache hit 后 decode | 17,875 | 16 | 7.121 s | `fire_async`、`correct_true_topk` 均出现 |

关键日志：

```text
IndexerSSDManager: reuse prefetch seeded 30 CSA layers ... lmcache_tokens=16384 compressed_tokens~=4096
IndexerSSDTiming: event=fire_async layer=28 proxy_ms≈0.8-1.0 missing≈15
IndexerSSDTiming: event=correct_true_topk layer=28 true=1024 missing=0 total_ms≈1.6-1.8
```

注意：这个结果证明语义链路可跑，不代表最终性能。当前仍有大量诊断日志，且 prototype
读/insert 仍是 Python 路径。reuse seed 已限制为 local worker 0，避免每个 TP rank 重复 seed。

旁路性能采样补充：约 22K prompt 的 `max_tokens=16` 命中 16,384 tokens 后约 4.354 s；
日志聚合显示 `fire_async` median 约 1.5 ms，`correct_true_topk` median 约 4.8 ms、
p95 约 29 ms，`read_ms` median 约 0.017 ms。慢点主要在 prototype decode
correction/诊断日志，不在 SSD retrieve 本身。`max_tokens=64` 在该 22K prompt 上触发
CUDA OOM，不作为有效性能样本。

## 2026-05-28 30K/512 对比

同一台 gpu002、同一个 30,000 token prompt、`max_tokens=512`、TP8、`--enforce-eager`、
`--gpu-memory-utilization 0.88`、`--no-enable-prefix-caching`。LMCache 配置为
SSD-only、`use_layerwise=false`、`use_gpu_connector_v3=true`、
`save_only_first_rank=false`。

| 路径 | 第一次 elapsed | 第二次 full-hit elapsed | LMCache load | 结论 |
|---|---:|---:|---|---|
| 干净基线：同一修复镜像，所有 CSA/HCA prefetch env 显式关闭 | 42.613 s | 40.744 s | 29,952 tokens；每 rank 13.26 GB，约 1.59-1.88 s，7.04-8.35 GB/s | 当前可比的“正常 DSv4 + LMCache SSD-only”性能 |
| CSA reuse seed 开启、decode prefetch 关闭：`LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0` | 38.397 s | 36.606 s | 29,440 tokens；每 rank 13.03 GB，约 1.52-1.84 s，7.06-8.60 GB/s | 没有跑坏正常 DSv4；full-hit prefill reuse seed 可与正常 decode 共存 |
| CSA/HCA Python prototype 全开，decode prefetch/correction 每 token 执行 | 42.843 s | 147.380 s | 29,952 tokens；每 rank 13.26 GB，约 1.43-1.78 s，7.45-9.27 GB/s | 慢点在 decode 期 Python prototype，不在 LMCache SSD load |
| 旧镜像 `lmcache/vllm-openai:indexer-ssd` | 无有效数据 | 无有效数据 | 第一次 store 即崩 | `attempt_permute_to_contiguous_view` 非 contiguous view 错误，不是可用基线 |

## 2026-06-08 CSA/HCA 端到端 pipeline 验证

容器 `dsv4-256k-measure-tutti`，TP=8，DSV4-Pro，EnforceEager，SSD-only，
`LMCACHE_ENGINE_LOGICAL_BLOCK_SIZE=256`。

**激活状态（startup script `/tmp/startup_256k_tutti.sh`）**：

```bash
export LMCACHE_INDEXER_ENABLE_PREFETCH=1
export LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=1
export LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=1
export LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=1
export LMCACHE_HCA_ENABLE_PREFETCH=1
export LMCACHE_HCA_ENABLE_PINNED_BOUNCE=1
export LMCACHE_INDEXER_SSD_DIR=/mnt/nvme0/lmcache_csa
```

**注意**：`docker exec env` 显示部分 env var 为 0/旧路径，
这是容器 docker-run `-e` 参数遗留。startup script 里的 `export` 覆盖了这些值，
vllm 进程实际使用的是 script 里的正确值（从 `store=...lmcache_csa` 日志可验证）。

**关键初始化日志**：

```text
IndexerSSDManager: enabled CSA prefetch on 61 decoder layers and attached 30 CSA indexers,
  pool_size=4096, store=/mnt/nvme0/lmcache_csa/rank_0
HCAPrefetchManager: using transient pinned I/O slab size=64 MiB
HCAPrefetchManager: enabled pinned-transient HCA state; decoder_hooks=61 HCA caches=31
  store=/mnt/nvme0/lmcache_csa/hca/rank_0
```

**端到端测试**（prompt ≈3978 tokens，需 ≥1024 token hit 才能触发 seed 路径）：

```text
# 第一次（cold store）：0.85 s
# 第二次（warm hit，lmcache_tokens=3840）：2.90 s，触发 seed+fire
```

**warm hit 后关键日志**（来自 09:32:25 时刻）：

```text
IndexerSSDManager: reuse prefetch seeded 30 CSA layers  lmcache_tokens=3840 compressed_tokens~=960
HCAPrefetchManager: seeded layer 0 rows=30 row_bytes=584
HCAPrefetchManager: fire layer=0 rows=30 missing=30 mode=pinned_transient
HCAPrefetchManager: drain layer=0 written=30 mode=pinned_transient
... (所有 31 个 HCA 层类似)   总 drain 事件 248 个
```

**结论**：
- CSA IndexerSSDManager seed → fire → correct_true_topk 链路完整
- HCA seed → fire → drain 链路完整，30 rows 每层写入 HBM KV cache
- drain 使用 `mode=pinned_transient`（CPU 侧 pinned bounce buffer + file I/O，不走 Tutti）
- `lba_cache=0`（Tutti 初始化问题）**不影响** HCA/CSA pipeline（二者走独立文件 I/O 路径）

**待测**：

| 项目 | 状态 |
|---|---|
| decode 期 `fire_async_for_layer` 非 skipped（需第三次 same-prefix 请求） | 未测 |
| `correct_true_topk` 出现（CSA 投机预取命中率） | 未测 |
| 基线对比：CSA/HCA on vs off 的 decode token/s | 未测 |
| rank_suffix 问题：所有 worker 都写 rank_0 目录 | 待确认是否影响结果 |

关键判断：

1. LMCache SSD load 本身约 1.5-1.9 s，占 40 s 级总时延很小。
2. prototype 全开后 full-hit 变成 147 s，是每个 decode token、每个 CSA 层的
   `fire_async/correct_true_topk` Python 路径造成的吞吐崩塌。
3. 因此性能实验默认必须让 `LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0`，只保留
   LMCache full-hit prefill reuse seed；decode prefetch 只作为 accuracy/状态机诊断开关。
4. 这个镜像曾经 commit 过 prefetch env，干净基线必须显式设置
   `LMCACHE_INDEXER_ENABLE_PREFETCH=0`、`LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=0`、
   `LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=0`、`LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0`、
   `LMCACHE_HCA_ENABLE_PREFETCH=0`，并把 `LMCACHE_INDEXER_SSD_DIR`、`LMCACHE_HCA_SSD_DIR`
   置空。

decode-gate 验证日志要点：

```text
Reqid ... LMCache hit tokens: 29440, need to load: 29440
Retrieved 29440 out of 29440 required tokens ... cost 1516-1845 ms
IndexerSSDManager: reuse prefetch seeded 30 CSA layers ... compressed_tokens~=7360
Avg generation throughput: 14.8 tokens/s
```

中间曾测到 `121.010 s` 的样本无效：当时补丁只传到了 master 的 `/tmp`，没有传到
gpu002 的 `/tmp`，容器里仍是旧代码，日志仍在刷 `fire_async/correct_true_topk`。

## 2026-05-28 prefill/decode 分拆

同一 decode-gate 容器、同一类 29.4K token 源码 prompt、streaming 请求。CSA reuse seed
开启，decode prefetch 关闭：

```text
LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=0
```

| 测项 | max_tokens | 第一次 miss/store | 第二次 full-hit/retrieve | 说明 |
|---|---:|---:|---:|---|
| prefill/TTFT 稳定样本 | 1 | TTFT 3.532 s | TTFT 1.873 s | 两次之间等待 10 s，避免刚 store 完立刻 retrieve 的可见性竞态 |
| decode tail | 512 | TTFT 3.523 s；tail 34.580 s；约 14.78 tok/s | TTFT 1.867 s；tail 34.448 s；约 14.83 tok/s | decode 不跑 prefetch，基本就是正常 DSv4 decode 吞吐 |

对应日志：

```text
prefill full-hit: LMCache hit tokens: 29440, need to load: 29440
prefill retrieve: 每 rank 13.03 GB，约 1.42-1.73 s，7.54-9.19 GB/s
decode full-hit:  LMCache hit tokens: 29440, need to load: 29440
decode retrieve:  每 rank 13.03 GB，约 1.42-1.72 s，7.56-9.15 GB/s
reuse seed:       reuse prefetch seeded 30 CSA layers ... compressed_tokens~=7360
decode hooks:     event=fire_async / event=correct_true_topk 计数为 0
```

注意：曾经把 `max_tokens=1` 两次请求紧贴着跑，第二次出现 partial retrieve /
`KV load failure`，例如期望 23,552 tokens 但部分 rank 只拿到 22,528 或 23,296。
这更像 store 刚返回后马上 retrieve 的落盘/多 rank 可见性竞态，不应作为 prefill full-hit
性能样本。稳定测 prefill 命中时，两次请求之间至少等待数秒，或直接使用已经稳定存在的
缓存 key。

## 已知禁忌

1. 不要把 DSv4 改成 V2 residual proxy。
2. 不要在 DSv4 用 `hidden_states + residual` 当 proxy 输入。
3. 不要把 `kv_caches` 过滤成 61 个主层 cache 作为最终修复；V3 group discovery
   必须在 store/retrieve allocation 前发生。
4. 不要关掉 `use_gpu_connector_v3: true`。
5. 不要用 MP mode 替代 normal multi-NVMe SSD-only 路径。
6. 不要让 `LMCACHE_INDEXER_SSD_DIR` 单独启用 prefetch。
7. 不要默认开启 `LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION`。
8. 不要默认开启 `LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS`。
9. 不要默认开启 `LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH`；它是当前 Python
   state-machine/accuracy 诊断路径，不是性能路径。
10. 不要把 chunk-local row ID 写进全局 token/block ID 语义。
11. 不要把 pool-only scoring 当成生产 correctness path。
12. 不要对 DeepGEMM `fp8_fp4_paged_mqa_logits` 传 `schedule_metadata=None`。
13. 不要返回 pool-local slot IDs 给 vLLM；必须转回 global compressed token IDs。
14. 不要 claim heredoc 请求跑过，除非 `docker exec` 用了 `-i`。
15. 不要在 V3 `use_gpu` 路径做整块 flat buffer memcpy；DSv4 memory object/temporary
    buffer 必须按 heterogeneous group 的实际 shape 搬。
16. 不要把 HCA 当成 CSA residual proxy。HCA 是 `compress_ratio=128` 的确定性 C128A
    路径，当前 manager 只做 deterministic prefetch/drain，不替换官方 attention indices。

## 2026-05-31 HCA-on runtimefix 复测注意

### 2026-05-31 HCA 触发窗口修正

修正后的约束：HCA full-hit read set 只由 prefix 长度和 `compress_ratio=128` 决定，
不依赖 MoE expert 路由或 FFN 计算结果。因此一旦当前 request 的 HCA compressed
slot mapping 可用，就可以提交下一 HCA 层的 deterministic read；目标 HCA attention
前必须 drain。

已做修正：

1. `F:\vllm_dev\vllm\models\deepseek_v4\nvidia\model.py`、
   `F:\vllm_dev\vllm\models\deepseek_v4\amd\model.py` 和
   `F:\vllm_patch\vllm\model_executor\models\deepseek_v4.py` 在 FFN/MoE 前直接调用
   `_fire_hca_prefetch(positions)`，目标 HCA attention 前继续 `_drain_hca_prefetch()`。
2. `LMCacheConnectorV1Impl._maybe_seed_hca_reuse_prefetch()` 先记录当前 request 的
   compressed slot mapping；flat store 已就绪时立即 fire，异步 seed 未完成时由 seed
   线程完成后自动 fire。
3. `HCAPrefetchManager.submit_seed_after_reuse(..., fire_seq_len=...)` 避免 seed/fire
   竞态，防止日志显示 prepared 但实际因 `initialized_rows == 0` 空跑。
4. `prefire_first_hca()` 仍默认拒绝执行；第一层 HCA 不能在当前 request slot mapping
   稳定前写回。

远端同步状态：

```text
gpu002:/tmp/deepseek_v4.py                         已同步
gpu002:/tmp/lmcache_rewrite_deploy/deepseek_v4.py  已同步
gpu002:/tmp/vp3/deepseek_v4.py                     已同步
gpu002:/tmp/lmcache_patch/integration/vllm/vllm_v1_adapter.py 已同步
gpu002:/tmp/lmcache_patch/v1/hca_prefetch_manager.py          已同步
```

`/tmp/vllm_patches2/deepseek_v4.py` 当前普通用户不可写，未覆盖；启动脚本若改用该目录，
必须先修权限或换成已同步的 patch 路径。

当前有效启动值：

```text
LMCACHE_HCA_ENABLE_PREFETCH=1
LMCACHE_HCA_ENABLE_PINNED_BOUNCE=1
LMCACHE_DSV4_DEFER_HCA_TO_MOE=1
LMCACHE_HCA_PINNED_BUFFER_MB=64
LMCACHE_HCA_BLOCKING_DRAIN=0
extra_config.dsv4_defer_hca_to_moe: true
```

不要再把 `LMCACHE_HCA_PINNED_BUFFER_MB` 设成 `2048` 做默认性能实验；这次启动曾在
API ready 前卡住，后续主 PID 进入 zombie 状态，`docker restart` 无法回收，只能
`docker rm -f dsv4-256k` 后重建。默认 pinned-transient smoke 用 64 MiB。

HCA-on 252K runtimefix 结果：

```text
HCA off baseline:       full-hit 3.052 s, retrieve 2.07-2.12 s/rank
HCA on defer=false:     full-hit 3.651-3.740 s, retrieve 2.09-2.12 s/rank
HCA on defer=true:      full-hit 4.280-4.411 s, retrieve 2.78-2.83 s/rank
prompt_tokens:          252,001
hit/load:               251,904
Retrieved size/rank:    4.7653 GB
```

defer=true 的正确日志：

```text
LMCACHE_DSV4_DEFER_HCA_TO_MOE=1:
  seeded HCA flat store directly from LMCache retrieve and deferred normal HCA H2D
HCAPrefetchManager: reuse prefetch seeded 31 HCA layers ...
HCAPrefetchManager: fire layer=0 rows=1968 missing=1968 mode=pinned_transient
HCAPrefetchManager: drain layer=0 written=1968 mode=pinned_transient
```

当前结论：HCA pinned-transient Python 原型只作为状态机诊断保留，不作为性能默认开关。
它能触发 direct seed/fire/drain，但 252K full-hit 比 HCA off 慢；下一步性能工作应转向
LMCache/HCA SSD range、BAT/gio_uring/GDS 或 GPU-visible staging，不能继续靠 full-hit
后 Python flat-store seed 再逐层写回。

## 2026-05-31 SSD pSLC cache 检查

当前 LMCache 8 盘池是：

```text
/mnt/nvme0,2,3,4,5,6,8,9 -> KIOXIA KCD81PUG7T68, 7.68 TB, FW 1XET6103
```

另有两块 `Micron_7450_MTFDKBA960TFR` 960 GB 盘（`nvme1n1`、`nvme7n1`），不在当前
LMCache 8 盘 runtimefix pool 中。

从本机可见信息判断：

1. KIOXIA `KCD81PUG7T68` 是 CD8P-R read-intensive 企业 NVMe，公开资料/控制器信息显示
   TLC/enterprise sustained-write 取向；本机 NVMe 标准字段和 `nvme supported-log-pages`
   没有暴露 “pSLC cache size/remaining” 之类可查询字段。
2. `nvme-cli` 版本为 1.16，没有 `nvme ocp` 子命令；KIOXIA 盘支持若干 vendor log
   `0xc0-0xc5/0xca`，但当前未解析出 pSLC/cache 状态。
3. 标准 SMART 当前健康正常：KIOXIA 盘 `percentage_used=0%`、`available_spare=100%`、
   `media_errors=0`，温度约 35-48 C。
4. 因此不要把当前 8 盘读性能瓶颈解释成“pSLC cache 用完”。这类企业盘即使内部有写入缓冲，
   也没有通过标准 NVMe 字段暴露给我们；当前 full-hit 主要是 read path，pSLC cache 对这轮
   `Retrieved size/rank` 和 TTFT 解释力很弱。

## 当前状态提示

截至 2026-05-29：

- LMCache non-layerwise SSD-only 基础复用已经有过 DSv4 成功记录。
- residual proxy accuracy 在 DSv4 long prompt first decode 上有过约 0.91-0.93 的记录。
- 当前 prototype 仍不是最终 Tutti/gio_uring/BAT 系统。
- 当前 NVIDIA DSv4 hook 在 attention 后、FFN 前传 `residual` 给 manager；manager 使用目标
  CSA 层自己的 HC/attn_norm/indexer 计算 proxy top-K。
- AMD/ROCm hook 目前不是 residual proxy 展示路径。
- HCA deterministic overlap 已有 Python 原型：adapter 发现 HCA layers，LMCache retrieve 可
  seed HCA flat store，decoder 在上一层 attention 后、FFN/MoE 前提交下一 HCA 的
  deterministic read，并在目标 HCA attention 前 drain。第一层 prefire 仍禁用，因为
  当前 request slot mapping 尚未稳定。
- LMCache full-hit prefill tail 的 CSA prefetch 初始化已跑通 smoke，但仍是 prototype，
  需继续优化性能和长输出稳定性。
- 若容器里存在 `/tmp/lmcache_indexer_enable_prefill_eviction`，当前是在实验 prefill eviction path。
- 256K 上下文（`dsv4-256k`）：通过 runtime patch 绕过 assert 问题，≤10K tokens 通过，
  ~52K tokens 仍因 GPU connector contiguous slot offsets 错误失败；需新镜像含
  `_select_primary_block_ids(block_size=256)` 才能彻底解决（本地代码已有，待构建进镜像）。

## 2026-06-08 Tutti 容器 + CSA/HCA 激活状态

### 当前容器

```text
name:   dsv4-256k-measure-tutti
image:  lmcache/vllm-openai:indexer-ssd-hca-prefetch-decodegate-20260528_0630
启动脚本: /tmp/startup_256k_tutti.sh  (host: gpu002)
```

Tutti GPU-direct NVMe 已初始化（8 路 KIOXIA，workers 0/5 是 NVMe-oF skip，workers 1-4/6-7
是 PCIe）。CSA/HCA overlapping 代码已在镜像里，但尚未激活（env var 没设）。

### 已做的代码修复

`lmcache/integration/vllm/vllm_v1_adapter.py` — `_attach_indexer_prefetch()` 加了 rank suffix
（与 `_attach_hca_prefetch()` 对齐）：

```python
rank_suffix = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
store_dir = os.path.join(base_store_dir, f"rank_{rank_suffix}")
```

这个 fix 还没进镜像，需要通过 patch 目录部署到容器。

### 激活步骤（gpu002 上执行）

**1. 传文件**

```powershell
# 从 Windows 本机
scp F:\LMCache\lmcache\integration\vllm\vllm_v1_adapter.py master:/tmp/vllm_v1_adapter_csa.py
ssh master "sshpass -p 'Pass2025' scp /tmp/vllm_v1_adapter_csa.py zbuser02@172.16.8.32:/tmp/vllm_v1_adapter_csa.py"
scp C:\tmp\activate_csa_hca.sh master:/tmp/activate_csa_hca.sh
ssh master "sshpass -p 'Pass2025' scp /tmp/activate_csa_hca.sh zbuser02@172.16.8.32:/tmp/activate_csa_hca.sh"
```

**2. 在 gpu002 上 patch 启动脚本**

```bash
# 把新 vllm_v1_adapter.py 放进 patch 目录（会在容器启动时覆盖 site-packages）
cp /tmp/vllm_v1_adapter_csa.py /tmp/lmcache_patch/integration/vllm/vllm_v1_adapter.py

# 给 /tmp/startup_256k_tutti.sh 加 CSA/HCA env var
bash /tmp/activate_csa_hca.sh
sudo cp /home/zbuser02/startup_csa_hca.sh /tmp/startup_256k_tutti.sh
```

**3. 准备 SSD 目录**

```bash
# Workers 1-4 / 6-7 对应 /mnt/nvme2-9
for n in 2 3 4 5 6 7 8 9; do
  mkdir -p /mnt/nvme${n}/lmcache_csa 2>/dev/null || true
done
```

**4. 重启容器（用 sync_path_sharder.sh）**

```bash
bash /tmp/sync_path_sharder.sh
```

### 激活后应看到的日志

```text
# CSA
IndexerSSDManager: enabled CSA prefetch; csa_layers=30 pool_size=2048 ...
IndexerSSDManager: attached 30 CSA indexers

# HCA
HCAPrefetchManager: using transient pinned I/O slab size=64 MiB ...
HCAPrefetchManager: enabled pinned-transient HCA state; ...

# Tutti (already working)
TuttiDirectLoader initialised: worker=N device=/dev/ssnvmeN pci=... lba_cache=...
```

### 激活后的 env var 全集

```text
# Tutti
tutti_device_path=/dev/ssnvme<N>          (via lmcache yaml extra_config)
tutti_pci_bdfs=0000:10:00.0,...

# CSA
LMCACHE_INDEXER_ENABLE_PREFETCH=1
LMCACHE_INDEXER_SSD_DIR=/mnt/nvme0/lmcache_csa      # 每 worker 自动加 rank_N suffix
LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=1
LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=1             # ← 这个是创新点核心；开启后 HC-proxy overlapping 生效

# HCA
LMCACHE_HCA_ENABLE_PREFETCH=1
LMCACHE_HCA_ENABLE_PINNED_BOUNCE=1
LMCACHE_DSV4_DEFER_HCA_TO_MOE=1
```

**⚠️ 注意**：`LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=1` 是当前创新点的关键开关。之前所有容器
都把它关掉是因为 Python prototype 路径太慢（147 s decode）。现在要做的是验证这条路径是否
已经足够快（Tutti + overlapping），或者识别哪些环节还在阻塞。如果 decode 仍慢，需要继续
优化 `IndexerSSDManager.fire_async_for_layer()` 和 `drain()` 的 Python 开销。

### 还未完成的部分

1. `activate_csa_hca.sh` 还没在 gpu002 上实际执行 — 容器还在用旧启动脚本
2. `LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH=1` 路径的性能还没量过（上次全开跑到 147s）
3. Tutti + CSA overlapping 的真实时延还没测

## 后续修改规则

以后每次改代码，都必须同步更新
`F:\LMCache\docs\design\v1\dsv4_csa_hca_prefetch_source_of_truth.md` 或本文：

1. 改动改变系统语义、调用链、开关或实验结论时，写进主设计文档。
2. 改动只影响运行命令、服务器路径、诊断日志或禁忌项时，写进本文。
3. 不能只在聊天记录或临时 handoff 里说明思路；compact 后默认只信这两个文档。
