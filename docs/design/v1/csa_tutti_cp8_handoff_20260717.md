# CSA/Tutti CP8 工作对接文档（2026-07-17）

## 1. 文档范围和结论

本文只描述 GPU002 上最后完成的 `CP8 off vs CSA` 版本，不介绍已经从最新启动脚本和最终进程环境中删除的旧环境变量，也不把本地历史原型当成部署依赖。

- 上游比较基线：LMCache `4d2d4f373c2487f89378c0ef1bf3d95a5c6ceea3`，即当前分支与 `origin/dev` 的共同祖先（2026-05-19）。
- 最新部署目录：`/home/zbuser02/csa_tutti_latest/`。
- 最新可复现实验源目录：`/home/zbuser02/csa_cp8_ab_20260717/`。
- 最后完成的结果：`cp8_off_vs_csa_480k200_20260717_050431`。
- 部署方式：不重建基础镜像；容器启动时把最新 Python patch 和预编译 `c_ops` 覆盖到镜像的 `site-packages`。
- 当前实验开启的是 **CSA attention-KV filter + indexer prefetch + CP=8**。HCA prefetch/walker/decode hook 均关闭，因此当前版本不是“CSA、HCA、MoE 三者同时运行”。
- `profile80` 用跨层 lookahead 给 I/O 留出多个层的提前量；CP=8 只负责把 prefill proxy scoring 分摊到 8 个 rank，**不会凭空扩大单个 MoE 窗口**。
- 设计目标是把 indexer 预测和 Tutti I/O 前移并与后续计算重叠；但最后一次 480K+200 实测尚未达到目标：CSA CP8 的 hit 中位数为 7.231 s，Tutti-only 为 3.407 s，前者慢 3.824 s，即 2.123 倍。这个结果必须作为当前状态交接，不能写成已经加速。

## 2. 当前 pipeline

一次 cache-hit prefill 的关键路径如下：

```text
LMCache lookup/retrieve
  -> 识别 DeepSeek-V4 的 heterogeneous KV groups
  -> 从 KV object store 找到 CSA attention-KV 的 raw extent/chunk
  -> 注册 request/chunk 元数据到 CSAAttentionKVPrefetchManager
  -> 进入 decoder
       -> 在前置层的 FFN/MoE 附近触发 indexer proxy scoring
       -> CP=8：各 rank 只计算本地 K shard 的 top-k 配额，再转成全局 token id
       -> profile80：按目标层选择跨层 lookahead，而不是只依赖一个 MoE 窗口
       -> Tutti indexed SGL read：NVMe 直接读入 HBM staging
       -> CUDA scatter：把选中的 rows 写入目标 attention KV cache
       -> 目标 CSA attention 层消费前 drain；预测遗漏走 correction/miss read
  -> 正常 attention/FFN 计算
```

当前 HCA 路径虽然源码仍在最新 patch 包中，但所有 HCA 功能开关都为 0，所以不会参与上述运行。保留源码是为了后续实验，不代表它属于当前性能数字。

“Stage0 plan I/O”在当前交付中不是独立可见的运行阶段。实际有效工作是：预测结果生成后构造 indexed read 请求、提交 Tutti I/O、轮询完成并 scatter。任何只生成 plan、但没有成功提交和消费 I/O 的阶段都不会改善 TTFT。

## 3. 相比最早 LMCache 的文件改动

### 3.1 部署边界

最新远端 patch 包共有 25 个文件：24 个 Python/二进制资源加 1 个 `c_ops` 二进制。下表逐个说明所有最新部署中的 `.py` 文件。`csrc` 不直接拷入容器，而是编译进 `c_ops.cpython-312-x86_64-linux-gnu.so`；对应的 `.cu/.cuh/.cpp` 改动见 3.3。

本地曾经出现过 `csa_prefetcher.py`、`csa_ssd_pool.py`、`index_to_io_plan.py`、`index_to_io_profile.py` 和多批临时 benchmark/patch 文件，它们不在 `/home/zbuser02/csa_tutti_latest/patches`，不属于本次部署，也不应在其他机器上复制。

### 3.2 Python 文件

| 文件 | 相对最早 LMCache 的改动 | 当前运行中的作用 |
|---|---|---|
| `deepseek_v4.py` | 修改 vLLM 的 DeepSeek-V4 model：给 decoder layer 增加 layer id、manager attachment、CSA/HCA fire/prepare/drain 钩子和全局 decoder registry；在 attention 与 FFN/MoE 边界插入 prefetch 时机。 | CSA 开启时用于把 indexer scoring 和 I/O 放到目标层之前；HCA 钩子存在但本次关闭。此文件覆盖 vLLM 的多个 DeepSeek-V4 安装路径。 |
| `lmcache/integration/vllm/lmcache_connector_v1.py` | connector 增加 vLLM HMA 支持，注册异构 KV cache，并增加 `request_finished_all_groups`，把多 KV group 的完成通知兼容到 LMCache。 | 让 vLLM 0.20.2 的 hybrid KV cache manager 能把各 group 的 block 信息传给 LMCache。 |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | 加入 HMA block-id 归一化、scheduler invalid-block 兼容、DeepSeek decoder 发现/注册、FFN/HCA overlap hooks；创建并挂接 Indexer、CSA attention-KV、HCA managers；创建 Tutti indexer storage；把 HMA layout 和 transfer metadata 传到 connector。 | 当前 pipeline 的总编排层。CP、profile80、indexer/Tutti attachment 和请求生命周期主要在这里接通。 |
| `lmcache/utils.py` | 扩展 slot mapping、cache key 和 disk metadata 的处理，使异构 group、压缩映射及对象存储路径可表达。 | 为 HMA block 映射、object id 和存储元数据提供公共工具。 |
| `lmcache/v1/cache_engine.py` | 初始化/复用 Tutti loader；支持 raw extent 批量读取和写入；按 DeepSeek-V4 group role 过滤 retrieve/store shape；构建并注册 CSA/HCA chunks；接入 KV object store、Tutti warmup 和 batched get。 | cache hit 的入口：决定哪些数据同步恢复、哪些 CSA 数据交给异步 prefetch，并把 SSD 对象映射交给 manager。 |
| `lmcache/v1/csa_attention_kv_prefetch_manager.py` | 新增 CSA attention-KV manager、chunk/location/layer state；编译 layer-major/indexed tables；提交预测读和 miss correction；轮询 I/O；在 CUDA stream 上 scatter 到目标 KV cache；支持等待、drain、关闭和 timing。 | 当前 CSA attention-KV 数据面的核心。只有它实际提交 Tutti read 并 scatter，预测才会转化为可消费 KV。 |
| `lmcache/v1/csa_pipeline_nvtx.py` | 新增统一 NVTX range/mark/I/O 标记封装，包含 request、layer、stage、I/O 元数据标签。 | 仅在 `LMCACHE_CSA_PIPELINE_NVTX=1` 时用于 Nsight 分析；最终性能运行关闭。 |
| `lmcache/v1/csa_prefetch_policy.py` | 新增可解析的跨层 lookahead policy，并提供 `profile80` recall profile 和 residual source 构造。 | 当前用 `profile80`，按目标层选择更早的 source layer，解决单一 MoE window 不够大的问题。 |
| `lmcache/v1/csa_prefill_cp_scorer.py` | 新增 prefill CP scoring：按 rank 切 K、计算本地 top-k 配额、全局化 token id、根据 logits 内存预算切 query range。 | `CP_SIZE=8` 时把 proxy indexer 的计算和临时 logits 分摊到 8 个 rank；它降低单 rank 工作量，但不增加 wall-clock lookahead。 |
| `lmcache/v1/gpu_connector/gpu_connectors.py` | 支持 DeepSeek-V4 优化 KV/HMA group transfer；登记 KV cache 指针；区分 CSA/HCA/普通 group；生成 direct seed/defer metadata；增加 batched raw-to-GPU 和多对象 transfer。 | 把 LMCache object、vLLM paged KV cache 与 Tutti HBM staging 接起来，并避免不需要的整组同步 copy。 |
| `lmcache/v1/gpu_connector/tutti_direct_loader.py` | 新增 Tutti GPU-direct NVMe loader：打开控制设备、解析 FIEMAP/extents、创建每 rank queue/slot、注册 GPU staging、提交 batch/indexed SGL read/write、poll completion，并支持多 extent 和批处理。 | 当前 NVMe→HBM 的真实数据通路；要求 host `snvme` 驱动和 `/dev/snvm_control`、`/dev/ssnvme*`。 |
| `lmcache/v1/gpu_connector/utils.py` | 扩展 shape descriptor 和 format-dispatch transfer，支持 HMA/DeepSeek-V4 的物理 chunk、raw payload 和 batched transfer。 | 给 GPU connector/CUDA kernel 提供一致的 shape 和 transfer 参数。 |
| `lmcache/v1/hca_prefetch_manager.py` | 新增 deterministic HCA row prefetch、object-source/slab batching、HBM backing、异步 prepare/fire/drain、stream 同步和资源回收。 | 本次环境全部关闭，当前结果不包含 HCA prefetch；只作为后续实验代码保留。 |
| `lmcache/v1/indexer_ssd_manager.py` | 把官方 Lightning Indexer 作为预测真值；增加 indexer K 的 SSD/Tutti store、小型 HBM LRU pool、proxy/residual/reuse prefetch、prefill CP scoring、真实 top-k correction、异步 fire/prepare/drain。 | 当前 proxy 预测控制面：产生候选 token ids，并通知 CSA manager 发 indexed reads。 |
| `lmcache/v1/indexer_tutti_backend.py` | 新增 indexer 专用 raw-region 布局、layer slot、extent 映射、aligned write/read request 和 token-run coalescing。 | 将每层 indexer K 放在各 rank 的 512 MiB raw file 中，并为 Tutti 构造物理 extent 请求。 |
| `lmcache/v1/kv_layer_groups.py` | 扩展 heterogeneous/compressed KV group 描述，推导 compression metadata、physical chunk size 和每组 shape descriptor。 | 让 HMA 下不同 group 的逻辑 block、物理 chunk 和传输尺寸不再被当成同一种 KV。 |
| `lmcache/v1/kv_object_store/__init__.py` | 导出 object id、record/state、metadata store、pool layout 和 pool I/O API。 | object store 子系统的稳定入口。 |
| `lmcache/v1/kv_object_store/object_id.py` | 新增稳定的 KV object identity、序列化和 key 解析。 | 将 request/token range/group/rank 对应到可持久化对象。 |
| `lmcache/v1/kv_object_store/record.py` | 新增对象生命周期、byte ranges、raw extents、ready/evicted 状态及 JSON 序列化。 | 记录对象在普通文件和 Tutti raw region 中的精确位置，供 direct read 判断。 |
| `lmcache/v1/kv_object_store/metadata_store.py` | 新增线程安全 metadata index、批量查询、JSONL dump/load。 | 进程内查询可读对象，并可恢复已有 metadata。 |
| `lmcache/v1/kv_object_store/pool_layout.py` | 新增固定 slot/alignment 的确定性 pool allocator、容量检查和 reset。 | 把大量 KV chunk 放进预分配文件，避免逐对象创建文件。 |
| `lmcache/v1/kv_object_store/pool_io.py` | 新增 `pread/preadv/pwritev`、批量对象读写和 read-into。 | 普通文件 fallback/建库路径；Tutti direct raw read 使用 record 中映射出的 extent。 |
| `lmcache/v1/storage_backend/local_disk_backend.py` | 接入 KV object store；生成 HCA/CSA layer-major snapshot；管理 raw-region writer/extent cache；判断 object 是否可被 Tutti 读取；支持对象写入、扫描恢复和 batched get。 | SSD 元数据与物理布局的主要 owner，是 cache engine 和 Tutti loader 之间的桥。 |
| `lmcache/v1/storage_backend/storage_manager.py` | 扩展 batch allocate/get/put、后端调度和请求取消，使 object/batched retrieve 能通过现有 storage manager 生命周期运行。 | 为 cache engine 提供统一异步存储接口，避免 CSA 路径绕开后端资源管理。 |

### 3.3 C++/CUDA 文件和二进制

| 文件 | 改动 | Python 暴露/效果 |
|---|---|---|
| `csrc/mp_mem_kernels.cu` | 新增多对象、多 layer 的 batched paged-KV transfer kernel；新增按 object pointer 和 row id 的向量化 scatter，支持对齐时 `uint64_t` copy、未对齐时 byte copy。 | `multi_layer_block_kv_transfer_batched`、`scatter_rows_from_object_ptrs`；减少 Python 循环和每对象 kernel launch。 |
| `csrc/mp_mem_kernels.cuh` | 声明上述 batched transfer/scatter host wrappers。 | 供 pybind 和其他 CUDA translation unit 使用。 |
| `csrc/tutti_kv_ops.cu` | 新增 GPU 侧 NVMe SQE/SGL 构造、indexed SGL read、batch read/write、CQ polling/backoff 和 queue state 包装。 | `tutti_submit_batch_sgl_read`、`tutti_submit_indexed_sgl_read`、`tutti_submit_batch_sgl_write`、`tutti_poll_batch`；让提交和完成轮询不经 CPU payload copy。 |
| `csrc/tutti_kv_ops.cuh` | 定义上述 Tutti host API 和参数契约。 | 与 loader/pybind 保持 ABI。 |
| `csrc/pybind.cpp` | 注册 batched KV transfer、row scatter 和 4 个 Tutti I/O API。 | 形成 `lmcache.c_ops` 的 Python 入口。 |
| `patches/c_ops.cpython-312-x86_64-linux-gnu.so` | 上述 C++/CUDA 的预编译产物。 | 只兼容当前交付的 CPython 3.12、Linux x86_64、镜像内 PyTorch/CUDA ABI；换 Python/PyTorch/CUDA 后必须重编。 |

最新二进制的 SHA256 为 `a12bbe1c60d04dbb8051995cf0062d45c564f9d61caab8c61576d4c263efd5cf`。启动脚本会在 CSA case 启动前检查 4 个关键 Tutti/scatter symbol；缺失即 fail closed。

### 3.4 不在最新部署包中的本地改动

以下内容曾用于探索，但不属于最新交付：

- `lmcache/v1/csa_prefetcher.py`、`lmcache/v1/csa_ssd_pool.py`：早期 HC proxy/SSD-HBM pool 原型。
- `lmcache/v1/index_to_io_plan.py`、`lmcache/v1/index_to_io_profile.py`：独立 index-to-I/O 规划/分析原型。
- 根目录 `bench_*.py`、`test_*.py` 和 `tmp_gpu002_logs/` 中的 staged/patch 副本：实验过程资产，不是运行时 source of truth。
- `vllm_service_factory.py`、lookup/metadata/path-sharder 等本地历史修改：没有进入最终 CP8 patch manifest，不应靠复制本地工作树来复现。

如果后续要把最新实现正式合并回 LMCache，应从远端 patch manifest 反向整理成干净提交，而不是直接提交当前含历史原型的工作树。

## 4. 驱动切换

### 4.1 当前 CP8/Tutti 模式

CP8 容器必须看到：

```text
/dev/snvm_control
/dev/ssnvme0 ... /dev/ssnvme7（实际编号以主机为准）
```

Tutti 仓库位于 `/home/zbuser02/Tutti`。安全重载脚本是：

```bash
cd /home/zbuser02/Tutti
sudo bash scripts/reset_snvme.sh
```

脚本流程是 unbind controller、检查 fd holder、卸载 `snvme/snvme_core`、重新插入模块。若 `rmmod` 失败会直接停止；若出现 duplicate sysfs filename，不应反复 `insmod`，需要重启机器。

`reset_snvme.sh --force-cleanup` 会 SIGKILL 所有占用 `/dev/snvm*` 的进程，只能在确认没有他人任务时使用。`--no-insmod` 只用于重编驱动前卸载模块。

### 4.2 切换到普通 NVMe/GDS 模式

使用本地留档的 `tmp_gpu002_logs/switch_snvme_to_nvme_gds.sh`，远端运行时应放到一个固定运维目录：

```bash
bash switch_snvme_to_nvme_gds.sh
```

它会先确认没有运行容器/GPU 进程，然后卸载 8 个 mount、调用 Tutti reset 的 `--no-insmod`、加载内核 `nvme`、对 8 个 BDF 执行 probe、`mount -a` 并逐盘验证 driver/mount。

相关脚本职责：

| 脚本 | 用途 | 注意 |
|---|---|---|
| `switch_snvme_to_nvme_gds.sh` | 从 Tutti/snvme 完整切回原生 NVMe/GDS。 | 推荐的整机切换入口；要求容器和 GPU 空闲。 |
| `complete_nvme_rebind.sh` | 已完成 snvme unbind 后，清理 `driver_override`、probe 8 个 BDF、挂盘并验证。 | 是中断恢复脚本，不替代前置 fd/mount 安全检查。 |
| `mount_gds_nvmes.sh` | 只挂载并检查 8 个原生 NVMe mount。 | 不切驱动。 |
| `check_after_snvme_unbind.sh` | 查看 module、每个 BDF driver、fd holder 和 mount。 | 只读诊断。 |
| `restore_gpu002_nvme0.sh` | 只修复 BDF `0000:6f:00.0` 和 `/mnt/nvme0`。 | 单盘故障恢复，不可当作 8 盘切换脚本。 |

8 个固定 BDF 为：

```text
0000:6f:00.0  0000:10:00.0  0000:1c:00.0  0000:4b:00.0
0000:e4:00.0  0000:88:00.0  0000:cc:00.0  0000:a2:00.0
```

对应 mount 为 `/mnt/nvme0,2,3,4,5,6,8,9`。切驱动是整机级破坏性操作，不能与其他 NVMe/GPU 作业并行。

### 4.3 跑实验前是否需要编译 Linux 内核

**不需要编译 Linux 内核。** 实验依赖的是 Tutti 的 out-of-tree 内核模块和 LMCache 的 CUDA 扩展，这两个产物与 Linux 内核本体是三件不同的事。

- 在当前 GPU002 上，如果 `/dev/snvm_control`、`/dev/ssnvme*` 已存在，且 `snvme`、`snvme_core` 已加载，直接运行实验，不需要重新编译。
- 换到另一台机器且内核版本或内核配置不同，只需要针对目标机当前内核重新编译 `snvme_core.ko` 和 `snvme.ko`，不需要重新编译整个 Linux 内核。
- 复用已有 `.ko` 前必须比较 `modinfo` 的 `vermagic` 与 `uname -r`。不匹配时禁止直接加载，应安装 `linux-headers-$(uname -r)` 后在目标机重编模块。
- `c_ops.cpython-312-x86_64-linux-gnu.so` 是单独的 LMCache C++/CUDA 扩展。只有 CPython 3.12、Linux x86_64、PyTorch 和 CUDA ABI 与交付镜像兼容时才能直接复用；更换 Python、PyTorch、CUDA 或基础镜像后需要重新编译 `c_ops`，但仍不需要编译 Linux 内核。

当前 GPU002 的跑前检查：

```bash
uname -r
lsmod | grep -E 'snvme|snvme_core'
ls -l /dev/snvm_control /dev/ssnvme*
```

目标机器复用已有内核模块前还要检查：

```bash
uname -r
modinfo /path/to/snvme_core.ko | grep vermagic
modinfo /path/to/snvme.ko | grep vermagic
```

如果设备节点已经存在且版本检查通过，不要为了“保险”在每次实验前重新编译或重载模块。驱动重载会影响整机 NVMe 状态，只应在设备未就绪、模块需要更新或明确排障时执行。

## 5. 容器和实验脚本

最新脚本位于 `/home/zbuser02/csa_tutti_latest/scripts/`；其中 `current_tag.txt` 和 `completed_tag.txt` 只是结果 tag，不是可执行脚本。

| 脚本 | 作用 | 是否直接运行 |
|---|---|---|
| `run_cp8_ab.sh` | 完整 A/B harness：检查 GPU 空闲，依次运行 `tutti_only off` 与 `tutti_csa_cp8 on profile80`，采集进程环境、日志、4 次 hit，验证 8 rank loader/manager/policy，最后生成 `comparison.json`。 | **推荐入口**。 |
| `run_container_cp8_ab.sh [off/on]` | 启动单个容器，传入设备、patch、环境变量并等待 8000 端口 ready。 | 调试单 case 时运行；正式 A/B 由上一个脚本调用。 |
| `startup_cp8_ab.sh` | 容器 entrypoint：覆盖 site-packages、验证 ABI/symbol、生成 LMCache YAML、创建 raw region、导出最终环境并启动 vLLM。 | 不在 host 手工运行。 |
| `run_hermes_trial2_480k200.py` | 构造固定 Hermes prompt，执行一次 cold store、一次不计入结果的 warmup hit 和 4 次计时 hit。 | 可在 server ready 后单独运行。 |

完整运行：

```bash
ssh gpu002-vscode
cd /home/zbuser02/csa_tutti_latest/scripts
bash run_cp8_ab.sh
```

单独启动 CSA case：

```bash
cd /home/zbuser02/csa_tutti_latest/scripts
bash run_container_cp8_ab.sh on
curl -s http://127.0.0.1:8000/v1/models
```

单独启动 Tutti-only baseline：

```bash
bash run_container_cp8_ab.sh off
```

基础镜像固定为 `lmcache/vllm-openai:indexer-ssd-hca-prefetch-decodegate-20260528_0630`，本次记录的 image ID 为 `sha256:1e087...`、vLLM 版本为 0.20.2。镜像没有 registry digest；迁移到其他机器时必须用完整镜像 tar 或重新生成可校验 digest，不能只凭 tag 假设内容相同。

容器参数包括 8 GPU、host network/PID/IPC、`--privileged`、`SYS_ADMIN`、`SYS_RAWIO`、64 GiB shm，并挂载 `/sys`、`/mnt`、`/tmp`、模型、patch 和 startup。模型固定为 `/mnt/nvme0/models/DeepSeek-V4-flash`。

## 6. 最新版本仍在使用的环境变量

本节以最后成功 case 的 `process_env.txt` 和当前三份启动脚本为准。已从最新版本删除的旧变量不列出。

### 6.1 Host A/B harness 输入

这些变量由 host 脚本消费；`LMCACHE_ABLATION_*` 是实验包装参数，不是新的 engine feature flag。

| 变量 | 最终值 | 何时设置/作用 |
|---|---:|---|
| `LMCACHE_ABLATION_PATCH_DIR` | `${root}/patches` | patch 不在默认目录时设置；正式 harness 自动设置。 |
| `LMCACHE_ABLATION_STARTUP_SCRIPT` | `${root}/startup_cp8_ab.sh` | 指定容器 entrypoint；正式 harness 自动设置。 |
| `LMCACHE_ABLATION_CSA_ATTENTION_KV_FILTER` | baseline `0`，CSA `1` | A/B 总开关；`1` 派生 indexer/CSA 的 engine 开关。 |
| `LMCACHE_ABLATION_MAX_MODEL_LEN` | `530000` | 480K context 实验需要；转成 vLLM `--max-model-len`。 |
| `LMCACHE_ABLATION_MAX_BATCHED_TOKENS` | `65536` | 控制 chunked prefill 每批 token 上限。 |
| `LMCACHE_ABLATION_GPU_UTIL` | `0.60` | 转成 vLLM `--gpu-memory-utilization`，给 staging/object/indexer pool 留 HBM。 |
| `LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES` | 空 | 只有需要精确覆盖 vLLM KV cache 字节数时填写；空值不传该参数。 |
| `LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB` | `8` | 生成 object-store slot 大小。 |
| `LMCACHE_ABLATION_KV_OBJECT_STORE_CAPACITY` | `48000` | 生成 object-store slot 数量/容量上限。 |
| `LMCACHE_ABLATION_TUTTI_N_SLOTS` | `4` | 每 rank Tutti 并发 staging/queue slot 数。 |
| `LMCACHE_ABLATION_TUTTI_SLOT_MB` | `128` | 每个 Tutti staging slot 的 HBM 大小。 |
| `LMCACHE_ABLATION_TUTTI_STARTUP_DELAY` | `120` | server 启动后 Tutti warmup 前等待秒数。 |
| `LMCACHE_ABLATION_TUTTI_AFTER_STORE_DELAY` | `10` | cold store 后 warmup/read 前等待秒数。 |

### 6.2 CSA、CP 和 indexer

| 变量 | CSA case 最终值 | 何时开启/含义 |
|---|---:|---|
| `LMCACHE_DSV4_CSA_ATTENTION_KV_FILTER` | `1` | CSA attention-KV 异步筛选总开关。只有准备好真实 Tutti raw extent 和新 `c_ops` 时开；baseline 为 0。 |
| `LMCACHE_INDEXER_ENABLE_PREFETCH` | `1` | 启用 indexer 预测。由 A/B 总开关派生；baseline 为 0。 |
| `LMCACHE_INDEXER_FULL_OVERLAP` | `1` | 允许预测/读请求尽早与 decoder 计算重叠；CSA case 开。 |
| `LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY` | `1` | 用前置 residual hidden state 做下一 CSA 层 proxy scoring。CSA case 开。 |
| `LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH` | `1` | cache reuse/prefill 路径继续使用预测 prefetch。CSA case 开。 |
| `LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER` | `profile80` | 使用按层拟合的 lookahead；当前解决“单个 MoE 窗口不够”的核心策略。baseline 不设置。 |
| `LMCACHE_CSA_PREFETCH_CP_SIZE` | `8` | prefill scoring 使用 8-way CP/K-shard。只能在 TP8/8 rank 的当前布局开；baseline 不设置。 |
| `LMCACHE_CSA_PREFETCH_CP_INTERLEAVE` | `64` | CP K-index 的交错粒度；保持各 rank shard 分布均衡。 |
| `LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE` | `1` | 本地 top-k 配额的 oversubscribe 系数；1 表示不额外放大候选。 |
| `LMCACHE_CSA_ATTENTION_KV_PROXY_MICROBATCH_ROWS` | `64` | proxy scoring 的 query row microbatch，控制临时 logits/HBM 与 launch 数。 |
| `LMCACHE_INDEXER_IO_WORKERS` | `8` | indexer 后台 I/O worker 数。当前按 8 rank/8 SSD 配置。 |
| `LMCACHE_INDEXER_MAX_SEQ_LEN` | `131072` | 每层 indexer raw slot 可容纳的最大 token 数；布局参数，改后必须重建/校验 raw file。 |
| `LMCACHE_INDEXER_SSD_DIR` | 8 个 `/mnt/nvme*/lmcache_csa` 路径 | 普通文件/fallback 的按盘目录。 |
| `LMCACHE_INDEXER_TUTTI_BACKEND` | `1` | indexer K 使用 Tutti raw backend；只有 snvme 设备、raw file 和 extents 都就绪时开。 |
| `LMCACHE_INDEXER_TUTTI_RAW_REGION_PATH` | 8 个 `indexer_raw_region_512m.bin` | 每 rank indexer raw file。启动脚本首次创建 512 MiB 文件。 |
| `LMCACHE_REUSE_PREFETCH_ASYNC` | `0` | 当前关闭额外 reuse 异步层，避免与已存在的 manager 生命周期叠加。 |
| `LMCACHE_INDEXER_CROSS_LAYER_PREFETCH` | `0` | 旧的通用 cross-layer 入口关闭；当前只使用显式 `profile80` policy。 |
| `LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH` | `0` | decode 阶段预测关闭；当前只验证 prefill hit。 |
| `LMCACHE_INDEXER_ENABLE_PREFILL_RESIDUAL_PROXY` | `0` | 另一条 prefill residual 原型入口关闭，避免重复 fire。 |
| `LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION` | `0` | prefill 结束后 indexer pool eviction 关闭。 |
| `LMCACHE_INDEXER_ENABLE_POOL_SCORING` | `0` | 只对 HBM pool 候选 scoring 的实验路径关闭。 |
| `LMCACHE_INDEXER_REUSE_RESIDUAL_TOPK` | `0` | 复用上一层 residual top-k 的实验路径关闭。 |
| `LMCACHE_INDEXER_EXPERIMENTAL_RESIDUAL_LOOKAHEAD` | `0` | 实验性 residual lookahead 关闭；当前 policy 只由 profile80 决定。 |
| `LMCACHE_INDEXER_PROFILE_ACCURACY` | CSA `1`，baseline `0` | CSA case 必须开启：把预测 compressed blocks 与官方 Lightning Indexer true tokens 映射出的 true blocks 比较，输出 `block_recall`；harness 在没有 accuracy 记录时直接判失败。 |

### 6.3 KV 布局、object store 和 Tutti

| 变量 | 最终值 | 何时开启/含义 |
|---|---:|---|
| `LMCACHE_CONFIG_FILE` | `/tmp/lmcache_ssd_tutti_kvobj.yaml` | startup 动态生成的唯一 LMCache 配置入口。 |
| `LMCACHE_ENGINE_LOGICAL_BLOCK_SIZE` | `256` | LMCache logical chunk/block 大小；必须与 YAML `chunk_size`、HMA 映射一致。 |
| `LMCACHE_DSV4_OPTIMIZED_KV` | `1` | 开启 DeepSeek-V4 heterogeneous KV 优化布局。 |
| `LMCACHE_DSV4_OPTIMIZED_TAIL_TOKENS` | `256` | 尾部最多 256 token 使用优化 transfer policy。 |
| `LMCACHE_KV_OBJECT_STORE_ENABLE` | `1` | 启用固定 pool 的 KV object store。 |
| `LMCACHE_KV_OBJECT_STORE_SLOT_MB` | `8` | 每 object slot 8 MiB；由 ablation 输入派生。 |
| `LMCACHE_KV_OBJECT_STORE_CAPACITY` | `48000` | object store 容量参数；由 ablation 输入派生。 |
| `LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE` | `1` | 为 object record 建立 Tutti 可读 raw extent。只有使用 snvme direct path 时开。 |
| `LMCACHE_TUTTI_STARTUP_WARMUP_DELAY_SEC` | `120` | engine 内 Tutti startup warmup 延迟。 |
| `LMCACHE_TUTTI_WARMUP_AFTER_STORE_DELAY_SEC` | `10` | store 后 warmup 延迟。 |

YAML 中另有不是环境变量、但同样固定的 Tutti 参数：control path `/dev/snvm_control`、device `/dev/ssnvme0`、8 个 PCI BDF、NSID=1、4 个 128 MiB slot，以及 8 个 16 GiB `rank_raw_region_3g.bin`。文件名中的 `3g` 已与实际 16 GiB 大小不一致，只是历史命名，判断容量必须看文件实际大小和 extent，不能看名字。

### 6.4 明确关闭的 HCA 路径

这些变量仍在最终进程环境中，作用是保证当前 A/B 不混入 HCA；不是被删除的旧变量。

| 变量 | 最终值 | 含义 |
|---|---:|---|
| `LMCACHE_DSV4_HCA_WALKER` | `0` | 不启用 HCA walker。 |
| `LMCACHE_HCA_ENABLE_PREFETCH` | `0` | 不启用 HCA manager 数据预取。 |
| `LMCACHE_HCA_ENABLE_PINNED_BOUNCE` | `0` | 不使用 HCA CPU pinned bounce。 |
| `LMCACHE_DSV4_DEFER_HCA_TO_MOE` | `0` | 不把 HCA retrieve 延迟到 MoE window。 |
| `LMCACHE_HCA_ENABLE_DECODE_HOOK` | `0` | decode HCA hook 关闭。 |

因此，当前 CP8 结果不能用于证明 HCA+CSA+MoE 的三路 overlap，也没有验证单个 MoE window 是否足以覆盖 HCA I/O。

### 6.5 Profiling 和通用进程变量

| 变量 | 最终值 | 使用方式 |
|---|---:|---|
| `LMCACHE_INDEXER_TIMING` | `0` | 调 indexer 阶段耗时时临时设 1；正式 timing 必须为 0，避免日志扰动。 |
| `LMCACHE_INDEXER_TIMING_LIMIT` | `20000` | timing 开启时的最大记录数。 |
| `LMCACHE_CSA_ATTENTION_KV_TIMING` | `0` | 调 CSA submit/poll/scatter 时临时设 1。 |
| `LMCACHE_TUTTI_PROFILE` | `0` | 调 Tutti queue/read 时临时设 1。 |
| `LMCACHE_TTFT_STAGE_PROFILE` | `0` | 调 TTFT 分阶段时临时设 1。 |
| `LMCACHE_CSA_PIPELINE_NVTX` | `0` | Nsight Systems 采集时设 1；非 profiling 关闭。 |
| `LMCACHE_HCA_TIMING` | `0` | HCA timing；当前 HCA 关闭。 |
| `PYTHONHASHSEED` | `0` | 保持 Python hash 顺序可复现。 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 降低长上下文/多 staging buffer 下的 CUDA allocator 碎片。 |
| `MODEL_PATH` | `/pro_model` | 容器内模型只读挂载点。 |

### 6.6 Workload 输入

| 变量 | 最终值 | 含义 |
|---|---:|---|
| `BASE_TOKENS` | `480000` | cold prefix 长度。 |
| `RECOMPUTE_TOKENS` | `200` | hit 时新增/重算 token 数。 |
| `NUM_WARMUP_HITS` | `1` | 不计入比较的 warmup hit 数。 |
| `NUM_HITS` | `4` | 计时 hit 数。 |
| `HIT_WAIT_S` | `5` | hit 之间等待秒数。 |
| `ENABLE_TORCH_PROFILE` | `0` | 正式比较关闭 profiler。 |

## 7. 预期效果与当前实测

### 7.1 预期效果

按设计，各部分应带来：

1. HMA/optimized-KV：只搬运当前 group 真正需要的 payload，避免把 DeepSeek-V4 的所有 KV 当成同构 full layer。
2. KV object store/raw extent：把 token/layer 对象映射到预分配文件和物理 extent，避免大量小文件和 CPU 拼包。
3. Tutti direct loader：NVMe→HBM，减少 page cache/CPU bounce 和 Python per-I/O 调度。
4. CP=8 scorer：每 rank 只处理本地 K shard，降低单 rank logits 计算量和峰值显存。
5. profile80：跨多个 decoder layer 提前触发，给 NVMe read+poll+scatter 足够 wall-clock 时间，而不是只赌一个 MoE window。
6. indexed read+scatter：只读预测 top-k rows，理论 I/O 量低于整层 attention-KV 恢复。
7. correction：预测漏项时补读，保证正确性；代价取决于 recall 和补读是否落在关键路径。


两组 `correction_records` 都是 0；CSA case 记录到 8 个 manager attachment 和 8 个 profile80 policy attachment。说明功能接通了，但当前开销/等待没有被计算覆盖。

当前最可能需要继续拆分的时间包括：CP scorer 本身、每层 fire/prepare 频率、indexed read 数量与 coalescing、CQ polling、scatter、目标层 drain 等待，以及首次 hit 的建表/建 cache 成本。下一轮优化应先开 timing/NVTX 做单变量 profile，不能同时打开所有 profiler 后直接比较 TTFT。

## 8. 在其他机器复现的最低条件

1. 8 张同等级 GPU；当前记录为 8×H200 143771 MiB。
2. host driver 580.126.20、kernel 5.15.0-185 或经过 ABI 验证的等价环境。
3. 完整 Docker image，而不只是 tag；容器内 CPython 3.12、PyTorch/CUDA ABI 必须匹配 `c_ops.so`。
4. 8 块 NVMe，BDF/mount 与脚本一致，或同步修改 startup YAML、raw paths 和 driver 脚本。
5. 与 host kernel 匹配的 Tutti `snvme/snvme_core`，并能生成 `/dev/snvm_control` 和 `/dev/ssnvme*`。
6. DeepSeek-V4-flash 模型挂在 `/mnt/nvme0/models/DeepSeek-V4-flash`，以及相同 Hermes 数据集/构造脚本。
7. 完整复制 `/home/zbuser02/csa_tutti_latest/patches` 和 `scripts`，并校验 SHA256 manifest。
8. 先跑 `off` 验证真实 Tutti path，再跑 `on`；不能拿 page-cache/普通文件 fallback 当 Tutti baseline。

当前 Tutti 仓库记录的 commit 是 `b9eb759...`，但工作树带未提交的 kernel 修改。这个状态是复现风险：只 clone 该 commit 不能保证得到 GPU002 上实际驱动。正式移交前应把 Tutti kernel diff 单独提交/打 patch，并记录 `.ko` SHA256、`modinfo` 和 build kernel headers。

## 9. 交接验收清单

- `sha256sum patches/c_ops*.so` 与本文一致。
- 容器日志出现 8 次 `TuttiDirectLoader initialised:`。
- CSA case 的进程环境有 `CP_SIZE=8`、`CP_INTERLEAVE=64`、`CP_OVERSUBSCRIBE=1`；baseline 中四个 CP/policy 变量不存在。
- CSA case 出现至少 8 次 manager attach 和 8 次 `canonical L2 policy=profile80`。
- CSA case 的 `LMCACHE_INDEXER_PROFILE_ACCURACY=1`，且日志至少出现一条 `attention_true_topk_profile`；baseline 为 0 且不得出现 accuracy 记录。
- baseline 的 `LMCACHE_INDEXER_ENABLE_PREFETCH=0`，CSA case 为 1。
- HCA 相关变量全部为 0。
- 日志无 OOM、illegal access、Traceback、`native_full_layer` 和 overlap hook failure。
- workload 是 480000 base + 200 recompute、1 warmup + 4 measured hits。
- `comparison.json`、两个 `case_summary.json`、`process_env.txt`、`server.log` 和 `container_inspect.json` 齐全。
- 结果必须同时报告绝对秒数和相对 Tutti-only ratio；当前版本应明确标为性能回退、待优化。
