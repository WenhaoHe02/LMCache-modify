# 交接文档:DSv4 CSA/HCA Prefetch(2026-07-08)

> 写给下一个接手的人(或下一个 session 的我)。诚实版:任务是什么、
> 代码被我改成了什么样、哪些可信、哪些是弯路、哪些直接是错的。

---

## 0. 一句话现状

**论文主线是"预测式预取"(HC-proxy 预测 top-K → 提前读 → 藏进 MoE 窗口)。
我这两天跑偏去做了"bulk 全读 + HBM 驻留复用"(V24-V29),端到端数字好看
但偏离论文命题;会议已纠偏。预测路径代码完好(一个环境变量即可切回),
当前正在做会议布置的阈值扫描,发现并需要先修一个测量污染 bug。**

---

## 1. 会议定下的任务(当前的真任务)

来源:2026-07-08 会议纪要。

1. **阈值扫描**:明确"重算 token 数达到多少时预测方案有效"——比较
   T_load(预测读时间)与 T_MoE(MoE 窗口),给出随上下文变化的判定公式。
   **不要大改现有方案。**
2. **精细 profiling**:拉 timeline 排查实现问题(代码空跑、计时污染等)。
3. **跨层预测验证**:离线验证隔 2/3/4 层预测的准确率(GLM 5.2 四层共享
   indexer 的思路),复用现有脚本,无需端到端。
4. indexer 多算子 launch 开销优化(CUDA graph 方向),先由 profiling 定量。

### 任务 1 当前进度

- 实验矩阵:两臂(`LMCACHE_CSA_BULK_PREDICTED=0` 预测 / `=1` bulk)
  × extra={500,1K,2K,4K,8K} × base={64K,480K},脚本 `/tmp/thsweep.sh`
  (gpu002),链式驱动 `/tmp/thchain.sh`,汇总 `/tmp/thsum.sh`。
- **已完成格**:预测/64K(hit 数据在,worker 指标因重定向 bug 丢失,已
  排队重跑)、bulk/64K。480K 两格在链上。
- **已拟合(可信)**:load 时间公式,1872 个实测点:
  - `T_NVMe(ms) = 0.57 + MB/11.05`(纯 DMA,poll_sync 口径)
  - `T_load(ms) = 3.85 + MB/10.97`(含 host,每调用固定开销 3.85ms)
  - 散块附加 ~0.02ms/IO。带宽 11 GB/s/rank 与 H200 平台一致。
- **已作废(我报错的)**:`T_MoE = 4.1 + 0.67·Q_k` ——用被预测读污染的
  均值拟的,无误差报告。真实分布双峰(普通层 p50≈3ms,重层 p90≈20ms),
  std 是 mean 的 1.5-8 倍。**教训:必须用 bulk 臂/OFF 测窗口,预测臂的
  MOE_TIMING 计时段混入了预测读 dispatch。**
- **待修 bug(阻塞任务 1)**:`deepseek_v4.py` 的 MOE_TIMING 计时窗口
  包含了同段执行的预测读 dispatch → 预测臂窗口测量被自己的读污染。
  修法:给 dispatch 单独计时或从计时段摘除(一行级)。
- 初步结论(用干净口径重验前仅供参考):几十 token 重算场景,预算
  `s·B < ~12 块`,固定开销 3.85ms 占预算 93% → **出路是砍 launch 开销
  (graph 化,3.85→<1ms)+ 跨层扩窗,不是调预取策略**。

---

## 2. 代码现状:我改了什么(全部 flag-gated,默认关)

分支 `codex/tutti-lazy-multiextent`,~35 个未 push 的本地 commit。
gpu002 部署副本在 `/home/zbuser02/codex_sync_overlap_fix/patches/`
(有 .bak_v25/.bak_prev28/.bak_prev29/.v27_win_saved 各阶段备份)。

### 2.1 预测路径(论文主线,完好,未动语义)

- 入口 `fire_predicted_reads`(csa_attention_kv_prefetch_manager.py),
  HC-proxy 在 indexer_ssd_manager.py。
- **`LMCACHE_CSA_BULK_PREDICTED=0` 即恢复预测路径,零代码改动。**
- smoke 已验:64K+2K 下 1176 次 dispatch、miss 大多为 0、hit2-4≈1.1s。
- 这两天修的底座 bug 预测路径直接受益:staging 池 4→16 槽(mega-batch
  11GB/s)、extent bisect 索引、io_lock 死锁、GPU bitmap、drain 泄漏、
  物理行寻址(slot_mapping)。

### 2.2 我加的三层"驻留式"机制(偏离主线的产物,但可作为大重算 fallback)

按环境变量分层,全部默认关:

| Flag | 名字 | 干什么 | 状态 |
|---|---|---|---|
| `LMCACHE_CSA_BULK_PREDICTED=1`(注意默认是 1!) | bulk walker | 不预测,layer-major 全量读藏进 compute;armed at register、started at first gate;单层单 Tutti 调用是硬不变量(违反=illegal access,两次实测) | 稳定 |
| `LMCACHE_CSA_WALKER_RESIDENT_SKIP=1` | V27 行搬迁 | 同前缀重复请求:chunk key 相同但 vLLM 分配器换了物理行(实测 100% churn)→ GPU 内 gather 旧行 scatter 新行(~250ms)替代重读 | 稳定,串行负载专用 |
| `LMCACHE_DSV4_HCA_WALKER=1` | V28 HCA 接管 | walker 扩到 HCA 20 层(占 ON 剩余同步读 84%);解了 512B 对齐(payload_skip 窗口)、cr=128 粒度(4 chunk 共享一物理块,压缩条目行寻址)、非连续张量(block/slot index_put_);gate 走 vLLM 每层 wait_for_layer_load 钩子 | 稳定 |
| `LMCACHE_DSV4_VECTORIZED_CONSUME=1` | V29 向量化搬运 | retrieve 的 per-chunk scatter(1874 次小 kernel≈1.9s)合批,kernel 一次最多 4 指针 | **16K/32K 验证过;48K 崩过 illegal access,已隔离;有 failsafe 自动回退** |

这套东西的端到端数字(如果论文要用"大重算 fallback"叙事):
16K 稳态 4.6-5.3s vs OFF 6.0-6.8(-25%);32K 6.2-7.7 vs 7.7-8.5;
repeat 3.05s 历史最低。**但注意:稳态赢依赖上一请求的 HBM 残留,
纯盘上路径(hit-1)仍慢(7-19s,含每 boot 一次 ~10s 的 miss 自读等锁,
根因=walker 起跑撞多步 retrieve 的 io_lock,未修)。**

### 2.3 我搞出来的烂账(按严重度)

1. **方向跑偏本身**:V24-V29 一周做的是"不预测"路线,和论文命题相反。
   会议已纠偏。这些代码留作 fallback,别再当主线投入。
2. **MOE_TIMING 计时污染**(见 1 节,阻塞当前任务,必须先修)。
3. **错误拟合并当结论报告**:无误差的 T_MoE 线性公式;"10.3s 是 indexer
   warmup"的错误归因(实为 miss 自读等 io_lock);"pre-retrieve 2.84s
   是 tokenize"的无测量推断(实际未拆分,只是间隙)。文档里已部分修正,
   但 docs/design/v1/csa_prefetch_v24_v25_findings.md 里仍有这些痕迹。
4. **V29 的 48K illegal access 未根因**(嫌疑:尾 chunk/prefix-skip 混入
   批量路径),该 flag 在 48K 禁用。
5. **胡编过一个概念**:"MLA 主 latent KV 组"——DSv4-Flash 没有这个组
   (SWA43+CSA21+HCA20,组表实测在 csa_prefetch_gpu_centric_plan.md),
   相关死代码已删,但看旧 commit 信息时注意别被误导。
6. **thsweep.sh 首格的重定向 bug**(`2>&1 > file` 丢 stderr),已修,
   首格已排队重跑。

---

## 3. 环境与操作(不看会浪费半天的事)

- **SSH 三跳**:`ssh master "sshpass -p 'Pass2025' ssh zbuser02@172.16.8.32 'CMD'"`
  (PowerShell 发出;引号嵌套极易翻车,复杂命令一律写成脚本 scp 过去跑)。
- **UVM 泄漏**:每容器周期泄 ~1-2GB/卡;`memory.used > 10GB` 时实验作废
  概率高(walker 线程会在 set_stream OOM)。**唯一解是 reboot**(rmmod
  nvidia_uvm/gpu-reset/杀 DCGM 全试过,refcnt 是内核态的,失败)。
- **reboot 后**:/tmp 清空、snvme 模块没了。跑
  `bash /tmp/recover_gpu002.sh`(先从 master scp 过去):insmod snvme
  → mount -a → 校验 8 个挂载点。
- **基准协议**:一次跑 = 清 cache 目录 → 起容器(~18min 模型加载)→
  cold_store → sleep → hit-1..5 → repeat×2。**每 boot 第一个 hit-1
  必有 ~10s 一次性开销**(miss 自读等 io_lock,协议丢弃首 hit,但
  论文如果讲盘上路径就必须修它)。pkill 匹配自身命令行会自杀,
  kill 用显式 PID。
- **数据位置(gpu002 /tmp,reboot 即失)**:th_a*_b*(阈值扫描)、
  thsmoke、v29r16/v29r32(V29)、v28r48/big48/winchain(V28+基线)、
  off16/off32。关键数字都已抄进 findings 文档。

## 4. 文档地图

- `docs/design/v1/csa_prefetch_gpu_centric_plan.md` — 组表 ground truth、
  对齐/粒度约束、V28/V29 实现地图(其中"阶段 2.5/2.6"的叙事是驻留路线,
  按会议口径降级为 fallback)。
- `docs/design/v1/csa_prefetch_v24_v25_findings.md` — V24-V29 全部实验
  数据。**注意其中 T_MoE 拟合与 pre-retrieve 归因已被证伪,以本文为准。**
- `dsv4_csa_hca_prefetch_source_of_truth.md` / `runbook` / 
  `implementation` — 预测路径的原始设计与健康信号(已加 superseded
  notice 说明 bulk 是当时默认;现在按会议要求切回预测,notice 里的
  健康信号清单两套都列了)。
- Memory(跨 session):`project_csa_v27_relocation_win.md`、
  `project_csa_v28_hca_walker_win.md`、`project_csa_session_2026070*` —
  记的是驻留路线的战果,读时带着"这是 fallback 不是主线"的滤镜。

## 5. 接手后的第一件事(按顺序)

1. 修 MOE_TIMING 计时污染(deepseek_v4.py,把预测读 dispatch 摘出计时段)。
2. 等/收 480K 两格 + 重跑的 64K 首格,用 bulk 臂窗口 + 预测臂选择性
   重建阈值公式,**每个数字带 n/std/分位数**。
3. 跨层预测离线验证(会议任务 3,delta-select 脚本改比较层 L vs L+2/3/4)。
4. nsys 拉一次 timeline(会议任务 2),定量 indexer 算子链和 launch 气泡,
   决定 graph 化收益。

---

## 6. 接手后补充(2026-07-08)

### 6.1 MOE_TIMING 污染修复已完成

gpu002 持久补丁文件:
`/home/zbuser02/codex_sync_overlap_fix/patches/deepseek_v4.py`。

- 旧文件备份:
  `/home/zbuser02/codex_sync_overlap_fix/patches/deepseek_v4.py.bak_moe_timing_sync_20260708`
  (`sha256=01276659dcbdf32ec4bf0626a3a9dd7ba58a80c25acd27fbf18ffb351858766b`)。
- 新补丁:
  `sha256=3b2446c20eceb90827bbad80ff5a3c3683b1b5a2031cbea520542a346f246c31`。
- 修法:新增 `_sync_current_cuda_stream()`，用
  `torch.cuda.current_stream().synchronize()` 替换 `LMCACHE_MOE_TIMING`
  计时段附近所有 `torch.cuda.synchronize()`。目的不是改变 MoE 语义,
  而是避免 device-wide sync 把独立 predicted-read stream 的等待计入
  MoE 窗口。
- 已通过 `python3 -m py_compile`。容器启动后确认三处 vLLM 路径 hash
  都是新 hash:
  `/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v4.py`,
  `/usr/local/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/model.py`,
  `/usr/local/lib/python3.12/site-packages/vllm/models/deepseek_v4/amd/model.py`。

### 6.2 重启前的异常与归档

重启前环境已经不可信:

- `th_a0_b64k` 在新 timing 补丁下跑过 `extra=500`,但预测读在 layer 2
  报 `failed to issue predicted reads`，trace 落在
  `torch.cuda.stream(scatter_stream)` 的 CUDA OOM。当时 GPU 1-7 仅剩约
  605 MB free。
- 同一格进入 `extra=1000` 后 `cold_store` 300s HTTP 500,随后
  `ENGINE_DEAD`。
- 降到 `LMCACHE_ABLATION_GPU_UTIL=0.70` 后仍在 `cold_store` 崩溃,但根因
  是 `deepseek_compressor.py` 的 Triton attention/compressor OOM,不是预测读。

已在重启前归档 `/tmp` 到:
`/home/zbuser02/codex_sync_overlap_fix/reboot_archive_20260708_1535`
(约 399 MB)。其中 `th_a1_b480k` 是 timing 修复前完成的数据,可用于
load/Tutti 侧参考,不可用于干净 `T_MoE` 结论。

### 6.3 重启后状态

按用户指示已重启 gpu002。重启后确认:

- `uptime` 约 2 分钟,8 张 GPU 初始 `memory.used=0`。
- 从 `master:/tmp/recover_gpu002.sh` 复制恢复脚本到 gpu002,执行后
  `snvme` 模块和 8 个 `/mnt/nvme*` mount 全部恢复。
- 从归档恢复 `/tmp/thsweep.sh`, `/tmp/thchain.sh`, `/tmp/thsum.sh` 等脚本。
- 启动 `bash /tmp/thsweep.sh 0 64000`。容器 ready 后确认三处
  `deepseek_v4.py` 都是新 hash `3b2446...`。

干净烟测结果:

- `a0/b64k/e500` 全部请求 HTTP 200:
  `cold_store=8.741867s`, `hit-1=17.795282s`,
  `hit-2=1.106813s`, `hit-3=1.062072s`, `hit-4=1.048769s`,
  `hit-5=1.047786s`, `repeat-5a=7.327753s`,
  `repeat-5b=1.019523s`。
- `seg_e500.log`: `illegal: 0`,未发现
  `failed to issue predicted reads`、CUDA OOM、`ENGINE_DEAD`。
- `a0/b64k/e1000` 也已完整通过:
  `cold_store=2.784405s`, `hit-1=6.736567s`,
  `hit-2=1.086095s`, `hit-3=1.065047s`, `hit-4=1.064560s`,
  `hit-5=1.077189s`, `repeat-5a=5.610580s`,
  `repeat-5b=1.039928s`, `illegal: 0`。

### 6.4 重启后阈值矩阵已完整跑完

完整结果已归档到 gpu002:
`/home/zbuser02/codex_sync_overlap_fix/postreboot_threshold_20260708`。

四格都已完成并归档:

- `th_a0_b64k`: predicted arm, base 64K, extras 500/1000/2000/4000/8000。
- `th_a1_b64k`: bulk arm, base 64K, extras 500/1000/2000/4000/8000。
- `th_a0_b480k`: predicted arm, base 480K, extras 500/1000/2000/4000/8000。
- `th_a1_b480k`: bulk arm, base 480K, extras 500/1000/2000/4000/8000。

清洁性检查:

- 每格每个 extra 的 `e*.jsonl` 都有 8 条 HTTP 200 记录。
- 每个 segment log 都报告 `illegal: 0`。
- 每格归档前 fatal grep 都为空:
  `failed to issue predicted reads|CUDA error|out of memory|ENGINE_DEAD|illegal memory access|Traceback|ERROR`。
- `README.txt` 已写入同一归档目录,记录补丁 hash、四格完成状态和清洁性规则。
- 另有探索性汇总 `threshold_summary.md` 和 `threshold_summary.csv`。其中
  steady hit 暂定为 `hit-2..hit-5 + repeat-5b`。MoE delta bucket 只是
  compute-window 参考尺子,不是 LMCache 侧待办;不接 LMCache 也能单独跑。
  这个汇总只作后续阈值拟合入口,最终公式应优先围绕 LMCache 的
  predicted read/selectivity/wait 行为重算。

分析建议:

- `T_load` 继续用已验证的 `3.85 + MB/10.97` 和 scatter 0.02ms/IO 口径。
- `T_MoE` 是外部 compute window 参考,不要把它写成 LMCache 侧未完成任务。
  如果要用它做阈值公式,只需给出稳定口径和 `n/std/p50/p90`。
- 预测臂只用于选择性、miss 分布、端到端 hit 行为;不要再拿污染过的
  旧预测臂 `MOE_TIMING` 拟合窗口。

---

## 7. 会议任务终局(2026-07-09/10 补充)

三个任务全部完成,最终结论集中归档在
[csa_prefetch_v24_v25_findings.md](csa_prefetch_v24_v25_findings.md)
顶部三节("会议任务 1/2/3 最终结论")。此处只留索引和物料位置:

- **任务 1(阈值)**:`可藏 ⟺ N ≤ 66 + 10.4·Q_k(块/层)`,两边全实测
  (碎读 21µs/块 = 1.7GB/s,**不是** 11GB/s;T_MoE_p50=1.80+0.219·Q_k,
  n=1720/桶)。当前实现下预测方案有效阈值不存在;量化出三条活路:
  graph 化固定开销、跨层扩窗、碎读连续化(6.5×)。
  数据:gpu002 `postreboot_threshold_20260708/`、`/tmp/moemeas`、
  `/tmp/load_pairs.txt`(已镜像到 `vprof_archive_20260710/`)。
- **任务 3(跨层)**:官方推理代码逐行 recall。age1=92.7%(锚点吻合
  生产 93%);深层(L26-42)隔 2-4 层共享可行(86-95%),浅层不可行
  (57-81%)→ 深度分段共享。脚本 `offline_xlayer.py`(gpu002
  `codex_sync_overlap_fix/` + `offxl_20260709/`)。
- **任务 2(profiling)**:生产栈(vLLM+LMCache)torch.profiler trace
  两份:64K+2K 真稳态 hit、480K+2K cold_store 第 4 chunk(深度~23 万,
  见 findings 里的口径警告)。**480K 下 indexer 打分 mqa_logits 占
  21.9% GPU,是第二大消耗** → 跨层共享(砍打分次数)收益远大于
  graph 化(launch 项合计 ~250ms)。trace 本地
  `Desktop\vllm_traces\vllm_trace_{64k2k,480k2k}.rank0.json`,gpu002
  镜像 `vprof_archive_20260710/`。nsys 口径(cudaProfilerApi
  capture-range,同一 gate 状态机)已部署:manager 的
  `_torch_profile_hook` 加了 `LMCACHE_NSYS_CAPTURE=1` 分支
  (备份 .bak_pre_nsys),run_container.sh 挂载宿主机
  `/opt/nvidia/nsight-systems` 并透传 `LMCACHE_EXEC_PREFIX`
  (startup 脚本 exec 前缀)。采集脚本 `/tmp/vnsys.sh`。

已知开放 bug(不阻塞会议结论):BULK=0 预测臂在 480K cold_store
偶发 illegal access(写路径,`KV_OBJECT_STORE_PROFILE op=write
status=failed error=illegal memory access` → shm_broadcast cancelled
→ 引擎死)。20260709 的 480K profiling run 死于此;64K 同配置全绿。
与 bulk 臂的 illegal-access 家族(V29/batched=4096)可能同源。
