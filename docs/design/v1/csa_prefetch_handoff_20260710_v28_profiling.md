# 交接说明:V28 profiling 重测(2026-07-10)

> 2026-07-10 追加:本交接中的 V28 nsys 臂已补跑完成。结果已写入
> [csa_prefetch_v24_v25_findings.md](csa_prefetch_v24_v25_findings.md)
> 顶部 "V28 profiling 补充结论" 小节。远端归档:
> `~/vprof_archive_20260710/v28_480k16k.nsys-rep`;本地已有
> `Desktop\vllm_traces\v28_480k16k.nsys-rep` 和 CSV 汇总。torch trace
> 远端归档完整,本地大文件拉取曾因跳板机 SSH 超时中断,需要时可重拉。

> 写给接手的人(GPT/Codex/下一个 session)。当前工作做到一半,
> gpu002 正在第二次 reboot。本文档自包含:任务、进度、脚本位置、
> 环境铁律、参考文档全在这里。

---

## 0. 一句话现状

**用户纠偏:昨晚的全部生产栈 profiling(torch+nsys 四份)跑的是
BULK=0 预测臂——错了,要的是 V28 配置。V28 重测两臂之一(torch)
已完成且全绿,nsys 臂因 GPU 残留 12.3GB 被脚本主动 ABORT,已发起
reboot,转交时机器重启中。**

---

## 1. 当前任务:V28 profiling(torch + nsys 双口径)

### 1.1 目标配置(用户明确指定)

- **V28 臂** = `LMCACHE_CSA_BULK_PREDICTED=1` +
  `LMCACHE_CSA_WALKER_RESIDENT_SKIP=1` + `LMCACHE_DSV4_HCA_WALKER=1`
  + `LMCACHE_DSV4_VECTORIZED_CONSUME=0`(V29 在 48K 崩过,保持关)。
- 形状:**480K base + 16K 增量**(V28 决胜形状,稳态赢 OFF 的那个)。
- indexer 数字必须出自 V28 的 vLLM trace,不是预测臂、不是官方参考
  实现、不是 microbench。
- 两种工具都要:torch.profiler 和 nsys(用户原话"都要测啊")。

### 1.2 torch 臂:已完成 ✅(数字未入库、未 commit)

Run 全绿(2026-07-10 05:46-05:57 UTC,illegal=0):
cold 49.24s / hit-1 18.39s / **hit-2 5.77s** / hit-3 8.85s /
hit-4 6.42s / hit-5 6.51s / repeat-5a 6.15s / repeat-5b 4.45s。

trace(SKIP=9 → 捕获窗口 = hit-2 的稳态 forward,含 V27/V28 搬迁):
- gpu002 `/tmp/v28prof/trace_v28_480k16k.rank0.json`(191MB)
- 已镜像 `~/vprof_archive_20260710/trace_v28_480k16k.rank0.json`

**初步 kernel 表(rank0,GPU busy 总 2821ms,6312 kernels)**:

| kernel | ms | 占比 | n |
|---|---|---|---|
| nccl all_reduce | 865 | 30.7% | 87 |
| **deep_gemm mqa_logits(indexer 打分)** | **721** | **25.6%** | 336 |
| sparse_attn | 330 | 11.7% | 43 |
| k_poll_batch(Tutti 轮询) | 254 | 9.0% | 128 |
| Marlin MoE | 114.5 | 4.1% | 86 |
| dequantize_gather_k | 104 | 3.7% | 84 |
| topKPerRowPrefill | 94 | 3.3% | 336 |
| multi_layer_block_transfer(walker/搬迁 scatter) | 77 | 2.7% | 2067 |

⚠ **all_reduce 的 865ms 不可直接采信**:torch in-process profiler
8 进程互拖会把通信 kernel 虚高 2-8×(预测臂上已被 nsys 证实:
64K 435→54ms,480K 1270→617ms)。等 nsys 臂出来后以 nsys 为准。
mqa/attention/MoE/poll 在预测臂上双口径吻合(±5%),大概率可信。

### 1.3 nsys 臂:未跑 ❌(接手后第一件事)

第一次尝试被 `/tmp/v28prof.sh` 的安全检查主动 ABORT:torch 臂跑完
GPU 残留 12.3GB(UVM 泄漏,无进程,只能 reboot)。已发起 reboot。

**接手步骤(按顺序)**:

1. 等 gpu002 回来:
   `ssh master "sshpass -p 'Pass2025' ssh zbuser02@172.16.8.32 'uptime'"`
2. **reboot 清空 /tmp**——从本地 `C:\tmp\` 重推脚本:
   `postreboot.sh`、`v28nsys.sh`、`trace_sum.py`、`nsys_sum.sh`、
   `nsys_dev3.sh`(推法:先 `scp X master:/tmp/`,再在 master 上
   `sshpass -p 'Pass2025' scp /tmp/X zbuser02@172.16.8.32:/tmp/`)。
3. 恢复环境:`bash /tmp/postreboot.sh`(insmod snvme ×2 + mount -a +
   校验 8 挂载 + GPU 残留 <10GB 才放行)。
4. 启动:`nohup bash /tmp/v28nsys.sh >/dev/null 2>&1 &`
   (脚本自含:V28 env + nsys `--capture-range=cudaProfilerApi` 包
   api_server + SKIP=9 同窗口 + 480K+16K benchmark + 自动归档)。
5. 轮询:`grep -aE "V28NSYS END|LOAD_FAIL|TIMEOUT|ABORT" /tmp/v28nsys/log`
   (全程 30-40 分钟:模型加载 ~13min + cold ~1min + hits ~2min)。
6. 汇总:`nsys stats --report cuda_gpu_kern_sum /tmp/v28nsys/v28_480k16k.nsys-rep`
   (宿主机 `/opt/nvidia/nsight-systems/2025.3.2/bin/nsys`;逐设备
   分析用 python sqlite3,参考 `nsys_dev3.sh`,宿主机没有 sqlite3 CLI)。
7. 与 torch 臂逐 kernel 对照(重点 all_reduce 折扣率),更新
   `csa_prefetch_v24_v25_findings.md`,commit,拉 rep 和 trace 到
   本地 `Desktop\vllm_traces\`。

### 1.4 nsys 接入机制(已部署,不用重做)

- manager 补丁 `patches/v1/csa_attention_kv_prefetch_manager.py` 的
  `_torch_profile_hook` 里加了 `LMCACHE_NSYS_CAPTURE=1` 分支:
  同一个 skip/gate 状态机,调 `torch.cuda.profiler.start()/stop()`
  而非 torch.profiler(备份 `.bak_pre_nsys`)。
- `run_container.sh`:挂载宿主机 `/opt/nvidia/nsight-systems`(ro)
  + 透传 `LMCACHE_NSYS_CAPTURE`/`LMCACHE_EXEC_PREFIX`
  (备份 `.bak_pre_nsys`)。
- `startup_csa_prefill_tutti.sh`:`exec ${LMCACHE_EXEC_PREFIX:-} python3 -m vllm...`
  (备份 `.bak_pre_nsys`)。
- 容器镜像里没有 nsys;宿主机 nsys 2025.3.2 挂进去用。

---

## 2. 昨晚的预测臂 profiling(被判"跑错对象",留作方法论参考)

四份文件在本地 `Desktop\vllm_traces\`:
`vllm_trace_{64k2k,480k2k}.rank0.json` + `prod{64k,480k}.nsys-rep`,
gpu002 镜像 `~/vprof_archive_20260710/`。结论正文在
[csa_prefetch_v24_v25_findings.md](csa_prefetch_v24_v25_findings.md)
"会议任务 2"两节——**其中的 kernel 排名是 BULK=0 预测臂的,不代表
V28**。仍然有效的部分:

1. **torch profiler 通信虚高机制**(跨臂通用):8 worker 同开
   in-process profiler,CUPTI 开销错开步调,自旋等 peer 的通信
   kernel 把等待记进时长。all_reduce 64K 虚高 8×、480K 虚高 2×。
   **最终数字以 nsys 为准。**
2. **mqa_logits 打分是结构性重负载**(双口径吻合):O(S×T) 随深度
   线性,480K 深度 ~23 万时 1.78ms/call。
3. **官方参考实现更慢 3 倍**(同 H200 同形状 A/B 实测):官方
   model.py 的 bf16 einsum 链 5.46ms/call(物化 1.89GB logits,
   GEMM 本身只 0.64ms,其余是内存带宽),生产 fused 1.80ms。
   优化空间不在写更快 kernel,在少打分(跨层共享)。
4. 工具版本:torch 2.11.0+cu129 / python 3.12.13 / vLLM 0.20.2
   (容器),nsys 2025.3.2 + driver 580.126.20(宿主机),
   **ncu 2025.3.1 已装未用**。

---

## 3. 还没做的(优先级排序)

1. **V28 nsys 臂**(§1.3,最紧迫)。
2. V28 findings 更新 + commit(torch 臂数字也还没入库)。
3. ncu 单 kernel 分析(mqa_logits / topKPerRowPrefill 的 SM 利用率
   和内存瓶颈)——ncu 在宿主机 `/usr/local/cuda-13.0/bin/ncu`。
4. V4-Pro 的 T_MoE(会议任务 1 Pro 侧,曾 LOAD_FAIL 未解决,搁置)。
5. 开放 bug:①BULK=0 480K store 路径 illegal access(非确定性,
   干净环境未复现);②rep3+ lookup 静默失效;③UVM 泄漏机制
   (每次跑完残留 8-19GB,无进程,只能 reboot)。

---

## 4. 参考文档地图

| 文档 | 内容 |
|---|---|
| [csa_prefetch_handoff_20260708.md](csa_prefetch_handoff_20260708.md) | 总交接:会议任务、全部 flag 清单、我犯过的错、§7 三任务结果索引 |
| [csa_prefetch_v24_v25_findings.md](csa_prefetch_v24_v25_findings.md) | 核心数据:会议任务 1/2/3 最终结论 + V24-V28 全部实验数据(任务 2 节是预测臂口径,V28 待补) |
| [csa_prefetch_gpu_centric_plan.md](csa_prefetch_gpu_centric_plan.md) | 组表 ground truth(SWA43+CSA21+HCA20,无主 MLA 组)、csrc 三阶段计划 |
| 本文档 §5-6 | V28 机制摘要 + 环境操作(原在 Claude 记忆里,GPT 看不见,故落盘) |

gpu002 上的数据归档:`~/vprof_archive_20260710/`(全部 trace/rep +
moemeas + r1.jsonl)、`~/postreboot_threshold_20260708/`(任务 1
四格矩阵)、`~/offxl_20260709/`(任务 3 跨层 recall JSON)、
`~/codex_sync_overlap_fix/`(部署补丁 + benchmark 脚本
`run_incremental_v2.py` + `offline_xlayer.py`)。

---

## 5. V28 是什么(从记忆落盘,供不了解上下文的接手者)

`LMCACHE_DSV4_HCA_WALKER=1`:把 CSA 组验证过的 walker(读藏进
compute)/gate(每层等待)/V27 搬迁(HBM 内 gather→scatter 替代
NVMe 重读)扩展到 hca_attention_kv 组(5.6GB,占 CSA filter 后剩余
同步读的 84%)。

**结果矩阵(480K base,同 boot 新鲜对照,illegal=0)**:

| 增量 | OFF | V28 | repeat(V28 vs OFF) |
|---|---|---|---|
| 16K | 6.04-6.76 | **5.48-6.20 赢** | **4.04** vs 4.53 |
| 32K | 7.71-8.52 | **7.10-7.98 赢**(中位 7.5 vs 7.9) | **4.11** vs 4.73 |
| 48K | ~9.6 | 8.9-9.8 平(compute 稀释) | **4.35** vs 4.86 |

**赢的机制**:OFF 每 hit 无条件重读 12.8GB;V28 稳态 hit 利用跨请求
HBM 残留(内容没变,vLLM 分配器只是换了行)→ ~250ms GPU 内搬迁替代
11.5GB 读,同步路径只剩 indexer 组 1.35GB。稳态赢的绝对值 ≈ OFF 多读
11.5GB 的时间,与 compute 无关(48K 被稀释成 parity 的原因);repeat
三形状全赢。

**稳态 hit 解剖(r2 hit-5 = 5.586s 全实测)**:pre-retrieve 2.84s
(51%,HTTP+tokenize 496K,与 KV 系统无关)+ retrieve 2.52s(45%,
1.35GB 碎成 1874 个小读 @0.6GB/s)+ compute 0.23s(21 个 gate 全
<0.1ms)。**下一杠杆:indexer 组合并大读或进 walker,可再省 ~2s。**

三个已解的硬问题(细节在 findings):512B 对齐(payload_skip)、
cr=128 粒度(压缩条目行寻址)、非连续张量((block,slot) index_put_)。

---

## 6. 环境铁律(每条都踩过坑)

1. **跑实验前查 GPU 残留**:`nvidia-smi --query-gpu=memory.used`
   最大值 >10GB → 先 reboot。UVM 泄漏无进程可杀,nvidia-smi 看不到
   持有者。跑完一轮实验通常残留 8-19GB。
2. **reboot 后必跑 postreboot.sh**:insmod snvme-core.ko + snvme.ko
   (`~/Tutti/backends/local/kernel_modules/snvme-5.15.0-public/`)+
   `mount -a` + 校验 `mount | grep -cE "n?s?nvme.*n[0-9]+ on /mnt/nvme"`
   ≥8。**/tmp 会被清空**,脚本/归档提前 cp 到 `~`。
3. **run_container.sh 的 -e 转发**:新环境变量必须显式加进列表,
   否则容器内看不见(踩过 4 次)。
4. **8 个 vLLM worker 的 RANK 全是 0**:trace 输出文件名必须带
   `pid{os.getpid()}` 后缀,否则互相覆盖。
5. **torch.profiler 的通信 kernel 时长不可信**(§2.1),以 nsys 为准。
6. **PowerShell 三跳 SSH 引号坑**:嵌套单引号会炸;复杂命令一律写
   .sh 文件两跳 scp 推过去 bash 执行。连接链:
   `ssh master "sshpass -p 'Pass2025' ssh zbuser02@172.16.8.32 'CMD'"`。
7. **汇报纪律**(用户多次强调):每个数字带 n/std/分位;hit 时间
   ≠load 时间;不许无误差拟合;不确定就说不确定,不许把推测当测量。
