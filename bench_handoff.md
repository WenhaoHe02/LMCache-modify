# Bench Handoff: GLP O_DIRECT tp=8 Benchmark

## 目标

测量 LMCache group-level layer prefetch（GLP）在 O_DIRECT=True、tp=8 下的效果：
- **gs=1**（基线，逐层串行读）vs **gs=4**（并发预取 4 层）的 retrieve 延迟对比
- 跑 32K / 64K / 128K / 256K / 512K 五个长度
- 预期：O_DIRECT 下 gs=4 应该比 gs=1 快（因为每个 chunk syscall 开销暴露，并发读摊销）

---

## 机器状态（gpu002）

- 8× H100/H200（143 GiB each），全部空闲
- sglang-decode **已 kill**（进程 23356/23357/23358/23359 已 kill -9，Docker 容器也 rm -f）
- GPU 0-7 均 ~4 MiB 占用（全空）

---

## 已有文件（gpu002）

| 路径 | 说明 |
|------|------|
| `/root/qwen14b1m_tp8_odirect.sh` | tp=8 O_DIRECT 容器启动脚本 |
| `/tmp/bench_compare_odirect_tp8.py` | 两阶段 bench 脚本 |
| `/mnt/nvme0/lmcache_ssd_odirect.yaml` | LMCache O_DIRECT 配置 |
| `/tmp/bench_odirect_tp8.log` | 最近一次运行日志 |

启动脚本关键参数：
- `--gpus all`，`--tensor-parallel-size 8`，`--gpu-memory-utilization 0.90`
- `-e VLLM_RPC_TIMEOUT=300000`（5 分钟）
- `-e LMCACHE_USE_EXPERIMENTAL=True`
- `use_odirect: True`，`use_layerwise: true`，`layer_group_size` 由 sed 动态修改
- 容器名 `qwen14b1m-tp8`，模型 `qwen14b1m`（Qwen2.5-14B-Instruct-1M）

---

## 当前问题

**bench 脚本每次跑到 32K gs=1 retrieve 时，vLLM worker 挂死恰好 300s 后 RPC timeout：**

```
TimeoutError: RPC call to sample_tokens timed out.
```

LMCache 日志显示 retrieve 请求进来时：
```
Total tokens 32501, Inference Engine computed tokens: 0,
LMCache hit tokens: 0, need to load: 0
```

"need to load: 0" 意味着 LMCache 没有调度 SSD 读。但 worker 仍然卡死，
然后 vLLM EngineCore 在 5 分钟后把 worker 杀掉。

**最可能的原因**：bench 脚本在 host 侧调用了 `drop_page_cache()`：
```bash
sync && echo 3 > /proc/sys/vm/drop_caches
```
O_DIRECT 本来就绕过 page cache，这个命令对 O_DIRECT 无意义，但它同时会清除 dentry/inode cache，
可能破坏了容器内 LMCache 的某些文件句柄或内存映射状态，导致 SSD 读路径 hang。

---

## 修复步骤

### Step 1：去掉 drop_page_cache（最可能解决问题）

修改 `/tmp/bench_compare_odirect_tp8.py` 的 `measure_retrieve` 函数：

```python
def measure_retrieve(prompt, tag, n=N_RETRIEVE):
    times = []
    for i in range(n):
        # drop_page_cache()   # O_DIRECT 不需要，去掉
        t, _ = do_request(prompt, f"retrieve-{i+1}")
        times.append(t)
        if i < n - 1:
            flush_gpu()
    avg = sum(times) / len(times)
    print(f"    [{tag} avg={avg:.3f}s]", flush=True)
    return times, avg
```

（`drop_page_cache` 函数本身可以保留，只是不调用它）

### Step 2：重新启动 bench

```bash
# 在 gpu002 上执行

# 清理旧容器和 IPC
docker rm -f qwen14b1m-tp8 2>/dev/null || true
for s in $(ipcs -s | awk '/^0x/ {print $2}'); do ipcrm -s $s 2>/dev/null || true; done
for m in $(ipcs -m | awk '/^0x/ {print $2}'); do ipcrm -m $m 2>/dev/null || true; done

# 清 SSD 缓存
for d in /mnt/nvme0 /mnt/nvme2 /mnt/nvme3 /mnt/nvme4 /mnt/nvme5 /mnt/nvme6 /mnt/nvme8 /mnt/nvme9; do
  find $d/lmcache/qwen14b1m -mindepth 1 -delete 2>/dev/null || true
done

# 跑（容器启动需要约 13-15 分钟 torch.compile + profiling，等 wait_ready 900s 超时够用）
nohup python3 /tmp/bench_compare_odirect_tp8.py > /tmp/bench_odirect_tp8.log 2>&1 &
echo "PID: $!"

# 监控
tail -f /tmp/bench_odirect_tp8.log
```

---

## bench 脚本逻辑（两阶段）

```
Phase 1 (gs=1):
  对每个长度 [32K, 64K, 128K, 256K, 512K]：
    1. populate（首次请求，触发 GPU 计算 + 写入 SSD）
    2. 等待 50s 让 SSD 写完
    3. flush_gpu()（发 "Hello." 清 GPU KV cache）
    4. retrieve × 3（从 SSD 读，取平均）

Phase 2 (gs=4, ONE restart):
  bash /root/qwen14b1m_tp8_odirect.sh 4   # 改 layer_group_size=4，重启容器
  对每个长度：
    retrieve × 3（用 Phase 1 写好的 SSD 数据）

输出：Length | Tokens | Populate | gs=1 ret | gs=4 ret | Speedup
```

---

## 参考：buffered IO 基准结果（已测完，供对比）

```
Length   Tokens   Populate   gs=1 ret   gs=4 ret   Speedup
32K       32501     1.662      1.348      0.715      1.88x
64K       80001     7.094      3.095      3.929      0.79x
128K     150001    15.913      5.343      7.623      0.70x
256K     280001    37.730      9.570     18.295      0.52x
512K     494002    87.052     18.186     35.288      0.52x
```

buffered IO 下 gs=4 在 64K+ 反而更慢（OS readahead 使 gs=1 已经很快），
O_DIRECT 预期 gs=4 应该更快。

---

## 如果 Step 1 不解决问题（备用排查）

检查 LMCache GLP 分支的 O_DIRECT asyncio 读取逻辑：
```
/root/lmcache-glp/lmcache/v1/storage_backend/
```
重点看 disk backend 里的 O_DIRECT read path，以及 asyncio event loop 在
retrieve 时是否有死锁（例如等待一个永远不会 set 的 asyncio.Event）。

VLLM_RPC_TIMEOUT 可以临时调大（比如 600000 = 10min）来排查是"纯粹太慢"还是"真死锁"：
```bash
# 在 /root/qwen14b1m_tp8_odirect.sh 中修改：
-e VLLM_RPC_TIMEOUT=600000
```

---

## 参考：CPU retrieve 基准结果

| Length | Tokens | Prefill | CPU_ret | Speedup |
|--------|--------|---------|---------|---------|
| 8K | 7,996 | 0.133s | 0.040s | 3.3x |
| 16K | 15,991 | 0.154s | 0.056s | 2.8x |
| 32K | 31,994 | 0.330s | 0.092s | 3.6x |
| 48K | 47,997 | 0.414s | 0.126s | 3.3x |
| 64K | 64,000 | 0.570s | 0.225s | 2.5x |
| 80K | 79,990 | 0.609s | 0.218s | 2.8x |
| 96K | 95,993 | 0.702s | 0.259s | 2.7x |
| 112K | 111,996 | 0.810s | 0.303s | 2.7x |
| 128K | 127,999 | 0.933s | 0.421s | 2.2x |
| 144K | 143,989 | 1.055s | 0.437s | 2.4x |
| 160K | 159,992 | 1.151s | 0.491s | 2.3x |
| 192K | 191,998 | 2.119s | 0.576s | 3.7x |
| 224K | 223,991 | 2.444s | 0.699s | 3.5x |
| 256K | 255,997 | 2.792s | 0.805s | 3.5x |
| 320K | 319,996 | 5.656s | 1.040s | 5.4x |
| 384K | 383,995 | 6.734s | 1.247s | 5.4x |
| 448K | 447,994 | 7.850s | 1.496s | 5.2x |

## 参考：CPU_ret vs SSD_prev vs Recompute 对比

```
  Approx   Recompute   SSD_prev   CPU_curr   vsCPU    vsSSD
  -----------------------------------------------------------
  16K        0.341      0.129      0.131      2.6x    2.6x
  32K        0.756      0.223      0.240      3.2x    3.4x
  48K        1.209      0.308      0.343      3.5x    3.9x
  64K        1.675      0.390      0.243*     6.9x*   4.3x
  80K        2.540      0.495      0.760      3.3x    5.1x
  96K        3.387      0.610      0.693      4.9x    5.6x
  112K       4.158      0.696      0.855      4.9x    6.0x
  128K       5.093      0.794      0.932      5.5x    6.4x
  144K       6.032      0.888      1.076      5.6x    6.8x
  160K       7.332      0.997      1.362      5.4x    7.4x
  201K      10.854      1.267      1.633      6.6x    8.6x
  224K      13.442      1.424      1.925      7.0x    9.4x
  256K      16.465      1.629      2.145      7.7x   10.1x
  320K      25.684      2.056      2.710      9.5x   12.5x
  366K      31.128      2.300      3.269      9.5x   13.5x
```
