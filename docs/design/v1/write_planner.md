# Tutti 写规划器

## 目标

Tutti 的读写共享一套 NVMe SQ/CQ、HBM staging 和 I/O 锁。后台写不能与关键路径读取同时提交，但也不应等到整个缓存对象一次写完才让出队列。写规划器把一次逻辑快照拆成有界的物理 wave，并在每个 wave 前重新做 admission：

```text
256-token LMCache chunks（逻辑格式不变）
        │
        ▼
request layer-major snapshot
        │
        ▼
32/64/... MiB physical waves
        │
        ├─ demand read waiting ──► park
        ├─ tool/compute slack ───► write one wave
        └─ implicit idle gap ────► write one wave
                                  │
                                  └─ re-check before next wave
```

只有全部 physical records 写成功后，`LocalDiskBackend` 才把 records 加入 metadata store，随后上层才发布 compact object/manifest。因此 wave 之间可以等待，未完成的一代仍不可见，读取只会命中上一代或 cache miss。

## 粒度

`LMCACHE_DSV4_WRITE_QUANTUM_MB` 控制物理 wave 上限。它不修改 `chunk_size=256`，也不修改磁盘对象格式。未设置时回退到现有的 `LMCACHE_DSV4_RAW_WRITE_WAVE_MB`，所以已有服务配置保持原行为。

一次已经提交到 NVMe 的 wave 不能被中途取消；quantum 因而也是新 demand read 的最大写阻塞粒度。当前 layer-major 打包以一个物理 layer sidecar 为最小单元，所以有效 quantum 不会小于该请求中最大的单 layer sidecar。

## Slack 来源

### 无外部 KV 读取的计算

vLLM worker 在 `start_load_kv` 检查整个 prefill batch。只有当所有请求都不需要从 LMCache 读取额外 KV 时，才从 `start_load_kv` 到 `wait_for_save` 打开 `compute_no_kv` 窗口。窗口关闭时会重新启动 idle guard，避免尾部写紧贴下一次读取。

### 工具调用

长工具调用天然形成 inter-request idle gap，现有 50 ms idle 检测可自动利用。知道工具生命周期的 agent runtime 也可调用显式接口，使写无需先等待 idle guard：

```python
token = engine.begin_tutti_write_slack("tool_call", expected_duration_s=2.0)
try:
    run_tool()
finally:
    if token is not None:
        engine.end_tutti_write_slack(token)
```

有限窗口会用写带宽 EWMA 估计 wave 是否能在 deadline 前完成；预计放不下时不会启动该 wave。
窗口到期后无需依赖新的写请求触发清理：admission 和状态查询都会淘汰过期 token。`POST /write_slack/end` 是幂等操作；token 已到期时返回 `removed=false`，但仍返回 HTTP 200，方便 frontend 在下一轮请求前安全收尾。

生产 serving 由 OpenAI/agent frontend 读取流式响应。只有解析到原生 `tool_calls`、`function_call` 或服务采用的 `<tool_call>` 协议后才开窗；请求输入中仅仅存在 `tools` 字段不能作为开窗依据。LMCache worker 暴露：

- `POST /write_slack/begin`
- `POST /write_slack/end`
- `GET /write_slack/status`

TP8 默认 worker 端口为 7000–7007。当前 vLLM OpenAI frontend 会在流式或非流式响应中确认原生 `tool_calls`；对 AgentBench 使用的 XML 协议，也会识别 `<tool_call>`。请求应携带稳定的 session header，下一轮同 session 请求到达时会在调度前关闭窗口：

```text
x-lmcache-agent-session-id: <stable-agent-session-id>
x-lmcache-tool-slack-seconds: <expected-tool-duration>
```

不携带 session header 时，原生 OpenAI `tool_call_id` 仍可完成关联；既没有 ID 又没有 session header 时，为避免无法关闭的全局窗口，只使用隐式 idle slack。frontend 也可直接使用 fan-out 客户端：

```python
from lmcache.v1.write_slack_client import TuttiWriteSlackFanoutClient

write_slack = TuttiWriteSlackFanoutClient.from_worker_ports(worker_count=8)

# 流式响应已经确认模型调用工具。
handle = write_slack.begin_tool_call(expected_duration_s=tool_timeout_s)
try:
    tool_result = run_tool()
finally:
    # 必须在发送包含 tool_result 的下一轮请求前关闭。
    write_slack.end(handle)
```

fan-out begin 在任一 worker 失败时会回滚其他 worker 已打开的窗口。工具窗口是服务全局信号；为了避免 agent A 的工具执行影响 agent B，vLLM adapter 会把需要外部 KV 的 forward 标为 `read_sensitive`。该状态具有硬优先级，在 forward 结束前禁止所有后台写，即使另一个 session 的工具窗口仍然打开。

## 优先级与观测

优先级固定为：demand read > read-sensitive forward > speculative read（写超时前）> 已知 slack 写 > implicit idle 写。最大等待时间只允许写越过 speculative reader，永远不能越过 demand read 或 read-sensitive forward。

`LMCacheEngine.get_tutti_write_plan_snapshot()` 返回 active slack、排队/执行 wave 数、排队字节、成功/失败计数和估计写带宽，便于 serving 实验记录控制面与数据面行为。

2026-08-03 的 TP8 SNVMe 验证保持逻辑 chunk 为 256 tokens、物理 wave 为 256 MiB：

- 480K+8192 三次精确命中为 1.344、1.358、1.364 秒，全部命中 480000 tokens。
- AgentBench 0.4 req/s 的 22 个续轮请求全部精确命中；hit TTFT p50/p95 为 1.257/1.472 秒，失败数为 0。
- 每个 rank 完成 77 个写 wave，失败数为 0；在线 EWMA 为 4.12–4.40 GiB/s。
- 同一 480K 快照从 cold response 完成到 TP8 全部 terminal record 发布的尾部为 1.325 秒；旧的逐 prefix alias SSD 路径为 11.224 秒。该 8.47 倍是本项目旧路径对比，不能标成上游官方 LMCache 的通用写性能。

相关配置：

- `LMCACHE_DSV4_WRITE_QUANTUM_MB`：可选的物理写 wave 上限。未设置时保留快速写版本已有的 `LMCACHE_DSV4_RAW_WRITE_WAVE_MB`；不要在无 A/B 数据时改变生产默认值。
- `LMCACHE_TUTTI_WRITE_SLACK_SEC`：隐式 idle guard，默认 0.05 秒。
- `LMCACHE_TUTTI_WRITE_MAX_DELAY_SEC`：后台写最大等待，默认 2 秒。
- `LMCACHE_TUTTI_WRITE_INITIAL_MIBPS`：带宽 EWMA 初值，默认 4096 MiB/s。
- `LMCACHE_TUTTI_WRITE_DEADLINE_GUARD_SEC`：有限 slack deadline 安全边界，默认 0.01 秒。
