# DSv4 CSA Attention-KV Prefetch Runbook

This is the short operational runbook for GPU002. Keep it current and avoid
adding raw experiment transcripts here.

## Machine

| Item | Value |
|---|---|
| host | `gpu002` |
| address | `172.16.8.32` |
| model on host | `/mnt/nvme0/models/DeepSeek-V4-Pro` |
| model in container | `/pro_model` |
| workspace | `/home/zbuser02/codex_sync_overlap_fix` |
| endpoint | `http://127.0.0.1:8000` |

Do not use `/home/zbuser02/models/deepseek-v4-pro` for DSv4-Pro. The current
model source is NVMe0.

## Model Mount Order

Before `docker run`, confirm the host path exists:

```bash
ssh gpu002 'test -f /mnt/nvme0/models/DeepSeek-V4-Pro/config.json'
```

Then bind it into the container:

```bash
-v /mnt/nvme0/models/DeepSeek-V4-Pro:/pro_model:ro
```

The vLLM startup command should use:

```text
--model /pro_model
```

Do not start the container first and then try to discover the model inside it.
vLLM validates `--model` during startup and fails immediately if the path is
wrong.

`/tmp` is not durable on GPU002. Keep scripts, patches, logs, and summaries in
`/home/zbuser02/codex_sync_overlap_fix` or another persistent directory.

## Container Rules

Use all 8 H200 GPUs for these runs, so stop/remove old DSv4 containers first.
Check before starting:

```bash
ssh gpu002 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"'
ssh gpu002 'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv'
```

If a startup log says:

```text
LMCacheConnectorV1 does not support HMA
```

do not switch to `--disable-hybrid-kv-cache-manager` as the first response. The
old working path used HMA; this error usually means the vLLM-side connector
patch did not land.

## Tutti Device Requirements

The CSA-on path must not silently fall back to the file backend. Before starting
`dsv4-csa-prefill-on`, the host must expose the Tutti/SNVM control device:

```bash
ssh gpu002 'ls -l /dev/snvm_control'
```

If it is missing, fix the host Tutti/SNVM setup first. Do not start the CSA-on
container and hope the loader appears later; the run is invalid if logs contain:

```text
LMCACHE_INDEXER_TUTTI_BACKEND is set but no Tutti loader is available; falling back to the file backend
TuttiDirectLoader init failed ... No such file or directory: '/dev/snvm_control'
```

`/dev/ssnvme<N>` does not need to exist before `docker run`: `TuttiDirectLoader`
opens `/dev/snvm_control`, issues `SNVM_CHRDEV_CREATE`, and creates the matching
`/dev/ssnvme<N>` node inside the privileged container. If `/dev/ssnvme*` already
exists on the host, `run_container.sh` passes those devices through as well.

`run_container.sh on` now fails fast when `/dev/snvm_control` is missing. The
docker run must include the Tutti device/capability parameters:

```text
--device /dev/snvm_control:/dev/snvm_control
--pid host
--cap-add SYS_ADMIN
--cap-add SYS_RAWIO
--privileged
-v /sys:/sys
-v /tmp:/tmp
```

`--privileged` alone is not the contract for this experiment; the device mapping
must be visible in the startup command so the run is auditable.

Host recovery after reboot:

```bash
ssh gpu002 'cd /home/zbuser02/Tutti/backends/local/kernel_modules/snvme-5.15.0-public && sudo make insmod'
ssh gpu002 'lsmod | grep -E "^(snvme|snvme_core)\s" && ls -l /dev/snvm_control'
```

Do not use destructive `--bind` smoke tests just to create `/dev/ssnvme*`; that
can detach the mounted NVMe filesystems used for `/pro_model` and LMCache disk
paths. The loader creates the per-controller char device itself.

## Docker Root Must Not Live on a Tutti/snvme Drive

**Symptom.** `cold_store` (and any real request) returns HTTP 500 shortly after
the server reaches `/v1/models` ready, and the container becomes unusable:

```text
500 EngineCore encountered an issue
dockerd ... Input/output error   (docker logs itself fails to read overlay2)
EXT4-fs error (device snvmeXnY): Detected aborted journal
block snvmeXnY: no available path - failing I/O
```

**Root cause.** The 8 KIOXIA cache drives are managed by the `snvme` kernel
driver. When Tutti starts GPU-direct I/O for a rank, `snvme` **rebinds the
driver on that device** (dmesg: `(snvm_rebind_driver): disable device for bind
new driver`), which pulls the block path out from under the still-mounted ext4
filesystem. Anything doing **continuous ext4 writes** on that drive dies with
EIO. Docker's `data-root` (overlay2) is a continuous writer, so if it sits on a
Tutti drive (`/mnt/nvme0/docker`), the whole container crashes the moment Tutti
binds `nvme0`.

Not everything on a Tutti drive conflicts — only continuous ext4 writers:

| Data on the Tutti drive | Survives Tutti rebind? |
|---|---|
| **docker `data-root`** | **No** — continuous writer, dies with EIO |
| Pro model weights (read once at load, before bind) | Yes — loaded into GPU before Tutti binds |
| LMCache `local_disk` KV cache | Yes — written before bind, read raw after (by design) |

**Fix.** Put docker `data-root` on a **non-Tutti** drive. On GPU002 the spare
`nvme0n1` (960 GB Micron, BDF `0000:41:00.0`, no co-located GPU so it is *not* a
Tutti drive) is used for exactly this:

```bash
# one-time: format the spare Micron and mount it
sudo mkfs.ext4 -F -L dockerdisk /dev/nvme0n1
# fstab: UUID=<...>  /mnt/dockerdisk  ext4  defaults,noatime,nofail  0  2
sudo mount /mnt/dockerdisk

# copy the image bundle off the KIOXIA (preserve overlay2 hardlinks/xattrs)
sudo systemctl stop docker
sudo rsync -aHAX --numeric-ids /mnt/nvme0/docker/ /mnt/dockerdisk/docker/

# point docker at the non-Tutti drive
# /etc/docker/daemon.json:  "data-root": "/mnt/dockerdisk/docker"
sudo systemctl start docker
docker images | grep vllm-openai   # confirm the image survived the move
```

Verify: `docker inspect <ctr> --format '{{.GraphDriver.Data.MergedDir}}'` must
point under `/mnt/dockerdisk`, and after a full run
`sudo touch /mnt/nvme0/.iotest` must still succeed (no EIO).

Do **not** try to make the spare Micron a Tutti drive to avoid the move: it has
no GPU on its PCIe root complex (`pci0000:40`), so GPU-direct P2P to it crosses
root complexes and is slow. Each of the 8 KIOXIA shares a root complex with
exactly one GPU (that is the fast-path pairing); the Micron cannot substitute.

## Host Resource Hazards (GPU UVM leak, root-disk shadow leak)

**GPU memory leak (~19-20 GiB/GPU, no process).** After Tutti/GPU-direct runs,
every H200 shows ~19-20 GiB used with an **empty** `nvidia-smi
--query-compute-apps` list. These are orphaned `nvidia_uvm` kernel contexts
(refcount stuck, no owning process). `gpu-reset` fails ("in use by another
client"); clean `rmmod nvidia_uvm` fails (refcount != 0). **Only a reboot clears
it.** Consequence: it caps usable `--gpu-memory-utilization`. With the leak,
`util=0.85` can fail vLLM's startup precheck
(`Free memory ... less than desired gpu_memory_utilization`); drop to `0.82`.

**Prevention (measured 2026-07-03).** The leak is created by hard-killing vLLM
workers. A graceful `docker stop -t 60` (SIGTERM + grace) released the workers'
~120 GiB/GPU correctly (`nvidia_uvm` refcount 84 -> 12), while every prior
`pkill -9` / `docker rm -f` of live workers left orphaned UVM contexts that
accumulate until reboot. Rules:

- To stop a serving container: `docker stop -t 60 <ctr>`, then `docker rm`.
- Never `pkill -9` vLLM workers or `docker rm -f` a running container except as
  a last resort after a crash — and expect leaked GPU memory when you do.
- After a reboot: verify `nvidia-smi` shows ~0 used, re-run the snvme insmod
  (see "Host recovery after reboot"), and confirm the 9 fstab mounts and docker
  `data-root` on the non-Tutti drive all came back.

**Root-disk fills to 100%.** `run_container.sh` mounts only `/mnt/nvme0`; the
other cache drives may be unmounted when Tutti writes to their mountpoints, so
`tutti_raw_reserve` + `lmcache_dsv4_cache` land on the **root fs** shadowed under
`/mnt/nvmeN`, filling `/`. Reclaim while unmounted:

```bash
for n in 2 3 4 5 6 8 9; do mountpoint -q /mnt/nvme$n || sudo rm -rf /mnt/nvme$n/*; done
sudo mount -a   # remount the real drives per fstab
```

## Startup Memory Config (skip profiling to avoid fragmentation OOM)

With the GPU UVM leak eating headroom, vLLM's automatic memory profiling can OOM
on a transient allocation even though steady-state fits. Set an explicit KV-cache
byte budget so profiling is skipped:

```bash
LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES=11811160064   # 11 GiB; must be >= min-KV for max_model_len
LMCACHE_ABLATION_GPU_UTIL=0.82
LMCACHE_ABLATION_MAX_BATCHED_TOKENS=1024             # small activation peak
LMCACHE_ABLATION_KV_OBJECT_STORE_SLOT_MB=32          # staging 4x32=128 MiB, fits physical headroom
```

The 11 GiB value must exceed the min-KV vLLM computes for `max_model_len`
(9.06 GiB at 16000). Weights are ~103 GiB/rank; `103 + 11 + ~3 overhead` fits in
the `~120 GiB` left after the leak.

## Indexer SSD Dir Sharding

`LMCACHE_INDEXER_SSD_DIR` accepts a **comma-separated list**, sharded across
drives `by_gpu` (same policy as `local_disk`), so each rank's indexer lands on
its own drive instead of all ranks sharing one (single point of failure + I/O
contention). The startup default lists all 8 drives; rank `r` uses `paths[r % N]`
via `PathSharder`. Do not revert it to a single path.

## Required Patch Bundle

The runtime image may not contain all local changes after `docker rm`. The
durable patch bundle under the GPU002 workspace must include at least:

```text
patches/c_ops.cpython-312-x86_64-linux-gnu.so
patches/v1/cache_engine.py
patches/v1/indexer_ssd_manager.py
patches/v1/csa_attention_kv_prefetch_manager.py
patches/v1/indexer_tutti_backend.py
patches/v1/gpu_connector/tutti_direct_loader.py
patches/v1/gpu_connector/gpu_connectors.py
patches/v1/gpu_connector/utils.py
patches/v1/storage_backend/local_disk_backend.py
patches/v1/storage_backend/storage_manager.py
patches/integration/vllm/vllm_v1_adapter.py
patches/integration/vllm/lmcache_connector_v1.py
```

Common missing-file signatures:

```text
No module named 'lmcache.v1.gpu_connector.tutti_direct_loader'
missing attempt_permute_to_contiguous_view
lmcache.c_ops.tutti_submit_batch_sgl_read not found
No module named 'lmcache.v1.kv_object_store'
StorageManager.batched_put does not support on_complete_callback
```

Fix the patch bundle before benchmarking.

## 32K Token Run

To test with a 32K prefill sequence instead of 8K, increase batched-token budget and
GPU memory utilization before calling `run_container.sh`:

```bash
LMCACHE_ABLATION_MAX_BATCHED_TOKENS=32768 \
LMCACHE_ABLATION_GPU_UTIL=0.90 \
LMCACHE_LOG_MOE_TIMING=1 \
./run_container.sh on
```

`MAX_BATCHED_TOKENS=32768` lets the prefill run as a single chunk so the MoE window
contains all 32768 tokens. `GPU_UTIL=0.90` is required to accommodate the larger KV
cache. `LMCACHE_LOG_MOE_TIMING=1` enables per-layer MoE timing (see below).

## MoE and EP Timing

When `LMCACHE_LOG_MOE_TIMING=1` is set (passed via `run_container.sh`), each MoE
forward emits two log lines per layer:

```text
LMCACHE_MOE_TIMING layer=N tokens=T gate_ms=G ep_expert_ms=E shared_ms=S total_ms=X
LMCACHE_MOE_KERNEL layer=N tokens=T stage_ms=A kernel_ms=B
```

Field meanings:

| Field | What it covers |
|---|---|
| `gate_ms` | Router gate linear + `fused_topk_bias` (local, no comm) |
| `ep_expert_ms` | **EP all-to-all dispatch + expert compute + EP all-to-all combine** |
| `shared_ms` | Shared expert MLP (runs concurrently with routed compute on HW) |
| `total_ms` | `gate + ep_expert + shared` |
| `stage_ms` | Input staging for mega_moe kernel (`_stage_deepseek_v4_mega_moe_inputs`) |
| `kernel_ms` | `fp8_fp4_mega_moe` = EP dispatch + expert GEMM + EP combine |

EP communication and expert compute are fused inside `deep_gemm.fp8_fp4_mega_moe` and
cannot be separated from Python. To isolate EP communication cost, compare `kernel_ms`
with EP=8 against a reference run with EP=1 (TP=8, all experts local).

The CSA prefetch overlap target: `TUTTI_PROFILE load_total total_ms` for target layer
`m+1` must be less than `LMCACHE_MOE_TIMING total_ms` for layer `m`. If Tutti
finishes inside the MoE window, the predicted reads are fully hidden.

## Benchmark Cases

Only two cases are needed for the current report:

```text
off: LMCACHE_INDEXER_ENABLE_PREFETCH=0
on:  LMCACHE_INDEXER_ENABLE_PREFETCH=1
```

Run `off` once as baseline. Then run `on`. Between rounds, unload/reset the
cache data as required by the script and sleep 10 seconds.

Do not drop Linux page cache for this comparison. Page-cache dropping belongs to
separate cold-NVMe bandwidth experiments and changes the meaning of hit-N
timing.

For the 8192-token smoke path, use:

```text
LMCACHE_ABLATION_MAX_BATCHED_TOKENS=1024
LMCACHE_ABLATION_GPU_UTIL=0.75
```

There is no separate vLLM activation-memory cap. `--gpu-memory-utilization`
covers weights, activation/workspace, and KV cache budget for the instance.

## Expected Request Flow

Start the server, then wait for:

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

Run the benchmark from the GPU002 workspace. Keep stdout JSONL, summary JSON,
and docker logs in persistent paths.

## What To Check

Do not judge the run only by:

```text
lmcache_hit_tokens_tail
retrieve_count
```

Those are ordinary LMCache retrieve-side fields. CSA attention-KV prefetch must
be checked through CSA/Tutti logs.

> **2026-07-08 update.** The signal lists below describe the LEGACY
> per-layer predicted-read mode (`LMCACHE_CSA_BULK_PREDICTED=0`).  In the
> default bulk-walker mode (V24+, `LMCACHE_CSA_BULK_PREDICTED=1`),
> `dispatch_csa_attention_kv_predicted` and `_submit_reads label=predicted`
> are intentionally absent — `fire_predicted_reads` no-ops.  Healthy bulk
> signals instead:
>
> ```text
> CSAAttentionKVPrefetchManager: bulk read-ahead req=... chunks=... layers=21 total_ms=...
> CSAAttentionKVPrefetchManager: layer-major walk layers=[N] ... landed=...
> CSAAttentionKVPrefetchManager: resident-chunk skip req=... matched=M/M  (V27, repeat hits)
> CSAAttentionKVPrefetchManager: correction ... miss_blocks=0
> TUTTI_PROFILE load_total keys=... loaded=...
> ```
>
> Problem signals in bulk mode: `miss_blocks>0` at gates, `illegal memory
> access`, `bulk read-ahead ... aborted (request changed)` on every request,
> walker `total_ms` in the tens of seconds (io_lock starvation), and
> `resident-chunk skip ... matched=0` on a repeat request (signature loss).

Required successful signals (legacy predicted mode only):

```text
LMCACHE_TTFT_STAGE event=csa_overlap_hook_fire source=hc_post
IndexerSSDTiming: event=dispatch_csa_attention_kv_predicted
IndexerSSDTiming: event=prefill_fire_async ... mode=csa_attention_kv_microbatch_finish
CSAAttentionKVPrefetchManager: _submit_reads label=predicted
TUTTI_PROFILE load_total keys=... loaded=...
CSAAttentionKVPrefetchManager: correction ... miss_blocks=0
```

Problem signals:

```text
dispatch_csa_attention_kv_predicted = 0
_submit_reads label=predicted = 0
_submit_reads label=miss is high
correction miss_blocks>0 and total_ms is large
failed to issue miss reads
Tutti extents cover 0/... bytes
Tutti direct load found no readable KV extents
CUDA illegal memory access during proxy microbatch
```

The 2026-06-30 layer-24 failure is the pattern to remember: Tutti itself showed
fast reads (`load_total keys=31 loaded=31 total_ms~=1.3 ms`), but CSA correction
still had `miss_blocks=31` on some ranks, followed by `failed to issue miss
reads` and CUDA illegal memory access. That is not an LMCache retrieve miss; it
is a CSA prefetch/correction failure.

If logs show `residual_proxy_skip ... reason=indexer_op_read_only_unsupported`,
the runtime SparseAttnIndexer op cannot be used for the high-recall proxy path
yet. Do not bypass that guard by calling the normal op: normal calls can write
indexer K-cache rows or trigger the patched CSA correction wrapper. Fix the
vLLM op/patch bundle so the op exposes a skip-insert attribute or kwarg.

## Correct Method Reminder

The `on` case must use exactly one target-layer prediction at a time:

```text
layer m attention HC_post
  -> run target layer m+1 HC_pre + attn_norm + indexer projection
  -> topK over all valid token rows
  -> unique compressed blocks
  -> fire predicted CSA attention-KV Tutti reads for layer m+1
  -> overlap with layer m MoE/FFN
```

Never submit all 30 CSA layers at once. Never use previous CSA layer true topK
as next-layer prediction. Never use tail-row-only proxy for the high-recall
scheme.

## Useful Commands

Inspect current logs:

```bash
ssh gpu002 'ls -lt /home/zbuser02/codex_sync_overlap_fix | head -30'
ssh gpu002 'grep -E "TUTTI_PROFILE|CSAAttentionKV|dispatch_csa_attention|csa_overlap_hook|illegal memory|failed to issue" /home/zbuser02/codex_sync_overlap_fix/*.log | tail -200'
```

Copy a log package locally when needed:

```bash
scp gpu002:/home/zbuser02/codex_sync_overlap_fix/<log-or-tgz> ./tmp_gpu002_logs/
```
