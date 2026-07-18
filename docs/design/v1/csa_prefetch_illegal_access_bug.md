# Bug: CSA predicted prefetch crashes/hangs on point D (base=2000, extra=10000)

Status: **root-caused 2026-07-03 — final fix (util=0.85) verification in progress**

## Symptom evolution (same underlying bug, different disguises)

| Run | Symptom |
|---|---|
| pre-reboot (leak present) | `CUDA error: out of memory` in `_submit_reads` layer 6 |
| post-reboot util=0.90 | `CUDA error: an illegal memory access` in `_submit_reads` layer 6 → worker death → NCCL cascade |
| after round-1/2 fixes | hit-1 hangs 300 s → `TimeoutError: RPC call to sample_tokens timed out` → EngineDead |
| A-point variant | all-8-rank 14.8 s freeze mid-hit-1, then self-recovery (proxy wall-clock 14 845 ms vs `proxy_gpu_ms` 15 ms) |

## True root cause (live py-spy capture, t=+130 s into the hang)

```
rank0   : OOM'd out of forward, back in worker_busy_loop waiting for RPC
          torch.nn.functional.pad -> torch.OutOfMemoryError: 64 MiB wanted, 54 MiB free
          (this process has 130.69 GiB in use; PyTorch allocated 124.95 GiB)
rank1-7 : stuck in torch.cuda.synchronize() after layer-7/8 fused MoE
          -> their EP all-to-all is waiting for rank0, forever
```

**Point D (extra=10000) has the largest single-prefill activation of the sweep
grid, and ON mode adds the prefetch pool + proxy buffers on top. At
`gpu_memory_utilization=0.90` there is no headroom left; a routine 64 MiB
`F.pad` allocation on rank0 fails, rank0 aborts its forward, and the other
seven ranks deadlock in collectives.** A/B/C (extra <= 8000) fit, D does not —
a pure memory-watermark problem, not a logic bug in the prefetch path.

The earlier OOM/illegal-access flavors were the same watermark breach
surfacing at different allocation sites (higher pressure = earlier failure,
async reporting = misleading stack lines).

## Real defects found and fixed along the way (all deployed, all still valid)

1. **`_issue_reads` returned bare `None`** when all candidate ids fell outside
   the chunk map, violating its `(event, objs)` contract
   ([csa_attention_kv_prefetch_manager.py](../../../lmcache/v1/csa_attention_kv_prefetch_manager.py)).
   → `return None, []`.
2. **Predicted path lacked the covered-range clamp** the miss path has
   (`_miss_ids_for_topk` drops uncovered ids; `_submit_reads` only clamped to
   bitmap size). Short prefixes (D: chunks cover blocks 0..6, prediction emits
   0..10) flooded per-id "not covered by any chunk" warnings and booked
   uncovered ids into `pending_reads`. → clamp to
   `state.chunks[-1].end_compressed_block` before any bookkeeping.
3. **No bounds check on the staging slice** in `_issue_reads`; a short Tutti
   read would silently truncate the Python slice and `.view()/copy_` a
   wrong-shaped region. → validate `byte_offset + blk_bytes <= flat.numel()`.
4. **`TuttiDirectLoader` had no synchronisation at all** (zero locks) while
   CSA prefetch calls `load_chunks_to_hbm` from proxy executor threads
   concurrently with main-thread retrieve loads and raw stores. One SQ/CQ
   ring, one doorbell, one staging pool → concurrent submit/poll interleaves
   ring updates and staging reuse
   ([tutti_direct_loader.py](../../../lmcache/v1/gpu_connector/tutti_direct_loader.py)).
   → `_io_lock` serialising the whole submit/poll/consume cycle for loads and
   raw stores. This matches the NVMe single-queue contract; per-thread queue
   pairs are the future optimisation if lock contention ever shows up
   (prefetch reads are 2-8 ms, so it does not today).
5. **The spinning `tutti_poll_batch` kernel ran on the legacy default stream**
   (`stream_ptr=0`), hard-serialising NVMe polling with model-forward and
   NCCL kernels — the opposite of the overlap design intent, and the
   amplifier that turned one slow completion into an all-rank freeze.
   → dedicated per-loader `torch.cuda.Stream`; submit+poll launch there and
   only that stream is synchronised.

## Final configuration fix

`LMCACHE_ABLATION_GPU_UTIL=0.90 → 0.85` for ON-mode runs (~7 GiB headroom for
peak activation + prefetch buffers at extra=10000). OFF mode was never
affected (no prefetch pool, smaller peak).

## Verification checklist

- [ ] Point D (2000/10000): cold + hit-1..5 + repeat all 200, engine alive.
- [ ] Points A/B/C: no regression vs the util=0.90 run.
- [ ] No `illegal memory`, `failed to issue`, `poll timed out`,
      `blocked for more than` (dmesg) during the sweep.
- [ ] Overlap telemetry intact: `correction ... miss_blocks=0`,
      predicted `load_total` ~2-8 ms.

## Operational note

A worker that dies inside a CUDA collective leaves the container
unstoppable (`container ... PID is zombie and can not be killed`) — another
reason the runbook mandates graceful `docker stop` and treats hard kills as
leak sources. If it happens, the zombie holds no GPU memory once the other
ranks are stopped; a reboot clears the container record.
