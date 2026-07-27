# CSA/Tutti streaming prefetch: current handoff

This is the canonical design and recovery note for the GPU002 CSA prefetch
implementation. Historical handoff documents and intermediate experiment notes
were removed after their verified conclusions were consolidated here.

## Frozen reference

The protected reference is the `profile80_hybrid` CP8 deployment validated on
2026-07-27:

| Item | Value |
|---|---|
| Workload | 480,000 cached prefix tokens + 8,192 recompute tokens |
| Model limit / batched tokens | 530,000 / 65,536 |
| GPU topology | TP8 / CP8 |
| Lookahead policy | `profile80_hybrid` |
| CP interleave / oversubscribe | 64 / 1 |
| Prediction block budget | 2,048 |
| L1 proxy top-K | 2,048 |
| Per-layer override | layer 28: 2,048 |
| Native indexer window | Stage0 1 layer, window 21 layers |
| Warmup hit | 1.535166612 s |
| Timed hits | 1.482527982 / 1.483438875 / 1.498633588 / 1.510532623 s |
| Median hit TTFT | **1.491036232 s** |

All five warmup/timed hits reported `LMCache hit tokens: 480000`. The run
produced 840 accuracy records and no CUDA or Tutti errors. Nsys was disabled for
these TTFT measurements.

The immutable server archive is:

```text
/home/zbuser02/protected_lmcache_versions/
  csa_profile80_hybrid_1p491_20260727
```

The protected image is:

```text
lmcache/csa-prefetch:protected-1p491-20260727
sha256:8a3d4b5f314eda35cb8f19598bb76be2785520688e4234b5b32c28b6fadd0142
```

The archive contains the exact deployed sources, scripts, result, environment,
logs, image inspect data, and an offline `docker save` image. The binary archive
is deliberately not stored in Git.

## Pipeline

Cold admission executes the complete model and materializes the online layout
once:

```text
GPU -> CPU snapshot
  -> compact main object (metadata-only when its payload is empty)
  -> CSA attention KV, one layer-major object per layer
  -> HCA attention KV, one layer-major object per layer
  -> CSA indexer K, one layer-major object per layer
  -> validate every required object
  -> publish one READY generation manifest
```

A hit consumes only a complete READY generation. It must not repack, transpose,
backfill sidecars, or rewrite the cached prefix. A missing object, insufficient
coverage, incompatible mode, or invalid extent fails closed to the normal
retrieve/miss path.

The ON request path is:

```text
prefix lookup and pin
  -> streaming preflight and target-buffer registration
  -> skip generic full-prefix retrieve when compact main is metadata-only
  -> start 8,192-token recompute
  -> stream native indexer layers (Stage0 + rolling window)
  -> target-2 residual creates the CSA proxy query
  -> CP8 proxy prediction and bounded block-ID exchange
  -> predicted Tutti indexed read and scatter
  -> target true indexer
  -> correction read for true-topK misses
  -> CUDA completion gate
  -> compact selected-KV gather and attention
```

HCA uses deterministic per-layer walking. The legacy standalone HCA prefetch and
decode-hook paths remain disabled.

## CP8 prediction

CP8 partitions query rows, not the K cache. Each rank evaluates full K for about
one eighth of the query rows, compresses its selected blocks to a bounded
`int32` ID list, exchanges those lists, and builds a local Tutti read plan from
their union. It does not use `topK/8` and does not exchange a prefix-sized
bitmap.

The final attention selection always comes from the true indexer. Prediction
only changes when likely data is fetched:

```text
correction = true_topk_union - resident_blocks - in_flight_blocks
```

Low prediction recall may increase correction latency, but must not change the
selected attention KV. Top-K-only I/O is correct only together with the vLLM
compact-gather/page-table remap and the completion gate.

## Correctness contract

- Cold and hit must use the same server process and READY generation because
  the current manifest state is process-local.
- A hit suffix must differ from the cold suffix; otherwise suffix caching can be
  mistaken for prefix recovery.
- A performance run is valid only when every timed request reports the expected
  480,000-token hit and does not repeat cold proxy warmup or cached-prefix writes.
- ON and OFF comparisons use the same source files, image, prompt, decode count,
  and timing definition; only feature switches differ.
- Multi-token decode output or logits must be checked separately from TTFT.
- Nsys is for ordering and overlap analysis. Final TTFT is measured without
  Nsys, accuracy logging, or verbose timing instrumentation unless the purpose
  of the run explicitly requires them.

## Verified lessons from removed experiments

1. The generic LMCache full-prefix retrieve must not run before the streaming
   path. It adds about 1.5--1.7 seconds and destroys the intended overlap.
2. A long NVTX `io_in_flight` range is a service lifetime, not pure device I/O.
   Pure I/O requires submit/completion markers or Tutti/NVMe kernel timing.
3. A 1.67-second reference regressed to about 34 seconds when a diagnostic
   startup script forced `LMCACHE_INDEXER_TUTTI_BACKEND=0`. Those requests were
   full recomputes, not slow hits. Runtime scripts and environment are part of
   the version, not incidental metadata.
4. The path improved from about 1.67 seconds to about 1.49 seconds through the
   compact native-indexer stream, Stage0/window submission, compact block-ID
   exchange, `profile80_hybrid`, and larger budgets on difficult layers. The
   individual contribution of each change was not isolated, so no single
   component should be credited with the full improvement.
5. Prediction value is deadline movement, not only byte reduction. Even when
   many query rows make the block union large, a read completed inside the
   target-2 compute window can reduce the correction gate.
6. Physical compact objects are not the only offset-correct representation, but
   the online path must never reinterpret a short prefix of the old concatenated
   object after zero-shaping excluded KV groups.
7. A deployment version is the combination of Git source, vLLM overrides,
   startup script, environment, `c_ops`, storage layout, and generation state.
   Matching a few Python hashes is insufficient.

## Canonical entry points

| File | Purpose |
|---|---|
| `scripts/run_container_cp8_ab.sh` | Starts one OFF or ON container and mounts the exact patch/startup inputs. |
| `scripts/startup_cp8_ab.sh` | Installs overrides, validates symbols, writes LMCache config, exports the supported environment, and starts vLLM. |
| `scripts/run_hermes_trial2_480k200.py` | Runs the fixed 480K cold/warmup/four-hit workload. |
| `scripts/restore_csa_profile80_1p491.sh` | Restores the protected image deployment from the immutable server archive. |

Key implementation ownership:

| Module | Responsibility |
|---|---|
| `lmcache/v1/cache_engine.py` | admission snapshots, streaming preflight, retrieve/fallback, pin lifetime |
| `lmcache/v1/storage_backend/local_disk_backend.py` | compact/layer-major objects, generation manifest, raw extents |
| `lmcache/v1/indexer_ssd_manager.py` | native indexer stream, proxy lifecycle, block selection |
| `lmcache/v1/csa_prefill_cp_scorer.py` | query-sharded full-K CP scoring |
| `lmcache/v1/csa_attention_kv_prefetch_manager.py` | predicted/correction reads, resident state, scatter, gate |
| `lmcache/v1/gpu_connector/tutti_direct_loader.py` | Tutti raw NVMe-to-HBM submit and poll |
| `lmcache/integration/vllm/dsv4_compact_prefill.py` | compact page table, selected-KV gather, top-K remap |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | vLLM lifecycle and manager attachment |
| `lmcache/v1/csa_prefetch_policy.py` | layer lookahead policy |

## Recovery and validation

On GPU002:

```bash
cd /home/zbuser02/protected_lmcache_versions/csa_profile80_hybrid_1p491_20260727
sha256sum -c source_result_bundle.tar.zst.sha256
sha256sum -c protected_image.tar.zst.sha256
bash scripts/restore_container.sh
```

Required devices are `/dev/snvm_control` and `/dev/ssnvme0` through
`/dev/ssnvme7`. If they are absent, use the maintained Tutti driver reset flow;
do not repeatedly insert modules after a duplicate-sysfs error.

After startup:

```bash
curl -sf http://127.0.0.1:8000/v1/models
docker inspect dsv4-csa-cp8-on --format '{{.Image}}'
docker logs dsv4-csa-cp8-on 2>&1 | grep EXPERIMENT_CONFIG | tail -n 1
```

For every new candidate, retain only:

- the source commit and deployment manifest;
- cold/warmup/timed-hit TTFT plus explicit hit evidence;
- a correctness sample with meaningful decode;
- one representative Nsys trace when it establishes a new best result or a
  durable root-cause conclusion.

Temporary patches, failed traces, raw logs, duplicate exports, and superseded
handoffs should not be kept in the repository.

## Cleanup state on 2026-07-27

The local workspace retains only two representative traces:

```text
nsys_results/off_k8_full_nsys_480000p8192_20260718_070437
nsys_results/csa_profile80_hybrid_l1topk2048_l28topk2048_first_hit_20260726_085114
```

The first is the OFF comparison; the second is the best representative ON
profile, with a 1.516956632-second post-capture hit. The 1.491-second reference
is preserved as a no-Nsys result inside the protected archive.

On GPU002, `/home/zbuser02/csa_cp8_ab_20260717` is reduced to the live mounted
patch directory plus the small set of canonical runners. Old result trees,
patch generations, build exports, and June/July diagnostic workspaces were
removed. Old LMCache cache trees were also removed from all eight experiment
NVMe devices. The following paths were explicitly retained:

```text
/mnt/nvme*/models
/mnt/nvme*/tutti_raw_reserve
/mnt/nvme*/lmcache_dsv4_cache
/mnt/nvme*/lmcache_csa
/home/zbuser02/Tutti
/home/zbuser02/protected_lmcache_versions/csa_profile80_hybrid_1p491_20260727
```

The running `dsv4-csa-cp8-on` service and the zero-GPU protection sentinel were
not stopped or replaced during cleanup.
