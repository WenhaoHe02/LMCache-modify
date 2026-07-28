# CSA/Tutti profile80 1.491s colleague handoff

This document hands off the frozen `profile80_hybrid` CP8 implementation that
was validated on GPU002 on 2026-07-27. It describes the immutable image, exact
runtime configuration, recovery procedure, source revision, and every vLLM
file replaced or modified by the deployment.

## Frozen release identity

| Item | Frozen value |
|---|---|
| Git revision | `1049cefd` (`de7fa55c` contains the performance-preserving implementation; `1049cefd` only removes obsolete artifacts) |
| Git tag | `csa-profile80-hybrid-1p491-20260727` |
| Branch | `codex/csa-profile80-1p491-clean-20260727` |
| Base image | `lmcache/vllm-openai:indexer-ssd-hca-prefetch-decodegate-20260528_0630` |
| Protected image | `lmcache/csa-prefetch:protected-1p491-20260727` |
| Protected image ID | `sha256:8a3d4b5f314eda35cb8f19598bb76be2785520688e4234b5b32c28b6fadd0142` |
| Offline image archive | `protected_image.tar.zst`, 8,709,571,130 bytes |
| Image archive SHA256 | `568b9c455c0a1af87938f7c5a890547075bb1e74bb01071142dc798eb7f0936e` |
| Source/result archive SHA256 | `9d90e501d0dd439dbca81c10f099c10690d3492f52c754b2cf3e60990dd65624` |

The model weights and Hermes dataset are not included in either archive.

## Validated result

The frozen run used a 480,000-token cached prefix and a distinct 8,192-token
recompute suffix. Nsys and verbose timing were disabled for the TTFT result.

| Sample | TTFT |
|---|---:|
| Warmup hit | 1.535166612 s |
| Hit 1 | 1.482527982 s |
| Hit 2 | 1.483438875 s |
| Hit 3 | 1.498633588 s |
| Hit 4 | 1.510532623 s |
| Median | **1.491036232 s** |

Every warmup/timed hit reported `LMCache hit tokens: 480000`; the run produced
840 prediction-accuracy records and no CUDA or Tutti error.

## Requirements

- Eight GPUs visible to Docker; the validated deployment is TP8/EP8/CP8.
- CUDA 12.9-compatible NVIDIA driver.
- Docker with NVIDIA Container Toolkit.
- DeepSeek-V4-Flash at `/mnt/nvme0/models/DeepSeek-V4-flash`.
- Mounted data devices `/mnt/nvme0`, `nvme2`, `nvme3`, `nvme4`, `nvme5`,
  `nvme6`, `nvme8`, and `nvme9`.
- Tutti devices `/dev/snvm_control` and `/dev/ssnvme0` through
  `/dev/ssnvme7`.
- `zstd`, `sha256sum`, `curl`, and Python 3.

The Linux kernel does not need to be rebuilt before each run. The Tutti kernel
modules must already match the running kernel. If the devices are absent and
the machine is otherwise idle, restore them with:

```bash
cd /home/zbuser02/Tutti
sudo bash scripts/reset_snvme.sh
```

Do not repeatedly run `insmod` after a `duplicate sysfs filename` error; reboot
the host and run the reset script once. `reset_snvme.sh --force-cleanup` kills
every process holding `/dev/snvm*` and is only safe on an exclusively allocated
machine.

## Restore the image and start the service

The complete frozen release on GPU002 is:

```text
/home/zbuser02/protected_lmcache_versions/
  csa_profile80_hybrid_1p491_20260727/
```

Validate both archives before use:

```bash
cd /home/zbuser02/protected_lmcache_versions/csa_profile80_hybrid_1p491_20260727
sha256sum -c protected_image.tar.zst.sha256
sha256sum -c source_result_bundle.tar.zst.sha256
```

Load the image only when the tag is missing:

```bash
docker image inspect lmcache/csa-prefetch:protected-1p491-20260727 \
  >/dev/null 2>&1 || \
  zstd -dc protected_image.tar.zst | docker load
```

Confirm that no unrelated job owns the eight GPUs, then start the exact ON
configuration:

```bash
bash scripts/restore_container.sh
```

`restore_container.sh` checks the model and Tutti devices, forces the protected
image tag into the runner, mounts the frozen patches read-only, and exports the
validated environment. It replaces only the container named
`dsv4-csa-cp8-on`.

Check readiness and deployment identity:

```bash
curl -sf http://127.0.0.1:8000/v1/models
docker inspect dsv4-csa-cp8-on --format '{{.Image}}'
docker logs dsv4-csa-cp8-on 2>&1 | grep EXPERIMENT_CONFIG | tail -n 1
```

Expected image ID:

```text
sha256:8a3d4b5f314eda35cb8f19598bb76be2785520688e4234b5b32c28b6fadd0142
```

## Run the validated workload

Cold admission and hits must run against the same server process because the
frozen generation manifest is process-local. Do not restart the container
between the cold request and timed hits.

```bash
cd /home/zbuser02/protected_lmcache_versions/csa_profile80_hybrid_1p491_20260727

BASE_TOKENS=480000 \
RECOMPUTE_TOKENS=8192 \
NUM_WARMUP_HITS=1 \
NUM_HITS=4 \
WARMUP_WAIT_S=10 \
HIT_WAIT_S=5 \
python3 scripts/run_hermes_trial2_480k200.py 20 \
  | tee colleague_480k8192.jsonl
```

The timed suffix must differ from the cold/warmup suffix. A valid result has
HTTP 200, 488,192 prompt tokens, and `LMCache hit tokens: 480000` for every
timed request. TTFT comparisons are made without Nsys.

For a meaningful-output check, request at least 128 decode tokens and compare
the output text or token IDs against an OFF reference using the same prompt,
sampling parameters, image, and source files. One-token TTFT runs do not prove
decode correctness.

Useful log commands:

```bash
docker logs -f dsv4-csa-cp8-on
docker logs dsv4-csa-cp8-on 2>&1 | grep -E \
  'LMCache hit tokens|EXPERIMENT_CONFIG|accuracy|CUDA error|Tutti'
```

Stop the release with:

```bash
docker rm -f dsv4-csa-cp8-on
```

## Exact performance configuration

The important non-default values are:

| Setting | Value | Meaning |
|---|---:|---|
| `LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER` | `profile80_hybrid` | Frozen per-layer target/lookahead policy. |
| `LMCACHE_CSA_PREFETCH_CP_SIZE` | `8` | Eight-rank query-sharded proxy scoring. |
| `LMCACHE_CSA_PREFETCH_CP_INTERLEAVE` | `64` | Query-row interleave granularity. |
| `LMCACHE_CSA_PREFETCH_CP_OVERSUBSCRIBE` | `1` | No extra oversubscription. |
| `LMCACHE_CSA_PREFETCH_CP_EXCHANGE_IDS` | `1` | Exchange compact block-ID lists, not prefix bitmaps. |
| `LMCACHE_CSA_PREFETCH_BLOCK_BUDGET` | `2048` | Global predicted-block budget. |
| `LMCACHE_CSA_L1_PROXY_TOPK_TOKENS` | `2048` | L1 proxy top-K. |
| `LMCACHE_CSA_PROXY_TOPK_TOKENS_BY_LAYER` | `28:2048` | Frozen difficult-layer override. |
| `LMCACHE_NATIVE_INDEXER_STAGE0_LAYERS` | `1` | One indexer layer submitted at Stage0. |
| `LMCACHE_NATIVE_INDEXER_WINDOW_LAYERS` | `21` | Rolling native-indexer I/O window. |
| `LMCACHE_DSV4_HCA_WALKER` | `1` | Deterministic layerwise HCA restore. |
| `LMCACHE_INDEXER_PROFILE_ACCURACY` | `1` | Prediction accuracy validation enabled in the frozen run. |
| `LMCACHE_CSA_PREDICTION_GATE_TIMEOUT_SEC` | `5` | Fail-closed wait limit before CSA consumption. |
| `LMCACHE_ABLATION_MAX_MODEL_LEN` | `530000` | vLLM model length. |
| `LMCACHE_ABLATION_MAX_BATCHED_TOKENS` | `65536` | Chunked-prefill scheduling limit. |
| `LMCACHE_ABLATION_GPU_UTIL` | `0.55` | GPU-memory reservation used by the validated service. |

Profiling switches are deliberately zero in the performance run:
`LMCACHE_NSYS_CAPTURE`, `LMCACHE_NSYS_FULL_CAPTURE`,
`LMCACHE_TUTTI_PROFILE`, and `LMCACHE_TTFT_STAGE_PROFILE`.

The vLLM command is equivalent to:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /pro_model \
  --served-model-name deepseek-v4-pro \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --all2all-backend allgather_reducescatter \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --max-model-len 530000 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 65536 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.55 \
  --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --no-disable-hybrid-kv-cache-manager \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --port 8000
```

## vLLM changes

The base image uses a vLLM 0.20.2-era DeepSeek-V4 implementation. The protected
image contains a committed site-packages overlay, and `startup_cp8_ab.sh`
reapplies the same files from `/patches` so the runtime does not depend on a
mutable container layer.

### `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`

The startup script applies a small compatibility edit in place:

- imports `SupportsHMA`;
- changes `LMCacheConnectorV1` to inherit `SupportsHMA`;
- adds `request_finished_all_groups()` and forwards the primary HMA KV-group
  block IDs to the existing `request_finished()` implementation.

This exposes the hybrid-KV-cache-manager contract required by the DSv4 cache
groups. It does not implement the CSA prediction policy itself.

### `vllm/model_executor/models/deepseek_v4.py`

The file is replaced for both ON and OFF runs, and is also copied to
`vllm/models/deepseek_v4/nvidia/model.py` (plus the unused AMD location for
layout compatibility). Relevant additions are:

- a weak registry of live `DeepseekV4DecoderLayer` instances so the LMCache
  adapter can attach managers after model construction without scanning
  arbitrary modules;
- `attach_indexer_prefetch()` and `attach_hca_prefetch()` public hooks;
- an HCA completion drain immediately before the target attention consumes KV;
- CSA proxy/indexer submission at the FFN-entry window using the residual and
  positions from the target lookahead source;
- preparation of the next CSA and HCA layer while the current FFN/MoE executes;
- a shared top-K index buffer and the DeepSeek-V4 runtime fixes required by the
  validated model/kernel combination.

The decoder hook moves I/O submission earlier; it does not replace the true
indexer result. The attention gate still decides whether predicted data is
ready and issues correction reads for true-topK misses.

### `vllm/model_executor/layers/deepseek_v4_attention.py`

This override is installed only when CSA ON is selected. It adds the correctness
boundary needed for sparse top-K recovery:

- recognizes only compressed CSA K caches owned by the active LMCache manager;
- obtains the exact selected-page bitmap produced by true-indexer correction;
- builds a compact page table for those pages and remaps original top-K entry
  IDs into the packed workspace;
- gathers/dequantizes only selected CSA pages before sparse attention;
- keeps the SWA region and offsets intact;
- bypasses compact planning when correction proves the whole cached prefix is
  selected;
- leaves cold/recompute and non-owned caches on the normal full-cache path.

Without this file, loading only predicted top-K pages violates vLLM's ordinary
assumption that every registered prefix KV page is valid. ON results obtained
without the compact gather/remap are not a valid implementation of this release.

### `lmcache/integration/vllm/dsv4_compact_prefill.py`

This is an LMCache file but forms the other half of the vLLM override. It
validates query partitions, constructs compact block tables and sequence
lengths, remaps top-K indices, and calls the native CUDA planner when available.
It provides a checked PyTorch fallback for compatible layouts.

### Files deliberately not changed

The protected source bundle does not contain `sparse_attn_indexer.py`, so the
conditional startup copy for that filename is inactive. vLLM's scheduler and
attention API were not redesigned; integration relies on the explicit HMA
connector compatibility hook, decoder attachment hooks, per-layer load gates,
and the DSv4 compact-gather override above.

## LMCache and Tutti ownership

| File or area | Release responsibility |
|---|---|
| `lmcache/integration/vllm/vllm_v1_adapter.py` | Captures HMA metadata, attaches managers and decoder hooks, disables generic full-prefix retrieval for a valid streaming hit, and enforces layer gates. |
| `lmcache/v1/cache_engine.py` | Creates final admission snapshots, performs streaming preflight, pins request state, and fails closed to normal retrieval/recompute. |
| `lmcache/v1/indexer_ssd_manager.py` | Streams native indexer layers and owns residual-proxy scheduling. |
| `lmcache/v1/csa_prefill_cp_scorer.py` | Partitions query rows across CP8 and scores full K. |
| `lmcache/v1/csa_attention_kv_prefetch_manager.py` | Submits predicted reads, computes true-topK correction, tracks resident/in-flight pages, scatters data, and publishes CUDA completion. |
| `lmcache/v1/hca_prefetch_manager.py` | Deterministic layerwise HCA walking. |
| `lmcache/v1/storage_backend/local_disk_backend.py` | Compact/layer-major object layout and generation metadata. |
| `lmcache/v1/gpu_connector/tutti_direct_loader.py` | Tutti raw NVMe-to-HBM indexed reads, polling, and scatter. |
| `lmcache/c_ops.cpython-312-x86_64-linux-gnu.so` | Frozen CUDA extension containing the indexed I/O and compact-prefill primitives used by this image. |

## Correctness and operational limits

- The frozen READY-generation registry is process-local. A restart loses the
  in-memory manifest even if the raw objects remain on disk.
- Never mix sidecars from different generations or reuse an old cache directory
  after changing object layout.
- Hit-path admission must not rewrite the cached 480K prefix.
- Prediction affects timing only. Final attention selection comes from the true
  indexer plus a correction read.
- `io_in_flight` NVTX lifetime is not pure NVMe service time.
- The image contains a compiled CPython 3.12/x86-64/CUDA extension. Rebuilding
  Python, PyTorch, CUDA, or the base image requires rebuilding `c_ops`.
- The 1.491-second number is steady-state after one warmup. Report cold,
  first-hit/warmup, and timed-hit values separately.

## Package layout

The colleague delivery consists of two files plus this document:

```text
protected_image.tar.zst                 # complete offline Docker image
csa_profile80_1p491_handoff_20260728.tar.gz
  runtime/patches/                      # exact deployed source overrides
  runtime/scripts/                      # exact restore/start/workload scripts
  source/lmcache-...-1049cefd.tar.gz    # clean Git source archive
  evidence/                             # frozen result and environment metadata
  IMAGE_MANIFEST.txt
  README.md
  SHA256SUMS
```

The Docker archive is intentionally not committed to GitHub. GitHub contains
the clean source revision, tag, scripts, and this runbook.

GPU002 also has a preassembled, checksum-verified view at:

```text
/home/zbuser02/colleague_handoff/csa_profile80_1p491_20260728
```

Its `protected_image.tar.zst` is a link to the immutable 8.71GB archive, while
the handoff tarball and extracted runtime are ordinary files.

On a different machine, assemble the portable release as follows:

```bash
tar -xzf csa_profile80_1p491_handoff_20260728.tar.gz
mv protected_image.tar.zst \
  csa_profile80_1p491_handoff_20260728/runtime/
cd csa_profile80_1p491_handoff_20260728/runtime
sha256sum -c protected_image.tar.zst.sha256
bash scripts/restore_container.sh
```

The portable checksum file uses a relative filename. The original checksum
inside the immutable GPU002 archive records the absolute server path and is
retained only as provenance.
