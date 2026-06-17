# Claude Handoff: DSv4 CSA SSD Prefetch Rewrite

## Current conclusion

Do not switch this deployment to LMCache MP mode for the CSA SSD prefetch work.
The target setup is the normal vLLM + `LMCacheConnectorV1` path because it uses
the existing multi-NVMe SSD-only config.

Also do not reintroduce the previous 61-layer KV filtering patch. DeepSeek V4
registers all HCA/CSA-related KV tensors, and LMCache must see the normal
`list(self.kv_caches.values())` list so its DSv4 heterogeneous KV group logic can
discover the real layouts.

The earlier 32K failures were caused by runtime/config drift, not by official
DSv4 being unable to run:

- The container was running an older installed LMCache package/CUDA extension
  that did not contain the local DSv4 heterogeneous KV compatibility work.
- The normal V1 connector was later started without `use_gpu_connector_v3: true`,
  so pointer registration still tried to fit 243 vLLM KV tensors into 61 legacy
  pointer slots.

## Required base before testing

The runtime must include the local DSv4 heterogeneous KV fixes from commit
`cdee0078 feat(dsv4): add DeepSeek V4 compatibility for heterogeneous KV cache
groups`.

The important pieces are:

- `lmcache/v1/gpu_connector/utils.py`
  - accepts dim-0-padded KV tensors in `attempt_permute_to_contiguous_view`
  - populates `PageBufferShapeDesc.block_stride_elems`
- `lmcache/v1/kv_layer_groups.py`
  - groups by `(kv_size, num_heads, head_size, block_size, dtype)`
  - derives compression metadata for HCA/CSA physical block sizes
- `lmcache/v1/gpu_connector/gpu_connectors.py`
  - uses the updated group metadata on the normal connector path
- `csrc/mp_mem_kernels.*`, `csrc/pybind.cpp`, and
  `lmcache/non_cuda_equivalents.py`
  - expose/use `block_stride_elems`

Before running long prompts, verify inside the container that the installed
package and compiled extension are not stale:

```bash
/opt/venv/bin/python - <<'PY'
import lmcache
import lmcache.c_ops as lmc_ops
print(lmcache.__version__, lmcache.__file__)
desc = lmc_ops.PageBufferShapeDesc()
print("block_stride_elems:", hasattr(desc, "block_stride_elems"))
PY
```

If `block_stride_elems` is false/missing, reinstall or rebuild LMCache from the
current source before testing. Copying only Python adapter files is insufficient.

The current container has `/opt/venv/bin/python` but not a working `pip`
executable, so the working rebuild path was:

```bash
cd /tmp/lmcache_src
export MAX_JOBS=${MAX_JOBS:-8}
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.4.5.dev64
/opt/venv/bin/python setup.py build_ext --inplace
/opt/venv/bin/python setup.py build_py

SITE=/opt/venv/lib/python3.12/site-packages
cp -a "$SITE/lmcache" "/tmp/backup_before_codex/lmcache.$(date +%s)"
cp -a /tmp/lmcache_src/build/lib.linux-x86_64-cpython-312/lmcache "$SITE/"
```

After this rebuild, the container verified as:

```text
lmcache 0.4.5.dev64 /opt/venv/lib/python3.12/site-packages/lmcache/__init__.py
block_stride_elems True
c_ops /opt/venv/lib/python3.12/site-packages/lmcache/c_ops.cpython-312-x86_64-linux-gnu.so
```

## Runtime config

Keep the SSD-only normal-mode LMCache config:

```bash
LMCACHE_CONFIG_FILE=/etc/lmcache/lmcache_ssd_only.yaml
LMCACHE_INDEXER_ENABLE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=1
LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS=0
LMCACHE_INDEXER_SSD_DIR=/mnt/nvme0/csa_ssd_pool/indexer
LMCACHE_INDEXER_POOL_SIZE=4096
LMCACHE_INDEXER_IO_WORKERS=8
LMCACHE_INDEXER_MAX_SEQ_LEN=131072
```

`LMCACHE_INDEXER_ENABLE_PREFETCH` is the feature switch. Do not make
`LMCACHE_INDEXER_SSD_DIR` alone enable prefetch.

The SSD-only config previously used is:

```yaml
chunk_size: 256
local_cpu: false
local_disk: "/lmcache/nvme0/,/lmcache/nvme2/,/lmcache/nvme3/,/lmcache/nvme4/,/lmcache/nvme5/,/lmcache/nvme6/,/lmcache/nvme8/,/lmcache/nvme9/"
local_disk_path_sharding: "by_gpu"
max_local_disk_size: 4096.0
extra_config: {'use_odirect': True}
use_gpu_connector_v3: true
```

`use_gpu_connector_v3: true` is mandatory for this normal V1 DSv4 deployment.
Without it, the first long request can fail during `post_init` with:

```text
ValueError: could not broadcast input array from shape (243,) into shape (61,)
```

That error means the legacy connector pointer array is being used. Do not "fix"
it by filtering KV caches down to 61 entries; that hides HCA/CSA tensors from
LMCache and breaks official DSv4 layout discovery.

The bind-mounted host config currently used by the container is:

```text
/mnt/nvme0/lmcache_ssd_only.yaml -> /etc/lmcache/lmcache_ssd_only.yaml
```

The container environment currently used for prefetch testing is:

```bash
LMCACHE_CONFIG_FILE=/etc/lmcache/lmcache_ssd_only.yaml
LMCACHE_INDEXER_ENABLE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=1
LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS=0
LMCACHE_INDEXER_SSD_DIR=/mnt/nvme0/csa_ssd_pool/indexer
LMCACHE_INDEXER_POOL_SIZE=4096
LMCACHE_INDEXER_IO_WORKERS=8
LMCACHE_INDEXER_MAX_SEQ_LEN=131072
```

## Remote/container access

The DSv4 test container is reached through the two-hop SSH path:

```powershell
ssh -o StrictHostKeyChecking=no master "sshpass -p 'Pass2025' ssh -o StrictHostKeyChecking=no zbuser02@172.16.8.32 'COMMAND'"
```

For multi-line remote commands, use base64 to avoid quoting problems:

```powershell
$script = @'
set -euo pipefail
sudo docker ps
'@
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($script))
ssh -o StrictHostKeyChecking=no master "sshpass -p 'Pass2025' ssh -o StrictHostKeyChecking=no zbuser02@172.16.8.32 'echo $b64 | base64 -d | bash -s'"
```

Container details:

```text
name: dsv4-indexer-ssd
image: lmcache/vllm-openai:indexer-ssd
endpoint: http://127.0.0.1:8000
model id: /mnt/nvme0/models/DeepSeek-V4-Pro
entrypoint: /opt/venv/bin/vllm serve
```

The current serve command is:

```text
/mnt/nvme0/models/DeepSeek-V4-Pro
--tensor-parallel-size 8
--enforce-eager
--kv-cache-dtype fp8
--max-model-len 32768
--kv-transfer-config {"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}
--port 8000
--trust-remote-code
```

## Code changed in this pass

### Normal LMCache V3 DSv4 path

These fixes keep official vLLM + LMCache normal mode working for DeepSeek V4
HCA/CSA heterogeneous KV tensors. They are required even when prefetch is off.

- `F:\LMCache\lmcache\integration\vllm\vllm_service_factory.py`
  - passes `inference_engine_logical_block_size` from
    `self.vllm_config.cache_config.block_size` into `vllm_layout_hints()`
  - this lets LMCache derive compressed group physical chunk sizes for DSv4
- `F:\LMCache\lmcache\v1\metadata.py`
  - `get_shapes(num_tokens)` now honors each group `compress_ratio`
  - grouped allocations use `num_tokens // compress_ratio` where needed
- `F:\LMCache\lmcache\v1\gpu_connector\gpu_connectors.py`
  - `VLLMPagedMemGPUConnectorV3` registers the real 243 vLLM KV tensors
  - builds a `KVLayerGroupsManager` before post-init allocation
  - transfers each heterogeneous group with its own shape descriptor and
    `physical_chunk_size`
  - supports non-contiguous physical block IDs while requiring contiguous
    offsets inside each logical block
- `F:\LMCache\lmcache\utils.py`
  - `DiskCacheMetadata` records optional grouped `shapes` and `dtypes`
- `F:\LMCache\lmcache\v1\storage_backend\local_disk_backend.py`
  - stores/loads raw heterogeneous bytes without forcing the first group shape
  - persists grouped shape/dtype metadata for later retrieval

The important runtime group discovery log for this setup is seven groups, not
61 flat layers:

```text
group0: layers=61 bs=256 hs=584 uint8 ratio=1
group1: layers=31 bs=2 hs=584 uint8 ratio=128
group2: layers=31 bs=256 hs=1024 float32 ratio=1
group3: layers=30 bs=64 hs=132 uint8 ratio=4
group4: layers=30 bs=256 hs=512 float32 ratio=1
group5: layers=30 bs=64 hs=584 uint8 ratio=4
group6: layers=30 bs=256 hs=2048 float32 ratio=1
```

### LMCache adapter prefetch attach

File: `F:\LMCache\lmcache\integration\vllm\vllm_v1_adapter.py`

Added environment-gated CSA/indexer prefetch attachment after
`self._manager.post_init()` in `register_kv_caches()`.

The adapter now:

- leaves LMCache store/retrieve on official `list(self.kv_caches.values())`
- discovers vLLM DeepSeek V4/V2 decoder registries if present
- creates `IndexerSSDManager` only when `LMCACHE_INDEXER_ENABLE_PREFETCH` is true
- wires each CSA `SparseAttnIndexer` with `ssd_manager` and `csa_layer_id`
- calls decoder-layer `attach_indexer_prefetch(manager, next_csa_layer_id)` when
  available

It intentionally does not:

- filter HCA/CSA KV caches
- change `start_load_kv`, `save_kv_layer`, or `wait_for_save`
- switch to MP connector

### Indexer SSD manager

File: `F:\LMCache\lmcache\v1\indexer_ssd_manager.py`

This is the SSD-backed HBM pool manager. It is not official upstream LMCache;
deploy it with the adapter changes.

Final important behavior:

- `prepare_pool()` always calls `ready_fut.result()` when a post-prefill future
  exists, even if the future is already done; this surfaces background failures
- `_pread()` uses Python's `os.pread(fd, size, offset)` API correctly
- `_drain()` waits for submitted reads instead of treating normal long reads as
  failures with a 5 second timeout
- the HBM pool uses the real vLLM IndexerCache packed layout:
  `[num_blocks, 64, token_bytes]`; each block stores all value bytes first and
  the token scale bytes at the end
- SSD files store one interleaved `value+scale` record per logical compressed
  indexer token; reads/writes convert explicitly between SSD layout and packed
  HBM layout
- `evict_after_prefill()` batch-gathers packed IndexerCache rows and writes one
  contiguous SSD byte range instead of issuing one `pwrite` per token
- prefill initialization uses the vLLM compressed indexer `block_table` and
  expands it into logical token-to-physical slot mapping
- pool state is reset before loading a new prefill state
- `torch.frombuffer(bytearray(data), dtype=torch.uint8)` is used to avoid the
  non-writable bytes warning

Decode residual proxy:

- enabled only when `LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=1`
- uses the DSv4 attention HC-post state, not V2-style `hidden_states + residual`
- diagnostic short run after the packed-layout fix showed
  `residual_proxy_accuracy ... recall=0.8076`, close to the documented
  `attn+residual` proxy result

Prefill/chunked prefill note:

- vLLM may submit several prefill chunks for one long prompt.
- The safe default is `LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS=0`, so the
  residual-proxy indexer is **not** called from prefill/chunked-prefill rows.
- Directly slicing the tail prefill row and calling the proxy indexer reused
  vLLM's full prefill metadata with a single-row proxy tensor and caused CUDA
  illegal access. Keep this path disabled unless a separate proxy metadata path
  is implemented.
- Initial prefill cannot issue useful SSD reads before the target layer's SSD
  store has been initialized. `fire_async_for_layer()` therefore skips reads
  while the layer decode cursor is `<= 0`.
- Correct future design for chunked prefill: behavior must match non-chunked
  prefill. Accumulate logical compressed token IDs across chunks, only prefetch
  token IDs below the target layer's initialized cursor, and use metadata that
  matches the proxy tensor shape. Do not read from an empty SSD store and do not
  let chunk-local positions replace global compressed token IDs.

### vLLM runtime patches

The runtime container uses the older vLLM layout, not the local new-layout
`F:\vllm_dev` tree. Patch these actual files in site-packages:

```text
/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v4.py
/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py
```

`deepseek_v4.py` must have:

- `_DEEPSEEK_V4_DECODER_LAYER_REGISTRY`
- registration in `DeepseekV4DecoderLayer.__init__`
- `attach_indexer_prefetch(manager, next_csa_layer_id)`
- `fire_async_for_layer(next_csa_layer_id)` immediately before FFN execution

`sparse_attn_indexer.py` must have:

- decode-only routing into `_forward_cuda_pool`
- 4D `q_values` for `fp8_fp4_paged_mqa_logits`
- real DeepGEMM `schedule_metadata`, never `None`
- pool exposed as 64-token pages, not block size 1
- invalid pool slots masked before top-k
- pool-slot top-k translated back to global compressed indexer token IDs
- prefill eviction seeded from the last prefill query row
- prefill eviction submitted with the compressed block table

## Runtime status as of 2026-05-26

Completed:

- LMCache was rebuilt in the container from current local source plus the
  adapter/indexer overlays.
- `PageBufferShapeDesc.block_stride_elems` exists in the runtime extension.
- The LMCache adapter in site-packages uses `list(self.kv_caches.values())`.
- The container vLLM old-layout `deepseek_v4.py` has
  `attach_indexer_prefetch(...)` and the pre-FFN fire hook.
- The container `sparse_attn_indexer.py` has the DeepGEMM pool-path fixes.
- `/mnt/nvme0/lmcache_ssd_only.yaml` has `use_gpu_connector_v3: true`.
- After restart, each worker logs:

```text
IndexerSSDManager: enabled CSA prefetch on 61 decoder layers and attached 30 CSA indexers
```

Final validation:

```text
timestamp: 2026-05-26T10:33:19+00:00
request: prompt = "LMCache DeepSeek V4 thirty two thousand token validation paragraph. " * 2144
max_tokens: 600
ignore_eos: true
HTTP: 200
elapsed: 95.49s
usage: prompt_tokens=32161, completion_tokens=600, total_tokens=32761
finish_reason: length
```

Fresh log counts from that run:

```text
prepare_pool_any=240
prepare_pool_4096=240
record_topk=240
fire_active=240
prefill_submit=960
prefill_complete=240
bad_errors=0
```

`bad_errors=0` checked for:

```text
async read failed|fallback sync read|Traceback|RuntimeError|ValueError|CRITICAL|ERROR|UserWarning
```

The final run proves:

- normal vLLM + LMCache SSD-only DSv4 still runs a near-32K prompt
- output length is above the requested 512-token threshold
- prefetch is not just attached; pool prepare/top-k/fire all executed
- the post-restart runtime loaded the latest manager without warning/error logs

## Prefetch accuracy and performance snapshot

The old 2026-05-26 diagnostic that reported `pool_vs_full_recall ~= 0.158` is
obsolete. It was caused by a pool layout bug: the code treated vLLM's
IndexerCache as per-token interleaved bytes, but the real cache packs each
64-token block as all value bytes followed by all scale bytes. A raw
`kv_cache[block, offset]` byte comparison was misleading because it made the
same wrong layout assumption.

Current diagnostic controls:

```text
LMCACHE_INDEXER_PROFILE_ACCURACY=1
/tmp/lmcache_indexer_profile_accuracy exists inside the container
```

Current diagnostic meanings:

- `profile_full_flat`: official paged full-cache top-k compared with an
  independent flat-gather full-cache top-k; this should be 1.0.
- `profile_accuracy`: SSD/HBM pool top-k compared with official full-cache
  top-k.
- `profile_bytes`: compares the packed full-cache row with the packed pool slot
  after reconstructing interleaved `value+scale` bytes.
- `residual_proxy_accuracy`: compares DSv4 attention-HC-post proxy top-k with
  the full-cache top-k.

Verified after the packed-layout fix:

```text
short profile:
HTTP_OK elapsed=8.48 prompt=6002 completion=2 total=6004
profile_accuracy=240
profile_full_flat=240
errors=0
pool_recall_avg=1.000000 min=1.000000 max=1.000000
coverage_avg=1.000000
gather_k_diff_avg=0.000
gather_scale_diff_avg=0.000

short profile with decode residual proxy after the 2026-05-27 redeploy:
HTTP_OK elapsed=18.28 prompt=12001 completion=2 total=12003 finish=length
profile_accuracy=240
profile_full_flat=240
residual_proxy=240
residual_proxy_accuracy=240
prefill_disabled_skips=480
errors=0
pool_recall_avg=1.000000
residual_recall_avg=0.929883 min=0.7305 max=0.9941

long validation after batch prefill eviction:
HTTP_OK elapsed=326.21 prompt=31501 completion=512 total=32013 finish=length
prefill_disabled_skips=240
waiting_prefill=8
timeout=0
errors=0

long-context accuracy profile after restart:
HTTP_OK elapsed=61.91 prompt=31501 completion=2 total=31503 finish=length
profile_accuracy=240
profile_full_flat=240
residual_proxy=240
residual_proxy_accuracy=240
prepare_pool=240
record_topk=240
prefill_disabled_skips=480
waiting_prefill=8
timeout=0
errors=0
pool_recall_avg=0.961847 min=0.848600 max=0.998000 n=240
coverage_avg=0.961847 min=0.848600 max=0.998000 n=240
fullflat_recall_avg=1.000000 min=1.000000 max=1.000000 n=240
residual_recall_avg=0.905776 min=0.502900 max=0.988300 n=240

non-repeating source-code prompt accuracy profile:
prompt source: real files from the runtime container, tokenizer-truncated to
31500 tokens:
  - lmcache/v1/indexer_ssd_manager.py
  - vllm/model_executor/layers/sparse_attn_indexer.py
  - vllm/model_executor/models/deepseek_v4.py
  - vllm/model_executor/layers/deepseek_v4_attention.py
  - lmcache/integration/vllm/vllm_v1_adapter.py
  - lmcache/v1/gpu_connector/gpu_connectors.py
  - lmcache/v1/kv_layer_groups.py
  - lmcache/v1/metadata.py
HTTP_OK elapsed=66.21 prompt=31500 completion=2 total=31502 finish=length
profile_accuracy=240
profile_full_flat=240
residual_proxy=240
residual_proxy_accuracy=240
prepare_pool=240
record_topk=240
prefill_disabled_skips=480
waiting_prefill=8
timeout=0
errors=0
pool_recall_avg=0.961840 min=0.883800 p50=0.971700 max=0.991200 n=240
coverage_avg=0.961840 min=0.883800 p50=0.971700 max=0.991200 n=240
fullflat_recall_avg=1.000000 min=1.000000 max=1.000000 n=240
residual_recall_avg=0.930867 min=0.818400 p50=0.941400 max=0.983400 n=240
```

Interpretation:

- The pool scoring path is now exact against the official full-cache path when
  the pool contains the same tokens.
- On the 31.5K long-context first-decode profile, the 4096-slot pool covers and
  returns about 96.2% of the official full-cache top-k. The independent
  full-flat check remains exactly 1.0, so the residual miss is pool coverage,
  not scoring-layout drift.
- The residual proxy is no longer the broken low-recall path; latest short-run
  average recall is about 0.93 and the 31.5K first-decode average is about
  0.906 against full-cache top-k, matching the memory docs' high-accuracy
  HC-post proxy result.
- A non-repeating 31.5K source-code prompt produced similar or higher numbers:
  pool recall about 0.962 and residual-proxy recall about 0.931. The earlier
  high result is therefore not only an artifact of the repeated validation
  sentence, though it is still a first-decode-token measurement.
- Prefill proxy is intentionally disabled by default. The current long run
  validates that decode remains stable after chunked prefill and batch SSD
  initialization, not that prefill proxy prefetch is solved.
- The 31.5K/512 timing above is not an optimized production number; diagnostic
  work and full post-prefill SSD initialization are still expensive.

## Current residual-proxy patch status

Important correction from the 2026-05-26 night session:

- Do **not** implement DSv4 residual proxy in `deepseek_v2.py`.
- Do **not** use `hidden_states + residual` or `input_layernorm`; that is a V2
  mental model and is wrong for DSv4.
- DSv4 uses HCA/CSA with HC state. The usable `attn+residual` proxy is the
  attention HC-post state:

```text
residual_f = HC_post(attention_out_{L-1}, residual_a, post_a, comb_a)
proxy_hidden_L = attn_norm_L(HC_pre_L(residual_f, hc_attn_fn_L, ...))
spec_ids = DeepseekV4Indexer_L(proxy_hidden_L)[0:1024]
```

Relevant memory docs:

- `C:\Users\30141\.claude\projects\f--LMCache\memory\project_csa_prefetch_findings.md`
  says V4-Pro `attn+residual` proxy accuracy is about 83.8% and `hc_post(shared)`
  is about 92.9%.
- `C:\Users\30141\.claude\projects\f--LMCache\memory\project_system_design_csa_nvme.md`
  describes the same path in Chinese.

Current patch status:

- `F:\LMCache\lmcache\v1\indexer_ssd_manager.py`
  - adds `LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY`
  - stores decoder layers via `register_decoder_layer()`
  - when enabled, uses the target DSv4 decoder layer's own
    `hc_pre(...hc_attn...)` plus `attn_norm` to compute the proxy hidden state
  - builds V4 indexer inputs through the target attention module:
    `attn.mla_attn.attn_gemm_parallel_execute(proxy_hidden)`, `attn.q_norm(qr)`,
    then calls `DeepseekV4Indexer.forward(proxy_hidden, qr, kv_score, weights,
    positions, rotary_emb)`
  - temporarily detaches `indexer_op.ssd_manager` so proxy scoring uses the
    full official indexer-cache path, not the SSD pool path
  - saves and restores proxy-written compressor state rows and indexer K rows
    so proxy scoring does not pollute real KV state
  - logs `IndexerSSDManager: residual_proxy_accuracy ...` when
    `/tmp/lmcache_indexer_profile_accuracy` or
    `LMCACHE_INDEXER_PROFILE_ACCURACY=1` is enabled
- `F:\vllm_dev\vllm\models\deepseek_v4\nvidia\model.py`
  - registers `DeepseekV4DecoderLayer` instances in
    `_DEEPSEEK_V4_DECODER_LAYER_REGISTRY`
  - `attach_indexer_prefetch(manager, next_csa_layer_id)` stores the manager
  - `_forward_cuda()` calls
    `self._fire_indexer_prefetch(residual, positions)` after the attention
    `mhc_fused_post_pre(...hc_ffn...)` step, where `residual` is the DSv4
    attention HC-post state
  - `_forward_native()` does the same after `x = hc_post(attn_out, ...)` and
    before FFN
- `F:\vllm_patch\vllm\model_executor\models\deepseek_v2.py`
  - the incorrect V2 `hidden_states + residual` residual-proxy hook was removed
    locally and should not be deployed for DSv4.

The local syntax check passed:

```text
python -m py_compile F:\LMCache\lmcache\v1\indexer_ssd_manager.py
python -m py_compile F:\vllm_dev\vllm\models\deepseek_v4\nvidia\model.py
python -m py_compile F:\vllm_patch\vllm\model_executor\models\deepseek_v2.py
```

Next development steps:

1. Confirm the runtime vLLM DSv4 path before copying. Earlier notes said the
   container had old layout
   `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v4.py`,
   while the local source now has new layout
   `F:\vllm_dev\vllm\models\deepseek_v4\nvidia\model.py`. Patch the actual
   imported DSv4 file, not V2.
2. Copy the patched LMCache manager:
   - `/opt/venv/lib/python3.12/site-packages/lmcache/v1/indexer_ssd_manager.py`
3. Ensure the runtime has:

```text
LMCACHE_INDEXER_ENABLE_PREFETCH=1
LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=1
LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS=0
```

4. Restart `dsv4-indexer-ssd`.
5. For decode-only accuracy checks, one generated token is enough; longer decode
   can show whether accuracy drifts as the sequence grows.
6. Check for:
   - `IndexerSSDManager: residual_proxy layer ... spec_tokens=1024`
   - `IndexerSSDManager: residual_proxy_accuracy ... avg_recall=...`
7. Do not enable prefill proxy by setting `LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS`
   above 0 until the separate proxy metadata/chunked-prefill design is
   implemented.

## Required next design: prefill proxy with chunked-prefill equivalence

User requirement: prefill should also prefetch, but chunked prefill must behave
like non-chunked prefill. Do not implement a chunk-local shortcut that changes
which logical tokens are eligible.

The correct target behavior is:

- For a non-chunked prompt of length `N`, the proxy decision for a layer sees
  the same logical compressed indexer token IDs that a chunked run of the same
  prompt eventually sees after all chunks are accumulated.
- During chunked prefill, track global logical token IDs, not chunk-local rows.
- A prefill proxy request may only schedule SSD reads for token IDs `< cursor`
  of the target CSA layer. The cursor means the target layer's SSD store has
  already been initialized up to that logical token ID.
- If the target layer store is not initialized yet, save the proxy intent or
  skip safely; never issue reads from empty/uninitialized SSD.
- Use a proxy metadata object whose batch/sequence shape matches the proxy
  tensor. Reusing the original full prefill metadata with a sliced one-row
  tensor caused CUDA illegal access and must not be repeated.
- For the first full prompt prefill, the most useful work is likely seeding the
  pool and preparing intent for the first decode token; true I/O overlap during
  initial prefill only works for chunks whose referenced prefix is already on
  SSD.

Implementation sketch:

1. Keep `LMCACHE_INDEXER_PREFETCH_PREFILL_ROWS=0` as the default safe setting.
2. Add an explicit prefill-proxy state machine keyed by layer and sequence.
3. On each prefill chunk, compute proxy IDs with correct metadata, translate
   them to global compressed token IDs, and filter by `0 <= tid < cursor`.
4. Schedule filtered reads through the same async queue used by decode.
5. After the final chunk initializes SSD, seed the pool exactly as the
   non-chunked path would; then decode sees identical resident/global IDs.
6. Add a diagnostic comparing chunked vs artificially non-chunked prefill for
   the same prompt: final pool IDs, SSD byte rows, and first-decode top-k should
   match.

## Useful test commands

Use `docker exec -i` for heredoc Python scripts. Without `-i`, Python receives an
empty stdin and exits successfully without running the request body.

Short prefetch probe:

```bash
sudo docker exec -i dsv4-indexer-ssd python3 -u - <<'PY'
import json, time, urllib.request

payload = {
    "model": "/mnt/nvme0/models/DeepSeek-V4-Pro",
    "prompt": "CSA SSD pool prefetch verification prompt. " * 1800,
    "max_tokens": 80,
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
print("HTTP", r.status, "elapsed", round(time.time() - t0, 2))
print("usage", obj["usage"])
print("finish", obj["choices"][0]["finish_reason"])
PY
```

Long validation request:

```bash
sudo docker exec -i dsv4-indexer-ssd python3 -u - <<'PY'
import json, time, urllib.request

payload = {
    "model": "/mnt/nvme0/models/DeepSeek-V4-Pro",
    "prompt": "LMCache DeepSeek V4 thirty two thousand token validation paragraph. " * 2144,
    "max_tokens": 600,
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
print("HTTP", r.status, "elapsed", round(time.time() - t0, 2))
print("usage", obj["usage"])
print("finish", obj["choices"][0]["finish_reason"])
PY
```

Log count check:

```bash
LOG=/tmp/dsv4_final_32k_600_counts.log
sudo docker logs --since '2026-05-26T10:33:19+00:00' dsv4-indexer-ssd > "$LOG" 2>&1
grep -c 'prepare_pool layer .* valid_slots=4096' "$LOG"
grep -c 'record_global_topk layer .* count=1024' "$LOG"
grep -c 'fire_async_for_layer layer .* active' "$LOG"
grep -Ec 'async read failed|fallback sync read|Traceback|RuntimeError|ValueError|CRITICAL|ERROR|UserWarning' "$LOG"
```

## Do not regress

- Do not filter `self.kv_caches` down to 61 layers.
- Do not use MP mode unless the multi-NVMe SSD-only requirement changes.
- Do not remove `use_gpu_connector_v3: true` from the SSD-only config.
- Do not make `LMCACHE_INDEXER_SSD_DIR` alone enable prefetch; use
  `LMCACHE_INDEXER_ENABLE_PREFETCH`.
- Do not use `schedule_metadata=None` with `fp8_fp4_paged_mqa_logits`.
- Do not expose the indexer SSD pool to DeepGEMM as block size 1; H100 expects
  64-token pages for this path.
- Do not return pool-local top-k slot IDs to vLLM; translate back to global
  compressed indexer token IDs.
- Do not implement DSv4 residual proxy in `deepseek_v2.py` or with
  `hidden_states + residual`; use DSv4 attention HC-post `residual_f`.
- Do not claim a request test ran if the command used `docker exec` heredoc
  without `-i`.
