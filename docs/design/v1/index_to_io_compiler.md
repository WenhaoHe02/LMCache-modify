# Index-to-I/O Compiler for Native Sparse Attention

> Runtime status (2026-07-16): the content-addressed Stage0 replay described
> in parts of this document was rejected and removed. Dynamic CSA block
> selection is not a function of prefix identity alone. Production has no
> Stage0 plan I/O: profile-disabled layers stay native, while selected deep
> layers run one target-minus-two residual prediction followed by exact
> target-indexer correction. Stage0 references below are historical context.

## Status and scope

This document specifies a staged compiler and runtime for turning sparse
attention decisions into SSD reads and final KV-cache placement. It targets
both DSv4 CSA/HCA/SWA and GLM DSA with IndexShare. The reusable interfaces must
not encode either model's layout assumptions.

The design builds on the current staged HCA-to-CSA pipeline. It does not make
the residual predictor authoritative and does not weaken the target layer's
attention semantics by default.

The implementation goals are:

1. reuse a sparse decision across layers or requests when its correctness tag
   permits it;
2. lower the same decision into indexer-K and attention-KV SSD ranges;
3. choose sparse, range, or bulk I/O from measured cost rather than a fixed
   environment switch;
4. separate NVMe completion from GPU materialization so each resource can be
   scheduled against the target layer's deadline; and
5. preserve correctness under prediction failure, cancellation, concurrent
   requests, row reuse, partial I/O, and late completion.

The first implementation remains process-local and rank-local. Persisting
plans or coordinating them across ranks is out of scope until the local
correctness and benefit are established.

## Core thesis

Sparse compute indices do not automatically produce sparse storage I/O. During
a long prefill, the union of many per-query top-K results can cover most of the
prefix. Per-object host callbacks and immediate scatter can then dominate even
when the SSD data plane sustains 9--11 GB/s.

The proposed system treats a true or guarded sparse index as an immutable plan:

```text
 true anchor index / exact decision replay / speculative prediction
                              |
                              v
                    correctness-tagged Plan IR
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
     indexer-K physical ranges         attention-KV physical ranges
             |                                 |
             +---------- adaptive lowering ----+
                              |
                              v
        NVMe admission -> CQ_READY -> GPU materialization -> HMA_READY
```

One plan may therefore remove redundant indexer computation, remove the
corresponding indexer-K bytes, and prepare the attention-KV bytes. Those three
effects are only claimed when the plan's correctness class permits them.

## Correctness taxonomy

Every plan has exactly one correctness class.

| Class | Source | May prefetch KV | May replace per-row top-K | May skip indexer |
|---|---|---:|---:|---:|
| `EXACT_TRUE` | Current official indexer | Yes | Yes | Already executed |
| `ARCHITECTURAL_SHARED` | Model-defined IndexShare group | Yes | Yes | Yes |
| `EXACT_REPLAY` | Exact decision-key hit with cached per-row top-K | Yes | Yes | Yes |
| `CERTIFIED_EXACT` | Future proof/certificate | Yes | Yes | Yes |
| `SPECULATIVE` | Anchor reuse, residual prediction, heavy hitters | Yes | No | No |
| `QUALITY_GUARDED` | Calibrated approximate reuse | Yes | Opt-in | Opt-in |

Two boundaries are non-negotiable:

- A block-union bitmap is an I/O plan, not the per-row attention decision. It
  can never be returned as the target indexer's top-K.
- True-topK miss correction repairs KV residency only. It does not repair an
  approximate attention index. A speculative target must still run the
  official indexer exactly once.

`QUALITY_GUARDED` is an explicit approximate mode. Calibration, shadow
execution, and drift detection can limit its risk, but cannot make an
individual request exact. The default production path uses only
`SPECULATIVE` prefetch and exact target computation.

`ARCHITECTURAL_SHARED` means exact with respect to the shipped model's
inference semantics, not that independently computed indices at the consumer
layers would be identical. GLM-5.2, for example, declares Full and Shared
indexer layers in its model config. The compiler may skip the Shared-layer
indexer because the model itself says that the layer consumes its Full-layer
owner's index. It must not infer this authority from a high measured overlap.

## GLM DSA and IndexCache semantics

The compiler represents cross-layer index reuse as an explicit `IndexGroup`:

```text
IndexGroup owner: Full layer F
Index decision:   top-K(F, query span)
Consumers:        F, S1, S2, ...

                       one exact shared index
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
       DSA KV lowering(F) DSA KV lowering(S1) DSA KV lowering(S2)
       owner deadline      consumer deadline    consumer deadline
```

The index decision is shared, but the physical KV plan is still lowered once
per consumer. Layers have different KV objects, SSD extents, target rows, and
deadlines. Indexer-K is loaded for the owner only; attention-KV is loaded for
every consumer according to the same logical top-K.

There are two distinct ways to construct a group:

1. **Architectural group.** GLM-5.2's `indexer_types` config is authoritative.
   A `full` layer starts a group and following `shared` layers consume that
   owner's index. The public GLM-5.2 config contains 78 entries and, after the
   initial Full layers, uses one Full followed by three Shared layers. These
   groups are safe for deterministic indexer skip under the shipped model.
2. **Calibrated group.** Training-free IndexCache chooses F/S layers by greedy
   search on language-model loss; training-aware IndexCache changes training
   so retained indexers serve multiple layers. Such a profile is
   `QUALITY_GUARDED` unless the exact trained model/config declares the sharing
   pattern. Pairwise top-K overlap alone never upgrades a group to exact.

The structural profile also records `index_topk`, indexer RoPE semantics,
skip offsets, the F/S pattern, and whether MTP iterations share the index.
Changing any of them invalidates the profile and its cached plans.

## Architecture

The system has five logical components. `StaticProfiler` is the compiler front
end; it prevents layer distance, sparse/bulk thresholds, and overlap windows
from becoming hand-written constants.

```text
 model config ----> ModelTopologyProfile ----------------------+
 offline traces --> StaticPrefetchProfile ---------------------+
                                                               v
  PlanRecorder              PlanCompiler               PlanRuntime
  ------------              ------------               -----------
  true top-K scan  ---> immutable logical IR ---> bind request layout
  query/content key         codec and delta       resident/inflight filter
  union observations        static prior          deadline/admission override
                                                          |
                                                          v
                                                   TuttiDirectLoader
                                                   submit / CQ completion
                                                          |
                                                          v
  ResidencyTracker <--- generation-checked materialization / CUDA event
```

`PlanRecorder` observes work that the true indexer already performs. It must
not add a second scan of the full per-query top-K tensor. `PlanCompiler` owns
logical plans and turns them into immutable request-bound plans. `PlanRuntime`
owns execution, scheduling, sharing, and cancellation. `ResidencyTracker`
owns the public resident/in-flight view used by miss correction.

### Static profiler boundary

The static profiler produces two separate, versioned artifacts:

- `ModelTopologyProfile` is extracted deterministically from the exact model
  config and weight/adaptor fingerprint. It records attention family, Full or
  Shared indexer mode, index-group owner and consumers, top-K, and layout ABI.
  This artifact may grant `ARCHITECTURAL_SHARED` authority.
- `StaticPrefetchProfile` is fitted from a short offline matrix over
  `(layer, prefix-length bucket, query-length bucket, strategy, lookahead)`.
  Each candidate records P95 I/O-plus-materialization service time, P05
  available compute overlap, physical bytes, HBM footprint, and sample count.
  It is performance guidance only.

For each layer and workload bucket, the static selector minimizes

```text
predicted gate stall = max(0, service_time_p95 - compute_overlap_p05)
```

and breaks ties by SSD bytes, HBM bytes, and then shorter lookahead. This
directly answers whether prefetching two layers early is useful: it is selected
only when the additional layer contributes a measured overlap window large
enough to hide service time without unacceptable read/HBM amplification.

The profile is keyed by model/config hash, TP/EP layout, GPU and SSD class,
storage-layout ABI, dtype/compression, and calibration workload id. Missing or
stale profiles fall back to demand I/O and the official model index path.
Runtime EWMAs may adjust cost coefficients or reject admission, but may never
change index ownership or promote calibrated similarity to exactness.

The first profiler implementation is deliberately offline and read-only. It
consumes traces produced by normal correctness runs; it does not add a second
top-K scan or a profiler callback to the serving hot path.

## Plan IR

### Enums

```python
class Correctness(Enum):
    EXACT_TRUE = auto()
    ARCHITECTURAL_SHARED = auto()
    EXACT_REPLAY = auto()
    CERTIFIED_EXACT = auto()
    SPECULATIVE = auto()
    QUALITY_GUARDED = auto()


class LayerRole(Enum):
    INDEXER_K = auto()
    CSA_KV = auto()
    HCA_KV = auto()
    SWA_KV = auto()
    DSA_KV = auto()


class PlanRole(Enum):
    ANCHOR = auto()
    SHARED = auto()
    DELTA = auto()
    CORRECTION = auto()


class LogicalCodec(Enum):
    LIST = auto()
    BITMAP = auto()
    RANGES = auto()
    BULK = auto()
    XOR_DELTA = auto()


class PlacementKind(Enum):
    VLLM_ROWS = auto()
    BLOCK_SLOT = auto()
    INDEXER_POOL = auto()
    STAGING_ONLY = auto()
```

### Reusable logical plan

```python
@dataclass(frozen=True, slots=True)
class ContentKey:
    schema: int
    model_fingerprint: bytes
    tp_world_size: int
    rank: int
    prefix_root: bytes
    prefix_blocks: int


@dataclass(frozen=True, slots=True)
class DecisionKey:
    content: ContentKey
    target_layer: int
    query_root: bytes
    query_start: int
    query_count: int
    indexer_config_hash: bytes


@dataclass(frozen=True, slots=True)
class Coverage:
    logical_limit: int
    selected_count: int
    density: float
    query_start: int
    query_count: int
    observed_queries: int
    recall_lcb: float | None
    read_amplification_bound: float


@dataclass(frozen=True, slots=True)
class Confidence:
    correctness: Correctness
    calibration_id: str | None
    sample_count: int
    row_recall_lcb: float | None
    union_recall_lcb: float | None
    quality_bound: float | None


@dataclass(frozen=True, slots=True)
class LogicalSelection:
    codec: LogicalCodec
    block_limit: int
    payload: bytes
    base_plan_id: bytes | None


@dataclass(frozen=True, slots=True)
class LayerPlan:
    plan_id: bytes
    decision_key: DecisionKey
    target_layer: int
    anchor_layer: int
    layer_role: LayerRole
    plan_role: PlanRole
    selection: LogicalSelection
    coverage: Coverage
    confidence: Confidence
    per_row_topk_ref: bytes | None
    deadline_offset_us: int
    max_read_bytes: int
    max_hbm_bytes: int
```

`plan_id` is the hash of a canonical serialization of all semantic fields.
The reusable plan must not contain vLLM physical rows, SSD offsets, CUDA
events, request IDs, or mutable resident state.

`per_row_topk_ref` is required for `EXACT_REPLAY`. It refers to a separately
checksummed, shape-described top-K payload. A union-only plan must leave it
unset.

### Request-bound physical plan

```python
@dataclass(frozen=True, slots=True)
class PhysicalRange:
    source_path: str
    source_offset: int
    dma_bytes: int
    payload_skip: int
    logical_bytes: int
    logical_blocks: tuple[int, ...]
    target_rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BoundPlan:
    logical: LayerPlan
    request_id: str
    source_layout_id: bytes
    layout_generation: int
    ranges: tuple[PhysicalRange, ...]
    physical_strategy: LogicalCodec
    deadline_ns: int
    placement: PlacementKind
    estimated_io_us: int
    estimated_materialize_us: int
```

The compiler creates a `BoundPlan` from an immutable
`RequestLayoutSnapshot`. `layout_generation` prevents a late scatter from
writing into rows that vLLM has reassigned to a newer request.

### Example

Suppose true top-K for query rows 0--1023 selects compressed blocks
`{0, 1, 2, 5, 6}` in layer 18. The cache representation may be a bitmap. For
the active request, blocks 0--2 are resident, blocks 5--6 occupy one contiguous
source extent, and their target rows are 91 and 104.

```text
LogicalSelection: BITMAP {0,1,2,5,6}
Resident filter:  {5,6}
Physical strategy: RANGE
DMA: [source(block 5), source(end block 6))
Materialize: staging blocks [0,1] -> HMA rows [91,104]
```

The reusable bitmap remains unchanged. Only the bound plan contains the
current physical rows.

## Content and decision keys

### Merkle construction

Each ordered source chunk contributes a leaf:

```text
leaf = H(
    schema,
    CacheEngineKey logical content identity,
    logical_start,
    logical_end,
    bytes_per_block,
    layer-role layout schema
)
```

Physical rows, LBA addresses, pool offsets, and request IDs are excluded. The
ordered leaves form `prefix_root`. Segment roots are retained so append can be
updated in `O(log n)` and the longest common prefix can be found without
rehashing every chunk.

`model_fingerprint` covers model weights, the sparse indexer weights, layout
revision, TP size, and compression parameters. `indexer_config_hash` covers
top-K, RoPE/position semantics, compressor configuration, and operator
revision.

### Source reuse versus decision reuse

The caches are deliberately separate:

```text
ContentKey  -> SourceMap / extents / reusable source execution
DecisionKey -> logical selection and optional exact per-row top-K
```

Sharing only a prefix does not make a new suffix's top-K identical. The new
suffix changes the query rows and therefore changes `query_root`. Such a
request may reuse source extents, an in-flight DMA, or a speculative
heavy-hitter plan, but it cannot use `EXACT_REPLAY`.

An exact decision hit requires the same prefix, query content, query positions,
model fingerprint, target layer, and indexer configuration.

### Delta plans

A delta plan stores either an XOR bitmap or added/removed ranges relative to an
immutable base plan. The encoder chooses the smaller of:

```text
full bitmap
sorted added and removed block IDs
XOR bitmap, optionally run-length encoded
```

The base plan is strongly referenced by every live delta. Before evicting a
base, the cache must materialize dependent deltas or evict them together.
Delta encoding is a metadata and lowering optimization; a 235-byte bitmap
alone is too small to justify a systems claim.

## Anchor and shared-layer policy

### Calibration table

Offline calibration is keyed by:

```text
(model fingerprint, anchor layer, target layer,
 context-length bucket, query-depth bucket, layer role)
```

Each record includes sample count, row-recall distribution, union recall,
Jaccard similarity, added/removed block density, quality experiment ID, and a
lower confidence bound rather than only the mean.

Historical DSv4 measurements motivate depth-sensitive policy: shallow-layer
age-3 recall fell to 57--81%, while deeper layers retained 86--95%. A fixed
model-wide stride is therefore invalid.

### Runtime selection

```python
def choose_index_policy(ctx, anchor_layer, target_layer):
    if exact_decision_cache.contains(ctx.decision_key(target_layer)):
        return EXACT_REPLAY

    calibration = calibration_table.lookup(ctx, anchor_layer, target_layer)
    if calibration is None or calibration.samples < min_samples:
        return DISABLED
    if calibration.union_recall_lcb < prefetch_min_recall:
        return DISABLED

    if approximate_skip_enabled:
        if calibration.row_recall_lcb >= skip_min_recall:
            if quality_registry.accepts(calibration.quality_experiment_id):
                return QUALITY_GUARDED

    return SPECULATIVE
```

The initial implementation supports only `DISABLED`, `SPECULATIVE`, and
`EXACT_REPLAY`. `QUALITY_GUARDED` is added only after an explicit quality
evaluation and remains off by default.

In speculative mode, the anchor supplies candidate reads and the target's
official indexer remains authoritative. In exact replay, the cached per-row
top-K is returned after validating its checksum, shape, query span, and key.

An approximate mode needs periodic shadow true-indexer execution and a drift
circuit breaker. A calibration miss, a new model fingerprint, an unsupported
shape, or an exceeded error threshold immediately restores the official
indexer.

## Adaptive lowering

### Candidate strategies

The cached codec and physical strategy are distinct. Before lowering, the
compiler computes:

```text
missing = selected - resident - in_flight
```

It evaluates:

- `LIST`: sorted logical block IDs for a very sparse plan;
- `BITMAP`: GPU bitmap compacted to IDs without a large CPU transfer;
- `RANGES`: adjacent source blocks coalesced into aligned byte ranges;
- `BULK`: the full registered slab or a full chunk; and
- `XOR_DELTA`: decoded against its base, then lowered using one of the above.

Source-contiguous blocks may use one DMA even if target rows are scattered; a
single GPU materializer then places them. A direct contiguous copy is used
when both source and target are contiguous. Ranges never merge blindly across
paths or incompatible layer roles.

Every source read is rounded to the 512-byte NVMe boundary. `payload_skip` and
`logical_bytes` preserve the exact payload. GPU staging allocation uses its own
page alignment.

### Cost model

For candidate `p`:

```text
T_lower(p) = decode(codec)
           + compact(bitmap_blocks)
           + map(selected_blocks)
           + resolve(extents)

T_io(p) = NVMe_queue_wait
        + alpha_submit
        + beta_iocb * number_of_iocbs
        + dma_bytes / measured_NVMe_bandwidth

T_materialize(p) = SM_queue_wait
                 + alpha_launch
                 + beta_row * target_rows
                 + copied_bytes / measured_G2G_bandwidth
                 + phase_interference

T_finish(p) = now + T_lower(p) + T_io(p) + T_materialize(p)

Score(p) = T_finish(p)
         + lambda_waste * (dma_bytes - expected_consumed_bytes)
         + lambda_hbm * peak_staging_bytes
         + lambda_deadline * deadline_miss_probability
```

Rank-local exponentially weighted measurements update the constants. Initial
values come from the measured sparse-read relation
`T_load(N) = 0.42 + 0.021*N ms`, contiguous I/O near 3.3 microseconds per
block, and 9--11 GB/s aggregate data-plane bandwidth.

The selected candidate must satisfy deadline, staging, HBM, batch-byte, and
IOCB budgets. A 10--15% hysteresis prevents repeated sparse/bulk switching.

If no complete speculative plan can finish before the deadline, the compiler
ranks cancellable microbatches by expected avoided gate time per DMA byte. It
submits only the affordable prefix of that ranking. The remaining blocks are
left to exact demand correction; speculation never delays the gate merely to
finish itself.

## Dual-resource scheduling

### State machine

```text
                         cancel
                           |
                           v
CACHED -> BOUND -> ADMISSION_WAIT -> SUBMITTED -> CQ_READY
  |         |             |              |           |
  |         +-> CANCELLED <-+              +-> ORPHANED
  |                                                  |
  +-> EVICTED                              MATERIALIZE_WAIT
                                                     |
                                          +----------+---------+
                                          |                    |
                                          v                    v
                                      HMA_READY              FAILED
                                          |
                                          v
                                       CONSUMED
```

`CQ_READY` means SSD data is present in staging. It does not mean the target
HMA rows are valid. Only a completed, generation-checked materialization event
transitions the request child to `HMA_READY`.

### NVMe admission

The priority order is:

```text
foreground demand > cold store > speculative read
```

Speculation submits its next microbatch only when no higher-priority work is
announced, the token bucket has credit, and staging/IOCB reservations are
available. The initial caps remain eight I/Os and 8 MiB per speculative batch.
An SQ submission cannot be cancelled; only unsubmitted batches can be removed.

### GPU materialization

NVMe and GPU scheduling use separate queues and credits. A CQ-ready result can
remain in registered staging while the GPU is running MoE, EP/NCCL, the
official indexer, or a more urgent materializer.

Materialization uses earliest-deadline-first order with a guard:

```text
guard = p99(materialization time)
      + p99(materializer queue delay)
      + clock and launch margin
```

Before the guard, a request may defer scatter during a known conflicting
model phase. At the guard it either materializes or reports not-ready so the
foreground demand path can take over. The scheduler uses known layer/NVTX
phases; it must not introduce a synchronous online SM-utilization query.

The runtime exposes opaque `PlanTicket`/`PlanLease` objects with public
`query_io_ready`, `query_hma_ready`, `wait_hma`, `promote`, and `cancel`
operations. Callers do not inspect runtime queues.

## Concurrency, reference counts, and failure semantics

### Ownership hierarchy

```text
immutable LayerPlan
    -> request-local BoundPlan
        -> shared PlanExecution for source DMA
            -> one materialization child per target layout generation
                -> subscriber-local PlanLease
```

The execution deduplication key is `(plan_id, source_layout_id, rank)`. Two
requests may share a DMA into staging, but each distinct target-row generation
requires its own materialization child.

Reference counts are explicit for:

- cached plans and base/delta dependencies;
- execution subscribers;
- staging buffers;
- target materialization children; and
- HMA row residency epochs.

### Cancellation

- `BOUND` or `ADMISSION_WAIT`: remove the work and release reservations.
- `SUBMITTED`: mark the subscriber orphaned. If no subscribers remain, release
  the staging result immediately after CQ completion.
- `CQ_READY` or `MATERIALIZE_WAIT`: cancel only that target child; a shared
  source execution continues for other subscribers.
- `HMA_READY`: release the lease. Row residency remains governed by its epoch
  and row references.

A demand request may atomically promote an existing speculative lease and
inherit completed I/O. Promotion must not submit the same logical block twice.

### Partial failure

- Key, schema, checksum, decode, or lowering failure discards the plan and
  falls back to the official indexer and demand path.
- A partial NVMe failure marks only successfully completed ranges as eligible
  for materialization. Failed blocks remain missing and are retried by demand.
- A scatter or CUDA-event failure leaves the complete child non-resident.
- A timeout cancels unsubmitted speculation and falls back; it does not kill a
  shared worker thread.
- Exact top-K cache corruption fails closed and runs the official indexer.

### Required invariants

1. A block becomes resident only after its materialization event completes and
   its layout generation still matches.
2. A `SPECULATIVE` plan never changes the official top-K return value.
3. Only `EXACT_REPLAY` or `CERTIFIED_EXACT` may skip the official indexer.
4. One physical `(generation, layer, block)` has at most one materialization
   owner; other users subscribe to its event.
5. Staging remains referenced until every dependent CUDA event completes.
6. Late completion from a cancelled request cannot update a newer request's
   rows or resident bitmap.
7. A deadline miss may reduce performance but cannot reduce correctness.
8. Locks are never held across CUDA synchronization, `Future.result`, or I/O.

The global lock order is plan cache, execution table, request layout, then
layer residency.

## Public API sketch

```python
class PlanCompiler:
    def record_true_topk(
        self,
        context: RequestPlanContext,
        layer_id: int,
        query_span: tuple[int, int],
        topk: torch.Tensor,
    ) -> LayerPlan: ...

    def lookup(self, key: DecisionKey) -> LayerPlan | None: ...

    def bind(
        self,
        plan: LayerPlan,
        layout: RequestLayoutSnapshot,
    ) -> BoundPlan: ...

    def lower(
        self,
        plan: BoundPlan,
        residency: ResidencySnapshot,
    ) -> LoweredPlan: ...


class PlanRuntime:
    def submit(self, plan: LoweredPlan, priority: Priority) -> PlanLease: ...
    def promote(self, lease: PlanLease, priority: Priority) -> None: ...
    def wait_hma(self, lease: PlanLease, timeout_s: float) -> Completion: ...
    def cancel(self, lease: PlanLease) -> None: ...
```

All public methods have typed inputs and return values. No module reaches into
another class's underscore-prefixed fields.

## Mapping to the current implementation

### `index_to_io_profile.py`

This pure front-end module owns `ModelTopologyProfile`, `IndexGroupTopology`,
and `StaticPrefetchProfile`. `extract_glm_dsa_topology` reads the authoritative
`indexer_types` array and never guesses groups from layer distance. The static
selector consumes already-aggregated trace measurements and emits a prior; it
does not issue I/O or inspect mutable runtime state.

### `csa_block_plan_cache.py`

`CSABlockPlan` becomes a compatibility wrapper around
`LogicalSelection(BITMAP)`. `CSABlockPlanCache` evolves into an immutable,
checksummed `DecisionKey` LRU with base/delta dependency accounting.

The current `record` method ORs every observation under one content signature.
The new cache includes query span/root in the decision key or records an
explicit aggregate plan. Otherwise unrelated query observations silently
inflate union coverage.

The new cache also needs a non-refreshing `peek` operation. A check followed by
a lookup must not update LRU order twice.

### `CSAAttentionKVPrefetchManager`

`register_request_chunks` creates an immutable `RequestLayoutSnapshot` with a
Merkle content key and monotonic layout generation. Background work binds to
that snapshot instead of reading mutable `state.chunks`.

The existing GPU `seen` bitmap in `_miss_ids_for_topk` is the recorder input.
It already avoids a second scan of the very large top-K tensor. The compiler
records the query span and decision key before the normal exact miss-mask
calculation continues.

`fire_cached_block_plan` becomes `lookup -> bind -> lower -> submit`.
`_submit_reads` and `_issue_reads` initially remain adapters, then move behind
`PlanRuntime`. HCA's block-slot scatter becomes
`PlacementKind.BLOCK_SLOT`, not a compiler-wide special case.

The current resident bitmap remains the transitional backing store. It is
eventually exposed only through a public `ResidencyTracker` API.

### `IndexerSSDManager`

`fire_stage0_for_layer` invokes the compiler. Stage1 HCA residual prediction
produces a `SPECULATIVE` delta candidate.

The existing true-topK record/correction points publish anchor calibration
telemetry and exact decision entries. The first implementation never skips
the target indexer. A public `select_index_policy` later returns `DISABLED`,
`SPECULATIVE`, `EXACT_REPLAY`, or `QUALITY_GUARDED`.

Indexer K uses the same IR and lowering path as attention KV. This is required
for a shared index to eliminate both indexer computation and indexer-K SSD
bytes. The current Tutti indexer bytes API includes a GPU-to-CPU bounce; the
prototype may retain it, but the evaluated path must consume GPU tensors.

### `TuttiIndexerStorage` and `TuttiDirectLoader`

The current `load_chunks_to_hbm` priority, deadline, batch-byte, batch-IO, and
raw callback parameters are sufficient through adaptive-lowering prototype
stages.

Separating CQ readiness from materialization requires a non-blocking API:

```python
def submit_ranges(
    ranges: Sequence[PhysicalRange],
    reservation: IoReservation,
    priority: Priority,
) -> IoTicket: ...

def wait_cq(ticket: IoTicket) -> CqResult: ...

def release(ticket: IoTicket) -> None: ...
```

`CqResult` retains staging views, per-range status, and a completion event. The
materializer does not run inside `on_raw_batch_loaded`. Completion and scatter
must not hold the single SQ/CQ I/O lock.

## Implementation stages

Each stage is independently measurable and revertible.

### P0: telemetry only

Record true-topK query-span unions, layer-pair overlap, extent count, density,
resident state, per-strategy service time, conservative compute overlap, and
layer deadlines. Add no new read or synchronization.

### P1: IR and shadow compiler

Add canonical keys, Merkle construction, codecs, plan cache, and dry-run
lowering. Record plans from the existing true-topK scan. Continue to execute
the old Stage0 path.

### P1.5: topology extraction and static profiler

Extract model-defined index groups from config, starting with GLM DSA
`indexer_types`. Build a versioned offline calibration artifact for each layer
and workload bucket. Validate hash invalidation and fallback behavior before a
profile is allowed to influence runtime admission.

### P2: compiler parity

Route exact content-plan prefetch through the compiler. For every layer, its
logical block set must equal the old `fire_cached_block_plan` result. Do not
enable adaptive lowering or index skipping.

### P3: adaptive lowering

Enable list/range/bulk choice with measured costs. Continue using the current
blocking Tutti callback and scatter implementation. Compare fixed sparse,
fixed bulk, and adaptive policies.

### P4: split runtime

Introduce `IoTicket`, explicit `CQ_READY`, an EDF materializer, generation
checks, leases, reference counts, and cancellation. Validate one request before
enabling source execution sharing.

### P5: shared-anchor speculative prefetch

Load the calibration table and use selected anchor layers only for speculative
I/O. The target official indexer remains unchanged.

### P6: architectural IndexShare execution

For GLM DSA, compute/load indexer-K once at each Full owner, retain its exact
per-row top-K for the declared Shared consumers, and lower that selection into
each consumer's distinct KV layout and deadline. Compare against an
IndexCache-only baseline that skips compute but does not optimize SSD I/O.

### P7: exact decision replay

Cache per-row top-K for small shapes first. Skip the official indexer only on a
complete decision-key hit. Keep quality-guarded reuse behind a separate,
default-off experiment flag.

### P8: cross-request execution sharing

Add source-map reuse, shared in-flight DMA, segment deltas, and target-specific
materialization children.

### P9: exact query-tile DAG research branch

Pipeline official true indexer tile `i+1`, I/O for tile `i`, and attention for
tile `i-1`. This addresses whole-prefill union saturation without prediction,
but changes vLLM/operator boundaries and overlaps strongly with prior
microtask-DAG work. It remains a research branch until the offline simulator
passes its threshold.

## Test matrix

### Unit and property tests

- canonical serialization and `plan_id` stability;
- Merkle append, segment root, longest-common-prefix, and model invalidation;
- list, bitmap, ranges, bulk, and XOR-delta round trips;
- base eviction with live deltas;
- randomized logical-set to physical-range equivalence;
- 512-byte alignment, payload skip, and HCA block-slot placement;
- exact replay rejection on model, query, position, shape, or checksum mismatch;
- speculative plans never changing returned top-K;
- cost-model oracle and sparse/bulk hysteresis;
- fake-clock EDF, priority, token bucket, and deadline guard;
- cancellation from every state and demand promotion;
- partial CQ failure and fallback;
- layout-generation ABA and late-completion rejection;
- shared execution subscriber races; and
- staging retention until the final CUDA event.

Tests target public interfaces. They do not read private fields of another
class.

### Integration tests

| Scenario | Required result |
|---|---|
| Exact Stage0 parity | Same blocks and original top-K as old path |
| Plan miss | Official indexer and demand correction succeed |
| Partial speculative completion | Missing blocks fetched by demand |
| Same prefix, different suffix | Source reuse allowed, exact decision reuse denied |
| Same exact request | Exact replay allowed only with per-row payload |
| Two subscribers, one cancellation | Remaining subscriber completes |
| Row generation changes mid-I/O | Late scatter discarded |
| Demand arrives during speculation | Demand promotes or preempts unsubmitted work |
| Materializer misses deadline | Gate falls back without incorrect residency |
| Corrupt exact payload | Official indexer runs |

## Offline falsification workflow

### Inputs

Use Parquet or NPZ with:

- run, request, model, rank, layer, context length, query start, and query
  count;
- per-row true top-K compressed IDs;
- anchor/residual predicted top-K when available;
- chunk logical ranges, bytes per block, source path/extents, and target role;
- resident and in-flight timeline;
- NVTX timing for indexer, attention, MoE/NCCL, I/O submit/CQ, scatter, and
  gate; and
- a quality-run ID for approximate skip experiments.

### Outputs

```text
pair_calibration.json
    per layer pair/bucket: recall LCB, Jaccard, delta density, sample count

lowering_oracle.parquet
    each candidate: bytes, IOCBs, lower/io/materialize time, waste, finish

tile_sim.parquet
    tile size x HBM budget: union, reread bytes, predicted critical path

scheduler_replay.json
    eager versus EDF: gate p50/p99, kernel elongation, staging peak

reuse_report.json
    exact-decision, source-map, in-flight-sharing hit rates and delta sizes
```

### Success thresholds

- Cross-layer prefetch: cover at least 50% of target indexer calls, union
  recall lower bound at least 90%, and read amplification at most 1.25.
- Approximate index skipping: separate end-to-end quality evaluation with no
  statistically significant regression; recall alone is insufficient.
- Adaptive lowering: at least 10% lower predicted completion than the best
  fixed policy, or at least 2x fewer IOCBs with no more than 1.10x bytes.
- Scheduler: at least 20% lower gate p99, at most 3% elongation of MoE, NCCL,
  or indexer kernels, and staging within its explicit budget.
- Exact replay: at least 20% decision-key hit rate in a target workload before
  it becomes a main result.
- Query-tile DAG: no more than 1.25x bulk bytes within the HBM budget and at
  least 20% predicted critical-path reduction.

Failure of a threshold narrows the claim. For example, poor cross-layer recall
retains adaptive physical lowering but removes general index reuse from the
paper claim.

## Evaluation matrix

Evaluate at least two native sparse-attention models. DSv4 must include its 21
real CSA indexers and heterogeneous HCA/SWA roles; another model must use its
actual layer mapping rather than a synthetic all-CSA assumption.

Required workloads include short sanity cases, 480K plus 16K/32K/48K long
prefills, several new prefixes, exact repeats, delayed same-prefix hits, and
concurrent different-prefix requests. Report p50, p95, p99, sample count,
dispersion, and failures.

Metrics include TTFT, goodput, gate stall, indexer calls and GPU time, logical
and physical SSD bytes, IOCBs, read amplification, plan/lowering CPU time,
NVMe time, materialization time, SM interference, staging/HBM peak, and wasted
speculative bytes. A warm repeat alone is not evidence for first-hit benefit.

## OSDI claim and related-work boundary

### Proposed claim

> A correctness-tagged sparse index can be compiled once to jointly eliminate
> redundant index computation and lower the same decision into deadline-bound
> SSD ranges and heterogeneous final KV placements. Separating NVMe completion
> from SM-aware materialization converts storage sparsity into first-hit TTFT
> benefit without making speculative predictions authoritative.

The paper should present one chain, not five unrelated features:

```text
true or guarded shared index
       -> Plan IR
       -> adaptive physical lowering
       -> dual-resource deadline execution
```

The plan cache, delta codec, materializer, and scheduler are necessary parts of
that chain, not independent headline contributions.

### Related-work risks

- **IndexCache** already covers cross-layer top-K reuse. The distinction must
  be that the same index also removes indexer-K/attention-KV SSD bytes and is
  lowered into physical placement and deadlines.
- **ECHO** publicly claims lossless prediction and fused indexer/recall. This
  work should not claim the first prediction mechanism. Its full publication
  requires a new related-work audit; the storage physical-plan claim must
  survive that comparison.
- **Strata** covers hierarchical cache and delay-hit scheduling. The distinction
  is sharing a sparse decision and physical execution, plus improving a new
  prefix's first hit rather than only merging concurrent loads.
- **SolidAttention** includes historical prediction, working-set control, and
  a microtask DAG. P8 has the highest overlap risk. It needs exact native
  learned-indexer execution, heterogeneous layouts, and long-context SSD
  evidence to be distinct.
- **Swarm** uses co-activation for placement. Placement alone is not a main
  contribution here; it is an optional lowering decision driven by the shared
  sparse plan.
- **Tutti** already provides GPU-native object I/O and slack-aware mechanisms.
  Merely wiring the current indexer backend to Tutti is necessary engineering,
  not the paper's novelty.

### Main risks

1. Deep-layer recall may support prefetch but not index skipping.
2. Whole-prefill union saturation may make the oracle choose bulk almost
   everywhere, shrinking the algorithmic contribution.
3. Host materialization overhead may hide compiler improvements until the CQ
   and HMA states are separated.
4. Exact decision-key hit rate may be low for diverse suffixes.
5. A broad mechanism list may obscure the single compute-to-storage compiler
   contribution.

These risks are why P0/P1 offline falsification precedes invasive runtime or
operator changes.
