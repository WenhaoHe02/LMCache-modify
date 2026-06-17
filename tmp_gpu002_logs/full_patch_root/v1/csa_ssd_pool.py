# SPDX-License-Identifier: Apache-2.0
"""
csa_ssd_pool.py — SSD-backed HBM pool for CSA Indexer scoring.

After prefill, evicts each CSA layer's kv_cache to SSD (one flat file per layer).
During decode, maintains a small HBM pool of index_topk blocks and replaces
Indexer.forward's full kv_cache einsum with pool-only scoring.

Spec prefetch (MoE-window overlap):
  At the start of Block L's MoE FFN (~3300 µs), fire async NVMe reads for the
  DELTA blocks = (prev_step_topk[L_next] − current_pool[L_next]).
  Adjacent steps share ~91% of topk, so only ~92 blocks/layer need fetching.
  11 MB total delta / 3 GB/s NVMe ≈ 3.7 ms, which fits within the 3300 µs window
  when reads across layers are pipelined by the thread pool.

Flow per decode step, per CSA layer L:
  1. Block L-1 FFN start:
       spec_set = _spec_topk[L]       ← prev step's topk (91% accurate)
       delta    = spec_set − pool[L]  ← blocks not yet in HBM
       issue async reads for delta    ← NVMe reads start
  2. Block L Indexer runs (≥3300 µs later):
       drain completed reads into pool
       pool_score_fn scores pool (replaces full kv_cache einsum)
       miss = true_topk − spec_set    ← ~9% fallback reads
       _spec_topk[L] = true_topk      ← updated for next step
  3. Result: only ~92 fallback reads/layer/step; pool stays warm.

Integration:
    pool_mgr = CSASSDPoolManager.build(transformer, store_dir, compress_ratio,
                                        kv_lora_rank, max_seq_len, io_workers)
    pool_mgr.evict_after_prefill(transformer)   # once after prefill
    pool_mgr.patch_transformer(transformer)     # installs hooks
    pool_mgr.reset()                            # before each decode run
    ...decode loop...
    pool_mgr.print_stats()
    pool_mgr.unpatch_transformer()
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from lmcache.v1.csa_prefetcher import (
    CSABlockStore,
    CSABlockStoreConfig,
)


# ---------------------------------------------------------------------------
# HBM Pool
# ---------------------------------------------------------------------------


class CSAHBMPool:
    """Per-layer HBM pool: [pool_size, head_dim] tensor on CUDA.

    Tracks which global block IDs occupy each pool slot via pool_ids.

    Eviction policy (two-tier LRU):
      Resident slots (marked via protect()) have lower eviction priority than
      ordinary slots.  When space is needed, ordinary LRU is drained first;
      the resident LRU is only touched when no ordinary slots remain.
      Set LMCACHE_DISABLE_RESIDENT_HBM=1 to skip protection (flat LRU).
    """

    _RESIDENT_ENABLED: bool = (
        os.environ.get("LMCACHE_DISABLE_RESIDENT_HBM", "0") != "1"
    )

    def __init__(self, pool_size: int, head_dim: int,
                 device: torch.device, dtype: torch.dtype) -> None:
        self.pool_size = pool_size
        self.pool_tensor = torch.zeros(pool_size, head_dim,
                                       device=device, dtype=dtype)
        self.pool_ids = torch.full((pool_size,), -1,
                                   dtype=torch.int32, device=device)
        # Two-tier LRU: ordinary blocks evicted before resident blocks.
        self._lru_ordinary:  OrderedDict = OrderedDict()
        self._lru_resident:  OrderedDict = OrderedDict()
        self._resident_slots: Set[int] = set()
        self._id_to_slot: Dict[int, int] = {}
        self._free: List[int] = list(range(pool_size))

    def contains(self, block_id: int) -> bool:
        """Return True if block_id is currently in the pool."""
        return block_id in self._id_to_slot

    def ids_in_pool(self) -> Set[int]:
        """Return the set of all block IDs currently held in the pool."""
        return set(self._id_to_slot.keys())

    def protect(self, block_id: int) -> None:
        """Mark block_id as resident (lower eviction priority).

        No-op if LMCACHE_DISABLE_RESIDENT_HBM=1 or block not in pool.

        Args:
            block_id: Global compressed-block index to protect.
        """
        if not self._RESIDENT_ENABLED:
            return
        slot = self._id_to_slot.get(block_id)
        if slot is None or slot in self._resident_slots:
            return
        self._lru_ordinary.pop(slot, None)
        self._lru_resident[slot] = None
        self._resident_slots.add(slot)

    def _evict_one(self) -> int:
        """Evict one slot (ordinary first, resident fallback); return slot."""
        if self._lru_ordinary:
            slot, _ = self._lru_ordinary.popitem(last=False)
        else:
            slot, _ = self._lru_resident.popitem(last=False)
            self._resident_slots.discard(slot)
        old_id = int(self.pool_ids[slot].item())
        if old_id >= 0:
            del self._id_to_slot[old_id]
        return slot

    def insert(self, block_id: int, block_data: torch.Tensor) -> int:
        """Insert or refresh block_data into the pool; return its slot index.

        Args:
            block_id:   Global compressed-block index.
            block_data: 1-D tensor of shape [head_dim] (any device/dtype).

        Returns:
            Pool slot index where the block was placed.
        """
        if block_id in self._id_to_slot:
            slot = self._id_to_slot[block_id]
            if slot in self._resident_slots:
                self._lru_resident.move_to_end(slot)
            else:
                self._lru_ordinary.move_to_end(slot)
            return slot

        if self._free:
            slot = self._free.pop()
        else:
            slot = self._evict_one()

        self.pool_tensor[slot] = block_data.to(
            device=self.pool_tensor.device, dtype=self.pool_tensor.dtype)
        self.pool_ids[slot] = block_id
        self._id_to_slot[block_id] = slot
        self._lru_ordinary[slot] = None  # new blocks start as ordinary
        return slot

    def insert_raw(self, block_id: int, raw: bytes, head_dim: int) -> None:
        """Insert a block from raw bytes (output of an NVMe pread).

        Args:
            block_id: Global compressed-block index.
            raw:      Bytes encoding a bfloat16 [head_dim] vector.
            head_dim: Expected number of elements.
        """
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(head_dim)
        t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
        self.insert(block_id, t)

    def bulk_from_kvcache(self, block_ids: List[int],
                          kv_cache: torch.Tensor) -> None:
        """Seed the pool from a kv_cache tensor (typically just before eviction).

        Args:
            block_ids: List of block IDs to seed.
            kv_cache:  CPU tensor of shape [n_blocks, head_dim].
        """
        for bid in block_ids:
            if bid < kv_cache.size(0):
                self.insert(bid, kv_cache[bid])


# ---------------------------------------------------------------------------
# SSD Pool Manager
# ---------------------------------------------------------------------------


class CSASSDPoolManager:
    """SSD-backed HBM pool manager with MoE-window speculative prefetch.

    Patches Block.forward (to fire delta reads before FFN) and installs
    a _pool_score_fn hook on each CSA Indexer (via the model.py patch) to
    replace the full kv_cache einsum with pool-only scoring.

    Args:
        store:          Backing CSABlockStore instance.
        csa_layer_ids:  Layer IDs that contain CSA Indexer modules.
        index_topk:     Number of blocks selected per Indexer per step.
        pool_size:      HBM pool capacity in blocks (>= index_topk).
    """

    def __init__(
        self,
        store: CSABlockStore,
        csa_layer_ids: List[int],
        index_topk: int = 1024,
        pool_size: int = 2048,
        use_spec: bool = True,
    ) -> None:
        self._store = store
        self._csa_layer_ids: Set[int] = set(csa_layer_ids)
        self._index_topk = index_topk
        self._pool_size = pool_size

        # Per-layer HBM pools (populated by evict_after_prefill)
        self._pools: Dict[int, CSAHBMPool] = {}

        # Async read futures: (layer_id, block_id) → Future[bytes]
        self._pending: Dict[Tuple[int, int], Future] = {}
        self._pending_lock = threading.Lock()

        # Spec prefetch: previous step's topk per layer (91% accurate)
        self._spec_topk: Dict[int, Set[int]] = {}

        self._use_spec = use_spec

        # id(Block module) → layer_id  (robust; avoids relying on layer_id attr)
        self._block_lid_map: Dict[int, int] = {}

        # fire_for_next[lid] = next CSA layer id to prefetch during lid's FFN
        self._fire_for_next: Dict[int, int] = {}

        # Cached list of (name, indexer) for internal lookups
        self._indexer_list: List[Tuple[str, object]] = []

        # Cumulative stats (reset by reset())
        self._hit_total = 0
        self._true_total = 0
        self._fallback_total = 0
        self._step_count = 0

        self._orig_block_forward = None
        self._patched_block_cls = None
        self._patched = False

    # ------------------------------------------------------------------
    # Build from model
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        transformer,
        store_dir: str,
        compress_ratio: int,
        kv_lora_rank: int,
        max_seq_len: int,
        io_workers: int = 8,
        pool_size: int = 2048,
        use_spec: bool = True,
    ) -> "CSASSDPoolManager":
        """Discover CSA layers from transformer and construct a manager.

        Args:
            transformer:    DSV4 Transformer instance.
            store_dir:      Directory for SSD block files.
            compress_ratio: CSA compression ratio (usually 4).
            kv_lora_rank:   KV LoRA rank (fallback for head_dim).
            max_seq_len:    Maximum sequence length.
            io_workers:     Thread pool size for async NVMe reads.
            pool_size:      HBM pool capacity in blocks.

        Returns:
            Configured CSASSDPoolManager (store not yet written).
        """
        csa_lids: List[int] = []
        for name, module in transformer.named_modules():
            if type(module).__name__ == "Indexer":
                parts = name.split(".")
                try:
                    csa_lids.append(int(parts[parts.index("layers") + 1]))
                except (ValueError, IndexError):
                    pass
        csa_lids = sorted(set(csa_lids))

        head_dim = kv_lora_rank  # fallback
        for _, module in transformer.named_modules():
            if type(module).__name__ == "Indexer":
                head_dim = module.head_dim
                compress_ratio = module.compress_ratio
                break

        block_size_bytes = head_dim * 2   # bfloat16
        cfg = CSABlockStoreConfig(
            store_dir=store_dir,
            n_blocks=max_seq_len // compress_ratio,
            block_size_bytes=block_size_bytes,
            io_workers=io_workers,
        )
        store = CSABlockStore(cfg)
        return cls(store, csa_lids, pool_size=pool_size, use_spec=use_spec)

    # ------------------------------------------------------------------
    # Evict prefill kv_cache → SSD + initialise pools
    # ------------------------------------------------------------------

    def evict_after_prefill(
        self,
        transformer,
        last_topk: Optional[Dict[int, Set[int]]] = None,
    ) -> None:
        """Copy each CSA Indexer's kv_cache to SSD and initialise HBM pools.

        Args:
            transformer: The Transformer model.
            last_topk:   Optional per-layer topk sets from prefill's last step,
                         used to seed the pool with the most relevant blocks.
                         If None, seeds with the first index_topk block IDs.
        """
        for name, module in transformer.named_modules():
            if type(module).__name__ != "Indexer":
                continue
            parts = name.split(".")
            try:
                lid = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            if lid not in self._csa_layer_ids:
                continue

            kv = module.kv_cache        # [max_batch, n_comp_blocks, head_dim]
            head_dim = kv.size(2)
            n_blocks = kv.size(1)

            raw_cpu = kv[0, :n_blocks].contiguous().cpu().to(torch.bfloat16)
            self._store.write_blocks_from_tensor(lid, raw_cpu)

            device = kv.device
            pool = CSAHBMPool(self._pool_size, head_dim, device, torch.bfloat16)
            seed_ids = list(last_topk.get(lid, set())) if last_topk else []
            if not seed_ids:
                seed_ids = list(range(min(self._index_topk, n_blocks)))
            pool.bulk_from_kvcache(seed_ids, raw_cpu.to(device))
            for bid in seed_ids:
                pool.protect(bid)
            self._pools[lid] = pool

        print(f"[CSASSDPoolManager] evicted {len(self._pools)} CSA layers to SSD "
              f"at {self._store._cfg.store_dir}")

    # ------------------------------------------------------------------
    # Patch transformer
    # ------------------------------------------------------------------

    def patch_transformer(self, transformer) -> None:
        """Install pool-scoring hooks and the Block.forward MoE-overlap patch.

        Raises:
            RuntimeError: If already patched or Block/Indexer not found.
        """
        if self._patched:
            raise RuntimeError("Already patched.")

        # --- Locate Block class ---
        block_cls = None
        for _, m in transformer.named_modules():
            if type(m).__name__ == "Block":
                block_cls = type(m)
                break
        if block_cls is None:
            raise RuntimeError("Block class not found in transformer.")

        # --- Map id(block) → layer_id using named_modules (no layer_id attr) ---
        lid_order: List[int] = []
        for name, module in transformer.named_modules():
            if type(module).__name__ == "Block":
                parts = name.split(".")
                try:
                    lid = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    lid = -1
                self._block_lid_map[id(module)] = lid
                lid_order.append(lid)

        # Build fire_for_next: for each block, what is the next CSA layer?
        csa_sorted = sorted(self._csa_layer_ids)
        for lid in sorted(set(lid_order)):
            next_csa = next((c for c in csa_sorted if c > lid), None)
            if next_csa is not None:
                self._fire_for_next[lid] = next_csa

        # --- Tag Indexers and install _pool_score_fn ---
        for name, module in transformer.named_modules():
            if type(module).__name__ == "Indexer":
                parts = name.split(".")
                try:
                    lid = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    lid = -1
                module._csa_layer_id = lid
                if lid in self._csa_layer_ids:
                    module._pool_score_fn = self._make_pool_score_fn(lid, module)

        # --- Cache indexer list for fast internal lookup ---
        self._cache_indexer_list(transformer)

        # --- Patch Block.forward for MoE-window spec reads (spec mode only) ---
        self._orig_block_forward = block_cls.forward
        self._patched_block_cls = block_cls

        if not self._use_spec:
            self._patched = True
            return

        mgr = self

        def _patched_block_forward(self_blk, x, start_pos, input_ids):
            lid = mgr._block_lid_map.get(id(self_blk), -1)

            # Attention half
            residual_a = x
            x_a, post_a, comb_a = self_blk.hc_pre(
                x, self_blk.hc_attn_fn,
                self_blk.hc_attn_scale, self_blk.hc_attn_base)
            x_a = self_blk.attn_norm(x_a)
            x_a = self_blk.attn(x_a, start_pos)
            x = self_blk.hc_post(x_a, residual_a, post_a, comb_a)
            residual_f = x

            # FFN pre-processing
            x_f, post_f, comb_f = self_blk.hc_pre(
                x, self_blk.hc_ffn_fn,
                self_blk.hc_ffn_scale, self_blk.hc_ffn_base)
            x_f = self_blk.ffn_norm(x_f)

            # MoE-window overlap: fire delta reads BEFORE ffn() starts.
            # Works for decode (seq_len=1), incremental prefill (seq_len>1),
            # and full-reuse prefill — any step where prev_topk is available.
            if start_pos > 0:
                next_lid = mgr._fire_for_next.get(lid)
                if next_lid is not None:
                    try:
                        mgr._fire_spec_for_next(next_lid)
                    except Exception:
                        pass   # never crash the forward pass

            x = self_blk.ffn(x_f, input_ids)   # ~3300 µs; NVMe reads run here
            x = self_blk.hc_post(x, residual_f, post_f, comb_f)
            return x

        block_cls.forward = _patched_block_forward
        self._patched = True

    def unpatch_transformer(self) -> None:
        """Restore the original Block.forward and remove _pool_score_fn hooks."""
        if not self._patched:
            return
        if self._patched_block_cls is not None:
            self._patched_block_cls.forward = self._orig_block_forward
        self._patched = False

    # ------------------------------------------------------------------
    # Pool scoring function (installed on each CSA Indexer via model.py hook)
    # ------------------------------------------------------------------

    def _make_pool_score_fn(self, layer_id: int, indexer):
        """Return a pool-scoring closure for the given CSA layer.

        The closure is assigned to indexer._pool_score_fn; the model.py hook
        calls it instead of the full kv_cache einsum when start_pos > 0.

        Args:
            layer_id: CSA layer index.
            indexer:  The Indexer module for this layer.

        Returns:
            Callable[[q, weights, bsz, end_pos, offset], topk_global_tensor]
        """
        mgr = self
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        def pool_score_fn(
            q: torch.Tensor,
            weights: torch.Tensor,
            bsz: int,
            end_pos: int,
            offset: int,
        ) -> torch.Tensor:
            pool = mgr._pools.get(layer_id)
            ratio = getattr(indexer, "compress_ratio", 4)
            spec_set = mgr._spec_topk.get(layer_id, set())
            if pool is None:
                # Fallback: full kv_cache scoring (shouldn't happen after evict)
                kv = indexer.kv_cache[:bsz, :end_pos // ratio]
                score = torch.einsum("bshd,btd->bsht", q, kv)
                score = (score.relu_() * weights.unsqueeze(-1)).sum(2)
                if world_size > 1:
                    import torch.distributed as dist
                    dist.all_reduce(score)
                n_topk = min(indexer.index_topk, end_pos // ratio)
                return score.topk(n_topk, dim=-1)[1] + offset

            if not mgr._use_spec:
                # Naive (blocking) path: synchronously pread missing blocks now,
                # blocking the forward pass — identical I/O volume to spec but
                # no FFN-window overlap.
                need = spec_set or set(range(min(indexer.index_topk,
                                               end_pos // ratio)))
                missing_sync = {bid for bid in need if bid >= 0
                                and not pool.contains(bid)}
                head_dim_s = pool.pool_tensor.size(1)
                for bid in missing_sync:
                    raw = mgr._store.read_block_sync(layer_id, bid)
                    pool.insert_raw(bid, raw, head_dim_s)
            else:
                # Spec path: drain reads that were prefetched during prev FFN
                mgr._drain_pending_into_pool(layer_id, pool)

            # Insert the newly compressed block for the current token
            new_slot = (end_pos - 1) // ratio
            kv_size = indexer.kv_cache.size(1)
            if new_slot < kv_size:
                new_block = indexer.kv_cache[0, new_slot].detach()
                pool.insert(new_slot, new_block)
                # Persist to SSD asynchronously
                mgr._store._executor.submit(
                    mgr._store.write_block_sync,
                    layer_id,
                    new_slot,
                    new_block.cpu().to(torch.bfloat16).view(torch.int16).numpy().tobytes(),
                )

            # Pool scoring: q [bsz, 1, n_heads, head_dim] × pool [pool_size, head_dim]
            pt = pool.pool_tensor              # [pool_size, head_dim]
            score = torch.einsum("bshd,td->bsht", q, pt)   # [b,1,n_h,pool_size]
            score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)  # [b,1,pool_size]
            if world_size > 1:
                import torch.distributed as dist
                dist.all_reduce(score)

            n_topk = min(indexer.index_topk, pool.pool_size)
            topk_rel = score.topk(n_topk, dim=-1)[1]    # [b, 1, topk]
            topk_global = pool.pool_ids[topk_rel.reshape(-1)].reshape(topk_rel.shape)

            # Update spec stats and advance the spec prediction for next step
            # Filter out -1 (empty pool slots whose pool_ids == -1)
            true_set = {bid for bid in topk_global.reshape(-1).cpu().tolist()
                        if bid >= 0}
            mgr._hit_total += len(true_set & spec_set)
            mgr._true_total += len(true_set)
            mgr._fallback_total += len(true_set - spec_set)
            mgr._step_count += 1

            # Fallback reads for missed blocks (arrive before next step's scoring)
            miss = true_set - spec_set
            if miss:
                mgr._submit_reads(layer_id, miss)

            # Save true topk for next step's spec (91% overlap → 91% hit rate)
            mgr._spec_topk[layer_id] = true_set

            return topk_global + offset

        return pool_score_fn

    # ------------------------------------------------------------------
    # Spec prefetch: fire delta reads during current block's MoE FFN
    # ------------------------------------------------------------------

    def _fire_spec_for_next(self, next_lid: int) -> None:
        """Submit async reads for the predicted delta blocks for next_lid.

        Uses the previous decode step's topk as the spec prediction.
        Hit rate is ~91% (adjacent-step overlap).

        Args:
            next_lid: Layer ID of the next CSA layer to prefetch for.
        """
        spec_set = self._spec_topk.get(next_lid)
        if not spec_set:
            return
        pool = self._pools.get(next_lid)
        if pool is None:
            return
        delta = spec_set - pool.ids_in_pool()
        if delta:
            self._submit_reads(next_lid, delta)

    def _cache_indexer_list(self, transformer) -> None:
        """Pre-cache all Indexer (name, module) pairs for O(1) lookup."""
        self._indexer_list = [
            (name, m)
            for name, m in transformer.named_modules()
            if type(m).__name__ == "Indexer"
        ]

    # ------------------------------------------------------------------
    # Async I/O helpers
    # ------------------------------------------------------------------

    def _submit_reads(self, layer_id: int, block_ids: Set[int]) -> None:
        """Submit async NVMe reads, deduplicating against already-pending reads.

        Args:
            layer_id:  CSA layer index.
            block_ids: Set of block IDs to fetch.
        """
        with self._pending_lock:
            already = {bid for (lid, bid) in self._pending if lid == layer_id}
        new_blocks = {bid for bid in block_ids if bid >= 0} - already
        if not new_blocks:
            return
        futures = self._store.read_blocks_async(layer_id, new_blocks)
        with self._pending_lock:
            for bid, fut in futures.items():
                self._pending[(layer_id, bid)] = fut

    def _drain_pending_into_pool(self, layer_id: int,
                                  pool: CSAHBMPool) -> None:
        """Collect completed reads and insert blocks into the pool.

        Args:
            layer_id: CSA layer index.
            pool:     HBM pool to receive the blocks.
        """
        with self._pending_lock:
            done_keys = [
                (lid, bid) for (lid, bid) in self._pending
                if lid == layer_id and self._pending[(lid, bid)].done()
            ]
            results = {k: self._pending.pop(k).result() for k in done_keys}
        head_dim = pool.pool_tensor.size(1)
        for (_, bid), raw in results.items():
            pool.insert_raw(bid, raw, head_dim)

    def wait_all(self, timeout: float = 30.0) -> int:
        """Block until all pending reads complete.

        Args:
            timeout: Maximum wait per future in seconds.

        Returns:
            Number of futures collected.
        """
        with self._pending_lock:
            items = list(self._pending.items())
        n = 0
        for (lid, bid), fut in items:
            try:
                raw = fut.result(timeout=timeout)
                pool = self._pools.get(lid)
                if pool is not None:
                    pool.insert_raw(bid, raw, pool.pool_tensor.size(1))
                with self._pending_lock:
                    self._pending.pop((lid, bid), None)
                n += 1
            except Exception:
                pass
        return n

    # ------------------------------------------------------------------
    # Reset / stats
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset per-run state (call before each new decode run)."""
        self._spec_topk.clear()
        with self._pending_lock:
            self._pending.clear()
        self._hit_total = 0
        self._true_total = 0
        self._fallback_total = 0
        self._step_count = 0

    def print_stats(self) -> None:
        """Print spec hit rate and fallback statistics."""
        hr = self._hit_total / self._true_total if self._true_total else 0.0
        fb_mean = self._fallback_total / self._step_count if self._step_count else 0.0
        print(f"\n[CSASSDPoolManager] SSD-pool decode stats")
        print(f"  spec_hit_rate = {hr:.3f}  (prev-topk, expected ~0.91)")
        print(f"  fallback/step = {fb_mean:.1f}  "
              f"(expected ~{self._index_topk * 0.09:.0f})")
        print(f"  total_steps   = {self._step_count}")
