# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM sparse-attention indexer with an optional LMCache chunk hook."""

import os
from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32

# Optional LMCache hook.  Keeping the registry in the vLLM op lets LMCache
# observe a completed true-topK chunk without synchronizing the model thread.
# When no callback is registered (the normal vLLM/OFF path), the only cost is
# one dictionary lookup per prefill chunk.
_LMCACHE_TOPK_CHUNK_CALLBACKS: dict[
    str, Callable[[torch.Tensor, int], None]
] = {}

# Experimental exact context-parallel execution for the *official* prefill
# indexer.  DeepSeek V4 replicates both the Indexer K cache and Indexer weights
# on every TP rank, so query rows can be divided without sharding K.  The
# feature is opt-in while it is being qualified on the production topology.
_LMCACHE_TRUE_INDEXER_CP_WORKSPACES: dict[
    tuple[str, int | None, torch.dtype, int, int],
    tuple[int, torch.Tensor],
] = {}


def _lmcache_true_indexer_cp_context(
    num_rows: int,
) -> tuple[int, int, object] | None:
    """Return the TP rank, size, and process group for exact Indexer CP."""
    enabled = os.getenv("LMCACHE_TRUE_INDEXER_CP", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return None
    try:
        minimum_rows = max(
            1,
            int(os.getenv("LMCACHE_TRUE_INDEXER_CP_MIN_ROWS", "1024")),
        )
        expected_size = max(
            2,
            int(os.getenv("LMCACHE_TRUE_INDEXER_CP_SIZE", "8")),
        )
    except ValueError as exc:
        raise RuntimeError("invalid LMCache true-Indexer CP configuration") from exc
    if int(num_rows) < minimum_rows:
        return None

    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "LMCACHE_TRUE_INDEXER_CP requires initialized torch.distributed"
        )
    tp_group = get_tp_group()
    world_size = int(tp_group.world_size)
    rank = int(tp_group.rank_in_group)
    if world_size != expected_size:
        raise RuntimeError(
            "LMCACHE_TRUE_INDEXER_CP_SIZE does not match the TP group: "
            f"configured={expected_size} runtime={world_size}"
        )
    return rank, world_size, tp_group.device_group


def _lmcache_balanced_row_bounds(
    num_rows: int,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    """Return one contiguous, balanced query-row interval."""
    return (
        int(num_rows) * int(rank) // int(world_size),
        int(num_rows) * (int(rank) + 1) // int(world_size),
    )


def _lmcache_true_indexer_cp_workspace(
    reference: torch.Tensor,
    padded_rows: int,
    topk_tokens: int,
    world_size: int,
) -> torch.Tensor:
    """Return a reusable rank-local send buffer for exact Indexer CP."""
    key = (
        reference.device.type,
        reference.device.index,
        reference.dtype,
        int(topk_tokens),
        int(world_size),
    )
    cached = _LMCACHE_TRUE_INDEXER_CP_WORKSPACES.get(key)
    if cached is None or cached[0] < int(padded_rows):
        capacity = int(padded_rows)
        send = torch.empty(
            (capacity, int(topk_tokens)),
            dtype=reference.dtype,
            device=reference.device,
        )
        cached = (capacity, send)
        _LMCACHE_TRUE_INDEXER_CP_WORKSPACES[key] = cached
    _capacity, send = cached
    return send[: int(padded_rows)]


def register_lmcache_topk_chunk_callback(
    k_cache_prefix: str,
    callback: Callable[[torch.Tensor, int], None],
) -> None:
    """Register an optional observer for completed prefill top-K chunks.

    Args:
        k_cache_prefix: Resolved sparse-indexer cache prefix.
        callback: Function receiving the live top-K slice and zero-based
            chunk index. It must return without synchronizing the CUDA stream.
    """
    _LMCACHE_TOPK_CHUNK_CALLBACKS[str(k_cache_prefix)] = callback


def unregister_lmcache_topk_chunk_callback(k_cache_prefix: str) -> None:
    """Remove the optional observer for ``k_cache_prefix``.

    Args:
        k_cache_prefix: Resolved sparse-indexer cache prefix.
    """
    _LMCACHE_TOPK_CHUNK_CALLBACKS.pop(str(k_cache_prefix), None)


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        chunk_callback = _LMCACHE_TOPK_CHUNK_CALLBACKS.get(str(k_cache_prefix))

        def _score_prefill_rows(
            chunk: Any,
            row_start: int,
            row_end: int,
            output: torch.Tensor,
        ) -> None:
            """Score a contiguous subset of one native prefill chunk."""
            if int(row_start) >= int(row_end):
                return
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[int(row_start) : int(row_end)]
            q_scale_slice = (
                q_scale[int(row_start) : int(row_end)]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            logits = fp8_fp4_mqa_logits(
                (q_slice_cast, q_scale_slice),
                (k_quant_cast, k_scale_cast),
                weights[int(row_start) : int(row_end)],
                chunk.cu_seqlen_ks[
                    int(row_start) - int(chunk.token_start) : int(row_end)
                    - int(chunk.token_start)
                ],
                chunk.cu_seqlen_ke[
                    int(row_start) - int(chunk.token_start) : int(row_end)
                    - int(chunk.token_start)
                ],
                clean_logits=False,
            )
            num_rows = logits.shape[0]
            topk_indices = output[:num_rows, :topk_tokens]

            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_prefill(  # type: ignore[attr-defined]
                    logits,
                    chunk.cu_seqlen_ks[
                        int(row_start) - int(chunk.token_start) : int(row_end)
                        - int(chunk.token_start)
                    ],
                    chunk.cu_seqlen_ke[
                        int(row_start) - int(chunk.token_start) : int(row_end)
                        - int(chunk.token_start)
                    ],
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks[
                        int(row_start) - int(chunk.token_start) : int(row_end)
                        - int(chunk.token_start)
                    ],
                    chunk.cu_seqlen_ke[
                        int(row_start) - int(chunk.token_start) : int(row_end)
                        - int(chunk.token_start)
                    ],
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

        chunks = tuple(prefill_metadata.chunks)
        prefill_start = int(chunks[0].token_start) if chunks else 0
        prefill_end = int(chunks[-1].token_end) if chunks else prefill_start
        for previous, current in zip(chunks, chunks[1:], strict=False):
            if int(previous.token_end) != int(current.token_start):
                raise RuntimeError(
                    "LMCache true-Indexer CP requires contiguous prefill chunks"
                )
        prefill_rows = prefill_end - prefill_start
        cp_context = _lmcache_true_indexer_cp_context(prefill_rows)
        # The exact CP path below overwrites every pure-prefill output row with
        # the all-gather result.  Clearing the complete [Q, topk] buffer first
        # therefore launches a redundant 16 MiB fill at Q=8192 on every layer
        # and rank.  Preserve the official initialization for local execution
        # and mixed prefill/decode batches, where unused rows may remain.
        complete_prefill_output = (
            prefill_start == 0 and prefill_end == int(hidden_states.shape[0])
        )
        if cp_context is None or has_decode or not complete_prefill_output:
            topk_indices_buffer[: hidden_states.shape[0]] = -1
        if cp_context is None:
            for chunk_index, chunk in enumerate(chunks):
                chunk_topk = topk_indices_buffer[
                    int(chunk.token_start) : int(chunk.token_end), :topk_tokens
                ]
                _score_prefill_rows(
                    chunk,
                    int(chunk.token_start),
                    int(chunk.token_end),
                    chunk_topk,
                )
                if chunk_callback is not None:
                    chunk_callback(chunk_topk, chunk_index)
        else:
            import torch.distributed as dist

            cp_rank, cp_world_size, cp_group = cp_context
            logger.info_once(
                "LMCache true-indexer query CP is active: world_size=%d "
                "prefill_rows=%d topk=%d",
                cp_world_size,
                prefill_rows,
                topk_tokens,
            )
            relative_start, relative_end = _lmcache_balanced_row_bounds(
                prefill_rows,
                cp_rank,
                cp_world_size,
            )
            local_start = prefill_start + relative_start
            local_end = prefill_start + relative_end
            padded_rows = (prefill_rows + cp_world_size - 1) // cp_world_size
            local_topk = _lmcache_true_indexer_cp_workspace(
                topk_indices_buffer,
                padded_rows,
                topk_tokens,
                cp_world_size,
            )
            direct_output = topk_indices_buffer[
                prefill_start:prefill_end,
                :topk_tokens,
            ]
            direct_gather = (
                prefill_rows % cp_world_size == 0
                and direct_output.is_contiguous()
            )
            if direct_gather:
                gathered_topk = direct_output
            else:
                gathered_topk = torch.empty(
                    (padded_rows * cp_world_size, topk_tokens),
                    dtype=topk_indices_buffer.dtype,
                    device=topk_indices_buffer.device,
                )
            local_rows = local_end - local_start
            # Balanced partitions need padding only when Q is not divisible by
            # the CP world size.  Scoring overwrites all real local rows.
            if local_rows < padded_rows:
                local_topk[local_rows:].fill_(-1)
            for chunk in chunks:
                overlap_start = max(local_start, int(chunk.token_start))
                overlap_end = min(local_end, int(chunk.token_end))
                if overlap_start >= overlap_end:
                    continue
                output_start = overlap_start - local_start
                output_end = overlap_end - local_start
                _score_prefill_rows(
                    chunk,
                    overlap_start,
                    overlap_end,
                    local_topk[output_start:output_end],
                )

            with torch.cuda.nvtx.range(
                "lmcache.true_indexer_cp.all_gather_topk"
            ):
                dist.all_gather_into_tensor(
                    gathered_topk,
                    local_topk,
                    group=cp_group,
                )
            if not direct_gather:
                for owner in range(cp_world_size):
                    owner_start, owner_end = _lmcache_balanced_row_bounds(
                        prefill_rows,
                        owner,
                        cp_world_size,
                    )
                    owner_rows = owner_end - owner_start
                    source_start = owner * padded_rows
                    topk_indices_buffer[
                        prefill_start + owner_start : prefill_start + owner_end,
                        :topk_tokens,
                    ].copy_(
                        gathered_topk[source_start : source_start + owner_rows]
                    )
            if chunk_callback is not None:
                for chunk_index, chunk in enumerate(chunks):
                    chunk_callback(
                        topk_indices_buffer[
                            int(chunk.token_start) : int(chunk.token_end),
                            :topk_tokens,
                        ],
                        chunk_index,
                    )

    if has_decode:
        if not has_prefill:
            topk_indices_buffer[: hidden_states.shape[0]] = -1
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        logits = fp8_fp4_paged_mqa_logits(
            (padded_q_quant_cast, padded_q_scale),
            kv_cache,
            weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
            clean_logits=False,
        )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_decode(  # type: ignore[attr-defined]
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.skip_k_cache_insert, (
            "AMD platform doesn't support skip cache insert yet"
        )
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            raise RuntimeError(
                "Sparse attention indexer ROCm custom op requires ROCm "
                "Aiter ops to be enabled."
            )
