# SPDX-License-Identifier: Apache-2.0
"""Source-contract tests for the vLLM 0.26.0 LMCache overlay."""

# Standard
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from scripts.apply_vllm_0260_lmcache import (
    _MARKER,
    _balanced_causal_k_bounds,
    _compact_causal_k_union,
    _patch_api_server,
    _patch_cache_utils,
    _patch_connector,
    _patch_deepseek_v2_profile_nvtx,
    _patch_dsv4_model,
    _patch_file,
    _patch_flashmla,
    _patch_kv_cache_admission,
    _patch_scheduler_oversized_prefill,
    _patch_sparse_indexer,
)


def test_api_server_overlay_installs_tool_slack_bootstrap() -> None:
    patched = _patch_api_server('if __name__ == "__main__":\n    pass\n')

    assert "install_vllm_tool_slack_hook()" in patched
    assert patched.index("install_vllm_tool_slack_hook()") < patched.index(
        'if __name__ == "__main__":'
    )


def test_connector_overlay_adds_hma_contract_and_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lmcache_connector.py"
    path.write_text(
        """from base import (
    KVConnectorRole,
)
class LMCacheConnectorV1(KVConnectorBase_V1):
    def request_finished(
        self,
    ): ...
""",
        encoding="utf-8",
    )

    assert _patch_file(path, _patch_connector) is True
    patched = path.read_text(encoding="utf-8")
    assert "KVConnectorBase_V1, SupportsHMA" in patched
    assert "def request_finished_all_groups(" in patched
    assert _patch_file(path, _patch_connector) is False


def test_dsv4_overlay_fires_post_attention_residual_before_ffn() -> None:
    source = """class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__()

        config = vllm_config.model_config.hf_config

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ):
        residual = x

        x = self.attn(positions, x, None)

        x = self.ffn(x, input_ids)
"""

    patched = _patch_dsv4_model(source)

    assert _MARKER in patched
    assert "_DEEPSEEK_V4_DECODER_LAYER_REGISTRY = []" in patched
    assert "_LMCACHE_DSV4_PREFETCH_ACTIVE = False" in patched
    assert "_LMCACHE_CSA_PIPELINE_NVTX_ENABLED" in patched
    assert 'f"event=attention|layer={self.layer_idx}"' in patched
    assert 'f"event=ffn_total|layer={self.layer_idx}"' in patched
    # Decoder attention/FFN use two direct ranges; the profile-only MoE
    # wrapper helper contributes the third range-push call site.
    assert patched.count("torch.cuda.nvtx.range_push(") == 3
    assert patched.count("torch.cuda.nvtx.range_pop()") == 3
    for event in (
        "moe_gate",
        "moe_router",
        "moe_routed_experts",
        "moe_shared_experts",
        "moe_ep_dispatch",
        "moe_ep_combine",
        "moe_ep_final_reduce",
    ):
        assert f'"{event}"' in patched
    assert "_lmcache_install_moe_nvtx" in patched
    assert "_DEEPSEEK_V4_DECODER_LAYER_REGISTRY.append(self)" in patched
    assert 'self.layer_idx = int(prefix.rsplit(".layers.", 1)' in patched
    assert "fire_dsv4_prefetch_from_ffn_boundary" in patched
    assert "if _LMCACHE_DSV4_PREFETCH_ACTIVE:" in patched
    assert "wait_dsv4_attention_kv_from_decoder_boundary(self)" in patched
    assert patched.index("_lmcache_fire_pre_ffn_overlap(residual, positions)") < (
        patched.index("x = self.ffn(x, input_ids)")
    )


def test_deepseek_v2_overlay_installs_profile_only_glm_ranges() -> None:
    source = """import torch

class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, prefix):
        self.self_attn = Attention()
        self.mlp = MLP()
"""

    patched = _patch_deepseek_v2_profile_nvtx(source)

    assert _MARKER in patched
    assert '"LMCACHE_CSA_PIPELINE_NVTX", "0"' in patched
    assert '"attention"' in patched
    assert '"ffn_total"' in patched
    for event in (
        "moe_gate",
        "moe_router",
        "moe_routed_experts",
        "moe_shared_experts",
        "moe_ep_dispatch",
        "moe_ep_combine",
        "moe_ep_final_reduce",
    ):
        assert f'"{event}"' in patched


def _sparse_indexer_source() -> str:
    return """import torch
MXFP4_BLOCK_SIZE = 32

def score():
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            xpu_logits = score_xpu(
                weights[chunk.token_start : chunk.token_end]
            )
            cuda_logits = score_cuda(
                weights[chunk.token_start : chunk.token_end]
            )
            ops.top_k_per_row_prefill(
                logits,
                cu_seqlen_ks,
                cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=chunk.cu_seqlen_ks,
            )

def decode():
        if has_decode:
            _merge_dcp_topk_global(
                decode_logits,
                decode_topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=decode_row_starts,
            )

def forward_cuda():
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
        )
"""


def test_sparse_indexer_overlay_observes_chunks_after_global_merge() -> None:
    source = _sparse_indexer_source()
    patched = _patch_sparse_indexer(source)
    compile(patched, "patched_sparse_attn_indexer.py", "exec")
    tree = ast.parse(patched)
    original_decode = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "decode"
    )
    patched_decode = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "decode"
    )
    assert ast.dump(patched_decode) == ast.dump(original_decode)
    assert not any(
        isinstance(node, ast.Name) and node.id.startswith("compact_k")
        for node in ast.walk(patched_decode)
    )
    prefill = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "score"
    )
    restores = [
        node
        for node in ast.walk(prefill)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_lmcache_publish_k_cp_topk"
    ]
    merges = [
        node
        for node in ast.walk(prefill)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_merge_dcp_topk_global"
    ]
    assert len(restores) == len(merges) == 1
    assert restores[0].lineno < merges[0].lineno
    topk_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "top_k_per_row_prefill"
    ]
    assert len(topk_calls) == 1
    assert isinstance(topk_calls[0].args[-1], ast.Name)
    assert topk_calls[0].args[-1].id == "score_topk_tokens"

    assert "register_lmcache_topk_chunk_callback" in patched
    assert "for chunk_index, chunk in enumerate" in patched
    assert "_lmcache_true_indexer_cp_context(" in patched
    assert '"LMCACHE_PROXY_INDEXER_CP"' in patched
    assert '"LMCACHE_PROXY_INDEXER_K_CP"' in patched
    assert "_lmcache_proxy_indexer_k_cp_context(" in patched
    assert "_lmcache_contiguous_k_bounds(" in patched
    assert "(chunk_rows, score_topk_tokens)" in patched
    assert "topk_tokens + k_cp_world_size - 1" in patched
    assert "gather_prefill_cp_local_k_rows(" in patched
    assert "cu_seqlen_ks = cu_seqlen_ks - compact_k_global_start" in patched
    assert "k_cp_row_offsets = cu_seqlen_ks - original_row_starts" in patched
    assert "int(chunk.block_table.shape[0]) == 1" in patched
    assert "score_topk_tokens," in patched
    assert "proxy_mode=bool(skip_k_cache_insert)" in patched
    assert "q_quant[score_start:score_end]" in patched
    assert patched.count("weights[score_start:score_end]") == 2
    assert "dist.all_gather_into_tensor(" in patched
    assert '"LMCACHE_PROXY_INDEXER_CP_EXCHANGE"' in patched
    assert "complete_topk[:local_rows].copy_" in patched
    assert "row_starts=cu_seqlen_ks" in patched
    assert "event=true_indexer_compute" in patched
    assert "with _lmcache_true_indexer_nvtx_context(self.k_cache.prefix):" in patched
    assert patched.index("_merge_dcp_topk_global(") < patched.index(
        "chunk_callback(topk_indices, chunk_index)"
    )


@pytest.mark.parametrize("rank", [None, 0, 1, 7])
@pytest.mark.parametrize("compact", [False, True])
def test_generated_k_cp_obeys_native_sampler_contract(
    monkeypatch: pytest.MonkeyPatch, rank: int | None, compact: bool
) -> None:
    """Packed native writes and row-relative IDs survive K partitioning.

    Model vLLM ffd46bf csrc/libtorch_stable/sampler.cu: output advances by row*topk
    stride, and each returned index is relative to that query's rowStart.
    Execute the generated prefill loop, not a separate implementation of it.
    """
    tree = ast.parse(_patch_sparse_indexer(_sparse_indexer_source()))
    names = {"score", "_lmcache_contiguous_k_bounds", "_lmcache_publish_k_cp_topk"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    starts = torch.tensor([0, 0, 0, 0] if compact else [0, 1000, 1000, 2000])
    ends = starts + torch.tensor([1, 7, 39, 49])
    complete = torch.full((8, 16), 777, dtype=torch.int32)
    chunk = SimpleNamespace(
        token_start=2,
        token_end=6,
        cu_seqlen_ks=starts,
        cu_seqlen_ke=ends,
        block_table=torch.zeros((1 if compact else 2, 64), dtype=torch.int32),
        local_cu_seq_lens=torch.tensor([0, 2049]),
        max_local_total_seq_lens=2049,
        local_total_seq_lens=2049,
        skip_kv_gather=False,
    )
    gathered: list[torch.Tensor] = []

    def gather_rows(
        *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = args[2]
        assert isinstance(keys, torch.Tensor)
        gathered.append(keys.clone())
        return torch.zeros((len(keys), 4)), torch.zeros((len(keys), 4))

    monkeypatch.setitem(
        sys.modules,
        "lmcache.v1.csa_prefill_cp_scorer",
        SimpleNamespace(gather_prefill_cp_local_k_rows=gather_rows),
    )

    def native_sampler(
        logits: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
        output: torch.Tensor,
        rows: int,
        stride0: int,
        stride1: int,
        width: int,
    ) -> None:
        assert width == (16 if rank is None else 2)
        assert output.is_contiguous(), "native sampler ignores output strides"
        packed = output.as_strided((rows, width), (width, 1))
        packed.fill_(-1)
        for row in range(rows):
            # Strictly increasing scores: choose the last visible K columns.
            length = int(ke[row] - ks[row])
            count = min(length, width)
            packed[row, :count] = torch.arange(length - 1, length - count - 1, -1)

    ns = {
        "torch": torch,
        "prefill_metadata": SimpleNamespace(chunks=[chunk]),
        "_LMCACHE_TOPK_CHUNK_CALLBACKS": {},
        "k_cache_prefix": "test",
        "_lmcache_true_indexer_cp_context": lambda *a, **kw: None,
        "_lmcache_proxy_indexer_k_cp_context": (
            lambda *a: None if rank is None else (rank, 8)
        ),
        "dcp_world_size": 1,
        "dcp_rank": 0,
        "use_pcp": False,
        "skip_k_cache_insert": True,
        "topk_tokens": 16,
        "q_quant": torch.zeros((8, 4)),
        "q_scale": None,
        "weights": torch.zeros((8, 1)),
        "topk_indices_buffer": complete,
        "k_quant_full": torch.zeros((2049, 4)),
        "k_scale_full": torch.zeros((2049, 4)),
        "kv_cache": torch.zeros((64, 64, 8), dtype=torch.uint8),
        "head_dim": 4,
        "use_fp4_cache": False,
        "fp8_dtype": torch.uint8,
        "ops": SimpleNamespace(
            cp_gather_indexer_k_quant_cache=lambda *a: None,
            top_k_per_row_prefill=native_sampler,
        ),
        "score_xpu": lambda *a: None,
        "score_cuda": lambda *a: None,
        "logits": torch.zeros((4, 2049)),
        "num_rows": 4,
        "_merge_dcp_topk_global": lambda *a, **kw: None,
        "cp_kv_cache_interleave_size": 64,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "generated.py", "exec"), ns
    )
    ns["score"]()
    expected = torch.full((4, 16), -1, dtype=torch.int32)
    for row, (start, end) in enumerate(
        zip(starts.tolist(), ends.tolist(), strict=True)
    ):
        shard_start, shard_end = (
            (start, end)
            if rank is None
            else _balanced_causal_k_bounds(start, end, rank, 8)
        )
        count = min(16 if rank is None else 2, shard_end - shard_start)
        expected[row, :count] = torch.arange(
            shard_end - start - 1,
            shard_end - start - count - 1,
            -1,
        )
    torch.testing.assert_close(complete[2:6], expected)
    assert torch.all(complete[:2] == 777) and torch.all(complete[6:] == 777)
    assert bool(gathered) == (compact and rank is not None)


def test_balanced_causal_k_bounds_cover_nondivisible_intervals() -> None:
    """Every causal interval is covered exactly once across prediction ranks."""
    for start, end in ((0, 0), (0, 1), (3, 10), (17, 1042), (0, 131073)):
        shards = [_balanced_causal_k_bounds(start, end, rank, 8) for rank in range(8)]
        assert shards[0][0] == start
        assert shards[-1][1] == end
        assert all(
            left[1] == right[0] for left, right in zip(shards, shards[1:], strict=False)
        )
        assert sum(shard_end - shard_start for shard_start, shard_end in shards) == (
            end - start
        )


def test_balanced_causal_k_bounds_follow_each_query_causal_end() -> None:
    """A later query's append tail is partitioned as part of its total K range."""
    early = [_balanced_causal_k_bounds(0, 1001, rank, 8) for rank in range(8)]
    later = [_balanced_causal_k_bounds(0, 1007, rank, 8) for rank in range(8)]

    assert early[-1][1] == 1001
    assert later[-1][1] == 1007
    assert later != early


def test_compact_causal_union_preserves_every_local_visible_set() -> None:
    """Rebased compact bounds reconstruct exact global K shards per query."""
    starts = (0, 0, 0, 0, 0)
    ends = (1001, 1002, 1003, 1004, 1009)
    for rank in range(8):
        union_start, union_end, local_starts, local_ends = _compact_causal_k_union(
            starts, ends, rank, 8
        )
        assert 0 <= union_start <= union_end
        for start, end, local_start, local_end in zip(
            starts, ends, local_starts, local_ends, strict=True
        ):
            expected = _balanced_causal_k_bounds(start, end, rank, 8)
            assert (local_start + union_start, local_end + union_start) == expected


def test_compact_causal_union_covers_nonzero_and_nondivisible_prefix() -> None:
    """Sequence offsets and append tails retain exact rank-local boundaries."""
    starts = (17, 17, 17)
    ends = (131071, 131072, 131079)
    union_start, union_end, local_starts, local_ends = _compact_causal_k_union(
        starts, ends, rank=5, world_size=8
    )
    assert union_end > union_start
    assert tuple(value + union_start for value in local_starts) == tuple(
        _balanced_causal_k_bounds(start, end, 5, 8)[0]
        for start, end in zip(starts, ends, strict=True)
    )
    assert tuple(value + union_start for value in local_ends) == tuple(
        _balanced_causal_k_bounds(start, end, 5, 8)[1]
        for start, end in zip(starts, ends, strict=True)
    )


def test_flashmla_overlay_remaps_only_compact_csa_rows() -> None:
    source = """from vllm.v1.worker.workspace import current_workspace_manager

def prefill():
        workspace_manager = current_workspace_manager()
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )
            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
            )
"""

    patched = _patch_flashmla(source)

    assert _MARKER in patched
    assert "build_compact_csa_prefill_page_plan" in patched
    assert "chunk_topk_indices = topk_indices[query_start:query_end]" in patched
    assert "combine_topk_swa_indices(\n                chunk_topk_indices," in patched
    assert "page_to_compact=compact_page_to_compact" in patched


def test_cache_utils_fuses_bounded_compact_page_translation() -> None:
    source = """def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    for token_idx in range(0, 1, 1):
        offset = tl.arange(0, PADDED_TOP_K)
        mask = offset < topk_len
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
        )
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + offset,
            topk_indices + M * batch_idx,
            mask=mask,
        )
"""

    patched = _patch_cache_utils(source)

    assert "PAGE_MAP_WIDTH=compact_page_map.shape[1]" in patched
    assert "& (topk_indices >= 0)" in patched
    assert "& (logical_page < PAGE_MAP_WIDTH)" in patched
    assert "other=-1" in patched
    assert "compact_page >= 0" in patched


def test_kv_admission_overlay_is_explicit_and_preserves_zero_memory_error() -> None:
    source = """    # Check if the available memory is enough per worker.
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )
"""

    patched = _patch_kv_cache_admission(source)

    assert _MARKER in patched
    assert "LMCACHE_ALLOW_OVERSIZED_KV_CACHE" in patched
    assert "lmcache_allow_oversized and avail_mem > 0" in patched
    assert patched.count("_check_enough_kv_cache_memory(") == 1


def test_scheduler_serializes_only_opt_in_oversized_prefill_chunks() -> None:
    source = """import itertools
import time

logger = init_logger(__name__)

def schedule(self):
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
            ):
                pass
"""

    patched = _patch_scheduler_oversized_prefill(source)

    assert _MARKER in patched
    assert "LMCACHE_ALLOW_OVERSIZED_KV_CACHE" in patched
    assert "request.num_in_flight_tokens > 0" in patched
    assert patched.index("request.num_in_flight_tokens > 0") < patched.index(
        "request.num_output_placeholders > 0"
    )
