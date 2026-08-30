# SPDX-License-Identifier: Apache-2.0
"""Apply LMCache's versioned vLLM 0.26.0 integration overlay."""

from __future__ import annotations

# Standard
import argparse
from pathlib import Path
from typing import Final
import re


_MARKER: Final = "LMCACHE_VLLM_0260_OVERLAY"


class PatchError(RuntimeError):
    """Raised when a vLLM source tree does not match the supported layout."""


def _balanced_causal_k_bounds(
    k_start: int,
    k_end: int,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    """Split one causal K interval into contiguous, non-overlapping shards."""
    if k_start < 0 or k_end < k_start:
        raise ValueError("K bounds must satisfy 0 <= start <= end")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid K partition rank or world size")
    span = k_end - k_start
    return (
        k_start + span * rank // world_size,
        k_start + span * (rank + 1) // world_size,
    )


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        raise PatchError(f"vLLM 0.26.0 marker not found for {label}")
    if source.count(old) != 1:
        raise PatchError(f"vLLM 0.26.0 marker is ambiguous for {label}")
    return source.replace(old, new, 1)


def _patch_file(path: Path, transform: object) -> bool:
    source = path.read_text(encoding="utf-8")
    if _MARKER in source:
        return False
    if not callable(transform):
        raise TypeError("transform must be callable")
    patched = transform(source)
    if patched == source:
        raise PatchError(f"overlay made no changes to {path}")
    path.write_text(patched, encoding="utf-8")
    return True


def _patch_connector(source: str) -> str:
    source = _replace_once(
        source,
        "    KVConnectorRole,\n)",
        "    KVConnectorRole,\n    SupportsHMA,\n)",
        "LMCache connector SupportsHMA import",
    )
    source = _replace_once(
        source,
        "class LMCacheConnectorV1(KVConnectorBase_V1):",
        f"# {_MARKER}\nclass LMCacheConnectorV1(KVConnectorBase_V1, SupportsHMA):",
        "LMCache connector HMA inheritance",
    )
    marker = "    def request_finished(\n"
    method = '''    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Notify LMCache using the primary heterogeneous KV group."""
        primary_block_ids = block_ids[0] if block_ids else []
        return self.request_finished(request, primary_block_ids)

'''
    return _replace_once(
        source,
        marker,
        method + marker,
        "LMCache connector HMA request completion",
    )


def _patch_api_server(source: str) -> str:
    marker = 'if __name__ == "__main__":\n'
    hook = f"""# {_MARKER}
from lmcache.integration.vllm.tool_slack_hook import (
    install_vllm_tool_slack_hook,
)

install_vllm_tool_slack_hook()

"""
    return _replace_once(
        source,
        marker,
        hook + marker,
        "OpenAI API server tool-slack bootstrap",
    )


def _patch_deepseek_v2_profile_nvtx(source: str) -> str:
    """Install profile-only GLM decoder and MoE NVTX wrappers.

    GLM-5.2 uses ``DeepseekV2DecoderLayer`` in vLLM 0.26.0.  The connector is
    constructed after the model, so these wrappers must be installed in the
    model module before decoder instances are created.  They are completely
    absent from the call path unless ``LMCACHE_CSA_PIPELINE_NVTX=1``.
    """
    if "class DeepseekV2DecoderLayer(" not in source:
        raise PatchError("vLLM 0.26.0 DeepSeek V2 decoder class not found")
    profile_hooks = f"""

# {_MARKER}: profile-only GLM decoder and MoE stage ranges.
_LMCACHE_GLM_PIPELINE_NVTX_ENABLED = __import__("os").environ.get(
    "LMCACHE_CSA_PIPELINE_NVTX", "0"
).lower() in {{"1", "on", "true", "yes"}}

if _LMCACHE_GLM_PIPELINE_NVTX_ENABLED:
    from functools import wraps as _lmcache_glm_wraps

    def _lmcache_glm_layer_id(owner):
        layer_id = getattr(owner, "layer_idx", None)
        if isinstance(layer_id, int):
            return layer_id
        layer_id = getattr(owner, "_lmcache_profile_layer_idx", None)
        return layer_id if isinstance(layer_id, int) else -1

    def _lmcache_glm_wrap_component(owner, module, event):
        if module is None:
            return
        original = getattr(module, "forward", None)
        if not callable(original) or getattr(original, "_lmcache_glm_nvtx", False):
            return

        @_lmcache_glm_wraps(original)
        def profiled(*args, **kwargs):
            layer_id = _lmcache_glm_layer_id(owner)
            torch.cuda.nvtx.range_push(f"event={{event}}|layer={{layer_id}}")
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        profiled._lmcache_glm_nvtx = True
        module.forward = profiled

    _lmcache_glm_original_init = DeepseekV2DecoderLayer.__init__

    @_lmcache_glm_wraps(_lmcache_glm_original_init)
    def _lmcache_glm_profiled_init(self, *args, **kwargs):
        _lmcache_glm_original_init(self, *args, **kwargs)
        prefix = kwargs.get("prefix")
        if not isinstance(prefix, str):
            prefix = next(
                (
                    value
                    for value in reversed(args)
                    if isinstance(value, str) and ".layers." in value
                ),
                "",
            )
        if ".layers." in prefix:
            try:
                self._lmcache_profile_layer_idx = int(
                    prefix.rsplit(".layers.", 1)[1].split(".", 1)[0]
                )
            except (IndexError, ValueError):
                self._lmcache_profile_layer_idx = -1
        _lmcache_glm_wrap_component(
            self,
            getattr(self, "self_attn", None),
            "attention",
        )
        _lmcache_glm_wrap_component(
            self,
            getattr(self, "mlp", None),
            "ffn_total",
        )

    DeepseekV2DecoderLayer.__init__ = _lmcache_glm_profiled_init

    try:
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts as _LMCacheGLMRoutedExperts,
        )
        from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
            FusedMoERouter as _LMCacheGLMFusedMoERouter,
        )
        from vllm.model_executor.layers.fused_moe.router.gate_linear import (
            GateLinear as _LMCacheGLMGateLinear,
        )
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
            MoERunner as _LMCacheGLMMoERunner,
        )
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExperts as _LMCacheGLMSharedExperts,
        )
    except ImportError:
        pass
    else:
        def _lmcache_glm_install_moe_nvtx(cls, method_name, event):
            original = getattr(cls, method_name)
            if getattr(original, "_lmcache_moe_nvtx", False):
                return

            @_lmcache_glm_wraps(original)
            def profiled(self, *args, **kwargs):
                layer_name = getattr(self, "layer_name", type(self).__name__)
                torch.cuda.nvtx.range_push(
                    f"event={{event}}|layer_name={{layer_name}}|method={{method_name}}"
                )
                try:
                    return original(self, *args, **kwargs)
                finally:
                    torch.cuda.nvtx.range_pop()

            profiled._lmcache_moe_nvtx = True
            setattr(cls, method_name, profiled)

        _lmcache_glm_install_moe_nvtx(
            _LMCacheGLMGateLinear, "forward", "moe_gate"
        )
        _lmcache_glm_install_moe_nvtx(
            _LMCacheGLMFusedMoERouter, "select_experts", "moe_router"
        )
        _lmcache_glm_install_moe_nvtx(
            _LMCacheGLMRoutedExperts, "forward_modular", "moe_routed_experts"
        )
        _lmcache_glm_install_moe_nvtx(
            _LMCacheGLMRoutedExperts, "forward_monolithic", "moe_routed_experts"
        )
        _lmcache_glm_install_moe_nvtx(
            _LMCacheGLMSharedExperts, "forward", "moe_shared_experts"
        )
        _lmcache_glm_install_moe_nvtx(
            _LMCacheGLMMoERunner, "_maybe_dispatch", "moe_ep_dispatch"
        )
        _lmcache_glm_install_moe_nvtx(
            _LMCacheGLMMoERunner, "_maybe_combine", "moe_ep_combine"
        )
        _lmcache_glm_install_moe_nvtx(
            _LMCacheGLMMoERunner,
            "_maybe_reduce_final_output",
            "moe_ep_final_reduce",
        )
"""
    return source + profile_hooks


def _patch_dsv4_model(source: str) -> str:
    class_marker = "class DeepseekV4DecoderLayer("
    class_start = source.find(class_marker)
    if class_start < 0:
        raise PatchError("vLLM 0.26.0 DeepSeek V4 decoder class not found")
    init_marker = (
        "        super().__init__()\n\n"
        "        config = vllm_config.model_config.hf_config\n"
    )
    init_hook = """        super().__init__()

        # LMCache's connector is constructed after the vLLM model, so its
        # registry hook cannot observe decoder construction on vLLM 0.26.0.
        # Preserve the layer id on the live module for the lazy GC discovery
        # path used when the Tutti loader becomes ready.
        self.layer_idx = int(prefix.rsplit(".layers.", 1)[1].split(".", 1)[0])
        _DEEPSEEK_V4_DECODER_LAYER_REGISTRY.append(self)

        config = vllm_config.model_config.hf_config
"""
    forward_marker = "    def forward(\n        self,\n        x: torch.Tensor,\n"
    forward_start = source.find(forward_marker, class_start)
    if forward_start < 0:
        raise PatchError("vLLM 0.26.0 DeepSeek V4 decoder forward not found")
    decoder_init = _replace_once(
        source[class_start:forward_start],
        init_marker,
        init_hook,
        "DeepSeek V4 decoder layer id",
    )
    source = source[:class_start] + decoder_init + source[forward_start:]
    registry = """\
# Decoder instances are created before LMCache's worker connector on vLLM
# 0.26.0. Keep a module-owned strong registry so the deferred Tutti attach can
# deterministically recover the live layer objects without a GC-order guess.
_DEEPSEEK_V4_DECODER_LAYER_REGISTRY = []
_LMCACHE_DSV4_PREFETCH_ACTIVE = False
_LMCACHE_CSA_PIPELINE_NVTX_ENABLED = __import__("os").environ.get(
    "LMCACHE_CSA_PIPELINE_NVTX", "0"
).lower() in {"1", "on", "true", "yes"}


class DeepseekV4DecoderLayer(nn.Module):
"""
    source = _replace_once(
        source,
        "class DeepseekV4DecoderLayer(nn.Module):\n",
        registry,
        "DeepSeek V4 decoder registry",
    )
    method = f'''    # {_MARKER}
    def _lmcache_fire_pre_ffn_overlap(
        self,
        residual_after_attention: torch.Tensor,
        positions: torch.Tensor,
    ) -> bool:
        """Fire LMCache work after attention and before the FFN."""
        from lmcache.integration.vllm.vllm_v1_adapter import (
            fire_dsv4_prefetch_from_ffn_boundary,
        )

        return fire_dsv4_prefetch_from_ffn_boundary(
            self,
            residual_after_attention,
            positions,
        )

'''
    source = _replace_once(
        source,
        forward_marker,
        method + forward_marker,
        "DeepSeek V4 native pre-FFN method",
    )
    ffn_marker = "\n        x = self.ffn(x, input_ids)\n"
    ffn_hook = """
        # This is the earliest point at which the post-attention residual is
        # available and the MoE has not started. Decode bypasses the Python
        # dispatcher entirely; the connector toggles this module-owned flag
        # once per model forward.
        if _LMCACHE_DSV4_PREFETCH_ACTIVE:
            self._lmcache_pre_ffn_overlap_fired = False
            self._lmcache_python_pre_ffn_overlap_fired = False
            self._lmcache_fire_pre_ffn_overlap(residual, positions)

        if _LMCACHE_CSA_PIPELINE_NVTX_ENABLED:
            torch.cuda.nvtx.range_push(
                f"event=ffn_total|layer={self.layer_idx}"
            )
            try:
                x = self.ffn(x, input_ids)
            finally:
                torch.cuda.nvtx.range_pop()
        else:
            x = self.ffn(x, input_ids)
"""
    source = _replace_once(
        source,
        ffn_marker,
        ffn_hook,
        "DeepSeek V4 pre-FFN call",
    )
    attention_marker = "\n        x = self.attn(positions, x, None)\n"
    attention_gate = """
        if _LMCACHE_DSV4_PREFETCH_ACTIVE:
            from lmcache.integration.vllm.vllm_v1_adapter import (
                wait_dsv4_attention_kv_from_decoder_boundary,
            )

            wait_dsv4_attention_kv_from_decoder_boundary(self)

        if _LMCACHE_CSA_PIPELINE_NVTX_ENABLED:
            torch.cuda.nvtx.range_push(
                f"event=attention|layer={self.layer_idx}"
            )
            try:
                x = self.attn(positions, x, None)
            finally:
                torch.cuda.nvtx.range_pop()
        else:
            x = self.attn(positions, x, None)
"""
    source = _replace_once(
        source,
        attention_marker,
        attention_gate,
        "DeepSeek V4 pre-attention HCA gate",
    )
    moe_profile_hooks = """

# Profile-only DeepSeek-V4 MoE stage hooks. Installing wrappers only when the
# NVTX switch is enabled keeps the production call path byte-for-byte original.
if _LMCACHE_CSA_PIPELINE_NVTX_ENABLED:
    from functools import wraps as _lmcache_wraps

    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts as _LMCacheRoutedExperts,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter as _LMCacheFusedMoERouter,
    )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import (
        GateLinear as _LMCacheGateLinear,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
        MoERunner as _LMCacheMoERunner,
    )
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts as _LMCacheSharedExperts,
    )

    def _lmcache_install_moe_nvtx(cls, method_name, event):
        original = getattr(cls, method_name)
        if getattr(original, "_lmcache_moe_nvtx", False):
            return

        @_lmcache_wraps(original)
        def profiled(self, *args, **kwargs):
            layer_name = getattr(self, "layer_name", type(self).__name__)
            torch.cuda.nvtx.range_push(
                f"event={event}|layer_name={layer_name}|method={method_name}"
            )
            try:
                return original(self, *args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        profiled._lmcache_moe_nvtx = True
        setattr(cls, method_name, profiled)

    _lmcache_install_moe_nvtx(_LMCacheGateLinear, "forward", "moe_gate")
    _lmcache_install_moe_nvtx(
        _LMCacheFusedMoERouter,
        "select_experts",
        "moe_router",
    )
    _lmcache_install_moe_nvtx(
        _LMCacheRoutedExperts,
        "forward_modular",
        "moe_routed_experts",
    )
    _lmcache_install_moe_nvtx(
        _LMCacheRoutedExperts,
        "forward_monolithic",
        "moe_routed_experts",
    )
    _lmcache_install_moe_nvtx(
        _LMCacheSharedExperts,
        "forward",
        "moe_shared_experts",
    )
    _lmcache_install_moe_nvtx(
        _LMCacheMoERunner,
        "_maybe_dispatch",
        "moe_ep_dispatch",
    )
    _lmcache_install_moe_nvtx(
        _LMCacheMoERunner,
        "_maybe_combine",
        "moe_ep_combine",
    )
    _lmcache_install_moe_nvtx(
        _LMCacheMoERunner,
        "_maybe_reduce_final_output",
        "moe_ep_final_reduce",
    )
"""
    return source + moe_profile_hooks


def _patch_sparse_indexer(source: str) -> str:
    source = _replace_once(
        source,
        "import torch\n",
        "import os\nimport re\n\nfrom collections.abc import Callable\n"
        "from contextlib import nullcontext\n\nimport torch\n",
        "sparse indexer callback import",
    )
    registry_marker = "MXFP4_BLOCK_SIZE = 32\n\n"
    registry = f'''MXFP4_BLOCK_SIZE = 32

# {_MARKER}
_LMCACHE_TOPK_CHUNK_CALLBACKS: dict[
    str, Callable[[torch.Tensor, int], None]
] = {{}}


def register_lmcache_topk_chunk_callback(
    k_cache_prefix: str,
    callback: Callable[[torch.Tensor, int], None],
) -> None:
    """Register an observer for one completed true-topK prefill chunk."""
    _LMCACHE_TOPK_CHUNK_CALLBACKS[str(k_cache_prefix)] = callback


def unregister_lmcache_topk_chunk_callback(k_cache_prefix: str) -> None:
    """Remove a previously registered LMCache chunk observer."""
    _LMCACHE_TOPK_CHUNK_CALLBACKS.pop(str(k_cache_prefix), None)


_LMCACHE_TRUE_INDEXER_CP_WORKSPACES: dict[
    tuple[str, int | None, torch.dtype, int, int],
    tuple[int, torch.Tensor],
] = {{}}


def _lmcache_true_indexer_nvtx_context(k_cache_prefix: object):
    """Return the opt-in detailed true-indexer NVTX context."""
    enabled = os.getenv("LMCACHE_CSA_DETAILED_IO_NVTX", "0").lower() in {{
        "1",
        "true",
        "yes",
        "on",
    }}
    if not enabled:
        return nullcontext()
    match = re.search(
        r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)",
        str(k_cache_prefix),
    )
    layer_id = int(match.group(1)) if match is not None else -1
    return torch.cuda.nvtx.range(
        f"event=true_indexer_compute|layer={{layer_id}}|target={{layer_id}}|"
        "kind=dsa_indexer"
    )


def _lmcache_true_indexer_cp_context(
    num_rows: int,
    dcp_world_size: int,
    use_pcp: bool,
    *,
    proxy_mode: bool = False,
) -> tuple[int, int, object] | None:
    """Return TP rank, size, and process group for exact query-row CP."""
    prefix = (
        "LMCACHE_PROXY_INDEXER_CP"
        if proxy_mode
        else "LMCACHE_TRUE_INDEXER_CP"
    )
    enabled = os.getenv(prefix, "0").lower() in {{
        "1",
        "true",
        "yes",
        "on",
    }}
    if not enabled:
        return None
    if proxy_mode and os.getenv(
        "LMCACHE_PROXY_INDEXER_K_CP", "0"
    ).lower() in {{"1", "true", "yes", "on"}}:
        # Prediction-only K-CP owns all query rows; query-CP would otherwise
        # discard seven eighths of them before the K partition is applied.
        return None
    try:
        minimum_rows = max(
            1,
            int(os.getenv(f"{{prefix}}_MIN_ROWS", "1024")),
        )
        expected_size = max(
            2,
            int(os.getenv(f"{{prefix}}_SIZE", "8")),
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid {{prefix}} configuration") from exc
    if int(num_rows) < minimum_rows:
        return None
    if int(dcp_world_size) != 1 or bool(use_pcp):
        raise RuntimeError(
            f"{{prefix}} requires vLLM DCP=1 and PCP disabled"
        )

    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            f"{{prefix}} requires initialized torch.distributed"
        )
    tp_group = get_tp_group()
    world_size = int(tp_group.world_size)
    rank = int(tp_group.rank_in_group)
    if world_size != expected_size:
        raise RuntimeError(
            f"{{prefix}}_SIZE does not match the TP group: "
            f"configured={{expected_size}} runtime={{world_size}}"
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


def _lmcache_proxy_indexer_k_cp_context(
    skip_k_cache_insert: bool,
    dcp_world_size: int,
    use_pcp: bool,
) -> tuple[int, int] | None:
    """Return TP rank/size for prediction-only contiguous K partitioning."""
    enabled = os.getenv("LMCACHE_PROXY_INDEXER_K_CP", "0").lower() in {{
        "1",
        "true",
        "yes",
        "on",
    }}
    if not bool(skip_k_cache_insert) or not enabled:
        return None
    if int(dcp_world_size) != 1 or bool(use_pcp):
        raise RuntimeError(
            "LMCACHE_PROXY_INDEXER_K_CP requires vLLM DCP=1 and PCP disabled"
        )
    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "LMCACHE_PROXY_INDEXER_K_CP requires initialized torch.distributed"
        )
    tp_group = get_tp_group()
    world_size = int(tp_group.world_size)
    rank = int(tp_group.rank_in_group)
    expected_size = max(
        2,
        int(os.getenv("LMCACHE_PROXY_INDEXER_K_CP_SIZE", "8")),
    )
    if world_size != expected_size:
        raise RuntimeError(
            "LMCACHE_PROXY_INDEXER_K_CP_SIZE does not match the TP group: "
            f"configured={{expected_size}} runtime={{world_size}}"
        )
    return rank, world_size


def _lmcache_contiguous_k_bounds(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split every row's complete causal K interval across TP ranks."""
    spans = row_ends - row_starts
    local_starts = row_starts + torch.div(
        spans * int(rank), int(world_size), rounding_mode="floor"
    )
    local_ends = row_starts + torch.div(
        spans * (int(rank) + 1), int(world_size), rounding_mode="floor"
    )
    return local_starts, local_ends


def _lmcache_true_indexer_cp_workspace(
    reference: torch.Tensor,
    padded_rows: int,
    topk_tokens: int,
    world_size: int,
) -> torch.Tensor:
    """Return a reusable rank-local all-gather send buffer."""
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

'''
    source = _replace_once(
        source,
        registry_marker,
        registry,
        "sparse indexer callback registry",
    )
    loop_marker = "        for chunk in prefill_metadata.chunks:\n"
    source = _replace_once(
        source,
        loop_marker,
        "        chunk_callback = _LMCACHE_TOPK_CHUNK_CALLBACKS.get(\n"
        "            str(k_cache_prefix)\n"
        "        )\n"
        "        for chunk_index, chunk in enumerate(prefill_metadata.chunks):\n",
        "sparse indexer chunk enumeration",
    )
    chunk_header = """\
        for chunk_index, chunk in enumerate(prefill_metadata.chunks):
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
"""
    cp_chunk_header = """\
        for chunk_index, chunk in enumerate(prefill_metadata.chunks):
            chunk_start = int(chunk.token_start)
            chunk_end = int(chunk.token_end)
            chunk_rows = chunk_end - chunk_start
            cp_context = _lmcache_true_indexer_cp_context(
                chunk_rows,
                dcp_world_size,
                use_pcp,
                proxy_mode=bool(skip_k_cache_insert),
            )
            k_cp_context = _lmcache_proxy_indexer_k_cp_context(
                bool(skip_k_cache_insert),
                dcp_world_size,
                use_pcp,
            )
            if cp_context is None:
                score_start = chunk_start
                score_end = chunk_end
            else:
                cp_rank, cp_world_size, _cp_group = cp_context
                relative_start, relative_end = _lmcache_balanced_row_bounds(
                    chunk_rows,
                    cp_rank,
                    cp_world_size,
                )
                score_start = chunk_start + relative_start
                score_end = chunk_start + relative_end
            relative_start = score_start - chunk_start
            relative_end = score_end - chunk_start
            cu_seqlen_ks = chunk.cu_seqlen_ks[relative_start:relative_end]
            cu_seqlen_ke = chunk.cu_seqlen_ke[relative_start:relative_end]
            score_topk_tokens = topk_tokens
            if k_cp_context is not None:
                k_cp_rank, k_cp_world_size = k_cp_context
                cu_seqlen_ks, cu_seqlen_ke = _lmcache_contiguous_k_bounds(
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    k_cp_rank,
                    k_cp_world_size,
                )
                score_topk_tokens = (
                    topk_tokens + k_cp_world_size - 1
                ) // k_cp_world_size
"""
    source = _replace_once(
        source,
        chunk_header,
        cp_chunk_header,
        "sparse indexer query CP chunk bounds",
    )
    query_output = """            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
"""
    cp_query_output = """            q_slice = q_quant[score_start:score_end]
            q_scale_slice = (
                q_scale[score_start:score_end]
                if q_scale is not None
                else None
            )
            complete_topk = topk_indices_buffer[
                chunk_start:chunk_end, :topk_tokens
            ]
            if k_cp_context is not None:
                complete_topk.fill_(-1)
                topk_indices = complete_topk[:, :score_topk_tokens]
            elif cp_context is None:
                topk_indices = complete_topk
            else:
                padded_rows = (chunk_rows + cp_world_size - 1) // cp_world_size
                topk_indices = _lmcache_true_indexer_cp_workspace(
                    complete_topk,
                    padded_rows,
                    topk_tokens,
                    cp_world_size,
                )
                local_rows = score_end - score_start
                if local_rows < padded_rows:
                    topk_indices[local_rows:].fill_(-1)
"""
    source = _replace_once(
        source,
        query_output,
        cp_query_output,
        "sparse indexer query CP slices",
    )
    weight_slice = "weights[chunk.token_start : chunk.token_end]"
    if source.count(weight_slice) != 2:
        raise PatchError(
            "vLLM 0.26.0 marker count changed for sparse indexer weight slices"
        )
    source = source.replace(weight_slice, "weights[score_start:score_end]", 2)
    source, replacements = re.subn(
        r"(ops\.top_k_per_row_prefill\(.*?logits\.stride\(1\),\s*)"
        r"topk_tokens,",
        r"\1score_topk_tokens,",
        source,
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise PatchError(
            "vLLM 0.26.0 marker not found for sparse indexer "
            "prediction K-CP top-k width"
        )
    merge_marker = """            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=chunk.cu_seqlen_ks,
            )
"""
    cp_merge = """            _merge_dcp_topk_global(
                logits,
                topk_indices,
                score_topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=cu_seqlen_ks,
            )
            if cp_context is not None:
                exchange_proxy_rows = os.getenv(
                    "LMCACHE_PROXY_INDEXER_CP_EXCHANGE",
                    "1",
                ).lower() in {"1", "true", "yes", "on"}
                if bool(skip_k_cache_insert) and not exchange_proxy_rows:
                    # Prediction consumes only the per-rank candidate union,
                    # so retain local query rows and avoid the exact top-K
                    # AllGather. The physical owner-gather exchanges the
                    # selected KV rows later at the aligned consumer gate.
                    complete_topk[:local_rows].copy_(topk_indices[:local_rows])
                    complete_topk[local_rows:].fill_(-1)
                    topk_indices = complete_topk[:local_rows]
                else:
                    import torch.distributed as dist

                    direct_gather = (
                        chunk_rows % cp_world_size == 0
                        and complete_topk.is_contiguous()
                    )
                    if direct_gather:
                        gathered_topk = complete_topk
                    else:
                        gathered_topk = torch.empty(
                            (padded_rows * cp_world_size, topk_tokens),
                            dtype=complete_topk.dtype,
                            device=complete_topk.device,
                        )
                    logger.info_once(
                        "LMCache indexer query CP is active: world_size=%d "
                        "rows=%d topk=%d proxy=%d",
                        cp_world_size,
                        chunk_rows,
                        topk_tokens,
                        int(bool(skip_k_cache_insert)),
                    )
                    with torch.cuda.nvtx.range(
                        "lmcache.indexer_cp.all_gather_topk"
                    ):
                        dist.all_gather_into_tensor(
                            gathered_topk,
                            topk_indices,
                            group=_cp_group,
                        )
                    if not direct_gather:
                        for owner in range(cp_world_size):
                            owner_start, owner_end = _lmcache_balanced_row_bounds(
                                chunk_rows,
                                owner,
                                cp_world_size,
                            )
                            owner_rows = owner_end - owner_start
                            source_start = owner * padded_rows
                            complete_topk[owner_start:owner_end].copy_(
                                gathered_topk[
                                    source_start : source_start + owner_rows
                                ]
                            )
                    topk_indices = complete_topk
"""
    source = _replace_once(
        source,
        merge_marker,
        cp_merge,
        "sparse indexer exact query CP gather",
    )
    source = _replace_once(
        source,
        cp_merge,
        cp_merge
        + "            if chunk_callback is not None:\n"
        + "                chunk_callback(topk_indices, chunk_index)\n",
        "sparse indexer completed-chunk callback",
    )
    return _replace_once(
        source,
        "        return torch.ops.vllm.sparse_attn_indexer(\n",
        "        with _lmcache_true_indexer_nvtx_context(self.k_cache.prefix):\n"
        "            return torch.ops.vllm.sparse_attn_indexer(\n",
        "sparse indexer detailed true-indexer range",
    )


def _patch_cache_utils(source: str) -> str:
    """Fuse compact-page translation into the existing sparse-index combiner."""
    signature = """def combine_topk_swa_indices(
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
"""
    source = _replace_once(
        source,
        signature,
        """def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    page_to_compact: torch.Tensor | None = None,
    compact_block_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
""",
        "compact-page combiner signature",
    )
    launch_marker = """    NUM_WORKERS = 128
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
"""
    source = _replace_once(
        source,
        launch_marker,
        """    use_compact_page_map = page_to_compact is not None
    if use_compact_page_map:
        assert page_to_compact is not None
        if page_to_compact.dtype != torch.int32 or page_to_compact.ndim != 2:
            raise ValueError("page_to_compact must be a two-dimensional int32 tensor")
        if int(page_to_compact.shape[0]) != int(num_reqs):
            raise ValueError("page_to_compact request count does not match seq_lens")
        if compact_block_size <= 0:
            raise ValueError("compact_block_size must be positive")
        compact_page_map = page_to_compact
    else:
        compact_page_map = topk_indices

    NUM_WORKERS = 128
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        compact_page_map,
        compact_page_map.stride(0),
        query_start_loc,
""",
        "compact-page combiner launch",
    )
    source = _replace_once(
        source,
        """        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
""",
        """        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
        PAGE_MAP_WIDTH=compact_page_map.shape[1],
        COMPACT_BLOCK_SIZE=compact_block_size,
        USE_COMPACT_PAGE_MAP=use_compact_page_map,
    )
""",
        "compact-page combiner constexprs",
    )
    source = _replace_once(
        source,
        """    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
""",
        """    topk_indices_ptr,
    topk_indices_stride,
    page_to_compact_ptr,
    page_to_compact_stride,
    query_start_loc_ptr,
""",
        "compact-page combiner kernel arguments",
    )
    source = _replace_once(
        source,
        """    PADDED_TOP_K: tl.constexpr,
):
""",
        """    PADDED_TOP_K: tl.constexpr,
    PAGE_MAP_WIDTH: tl.constexpr,
    COMPACT_BLOCK_SIZE: tl.constexpr,
    USE_COMPACT_PAGE_MAP: tl.constexpr,
):
""",
        "compact-page combiner kernel constexprs",
    )
    return _replace_once(
        source,
        """        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
        )
        tl.store(
""",
        """        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
        )
        if USE_COMPACT_PAGE_MAP:
            logical_page = topk_indices // COMPACT_BLOCK_SIZE
            page_offset = topk_indices % COMPACT_BLOCK_SIZE
            page_mask = (
                mask
                & (topk_indices >= 0)
                & (logical_page < PAGE_MAP_WIDTH)
            )
            compact_page = tl.load(
                page_to_compact_ptr
                + batch_idx * page_to_compact_stride
                + logical_page,
                mask=page_mask,
                other=-1,
            )
            topk_indices = tl.where(
                compact_page >= 0,
                compact_page * COMPACT_BLOCK_SIZE + page_offset,
                -1,
            )
            topk_indices = compact_page * COMPACT_BLOCK_SIZE + page_offset
        tl.store(
""",
        "compact-page translation in sparse-index consumer",
    )


def _patch_flashmla(source: str) -> str:
    import_marker = """from vllm.v1.worker.workspace import current_workspace_manager

"""
    helpers = f'''from vllm.v1.worker.workspace import current_workspace_manager

from lmcache.integration.vllm.dsv4_compact_prefill import (
    build_compact_csa_prefill_page_plan,
)

# {_MARKER}
def _lmcache_compact_csa_gather_enabled(
    compressed_k_cache: torch.Tensor | None,
) -> bool:
    """Return whether LMCache published exact selected CSA pages."""
    if compressed_k_cache is None:
        return False
    try:
        from lmcache.v1.csa_attention_kv_prefetch_manager import (
            get_csa_attention_kv_prefetch_manager,
        )

        manager = get_csa_attention_kv_prefetch_manager()
        return bool(
            manager is not None
            and manager.owns_k_cache(compressed_k_cache)
            and manager.true_selected_blocks_for_cache(compressed_k_cache) is not None
        )
    except (ImportError, AttributeError):
        return False


def _lmcache_compact_csa_gather_plan(
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    query_row_offsets: torch.Tensor,
    block_size: int,
    compressed_k_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build a compact gather plan from LMCache's correction bitmap."""
    selected_page_bitmap = None
    cached_prefix_fully_selected = False
    try:
        from lmcache.v1.csa_attention_kv_prefetch_manager import (
            get_csa_attention_kv_prefetch_manager,
        )

        manager = get_csa_attention_kv_prefetch_manager()
        if manager is not None:
            selected_page_bitmap = manager.true_selected_blocks_for_cache(
                compressed_k_cache
            )
            covers_prefix = getattr(
                manager,
                "true_selected_covers_cached_prefix_for_cache",
                None,
            )
            if callable(covers_prefix):
                cached_prefix_fully_selected = bool(
                    covers_prefix(compressed_k_cache)
                )
    except (ImportError, AttributeError):
        selected_page_bitmap = None
    if cached_prefix_fully_selected:
        return block_table, compressed_seq_lens, None
    if selected_page_bitmap is None:
        raise RuntimeError("LMCache compact gather has no selected-page bitmap")
    return build_compact_csa_prefill_page_plan(
        topk_indices,
        block_table,
        compressed_seq_lens,
        query_row_offsets,
        block_size,
        selected_page_bitmap,
    )

'''
    source = _replace_once(
        source,
        import_marker,
        helpers,
        "FlashMLA compact-gather helpers",
    )
    loop_marker = """        workspace_manager = current_workspace_manager()
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
"""
    source = _replace_once(
        source,
        loop_marker,
        """        workspace_manager = current_workspace_manager()
        compact_csa_gather = (
            not swa_only
            and self.compress_ratio == 4
            and _lmcache_compact_csa_gather_enabled(compressed_k_cache)
        )
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
""",
        "FlashMLA compact-gather mode",
    )
    gather_marker = """            if not swa_only:
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
"""
    gather_replacement = """            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            chunk_topk_indices = topk_indices[query_start:query_end]
            compact_page_to_compact = None

            if not swa_only:
                # Gather compressed KV. On cache hits LMCache can replace the
                # full table with the exact true-topK page union.
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                gather_block_table = block_table[chunk_start:chunk_end]
                gather_seq_lens = (
                    seq_lens[chunk_start:chunk_end] // self.compress_ratio
                )
                if compact_csa_gather:
                    query_row_offsets = query_start_loc[
                        num_decodes + chunk_start : num_decodes + chunk_end + 1
                    ]
                    compact_device = chunk_topk_indices.device
                    gather_block_table = gather_block_table.to(
                        device=compact_device,
                        non_blocking=True,
                    )
                    gather_seq_lens = gather_seq_lens.to(
                        device=compact_device,
                        non_blocking=True,
                    )
                    query_row_offsets = query_row_offsets.to(
                        device=compact_device,
                        non_blocking=True,
                    )
                    (
                        gather_block_table,
                        gather_seq_lens,
                        compact_page_to_compact,
                    ) = _lmcache_compact_csa_gather_plan(
                        chunk_topk_indices,
                        gather_block_table,
                        gather_seq_lens,
                        query_row_offsets,
                        attn_metadata.block_size // self.compress_ratio,
                        compressed_k_cache,
                    )
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=gather_seq_lens,
                    gather_lens=None,
                    block_table=gather_block_table,
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )
"""
    source = _replace_once(
        source,
        gather_marker,
        gather_replacement,
        "FlashMLA compact gather",
    )
    old_query = """\
            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
"""
    new_query = """\
            # Combine indices and translate compact CSA pages in one pass.
            combined_indices, combined_lens = combine_topk_swa_indices(
                chunk_topk_indices,
"""
    source = _replace_once(
        source,
        old_query,
        new_query,
        "FlashMLA compact top-K input",
    )
    combine_tail = """                chunk_M,
                chunk_N,
            )
"""
    return _replace_once(
        source,
        combine_tail,
        """                chunk_M,
                chunk_N,
                page_to_compact=compact_page_to_compact,
                compact_block_size=(
                    attn_metadata.block_size // self.compress_ratio
                    if compact_page_to_compact is not None
                    else 0
                ),
            )
""",
        "FlashMLA fused compact-page translation",
    )


def _patch_kv_cache_admission(source: str) -> str:
    admission_check = """    # Check if the available memory is enough per worker.
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
    external_cache_check = f"""    # Check if the available memory is enough per worker.
    # {_MARKER}: the DSV4 SSD path intentionally keeps only a bounded working
    # set in vLLM's GPU cache while LMCache owns the complete logical sequence.
    # vLLM 0.26 otherwise rejects the old, proven 530K configuration before
    # the connector can participate. Keep this fail-closed and opt-in.
    lmcache_allow_oversized = os.getenv(
        "LMCACHE_ALLOW_OVERSIZED_KV_CACHE", "0"
    ).lower() in {{"1", "on", "true", "yes"}}
    if lmcache_allow_oversized:
        logger.warning(
            "LMCache external-KV mode is bypassing the max-model-len local "
            "KV admission check; available GPU KV memory still determines "
            "the resident working-set capacity."
        )
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        if lmcache_allow_oversized and avail_mem > 0:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )
"""
    return _replace_once(
        source,
        admission_check,
        external_cache_check,
        "LMCache external-KV admission policy",
    )


def _patch_scheduler_oversized_prefill(source: str) -> str:
    source = _replace_once(
        source,
        "import itertools\nimport time\n",
        "import itertools\nimport os\nimport time\n",
        "scheduler LMCache environment import",
    )
    logger_marker = "logger = init_logger(__name__)\n"
    opt_in = f"""logger = init_logger(__name__)

# {_MARKER}: vLLM 0.26 can schedule the next prefill chunk before the
# preceding GPU step has settled. An oversized hybrid request may then fail
# its next allocation and preempt itself before sparse-window blocks become
# safe to recycle. Serialize chunks only for the explicit LMCache external-KV
# mode; ordinary vLLM scheduling is unchanged.
_LMCACHE_SERIALIZE_OVERSIZED_PREFILL = os.getenv(
    "LMCACHE_ALLOW_OVERSIZED_KV_CACHE", "0"
).lower() in {{"1", "on", "true", "yes"}}
"""
    source = _replace_once(
        source,
        logger_marker,
        opt_in,
        "scheduler LMCache oversized-prefill flag",
    )
    running_marker = """\
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
"""
    serialized = """        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                _LMCACHE_SERIALIZE_OVERSIZED_PREFILL
                and request.is_prefill_chunk
                and request.num_in_flight_tokens > 0
            ):
                # Wait until update_from_output marks the previous chunk safe.
                # This lets allocate_slots recycle sparse-window blocks instead
                # of preempting and restarting the request from token zero.
                req_index += 1
                continue

            if (
"""
    return _replace_once(
        source,
        running_marker,
        serialized,
        "scheduler LMCache oversized-prefill serialization",
    )


def patch_vllm_0260(vllm_root: Path) -> tuple[Path, ...]:
    """Apply the LMCache overlay to a vLLM 0.26.0 package tree.

    Args:
        vllm_root: Directory containing the installed ``vllm`` package.

    Returns:
        Paths changed by this invocation. An empty tuple means the complete
        overlay was already present.

    Raises:
        FileNotFoundError: If a required vLLM 0.26.0 source file is missing.
        PatchError: If source markers do not match the supported 0.26.0 tree.
    """
    targets = (
        (
            vllm_root / "distributed/kv_transfer/kv_connector/v1/lmcache_connector.py",
            _patch_connector,
        ),
        (vllm_root / "entrypoints/openai/api_server.py", _patch_api_server),
        (
            vllm_root / "model_executor/models/deepseek_v2.py",
            _patch_deepseek_v2_profile_nvtx,
        ),
        (vllm_root / "models/deepseek_v4/nvidia/model.py", _patch_dsv4_model),
        (
            vllm_root / "model_executor/layers/sparse_attn_indexer.py",
            _patch_sparse_indexer,
        ),
        (
            vllm_root / "models/deepseek_v4/common/ops/cache_utils.py",
            _patch_cache_utils,
        ),
        (vllm_root / "models/deepseek_v4/nvidia/flashmla.py", _patch_flashmla),
        (vllm_root / "v1/core/kv_cache_utils.py", _patch_kv_cache_admission),
        (
            vllm_root / "v1/core/sched/scheduler.py",
            _patch_scheduler_oversized_prefill,
        ),
    )
    missing = [path for path, _ in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing vLLM 0.26.0 files: {missing}")

    changed: list[Path] = []
    for path, transform in targets:
        if _patch_file(path, transform):
            changed.append(path)
    return tuple(changed)


def main() -> int:
    """Apply the overlay selected by the command-line vLLM package path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-root",
        type=Path,
        required=True,
        help="path to the installed vllm Python package",
    )
    args = parser.parse_args()
    changed = patch_vllm_0260(args.vllm_root.resolve())
    for path in changed:
        print(f"patched {path}")
    if not changed:
        print("vLLM 0.26.0 LMCache overlay already applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
