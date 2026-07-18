# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING, Any, Generator, Optional, Union
import gc
import inspect
import math
import os
import re
import threading
import time
import weakref

# Third Party
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm.version import __version__ as VLLM_VERSION
import torch

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache import utils
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    lmcache_get_or_create_config,
)
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheStoreEvent, _lmcache_nvtx_annotate, cdiv
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import validate_and_set_config_value
from lmcache.v1.csa_pipeline_nvtx import CsaNvtxEvent, csa_pipeline_nvtx
from lmcache.v1.csa_prefetch_policy import (
    CSAPrefetchLookaheadPolicy,
    build_residual_prefetch_sources,
)
from lmcache.v1.manager import LMCacheManager

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    # First Party
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


_INDEXER_PREFETCH_MANAGER: Any = None
_CSA_ATTENTION_KV_PREFETCH_MANAGER: Any = None
# Tutti loader recorded by the loader-ready callback (may run on a background
# daemon thread). Main-thread lazy attach consumes it for the independent
# indexer and CSA attention-KV managers.
_CSA_PREFETCH_PENDING_LOADER: Any = None
_CSA_PREFETCH_PENDING_LOCK = threading.Lock()
_INDEXER_PREFETCH_PENDING_LOADER: Any = None
_INDEXER_PREFETCH_PENDING_LOCK = threading.Lock()
_DEEPSEEK_DECODER_LAYER_CACHE: tuple[Any, ...] = ()
_LMCACHE_DEEPSEEK_DECODER_LAYER_REGISTRY: Any = weakref.WeakSet()
_DEEPSEEK_DECODER_REGISTRY_HOOK_INSTALLED: bool = False
_OVERLAP_HOOK_ERROR_LOGGED: set[tuple[str, int, str]] = set()
_SCHEDULER_HMA_INVALID_BLOCK_PATCH_INSTALLED: bool = False


def _block_ids_at_index(block_id_groups: tuple[list[int], ...], idx: int) -> set[int]:
    """Return every HMA group block ID present at a logical block index."""
    block_ids: set[int] = set()
    for group_block_ids in block_id_groups:
        if idx < len(group_block_ids):
            block_ids.add(int(group_block_ids[idx]))
    return block_ids


def _install_scheduler_hma_invalid_block_patch() -> None:
    """Patch vLLM's KV-load failure path to handle hybrid KV cache groups."""
    global _SCHEDULER_HMA_INVALID_BLOCK_PATCH_INSTALLED

    if _SCHEDULER_HMA_INVALID_BLOCK_PATCH_INSTALLED:
        return
    try:
        module = __import__(
            "vllm.v1.core.sched.scheduler",
            fromlist=["Scheduler"],
        )
    except ImportError:
        return
    scheduler_cls = getattr(module, "Scheduler", None)
    if scheduler_cls is None:
        return
    if getattr(
        scheduler_cls,
        "_lmcache_hma_invalid_block_patch_installed",
        False,
    ):
        _SCHEDULER_HMA_INVALID_BLOCK_PATCH_INSTALLED = True
        return
    original = getattr(
        scheduler_cls,
        "_update_requests_with_invalid_blocks",
        None,
    )
    if original is None:
        return

    def _lmcache_update_requests_with_invalid_blocks(
        self: Any,
        requests: Iterable[Any],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            req_block_id_groups = self.kv_cache_manager.get_block_ids(req_id)
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )
            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size

            if len(req_block_id_groups) == 1:
                req_block_ids = req_block_id_groups[0]
                for idx, block_id in zip(
                    range(req_num_computed_blocks),
                    req_block_ids,
                ):
                    if block_id not in invalid_block_ids:
                        continue

                    is_affected = True

                    if block_id in marked_invalid_block_ids:
                        continue

                    marked_invalid_block_ids.add(block_id)

                    if marked_invalid_block:
                        continue

                    marked_invalid_block = True
                    request.num_computed_tokens = idx * self.block_size
                    total_affected_tokens += (
                        req_num_computed_tokens - request.num_computed_tokens
                    )

                    if evict_blocks:
                        blocks_to_evict.update(req_block_ids[idx:])
            else:
                for idx in range(req_num_computed_blocks):
                    block_ids_at_idx = _block_ids_at_index(req_block_id_groups, idx)
                    invalid_at_idx = block_ids_at_idx & invalid_block_ids
                    if not invalid_at_idx:
                        continue

                    is_affected = True
                    newly_marked = invalid_at_idx - marked_invalid_block_ids
                    if not newly_marked:
                        continue

                    marked_invalid_block_ids.update(newly_marked)

                    if marked_invalid_block:
                        continue

                    marked_invalid_block = True
                    request.num_computed_tokens = idx * self.block_size
                    total_affected_tokens += (
                        req_num_computed_tokens - request.num_computed_tokens
                    )

                    if evict_blocks:
                        for group_block_ids in req_block_id_groups:
                            if idx < len(group_block_ids):
                                blocks_to_evict.update(
                                    int(block_id)
                                    for block_id in group_block_ids[idx:]
                                )

            if is_affected:
                if not marked_invalid_block:
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    scheduler_cls._lmcache_original_update_requests_with_invalid_blocks = original
    scheduler_cls._update_requests_with_invalid_blocks = (
        _lmcache_update_requests_with_invalid_blocks
    )
    scheduler_cls._lmcache_hma_invalid_block_patch_installed = True
    _SCHEDULER_HMA_INVALID_BLOCK_PATCH_INSTALLED = True
    logger.info("LMCache installed vLLM HMA invalid-block scheduler patch")


def _layer_idx_from_prefix(prefix: Any) -> int:
    """Best-effort DeepSeek decoder layer id extraction from a vLLM prefix."""
    if not isinstance(prefix, str):
        return -1
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
    if match is None:
        return -1
    return int(match.group(1))


def _infer_decoder_layer_idx(decoder_layer: Any) -> int:
    """Infer and cache the vLLM layer id on a DeepSeek decoder layer."""
    layer_idx = getattr(decoder_layer, "layer_idx", -1)
    if isinstance(layer_idx, int) and layer_idx >= 0:
        return layer_idx

    prefixes = [
        getattr(decoder_layer, "prefix", None),
    ]
    attn = getattr(decoder_layer, "self_attn", None)
    if attn is None:
        attn = getattr(decoder_layer, "attn", None)
    if attn is not None:
        prefixes.extend(
            [
                getattr(attn, "prefix", None),
                getattr(getattr(attn, "mla_attn", None), "prefix", None),
                getattr(
                    getattr(getattr(attn, "mla_attn", None), "mla_attn", None),
                    "prefix",
                    None,
                ),
            ]
        )
    for prefix in prefixes:
        inferred = _layer_idx_from_prefix(prefix)
        if inferred >= 0:
            try:
                setattr(decoder_layer, "layer_idx", inferred)
            except Exception:
                pass
            return inferred
    return -1


def _env_flag(name: str) -> bool:
    """Return True when an environment variable is set to a truthy value."""
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _normalize_block_ids_by_group(
    block_ids: Optional[Union[tuple[list[int], ...], list[int], list[list[int]]]],
) -> tuple[list[int], ...]:
    """Normalize vLLM block ids into ``tuple[group][block]`` form."""
    if block_ids is None:
        return ()
    if len(block_ids) == 0:
        return ()
    if isinstance(block_ids, tuple):
        return tuple(list(group) for group in block_ids)
    if isinstance(block_ids, list):
        if block_ids and all(isinstance(group, list) for group in block_ids):
            return tuple(list(group) for group in block_ids)
        return (list(block_ids),)
    raise ValueError(f"Unsupported block_ids type {type(block_ids)}")


def _select_primary_block_ids(
    block_ids_by_group: tuple[list[int], ...],
    num_tokens: int,
    block_size: Optional[int],
) -> list[int]:
    """Pick the KV group that can represent the original token sequence.

    Under vLLM HMA, DSv4 exposes several KV cache groups. The first vLLM
    group is the canonical full-MLA group and carries the engine-logical token
    block size. LMCache's legacy ``slot_mapping`` is still token-addressed
    with that block size, so using ``vllm_config.cache_config.block_size``
    directly can be wrong when the global config reports the smallest
    compressed-group block size.
    """
    if not block_ids_by_group:
        return []
    if block_size is not None:
        for group in block_ids_by_group:
            if len(group) * block_size >= num_tokens:
                return list(group)
    return list(block_ids_by_group[0])


def _engine_logical_block_size(vllm_config: Any, parent: Any) -> int:
    """Return the token-addressed vLLM block size used by LMCache metadata."""
    override = os.environ.get("LMCACHE_ENGINE_LOGICAL_BLOCK_SIZE")
    if override:
        try:
            value = int(override)
        except ValueError:
            logger.warning(
                "Invalid LMCACHE_ENGINE_LOGICAL_BLOCK_SIZE=%r; ignoring",
                override,
            )
        else:
            if value > 0:
                return value

    cache_config = getattr(vllm_config, "cache_config", None)
    fallback = int(getattr(cache_config, "block_size", 0) or 0)

    kv_cache_config = getattr(parent, "_kv_cache_config", None)
    groups = getattr(kv_cache_config, "kv_cache_groups", ()) or ()
    if not groups:
        groups = getattr(cache_config, "kv_cache_groups", ()) or ()

    group_block_sizes: list[int] = []
    for group in groups:
        spec = getattr(group, "kv_cache_spec", None)
        block_size = int(getattr(spec, "block_size", 0) or 0)
        if block_size > 0:
            group_block_sizes.append(block_size)

    if group_block_sizes:
        logical_block_size = group_block_sizes[0]
        if fallback > 0 and logical_block_size != fallback:
            logger.info(
                "LMCache using vLLM HMA group0 block_size=%d as the "
                "engine-logical block size; cache_config.block_size=%d",
                logical_block_size,
                fallback,
            )
        return logical_block_size

    if fallback <= 0:
        raise ValueError("vLLM cache_config.block_size must be a positive integer")
    return fallback


def _indexer_prefetch_enabled() -> bool:
    """Return whether SSD-backed indexer prefetch is explicitly enabled."""
    value = os.environ.get("LMCACHE_INDEXER_ENABLE_PREFETCH", "")
    return value.lower() in {"1", "true", "yes", "on"}


def close_vllm_prefetch_managers() -> None:
    """Close and detach process-local Tutti prefetch managers.

    The operation is idempotent and clears both adapter-owned and
    manager-module global references before releasing resources.
    """
    global _CSA_ATTENTION_KV_PREFETCH_MANAGER, _INDEXER_PREFETCH_MANAGER

    csa_manager = _CSA_ATTENTION_KV_PREFETCH_MANAGER
    _CSA_ATTENTION_KV_PREFETCH_MANAGER = None
    try:
        from lmcache.v1.csa_attention_kv_prefetch_manager import (
            set_csa_attention_kv_prefetch_manager,
        )

        set_csa_attention_kv_prefetch_manager(None)
    except ImportError:
        pass
    if csa_manager is not None:
        try:
            csa_manager.close()
        except Exception:
            logger.exception("Failed to close the CSA/HCA Tutti prefetch manager")

    indexer_manager = _INDEXER_PREFETCH_MANAGER
    _INDEXER_PREFETCH_MANAGER = None
    try:
        from lmcache.v1.indexer_ssd_manager import set_indexer_ssd_manager

        set_indexer_ssd_manager(None)
    except ImportError:
        pass
    if indexer_manager is not None:
        try:
            indexer_manager.close()
        except Exception:
            logger.exception("Failed to close IndexerSSDManager")


def _csa_prefetch_lookahead_policy() -> CSAPrefetchLookaheadPolicy:
    """Return the per-target policy, defaulting to deep-only L2 prefetch."""
    value = os.environ.get(
        "LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER",
        "profile80",
    )
    return CSAPrefetchLookaheadPolicy(value)


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable with a safe fallback."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d", name, value, default)
        return default


def _deepseek_decoder_registries() -> list[Any]:
    """Return vLLM DeepSeek decoder registries that are present in-process."""
    registry_specs = (
        (
            "vllm.models.deepseek_v4.nvidia.model",
            "_DEEPSEEK_V4_DECODER_LAYER_REGISTRY",
        ),
        (
            "vllm.models.deepseek_v4.amd.model",
            "_DEEPSEEK_V4_DECODER_LAYER_REGISTRY",
        ),
        (
            "vllm.model_executor.models.deepseek_v4",
            "_DEEPSEEK_V4_DECODER_LAYER_REGISTRY",
        ),
    )
    registries: list[Any] = [_LMCACHE_DEEPSEEK_DECODER_LAYER_REGISTRY]
    for module_name, registry_name in registry_specs:
        try:
            module = __import__(module_name, fromlist=[registry_name])
        except ImportError:
            continue
        registry = getattr(module, registry_name, None)
        if registry is not None:
            registries.append(registry)
    return registries


def _install_deepseek_decoder_registry_hook() -> None:
    """Register DeepSeek decoder layers constructed by vLLM in this process."""
    global _DEEPSEEK_DECODER_REGISTRY_HOOK_INSTALLED

    if _DEEPSEEK_DECODER_REGISTRY_HOOK_INSTALLED:
        return
    try:
        module = __import__(
            "vllm.model_executor.models.deepseek_v4",
            fromlist=["DeepseekV4DecoderLayer"],
        )
    except ImportError:
        return
    module_registry = getattr(module, "_DEEPSEEK_V4_DECODER_LAYER_REGISTRY", None)
    if module_registry is None:
        module_registry = weakref.WeakSet()
        try:
            setattr(module, "_DEEPSEEK_V4_DECODER_LAYER_REGISTRY", module_registry)
        except Exception:
            module_registry = None
    layer_cls = getattr(module, "DeepseekV4DecoderLayer", None)
    if layer_cls is None:
        return
    if getattr(layer_cls, "_lmcache_decoder_registry_hook_installed", False):
        _DEEPSEEK_DECODER_REGISTRY_HOOK_INSTALLED = True
        return
    original_init = getattr(layer_cls, "__init__", None)
    if original_init is None:
        return

    def _lmcache_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = kwargs.get("prefix")
        if prefix is None and len(args) >= 2:
            prefix = args[1]
        layer_idx = _layer_idx_from_prefix(prefix)
        if layer_idx >= 0:
            try:
                setattr(self, "layer_idx", layer_idx)
            except Exception:
                pass
        try:
            _LMCACHE_DEEPSEEK_DECODER_LAYER_REGISTRY.add(self)
        except TypeError:
            pass
        if module_registry is not None:
            try:
                module_registry.add(self)
            except TypeError:
                pass

    layer_cls._lmcache_original_init = original_init
    layer_cls.__init__ = _lmcache_init
    layer_cls._lmcache_decoder_registry_hook_installed = True
    _DEEPSEEK_DECODER_REGISTRY_HOOK_INSTALLED = True
    logger.info("LMCache installed DeepSeek decoder layer registry hook")


def _decoder_csa_indexer(decoder_layer: Any) -> Any:
    """Return the SparseAttnIndexer op owned by a DeepSeek decoder layer."""
    attn = getattr(decoder_layer, "self_attn", None)
    if attn is None:
        attn = getattr(decoder_layer, "attn", None)
    indexer = getattr(attn, "indexer", None)
    return getattr(indexer, "indexer_op", None)


def _decoder_hca_attention(decoder_layer: Any) -> Any:
    """Return the HCA MLA attention object owned by a DeepSeek decoder layer."""
    attn = getattr(decoder_layer, "self_attn", None)
    if attn is None:
        attn = getattr(decoder_layer, "attn", None)
    if attn is None:
        return None
    wrapper = getattr(attn, "mla_attn", None)
    candidates = [
        getattr(wrapper, "mla_attn", None),
        wrapper,
        attn,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        compress_ratio = int(getattr(candidate, "compress_ratio", 1))
        kv_cache = getattr(candidate, "kv_cache", None)
        if compress_ratio == 128 and isinstance(kv_cache, torch.Tensor):
            return candidate
    return None


def _decoder_csa_attention(decoder_layer: Any) -> Any:
    """Return the CSA MLA attention object owned by a DeepSeek decoder layer.

    The returned object has a populated ``kv_cache`` tensor of shape
    ``[num_blocks, compressed_block_size, token_bytes]`` and
    ``compress_ratio == 4``.  Returns ``None`` when the layer is HCA or the
    cache has not been materialised yet.
    """
    attn = getattr(decoder_layer, "self_attn", None)
    if attn is None:
        attn = getattr(decoder_layer, "attn", None)
    if attn is None:
        return None
    wrapper = getattr(attn, "mla_attn", None)
    candidates = [
        getattr(wrapper, "mla_attn", None),
        wrapper,
        attn,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        compress_ratio = int(getattr(candidate, "compress_ratio", 1))
        kv_cache = getattr(candidate, "kv_cache", None)
        if compress_ratio == 4 and isinstance(kv_cache, torch.Tensor) and kv_cache.numel() > 0:
            return candidate
    return None


def _is_deepseek_decoder_layer_candidate(obj: Any) -> bool:
    """Return whether ``obj`` looks like a live DeepSeek V4 decoder layer."""
    try:
        obj_type = type(obj)
        type_name = getattr(obj_type, "__name__", "")
        module_name = getattr(obj_type, "__module__", "")
        is_deepseek_decoder_type = (
            type_name == "DeepseekV4DecoderLayer"
            and module_name == "vllm.model_executor.models.deepseek_v4"
        )
        if (
            not is_deepseek_decoder_type
            and "deepseek" not in module_name.lower()
            and "deepseek" not in type_name.lower()
        ):
            return False
        layer_id = _infer_decoder_layer_idx(obj)
        if not isinstance(layer_id, int) or layer_id < 0:
            return False
        has_attention = getattr(obj, "self_attn", None) is not None or getattr(
            obj,
            "attn",
            None,
        ) is not None
        if not has_attention:
            return False
        if is_deepseek_decoder_type:
            return True
        if not callable(getattr(obj, "hc_pre", None)):
            return False
        return getattr(obj, "hc_ffn_fn", None) is not None
    except Exception:
        return False


def _deepseek_decoder_layers() -> list[Any]:
    """Return live DeepSeek decoder layers, with a GC fallback for patched vLLM."""
    global _DEEPSEEK_DECODER_LAYER_CACHE

    if _DEEPSEEK_DECODER_LAYER_CACHE:
        return list(_DEEPSEEK_DECODER_LAYER_CACHE)

    decoder_layers: list[Any] = []
    seen: set[int] = set()
    source = "registry"
    for registry in _deepseek_decoder_registries():
        for decoder_layer in list(registry):
            obj_id = id(decoder_layer)
            if obj_id in seen:
                continue
            if not _is_deepseek_decoder_layer_candidate(decoder_layer):
                continue
            seen.add(obj_id)
            decoder_layers.append(decoder_layer)

    if not decoder_layers:
        source = "gc"
        try:
            objects = gc.get_objects()
        except Exception as exc:
            logger.debug("Unable to scan GC for DeepSeek decoder layers: %r", exc)
            objects = ()
        for obj in objects:
            obj_id = id(obj)
            if obj_id in seen:
                continue
            if not _is_deepseek_decoder_layer_candidate(obj):
                continue
            seen.add(obj_id)
            decoder_layers.append(obj)

    decoder_layers.sort(key=lambda layer: getattr(layer, "layer_idx", -1))
    if decoder_layers:
        _DEEPSEEK_DECODER_LAYER_CACHE = tuple(decoder_layers)
        logger.info(
            "LMCache discovered %d DeepSeek decoder layers via %s fallback",
            len(decoder_layers),
            source,
        )
    return list(_DEEPSEEK_DECODER_LAYER_CACHE)


def _same_callable(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    if left is right:
        return True
    left_func = getattr(left, "__func__", None)
    right_func = getattr(right, "__func__", None)
    left_self = getattr(left, "__self__", None)
    right_self = getattr(right, "__self__", None)
    if left_func is not None and right_func is not None:
        return left_func is right_func and left_self is right_self
    try:
        return bool(left == right)
    except Exception:
        return False


def _hc_pre_fn_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    for name in ("fn", "hc_fn", "kernel", "op"):
        if name in kwargs:
            return kwargs[name]
    return None


def _is_ffn_hc_pre_call(
    decoder_layer: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    return _same_callable(
        _hc_pre_fn_arg(args, kwargs),
        getattr(decoder_layer, "hc_ffn_fn", None),
    )


def _mhc_fused_post_pre_fn_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Return the HC function argument from a fused post/pre call."""
    if len(args) >= 5:
        return args[4]
    for name in ("fn", "hc_fn", "kernel", "op"):
        if name in kwargs:
            return kwargs[name]
    return None


def _is_ffn_mhc_fused_post_pre_call(
    decoder_layer: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    """Return whether a fused MHC post/pre call is entering the FFN block."""
    return _same_callable(
        _mhc_fused_post_pre_fn_arg(args, kwargs),
        getattr(decoder_layer, "hc_ffn_fn", None),
    )


def _overlap_token_rows(hidden_states: torch.Tensor) -> int:
    """Return the token-row count represented by a decoder hidden tensor."""
    if hidden_states.ndim == 0:
        return 1
    if hidden_states.ndim >= 3 and int(hidden_states.shape[0]) == 1:
        return int(hidden_states.shape[1])
    return int(hidden_states.shape[0])


def _flatten_overlap_hidden_for_positions(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Align a decoder hidden tensor with flattened position rows."""
    flat_positions = positions.reshape(-1)
    if hidden_states.ndim >= 2:
        if int(hidden_states.shape[0]) == int(flat_positions.shape[0]):
            return hidden_states, flat_positions, int(flat_positions.shape[0])
        flat_hidden = hidden_states.reshape(-1, int(hidden_states.shape[-1]))
        max_rows = min(int(flat_hidden.shape[0]), int(flat_positions.shape[0]))
        return (
            flat_hidden[:max_rows].contiguous(),
            flat_positions[:max_rows].contiguous(),
            max_rows,
        )
    return hidden_states, flat_positions[:1].contiguous(), 1


def _positions_from_forward_frame(
    frame: Any,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    device = hidden_states.device
    rows = _overlap_token_rows(hidden_states)
    current = frame
    for _ in range(6):
        if current is None:
            return None
        locals_map = current.f_locals
        for name in ("positions", "position_ids", "input_positions"):
            value = locals_map.get(name)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return value.reshape(-1).to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True,
                )
        for name in ("start_pos", "start_position"):
            value = locals_map.get(name)
            if isinstance(value, int):
                return torch.arange(
                    value,
                    value + rows,
                    device=device,
                    dtype=torch.long,
                )
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                start = int(value.reshape(-1)[0].detach().cpu().item())
                return torch.arange(
                    start,
                    start + rows,
                    device=device,
                    dtype=torch.long,
                )
        current = current.f_back
    return None


def _forward_position_source(
    signature: inspect.Signature | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    values: dict[str, Any] = dict(kwargs)
    if signature is not None:
        try:
            values.update(signature.bind_partial(*args, **kwargs).arguments)
        except TypeError:
            pass
    for name in ("positions", "position_ids", "input_positions"):
        value = values.get(name)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            return value
    for name in ("start_pos", "start_position"):
        value = values.get(name)
        if isinstance(value, (int, torch.Tensor)):
            return value
    for value in args:
        if isinstance(value, int):
            return value
    return None


def _positions_from_source(
    source: Any,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    device = hidden_states.device
    rows = _overlap_token_rows(hidden_states)
    if isinstance(source, torch.Tensor) and source.numel() > 0:
        return source.reshape(-1).to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
    if isinstance(source, int):
        return torch.arange(
            source,
            source + rows,
            device=device,
            dtype=torch.long,
        )
    return None


def _positions_for_overlap(
    decoder_layer: Any,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    """Return current forward positions for a decoder-layer overlap hook."""
    positions = _positions_from_source(
        getattr(decoder_layer, "_lmcache_forward_position_source", None),
        hidden_states,
    )
    if positions is None:
        frame = inspect.currentframe()
        try:
            caller = frame.f_back if frame is not None else None
            positions = _positions_from_forward_frame(caller, hidden_states)
        finally:
            del frame
    return positions


def _install_decoder_forward_position_hook(decoder_layer: Any) -> bool:
    if getattr(decoder_layer, "_lmcache_forward_position_installed", False):
        return True
    original_forward = getattr(decoder_layer, "forward", None)
    if not callable(original_forward):
        return False
    try:
        signature = inspect.signature(original_forward)
    except (TypeError, ValueError):
        signature = None

    def _lmcache_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous_source = getattr(self, "_lmcache_forward_position_source", None)
        previous_overlap_fired = getattr(
            self,
            "_lmcache_pre_ffn_overlap_fired",
            None,
        )
        previous_python_overlap_fired = getattr(
            self,
            "_lmcache_python_pre_ffn_overlap_fired",
            None,
        )
        previous_hc_post_index = getattr(
            self,
            "_lmcache_hc_post_call_index",
            None,
        )
        self._lmcache_forward_position_source = _forward_position_source(
            signature,
            args,
            kwargs,
        )
        self._lmcache_pre_ffn_overlap_fired = False
        self._lmcache_python_pre_ffn_overlap_fired = False
        self._lmcache_hc_post_call_index = 0
        try:
            return original_forward(*args, **kwargs)
        finally:
            full_nsys_manager = _CSA_ATTENTION_KV_PREFETCH_MANAGER
            full_nsys_scope = os.getenv(
                "LMCACHE_NSYS_FULL_CAPTURE_SCOPE",
                "decoder",
            ).lower()
            if (
                full_nsys_scope == "decoder"
                and getattr(
                    self,
                    "_lmcache_nsys_full_capture_last_layer",
                    False,
                )
                and full_nsys_manager is not None
            ):
                finish_capture = getattr(
                    full_nsys_manager,
                    "finish_full_nsys_capture",
                    None,
                )
                if callable(finish_capture):
                    finish_capture()
            if previous_source is None:
                try:
                    delattr(self, "_lmcache_forward_position_source")
                except AttributeError:
                    pass
            else:
                self._lmcache_forward_position_source = previous_source
            if previous_overlap_fired is None:
                try:
                    delattr(self, "_lmcache_pre_ffn_overlap_fired")
                except AttributeError:
                    pass
            else:
                self._lmcache_pre_ffn_overlap_fired = previous_overlap_fired
            if previous_python_overlap_fired is None:
                try:
                    delattr(self, "_lmcache_python_pre_ffn_overlap_fired")
                except AttributeError:
                    pass
            else:
                self._lmcache_python_pre_ffn_overlap_fired = (
                    previous_python_overlap_fired
                )
            if previous_hc_post_index is None:
                try:
                    delattr(self, "_lmcache_hc_post_call_index")
                except AttributeError:
                    pass
            else:
                self._lmcache_hc_post_call_index = previous_hc_post_index

    decoder_layer._lmcache_original_forward = original_forward
    decoder_layer.forward = MethodType(_lmcache_forward, decoder_layer)
    decoder_layer._lmcache_forward_position_installed = True
    return True


def _log_overlap_hook_error_once(
    kind: str,
    layer_id: int,
    exc: Exception,
) -> None:
    reason = type(exc).__name__
    key = (kind, layer_id, reason)
    if key in _OVERLAP_HOOK_ERROR_LOGGED:
        return
    _OVERLAP_HOOK_ERROR_LOGGED.add(key)
    logger.warning(
        "LMCache %s early-overlap hook failed at decoder layer %d: %r",
        kind,
        layer_id,
        exc,
    )


def _fire_decoder_ffn_overlap(
    decoder_layer: Any,
    residual_after_attention: torch.Tensor,
    positions: torch.Tensor | None,
    source: str,
) -> bool:
    layer_id = getattr(decoder_layer, "layer_idx", -1)
    fired = False

    indexer_manager = getattr(
        decoder_layer,
        "_lmcache_indexer_prefetch_manager",
        None,
    )
    next_csa = getattr(decoder_layer, "_lmcache_next_csa_layer_id", -1)
    csa_prefetch_level = getattr(
        decoder_layer, "_lmcache_next_csa_prefetch_level", 1
    )
    if indexer_manager is not None and isinstance(next_csa, int) and next_csa >= 0:
        try:
            if positions is not None:
                residual_after_attention, positions, rows = (
                    _flatten_overlap_hidden_for_positions(
                        residual_after_attention,
                        positions,
                    )
                )
            else:
                rows = _overlap_token_rows(residual_after_attention)
            if rows > 1:
                csa_kv_attached_fn = getattr(
                    indexer_manager,
                    "csa_attention_kv_prefetch_attached",
                    None,
                )
                csa_kv_attached = (
                    bool(csa_kv_attached_fn())
                    if callable(csa_kv_attached_fn)
                    else False
                )
                if not csa_kv_attached:
                    indexer_manager = None
            if indexer_manager is not None:
                fire_residual = getattr(
                    indexer_manager,
                    "fire_residual_prefetch_for_layer",
                    None,
                )
                if callable(fire_residual):
                    fire_residual(
                        next_csa,
                        residual_after_attention,
                        positions,
                        lookahead=int(csa_prefetch_level),
                    )
                else:
                    raise RuntimeError(
                        "IndexerSSDManager lacks the canonical L2 prefetch API"
                    )
                fired = True
        except Exception as exc:
            _log_overlap_hook_error_once("CSA", int(layer_id), exc)

    hca_manager = getattr(decoder_layer, "_lmcache_hca_prefetch_manager", None)
    hca_targets = getattr(decoder_layer, "_lmcache_next_hca_layer_ids", ())
    if hca_manager is not None and isinstance(hca_targets, tuple) and hca_targets:
        try:
            hca_fired = getattr(hca_manager, "layer_fired_for_active_request", None)
            fire_layers = getattr(hca_manager, "fire_async_for_layers", None)
            pending_targets = tuple(
                next_hca
                for next_hca in hca_targets
                if not (callable(hca_fired) and hca_fired(next_hca))
            )
            if positions is not None and callable(fire_layers):
                if pending_targets:
                    fire_layers(pending_targets, positions)
            elif positions is not None:
                for next_hca in pending_targets:
                    hca_manager.fire_async_for_layer(next_hca, positions)
            prepare_hca_layers = getattr(hca_manager, "prepare_layers_async", None)
            if callable(prepare_hca_layers):
                prepare_hca_layers(hca_targets)
            else:
                prepare_hca = getattr(hca_manager, "prepare_layer_async", None)
                if callable(prepare_hca):
                    for next_hca in hca_targets:
                        prepare_hca(next_hca)
            fired = True
        except Exception as exc:
            _log_overlap_hook_error_once("HCA", int(layer_id), exc)
    return fired


def _maybe_fire_decoder_ffn_overlap(
    decoder_layer: Any,
    residual_after_attention: torch.Tensor,
    positions: torch.Tensor | None,
    source: str,
) -> None:
    if positions is None:
        return
    if getattr(decoder_layer, "_lmcache_pre_ffn_overlap_fired", False):
        return
    if getattr(decoder_layer, "_lmcache_python_pre_ffn_overlap_fired", False):
        return
    if _fire_decoder_ffn_overlap(
        decoder_layer,
        residual_after_attention,
        positions,
        source,
    ):
        decoder_layer._lmcache_python_pre_ffn_overlap_fired = True
        decoder_layer._lmcache_pre_ffn_overlap_fired = True


def _install_decoder_hc_pre_overlap_hook(decoder_layer: Any) -> bool:
    """Install an FFN-entry hook that fires CSA/HCA prefetch before MoE."""
    hc_pre = getattr(decoder_layer, "hc_pre", None)
    if not callable(hc_pre):
        return False
    if getattr(decoder_layer, "_lmcache_hc_pre_overlap_installed", False):
        return True

    original_hc_pre = hc_pre

    def _lmcache_hc_pre(
        self: Any,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if isinstance(hidden_states, torch.Tensor) and _is_ffn_hc_pre_call(
            self,
            args,
            kwargs,
        ):
            positions = _positions_for_overlap(self, hidden_states)
            _maybe_fire_decoder_ffn_overlap(self, hidden_states, positions, "hc_pre")
        return original_hc_pre(hidden_states, *args, **kwargs)

    _install_decoder_forward_position_hook(decoder_layer)
    decoder_layer._lmcache_original_hc_pre = original_hc_pre
    decoder_layer.hc_pre = MethodType(_lmcache_hc_pre, decoder_layer)
    decoder_layer._lmcache_hc_pre_overlap_installed = True
    return True


def _install_decoder_hc_post_overlap_hook(decoder_layer: Any) -> bool:
    """Install an attention-post hook that fires before the FFN HC step."""
    hc_post = getattr(decoder_layer, "hc_post", None)
    if not callable(hc_post):
        return False
    if getattr(decoder_layer, "_lmcache_hc_post_overlap_installed", False):
        return True

    original_hc_post = hc_post

    def _lmcache_hc_post(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_hc_post(*args, **kwargs)
        call_index = int(getattr(self, "_lmcache_hc_post_call_index", 0))
        self._lmcache_hc_post_call_index = call_index + 1
        if call_index == 0 and isinstance(result, torch.Tensor):
            positions = _positions_for_overlap(self, result)
            _maybe_fire_decoder_ffn_overlap(self, result, positions, "hc_post")
        return result

    _install_decoder_forward_position_hook(decoder_layer)
    decoder_layer._lmcache_original_hc_post = original_hc_post
    decoder_layer.hc_post = MethodType(_lmcache_hc_post, decoder_layer)
    decoder_layer._lmcache_hc_post_overlap_installed = True
    return True


def _install_decoder_mhc_fused_post_pre_overlap_hook(decoder_layer: Any) -> bool:
    """Install a CUDA fused attention-post/FFN-pre overlap hook."""
    fused_op = getattr(decoder_layer, "mhc_fused_post_pre", None)
    original_forward = getattr(fused_op, "forward", None)
    if not callable(original_forward):
        return False
    if getattr(fused_op, "_lmcache_mhc_fused_overlap_installed", False):
        return True

    def _lmcache_mhc_fused_post_pre(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_forward(*args, **kwargs)
        if not _is_ffn_mhc_fused_post_pre_call(decoder_layer, args, kwargs):
            return result
        residual_after_attention = None
        if isinstance(result, tuple) and result:
            candidate = result[0]
            if isinstance(candidate, torch.Tensor):
                residual_after_attention = candidate
        if residual_after_attention is None:
            return result
        positions = _positions_for_overlap(decoder_layer, residual_after_attention)
        _maybe_fire_decoder_ffn_overlap(
            decoder_layer,
            residual_after_attention,
            positions,
            "mhc_fused",
        )
        return result

    _install_decoder_forward_position_hook(decoder_layer)
    fused_op._lmcache_original_forward = original_forward
    fused_op.forward = _lmcache_mhc_fused_post_pre
    fused_op._lmcache_mhc_fused_overlap_installed = True
    return True


def _install_decoder_native_indexer_prefetch_guard(decoder_layer: Any) -> None:
    """Guard native vLLM indexer prefetch against Python fallback double-fire."""
    native_fire = getattr(decoder_layer, "_fire_indexer_prefetch", None)
    if not callable(native_fire):
        return
    if getattr(decoder_layer, "_lmcache_native_indexer_guard_installed", False):
        return

    original_native_fire = native_fire

    def _lmcache_fire_indexer_prefetch(
        self: Any,
        residual_f: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> Any:
        if getattr(self, "_lmcache_pre_ffn_overlap_fired", False):
            return None
        self._lmcache_pre_ffn_overlap_fired = True
        return original_native_fire(residual_f, positions)

    _install_decoder_forward_position_hook(decoder_layer)
    decoder_layer._lmcache_original_fire_indexer_prefetch = original_native_fire
    decoder_layer._fire_indexer_prefetch = MethodType(
        _lmcache_fire_indexer_prefetch,
        decoder_layer,
    )
    decoder_layer._lmcache_native_indexer_guard_installed = True


def _install_decoder_pre_ffn_overlap_hooks(decoder_layer: Any) -> bool:
    """Install best-effort Python hooks for the pre-FFN overlap window."""
    _install_decoder_native_indexer_prefetch_guard(decoder_layer)
    installed = False
    if _install_decoder_hc_post_overlap_hook(decoder_layer):
        installed = True
    if _install_decoder_mhc_fused_post_pre_overlap_hook(decoder_layer):
        installed = True
    if _install_decoder_hc_pre_overlap_hook(decoder_layer):
        installed = True
    return installed


def _configure_decoder_csa_overlap(
    decoder_layer: Any,
    manager: Any,
    next_csa_layer_id: int,
    prefetch_level: int = 1,
) -> bool:
    decoder_layer._lmcache_indexer_prefetch_manager = manager
    decoder_layer._lmcache_next_csa_layer_id = next_csa_layer_id
    decoder_layer._lmcache_next_csa_prefetch_level = int(prefetch_level)
    native_hook = callable(
        getattr(decoder_layer, "_lmcache_fire_pre_ffn_overlap", None)
    )
    if not isinstance(next_csa_layer_id, int) or next_csa_layer_id < 0:
        return native_hook
    return _install_decoder_pre_ffn_overlap_hooks(decoder_layer) or native_hook


def _configure_decoder_hca_overlap(
    decoder_layer: Any,
    manager: Any,
    next_hca_layer_id: int,
    next_hca_layer_ids: tuple[int, ...] = (),
) -> bool:
    decoder_layer._lmcache_hca_prefetch_manager = manager
    decoder_layer._lmcache_next_hca_layer_id = next_hca_layer_id
    decoder_layer._lmcache_next_hca_layer_ids = next_hca_layer_ids
    native_hook = callable(
        getattr(decoder_layer, "_lmcache_fire_pre_ffn_overlap", None)
    )
    if (
        (not isinstance(next_hca_layer_id, int) or next_hca_layer_id < 0)
        and not next_hca_layer_ids
    ):
        return native_hook
    return _install_decoder_pre_ffn_overlap_hooks(decoder_layer) or native_hook


def _indexer_tutti_backend_enabled() -> bool:
    """Return whether the CSA indexer should use the Tutti GPU-direct backend.

    Default is off; the legacy per-layer ``.bin`` file backend is preserved
    until the operator opts in by setting ``LMCACHE_INDEXER_TUTTI_BACKEND=1``.
    """
    value = os.environ.get("LMCACHE_INDEXER_TUTTI_BACKEND", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _indexer_worker_rank(tutti_loader: Optional[Any] = None) -> int:
    """Return the worker rank, preferring the loader's actual CUDA device.

    Args:
        tutti_loader: Optional active Tutti loader.

    Returns:
        Non-negative worker rank.

    Raises:
        ValueError: If neither the loader nor rank environment is valid.
    """
    loader_device = getattr(tutti_loader, "cuda_device", None)
    if isinstance(loader_device, int) and loader_device >= 0:
        return loader_device
    rank_value = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
    try:
        rank = int(rank_value)
    except ValueError as exc:
        raise ValueError(f"invalid indexer worker rank {rank_value!r}") from exc
    if rank < 0:
        raise ValueError(f"invalid negative indexer worker rank {rank}")
    return rank


def _maybe_build_indexer_tutti_storage(
    tutti_loader: Optional[Any],
    csa_layer_ids: list[int],
    token_bytes: int,
    max_seq_len: int,
) -> Optional[Any]:
    """Build a :class:`TuttiIndexerStorage` if the Tutti backend is requested.

    Args:
        tutti_loader: Active :class:`TuttiDirectLoader`.  When ``None``, the
            file backend is used regardless of the env flag.
        csa_layer_ids: Sorted CSA layer ids.
        token_bytes: Bytes per logical token K vector.
        max_seq_len: Logical token capacity per layer.

    Returns:
        A configured :class:`TuttiIndexerStorage` when all of the following
        hold: the env flag is on, ``tutti_loader`` is non-None, and a raw
        region path is configured via ``LMCACHE_INDEXER_TUTTI_RAW_REGION_PATH``.
        Returns ``None`` otherwise; callers fall back to the file backend.
    """
    if not _indexer_tutti_backend_enabled():
        return None
    if tutti_loader is None:
        logger.warning(
            "LMCACHE_INDEXER_TUTTI_BACKEND is set but no Tutti loader is "
            "available; falling back to the file backend"
        )
        return None
    raw_region_csv = os.environ.get("LMCACHE_INDEXER_TUTTI_RAW_REGION_PATH", "")
    raw_region_paths = [
        path.strip() for path in raw_region_csv.split(",") if path.strip()
    ]
    if not raw_region_paths:
        logger.warning(
            "LMCACHE_INDEXER_TUTTI_BACKEND is set but "
            "LMCACHE_INDEXER_TUTTI_RAW_REGION_PATH is empty; falling back to "
            "the file backend"
        )
        return None
    try:
        rank_index = _indexer_worker_rank(tutti_loader)
    except ValueError as exc:
        logger.warning("Cannot select indexer raw region: %s", exc)
        return None
    raw_region_path = raw_region_paths[rank_index % len(raw_region_paths)]
    try:
        from lmcache.v1.gpu_connector.tutti_direct_loader import FiemapHelper
        from lmcache.v1.indexer_tutti_backend import TuttiIndexerStorage
    except ImportError as exc:
        logger.warning(
            "Tutti indexer backend unavailable: %s; falling back to file backend",
            exc,
        )
        return None

    cached_records = getattr(tutti_loader, "get_lba_records", lambda _path: [])(
        raw_region_path
    )
    if cached_records:
        lba_records = cached_records
    else:
        try:
            lba_records = FiemapHelper.query_extents(raw_region_path)
        except Exception as exc:
            logger.warning(
                "Failed to query FIEMAP for indexer raw region %s: %s; falling "
                "back to file backend",
                raw_region_path,
                exc,
            )
            return None
    if not lba_records:
        logger.warning(
            "Indexer raw region %s reported no LBA extents; falling back to "
            "file backend",
            raw_region_path,
        )
        return None

    raw_extents = [
        (int(record.file_offset), int(record.slba), int(record.n_sectors))
        for record in lba_records
    ]
    rank_suffix = str(rank_index)
    synthetic_path = f"tutti://csa_indexer_rank_{rank_suffix}"

    try:
        storage = TuttiIndexerStorage(
            tutti_loader=tutti_loader,
            raw_region_path=synthetic_path,
            raw_region_extents=raw_extents,
            layer_ids=csa_layer_ids,
            token_bytes=token_bytes,
            max_seq_len=max_seq_len,
        )
    except ValueError as exc:
        logger.warning(
            "Failed to construct TuttiIndexerStorage from %s: %s; falling "
            "back to file backend",
            raw_region_path,
            exc,
        )
        return None
    return storage


def _shard_indexer_ssd_dir(
    base_csv: str,
    rank_suffix: str,
    *,
    create_dirs: bool = True,
) -> str:
    """Select this worker's indexer SSD base directory from a CSV of paths.

    ``LMCACHE_INDEXER_SSD_DIR`` (and ``LMCACHE_HCA_SSD_DIR``) may be a single
    path or a comma-separated list of one path per drive. This shards the list
    across GPUs with the same ``by_gpu`` policy the KV cache ``local_disk``
    backend uses, so each rank's indexer lands on its own drive rather than all
    ranks sharing one -- which would be a single point of failure and an I/O
    bottleneck. A single-path value is returned unchanged.

    Args:
        base_csv: One path, or a comma-separated list of per-drive paths.
        rank_suffix: This worker's local rank as a string, used to derive the
            GPU device index for ``by_gpu`` sharding.
        create_dirs: Whether to create the configured directories. Tutti
            callers disable this after the backing filesystems are unmounted
            for direct device binding.

    Returns:
        The single base directory selected for this worker (no ``rank_N``
        suffix appended).

    Raises:
        ValueError: If ``base_csv`` contains no non-empty paths.
    """
    # Local import: keep the storage backend optional for callers that never
    # reach the prefetch attach paths.
    from lmcache.v1.storage_backend.path_sharder import PathSharder

    sharder = PathSharder(
        base_csv,
        strategy="by_gpu",
        dst_device=f"cuda:{rank_suffix}",
        create_dirs=create_dirs,
    )
    return sharder.selected


def _attach_indexer_prefetch(tutti_loader: Optional[Any] = None) -> None:
    """Attach SSD-backed CSA indexer prefetch when enabled by environment.

    Args:
        tutti_loader: Optional active :class:`TuttiDirectLoader`.  When the
            environment requests the Tutti backend (see
            ``LMCACHE_INDEXER_TUTTI_BACKEND``), this loader is used to allocate
            a shared :class:`TuttiIndexerStorage` and route CSA indexer I/O
            through Tutti's GPU-direct NVMe path.  When ``None`` or the
            environment requests the file backend, the legacy per-layer
            ``.bin`` file backend is used.
    """
    global _INDEXER_PREFETCH_MANAGER

    if not _indexer_prefetch_enabled():
        return
    if _INDEXER_PREFETCH_MANAGER is not None:
        return
    if _indexer_tutti_backend_enabled() and tutti_loader is None:
        logger.debug(
            "IndexerSSDManager waiting for the Tutti loader before attach"
        )
        return

    base_store_dir = os.environ.get("LMCACHE_INDEXER_SSD_DIR", "")
    if not base_store_dir:
        logger.warning(
            "LMCACHE_INDEXER_ENABLE_PREFETCH is set, but "
            "LMCACHE_INDEXER_SSD_DIR is empty; skipping CSA prefetch"
        )
        return

    try:
        rank_suffix = str(_indexer_worker_rank(tutti_loader))
    except ValueError as exc:
        logger.warning("Cannot attach IndexerSSDManager: %s", exc)
        return
    selected_base = _shard_indexer_ssd_dir(
        base_store_dir,
        rank_suffix,
        create_dirs=not _indexer_tutti_backend_enabled(),
    )
    store_dir = os.path.join(selected_base, f"rank_{rank_suffix}")

    try:
        from lmcache.v1.indexer_ssd_manager import (
            IndexerSSDManager,
            set_indexer_ssd_manager,
        )
    except ImportError as exc:
        logger.warning("IndexerSSDManager is unavailable: %s", exc)
        return

    decoder_layers = _deepseek_decoder_layers()

    if not decoder_layers:
        logger.warning(
            "CSA prefetch requested, but no registered DeepSeek decoder "
            "layers were found"
        )
        return

    csa_info: list[tuple[int, Any]] = []
    for decoder_layer in decoder_layers:
        layer_id = getattr(decoder_layer, "layer_idx", -1)
        indexer_op = _decoder_csa_indexer(decoder_layer)
        if isinstance(layer_id, int) and layer_id >= 0 and indexer_op is not None:
            csa_info.append((layer_id, indexer_op))

    if not csa_info:
        logger.warning(
            "CSA prefetch requested, but registered DeepSeek decoder layers "
            "have no CSA indexers"
        )
        return

    csa_info.sort(key=lambda item: item[0])
    csa_layer_ids = [layer_id for layer_id, _ in csa_info]

    token_bytes = 144
    device = torch.device("cuda")
    for _, indexer_op in csa_info:
        kv_cache = getattr(getattr(indexer_op, "k_cache", None), "kv_cache", None)
        if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() > 0:
            token_bytes = int(kv_cache.shape[-1])
            device = kv_cache.device
            break

    pool_size = int(os.environ.get("LMCACHE_INDEXER_POOL_SIZE", "2048"))
    io_workers = int(os.environ.get("LMCACHE_INDEXER_IO_WORKERS", "8"))
    max_seq_len = int(os.environ.get("LMCACHE_INDEXER_MAX_SEQ_LEN", "131072"))

    tutti_storage = _maybe_build_indexer_tutti_storage(
        tutti_loader=tutti_loader,
        csa_layer_ids=csa_layer_ids,
        token_bytes=token_bytes,
        max_seq_len=max_seq_len,
    )

    manager = IndexerSSDManager(
        csa_layer_ids=csa_layer_ids,
        store_dir=store_dir,
        pool_size=pool_size,
        token_bytes=token_bytes,
        max_seq_len=max_seq_len,
        io_workers=io_workers,
        device=device,
        tutti_storage=tutti_storage,
    )

    lookahead_policy = _csa_prefetch_lookahead_policy()
    disabled_csa_layers = (
        lookahead_policy.disabled_targets(csa_layer_ids)
        if lookahead_policy is not None
        else set()
    )
    for layer_id, indexer_op in csa_info:
        if layer_id in disabled_csa_layers:
            continue
        indexer_op.ssd_manager = manager
        indexer_op.csa_layer_id = layer_id

    _INDEXER_PREFETCH_MANAGER = manager
    set_indexer_ssd_manager(manager)

    for decoder_layer in decoder_layers:
        layer_id = getattr(decoder_layer, "layer_idx", -1)
        if isinstance(layer_id, int) and layer_id >= 0:
            register = getattr(manager, "register_decoder_layer", None)
            if callable(register):
                register(layer_id, decoder_layer)

    attached_decoders = 0
    early_overlap_hooks = 0
    source_prefetch: dict[int, tuple[int, int]] = {}
    source_prefetch = build_residual_prefetch_sources(
        csa_layer_ids,
        lookahead_policy,
    )
    manager.configure_prefetch_lookahead(
        {
            target: lookahead_policy.lookahead_for(target)
            for target in csa_layer_ids
        }
    )
    for decoder_layer in decoder_layers:
        decoder_layer_id = getattr(decoder_layer, "layer_idx", -1)
        next_csa, prefetch_level = source_prefetch.get(
            int(decoder_layer_id), (-1, 2)
        )
        attach = getattr(decoder_layer, "attach_indexer_prefetch", None)
        if callable(attach):
            # Native adjacent-L1 prediction was removed. The Python FFN hook
            # owns the one canonical L2 scheduling point.
            attach(manager, -1)
            attached_decoders += 1
        if _configure_decoder_csa_overlap(
            decoder_layer,
            manager,
            next_csa,
            prefetch_level,
        ):
            early_overlap_hooks += 1

    if os.getenv("LMCACHE_NSYS_FULL_CAPTURE", "0").lower() in {
        "1",
        "on",
        "true",
        "yes",
    }:
        indexed_decoders = [
            layer
            for layer in decoder_layers
            if isinstance(getattr(layer, "layer_idx", None), int)
        ]
        if indexed_decoders:
            last_decoder = max(
                indexed_decoders,
                key=lambda layer: int(layer.layer_idx),
            )
            last_decoder._lmcache_nsys_full_capture_last_layer = True
            _install_decoder_forward_position_hook(last_decoder)
            logger.info(
                "IndexerSSDManager: full nsys capture stop hook attached "
                "to decoder layer %d",
                int(last_decoder.layer_idx),
            )

    if attached_decoders != len(decoder_layers) and early_overlap_hooks == 0:
        logger.warning(
            "CSA prefetch requested, but only %d/%d DeepSeek decoder layers "
            "expose attach_indexer_prefetch(); FFN-window prefetch may be "
            "disabled for the remaining layers",
            attached_decoders,
            len(decoder_layers),
        )

    logger.info(
        "IndexerSSDManager: enabled CSA prefetch on %d native decoder hooks, "
        "%d FFN-entry early-overlap hooks, managed_indexers=%d "
        "native_indexers=%d pool_size=%d, store=%s",
        attached_decoders,
        early_overlap_hooks,
        len(csa_layer_ids) - len(disabled_csa_layers),
        len(disabled_csa_layers),
        pool_size,
        store_dir,
    )
    logger.info(
        "IndexerSSDManager: canonical L2 policy=%s two_layer_targets=%s "
        "demand_only_targets=%s",
        lookahead_policy.specification,
        sorted(lookahead_policy.two_layer_targets(csa_layer_ids)),
        sorted(lookahead_policy.disabled_targets(csa_layer_ids)),
    )


def _csa_attention_kv_prefetch_enabled() -> bool:
    """Return whether the canonical Tutti CSA/HCA pipeline is enabled.

    ``LMCACHE_INDEXER_ENABLE_PREFETCH`` is the single feature gate. Manager
    attachment and retrieve filtering are enabled or disabled together.
    """
    return _indexer_prefetch_enabled()


def _attach_csa_attention_kv_prefetch(tutti_loader: Optional[Any] = None) -> None:
    """Attach the CSA attention KV prefetcher when enabled by environment.

    Wires up a :class:`CSAAttentionKVPrefetchManager` covering all DSv4 CSA
    decoder layers, registers each layer's vLLM MLA K cache tensor, patches
    each ``DeepseekV4Indexer.forward`` so the true_topk drives miss
    correction + drain, and attaches the manager onto the active
    :class:`IndexerSSDManager` so predicted top-K is mirrored across the
    indexer-cache and attention-KV prefetchers.

    Args:
        tutti_loader: Active :class:`TuttiDirectLoader`.  Reads route through
            Tutti GPU-direct DMA; when no loader is available the manager
            is not attached and the legacy synchronous scatter remains in
            effect.
    """
    global _CSA_ATTENTION_KV_PREFETCH_MANAGER

    if not _csa_attention_kv_prefetch_enabled():
        return
    if tutti_loader is None:
        logger.debug(
            "CSA attention KV prefetch waiting for Tutti loader before attach"
        )
        return
    if _CSA_ATTENTION_KV_PREFETCH_MANAGER is not None:
        return

    try:
        from lmcache.v1.csa_attention_kv_prefetch_manager import (
            CSAAttentionKVPrefetchManager,
            set_csa_attention_kv_prefetch_manager,
        )
        from lmcache.v1.indexer_ssd_manager import get_indexer_ssd_manager
    except ImportError as exc:
        logger.warning(
            "CSA attention KV prefetcher is unavailable: %s; skipping attach",
            exc,
        )
        return

    indexer_manager = get_indexer_ssd_manager()
    if indexer_manager is None:
        logger.warning(
            "CSA attention KV prefetch requested but no IndexerSSDManager is "
            "attached; skipping attach"
        )
        return

    decoder_layers = _deepseek_decoder_layers()
    if not decoder_layers:
        logger.warning(
            "CSA attention KV prefetch requested but no DeepSeek decoder "
            "layers were found; skipping attach"
        )
        return

    csa_layer_entries: list[tuple[int, Any, Any, Any]] = []
    for decoder_layer in decoder_layers:
        layer_id = getattr(decoder_layer, "layer_idx", -1)
        if not isinstance(layer_id, int) or layer_id < 0:
            continue
        attention = _decoder_csa_attention(decoder_layer)
        if attention is None:
            continue
        indexer_op = _decoder_csa_indexer(decoder_layer)
        if indexer_op is None:
            continue
        csa_layer_entries.append((layer_id, attention, indexer_op, decoder_layer))

    if not csa_layer_entries:
        logger.warning(
            "CSA attention KV prefetch requested but no CSA layers exposed "
            "both attention.kv_cache and indexer modules; skipping attach"
        )
        return

    csa_layer_entries.sort(key=lambda entry: entry[0])
    csa_layer_ids = [layer_id for layer_id, _, _, _ in csa_layer_entries]
    # Probe the first attention to derive token_bytes/compressed_block_size.
    probe_kv_cache = getattr(csa_layer_entries[0][1], "kv_cache", None)
    if not isinstance(probe_kv_cache, torch.Tensor) or probe_kv_cache.ndim != 3:
        logger.warning(
            "CSA attention KV prefetch: expected [num_blocks, block_size, "
            "token_bytes] K cache tensor; got shape %s; skipping attach",
            None if not isinstance(probe_kv_cache, torch.Tensor) else
            tuple(probe_kv_cache.shape),
        )
        return
    compressed_block_size = int(probe_kv_cache.shape[1])
    token_bytes = int(probe_kv_cache.shape[2])

    manager = CSAAttentionKVPrefetchManager(
        tutti_loader=tutti_loader,
        csa_layer_ids=csa_layer_ids,
        compressed_block_size=compressed_block_size,
        token_bytes=token_bytes,
    )

    patched_layers = 0
    for layer_id, attention, indexer_op, decoder_layer in csa_layer_entries:
        try:
            kv_cache = getattr(attention, "kv_cache", None)
            if not isinstance(kv_cache, torch.Tensor):
                continue
            manager.register_layer(int(layer_id), kv_cache)
            # patch_indexer_forward wraps the SparseAttnIndexer op exposed
            # via ``attn.indexer.indexer_op`` (the leaf module that runs the
            # actual Lightning Indexer kernel and returns top-K indices).
            manager.patch_indexer_forward(indexer_op, int(layer_id))
            patched_layers += 1
        except Exception:
            logger.exception(
                "Failed to register CSA attention KV prefetch for layer %d",
                layer_id,
            )

    if patched_layers == 0:
        logger.warning(
            "CSA attention KV prefetch attach failed for all %d CSA layers; "
            "rolling back",
            len(csa_layer_entries),
        )
        manager.close()
        return

    # V28: register HCA layers so the layer-major walker also owns their
    # attention-KV bytes.  Gate is the connector's per-layer
    # wait_for_layer_load hook (HCA has no indexer forward to patch).
    hca_registered = 0
    if _env_flag("LMCACHE_DSV4_HCA_WALKER"):
        # One-shot layer map: for every decoder layer log the best
        # candidate's compress_ratio and kv_cache population so a finder
        # mismatch (wrong cr constant, unmaterialised cache, different
        # attribute path) is diagnosable from a single line.
        layer_map: list[str] = []
        for decoder_layer in decoder_layers:
            lid = getattr(decoder_layer, "layer_idx", -1)
            attn = getattr(decoder_layer, "self_attn", None) or getattr(
                decoder_layer, "attn", None
            )
            wrapper = getattr(attn, "mla_attn", None)
            best = ""
            for cand in (getattr(wrapper, "mla_attn", None), wrapper, attn):
                if cand is None:
                    continue
                cr = getattr(cand, "compress_ratio", None)
                kv = getattr(cand, "kv_cache", None)
                numel = kv.numel() if isinstance(kv, torch.Tensor) else -1
                best = f"{lid}:cr{cr}:kv{numel}"
                if isinstance(cr, int) and cr > 1:
                    break
            layer_map.append(best)
        logger.info("HCA_WALKER_LAYER_MAP %s", " ".join(layer_map))
        for decoder_layer in decoder_layers:
            layer_id = getattr(decoder_layer, "layer_idx", -1)
            if not isinstance(layer_id, int) or layer_id < 0:
                continue
            hca_attention = _decoder_hca_attention(decoder_layer)
            if hca_attention is None:
                continue
            kv_cache = getattr(hca_attention, "kv_cache", None)
            if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
                continue
            try:
                manager.register_hca_layer(int(layer_id), kv_cache)
                hca_registered += 1
            except Exception:
                logger.exception(
                    "Failed to register HCA layer %d with the CSA/HCA "
                    "prefetch walker",
                    layer_id,
                )
        logger.info(
            "CSAAttentionKVPrefetchManager: HCA walker registered %d HCA "
            "layers",
            hca_registered,
        )

    indexer_manager.attach_csa_attention_kv_manager(manager)
    if hca_registered:
        hca_set = set(manager.hca_layer_ids)
        for decoder_layer in decoder_layers:
            layer_id = getattr(decoder_layer, "layer_idx", -1)
            if not isinstance(layer_id, int):
                continue
            native_attach = getattr(decoder_layer, "attach_hca_prefetch", None)
            if callable(native_attach):
                try:
                    # The legacy native mapping chooses the next HCA anywhere
                    # ahead. Disable it; the Python staged hook below owns the
                    # exact adjacent-layer schedule through the unified manager.
                    native_attach(None, -1, -1)
                except Exception:
                    logger.debug(
                        "Could not detach legacy native HCA prefetch for "
                        "layer %d",
                        layer_id,
                        exc_info=True,
                    )
            next_hca = layer_id + 1
            if next_hca in hca_set:
                _configure_decoder_hca_overlap(
                    decoder_layer,
                    indexer_manager,
                    next_hca,
                    (next_hca,),
                )
            else:
                # Disable the legacy "next HCA anywhere ahead" mapping. In
                # staged mode HCA is deterministic and is submitted only from
                # its immediately preceding CSA FFN, so it cannot steal I/O
                # from a nearer target CSA prediction deadline.
                _configure_decoder_hca_overlap(
                    decoder_layer,
                    None,
                    -1,
                    (),
                )
    set_csa_attention_kv_prefetch_manager(manager)
    _CSA_ATTENTION_KV_PREFETCH_MANAGER = manager
    logger.info(
        "CSAAttentionKVPrefetchManager: attached with %d/%d CSA layers, "
        "compressed_block_size=%d token_bytes=%d",
        patched_layers,
        len(csa_layer_entries),
        compressed_block_size,
        token_bytes,
    )


def _ensure_csa_attention_kv_prefetch_attached(
    tutti_loader: Optional[Any],
) -> Any:
    """Lazy-attach the CSA attention KV prefetcher when Tutti becomes ready.

    ``_attach_csa_attention_kv_prefetch`` is initially called from
    ``post_init`` but Tutti's loader is bootstrapped asynchronously and is
    often still ``None`` at post_init time.  Retrieve paths call this helper
    to retry the attach once the loader has finished warming up.

    Returns the active :class:`CSAAttentionKVPrefetchManager` or ``None``.
    """
    global _CSA_ATTENTION_KV_PREFETCH_MANAGER

    if _CSA_ATTENTION_KV_PREFETCH_MANAGER is not None:
        return _CSA_ATTENTION_KV_PREFETCH_MANAGER
    if tutti_loader is None:
        return None
    _attach_csa_attention_kv_prefetch(tutti_loader=tutti_loader)
    return _CSA_ATTENTION_KV_PREFETCH_MANAGER


def _prefetch_set_pending_loader(tutti_loader: Any) -> None:
    """Record the Tutti loader for later main-thread prefetch attachment.

    Registered with the cache engine via ``register_tutti_loader_ready_callback``
    and may be invoked from the Tutti warmup daemon thread. It therefore only
    stores the loader reference under a lock and performs no CUDA work; the
    actual attach (which allocates CUDA bitmaps and patches indexer forwards)
    runs on the main thread in :func:`_maybe_lazy_attach_csa_prefetch`.

    Args:
        tutti_loader: The freshly created :class:`TuttiDirectLoader`.
    """
    global _CSA_PREFETCH_PENDING_LOADER, _INDEXER_PREFETCH_PENDING_LOADER
    with _CSA_PREFETCH_PENDING_LOCK:
        _CSA_PREFETCH_PENDING_LOADER = tutti_loader
    with _INDEXER_PREFETCH_PENDING_LOCK:
        _INDEXER_PREFETCH_PENDING_LOADER = tutti_loader


def _maybe_lazy_attach_indexer_prefetch(engine: Any) -> None:
    """Attach the Tutti indexer manager after the loader becomes available.

    Args:
        engine: Active LMCache engine used as a fallback loader source.
    """
    if _INDEXER_PREFETCH_MANAGER is not None:
        return
    with _INDEXER_PREFETCH_PENDING_LOCK:
        loader = _INDEXER_PREFETCH_PENDING_LOADER
    if loader is None:
        loader = getattr(engine, "_tutti_loader", None) if engine is not None else None
    if loader is None:
        return
    _attach_indexer_prefetch(tutti_loader=loader)


def _maybe_lazy_attach_csa_prefetch(engine: Any) -> None:
    """Attach the CSA attention-KV prefetch manager on the main thread.

    Called per request from ``start_load_kv`` before the model forward. When the
    Tutti loader was created after ``register_kv_caches`` (the common case, due
    to the warmup delay), this performs the deferred attach exactly once. It is
    a fast no-op after the manager is attached.

    Args:
        engine: The active LMCache engine, or ``None``.
    """
    if _CSA_ATTENTION_KV_PREFETCH_MANAGER is not None:
        return
    with _CSA_PREFETCH_PENDING_LOCK:
        loader = _CSA_PREFETCH_PENDING_LOADER
    if loader is None:
        # The callback may not have fired yet (or was never registered); fall
        # back to reading the loader directly off the engine.
        loader = getattr(engine, "_tutti_loader", None) if engine is not None else None
    if loader is None:
        return
    _ensure_csa_attention_kv_prefetch_attached(loader)


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0
    total_chunks: int = 0
    receiver_query_port: Optional[list[int]] = None


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params and sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    allocated_block_ids: list[int]
    allocated_block_ids_by_group: tuple[list[int], ...] = field(
        default_factory=tuple
    )

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    # The number of tokens that are cached in LMCache for this request
    num_lmcache_cached_tokens: int = 0

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
        block_size: Optional[int] = None,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # tuple[list[int]]
        # Need to check the type of request.block_ids

        block_ids_by_group = _normalize_block_ids_by_group(new_request.block_ids)
        unfolded_block_ids = _select_primary_block_ids(
            block_ids_by_group,
            num_tokens_to_compute,
            block_size,
        )

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            allocated_block_ids_by_group=block_ids_by_group,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
            num_lmcache_cached_tokens=lmcache_cached_tokens,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
        preempted: bool = False,
        lmcache_cached_tokens: int = 0,
        vllm_cached_tokens: int = 0,
        all_token_ids: Optional[list[int]] = None,
        block_size: Optional[int] = None,
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again

        vllm_cached_tokens: the number of tokens that are cached in vLLM
        is only used for preempted requests
        all_token_ids: the full token list from the vLLM request, used to
        restore token_ids for preempted requests to ensure chunk keys match
        block_size: the logical vLLM block size used to select the primary HMA
        KV group for legacy slot_mapping metadata
        """

        new_block_ids_by_group = _normalize_block_ids_by_group(new_block_ids)
        new_block_ids = _select_primary_block_ids(
            new_block_ids_by_group,
            len(new_token_ids),
            block_size=block_size,
        )

        if preempted:
            assert all_token_ids is not None, (
                f"Preempted request {self.req_id} has no all_token_ids"
            )
            # the block ids will change after preemption
            self.allocated_block_ids = new_block_ids
            self.allocated_block_ids_by_group = new_block_ids_by_group
            # reset the number of saved tokens
            self.num_saved_tokens = lmcache_cached_tokens
            num_computed_tokens = max(lmcache_cached_tokens, vllm_cached_tokens)

            # FIX: For preempted requests, restore token_ids from the full
            # token list to ensure chunk keys match what was used during
            # lookup. The lookup uses request.all_token_ids, so we need the
            # same tokens for retrieve.
            num_tokens_needed = max(
                num_computed_tokens + len(new_token_ids),
                lmcache_cached_tokens,
            )
            self.token_ids = all_token_ids[:num_tokens_needed]
        else:
            self.allocated_block_ids.extend(new_block_ids)
            if new_block_ids_by_group:
                if not self.allocated_block_ids_by_group:
                    self.allocated_block_ids_by_group = tuple(
                        [] for _ in new_block_ids_by_group
                    )
                if len(new_block_ids_by_group) != len(
                    self.allocated_block_ids_by_group
                ):
                    logger.warning(
                        "Request %s KV group count changed from %d to %d; "
                        "using current scheduler block ids",
                        self.req_id,
                        len(self.allocated_block_ids_by_group),
                        len(new_block_ids_by_group),
                    )
                    self.allocated_block_ids_by_group = new_block_ids_by_group
                else:
                    self.allocated_block_ids_by_group = tuple(
                        list(old) + list(new)
                        for old, new in zip(
                            self.allocated_block_ids_by_group,
                            new_block_ids_by_group,
                            strict=True,
                        )
                    )
            self.token_ids.extend(new_token_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor
    # vLLM HMA block ids in kv_cache_group order. The legacy slot_mapping above
    # is built from the full-MLA group; GPU transfer can use this richer form
    # to address non-full groups with their own block tables.
    block_ids_by_group: tuple[list[int], ...] = field(default_factory=tuple)

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len >= tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        token_ids = input_token_ids[:num_tokens_to_save]

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            # TODO: Optimize this
            token_ids = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids.tolist()

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size:
            max_capacity = num_blocks * block_size
            raise ValueError(
                "Request %s: num_tokens=%d exceeds primary-group capacity=%d "
                "(num_blocks=%d, block_size=%d). This means LMCache is using "
                "the wrong vLLM HMA block-id group or block size; refusing to "
                "truncate token_ids because that would create a false partial "
                "SSD hit.",
                tracker.req_id,
                len(token_ids),
                max_capacity,
                num_blocks,
                block_size,
            )

        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: log if the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens (%d cached in vLLM) for request %s",
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
                tracker.req_id,
            )

        # For disagg requests, compute total_chunks for sender admission control.
        if tracker.disagg_spec is not None and tracker.disagg_spec.total_chunks == 0:
            # Only compute once (on first batch)
            total_chunks_for_req = math.ceil(tracker.prompt_len / lmcache_chunk_size)
            tracker.disagg_spec.total_chunks = total_chunks_for_req

        # Note: We keep load_spec even when can_load=False to pass metrics to worker
        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            block_ids_by_group=tracker.allocated_block_ids_by_group,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
        )


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        _install_scheduler_hma_invalid_block_patch()
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size

        # Load and configure LMCache config
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        self._apply_extra_config(config, vllm_config)
        self.config = config

        service_factory = VllmServiceFactory(config, vllm_config, role.name.lower())
        self._manager = LMCacheManager(config, service_factory, connector=self)

        # Start services managed by LMCacheManager
        self._manager.start_services()

        # Initialize connector-specific state
        self._init_connector_state(role, vllm_config, config)

        # Setup metrics for monitoring data structures
        self._setup_metrics()

        logger.info(
            "LMCache initialized for role %s with version %s, "
            "vllm version %s, lmcache cache_engine metadata: %s",
            role,
            utils.get_version(),
            VLLM_VERSION,
            getattr(self.lmcache_engine, "metadata", None),
        )

    def _apply_extra_config(
        self, config: LMCacheEngineConfig, vllm_config: "VllmConfig"
    ) -> None:
        """Apply extra config from vLLM to LMCache config."""
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            "Updated config %s from vLLM extra config",
                            config_key,
                        )

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        """Initialize connector-specific state variables."""
        _install_deepseek_decoder_registry_hook()

        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            tuple[Generator[Optional[torch.Tensor], None, None], ReqMeta]
        ] = []
        self._layerwise_save_storers: dict[
            str, Generator[Optional[torch.Tensor], None, None]
        ] = {}
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()

        # Role-specific initialization
        if role == KVConnectorRole.SCHEDULER:
            self._unfinished_requests: dict[str, "Request"] = {}
        else:
            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending

            if self.enable_blending:
                assert self.lmcache_engine is not None
                assert self.lmcache_engine.gpu_connector is not None, (
                    "GPU connector must be available for blending"
                )
                self.blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
                )

        # Legacy compatibility check
        self._check_legacy_register_kv_caches()

        self.kv_caches: dict[str, torch.Tensor] = {}
        self._block_size = _engine_logical_block_size(vllm_config, self._parent)
        self.load_specs: dict[str, LoadSpec] = {}
        self.kv_cache_manager: Optional["KVCacheManager"] = None
        self._request_trackers: dict[str, RequestTracker] = {}

        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
        )

        self._lmcache_chunk_size = config.chunk_size

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))
        self._requests_priority: dict[str, int] = {}
        self._invalid_block_ids: set[int] = set()
        self._kv_cache_layer_names: tuple[str, ...] = ()
        self._vllm_kv_cache_group_layer_names: tuple[tuple[str, ...], ...] = ()
        self._vllm_kv_cache_group_block_sizes: tuple[int, ...] = ()

    def _check_legacy_register_kv_caches(self) -> None:
        """Check for legacy connector without register_kv_caches implementation."""
        if self.lmcache_engine is None:
            return

        child_class = self._parent.__class__
        parent_class = KVConnectorBase_V1
        child_method = getattr(child_class, "register_kv_caches", None)
        parent_method = getattr(parent_class, "register_kv_caches", None)

        if child_method is None or parent_method is None:
            implements = False
        else:
            implements = child_method is not parent_method

        if not implements:
            logger.warning(
                "Please use the latest lmcache connector, otherwise some "
                "features may not work, such as DSA"
            )
            self._manager.post_init()

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance from manager."""
        return self._manager.lmcache_engine

    @property
    def lmcache_engine_metadata(self):
        """Get the LMCache engine metadata from manager."""
        return self._manager.lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional["LookupClientInterface"]:
        """Get the lookup client from manager."""
        return self._manager.lookup_client

    @property
    def lookup_server(self):
        """Get the lookup server from manager."""
        return self._manager.lookup_server

    def _setup_metrics(self):
        """Setup metrics for monitoring data structures in the connector."""
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is None:
            logger.warning(
                "PrometheusLogger is not initialized, "
                "connector metrics will not be collected"
            )
            return

        # Set up metrics for scheduler-specific and general data structures
        metrics_map = {
            "_unfinished_requests": "scheduler_unfinished_requests_count",
            "load_specs": "connector_load_specs_count",
            "_request_trackers": "connector_request_trackers_count",
            "kv_caches": "connector_kv_caches_count",
            "layerwise_retrievers": "connector_layerwise_retrievers_count",
            "_invalid_block_ids": "connector_invalid_block_ids_count",
            "_requests_priority": "connector_requests_priority_count",
        }

        for attr_name, metric_name in metrics_map.items():
            if hasattr(self, attr_name):
                metric = getattr(prometheus_logger, metric_name)
                # Use a default argument in the lambda to capture
                # the current value of `attr_name`
                # to avoid issues with late binding in closures.
                metric.set_function(lambda name=attr_name: len(getattr(self, name)))

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    # TODO(chunxiaozheng): in the latest lmcache_connector, we use `register_kv_caches`
    #  to init self.kv_caches, we keep it in order to be compatible with old versions
    #  and will be removed in the future.
    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

    ####################
    # Worker side APIs
    ####################
    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        self._kv_cache_layer_names = tuple(kv_caches.keys())
        self._capture_vllm_hma_layout()
        self._manager.post_init()
        engine_tutti_loader = (
            getattr(self.lmcache_engine, "_tutti_loader", None)
            if self.lmcache_engine is not None
            else None
        )
        _attach_indexer_prefetch(tutti_loader=engine_tutti_loader)
        _attach_csa_attention_kv_prefetch(tutti_loader=engine_tutti_loader)
        # The Tutti loader is usually still None here (created lazily after the
        # warmup delay), so the attach above no-ops. Register a callback so the
        # CSA attention-KV manager attaches on the main thread once the loader
        # is created. See _maybe_lazy_attach_csa_prefetch / start_load_kv.
        if (
            self.lmcache_engine is not None
            and (
                (
                    _CSA_ATTENTION_KV_PREFETCH_MANAGER is None
                    and _csa_attention_kv_prefetch_enabled()
                )
                or (
                    _INDEXER_PREFETCH_MANAGER is None
                    and _indexer_prefetch_enabled()
                    and _indexer_tutti_backend_enabled()
                )
            )
        ):
            register_cb = getattr(
                self.lmcache_engine,
                "register_tutti_loader_ready_callback",
                None,
            )
            if callable(register_cb):
                register_cb(_prefetch_set_pending_loader)

    def _capture_vllm_hma_layout(self) -> None:
        """Capture vLLM HMA group metadata for LMCache GPU transfers."""
        kv_cache_config = getattr(self._parent, "_kv_cache_config", None)
        groups = getattr(kv_cache_config, "kv_cache_groups", ()) or ()

        group_layer_names: list[tuple[str, ...]] = []
        group_block_sizes: list[int] = []
        layer_name_to_group: dict[str, int] = {}
        layer_name_to_block_size: dict[str, int] = {}
        for group in groups:
            names = tuple(getattr(group, "layer_names", ()))
            group_layer_names.append(names)
            group_idx = len(group_layer_names) - 1
            layer_name_to_group.update((name, group_idx) for name in names)
            spec = getattr(group, "kv_cache_spec", None)
            block_size = int(getattr(spec, "block_size", self._block_size))
            group_block_sizes.append(block_size)
            layer_name_to_block_size.update((name, block_size) for name in names)

        self._vllm_kv_cache_group_layer_names = tuple(group_layer_names)
        self._vllm_kv_cache_group_block_sizes = tuple(group_block_sizes)
        if self.lmcache_engine is not None:
            gpu_connector = getattr(self.lmcache_engine, "gpu_connector", None)
            layout_hints = getattr(gpu_connector, "layout_hints", None)
            if isinstance(layout_hints, dict) and layer_name_to_group:
                layout_hints["vllm_kv_cache_group_ids"] = [
                    layer_name_to_group.get(name, -1)
                    for name in self._kv_cache_layer_names
                ]
                layout_hints["vllm_kv_cache_layer_block_sizes"] = [
                    layer_name_to_block_size.get(name, self._block_size)
                    for name in self._kv_cache_layer_names
                ]
        if group_layer_names:
            logger.info(
                "LMCache captured vLLM HMA layout: kv_cache_layers=%d, "
                "groups=%s",
                len(self._kv_cache_layer_names),
                [
                    {
                        "group": idx,
                        "layers": len(names),
                        "block_size": group_block_sizes[idx],
                    }
                    for idx, names in enumerate(group_layer_names)
                ],
            )

    def _hma_transfer_kwargs(self, request: ReqMeta) -> dict[str, Any]:
        """Return HMA metadata consumed by VLLMPagedMemGPUConnectorV3."""
        if not request.block_ids_by_group:
            return {}
        return {
            "block_ids_by_group": request.block_ids_by_group,
            "kv_cache_layer_names": self._kv_cache_layer_names,
            "vllm_kv_cache_group_layer_names": (
                self._vllm_kv_cache_group_layer_names
            ),
            "vllm_kv_cache_group_block_sizes": (
                self._vllm_kv_cache_group_block_sizes
            ),
        }

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation
        """
        try:
            self._start_load_kv_impl(forward_context, **kwargs)
        except Exception:
            import traceback as _tb
            logger.error(
                "start_load_kv unhandled exception (worker traceback):\n%s",
                _tb.format_exc(),
            )
            raise

    def _start_load_kv_impl(
        self, forward_context: "ForwardContext", **kwargs
    ) -> None:
        self.current_layer = 0

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        if not isinstance(metadata, LMCacheConnectorMetadata):
            raise ValueError(
                f"Expected LMCacheConnectorMetadata, got {type(metadata)}"
            )

        if len(self.kv_caches) == 0:
            raise ValueError(
                "start_load_kv: kv_caches still empty after init attempt"
            )
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        if self.lmcache_engine is None:
            logger.warning(
                "start_load_kv: lmcache_engine is None (degraded mode), skipping KV load"
            )
            return

        # Attach the CSA attention-KV prefetch manager if the Tutti loader
        # became ready after register_kv_caches. Runs on the main thread before
        # the model forward; idempotent no-op once attached.
        if _INDEXER_PREFETCH_MANAGER is None:
            _maybe_lazy_attach_indexer_prefetch(self.lmcache_engine)
        if _CSA_ATTENTION_KV_PREFETCH_MANAGER is None:
            _maybe_lazy_attach_csa_prefetch(self.lmcache_engine)

        full_nsys_manager = _CSA_ATTENTION_KV_PREFETCH_MANAGER
        if full_nsys_manager is not None:
            start_capture = getattr(
                full_nsys_manager,
                "start_full_nsys_capture_for_request",
                None,
            )
            if callable(start_capture):
                for request in metadata.requests:
                    load_spec = request.load_spec
                    if load_spec is not None and load_spec.can_load:
                        start_capture(str(request.req_id))
                        break

        self.layerwise_retrievers = []

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            # Update metrics for all requests that have a load_spec
            if request.load_spec is not None:
                self._stats_monitor.update_interval_vllm_hit_tokens(
                    request.load_spec.vllm_cached_tokens
                )
                self._stats_monitor.update_interval_prompt_tokens(
                    len(request.token_ids)
                )

            if request.load_spec is None or not request.load_spec.can_load:
                continue

            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.to(self.device)
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            hma_kwargs = self._hma_transfer_kwargs(request)
            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                        **hma_kwargs,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append((layerwise_retriever, request))
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                    **hma_kwargs,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s"
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!",
                        request.req_id,
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    """
                    Report failed block IDs in case of partial failure.
                    """
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping[:lmcache_cached_tokens],
                    )
                    self._invalid_block_ids.update(missing_blocks)

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> set[int]:
        """Record block IDs associated with failed load attempts.

        Args:
            request_id: request id from vLLM.
            expected_mask: Boolean tensor indicating which tokens were expected to
                be loaded from LMCache. True means the token should be loaded,
                False means the token is already cached in vLLM and does not need
                to be loaded from LMCache.
            ret_mask: Boolean tensor indicating which tokens were actually
                successfully retrieved from LMCache. True means the token was
                successfully loaded. For example, if 256 tokens are expected to be
                loaded, but only 192 tokens are successfully loaded, then the
                ret_mask will be a tensor of 256 items like [T, T, ..., F, F, ...]
                where the first 192 elements are True and the last 64 elements
                are False.
            slot_mapping: Tensor indicating slot IDs for each token. The block
                ID is computed by dividing the slot ID by the block size.

        Example:
            expected_mask = [F, T, T, T] meaning the 1st is in vLLM cache
            ret_mask = [F, T, F, F] meaning failure from loading the 3rd
            missing_mask = expected_mask & ~ret_mask = [F, F, T, T]
            missing_indices = [2, 3]
            then missing_blocks is calculated from slot_mapping and missing_indices

        Returns:
            set[int]: Set of block IDs that failed to load.
        """

        if expected_mask.numel() == 0:
            return set()

        expected_mask_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
        ret_mask_cpu = ret_mask.to(device="cpu", dtype=torch.bool)

        if ret_mask_cpu.shape[0] != expected_mask_cpu.shape[0]:
            logger.debug("expected_mask_cpu.shape[0] != ret_mask_cpu.shape[0]")
            return set()

        missing_mask = expected_mask_cpu & ~ret_mask_cpu
        if not torch.any(missing_mask):
            return set()

        missing_indices = torch.nonzero(missing_mask, as_tuple=False).view(-1)
        if missing_indices.numel() == 0:
            return set()

        slot_mapping_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
        if slot_mapping_cpu.shape[0] > missing_mask.shape[0]:
            slot_mapping_cpu = slot_mapping_cpu[: missing_mask.shape[0]]

        missing_blocks_tensor = torch.unique(
            slot_mapping_cpu[missing_indices] // self._block_size
        )
        missing_blocks = {int(block.item()) for block in missing_blocks_tensor}

        if not missing_blocks:
            return set()

        logger.warning(
            "Request %s failed to load %d tokens across %d blocks",
            request_id,
            missing_indices.numel(),
            len(missing_blocks),
        )
        return missing_blocks

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        # V28 HCA walker gate: HCA layers have no indexer forward to patch,
        # so this per-layer connector hook blocks until the layer-major
        # walker has landed the layer's attention-KV bytes.  No-op for CSA
        # layers (their gate is the patched indexer forward) and for layers
        # the walker does not own.
        manager = _CSA_ATTENTION_KV_PREFETCH_MANAGER
        if manager is not None and _env_flag("LMCACHE_DSV4_HCA_WALKER"):
            layer_id = _layer_idx_from_prefix(layer_name)
            if layer_id >= 0 and layer_id in getattr(
                manager, "hca_layer_ids", ()
            ):
                try:
                    if not manager.wait_for_layer(layer_id):
                        raise RuntimeError(
                            f"HCA walker did not complete layer {layer_id}"
                        )
                except Exception as exc:
                    logger.exception(
                        "HCA walker gate failed for layer %s; aborting "
                        "request before attention",
                        layer_name,
                    )
                    raise RuntimeError(
                        f"HCA KV is unavailable for {layer_name}"
                    ) from exc

        if self.layerwise_retrievers:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        # Wait for the layer to be loaded
        for layerwise_retriever, request in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)
            if ret_token_mask is not None:
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")

        if self.layerwise_retrievers:
            self.current_layer += 1

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        if self.lmcache_engine is None:
            logger.warning("save_kv_layer: lmcache_engine is None, skipping save")
            return

        if not self.use_layerwise and not self.layerwise_retrievers:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        if not isinstance(connector_metadata, LMCacheConnectorMetadata):
            logger.warning(
                "save_kv_layer: expected LMCacheConnectorMetadata, got %s",
                type(connector_metadata),
            )
            return

        if len(self.kv_caches) == 0:
            logger.warning("save_kv_layer: kv_caches is empty, skipping")
            return

        kvcaches = list(self.kv_caches.values())
        is_first = True

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            layerwise_storer = self._layerwise_save_storers.get(request.req_id)
            if layerwise_storer is None:
                token_ids = request.token_ids
                assert isinstance(token_ids, list)

                slot_mapping = request.slot_mapping
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.to(self.device)

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    assert save_spec is not None
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.debug(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                    req_id=request.req_id,
                    **self._hma_transfer_kwargs(request),
                )
                self._layerwise_save_storers[request.req_id] = layerwise_storer
                if is_first:
                    is_first = False

            next(layerwise_storer)

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            # But still need to unpin the kv caches according to req_id
            # to balance the pin count from contains()
            assert self.lmcache_engine is not None, (
                "LMCacheEngine must be initialized to unpin requests."
            )
            for request in connector_metadata.requests:
                self.lmcache_engine.lookup_unpin(request.req_id)

            return

        if self.use_layerwise:
            for request in connector_metadata.requests:
                layerwise_storer = self._layerwise_save_storers.pop(
                    request.req_id, None
                )
                if layerwise_storer is not None:
                    next(layerwise_storer)
                # unpin the kv caches according to req_id
                self.lmcache_engine.lookup_unpin(request.req_id)
            return

        if len(self.kv_caches) == 0:
            logger.warning("wait_for_save: kv_caches is empty, skipping")
            return
        kvcaches = list(self.kv_caches.values())

        if self.lmcache_engine is None:
            logger.warning("wait_for_save: lmcache_engine is None, skipping save")
            return

        # Probe decoder cache before store if bidirectional mode is enabled
        bidir_enabled = getattr(self.config, "pd_bidirectional", False)

        for request in connector_metadata.requests:
            # unpin the kv caches according to req_id
            self.lmcache_engine.lookup_unpin(request.req_id)

            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.to(self.device)

            skip_leading_tokens = save_spec.skip_leading_tokens
            # shared storage disaggregation will not have a disagg_spec passed in
            if self.kv_role == "kv_producer" and request.disagg_spec:
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.debug(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                if not self.enable_blending:
                    token_len = len(token_ids)
                    aligned_token_len = (
                        token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

            # Probe decoder cache before store
            if bidir_enabled and request.disagg_spec is not None:
                try:
                    self._probe_decoder_cache(request, token_ids)
                except Exception as e:
                    logger.warning(
                        "Bidirectional NIXL cache probe failed for %s: %s",
                        request.req_id,
                        e,
                    )

            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
                req_id=request.req_id,
                is_last_prefill=is_last_prefill,
                **self._hma_transfer_kwargs(request),
            )

            # Probe decoder cache after store
            if (
                bidir_enabled
                and request.disagg_spec is not None
                and request.disagg_spec.receiver_query_port is not None
            ):
                try:
                    self._probe_decoder_cache(request, token_ids)
                except Exception as e:
                    logger.warning(
                        "Bidirectional NIXL cache probe failed for %s: %s",
                        request.req_id,
                        e,
                    )

            # Update skip_leading_tokens only on last rank to ensure
            # each PP stage stores its own KV cache
            if get_pp_group().is_last_rank:
                # NOTE(Jiayi): We assume all tokens are saved
                save_spec.skip_leading_tokens = len(token_ids)
                if request.disagg_spec:
                    request.disagg_spec.num_transferred_tokens = len(token_ids)

    def _probe_decoder_cache(self, request: ReqMeta, token_ids: list[int]) -> None:
        """Query the decoder's cache to check which blocks are already cached.

        This is the bidirectional NIXL cache probe: the prefiller queries the
        decoder via ZMQ to find out which KV blocks are already in the
        decoder's GPU memory. This validates the cache query channel works
        E2E through the real inference path.

        In the future, this information can be used to skip prefill
        computation for cached blocks.
        """
        sm = self.lmcache_engine.storage_manager  # type: ignore[union-attr]
        if sm is None or sm.allocator_backend is None:
            return
        pd_backend = sm.allocator_backend
        if not hasattr(pd_backend, "query_remote_cache"):
            return
        if not hasattr(pd_backend, "cache_query_sockets"):
            return

        # Get query port from LMCache config (pd_peer_query_port)
        query_ports = self.config.pd_peer_query_port
        if query_ports is None:
            return

        # Build cache keys using the token database's process_tokens
        td = self.lmcache_engine.token_database  # type: ignore[union-attr]
        if td is None:
            return

        chunk_keys = []
        for _start, _end, key in td.process_tokens(
            tokens=token_ids, mask=None, make_key=True
        ):
            chunk_keys.append(key)

        if not chunk_keys:
            return

        # Build receiver_id from disagg_spec
        disagg = request.disagg_spec
        init_port = disagg.receiver_init_port  # type: ignore[union-attr]
        if isinstance(init_port, list):
            init_port = init_port[pd_backend.tp_rank]  # type: ignore[union-attr]
        receiver_id = disagg.receiver_host + str(init_port)  # type: ignore[union-attr]

        # Ensure peer and cache query connections
        alloc_port = disagg.receiver_alloc_port  # type: ignore[union-attr]
        if isinstance(alloc_port, list):
            alloc_port = alloc_port[pd_backend.tp_rank]  # type: ignore[union-attr]
        query_port = query_ports[pd_backend.tp_rank]  # type: ignore[union-attr]

        pd_backend._ensure_peer_connection(  # type: ignore[union-attr]
            receiver_id=receiver_id,
            receiver_host=disagg.receiver_host,  # type: ignore[union-attr]
            receiver_init_port=init_port,
            receiver_alloc_port=alloc_port,
        )
        pd_backend._ensure_cache_query_connection(  # type: ignore[union-attr]
            receiver_id=receiver_id,
            receiver_host=disagg.receiver_host,  # type: ignore[union-attr]
            receiver_query_port=query_port,
        )

        # Query decoder cache
        cache_resp = pd_backend.query_remote_cache(receiver_id, chunk_keys)

        logger.info(
            "Bidirectional NIXL cache probe: req=%s, "
            "queried %d chunks, decoder has %d cached "
            "(%.0f%% hit rate)",
            request.req_id,
            len(chunk_keys),
            len(cache_resp.cached_keys),
            100.0 * len(cache_resp.cached_keys) / len(chunk_keys) if chunk_keys else 0,
        )

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if (
            os.getenv("LMCACHE_NSYS_FULL_CAPTURE_SCOPE", "decoder").lower()
            == "connector"
        ):
            manager = _CSA_ATTENTION_KV_PREFETCH_MANAGER
            finish_capture = getattr(manager, "finish_full_nsys_capture", None)
            if callable(finish_capture) and bool(
                getattr(manager, "full_nsys_capture_active", False)
            ):
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.barrier(
                        device_ids=[torch.cuda.current_device()],
                    )
                finish_capture()
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
        return invalid_blocks

    @_lmcache_nvtx_annotate
    def shutdown(self):
        """Shutdown the connector by delegating to LMCacheManager."""
        logger.info("Starting LMCacheConnector shutdown...")
        close_vllm_prefetch_managers()
        self._manager.stop_services()

    ###################
    # Scheduler side APIs
    ####################

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # Ignore DP attention mock requests
        if request.request_id.startswith("mock_req"):
            return 0
        # to handle preempted requests, we want `get_num_new_matched_tokens` to be
        # idempotent under the condition that `update_state_after_alloc` is NOT called
        # then the two side-effects that must be idempotent are:
        # 1. lookup_client caches a result
        #     uncached in `update_state_after_alloc` if this request can be scheduled
        # 2. cache engine will pin the KV caches for the request
        #     unpinned in `wait_for_save` if this request can be scheduled
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        req_id = request.request_id

        if self.lookup_client is None:
            logger.error(
                "get_num_new_matched_tokens: lookup_client is None (scheduler "
                "init failed?); returning 0 for request %s. "
                "init_failed=%s init_failed_reason=%s",
                req_id,
                getattr(self._manager, "_init_failed", "?"),
                getattr(self._manager, "_init_failed_reason", "?"),
            )
            return 0

        if (
            num_external_hit_tokens := self.lookup_client.lookup_cache(lookup_id=req_id)
        ) != -1:
            # -1 means no result cached
            # None or int means ongoing (async) or cached result
            logger.debug(
                f"Found {num_external_hit_tokens} hit tokens for request"
                f" {req_id} in the lookup cache."
            )
        else:
            logger.debug(f"Looking up cache for the first time for request {req_id}!")
            self._requests_priority[req_id] = getattr(request, "priority", 0)

            # token_ids = request.prompt_token_ids
            # all token ids covers the preemption case
            token_ids = request.all_token_ids

            # If the request has multimodal hashes, apply them to the token ids
            mm_hashes, mm_positions = extract_mm_features(request)
            if mm_hashes and mm_positions:
                # TODO(Jiayi): Optimize this
                token_ids = torch.tensor(request.prompt_token_ids)
                apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
                token_ids = token_ids.tolist()

            request_configs = extract_request_configs(request.sampling_params)
            if self.skip_last_n_tokens > 0:
                token_ids = token_ids[: -self.skip_last_n_tokens]

            num_external_hit_tokens = self.lookup_client.lookup(
                token_ids,
                lookup_id=req_id,
                request_configs=request_configs,
            )

        if num_external_hit_tokens is None:
            logger.debug(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: None (async lookup pending, treating as miss).",
                req_id,
                request.num_tokens,
                num_computed_tokens,
            )
            return 0

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            need_to_allocate -= 1

        # Check if hit tokens meet the minimum for retrieve
        # If below minimum, skip retrieve but still record hit tokens
        # for skip_leading_tokens to avoid re-storing existing chunks
        min_retrieve = self.config.min_retrieve_tokens
        below_min_retrieve = min_retrieve > 0 and need_to_allocate < min_retrieve

        if below_min_retrieve:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, but need to load: %d < min_retrieve %d, "
                "skip retrieve but record for save skip",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
                min_retrieve,
            )
        else:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, need to load: %d",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
            )

        self.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        if below_min_retrieve or need_to_allocate <= 0:
            return 0

        # TODO: Align to vLLM block size. Should test whether it can be removed
        # need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size

        return need_to_allocate

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """

        # Clear local status in lookup client when a new request is
        # successfully scheduled.
        if self.lookup_client is None:
            logger.warning(
                "update_state_after_alloc: lookup_client is None, skipping "
                "clear_lookup_status for request %s",
                request.request_id,
            )
            return
        self.lookup_client.clear_lookup_status(request.request_id)

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
                receiver_query_port=req_disagg_spec.get("receiver_query_port"),
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec
        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        recalc_last = (
            1
            if (
                self.load_specs[request.request_id].lmcache_cached_tokens
                == request.num_tokens
            )
            else 0
        )
        assert (
            num_external_tokens
            == self.load_specs[request.request_id].lmcache_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
            - recalc_last
        ), (
            f"Mismatch in tokens to load: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} "
            "(tokens in lmcache) - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens} "
            "(tokens in vllm) - "
            f"{recalc_last} "
            "(full lmcache hits subtracts last token to recalculate logits)"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        # We should load KV for:
        # 1. new requests
        # 2. preempted requests (once per recovery)
        # can_load will only be True if `update_state_after_alloc` has been called
        # which only happens when vLLM's KV manager has space to receive KV from LMCache
        for request in scheduler_output.scheduled_new_reqs:
            # Ignore DP attention mock requests
            if request.req_id.startswith("mock_req"):
                continue
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
                block_size=self._block_size,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                load_spec = self.load_specs.pop(req.req_id, None)
                lmcache_cached_tokens = 0
                vllm_cached_tokens = 0
                if load_spec is not None:
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                request_tracker = self._request_trackers[req.req_id]

                # Pass all_token_ids for preempted requests to restore
                # token_ids correctly for chunk key computation
                all_token_ids = None
                if req.resumed_from_preemption:
                    vllm_request = self._unfinished_requests.get(req.req_id)
                    assert vllm_request is not None, (
                        f"Preempted request {req.req_id} not found "
                        "in _unfinished_requests"
                    )
                    all_token_ids = list(vllm_request.all_token_ids)

                request_tracker.update(
                    req.new_token_ids,
                    req.new_block_ids,
                    req.resumed_from_preemption,
                    lmcache_cached_tokens=lmcache_cached_tokens,
                    vllm_cached_tokens=vllm_cached_tokens,
                    all_token_ids=all_token_ids,
                    block_size=self._block_size,
                )

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self.config.save_decode_cache,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: this is a dangerous reference to the request object inside vllm
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = request.num_computed_tokens
                # tracker_len < num_computed_tokens during decode
                #   (important for save_decode_cache).
                # num_computed_tokens < tracker_len after preemption.
                tracker_len = len(request_tracker.token_ids)
                slice_base = min(num_current_tokens, tracker_len)
                new_token_ids = request.all_token_ids[
                    slice_base : slice_base + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            load_spec = self.load_specs.pop(req_id, None)
            lmcache_cached_tokens = 0
            vllm_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                vllm_cached_tokens = load_spec.vllm_cached_tokens

            # Handle both old and new versions of CachedRequestData
            if hasattr(cached_reqs, "resumed_req_ids"):
                # New version with resumed_req_ids
                preempted = req_id in cached_reqs.resumed_req_ids
            elif hasattr(cached_reqs, "resumed_from_preemption"):
                # Old version with resumed_from_preemption
                preempted = cached_reqs.resumed_from_preemption[i]
            else:
                # This case should not be reached with supported vLLM versions.
                # Raising an error is safer than assuming not preempted.
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}. "
                    f"This might be due to an unsupported vLLM version."
                )
            if preempted:
                assert load_spec is not None, (
                    f"Request {req_id} is preempted but was not given a load spec"
                )
                # num_computed_tokens should be reset to 0 during preemption
                # and then set to the number of already cached tokens (maxxing
                # prefix caching and lmcache)
                # this assumption is crucial for the update() call of RequestTracker
                # On full cache hit, get_num_new_matched_tokens subtracts 1
                # to force last-token recomputation. This only affects
                # num_computed_tokens when lmcache has all tokens AND
                # provides more than vLLM's local cache.
                expected = max(lmcache_cached_tokens, load_spec.vllm_cached_tokens)
                full_hit_adj = (
                    lmcache_cached_tokens == len(request.all_token_ids)
                    and lmcache_cached_tokens > load_spec.vllm_cached_tokens
                )
                if full_hit_adj:
                    expected -= 1
                assert request.num_computed_tokens == expected, (
                    f"Preempted request {req_id} has "
                    f"num_computed_tokens {request.num_computed_tokens} "
                    f"but expected {expected} "
                    f"(full_hit_adj={full_hit_adj})"
                )

            # When retrieve fail, vllm will call _handle_invalid_blocks to
            # reset request.num_computed_tokens, this will lead to
            # request_tracker.token_ids being not matched with vllm
            if num_current_tokens < len(request_tracker.token_ids):
                logger.warning(
                    "Request %s rolled back from %d to %d tokens; "
                    "truncating tracker state.",
                    req_id,
                    len(request_tracker.token_ids),
                    num_current_tokens,
                )
                num_token_slots = (
                    len(request_tracker.allocated_block_ids) * self._block_size
                )
                tokens_to_keep = num_current_tokens
                if num_token_slots < num_current_tokens:
                    logger.warning(
                        "Request %s tracker has %d token slots but %d tokens; "
                        "capping token_ids to slot capacity.",
                        req_id,
                        num_token_slots,
                        num_current_tokens,
                    )
                    tokens_to_keep = num_token_slots

                request_tracker.token_ids = list(request.all_token_ids[:tokens_to_keep])
                request_tracker.num_saved_tokens = min(
                    request_tracker.num_saved_tokens, tokens_to_keep
                )

            # Pass all_token_ids for preempted requests to restore
            # token_ids correctly for chunk key computation
            all_token_ids = list(request.all_token_ids) if preempted else None

            request_tracker.update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
                block_size=self._block_size,
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Layerwise save uses request-scoped generators. If request finishes
        # without entering wait_for_save (abort/error/evict path), make sure
        # we release the generator entry to avoid leaking state.
        if getattr(self, "use_layerwise", False) and hasattr(
            self, "_layerwise_save_storers"
        ):
            self._layerwise_save_storers.pop(request.request_id, None)

        # Cleanup if request was aborted
        if request.status == RequestStatus.FINISHED_ABORTED:
            # Abort notifications can run on scheduler-side connector objects
            # that do not own an LMCache engine. Treat cleanup as best-effort so
            # an interrupted benchmark does not crash the serving process.
            if self.lmcache_engine is not None:
                sm = self.lmcache_engine.storage_manager
                if sm is not None:
                    sm.cancel_request(request.request_id)

            if self.async_loading:
                # Cancel any ongoing async lookup and prefetch tasks on workers
                lookup_id = request.request_id
                if self.lookup_client is not None:
                    self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[attr-defined]

        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        if self.config.get_extra_config_value(
            "enable_cache_usage_details_in_response", False
        ):
            request_tracker = self._request_trackers.get(request.request_id)
            if request_tracker:
                return_params = return_params or {}
                return_params["num_lmcache_cached_tokens"] = (
                    request_tracker.num_lmcache_cached_tokens
                )

        return False, return_params

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.lmcache_engine is not None:
            return self.lmcache_engine.get_kv_events()
        return []
