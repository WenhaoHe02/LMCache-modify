# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.26.0 bindings for GLM DSA predictive prefetch."""

from __future__ import annotations

# Standard
import gc
from typing import Any
import weakref

# First Party
from lmcache.integration.vllm.glm_dsa_vllm_0202 import (
    VLLM0202GLMDSAHooks,
    VLLM0202GLMDSAProxy,
    decoder_layer_map,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)

_SUPPORTED_VLLM_VERSION = "0.26.0"
_DECODER_REGISTRY: weakref.WeakSet[Any] = weakref.WeakSet()
_REGISTRY_HOOK_INSTALLED = False

# The Full-layer projection and SparseAttnIndexer APIs used by the predictor
# are unchanged.  vLLM 0.26.0 now implements IndexShare natively: Shared
# layers expose ``indexer=None``, which makes the 0.20.2 hook's compatibility
# cleanup an intentional no-op for those layers.
VLLM0260GLMDSAProxy = VLLM0202GLMDSAProxy
VLLM0260GLMDSAHooks = VLLM0202GLMDSAHooks


def is_supported_vllm_version(version: str) -> bool:
    """Return whether ``version`` identifies vLLM 0.26.0.

    Args:
        version: Value exported by ``vllm.version.__version__``.

    Returns:
        ``True`` for vLLM 0.26.0, including local build suffixes.
    """
    normalized = version.strip().removeprefix("v")
    return normalized == _SUPPORTED_VLLM_VERSION or normalized.startswith(
        f"{_SUPPORTED_VLLM_VERSION}+"
    )


def install_decoder_registry_hook(version: str) -> bool:
    """Record vLLM 0.26.0 DeepseekV2 decoder instances as they are built.

    GLM DSA continues to use ``DeepseekV2DecoderLayer`` in vLLM 0.26.0.
    Registering construction avoids a process-wide GC scan after compilation.

    Args:
        version: Runtime vLLM version.

    Returns:
        Whether the supported class was patched or was already patched.
    """
    global _REGISTRY_HOOK_INSTALLED

    if not is_supported_vllm_version(version):
        return False
    if _REGISTRY_HOOK_INSTALLED:
        return True
    try:
        from vllm.model_executor.models.deepseek_v2 import (
            DeepseekV2DecoderLayer,
        )
    except ImportError:
        return False
    if getattr(
        DeepseekV2DecoderLayer,
        "_lmcache_glm_dsa_0260_registry_hook_installed",
        False,
    ):
        _REGISTRY_HOOK_INSTALLED = True
        return True
    original_init = getattr(DeepseekV2DecoderLayer, "__init__", None)
    if not callable(original_init):
        return False

    def _lmcache_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        try:
            _DECODER_REGISTRY.add(self)
        except TypeError:
            logger.debug("GLM DSA decoder layer does not support weak references")

    DeepseekV2DecoderLayer._lmcache_glm_dsa_0260_original_init = original_init
    DeepseekV2DecoderLayer.__init__ = _lmcache_init
    DeepseekV2DecoderLayer._lmcache_glm_dsa_0260_registry_hook_installed = True
    _REGISTRY_HOOK_INSTALLED = True
    logger.info("LMCache installed the vLLM 0.26.0 GLM decoder registry hook")
    return True


def registered_glm_decoder_layers() -> tuple[Any, ...]:
    """Return registered decoder layers ordered by zero-based layer id."""
    # The LMCache connector is constructed after vLLM builds the model, so on
    # a stock 0.26.0 launch the constructor hook cannot observe those existing
    # decoder instances. Fall back once to the same process-local discovery
    # used by the DSV4 compatibility path, then retain weak references so later
    # calls stay cheap and deterministic.
    if not _DECODER_REGISTRY:
        try:
            from vllm.model_executor.models.deepseek_v2 import (
                DeepseekV2DecoderLayer,
            )

            discovered = 0
            for candidate in gc.get_objects():
                try:
                    is_decoder = isinstance(candidate, DeepseekV2DecoderLayer)
                except ReferenceError:
                    continue
                if not is_decoder:
                    continue
                try:
                    _DECODER_REGISTRY.add(candidate)
                    discovered += 1
                except TypeError:
                    continue
            if discovered:
                logger.info(
                    "LMCache discovered %d existing vLLM 0.26.0 GLM "
                    "decoder layers",
                    discovered,
                )
        except ImportError:
            pass
    layers = [
        layer
        for layer in list(_DECODER_REGISTRY)
        if isinstance(getattr(layer, "layer_idx", None), int)
        and getattr(layer, "self_attn", None) is not None
    ]
    layers.sort(key=lambda layer: int(layer.layer_idx))
    return tuple(layers)


__all__ = [
    "VLLM0260GLMDSAHooks",
    "VLLM0260GLMDSAProxy",
    "decoder_layer_map",
    "install_decoder_registry_hook",
    "is_supported_vllm_version",
    "registered_glm_decoder_layers",
]
