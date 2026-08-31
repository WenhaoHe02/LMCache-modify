# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.20.2 hooks for GLM DSA predictive prefetch experiments."""

from __future__ import annotations

# Standard
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MethodType
from typing import Any
import os
import threading
import weakref

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.glm_dsa_predictive_prefetch import (
    GLMDSAPredictionSchedule,
    GLMDSAPredictivePrefetchManager,
)

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class GLMDSAAsyncPrediction:
    """Top-K output and completion event for one private-stream prediction."""

    topk_indices: torch.Tensor
    done_event: torch.cuda.Event
    phase_events: tuple[torch.cuda.Event, ...] = ()


_SUPPORTED_VLLM_VERSION = "0.20.2"
_DECODER_REGISTRY: weakref.WeakSet[Any] = weakref.WeakSet()
_REGISTRY_HOOK_INSTALLED = False


def is_supported_vllm_version(version: str) -> bool:
    """Return whether *version* identifies the SSD experiment's vLLM build.

    Args:
        version: Value exported by ``vllm.version.__version__``.

    Returns:
        ``True`` for vLLM 0.20.2, including local build suffixes.
    """
    normalized = version.strip().removeprefix("v")
    return normalized == _SUPPORTED_VLLM_VERSION or normalized.startswith(
        f"{_SUPPORTED_VLLM_VERSION}+"
    )


def install_decoder_registry_hook(version: str) -> bool:
    """Record vLLM 0.20.2 DeepseekV2 decoder instances as they are built.

    Args:
        version: Runtime vLLM version.

    Returns:
        Whether the exact supported class was patched or already registered.
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
        "_lmcache_glm_dsa_registry_hook_installed",
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

    DeepseekV2DecoderLayer._lmcache_glm_dsa_original_init = original_init
    DeepseekV2DecoderLayer.__init__ = _lmcache_init
    DeepseekV2DecoderLayer._lmcache_glm_dsa_registry_hook_installed = True
    _REGISTRY_HOOK_INSTALLED = True
    logger.info("LMCache installed the vLLM 0.20.2 GLM decoder registry hook")
    return True


def registered_glm_decoder_layers() -> tuple[Any, ...]:
    """Return registered decoder layers ordered by zero-based layer id."""
    layers = [
        layer
        for layer in list(_DECODER_REGISTRY)
        if isinstance(getattr(layer, "layer_idx", None), int)
        and getattr(layer, "self_attn", None) is not None
    ]
    layers.sort(key=lambda layer: int(layer.layer_idx))
    return tuple(layers)


class VLLM0202GLMDSAProxy:
    """Run target Full indexers read-only using an early residual proxy.

    Args:
        decoder_layers: Live vLLM decoder layers keyed by layer id.
    """

    def __init__(self, decoder_layers: Mapping[int, Any]) -> None:
        self._decoder_layers = dict(decoder_layers)
        self._prediction_stream: torch.cuda.Stream | None = None
        self._prediction_device: torch.device | None = None
        self._launch_lock = threading.Lock()

    def validate_schedule(
        self,
        schedules: tuple[GLMDSAPredictionSchedule, ...],
    ) -> None:
        """Validate every scheduled target against the vLLM 0.20.2 API.

        Args:
            schedules: Prediction edges that will be attached.

        Raises:
            RuntimeError: If the live server objects differ from the supported
                vLLM 0.20.2 GLM/DeepseekV2 layout.
        """
        for schedule in schedules:
            layer = self._decoder_layers.get(schedule.target_layer)
            if layer is None:
                raise RuntimeError(
                    f"missing target decoder layer {schedule.target_layer}"
                )
            attention = getattr(layer, "self_attn", None)
            required_attention = (
                "fused_qkv_a_proj",
                "q_a_layernorm",
                "q_lora_rank",
                "kv_lora_rank",
                "qk_rope_head_dim",
                "indexer",
                "indexer_rope_emb",
            )
            missing = [
                name
                for name in required_attention
                if getattr(attention, name, None) is None
            ]
            if missing:
                raise RuntimeError(
                    f"target layer {schedule.target_layer} is missing "
                    f"vLLM 0.20.2 attributes: {missing}"
                )
            indexer_op = getattr(attention.indexer, "indexer_op", None)
            if indexer_op is None:
                raise RuntimeError(
                    f"target layer {schedule.target_layer} has no indexer_op"
                )
            if not hasattr(indexer_op, "skip_k_cache_insert"):
                raise RuntimeError(
                    f"target layer {schedule.target_layer} cannot score read-only"
                )
            if not isinstance(
                getattr(indexer_op, "topk_indices_buffer", None),
                torch.Tensor,
            ):
                raise RuntimeError(
                    f"target layer {schedule.target_layer} has no top-K buffer"
                )

    def predict(
        self,
        schedule: GLMDSAPredictionSchedule,
        activation: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a target Full layer without inserting proxy indexer K rows.

        Args:
            schedule: Edge identifying the target Full layer.
            activation: Logical post-MLP source-layer state.
            positions: Query token positions.

        Returns:
            A private ``[query_rows, index_topk]`` int32 tensor.

        Raises:
            RuntimeError: If CUDA graph capture or the supported object API
                prevents a safe read-only proxy.
        """
        if activation.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("GLM DSA proxy is not supported during CUDA capture")
        return self._predict_activation(schedule, activation, positions)

    def predict_async(
        self,
        schedule: GLMDSAPredictionSchedule,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> GLMDSAAsyncPrediction:
        """Enqueue activation and prediction on one ordered private stream."""
        if not hidden_states.is_cuda:
            raise RuntimeError("asynchronous GLM prediction requires CUDA tensors")
        device = hidden_states.device
        with self._launch_lock, torch.cuda.device(device):
            if self._prediction_stream is None:
                self._prediction_stream = torch.cuda.Stream(device=device)
                self._prediction_device = device
            elif self._prediction_device != device:
                raise RuntimeError("GLM prediction stream device changed")
            model_stream = torch.cuda.current_stream(device)
            source_ready = torch.cuda.Event()
            source_ready.record(model_stream)
            self._prediction_stream.wait_event(source_ready)
            hidden_states.record_stream(self._prediction_stream)
            positions.record_stream(self._prediction_stream)
            if residual is not None:
                residual.record_stream(self._prediction_stream)
            with torch.cuda.stream(self._prediction_stream):
                diag_enabled = os.path.exists("/tmp/lmcache_glm_kcp_phase_diag")
                phase_events = (
                    tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))
                    if diag_enabled
                    else ()
                )
                if phase_events:
                    phase_events[0].record(self._prediction_stream)
                activation = (
                    hidden_states if residual is None else hidden_states + residual
                )
                topk = self._predict_activation(
                    schedule,
                    activation,
                    positions,
                    phase_events=(
                        (phase_events[1], phase_events[2], phase_events[3])
                        if phase_events
                        else None
                    ),
                )
                topk.record_stream(self._prediction_stream)
                done = torch.cuda.Event()
                done.record(self._prediction_stream)
        return GLMDSAAsyncPrediction(
            topk,
            done,
            phase_events,
        )

    def _predict_activation(
        self,
        schedule: GLMDSAPredictionSchedule,
        activation: torch.Tensor,
        positions: torch.Tensor,
        phase_events: tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]
        | None = None,
    ) -> torch.Tensor:
        """Launch one read-only target indexer for a prepared activation."""
        target = self._decoder_layers[schedule.target_layer]
        attention = target.self_attn
        proxy_hidden = target.input_layernorm(activation)
        if phase_events is not None:
            phase_events[0].record()
        projected = attention.fused_qkv_a_proj(proxy_hidden)
        if isinstance(projected, tuple):
            projected = projected[0]
        if not isinstance(projected, torch.Tensor):
            raise RuntimeError("target fused_qkv_a_proj did not return a tensor")
        q_width = int(attention.q_lora_rank)
        kv_width = int(attention.kv_lora_rank) + int(attention.qk_rope_head_dim)
        q_c, _ = projected.split([q_width, kv_width], dim=-1)
        q_c = attention.q_a_layernorm(q_c)
        if phase_events is not None:
            phase_events[1].record()

        indexer = attention.indexer
        indexer_op = indexer.indexer_op
        reference = indexer_op.topk_indices_buffer
        topk_width = int(getattr(indexer, "topk_tokens", reference.shape[1]))
        private_topk = torch.empty(
            (int(proxy_hidden.shape[0]), topk_width),
            dtype=reference.dtype,
            device=reference.device,
        )
        old_buffer = indexer_op.topk_indices_buffer
        old_skip_insert = indexer_op.skip_k_cache_insert
        try:
            indexer_op.topk_indices_buffer = private_topk
            indexer_op.skip_k_cache_insert = True
            result = indexer(
                proxy_hidden,
                q_c,
                positions,
                attention.indexer_rope_emb,
            )
            if (
                isinstance(result, torch.Tensor)
                and result.data_ptr() != private_topk.data_ptr()
            ):
                private_topk.copy_(result[: private_topk.shape[0], :topk_width])
        finally:
            indexer_op.topk_indices_buffer = old_buffer
            indexer_op.skip_k_cache_insert = old_skip_insert
        if phase_events is not None:
            phase_events[2].record()
        return private_topk


class VLLM0202GLMDSAHooks:
    """Patch and restore vLLM 0.20.2 source layers and Full indexers.

    Args:
        decoder_layers: Live decoder layers keyed by layer id.
        manager: Version-independent predictive prefetch manager.
        indexer_types: Model-declared Full/Shared pattern.  Shared vLLM 0.20.2
            wrapper calls are disabled so the preceding Full top-K is reused.
        forward_enabled: Optional callback that identifies an external-KV
            prefill forward. Disabled forwards bypass every LMCache gate,
            prediction, and authoritative-observation hook.
        disabled_authoritative_observer: Optional profile-only callback for
            authoritative Full-index output from disabled (normally decode)
            forwards. It must not submit I/O or alter model state.
    """

    def __init__(
        self,
        decoder_layers: Mapping[int, Any],
        manager: GLMDSAPredictivePrefetchManager,
        *,
        indexer_types: tuple[str, ...] = (),
        authoritative_observer: Callable[[int, torch.Tensor], None] | None = None,
        disabled_authoritative_observer: Callable[[int, torch.Tensor], None]
        | None = None,
        consumer_waiter: Callable[[int], bool] | None = None,
        forward_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._decoder_layers = dict(decoder_layers)
        self._manager = manager
        self._indexer_types = indexer_types
        self._authoritative_observer = authoritative_observer
        self._disabled_authoritative_observer = disabled_authoritative_observer
        self._consumer_waiter = consumer_waiter
        self._forward_enabled = forward_enabled
        self._patched_decoders: list[Any] = []
        self._patched_indexers: list[Any] = []
        self._shared_indexers: list[tuple[Any, Any]] = []

    def attach(self) -> None:
        """Attach post-MLP prediction and authoritative Full-index hooks."""
        self._apply_indexshare()
        hooked_decoder_ids = {
            layer_id
            for schedule in self._manager.schedules
            for layer_id in (schedule.source_layer, *schedule.consumer_layers)
        }
        for layer_id in sorted(hooked_decoder_ids):
            decoder = self._decoder_layers.get(layer_id)
            if decoder is not None:
                self._patch_source_decoder(decoder, layer_id)

        full_layer_ids = {schedule.target_layer for schedule in self._manager.schedules}
        full_layer_ids.update(
            layer_id
            for layer_id, indexer_type in enumerate(self._indexer_types)
            if indexer_type == "full"
        )
        for layer_id in sorted(full_layer_ids):
            target_attention = self._decoder_layers[layer_id].self_attn
            indexer = getattr(target_attention, "indexer", None)
            if indexer is not None:
                self._patch_true_indexer(indexer, layer_id)

    def close(self) -> None:
        """Restore every patched vLLM forward method; safe to call repeatedly."""
        for wrapper, indexer in self._shared_indexers:
            wrapper.indexer = indexer
        for decoder in self._patched_decoders:
            original = getattr(
                decoder,
                "_lmcache_glm_dsa_original_forward",
                None,
            )
            if callable(original):
                decoder.forward = original
                delattr(decoder, "_lmcache_glm_dsa_original_forward")
        for indexer in self._patched_indexers:
            original = getattr(
                indexer,
                "_lmcache_glm_dsa_original_forward",
                None,
            )
            if callable(original):
                indexer.forward = original
                delattr(indexer, "_lmcache_glm_dsa_original_forward")
        self._patched_decoders.clear()
        self._patched_indexers.clear()
        self._shared_indexers.clear()

    def _apply_indexshare(self) -> None:
        if not self._indexer_types:
            return
        if len(self._indexer_types) != len(self._decoder_layers):
            raise RuntimeError("indexer_types does not match live decoder layers")
        skipped: list[int] = []
        for layer_id, indexer_type in enumerate(self._indexer_types):
            if indexer_type == "full":
                continue
            if indexer_type != "shared":
                raise RuntimeError(f"unsupported indexer type {indexer_type!r}")
            decoder = self._decoder_layers.get(layer_id)
            if decoder is None:
                raise RuntimeError(f"missing shared decoder layer {layer_id}")
            attention = decoder.self_attn
            wrapper = getattr(attention, "mla_attn", None)
            if wrapper is None:
                raise RuntimeError(f"shared layer {layer_id} has no MLA wrapper")
            indexer = getattr(wrapper, "indexer", None)
            if indexer is None:
                continue
            self._shared_indexers.append((wrapper, indexer))
            wrapper.indexer = None
            skipped.append(layer_id)
        if skipped:
            logger.info(
                "LMCache enabled GLM IndexShare on vLLM 0.20.2 shared layers=%s",
                skipped,
            )

    def _patch_source_decoder(self, decoder: Any, source_layer: int) -> None:
        if hasattr(decoder, "_lmcache_glm_dsa_original_forward"):
            return
        original = decoder.forward
        manager = self._manager

        def _forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
            enabled = self._forward_enabled is None or self._forward_enabled()
            full_indexer_layer = bool(
                self._indexer_types and self._indexer_types[source_layer] == "full"
            )
            if (
                enabled
                and not full_indexer_layer
                and not manager.wait_for_prediction(source_layer)
            ):
                raise RuntimeError(
                    f"GLM DSA prediction timed out at layer {source_layer}"
                )
            if (
                enabled
                and not full_indexer_layer
                and self._consumer_waiter is not None
                and not self._consumer_waiter(source_layer)
            ):
                raise RuntimeError(
                    f"GLM DSA attention KV gate timed out at layer {source_layer}"
                )
            result = original(*args, **kwargs)
            positions = args[0] if args else kwargs.get("positions")
            if (
                enabled
                and isinstance(result, tuple)
                and len(result) >= 2
                and isinstance(result[0], torch.Tensor)
                and isinstance(result[1], torch.Tensor)
                and isinstance(positions, torch.Tensor)
            ):
                manager.after_source_layer(
                    source_layer,
                    result[0],
                    result[1],
                    positions,
                )
            return result

        decoder._lmcache_glm_dsa_original_forward = original
        decoder.forward = MethodType(_forward, decoder)
        self._patched_decoders.append(decoder)

    def _patch_true_indexer(self, indexer: Any, target_layer: int) -> None:
        if hasattr(indexer, "_lmcache_glm_dsa_original_forward"):
            return
        original = indexer.forward
        manager = self._manager

        def _forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if bool(
                getattr(
                    getattr(instance, "indexer_op", None), "skip_k_cache_insert", False
                )
            ):
                # The proxy calls this same module with private output storage.
                # It must never join its own prediction future or publish an
                # approximate result through the authoritative observer.
                return original(*args, **kwargs)
            result = original(*args, **kwargs)
            hidden = args[0] if args else kwargs.get("hidden_states")
            if isinstance(result, torch.Tensor) and isinstance(hidden, torch.Tensor):
                rows = int(hidden.shape[0])
                true_topk = result[:rows]
                if self._forward_enabled is not None and not self._forward_enabled():
                    if self._disabled_authoritative_observer is not None:
                        self._disabled_authoritative_observer(target_layer, true_topk)
                    return result
                # Full-indexer computation needs indexer K, not the attention
                # KV being prefetched. Join only now, before correction and
                # sparse attention consume that KV, to retain its overlap.
                if not manager.wait_for_prediction(target_layer):
                    raise RuntimeError(
                        f"GLM DSA prediction timed out at layer {target_layer}"
                    )
                if self._consumer_waiter is not None and not self._consumer_waiter(
                    target_layer
                ):
                    raise RuntimeError(
                        f"GLM DSA attention KV gate timed out at layer {target_layer}"
                    )
                manager.observe_true_topk(target_layer, true_topk)
                if self._authoritative_observer is not None:
                    self._authoritative_observer(target_layer, true_topk)
            return result

        indexer._lmcache_glm_dsa_original_forward = original
        indexer.forward = MethodType(_forward, indexer)
        self._patched_indexers.append(indexer)


def decoder_layer_map(layers: tuple[Any, ...]) -> Mapping[int, Any]:
    """Validate and index live decoder layers by unique layer id.

    Args:
        layers: Registered vLLM decoder layer instances.

    Returns:
        Mapping from zero-based layer id to decoder object.

    Raises:
        ValueError: If layer ids are invalid or duplicated.
    """
    indexed: dict[int, Any] = {}
    for layer in layers:
        layer_id = getattr(layer, "layer_idx", None)
        if not isinstance(layer_id, int) or layer_id < 0:
            raise ValueError("decoder layers require non-negative integer layer_idx")
        if layer_id in indexed:
            raise ValueError(f"duplicate decoder layer id {layer_id}")
        indexed[layer_id] = layer
    return indexed


PrefetchSubmitterFactory = Callable[[Any], Callable[[Any], None] | None]
