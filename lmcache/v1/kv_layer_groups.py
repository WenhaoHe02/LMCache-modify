# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import DiscoverableKVCache

logger = init_logger(__name__)


# The 5-tuple that uniquely identifies a set of kernel-equivalent layers:
# ``(kv_size, num_heads, head_size, block_size, dtype)``. Two layers share a
# transfer-kernel launch iff they share this identity — see the grouping loop
# in :meth:`KVLayerGroupsManager.__init__` for the derivation.
# ``block_size`` is the 4th element to support DeepSeek V4 mixed-compression
# layouts where MLA layers may have a different physical block size than
# standard attention layers.
LayerGroupIdentity = tuple[int, int, int, int, torch.dtype]


@dataclass
class KVLayerGroupInfo:
    """A single transfer-kernel dispatch unit: a set of KV layers that can
    ride one kernel launch with one ``PageBufferShapeDesc``.

    Membership is decided by :class:`KVLayerGroupsManager` according to
    :data:`LayerGroupIdentity`; every layer referenced by
    ``layer_indices`` shares the same ``(kv_size, num_heads, head_size,
    block_size, dtype)`` signature. Consumers use ``layer_indices`` to pull
    the matching device pointers out of ``kv_caches`` (via
    :func:`~lmcache.v1.gpu_connector.utils.get_group_data_ptrs`) and feed
    them to the kernel alongside ``shape_desc``.

    ``dtype`` is carried alongside ``shape_desc`` because
    ``PageBufferShapeDesc.element_size`` is a byte width, which cannot
    distinguish dtypes that share a byte count (e.g. bfloat16 and
    float16 are both 2 bytes). Kernel template instantiation keys on the
    torch dtype, not the byte width, so we keep it explicit.

    Treat instances as immutable after construction; callers may hold
    references for the lifetime of the manager.
    """

    layer_indices: list[int]
    """0-based layer indices belonging to this group, in the order the
    kernel should iterate them. Fed to ``get_group_data_ptrs`` to build
    the per-group pointer array."""
    shape_desc: "lmc_ops.PageBufferShapeDesc"
    """Kernel-facing shape descriptor shared by every layer in the group.
    All eight fields (``kv_size, nl, nb, bs, nh, hs, element_size,
    block_stride_elems``) are stamped once at construction."""
    dtype: torch.dtype
    """Torch dtype of the KV cache tensors for this group. Used for
    kernel template instantiation; see class docstring for why we keep
    this alongside ``shape_desc.element_size``."""

    def __repr__(self) -> str:
        if not self.layer_indices:
            indices_repr = "[]"
        else:
            indices_repr = f"{self.layer_indices[0]}-{self.layer_indices[-1]}"
        sd = self.shape_desc
        return (
            f"KVLayerGroupInfo(layers={len(self.layer_indices)}, "
            f"indices={indices_repr}, "
            f"shape_desc=(kv={sd.kv_size}, nl={sd.nl}, nb={sd.nb}, "
            f"bs={sd.bs}, nh={sd.nh}, hs={sd.hs}, "
            f"element_size={sd.element_size}), dtype={self.dtype})"
        )

    @property
    def num_layers(self) -> int:
        """Number of layers in this group."""
        return len(self.layer_indices)

    @property
    def hidden_dim_size(self) -> int:
        """Hidden dimension size (``num_heads * head_size``)."""
        return self.shape_desc.nh * self.shape_desc.hs


class KVLayerGroupsManager:
    """Partition a model's KV layers into transfer-kernel dispatch units.

    At construction time, every layer in ``kv_caches`` is bucketed by its
    :data:`LayerGroupIdentity` (``(kv_size, num_heads, head_size,
    block_size, dtype)``). Each bucket becomes one :class:`KVLayerGroupInfo`
    holding the layer indices, a shared :class:`PageBufferShapeDesc`, and the
    group's torch dtype.

    Downstream consumers (``VLLMPagedMemGPUConnectorV3``,
    ``GPUCacheContext``, the multiprocess server) iterate
    ``self.kv_layer_groups`` and issue one transfer-kernel launch per
    group. The manager itself is a pure metadata object — it does not
    own any GPU buffers or perform any transfers.

    Layout parsing is delegated entirely to
    :mod:`lmcache.v1.gpu_connector.utils`; this class only drives the
    grouping and look-up.
    """

    # Formats where dim 0 of the per-layer tensor is the block axis.
    # For these formats we check the actual dim-0 stride to detect padding
    # (DeepSeek V4 mixed-compression layouts).
    _BLOCK_AXIS_FORMATS: frozenset = frozenset(
        {
            lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
            lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
        }
    )

    def __init__(
        self,
        kv_caches: "DiscoverableKVCache",
        gpu_kv_format: "lmc_ops.GPUKVFormat",
        num_blocks: int,
    ) -> None:
        """Partition layers into groups keyed by
        :data:`LayerGroupIdentity`.

        For each layer ``i`` in ``kv_caches``, read
        ``(kv_size, num_heads, head_size, block_size, dtype)`` via the
        format-aware accessors in ``utils.py``. Layers with identical
        identities are bucketed together; each bucket becomes one
        :class:`KVLayerGroupInfo`.

        Groups are emitted in the order of their first-appearing layer,
        so group indices are deterministic across runs.

        Args:
            kv_caches: KV cache structure accepted by
                :func:`normalize_kv_and_discover_format`.
            gpu_kv_format: Format returned by
                :func:`normalize_kv_and_discover_format`.
            num_blocks: Number of paged blocks. Stamped into every
                ``shape_desc.nb``.
        """
        # Import here to break a circular import via
        # lmcache.v1.gpu_connector.__init__ → metadata → kv_layer_groups.
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            get_block_size,
            get_dtype,
            get_head_size,
            get_num_heads,
            get_num_layers,
            is_mla,
            make_page_buffer_shape_desc,
        )

        self.kv_layer_groups: list[KVLayerGroupInfo] = []

        num_layers = get_num_layers(kv_caches, gpu_kv_format)
        if num_layers == 0:
            logger.debug("No KV caches available, skipping KV layer groups building")
            return

        # Temporary accumulator: maps each LayerGroupIdentity to the list
        # of layer indices that share it. Built in one linear pass over
        # all layers, then drained into KVLayerGroupInfo objects below.
        # The index lists are passed by reference into the infos, so
        # after __init__ returns this dict is garbage-collected while
        # the lists stay alive on each group.
        mla = is_mla(gpu_kv_format)
        kv_size = 1 if mla else 2
        groups_dict: dict[LayerGroupIdentity, list[int]] = defaultdict(list)
        for idx in range(num_layers):
            nh = 1 if mla else get_num_heads(kv_caches, gpu_kv_format, idx)
            hs = get_head_size(kv_caches, gpu_kv_format, idx)
            dt = get_dtype(kv_caches, gpu_kv_format, idx)
            # Per-layer block_size: DSV4 MLA layers may differ from MHA layers.
            bs = get_block_size(kv_caches, gpu_kv_format, layer_idx=idx)
            groups_dict[(kv_size, nh, hs, bs, dt)].append(idx)

        # Emit groups in order of their first-appearing layer.
        for (_, _, _, bs, dt), indices in sorted(
            groups_dict.items(), key=lambda kv: kv[1][0]
        ):
            # Detect padded dim-0 stride for formats where the block axis
            # is dim 0 of the per-layer tensor (DSV4 mixed-compression).
            block_stride_elems: int | None = None
            if gpu_kv_format in self._BLOCK_AXIS_FORMATS:
                rep_tensor = kv_caches[indices[0]]
                # Tight stride = product of all dims that follow dim 0.
                tight = 1
                for d in range(1, rep_tensor.ndim):
                    tight *= rep_tensor.shape[d]
                actual_stride0 = rep_tensor.stride(0)
                if actual_stride0 > tight:
                    block_stride_elems = actual_stride0
                    logger.debug(
                        "KVLayerGroupsManager: group starting at layer %d "
                        "has padded dim-0 stride %d > tight %d "
                        "(block_stride_elems=%d)",
                        indices[0],
                        actual_stride0,
                        tight,
                        block_stride_elems,
                    )

            shape_desc = make_page_buffer_shape_desc(
                kv_caches,
                gpu_kv_format,
                layer_idx=indices[0],
                num_layers_in_group=len(indices),
                num_blocks=num_blocks,
                block_size=bs,
                block_stride_elems=block_stride_elems,
            )
            self.kv_layer_groups.append(
                KVLayerGroupInfo(
                    layer_indices=indices,
                    shape_desc=shape_desc,
                    dtype=dt,
                )
            )

        logger.info("KV layer groups: %s", self.kv_layer_groups)

    @property
    def num_groups(self) -> int:
        """Number of :class:`KVLayerGroupInfo` entries.

        Zero if ``kv_caches`` had no layers at construction time.
        """
        return len(self.kv_layer_groups)

    def get_shape_desc(self, group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Return the :class:`PageBufferShapeDesc` for *group_idx*.

        Equivalent to ``self.kv_layer_groups[group_idx].shape_desc``.

        Args:
            group_idx: 0-based group index.

        Raises:
            IndexError: If *group_idx* is out of range.
        """
        return self.kv_layer_groups[group_idx].shape_desc
