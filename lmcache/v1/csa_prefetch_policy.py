# SPDX-License-Identifier: Apache-2.0
"""Per-layer lookahead policy for speculative CSA KV prefetch."""

# Standard
from collections.abc import Iterable
from dataclasses import dataclass, field
import re


_PROFILE80_SPEC = "default:0,26-42:2"
_PROFILE80_HYBRID_SPEC = "default:0,26-34:2,36:1,38-40:2,42:1"


@dataclass(frozen=True)
class CSAPrefetchLookaheadPolicy:
    """Select demand-only or two-layer prefetch by target layer.

    The specification is a comma-separated list of ``layer:value``,
    ``start-end:value``, or ``default:value`` entries. Zero disables
    prediction and preserves true-indexer demand loading; one and two select
    a single prediction one or two decoder layers before the target. Without
    an explicit default, omitted layers are demand-only.
    ``profile80`` selects only the profiled deep layers with two-layer
    lookahead. ``profile80_hybrid`` moves the two correction-heavy targets,
    layers 36 and 42, to a closer one-layer source.

    Args:
        specification: Per-target-layer lookahead specification.
    """

    specification: str
    _by_layer: dict[int, int] = field(init=False, repr=False, compare=False)
    _default_lookahead: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the specification when the policy is constructed."""
        default_lookahead, by_layer = self._parse(self.specification)
        object.__setattr__(self, "_default_lookahead", default_lookahead)
        object.__setattr__(self, "_by_layer", by_layer)

    def lookahead_for(self, target_layer_id: int) -> int:
        """Return the configured lookahead for a target CSA layer.

        Args:
            target_layer_id: Transformer layer id of the target CSA.

        Returns:
            ``0`` when prediction is disabled, otherwise the configured
            source-to-target distance (one or two layers).
        """
        return self._by_layer.get(int(target_layer_id), self._default_lookahead)

    def two_layer_targets(self, target_layer_ids: list[int]) -> set[int]:
        """Return target CSA layers configured for two-layer prefetch.

        Args:
            target_layer_ids: Available target CSA transformer layer ids.

        Returns:
            The subset whose configured lookahead is two.
        """
        return {
            int(layer_id)
            for layer_id in target_layer_ids
            if self.lookahead_for(layer_id) == 2
        }

    def one_layer_targets(self, target_layer_ids: list[int]) -> set[int]:
        """Return target CSA layers configured for one-layer prefetch.

        Args:
            target_layer_ids: Available target CSA transformer layer ids.

        Returns:
            The subset whose configured lookahead is one.
        """
        return {
            int(layer_id)
            for layer_id in target_layer_ids
            if self.lookahead_for(layer_id) == 1
        }

    def disabled_targets(self, target_layer_ids: list[int]) -> set[int]:
        """Return target CSA layers whose prediction is disabled.

        Args:
            target_layer_ids: Available target CSA transformer layer ids.

        Returns:
            The subset configured with lookahead zero.
        """
        return {
            int(layer_id)
            for layer_id in target_layer_ids
            if self.lookahead_for(layer_id) == 0
        }

    @classmethod
    def from_recall_profile(
        cls,
        recall_by_target_layer: dict[int, float],
        min_recall: float = 0.8,
    ) -> "CSAPrefetchLookaheadPolicy":
        """Build a policy from offline two-layer prediction recall.

        Args:
            recall_by_target_layer: Recall measured for each target CSA layer.
            min_recall: Inclusive threshold for enabling two-layer lookahead.

        Returns:
            A policy that assigns lookahead two exactly to layers meeting the
            threshold. Unprofiled layers disable speculative prediction.

        Raises:
            ValueError: If the threshold or a recall value is outside [0, 1].
        """
        if not 0.0 <= min_recall <= 1.0:
            raise ValueError("min_recall must be within [0, 1]")
        selected: list[str] = []
        for layer_id, recall in sorted(recall_by_target_layer.items()):
            if not 0.0 <= recall <= 1.0:
                raise ValueError("profile recall values must be within [0, 1]")
            if recall >= min_recall:
                selected.append(f"{int(layer_id)}:2")
        return cls(",".join(["default:0", *selected]))

    @staticmethod
    def _parse(specification: str) -> tuple[int, dict[int, int]]:
        spec = specification.strip().lower()
        if not spec:
            return 0, {}
        if spec == "profile80":
            spec = _PROFILE80_SPEC
        elif spec == "profile80_hybrid":
            spec = _PROFILE80_HYBRID_SPEC

        default_lookahead = 0
        default_seen = False
        by_layer: dict[int, int] = {}
        entry_pattern = re.compile(r"^(\d+)(?:-(\d+))?\s*:\s*([012])$")
        default_pattern = re.compile(r"^default\s*:\s*([012])$")
        for raw_entry in spec.split(","):
            entry = raw_entry.strip()
            default_match = default_pattern.fullmatch(entry)
            if default_match is not None:
                if default_seen:
                    raise ValueError(
                        "LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER may contain "
                        "only one default entry"
                    )
                default_seen = True
                default_lookahead = int(default_match.group(1))
                continue
            match = entry_pattern.fullmatch(entry)
            if match is None:
                raise ValueError(
                    "LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER entries must be "
                    "DEFAULT:0|1|2, LAYER:0|1|2, or START-END:0|1|2; got "
                    + repr(raw_entry)
                )
            start = int(match.group(1))
            end = int(match.group(2) or start)
            lookahead = int(match.group(3))
            if end < start:
                raise ValueError(
                    "LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER range end must "
                    f"not precede its start: {entry!r}"
                )
            for layer_id in range(start, end + 1):
                previous = by_layer.get(layer_id)
                if previous is not None and previous != lookahead:
                    raise ValueError(
                        "LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER has "
                        f"conflicting values for layer {layer_id}"
                    )
                by_layer[layer_id] = lookahead
        return default_lookahead, by_layer


def build_residual_prefetch_sources(
    target_layer_ids: list[int],
    policy: CSAPrefetchLookaheadPolicy,
    *,
    excluded_target_layer_ids: Iterable[int] = (),
) -> dict[int, tuple[int, int]]:
    """Build one exact-lookahead source hook per enabled target layer.

    Args:
        target_layer_ids: Transformer layer ids containing CSA targets.
        policy: Per-target lookahead policy.
        excluded_target_layer_ids: Targets owned by a deterministic retrieval
            path, such as early dense shard-gather. Excluded targets never
            receive a speculative source hook.

    Returns:
        Mapping ``source_layer -> (target_layer, prefetch_level)``. Each
        enabled target has exactly one source at ``target - lookahead``.
        Disabled targets have no source.

    Raises:
        ValueError: If two targets require incompatible work from one source
            layer. The current decoder hook supports one target per source.
    """
    sources: dict[int, tuple[int, int]] = {}
    excluded = {int(layer_id) for layer_id in excluded_target_layer_ids}
    for target_layer_id in target_layer_ids:
        target = int(target_layer_id)
        if target in excluded:
            continue
        lookahead = policy.lookahead_for(target)
        if lookahead == 0:
            continue
        source = target - lookahead
        work = (target, lookahead)
        previous = sources.get(source)
        if previous is not None and previous != work:
            raise ValueError(
                f"source layer {source} maps to both {previous} and {work}"
            )
        sources[source] = work
    return sources
