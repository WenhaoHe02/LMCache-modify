# SPDX-License-Identifier: Apache-2.0
"""Pure logical planning primitives for lowering sparse indices to I/O.

This module deliberately has no dependency on CUDA, vLLM, or a storage
backend.  It represents content-addressed sparse selections, binds them to
physical byte ranges, and chooses a lowering with an injectable cost model.
The resulting plan can be consumed by a runtime without granting the planner
access to runtime-private state.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


class Correctness(str, Enum):
    """Correctness guarantee attached to a logical plan."""

    EXACT = "exact"
    SPECULATIVE = "speculative"


class LayerRole(str, Enum):
    """Attention-layer family that consumes a plan."""

    CSA = "csa"
    HCA = "hca"
    SWA = "swa"
    DSA = "dsa"


class PlanRole(str, Enum):
    """Logical tensor family materialized by a plan."""

    ATTENTION_KV = "attention_kv"
    INDEXER_K = "indexer_k"


class LogicalCodec(str, Enum):
    """Canonical encoding used by :class:`LogicalSelection`."""

    LIST = "list"
    BITMAP = "bitmap"
    RANGES = "ranges"
    XOR_DELTA = "xor_delta"


class PlacementKind(str, Enum):
    """Physical lowering selected by the cost model."""

    LIST = "list"
    RANGE = "range"
    BULK = "bulk"


@dataclass(frozen=True, slots=True)
class ContentKey:
    """Content identity shared by all decisions for one cached prefix.

    Args:
        model_fingerprint: Stable digest of model weights and adapters.
        prefix_root: Ordered Merkle root of the cached logical chunks.
        token_count: Number of tokens represented by ``prefix_root``.
        layout_fingerprint: Stable attention/KV layout ABI digest.
    """

    model_fingerprint: str
    prefix_root: str
    token_count: int
    layout_fingerprint: str

    def __post_init__(self) -> None:
        if not self.model_fingerprint:
            raise ValueError("model_fingerprint must be non-empty")
        if not self.prefix_root:
            raise ValueError("prefix_root must be non-empty")
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if not self.layout_fingerprint:
            raise ValueError("layout_fingerprint must be non-empty")


@dataclass(frozen=True, slots=True)
class DecisionKey:
    """Exact identity of one layer/query-span planning decision.

    Args:
        content: Content-addressed prefix identity.
        layer_id: Transformer layer id.
        layer_role: Attention family of the target layer.
        plan_role: Tensor family to materialize.
        query_start: Absolute position of the first query row.
        query_count: Number of query rows covered by the decision.
        query_digest: Digest of query tokens in the covered span.
    """

    content: ContentKey
    layer_id: int
    layer_role: LayerRole
    plan_role: PlanRole
    query_start: int
    query_count: int
    query_digest: str

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if self.query_start < 0:
            raise ValueError("query_start must be non-negative")
        if self.query_count < 0:
            raise ValueError("query_count must be non-negative")
        if not self.query_digest:
            raise ValueError("query_digest must be non-empty")


@dataclass(frozen=True, slots=True)
class Coverage:
    """Logical selection density.

    Args:
        selected: Number of selected logical units.
        universe: Number of addressable logical units.
    """

    selected: int
    universe: int

    def __post_init__(self) -> None:
        if self.universe < 0:
            raise ValueError("universe must be non-negative")
        if self.selected < 0 or self.selected > self.universe:
            raise ValueError("selected must lie in [0, universe]")

    @property
    def ratio(self) -> float:
        """Return selected density, with an empty universe defined as zero."""
        return 0.0 if self.universe == 0 else self.selected / self.universe


@dataclass(frozen=True, slots=True)
class Confidence:
    """Empirical confidence attached to a speculative selection.

    Args:
        recall_lower_bound: Estimated lower bound in the inclusive range
            ``[0, 1]``.
        samples: Number of observations supporting the estimate.
    """

    recall_lower_bound: float
    samples: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.recall_lower_bound <= 1.0:
            raise ValueError("recall_lower_bound must lie in [0, 1]")
        if self.samples < 0:
            raise ValueError("samples must be non-negative")


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varints cannot encode negative values")
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _decode_varints(payload: bytes) -> tuple[int, ...]:
    values: list[int] = []
    value = 0
    shift = 0
    for byte in payload:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            if shift > 63:
                raise ValueError("varint is too large")
            continue
        values.append(value)
        value = 0
        shift = 0
    if shift:
        raise ValueError("truncated varint payload")
    return tuple(values)


def _validate_ids(ids: Iterable[int], universe_size: int) -> tuple[int, ...]:
    if universe_size < 0:
        raise ValueError("universe_size must be non-negative")
    normalized = tuple(sorted({int(logical_id) for logical_id in ids}))
    if normalized and normalized[0] < 0:
        raise ValueError("logical ids must be non-negative")
    if normalized and normalized[-1] >= universe_size:
        raise ValueError("logical id lies outside universe_size")
    return normalized


def _encode_list(ids: Sequence[int]) -> bytes:
    payload = bytearray(_encode_varint(len(ids)))
    previous = -1
    for logical_id in ids:
        payload.extend(_encode_varint(logical_id - previous))
        previous = logical_id
    return bytes(payload)


def _decode_list(payload: bytes) -> tuple[int, ...]:
    values = _decode_varints(payload)
    if not values:
        raise ValueError("LIST payload has no item count")
    count = values[0]
    if len(values) != count + 1:
        raise ValueError("LIST item count does not match payload")
    ids: list[int] = []
    previous = -1
    for delta in values[1:]:
        if delta <= 0:
            raise ValueError("LIST deltas must be positive")
        previous += delta
        ids.append(previous)
    return tuple(ids)


def _encode_ranges(ids: Sequence[int]) -> bytes:
    ranges: list[tuple[int, int]] = []
    if ids:
        start = ids[0]
        previous = start
        for logical_id in ids[1:]:
            if logical_id == previous + 1:
                previous = logical_id
                continue
            ranges.append((start, previous - start + 1))
            start = logical_id
            previous = logical_id
        ranges.append((start, previous - start + 1))
    payload = bytearray(_encode_varint(len(ranges)))
    previous_end = 0
    for start, length in ranges:
        payload.extend(_encode_varint(start - previous_end))
        payload.extend(_encode_varint(length))
        previous_end = start + length
    return bytes(payload)


def _decode_ranges(payload: bytes) -> tuple[int, ...]:
    values = _decode_varints(payload)
    if not values:
        raise ValueError("RANGES payload has no range count")
    count = values[0]
    if len(values) != 1 + count * 2:
        raise ValueError("RANGES item count does not match payload")
    ids: list[int] = []
    previous_end = 0
    for index in range(count):
        start_delta = values[1 + index * 2]
        length = values[2 + index * 2]
        if length <= 0:
            raise ValueError("RANGES lengths must be positive")
        start = previous_end + start_delta
        ids.extend(range(start, start + length))
        previous_end = start + length
    return tuple(ids)


@dataclass(frozen=True, slots=True)
class LogicalSelection:
    """Immutable encoded set of logical block or chunk ids.

    Args:
        codec: Canonical representation used by ``payload``.
        universe_size: Exclusive upper bound for represented ids.
        payload: Codec-specific canonical bytes.
        base_checksum: Required base selection checksum for ``XOR_DELTA``.
    """

    codec: LogicalCodec
    universe_size: int
    payload: bytes
    base_checksum: str | None = None

    def __post_init__(self) -> None:
        if self.universe_size < 0:
            raise ValueError("universe_size must be non-negative")
        if self.codec is LogicalCodec.BITMAP:
            expected = (self.universe_size + 7) // 8
            if len(self.payload) != expected:
                raise ValueError("BITMAP payload length does not match universe_size")
            if self.universe_size % 8 and self.payload:
                valid_mask = (1 << (self.universe_size % 8)) - 1
                if self.payload[-1] & ~valid_mask:
                    raise ValueError("BITMAP sets bits outside universe_size")
        if self.codec is LogicalCodec.XOR_DELTA:
            if not self.base_checksum:
                raise ValueError("XOR_DELTA requires base_checksum")
        elif self.base_checksum is not None:
            raise ValueError("base_checksum is only valid for XOR_DELTA")

    @classmethod
    def from_ids(
        cls,
        ids: Iterable[int],
        universe_size: int,
        codec: LogicalCodec = LogicalCodec.LIST,
        *,
        base: "LogicalSelection | None" = None,
    ) -> "LogicalSelection":
        """Encode logical ids with a selected codec.

        Args:
            ids: Logical ids to encode.
            universe_size: Exclusive upper bound for ids.
            codec: Desired canonical codec.
            base: Base selection required by ``XOR_DELTA``.

        Returns:
            Immutable encoded selection.

        Raises:
            ValueError: If ids are invalid or the delta base is incompatible.
        """
        normalized = _validate_ids(ids, universe_size)
        base_checksum: str | None = None
        encoded_ids = normalized
        if codec is LogicalCodec.XOR_DELTA:
            if base is None:
                raise ValueError("XOR_DELTA requires a base selection")
            if base.universe_size != universe_size:
                raise ValueError("delta base universe_size mismatch")
            encoded_ids = tuple(sorted(set(normalized) ^ set(base.ids())))
            base_checksum = base.checksum()
            payload = _encode_list(encoded_ids)
        elif codec is LogicalCodec.LIST:
            payload = _encode_list(encoded_ids)
        elif codec is LogicalCodec.RANGES:
            payload = _encode_ranges(encoded_ids)
        elif codec is LogicalCodec.BITMAP:
            bitmap = bytearray((universe_size + 7) // 8)
            for logical_id in encoded_ids:
                bitmap[logical_id >> 3] |= 1 << (logical_id & 7)
            payload = bytes(bitmap)
        else:  # pragma: no cover - exhaustive enum guard
            raise ValueError(f"unsupported codec {codec}")
        return cls(codec, universe_size, payload, base_checksum)

    def ids(self, *, base: "LogicalSelection | None" = None) -> tuple[int, ...]:
        """Decode selected ids in ascending order.

        Args:
            base: Required base selection for ``XOR_DELTA``.

        Returns:
            Sorted unique logical ids.

        Raises:
            ValueError: If payload bytes or the supplied delta base are invalid.
        """
        if self.codec is LogicalCodec.BITMAP:
            decoded = tuple(
                byte_index * 8 + bit_index
                for byte_index, byte in enumerate(self.payload)
                for bit_index in range(8)
                if byte & (1 << bit_index)
            )
        elif self.codec is LogicalCodec.RANGES:
            decoded = _decode_ranges(self.payload)
        else:
            decoded = _decode_list(self.payload)
        if self.codec is LogicalCodec.XOR_DELTA:
            if base is None:
                raise ValueError("decoding XOR_DELTA requires its base")
            if base.universe_size != self.universe_size:
                raise ValueError("delta base universe_size mismatch")
            if base.checksum() != self.base_checksum:
                raise ValueError("delta base checksum mismatch")
            decoded = tuple(sorted(set(decoded) ^ set(base.ids())))
        return _validate_ids(decoded, self.universe_size)

    def checksum(self) -> str:
        """Return the canonical SHA-256 checksum of this encoded selection."""
        return canonical_checksum(self)

    def coverage(self, *, base: "LogicalSelection | None" = None) -> Coverage:
        """Return logical coverage represented by this selection.

        Args:
            base: Required base for an ``XOR_DELTA`` selection.

        Returns:
            Selected and universe cardinalities.
        """
        return Coverage(len(self.ids(base=base)), self.universe_size)


@dataclass(frozen=True, slots=True)
class LayerPlan:
    """Content-addressed logical selection for one target layer.

    Args:
        key: Exact decision identity.
        correctness: Whether the plan may affect semantics or only prefetch.
        selection: Encoded logical selection.
        coverage: Declared selection coverage.
        confidence: Empirical confidence for the selection.
    """

    key: DecisionKey
    correctness: Correctness
    selection: LogicalSelection
    coverage: Coverage
    confidence: Confidence

    def __post_init__(self) -> None:
        if self.selection.codec is not LogicalCodec.XOR_DELTA:
            actual = self.selection.coverage()
            if actual != self.coverage:
                raise ValueError("coverage does not match logical selection")

    @property
    def plan_hash(self) -> str:
        """Return the canonical content hash of this plan."""
        return canonical_checksum(self)

    def require_key(self, expected: DecisionKey) -> None:
        """Reject reuse under a different exact decision key.

        Args:
            expected: Decision key required by the caller.

        Raises:
            ValueError: If the stored and expected keys differ.
        """
        if self.key != expected:
            raise ValueError("exact decision key mismatch")


@dataclass(frozen=True, slots=True)
class LogicalBlockSource:
    """Physical source bytes for one logical id.

    Args:
        logical_id: Logical block or chunk id.
        path: Storage object or raw-region path.
        chunk_id: Immutable source chunk identity within ``path``.
        offset: Byte offset of the logical payload.
        length: Logical payload length in bytes.
    """

    logical_id: int
    path: str
    chunk_id: str
    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.logical_id < 0:
            raise ValueError("logical_id must be non-negative")
        if not self.path or not self.chunk_id:
            raise ValueError("path and chunk_id must be non-empty")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")


@dataclass(frozen=True, slots=True)
class PhysicalRange:
    """Aligned physical read that carries one or more logical payloads.

    Args:
        path: Storage object or raw-region path.
        chunk_id: Source chunk identity. Ranges never merge across this field.
        offset: Aligned physical byte offset.
        length: Aligned physical read length.
        payload_skip: Bytes skipped before the first requested payload.
        payload_length: Contiguous requested payload length.
        logical_ids: Logical ids represented by the range.
    """

    path: str
    chunk_id: str
    offset: int
    length: int
    payload_skip: int
    payload_length: int
    logical_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.path or not self.chunk_id:
            raise ValueError("path and chunk_id must be non-empty")
        if self.offset < 0 or self.length <= 0:
            raise ValueError("physical range must have non-negative offset and length")
        if self.payload_skip < 0 or self.payload_length <= 0:
            raise ValueError("payload bounds must be positive and contained")
        if self.payload_skip + self.payload_length > self.length:
            raise ValueError("payload lies outside physical range")
        if tuple(sorted(set(self.logical_ids))) != self.logical_ids:
            raise ValueError("logical_ids must be sorted and unique")


@dataclass(frozen=True, slots=True)
class BoundPlan:
    """Logical layer plan lowered to physical reads.

    Args:
        logical_plan: Source logical layer plan.
        placement: Selected physical lowering kind.
        ranges: Aligned physical reads.
        estimated_cost: Cost-model score for the lowering.
    """

    logical_plan: LayerPlan
    placement: PlacementKind
    ranges: tuple[PhysicalRange, ...]
    estimated_cost: float

    @property
    def checksum(self) -> str:
        """Return the canonical SHA-256 checksum of the bound plan."""
        return canonical_checksum(self)


@dataclass(frozen=True, slots=True)
class LoweringCostModel:
    """Injectable linear cost model for physical lowering candidates.

    Args:
        list_base: Fixed cost of LIST lowering.
        list_per_range: Per-read LIST submission cost.
        range_base: Fixed cost of RANGE lowering.
        range_per_range: Per-read RANGE submission cost.
        bulk_base: Fixed cost of BULK lowering.
        bulk_per_range: Per-read BULK submission cost.
        byte_cost: Cost per physical byte read.
        overread_cost: Additional cost per byte beyond selected payload bytes.
    """

    list_base: float = 0.0
    list_per_range: float = 1.0
    range_base: float = 0.0
    range_per_range: float = 1.0
    bulk_base: float = 0.0
    bulk_per_range: float = 1.0
    byte_cost: float = 0.0
    overread_cost: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.list_base,
            self.list_per_range,
            self.range_base,
            self.range_per_range,
            self.bulk_base,
            self.bulk_per_range,
            self.byte_cost,
            self.overread_cost,
        )
        if any(value < 0 for value in values):
            raise ValueError("cost model parameters must be non-negative")

    def estimate(
        self,
        placement: PlacementKind,
        ranges: Sequence[PhysicalRange],
        selected_bytes: int,
    ) -> float:
        """Estimate one lowering candidate.

        Args:
            placement: Candidate placement kind.
            ranges: Candidate physical ranges.
            selected_bytes: Bytes belonging to selected logical ids.

        Returns:
            Scalar cost; lower values are preferred.
        """
        if selected_bytes < 0:
            raise ValueError("selected_bytes must be non-negative")
        base, per_range = {
            PlacementKind.LIST: (self.list_base, self.list_per_range),
            PlacementKind.RANGE: (self.range_base, self.range_per_range),
            PlacementKind.BULK: (self.bulk_base, self.bulk_per_range),
        }[placement]
        physical_bytes = sum(read_range.length for read_range in ranges)
        overread = max(0, physical_bytes - selected_bytes)
        return (
            base
            + per_range * len(ranges)
            + self.byte_cost * physical_bytes
            + self.overread_cost * overread
        )


_EMPTY_PREFIX_ROOT = hashlib.sha256(b"lmcache:index-to-io:empty:v1").digest()


@dataclass(frozen=True, slots=True)
class OrderedMerklePrefix:
    """Persistent ordered prefix accumulator with append and LCP support.

    ``roots[i]`` is the domain-separated hash-chain root after ``i + 1``
    chunk digests.  Keeping cumulative roots makes append immutable and lets
    two prefixes find their longest common prefix without exposing chunks.

    Args:
        roots: Cumulative binary roots, one per appended chunk digest.
    """

    roots: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        if any(len(root) != hashlib.sha256().digest_size for root in self.roots):
            raise ValueError("every Merkle root must be a SHA-256 digest")

    @property
    def chunk_count(self) -> int:
        """Return the number of appended logical chunks."""
        return len(self.roots)

    @property
    def root(self) -> str:
        """Return the current root as lowercase hexadecimal."""
        return (self.roots[-1] if self.roots else _EMPTY_PREFIX_ROOT).hex()

    def append(self, chunk_digest: bytes | str) -> "OrderedMerklePrefix":
        """Return a new prefix with one ordered chunk digest appended.

        Args:
            chunk_digest: Stable digest bytes or hexadecimal/text digest.

        Returns:
            New immutable prefix accumulator.

        Raises:
            ValueError: If the digest is empty.
        """
        digest = (
            chunk_digest.encode("utf-8")
            if isinstance(chunk_digest, str)
            else bytes(chunk_digest)
        )
        if not digest:
            raise ValueError("chunk_digest must be non-empty")
        previous = self.roots[-1] if self.roots else _EMPTY_PREFIX_ROOT
        hasher = hashlib.sha256()
        hasher.update(b"lmcache:index-to-io:append:v1\x00")
        hasher.update(previous)
        hasher.update(len(digest).to_bytes(8, "big"))
        hasher.update(digest)
        return OrderedMerklePrefix(self.roots + (hasher.digest(),))

    def extend(self, chunk_digests: Iterable[bytes | str]) -> "OrderedMerklePrefix":
        """Return a new prefix with all chunk digests appended in order.

        Args:
            chunk_digests: Ordered stable chunk digests.

        Returns:
            New immutable prefix accumulator.
        """
        result = self
        for chunk_digest in chunk_digests:
            result = result.append(chunk_digest)
        return result

    def prefix_root(self, chunk_count: int) -> str:
        """Return the root of the first ``chunk_count`` chunks.

        Args:
            chunk_count: Prefix length in chunks.

        Returns:
            Hexadecimal ordered prefix root.

        Raises:
            ValueError: If the requested prefix is outside this accumulator.
        """
        if chunk_count < 0 or chunk_count > len(self.roots):
            raise ValueError("chunk_count lies outside the accumulated prefix")
        if chunk_count == 0:
            return _EMPTY_PREFIX_ROOT.hex()
        return self.roots[chunk_count - 1].hex()

    def longest_common_prefix(self, other: "OrderedMerklePrefix") -> int:
        """Return the number of leading chunks shared with ``other``.

        Args:
            other: Prefix accumulator to compare.

        Returns:
            Longest common prefix length in chunks.
        """
        common = 0
        for left, right in zip(self.roots, other.roots, strict=False):
            if left != right:
                break
            common += 1
        return common


def canonical_checksum(value: object) -> str:
    """Return a stable SHA-256 checksum for supported plan values.

    Args:
        value: Nested frozen dataclass, mapping with string keys, sequence,
            enum, bytes, or JSON scalar.

    Returns:
        Lowercase SHA-256 hexadecimal digest.

    Raises:
        TypeError: If a value has no canonical representation.
    """
    canonical = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _canonical_value(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mapping keys must be strings")
        return {
            key: _canonical_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical value {type(value).__name__}")


def logical_missing(
    selection: LogicalSelection,
    resident: Iterable[int],
    inflight: Iterable[int],
    *,
    base: LogicalSelection | None = None,
    codec: LogicalCodec = LogicalCodec.LIST,
) -> LogicalSelection:
    """Return ``selection - (resident union inflight)``.

    Args:
        selection: Requested logical selection.
        resident: Logical ids already materialized.
        inflight: Logical ids with reads already submitted.
        base: Required base when ``selection`` is ``XOR_DELTA``.
        codec: Codec for the returned missing selection.

    Returns:
        Immutable missing logical selection.
    """
    selected = set(selection.ids(base=base))
    unavailable = set(_validate_ids(resident, selection.universe_size))
    unavailable.update(_validate_ids(inflight, selection.universe_size))
    return LogicalSelection.from_ids(
        selected - unavailable,
        selection.universe_size,
        codec,
    )


def bind_logical_selection(
    selection: LogicalSelection,
    sources: Mapping[int, LogicalBlockSource],
    *,
    alignment: int,
    merge_ranges: bool = True,
    base: LogicalSelection | None = None,
) -> tuple[PhysicalRange, ...]:
    """Bind a logical selection to aligned physical source ranges.

    Merging is allowed only when payload bytes are contiguous and both ranges
    have the same ``path`` and ``chunk_id``.  The union of returned
    ``logical_ids`` is guaranteed to equal the input logical selection.

    Args:
        selection: Logical ids to bind.
        sources: Public logical-id to physical-source map.
        alignment: Required physical offset and length alignment.
        merge_ranges: Whether contiguous payloads may be coalesced.
        base: Required delta base for ``XOR_DELTA``.

    Returns:
        Deterministically ordered aligned physical ranges.

    Raises:
        ValueError: If alignment is invalid, a source is missing/mismatched,
            or the resulting logical set is not equivalent.
    """
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    selected_ids = selection.ids(base=base)
    selected_sources: list[LogicalBlockSource] = []
    for logical_id in selected_ids:
        source = sources.get(logical_id)
        if source is None:
            raise ValueError(f"missing physical source for logical id {logical_id}")
        if source.logical_id != logical_id:
            raise ValueError("physical source key does not match logical_id")
        selected_sources.append(source)
    selected_sources.sort(
        key=lambda source: (source.path, source.chunk_id, source.offset)
    )

    groups: list[list[LogicalBlockSource]] = []
    for source in selected_sources:
        if (
            merge_ranges
            and groups
            and groups[-1][-1].path == source.path
            and groups[-1][-1].chunk_id == source.chunk_id
            and groups[-1][-1].offset + groups[-1][-1].length == source.offset
        ):
            groups[-1].append(source)
        else:
            groups.append([source])
    ranges = tuple(_range_from_sources(group, alignment) for group in groups)
    represented = tuple(
        sorted(
            logical_id for read_range in ranges for logical_id in read_range.logical_ids
        )
    )
    if represented != selected_ids:
        raise ValueError("physical binding is not set-equivalent to logical selection")
    return ranges


def _range_from_sources(
    sources: Sequence[LogicalBlockSource], alignment: int
) -> PhysicalRange:
    first = sources[0]
    payload_start = first.offset
    payload_end = sources[-1].offset + sources[-1].length
    aligned_start = payload_start - payload_start % alignment
    aligned_end = ((payload_end + alignment - 1) // alignment) * alignment
    return PhysicalRange(
        path=first.path,
        chunk_id=first.chunk_id,
        offset=aligned_start,
        length=aligned_end - aligned_start,
        payload_skip=payload_start - aligned_start,
        payload_length=payload_end - payload_start,
        logical_ids=tuple(sorted(source.logical_id for source in sources)),
    )


def _bulk_ranges(
    selected_ids: Sequence[int],
    sources: Mapping[int, LogicalBlockSource],
    alignment: int,
) -> tuple[PhysicalRange, ...]:
    selected_by_group: dict[tuple[str, str], list[int]] = {}
    all_by_group: dict[tuple[str, str], list[LogicalBlockSource]] = {}
    for source in sources.values():
        key = (source.path, source.chunk_id)
        all_by_group.setdefault(key, []).append(source)
    for logical_id in selected_ids:
        source = sources[logical_id]
        selected_by_group.setdefault((source.path, source.chunk_id), []).append(
            logical_id
        )

    result: list[PhysicalRange] = []
    for key in sorted(selected_by_group):
        group_sources = sorted(all_by_group[key], key=lambda source: source.offset)
        first = group_sources[0]
        payload_start = first.offset
        payload_end = max(source.offset + source.length for source in group_sources)
        aligned_start = payload_start - payload_start % alignment
        aligned_end = ((payload_end + alignment - 1) // alignment) * alignment
        result.append(
            PhysicalRange(
                path=key[0],
                chunk_id=key[1],
                offset=aligned_start,
                length=aligned_end - aligned_start,
                payload_skip=payload_start - aligned_start,
                payload_length=payload_end - payload_start,
                logical_ids=tuple(sorted(selected_by_group[key])),
            )
        )
    return tuple(result)


def lower_layer_plan(
    plan: LayerPlan,
    sources: Mapping[int, LogicalBlockSource],
    *,
    alignment: int,
    cost_model: LoweringCostModel,
    base: LogicalSelection | None = None,
) -> BoundPlan:
    """Choose the cheapest LIST, RANGE, or BULK physical lowering.

    Args:
        plan: Logical layer plan to lower.
        sources: Complete public source map for the logical universe relevant
            to BULK, including every selected id.
        alignment: Required physical read alignment.
        cost_model: Injectable pure cost model.
        base: Required base when the plan selection is ``XOR_DELTA``.

    Returns:
        Immutable bound plan with deterministic tie-breaking in
        LIST, RANGE, BULK order.
    """
    selected_ids = plan.selection.ids(base=base)
    list_ranges = bind_logical_selection(
        plan.selection,
        sources,
        alignment=alignment,
        merge_ranges=False,
        base=base,
    )
    range_ranges = bind_logical_selection(
        plan.selection,
        sources,
        alignment=alignment,
        merge_ranges=True,
        base=base,
    )
    bulk_ranges = _bulk_ranges(selected_ids, sources, alignment)
    selected_bytes = sum(sources[logical_id].length for logical_id in selected_ids)
    candidates = (
        (PlacementKind.LIST, list_ranges),
        (PlacementKind.RANGE, range_ranges),
        (PlacementKind.BULK, bulk_ranges),
    )
    placement, ranges = min(
        candidates,
        key=lambda candidate: cost_model.estimate(
            candidate[0], candidate[1], selected_bytes
        ),
    )
    represented = tuple(
        sorted(
            logical_id for read_range in ranges for logical_id in read_range.logical_ids
        )
    )
    if represented != selected_ids:
        raise ValueError("lowered physical plan is not set-equivalent")
    return BoundPlan(
        logical_plan=plan,
        placement=placement,
        ranges=ranges,
        estimated_cost=cost_model.estimate(placement, ranges, selected_bytes),
    )
