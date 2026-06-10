# SPDX-License-Identifier: Apache-2.0
# Standard
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json


@dataclass(frozen=True, slots=True)
class KVObjectId:
    """Stable identifier for one layer/block KV object.

    Args:
        model_id: Model or cache namespace owning the object.
        parallel_config_id: Parallel-layout namespace, such as TP size and rank map.
        rank: Local rank that owns the object shard.
        layer_id: Transformer layer index.
        role: KV role, for example ``csa``, ``hca``, ``swa`` or ``full``.
        block_id: vLLM/LMCache block identifier within the layer and role.
        schema_version: Identifier schema version.
    """

    model_id: str
    parallel_config_id: str
    rank: int
    layer_id: int
    role: str
    block_id: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Validate fields that would make the object ambiguous."""
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.parallel_config_id:
            raise ValueError("parallel_config_id must be non-empty")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if not self.role:
            raise ValueError("role must be non-empty")
        if not self.block_id:
            raise ValueError("block_id must be non-empty")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the identifier."""
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "parallel_config_id": self.parallel_config_id,
            "rank": self.rank,
            "layer_id": self.layer_id,
            "role": self.role,
            "block_id": self.block_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KVObjectId":
        """Build an identifier from a JSON-compatible dictionary.

        Args:
            value: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed object identifier.

        Raises:
            KeyError: If a required field is missing.
            ValueError: If a field value is invalid.
        """
        return cls(
            model_id=str(value["model_id"]),
            parallel_config_id=str(value["parallel_config_id"]),
            rank=int(value["rank"]),
            layer_id=int(value["layer_id"]),
            role=str(value["role"]),
            block_id=str(value["block_id"]),
            schema_version=int(value.get("schema_version", 1)),
        )

    def to_key(self) -> str:
        """Return a deterministic string key suitable for metadata indexes."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_key(cls, key: str) -> "KVObjectId":
        """Parse an identifier from :meth:`to_key` output.

        Args:
            key: Deterministic JSON key string.

        Returns:
            The reconstructed object identifier.

        Raises:
            json.JSONDecodeError: If the key is not valid JSON.
            KeyError: If a required field is missing.
            ValueError: If a field value is invalid.
        """
        value = json.loads(key)
        if not isinstance(value, dict):
            raise ValueError("KV object key must decode to a dictionary")
        return cls.from_dict(value)
