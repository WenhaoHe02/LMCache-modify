# SPDX-License-Identifier: Apache-2.0
# Standard
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import json
import threading

# First Party
from lmcache.v1.kv_object_store.object_id import KVObjectId
from lmcache.v1.kv_object_store.record import KVObjectRecord, KVObjectState


class KVObjectMetadataStore:
    """Thread-safe in-memory index for KV object metadata.

    The store is intentionally storage-engine agnostic: overlap code can look up
    layer/block objects here, while a lower layer decides whether reads use
    Tutti, normal LMCache CPU reads, or another backend.
    """

    def __init__(self) -> None:
        """Create an empty metadata store."""
        self._records: dict[str, KVObjectRecord] = {}
        self._lock = threading.RLock()

    def put(self, record: KVObjectRecord) -> None:
        """Insert or replace one metadata record.

        Args:
            record: Metadata record to store.
        """
        with self._lock:
            self._records[record.object_id.to_key()] = record

    def get(self, object_id: KVObjectId) -> KVObjectRecord | None:
        """Return one record if present.

        Args:
            object_id: Object identifier to look up.

        Returns:
            The matching record, or ``None`` when absent.
        """
        with self._lock:
            return self._records.get(object_id.to_key())

    def get_many(
        self,
        object_ids: Sequence[KVObjectId],
        *,
        ready_only: bool = True,
    ) -> list[KVObjectRecord | None]:
        """Look up many objects in request order.

        Args:
            object_ids: Object identifiers to look up.
            ready_only: When true, allocated or evicted records are treated as
                misses.

        Returns:
            One result per input object id; absent entries are ``None``.
        """
        with self._lock:
            records = [
                self._records.get(object_id.to_key()) for object_id in object_ids
            ]
        if not ready_only:
            return records
        return [
            record
            if record is not None and record.state == KVObjectState.READY
            else None
            for record in records
        ]

    def delete(self, object_id: KVObjectId) -> KVObjectRecord | None:
        """Remove one record from the store.

        Args:
            object_id: Object identifier to remove.

        Returns:
            The removed record, or ``None`` when absent.
        """
        with self._lock:
            return self._records.pop(object_id.to_key(), None)

    def records(self) -> list[KVObjectRecord]:
        """Return a snapshot of all records."""
        with self._lock:
            return list(self._records.values())

    def ready_records(self) -> list[KVObjectRecord]:
        """Return a snapshot of records that are ready for retrieval."""
        with self._lock:
            return [
                record
                for record in self._records.values()
                if record.state == KVObjectState.READY
            ]

    def dump_jsonl(self, path: Path) -> None:
        """Write all records to a JSONL metadata file.

        Args:
            path: Destination JSONL path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = list(self._records.values())
        with path.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record.to_dict(), sort_keys=True))
                output.write("\n")

    @classmethod
    def load_jsonl(cls, path: Path) -> "KVObjectMetadataStore":
        """Load metadata records from a JSONL file.

        Args:
            path: Source JSONL path.

        Returns:
            A metadata store populated with the file contents.
        """
        store = cls()
        if not path.exists():
            return store
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    store.put(KVObjectRecord.from_dict(value))
        return store

    def extend(self, records: Iterable[KVObjectRecord]) -> None:
        """Insert multiple records.

        Args:
            records: Records to insert or replace.
        """
        with self._lock:
            for record in records:
                self._records[record.object_id.to_key()] = record
