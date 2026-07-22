# SPDX-License-Identifier: Apache-2.0
"""Produce an auditable critical-path summary from an Nsys SQLite export.

The script reads the schema before querying it, uses nanoseconds as exported
by Nsight Systems, and saves every custom SQL statement next to the JSON
result. It intentionally reports time-union metrics rather than summing
overlapping streams into wall time.
"""

# Standard
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from statistics import median
from typing import Any, Iterable, Sequence


Interval = tuple[int, int]
GpuEvent = tuple[int, int, str, str, int, int, int]


def _merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Return the sorted union of half-open nanosecond intervals."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _interval_total(intervals: Iterable[Interval]) -> int:
    """Return union duration in nanoseconds."""
    return sum(end - start for start, end in _merge_intervals(intervals))


def _intersection_total(left: Iterable[Interval], right: Iterable[Interval]) -> int:
    """Return intersection duration of two interval unions in nanoseconds."""
    left_union = _merge_intervals(left)
    right_union = _merge_intervals(right)
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left_union) and right_index < len(right_union):
        left_start, left_end = left_union[left_index]
        right_start, right_end = right_union[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _percentile(values: Sequence[int], percentile: float) -> int:
    """Return a nearest-rank percentile from an integer sequence."""
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(percentile * len(ordered)) - 1))
    return ordered[index]


def _ns_to_ms(value: int) -> float:
    """Convert nanoseconds to milliseconds."""
    return value / 1_000_000.0


class NsysSqliteAnalysis:
    """Schema-aware analyzer for one Nsight Systems SQLite export."""

    def __init__(self, database_path: Path) -> None:
        """Open an Nsys SQLite database.

        Args:
            database_path: Existing SQLite export from ``nsys export``.

        Raises:
            FileNotFoundError: If the export does not exist.
            RuntimeError: If required CUDA activity tables are absent.
        """
        if not database_path.is_file():
            raise FileNotFoundError(database_path)
        self.database_path = database_path
        self.connection = sqlite3.connect(str(database_path))
        self.connection.row_factory = sqlite3.Row
        self.tables = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
        missing = sorted(required - self.tables)
        if missing:
            raise RuntimeError(f"missing required Nsys tables: {missing}")
        self.queries: dict[str, str] = {}

    def close(self) -> None:
        """Close the SQLite connection."""
        self.connection.close()

    def analyze(self, minimum_gap_ms: float, top: int) -> dict[str, Any]:
        """Build critical-path, kernel, synchronization, and NVTX summaries.

        Args:
            minimum_gap_ms: Minimum fully idle GPU gap to retain.
            top: Maximum rows in each ranked section.

        Returns:
            JSON-serializable analysis dictionary.
        """
        events = self._gpu_events()
        if not events:
            raise RuntimeError("the export contains no GPU activities")
        start_ns = min(event[0] for event in events)
        end_ns = max(event[1] for event in events)
        all_intervals = [(event[0], event[1]) for event in events]
        compute = [
            (start, end)
            for start, end, kind, name, _device, _stream, _pid in events
            if kind == "kernel" and "nccl" not in name.lower()
        ]
        communication = [
            (start, end)
            for start, end, kind, name, _device, _stream, _pid in events
            if kind == "kernel" and "nccl" in name.lower()
        ]
        memory = [
            (start, end)
            for start, end, kind, _name, _device, _stream, _pid in events
            if kind in {"memcpy", "memset"}
        ]
        active_ns = _interval_total(all_intervals)
        compute_ns = _interval_total(compute)
        communication_ns = _interval_total(communication)
        memory_ns = _interval_total(memory)
        compute_communication_overlap_ns = _intersection_total(
            compute,
            communication,
        )
        wall_ns = end_ns - start_ns
        result: dict[str, Any] = {
            "source": str(self.database_path),
            "time_unit": "nanoseconds in SQLite; milliseconds in this report",
            "schema": self._schema_summary(),
            "critical_path": {
                "start_ns": start_ns,
                "end_ns": end_ns,
                "wall_ms": _ns_to_ms(wall_ns),
                "gpu_any_active_ms": _ns_to_ms(active_ns),
                "gpu_fully_idle_ms": _ns_to_ms(max(0, wall_ns - active_ns)),
                "compute_union_ms": _ns_to_ms(compute_ns),
                "communication_union_ms": _ns_to_ms(communication_ns),
                "memcpy_memset_union_ms": _ns_to_ms(memory_ns),
                "compute_communication_overlap_ms": _ns_to_ms(
                    compute_communication_overlap_ns
                ),
                "exposed_communication_ms": _ns_to_ms(
                    max(0, communication_ns - compute_communication_overlap_ns)
                ),
                "communication_overlap_ratio": (
                    compute_communication_overlap_ns / communication_ns
                    if communication_ns
                    else 0.0
                ),
            },
            "per_process_device_critical_path": self._per_device_metrics(events),
            "largest_global_gpu_gaps": self._gpu_gaps(
                events,
                minimum_gap_ms,
                top,
            ),
            "top_kernels": self._kernel_summary(top),
            "top_cuda_runtime": self._runtime_summary(top),
            "top_os_runtime": self._os_runtime_summary(top),
            "longest_nvtx_ranges": self._nvtx_summary(top),
        }
        return result

    def save_queries(self, path: Path) -> None:
        """Save the exact custom SQL used by this analysis.

        Args:
            path: Destination ``.sql`` file.
        """
        sections = [
            f"-- {name}\n{query.strip()};\n" for name, query in self.queries.items()
        ]
        path.write_text("\n".join(sections), encoding="utf-8")

    def _schema_summary(self) -> dict[str, Any]:
        """Return table row counts and columns used for reproducibility."""
        relevant = [
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "NVTX_EVENTS",
            "OSRT_API",
            "StringIds",
        ]
        summary: dict[str, Any] = {}
        for table in relevant:
            if table not in self.tables:
                continue
            columns = [
                str(row[1])
                for row in self.connection.execute(f"PRAGMA table_info({table})")
            ]
            count = int(
                self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            summary[table] = {"rows": count, "columns": columns}
        return summary

    def _gpu_events(self) -> list[GpuEvent]:
        """Return normalized kernel, memcpy, and memset events."""
        kernel_query = """
            SELECT k.start, k.end, COALESCE(s.value, '<unknown>') AS name,
                   k.deviceId, k.streamId, COALESCE(k.globalPid, -1) AS globalPid
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            LEFT JOIN StringIds AS s ON s.id = k.demangledName
            ORDER BY k.start
        """
        self.queries["gpu_kernels"] = kernel_query
        events = [
            (
                int(row["start"]),
                int(row["end"]),
                "kernel",
                str(row["name"]),
                int(row["deviceId"]),
                int(row["streamId"]),
                int(row["globalPid"]),
            )
            for row in self.connection.execute(kernel_query)
        ]
        for table, kind in (
            ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
        ):
            if table not in self.tables:
                continue
            query = (
                "SELECT start, end, deviceId, streamId, "
                f"COALESCE(globalPid, -1) AS globalPid FROM {table} ORDER BY start"
            )
            self.queries[f"gpu_{kind}"] = query
            events.extend(
                (
                    int(row["start"]),
                    int(row["end"]),
                    kind,
                    kind,
                    int(row["deviceId"]),
                    int(row["streamId"]),
                    int(row["globalPid"]),
                )
                for row in self.connection.execute(query)
            )
        return sorted(events)

    def _per_device_metrics(self, events: Sequence[GpuEvent]) -> list[dict[str, Any]]:
        """Return activity unions per process and CUDA device."""
        process_names: dict[int, str] = {}
        if "PROCESSES" in self.tables:
            process_names = {
                int(row["globalPid"]): str(row["name"])
                for row in self.connection.execute(
                    "SELECT globalPid, name FROM PROCESSES"
                )
            }
        grouped: dict[tuple[int, int], list[GpuEvent]] = defaultdict(list)
        for event in events:
            grouped[(event[6], event[4])].append(event)
        rows: list[dict[str, Any]] = []
        for (global_pid, device), group in sorted(grouped.items()):
            start = min(event[0] for event in group)
            end = max(event[1] for event in group)
            all_intervals = [(event[0], event[1]) for event in group]
            compute = [
                (event[0], event[1])
                for event in group
                if event[2] == "kernel" and "nccl" not in event[3].lower()
            ]
            communication = [
                (event[0], event[1])
                for event in group
                if event[2] == "kernel" and "nccl" in event[3].lower()
            ]
            memory = [
                (event[0], event[1])
                for event in group
                if event[2] in {"memcpy", "memset"}
            ]
            active = _interval_total(all_intervals)
            communication_time = _interval_total(communication)
            overlap = _intersection_total(compute, communication)
            rows.append(
                {
                    "global_pid": global_pid,
                    "process": process_names.get(global_pid, "<unknown>"),
                    "device": device,
                    "wall_ms": _ns_to_ms(end - start),
                    "gpu_active_ms": _ns_to_ms(active),
                    "gpu_idle_ms": _ns_to_ms(max(0, end - start - active)),
                    "compute_union_ms": _ns_to_ms(_interval_total(compute)),
                    "communication_union_ms": _ns_to_ms(communication_time),
                    "memcpy_memset_union_ms": _ns_to_ms(_interval_total(memory)),
                    "compute_communication_overlap_ms": _ns_to_ms(overlap),
                    "exposed_communication_ms": _ns_to_ms(
                        max(0, communication_time - overlap)
                    ),
                }
            )
        return rows

    def _gpu_gaps(
        self,
        events: Sequence[GpuEvent],
        minimum_gap_ms: float,
        top: int,
    ) -> list[dict[str, Any]]:
        """Return gaps where every captured GPU is idle."""
        merged = _merge_intervals((event[0], event[1]) for event in events)
        threshold_ns = int(minimum_gap_ms * 1_000_000)
        gaps: list[dict[str, Any]] = []
        for index in range(len(merged) - 1):
            start = merged[index][1]
            end = merged[index + 1][0]
            if end - start < threshold_ns:
                continue
            before = max(
                (event for event in events if event[1] <= start),
                key=lambda event: event[1],
                default=None,
            )
            after = min(
                (event for event in events if event[0] >= end),
                key=lambda event: event[0],
                default=None,
            )
            gaps.append(
                {
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ms": _ns_to_ms(end - start),
                    "before": self._event_identity(before),
                    "after": self._event_identity(after),
                }
            )
        return sorted(gaps, key=lambda gap: gap["duration_ms"], reverse=True)[:top]

    @staticmethod
    def _event_identity(
        event: GpuEvent | None,
    ) -> dict[str, Any] | None:
        """Return a compact event identity for a gap boundary."""
        if event is None:
            return None
        start, end, kind, name, device, stream, global_pid = event
        return {
            "kind": kind,
            "name": name,
            "device": device,
            "stream": stream,
            "global_pid": global_pid,
            "start_ns": start,
            "end_ns": end,
        }

    def _kernel_summary(self, top: int) -> list[dict[str, Any]]:
        """Aggregate kernel duration distributions by demangled name."""
        query = """
            SELECT COALESCE(s.value, '<unknown>') AS name, k.end - k.start AS duration
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            LEFT JOIN StringIds AS s ON s.id = k.demangledName
        """
        self.queries["kernel_durations"] = query
        durations: dict[str, list[int]] = defaultdict(list)
        for row in self.connection.execute(query):
            durations[str(row["name"])].append(int(row["duration"]))
        rows = []
        for name, values in durations.items():
            rows.append(
                {
                    "name": name,
                    "count": len(values),
                    "sum_ms_overlapping_streams": _ns_to_ms(sum(values)),
                    "median_ms": _ns_to_ms(int(median(values))),
                    "p95_ms": _ns_to_ms(_percentile(values, 0.95)),
                    "max_ms": _ns_to_ms(max(values)),
                }
            )
        return sorted(
            rows,
            key=lambda row: row["sum_ms_overlapping_streams"],
            reverse=True,
        )[:top]

    def _runtime_summary(self, top: int) -> list[dict[str, Any]]:
        """Aggregate CUDA Runtime API wall durations by function name."""
        if "CUPTI_ACTIVITY_KIND_RUNTIME" not in self.tables:
            return []
        query = """
            SELECT COALESCE(s.value, '<unknown>') AS name,
                   COUNT(*) AS count, SUM(r.end - r.start) AS duration
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            LEFT JOIN StringIds AS s ON s.id = r.nameId
            GROUP BY name
            ORDER BY duration DESC
            LIMIT ?
        """
        self.queries["cuda_runtime_summary"] = query.replace("?", str(top))
        return [
            {
                "name": str(row["name"]),
                "count": int(row["count"]),
                "sum_ms_across_threads": _ns_to_ms(int(row["duration"] or 0)),
            }
            for row in self.connection.execute(query, (top,))
        ]

    def _os_runtime_summary(self, top: int) -> list[dict[str, Any]]:
        """Aggregate OS runtime durations by function name."""
        if "OSRT_API" not in self.tables:
            return []
        query = """
            SELECT COALESCE(s.value, '<unknown>') AS name,
                   COUNT(*) AS count, SUM(o.end - o.start) AS duration
            FROM OSRT_API AS o
            LEFT JOIN StringIds AS s ON s.id = o.nameId
            GROUP BY name
            ORDER BY duration DESC
            LIMIT ?
        """
        self.queries["os_runtime_summary"] = query.replace("?", str(top))
        return [
            {
                "name": str(row["name"]),
                "count": int(row["count"]),
                "sum_ms_across_threads": _ns_to_ms(int(row["duration"] or 0)),
            }
            for row in self.connection.execute(query, (top,))
        ]

    def _nvtx_summary(self, top: int) -> list[dict[str, Any]]:
        """Return the longest completed NVTX ranges with resolved text."""
        if "NVTX_EVENTS" not in self.tables:
            return []
        query = """
            SELECT n.start, n.end,
                   COALESCE(n.text, s.value, '<unnamed>') AS name,
                   n.globalTid, n.endGlobalTid
            FROM NVTX_EVENTS AS n
            LEFT JOIN StringIds AS s ON s.id = n.textId
            WHERE n.end IS NOT NULL AND n.end > n.start
            ORDER BY n.end - n.start DESC
            LIMIT ?
        """
        self.queries["longest_nvtx_ranges"] = query.replace("?", str(top))
        return [
            {
                "name": str(row["name"]),
                "duration_ms": _ns_to_ms(int(row["end"]) - int(row["start"])),
                "start_ns": int(row["start"]),
                "end_ns": int(row["end"]),
                "start_global_tid": row["globalTid"],
                "end_global_tid": row["endGlobalTid"],
            }
            for row in self.connection.execute(query, (top,))
        ]


def main() -> None:
    """Parse command-line arguments and write JSON plus SQL evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-gap-ms", type=float, default=0.1)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    output = args.output or args.sqlite.with_suffix(".critical_path.json")
    sql_output = output.with_suffix(".sql")
    analysis = NsysSqliteAnalysis(args.sqlite)
    try:
        result = analysis.analyze(args.minimum_gap_ms, args.top)
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        analysis.save_queries(sql_output)
    finally:
        analysis.close()
    print(output)
    print(sql_output)


if __name__ == "__main__":
    main()
