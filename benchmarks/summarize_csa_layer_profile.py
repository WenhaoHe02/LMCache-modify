# SPDX-License-Identifier: Apache-2.0
"""Summarize per-layer CSA/Tutti timing lines from a full server log."""

# Standard
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


_EVENT_RE = re.compile(
    r"(?P<event>TUTTI_LAYER_PROFILE|CSA_LAYER_PROFILE|"
    r"CSAAttentionKVPrefetchManager: correction)\s+(?P<fields>.*)"
)
_FIELD_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s,]+)")


def _coerce(value: str) -> Any:
    """Convert an unquoted profile value to int/float when possible."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_profile(path: Path) -> list[dict[str, Any]]:
    """Parse supported profile records from ``path``."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            match = _EVENT_RE.search(line)
            if match is None:
                continue
            record = {
                "event": match.group("event"),
                "line": line_number,
            }
            record.update(
                {
                    field.group("key"): _coerce(field.group("value"))
                    for field in _FIELD_RE.finditer(match.group("fields"))
                }
            )
            records.append(record)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numeric fields by event and transformer layer."""
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["event"]), record.get("layer", "none"))].append(record)

    groups = []
    for (event, layer), samples in sorted(
        grouped.items(), key=lambda item: (str(item[0][1]), item[0][0])
    ):
        numeric_keys = sorted(
            {
                key
                for sample in samples
                for key, value in sample.items()
                if isinstance(value, (int, float))
                and key not in {"device", "line", "layer"}
            }
        )
        metrics = {}
        for key in numeric_keys:
            values = [float(sample[key]) for sample in samples if key in sample]
            metrics[key] = {
                "min": min(values),
                "p50": statistics.median(values),
                "max": max(values),
            }
        groups.append(
            {
                "event": event,
                "layer": layer,
                "samples": len(samples),
                "devices": sorted(
                    {str(sample["device"]) for sample in samples if "device" in sample}
                ),
                "metrics": metrics,
            }
        )
    return {"records": len(records), "groups": groups}


def main() -> None:
    """Parse one server log and print a JSON layer summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(parse_profile(args.log))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
