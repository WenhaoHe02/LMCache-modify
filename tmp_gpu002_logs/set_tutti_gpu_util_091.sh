# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
set -euo pipefail

tmp_file=$(mktemp)
python3 - "${tmp_file}" <<'PY'
import sys
from pathlib import Path

path = Path("/tmp/startup_256k_tutti.sh")
text = path.read_text()
old = "--gpu-memory-utilization 0.92"
new = "--gpu-memory-utilization 0.91"
if old not in text and new not in text:
    raise SystemExit("gpu-memory-utilization flag not found")
Path(sys.argv[1]).write_text(text.replace(old, new))
PY

sudo cp "${tmp_file}" /tmp/startup_256k_tutti.sh
rm -f "${tmp_file}"

grep -n -- "--gpu-memory-utilization" /tmp/startup_256k_tutti.sh
