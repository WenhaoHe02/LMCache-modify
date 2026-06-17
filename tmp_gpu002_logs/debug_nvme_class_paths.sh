#!/usr/bin/env bash
set -euo pipefail

for c in /sys/class/nvme/nvme*; do
  [ -e "$c" ] || continue
  echo "CTRL $(basename "$c") -> $(readlink -f "$c")"
  find "$c" -maxdepth 4 -type l -o -type d | sed 's#^#  #'
done

echo "== block symlinks =="
for b in /sys/class/block/nvme*; do
  [ -e "$b" ] || continue
  echo "$(basename "$b") -> $(readlink -f "$b")"
done
