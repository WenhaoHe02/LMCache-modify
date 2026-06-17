#!/bin/bash
set -e
MODDIR=/home/zbuser02/Tutti/backends/local/kernel_modules/snvme-5.15.0-public
cd "$MODDIR"
echo "=== Building (KBUILD_MODPOST_WARN=1) ==="
sudo KBUILD_MODPOST_WARN=1 make -j$(nproc) 2>&1
echo "=== Build exit: $? ==="
ls -lh *.ko 2>/dev/null || echo "No .ko files found"
