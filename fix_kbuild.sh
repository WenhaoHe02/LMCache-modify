#!/bin/bash
# Workaround for Ubuntu kernel-headers missing kernel/time/timeconst.bc.
# The Kbuild rule requires this file, but Ubuntu ships only the header.
# Fix: create a stub .bc file and remove FORCE from the rule so make
# won't attempt to regenerate timeconst.h (which already exists).
set -e

HDR=/usr/src/linux-headers-5.15.0-179-generic

# Create the stub kernel/time directory and timeconst.bc
sudo mkdir -p "$HDR/kernel/time"
# This is a minimal bc script that outputs the existing timeconst.h content.
# Since filechk compares output to existing file, if output matches, no write.
# For a stub, we just copy the existing header content into a bc script that
# uses print() to emit it — but the simplest approach is to patch out FORCE.
sudo touch "$HDR/kernel/time/timeconst.bc"

# Make timeconst.bc appear newer than timeconst.h so make is satisfied
sudo touch "$HDR/kernel/time/timeconst.bc"

# Also remove FORCE from the timeconst rule in Kbuild so make won't
# unconditionally try to rebuild it. Backup Kbuild first.
sudo cp "$HDR/Kbuild" "$HDR/Kbuild.bak"
sudo sed -i 's|$(timeconst-file): kernel/time/timeconst.bc FORCE|$(timeconst-file): kernel/time/timeconst.bc|g' "$HDR/Kbuild"

echo "=== Patched Kbuild ==="
grep -A2 timeconst-file "$HDR/Kbuild" | head -8
