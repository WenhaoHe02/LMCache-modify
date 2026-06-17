#!/bin/bash
HDR=/usr/src/linux-headers-5.15.0-179-generic
sudo touch \
    "$HDR/arch/x86/kernel/asm-offsets.s" \
    "$HDR/include/generated/timeconst.h" \
    "$HDR/include/generated/bounds.h" \
    "$HDR/include/generated/asm-offsets.h"
echo "Touched all generated headers"
