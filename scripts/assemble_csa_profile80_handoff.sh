#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

delivery_root=${DELIVERY_ROOT:-/home/zbuser02/colleague_handoff/csa_profile80_1p491_20260728}
upload_root=${UPLOAD_ROOT:-/home/zbuser02}
protected_root=${LMCACHE_PROTECTED_ROOT:-/home/zbuser02/protected_lmcache_versions/csa_profile80_hybrid_1p491_20260727}
bundle_name=csa_profile80_1p491_handoff_20260728.tar.gz
extracted_name=csa_profile80_1p491_handoff_20260728

mkdir -p "${delivery_root}/extracted"
cp -f "${upload_root}/${bundle_name}" "${delivery_root}/${bundle_name}"
cp -f "${upload_root}/FULL_PACKAGE_SHA256SUMS" \
  "${delivery_root}/FULL_PACKAGE_SHA256SUMS"
ln -sfn "${protected_root}/protected_image.tar.zst" \
  "${delivery_root}/protected_image.tar.zst"

tar -xzf "${delivery_root}/${bundle_name}" -C "${delivery_root}/extracted"
runtime="${delivery_root}/extracted/${extracted_name}/runtime"
ln -sfn "${delivery_root}/protected_image.tar.zst" \
  "${runtime}/protected_image.tar.zst"

bash -n \
  "${runtime}/scripts/restore_container.sh" \
  "${runtime}/scripts/run_container_cp8_ab.sh" \
  "${runtime}/scripts/startup_cp8_ab.sh"
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  "${runtime}/scripts/run_hermes_trial2_480k200.py"

(
  cd "${delivery_root}"
  sha256sum -c FULL_PACKAGE_SHA256SUMS
)

printf 'HANDOFF_READY=%s\n' "${delivery_root}"
