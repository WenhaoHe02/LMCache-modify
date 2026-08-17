# DSv4 CP8 Q×K Fusion Final V1

This branch freezes the gpu002 production source validated on 2026-08-18.
It builds on the proven GLM/DSv4 V5 version and makes two exact DSv4 read-path
optimizations:

- accumulate selected pages while the official true indexer emits its chunks,
  reducing final miss correction from another Q×K scan to a page-bitmap scan;
- translate logical pages to compact pages inside the existing sparse-index
  combiner, removing the separate Q×K remap pass.

The full-scan correction remains as a fail-closed fallback. Compact page-map
loads reject negative, out-of-range, and unselected entries.

## gpu002 validation

Each result contains one warmup and three formal repetitions.

| Cell | Final mean | Proven V5 mean | Change |
|---|---:|---:|---:|
| 32K prefix + 8K recompute | 0.397595 s | 0.399504 s | -1.909 ms |
| 32K prefix + 64K recompute | 2.449056 s | 2.576579 s | -127.523 ms |
| 128K prefix + 64K recompute | 3.389816 s | 3.494839 s | -105.023 ms |

At 128K+64K, Tutti no-prefetch was 3.482026 s and SSD-only was
3.940108 s, so the final version is respectively 92.210 ms and 550.292 ms
faster.

The immutable gpu002 deployment directory is:

```text
/home/zbuser02/lmcache_v026_20260806_run/versions/20260818_dsv4_qk_fusion_final_v1
```

Its `validation/` directory contains the complete per-request artifacts. The
rollback directory is `20260817_proven_glm_dsv4_cp8_v5`. No additional content
hashes were calculated; version identity is the frozen directory and Git
branch name.
