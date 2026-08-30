# Exact Indexer-K sharding v56

The v56 path partitions the cached sequence/K dimension into eight contiguous
slices. Each TP rank scores the full append query against its own `P/8` keys,
keeps exactly `k/8` candidates, reads only the corresponding rank-local KV
rows, and reconstructs the attention-visible KV set with a gate-aligned
AllGather. It never merges candidate IDs or computes an exact distributed
top-k.

The production configuration uses full query scoring
(`LMCACHE_CSA_PREFETCH_CP_QUERY_SAMPLE_STRIDE=1`) and disables the independent
HCA walker. The latter was not part of Indexer-K sharding and added roughly
70 ms at 98K history + 256 append by competing with CSA reads and collectives.

On DSv4-Flash TP8, three formal repetitions at 98,304 history + 256 append
measured 0.380405, 0.386014, and 0.382245 seconds. The same-container Tutti
no-prefetch control measured 0.385951, 0.386774, and 0.389326 seconds. At
131,072 history + 2,048 append, the two paths were effectively tied (within
3.6 ms).

Use `scripts/run_dsv4_exact_key_sharded_v56.sh` to apply the fixed environment
to an existing DSv4 launcher.
