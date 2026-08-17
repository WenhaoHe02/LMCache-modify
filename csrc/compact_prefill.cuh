// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

#include <vector>

// Build a compact per-request page table and remap CSA top-K entries entirely
// on the current CUDA stream. No tensor value is copied to the host.
std::vector<at::Tensor> build_compact_csa_prefill_gather_plan_cuda(
    at::Tensor topk_indices, at::Tensor block_table,
    at::Tensor compressed_seq_lens, at::Tensor query_row_offsets,
    int64_t block_size);

// Build the same compact gather plan from a precomputed one-request logical
// page bitmap. This avoids rescanning millions of top-K entries after the
// correction path has already produced their exact union.
std::vector<at::Tensor>
build_compact_csa_prefill_gather_plan_from_page_seen_cuda(
    at::Tensor topk_indices, at::Tensor block_table,
    at::Tensor compressed_seq_lens, at::Tensor query_row_offsets,
    int64_t block_size, at::Tensor page_seen, bool remap_topk = true);

// Return sorted unique logical blocks selected by top-K but absent from the
// resident bitmap. The 4M-entry prefill union is scanned by one CUDA kernel;
// only the small block mask is compacted afterwards.
at::Tensor select_missing_csa_blocks_cuda(at::Tensor topk_indices,
                                          at::Tensor resident_blocks,
                                          int64_t max_blocks,
                                          int64_t block_size);

// Return both the missing block ids and the exact selected-page bitmap. The
// latter can be reused by compact prefill gather planning.
std::vector<at::Tensor> select_missing_csa_blocks_with_seen_cuda(
    at::Tensor topk_indices, at::Tensor resident_blocks, int64_t max_blocks,
    int64_t selected_max_blocks, int64_t block_size);

// Accumulate the logical-page union of one completed true-topK chunk into an
// existing int32 bitmap. The caller owns bitmap zeroing and stream ordering.
void mark_csa_selected_blocks_into_cuda(at::Tensor topk_indices,
                                        at::Tensor selected_blocks,
                                        int64_t selected_max_blocks,
                                        int64_t block_size);
