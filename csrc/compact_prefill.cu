// SPDX-License-Identifier: Apache-2.0

#include "compact_prefill.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_scan.cuh>
#include <torch/all.h>

namespace {

constexpr int kThreads = 256;

template <typename index_t>
__global__ void mark_selected_pages_kernel(
    const index_t* topk_indices, int64_t num_entries, int64_t topk_width,
    const int64_t* compressed_seq_lens, const int64_t* query_row_offsets,
    int64_t num_requests, int64_t max_pages, int64_t block_size,
    int32_t* page_seen) {
  const int64_t entry_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (entry_index >= num_entries) return;

  const int64_t row = entry_index / topk_width + query_row_offsets[0];
  int64_t request_index = -1;
  for (int64_t request = 0; request < num_requests; ++request) {
    if (row >= query_row_offsets[request] &&
        row < query_row_offsets[request + 1]) {
      request_index = request;
      break;
    }
  }
  if (request_index < 0) return;

  const int64_t entry = static_cast<int64_t>(topk_indices[entry_index]);
  const int64_t compressed_len = compressed_seq_lens[request_index];
  if (entry < 0 || entry >= compressed_len) return;
  const int64_t page = entry / block_size;
  if (page < max_pages) {
    atomicExch(page_seen + request_index * max_pages + page, 1);
  }
}

template <typename index_t>
__global__ void mark_missing_blocks_kernel(const index_t* topk_indices,
                                           int64_t num_entries,
                                           const bool* resident_blocks,
                                           int64_t max_blocks,
                                           int64_t block_size,
                                           int32_t* missing_blocks) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_entries) return;
  const int64_t entry = static_cast<int64_t>(topk_indices[index]);
  if (entry < 0) return;
  const int64_t block = entry / block_size;
  if (block < max_blocks && !resident_blocks[block]) {
    atomicExch(missing_blocks + block, 1);
  }
}

template <typename index_t>
__global__ void mark_selected_and_missing_blocks_kernel(
    const index_t* topk_indices, int64_t num_entries,
    const bool* resident_blocks, int64_t selected_max_blocks,
    int64_t missing_max_blocks, int64_t block_size, int32_t* selected_blocks,
    int32_t* missing_blocks) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_entries) return;
  const int64_t entry = static_cast<int64_t>(topk_indices[index]);
  if (entry < 0) return;
  const int64_t block = entry / block_size;
  if (block >= selected_max_blocks) return;
  atomicExch(selected_blocks + block, 1);
  // Only cached-prefix blocks can require NVMe correction. Top-K entries in
  // the active recompute suffix are already resident, but they must still be
  // represented in selected_blocks so compact gather does not drop them.
  if (block < missing_max_blocks && !resident_blocks[block]) {
    atomicExch(missing_blocks + block, 1);
  }
}

template <typename index_t>
__global__ void mark_selected_blocks_into_kernel(
    const index_t* topk_indices, int64_t num_entries,
    int64_t selected_max_blocks, int64_t block_size,
    int32_t* selected_blocks) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_entries) return;
  const int64_t entry = static_cast<int64_t>(topk_indices[index]);
  if (entry < 0) return;
  const int64_t block = entry / block_size;
  if (block >= selected_max_blocks) return;
  // Every writer stores the same aligned 32-bit value. No thread ever clears
  // the bitmap inside this kernel, so an atomic exchange only serializes hot
  // pages without changing the result.
  selected_blocks[block] = 1;
}

template <typename block_t, typename length_t>
__global__ void compact_selected_pages_kernel(
    const int32_t* page_seen, const block_t* block_table, int64_t max_pages,
    int64_t block_size, block_t* compact_block_table,
    length_t* compact_seq_lens, int32_t* page_to_compact) {
  using BlockScan = cub::BlockScan<int32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int32_t running_count;
  __shared__ int32_t tile_count;

  const int64_t request = blockIdx.x;
  if (threadIdx.x == 0) running_count = 0;
  __syncthreads();

  const int64_t request_offset = request * max_pages;
  for (int64_t tile = 0; tile < max_pages; tile += kThreads) {
    const int64_t page = tile + threadIdx.x;
    const int32_t selected =
        page < max_pages ? page_seen[request_offset + page] : 0;
    int32_t local_offset = 0;
    int32_t aggregate = 0;
    BlockScan(scan_storage).ExclusiveSum(selected, local_offset, aggregate);
    if (threadIdx.x == 0) tile_count = aggregate;
    __syncthreads();

    if (selected != 0) {
      const int32_t compact_page = running_count + local_offset;
      compact_block_table[request_offset + compact_page] =
          block_table[request_offset + page];
      page_to_compact[request_offset + page] = compact_page;
    }
    __syncthreads();
    if (threadIdx.x == 0) running_count += tile_count;
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    compact_seq_lens[request] =
        static_cast<length_t>(static_cast<int64_t>(running_count) * block_size);
  }
}

template <typename index_t>
__global__ void remap_topk_entries_kernel(
    const index_t* topk_indices, int64_t num_entries, int64_t topk_width,
    const int64_t* compressed_seq_lens, const int64_t* query_row_offsets,
    int64_t num_requests, int64_t max_pages, int64_t block_size,
    const int32_t* page_to_compact, index_t* remapped_topk) {
  const int64_t entry_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (entry_index >= num_entries) return;

  const int64_t row = entry_index / topk_width + query_row_offsets[0];
  int64_t request_index = -1;
  for (int64_t request = 0; request < num_requests; ++request) {
    if (row >= query_row_offsets[request] &&
        row < query_row_offsets[request + 1]) {
      request_index = request;
      break;
    }
  }
  if (request_index < 0) return;

  const int64_t entry = static_cast<int64_t>(topk_indices[entry_index]);
  const int64_t compressed_len = compressed_seq_lens[request_index];
  if (entry < 0 || entry >= compressed_len) return;
  const int64_t page = entry / block_size;
  if (page >= max_pages) return;
  const int32_t compact_page =
      page_to_compact[request_index * max_pages + page];
  if (compact_page < 0) return;
  remapped_topk[entry_index] = static_cast<index_t>(
      static_cast<int64_t>(compact_page) * block_size + entry % block_size);
}

void check_integral_metadata(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong,
      name, " must use int32 or int64");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<at::Tensor> build_compact_csa_prefill_gather_plan_cuda(
    at::Tensor topk_indices, at::Tensor block_table,
    at::Tensor compressed_seq_lens, at::Tensor query_row_offsets,
    int64_t block_size) {
  check_integral_metadata(topk_indices, "topk_indices");
  check_integral_metadata(block_table, "block_table");
  check_integral_metadata(compressed_seq_lens, "compressed_seq_lens");
  check_integral_metadata(query_row_offsets, "query_row_offsets");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be two-dimensional");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be two-dimensional");
  TORCH_CHECK(compressed_seq_lens.dim() == 1,
              "compressed_seq_lens must be one-dimensional");
  TORCH_CHECK(query_row_offsets.dim() == 1,
              "query_row_offsets must be one-dimensional");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  const int64_t num_requests = block_table.size(0);
  const int64_t max_pages = block_table.size(1);
  TORCH_CHECK(compressed_seq_lens.numel() == num_requests,
              "compressed_seq_lens must contain one value per request");
  TORCH_CHECK(query_row_offsets.numel() == num_requests + 1,
              "query_row_offsets must contain num_requests + 1 values");
  TORCH_CHECK(topk_indices.device() == block_table.device() &&
                  topk_indices.device() == compressed_seq_lens.device() &&
                  topk_indices.device() == query_row_offsets.device(),
              "all compact prefill tensors must use the same CUDA device");

  c10::cuda::CUDAGuard device_guard(topk_indices.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto int_options = topk_indices.options().dtype(at::kInt);
  auto page_seen = at::zeros({num_requests, max_pages}, int_options);
  auto page_to_compact = at::full({num_requests, max_pages}, -1, int_options);
  auto compact_block_table = at::zeros_like(block_table);
  auto compact_seq_lens = at::zeros_like(compressed_seq_lens);
  auto remapped_topk = at::full_like(topk_indices, -1);

  // These tensors contain at most PREFILL_CHUNK_SIZE + 1 values. Converting
  // them on-device avoids dtype-specialising the large top-K kernels and does
  // not introduce a host synchronisation.
  auto lengths64 = compressed_seq_lens.to(at::kLong);
  auto offsets64 = query_row_offsets.to(at::kLong);
  const int64_t num_entries = topk_indices.numel();
  const int64_t topk_width = topk_indices.size(1);

  if (num_entries > 0 && num_requests > 0 && max_pages > 0) {
    const int64_t blocks = (num_entries + kThreads - 1) / kThreads;
    AT_DISPATCH_INTEGRAL_TYPES(
        topk_indices.scalar_type(), "mark_compact_csa_pages", [&] {
          mark_selected_pages_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
              topk_indices.data_ptr<scalar_t>(), num_entries, topk_width,
              lengths64.data_ptr<int64_t>(), offsets64.data_ptr<int64_t>(),
              num_requests, max_pages, block_size,
              page_seen.data_ptr<int32_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (num_requests > 0) {
    AT_DISPATCH_INTEGRAL_TYPES(
        block_table.scalar_type(), "compact_csa_block_table", [&] {
          using block_t = scalar_t;
          AT_DISPATCH_INTEGRAL_TYPES(
              compressed_seq_lens.scalar_type(), "compact_csa_seq_lens", [&] {
                compact_selected_pages_kernel<block_t, scalar_t>
                    <<<num_requests, kThreads, 0, stream>>>(
                        page_seen.data_ptr<int32_t>(),
                        block_table.data_ptr<block_t>(), max_pages, block_size,
                        compact_block_table.data_ptr<block_t>(),
                        compact_seq_lens.data_ptr<scalar_t>(),
                        page_to_compact.data_ptr<int32_t>());
              });
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (num_entries > 0 && num_requests > 0 && max_pages > 0) {
    const int64_t blocks = (num_entries + kThreads - 1) / kThreads;
    AT_DISPATCH_INTEGRAL_TYPES(
        topk_indices.scalar_type(), "remap_compact_csa_topk", [&] {
          remap_topk_entries_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
              topk_indices.data_ptr<scalar_t>(), num_entries, topk_width,
              lengths64.data_ptr<int64_t>(), offsets64.data_ptr<int64_t>(),
              num_requests, max_pages, block_size,
              page_to_compact.data_ptr<int32_t>(),
              remapped_topk.data_ptr<scalar_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {compact_block_table, compact_seq_lens, remapped_topk};
}

std::vector<at::Tensor>
build_compact_csa_prefill_gather_plan_from_page_seen_cuda(
    at::Tensor topk_indices, at::Tensor block_table,
    at::Tensor compressed_seq_lens, at::Tensor query_row_offsets,
    int64_t block_size, at::Tensor page_seen, bool remap_topk) {
  check_integral_metadata(topk_indices, "topk_indices");
  check_integral_metadata(block_table, "block_table");
  check_integral_metadata(compressed_seq_lens, "compressed_seq_lens");
  check_integral_metadata(query_row_offsets, "query_row_offsets");
  TORCH_CHECK(page_seen.is_cuda(), "page_seen must be a CUDA tensor");
  TORCH_CHECK(page_seen.is_contiguous(), "page_seen must be contiguous");
  TORCH_CHECK(page_seen.scalar_type() == at::kInt,
              "page_seen must use int32 dtype");
  TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be two-dimensional");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be two-dimensional");
  TORCH_CHECK(compressed_seq_lens.dim() == 1,
              "compressed_seq_lens must be one-dimensional");
  TORCH_CHECK(query_row_offsets.dim() == 1,
              "query_row_offsets must be one-dimensional");
  TORCH_CHECK(page_seen.dim() == 2, "page_seen must be two-dimensional");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  const int64_t num_requests = block_table.size(0);
  const int64_t max_pages = block_table.size(1);
  TORCH_CHECK(compressed_seq_lens.numel() == num_requests,
              "compressed_seq_lens must contain one value per request");
  TORCH_CHECK(query_row_offsets.numel() == num_requests + 1,
              "query_row_offsets must contain num_requests + 1 values");
  TORCH_CHECK(
      page_seen.size(0) == num_requests && page_seen.size(1) == max_pages,
      "page_seen must match block_table shape");
  TORCH_CHECK(topk_indices.device() == block_table.device() &&
                  topk_indices.device() == compressed_seq_lens.device() &&
                  topk_indices.device() == query_row_offsets.device() &&
                  topk_indices.device() == page_seen.device(),
              "all compact prefill tensors must use the same CUDA device");

  c10::cuda::CUDAGuard device_guard(topk_indices.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto int_options = topk_indices.options().dtype(at::kInt);
  auto page_to_compact = at::full({num_requests, max_pages}, -1, int_options);
  auto compact_block_table = at::zeros_like(block_table);
  auto compact_seq_lens = at::zeros_like(compressed_seq_lens);
  auto remapped_topk = remap_topk
                            ? at::full_like(topk_indices, -1)
                            : at::empty({0}, topk_indices.options());
  auto lengths64 = compressed_seq_lens.to(at::kLong);
  auto offsets64 = query_row_offsets.to(at::kLong);
  const int64_t num_entries = topk_indices.numel();
  const int64_t topk_width = topk_indices.size(1);

  if (num_requests > 0) {
    AT_DISPATCH_INTEGRAL_TYPES(
        block_table.scalar_type(), "compact_csa_block_table_from_seen", [&] {
          using block_t = scalar_t;
          AT_DISPATCH_INTEGRAL_TYPES(
              compressed_seq_lens.scalar_type(),
              "compact_csa_seq_lens_from_seen", [&] {
                compact_selected_pages_kernel<block_t, scalar_t>
                    <<<num_requests, kThreads, 0, stream>>>(
                        page_seen.data_ptr<int32_t>(),
                        block_table.data_ptr<block_t>(), max_pages, block_size,
                        compact_block_table.data_ptr<block_t>(),
                        compact_seq_lens.data_ptr<scalar_t>(),
                        page_to_compact.data_ptr<int32_t>());
              });
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (remap_topk && num_entries > 0 && num_requests > 0 && max_pages > 0) {
    const int64_t blocks = (num_entries + kThreads - 1) / kThreads;
    AT_DISPATCH_INTEGRAL_TYPES(
        topk_indices.scalar_type(), "remap_compact_csa_topk_from_seen", [&] {
          remap_topk_entries_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
              topk_indices.data_ptr<scalar_t>(), num_entries, topk_width,
              lengths64.data_ptr<int64_t>(), offsets64.data_ptr<int64_t>(),
              num_requests, max_pages, block_size,
              page_to_compact.data_ptr<int32_t>(),
              remapped_topk.data_ptr<scalar_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {compact_block_table, compact_seq_lens, remapped_topk,
          page_to_compact};
}

at::Tensor select_missing_csa_blocks_cuda(at::Tensor topk_indices,
                                          at::Tensor resident_blocks,
                                          int64_t max_blocks,
                                          int64_t block_size) {
  check_integral_metadata(topk_indices, "topk_indices");
  TORCH_CHECK(resident_blocks.is_cuda(),
              "resident_blocks must be a CUDA tensor");
  TORCH_CHECK(resident_blocks.is_contiguous(),
              "resident_blocks must be contiguous");
  TORCH_CHECK(resident_blocks.scalar_type() == at::kBool,
              "resident_blocks must use bool dtype");
  TORCH_CHECK(topk_indices.device() == resident_blocks.device(),
              "topk_indices and resident_blocks must use one CUDA device");
  TORCH_CHECK(max_blocks >= 0 && max_blocks <= resident_blocks.numel(),
              "max_blocks must fit resident_blocks");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  c10::cuda::CUDAGuard device_guard(topk_indices.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto missing_blocks =
      at::zeros({max_blocks}, topk_indices.options().dtype(at::kInt));
  const int64_t num_entries = topk_indices.numel();
  if (num_entries > 0 && max_blocks > 0) {
    const int64_t blocks = (num_entries + kThreads - 1) / kThreads;
    AT_DISPATCH_INTEGRAL_TYPES(
        topk_indices.scalar_type(), "mark_missing_csa_blocks", [&] {
          mark_missing_blocks_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
              topk_indices.data_ptr<scalar_t>(), num_entries,
              resident_blocks.data_ptr<bool>(), max_blocks, block_size,
              missing_blocks.data_ptr<int32_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return at::nonzero(missing_blocks).reshape({-1});
}

std::vector<at::Tensor> select_missing_csa_blocks_with_seen_cuda(
    at::Tensor topk_indices, at::Tensor resident_blocks, int64_t max_blocks,
    int64_t selected_max_blocks, int64_t block_size) {
  check_integral_metadata(topk_indices, "topk_indices");
  TORCH_CHECK(resident_blocks.is_cuda(),
              "resident_blocks must be a CUDA tensor");
  TORCH_CHECK(resident_blocks.is_contiguous(),
              "resident_blocks must be contiguous");
  TORCH_CHECK(resident_blocks.scalar_type() == at::kBool,
              "resident_blocks must use bool dtype");
  TORCH_CHECK(topk_indices.device() == resident_blocks.device(),
              "topk_indices and resident_blocks must use one CUDA device");
  TORCH_CHECK(max_blocks >= 0 && max_blocks <= resident_blocks.numel(),
              "max_blocks must fit resident_blocks");
  TORCH_CHECK(selected_max_blocks >= max_blocks &&
                  selected_max_blocks <= resident_blocks.numel(),
              "selected_max_blocks must cover max_blocks and fit "
              "resident_blocks");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  c10::cuda::CUDAGuard device_guard(topk_indices.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto int_options = topk_indices.options().dtype(at::kInt);
  auto selected_blocks = at::zeros({resident_blocks.numel()}, int_options);
  auto missing_blocks = at::zeros({max_blocks}, int_options);
  const int64_t num_entries = topk_indices.numel();
  if (num_entries > 0 && max_blocks > 0) {
    const int64_t blocks = (num_entries + kThreads - 1) / kThreads;
    AT_DISPATCH_INTEGRAL_TYPES(
        topk_indices.scalar_type(), "mark_selected_and_missing_csa_blocks",
        [&] {
          mark_selected_and_missing_blocks_kernel<scalar_t>
              <<<blocks, kThreads, 0, stream>>>(
                  topk_indices.data_ptr<scalar_t>(), num_entries,
                  resident_blocks.data_ptr<bool>(), selected_max_blocks,
                  max_blocks, block_size, selected_blocks.data_ptr<int32_t>(),
                  missing_blocks.data_ptr<int32_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {at::nonzero(missing_blocks).reshape({-1}), selected_blocks};
}

void mark_csa_selected_blocks_into_cuda(at::Tensor topk_indices,
                                        at::Tensor selected_blocks,
                                        int64_t selected_max_blocks,
                                        int64_t block_size) {
  check_integral_metadata(topk_indices, "topk_indices");
  TORCH_CHECK(selected_blocks.is_cuda(),
              "selected_blocks must be a CUDA tensor");
  TORCH_CHECK(selected_blocks.is_contiguous(),
              "selected_blocks must be contiguous");
  TORCH_CHECK(selected_blocks.scalar_type() == at::kInt,
              "selected_blocks must use int32 dtype");
  TORCH_CHECK(topk_indices.device() == selected_blocks.device(),
              "topk_indices and selected_blocks must use one CUDA device");
  TORCH_CHECK(selected_max_blocks >= 0 &&
                  selected_max_blocks <= selected_blocks.numel(),
              "selected_max_blocks must fit selected_blocks");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  c10::cuda::CUDAGuard device_guard(topk_indices.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_entries = topk_indices.numel();
  if (num_entries > 0 && selected_max_blocks > 0) {
    const int64_t blocks = (num_entries + kThreads - 1) / kThreads;
    AT_DISPATCH_INTEGRAL_TYPES(
        topk_indices.scalar_type(), "mark_csa_selected_blocks_into", [&] {
          mark_selected_blocks_into_kernel<scalar_t>
              <<<blocks, kThreads, 0, stream>>>(
                  topk_indices.data_ptr<scalar_t>(), num_entries,
                  selected_max_blocks, block_size,
                  selected_blocks.data_ptr<int32_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
