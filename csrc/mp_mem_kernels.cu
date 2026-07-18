// SPDX-License-Identifier: Apache-2.0

#include "mp_mem_kernels.cuh"

#include <cstdint>

namespace {

/**
 * Key logic in the kernel implementation:
 * 1. Each thread block is for (BS, NH, HS) part (i.e., a single block in the
 * paged buffer)
 * 2. Within a thread block, each warp is for a single head. Number of warps
 * in a thread block is equal to the number of heads (NH).
 * 3. Within a thread block, we do the loop over the BS (i.e., number of tokens
 * in the block) dimension.
 * 4. The grid will take over (2, NB, NL) dimensions. No matter what the actual
 * layout in memory is, we will calculate the global offset for the start of the
 * block
 * 5. For LMCache, we assume it is always using 2LTD layout, e.g.,
 * [2, L, 256, NH * HS], where 256 means that 256 tokens
 */

/**
 * Calculate the offset for the current block in the paged buffer
 */
template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_engine_global_offset(
    const int k_or_v, const int engine_block_idx, const int layer_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_block = shape_desc.scalars_per_block<ScalarType>();
  if constexpr (format == GPUKVFormat::NB_NL_TWO_BS_NH_HS) {
    // Cross-layer: single tensor [NB, NL, 2, BS, NH, HS]
    return k_or_v * scalars_per_block +
           layer_idx * shape_desc.kv_size * scalars_per_block +
           engine_block_idx * shape_desc.kv_size * scalars_per_block *
               shape_desc.nl;
  } else if constexpr (format == GPUKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    // Normal: L tensors [2, NB, BS, NH, HS]
    return engine_block_idx * scalars_per_block +
           k_or_v * shape_desc.nb * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    // Flash Infer: L tensors [NB, 2, BS, NH, HS]
    return engine_block_idx * shape_desc.kv_size * scalars_per_block +
           k_or_v * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::NL_X_NB_BS_HS) {
    // MLA: L tensors [NB, BS, HS]
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS) {
    // SGLang MHA: 2L tensors [NBBS, NH, HS] — K/V via separate tensor ptrs
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::NL_X_NBBS_ONE_HS) {
    // SGLang MLA: L tensors [NBBS, 1, HS]
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::NB_NL_TWO_NH_BS_HS) {
    // TRT-LLM cross-layer HND: single tensor [NB, NL, 2, NH, BS, HS]
    // same block-level strides as NB_NL_TWO_BS_NH_HS
    return k_or_v * scalars_per_block +
           layer_idx * shape_desc.kv_size * scalars_per_block +
           engine_block_idx * shape_desc.kv_size * scalars_per_block *
               shape_desc.nl;
  }
}

/**
 * Calculate the offset for the current token against the start
 * of the block in the paged buffer.
 */
template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_engine_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  if constexpr (format == GPUKVFormat::NB_NL_TWO_NH_BS_HS) {
    // HND: [NH, BS, HS] — heads are outermost within a block
    size_t scalars_per_head_block =
        shape_desc.bs * scalars_per_head;  // BS * HS
    return head_idx * scalars_per_head_block + token_offset * scalars_per_head;
  } else {
    // NHD: [BS, NH, HS] — tokens are outermost within a block
    return head_idx * scalars_per_head + token_offset * scalars_per_token;
  }
}

/**
 * Calculate the global offset for the current `block` in the LMCache object.
 * The `block` here is the memory region corresponding to a thread-block.
 */
template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_lmcache_global_offset(
    const int k_or_v,
    const int
        token_offset_in_lmcache_object,  // 0~255 if LMCache chunk size is 256
    const int layer_idx,
    const int lmcache_chunk_size,  // e.g., 256
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  // LMCache is using 2LTD all the times
  return token_offset_in_lmcache_object * scalars_per_token +
         layer_idx * lmcache_chunk_size * scalars_per_token +
         k_or_v * shape_desc.nl * lmcache_chunk_size * scalars_per_token;
}

/**
 * Calculate the local offset for the current token against the start of the
 * block in the LMCache object.
 */
template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_lmcache_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  return head_idx * scalars_per_head + token_offset * scalars_per_token;
}

__device__ inline uint4 ld_cs(const uint4* addr) {
#ifdef __CUDA_ARCH__
  uint4 val;
  asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  return val;
#else
  return *addr;
#endif
}

__device__ inline void st_cs(uint4* addr, uint4 val) {
#ifdef __CUDA_ARCH__
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else
  *addr = val;
#endif
}

template <typename ScalarType>
__device__ inline void warp_copy(ScalarType* __restrict__ dst,
                                 const ScalarType* __restrict__ src,
                                 size_t num_elements) {
  int idx = threadIdx.x;
  int stride = blockDim.x;
  if constexpr (std::is_same_v<ScalarType, uint4>) {
    for (size_t i = idx; i < num_elements; i += stride) {
      st_cs(dst + i, ld_cs(src + i));
    }
  } else {
    for (size_t i = idx; i < num_elements; i += stride) {
      dst[i] = src[i];
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine, GPUKVFormat format>
__device__ void multi_layer_block_transfer_single_block(
    ScalarType* __restrict__ lmcache_object,
    ScalarType** __restrict__ paged_buffer_ptrs, const int engine_block_idx,
    const int offset_in_lmcache_block, const PageBufferShapeDesc shape_desc,
    const int lmcache_chunk_size  // e.g., 256, used to calculate global offset
                                  // in LMCache object
) {
  const int head_idx = threadIdx.y;
  const int k_or_v = blockIdx.x;
  const int layer_idx = blockIdx.z;

  const size_t engine_global_offset =
      calculate_engine_global_offset<ScalarType, format>(
          k_or_v, engine_block_idx, layer_idx, shape_desc);
  const size_t lmcache_global_offset =
      calculate_lmcache_global_offset<ScalarType, format>(
          k_or_v, offset_in_lmcache_block, layer_idx, lmcache_chunk_size,
          shape_desc);
  ScalarType* paged_buffer_layer_ptr;
  if constexpr (format == GPUKVFormat::NB_NL_TWO_BS_NH_HS ||
                format == GPUKVFormat::NB_NL_TWO_NH_BS_HS) {
    paged_buffer_layer_ptr = (ScalarType*)paged_buffer_ptrs[0];
  } else if constexpr (format == GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS) {
    // SGLang MHA: ptrs[0..NL-1] = K per layer, ptrs[NL..2NL-1] = V per layer
    paged_buffer_layer_ptr =
        (ScalarType*)paged_buffer_ptrs[k_or_v * shape_desc.nl + layer_idx];
  } else {
    paged_buffer_layer_ptr = (ScalarType*)paged_buffer_ptrs[layer_idx];
  }

  for (int token_offset = 0; token_offset < shape_desc.bs; ++token_offset) {
    const size_t engine_local_offset =
        calculate_engine_local_offset<ScalarType, format>(token_offset,
                                                          head_idx, shape_desc);
    const size_t lmcache_local_offset =
        calculate_lmcache_local_offset<ScalarType, format>(
            token_offset, head_idx, shape_desc);
    ScalarType* engine_ptr =
        paged_buffer_layer_ptr + engine_global_offset + engine_local_offset;
    ScalarType* lmcache_ptr =
        lmcache_object + lmcache_global_offset + lmcache_local_offset;
    if constexpr (lmcache_to_engine) {
      warp_copy<ScalarType>(engine_ptr, lmcache_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    } else {
      warp_copy<ScalarType>(lmcache_ptr, engine_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine, GPUKVFormat format>
__global__ void multi_layer_block_transfer_kernel(
    MemoryObj4<ScalarType> lmcache_objects,
    ScalarType** __restrict__ paged_buffer_ptrs,
    const int64_t* engine_block_ids,
    const int num_blocks_per_object,  // e.g. 16 for lmcache chunk size =
                                      // 256 and block size = 16
    const PageBufferShapeDesc shape_desc,
    const int lmcache_chunk_size,  // e.g., 256, used to calculate global offset
                                   // in LMCache object
    const int skip_prefix_n_blocks) {
  // blockIdx.y spans all blocks across all objects (total_blocks).
  // Derive which object and local block index from the flat index.
  const int flat_block_idx = blockIdx.y;
  if (flat_block_idx < skip_prefix_n_blocks) {
    return;
  }
  const int obj_idx = flat_block_idx / num_blocks_per_object;
  const int block_idx_in_object = flat_block_idx % num_blocks_per_object;

  const int engine_block_idx = engine_block_ids[flat_block_idx];
  multi_layer_block_transfer_single_block<ScalarType, lmcache_to_engine,
                                          format>(
      lmcache_objects.objects[obj_idx], paged_buffer_ptrs, engine_block_idx,
      block_idx_in_object * shape_desc.bs,  // offset in LMCache object
      shape_desc, lmcache_chunk_size);
}

template <typename ScalarType, bool lmcache_to_engine, GPUKVFormat format>
__global__ void multi_layer_block_transfer_batched_kernel(
    const int64_t* __restrict__ lmcache_object_ptrs,
    ScalarType** __restrict__ paged_buffer_ptrs,
    const int64_t* engine_block_ids, const int num_blocks_per_object,
    const PageBufferShapeDesc shape_desc, const int lmcache_chunk_size,
    const int skip_prefix_n_blocks) {
  const int flat_block_idx = blockIdx.y;
  if (flat_block_idx < skip_prefix_n_blocks) {
    return;
  }
  const int obj_idx = flat_block_idx / num_blocks_per_object;
  const int block_idx_in_object = flat_block_idx % num_blocks_per_object;
  const int engine_block_idx = engine_block_ids[flat_block_idx];
  auto* lmcache_object =
      reinterpret_cast<ScalarType*>(lmcache_object_ptrs[obj_idx]);
  multi_layer_block_transfer_single_block<ScalarType, lmcache_to_engine,
                                          format>(
      lmcache_object, paged_buffer_ptrs, engine_block_idx,
      block_idx_in_object * shape_desc.bs, shape_desc, lmcache_chunk_size);
}

template <typename ScalarType>
__global__ void scatter_rows_from_object_ptrs_kernel(
    const int64_t* __restrict__ source_ptrs, uint8_t* __restrict__ destination,
    const int64_t* __restrict__ destination_rows, int64_t total_rows,
    int rows_per_object, int scalars_per_row, int row_bytes,
    int64_t destination_rows_limit, int64_t destination_stride0,
    int64_t destination_stride1, int logical_slots_per_block) {
  const int64_t scalar_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_scalars = total_rows * scalars_per_row;
  if (scalar_idx >= total_scalars) {
    return;
  }

  const int64_t row_idx = scalar_idx / scalars_per_row;
  const int scalar_in_row = scalar_idx % scalars_per_row;
  const int64_t object_idx = row_idx / rows_per_object;
  const int row_in_object = row_idx % rows_per_object;
  const int64_t destination_row = destination_rows[row_idx];
  if (destination_row < 0 || destination_row >= destination_rows_limit) {
    return;
  }

  const auto* source = reinterpret_cast<const uint8_t*>(
      static_cast<uintptr_t>(source_ptrs[object_idx]));
  const auto* source_row =
      reinterpret_cast<const ScalarType*>(source + row_in_object * row_bytes);

  int64_t destination_offset;
  if (logical_slots_per_block > 0) {
    const int64_t block_idx = destination_row / logical_slots_per_block;
    const int64_t slot_idx = destination_row % logical_slots_per_block;
    destination_offset =
        block_idx * destination_stride0 + slot_idx * destination_stride1;
  } else {
    destination_offset = destination_row * destination_stride0;
  }
  auto* destination_row_ptr =
      reinterpret_cast<ScalarType*>(destination + destination_offset);
  destination_row_ptr[scalar_in_row] = source_row[scalar_in_row];
}

#define LAUNCH_KERNEL(DIRECTION, FORMAT)                                 \
  multi_layer_block_transfer_kernel<ScalarType, DIRECTION, FORMAT>       \
      <<<grid, block, 0, stream>>>(lmcache_obj4, paged_buffer_ptrs,      \
                                   block_ids_ptr, num_blocks_per_object, \
                                   shape_desc, lmcache_chunk_size,       \
                                   skip_prefix_n_blocks);                \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define DISPATCH_FORMAT(DIRECTION)                                  \
  switch (gpu_kv_format) {                                          \
    case GPUKVFormat::NB_NL_TWO_BS_NH_HS:                           \
      LAUNCH_KERNEL(DIRECTION, GPUKVFormat::NB_NL_TWO_BS_NH_HS);    \
      break;                                                        \
    case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:                         \
      LAUNCH_KERNEL(DIRECTION, GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);  \
      break;                                                        \
    case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:                         \
      LAUNCH_KERNEL(DIRECTION, GPUKVFormat::NL_X_NB_TWO_BS_NH_HS);  \
      break;                                                        \
    case GPUKVFormat::NL_X_NB_BS_HS:                                \
      LAUNCH_KERNEL(DIRECTION, GPUKVFormat::NL_X_NB_BS_HS);         \
      break;                                                        \
    case GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS:                        \
      LAUNCH_KERNEL(DIRECTION, GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS); \
      break;                                                        \
    case GPUKVFormat::NL_X_NBBS_ONE_HS:                             \
      LAUNCH_KERNEL(DIRECTION, GPUKVFormat::NL_X_NBBS_ONE_HS);      \
      break;                                                        \
    case GPUKVFormat::NB_NL_TWO_NH_BS_HS:                           \
      LAUNCH_KERNEL(DIRECTION, GPUKVFormat::NB_NL_TWO_NH_BS_HS);    \
      break;                                                        \
    default:                                                        \
      TORCH_CHECK(false, "Unsupported GPUKVFormat: ",               \
                  static_cast<int>(gpu_kv_format));                 \
  }

template <typename ScalarType>
void multi_layer_block_kv_transfer_templated(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    GPUKVFormat gpu_kv_format, int skip_prefix_n_blocks) {
  // --- Validation ---
  int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1-4 LMCache objects, got ", num_objects);

  int total_blocks = block_ids.size(0);
  TORCH_CHECK(total_blocks % num_objects == 0, "block_ids length (",
              total_blocks, ") must be divisible by num_objects (", num_objects,
              ")");
  int num_blocks_per_object = total_blocks / num_objects;

  TORCH_CHECK(num_blocks_per_object * shape_desc.bs == lmcache_chunk_size,
              "blocks_per_object * block_size (",
              num_blocks_per_object * shape_desc.bs,
              ") must equal lmcache_chunk_size (", lmcache_chunk_size, ")");

  // --- Build MemoryObj4 ---
  MemoryObj4<ScalarType> lmcache_obj4;
  lmcache_obj4.num_objects = num_objects;
  for (int i = 0; i < 4; ++i) {
    lmcache_obj4.objects[i] =
        (i < num_objects)
            ? reinterpret_cast<ScalarType*>(lmcache_objects_ptrs[i])
            : nullptr;
  }

  // --- Build paged buffer pointer array ---
  ScalarType** paged_buffer_ptrs =
      reinterpret_cast<ScalarType**>(paged_buffer_ptrs_tensor.data_ptr());

  const at::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // --- block_ids is a GPU int64 tensor, read directly ---
  const int64_t* block_ids_ptr = block_ids.data_ptr<int64_t>();

  // --- Grid and block dimensions ---
  int elements_per_head = shape_desc.hs * shape_desc.element_size /
                          static_cast<int>(sizeof(ScalarType));
  int thread_dim_x = std::min(elements_per_head, 32);
  int thread_dim_y = shape_desc.nh;

  dim3 block(thread_dim_x, thread_dim_y);
  dim3 grid(shape_desc.kv_size, total_blocks, shape_desc.nl);

  if (direction == TransferDirection::H2D) {
    DISPATCH_FORMAT(true);
  } else {
    DISPATCH_FORMAT(false);
  }
}

#define LAUNCH_BATCHED_KERNEL(DIRECTION, FORMAT)                           \
  multi_layer_block_transfer_batched_kernel<ScalarType, DIRECTION, FORMAT> \
      <<<grid, block, 0, stream>>>(lmcache_object_ptrs, paged_buffer_ptrs, \
                                   block_ids_ptr, num_blocks_per_object,   \
                                   shape_desc, lmcache_chunk_size,         \
                                   skip_prefix_n_blocks);                  \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define DISPATCH_BATCHED_FORMAT(DIRECTION)                                  \
  switch (gpu_kv_format) {                                                  \
    case GPUKVFormat::NB_NL_TWO_BS_NH_HS:                                   \
      LAUNCH_BATCHED_KERNEL(DIRECTION, GPUKVFormat::NB_NL_TWO_BS_NH_HS);    \
      break;                                                                \
    case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:                                 \
      LAUNCH_BATCHED_KERNEL(DIRECTION, GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);  \
      break;                                                                \
    case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:                                 \
      LAUNCH_BATCHED_KERNEL(DIRECTION, GPUKVFormat::NL_X_NB_TWO_BS_NH_HS);  \
      break;                                                                \
    case GPUKVFormat::NL_X_NB_BS_HS:                                        \
      LAUNCH_BATCHED_KERNEL(DIRECTION, GPUKVFormat::NL_X_NB_BS_HS);         \
      break;                                                                \
    case GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS:                                \
      LAUNCH_BATCHED_KERNEL(DIRECTION, GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS); \
      break;                                                                \
    case GPUKVFormat::NL_X_NBBS_ONE_HS:                                     \
      LAUNCH_BATCHED_KERNEL(DIRECTION, GPUKVFormat::NL_X_NBBS_ONE_HS);      \
      break;                                                                \
    case GPUKVFormat::NB_NL_TWO_NH_BS_HS:                                   \
      LAUNCH_BATCHED_KERNEL(DIRECTION, GPUKVFormat::NB_NL_TWO_NH_BS_HS);    \
      break;                                                                \
    default:                                                                \
      TORCH_CHECK(false, "Unsupported GPUKVFormat: ",                       \
                  static_cast<int>(gpu_kv_format));                         \
  }

template <typename ScalarType>
void multi_layer_block_kv_transfer_batched_templated(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    const torch::Tensor& lmcache_objects_ptrs_tensor,
    const torch::Tensor& block_ids, const torch::Device& device,
    TransferDirection direction, PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size, GPUKVFormat gpu_kv_format,
    int skip_prefix_n_blocks) {
  TORCH_CHECK(paged_buffer_ptrs_tensor.is_cuda(),
              "paged_buffer_ptrs_tensor must be a CUDA tensor");
  TORCH_CHECK(paged_buffer_ptrs_tensor.scalar_type() == torch::kInt64,
              "paged_buffer_ptrs_tensor must be int64");
  TORCH_CHECK(paged_buffer_ptrs_tensor.is_contiguous(),
              "paged_buffer_ptrs_tensor must be contiguous");
  TORCH_CHECK(lmcache_objects_ptrs_tensor.is_cuda(),
              "lmcache_objects_ptrs_tensor must be a CUDA tensor");
  TORCH_CHECK(lmcache_objects_ptrs_tensor.scalar_type() == torch::kInt64,
              "lmcache_objects_ptrs_tensor must be int64");
  TORCH_CHECK(lmcache_objects_ptrs_tensor.is_contiguous(),
              "lmcache_objects_ptrs_tensor must be contiguous");
  TORCH_CHECK(block_ids.is_cuda(), "block_ids must be a CUDA tensor");
  TORCH_CHECK(block_ids.scalar_type() == torch::kInt64,
              "block_ids must be int64");
  TORCH_CHECK(block_ids.is_contiguous(), "block_ids must be contiguous");
  TORCH_CHECK(paged_buffer_ptrs_tensor.device() == block_ids.device() &&
                  lmcache_objects_ptrs_tensor.device() == block_ids.device(),
              "pointer tensors and block_ids must be on the same device");

  const int num_objects = static_cast<int>(lmcache_objects_ptrs_tensor.numel());
  TORCH_CHECK(num_objects >= 1, "Expected at least one LMCache object");
  const int total_blocks = static_cast<int>(block_ids.numel());
  TORCH_CHECK(total_blocks > 0, "block_ids must not be empty");
  TORCH_CHECK(total_blocks <= 65535,
              "block_ids length exceeds CUDA grid y limit: ", total_blocks);
  TORCH_CHECK(total_blocks % num_objects == 0, "block_ids length (",
              total_blocks, ") must be divisible by num_objects (", num_objects,
              ")");
  const int num_blocks_per_object = total_blocks / num_objects;
  TORCH_CHECK(num_blocks_per_object * shape_desc.bs == lmcache_chunk_size,
              "blocks_per_object * block_size (",
              num_blocks_per_object * shape_desc.bs,
              ") must equal lmcache_chunk_size (", lmcache_chunk_size, ")");
  TORCH_CHECK(skip_prefix_n_blocks >= 0 && skip_prefix_n_blocks <= total_blocks,
              "skip_prefix_n_blocks must be in [0, ", total_blocks, "]");

  const at::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto** paged_buffer_ptrs = reinterpret_cast<ScalarType**>(
      paged_buffer_ptrs_tensor.data_ptr<int64_t>());
  const int64_t* lmcache_object_ptrs =
      lmcache_objects_ptrs_tensor.data_ptr<int64_t>();
  const int64_t* block_ids_ptr = block_ids.data_ptr<int64_t>();

  const int elements_per_head = shape_desc.hs * shape_desc.element_size /
                                static_cast<int>(sizeof(ScalarType));
  const int thread_dim_x = std::min(elements_per_head, 32);
  const int thread_dim_y = shape_desc.nh;
  const dim3 block(thread_dim_x, thread_dim_y);
  const dim3 grid(shape_desc.kv_size, total_blocks, shape_desc.nl);

  if (direction == TransferDirection::H2D) {
    DISPATCH_BATCHED_FORMAT(true);
  } else {
    DISPATCH_BATCHED_FORMAT(false);
  }
}

#undef DISPATCH_BATCHED_FORMAT
#undef LAUNCH_BATCHED_KERNEL

#undef DISPATCH_FORMAT
#undef LAUNCH_KERNEL

}  // namespace

#define LAUNCH_TEMPLATED(type)                                             \
  do {                                                                     \
    multi_layer_block_kv_transfer_templated<type>(                         \
        paged_buffer_ptrs_tensor, lmcache_objects_ptrs, block_ids, device, \
        direction, shape_desc, lmcache_chunk_size, gpu_kv_format,          \
        skip_prefix_n_blocks);                                             \
  } while (0)

#define LAUNCH_BATCHED_TEMPLATED(type)                                    \
  do {                                                                    \
    multi_layer_block_kv_transfer_batched_templated<type>(                \
        paged_buffer_ptrs_tensor, lmcache_objects_ptrs_tensor, block_ids, \
        device, direction, shape_desc, lmcache_chunk_size, gpu_kv_format, \
        skip_prefix_n_blocks);                                            \
  } while (0)

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    GPUKVFormat gpu_kv_format, int skip_prefix_n_blocks) {
  int head_bytes = shape_desc.hs * shape_desc.element_size;
  TORCH_CHECK(head_bytes % sizeof(uint16_t) == 0, "head_size * element_size (",
              head_bytes, ") must be divisible by 2 for vectorized access");

  if (head_bytes % sizeof(uint4) == 0) {
    LAUNCH_TEMPLATED(uint4);  // 16 bytes per copy
  } else if (head_bytes % sizeof(uint32_t) == 0) {
    LAUNCH_TEMPLATED(uint32_t);  // 4 bytes per copy
  } else {
    LAUNCH_TEMPLATED(uint16_t);  // 2 bytes per copy (minimum granularity)
  }
}

void multi_layer_block_kv_transfer_batched(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    const torch::Tensor& lmcache_objects_ptrs_tensor,
    const torch::Tensor& block_ids, const torch::Device& device,
    TransferDirection direction, PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size, GPUKVFormat gpu_kv_format,
    int skip_prefix_n_blocks) {
  const int head_bytes = shape_desc.hs * shape_desc.element_size;
  TORCH_CHECK(head_bytes % sizeof(uint16_t) == 0, "head_size * element_size (",
              head_bytes, ") must be divisible by 2 for vectorized access");

  if (head_bytes % sizeof(uint4) == 0) {
    LAUNCH_BATCHED_TEMPLATED(uint4);
  } else if (head_bytes % sizeof(uint32_t) == 0) {
    LAUNCH_BATCHED_TEMPLATED(uint32_t);
  } else {
    LAUNCH_BATCHED_TEMPLATED(uint16_t);
  }
}

void scatter_rows_from_object_ptrs(const torch::Tensor& source_ptrs,
                                   const torch::Tensor& destination,
                                   const torch::Tensor& destination_rows,
                                   int rows_per_object, int row_bytes,
                                   int logical_slots_per_block,
                                   bool source_ptrs_aligned) {
  TORCH_CHECK(source_ptrs.is_cuda(), "source_ptrs must be a CUDA tensor");
  TORCH_CHECK(source_ptrs.scalar_type() == torch::kInt64,
              "source_ptrs must be int64");
  TORCH_CHECK(source_ptrs.is_contiguous(), "source_ptrs must be contiguous");
  TORCH_CHECK(destination.is_cuda(), "destination must be a CUDA tensor");
  TORCH_CHECK(destination.scalar_type() == torch::kUInt8,
              "destination must be uint8");
  TORCH_CHECK(destination_rows.is_cuda(),
              "destination_rows must be a CUDA tensor");
  TORCH_CHECK(destination_rows.scalar_type() == torch::kInt64,
              "destination_rows must be int64");
  TORCH_CHECK(destination_rows.is_contiguous(),
              "destination_rows must be contiguous");
  TORCH_CHECK(source_ptrs.device() == destination.device() &&
                  source_ptrs.device() == destination_rows.device(),
              "source_ptrs, destination, and destination_rows must share a "
              "CUDA device");
  TORCH_CHECK(source_ptrs.dim() == 1, "source_ptrs must be one-dimensional");
  TORCH_CHECK(destination_rows.dim() == 1,
              "destination_rows must be one-dimensional");
  TORCH_CHECK(rows_per_object > 0, "rows_per_object must be positive");
  TORCH_CHECK(row_bytes > 0, "row_bytes must be positive");
  TORCH_CHECK(logical_slots_per_block >= 0,
              "logical_slots_per_block must be non-negative");
  TORCH_CHECK(destination.dim() >= 2,
              "destination must have at least two dimensions");
  TORCH_CHECK(destination.stride(destination.dim() - 1) == 1,
              "destination innermost dimension must be contiguous");

  const int64_t num_objects = source_ptrs.numel();
  const int64_t total_rows = destination_rows.numel();
  TORCH_CHECK(num_objects > 0, "source_ptrs must not be empty");
  TORCH_CHECK(total_rows == num_objects * rows_per_object,
              "destination_rows length (", total_rows,
              ") must equal source_ptrs length * rows_per_object (",
              num_objects * rows_per_object, ")");

  int64_t destination_rows_limit = destination.size(0);
  int64_t destination_stride1 = 0;
  if (logical_slots_per_block > 0) {
    TORCH_CHECK(destination.dim() >= 3,
                "slot-addressed destination must have at least 3 dimensions");
    TORCH_CHECK(destination.size(1) >= logical_slots_per_block,
                "destination slot dimension is smaller than "
                "logical_slots_per_block");
    destination_rows_limit =
        destination.size(0) * static_cast<int64_t>(logical_slots_per_block);
    destination_stride1 = destination.stride(1);
  }
  TORCH_CHECK(destination.size(destination.dim() - 1) >= row_bytes,
              "destination row width is smaller than row_bytes");

  const at::cuda::OptionalCUDAGuard device_guard(destination.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  constexpr int vector_bytes = sizeof(uint64_t);
  const auto destination_ptr =
      reinterpret_cast<uintptr_t>(destination.data_ptr<uint8_t>());
  const bool destination_aligned =
      destination_ptr % vector_bytes == 0 &&
      destination.stride(0) % vector_bytes == 0 &&
      (logical_slots_per_block == 0 || destination_stride1 % vector_bytes == 0);
  const int64_t scalar_size = source_ptrs_aligned && destination_aligned &&
                                      row_bytes % vector_bytes == 0
                                  ? static_cast<int64_t>(vector_bytes)
                                  : static_cast<int64_t>(sizeof(uint8_t));
  const int scalars_per_row = row_bytes / scalar_size;
  const int64_t total_scalars = total_rows * scalars_per_row;
  const int blocks = static_cast<int>((total_scalars + threads - 1) / threads);

  if (scalar_size == static_cast<int64_t>(sizeof(uint64_t))) {
    scatter_rows_from_object_ptrs_kernel<uint64_t>
        <<<blocks, threads, 0, stream>>>(
            source_ptrs.data_ptr<int64_t>(), destination.data_ptr<uint8_t>(),
            destination_rows.data_ptr<int64_t>(), total_rows, rows_per_object,
            scalars_per_row, row_bytes, destination_rows_limit,
            destination.stride(0), destination_stride1,
            logical_slots_per_block);
  } else {
    scatter_rows_from_object_ptrs_kernel<uint8_t>
        <<<blocks, threads, 0, stream>>>(
            source_ptrs.data_ptr<int64_t>(), destination.data_ptr<uint8_t>(),
            destination_rows.data_ptr<int64_t>(), total_rows, rows_per_object,
            scalars_per_row, row_bytes, destination_rows_limit,
            destination.stride(0), destination_stride1,
            logical_slots_per_block);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef LAUNCH_TEMPLATED
#undef LAUNCH_BATCHED_TEMPLATED
