// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mem_kernels.cuh"  // TransferDirection, GPUKVFormat

#include <c10/cuda/CUDAGuard.h>
#include <vector>

struct PageBufferShapeDesc {
  int kv_size;       // 1 or 2
  int nl;            // num layers
  int nb;            // num blocks
  int bs;            // block size
  int nh;            // num heads
  int hs;            // head size
  int element_size;  // bytes (1 or 2)
  // 0 = dense (use scalars_per_block); > 0 = padded stride in element units.
  // Used for DeepSeek V4 mixed-compression KV layouts where vLLM pads the
  // block dimension so that dim-0 strides differ from the tight product.
  int block_stride_elems;

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_head() const {
    return hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_token() const {
    return nh * hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  __host__ __device__ inline size_t scalars_per_block() const {
    return bs * nh * hs * element_size / sizeof(ScalarType);
  }

  /**
   * Return the stride (in ScalarType units) between consecutive dim-0
   * entries in the paged buffer.
   *
   * If ``block_stride_elems == 0`` (dense layout) this is identical to
   * ``tight_in_scalar_type``; otherwise it converts the element-unit
   * padded stride to ScalarType units.
   *
   * @param tight_in_scalar_type  The dense/tight stride expressed in
   *                              ScalarType units, as computed by the
   *                              caller from ``scalars_per_block()``.
   */
  template <typename ScalarType>
  __host__ __device__ inline size_t block_stride_or(
      size_t tight_in_scalar_type) const {
    if (block_stride_elems == 0) return tight_in_scalar_type;
    return (size_t)block_stride_elems * element_size / sizeof(ScalarType);
  }
};

template <typename ScalarType>
struct MemoryObj4 {
  ScalarType* objects[4];
  int num_objects;  // 0 - 4
};

/**
 * Block-level multi-layer KV transfer between vLLM paged buffers and
 * LMCache contiguous memory objects.
 *
 * @param paged_buffer_ptrs_tensor  GPU int64 tensor of data pointers into
 *                                  vLLM paged buffers (one per tensor)
 * @param lmcache_objects_ptrs      Raw pointers to LMCache memory objects
 * @param block_ids                 GPU int64 tensor of block indices in vLLM
 *                                  paged buffer
 * @param device                    CUDA device of vLLM tensors
 * @param direction                 H2D (LMCache->vLLM) or D2H (vLLM->LMCache)
 * @param shape_desc                Shape descriptor for the paged buffer
 * @param lmcache_chunk_size        Tokens per LMCache memory object
 * @param gpu_kv_format             GPUKVFormat identifier
 * @param skip_prefix_n_blocks      Number of blocks to skip at the beginning
 */
void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    GPUKVFormat gpu_kv_format, int skip_prefix_n_blocks);
