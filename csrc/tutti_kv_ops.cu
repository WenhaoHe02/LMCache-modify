// SPDX-License-Identifier: Apache-2.0
//
// GPU-direct NVMe batch I/O kernels for the Tutti/snvme integration.
//
// Most kernels are batch versions of the single-I/O k_submit_rw /
// k_poll_one kernels in snvme_smoke_gpu.cu. The fixed-size indexed CSA
// submit and the CQ poll paths run cooperatively; each publishes one queue
// doorbell per batch.
//
// SGL encoding (NVMe Base Spec 1.4 §4.4.1, SGL Data Block Descriptor):
//   prp1 = dptr0 = IOVA of the HBM staging buffer
//   prp2 = dptr1 = (byte_len & 0xFFFFFFFF) | (0x00ULL << 56)
//   flags bit [7:6] = PSDT = 0b01 → SGL Data Block
//
// References:
//   NVMe Base Spec 1.4, figures 26 (CQE), 105 (SQE), 400 (SGL)
//   Tutti paper §3.2 (GPU-centric KV Object Store, SGL vs PRP)
//   snvme_smoke_gpu.cu (k_submit_rw / k_poll_one reference)

#include <cuda_runtime.h>
#include <torch/all.h>

#include "tutti_kv_ops.cuh"

// ---------------------------------------------------------------------------
// NVMe protocol constants (NVMe Base Spec 1.4)
// ---------------------------------------------------------------------------

#define NVME_OPC_WRITE       0x01u
#define NVME_OPC_READ        0x02u
#define NVME_SQE_SIZE        64u
#define NVME_CQE_SIZE        16u

// CDW0 bits [15:14] = PSDT.  SGL Data Block = 01b → lives at flags[7:6].
#define NVME_FLAG_PSDT_SGL   (1u << 6)

// SGL Data Block descriptor subtype byte (byte 15 of the descriptor).
// 0x00 = Data Block, no keyed access.
#define NVME_SGL_BYTE15      0x00u

// NVMe logical block size assumed by snvme (map.c).
#define NVME_LBS             512u

// ---------------------------------------------------------------------------
// NVMe queue entry structures (must match controller DMA layout exactly)
// ---------------------------------------------------------------------------

struct nvme_sqe {
    uint8_t  opcode;
    uint8_t  flags;
    uint16_t cid;
    uint32_t nsid;
    uint64_t rsvd_2_3;
    uint64_t metadata;
    uint64_t prp1;   // SGL dptr0: staging IOVA when PSDT_SGL
    uint64_t prp2;   // SGL dptr1: length descriptor when PSDT_SGL
    uint32_t cdw10;  // SLBA[31:0]
    uint32_t cdw11;  // SLBA[63:32]
    uint32_t cdw12;  // NLB zero-based (nlb = bytes/512 - 1)
    uint32_t cdw13;
    uint32_t cdw14;
    uint32_t cdw15;
} __attribute__((packed));

static_assert(sizeof(nvme_sqe) == NVME_SQE_SIZE, "nvme_sqe must be 64 bytes");

struct nvme_cqe {
    uint32_t result;
    uint32_t rsvd;
    uint16_t sq_head;
    uint16_t sq_id;
    uint16_t cid;
    uint16_t status;
} __attribute__((packed));

static_assert(sizeof(nvme_cqe) == NVME_CQE_SIZE, "nvme_cqe must be 16 bytes");

// NVMe completions arrive through PCIe DMA, outside the GPU's normal store
// path. Plain C++ volatile prevents compiler hoisting but can keep an old CQ
// cache line in L2 when many future slots are polled before the controller
// writes them. PTX cache-volatile loads invalidate a matching L2 line before
// refetching it, which is required for forward progress of parallel polling.
__device__ __forceinline__ uint16_t load_cq_status_cv(const nvme_cqe* slot) {
  uint16_t status;
  const void* address = reinterpret_cast<const char*>(slot) + 14;
  asm volatile("ld.global.cv.u16 %0, [%1];" : "=h"(status) : "l"(address));
  return status;
}

__device__ __forceinline__ uint16_t load_cq_cid_cv(const nvme_cqe* slot) {
  uint16_t cid;
  const void* address = reinterpret_cast<const char*>(slot) + 12;
  asm volatile("ld.global.cv.u16 %0, [%1];" : "=h"(cid) : "l"(address));
  return cid;
}

// Per-queue device-side state passed by value into kernels.
struct tutti_queue_dev {
    nvme_sqe*          sq;      // device VA: SQ ring base
    nvme_cqe*          cq;      // device VA: CQ ring base
    volatile uint32_t* sq_db;   // GPU VA: SQ tail doorbell in BAR0
    volatile uint32_t* cq_db;   // GPU VA: CQ head doorbell in BAR0
    uint16_t           q_depth;
    uint16_t           qid;
};

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// `%globaltimer` is comparable across kernels on one GPU and is reported in
// nanoseconds on Hopper. Absolute values must never be compared across GPUs.
__device__ __forceinline__ uint64_t tutti_globaltimer_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// Submit n_ios NVMe SGL READ/WRITE commands from a single GPU thread.
// Rings the SQ doorbell after every SQE so the controller can begin DMA
// for earlier I/Os while the kernel is still building later SQEs.
template <bool Profiled>
__global__ void k_submit_batch_sgl_rw(
    tutti_queue_dev   qd,
    uint16_t*         sq_tail_io,
    uint32_t          nsid,
    uint8_t           opcode,
    const uint64_t*   staging_iovas,
    const uint64_t*   slbas,
    const uint32_t*   byte_lens,
    int               n_ios,
    uint64_t*          doorbell_timestamps)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint16_t tail = *sq_tail_io;

    for (int i = 0; i < n_ios; i++) {
        nvme_sqe* slot = &qd.sq[tail];

        // Zero the 64-byte SQE slot via explicit per-byte stores.
        // Avoids wider CUDA store instructions that could overlap with
        // concurrent controller DMA on adjacent entries.
        uint8_t* p = reinterpret_cast<uint8_t*>(slot);
        #pragma unroll 8
        for (int j = 0; j < static_cast<int>(sizeof(nvme_sqe)); j++) p[j] = 0;

        // SGL Data Block descriptor:
        //   dptr0 (prp1): IOVA of the HBM staging buffer
        //   dptr1 (prp2): (length & 0xFFFFFFFF) | (desc_type << 56)
        const uint64_t sgl_dptr1 =
            static_cast<uint64_t>(byte_lens[i] & 0xffffffffu) |
            (static_cast<uint64_t>(NVME_SGL_BYTE15) << 56);

        slot->opcode = opcode;
        slot->flags  = NVME_FLAG_PSDT_SGL;
        slot->cid    = static_cast<uint16_t>(i);  // 0-based within this batch
        slot->nsid   = nsid;
        slot->prp1   = staging_iovas[i];
        slot->prp2   = sgl_dptr1;
        slot->cdw10  = static_cast<uint32_t>(slbas[i] & 0xffffffffu);
        slot->cdw11  = static_cast<uint32_t>(slbas[i] >> 32);
        // NLB is zero-based: (byte_count / 512) - 1.
        slot->cdw12  = static_cast<uint32_t>(byte_lens[i] / NVME_LBS - 1u);

        // Ensure all SQE bytes are visible to the NVMe controller DMA engine
        // before the doorbell write propagates to the PCIe device.
        __threadfence_system();

        uint16_t new_tail = static_cast<uint16_t>((tail + 1) % qd.q_depth);
        if constexpr (Profiled) {
            // A pre-doorbell timestamp makes CQE-observed - doorbell_ts a
            // conservative upper envelope for controller-visible latency.
            doorbell_timestamps[i] = tutti_globaltimer_ns();
        }
        *qd.sq_db = new_tail;
        tail = new_tail;
    }

    *sq_tail_io = tail;
}

// Submit fixed-size reads selected through a GPU-resident LBA lookup table.
// This is the CSA fast path: Python no longer builds staging_iovas/slbas/
// byte_lens lists for every layer. Each object occupies one 64-KiB-aligned
// staging row, so its IOVA can be resolved through the registered GPU-page
// IOVA table without assuming that adjacent pages have contiguous IOVAs.
template <bool Profiled>
__global__ void k_submit_indexed_sgl_read(
    tutti_queue_dev qd,
    uint16_t* sq_tail_io,
    uint32_t nsid,
    const uint64_t* staging_page_iovas,
    uint64_t staging_stride,
    const uint64_t* slba_table,
    const int64_t* selected_ids,
    uint32_t byte_len,
    int n_ios,
    uint64_t* doorbell_timestamps)
{
    __shared__ uint16_t base_tail;
    if (threadIdx.x == 0) base_tail = *sq_tail_io;
    __syncthreads();

    for (int i = threadIdx.x; i < n_ios; i += blockDim.x) {
        const uint16_t slot_index = static_cast<uint16_t>(
            (static_cast<uint32_t>(base_tail) + static_cast<uint32_t>(i)) %
            qd.q_depth);
        nvme_sqe* slot = &qd.sq[slot_index];
        // Unlike the legacy path, the controller sees no doorbell until the
        // whole batch is ready. Each thread exclusively owns its 64-byte,
        // naturally aligned slot, so four vector stores safely replace the
        // old 64 byte stores.
        uint4* words = reinterpret_cast<uint4*>(slot);
        #pragma unroll
        for (int j = 0; j < 4; j++) words[j] = make_uint4(0, 0, 0, 0);

        const uint64_t staging_offset =
            static_cast<uint64_t>(i) * staging_stride;
        const uint64_t page_index = staging_offset / 65536u;
        const uint64_t page_offset = staging_offset % 65536u;
        const int64_t selected_id = selected_ids[i];
        const uint64_t sgl_dptr1 =
            static_cast<uint64_t>(byte_len) |
            (static_cast<uint64_t>(NVME_SGL_BYTE15) << 56);
        const uint64_t slba = slba_table[selected_id];

        slot->opcode = NVME_OPC_READ;
        slot->flags = NVME_FLAG_PSDT_SGL;
        slot->cid = static_cast<uint16_t>(i);
        slot->nsid = nsid;
        slot->prp1 = staging_page_iovas[page_index] + page_offset;
        slot->prp2 = sgl_dptr1;
        slot->cdw10 = static_cast<uint32_t>(slba & 0xffffffffu);
        slot->cdw11 = static_cast<uint32_t>(slba >> 32);
        slot->cdw12 = byte_len / NVME_LBS - 1u;
    }

    // Every producer makes its SQE stores visible to the PCIe device before
    // thread 0 advances the hardware tail. One doorbell per batch avoids the
    // previous O(n_ios) MMIO writes while preserving NVMe queue semantics.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint16_t new_tail = static_cast<uint16_t>(
          (static_cast<uint32_t>(base_tail) + static_cast<uint32_t>(n_ios)) %
          qd.q_depth);
      *sq_tail_io = new_tail;
      if constexpr (Profiled) {
        const uint64_t timestamp = tutti_globaltimer_ns();
        for (int i = 0; i < n_ios; ++i) {
          doorbell_timestamps[i] = timestamp;
        }
      }
      *qd.sq_db = new_tail;
    }
}

// Poll one queue-depth-bounded CQ batch in parallel. Each CUDA thread owns a
// strided subset of CQ slots; after every expected phase bit is visible,
// thread 0 advances the CQ head and rings the doorbell once for the batch.
//
// The legacy max_iters value was calibrated for one serial polling thread.
// Starting that raw iteration budget independently for every CQ slot makes
// later completions time out almost immediately when all slots are polled in
// parallel. Use it as a coarse wall-clock budget instead, and back off between
// MMIO reads to avoid flooding the PCIe BAR while the device is working.
template <bool Profiled>
__global__ void k_poll_batch(tutti_queue_dev qd, uint16_t* cq_head_io,
                             uint8_t* cq_phase_io, int n_ios,
                             uint32_t* status_out, int* timed_out,
                             uint64_t max_iters,
                             uint64_t* cqe_timestamps,
                             int32_t* observed_cids) {
  if (blockIdx.x != 0) return;

  const uint16_t head = *cq_head_io;
  const uint8_t phase = *cq_phase_io;
  if (threadIdx.x == 0) *timed_out = 0;
  __syncthreads();

  constexpr uint64_t kTimeoutCyclesPerLegacyIter = 512;
  constexpr unsigned int kPollBackoffNs = 256;
  const uint64_t start_clock = clock64();
  const uint64_t timeout_cycles =
      max_iters > UINT64_MAX / kTimeoutCyclesPerLegacyIter
          ? UINT64_MAX
          : max_iters * kTimeoutCyclesPerLegacyIter;

  for (int i = threadIdx.x; i < n_ios; i += blockDim.x) {
    const uint32_t absolute_slot =
        static_cast<uint32_t>(head) + static_cast<uint32_t>(i);
    const uint16_t slot_index = static_cast<uint16_t>(
        absolute_slot % static_cast<uint32_t>(qd.q_depth));
    const uint8_t expected_phase = static_cast<uint8_t>(
        phase ^ (absolute_slot >= static_cast<uint32_t>(qd.q_depth)));
    const nvme_cqe* slot = &qd.cq[slot_index];

    uint16_t status = 0;
    for (;;) {
      status = load_cq_status_cv(slot);
      if (static_cast<uint8_t>(status & 0x1u) == expected_phase) {
        if constexpr (Profiled) {
          // CQ slots are arrival ordered, not CID ordered. Preserve both the
          // observed slot order and a CID-indexed first-observed timestamp.
          const uint16_t cid = load_cq_cid_cv(slot);
          observed_cids[i] = static_cast<int32_t>(cid);
          if (cid < static_cast<uint16_t>(n_ios)) {
            cqe_timestamps[cid] = tutti_globaltimer_ns();
          }
        }
        break;
      }
      if (clock64() - start_clock >= timeout_cycles) {
        atomicExch(timed_out, 1);
        break;
      }
      __nanosleep(kPollBackoffNs);
    }
    status_out[i] = static_cast<uint32_t>(status);
    // Order this thread's CQE read before the batched head doorbell.
    __threadfence_system();
  }
  __syncthreads();

  if (threadIdx.x == 0 && *timed_out == 0) {
    const uint32_t absolute_head =
        static_cast<uint32_t>(head) + static_cast<uint32_t>(n_ios);
    const uint16_t new_head = static_cast<uint16_t>(
        absolute_head % static_cast<uint32_t>(qd.q_depth));
    const uint8_t new_phase = static_cast<uint8_t>(
        phase ^ (absolute_head >= static_cast<uint32_t>(qd.q_depth)));
    *cq_head_io = new_head;
    *cq_phase_io = new_phase;
    *qd.cq_db = new_head;
  }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

static tutti_queue_dev make_qd(int64_t sq_dev_ptr, int64_t cq_dev_ptr,
                               int64_t sq_db_ptr, int64_t cq_db_ptr,
                               int q_depth, int qid) {
  tutti_queue_dev qd;
  qd.sq = reinterpret_cast<nvme_sqe*>(sq_dev_ptr);
  qd.cq = reinterpret_cast<nvme_cqe*>(cq_dev_ptr);
  qd.sq_db = reinterpret_cast<volatile uint32_t*>(sq_db_ptr);
  qd.cq_db = reinterpret_cast<volatile uint32_t*>(cq_db_ptr);
  qd.q_depth = static_cast<uint16_t>(q_depth);
  qd.qid = static_cast<uint16_t>(qid);
  return qd;
}

// ---------------------------------------------------------------------------
// Public host wrappers (called from Python via pybind11)
// ---------------------------------------------------------------------------

static void tutti_submit_batch_sgl_rw(
    int64_t   sq_dev_ptr,
    int64_t   cq_dev_ptr,
    int64_t   sq_db_ptr,
    int64_t   cq_db_ptr,
    int64_t   sq_tail_ptr,
    int       q_depth,
    int       qid,
    int64_t   nsid,
    at::Tensor staging_iovas,
    at::Tensor slbas,
    at::Tensor byte_lens,
    uint8_t    opcode,
    at::Tensor* doorbell_timestamps,
    int64_t   stream_ptr)
{
    TORCH_CHECK(staging_iovas.dtype() == at::kLong,
                "staging_iovas must be int64 (uint64 reinterpret)");
    TORCH_CHECK(slbas.dtype() == at::kLong,
                "slbas must be int64 (uint64 reinterpret)");
    TORCH_CHECK(byte_lens.dtype() == at::kInt,
                "byte_lens must be int32 (uint32 reinterpret)");
    TORCH_CHECK(staging_iovas.is_contiguous(), "staging_iovas must be contiguous");
    TORCH_CHECK(slbas.is_contiguous(),         "slbas must be contiguous");
    TORCH_CHECK(byte_lens.is_contiguous(),     "byte_lens must be contiguous");

    const int n_ios = static_cast<int>(staging_iovas.numel());
    TORCH_CHECK(n_ios > 0, "n_ios must be positive");
    TORCH_CHECK(n_ios < q_depth,
                "n_ios (", n_ios, ") must be smaller than q_depth (",
                q_depth, ") so the NVMe queue retains one empty slot");

    tutti_queue_dev qd = make_qd(
        sq_dev_ptr, cq_dev_ptr, sq_db_ptr, cq_db_ptr, q_depth, qid);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    if (doorbell_timestamps != nullptr) {
      TORCH_CHECK(doorbell_timestamps->is_cuda(),
                  "doorbell_timestamps must be a CUDA tensor");
      TORCH_CHECK(doorbell_timestamps->dtype() == at::kLong,
                  "doorbell_timestamps must be int64");
      TORCH_CHECK(doorbell_timestamps->is_contiguous(),
                  "doorbell_timestamps must be contiguous");
      TORCH_CHECK(doorbell_timestamps->numel() >= n_ios,
                  "doorbell_timestamps is too small");
      k_submit_batch_sgl_rw<true><<<1, 1, 0, stream>>>(
          qd, reinterpret_cast<uint16_t*>(sq_tail_ptr),
          static_cast<uint32_t>(nsid), opcode,
          reinterpret_cast<const uint64_t*>(staging_iovas.data_ptr<int64_t>()),
          reinterpret_cast<const uint64_t*>(slbas.data_ptr<int64_t>()),
          reinterpret_cast<const uint32_t*>(byte_lens.data_ptr<int32_t>()),
          n_ios, reinterpret_cast<uint64_t*>(
                     doorbell_timestamps->data_ptr<int64_t>()));
    } else {
      k_submit_batch_sgl_rw<false><<<1, 1, 0, stream>>>(
          qd, reinterpret_cast<uint16_t*>(sq_tail_ptr),
          static_cast<uint32_t>(nsid), opcode,
          reinterpret_cast<const uint64_t*>(staging_iovas.data_ptr<int64_t>()),
          reinterpret_cast<const uint64_t*>(slbas.data_ptr<int64_t>()),
          reinterpret_cast<const uint32_t*>(byte_lens.data_ptr<int32_t>()),
          n_ios, nullptr);
    }
}

void tutti_submit_batch_sgl_read(
    int64_t   sq_dev_ptr,
    int64_t   cq_dev_ptr,
    int64_t   sq_db_ptr,
    int64_t   cq_db_ptr,
    int64_t   sq_tail_ptr,
    int       q_depth,
    int       qid,
    int64_t   nsid,
    at::Tensor staging_iovas,
    at::Tensor slbas,
    at::Tensor byte_lens,
    int64_t   stream_ptr)
{
    tutti_submit_batch_sgl_rw(
        sq_dev_ptr, cq_dev_ptr, sq_db_ptr, cq_db_ptr, sq_tail_ptr,
        q_depth, qid, nsid, staging_iovas, slbas, byte_lens,
        NVME_OPC_READ, nullptr, stream_ptr);
}

void tutti_submit_batch_sgl_read_profiled(
    int64_t sq_dev_ptr, int64_t cq_dev_ptr, int64_t sq_db_ptr,
    int64_t cq_db_ptr, int64_t sq_tail_ptr, int q_depth, int qid,
    int64_t nsid, at::Tensor staging_iovas, at::Tensor slbas,
    at::Tensor byte_lens, at::Tensor doorbell_timestamps,
    int64_t stream_ptr) {
  tutti_submit_batch_sgl_rw(
      sq_dev_ptr, cq_dev_ptr, sq_db_ptr, cq_db_ptr, sq_tail_ptr, q_depth,
      qid, nsid, staging_iovas, slbas, byte_lens, NVME_OPC_READ,
      &doorbell_timestamps, stream_ptr);
}

void tutti_submit_indexed_sgl_read(
    int64_t sq_dev_ptr,
    int64_t cq_dev_ptr,
    int64_t sq_db_ptr,
    int64_t cq_db_ptr,
    int64_t sq_tail_ptr,
    int q_depth,
    int qid,
    int64_t nsid,
    at::Tensor staging_page_iovas,
    int64_t staging_stride,
    at::Tensor slba_table,
    at::Tensor selected_ids,
    int byte_len,
    int64_t stream_ptr)
{
    TORCH_CHECK(staging_page_iovas.dtype() == at::kLong,
                "staging_page_iovas must be int64");
    TORCH_CHECK(slba_table.dtype() == at::kLong,
                "slba_table must be int64");
    TORCH_CHECK(selected_ids.dtype() == at::kLong,
                "selected_ids must be int64");
    TORCH_CHECK(staging_page_iovas.is_cuda(),
                "staging_page_iovas must be a CUDA tensor");
    TORCH_CHECK(slba_table.is_cuda(), "slba_table must be a CUDA tensor");
    TORCH_CHECK(selected_ids.is_cuda(), "selected_ids must be a CUDA tensor");
    TORCH_CHECK(staging_page_iovas.is_contiguous(),
                "staging_page_iovas must be contiguous");
    TORCH_CHECK(slba_table.is_contiguous(), "slba_table must be contiguous");
    TORCH_CHECK(selected_ids.is_contiguous(),
                "selected_ids must be contiguous");
    TORCH_CHECK(staging_stride > 0 && staging_stride % 65536 == 0,
                "staging_stride must be a positive multiple of 65536");
    TORCH_CHECK(byte_len > 0 && byte_len % NVME_LBS == 0,
                "byte_len must be a positive multiple of 512");
    TORCH_CHECK(byte_len <= staging_stride,
                "byte_len must not exceed staging_stride");

    const int n_ios = static_cast<int>(selected_ids.numel());
    TORCH_CHECK(n_ios > 0, "n_ios must be positive");
    TORCH_CHECK(n_ios < q_depth,
                "n_ios (", n_ios, ") must be smaller than q_depth (",
                q_depth, ") for one batched doorbell");
    TORCH_CHECK(
        static_cast<int64_t>(n_ios) * staging_stride <=
            staging_page_iovas.numel() * 65536,
        "indexed staging plan exceeds registered staging pages");

    tutti_queue_dev qd = make_qd(
        sq_dev_ptr, cq_dev_ptr, sq_db_ptr, cq_db_ptr, q_depth, qid);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int kSubmitThreads = 256;
    k_submit_indexed_sgl_read<false><<<1, kSubmitThreads, 0, stream>>>(
        qd,
        reinterpret_cast<uint16_t*>(sq_tail_ptr),
        static_cast<uint32_t>(nsid),
        reinterpret_cast<const uint64_t*>(
            staging_page_iovas.data_ptr<int64_t>()),
        static_cast<uint64_t>(staging_stride),
        reinterpret_cast<const uint64_t*>(slba_table.data_ptr<int64_t>()),
        selected_ids.data_ptr<int64_t>(),
        static_cast<uint32_t>(byte_len),
        n_ios,
        nullptr);
}

void tutti_submit_indexed_sgl_read_profiled(
    int64_t sq_dev_ptr, int64_t cq_dev_ptr, int64_t sq_db_ptr,
    int64_t cq_db_ptr, int64_t sq_tail_ptr, int q_depth, int qid,
    int64_t nsid, at::Tensor staging_page_iovas, int64_t staging_stride,
    at::Tensor slba_table, at::Tensor selected_ids, int byte_len,
    at::Tensor doorbell_timestamps, int64_t stream_ptr) {
  TORCH_CHECK(doorbell_timestamps.is_cuda(),
              "doorbell_timestamps must be a CUDA tensor");
  TORCH_CHECK(doorbell_timestamps.dtype() == at::kLong,
              "doorbell_timestamps must be int64");
  TORCH_CHECK(doorbell_timestamps.is_contiguous(),
              "doorbell_timestamps must be contiguous");
  const int n_ios = static_cast<int>(selected_ids.numel());
  TORCH_CHECK(doorbell_timestamps.numel() >= n_ios,
              "doorbell_timestamps is too small");

  // Reuse the production wrapper's input validation before launching the
  // profiled specialization. The duplicate launch is avoided by keeping the
  // validation below explicit rather than calling that wrapper.
  TORCH_CHECK(staging_page_iovas.dtype() == at::kLong &&
                  slba_table.dtype() == at::kLong &&
                  selected_ids.dtype() == at::kLong,
              "indexed descriptor tensors must be int64");
  TORCH_CHECK(staging_page_iovas.is_cuda() && slba_table.is_cuda() &&
                  selected_ids.is_cuda(),
              "indexed descriptor tensors must be CUDA tensors");
  TORCH_CHECK(staging_page_iovas.is_contiguous() &&
                  slba_table.is_contiguous() && selected_ids.is_contiguous(),
              "indexed descriptor tensors must be contiguous");
  TORCH_CHECK(n_ios > 0 && n_ios < q_depth,
              "n_ios must be positive and smaller than q_depth");
  TORCH_CHECK(staging_stride > 0 && staging_stride % 65536 == 0,
              "staging_stride must be a positive multiple of 65536");
  TORCH_CHECK(byte_len > 0 && byte_len % NVME_LBS == 0 &&
                  byte_len <= staging_stride,
              "invalid indexed byte_len");
  TORCH_CHECK(static_cast<int64_t>(n_ios) * staging_stride <=
                  staging_page_iovas.numel() * 65536,
              "indexed staging plan exceeds registered staging pages");

  tutti_queue_dev qd = make_qd(
      sq_dev_ptr, cq_dev_ptr, sq_db_ptr, cq_db_ptr, q_depth, qid);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  constexpr int kSubmitThreads = 256;
  k_submit_indexed_sgl_read<true><<<1, kSubmitThreads, 0, stream>>>(
      qd, reinterpret_cast<uint16_t*>(sq_tail_ptr),
      static_cast<uint32_t>(nsid),
      reinterpret_cast<const uint64_t*>(
          staging_page_iovas.data_ptr<int64_t>()),
      static_cast<uint64_t>(staging_stride),
      reinterpret_cast<const uint64_t*>(slba_table.data_ptr<int64_t>()),
      selected_ids.data_ptr<int64_t>(), static_cast<uint32_t>(byte_len),
      n_ios,
      reinterpret_cast<uint64_t*>(doorbell_timestamps.data_ptr<int64_t>()));
}

void tutti_submit_batch_sgl_write(
    int64_t   sq_dev_ptr,
    int64_t   cq_dev_ptr,
    int64_t   sq_db_ptr,
    int64_t   cq_db_ptr,
    int64_t   sq_tail_ptr,
    int       q_depth,
    int       qid,
    int64_t   nsid,
    at::Tensor staging_iovas,
    at::Tensor slbas,
    at::Tensor byte_lens,
    int64_t   stream_ptr)
{
  tutti_submit_batch_sgl_rw(sq_dev_ptr, cq_dev_ptr, sq_db_ptr, cq_db_ptr,
                            sq_tail_ptr, q_depth, qid, nsid, staging_iovas,
                            slbas, byte_lens, NVME_OPC_WRITE, nullptr,
                            stream_ptr);
}

void tutti_poll_batch(int64_t sq_dev_ptr, int64_t cq_dev_ptr, int64_t sq_db_ptr,
                      int64_t cq_db_ptr, int64_t cq_head_ptr,
                      int64_t cq_phase_ptr, int q_depth, int n_ios,
                      at::Tensor status_out, int64_t timed_out_ptr,
                      int64_t max_iters, int64_t stream_ptr) {
  TORCH_CHECK(status_out.dtype() == at::kInt, "status_out must be int32");
  TORCH_CHECK(status_out.is_contiguous(), "status_out must be contiguous");
  TORCH_CHECK(static_cast<int>(status_out.numel()) >= n_ios,
              "status_out too small (", status_out.numel(), " < ", n_ios, ")");
  TORCH_CHECK(max_iters > 0, "max_iters must be positive");

  tutti_queue_dev qd =
      make_qd(sq_dev_ptr, cq_dev_ptr, sq_db_ptr, cq_db_ptr, q_depth, 0);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  constexpr int kPollThreads = 256;
  k_poll_batch<false><<<1, kPollThreads, 0, stream>>>(
      qd, reinterpret_cast<uint16_t*>(cq_head_ptr),
      reinterpret_cast<uint8_t*>(cq_phase_ptr), n_ios,
      reinterpret_cast<uint32_t*>(status_out.data_ptr<int32_t>()),
      reinterpret_cast<int*>(timed_out_ptr), static_cast<uint64_t>(max_iters),
      nullptr, nullptr);
}


void tutti_poll_batch_profiled(
    int64_t sq_dev_ptr, int64_t cq_dev_ptr, int64_t sq_db_ptr,
    int64_t cq_db_ptr, int64_t cq_head_ptr, int64_t cq_phase_ptr,
    int q_depth, int n_ios, at::Tensor status_out, int64_t timed_out_ptr,
    int64_t max_iters, at::Tensor cqe_timestamps, at::Tensor observed_cids,
    int64_t stream_ptr) {
  TORCH_CHECK(status_out.dtype() == at::kInt && status_out.is_contiguous(),
              "status_out must be contiguous int32");
  TORCH_CHECK(cqe_timestamps.is_cuda() &&
                  cqe_timestamps.dtype() == at::kLong &&
                  cqe_timestamps.is_contiguous(),
              "cqe_timestamps must be contiguous CUDA int64");
  TORCH_CHECK(observed_cids.is_cuda() && observed_cids.dtype() == at::kInt &&
                  observed_cids.is_contiguous(),
              "observed_cids must be contiguous CUDA int32");
  TORCH_CHECK(status_out.numel() >= n_ios &&
                  cqe_timestamps.numel() >= n_ios &&
                  observed_cids.numel() >= n_ios,
              "profile output tensor is too small");
  TORCH_CHECK(max_iters > 0, "max_iters must be positive");

  tutti_queue_dev qd =
      make_qd(sq_dev_ptr, cq_dev_ptr, sq_db_ptr, cq_db_ptr, q_depth, 0);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  constexpr int kPollThreads = 256;
  k_poll_batch<true><<<1, kPollThreads, 0, stream>>>(
      qd, reinterpret_cast<uint16_t*>(cq_head_ptr),
      reinterpret_cast<uint8_t*>(cq_phase_ptr), n_ios,
      reinterpret_cast<uint32_t*>(status_out.data_ptr<int32_t>()),
      reinterpret_cast<int*>(timed_out_ptr), static_cast<uint64_t>(max_iters),
      reinterpret_cast<uint64_t*>(cqe_timestamps.data_ptr<int64_t>()),
      observed_cids.data_ptr<int32_t>());
}
