// SPDX-License-Identifier: Apache-2.0
//
// GPU-direct NVMe batch I/O kernels for the Tutti/snvme integration.
//
// These kernels are batch versions of the single-I/O k_submit_rw /
// k_poll_one kernels in snvme_smoke_gpu.cu.  Both kernels run
// single-threaded (<<<1,1>>>) so the SQ/CQ ring state needs no
// synchronisation primitives.
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

// Submit n_ios NVMe SGL READ commands from a single GPU thread.
// Rings the SQ doorbell after every SQE so the controller can begin DMA
// for earlier I/Os while the kernel is still building later SQEs.
__global__ void k_submit_batch_sgl_read(
    tutti_queue_dev   qd,
    uint16_t*         sq_tail_io,
    uint32_t          nsid,
    const uint64_t*   staging_iovas,
    const uint64_t*   slbas,
    const uint32_t*   byte_lens,
    int               n_ios)
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

        slot->opcode = NVME_OPC_READ;
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
        *qd.sq_db = new_tail;
        tail = new_tail;
    }

    *sq_tail_io = tail;
}

// Poll for n_ios NVMe CQE completions in submission order.
// On per-CQE timeout, sets *timed_out = 1 and returns immediately,
// leaving head/phase at the last successfully consumed CQE position.
__global__ void k_poll_batch(
    tutti_queue_dev qd,
    uint16_t*       cq_head_io,
    uint8_t*        cq_phase_io,
    int             n_ios,
    uint32_t*       status_out,
    int*            timed_out,
    uint64_t        max_iters)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint16_t head  = *cq_head_io;
    uint8_t  phase = *cq_phase_io;
    *timed_out = 0;

    for (int i = 0; i < n_ios; i++) {
        volatile nvme_cqe* slot = &qd.cq[head];

        uint64_t iter = 0;
        uint16_t status;
        for (;;) {
            status = slot->status;
            if (static_cast<uint8_t>(status & 0x1u) == phase) break;
            if (++iter >= max_iters) {
                *timed_out = 1;
                return;
            }
        }

        status_out[i] = static_cast<uint32_t>(status);

        const uint16_t new_head = static_cast<uint16_t>((head + 1) % qd.q_depth);
        // Phase bit toggles each time the head pointer wraps around.
        if (new_head == 0) phase ^= 1u;

        // Ensure the CQE read completes before writing the head doorbell.
        __threadfence_system();
        *qd.cq_db = new_head;
        head = new_head;
    }

    *cq_head_io  = head;
    *cq_phase_io = phase;
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

static tutti_queue_dev make_qd(
    int64_t sq_dev_ptr, int64_t cq_dev_ptr,
    int64_t sq_db_ptr,  int64_t cq_db_ptr,
    int     q_depth,    int qid)
{
    tutti_queue_dev qd;
    qd.sq      = reinterpret_cast<nvme_sqe*>(sq_dev_ptr);
    qd.cq      = reinterpret_cast<nvme_cqe*>(cq_dev_ptr);
    qd.sq_db   = reinterpret_cast<volatile uint32_t*>(sq_db_ptr);
    qd.cq_db   = reinterpret_cast<volatile uint32_t*>(cq_db_ptr);
    qd.q_depth = static_cast<uint16_t>(q_depth);
    qd.qid     = static_cast<uint16_t>(qid);
    return qd;
}

// ---------------------------------------------------------------------------
// Public host wrappers (called from Python via pybind11)
// ---------------------------------------------------------------------------

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
    TORCH_CHECK(n_ios <= q_depth,
                "n_ios (", n_ios, ") exceeds q_depth (", q_depth, "); "
                "split into smaller batches");

    tutti_queue_dev qd = make_qd(
        sq_dev_ptr, cq_dev_ptr, sq_db_ptr, cq_db_ptr, q_depth, qid);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    k_submit_batch_sgl_read<<<1, 1, 0, stream>>>(
        qd,
        reinterpret_cast<uint16_t*>(sq_tail_ptr),
        static_cast<uint32_t>(nsid),
        reinterpret_cast<const uint64_t*>(staging_iovas.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(slbas.data_ptr<int64_t>()),
        reinterpret_cast<const uint32_t*>(byte_lens.data_ptr<int32_t>()),
        n_ios);
}

void tutti_poll_batch(
    int64_t   sq_dev_ptr,
    int64_t   cq_dev_ptr,
    int64_t   sq_db_ptr,
    int64_t   cq_db_ptr,
    int64_t   cq_head_ptr,
    int64_t   cq_phase_ptr,
    int       q_depth,
    int       n_ios,
    at::Tensor status_out,
    int64_t   timed_out_ptr,
    int64_t   max_iters,
    int64_t   stream_ptr)
{
    TORCH_CHECK(status_out.dtype() == at::kInt,
                "status_out must be int32");
    TORCH_CHECK(status_out.is_contiguous(), "status_out must be contiguous");
    TORCH_CHECK(static_cast<int>(status_out.numel()) >= n_ios,
                "status_out too small (", status_out.numel(), " < ", n_ios, ")");
    TORCH_CHECK(max_iters > 0, "max_iters must be positive");

    tutti_queue_dev qd = make_qd(
        sq_dev_ptr, cq_dev_ptr, sq_db_ptr, cq_db_ptr, q_depth, 0);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    k_poll_batch<<<1, 1, 0, stream>>>(
        qd,
        reinterpret_cast<uint16_t*>(cq_head_ptr),
        reinterpret_cast<uint8_t*>(cq_phase_ptr),
        n_ios,
        reinterpret_cast<uint32_t*>(status_out.data_ptr<int32_t>()),
        reinterpret_cast<int*>(timed_out_ptr),
        static_cast<uint64_t>(max_iters));
}
