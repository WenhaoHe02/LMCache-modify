// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

// Batch-submit N NVMe SGL READ commands from a single GPU thread.
//
// Each I/O reads byte_lens[i] bytes from the NVMe namespace at logical
// block address slbas[i], depositing the payload directly into the
// HBM staging buffer whose NVMe IOVA is staging_iovas[i].
//
// The kernel runs single-threaded (<<<1,1>>> launch) so that the SQ
// tail, doorbell writes, and counters need no synchronisation.
//
// Args:
//   sq_dev_ptr:     Device VA of SQ ring (array of nvme_sqe[q_depth]).
//   cq_dev_ptr:     Device VA of CQ ring (array of nvme_cqe[q_depth]).
//   sq_db_ptr:      GPU VA of SQ tail doorbell in BAR0 (volatile uint32_t).
//   cq_db_ptr:      GPU VA of CQ head doorbell in BAR0 (volatile uint32_t).
//   sq_tail_ptr:    Managed/pinned uint16_t: current SQ tail.
//   q_depth:        Hardware queue depth (power of 2, usually 64).
//   qid:            NVMe I/O queue ID (informational only).
//   nsid:           NVMe namespace ID (from NVM_GET_DEV_INFO.block_size).
//   staging_iovas:  [n_ios] int64_t – NVMe IOVAs of HBM staging buffers.
//   slbas:          [n_ios] int64_t – starting LBA per chunk.
//   byte_lens:      [n_ios] int32_t – transfer size in bytes per chunk.
//                   Must be a multiple of 512 (NVMe logical block size).
//   stream_ptr:     cudaStream_t cast to int64_t (0 = default stream).
void tutti_submit_batch_sgl_read(
    int64_t sq_dev_ptr,
    int64_t cq_dev_ptr,
    int64_t sq_db_ptr,
    int64_t cq_db_ptr,
    int64_t sq_tail_ptr,
    int q_depth,
    int qid,
    int64_t nsid,
    at::Tensor staging_iovas,
    at::Tensor slbas,
    at::Tensor byte_lens,
    int64_t stream_ptr);

// Poll for n_ios NVMe CQE completions in submission order.
//
// Spins on the CQE phase bit for each slot in sequence.  On timeout
// (*timed_out = 1), returns immediately leaving head/phase at the last
// successfully consumed CQE position.
//
// Args:
//   sq_dev_ptr / cq_dev_ptr / sq_db_ptr / cq_db_ptr: same as submit.
//   cq_head_ptr:    Managed/pinned uint16_t: current CQ head.
//   cq_phase_ptr:   Managed/pinned uint8_t:  current phase bit.
//   q_depth:        Hardware queue depth.
//   n_ios:          Number of completions to wait for.
//   status_out:     [n_ios] int32_t – NVMe status word per CQE.
//                   Zero indicates success; non-zero encodes SC/SCT.
//   timed_out_ptr:  Managed/pinned int32_t – set to 1 on timeout.
//   max_iters:      Per-CQE spin budget before declaring a timeout.
//   stream_ptr:     cudaStream_t cast to int64_t (0 = default stream).
void tutti_poll_batch(
    int64_t sq_dev_ptr,
    int64_t cq_dev_ptr,
    int64_t sq_db_ptr,
    int64_t cq_db_ptr,
    int64_t cq_head_ptr,
    int64_t cq_phase_ptr,
    int q_depth,
    int n_ios,
    at::Tensor status_out,
    int64_t timed_out_ptr,
    int64_t max_iters,
    int64_t stream_ptr);
