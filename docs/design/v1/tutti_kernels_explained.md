# Tutti CUDA Kernel 详解

> 来源：`F:\LMCache\Tutti\backends\local\kernel_modules\test\snvme_smoke_gpu.cu`
>
> 这个文件是 Tutti 整个 GPU-centric NVMe I/O 系统的核心验证程序，包含 4 个 CUDA kernel
> 和完整的 host-side 控制面代码。理解它等于理解 gio_uring 的完整数据面。

---

## 背景：为什么 GPU 可以直接驱动 NVMe

标准 NVMe 的 I/O 流程是：CPU 写 SQE → 写 BAR0 doorbell → 等 CQE → CPU 读 CQE。
整条路径都在 CPU 上，GPU 必须等 CPU 中转。

Tutti 的改动：
1. **snvme 内核驱动**：通过 `nvidia_p2p_get_pages` 把 GPU HBM 的物理地址（IOVA）
   注册为 NVMe DMA 目标。NVMe 控制器的 DMA 引擎直接往 GPU HBM 写数据，绕过 CPU DRAM。
2. **BAR0 cudaHostRegister**：把 NVMe 控制器的 PCIe BAR0（包含 doorbell 寄存器）
   注册为 CUDA IO memory，让 GPU 可以直接写 doorbell。
3. **GPU-resident SQ/CQ rings**：SQ 和 CQ 的 ring buffer 本身也放在 GPU HBM。

有了这三点，GPU 上的 CUDA kernel 就能完成整条 I/O 路径：
填 SQE → 写 doorbell → 等 CQE，全程不经过 CPU。

---

## 关键数据结构

```c
// NVMe Submission Queue Entry (64 bytes, 与 NVMe 1.4 规范完全对应)
struct nvme_sqe {
    uint8_t  opcode;     // 0x01=WRITE, 0x02=READ
    uint8_t  flags;      // bits[7:6]=PSDT: 00=PRP, 01=SGL
    uint16_t cid;        // command ID，用于匹配 CQE
    uint32_t nsid;       // namespace ID，通常是 1
    uint64_t rsvd_2_3;
    uint64_t metadata;
    uint64_t prp1;       // 数据 buffer 第一个 4KiB 页的 IOVA (DMA 地址)
    uint64_t prp2;       // 第二个页 IOVA，或 PRP_List 的 IOVA
    uint32_t cdw10;      // SLBA[31:0]   起始逻辑块地址低32位
    uint32_t cdw11;      // SLBA[63:32]  高32位
    uint32_t cdw12;      // NLB (zero-based): 传输的 LBA 数量 - 1
    ...
};

// NVMe Completion Queue Entry (16 bytes)
struct nvme_cqe {
    uint32_t result;     // 命令结果
    uint32_t rsvd;
    uint16_t sq_head;    // SQ head pointer (告诉主机 SQ 消耗到哪了)
    uint16_t sq_id;
    uint16_t cid;        // 对应 SQE 的 command ID
    uint16_t status;     // bit[0]=phase bit; bits[15:1]=status code
};

// GPU 上的每个队列状态（传值给 kernel）
struct test_queue_dev {
    nvme_sqe*           sq;       // GPU HBM 里的 SQ ring 起始地址
    nvme_cqe*           cq;       // GPU HBM 里的 CQ ring 起始地址
    volatile uint32_t*  sq_db;    // BAR0 里 SQ doorbell 的 GPU VA
    volatile uint32_t*  cq_db;    // BAR0 里 CQ doorbell 的 GPU VA
    uint16_t            q_depth;  // 队列深度 (通常 64)
    uint16_t            qid;      // 队列 ID
};
```

---

## Kernel 1：`k_submit_rw` — GPU 提交一个 NVMe I/O 命令

```cuda
__global__ void k_submit_rw(
    test_queue_dev qd,
    uint16_t* sq_tail_io,        // unified memory，当前 SQ tail 指针
    uint16_t cid,                // command ID
    uint8_t  opcode,             // NVME_OPC_WRITE=0x01 or READ=0x02
    uint8_t  flags,              // PSDT: PRP(0x00) or SGL(0x40)
    uint32_t nsid,
    uint64_t dptr0,              // PRP1: 数据 buffer 首页 IOVA
    uint64_t dptr1,              // PRP2: 第二页 IOVA 或 PRP_List IOVA
    uint64_t slba,               // 起始 LBA
    uint16_t nlb_zero_based      // 传输 LBA 数 - 1
)
```

**执行逻辑（单线程 kernel，`<<<1,1>>>`）：**

```
1. 只让 thread(0,0) 执行（其余线程立刻 return）

2. 从 sq_tail_io 读取当前 tail，找到 SQ ring 里下一个空 slot：
   nvme_sqe* slot = &qd.sq[tail]

3. 用字节循环把 64-byte slot 清零（volatile 语义，确保 controller 看到
   的不是旧数据残留）

4. 填写 SQE 字段：
   - opcode, flags, cid, nsid
   - prp1 = dptr0, prp2 = dptr1
   - cdw10 = slba[31:0], cdw11 = slba[63:32], cdw12 = nlb_zero_based

5. __threadfence_system()
   ← 关键！保证 SQE 的所有字节在 doorbell 写出前对 NVMe 控制器的
   DMA 引擎可见（CPU-visible store ordering，跨 PCIe 边界）

6. new_tail = (tail + 1) % q_depth
   *qd.sq_db = new_tail        ← 写 BAR0 doorbell，通知控制器有新 SQE
   *sq_tail_io = new_tail      ← 更新 unified memory 里的 tail 指针
```

**关键点：`__threadfence_system()`**

普通的 `__threadfence()` 只保证 GPU 内部的 L1/L2 缓存一致性。
`__threadfence_system()` 额外保证 GPU 的写操作对 CPU 和 PCIe 设备可见，
相当于一个系统级内存屏障。没有它，控制器可能在 SQE 字段还没写完的时候
就看到了 doorbell 的更新，导致命令数据损坏。

**数据格式详解：PRP vs SGL**

- **PRP（Physical Region Page）**：每个 4KiB 页用一个 64-bit IOVA 表示
  - PRP1 = 第一个页
  - PRP2 = 第二个页（8KiB 传输），或者 PRP_List 的地址（>8KiB）
  - PRP_List：GPU HBM 里的一个 uint64_t 数组，每项是一个 4KiB 页的 IOVA

- **SGL（Scatter-Gather List）**：NVMe 1.4 扩展，描述符更灵活
  - dptr0 = SGL 数据 descriptor 的起始地址 IOVA
  - dptr1 = (length[31:0]) | (descriptor_type << 56)
  - Tutti 论文中用 SGL 替代 PRP 是核心贡献之一（SGL 支持非连续、变长 I/O）

---

## Kernel 2：`k_poll_one` — GPU 等待一个 NVMe 命令完成

```cuda
__global__ void k_poll_one(
    test_queue_dev qd,
    uint16_t* cq_head_io,        // unified memory，当前 CQ head
    uint8_t*  cq_phase_io,       // unified memory，当前期望的 phase bit (0 或 1)
    nvme_cqe* out_cqe,           // unified memory，输出 CQE
    int*      timed_out,         // unified memory，超时标志
    uint64_t  max_iters          // 最大自旋次数
)
```

**执行逻辑（单线程 kernel，`<<<1,1>>>`）：**

```
1. 只让 thread(0,0) 执行

2. 从 cq_head_io 和 cq_phase_io 读取当前 head 和期望 phase

3. 自旋等待 CQE:
   for (;;) {
       volatile nvme_cqe* slot = &qd.cq[head]
       uint16_t status = slot->status
       uint8_t phase = status & 0x1      ← phase bit 是 status 的第0位

       if (phase == expected) {
           // CQE 到了！NVMe 控制器把 phase bit 翻转来表示新 CQE
           复制 CQE 到 out_cqe
           
           new_head = (head + 1) % q_depth
           if (new_head == 0) expected ^= 1   ← ring 回绕时翻转期望 phase
           
           __threadfence_system()   ← 确保 CQE 读取对后续操作可见
           *qd.cq_db = new_head     ← 写 CQ head doorbell，归还 CQ credit
           *cq_head_io = new_head
           *cq_phase_io = expected
           *timed_out = 0
           return
       }
       
       if (++i >= max_iters) {
           *timed_out = 1
           return
       }
   }
```

**Phase bit 机制（NVMe 环形队列协议）：**

NVMe CQ 是一个环形 buffer，控制器和主机没有独立的"满/空"指针，而是用
phase bit 来区分"新 CQE"和"旧/空 CQE"：

- 初始化时所有 CQE 的 phase = 0，主机期望 phase = 1
- 控制器写入新 CQE 时，把该 slot 的 phase bit 设为 1（当前期望值）
- 主机看到 phase == expected → 确认是新 CQE
- 当 head 绕过 ring 末尾回到 0 时，期望 phase 翻转为 0
- 控制器下一轮写入时，新 CQE 的 phase 又设为 0，以此类推

这个机制完全不需要互斥锁，只用一个 bit 就实现了生产者-消费者同步。

**为什么要 `volatile nvme_cqe*`：**

GPU L1 缓存默认会缓存全局内存读取。如果不用 `volatile`，GPU 会缓存
第一次读到的 `status`（phase=0），之后的循环读的都是缓存值，
永远看不到控制器更新的 phase=1。`volatile` 强制每次都去内存读。

---

## Kernel 3：`k_fill_pattern` — GPU 填充测试数据

```cuda
__global__ void k_fill_pattern(uint8_t* buf, size_t bytes, uint8_t pat) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes) return;
    buf[idx] = pat ^ (uint8_t)(idx >> 12);
}
```

**作用：** 用确定性的字节模式填充 GPU 数据 buffer（在写入 NVMe 之前）。

**模式：** `pat ^ (idx >> 12)`
- `pat` 由 `round_idx ^ qid ^ io_index` 生成，不同轮次/队列/IO 不同
- `idx >> 12` 每 4KiB（一个 NVMe LBA）变化一次，区分同一 IO 里不同页

这样读回来验证时，任何字节损坏都能精确定位到"哪个 round，哪个 queue，
哪个 IO，哪个 4KiB 页，哪个字节偏移"。

**启动配置：** `<<<blocks, 256>>>` 其中 `blocks = ceil(bytes / 256)`，
标准 1D 并行，每个线程填一个字节。

---

## Kernel 4：`k_verify_pattern` — GPU 验证读回数据

```cuda
__global__ void k_verify_pattern(
    const uint8_t* buf, size_t bytes, uint8_t pat, int* mismatch_idx)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes) return;
    uint8_t expect = pat ^ (uint8_t)(idx >> 12);
    if (buf[idx] != expect) {
        atomicCAS(mismatch_idx, -1, (int)idx);
    }
}
```

**作用：** 验证从 NVMe 读回的数据与写入时的模式完全一致。

**`atomicCAS(mismatch_idx, -1, (int)idx)`：**

Compare-And-Swap：如果 `*mismatch_idx == -1`（初始值），就把它设为 `idx`。
这是一个"第一个发现不匹配的线程赢"的并发写入模式：
- host 初始化 `*mismatch_idx = -1`
- 多个线程并发检查各自的字节
- 第一个发现不匹配的线程原子地写入自己的 `idx`
- 其余发现不匹配的线程的 CAS 失败（`*mismatch_idx` 已经不是 -1），不覆盖

不需要精确知道"所有"不匹配字节，只需要知道"有没有不匹配"，
`atomicCAS` 是最低开销的实现（没有 `atomicAdd`，没有全局归约）。

---

## 完整 I/O 流水线（host-side 编排 + kernel 序列）

```
host:  k_fill_pattern<<<>>> (填充 wbuf，GPU 上，256 线程/block)
host:  cudaDeviceSynchronize()
         ↓
host:  k_submit_rw<<<1,1>>> (GPU 写 SQE + ring SQ doorbell)
         ↓ (同一 CUDA stream，隐式串行)
host:  k_poll_one<<<1,1>>>  (GPU 自旋等 CQE，ring CQ head doorbell)
host:  cudaDeviceSynchronize() ← 等 k_poll_one 返回

底层硬件并发（在 k_submit_rw 和 k_poll_one 之间）：
   NVMe 控制器:  从 GPU HBM (wbuf) DMA 读数据 → 写到 NVMe flash
   GPU:          k_poll_one 在自旋等 phase bit 翻转

host:  cudaMemset(rbuf, 0, ...)  ← 清空读 buffer
host:  k_submit_rw<<<1,1>>> (READ 命令)
host:  k_poll_one<<<1,1>>>   (等 READ 完成)
host:  cudaDeviceSynchronize()

底层硬件并发：
   NVMe 控制器:  从 flash 读数据 → DMA 写到 GPU HBM (rbuf)
   GPU:          自旋等 phase bit

host:  k_verify_pattern<<<>>> (GPU 验证 rbuf)
host:  cudaDeviceSynchronize()
host:  检查 *mismatch_idx == -1 → PASS/FAIL
```

---

## SGL：Tutti 的核心 I/O 描述符格式

**SGL（Scatter-Gather List）是 Tutti 生产路径的主要格式**，smoke test 里的 Tier 1-3
PRP 测试只是兼容性验证，不代表 Tutti 的实际 I/O 策略。

### SGL 在 SQE 里的编码

当 `flags = NVME_FLAG_PSDT_SGL (0x40)` 时，控制器把 SQE 里的 `prp1/prp2`
这 16 个字节**整体**解释成一个 SGL Data Block 描述符：

```
SQE dptr 字段（16 bytes）:
  prp1/dptr0 [bytes  0-7]  = Address  ← 数据 buffer 的 IOVA（任意对齐）
  prp2/dptr1 [bytes  8-11] = Length   ← 传输字节数（任意大小）
             [bytes 12-14] = Reserved
             [byte     15] = Type/Subtype = 0x00 = SGL Data Block

代码（Tier 4）:
  sgl_addr = wpage(0)                         // dptr0 = 数据 IOVA
  sgl_meta = (io_bytes & 0xffffffff)          // dptr1 低32位 = 字节长度
           | ((uint64_t)NVME_SGL_DESC_BYTE15 << 56) // 高8位 = 0x00
```

描述符完全装在 SQE 的 16 字节 dptr 字段里，零额外 DMA 开销。

### PRP vs SGL：为什么 SGL 对 KV cache 更合适

| | PRP | SGL Data Block |
|-|-----|----------------|
| 寻址粒度 | 固定 4 KiB 页 | 任意字节数 |
| >8 KiB 传输 | 需要 PRP_List（额外一次 DMA 读取 List 数组） | 单描述符，零额外开销 |
| 非 4 KiB 对齐 | 不支持 | 支持 |
| KV block 512 bytes | 要 padding 到 4 KiB，8× 浪费 | 精确传 512 bytes |

CSA 每个 compressed block = 512 bytes，HCA compressed row = 512 bytes，
都远小于 4 KiB。PRP 必须把每个 block padding 到一个 4 KiB 页，
NVMe 读取的是 4 KiB 但有效数据只有 512 bytes，带宽利用率 12.5%。
SGL 精确指定字节数，带宽利用率 100%。

这就是论文里 "SGL vs PRP 31×/91.3× 读写提升" 的根本原因。

### 四个 Smoke Test Tier（由简到难的兼容性验证）

| Tier | 大小 | 方式 | 说明 |
|------|------|------|------|
| 1 | 4 KiB | PRP1 only | dptr0=数据页IOVA, dptr1=0 |
| 2 | 8 KiB | PRP1+PRP2 | dptr0=第1页, dptr1=第2页 |
| 3 | 16 KiB | PRP1+PRP_List | dptr1=PRP_List IOVA（GPU HBM 里的 uint64_t[3]，需额外DMA读） |
| 4 | 4 KiB | **SGL Data Block** | dptr0=addr, dptr1=(length\|type<<56)，零额外DMA |

Tier 4 在控制器支持（`sgl_supported != 0`）时才运行，这是 Tutti 生产路径的基础。

---

## 动态 alloc/free 压力测试（N rounds）

每个 round 完整经历：

```
NVM_CREATE_QUEUE_GROUP
  → cudaMalloc(SQ ring + CQ ring, 64KiB GPU page each)
  → NVM_MAP_DEVICE_MEMORY(SQ ring, kind=RING_SQ, group_id=round_gid)
  → NVM_MAP_DEVICE_MEMORY(CQ ring, kind=RING_CQ, group_id=round_gid)
  → NVM_ADD_USER_QUEUE → controller 执行 Create I/O SQ + Create I/O CQ
  → 跑 Tier 1/2/3/4 + wrap 压力测试
  → NVM_DESTROY_QUEUE_GROUP → controller 执行 Delete I/O SQ/CQ
                             → snvme 释放 RING_SQ/RING_CQ map 的 p2p pin
  → cudaFree(SQ ring + CQ ring)
```

数据 buffer（wbuf/rbuf/PRP_List）用 `kind=DATA, group_id=0`，
只注册一次，跨所有 round 复用，在 `close(fd_dev)` 时由
`snvm_dev_release` 统一释放。

这个测试验证：
1. snvme 的 user QID pool 在反复 CREATE/DESTROY 后无 QID 泄漏
2. `nvidia_p2p_get_pages` / `put_pages` 的 refcount 在多轮后归零
3. 控制器接受在同一次 bind 内多次 Create/Delete I/O SQ/CQ

---

## Map Kind 标签（B6 ABI）

```c
enum nvm_map_kind {
    NVM_MAP_KIND_UNSPEC  = 0,
    NVM_MAP_KIND_RING_SQ = 1,   // SQ ring：生命期 = queue group
    NVM_MAP_KIND_RING_CQ = 2,   // CQ ring：生命期 = queue group
    NVM_MAP_KIND_DATA    = 3,   // 数据 buffer：生命期 = fd
};
```

内核用 `kind` 来区分两种生命期：
- `RING_SQ/RING_CQ`：绑定到某个 queue group，group destroy 时级联释放
- `DATA`（group_id=0）：绑定到 fd，close(fd) 时释放

这个设计的目的：允许一个应用在同一个 DMA 内存池（DATA maps）上，
高频创建/销毁 I/O 队列（RING maps），而不需要每次重新 pin 数据 buffer。
这正是 Tutti 论文里 §3.3 "GPU-centric object store"的核心工程设计。

---

## 与 Tutti 论文的对应关系

| 论文术语 | 代码实现 |
|---------|---------|
| gio_uring | `k_submit_rw` + `k_poll_one`（GPU 直接写 SQE/doorbell，自旋 CQE）|
| GPU P2P DMA | `NVM_MAP_DEVICE_MEMORY` + `nvidia_p2p_get_pages` |
| SGL（vs PRP） | Tier 4：`NVME_FLAG_PSDT_SGL`，`dptr1 = (length \| type<<56)` |
| GPU-centric object store | `kind=DATA, group_id=0` 持久 DMA 池 |
| Queue group lifecycle | `NVM_CREATE/DESTROY_QUEUE_GROUP` + ring map 级联释放 |
| Green Context（SM 分区） | 未在此文件中实现（需要上层调度层，未开源）|
| Slack-aware I/O scheduler | 未在此文件中实现（未开源）|

---

## 总结

Tutti 的 4 个 CUDA kernel 分工极简：

| Kernel | 线程数 | 功能 |
|--------|--------|------|
| `k_submit_rw` | 1 | 填 64-byte SQE，`__threadfence_system()`，写 doorbell |
| `k_poll_one` | 1 | 自旋等 phase bit，读 CQE，写 CQ head doorbell |
| `k_fill_pattern` | N（并行） | 生成确定性测试数据 |
| `k_verify_pattern` | N（并行） | 验证读回数据，`atomicCAS` 报告第一个错误字节 |

核心语义：**GPU 自己完成 NVMe 命令提交和完成轮询，CPU 只做控制面（bind/group/map）**。
数据面零 CPU 参与。这是 gio_uring 的实质。
