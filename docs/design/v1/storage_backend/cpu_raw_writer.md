# CPU raw write path for Tutti-backed local storage

## Purpose

The CPU raw write path lets GPU-direct Tutti reads and background KV writes
use separate submission queues. Reads continue to DMA from NVMe into HBM;
writes use the Linux snvme block device with `O_DIRECT`.

The default policy is deliberately conservative:

- one outstanding kernel write per rank-local SSD (QD1);
- a 4 MiB non-preemptible block;
- a 512 MiB/s per-device rate limit;
- a fresh write-planner admission decision before every block.

This leaves queue-idle gaps for latency-sensitive reads. The logical object
layout, 64 MiB write waves, metadata, and read path do not change.

## Data flow

```text
CPU layer-major snapshot
        |
        v
double-buffered CPU pack
        |
        v
CPURawBlockWriter (4 MiB, QD1, rate limited)
        |
        v
snvme kernel block device -> SSD LBAs
        |
        v
all wave writes succeed -> publish READY metadata
```

The loader resolves the kernel namespace by PCI BDF through sysfs. This is
required because snvme controller numbers are discovery-order dependent and
the namespace is created after the container starts. If the block node is not
visible in the container, the loader creates the matching node under `/tmp`
using the sysfs major/minor numbers.

## Configuration

Enable the path with:

```bash
export LMCACHE_DSV4_CPU_RAW_WRITE=1
export LMCACHE_DSV4_CPU_RAW_WRITE_MIBPS=512
export LMCACHE_DSV4_CPU_RAW_WRITE_BLOCK_MB=4
```

Equivalent `extra_config` keys are:

```yaml
kv_object_store_cpu_raw_write_enable: true
kv_object_store_cpu_raw_write_mibps: 512
kv_object_store_cpu_raw_write_block_mb: 4
```

The feature additionally requires the existing Tutti raw object store. If the
CPU writer cannot initialize, LMCache retains Tutti raw writes. If a CPU write
fails after partially writing a wave, LMCache disables that CPU writer and
rewrites the complete wave through Tutti before publishing metadata.

## Scheduling and correctness

The existing write planner remains the control plane. Demand readers and
read-sensitive forwards have strict priority. Tool-call and compute-without-KV
windows can admit writes immediately; otherwise idle-gap and maximum-delay
rules apply. A CPU wave does not mark Tutti's shared user queue as occupied,
so GPU-direct reads can execute concurrently through their own queue.

Metadata becomes `READY` only after every physical write in the wave returns
successfully. A partial or failed CPU write is therefore never lookup-visible.

## Initial validation

On the eight-rank 480K-context plus 8192-token workload, all ranks used distinct
PCI-resolved block devices with zero failed waves. No-save TTFT remained at
1.345 seconds median. Save TTFT was 1.388 seconds median, including gather,
pack, metadata, and data-plane overhead. Effective per-device write bandwidth
was 460-468 MiB/s with the 512 MiB/s limit.
