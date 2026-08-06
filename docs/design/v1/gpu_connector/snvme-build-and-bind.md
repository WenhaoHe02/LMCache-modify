# SNVMe 编译和绑定步骤

下面命令均在目标服务器执行。`.o` 是中间文件；真正加载的是
`snvme-core.ko` 和 `snvme.ko`。

## 1. 检查环境

```bash
cd /path/to/LMCache/Tutti

uname -r
test -d "/lib/modules/$(uname -r)/build"
find /usr/src -path '*/nv-p2p.h' -print
nvidia-smi
```

首次部署可先安装 Tutti 的构建依赖：

```bash
bash scripts/prepare_env.sh
```

## 2. 编译 `.o` 和 `.ko`

```bash
cd /path/to/LMCache/Tutti

# 自动按 uname -r 选择 snvme baseline。
cmake -S . -B build
cmake --build build --target modules --parallel "$(nproc)"
```

自动匹配失败时显式指定 baseline：

```bash
cmake -S . -B build \
  -DSNVME_KERNEL_VERSION=5.15.0-public
cmake --build build --target modules --parallel "$(nproc)"
```

检查产物：

```bash
find build/module -maxdepth 1 \
  -type f \( -name '*.o' -o -name '*.ko' \) -printf '%f\n' | sort
modinfo build/module/snvme-core.ko
modinfo build/module/snvme.ko
```

## 3. 加载模块

```bash
cd /path/to/LMCache/Tutti
cmake --build build --target insmod
```

等价于：

```bash
sudo insmod build/module/snvme-core.ko
sudo insmod build/module/snvme.ko
```

验证：

```bash
lsmod | grep -E '^(snvme|snvme_core)\b'
ls -l /dev/snvm_control
dmesg -T | tail -n 100
```

此时只是加载驱动，还没有把 SSD 绑定给 `snvme`。

## 4. 确认目标 SSD

```bash
lspci -Dnnk -d ::0108
lsblk -o NAME,PATH,SIZE,FSTYPE,MOUNTPOINTS
findmnt /mnt/nvme0
```

设置目标 BDF 和挂载点，并确认二者属于同一块盘：

```bash
PCI_BDF=0000:4b:00.0
CACHE_MOUNT=/mnt/nvme0
```

## 5. bind 前准备

LMCache 必须先完成 cold-store 落盘、`scan_existing_entries()` 和 FIEMAP
预扫描。FIEMAP 必须在 unmount/bind 之前完成。

停止自动 remount 服务并同步卸载：

```bash
sudo systemctl stop lmcache-remount.service 2>/dev/null || true
sudo umount "$CACHE_MOUNT"

if findmnt -nr -M "$CACHE_MOUNT" | grep -q .; then
  echo "mount is still active: $CACHE_MOUNT" >&2
  exit 1
fi
```

不能使用 `umount -l`。

TP8/CP8 容器应直接使用 `scripts/run_container_cp8_ab.sh`。该脚本会等待
8 个 `mnt_nvme*.ready`，同步卸载 host mounts，最后创建 `host.ready`。
`SNVM_DEVICE_BIND` 在 barrier 完成前会被拒绝。

## 6. 执行 bind

正常 LMCache 路径不手工调用 ioctl。创建 `TuttiDirectLoader` 时，
`SnvmeSession._setup()` 自动执行：

```text
open(/dev/snvm_control)
-> SNVM_CHRDEV_CREATE(PCI_BDF)
-> open(/dev/ssnvme<N>)
-> NVM_SET_KERNEL_IOQ_CAP
-> 检查 rank/host handoff barrier
-> SNVM_DEVICE_BIND(PCI_BDF)
-> NVM_GET_DEV_INFO
-> 映射 BAR0/HBM/SQ/CQ
-> NVM_ADD_USER_QUEUE
```

容器至少需要：

```bash
docker run ... \
  --privileged \
  --cap-add SYS_ADMIN \
  --cap-add SYS_RAWIO \
  --device /dev/snvm_control:/dev/snvm_control \
  -v /sys:/sys \
  -v /tmp:/tmp \
  ...
```

已存在的 `/dev/ssnvme<N>` 也要通过 `--device` 传入。

## 7. 验证

```bash
lspci -Dnnk -s "$PCI_BDF"
ls -l /dev/snvm_control /dev/ssnvme*
dmesg -T | grep -E 'snvme|queue squeeze|user QID|NVM_ADD_USER_QUEUE'
```

`lspci` 应显示：

```text
Kernel driver in use: snvme
```

LMCache 日志应出现：

```text
NVM_GET_DEV_INFO: ...
BAR0 mmap: ...
cudaHostRegister BAR0: ...
```

出现这些日志且 user queue 创建成功后，GPU Direct I/O 才可用。

## 8. 回退到内核 `nvme`

先停止 LMCache/Tutti，并确认没有 fd holder：

```bash
sudo lsof /dev/snvm_control /dev/ssnvme* 2>/dev/null || true
```

解绑并卸载模块：

```bash
cd /path/to/LMCache/Tutti
sudo scripts/reset_snvme.sh --no-insmod
```

确认设备回到内核 `nvme`：

```bash
lspci -Dnnk -s "$PCI_BDF"
lsblk
```

若未自动回到 `nvme`：

```bash
sudo scripts/bind_nvme_device.sh "$PCI_BDF"
sudo mount "$CACHE_MOUNT"
findmnt "$CACHE_MOUNT"
```

如果出现 duplicate sysfs filename，不要反复 `insmod`，应停止操作并重启。

## 代码位置

- 构建：`Tutti/CMakeLists.txt`
- Kbuild：`Tutti/backends/local/kernel_modules/snvme-*/Makefile.in`
- bind：`lmcache/v1/gpu_connector/tutti_direct_loader.py`
- mount handoff：`lmcache/v1/cache_engine.py`
- TP8/CP8 barrier：`scripts/run_container_cp8_ab.sh`
- 回退：`Tutti/scripts/reset_snvme.sh`
