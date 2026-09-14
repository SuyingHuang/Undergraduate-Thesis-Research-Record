# GPU 硬件故障事件（2026-09-13 发现）

## 事件

宿主机的一张 RTX 4090 处于不可用状态，正式实验因此阻塞。

```text
$ nvidia-smi -L
CUDA_VISIBLE_DEVICES='<unset>'
Unable to determine the device handle for gpu 0000:3B:00.0: Unknown Error
GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-9bd37703-4d1f-1565-6e12-6630229223e5)
```

| PCI 地址 | device minor | UUID | 状态 |
| --- | ---: | --- | --- |
| `0000:3b:00.0` | 0 | `GPU-33f5f413-57a2-5cf8-5aa0-7d60d5ee907a` | 故障（`Unknown Error`） |
| `0000:86:00.0` | 1 | `GPU-9bd37703-4d1f-1565-6e12-6630229223e5` | 可用 |

`CUDA_VISIBLE_DEVICES` **未设置**，因此这不是环境变量限制。内核仍注册两张卡
（`/proc/driver/nvidia/gpus/` 有两个条目），但 CUDA 只枚举出 1 张
（`runtime.cuda_device_count=1`）。

## 与 2026-09-10 故障的关系

2026-09-10 的 `Exp6_Bc` 批次曾失去 94/128 个任务，报错为
`RuntimeError: CUDA error: unknown error`，并伴随 8 个 MTD 运行同时停摆 22.9 小时。
当时该故障未被解释，现已由上方“已确认的根因”给出的 Xid 79 定案。完整链条为：

1. **2026-09-08/09**：4096 帧日志里 `device=cuda:1` 出现 233 次、`cuda:0` 出现
   227 次 —— 两张卡当时都健康。
2. **2026-09-10 10:47:46**：`0000:3b:00.0` 报 Xid 79 掉卡（内核日志直接证据）。
3. **2026-09-11 00:33 起**：后续任务开始失败，失败算法分布为
   24 AC / 24 COB / 22 LDA / 24 MTD —— **连不使用网络的 COB/MTD 一起失败**。
   失败堆栈落在 `torch.cuda.manual_seed_all`，而该调用会遍历**所有**设备，
   因此一张掉卡的设备足以让每个触碰 CUDA 的 worker 报错。掉卡与失败之间约 14 小时
   的间隔，与“任务按组推进、下一批 worker 才触碰 CUDA”的调度方式吻合。
4. **2026-09-13**：确认 `0000:3b:00.0` 已无法取得设备句柄。

「遍历所有设备的一次失败 → 全批失败」这一机制，正好解释了当时为何连纯 CPU 算法也
无法幸免；该机制已由 `set_seed(use_cuda=False)` 修复消除（见
[`20260913_cuda_containment_summary.md`](20260913_cuda_containment_summary.md)）。

注意边界：Xid 79 **解释了 9-10 的失败**，但没有解释为什么这张卡会在第 21 天掉卡；
供电、接头、转接线与板卡自身故障都可能，需按下方清单排查。

## 为什么 2026-09-13 的 GPU pilot 仍然通过了

`Exp1_J`（48/48）与 `Exp6_Bc`（32/32）都在宿主机 GPU 上完成、零失败、零重试。但这
**不能**说明故障已消失：CUDA 只枚举出 1 张卡，pilot 记录的 `cuda:0` 就是那张**幸存
卡**，故障卡从未被使用，也就不可能被这些 pilot 暴露。

这是本次把就绪状态从“GPU 门禁通过”下调为“阻塞”的直接理由。

## 影响与处理

- **影响**：正式 4096 帧运行（约 22–30 小时、10 worker）不应建立在一台正在报
  `Unknown Error` 的机器上。单卡本身可用，但驱动级故障会同时打断所有 LDA/AC
  worker，`--task-retries` 只能覆盖一次瞬时抖动。
- **处理顺序**：见
  [`../docs/FORMAL_READINESS_20260913.md`](../docs/FORMAL_READINESS_20260913.md)
  的“阻塞项：GPU 硬件故障”一节（先取证 `nvidia-smi -q -i 0` 与 `dmesg | grep Xid`，
  再尝试 `--gpu-reset`，失败则重启宿主机；重启后重跑 GPU pilot 并确认
  `cuda_device_count=2`）。
- **若最终只能单卡运行**：应在论文威胁有效性一节披露设备分配退化为单卡，以及训练
  吞吐低于双卡配置。

## 已确认的根因：Xid 79（2026-09-10 10:47:46）

上一开机的内核日志保留了决定性证据：

```text
Aug 20 01:33:31 1013 kernel: NVRM: loading NVIDIA UNIX x86_64 Kernel Module  535.230.02
Sep 10 10:47:46 1013 kernel: NVRM: GPU at PCI:0000:3b:00: GPU-33f5f413-57a2-5cf8-5aa0-7d60d5ee907a
Sep 10 10:47:46 1013 kernel: NVRM: Xid (PCI:0000:3b:00): 79, pid='<unknown>', name=<unknown>, GPU has fallen off the bus.
Sep 10 10:47:46 1013 kernel: NVRM: GPU 0000:3b:00.0: GPU has fallen off the bus.
```

**Xid 79 = “GPU has fallen off the bus”**，发生在 `0000:3b:00.0`，即 9-13 报
`Unknown Error` 的同一张卡。这三点由此确定：

1. 故障是**硬件/供电/PCIe 层面的掉卡**，不是驱动或用户态状态残留；
2. 掉卡时刻固定在 `2026-09-10 10:47:46`，**早于**任务开始大面积失败
   （失败日志首现于 `2026-09-11 00:33`）约 14 小时。这解释了为什么当时只看到
   “MTD 停摆 22.9 小时 + 后续整批失败”，而没有一个即时的错误时间点；
3. 因此 9-10 批量失败与本次 `Unknown Error` 是同一个硬件事件的两种表现，
   先前的“强关联”升级为**已确认的因果**。

**uptime 口径需要更正**：`journalctl --list-boots` 不加 `sudo` 时显示上一开机始于
`2026-09-04 01:59`，但那是受限可见范围；带 `sudo` 的 `-b -1` 显示 NVRM 实际加载于
`Aug 20 01:33:31`。也就是说上一开机连续运行约 **25 天**，掉卡发生在其第 **21 天**。

## 更新：2026-09-14 00:35 重启后

宿主机已于 2026-09-14 00:27 关机、00:35 重新开机。

重启后本次开机的内核日志显示**两张卡都干净地完成初始化，且没有任何 Xid、NVRM 错误
或 PCIe AER 错误**：

```text
nvidia 0000:3b:00.0: enabling device (0140 -> 0143)
nvidia 0000:86:00.0: enabling device (0140 -> 0143)
[drm] Initialized nvidia-drm 0.0.0 20160202 for 0000:3b:00.0 on minor 1
[drm] Initialized nvidia-drm 0.0.0 20160202 for 0000:86:00.0 on minor 2
nvidia-uvm: Loaded the UVM driver, major device number 509.
```

（`acpi ... _OSC: platform does not support [... AER ...]` 与 `ata*: SATA link down`
是常规能力协商和空 SATA 口，不是错误。）

**但重启并不能自行证明故障已消除**：`nvidia-smi -L` 与 `torch.cuda.device_count()`
必须重新确认。在确认之前，就绪状态仍保持阻塞。

### 待确认

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,pcie.link.gen.current,pcie.link.width.current \
  --format=csv
```

两项都要看：①两张卡是否都列出；②**CUDA 索引与 PCI 地址的对应关系**。

### 关键陷阱：重启后 `cuda:0` 很可能又是那张掉过卡的卡

重启后两张卡都回来了，而 CUDA 通常按 PCI 顺序枚举，即
`cuda:0 = 0000:3b:00.0`（掉过卡）、`cuda:1 = 0000:86:00.0`（从未出问题）。
如果直接按默认 `LDA_DEVICE=auto` 启动正式运行，`_assign_worker_dnn_device` 会把
worker 轮转分配到**那张刚掉过 bus 的卡**上。

因此正式运行前应显式按 **UUID** 固定设备（UUID 不受索引重排影响）：

```bash
CUDA_VISIBLE_DEVICES=GPU-9bd37703-4d1f-1565-6e12-6630229223e5 \
PYTHON_BIN=/home/hp/miniconda3/envs/sagin/bin/python \
  scripts/ldactl sweep --experiments Exp7_Bsat --frames 512 --seeds 42 123 --workers 10
```

这样把整轮运行锁在从未故障的 `0000:86:00.0` 上。service 文件也需要同步加入该环境
变量，否则 `formal start` 会绕过这个限制。

### 复发风险

Xid 79 属于硬件/供电/PCIe 类别，重启能让卡回来，但**不降低复发概率**。在长跑前建议
按以下顺序排查物理原因：

1. 重新插拔显卡与 PCIe 供电（4090 的 12VHPWR 接头未完全到位是 Xid 79 的常见诱因，
   也需检查接头有无过热/变色痕迹）；
2. 确认电源余量（两张 4090 的瞬态尖峰可远超额定均值）；
3. 若用了 PCIe 转接线/延长线，检查接触与线材规格；
4. 检查机箱风道与进风口温度；
5. 以上都排除后仍复发，按硬件故障走保修。

运行期间还应：

- 在正式启动前记录 `host_uptime_s`，避免在已运行多日的机器上开始长实验；
- 确认 `nvidia-persistenced` 已启用；
- 用一个独立监视持续检查 Xid，而不是只在 22–30 小时后看结果——宿主 `dmesg_restrict=0`，
  受限沙箱里也能直接读到内核日志，因此可以做到运行期告警。

