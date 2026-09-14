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
当时该故障未被解释。现有证据把它与本次硬件故障连成一条链：

1. **2026-09-08/09**：4096 帧日志里 `device=cuda:1` 出现 233 次、`cuda:0` 出现
   227 次 —— 两张卡当时都健康。
2. **2026-09-10 00:33 起**：所有后续任务开始失败，且失败算法分布为
   24 AC / 24 COB / 22 LDA / 24 MTD —— **连不使用网络的 COB/MTD 一起失败**。
   失败堆栈落在 `torch.cuda.manual_seed_all`，而该调用会遍历**所有**设备，
   因此一张坏卡足以让每个触碰 CUDA 的 worker 报错。
3. **2026-09-13**：确认 `0000:3b:00.0` 已无法取得设备句柄。

「遍历所有设备的一次失败 → 全批失败」这一机制，正好解释了当时为何连纯 CPU 算法也
无法幸免；该机制已由 `set_seed(use_cuda=False)` 修复消除（见
[`20260913_cuda_containment_summary.md`](20260913_cuda_containment_summary.md)）。

需要明确的是：**这是时序上的强关联与机制吻合，不是已被证明的因果**。当时没有采集
`dmesg`/Xid 证据，本次也未做替换硬件或复位后的对照，所以不能断言 9-10 的故障百分百
由这张卡引起。

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

## 未采集的证据

以下信息没有留存，因此这条因果链只能停留在“强关联”。**上一次开机的日志仍在**
（`journalctl -b -1`），但读取需要权限，应优先补采：

- `sudo journalctl -k -b -1 | grep -iE 'nvrm|xid' | tail -40`（Xid 编号可区分掉卡、
  ECC 与非法访问）——这是最关键的一条；
- `nvidia-smi -q -i 0` 在故障期间（重启前）的完整输出；
- 故障卡在 2026-09-08/09 与 09-10 之间是否发生过自动复位。

## 更新：2026-09-14 00:27 重启

宿主机已于 2026-09-14 00:27 关机、00:35 重新开机（`journalctl --list-boots`）。

**上一次开机连续运行了 10 天**：`2026-09-04 01:59` → `2026-09-14 00:27`，完整覆盖
了 9-10 的失败与 9-13 的检测。故障出现在这段长 uptime 的中段（约第 6 天），而不是
开机时——这既符合硬件在长期运行中劣化，也符合驱动/内核状态在长 uptime 下累积异常
这两类解释，现有数据不足以区分。这也是把 `host_uptime_s` 记入 manifest 的直接原因。

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
nvidia-smi -L                     # 期望列出 GPU 0 与 GPU 1
sudo journalctl -k -b -1 | grep -iE 'nvrm|xid' | tail -40   # 故障期证据
```

两者若都正常（两张卡可见 + 昨天无 Xid 或仅有可解释的告警），应重跑一次 GPU pilot
确认 `runtime.cuda_device_count=2`，再把就绪状态改回 GO。

### 需要留意的复发风险

故障出现在长 uptime 中段，而正式运行本身要 22–30 小时。因此即使本次恢复，也应：

- 在正式启动前记录 `host_uptime_s`，避免在已运行多日的机器上开始长实验；
- 确认 `nvidia-persistenced` 已启用（保持驱动初始化状态，可减少部分掉卡类问题）；
- 运行期间用一个独立的轻量监视检查 `nvidia-smi -L` 与 Xid，而不是只在结束时看结果。

