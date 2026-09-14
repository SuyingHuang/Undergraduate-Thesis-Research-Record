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

以下信息当时没有留存，应在处理时补采，否则这条因果链只能停留在“强关联”：

- `sudo dmesg | grep -iE 'nvrm|xid'` 的完整输出（Xid 编号可区分掉卡、ECC 与非法访问）；
- `nvidia-smi -q -i 0` 的完整输出；
- 故障卡在 2026-09-08/09 与 09-10 之间是否发生过自动复位。
