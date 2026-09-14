# 正式实验启动就绪记录（2026-09-13）

## 结论

**当前状态：阻塞，不启动 4096 帧正式运行。** 代码、配置、复现链路与短程数值行为
都已达到启动门槛（门槛 1–11 全部通过，含宿主机 GPU pilot），但宿主机的一张
RTX 4090（`0000:3b:00.0`）处于 `Unknown Error` 状态，且与 2026-09-10 那次
失去 94/128 任务的报错同类。详见“阻塞项：GPU 硬件故障”。

该结论不表示算法已经收敛，也不预先保证所有算法满足长期能量约束。故障处理完成后
（GPU 恢复，或明确决定并记录为单卡运行），才应把状态改回 GO。

**设备敏感性结论仍需保留。** 受控 A/B 已证明 COB/MTD 跨设备逐帧一致，而 LDA/AC
会因决策翻转产生数个百分点分歧；无论最终单卡还是双卡运行，同一次比较内都不得混用
设备。

冻结提交与标签的关系：标签 `formal-baseline-20260913` 指向 `3bc2aa2`，即本记录最初
核验的提交；随后为收敛 GPU 故障半径修改了 `set_seed`、agent 设备声明和 sweep 重试
逻辑，代码基线已经前移。因此：

- 启动前应确认**本地分支与远端分支**指向同一提交且工作树为空（当前为 `63b4590`）；
- 旧标签保持不变，作为“CPU 路径首次核验”的历史标记；
- **新的正式基线标签应在硬件阻塞解除之后、正式运行之前再打**，不要在已知硬件故障
  未处理时提前冻结一个声称“已验证”的基线。

这样处理是刻意的：把标签放在 GPU 门槛之后，可以避免标签本身成为一条未经检验的
就绪声明。

## 已通过门槛

1. 指标定义已冻结为“PAoI 导向的加权完成时延惩罚”，公式和论文用语见
   [`METRIC_DEFINITION.md`](METRIC_DEFINITION.md)；算法目标未因命名调整而改变。
2. 当前配置已用固定 5 种子 × 200 帧重新标定，三项绝对贡献占比约为
   31.6% / 35.3% / 33.1%，见 [`CALIBRATION_20260913.md`](CALIBRATION_20260913.md)。
3. 完整单元测试 `106/106` 通过（原 96 项 + 本次新增 10 项 CUDA 收敛与重试回归）。
4. coupled 数值 oracle 的 40 个实例全部通过，相对可行稠密网格的最大正 gap 为 0。
5. 干净提交 smoke 的 8/8 任务成功，manifest 记录：
   - `git_commit=cf467970a330279f35b7a13c0330f4f1fafac04d`；
   - `git_worktree_dirty=false`；
   - `calibration_id=global_state_v3_coupled_legacy_5x200_20260913`；
   - 结果目录：`results/sweep/20260913_103304_293127_Smoke_K`。
6. Exp1_J 的 64 帧单种子 pilot 为 24/24 成功，结果目录：
   `results/sweep/20260913_103417_387843_Exp1_J`。
7. Exp1_J 的 512 帧双种子 pilot 为 48/48 成功，manifest 保持相同干净提交和
   标定 ID，结果目录：`results/sweep/20260913_103956_445823_Exp1_J`。
8. **修改 `set_seed`、agent 设备声明与 sweep 重试逻辑后重新核验（提交
   `c408cee`）**：
   - 64 帧单种子 pilot 24/24 成功，`git_worktree_dirty=false`，结果目录
     `results/sweep/20260913_120328_873292_Exp1_J`；
   - 512 帧双种子 pilot 48/48 成功，12 个场景哈希一致，manifest 记录
     `task_retries=1` 且未触发任何重试，结果目录
     `results/sweep/20260913_120859_367608_Exp1_J`；
   - 任务日志确认设备收敛生效：LDA/AC 为 `CUDA: enabled`，COB/MTD 为
     `CUDA: not used (CPU-only agent)`。
9. 512 帧 pilot 中 LDA 的 12/12 个运行均未越过 1800 J 的 BS 能量虚拟队列
   诊断阈值；最大末值约 672 J。COB 的 12/12 个运行也未触发。
10. **宿主机 GPU 门禁（提交 `6cd162f`）**：
    - `Exp1_J` 512 帧双种子 pilot：48/48 成功，12 个场景哈希一致，未触发重试；
      `runtime.cuda_available=true`、`cuda_health={ok: true}`，48 个学习任务全部
      记录 `device=cuda:0`，COB/MTD 记录 `CUDA: not used (CPU-only agent)`；
      结果目录 `results/sweep/20260913_220855_456859_Exp1_J`；
    - `Exp6_Bc` 512 帧双种子 pilot：32/32 成功，未触发重试，结果目录
      `results/sweep/20260913_224608_730350_Exp6_Bc`。该组正是 2026-09-10 失去
      94/128 任务的那一组，本次在 GPU 上完整通过；
    - 两臂的已知信号与 CPU pilot 一致：`Exp1_J` 中 AC 9/12、MTD 2/12、LDA 与 COB
      0/12 触发能量队列告警。
11. **设备敏感性受控 A/B**：同一提交、同一入口、同一线程环境下仅改变
    `LDA_DEVICE`，两臂各 48/48 成功。COB/MTD 逐帧完全一致；LDA 最大差 1.54%、
    AC 最大差 9.36%，分歧首发于第 30–49 帧（早于第 260 帧首次训练）。完整证据见
    [`../analysis/20260913_device_sensitivity_summary.md`](../analysis/20260913_device_sensitivity_summary.md)。

## 已知信号与解释

- AC 有 10/12 个运行触发高能量队列提示，最大末值约 9139.6 J；MTD 有 2/12
  个运行触发，均在 `J=4`，最大末值约 12991.6 J。这些是对照/消融的约束可行性
  结果，不是 worker failure。正式分析必须先报告可行性，再比较时延惩罚。
- LDA 的 12 个运行中，时延惩罚后半程四分块相对范围都不超过 10%。由于 512 帧
  只是 pilot，物理队列和能量队列仅有 3/12 个运行同时满足严格 10% 分块范围；
  因此正式 4096 帧仍必须检查稳定性，必要时统一延长到 8192 帧，不能删除成功
  但收敛较慢的种子。
- 本机内核注册 **2 × NVIDIA GeForce RTX 4090**（驱动 535.230.02，CUDA 12.2），但其中
  `0000:3b:00.0`（device minor 0，UUID `GPU-33f5f413-57a2-5cf8-5aa0-7d60d5ee907a`）
  **已故障**。2026-09-13 在宿主机执行 `nvidia-smi -L` 得到：

  ```text
  CUDA_VISIBLE_DEVICES='<unset>'
  Unable to determine the device handle for gpu 0000:3B:00.0: Unknown Error
  GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-9bd37703-4d1f-1565-6e12-6630229223e5)
  ```

  `CUDA_VISIBLE_DEVICES` 未设置，所以这不是环境变量限制，而是设备/驱动故障；CUDA 因此
  只枚举出 1 张卡（`runtime.cuda_device_count=1`）。第 5–8 条 pilot 的
  `cuda_available=false`、`[DNN] device=cpu` 是另一回事（受限沙箱看不到设备节点）。
  这构成正式运行的阻塞项，见下节。
- 64/512 帧实测显示 `J=14` 的 LDA 是 Exp1 的明显长尾；正式排期应按高 `J`
  LDA 估算，而不是按总任务数线性平均。
- **设备是实验环境的一部分。** 同一提交、同一环境种子在不同设备上不保证相同数字：
  受控 A/B 显示 COB/MTD 逐帧完全一致，而 LDA 最大差 1.54%、AC 最大差 9.36%。
  分歧在第 30–49 帧就出现，早于第 260 帧的首次训练更新，机制是设备相关的 DNN
  前向舍入翻转了近似并列的候选选择。完整证据见
  [`../analysis/20260913_device_sensitivity_summary.md`](../analysis/20260913_device_sensitivity_summary.md)。
  因此：**同一次比较内禁止混用设备**，且论文与图注必须记录设备。

## 阻塞项：GPU 硬件故障

`0000:3b:00.0` 处于 `Unknown Error` 状态，**在与 2026-09-10 故障同类的报错文本下**。
两者指向同一条因果链：

1. 2026-09-08/09 的 4096 帧日志中 `cuda:0` 与 `cuda:1` 都出现过（233 / 227 次），
   说明当时两张卡都健康；
2. 2026-09-10 的 `Exp6_Bc` 出现 `RuntimeError: CUDA error: unknown error`，失去
   94/128 任务，并伴随 8 个 MTD 运行同时停摆 22.9 小时。失败堆栈落在
   `torch.cuda.manual_seed_all`，而该调用会遍历**所有**设备——一张坏卡足以让每个
   触碰 CUDA 的 worker 报错，这与“连 COB/MTD 一起失败”的分布吻合；
3. 2026-09-13 确认 `0000:3b:00.0` 已无法取得设备句柄。

因此第 9–10 条 pilot 之所以通过，很可能只是因为它们落在**幸存的那张卡**上
（CUDA 只枚举出 1 张，`cuda:0` 即幸存卡），并未证明故障已消失。这是把状态从
“GPU 门禁通过”下调为“阻塞”的理由。

### 处理顺序

1. 先确认故障性质与是否持久：

   ```bash
   nvidia-smi -q -i 0 | head -40
   sudo dmesg | grep -iE 'nvrm|xid|nvidia' | tail -40
   ```

   重点关注 `Xid`：79 通常表示掉卡，48/63 表示 ECC/受限错误，13/31 表示非法访问。

2. 尝试恢复：先 `sudo nvidia-smi --gpu-reset -i 0`（需无进程占用，掉卡时通常失败），
   失败则**重启宿主机**——这是清除 `Unknown Error` 最可靠的手段。

3. 重启后重新执行 `nvidia-smi -L`：若两张卡恢复，重跑下面的 GPU 门禁 pilot 并确认
   `runtime.cuda_device_count=2`；若故障依旧，应视为硬件问题（保修/RMA），并明确
   决定是否接受单卡运行。

4. **在故障处理完成前不启动 4096 帧正式运行。** 单卡本身是可用的（第 9–10 条已证明），
   但一次 22–30 小时的运行不能建立在“第二张卡正在报 Unknown Error”的机器上：驱动级
   故障会同时打断所有 LDA/AC worker，`--task-retries` 只能覆盖一次瞬时抖动。

### 仍需补验

- **`Exp7_Bsat`** 自 2026-09-10 批次起未在当前代码上运行过；建议在 GPU 恢复后补一次
  512 帧 pilot。
- 若最终确定只能单卡运行，应在论文的威胁有效性一节披露：设备分配退化为单卡、
  且训练吞吐低于双卡配置。

## 启动命令

在具有用户级 systemd bus 的宿主机 shell 中，从仓库根目录执行：

```bash
git switch codex/linux-server
git pull --ff-only origin codex/linux-server
git status --short
scripts/ldactl formal start
scripts/ldactl formal status
```

`git status --short` 必须无输出。当前 service 会执行 Exp1–Exp7、4096 帧、8 个固定
环境种子和 10 个 worker。若 service 尚未安装，先把
`systemd/lda-experiments.service` 链接到用户 systemd 配置并执行
`systemctl --user daemon-reload`。

本次受限执行环境无法连接用户级 systemd bus，且会回收脱离会话的普通后台进程，
因此没有在这里伪装成“已启动”；正式任务应由上述宿主机 service 托管，才能保证
SSH 或当前会话结束后继续运行。
