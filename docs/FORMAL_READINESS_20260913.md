# 正式实验启动就绪记录（2026-09-13）

## 结论

当前 `coupled + legacy` 主实现可以启动新的 8 种子 × 4096 帧正式 sweep。
该结论表示代码、配置、复现链路和短程数值行为达到启动门槛，不表示算法已经收敛，
也不预先保证所有算法满足长期能量约束。

**该结论的证据全部来自 CPU 路径。** 就绪核验所处的受限沙箱看不到 NVIDIA 设备
节点，所以 GPU 设备分配与 GPU 故障恢复没有经过验证；正式启动前必须补做
“尚未覆盖的路径”一节中的宿主机 pilot，否则不能把状态记为完整 GO。

冻结提交与标签的关系需要重新确认，不能沿用旧口径：标签
`formal-baseline-20260913` 指向 `3bc2aa2`，即本记录最初核验的提交；随后为收敛
GPU 故障半径修改了 `set_seed`、agent 设备声明和 sweep 重试逻辑，代码基线已经前移。
因此：

- 启动前应确认**本地分支与远端分支**指向同一提交且工作树为空（当前为
  `c408cee`，见下方门槛记录）；
- 旧标签保持不变，作为“CPU 路径首次核验”的历史标记；
- **新的正式基线标签应在宿主机 GPU pilot 通过之后再打**，不要在 GPU 路径未验证
  时提前冻结一个声称“已验证”的基线。

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
- 本机有 **2 × NVIDIA GeForce RTX 4090**（驱动 535.230.02，CUDA 12.2）。注意：第 5–8 条
  pilot 是在一个无法访问 NVIDIA 设备节点的受限沙箱中完成的，其
  `cuda_available=false`、`[DNN] device=cpu` 是沙箱假象；宿主机 GPU 路径已由第 9–10
  条单独核验。但 CUDA 目前只报告 **1** 张设备（宿主内核注册 2 张），原因未确认，
  见“尚未覆盖的路径”第 3 条。
- 64/512 帧实测显示 `J=14` 的 LDA 是 Exp1 的明显长尾；正式排期应按高 `J`
  LDA 估算，而不是按总任务数线性平均。
- **设备是实验环境的一部分。** 同一提交、同一环境种子在不同设备上不保证相同数字：
  受控 A/B 显示 COB/MTD 逐帧完全一致，而 LDA 最大差 1.54%、AC 最大差 9.36%。
  分歧在第 30–49 帧就出现，早于第 260 帧的首次训练更新，机制是设备相关的 DNN
  前向舍入翻转了近似并列的候选选择。完整证据见
  [`../analysis/20260913_device_sensitivity_summary.md`](../analysis/20260913_device_sensitivity_summary.md)。
  因此：**同一次比较内禁止混用设备**，且论文与图注必须记录设备。

## 尚未覆盖的路径

宿主机 GPU 门禁已于 `6cd162f` 通过（见第 9–10 条），但以下三点仍未覆盖：

1. **`Exp7_Bsat`** 自 2026-09-10 批次起就没有在当前代码上运行过；正式运行前应补
   一次 512 帧 pilot。
2. **GPU 故障模式**：2026-09-10 的 `Exp6_Bc` 曾因
   `RuntimeError: CUDA error: unknown error` 失去 94/128 任务，并伴随 8 个 MTD 运行
   同时停摆 22.9 小时。`Exp6_Bc` 已在 `6cd162f` 上以 32/32 通过，说明该故障没有在
   本次复现，但这**不等于根因已排除**——它更可能是一次设备/驱动瞬态故障。
3. **第二张 GPU 的去向**：宿主内核注册 2 张 4090，CUDA 只报告 1 张
   （`runtime.cuda_device_count=1`）。2026-09-08/09 的日志两张卡都用过，说明这是
   后来发生的变化。若第二张卡处于异常态，它很可能就是第 2 条故障的根因。启动前应
   确认：

   ```bash
   echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}'"
   ls -l /dev/nvidia*; nvidia-smi -L
   sudo dmesg | grep -iE 'nvrm|xid' | tail -20
   ```

   若只是环境变量限制则无害；若是设备掉线，必须先处理硬件/驱动。

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
