# 正式实验启动就绪记录（2026-09-13）

## 结论

当前 `coupled + legacy` 主实现可以启动新的 8 种子 × 4096 帧正式 sweep。
该结论表示代码、配置、复现链路和短程数值行为达到启动门槛，不表示算法已经收敛，
也不预先保证所有算法满足长期能量约束。

冻结提交在本记录提交后由 Git 标签 `formal-baseline-20260913` 标识；启动前应确认
本地分支、远端分支和该标签指向同一提交，且工作树为空。

## 已通过门槛

1. 指标定义已冻结为“PAoI 导向的加权完成时延惩罚”，公式和论文用语见
   [`METRIC_DEFINITION.md`](METRIC_DEFINITION.md)；算法目标未因命名调整而改变。
2. 当前配置已用固定 5 种子 × 200 帧重新标定，三项绝对贡献占比约为
   31.6% / 35.3% / 33.1%，见 [`CALIBRATION_20260913.md`](CALIBRATION_20260913.md)。
3. 完整单元测试 `96/96` 通过。
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
8. 512 帧 pilot 中 LDA 的 12/12 个运行均未越过 1800 J 的 BS 能量虚拟队列
   诊断阈值；最大末值约 672 J。COB 的 12/12 个运行也未触发。

## 已知信号与解释

- AC 有 10/12 个运行触发高能量队列提示，最大末值约 9139.6 J；MTD 有 2/12
  个运行触发，均在 `J=4`，最大末值约 12991.6 J。这些是对照/消融的约束可行性
  结果，不是 worker failure。正式分析必须先报告可行性，再比较时延惩罚。
- LDA 的 12 个运行中，时延惩罚后半程四分块相对范围都不超过 10%。由于 512 帧
  只是 pilot，物理队列和能量队列仅有 3/12 个运行同时满足严格 10% 分块范围；
  因此正式 4096 帧仍必须检查稳定性，必要时统一延长到 8192 帧，不能删除成功
  但收敛较慢的种子。
- 当前机器没有可用 CUDA/NVML，正式运行会回退 CPU。候选搜索本来主要受 CPU
  限制，但 DNN 训练部分不会获得 GPU 加速。
- 64/512 帧实测显示 `J=14` 的 LDA 是 Exp1 的明显长尾；正式排期应按高 `J`
  LDA 估算，而不是按总任务数线性平均。

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
