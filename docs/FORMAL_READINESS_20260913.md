# 正式实验启动就绪记录（2026-09-13）

## 结论

当前 `coupled + legacy` 主实现可以启动新的 8 种子 × 4096 帧正式 sweep。
该结论表示代码、配置、复现链路和短程数值行为达到启动门槛，不表示算法已经收敛，
也不预先保证所有算法满足长期能量约束。

**该结论的证据全部来自 CPU 路径。** 就绪核验所处的受限沙箱看不到 NVIDIA 设备
节点，所以 GPU 设备分配与 GPU 故障恢复没有经过验证；正式启动前必须补做
“尚未覆盖的路径”一节中的宿主机 pilot，否则不能把状态记为完整 GO。

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
- 本机有 **2 × NVIDIA GeForce RTX 4090**（驱动 535.230.02，CUDA 12.2），正式
  运行会按 `LDA_DEVICE=auto` 使用 `cuda:0`/`cuda:1`。注意：本次就绪核验是在一个
  无法访问 NVIDIA 设备节点的受限沙箱中完成的，因此第 5–8 条 pilot 的
  `cuda_available=false`、`[DNN] device=cpu` 是沙箱假象，**不代表机器能力，也不
  构成对 GPU 路径的验证**。GPU 设备分配与故障恢复必须在正式启动前用一次宿主机
  pilot 单独确认，见第 4 节。
- 64/512 帧实测显示 `J=14` 的 LDA 是 Exp1 的明显长尾；正式排期应按高 `J`
  LDA 估算，而不是按总任务数线性平均。

## 尚未覆盖的路径（启动前必须补验）

本次核验全部在无 GPU 的受限沙箱中完成，因此以下三条**没有被验证**，不能作为
“已就绪”的一部分：

1. GPU 设备分配：`cuda:0`/`cuda:1` 的 worker 轮转；
2. GPU 长期稳定性：2026-09-10 的 `Exp6_Bc` 批次曾出现 94/128 任务因
   `RuntimeError: CUDA error: unknown error` 失败，并伴随 8 个 MTD 运行同时停摆
   22.9 小时；`Exp7_Bsat` 在该批次整体缺失。该故障模式尚未在当前提交上复现或排除。
3. 该故障的根因之一是 `utils/reproducibility.py::set_seed` 对**每个** worker 都调用
   `torch.manual_seed`（内部转调 `torch.cuda.manual_seed_all`），使 COB/MTD 也创建
   CUDA 上下文，从而把故障半径扩大到整批任务。已改为按 `uses_dnn` 区分，并新增
   `--task-retries` 与 manifest 中的 `runtime.cuda_health`；但该修复本身同样只在
   CPU 沙箱中测试过。

补验命令（在宿主机 shell，非沙箱）：

```bash
PYTHON_BIN=/home/hp/miniconda3/envs/sagin/bin/python \
  scripts/ldactl sweep --experiments Exp1_J Exp6_Bc \
  --frames 512 --seeds 42 123 --workers 10
```

通过标准：任务日志出现 `[DNN] device=cuda:0`/`cuda:1`，manifest 的
`runtime.cuda_available=true` 且 `runtime.cuda_health.ok=true`，且无 worker failure。

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
