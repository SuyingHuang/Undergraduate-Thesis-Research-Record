# 实验运行指南

## 1. 当前冻结口径

- 正式主实现：`resource_solver='coupled'`、`old_bs_policy='legacy'`；
- 学习算法：LDA1；LDA2 是同时去除上下层时延惩罚的消融，不是 Actor-Critic；
- 对照：COB、MTD；
- 主统计窗口：所有算法统一使用后半程，成功种子不按表现删除；
- 主指标：加权完成时延惩罚、物理队列、逐 BS 平均能耗、能量虚拟队列；
- 独立重复：8 个环境种子 `42 123 456 789 1000 2003 3141 6283`；
- 当前 `budgeted-75%` 只作为机制基线，`joint_dpp` 路线不进入正式主实验。

指标定义以 [`METRIC_DEFINITION.md`](METRIC_DEFINITION.md) 为准。代码和历史 JSON
仍使用 `PAoI`/`Cost` 字段，出图与论文解释时应按“时延惩罚”命名。

## 2. 启动门槛

正式长实验只在以下条件全部满足后启动：

1. 工作树干净，远端分支能定位到同一提交；
2. `scripts/ldactl test` 全部通过；
3. 当前状态、目标和物理参数对应的参考尺度已经标定，`calibration_id` 不含
   `requires_recalibration`；
4. `scripts/ldactl smoke` 通过，manifest 中提交号正确且
   `git_worktree_dirty=false`；
5. 64/512 帧 pilot 的哈希、内存、指标有限性和吞吐检查通过，且**在将执行正式
   任务的同一台宿主机上**确认了设备分配：任务日志出现 `[DNN] device=cuda:N`，
   manifest 的 `runtime.cuda_available=true`、`runtime.cuda_health.ok=true`；
6. 正式 service 的命令、并发数、种子和输出盘空间已经复核。

短程机制实验通过不能替代这些门槛。在无法访问 GPU 设备节点的受限环境里跑 pilot，
只能验证 CPU 路径；此时门槛 5 视为未通过。

2026-09-13 的核验结论为“CPU 路径 GO，GPU 路径待补验”；证据、限制和补验命令见
[`FORMAL_READINESS_20260913.md`](FORMAL_READINESS_20260913.md)。

## 3. 统一入口

```bash
# 查看所有命令
scripts/ldactl help

# 回归测试、数值 oracle 和 smoke
scripts/ldactl test
scripts/ldactl oracle
scripts/ldactl smoke

# 目标尺度标定；正式采用固定的五个种子
scripts/ldactl calibrate --frames 200 --seeds 42,123,456,789,1024 --workers 5

# 前台运行显式指定的 sweep
scripts/ldactl sweep \
  --experiments Exp1_J Exp2_L \
  --frames 4096 \
  --seeds 42 123 456 789 1000 2003 3141 6283 \
  --workers 8

# systemd 后台正式实验
scripts/ldactl formal start
scripts/ldactl formal status
scripts/ldactl formal watch
scripts/ldactl formal logs
```

统一入口只做参数检查和转发。直接调用 `run_sweeps.py`、`collect_calibration.py`
或 `scripts/trainctl` 仍受支持。

## 4. 分阶段启动

建议先做不用于论文结论的 pilot：

```bash
scripts/ldactl sweep --experiments Exp1_J --frames 64 --seeds 42 --workers 1
scripts/ldactl sweep --experiments Exp1_J --frames 512 --seeds 42 123 --workers 2
```

pilot 后检查最新 `results/sweep/<run>/manifest.json` 和日志，至少确认：

- `git_commit` 与准备发布的提交一致；
- `git_worktree_dirty` 为 `false`；
- 所有同参数同种子的 `scenario_hash` 一致；
- 学习任务的 `[DNN] device=` 与预期设备一致，manifest 中 `runtime.cuda_health`
  为 `{available: true, ok: true}`（若本机有 GPU）；
- 没有 worker failure、NaN/Inf 或物理断言异常；
- 预计总运行时间与磁盘占用可接受。

失败任务默认重试 1 次（`--task-retries`，或用 `LDA_TASK_RETRIES` 覆盖）。重试是
为环境故障（GPU 进入错误态、设备瞬时异常）准备的；被替换的失败日志保留为
`*.attemptN`。若某任务重试后仍失败，应按 worker failure 处理，不要当作已通过。

正式运行默认由 `systemd/lda-experiments.service` 执行 Exp1–Exp7、4096 帧和 8 个
环境种子。service 文件当前显式配置 10 个 worker；改变并发前应先用 pilot 测得
吞吐和内存峰值，并同步更新 service 描述与本文档。

## 5. 历史与专项入口

| 入口 | 用途 | 是否作为当前正式入口 |
| --- | --- | --- |
| `main.py` | 单次仿真和交互式绘图 | 否 |
| `run_experiments.py` | 旧单种子四算法比较，会覆盖部分结果 | 否 |
| `run_multi_seed_experiment.py` | 默认点多种子轨迹 | 否 |
| `run_sweeps.py` | 参数扫描底层入口 | 是，由 `ldactl sweep` 转发 |
| `analysis/run_diagnostic_ablation.py` | coupled 修复诊断 | 历史/专项 |
| `analysis/run_old_bs_pipeline.py` | 旧 BS 阶段 B 流水线 | 历史复现 |
| `analysis/run_old_bs_generalization.py` | 旧任务跨负载筛选 | 历史复现 |
| `analysis/run_instability_ablations.py` | 不稳定性机制消融 | 专项 |
| `analysis/run_seed_sensitivity.py` | 策略种子敏感性 | 专项 |

对应的预注册和结果报告见 [`README.md`](README.md) 的文档导航。

## 6. 结果解释

- `fixed_half` 是主视图；`raw` 和 `cleaned` 只用于追溯与敏感性分析；
- 候选窗口缩小不等于模型已经收敛；
- BS 能量虚拟队列积压是长期约束的诊断，不是单帧能耗硬越界；
- 4096 帧仍需查看后半程分块稳定性；若不足，统一延长而不删除成功种子；
- **设备与线程设置属于实验环境**：同一提交、同一环境种子换设备不保证相同数字。
  受控 A/B 显示 COB/MTD 跨设备逐帧完全一致，而 LDA 最大差 1.54%、AC 最大差 9.36%，
  分歧在第 30–49 帧即出现（早于首次训练），机制是设备相关的 DNN 前向舍入翻转了
  近似并列的候选。因此**同一次比较内不得混用 CPU 与 GPU 运行**，图注必须标注
  设备。证据见
  [`../analysis/20260913_device_sensitivity_summary.md`](../analysis/20260913_device_sensitivity_summary.md)；
- 判定一次运行的可比性时，至少核对 manifest 的 `git_commit`、`runtime.thread_env`、
  `runtime.dnn_device_request` 与 `runtime.cuda_health`；
- 任何正式图表均应记录提交号、配置、种子、统计窗口、误差条和单位。
