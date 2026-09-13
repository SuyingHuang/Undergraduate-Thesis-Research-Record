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
5. 64/512 帧 pilot 的哈希、设备、内存、指标有限性和吞吐检查通过；
6. 正式 service 的命令、并发数、种子和输出盘空间已经复核。

短程机制实验通过不能替代这些门槛。

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
- 没有 worker failure、NaN/Inf 或物理断言异常；
- 预计总运行时间与磁盘占用可接受。

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
- 任何正式图表均应记录提交号、配置、种子、统计窗口、误差条和单位。
