# 旧 BS 泛化实验：阶段 B 暂停记录

## 状态

阶段 B 按冻结方案启动后，应用户要求在第一个完整运行写盘时安全停止。

- 结果目录：`results/old_bs_generalization/20260911_101254_224191/`；
- 计划运行：60；完整运行：1；未完成：59；
- 主运行器与所有 worker 已停止；
- 已完成运行的 JSON、NPZ 和日志均完整保留；
- 其余在途运行只有日志头，没有结果 JSON/NPZ，恢复时会重新计算；
- `manifest.json` 和 `source.zip` 已保留，恢复会校验参数及源码哈希。

冻结设计见
[`old_bs_generalization_phase_b_preregistration.md`](old_bs_generalization_phase_b_preregistration.md)。

## 已完成样本

样本：L=10 Mbit、`budgeted-75%`、环境种子 196613、策略种子 456。

| 指标 | 数值 | 冻结判定 |
|---|---:|---|
| PAoI 代理 | 2.221924 | 仅记录，非约束 |
| 队列均值 | 0.037336 Mbit/用户 | 仅记录 |
| BS 总能耗 | 48.810515 J/节点 | 通过 |
| 旧/新任务能耗 | 3.076172 / 45.734343 J/节点 | 仅记录 |
| 最大能量队列斜率 | 0.0004325 J/帧 | 通过（阈值 0.01） |
| 物理队列斜率 | +0.000015115 Mbit/(用户·帧) | **严格失败**（要求 \(\le0\)） |
| 总筛选 | `False` | **失败** |

这个正斜率很小，但不能在看过确认数据后增加容差。依据冻结决策规则，阶段 B
已经出现一次严格失败；即使以后补齐其余运行，也不能再声称“30/30 全部通过”，
也不能直接进入 4096 帧确认。补齐剩余样本仍有价值：它可以判断这是孤立的
近零边界波动，还是可重复的容量不足，并为动态联合策略提供适用域证据。

## 恢复命令

下次继续时使用原目录恢复，不创建新的实验批次：

```bash
python analysis/run_old_bs_generalization.py \
  --frames 2048 \
  --loads-mbit 10 12 16 \
  --task-std-mbit 3 \
  --users-per-bs 10 \
  --scenario-seeds 155921 196613 238919 275015 314159 \
  --policy-seeds 456 789 \
  --budget-fractions 0.75 \
  --workers 8 --device cpu \
  --resume results/old_bs_generalization/20260911_101254_224191
```

恢复前不得修改 manifest 记录的 Python 源文件；若源码确需改变，应保留当前目录，
另开新实验并明确标记为不同版本。

## 自动流水线

现已增加不修改上述 manifest 源文件的外层控制器。它会自动补齐 2048 帧阶段、
审计全部产物并执行冻结门槛；仅当 30 个 budgeted 运行全部通过时才启动 4096 帧。
当前已完成样本严格失败，因此补齐短阶段后应自动停在
`stopped_strict_gate_failed_2048`，不会误启长阶段。

```bash
python analysis/run_old_bs_pipeline.py \
  --control-dir results/old_bs_pipeline/phase_b_current \
  --resume-short results/old_bs_generalization/20260911_101254_224191 \
  --workers 8 --device cpu
```

同一命令可在中断后恢复；若只想重新生成状态与 Markdown 报告而不运行仿真，追加
`--evaluate-only`。流水线在操作系统进程内完成等待和分支判断，不要求 Codex
持续轮询。
