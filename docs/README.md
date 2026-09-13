# 文档导航

本页是研究仓库的文档总入口。历史脚本和报告继续保留以保证复现，但新实验应从
“当前规范”开始，不应从日期最早或文件名最像入口的脚本猜测用法。

## 当前规范

| 文档 | 用途 | 状态 |
| --- | --- | --- |
| [`../README.md`](../README.md) | 项目概览、算法和工程边界 | 当前 |
| [`METRIC_DEFINITION.md`](METRIC_DEFINITION.md) | 时延惩罚的精确定义与论文推荐表述 | 当前，正式实验口径 |
| [`EXPERIMENT_GUIDE.md`](EXPERIMENT_GUIDE.md) | 测试、标定、pilot、正式 sweep、后台状态 | 当前，运行入口 |
| [`CALIBRATION_20260913.md`](CALIBRATION_20260913.md) | 当前目标参考尺度的命令、样本和结果 | 当前，正式实验依据 |
| [`../PROJECT_REVIEW_AND_ROADMAP.md`](../PROJECT_REVIEW_AND_ROADMAP.md) | 项目审计、证据边界和后续路线 | 当前 |
| [`old_bs_cross_frame_theory.md`](old_bs_cross_frame_theory.md) | 旧 BS 任务跨帧调度的理论边界 | 当前参考 |

## 实验记录

`analysis/` 下的文档分为三类：

- `*_preregistration.md`：运行前冻结的设计、种子和停止规则；
- `20*_results.md`：已经完成的阶段结果和结论边界；
- `20*_pause.md`、`*_status.md`、`*_summary.md`：过程记录或阶段性汇总。

当前正式主线使用 `coupled + legacy`。`budgeted-75%` 只作为机制基线；
`joint_dpp` 及 witness 路线已按预注册规则终止。要复现实验历史时再直接使用
`analysis/run_*.py`，不要把这些入口当作新一轮正式 sweep 的默认入口。

## 历史问题文档

- `lyapunov_optimizer_problem.md` 记录早期量纲问题，里面的旧权重和算法描述不代表
  当前实现。
- `”短视陷阱“.md` 是理论探索笔记，其中带有未整理的检索标记，不作为论文可直接
  引用的参考文献表。

两份文档只为追溯研究过程保留；当前结论以本页“当前规范”所列文档为准。

## 唯一推荐工程入口

```bash
scripts/ldactl help
```

该入口统一转发测试、smoke、标定、数值 oracle、前台 sweep 和后台正式实验控制；
原脚本继续保留，因此既有命令和历史 manifest 不受影响。
