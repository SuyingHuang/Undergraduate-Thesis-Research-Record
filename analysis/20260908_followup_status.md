# 资源分配修复与消融续跑记录

状态：诊断批次已完成，最终结论见 `analysis/20260909_diagnostic_results.md`。

- 已重新执行 67 项 unittest，全部通过，日志见 `analysis/test_results.txt`。
- 已重新执行 40 个 BS/LEO 双用户密集网格对照，最大正目标差为 0。
- 已验证完整组复用、恢复参数不一致拒绝、未完成矩阵默认拒绝正式导出。
- 主批次：`results/diagnostic/20260908_210103_086927`，66/66 组成功，零失败。
- 规模：3 场景 × 11 变体 × 2 种子（42、123）× 512 帧；12 个 worker。
- 运行日志：`/tmp/lda-diagnostic-512.log`。
- 原 64 帧预跑仅完成 12/33 组，零训练更新，已导出明确标注 PARTIAL 的初步表格。

重新生成汇总时执行：

```bash
python analysis/summarize_diagnostic.py results/diagnostic/20260908_210103_086927
```

如主批次中断，源代码和实验参数保持一致时执行：

```bash
python analysis/run_diagnostic_ablation.py --frames 512 --seeds 42 123 --workers 12 --resume results/diagnostic/20260908_210103_086927
```

验收：66/66 组成功、轨迹与来源一致、形成配对差值/对比图/时间曲线和中文结论；有限时域诊断不能宣称长期收敛，也未完成正式重新标定。
