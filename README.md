# 空天地一体化网络资源调度（LDA）

本仓库提供 LDA、LDA2 消融变体及 COB/MTD 基线的仿真代码。

当前是**工程清理与模型审查基线**，不是已经通过全部物理约束验证的论文复现版本。已知问题见下文；正式重跑论文实验前应先修正模型记账问题。论文正文、实验数据和答辩材料不随代码提交。

## 环境与入口

本次验证环境：Python 3.9.25、PyTorch 2.5.1、NumPy 2.0.1。依赖见 requirements.txt；测试使用标准库 unittest，无需 pytest。现有 Conda 环境不需要重新配置。

```powershell
# 从项目根目录执行
& 'D:/Anaconda/envs/thesis_env/python.exe' -X utf8 -m unittest discover -s tests -p 'test_*.py' -v

# 小规模端到端测试：2 BS × 3 UE、4 算法、2 种子、每次 20 帧
& 'D:/Anaconda/envs/thesis_env/python.exe' -X utf8 run_sweeps.py --smoke

# 选择实验（长实验；解决已知模型问题后再用于论文）
& 'D:/Anaconda/envs/thesis_env/python.exe' -X utf8 run_sweeps.py --experiments Exp1_J Exp2_L --frames 4096 --seeds 42 123 456 789 1000 2003 3141 6283 --workers 4
```

其他入口：

- main.py：单次 LDA 仿真及绘图。
- run_experiments.py：单种子四算法比较，仍会覆盖 results/simulation_results_*.pkl。
- run_multi_seed_experiment.py：默认配置的多种子时间轨迹及均值曲线。
- run_sweeps.py：七组参数扫描；建议作为正式实验统一入口。
- analyze_metrics.py：读取指定 logs/sweep 子目录的历史指标，默认展示历史自适应窗口，不等于新主统计视图。
- collect_calibration.py：未训练策略的参考尺度标定；在 env.step 之前采集，结果不会自动覆盖 config.py。

每次仿真都会创建新网络、Adam 和回放池，不自动加载旧模型，属于从头训练。当前没有持久化 checkpoint 或断点续训接口。

## 模块与算法

执行关系：入口 → 环境生成任务/信道 → Agent 生成候选 → BS/LEO 优化器分配频率 → 候选评分 → 环境记账 → 回放训练。

| 代码标识 | 图例 | 实际含义 |
| --- | --- | --- |
| LDA | LDA1 | 含 PAoI、队列和能耗项的候选评估，监督学习候选优胜动作 |
| AC | LDA2 | 仅从候选评分中去掉 PAoI；下层资源分配仍含 K_p；没有 Critic |
| COB | COB | 一帧内能本地完成则本地处理，其余任务全部给 BS |
| MTD | MTD | 本地判断后，每个 BS 将卫星传输时间最短的至多 2 个用户给 LEO，其余给 BS |

每个 BS 独立 Actor，输入依次为 L_t、Q_bs、Q_sat_total、E_BS、T_left、R_BS、R_sat，共 5J+2 维。当前任务量以 Mbit 缩放；旧 4J+2 维权重不兼容。加入 L_t 不代表已证明状态满足马尔可夫性。

网络为 LayerNorm + 两个等宽残差块。回放采用均匀 random.sample，不是优先经验回放。默认 FocalLoss(gamma=0, alpha=0.5) 为逐元素 BCE 均值的 0.5 倍。默认回放容量 1024、batch 64、训练间隔 10；至少积累 256 条经验才训练。

BS/LEO 优化器保留标量、向量化和多候选版本，用于数值交叉验证，不是可随意删除的重复代码。

## 参数扫描和统计口径

可选实验：Exp1_J、Exp2_L、Exp3_fUE、Exp4_K、Exp5_UAV、Exp6_Bc、Exp7_Bsat。无参数运行时仍只执行 Exp1、Exp2；其余通过 --experiments 选择。

每次扫描输出到新的时间戳目录 logs/sweep 和 results/sweep，包含配置、代码提交号、运行环境、任务序列哈希、逐种子指标、日志和图表。每算法/种子独立初始化；同一参数/种子共用任务序列并重置信道 RNG。

- 默认 --view fixed_half：所有算法统一取后半段，保留全部成功种子；失败保持缺失，不填补、不因收敛慢剔除种子。
- --view raw：历史自适应窗口，保留全部成功种子。
- --view cleaned：历史筛选规则，仅供追溯/敏感性分析，不作为主要结论。
- delta_t 变小只是探索窗口诊断，不能据此宣称损失、策略或队列已经收敛。
- ci95 是 1.96 × 样本标准差 / sqrt(n) 的正态近似半宽；n<2 时不可估计，保存为 NaN。小样本不应把这个近似当作严格置信保证。
- Q 原始存储单位为 bit，绘图转换为 Mbit。训练 Loss 以真实训练帧记录。

固定后半段也是预设的比较窗口，不是自动收敛认证；正式结果需要更长时域、完整种子和稳定性诊断。主统计窗口与原论文“学习算法稳态、启发式全程”的口径不同，不能直接把新表格当旧论文图的复现。

## 已知模型问题（尚未修正）

独立物理断言目前有 6 项失败，用以下命令复现，预期退出码为 1：

```powershell
& 'D:/Anaconda/envs/thesis_env/python.exe' -X utf8 -m unittest tests.known_model_issues -v
```

1. Q_sat 已包含旧账本残余，再加 Q_sat_pending 会重复计数。
2. 旧卫星尾任务频率按 f_max_Sat 而非能量受限频率计算，存在瞬时能耗超限。
3. Agent 在 step 前读到上一帧的旧卫星服务/能耗；本帧清算发生在 step 内。
4. 卫星平均能耗的分母在账本更新后统计，可能重复计入新卫星并遗漏刚清空的旧卫星。
5. BS 少量旧债未清完时，队列仍有余额，但物理残余账本会被本帧新残余覆盖。
6. history['Drift'] 当前是队列增量的平方项，并非实际 Lyapunov 函数前后差。

此外：各 BS 候选按最短列表同下标拼接会丢弃其他 BS 的候选；只输入本 BS 聚合积压不足以保证共享卫星的完整状态；归一化候选评分与下层优化器的权重口径未统一；旧卫星能量约束下的 PAoI 预测仍用最大频率估时。这些需要模型级修订和重新实验，不能通过清理代码解决。

常规回归测试通过，只表示所测试的工程行为和数值一致性通过；不会把上述失败隐藏为“预期成功”，也不声称已复现整篇论文。

## 版本管理

仅提交源代码、测试及必要说明。结果、模型权重、论文 PDF、PPT、缓存和本地审查备份均被忽略。曾被跟踪的结果/参考 PDF 从当前 Git 树移除，但本地文件保留，既有 Git 历史不重写。
