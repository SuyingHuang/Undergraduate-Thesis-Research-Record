# 空天地一体化网络资源调度（LDA）

本仓库提供 LDA、LDA2 消融变体及 COB/MTD 基线的仿真代码。

当前版本完成了工程清理、共享卫星状态扩展、物理记账修正和回归测试。它是后续重新标定与正式实验的代码基线；论文正文、实验数据和答辩材料不随代码提交。

## 环境与入口

本次验证环境：Python 3.9.25、PyTorch 2.5.1、NumPy 2.0.1。依赖见 requirements.txt；测试使用标准库 unittest，无需 pytest。现有 Conda 环境不需要重新配置。

```powershell
# 从项目根目录执行
& 'D:/Anaconda/envs/thesis_env/python.exe' -X utf8 -m unittest discover -s tests -p 'test_*.py' -v

# 小规模端到端测试：2 BS × 3 UE、4 算法、2 种子、每次 20 帧
& 'D:/Anaconda/envs/thesis_env/python.exe' -X utf8 run_sweeps.py --smoke

# 选择正式实验（长实验；建议在服务器运行）
& 'D:/Anaconda/envs/thesis_env/python.exe' -X utf8 run_sweeps.py --experiments Exp1_J Exp2_L Exp3_fUE Exp4_K Exp5_UAV Exp6_Bc Exp7_Bsat --frames 4096 --seeds 42 123 456 789 1000 2003 3141 6283 --workers 8
```

其他入口：

- main.py：单次 LDA 仿真及绘图。
- run_experiments.py：单种子四算法比较，仍会覆盖 results/simulation_results_*.pkl。
- run_multi_seed_experiment.py：默认配置的多种子时间轨迹及均值曲线。
- run_sweeps.py：七组参数扫描；建议作为正式实验统一入口。
- analyze_metrics.py：读取指定 logs/sweep 子目录的历史指标，默认展示历史自适应窗口，不等于新主统计视图。
- collect_calibration.py：未训练策略的参考尺度标定；在 env.step 之前采集，结果不会自动覆盖 config.py。

每次仿真都会创建新网络、Adam 和回放池，不自动加载旧模型，属于从头训练。当前没有持久化 checkpoint 或断点续训接口。

### 资源分配修复与诊断消融

默认 `resource_solver='coupled'` 使用与上层评分一致的共享残余任务时延目标，并保留旧求解器的可行解作为比较起点。小节点枚举完成任务集合，大节点使用有限集合近似，不能据此宣称全局最优。`include_baseline_candidates=True` 将 COB/MTD 动作加入候选，只保证同状态的单步目标不更差。

Coupled 路径会在同一帧缓存重复联合动作，批量计算旧求解器 warm start，并用目标下界跳过不可能改善当前解的完成集合；这些优化不删减候选集合。Linux 启动脚本默认把 OMP/MKL/OpenBLAS/NumExpr 内部线程数限制为 1，避免多进程扫描时嵌套线程争用。默认 `J=10、delta_t=0.5` 的隔离 CPU 基准从约 6.0 秒/帧降至约 1.6 秒/帧；实际速度取决于 CPU、候选数和探索窗口。

```bash
python -m unittest discover -s tests
python analysis/check_coupled_oracle.py
python analysis/run_diagnostic_ablation.py --frames 512 --seeds 42 123 --workers 12
# 将下面路径替换为运行时打印的目录
python analysis/summarize_diagnostic.py results/diagnostic/<run-directory>
# 中断后复用完整运行；未完成的单组实验从头运行。参数与源代码必须保持一致。
python analysis/run_diagnostic_ablation.py --frames 512 --seeds 42 123 --workers 12 --resume results/diagnostic/<run-directory>
```

诊断矩阵包括默认点、`L_mean=10 Mb`、`J=8` 三个场景，分别对比旧/新求解器、基线候选、仅上层/仅下层/同时去 PAoI，以及旧 BS 任务的能量感知调度。每组保留源代码快照、参数、工作负载哈希、轨迹和训练更新次数；汇总默认拒绝未完成矩阵，可用 `--allow-partial` 明确导出预览。

统计固定使用后半程；图中黑点代表各个种子。512 帧仅覆盖初步训练（默认至少 256 条经验才开始），不代替长时域、多种子的正式验证。诊断固定原参考尺度，不自动重新标定。当前 `Cost` 是完成时延代理指标，尚未独立重建真实年龄峰值。

### Linux 服务器

`codex/linux-server` 分支对无桌面 Linux 自动启用 Matplotlib `Agg` 后端，并让多进程入口显式使用 `spawn`，避免 PyTorch/科学计算库在 `fork` 后出现线程状态问题。路径均由项目目录动态生成，没有依赖 Windows 盘符。

```bash
# 新克隆；已有仓库首次使用该分支时运行：
# git fetch origin && git switch --track origin/codex/linux-server
git clone --branch codex/linux-server \
  git@github.com:SuyingHuang/Undergraduate-Thesis-Research-Record.git
cd Undergraduate-Thesis-Research-Record

conda create -n thesis_env python=3.11 -y
conda activate thesis_env
python -m pip install -r requirements.txt

# 先验证环境
python -X utf8 -m unittest discover -s tests -p 'test_*.py' -v

# 启动正式实验；脚本无参数时会拒绝启动，避免误跑默认长实验
PYTHON_BIN=python scripts/run_linux.sh \
  --experiments Exp1_J Exp2_L Exp3_fUE Exp4_K Exp5_UAV Exp6_Bc Exp7_Bsat \
  --frames 4096 \
  --seeds 42 123 456 789 1000 2003 3141 6283 \
  --workers 8
```

服务器已启用用户级 systemd linger 时，可用项目自带的短命令在后台运行全部 Exp1–Exp7（默认 8 并发）：

```bash
scripts/trainctl start    # 启动，SSH 断开后继续
scripts/trainctl status   # 查看当前实验、worker、任务和帧进度
scripts/trainctl watch    # 每 10 秒刷新详细进度
scripts/trainctl logs     # 跟踪日志，Ctrl+C 不会停止训练
scripts/trainctl stop     # 停止主进程及所有 worker
```

服务器没有桌面时无需安装 X11。可用 `LDA_HEADLESS=0` 强制允许交互绘图，或通过 `MPLBACKEND` 自行选择后端。DNN 推理与训练默认使用 `LDA_DEVICE=auto`：检测到 CUDA 时使用 GPU，否则回退 CPU；也可显式设置为 `cpu`、`cuda` 或 `cuda:N`。多进程参数扫描会把 `auto` 模式的 LDA/LDA2 worker 轮流分配到可见 GPU，实际设备记录在任务日志的 `[DNN] device=...` 行。候选搜索和解析优化仍主要使用 CPU，因此 GPU 迁移不会让整段仿真按纯神经网络训练的比例加速。

## 模块与算法

执行关系：入口 → 环境生成任务/信道 → Agent 生成候选 → BS/LEO 优化器分配频率 → 候选评分 → 环境记账 → 回放训练。

| 代码标识 | 图例 | 实际含义 |
| --- | --- | --- |
| LDA | LDA1 | 含 PAoI、队列和能耗项的候选评估，监督学习候选优胜动作 |
| AC | LDA2 | 从候选评分和下层 BS/LEO 资源分配中同时去掉 PAoI；没有 Critic |
| COB | COB | 一帧内能本地完成则本地处理，其余任务全部给 BS |
| MTD | MTD | 本地判断后，每个 BS 将卫星传输时间最短的至多 2 个用户给 LEO，其余给 BS |

每个 BS 使用一个 Actor。其本地部分为 L_t、Q_bs、Q_sat、E_BS、T_left、R_BS、R_sat，共 `5J+2` 维；另加入全局卫星积压、当帧旧卫星服务计划、有限个逐星负载槽、旧卫星数量和旧卫星能耗，共 `2IJ+S+2` 维。因此总输入维度为 `5J+2+2IJ+S+2`；默认 `I=3、J=10、S=8` 时是 122 维。当前任务量和队列以 Mbit 缩放。旧输入维度的模型权重不兼容，必须从头训练。

网络为 LayerNorm + 两个等宽残差块。扩展输入后默认隐藏维度由 512 增至 640。回放采用均匀 `random.sample`，不是优先经验回放；只对实际进入卸载决策的用户计算监督损失，本地执行用户的无意义 `b` 位不参与训练。默认回放容量 1024、batch 64、训练间隔 10；至少积累 256 条经验才训练。

候选组合采用受限坐标搜索：以当前联合动作出发，逐个 BS 替换候选并重复有限轮次，避免原来的“各候选列表按同一下标截断”问题，同时控制笛卡尔积爆炸。默认最多 3 轮，它仍是近似搜索而非全局最优证明。

候选评分与 BS/LEO 下层优化器共用同一组队列、PAoI、能量归一化系数。BS/LEO 优化器保留标量、向量化和多候选版本，用于数值交叉验证，不是可随意删除的重复代码。

## 参数扫描和统计口径

可选实验：Exp1_J、Exp2_L、Exp3_fUE、Exp4_K、Exp5_UAV、Exp6_Bc、Exp7_Bsat。无参数运行时仍只执行 Exp1、Exp2；其余通过 --experiments 选择。

每次扫描输出到新的时间戳目录 logs/sweep 和 results/sweep，包含配置、代码提交号、运行环境、任务序列哈希、逐种子指标、日志和图表。每算法/种子独立初始化；同一参数/种子共用任务序列并重置信道 RNG。

- 默认 --view fixed_half：所有算法统一取后半段，保留全部成功种子；失败保持缺失，不填补、不因收敛慢剔除种子。
- --view raw：历史自适应窗口，保留全部成功种子。
- --view cleaned：历史筛选规则，仅供追溯/敏感性分析，不作为主要结论。
- delta_t 变小只是探索窗口诊断，不能据此宣称损失、策略或队列已经收敛。
- ci95 使用双侧 95% Student-t 区间半宽；n<2 时不可估计，保存为 NaN。
- 另保存同一种子下“其他算法减 LDA1”的配对差值及其 Student-t 区间，减少任务序列差异造成的方差。
- 固定后半段会做分块稳定性诊断，但诊断只提示、不删除成功种子。
- Q 原始存储单位为 bit，绘图转换为 Mbit。训练 Loss 以真实训练帧记录。

固定后半段也是预设的比较窗口，不是自动收敛认证；正式结果需要更长时域、完整种子和稳定性诊断。主统计窗口与原论文“学习算法稳态、启发式全程”的口径不同，不能直接把新表格当旧论文图的复现。

## 已修正项与剩余边界

常规测试现已覆盖：卫星积压单一来源、旧卫星逐星能耗约束、决策时刻状态一致性、卫星平均能耗分母、BS 跨帧残余、实际 Lyapunov 漂移、BS/卫星队列守恒、候选组合、掩码监督，以及三种优化器实现之间和稠密网格之间的数值一致性。环境在运行时也会检查队列守恒与逐星能耗。

仍需谨慎解释的边界：

1. 逐星状态使用固定 `S` 个负载槽，超出槽位的卫星通过全局积压、计划服务和数量汇总表达；这是有界近似状态，不是严格的马尔可夫充分性证明。
2. 受限坐标搜索控制了计算量，但不等同于完整联合候选笛卡尔积。
3. 代码的逐用户/逐 BS 队列和跨帧卫星账本比论文中的聚合公式更细；重新实验后需要按实际实现统一论文符号与说明。
4. 当前 `Q_ref、PAoI_ref、E_ref` 已按重构后的 5 种子 × 200 帧未训练探索轨迹重新标定；稀疏能量项按非零活跃帧统计，并对尺度反馈采用对数空间阻尼。更改状态、目标函数或物理参数后必须重新标定，且不能把标定或冒烟测试当论文结果。
5. BS 能量采用长期平均约束对应的虚拟队列；`run_sweeps.py` 的高积压提示是稳定性诊断，不代表单帧物理能耗越界。卫星能量才按逐颗、逐帧硬约束检查。

测试通过表示上述工程行为和数值断言通过，不等同于已经复现论文中的全部曲线或证明算法全局最优。

## 版本管理

仅提交源代码、测试及必要说明。结果、模型权重、论文 PDF、PPT、缓存和本地审查备份均被忽略。曾被跟踪的结果/参考 PDF 从当前 Git 树移除，但本地文件保留，既有 Git 历史不重写。
