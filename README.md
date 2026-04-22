# 空天地地一体化网络资源调度系统
# (Satellite-Ground Integrated Network Resource Scheduling)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Optimization](https://img.shields.io/badge/Optimization-Lyapunov-orange.svg)]()

> **关键词**：空天地一体化网络（SAGIN）、深度强化学习、Lyapunov优化、资源调度、PAoI最小化

### 项目简介
本项目研究**空天地一体化网络（SAGIN）**场景下的实时资源调度问题，目标是最小化信息年龄（Peak Age of Information, PAoI）。系统实现了基于Lyapunov优化的传统调度算法与基于深度强化学习（LDA）的智能决策算法，并支持多种基线算法的对比评估。

---

## 项目背景与挑战

**典型场景**：在空天地一体化网络中，海量终端设备产生计算任务，需在毫秒级时间窗内决定任务分配策略——本地执行、卸载至地面基站（BS）、或卸载至低轨卫星（LEO）。

**核心挑战**：
1. **时敏性**：PAoI最小化要求调度算法具备实时决策能力
2. **随机性**：Shadowed-Rician衰落信道与随机业务到达
3. **复杂性**：计算资源（CPU频率）、通信资源（功率/带宽）、长期能量约束的联合优化

## 核心算法实现

### 1. 传统优化方法（Lyapunov Optimization）
- **问题解耦**：利用Lyapunov优化将长期约束解耦为单帧确定性优化
- **子问题求解**：P3/P4子问题基于拉格朗日乘子法与牛顿迭代求解
- **文件**：`core/optimizers/leo_optimizer.py`, `core/optimizers/bs_optimizer.py`

### 2. 深度强化学习方法（LDA）
- **网络结构**：类似于经典的DPOO算法（Deep Reinforcement learning-based Online Offloading），每基站独立Actor网络
- **训练策略**：Focal Loss + Priority Experience Replay
- **文件**：`core/agents/lda_agent.py`, `core/models/dnn_model.py`

### 3. 基线算法
- **AC**（All Computing）：任务全部本地执行
- **COB**（Computation Offloading to BS）：任务全部卸载至基站
- **MTD**（Most Transmit Diversity）：基于最大传输分集策略
- **文件**：`core/agents/baselines.py`

## 项目结构

```
.
├── config.py                 # 系统参数配置
├── main.py                   # 主入口：运行LDA算法
├── run_experiments.py        # 批量实验脚本
├── run_sweeps.py             # 参数扫描脚本
├── core/
│   ├── env.py                # SAGIN仿真环境
│   ├── agents/
│   │   ├── lda_agent.py      # LDA强化学习智能体
│   │   └── baselines.py       # 基线算法（AC/COB/MTD）
│   ├── optimizers/
│   │   ├── leo_optimizer.py  # 卫星端优化器
│   │   └── bs_optimizer.py    # 基站端优化器
│   ├── models/
│   │   ├── dnn_model.py       # Actor网络与Focal Loss
│   │   └── tcopq.py          # 任务分配候选生成
│   └── channels/
│       ├── satellite_channel.py  # Shadowed-Rician卫星信道
│       ├── bs_channel.py        # 基站信道模型
│       └── uavr_channel.py       # UAV中继信道
├── utils/
│   ├── math_utils.py         # 数值计算工具
│   └── plotter.py            # 结果可视化
└── tests/
    ├── test_alg3_leo.py      # LEO优化器测试
    └── test_alg2_bs.py        # 基站优化器测试
```

## 快速开始

```bash
# 运行LDA算法
python main.py

# 运行基线算法对比实验
python run_experiments.py

# 参数扫描
python run_sweeps.py
```

## 技术栈
- **Python 3.8+**
- **PyTorch 2.0+**：深度强化学习
- **NumPy**：数值计算
- **SciPy**：科学计算
- **Matplotlib**：可视化

## 参考文献
> 《Age-Critical Joint Communication and Computation Offloading for Satellite-Integrated Internet》
