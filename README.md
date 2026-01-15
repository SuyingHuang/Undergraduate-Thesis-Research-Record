# 时敏网络下的分布式资源调度与优化系统
# (Time-Sensitive Distributed Resource Optimization System)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Optimization](https://img.shields.io/badge/Optimization-Convex_%26_Lyapunov-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Active_Development-orange.svg)]()

> **关键词**：凸优化、李雅普诺夫优化、数值计算、随机信道建模、资源调度

### 说明
本项目是我的毕设课题。此项目的灵感主要来自于《Age-Critical Joint Communication and Computation Offloading for Satellite-Integrated Internet》，目前已经做了P3、P4的求解工作，未来预计在全部复现的基础上使这个系统更加”复杂“，以使这个系统更符合实际的应用情况。
## 项目背景与挑战 (Context)

本项目旨在解决**高动态、强约束环境下的资源分配问题**。

在一个典型的空天地一体化网络（SAGIN）场景中，我们需要在毫秒级的时间窗内，为海量并发任务分配计算资源（CPU频率）和通信资源（功率/带宽）。这是一个典型的**非凸、非线性、且带有长期平均约束的随机优化问题**。

**核心难点：**
1.  **极端的时敏性要求**：目标是最小化“信息年龄”（Peak Age of Information, PAoI），而非传统的吞吐量，对调度算法的实时性要求极高。
2.  **严格的物理约束**：系统必须满足长期能量预算（Energy Budget）和瞬时频率上限（Frequency Cap）。
3.  **环境的高度随机性**：通信链路受到 Shadowed-Rician 衰落影响，信道质量随时间剧烈波动。

## 核心解决方案 (Methodology)

为了在不依赖黑盒求解器的情况下实现高效求解，本项目从底层推导并实现了一套**基于模型（Model-based）的优化引擎**。

### 1. 复杂约束解耦 (Lyapunov Optimization)
利用**李雅普诺夫优化（Lyapunov Optimization）**框架，构建虚拟队列（Virtual Queues）来追踪长期能量积压和任务积压。
* **作用**：将一个长期的时间耦合优化问题，解耦为一系列独立的、可并行求解的**单帧确定性优化问题**。
**优势**：将算法复杂度从指数级降低至多项式级，保证了系统的稳定性。

### 2. 定制化数值求解器 (Custom Numerical Solvers)
针对解耦后的子问题（P3 & P4），实现了基于**拉格朗日乘子法**和**牛顿迭代法**的高性能求解器。
* **闭式解推导**：针对目标函数推导出了基于三次方程（Cubic Equation）的半闭式解（Type A/B 分类讨论）。
* **牛顿法实现 (`core/math_utils.py`)**：手写实现了稳健的牛顿迭代算法 (`solve_cubic_newton`)，相比通用求根库，针对特定物理模型进行了初始化优化。
* **双层二分搜索**：设计了嵌套的二分查找算法（Bisection Search），用于寻找满足总功率约束的最优拉格朗日乘子 $\lambda, \mu, \nu$。

### 3. 高保真环境仿真 (Simulation Environment)
* **随机信道建模**：实现了 **Shadowed-Rician** 信道模型，使用**接受-拒绝采样（Accept-Reject Sampling）**生成符合特定概率密度函数（PDF）的随机样本，模拟真实的物理链路衰落。
* **OOP 架构**：采用面向对象设计，将配置 (`config`)、优化器 (`optimizer`)、信道 (`channel`) 和主循环逻辑解耦，便于后续接入强化学习（RL）Agent。

## 目前的项目结构 (Structure)

```text
.
├── config.py               # 系统参数配置中心 (单例模式管理物理参数)
├── core/
│   ├── math_utils.py       # 底层数值计算库 (牛顿法求解器)
│   ├── leo_optimizer.py    # 卫星端资源分配优化器 (算法核心)
│   ├── bs_optimizer.py     # 地面基站资源分配优化器
│   └── satellite_channel.py # Shadowed-Rician 信道建模与采样
├── test_alg3_leo.py        # 集成测试脚本 (算法收敛性验证)
├── test_alg2_bs.py         # 基准算法验证
└── README.md
```
## 初步成果与验证（Results）
目前已完成核心优化算子的开发与验证（Phase 1 & 2）。
* **收敛性验证**：在`test_alg3_leo.py`的仿真中，算法能够动态调节计算频率。当任务积压（Backlog）增加时，虚拟队列能够驱动优化器分配更多资源，并在积压消除后自动回落，验证了李雅普诺夫控制的有效性。(`test_alg2_bs.py`同理)
* **约束满足**：仿真数据显示，长期运行下的平均能耗严格控制在预设的`E_max`(60J/120J) 之下。

## 路线图（Roadmap）
本项目目前处于**核心算法验证**与**智能决策层开发**的过渡阶段：
* **Phase 1:物理层环境构建**：完成高保真信道建模与系统参数配置。
* **Phase 2:数值优化引擎 (Current Stage)**：
  -* 完成基于Lyapunov 的问题解耦。
  -* 实现基于牛顿法的 P3/P4 子问题求解器。
  -* 通过单元测试验证数学推导的正确性。
* **Phase 3: 智能化升级 (In Progress)**:
  -* 引入**深度强化学习（Deep Reinforcement Learning）**
  -* 计划实现 LDA (Lyapunov-enhanced Decoupled Actor-Critic) 算法：利用神经网络处理离散的“卸载决策”变量，与当前的连续变量优化器协同工作，进一步降低平均 PAoI。
## 技术栈
-* Python: 主要开发语言
-* NumPy: 高性能矩阵运算
-* SciPy: 科学计算与特殊函数 (hyp1f1)
-* Matplotlib: 数据可视化与分析
