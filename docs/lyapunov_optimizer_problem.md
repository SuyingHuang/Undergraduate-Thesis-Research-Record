# Lyapunov优化目标函数量纲均衡问题

## 1. 问题背景

### 1.1 项目概述

本项目研究**星地一体化网络（SAGIN）中的任务卸载与资源分配**问题：
- 场景：3个地面基站（BS）+ 1颗LEO卫星 + 30个用户设备（UE）
- 目标：最小化系统年龄信息（PAoI），同时保证队列稳定性与能量约束
- 算法：基于深度强化学习（DRL）的LDA算法 vs 多个基线算法

### 1.2 Lyapunov优化框架

LDA算法采用Lyapunov优化框架，将队列稳定性约束纳入目标函数：

```
G1 = term_q + term_p + term_e
```

其中：
- **term_q**：队列漂移项（Queue Drift）—— 保证队列稳定性
- **term_p**：PAoI惩罚项（Age of Information）—— 优化信息新鲜度
- **term_e**：能量惩罚项（Energy Constraint）—— 满足基站能量预算

---

## 2. G1目标函数详解

### 2.1 完整公式

```python
G1 = term_q * 0.01 + term_p + term_e * 10
```

### 2.2 各分项物理含义与计算

#### (1) 队列漂移项 term_q

```python
term_q_bs = np.sum((env.Q_bs / 1e5) * ((l_left_bs_new - l_proc_old_bs) / 1e4))
term_q_sat = np.sum((env.Q_sat / 1e5) * ((l_left_sat_new - env.current_q_sat_reduction_mat) / 1e4))
term_q = term_q_bs + term_q_sat
```

**物理含义**：Lyapunov漂移，本质是 $Q(t) \cdot [A(t) - S(t)]$

| 变量 | 含义 | 典型量级 |
|------|------|---------|
| `env.Q_bs` | 基站队列积压 | ~10⁷ bits |
| `env.Q_sat` | 卫星队列积压 | ~10⁶ bits |
| `l_left_bs_new` | 本帧新任务残留 | ~10⁶ bits |
| `l_proc_old_bs` | 旧任务被清理量 | ~10⁶ bits |
| `current_q_sat_reduction_mat` | 卫星账本自然流逝量 | ~10⁵ bits |

#### (2) PAoI惩罚项 term_p

```python
term_p = K_p * np.sum(paoi_total)
# K_p = 100
```

**物理含义**：Age of Information时间惩罚，衡量信息从产生到被处理的时间间隔。

| 变量 | 含义 | 典型量级 |
|------|------|---------|
| `paoi_total` | 系统总PAoI | ~100-1000 秒 |
| `K_p` | PAoI权重系数 | 100 |

#### (3) 能量惩罚项 term_e

```python
term_e_bs = np.sum(env.E_BS * (e_bs_total - E_max_BS))
# E_max_BS = 160.0 Joules
```

**物理含义**：虚拟能量队列漂移，惩罚能量超限行为。

| 变量 | 含义 | 典型量级 |
|------|------|---------|
| `env.E_BS` | 基站虚拟能量队列（累积超额） | ~10²-10³ |
| `e_bs_total` | 当帧总能耗 | ~10²-10³ Joules |
| `E_max_BS` | 基站能量预算上限 | 160 Joules |

---

## 3. 当前问题：量纲失衡

### 3.1 实际测量数据

运行 `python main.py` 后的G1分解数据：

| 时刻 | term_q×0.01 | term_p | term_e×10 | G1 |
|------|-------------|--------|-----------|-----|
| Fr 500 | -6,952 | 12,702 | -34,080 | **-28,331** |
| Fr 1000 | -14,194 | 14,693 | 0 | **+499** |
| Fr 1500 | -13,651 | 14,553 | -24,009 | **-23,107** |
| Fr 4000 | -12,129 | 15,804 | -38,217 | **-34,542** |

### 3.2 问题分析

1. **term_e（能量项）主导G1**：-2万~-3万的量级，符号为负
2. **term_p（PAoI项）次之**：+1万量级，符号为正
3. **term_q（队列项）几乎无影响**：-100量级，系数0.01进一步削弱

### 3.3 问题本质

三项的**物理量纲不同**：
- term_q：bits² 的归一化值
- term_p：秒（时间）
- term_e：Joules² 的归一化值

直接相加没有物理意义，当前通过**人工系数（0.01, 10）强行调平**，但：
- 系数缺乏理论依据
- 系数不随系统状态自适应
- 导致队列稳定性几乎不被考虑

---

## 4. LDA vs AC 算法对比

### 4.1 目标函数差异

**LDA（完整Lyapunov优化）**：
```python
G1_lda = term_q * 0.01 + term_p + term_e * 10
```

**AC（Actor-Critic，剥离队列稳定性）**：
```python
G1_ac = term_p + term_e * 10  # 无term_q项
```

### 4.2 实际表现

由于term_q为负值：
- LDA的G1比AC更负（多减去了一个负数）
- LDA理论上应该更积极地清空队列

### 4.3 观察到的现象

运行 `run_sweeps.py` 参数扫描后发现：
- **LDA和AC的性能曲线几乎完全重合**
- 说明队列项的权重太轻，没有产生可观测的差异

---

## 5. 理论背景：Lyapunov优化

### 5.1 标准Lyapunov函数

```python
L(t) = (Q_bs² + Q_sat²) / (2*γ_Q) + E_bs² / (2*γ_E)
```

### 5.2 Drift-Plus-Penalty形式

```python
ΔL(t) + K_p * PAoI(t)
```

其中ΔL(t) = L(t+1) - L(t)是Lyapunov漂移。

### 5.3 当前实现的偏差

当前term_q实现为：
```python
term_q = np.sum((Q / 1e5) * ((l_left - l_proc_old) / 1e4))
```

这是对Lyapunov漂移的**工程近似**，归一化系数1e5/1e4缺乏理论依据。

---

## 6. 改进方向建议

### 6.1 量纲统一（最紧迫）

将三项归一化到同一量纲：

| 方案 | 描述 | 优缺点 |
|------|------|--------|
| A | 除以各自典型值 | 简单，但需先验知识 |
| B | 使用自适应权重 | 更鲁棒，但需额外学习 |
| C | 理论推导归一化系数 | 最严谨，但难度大 |

### 6.2 归一化目标函数建议

```python
# 方案A示例
G1 = (term_q / Q_ref) + K_p * (term_p / PAoI_ref) + K_e * (term_e / E_ref)
```

其中 Q_ref, PAoI_ref, E_ref 可以是：
- 各自的历史均值
- 各自的物理上限
- 通过理论分析确定

### 6.3 自适应权重机制

参考PPO等算法中的advantage normalization思想：
```python
term_q_normalized = (term_q - mean_q) / (std_q + eps)
term_p_normalized = (term_p - mean_p) / (std_p + eps)
term_e_normalized = (term_e - mean_e) / (std_e + eps)

G1 = w_q * term_q_norm + w_p * term_p_norm + w_e * term_e_norm
```

---

## 7. 相关文件索引

| 文件 | 关键代码位置 | 内容 |
|------|------------|------|
| `core/agents/lda_agent.py` | lines 200-245 | G1目标函数计算 |
| `core/agents/baselines.py` | lines 111-130 | ACAgent目标函数 |
| `core/env.py` | lines 255-300 | 队列与能量更新 |
| `config.py` | line 73 | K_p参数定义 |

---

## 8. 待解决问题

1. **核心问题**：如何设计归一化方案，使三项具有可比性？
2. **理论问题**：Lyapunov漂移项的理论形式与当前工程实现的关系？
3. **实践问题**：如何确保LDA和AC在参数扫描中展现可观测的差异？
4. **自适应问题**：是否需要引入自适应权重机制？

---

*文档生成时间：2026-04-26*
*问题抽象者：Claude Code*
