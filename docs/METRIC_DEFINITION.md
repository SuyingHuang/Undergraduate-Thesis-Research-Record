# PAoI 导向的加权完成时延惩罚

## 1. 定位

代码历史上把该指标记为 `PAoI`，把逐帧均值存入 `history['Cost']`。它的设计目的
是让调度器偏好尽快完成任务，并对无法在当前帧完成、需要延期到后续帧的任务施加
更强惩罚。它不是由信息包生成时间、交付时间和接收端年龄轨迹重建的峰值信息年龄
（Peak Age of Information, PAoI）。

从本版本起，论文、图注和结果分析统一使用以下名称：

> **PAoI 导向的加权完成时延惩罚**（PAoI-oriented weighted completion-delay
> penalty），简称**时延惩罚**（delay penalty）。

代码字段名暂不改动，以保持历史结果文件、脚本和测试兼容。

## 2. 实际计算定义

设帧长为 $\tau$，每 bit 所需 CPU cycle 为 $\phi$。对当前帧产生的任务 $j$，
若任务在本帧完成，其惩罚为从帧起点到完成的估计时间

$$
d_j=\tau-t_j^{\mathrm{avail}}+\frac{\phi L_j}{f_j},
$$

其中 $t_j^{\mathrm{avail}}$ 是扣除传输、传播或旧任务占用后可用于计算的时间。
本地完成任务没有这些前置占用，因此对应 $d_j=\phi L_j/f_j$。

若任务在本帧结束时仍有残量，代码使用

$$
d_j=\tau+w\frac{\phi\left(W^{\mathrm{old}}+
\sum_k W_k^{\mathrm{left}}\right)}{f^{\mathrm{future}}},
\qquad w=2,
$$

其中 $W^{\mathrm{old}}$ 是同一计算节点尚未完成的旧任务总量，
$\sum_k W_k^{\mathrm{left}}$ 是该节点本帧新任务的总残量，
$f^{\mathrm{future}}$ 是后续处理频率的保守估计。共享残量会计入该节点每个未完成
任务的惩罚，$w=2$ 进一步放大跨帧延期代价；这两点都是有意的风险厌恶设计，
不是对真实 PAoI 的等式推导。

每帧报告的 `Cost` 为全部用户位置上 $d_j$ 的算术平均；没有分配到某类节点的
位置在对应分项中为零，三种处理路径合并后每个任务只贡献一次。

## 3. 在优化目标中的作用

时延惩罚作为 drift-plus-penalty 目标中的 penalty 项：

$$
G_1=\widetilde G_Q+
\frac{K_p}{D_{\mathrm{ref}}}\sum_j d_j+
\widetilde G_E.
$$

$K_p$ 控制时延偏好，$D_{\mathrm{ref}}$ 是量级标定尺度；队列漂移和能量漂移
分别由 $\widetilde G_Q$ 与 $\widetilde G_E$ 表示。由此可以声称算法优化了定义明确
的加权完成时延 surrogate，但不能从该式直接推出真实 PAoI 最优性。

## 4. 论文推荐表述

可在系统模型或问题定义中使用：

> 为避免仅最小化当帧已完成任务的处理时间而低估跨帧延期风险，本文定义一个
> PAoI 导向的加权完成时延惩罚。对当帧完成的任务，该指标取其估计完成时延；
> 对当帧未完成的任务，该指标以共享残余工作量估计后续完成时间，并使用
> $w=2$ 放大延期代价。该指标是风险厌恶的调度 surrogate，而非由接收端年龄
> 时间线直接采样得到的真实峰值 AoI。

英文可表述为：

> We optimize a PAoI-oriented weighted completion-delay penalty. It uses the
> estimated completion delay for tasks finished within the current frame and
> a shared-residual extrapolation with an overdue weight $w=2$ for unfinished
> tasks. The metric is a risk-averse scheduling surrogate rather than an
> event-level peak AoI reconstructed from receiver-side timestamps.

## 5. 结果命名规则

- 表格列名：`Weighted completion-delay penalty (s)`；
- 图纵轴：`Average weighted delay penalty (s)`；
- 中文正文：首次写全称，后续简称“时延惩罚”；
- 讨论历史代码或产物时可写“代码字段 `PAoI`/`Cost`”；
- 不使用“真实 PAoI”“接收端峰值年龄”或“PAoI 测量值”描述该指标。

已有实验数值无需因名称调整而重算，但跨版本比较仍必须使用相同提交、配置、种子
和统计窗口。
