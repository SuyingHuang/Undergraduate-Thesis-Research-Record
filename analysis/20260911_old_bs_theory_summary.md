# 旧 BS 调度、长期能量约束与跨帧反馈总结

## 结论定位

当前结论不应表述成“`L=16 Mbit` 时给旧任务 50% 能量最好”。更准确的表述是：

> 按剩余工作量比例分配频率能够在固定总频率下最小化旧任务的最晚完成时间；
> 但 CPU 能耗对频率是凸函数，在长期平均能量约束下，旧任务满频清算可能通过
> 能量虚拟队列抑制后续新任务服务，从而形成跨帧正反馈。固定能量份额策略能够
> 截断这条反馈，但固定份额本身不是联合 PAoI 问题的理论最优解。

完整推导见 [`docs/old_bs_cross_frame_theory.md`](../docs/old_bs_cross_frame_theory.md)。

## 1. 比例分配严格优化的对象

设旧任务为 \(w_j\)，总频率约束为 \(\sum_jf_j\le F\)，每 bit 需要 \(\phi\)
个 cycle。比例分配

\[
f_j=F\frac{w_j}{\sum_kw_k}
\]

求解的是

\[
\min_{\boldsymbol f}\max_j\frac{\phi w_j}{f_j},
\]

即最小化最晚完成时间。其最优值为 \(\phi\sum_jw_j/F\)，所有任务同时完成。

比例分配并不一般性地最小化总完成时间或总 PAoI。忽略能耗时，最小化
\(\sum_j\phi w_j/f_j\) 的解为 \(f_j\propto\sqrt{w_j}\)。所以比例分配的
理论含义是 makespan 公平性，而不是 PAoI 全局最优性。

## 2. 凸性能耗与最大频率清算

令 \(W=\sum_jw_j\)、\(r_j=w_j/W\)，并令 \(f_j=r_jF\)。旧任务能耗为

\[
E^o(F)=
\begin{cases}
\kappa\tau F^3\sum_jr_j^3,&F<\phi W/\tau,\\
\kappa\phi WF^2\sum_jr_j^3,&F\ge\phi W/\tau.
\end{cases}
\]

该函数连续且随 \(F\) 单调增加。未完成区按 \(F^3\) 增长，完成区按 \(F^2\)
增长。因此，使用最大频率换取少量即时完成时间缩短，可能付出超线性能耗。

对固定 cycle 总量 \(C\)，若允许在时间 \(T\) 内以常频完成，则

\[
E=\kappa\frac{C^3}{T^2}.
\]

这说明在没有硬截止期时，跨帧平滑服务可能比“当前满频、以后空闲”节能。

## 3. 长期能量约束为什么会产生跨帧反馈

长期平均能量约束由虚拟队列表示：

\[
Z_{t+1}=[Z_t+E^o_t+E^n_t-\bar E]^+.
\]

对任意样本路径都有

\[
Z_T\ge Z_0+\sum_{t=0}^{T-1}(E^o_t+E^n_t-\bar E).
\]

所以若平均能耗长期超过预算，\(Z_T\) 必然至少线性增长。最终某一点低于报警
阈值不能证明稳定，必须检查平均能耗和能量队列增长率。

如果新任务服务量满足“可用时间越长则服务越多、能量债务越高则服务越少”，
系统会产生一般性的反馈链：

\[
Q\uparrow\Rightarrow E^o\uparrow\Rightarrow Z\uparrow
\Rightarrow S^n\downarrow\Rightarrow Q\uparrow.
\]

在平衡点附近写成

\[
\begin{bmatrix}\delta Q_{t+1}\\\delta Z_{t+1}\end{bmatrix}
=J\begin{bmatrix}\delta Q_t\\\delta Z_t\end{bmatrix},
\]

则离散系统局部稳定要求 \(\rho(J)<1\)。因此，真正需要跨场景检验的是反馈
增益和容量区域，而不是某一个任务量点的最终队列数值。

## 4. 固定预算策略能保证什么

给旧任务名义份额 \(\beta\bar E\)，定义

\[
F_\beta=\max\{F\le F_{\max}:E^o(F)\le\beta\bar E\}.
\]

由于能耗随频率单调增加，\(F_\beta\) 可以解析反解。它严格保证：

1. 旧任务能耗不超过 \(\beta\bar E\)；
2. 在比例分配策略类内，预算下的旧任务服务量最大、占用时间最短。

它不保证：

- \(\beta=0.5\) 是最优份额；
- 新旧任务总能耗逐帧不超过 \(\bar E\)；
- 总 PAoI 全局最优；
- 无限时域下物理队列和能量队列必然稳定。

固定份额存在基本可行区间：

\[
\beta>
\frac{\kappa\phi^3\sigma\lambda^3}{\tau^2\bar E},
\qquad
\beta\le1-\frac{\bar E_n}{\bar E}.
\]

左侧来自旧任务服务容量，右侧来自给新任务保留的长期能量。如果区间为空，
固定切分不可能同时满足两方面要求，应改为动态联合调度。

## 5. 训练前容量包络

若保守地把某帧某 BS 的全部原始到达都视为下一帧旧任务，则一帧清空所需的
预算份额为

\[
\beta_{\mathrm{req},t}
=\frac{\kappa\phi^3\sum_jL_{j,t}^3}{\tau^2\bar E}.
\]

任意实际 BS 卸载子集的所需预算不会高于该值。因此，高于该包络是充分条件；
低于它并不代表一定不可行，只能作为容量边界探针。

对阶段 A 的两个预注册环境种子，训练前得到：

| 平均任务量 | 包络均值 | 95% 分位 | 99% 分位 | 样本最大值 |
|---|---:|---:|---:|---:|
| 10 Mbit | 5.64% | 8.3% | 9.6% | 11.8%–12.9% |
| 12 Mbit | 9.12% | 12.7% | 14.4% | 17.4%–18.6% |
| 16 Mbit | 20.1% | 26.1%–26.2% | 28.9% | 33.6%–34.9% |

因此保留 25%/50%/75%：25% 位于高负载容量转折附近；50% 和 75% 位于本批次
样本最大包络以上，用于检验能量余量和 PAoI 的权衡。

## 6. 更一般的最终策略

固定份额只是机制诊断。更一般的策略应联合选择旧任务频率、新任务频率和卸载
动作，最小化一步 Lyapunov drift-plus-penalty 上界：

\[
\min
\left\{
-\sum_jQ_{j,t}S_{j,t}
+Z_t(E^o_t+E^n_t)
+VP_t
\right\}.
\]

此时 \(Z_t\) 是动态能量影子价格，新旧任务通过同一个价格竞争资源，不再依赖
人工固定百分比。在标准平稳性、有界二阶矩和严格可行性条件下，该形式能够
得到时间平均目标 \(O(1/V)\) 最优性差距与 \(O(V)\) 队列权衡。该保证只针对
目标中明确定义的 \(P_t\)；如果 `PAoI` 仍是完成时间代理，就不能自动解释成
真实峰值 AoI 保证。

## 7. 实验进展与下一阶段

阶段 A 已按以下预注册方案完成：

- 负载：10、12、16 Mbit；
- 环境种子：104729、130363；
- 策略种子：42、123，嵌套在环境内；
- 处理：legacy、25%、50%、75%；
- 每次 2048 帧，统计后半 1024 帧；
- 先检查逐 BS 能耗、能量队列斜率和物理队列斜率，再比较 PAoI。

完整方案见
[`analysis/old_bs_generalization_preregistration.md`](old_bs_generalization_preregistration.md)，
执行入口为
[`analysis/run_old_bs_generalization.py`](run_old_bs_generalization.py)。

48/48 次运行成功。按严格筛选规则，25% 通过 8/12 次、50% 通过 10/12 次、
75% 通过 12/12 次，因此冻结 75% 进入独立环境确认。详细数值和结论边界见
[`analysis/20260911_old_bs_generalization_results.md`](20260911_old_bs_generalization_results.md)，
阶段 B 冻结方案见
[`analysis/old_bs_generalization_phase_b_preregistration.md`](old_bs_generalization_phase_b_preregistration.md)。
阶段 B 已启动并在首个完整样本写盘后按用户要求暂停；恢复点与初步边界信号见
[`analysis/20260911_old_bs_phase_b_pause.md`](20260911_old_bs_phase_b_pause.md)。
