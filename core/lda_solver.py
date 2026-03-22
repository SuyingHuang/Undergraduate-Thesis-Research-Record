
import numpy as np
import torch
import torch.optim as optim
import random
from collections import deque # [新增] 用于经验回放

from config import SystemConfig
from core.dnn_model import OffloadingActor, get_input_vector,FocalLoss
from core.bs_optimizer import BS_Optimizer
from core.leo_optimizer import LEO_Optimizer
from core.bs_channel import BSChannel
from core.satellite_channel import SatelliteChannel
from core.tcopq import generate_candidates, check_local_feasibility

from core.physics_validator import (
    validate_sat_time_constraint,
    validate_bs_time_constraint,
    validate_task_span_constraint
)

class LDASolver:
    """
        LDA 算法总控器 (The Solver)
        实现论文 Algorithm 1 的主流程。
        """

    def __init__(self, cfg):
        self.cfg = cfg

        # 1. 初始化各组件
        self.actors = torch.nn.ModuleList([OffloadingActor(cfg.J) for _ in range(cfg.I)])

        self.bs_opt = BS_Optimizer(cfg)
        self.leo_opt = LEO_Optimizer(cfg)
        self.bs_channel = BSChannel(cfg)
        self.sat_channel = SatelliteChannel(cfg)

        # 2. 训练组件初始化
        # 每个 DNN 需要独立的优化器
        self.optimizers = [optim.Adam(actor.parameters(), lr=self.cfg.lr) for actor in self.actors]
        self.criterion = FocalLoss(alpha=self.cfg.focal_alpha, gamma=self.cfg.focal_gamma)

        # 每个基站维护独立的经验回放池 (论文 Algorithm 1 第 18 行：Sample mini-batch C_i)
        self.memories = [deque(maxlen=self.cfg.memory_capacity) for _ in range(cfg.I)]

        self.batch_size = self.cfg.batch_size
        self.train_interval = self.cfg.train_interval

        # 3. 初始化环境状态
        self.reset()

        # 4. 优化参数
        self.delta_t = 0.5
        self.gamma = 0.95
        self.frame_count = 0  # 计数器

    def reset(self):
        """重置环境状态"""
        I, J = self.cfg.I, self.cfg.J
        self.Q_bs = np.zeros((I, J))
        self.Q_sat = np.zeros((I, J))
        self.Q_total = np.zeros((I, J))

        self.E_BS = np.zeros(I)  # [修改] 每个基站独立的能量队列

        self.T_BS_left_prev = np.zeros(I)
        self.L_BS_left_prev_vec = np.zeros((I, J))
        self.L_Sat_left_prev_vec = np.zeros((I, J))

        self.l_proc_total_prev = np.zeros((I, J))
        #记录上一帧卫星的频率决策
        self.f_sat_prev = np.zeros((I,J))

        self.history = {
            'Q_total': [],
            'Q_bs': [],
            'Q_sat': [],
            'PAoI': [],
            'Cost': [],
            'E_virt_bs': [],
            'E_virt_sat': [],
            'Loss': [],
            'Drift': [],
            'Reward':[],
            'R_bs_max': [],
            'R_bs_min': [],
            'R_sat_max': [],
            'R_sat_min': []
        }
        self.frame_count = 0

    def step(self,L_t):
        """
        执行一帧的完整调度 (Algorithm 1)
        :param L_t: 当前帧新到达的任务量 (bits), shape=(J,)
        """
        I, J = self.cfg.I, self.cfg.J
        self.frame_count += 1

        # --- Step 1: 环境感知 (Channel & Rates) ---
        # 1. 基站信道：依然是随机分布 (用户在基站覆盖范围内移动)
        d_bs = np.random.uniform(self.cfg.d_min, self.cfg.d_max, (I, J))
        R_bs = self.bs_channel.calculate_uplink_rate(d_bs)

        # 2. 卫星信道：[修改点]
        # 假设：每一帧都接入一个新的、位于固定轨道高度的卫星
        # 距离 d 固定为 H_sat (忽略仰角带来的斜距变化，做最简处理)
        d_sat = np.full((I, J), self.cfg.H_sat)

        # 衰落 (Shadowing/Multipath) 依然随机
        # 虽然距离不变，但大气环境和多径效应依然是动态的
        h_sq = self.sat_channel.generate_channel_gain_samples(I * J).reshape(I, J)
        R_sat, T_prop = self.sat_channel.calculate_uplink_rate(d_sat, h_sq)

        self.history['R_bs_max'].append(np.max(R_bs))
        self.history['R_bs_min'].append(np.min(R_bs))
        self.history['R_sat_max'].append(np.max(R_sat))
        self.history['R_sat_min'].append(np.min(R_sat))

        # --- Step 2: 状态构建与 Actor 预测 ---
        # 构造状态向量 X_t (归一化交给 DNN 内部 BN 层)
        self.Q_total = self.Q_bs + self.Q_sat
        # state_tensor shape: (I, input_dim)
        state_tensor = get_input_vector(self.Q_total, self.E_BS, self.T_BS_left_prev, R_bs, R_sat)

        # Actor 前向传播
        prob_b = np.zeros((I, J))

        # [精准修正 1]：遍历 I 个独立的 DNN 进行推理
        for i in range(I):
            self.actors[i].eval()  # 针对第 i 个基站的网络调用 eval()
            with torch.no_grad():
                # state_tensor[i] 是 (input_dim,)，unsqueeze(0) 增加 batch 维度
                prob_i = self.actors[i](state_tensor[i].unsqueeze(0))
                prob_b[i] = prob_i.numpy().flatten()

        # --- Step 3: TCOPQ 候选生成 ---
        # 确定本地计算约束 (I, J) 矩阵
        f_local = np.ones((I, J)) * self.cfg.f_max_UE
        l_decisions = check_local_feasibility(L_t, f_local)

        bs_candidates = []
        # 生成候选集
        for i in range(I):
            # 传入当前基站的连续预测值 prob_b[i] 和 本地决策 l_decisions[i]
            cands_i = generate_candidates(prob_b[i], self.delta_t, l_decisions[i])
            bs_candidates.append(cands_i)

        # --- Step 4: 候选评估 (Parallel Solving) ---
        # [精准修正]：寻找所有基站中生成候选数量最少的那一个，作为对齐评估的上限
        K_candidates = min(len(cands) for cands in bs_candidates)
        # 增加一个安全校验，防止极端情况下某个基站有效候选为0
        if K_candidates == 0:
            # 如果没有有效候选，退化为全本地计算等基准策略，或进行异常处理
            pass

        best_G1 = float('inf')
        best_sol = None

        for k in range(K_candidates):
            # 将各基站的第 k 个局部决策堆叠为 (I, J) 的全局决策矩阵
            l_mat = np.array([bs_candidates[i][k][0] for i in range(I)])
            b_mat = np.array([bs_candidates[i][k][1] for i in range(I)])

            # 提取卸载掩码 (I, J)
            mask_bs = (l_mat == 0) & (b_mat == 1)
            mask_sat = (l_mat == 0) & (b_mat == 0)

            L_to_bs = np.where(mask_bs, L_t, 0.0)
            L_to_sat = np.where(mask_sat, L_t, 0.0)

            # --- A. 基站资源分配: 各基站解耦独立求解 ---
            f_bs = np.zeros((I, J))
            T_tran_bs = np.zeros((I, J))
            for i in range(I):
                T_tran_bs[i] = np.where(mask_bs[i], L_to_bs[i] / R_bs[i], 0.0)
                # 调用各个基站局部的底层求解器
                f_bs[i] = self.bs_opt.optimize(L_to_bs[i], self.Q_bs[i], self.E_BS[i], T_tran_bs[i],
                                               self.T_BS_left_prev[i])

            # --- B. 卫星资源分配: 集中式求解 (所有基站的任务拍扁后全局竞争) ---
            T_tran_sat = np.where(mask_sat, L_to_sat / R_sat, 0.0)

            # [物理防线] 拦截卫星时间穿透
            T_avail_sat_raw = self.cfg.tau - T_tran_sat - T_prop
            validate_sat_time_constraint(T_avail_sat_raw, mask_sat, T_prop, T_tran_sat)

            T_avail_sat = np.maximum(0, T_avail_sat_raw)

            f_sat_flat = self.leo_opt.optimize(L_to_sat.flatten(), self.Q_sat.flatten(), T_avail_sat.flatten())
            f_sat = f_sat_flat.reshape(I, J)

            # --- C. 计算全局目标函数 G1 ---
            G1, details = self.calculate_objective(
                L_t, self.Q_bs, self.Q_sat, self.E_BS,
                l_mat, mask_bs, mask_sat,
                f_bs, f_sat, f_local,
                T_tran_bs, T_avail_sat,
                self.T_BS_left_prev, self.L_BS_left_prev_vec
            )

            # 寻找使全局 G1 最小的第 k 个对齐策略
            if G1 < best_G1:
                best_G1 = G1
                best_sol = {
                    'l': l_mat, 'b': b_mat,
                    'f_bs': f_bs, 'f_sat': f_sat,
                    'details': details
                }

        self.history['Reward'].append(-best_G1)
        # --- [新增] Step 4.5: 存储经验 ---
        # 我们要让神经网络学习的是：在当前 state 下，best_sol['b'] 是最好的选择
        # 注意：只存 b (Offloading Decision)，因为 l (Local) 是由 feasibility 决定的硬约束
        best_action_b = best_sol['b']
        self.store_experience(state_tensor, best_action_b)

        # Step 4.6: 触发训练 ---
        # 1. 获取当前回放池的样本数量
        current_memory_size = len(self.memories[0])
        # 2. 获取最大容量的一半
        half_capacity = self.cfg.memory_capacity / 2.0
        if (current_memory_size >= half_capacity) and ( self.frame_count% self.cfg.train_interval == 0):
            loss_val = self.train_network()
            self.history['Loss'].append((self.frame_count, loss_val))
        # --- Step 5: 环境更新 ---
        self.update_env(best_sol, L_t)

        # --- Step 6: 动态阈值衰减 (Eq. 44) ---
        # 简单实现：每帧衰减，或者每隔 Delta T 衰减
        self.delta_t = max(0.1, self.delta_t * self.gamma)
        # ================= [新增调试信息] =================
        # 统计本帧的决策分布
        # l_vec: 1=Local, 0=Offload
        # b_vec: 1=BS, 0=Sat
        l = best_sol['l']
        b = best_sol['b']

        cnt_local = np.sum(l == 1)
        cnt_bs = np.sum((l == 0) & (b == 1))
        cnt_sat = np.sum((l == 0) & (b == 0))

        # 统计资源利用率
        # f_bs shape (J,), f_max_BS 是标量
        util_bs = np.sum(best_sol['f_bs']) / (self.cfg.I*self.cfg.f_max_BS + 1e-9)
        util_sat = np.sum(best_sol['f_sat']) / (self.cfg.f_max_Sat + 1e-9)

        # 统计吞吐量供需关系
        arrival_mb = np.sum(L_t) / 1e6
        served_mb = np.sum(best_sol['details']['l_proc_total']) / 1e6

        debug_info = {
            'dist': (cnt_local, cnt_bs, cnt_sat),  # 决策分布
            'util': (util_bs, util_sat),  # 频率占用率 (0~1)
            'flow': (arrival_mb, served_mb),  # 流量 (Mb)
            'q_trend': served_mb - arrival_mb,  # 净流量 (>0 表示在清理积压)
            'prob_mean': np.mean(prob_b)  # 神经网络平均输出值
        }
        best_sol['debug'] = debug_info
        # =================================================

        return best_sol

    def calculate_objective(self, L_t, Q_bs, Q_sat, E_BS,
                            l_vec, mask_bs, mask_sat,
                            f_bs, f_sat, f_local,
                            T_tran_bs, T_avail_sat, T_left_prev, L_left_prev_vec):
        """
        计算单帧目标函数 G1(t) (Eq. 39)
        修正版：全面适配 (I, J) 多基站架构，彻底修复数组广播和按轴求和(axis=1)问题。
        """
        phi = self.cfg.phi
        kappa1 = self.cfg.kappa1
        kappa2 = self.cfg.kappa2

        # --- A. 计算处理量与残留量 ---
        # 1. BS 处理情况 (只针对新任务 L_t)
        load_bs_new = np.where(mask_bs, L_t, 0.0)

        # [修正点] 解决广播问题：T_left_prev 从 (I,) 变为 (I, 1)，以匹配 T_tran_bs 的 (I, J)
        T_left_prev_mat = T_left_prev.reshape(-1, 1)
        # [物理防线] 拦截基站时间穿透
        t_bs_avail_raw = self.cfg.tau - np.maximum(T_tran_bs, T_left_prev_mat)
        validate_bs_time_constraint(t_bs_avail_raw, mask_bs, T_left_prev_mat)

        t_bs_avail_phys = np.maximum(0, t_bs_avail_raw)
        cap_bs = f_bs * t_bs_avail_phys / phi
        l_proc_bs_new = np.minimum(load_bs_new, cap_bs)
        l_left_bs_new = np.maximum(0, load_bs_new - l_proc_bs_new)

        # 2. Sat 处理情况 (只针对新任务)
        load_sat_new = np.where(mask_sat, L_t, 0.0)
        cap_sat = f_sat * T_avail_sat / phi
        l_proc_sat_new = np.minimum(load_sat_new, cap_sat)
        l_left_sat_new = np.maximum(0, load_sat_new - l_proc_sat_new)

        # 3. Local 处理情况
        l_proc_loc_new = np.where(l_vec == 1, L_t, 0.0)

        # 4. 旧任务的处理量 (Old Tasks Processing)
        total_l_prev_bs = np.sum(L_left_prev_vec, axis=1, keepdims=True)
        f_old_bs_vec = np.where(total_l_prev_bs > 1e-9,
                                self.cfg.f_max_BS * (L_left_prev_vec / (total_l_prev_bs + 1e-12)),
                                0.0)
        cap_old_bs = (f_old_bs_vec * self.cfg.tau) / phi
        l_proc_old_bs = np.minimum(L_left_prev_vec, cap_old_bs)

        # [物理防线] 拦截任务拖延至第三帧
        failed_to_clear_bs = (l_proc_old_bs < L_left_prev_vec) & (L_left_prev_vec > 1e-9)
        validate_task_span_constraint(failed_to_clear_bs, L_left_prev_vec, cap_old_bs)

        # Sat 旧任务: 全局卫星共享，直接 flat 累加
        l_proc_old_sat = np.zeros_like(L_t)
        total_l_sat_prev = np.sum(self.L_Sat_left_prev_vec)

        if total_l_sat_prev > 1e-9:
            f_old_sat_vec = self.cfg.f_max_Sat * (self.L_Sat_left_prev_vec / total_l_sat_prev)
            cap_old_sat = (f_old_sat_vec * self.cfg.tau) / phi
            l_proc_old_sat = np.minimum(self.L_Sat_left_prev_vec, cap_old_sat)

        l_proc_total = l_proc_bs_new + l_proc_sat_new + l_proc_loc_new + l_proc_old_bs + l_proc_old_sat

        # --- B. 计算能耗 ---
        # 1. BS 能耗 [修正点] 按基站累加，axis=1，得到形如 (I,) 的向量
        e_bs_new = np.sum(kappa1 * phi * (f_bs ** 2) * l_proc_bs_new, axis=1)

        # e_old 计算同理
        e_old = np.zeros(self.cfg.I)
        if np.sum(total_l_prev_bs) > 1e-9:
            e_old = np.sum(kappa1 * phi * (f_old_bs_vec ** 2) * L_left_prev_vec, axis=1)

        e_bs_total = e_bs_new + e_old  # 形状: (I,)

        # 2. Sat Energy (全局唯一卫星，直接全部求和得到标量)
        e_sat = np.sum(kappa2 * phi * (f_sat ** 2) * l_proc_sat_new)

        # --- C. 计算 PAoI ---
        # 1. 基站估算标量时间 [修正点] 按基站统计，axis=1, shape: (I, 1)
        total_left_bs_new = np.sum(l_left_bs_new, axis=1, keepdims=True)
        t_next_left_bs_scalar = np.where(total_left_bs_new > 1e-9,
                                         (phi * total_left_bs_new) / self.cfg.f_max_BS,
                                         0.0)
        # 广播赋值给 (I, J) 矩阵
        t_next_left_bs_est = np.where(l_left_bs_new > 1e-9, t_next_left_bs_scalar, 0.0)

        # 2. 卫星估算标量时间
        total_left_sat_new = np.sum(l_left_sat_new)
        t_next_left_sat_scalar = 0.0
        if total_left_sat_new > 1e-9:
            t_next_left_sat_scalar = (phi * total_left_sat_new) / self.cfg.f_max_Sat
        t_next_left_sat_est = np.where(l_left_sat_new > 1e-9, t_next_left_sat_scalar, 0.0)

        # 3. 本地 PAoI
        paoi_loc = np.where(l_vec == 1, (phi * L_t) / f_local, 0.0)

        # 4. BS PAoI [修正点] 采用 T_left_prev_mat 参与计算
        time_finish_bs = np.maximum(T_tran_bs, T_left_prev_mat) + (l_proc_bs_new * phi / (f_bs + 1e-9))
        paoi_bs = np.where(l_left_bs_new > 1e-9,
                           self.cfg.tau + self.cfg.w * t_next_left_bs_est,
                           time_finish_bs)
        paoi_bs = np.where(mask_bs, paoi_bs, 0.0)

        # 5. Sat PAoI
        paoi_sat = np.where(l_left_sat_new > 1e-9,
                            self.cfg.tau + self.cfg.w * t_next_left_sat_est,
                            (self.cfg.tau - T_avail_sat) + l_proc_sat_new * phi / (f_sat + 1e-9))
        paoi_sat = np.where(mask_sat, paoi_sat, 0.0)

        # 6. 合并总 PAoI
        paoi_total = paoi_bs + paoi_sat + paoi_loc

        # --- D. 组合 G1 ---
        # Drift Term: Q * (Arrival - Service)
        # 1. 基站漂移项：只看基站的队列 Q_bs 以及基站相关的处理量
        # 这里用上帧残留量作为惩罚基准(参考您原本逻辑)，减去本帧基站处理的新/旧任务总量
        term_q_bs = np.sum((Q_bs / 1e7) * ((self.L_BS_left_prev_vec - l_proc_bs_new - l_proc_old_bs) / 1e7))

        # 2. 卫星漂移项：只看卫星的队列 Q_sat 以及卫星相关的处理量
        term_q_sat = np.sum((Q_sat / 1e7) * ((self.L_Sat_left_prev_vec - l_proc_sat_new - l_proc_old_sat) / 1e7))

        # 3. 本地计算几乎不产生跨帧积压队列，因此无需单独的漂移惩罚，直接相加
        term_q = term_q_bs + term_q_sat

        term_p = self.cfg.K_p * np.sum(paoi_total)

        # [修正点] 能耗惩罚项变为内积运算，得到标量 G1
        term_e_bs = np.sum(E_BS * (e_bs_total - self.cfg.E_max_BS))

        G1 = 5*term_q + term_p + term_e_bs

        # real_drift 中的能量部分也必须是内积累加求和
        real_drift = 0.5 * np.sum((L_t - l_proc_total) ** 2) + 0.5 * np.sum((e_bs_total - self.cfg.E_max_BS) ** 2)

        # 更新 details 字典
        details = {
            'l_proc_total': l_proc_total,
            'l_proc_bs': l_proc_bs_new,
            'l_proc_sat': l_proc_sat_new,
            'l_proc_old_sat': l_proc_old_sat,
            'l_left_bs': l_left_bs_new,
            'l_left_sat': l_left_sat_new,
            'e_bs_total': e_bs_total,
            'e_sat': e_sat,
            'paoi': paoi_total,
            'real_drift': real_drift,
            't_next_left_bs_scalar': t_next_left_bs_scalar.flatten()  # [修正点] 拍平为 (I,) 一维数组，供 update_env 接收
        }

        return G1, details

    def update_env(self, sol, L_t):
        """
        根据最佳策略更新系统状态 (Algorithm 1)
        修正版：全面适配 (I, J) 多基站架构，处理 NumPy 数组形态的状态更新。
        """
        details = sol['details']
        l_vec = sol['l']
        b_vec = sol['b']

        # 提取卸载掩码
        mask_bs = (l_vec == 0) & (b_vec == 1)
        mask_sat = (l_vec == 0) & (b_vec == 0)

        # ==========================================================
        # 1. 更新虚拟任务队列 Q (Lyapunov Queue)
        # ==========================================================
        # BS Queue: Q(t+1) = max(0, Q(t) + Arrival - Service)
        # Service 包含处理的新任务和被处理掉的旧任务(上帧残留)
        service_bs_total = details['l_proc_bs'] + self.L_BS_left_prev_vec
        arrival_bs = np.where(mask_bs, L_t, 0.0)
        self.Q_bs = np.maximum(0, self.Q_bs + arrival_bs - service_bs_total)

        # Sat Queue:
        # Service 包含新卫星处理的新任务，以及旧卫星在后台处理掉的积压任务
        service_sat_total = details['l_proc_sat'] + details['l_proc_old_sat']
        arrival_sat = np.where(mask_sat, L_t, 0.0)
        self.Q_sat = np.maximum(0, self.Q_sat + arrival_sat - service_sat_total)

        # 汇总总积压，用于下一帧 DNN 的状态输入
        self.Q_total = self.Q_bs + self.Q_sat

        # ==========================================================
        # 2. 更新虚拟能量队列 E
        # ==========================================================
        # 基站 (固定不动，有长期的平均能耗约束，必须跨帧累加)
        e_bs_total = details['e_bs_total']  # 现在是形状为 (I,) 的数组

        # [核心修正点 1]：将 Python 内置的 max() 替换为 np.maximum()，支持 (I,) 数组的逐元素更新
        self.E_BS = np.maximum(0.0, self.E_BS + e_bs_total - self.cfg.E_max_BS)

        # 卫星单帧过境，只有瞬时能耗约束，无需跨帧维护虚拟能量队列

        # ==========================================================
        # 3. 更新物理状态 (为下一帧做准备)
        # ==========================================================
        # --- 基站 (BS) 状态更新 ---
        l_left_bs_next = details['l_left_bs']  # 提取本帧产生的新残留任务

        # [核心修正点 2]：默认取全零数组时，需要给定形状 (I,) 以匹配多基站数量
        self.T_BS_left_prev = details.get('t_next_left_bs_scalar', np.zeros(self.cfg.I))

        # 保存本帧的残留任务向量，它将在下一帧变为旧任务 L_BS_left_prev_vec
        if np.sum(l_left_bs_next) > 1e-9:
            self.L_BS_left_prev_vec = l_left_bs_next.copy()
        else:
            # [核心修正点 3]：清空残留队列时，必须重置为 (I, J) 的二维矩阵，而不是一维的 cfg.J
            self.L_BS_left_prev_vec = np.zeros((self.cfg.I, self.cfg.J))

        self.l_proc_total_prev = details['l_proc_total'].copy()

        # --- 卫星 (Sat) 状态更新 ---
        l_left_sat_next = details['l_left_sat']
        if np.sum(l_left_sat_next) > 1e-9:
            self.L_Sat_left_prev_vec = l_left_sat_next.copy()
        else:
            # [核心修正点 4]：同理，卫星旧任务重置也必须是 (I, J) 维度
            self.L_Sat_left_prev_vec = np.zeros((self.cfg.I, self.cfg.J))

        # ==========================================================
        # 4. 记录历史 (用于 matplotlib 画图)
        # ==========================================================
        self.history['Q_total'].append(np.mean(self.Q_total))
        self.history['Q_bs'].append(np.mean(self.Q_bs))
        self.history['Q_sat'].append(np.mean(self.Q_sat))
        self.history['Cost'].append(np.mean(details['paoi']))

        # [核心修正点 5]：self.E_BS 现在是数组，画图记录时取平均值（也可以用 np.max 取违约最严重的基站）
        self.history['E_virt_bs'].append(np.mean(self.E_BS))

        # 卫星能量不作为队列维护，但依然可以记录单帧真实消耗 e_sat 用于画图观察
        self.history['E_virt_sat'].append(details['e_sat'])
        self.history['Drift'].append(details['real_drift'])

    def store_experience(self, state_tensor, best_action_b):
        """将 (State, Best_Action) 存入 Replay Buffer"""
        # state_tensor: state_tensor: (I, input_dim) -> 拆解为 I 个 (input_dim,) 存入对应基站
        # best_action_b: (I, J) 0/1 vector
        states = state_tensor.detach().numpy()
        for i in range(self.cfg.I):
            s_i = states[i]
            a_i = best_action_b[i].copy()
            self.memories[i].append((s_i, a_i))

    def train_network(self):
        """分布式更新 3 个 DNN 的参数"""
        loss_vals = []
        # [核心修改]：分别对 I 个基站的 DNN 进行经验回放和梯度下降
        for i in range(self.cfg.I):
            if len(self.memories[i]) < self.batch_size:
                continue

            batch = random.sample(self.memories[i], self.batch_size)
            state_batch, action_batch = zip(*batch)

            states = torch.FloatTensor(np.array(state_batch))
            targets = torch.FloatTensor(np.array(action_batch))

            self.actors[i].train()
            self.optimizers[i].zero_grad()

            probs = self.actors[i](states)
            loss = self.criterion(probs, targets)

            loss.backward()
            self.optimizers[i].step()
            loss_vals.append(loss.item())

        # 返回 3 个网络的平均 Loss 用于监控
        return np.mean(loss_vals) if loss_vals else 0.0