# core/agents/lda_agent.py

import numpy as np
import torch
import torch.optim as optim
import random
from collections import deque

from core.models.dnn_model import OffloadingActor, get_input_vector, FocalLoss
from core.optimizers.bs_optimizer import BS_Optimizer
from core.optimizers.leo_optimizer import LEO_Optimizer
from core.models.tcopq import generate_candidates, check_local_feasibility
from utils.physics_validator import (
    validate_sat_time_constraint,
    validate_bs_time_constraint,
    validate_task_span_constraint
)


class LDAAgent:
    """
    LDA 算法的智能体
    专职负责：观测环境状态 -> DNN 推理 -> 生成策略候选 -> 寻找最优解 -> 收集经验并自我训练
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # 1. 初始化各组件
        self.actors = torch.nn.ModuleList([OffloadingActor(cfg.J) for _ in range(cfg.I)])
        self.bs_opt = BS_Optimizer(cfg)
        self.leo_opt = LEO_Optimizer(cfg)

        # 2. 训练组件初始化
        self.optimizers = [optim.Adam(actor.parameters(), lr=self.cfg.lr) for actor in self.actors]
        self.criterion = FocalLoss(alpha=self.cfg.focal_alpha, gamma=self.cfg.focal_gamma)

        # 每个基站维护独立的经验回放池
        self.memories = [deque(maxlen=self.cfg.memory_capacity) for _ in range(cfg.I)]

        self.batch_size = self.cfg.batch_size
        self.train_interval = self.cfg.train_interval

        # 3. 算法特定参数
        self.delta_t = 0.5
        self.gamma = 0.95

        # 记录训练过程中的 Loss
        self.loss_history = []

    def select_action(self, env, L_t, R_bs, R_sat, T_prop):
        """
        核心决策逻辑：根据环境给出的状态，输出本帧的最佳卸载与资源分配策略
        """
        I, J = self.cfg.I, self.cfg.J

        # --- Step 1: 状态构建与 Actor 预测 ---
        # 获取物理环境的真实状态
        state_tensor = get_input_vector(env.Q_total, env.E_BS, env.T_BS_left_prev, R_bs, R_sat)

        prob_b = np.zeros((I, J))
        for i in range(I):
            self.actors[i].eval()
            with torch.no_grad():
                prob_i = self.actors[i](state_tensor[i].unsqueeze(0))
                prob_b[i] = prob_i.numpy().flatten()

        # --- Step 2: TCOPQ 候选生成 ---
        f_local = np.ones((I, J)) * self.cfg.f_max_UE
        l_decisions = check_local_feasibility(L_t, f_local)

        bs_candidates = []
        for i in range(I):
            cands_i = generate_candidates(prob_b[i], self.delta_t, l_decisions[i])
            bs_candidates.append(cands_i)

        # --- Step 3: 候选评估 (Parallel Solving) ---
        K_candidates = min(len(cands) for cands in bs_candidates)
        if K_candidates == 0:
            pass  # 防止极端情况

        best_G1 = float('inf')
        best_sol = None

        for k in range(K_candidates):
            l_mat = np.array([bs_candidates[i][k][0] for i in range(I)])
            b_mat = np.array([bs_candidates[i][k][1] for i in range(I)])

            mask_bs = (l_mat == 0) & (b_mat == 1)
            mask_sat = (l_mat == 0) & (b_mat == 0)

            L_to_bs = np.where(mask_bs, L_t, 0.0)
            L_to_sat = np.where(mask_sat, L_t, 0.0)

            # A. 基站资源分配
            f_bs = np.zeros((I, J))
            T_tran_bs = np.zeros((I, J))
            for i in range(I):
                T_tran_bs[i] = np.where(mask_bs[i], L_to_bs[i] / R_bs[i], 0.0)
                f_bs[i] = self.bs_opt.optimize(L_to_bs[i], env.Q_bs[i], env.E_BS[i], T_tran_bs[i],
                                               env.T_BS_left_prev[i])

            # B. 卫星资源分配
            T_tran_sat = np.where(mask_sat, L_to_sat / R_sat, 0.0)
            T_avail_sat_raw = self.cfg.tau - T_tran_sat - T_prop
            validate_sat_time_constraint(T_avail_sat_raw, mask_sat, T_prop, T_tran_sat)
            T_avail_sat = np.maximum(0, T_avail_sat_raw)

            f_sat_flat = self.leo_opt.optimize(L_to_sat.flatten(), env.Q_sat.flatten(), T_avail_sat.flatten())
            f_sat = f_sat_flat.reshape(I, J)

            # C. 计算全局目标函数 G1 (传入 env 以读取旧状态)
            G1, details = self.calculate_objective(
                env, L_t, l_mat, mask_bs, mask_sat,
                f_bs, f_sat, f_local, T_tran_bs, T_avail_sat
            )

            if G1 < best_G1:
                best_G1 = G1
                best_sol = {
                    'l': l_mat, 'b': b_mat,
                    'f_bs': f_bs, 'f_sat': f_sat,
                    'details': details,
                    'G1': G1  # 附加 G1，方便 env 记录 Reward
                }

        # --- Step 4: 存储经验与阈值衰减 ---
        best_action_b = best_sol['b']
        self.store_experience(state_tensor, best_action_b)
        self.delta_t = max(0.1, self.delta_t * self.gamma)

        # 收集调试统计信息
        self._attach_debug_info(best_sol, L_t, prob_b)

        return best_sol

    def calculate_objective(self, env, L_t, l_vec, mask_bs, mask_sat, f_bs, f_sat, f_local, T_tran_bs, T_avail_sat):
        """
        计算单帧目标函数 G1(t)。注意：这里通过读取 env 中的物理队列状态来进行计算。
        """
        phi = self.cfg.phi
        kappa1 = self.cfg.kappa1
        kappa2 = self.cfg.kappa2

        # 1. BS 处理情况
        load_bs_new = np.where(mask_bs, L_t, 0.0)
        T_left_prev_mat = env.T_BS_left_prev.reshape(-1, 1)

        t_bs_avail_raw = self.cfg.tau - np.maximum(T_tran_bs, T_left_prev_mat)
        validate_bs_time_constraint(t_bs_avail_raw, mask_bs, T_left_prev_mat)

        t_bs_avail_phys = np.maximum(0, t_bs_avail_raw)
        cap_bs = f_bs * t_bs_avail_phys / phi
        l_proc_bs_new = np.minimum(load_bs_new, cap_bs)
        l_left_bs_new = np.maximum(0, load_bs_new - l_proc_bs_new)

        # 2. Sat 处理情况
        load_sat_new = np.where(mask_sat, L_t, 0.0)
        cap_sat = f_sat * T_avail_sat / phi
        l_proc_sat_new = np.minimum(load_sat_new, cap_sat)
        l_left_sat_new = np.maximum(0, load_sat_new - l_proc_sat_new)

        # 3. Local 处理情况
        l_proc_loc_new = np.where(l_vec == 1, L_t, 0.0)

        # 4. 旧任务的处理量
        total_l_prev_bs = np.sum(env.L_BS_left_prev_vec, axis=1, keepdims=True)
        f_old_bs_vec = np.where(total_l_prev_bs > 1e-9,
                                self.cfg.f_max_BS * (env.L_BS_left_prev_vec / (total_l_prev_bs + 1e-12)),
                                0.0)
        cap_old_bs = (f_old_bs_vec * self.cfg.tau) / phi
        l_proc_old_bs = np.minimum(env.L_BS_left_prev_vec, cap_old_bs)

        failed_to_clear_bs = (l_proc_old_bs < env.L_BS_left_prev_vec) & (env.L_BS_left_prev_vec > 1e-9)
        validate_task_span_constraint(failed_to_clear_bs, env.L_BS_left_prev_vec, cap_old_bs)

        l_proc_old_sat = np.zeros_like(L_t)
        total_l_sat_prev = np.sum(env.L_Sat_left_prev_vec)
        if total_l_sat_prev > 1e-9:
            f_old_sat_vec = self.cfg.f_max_Sat * (env.L_Sat_left_prev_vec / total_l_sat_prev)
            cap_old_sat = (f_old_sat_vec * self.cfg.tau) / phi
            l_proc_old_sat = np.minimum(env.L_Sat_left_prev_vec, cap_old_sat)

        l_proc_total = l_proc_bs_new + l_proc_sat_new + l_proc_loc_new + l_proc_old_bs + l_proc_old_sat

        # B. 计算能耗
        e_bs_new = np.sum(kappa1 * phi * (f_bs ** 2) * l_proc_bs_new, axis=1)
        e_old = np.zeros(self.cfg.I)
        if np.sum(total_l_prev_bs) > 1e-9:
            e_old = np.sum(kappa1 * phi * (f_old_bs_vec ** 2) * env.L_BS_left_prev_vec, axis=1)
        e_bs_total = e_bs_new + e_old

        e_sat = np.sum(kappa2 * phi * (f_sat ** 2) * l_proc_sat_new)

        # C. 计算 PAoI
        total_left_bs_new = np.sum(l_left_bs_new, axis=1, keepdims=True)
        t_next_left_bs_scalar = np.where(total_left_bs_new > 1e-9, (phi * total_left_bs_new) / self.cfg.f_max_BS, 0.0)
        t_next_left_bs_est = np.where(l_left_bs_new > 1e-9, t_next_left_bs_scalar, 0.0)

        total_left_sat_new = np.sum(l_left_sat_new)
        t_next_left_sat_scalar = (phi * total_left_sat_new) / self.cfg.f_max_Sat if total_left_sat_new > 1e-9 else 0.0
        t_next_left_sat_est = np.where(l_left_sat_new > 1e-9, t_next_left_sat_scalar, 0.0)

        paoi_loc = np.where(l_vec == 1, (phi * L_t) / f_local, 0.0)

        time_finish_bs = np.maximum(T_tran_bs, T_left_prev_mat) + (l_proc_bs_new * phi / (f_bs + 1e-9))
        paoi_bs = np.where(l_left_bs_new > 1e-9, self.cfg.tau + self.cfg.w * t_next_left_bs_est, time_finish_bs)
        paoi_bs = np.where(mask_bs, paoi_bs, 0.0)

        paoi_sat = np.where(l_left_sat_new > 1e-9, self.cfg.tau + self.cfg.w * t_next_left_sat_est,
                            (self.cfg.tau - T_avail_sat) + l_proc_sat_new * phi / (f_sat + 1e-9))
        paoi_sat = np.where(mask_sat, paoi_sat, 0.0)

        paoi_total = paoi_bs + paoi_sat + paoi_loc

        # D. 组合 G1
        term_q_bs = np.sum((env.Q_bs / 1e7) * ((env.L_BS_left_prev_vec - l_proc_bs_new - l_proc_old_bs) / 1e7))
        term_q_sat = np.sum((env.Q_sat / 1e7) * ((env.L_Sat_left_prev_vec - l_proc_sat_new - l_proc_old_sat) / 1e7))
        term_q = term_q_bs + term_q_sat
        term_p = self.cfg.K_p * np.sum(paoi_total)
        term_e_bs = np.sum(env.E_BS * (e_bs_total - self.cfg.E_max_BS))

        G1 = 5 * term_q + term_p + term_e_bs
        real_drift = 0.5 * np.sum((L_t - l_proc_total) ** 2) + 0.5 * np.sum((e_bs_total - self.cfg.E_max_BS) ** 2)

        details = {
            'l_proc_total': l_proc_total, 'l_proc_bs': l_proc_bs_new, 'l_proc_sat': l_proc_sat_new,
            'l_proc_old_sat': l_proc_old_sat, 'l_left_bs': l_left_bs_new, 'l_left_sat': l_left_sat_new,
            'e_bs_total': e_bs_total, 'e_sat': e_sat, 'paoi': paoi_total, 'real_drift': real_drift,
            't_next_left_bs_scalar': t_next_left_bs_scalar.flatten()
        }

        return G1, details

    def store_experience(self, state_tensor, best_action_b):
        states = state_tensor.detach().numpy()
        for i in range(self.cfg.I):
            self.memories[i].append((states[i], best_action_b[i].copy()))

    def train(self, current_frame):
        """分布更新 DNN，仅在满足间隔与样本数量时触发"""
        if current_frame % self.cfg.train_interval != 0:
            return

        loss_vals = []
        half_capacity = self.cfg.memory_capacity / 2.0

        for i in range(self.cfg.I):
            if len(self.memories[i]) < half_capacity:
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

        if loss_vals:
            avg_loss = np.mean(loss_vals)
            self.loss_history.append((current_frame, avg_loss))

    def _attach_debug_info(self, sol, L_t, prob_b):
        """收集打印日志所需的统计信息"""
        l, b = sol['l'], sol['b']
        cnt_local = np.sum(l == 1)
        cnt_bs = np.sum((l == 0) & (b == 1))
        cnt_sat = np.sum((l == 0) & (b == 0))

        util_bs = np.sum(sol['f_bs']) / (self.cfg.I * self.cfg.f_max_BS + 1e-9)
        util_sat = np.sum(sol['f_sat']) / (self.cfg.f_max_Sat + 1e-9)

        arrival_mb = np.sum(L_t) / 1e6
        served_mb = np.sum(sol['details']['l_proc_total']) / 1e6

        sol['debug'] = {
            'dist': (cnt_local, cnt_bs, cnt_sat),
            'util': (util_bs, util_sat),
            'flow': (arrival_mb, served_mb),
            'q_trend': served_mb - arrival_mb,
            'prob_mean': np.mean(prob_b)
        }