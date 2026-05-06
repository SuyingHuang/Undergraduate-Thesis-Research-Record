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
    validate_task_span_constraint
)


class LDAAgent:
    """
    LDA 算法的智能体
    专职负责：观测环境状态 -> DNN 推理 -> 生成策略候选 -> 寻找最优解 -> 收集经验并自我训练
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.actors = torch.nn.ModuleList([OffloadingActor(cfg.J) for _ in range(cfg.I)])
        self.bs_opt = BS_Optimizer(cfg)
        self.leo_opt = LEO_Optimizer(cfg)

        self.optimizers = [optim.Adam(actor.parameters(), lr=self.cfg.lr) for actor in self.actors]
        self.criterion = FocalLoss(alpha=self.cfg.focal_alpha, gamma=self.cfg.focal_gamma)

        self.memories = [deque(maxlen=self.cfg.memory_capacity) for _ in range(cfg.I)]

        self.batch_size = self.cfg.batch_size
        self.train_interval = self.cfg.train_interval

        self.delta_t = self.cfg.delta_init
        self.delta_min = self.cfg.delta_min
        self.delta_max = self.cfg.delta_max
        self.loss_ema = None
        self.loss_ema_slow = None
        self.loss_history = []
        self.loss_history_per_bs = [[] for _ in range(cfg.I)]

    def select_action(self, env, L_t, R_bs, R_sat, T_prop, t=0):
        I, J = self.cfg.I, self.cfg.J

        state_tensor = get_input_vector(env.Q_bs, env.Q_sat_total, env.E_BS, env.T_BS_left_prev, R_bs, R_sat)

        prob_b = np.zeros((I, J))
        for i in range(I):
            self.actors[i].eval()
            with torch.no_grad():
                logits_i = self.actors[i](state_tensor[i].unsqueeze(0))
                prob_i = torch.sigmoid(logits_i)
                prob_b[i] = prob_i.numpy().flatten()

        f_local = np.ones((I, J)) * self.cfg.f_max_UE
        l_decisions = check_local_feasibility(L_t, f_local, self.cfg)

        bs_candidates = []
        for i in range(I):
            cands_i = generate_candidates(prob_b[i], self.delta_t, l_decisions[i])
            bs_candidates.append(cands_i)

        K_candidates = min(len(cands) for cands in bs_candidates)
        if K_candidates == 0:
            raise RuntimeError(
                "generate_candidates returned zero candidates: delta_t may have decayed to zero "
                "or all tasks are classified as local-only."
            )

        best_G1 = float('inf')
        best_sol = None

        # ---- Phase 1: 收集各候选的输入 ----
        K = K_candidates
        N = I * J
        l_all = np.zeros((K, I, J), dtype=int)
        b_all = np.zeros((K, I, J), dtype=int)
        L_to_bs_stack = np.zeros((K, N))
        T_tran_bs_stack = np.zeros((K, N))
        L_to_sat_stack = np.zeros((K, N))
        T_avail_sat_stack = np.zeros((K, N))
        mask_bs_list, mask_sat_list = [], []

        for k in range(K):
            l_mat = np.array([bs_candidates[i][k][0] for i in range(I)])
            b_mat = np.array([bs_candidates[i][k][1] for i in range(I)])
            l_all[k] = l_mat
            b_all[k] = b_mat

            mask_bs = (l_mat == 0) & (b_mat == 1)
            mask_sat = (l_mat == 0) & (b_mat == 0)
            mask_bs_list.append(mask_bs)
            mask_sat_list.append(mask_sat)

            L_to_bs = np.where(mask_bs, L_t, 0.0)
            L_to_bs_stack[k] = L_to_bs.ravel()

            T_tran_bs = np.where(mask_bs, L_to_bs / R_bs, 0.0)
            T_tran_bs_stack[k] = T_tran_bs.ravel()

            L_to_sat = np.where(mask_sat, L_t, 0.0)
            L_to_sat_stack[k] = L_to_sat.ravel()

            T_tran_sat = np.where(mask_sat, L_to_sat / R_sat, 0.0)
            T_avail_sat_raw = self.cfg.tau - T_tran_sat - T_prop
            validate_sat_time_constraint(T_avail_sat_raw, mask_sat, T_prop, T_tran_sat)
            T_avail_sat_stack[k] = np.maximum(0, T_avail_sat_raw).ravel()

        # ---- Phase 2: 批量优化 (所有候选一次求解) ----
        f_bs_all = self.bs_opt.optimize_multi_candidate(
            L_to_bs_stack, env.Q_bs.ravel(), env.E_BS,
            T_tran_bs_stack, env.T_BS_left_prev)           # (K, N)

        f_sat_all = self.leo_opt.optimize_multi_candidate(
            L_to_sat_stack, env.Q_sat.ravel(),
            T_avail_sat_stack)                              # (K, N)

        # ---- Phase 3: 逐候选计算 G1 ----
        for k in range(K):
            l_mat = l_all[k]
            b_mat = b_all[k]
            mask_bs = mask_bs_list[k]
            mask_sat = mask_sat_list[k]
            f_bs = f_bs_all[k].reshape(I, J)
            f_sat = f_sat_all[k].reshape(I, J)
            T_tran_bs = T_tran_bs_stack[k].reshape(I, J)
            T_avail_sat = T_avail_sat_stack[k].reshape(I, J)

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
                    'G1': G1
                }

        best_action_b = best_sol['b']

        # [DEBUG] 观察三项量级
        if t % 500 == 0:
            details = best_sol['details']
            term_q_bs = np.sum((env.Q_bs / 1e5) * ((details['l_left_bs'] - details['l_proc_old_bs']) / 1e4))
            term_q_sat = np.sum((env.Q_sat / 1e5) * ((details['l_left_sat'] - env.current_q_sat_reduction_mat) / 1e4))
            term_q = term_q_bs + term_q_sat
            term_p = self.cfg.K_p * np.sum(details['paoi'])
            term_e_bs = np.sum(env.E_BS * (details['e_bs_total'] - self.cfg.E_max_BS))
            print(f"[G1 Debug @ Fr {t}]")
            print(f"  Raw: term_q={float(term_q):12.4f} | term_p={float(term_p):12.4f} | term_e={float(term_e_bs):12.4f}")
            print(f"  Scaled: term_q/1e6={float(term_q/1e6):8.4f} | term_p/100={float(term_p/100):8.4f} | term_e/5e3={float(term_e_bs/5e3):8.4f}")

        self.store_experience(state_tensor, best_action_b)

        if t % 200 == 0:
            loss_str = f", loss_ema={self.loss_ema:.4f}" if self.loss_ema is not None else ""
            print(f"[Frame {t:04d}] delta_t={self.delta_t:.4f}{loss_str}")

        self._attach_debug_info(best_sol, L_t, prob_b)
        return best_sol

    def calculate_objective(self, env, L_t, l_vec, mask_bs, mask_sat, f_bs, f_sat, f_local, T_tran_bs, T_avail_sat):
        phi = self.cfg.phi
        kappa1 = self.cfg.kappa1
        kappa2 = self.cfg.kappa2

        # ==========================================
        # 1. 任务到达与分配预处理
        # ==========================================
        L_loc = np.where(l_vec == 1, L_t, 0.0)
        L_to_bs = np.where(mask_bs, L_t, 0.0)
        L_to_sat = np.where(mask_sat, L_t, 0.0)

        t_proc_loc_new = self.cfg.tau
        l_proc_loc_new = np.minimum(L_loc, (f_local * t_proc_loc_new) / phi)

        T_left_prev_mat = np.zeros_like(T_tran_bs)
        for i in range(self.cfg.I):
            T_left_prev_mat[i, :] = env.T_BS_left_prev[i]

        t_proc_bs_new = np.maximum(0, self.cfg.tau - np.maximum(T_tran_bs, T_left_prev_mat))
        l_proc_bs_new = np.minimum(L_to_bs, (f_bs * t_proc_bs_new) / phi)

        l_proc_sat_new = np.minimum(L_to_sat, (f_sat * T_avail_sat) / phi)

        l_left_bs_new = L_to_bs - l_proc_bs_new
        l_left_sat_new = L_to_sat - l_proc_sat_new

        # ==========================================
        # 2. 旧任务的处理量 (基站硬拦截 + 卫星矩阵账本融合)
        # ==========================================
        total_l_prev_bs = np.sum(env.L_BS_left_prev_vec, axis=1, keepdims=True)
        f_old_bs_vec = np.where(total_l_prev_bs > 1e-9,
                                self.cfg.f_max_BS * (env.L_BS_left_prev_vec / (total_l_prev_bs + 1e-12)),
                                0.0)
        cap_old_bs = (f_old_bs_vec * self.cfg.tau) / phi
        l_proc_old_bs = np.minimum(env.L_BS_left_prev_vec, cap_old_bs)

        # 触发基站跨帧物理约束报警器 (rtol=1e-6 容忍浮点舍入误差)
        failed_to_clear_bs = (l_proc_old_bs < env.L_BS_left_prev_vec * (1 - 1e-6)) & (env.L_BS_left_prev_vec > 1e-9)
        validate_task_span_constraint(failed_to_clear_bs, env.L_BS_left_prev_vec, cap_old_bs)

        # 系统当前帧总处理量矩阵：新任务 + 基站旧任务 + 卫星环境矩阵账本清算的量
        l_proc_total = l_proc_bs_new + l_proc_sat_new + l_proc_loc_new + l_proc_old_bs + env.current_q_sat_reduction_mat

        # ==========================================
        # 3. 计算系统真实能耗
        # ==========================================
        e_bs_new = np.sum(kappa1 * phi * (f_bs ** 2) * l_proc_bs_new, axis=1)
        e_old = np.zeros(self.cfg.I)
        if np.sum(total_l_prev_bs) > 1e-9:
            e_old = np.sum(kappa1 * phi * (f_old_bs_vec ** 2) * l_proc_old_bs, axis=1)
        e_bs_total = e_bs_new + e_old

        # 卫星总能耗 = 新任务能耗 + 账本自然清算的旧任务真实能耗
        e_sat_new = np.sum(kappa2 * phi * (f_sat ** 2) * l_proc_sat_new)
        e_sat = e_sat_new + env.current_e_sat_old

        # ==========================================
        # 4. PAoI 核算 (当期一次性结清)
        # ==========================================
        total_left_bs_new = np.sum(l_left_bs_new, axis=1, keepdims=True)
        t_next_left_bs_scalar = np.where(total_left_bs_new > 1e-9, (phi * total_left_bs_new) / self.cfg.f_max_BS, 0.0)
        t_next_left_bs_est = np.where(l_left_bs_new > 1e-9, t_next_left_bs_scalar, 0.0)

        paoi_loc = np.where(l_vec == 1, (phi * L_t) / f_local, 0.0)

        time_finish_bs = np.maximum(T_tran_bs, T_left_prev_mat) + (l_proc_bs_new * phi / (f_bs + 1e-9))
        paoi_bs = np.where(l_left_bs_new > 1e-9, self.cfg.tau + self.cfg.w * t_next_left_bs_est, time_finish_bs)
        paoi_bs = np.where(mask_bs, paoi_bs, 0.0)

        total_left_sat_new = np.sum(l_left_sat_new)
        t_next_left_sat_est = (phi * total_left_sat_new) / self.cfg.f_max_Sat if total_left_sat_new > 1e-9 else 0.0

        paoi_sat = np.where(l_left_sat_new > 1e-9,
                            self.cfg.tau + self.cfg.w * t_next_left_sat_est,
                            (self.cfg.tau - T_avail_sat) + l_proc_sat_new * phi / (f_sat + 1e-9))
        paoi_sat = np.where(mask_sat, paoi_sat, 0.0)

        paoi_total = paoi_bs + paoi_sat + paoi_loc

        # ==========================================
        # 5. 组装 G1 (量级对齐法)
        # ==========================================
        # 注意：term_q 的计算公式必须与 debug 输出和 ACAgent 保持一致
        # 公式：Q/1e5 × Δl/1e4，目的是将 ~1e13 量级归一化到 ~1 量级
        term_q_bs = np.sum((env.Q_bs / 1e5) * ((l_left_bs_new - l_proc_old_bs) / 1e4))
        term_q_sat = np.sum((env.Q_sat / 1e5) * ((l_left_sat_new - env.current_q_sat_reduction_mat) / 1e4))
        term_q = term_q_bs + term_q_sat
        term_p = self.cfg.K_p * np.sum(paoi_total)
        term_e_bs = np.sum(env.E_BS * (e_bs_total - self.cfg.E_max_BS))

        # [量级对齐法] 通过预设参考尺度对齐三项量纲
        # 归一化后典型值：term_q~1e6/1e6=1, term_p~186/100=1.86, term_e~1e4/5e3=2
        # 注：term_q 预归一化后量级约~1e6，故 Q_ref=1e6 使三项量级均衡
        Q_ref = 1e6   # 队列项参考尺度（从1e7调至1e6，使队列项量级与PAoI/能量项均衡）
        PAoI_ref = 100.0  # PAoI项参考尺度（基于实测 PAoI_sum 均值 ~186）
        E_ref = 5e3  # 能量项参考尺度

        G1 = term_q / Q_ref + term_p / PAoI_ref + term_e_bs / E_ref

        real_drift = 0.5 * np.sum((L_t - l_proc_total) ** 2) + 0.5 * np.sum((e_bs_total - self.cfg.E_max_BS) ** 2)

        details = {
            'l_proc_total': l_proc_total,
            'l_proc_bs': l_proc_bs_new,
            'l_proc_sat': l_proc_sat_new,
            'l_proc_old_bs': l_proc_old_bs,  # 给 Env 更新 BS Q 使用
            'l_left_bs': l_left_bs_new,
            'l_left_sat': l_left_sat_new,
            'e_bs_total': e_bs_total,
            'e_sat': e_sat,
            'paoi': paoi_total,
            'real_drift': real_drift,
            't_next_left_bs_scalar': t_next_left_bs_scalar.flatten()
        }

        return G1, details

    def _update_delta_t(self):
        """基于loss趋势自适应调整探索窗口 delta_t"""
        if self.loss_ema is None or self.loss_ema_slow is None:
            return

        ratio = self.loss_ema / (self.loss_ema_slow + 1e-9)
        if ratio < self.cfg.delta_ratio_lo:
            self.delta_t = max(self.delta_min, self.delta_t * self.cfg.delta_decay)
        elif ratio > (2.0 - self.cfg.delta_ratio_lo):
            self.delta_t = min(self.delta_max, self.delta_t * self.cfg.delta_grow)

    def store_experience(self, state_tensor, best_action_b):
        states = state_tensor.detach().numpy()
        for i in range(self.cfg.I):
            self.memories[i].append((states[i], best_action_b[i].copy()))

    def train(self, current_frame):
        if current_frame % self.cfg.train_interval != 0:
            return

        loss_vals = []
        loss_per_bs = {}
        min_samples = max(self.batch_size * 4, self.cfg.memory_capacity // 4)

        for i in range(self.cfg.I):
            if len(self.memories[i]) < min_samples:
                continue

            batch = random.sample(self.memories[i], self.batch_size)
            state_batch, action_batch = zip(*batch)

            states = torch.FloatTensor(np.array(state_batch))
            targets = torch.FloatTensor(np.array(action_batch))

            self.actors[i].train()
            self.optimizers[i].zero_grad()

            logits = self.actors[i](states)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizers[i].step()
            loss_vals.append(loss.item())
            loss_per_bs[i] = loss.item()

        if loss_vals:
            avg_loss = np.mean(loss_vals)
            self.loss_history.append((current_frame, avg_loss))
            for i in range(self.cfg.I):
                if i in loss_per_bs:
                    self.loss_history_per_bs[i].append((current_frame, loss_per_bs[i]))

            # 更新loss指数移动平均，用于驱动delta_t自适应
            if self.loss_ema is None:
                self.loss_ema = avg_loss
                self.loss_ema_slow = avg_loss
            else:
                alpha_f = 1.0 - self.cfg.delta_ema_fast
                alpha_s = 1.0 - self.cfg.delta_ema_slow
                self.loss_ema = self.cfg.delta_ema_fast * self.loss_ema + alpha_f * avg_loss
                self.loss_ema_slow = self.cfg.delta_ema_slow * self.loss_ema_slow + alpha_s * avg_loss

            self._update_delta_t()

    def _attach_debug_info(self, sol, L_t, prob_b):
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