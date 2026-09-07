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
from utils.physics_validator import validate_sat_time_constraint
from utils.objective import objective_coefficients


class LDAAgent:
    """
    LDA 算法的智能体
    专职负责：观测环境状态 -> DNN 推理 -> 生成策略候选 -> 寻找最优解 -> 收集经验并自我训练
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.actors = torch.nn.ModuleList([
            OffloadingActor(cfg.J, num_bs=cfg.I,
                            sat_state_slots=cfg.sat_state_slots,
                            hidden_dim=cfg.hidden_dim) for _ in range(cfg.I)
        ])
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
        sat_context = env.satellite_state_context()
        state_tensor = get_input_vector(
            L_t, env.Q_bs, env.Q_sat, env.E_BS,
            env.T_BS_left_prev, R_bs, R_sat,
            sat_service_plan=sat_context['service'],
            sat_ledger_loads=sat_context['ledger_loads'],
            sat_old_energy=sat_context['old_energy'],
            sat_state_slots=self.cfg.sat_state_slots,
        )

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

        if any(not candidates for candidates in bs_candidates):
            raise RuntimeError("generate_candidates returned zero candidates")

        # Bounded coordinate search: every BS gets to expose all of its local
        # candidates even when another BS has only a single candidate.
        current_b = np.array([candidates[0][1] for candidates in bs_candidates])
        best_sol = None
        for _ in range(self.cfg.coordinate_search_rounds):
            joint_candidates = self._coordinate_proposals(current_b, bs_candidates)
            round_best = self._evaluate_joint_candidates(
                env, L_t, R_bs, R_sat, T_prop, l_decisions,
                joint_candidates)
            if best_sol is not None and round_best['G1'] >= best_sol['G1'] - 1e-12:
                break
            best_sol = round_best
            current_b = best_sol['b'].copy()

        best_action_b = best_sol['b']

        # [DEBUG] 观察三项量级
        if t % 500 == 0:
            terms = best_sol['details']['objective_terms']
            print(f"[G1 Debug @ Fr {t}]")
            print(f"  Raw: term_q={terms['queue_raw']:12.4e} | "
                  f"term_p={terms['paoi_raw']:12.4f} | term_e={terms['energy_raw']:12.4f}")
            print(f"  Weighted: Q={terms['queue_weighted']:8.4f} | "
                  f"PAoI={terms['paoi_weighted']:8.4f} | E={terms['energy_weighted']:8.4f}")

        self.store_experience(state_tensor, best_action_b, l_decisions == 0)

        if t % 200 == 0:
            loss_str = f", loss_ema={self.loss_ema:.4f}" if self.loss_ema is not None else ""
            print(f"[Frame {t:04d}] delta_t={self.delta_t:.4f}{loss_str}")

        self._attach_debug_info(best_sol, L_t, prob_b)
        return best_sol

    @staticmethod
    def _coordinate_proposals(current_b, bs_candidates):
        proposals = [current_b.copy()]
        seen = {tuple(current_b.ravel())}
        for i, candidates in enumerate(bs_candidates):
            for _, b_i in candidates:
                proposal = current_b.copy()
                proposal[i] = b_i
                key = tuple(proposal.ravel())
                if key not in seen:
                    seen.add(key)
                    proposals.append(proposal)
        return proposals

    def _evaluate_joint_candidates(self, env, L_t, R_bs, R_sat, T_prop,
                                   l_mat, b_candidates):
        """Allocate resources for a bounded set of complete system actions."""
        I, J = self.cfg.I, self.cfg.J
        K = len(b_candidates)
        N = I * J
        l_all = np.zeros((K, I, J), dtype=int)
        b_all = np.zeros((K, I, J), dtype=int)
        L_to_bs_stack = np.zeros((K, N))
        T_tran_bs_stack = np.zeros((K, N))
        L_to_sat_stack = np.zeros((K, N))
        T_avail_sat_stack = np.zeros((K, N))
        mask_bs_list, mask_sat_list = [], []

        for k, b_mat in enumerate(b_candidates):
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
        best_G1, best_sol = float('inf'), None
        f_local = np.ones((I, J)) * self.cfg.f_max_UE
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

        l_left_old_bs = np.maximum(0.0, env.L_BS_left_prev_vec - l_proc_old_bs)
        l_left_bs_total = l_left_old_bs + l_left_bs_new

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
        total_left_bs_new = np.sum(l_left_bs_total, axis=1, keepdims=True)
        t_next_left_bs_scalar = np.where(total_left_bs_new > 1e-9, (phi * total_left_bs_new) / self.cfg.f_max_BS, 0.0)
        t_next_left_bs_est = np.where(l_left_bs_new > 1e-9, t_next_left_bs_scalar, 0.0)

        paoi_loc = np.where(l_vec == 1, (phi * L_t) / f_local, 0.0)

        time_finish_bs = np.maximum(T_tran_bs, T_left_prev_mat) + (l_proc_bs_new * phi / (f_bs + 1e-9))
        paoi_bs = np.where(l_left_bs_new > 1e-9, self.cfg.tau + self.cfg.w * t_next_left_bs_est, time_finish_bs)
        paoi_bs = np.where(mask_bs, paoi_bs, 0.0)

        total_left_sat_new = np.sum(l_left_sat_new)
        f_sat_energy_limit = (self.cfg.E_max_Sat / (kappa2 * self.cfg.tau)) ** (1 / 3)
        f_sat_effective = min(self.cfg.f_max_Sat, f_sat_energy_limit)
        t_next_left_sat_est = ((phi * total_left_sat_new) / f_sat_effective
                               if total_left_sat_new > 1e-9 else 0.0)

        paoi_sat = np.where(l_left_sat_new > 1e-9,
                            self.cfg.tau + self.cfg.w * t_next_left_sat_est,
                            (self.cfg.tau - T_avail_sat) + l_proc_sat_new * phi / (f_sat + 1e-9))
        paoi_sat = np.where(mask_sat, paoi_sat, 0.0)

        paoi_total = paoi_bs + paoi_sat + paoi_loc

        # ==========================================
        # 5. 组装 G1 (量级对齐法)
        # ==========================================
        term_q = np.sum(env.Q_bs * (l_left_bs_new - l_proc_old_bs))
        term_q += np.sum(env.Q_sat * (l_left_sat_new - env.current_q_sat_reduction_mat))
        term_p = np.sum(paoi_total)
        term_e_bs = np.sum(env.E_BS * (e_bs_total - self.cfg.E_max_BS))
        weights = objective_coefficients(self.cfg, include_paoi=True)
        weighted_q = weights['queue'] * term_q
        weighted_p = weights['paoi'] * term_p
        weighted_e = weights['energy'] * term_e_bs
        G1 = weighted_q + weighted_p + weighted_e

        drift_bound_quadratic = (0.5 * np.sum((L_t - l_proc_total) ** 2) +
                                 0.5 * np.sum((e_bs_total - self.cfg.E_max_BS) ** 2))

        details = {
            'l_proc_total': l_proc_total,
            'l_proc_bs': l_proc_bs_new,
            'l_proc_sat': l_proc_sat_new,
            'l_proc_old_bs': l_proc_old_bs,  # 给 Env 更新 BS Q 使用
            'l_left_old_bs': l_left_old_bs,
            'l_left_bs': l_left_bs_new,
            'l_left_bs_total': l_left_bs_total,
            'l_left_sat': l_left_sat_new,
            'e_bs_total': e_bs_total,
            'e_sat_new': e_sat_new,
            'e_sat': e_sat,
            'paoi': paoi_total,
            'drift_bound_quadratic': drift_bound_quadratic,
            'objective_terms': {
                'queue_raw': float(term_q), 'paoi_raw': float(term_p),
                'energy_raw': float(term_e_bs),
                'queue_weighted': float(weighted_q),
                'paoi_weighted': float(weighted_p),
                'energy_weighted': float(weighted_e),
            },
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

    def store_experience(self, state_tensor, best_action_b, offload_mask=None):
        states = state_tensor.detach().numpy()
        if offload_mask is None:
            offload_mask = np.ones_like(best_action_b, dtype=bool)
        for i in range(self.cfg.I):
            self.memories[i].append((states[i], best_action_b[i].copy(),
                                     offload_mask[i].copy()))

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
            state_batch, action_batch, mask_batch = zip(*batch)

            states = torch.FloatTensor(np.array(state_batch))
            targets = torch.FloatTensor(np.array(action_batch))
            masks = torch.FloatTensor(np.array(mask_batch))

            self.actors[i].train()
            self.optimizers[i].zero_grad()

            logits = self.actors[i](states)
            if masks.sum().item() == 0:
                continue
            loss = self.criterion(logits, targets, mask=masks)

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
