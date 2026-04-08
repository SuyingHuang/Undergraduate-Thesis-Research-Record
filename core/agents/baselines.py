
import numpy as np
from core.agents.lda_agent import LDAAgent


class HeuristicAgent(LDAAgent):
    """
    启发式基线算法的父类 (COB, MTD)
    复用 LDAAgent 的资源分配逻辑，但屏蔽 DRL 神经网络的训练和经验存储。
    """

    def train(self, current_frame):
        pass  # 启发式算法不需要训练

    def store_experience(self, state_tensor, best_action_b):
        pass  # 启发式算法不需要记录经验

    def _evaluate_fixed_action(self, env, L_t, R_bs, R_sat, T_prop, l_mat, b_mat):
        """
        给定固定的卸载决策 (l_mat, b_mat)，调用下层优化器分配资源并计算指标
        """
        I, J = self.cfg.I, self.cfg.J
        mask_bs = (l_mat == 0) & (b_mat == 1)
        mask_sat = (l_mat == 0) & (b_mat == 0)

        L_to_bs = np.where(mask_bs, L_t, 0.0)
        L_to_sat = np.where(mask_sat, L_t, 0.0)

        f_local = np.ones((I, J)) * self.cfg.f_max_UE

        # A. 基站资源分配
        f_bs = np.zeros((I, J))
        T_tran_bs = np.zeros((I, J))
        for i in range(I):
            T_tran_bs[i] = np.where(mask_bs[i], L_to_bs[i] / R_bs[i], 0.0)
            f_bs[i] = self.bs_opt.optimize(L_to_bs[i], env.Q_bs[i], env.E_BS[i], T_tran_bs[i], env.T_BS_left_prev[i])

        # B. 卫星资源分配
        T_tran_sat = np.where(mask_sat, L_to_sat / R_sat, 0.0)
        T_avail_sat = np.maximum(0, self.cfg.tau - T_tran_sat - T_prop)
        f_sat_flat = self.leo_opt.optimize(L_to_sat.flatten(), env.Q_sat.flatten(), T_avail_sat.flatten())
        f_sat = f_sat_flat.reshape(I, J)

        # C. 复用父类的计算目标函数逻辑
        G1, details = self.calculate_objective(
            env, L_t, l_mat, mask_bs, mask_sat,
            f_bs, f_sat, f_local, T_tran_bs, T_avail_sat
        )

        sol = {
            'l': l_mat, 'b': b_mat,
            'f_bs': f_bs, 'f_sat': f_sat,
            'details': details,
            'G1': G1
        }

        # 伪造 prob_b 防止父类 _attach_debug_info 报错
        self._attach_debug_info(sol, L_t, prob_b=b_mat)
        return sol


class COBAgent(HeuristicAgent):
    """
    基线算法 1: COB (Complete Offloading to BS)
    所有任务100%卸载给基站
    """

    def select_action(self, env, L_t, R_bs, R_sat, T_prop):
        l_mat = np.zeros((self.cfg.I, self.cfg.J))
        b_mat = np.ones((self.cfg.I, self.cfg.J))
        return self._evaluate_fixed_action(env, L_t, R_bs, R_sat, T_prop, l_mat, b_mat)


class MTDAgent(HeuristicAgent):
    """
    基线算法 2: MTD (Minimum Transmission Delay)
    每个基站挑选传输延迟最小的1个用户给卫星，其余全给基站
    """

    def select_action(self, env, L_t, R_bs, R_sat, T_prop):
        l_mat = np.zeros((self.cfg.I, self.cfg.J))
        b_mat = np.ones((self.cfg.I, self.cfg.J))

        # T_tran_sat = L_t / R_sat
        T_tran_sat = L_t / (R_sat + 1e-9)
        for i in range(self.cfg.I):
            best_j = np.argmin(T_tran_sat[i])
            b_mat[i, best_j] = 0  # 给卫星

        return self._evaluate_fixed_action(env, L_t, R_bs, R_sat, T_prop, l_mat, b_mat)


class ACAgent(LDAAgent):
    """
    基线算法 3: AC (Actor-Critic)
    缺少队列稳定性的强化学习算法。
    实现原理：重写目标函数 G1 计算逻辑，在求和时剥离掉 Lyapunov 的队列积压漂移项，直接优化 PAoI 和能量。
    """

    def calculate_objective(self, env, L_t, l_vec, mask_bs, mask_sat, f_bs, f_sat, f_local, T_tran_bs, T_avail_sat):
        # 先利用父类算出所有底层的指标详情
        G1_lda, details = super().calculate_objective(
            env, L_t, l_vec, mask_bs, mask_sat, f_bs, f_sat, f_local, T_tran_bs, T_avail_sat
        )

        # 剥离 Lyapunov 框架中的任务队列漂移项 (无视系统任务队列堵塞)
        term_p = self.cfg.K_p * np.sum(details['paoi'])
        term_e_bs = np.sum(env.E_BS * (details['e_bs_total'] - self.cfg.E_max_BS))

        # AC 的损失函数仅关注时延和能量
        G1_ac = term_p + term_e_bs

        return G1_ac, details