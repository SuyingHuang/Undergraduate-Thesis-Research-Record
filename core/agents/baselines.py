
import numpy as np
from core.agents.lda_agent import LDAAgent
from core.models.tcopq import check_local_feasibility


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

        # A. 基站资源分配 (批量: I*J 用户一次处理)
        T_tran_bs = np.where(mask_bs, L_to_bs / R_bs, 0.0)
        f_bs_all = self.bs_opt.optimize_batched(
            L_to_bs.ravel(), env.Q_bs.ravel(), env.E_BS,
            T_tran_bs.ravel(), env.T_BS_left_prev)
        f_bs = f_bs_all.reshape(I, J)

        # B. 卫星资源分配
        T_tran_sat = np.where(mask_sat, L_to_sat / R_sat, 0.0)
        T_avail_sat = np.maximum(0, self.cfg.tau - T_tran_sat - T_prop)
        f_sat_flat = self.leo_opt.optimize_vectorized(L_to_sat.flatten(), env.Q_sat.flatten(), T_avail_sat.flatten())
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

    def select_action(self, env, L_t, R_bs, R_sat, T_prop, t=0):
        # 1. 先判断哪些任务必须卸载，哪些可以本地处理
        f_local = np.ones((self.cfg.I, self.cfg.J)) * self.cfg.f_max_UE
        l_mat = check_local_feasibility(L_t, f_local, self.cfg)  # l=1 本地，l=0 必须卸载

        # 2. 对必须卸载的任务(l=0)，全部走基站
        b_mat = np.ones((self.cfg.I, self.cfg.J))  # b=1 表示 BS

        return self._evaluate_fixed_action(env, L_t, R_bs, R_sat, T_prop, l_mat, b_mat)


class MTDAgent(HeuristicAgent):
    """
    基线算法 2: MTD (Minimum Transmission Delay)
    每个基站优先让传输延迟最小的2个用户使用卫星链路，其余全给基站
    """

    def select_action(self, env, L_t, R_bs, R_sat, T_prop, k_sat=2, t=0):
        # 1. 先判断哪些任务必须卸载，哪些可以本地处理
        f_local = np.ones((self.cfg.I, self.cfg.J)) * self.cfg.f_max_UE
        l_mat = check_local_feasibility(L_t, f_local, self.cfg)  # l=1 本地，l=0 必须卸载

        # 2. 对必须卸载的任务(l=0)，选传输延迟最小的 k_sat 个给卫星，其余给基站
        b_mat = np.ones((self.cfg.I, self.cfg.J))  # b=1 表示 BS
        T_tran_sat = L_t / (R_sat + 1e-9)
        for i in range(self.cfg.I):
            # 只在必须卸载的任务中选择
            offloadable_mask = l_mat[i] == 0
            if not np.any(offloadable_mask):
                continue
            # 找出传输延迟最小的 k_sat 个
            T_tran_offloadable = np.where(offloadable_mask, T_tran_sat[i], np.inf)
            # argsort 得到排序后的索引，取前 k_sat 个
            sorted_indices = np.argsort(T_tran_offloadable)
            # 取有效的（不是 inf 的）前 k_sat 个
            valid_sorted = [idx for idx in sorted_indices if T_tran_offloadable[idx] < np.inf]
            for j_idx in valid_sorted[:k_sat]:
                b_mat[i, j_idx] = 0  # 给卫星

        return self._evaluate_fixed_action(env, L_t, R_bs, R_sat, T_prop, l_mat, b_mat)


class ACAgent(LDAAgent):
    """
    基线算法 3: AC (Actor-Critic)
    缺少PAoI优化的强化学习算法。
    实现原理：使用量级对齐法，但禁用PAoI项，仅优化队列和能量。
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def calculate_objective(self, env, L_t, l_vec, mask_bs, mask_sat, f_bs, f_sat, f_local, T_tran_bs, T_avail_sat):
        # 复用父类LDAAgent的计算获取details
        G1_lda, details = LDAAgent.calculate_objective(
            self, env, L_t, l_vec, mask_bs, mask_sat, f_bs, f_sat, f_local, T_tran_bs, T_avail_sat
        )

        # AC计算term值
        term_q_bs = np.sum((env.Q_bs / 1e5) * ((details['l_left_bs'] - details['l_proc_old_bs']) / 1e4))
        term_q_sat = np.sum((env.Q_sat / 1e5) * ((details['l_left_sat'] - env.current_q_sat_reduction_mat) / 1e4))
        term_q = term_q_bs + term_q_sat
        term_e_bs = np.sum(env.E_BS * (details['e_bs_total'] - self.cfg.E_max_BS))

        # AC: 量级对齐法（禁用PAoI项），参考尺度由 config.py 统一管理
        G1_ac = term_q / self.cfg.Q_ref + term_e_bs / self.cfg.E_ref

        return G1_ac, details