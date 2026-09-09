
import numpy as np
from core.agents.lda_agent import LDAAgent
from core.models.tcopq import check_local_feasibility
from core.optimizers.bs_optimizer import BS_Optimizer
from core.optimizers.leo_optimizer import LEO_Optimizer
from core.agents.heuristic_actions import baseline_actions


class HeuristicAgent(LDAAgent):
    """
    启发式基线算法的父类 (COB, MTD)
    复用 LDAAgent 的资源分配逻辑，不创建未使用的网络、优化器和回放池。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.bs_opt = BS_Optimizer(cfg)
        self.leo_opt = LEO_Optimizer(cfg)
        self._configure_paoi_ablation()

    def train(self, current_frame):
        pass  # 启发式算法不需要训练

    def store_experience(self, state_tensor, best_action_b, offload_mask=None):
        pass  # 启发式算法不需要记录经验

    def _evaluate_fixed_action(self, env, L_t, R_bs, R_sat, T_prop, l_mat, b_mat):
        """
        给定固定的卸载决策 (l_mat, b_mat)，调用下层优化器分配资源并计算指标
        """
        env.prepare_frame()
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
    能在一帧内完成的任务留在本地，其余全部卸载给基站。
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
        _, b_mat = baseline_actions(l_mat, L_t, R_sat, k_sat)

        return self._evaluate_fixed_action(env, L_t, R_bs, R_sat, T_prop, l_mat, b_mat)


class ACAgent(LDAAgent):
    """
    历史标识 AC，论文图例为 LDA2；并未实现 Critic 网络。
    从候选排序目标及下层 BS/LEO 资源分配中一致地去掉 PAoI 项。
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.bs_opt.paoi_weight = 0.0
        self.leo_opt.paoi_weight = 0.0

    def calculate_objective(self, env, L_t, l_vec, mask_bs, mask_sat, f_bs, f_sat, f_local, T_tran_bs, T_avail_sat):
        # 复用父类LDAAgent的计算获取details
        _, details = LDAAgent.calculate_objective(
            self, env, L_t, l_vec, mask_bs, mask_sat, f_bs, f_sat, f_local, T_tran_bs, T_avail_sat
        )

        terms = details['objective_terms']
        terms['paoi_weighted'] = 0.0
        G1_ac = terms['queue_weighted'] + terms['energy_weighted']

        return G1_ac, details
