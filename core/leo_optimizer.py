import numpy as np
from core.math_utils import solve_cubic_newton


class LEO_Optimizer:
    """
    实现论文 Algorithm 3: Computing Resource Optimization for LEOS
    [cite_start]解决 Problem P4 [cite: 548]
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def get_search_bounds(self, L_t, Q_t, T_avail):
        """
        根据当前帧的状态动态计算二分搜索的上界
        """
        # 提取参数
        phi = self.cfg.phi
        kappa2 = self.cfg.kappa2
        f_max = self.cfg.f_max_Sat
        K_p = self.cfg.K_p
        w = self.cfg.w

        # 1. 计算 nu_high
        # 找到所有用户中最小的 f_th (对应最大的 nu_max)
        # f_th = phi * L / T_avail
        # nu_max = K / (2 * k2 * f_th^3)
        # 简化：nu_max 正比于 (T_avail / L)^3

        # 防止除以零
        valid_mask = (L_t > 1e-6) & (T_avail > 1e-6)
        if not np.any(valid_mask):
            return 1e5, 1e8  # 默认值
        # 计算每个用户的理论 nu_max
        f_th_list = phi * L_t[valid_mask] / T_avail[valid_mask]
        # 加上一个安全系数 1.5 或 2.0
        nu_max_candidates = K_p / (2 * kappa2 * (f_th_list ** 3) + 1e-20)
        nu_high = np.max(nu_max_candidates) * 2.0

        # 限制一个合理的最小值和最大值，防止数值溢出
        nu_high = np.clip(nu_high, 1e5, 1e9)

        # 2. 计算 mu_high
        # mu_max ~ (Q/phi) * T_avail
        M_prime = (K_p * w) / f_max
        if np.any(valid_mask):
            term_b = Q_t[valid_mask] / phi + M_prime
            mu_max_candidates = term_b * T_avail[valid_mask]
            mu_high = np.max(mu_max_candidates) * 2.0
        else:
            mu_high = 1e8

        return nu_high, mu_high

    def optimize(self, L_t, Q_t, T_avail):
        nu_high_calc, mu_high_calc = self.get_search_bounds(L_t, Q_t, T_avail)
        """
        :param L_t: 当前帧卸载到卫星的任务量 (bits)
        :param Q_t: 卫星处的任务积压 (bits)
        :param T_avail: 留给计算的物理可用时间 = tau - T_tran - T_prop
        """
        # 参数提取
        phi = self.cfg.phi
        kappa2 = self.cfg.kappa2
        f_max = self.cfg.f_max_Sat
        E_max = self.cfg.E_max_Sat
        K_p = self.cfg.K_p
        w = self.cfg.w

        n_users = len(L_t)

        # 辅助常数 M' (Eq. 60 下方)
        M_prime = (K_p * w) / f_max

        # --- 双层二分搜索 ---
        # 外层循环：搜索能量乘子 nu (对应 E_max 约束)
        nu_low, nu_high = 0.0, nu_high_calc

        f_final = np.zeros(n_users)

        for _ in range(30):  # 外层迭代次数
            nu = (nu_low + nu_high) / 2
            if nu < 1e-15: nu = 1e-15

            # 内层循环：搜索频率乘子 mu (对应 f_max 约束)
            mu_low, mu_high = 0.0, mu_high_calc
            f_inner = np.zeros(n_users)

            for _ in range(40):  # 内层迭代次数
                mu = (mu_low + mu_high) / 2

                f_temp = np.zeros(n_users)

                # 遍历每个用户求解 optimal f
                for k in range(n_users):
                    L = L_t[k]
                    t_av = T_avail[k]
                    q = Q_t[k]

                    if L <= 1e-6: continue

                    # 1. 计算阈值频率 f_th (任务恰好做完的频率)
                    if t_av <= 1e-6:
                        f_th = 1e14  # 无时间，阈值无穷大
                    else:
                        f_th = phi * L / t_av

                    # 2. Type A (能做完): 解三次方程 Eq. 62
                    # 2*k2*phi*nu*L * f^3 + mu * f^2 - K*phi*L = 0
                    a = 2 * kappa2 * phi * nu * L
                    b = mu
                    d = -K_p * phi * L
                    # 使用你已有的 newton solver
                    f_A = solve_cubic_newton(a, b, d, iterations=self.cfg.newton_iter)

                    # 3. Type B (做不完): 闭式解 Eq. 65
                    # f_B = sqrt( (Q/phi + M')/(3*k2*nu) - mu/(3*k2*nu*t_av) )
                    denom = 3 * kappa2 * nu
                    term1 = (q / phi + M_prime) / denom

                    if t_av > 1e-6:
                        term2 = mu / (denom * t_av)
                        val = term1 - term2
                        f_B = np.sqrt(val) if val > 0 else 0.0
                    else:
                        f_B = 0.0

                    # 4. 决策逻辑 Eq. 66
                    # 如果最优的 f_B 小于阈值，说明确实做不完，选 f_B
                    if f_B < f_th:
                        f_temp[k] = f_B
                    else:
                        f_temp[k] = f_A

                # 内层 Check: 总频率约束
                if np.sum(f_temp) > f_max:
                    mu_low = mu  # 频率超标，增加惩罚 mu
                else:
                    mu_high = mu
                    f_inner = f_temp.copy()

            # 外层 Check: 总能量约束
            # 计算实际能耗 (需先算出实际处理量 L_proc)
            e_total = 0.0
            for k in range(n_users):
                if f_inner[k] < 1e-9: continue
                # [cite_start]实际处理量 [cite: 18]
                l_proc = min(L_t[k], f_inner[k] * T_avail[k] / phi) if T_avail[k] > 0 else 0
                e_total += kappa2 * phi * (f_inner[k] ** 2) * l_proc

            if e_total > E_max:
                nu_low = nu  # 能耗超标，增加惩罚 nu
            else:
                nu_high = nu
                f_final = f_inner.copy()

        return f_final
