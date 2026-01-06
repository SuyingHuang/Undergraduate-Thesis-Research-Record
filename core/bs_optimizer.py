import numpy as np
from core.math_utils import solve_cubic_newton


class BS_Optimizer:
    """
    Algorithm 2: Optimal Frequency Allocation for BS
    解决 Problem P3
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def optimize(self, L_t, Q_t, E_t, T_tran, T_left_prev):
        # 提取参数以简化公式书写
        J = self.cfg.J
        phi = self.cfg.phi
        kappa1 = self.cfg.kappa1
        tau = self.cfg.tau
        f_max = self.cfg.f_max_BS
        K_p = self.cfg.K_p
        w = self.cfg.w

        # 防止分母为0的安全能量值
        E_safe = max(E_t, 1e-5)

        # 辅助常数 M
        M = K_p * w / f_max

        # 1. Lambda 上界动态估算
        # 确保 Eq. 56 根号下非负
        lambda_limits = (Q_t / phi + M) * tau
        lambda_high = np.max(lambda_limits) + 1.0
        lambda_low = 0.0

        f_final = np.zeros(J)

        # 2. 二分查找 Lambda
        for _ in range(60):
            lam = (lambda_low + lambda_high) / 2
            if lam < 1e-10: lam = 1e-10

            f_temp = np.zeros(J)
            term_B_denom = 3 * E_safe * kappa1

            for j in range(J):
                L = L_t[j]
                if L <= 1e-6: continue  # 无任务跳过

                # --- A. 计算物理可用时间 ---
                # 必须扣除传输延迟和上一帧残留的阻塞时间
                delay_occupancy = max(T_tran[j], T_left_prev)
                t_avail = tau - delay_occupancy

                # --- B. 计算阈值频率 f_th ---
                if t_avail <= 1e-6:
                    f_th = float('inf')
                else:
                    f_th = phi * L / t_avail

                # --- C. 计算 Type A (能做完) 候选解 ---
                # Eq 55: 2*E*k*phi*L * f^3 + lam * f^2 - K*phi*L = 0
                a = 2 * E_t * kappa1 * phi * L
                b = lam
                d = - K_p * phi * L
                f_A = solve_cubic_newton(a, b, d, self.cfg.newton_iter)

                # --- D. 计算 Type B (做不完) 候选解 ---
                # Eq 56
                if t_avail <= 1e-6:
                    f_B = 0.0
                else:
                    num = (Q_t[j] / phi + M)
                    term_lam = lam / (term_B_denom * t_avail)
                    val = num / term_B_denom - term_lam
                    f_B = np.sqrt(val) if val > 0 else 0.0

                # --- E. 比较并选择 ---
                if f_B < f_th:
                    f_temp[j] = f_B  # 判定为 Type B
                else:
                    f_temp[j] = f_A  # 判定为 Type A

            # 3. 检查总算力约束并更新 Lambda
            if np.sum(f_temp) > f_max:
                lambda_low = lam  # 需求过大，涨价
            else:
                lambda_high = lam  # 满足约束，降价
                f_final = f_temp.copy()

        return f_final