import numpy as np


def solve_cubic_newton(a, b, d, iterations=10):
    """
    使用牛顿迭代法求解三次方程: a*x^3 + b*x^2 + d = 0 (c=0)
    对应论文 Eq. 55: 2*E*k*phi*L * f^3 + lam * f^2 - K*phi*L = 0
    单调递增函数，必有唯一正实根。
    """
    # 1. 初始猜测 [Snippet 1239]
    # 当 lambda=0 时，ax^3 + d = 0 => x = (-d/a)^(1/3)
    # 这通常是一个很好的起点（上限）
    if abs(a) < 1e-20:
        if abs(b) < 1e-20: return 0.0
        val = -d / b
        return np.sqrt(val) if val > 0 else 0.0

    x = (-d / a) ** (1 / 3)

    # 2. 迭代
    for _ in range(iterations):
        # f(x)
        fx = a * x ** 3 + b * x ** 2 + d
        # f'(x) = 3ax^2 + 2bx
        dfx = 3 * a * x ** 2 + 2 * b * x

        if abs(dfx) < 1e-10: break  # 防止除零

        x_new = x - fx / dfx

        if abs(x_new - x) < 1e-6:
            return x_new
        x = x_new

    return max(x, 0.0)


class BS_Optimizer:
    def __init__(self, cfg):
        self.cfg = cfg

    def optimize(self, L_t, Q_t, E_t, T_tran, T_left_prev):
        J = self.cfg.J
        phi = self.cfg.phi
        kappa1 = self.cfg.kappa1
        tau = self.cfg.tau
        f_max = self.cfg.f_max_BS
        K_p = self.cfg.K_p
        w = self.cfg.w

        E_safe = max(E_t, 1e-5)
        M = K_p * w / f_max

        # lambda 上界计算
        lambda_limits = (Q_t / phi + M) * tau
        lambda_high = np.max(lambda_limits) + 1.0
        lambda_low = 0.0

        f_final = np.zeros(J)

        for _ in range(60):  # 二分查找 lambda
            lam = (lambda_low + lambda_high) / 2
            if lam < 1e-10: lam = 1e-10

            f_temp = np.zeros(J)
            term_B_denom = 3 * E_safe * kappa1

            for j in range(J):
                L = L_t[j]
                if L <= 1e-6: continue

                # --- 1. 计算 f_th (基于物理可用时间) ---
                delay_occupancy = max(T_tran[j], T_left_prev)
                t_avail = tau - delay_occupancy

                if t_avail <= 1e-6:
                    f_th = float('inf')
                else:
                    f_th = phi * L / t_avail

                # --- 2. Type A: 牛顿迭代法 [Eq. 55] ---
                # 2*E*k*phi*L * f^3 + lam * f^2 - K*phi*L = 0
                # a*f^3 + b*f^2 + d = 0
                a = 2 * E_t * kappa1 * phi * L
                b = lam
                d = - K_p * phi * L
                f_A = solve_cubic_newton(a, b, d, iterations=self.cfg.newton_iter)

                # --- 3. Type B: 闭式解 [Eq. 56] ---
                if t_avail <= 1e-6:
                    f_B = 0.0
                else:
                    num = (Q_t[j] / phi + M)
                    term_lam = lam / (term_B_denom * t_avail)
                    val = num / term_B_denom - term_lam
                    f_B = np.sqrt(val) if val > 0 else 0.0

                # --- 4. 判决 ---
                if f_B < f_th:
                    f_temp[j] = f_B
                else:
                    f_temp[j] = f_A

            if np.sum(f_temp) > f_max:
                lambda_low = lam
            else:
                lambda_high = lam
                f_final = f_temp.copy()

        return f_final