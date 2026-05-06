import numpy as np
from utils.math_utils import solve_cubic_newton, solve_cubic_newton_vectorized


class LEO_Optimizer:
    """
    实现论文 Algorithm 3: Computing Resource Optimization for LEOS
    [cite_start]解决 Problem P4 [cite: 548]
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def get_search_bounds(self, L_t, Q_t, T_avail):
        """根据当前帧的状态动态计算二分搜索的上界"""
        phi = self.cfg.phi
        kappa2 = self.cfg.kappa2
        f_max = self.cfg.f_max_Sat
        K_p = self.cfg.K_p
        w = self.cfg.w

        valid_mask = (L_t > 1e-6) & (T_avail > 1e-6)
        if not np.any(valid_mask):
            return 1e5, 1e8

        f_th_list = phi * L_t[valid_mask] / T_avail[valid_mask]
        nu_max_candidates = K_p / (2 * kappa2 * (f_th_list ** 3) + 1e-20)
        nu_high = np.max(nu_max_candidates) * 2.0
        nu_high = np.clip(nu_high, 1e5, 1e10)

        M_prime = (K_p * w) / f_max
        if np.any(valid_mask):
            term_b = Q_t[valid_mask] / phi + M_prime
            mu_max_candidates = term_b * T_avail[valid_mask]
            mu_high = np.max(mu_max_candidates) * 2.0
        else:
            mu_high = 1e8

        return nu_high, mu_high

    def optimize(self, L_t, Q_t, T_avail):
        """原始逐用户循环版本（保留用于对照验证）"""
        nu_high_calc, mu_high_calc = self.get_search_bounds(L_t, Q_t, T_avail)

        phi = self.cfg.phi
        kappa2 = self.cfg.kappa2
        f_max = self.cfg.f_max_Sat
        E_max = self.cfg.E_max_Sat
        K_p = self.cfg.K_p
        w = self.cfg.w

        n_users = len(L_t)
        M_prime = (K_p * w) / f_max

        nu_low, nu_high = 0.0, nu_high_calc
        f_final = np.zeros(n_users)

        for _ in range(30):
            nu = (nu_low + nu_high) / 2
            if nu < 1e-15: nu = 1e-15

            mu_low, mu_high = 0.0, mu_high_calc
            f_inner = np.zeros(n_users)

            for _ in range(30):
                mu = (mu_low + mu_high) / 2
                f_temp = np.zeros(n_users)

                for k in range(n_users):
                    L = L_t[k]
                    t_av = T_avail[k]
                    q = Q_t[k]
                    if L <= 1e-6: continue

                    if t_av <= 1e-6:
                        f_th = 1e14
                    else:
                        f_th = phi * L / t_av

                    a = 2 * kappa2 * phi * nu * L
                    b = mu
                    d = -K_p * phi * L
                    f_A = solve_cubic_newton(a, b, d, iterations=self.cfg.newton_iter)

                    denom = 3 * kappa2 * nu
                    term1 = (q / phi + M_prime) / denom

                    if t_av > 1e-6:
                        term2 = mu / (denom * t_av)
                        val = term1 - term2
                        f_B = np.sqrt(val) if val > 0 else 0.0
                    else:
                        f_B = 0.0

                    if f_B < f_th:
                        f_temp[k] = f_B
                    else:
                        f_temp[k] = f_A

                if np.sum(f_temp) > f_max:
                    mu_low = mu
                else:
                    mu_high = mu
                    f_inner = f_temp.copy()

            e_total = 0.0
            for k in range(n_users):
                if f_inner[k] < 1e-9: continue
                l_proc = min(L_t[k], f_inner[k] * T_avail[k] / phi) if T_avail[k] > 0 else 0
                e_total += kappa2 * phi * (f_inner[k] ** 2) * l_proc

            if e_total > E_max:
                nu_low = nu
            else:
                nu_high = nu
                f_final = f_inner.copy()

        return f_final

    def optimize_vectorized(self, L_t, Q_t, T_avail):
        """向量化版本：批量求解所有用户的卫星频率分配，数值结果与逐用户循环一致。"""
        nu_high_calc, mu_high_calc = self.get_search_bounds(L_t, Q_t, T_avail)

        phi = self.cfg.phi
        kappa2 = self.cfg.kappa2
        f_max = self.cfg.f_max_Sat
        E_max = self.cfg.E_max_Sat
        K_p = self.cfg.K_p
        w = self.cfg.w

        n_users = len(L_t)
        M_prime = (K_p * w) / f_max

        # ---------- 预计算不依赖 nu/mu 的量 ----------
        mask = L_t > 1e-6                                          # (N,) bool
        L = L_t.copy()
        t_av = T_avail.copy()

        # 阈值频率
        f_th = np.where(mask & (t_av > 1e-6),
                        phi * L / t_av, 1e14)                      # (N,)

        # Type A 三次方程中与 nu/mu 无关的系数
        d_A_base = np.where(mask, -K_p * phi * L, 0.0)             # (N,) 不含 nu
        a_A_factor = 2.0 * kappa2 * phi * L                        # (N,)  a = a_A_factor * nu

        # Type B 中的常数分子
        num_B = np.where(mask, Q_t / phi + M_prime, 0.0)           # (N,)

        # ---------- 外层二分搜索 nu ----------
        nu_low, nu_high = 0.0, nu_high_calc
        f_final = np.zeros(n_users)

        for _ in range(30):
            nu = (nu_low + nu_high) / 2.0
            if nu < 1e-15:
                nu = 1e-15

            denom_B = 3.0 * kappa2 * nu                            # scalar, Type B 分母基础

            # ---------- 内层二分搜索 mu ----------
            mu_low, mu_high = 0.0, mu_high_calc
            f_inner = np.zeros(n_users)

            for _ in range(30):
                mu = (mu_low + mu_high) / 2.0

                # --- 向量化 Type A ---
                a_A = np.where(mask, a_A_factor * nu, 0.0)
                f_A = solve_cubic_newton_vectorized(a_A, mu, d_A_base, self.cfg.newton_iter)

                # --- 向量化 Type B ---
                # f_B = sqrt( (q/phi + M') / (3*k2*nu)  -  mu / (3*k2*nu * t_av) )
                term1_B = num_B / denom_B
                term2_B = np.where(mask & (t_av > 1e-6),
                                   mu / (denom_B * t_av), np.inf)
                val_B = term1_B - term2_B
                f_B = np.zeros(n_users)
                valid_B = mask & (val_B > 0.0)
                if np.any(valid_B):
                    f_B[valid_B] = np.sqrt(val_B[valid_B])

                # --- 选择 ---
                f_temp = np.where(mask & (f_B < f_th), f_B, f_A)

                # --- 内层更新 mu ---
                if np.sum(f_temp) > f_max:
                    mu_low = mu
                else:
                    mu_high = mu
                    f_inner = f_temp.copy()

            # ---------- 向量化能耗计算 ----------
            l_proc = np.where(mask & (t_av > 0),
                              np.minimum(L, f_inner * t_av / phi), 0.0)
            e_total = np.sum(kappa2 * phi * (f_inner ** 2) * l_proc)

            # ---------- 外层更新 nu ----------
            if e_total > E_max:
                nu_low = nu
            else:
                nu_high = nu
                f_final = f_inner.copy()

        return f_final

    def optimize_multi_candidate(self, L_stack, Q_all, T_avail_stack):
        """
        一次性求解 K 个候选的全部 LEO 频率分配。
        各候选独立 nu/mu 二分搜索，每轮迭代处理 K×N 用户。

        :param L_stack:        (K, N) 各候选各用户卫星任务量
        :param Q_all:         (N,)   共享的卫星队列积压
        :param T_avail_stack: (K, N) 各候选各用户可用计算时间
        :return: f_stack (K, N)
        """
        K, N = L_stack.shape
        phi = self.cfg.phi
        kappa2 = self.cfg.kappa2
        f_max = self.cfg.f_max_Sat
        E_max = self.cfg.E_max_Sat
        K_p = self.cfg.K_p
        w = self.cfg.w
        M_prime = K_p * w / f_max

        cand_idx = np.repeat(np.arange(K), N)                                 # (K*N,)

        # 每个候选独立计算搜索上界（与逐候选调用一致）
        nu_hi = np.zeros(K)
        mu_hi = np.zeros(K)
        for k in range(K):
            nh, mh = self.get_search_bounds(L_stack[k], Q_all, T_avail_stack[k])
            nu_hi[k] = nh
            mu_hi[k] = mh

        # ---- 预计算 ----
        L_flat = L_stack.ravel()
        Q_flat = np.tile(Q_all, K)
        t_av = T_avail_stack.ravel()
        mask = L_flat > 1e-6
        f_th = np.where(mask & (t_av > 1e-6), phi * L_flat / t_av, 1e14)
        d_A = np.where(mask, -K_p * phi * L_flat, 0.0)
        a_base = 2.0 * kappa2 * phi * L_flat                                 # a = a_base * nu
        num_B = np.where(mask, Q_flat / phi + M_prime, 0.0)

        # ---- 外层 nu (K 路独立) ----
        nu_low = np.zeros(K)
        nu_high = np.full(K, nu_hi)
        f_final = np.zeros(K * N)

        for _ in range(30):
            nu = np.maximum((nu_low + nu_high) / 2.0, 1e-15)
            nu_u = nu[cand_idx]
            denom = 3.0 * kappa2 * nu_u

            # ---- 内层 mu (K 路独立) ----
            mu_low = np.zeros(K)
            mu_high = np.full(K, mu_hi)
            f_inner = np.zeros(K * N)

            for _ in range(30):
                mu = (mu_low + mu_high) / 2.0
                mu_u = mu[cand_idx]

                a_A = np.where(mask, a_base * nu_u, 0.0)
                f_A = solve_cubic_newton_vectorized(a_A, mu_u, d_A, self.cfg.newton_iter)

                term1 = num_B / denom
                term2 = np.where(mask & (t_av > 1e-6),
                                 mu_u / (denom * t_av), np.inf)
                val = term1 - term2
                f_B = np.zeros(K * N)
                ok = mask & (val > 0.0)
                if np.any(ok):
                    f_B[ok] = np.sqrt(val[ok])

                f_temp = np.where(mask & (f_B < f_th), f_B, f_A)

                sum_f = np.bincount(cand_idx, weights=f_temp, minlength=K)
                exceed = sum_f > f_max
                mu_low = np.where(exceed, mu, mu_low)
                mu_high = np.where(~exceed, mu, mu_high)
                f_inner = np.where(exceed[cand_idx], f_inner, f_temp)

            # 能耗
            l_proc = np.where(mask & (t_av > 0),
                              np.minimum(L_flat, f_inner * t_av / phi), 0.0)
            e_total = np.bincount(cand_idx,
                                  weights=kappa2 * phi * (f_inner ** 2) * l_proc,
                                  minlength=K)
            exceed = e_total > E_max
            nu_low = np.where(exceed, nu, nu_low)
            nu_high = np.where(~exceed, nu, nu_high)
            f_final = np.where(exceed[cand_idx], f_final, f_inner)

        return f_final.reshape(K, N)
