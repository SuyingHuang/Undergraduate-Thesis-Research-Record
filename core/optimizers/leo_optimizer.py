import numpy as np
from utils.math_utils import solve_cubic_newton, solve_cubic_newton_vectorized, divide_where
from utils.objective import objective_coefficients, select_piecewise_frequency


class LEO_Optimizer:
    """
    实现论文 Algorithm 3: Computing Resource Optimization for LEOS
    解决卫星计算资源分配子问题。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        weights = objective_coefficients(cfg)
        self.queue_weight = weights['queue']
        self.paoi_weight = weights['paoi']
        energy_limited = (cfg.E_max_Sat / (cfg.kappa2 * cfg.tau)) ** (1 / 3)
        self.paoi_future_frequency = min(cfg.f_max_Sat, energy_limited)

    def get_search_bounds(self, L_t, Q_t, T_avail):
        """计算对偶变量上界的初始估计，实际求解前还会验证并扩展。"""
        phi = self.cfg.phi
        kappa2 = self.cfg.kappa2
        f_max = self.cfg.f_max_Sat
        K_p = self.paoi_weight
        w = self.cfg.w

        valid_mask = (L_t > 1e-6) & (T_avail > 1e-6)
        if not np.any(valid_mask):
            return 1.0, 1.0

        f_th_list = phi * L_t[valid_mask] / T_avail[valid_mask]
        nu_max_candidates = K_p / (2 * kappa2 * (f_th_list ** 3) + 1e-20)
        nu_high = np.max(nu_max_candidates) * 2.0
        nu_high = np.clip(nu_high, 1e-12, 1e10)

        M_prime = (K_p * w) / self.paoi_future_frequency
        term_b = self.queue_weight * Q_t[valid_mask] / phi + M_prime
        mu_max_candidates = term_b * T_avail[valid_mask]
        mu_high = np.max(mu_max_candidates) * 2.0

        return nu_high, mu_high

    def optimize(self, L_t, Q_t, T_avail):
        """原始逐用户循环版本（保留用于对照验证）"""
        nu_high_calc, mu_high_calc = self.get_search_bounds(L_t, Q_t, T_avail)

        phi = self.cfg.phi
        kappa2 = self.cfg.kappa2
        f_max = self.cfg.f_max_Sat
        E_max = self.cfg.E_max_Sat
        K_p = self.paoi_weight
        w = self.cfg.w

        n_users = len(L_t)
        M_prime = (K_p * w) / self.paoi_future_frequency

        def frequencies_at(nu, mu):
            f_temp = np.zeros(n_users)
            for k in range(n_users):
                L = L_t[k]
                t_av = T_avail[k]
                q = self.queue_weight * Q_t[k]
                if L <= 1e-6:
                    continue

                f_th = phi * L / t_av if t_av > 1e-6 else 1e14
                a = 2 * kappa2 * phi * nu * L
                d = -K_p * phi * L
                f_A = solve_cubic_newton(a, mu, d, iterations=self.cfg.newton_iter)

                denom = 3 * kappa2 * nu
                term1 = (q / phi + M_prime) / denom
                if t_av > 1e-6:
                    val = term1 - mu / (denom * t_av)
                    f_B = np.sqrt(val) if val > 0 else 0.0
                else:
                    f_B = 0.0

                f_temp[k] = float(select_piecewise_frequency(
                    L, Q_t[k], t_av, f_A, f_B, f_th, mu, nu,
                    self.cfg, self.queue_weight, K_p, kappa2, f_max,
                    self.paoi_future_frequency
                ))
            return f_temp

        def solve_capacity_dual(nu):
            mu_low = 0.0
            mu_high = max(float(mu_high_calc), 1e-18)
            f_high = frequencies_at(nu, mu_high)
            for _ in range(80):
                if np.sum(f_high) <= f_max:
                    break
                mu_low = mu_high
                mu_high *= 2.0
                f_high = frequencies_at(nu, mu_high)
            else:
                raise RuntimeError("Failed to bracket a feasible LEO resource dual")

            f_inner = f_high.copy()
            # Piecewise branch switches can make the feasible boundary very
            # sharp at the small physical scale of mu; retain enough binary
            # digits to avoid mistaking the feasible side for all-zero.
            for _ in range(60):
                mu = (mu_low + mu_high) / 2.0
                f_temp = frequencies_at(nu, mu)
                if np.sum(f_temp) > f_max:
                    mu_low = mu
                else:
                    mu_high = mu
                    f_inner = f_temp.copy()
            return f_inner

        def energy_of(frequency):
            l_proc = np.where(
                T_avail > 0,
                np.minimum(L_t, frequency * T_avail / phi),
                0.0,
            )
            return float(np.sum(kappa2 * phi * frequency ** 2 * l_proc))

        nu_low = 0.0
        nu_high = max(float(nu_high_calc), 1e-15)
        f_high = solve_capacity_dual(nu_high)
        for _ in range(80):
            if energy_of(f_high) <= E_max:
                break
            nu_low = nu_high
            nu_high *= 2.0
            f_high = solve_capacity_dual(nu_high)
        else:
            raise RuntimeError("Failed to bracket a feasible LEO energy dual")

        f_final = f_high.copy()
        for _ in range(30):
            nu = max((nu_low + nu_high) / 2.0, 1e-15)
            f_inner = solve_capacity_dual(nu)
            if energy_of(f_inner) > E_max:
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
        K_p = self.paoi_weight
        w = self.cfg.w

        n_users = len(L_t)
        M_prime = (K_p * w) / self.paoi_future_frequency

        # ---------- 预计算不依赖 nu/mu 的量 ----------
        mask = L_t > 1e-6                                          # (N,) bool
        L = L_t.copy()
        t_av = T_avail.copy()

        # 阈值频率
        f_th = divide_where(phi * L, t_av, mask & (t_av > 1e-6), 1e14)                      # (N,)

        # Type A 三次方程中与 nu/mu 无关的系数
        d_A_base = np.where(mask, -K_p * phi * L, 0.0)             # (N,) 不含 nu
        a_A_factor = 2.0 * kappa2 * phi * L                        # (N,)  a = a_A_factor * nu

        # Type B 中的常数分子
        num_B = np.where(mask, self.queue_weight * Q_t / phi + M_prime, 0.0)

        def frequencies_at(nu, mu):
            denom_B = 3.0 * kappa2 * nu                            # scalar, Type B 分母基础
            a_A = np.where(mask, a_A_factor * nu, 0.0)
            f_A = solve_cubic_newton_vectorized(a_A, mu, d_A_base, self.cfg.newton_iter)
            term1_B = num_B / denom_B
            term2_B = divide_where(mu, denom_B * t_av, mask & (t_av > 1e-6))
            val_B = term1_B - term2_B
            f_B = np.zeros(n_users)
            valid_B = mask & (val_B > 0.0)
            if np.any(valid_B):
                f_B[valid_B] = np.sqrt(val_B[valid_B])
            return select_piecewise_frequency(
                L, Q_t, t_av, f_A, f_B, f_th, mu, nu,
                self.cfg, self.queue_weight, K_p, kappa2, f_max,
                self.paoi_future_frequency
            )

        def solve_capacity_dual(nu):
            mu_low = 0.0
            mu_high = max(float(mu_high_calc), 1e-18)
            f_high = frequencies_at(nu, mu_high)
            for _ in range(80):
                if np.sum(f_high) <= f_max:
                    break
                mu_low = mu_high
                mu_high *= 2.0
                f_high = frequencies_at(nu, mu_high)
            else:
                raise RuntimeError("Failed to bracket a feasible LEO resource dual")

            f_inner = f_high.copy()
            for _ in range(60):
                mu = (mu_low + mu_high) / 2.0
                f_temp = frequencies_at(nu, mu)
                if np.sum(f_temp) > f_max:
                    mu_low = mu
                else:
                    mu_high = mu
                    f_inner = f_temp.copy()
            return f_inner

        def energy_of(frequency):
            l_proc = np.where(mask & (t_av > 0),
                              np.minimum(L, frequency * t_av / phi), 0.0)
            return float(np.sum(kappa2 * phi * frequency ** 2 * l_proc))

        # First bracket a frequency-feasible solution for mu and an
        # energy-feasible solution for nu.  The analytical bounds are only
        # starting guesses and are not assumed to be valid brackets.
        nu_low = 0.0
        nu_high = max(float(nu_high_calc), 1e-15)
        f_high = solve_capacity_dual(nu_high)
        for _ in range(80):
            if energy_of(f_high) <= E_max:
                break
            nu_low = nu_high
            nu_high *= 2.0
            f_high = solve_capacity_dual(nu_high)
        else:
            raise RuntimeError("Failed to bracket a feasible LEO energy dual")

        f_final = f_high.copy()
        for _ in range(30):
            nu = max((nu_low + nu_high) / 2.0, 1e-15)
            f_inner = solve_capacity_dual(nu)

            if energy_of(f_inner) > E_max:
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
        K_p = self.paoi_weight
        w = self.cfg.w
        M_prime = K_p * w / self.paoi_future_frequency

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
        f_th = divide_where(phi * L_flat, t_av, mask & (t_av > 1e-6), 1e14)
        d_A = np.where(mask, -K_p * phi * L_flat, 0.0)
        a_base = 2.0 * kappa2 * phi * L_flat                                 # a = a_base * nu
        num_B = np.where(mask, self.queue_weight * Q_flat / phi + M_prime, 0.0)

        def frequencies_at(nu, mu):
            nu_u = nu[cand_idx]
            mu_u = mu[cand_idx]
            denom = 3.0 * kappa2 * nu_u
            a_A = np.where(mask, a_base * nu_u, 0.0)
            f_A = solve_cubic_newton_vectorized(a_A, mu_u, d_A, self.cfg.newton_iter)
            term1 = num_B / denom
            term2 = divide_where(mu_u, denom * t_av, mask & (t_av > 1e-6))
            val = term1 - term2
            f_B = np.zeros(K * N)
            ok = mask & (val > 0.0)
            if np.any(ok):
                f_B[ok] = np.sqrt(val[ok])
            return select_piecewise_frequency(
                L_flat, Q_flat, t_av, f_A, f_B, f_th,
                mu_u, nu_u, self.cfg,
                self.queue_weight, K_p, kappa2, f_max,
                self.paoi_future_frequency
            )

        def solve_capacity_duals(nu):
            mu_low = np.zeros(K)
            mu_high = np.maximum(mu_hi.copy(), 1e-18)
            f_high = frequencies_at(nu, mu_high)
            for _ in range(80):
                sum_f = np.bincount(cand_idx, weights=f_high, minlength=K)
                exceed = sum_f > f_max
                if not np.any(exceed):
                    break
                mu_low = np.where(exceed, mu_high, mu_low)
                mu_high = np.where(exceed, mu_high * 2.0, mu_high)
                f_high = frequencies_at(nu, mu_high)
            else:
                raise RuntimeError("Failed to bracket feasible candidate LEO resource duals")

            f_inner = f_high.copy()
            for _ in range(60):
                mu = (mu_low + mu_high) / 2.0
                f_temp = frequencies_at(nu, mu)
                sum_f = np.bincount(cand_idx, weights=f_temp, minlength=K)
                exceed = sum_f > f_max
                mu_low = np.where(exceed, mu, mu_low)
                mu_high = np.where(~exceed, mu, mu_high)
                f_inner = np.where(exceed[cand_idx], f_inner, f_temp)
            return f_inner

        def energy_of(frequency):
            l_proc = np.where(mask & (t_av > 0),
                              np.minimum(L_flat, frequency * t_av / phi), 0.0)
            return np.bincount(
                cand_idx,
                weights=kappa2 * phi * frequency ** 2 * l_proc,
                minlength=K,
            )

        # ---- 先对每个候选独立括定 nu，再进行外层二分 ----
        nu_low = np.zeros(K)
        nu_high = np.maximum(nu_hi.copy(), 1e-15)
        f_high = solve_capacity_duals(nu_high)
        for _ in range(80):
            exceed = energy_of(f_high) > E_max
            if not np.any(exceed):
                break
            nu_low = np.where(exceed, nu_high, nu_low)
            nu_high = np.where(exceed, nu_high * 2.0, nu_high)
            f_high = solve_capacity_duals(nu_high)
        else:
            raise RuntimeError("Failed to bracket feasible candidate LEO energy duals")

        f_final = f_high.copy()
        for _ in range(30):
            nu = np.maximum((nu_low + nu_high) / 2.0, 1e-15)
            f_inner = solve_capacity_duals(nu)
            exceed = energy_of(f_inner) > E_max
            nu_low = np.where(exceed, nu, nu_low)
            nu_high = np.where(~exceed, nu, nu_high)
            f_final = np.where(exceed[cand_idx], f_final, f_inner)

        return f_final.reshape(K, N)
