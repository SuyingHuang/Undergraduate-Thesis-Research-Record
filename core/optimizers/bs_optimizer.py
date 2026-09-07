import numpy as np
from utils.math_utils import solve_cubic_newton, solve_cubic_newton_vectorized, divide_where
from utils.objective import objective_coefficients, select_piecewise_frequency


class BS_Optimizer:
    """
    Algorithm 2: Optimal Frequency Allocation for BS
    解决 Problem P3
    """

    def __init__(self, cfg):
        self.cfg = cfg
        weights = objective_coefficients(cfg)
        self.queue_weight = weights['queue']
        self.paoi_weight = weights['paoi']
        self.energy_weight = weights['energy']

    def optimize(self, L_t, Q_t, E_t, T_tran, T_left_prev):
        """原始逐用户循环版本（保留用于对照验证）"""
        J = self.cfg.J
        phi = self.cfg.phi
        kappa1 = self.cfg.kappa1
        tau = self.cfg.tau
        f_max = self.cfg.f_max_BS
        K_p = self.paoi_weight
        w = self.cfg.w

        E_safe = max(E_t * self.energy_weight, 1e-12)
        M = K_p * w / f_max

        lambda_limits = (self.queue_weight * Q_t / phi + M) * tau
        lambda_high = max(float(np.max(lambda_limits)) * 1.01, 1e-18)
        lambda_low = 0.0

        f_final = np.zeros(J)

        for _ in range(60):
            lam = (lambda_low + lambda_high) / 2
            if lam < 1e-20: lam = 1e-20

            f_temp = np.zeros(J)
            term_B_denom = 3 * E_safe * kappa1

            for j in range(J):
                L = L_t[j]
                if L <= 1e-6: continue
                delay_occupancy = max(T_tran[j], T_left_prev)
                t_avail = tau - delay_occupancy

                if t_avail <= 1e-6:
                    f_th = float('inf')
                else:
                    f_th = phi * L / t_avail

                a = 2 * E_safe * kappa1 * phi * L
                b = lam
                d = - K_p * phi * L
                f_A = solve_cubic_newton(a, b, d, self.cfg.newton_iter)

                if t_avail <= 1e-6:
                    f_B = 0.0
                else:
                    num = (self.queue_weight * Q_t[j] / phi + M)
                    term_lam = lam / (term_B_denom * t_avail)
                    val = num / term_B_denom - term_lam
                    f_B = np.sqrt(val) if val > 0 else 0.0

                f_temp[j] = float(select_piecewise_frequency(
                    L, Q_t[j], t_avail, f_A, f_B, f_th, lam, E_safe,
                    self.cfg, self.queue_weight, K_p, kappa1, f_max
                ))

            if np.sum(f_temp) > f_max:
                lambda_low = lam
            else:
                lambda_high = lam
                f_final = f_temp.copy()

        return f_final

    def optimize_vectorized(self, L_t, Q_t, E_t, T_tran, T_left_prev):
        """向量化版本：批量求解所有用户的频率分配，数值结果与逐用户循环一致。"""
        J = self.cfg.J
        phi = self.cfg.phi
        kappa1 = self.cfg.kappa1
        tau = self.cfg.tau
        f_max = self.cfg.f_max_BS
        K_p = self.paoi_weight
        w = self.cfg.w

        E_safe = max(E_t * self.energy_weight, 1e-12)
        M_scalar = K_p * w / f_max

        lambda_limits = (self.queue_weight * Q_t / phi + M_scalar) * tau
        lambda_high = max(float(np.max(lambda_limits)) * 1.01, 1e-18)
        lambda_low = 0.0

        f_final = np.zeros(J)

        # ---------- 预计算不依赖 lam 的量 ----------
        mask = L_t > 1e-6                                     # (J,) bool
        L = L_t.copy()
        delay_occ = np.maximum(T_tran, T_left_prev)           # scalar broadcast → (J,)
        t_avail = tau - delay_occ                             # (J,)

        # 阈值频率 f_th: 恰好做完任务的频率
        f_th = divide_where(phi * L, t_avail, mask & (t_avail > 1e-6))            # (J,)

        # Type A 三次方程的 d 系数（与 lam 无关）
        d_A = np.where(mask, -K_p * phi * L, 0.0)             # (J,)

        # Type B 分子（与 lam 无关）
        num_B = np.where(mask, self.queue_weight * Q_t / phi + M_scalar, 0.0)

        # Type B 分母中的常数因子
        denom_B_base = 3.0 * E_safe * kappa1                  # scalar

        # ---------- 二分搜索 lambda ----------
        for _ in range(60):
            lam = (lambda_low + lambda_high) / 2.0
            if lam < 1e-20:
                lam = 1e-20

            # --- 向量化 Type A ---
            a_A = np.where(mask, 2.0 * E_safe * kappa1 * phi * L, 0.0)
            f_A = solve_cubic_newton_vectorized(a_A, lam, d_A, self.cfg.newton_iter)

            # --- 向量化 Type B ---
            # expr: sqrt( num_B / denom_B_base  -  lam / (denom_B_base * t_avail) )
            term_lam_B = divide_where(lam, denom_B_base * t_avail, mask & (t_avail > 1e-6))
            val_B = num_B / denom_B_base - term_lam_B
            f_B = np.zeros(J)
            valid_B = mask & (val_B > 0.0)
            if np.any(valid_B):
                f_B[valid_B] = np.sqrt(val_B[valid_B])

            # 比较两个分段定义域内的候选，也允许最优点落在完成阈值处。
            f_temp = select_piecewise_frequency(
                L, Q_t, t_avail, f_A, f_B, f_th, lam, E_safe,
                self.cfg, self.queue_weight, K_p, kappa1, f_max
            )

            # --- Lambda 更新 ---
            if np.sum(f_temp) > f_max:
                lambda_low = lam
            else:
                lambda_high = lam
                f_final = f_temp.copy()

        self._last_lambda = (lambda_low + lambda_high) / 2
        return f_final

    def optimize_batched(self, L_all, Q_all, E_per_bs, T_tran_all, T_left_per_bs):
        """
        批量求解所有 BS 的频率分配 (I*J 用户一次性向量化)。
        各 BS 独立二分搜索各自的 lambda，每轮迭代所有用户并行计算。

        :param L_all:         形状 (N,)  所有用户任务量 (N = I*J)
        :param Q_all:         形状 (N,)  所有用户 BS 积压
        :param E_per_bs:      形状 (I,)  每个 BS 的能量队列
        :param T_tran_all:    形状 (N,)  所有用户传输延迟
        :param T_left_per_bs: 形状 (I,)  每个 BS 上一帧残留时间
        :return: f_all 形状 (N,)
        """
        I = self.cfg.I
        J = self.cfg.J
        N = I * J
        phi, kappa1, tau = self.cfg.phi, self.cfg.kappa1, self.cfg.tau
        f_max, K_p, w = self.cfg.f_max_BS, self.paoi_weight, self.cfg.w

        bs_idx = np.repeat(np.arange(I), J)                        # (N,)
        E_safe_per_bs = np.maximum(E_per_bs * self.energy_weight, 1e-12)
        E_safe = E_safe_per_bs[bs_idx]                             # (N,)
        M_scalar = K_p * w / f_max

        # 每个 BS 的 lambda 上界
        lam_limits = (self.queue_weight * Q_all / phi + M_scalar) * tau
        lam_high = np.array([max(float(np.max(lam_limits[bs_idx == i])) * 1.01, 1e-18)
                             if np.any(bs_idx == i) else 1e-18 for i in range(I)])
        lam_low = np.zeros(I)
        f_final = np.zeros(N)

        # ---- 预计算 ----
        mask = L_all > 1e-6
        L = L_all.copy()
        T_left = T_left_per_bs[bs_idx]
        t_avail = tau - np.maximum(T_tran_all, T_left)
        f_th = divide_where(phi * L, t_avail, mask & (t_avail > 1e-6))
        d_A = np.where(mask, -K_p * phi * L, 0.0)
        num_B = np.where(mask, self.queue_weight * Q_all / phi + M_scalar, 0.0)
        base_B = 3.0 * E_safe * kappa1
        a_factor = 2.0 * E_safe * kappa1 * phi * L

        # ---- 统一二分搜索 (60 轮) ----
        for _ in range(60):
            lam = np.maximum((lam_low + lam_high) / 2.0, 1e-20)   # (I,)
            lam_u = lam[bs_idx]                                     # (N,)

            # Type A
            a_A = np.where(mask, a_factor, 0.0)
            f_A = solve_cubic_newton_vectorized(a_A, lam_u, d_A, self.cfg.newton_iter)

            # Type B
            term_lam = divide_where(lam_u, base_B * t_avail, mask & (t_avail > 1e-6))
            val = num_B / base_B - term_lam
            f_B = np.zeros(N)
            ok = mask & (val > 0.0)
            if np.any(ok):
                f_B[ok] = np.sqrt(val[ok])

            f_temp = select_piecewise_frequency(
                L, Q_all, t_avail, f_A, f_B, f_th, lam_u, E_safe,
                self.cfg, self.queue_weight, K_p, kappa1, f_max
            )

            # 按 BS 分组求和 → 各自判断是否超额
            sum_f = np.bincount(bs_idx, weights=f_temp, minlength=I)
            exceed = sum_f > f_max
            lam_low = np.where(exceed, lam, lam_low)
            lam_high = np.where(~exceed, lam, lam_high)
            f_final = np.where(exceed[bs_idx], f_final, f_temp)

        self._last_lambda = float(np.mean((lam_low + lam_high) / 2))
        return f_final

    def optimize_multi_candidate(self, L_stack, Q_all, E_per_bs, T_tran_stack, T_left_per_bs):
        """
        一次性求解 K 个候选的全部 BS 频率分配。
        K×I 组 lambda 独立搜索，每轮处理 K×N 用户。

        :param L_stack:       (K, N)
        :param Q_all:        (N,)  所有候选共享的 BS 积压
        :param E_per_bs:     (I,)  共享的能量队列
        :param T_tran_stack: (K, N)
        :param T_left_per_bs:(I,)  共享的残留时间
        :return: f_stack (K, N)
        """
        K, N = L_stack.shape
        I, J = self.cfg.I, self.cfg.J
        phi, kappa1, tau = self.cfg.phi, self.cfg.kappa1, self.cfg.tau
        f_max, K_p, w = self.cfg.f_max_BS, self.paoi_weight, self.cfg.w

        # 组: g = k*I + i, 共 K*I 组, 每组 J 个用户
        bs_per_user = np.repeat(np.arange(I), J)                              # (N,)
        group_idx = (np.arange(K)[:, None] * I + bs_per_user[None, :]).ravel()  # (K*N,)

        E_safe_per_bs = np.maximum(E_per_bs * self.energy_weight, 1e-12)
        E_safe_flat = np.tile(E_safe_per_bs[bs_per_user], K)                   # (K*N,)
        M_scalar = K_p * w / f_max

        # Lambda 上界 per (candidate, BS) group
        lam_limits = (self.queue_weight * np.tile(Q_all, K) / phi + M_scalar) * tau
        lam_high = np.array([max(float(np.max(lam_limits[group_idx == g])) * 1.01, 1e-18)
                             if np.any(group_idx == g) else 1e-18
                             for g in range(K * I)])
        lam_low = np.zeros(K * I)
        f_final = np.zeros(K * N)

        # ---- 预计算 ----
        L_flat = L_stack.ravel()
        Q_flat = np.tile(Q_all, K)
        T_left_flat = np.tile(T_left_per_bs[bs_per_user], K)
        t_avail = tau - np.maximum(T_tran_stack.ravel(), T_left_flat)
        mask = L_flat > 1e-6
        f_th = divide_where(phi * L_flat, t_avail, mask & (t_avail > 1e-6))
        d_A = np.where(mask, -K_p * phi * L_flat, 0.0)
        num_B = np.where(mask, self.queue_weight * Q_flat / phi + M_scalar, 0.0)
        base_B = 3.0 * E_safe_flat * kappa1
        a_factor = 2.0 * E_safe_flat * kappa1 * phi * L_flat

        # ---- 统一二分 ----
        for _ in range(60):
            lam = np.maximum((lam_low + lam_high) / 2.0, 1e-20)               # (K*I,)
            lam_u = lam[group_idx]                                             # (K*N,)
            a_A = np.where(mask, a_factor, 0.0)
            f_A = solve_cubic_newton_vectorized(a_A, lam_u, d_A, self.cfg.newton_iter)
            term_lam = divide_where(lam_u, base_B * t_avail, mask & (t_avail > 1e-6))
            val = num_B / base_B - term_lam
            f_B = np.zeros(K * N)
            ok = mask & (val > 0.0)
            if np.any(ok):
                f_B[ok] = np.sqrt(val[ok])
            f_temp = select_piecewise_frequency(
                L_flat, Q_flat, t_avail, f_A, f_B, f_th,
                lam_u, E_safe_flat, self.cfg,
                self.queue_weight, K_p, kappa1, f_max
            )
            sum_f = np.bincount(group_idx, weights=f_temp, minlength=K * I)
            exceed = sum_f > f_max
            lam_low = np.where(exceed, lam, lam_low)
            lam_high = np.where(~exceed, lam, lam_high)
            f_final = np.where(exceed[group_idx], f_final, f_temp)

        self._last_lambda = float(np.mean((lam_low + lam_high) / 2))
        return f_final.reshape(K, N)
