import numpy as np


class SystemConfig:
    def __init__(self):
        # --- 拓扑参数 ---
        self.J = 10  # 用户数

        # --- [新增] 信道模型参数 (基于论文 Table II 和 描述) ---
        self.f_c = 6e9  # 载波频率 6 GHz [Snippet 1474]
        self.c = 3e8  # 光速
        self.wl = self.f_c / self.c  # 波长 lambda (注意: lambda_c = c/f)
        # 修正: 应该是 c/f = 0.05m
        self.wl_c = 3e8 / 6e9

        self.B_c = 500e6  # C-band 总带宽 500 MHz [Table II]
        self.sigma2 = 2.2e-12  # 噪声功率 (Watts) [Table II]
        self.p_tx = 0.2  # UE 发射功率 (假设 200mW/23dBm，论文未详述但为典型值)

        self.d_min = 300.0  # 距离分布下界 (m) [Snippet 1476]
        self.d_max = 600.0  # 距离分布上界 (m) [Snippet 1476]

        # --- 时间参数 ---
        self.tau = 5.0  # 帧长 (s)
        self.sim_frames = 5000

        # --- 计算与能耗参数 ---
        self.E_max_BS = 120.0
        self.f_max_BS = 4e9
        self.phi = 100
        self.kappa1 = 2e-26

        # --- 优化参数 ---
        self.w = 2.0  # PAoI 惩罚权重
        self.K_p = 300.0  # Lyapunov 参数

        # --- 任务参数 ---
        self.L_mean = 8e6
        self.L_std = 5e6

        # --- [新增] 牛顿法参数 ---
        self.newton_iter = 10  # 论文提到 3-5 次即可，给 10 次余量
        self.newton_tol = 1e-6