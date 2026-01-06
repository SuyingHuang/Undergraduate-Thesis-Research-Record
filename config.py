import numpy as np


class SystemConfig:
    def __init__(self):
        # --- 1. 拓扑与环境参数 ---
        self.J = 10  # 用户数 (UE)
        self.d_min = 300.0  # 用户到基站距离下界 (m)
        self.d_max = 600.0  # 用户到基站距离上界 (m)

        # --- 2. 信道模型 (Free Space Path Loss) ---
        self.f_c = 6e9  # 载波频率 6 GHz
        self.c = 3e8  # 光速
        self.wl_c = self.c / self.f_c  # 波长
        self.B_c = 500e6  # C-band 总带宽 500 MHz
        self.sigma2 = 2.2e-12  # 噪声功率 (Watts)
        self.p_tx = 0.2  # UE 发射功率 (Watts)

        # --- 3. 时间与仿真参数 ---
        self.tau = 5.0  # 帧长 (s)
        self.sim_frames = 1000  # 仿真总帧数

        # --- 4. 基站计算与能耗 (BS) ---
        self.E_max_BS = 120.0  # 能量预算 (J)
        self.f_max_BS = 4e9  # 最大频率 4GHz
        self.phi = 100  # 计算密度 cycles/bit
        self.kappa1 = 2e-26

        # --- 优化参数 ---
        self.w = 2.0  # PAoI 惩罚权重
        self.K_p = 300.0  # Lyapunov 参数

        # --- 任务参数 ---
        self.L_mean = 8e6
        self.L_std = 5e6

        # --- [新增] 牛顿法参数 ---
        self.newton_iter = 10  # 论文提到 3-5 次即可，给 10 次余量
        # --- 8. [预留] 卫星参数 (Satellite) ---
        # 待 Algorithm 3 确认后填充
        pass
