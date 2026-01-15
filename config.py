import numpy as np


class SystemConfig:
    def __init__(self):
        # --- 1. 拓扑与环境参数 ---
        self.I = 2 # BS数量
        self.J = 2  # 用户数 (UE)
        self.d_min = 300.0  # 用户到基站距离下界 (m)
        self.d_max = 600.0  # 用户到基站距离上界 (m)

        # --- 2. 信道模型 (Free Space Path Loss) ---
        self.f_c = 6e9  # 载波频率 6 GHz
        self.c = 3e8  # 光速
        self.wl_c = self.c / self.f_c  # 波长
        self.B_c = 500e6  # C-band 总带宽 500 MHz
        self.sigma1 = 2.2e-12  # 噪声功率 (Watts)
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
        self.K_p = 3e3  # Lyapunov 参数

        # --- 任务参数 ---
        self.L_mean = 15e6
        self.L_std = 3e6

        # --- [新增] 牛顿法参数 ---
        self.newton_iter = 10  # 论文提到 3-5 次即可，给 10 次余量
        # --- 8. 卫星参数 (Satellite) ---

        # 物理位置
        self.H_sat = 600e3  # 卫星高度 (这里我把卫星高度改成600 km, LEO)

        # 信道: Ka-band (通常卫星用 Ka 波段，约 28GHz，这里取30GHz)
        self.f_c_sat = 30e9
        self.B_sat = 800e6  # 卫星总带宽 (假设 800MHz,原论文的1Ghz有点大了)
        self.p_tx_ue_sat_dbm = 33.0  # UE -> Sat 发射功率 (卫星远，功率需大，假设 2W)
        self.G_tx_ue_dbi = 42.0  # UE 天线增益
        self.G_rx_sat_dbi = 44.0  # 卫星天线增益 (线性值, 30dBi)
        # 计算与能耗
        self.E_max_Sat = 60.0  # 卫星能耗限制(TABLE II)
        self.f_max_Sat =2e9   # 卫星最大频率 ()
        self.kappa2 =1e-26   # 卫星能耗系数 ()
