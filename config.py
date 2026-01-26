import numpy as np


class SystemConfig:
    def __init__(self):
        # --- 1. 拓扑与环境参数 ---
        self.I = 2  # BS数量 (论文 Table II 为 3)
        self.J = 3  # 每个BS的用户数 (论文 Table II 为 10)
        self.d_min = 500.0
        self.d_max = 1500.0

        # --- 2. 物理常数 (新增) ---
        self.k_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)
        self.T0 = 290.0  # 标准参考温度 (K)

        # --- 3. 信道与噪声模型 (C-band BS) ---
        self.f_c = 6e9
        self.c = 3e8
        self.wl_c = self.c / self.f_c
        self.B_c = 500e6  # C-band 总带宽 500 MHz
        self.p_tx = 0.2  # UE 发射功率 (Watts)

        # [新增] BS 噪声参数 (由用户指定)
        self.NF_BS_dB = 5.0  # BS 接收机噪声系数 (dB) - 典型值
        self.T_ant_BS = 290.0  # BS 天线温度 (K) - 视向地面环境温度
        self.G_rx_bs =316     #约等于25dBi
        # --- 4. 信道与噪声模型 (Ka-band Satellite) ---
        self.H_sat = 600e3
        self.f_c_sat = 30e9
        self.B_sat = 800e6  # 卫星总带宽

        # [新增] Satellite 噪声参数 (由用户指定)
        self.NF_Sat_dB = 3.0  # 卫星接收机噪声系数 (dB) - LNA通常较好
        self.T_ant_Sat = 290.0  # 卫星天线温度 (K) - 天线指向地球

        self.p_tx_ue_sat_dbm = 33.0
        self.G_tx_ue_dbi = 42.0
        self.G_rx_sat_dbi = 44.0

        # --- 5. 噪声功率计算 (初步阶段：平均分配) ---
        # 依据论文 Eq. (3): BS带宽由 J 个用户共享
        self.bw_per_user_bs = self.B_c / self.J
        self.sigma1 = self._calculate_noise_power(
            self.bw_per_user_bs, self.NF_BS_dB, self.T_ant_BS
        )

        # 依据论文 Eq. (7): Sat带宽由 I*J 个用户共享
        self.bw_per_user_sat = self.B_sat / (self.I * self.J)
        self.sigma2 = self._calculate_noise_power(
            self.bw_per_user_sat, self.NF_Sat_dB, self.T_ant_Sat
        )

        # 打印以供核对 (可选)
        # print(f"BS Noise Power (sigma1): {self.sigma1:.2e} W")
        # print(f"Sat Noise Power (sigma2): {self.sigma2:.2e} W")

        # --- 6. 时间与仿真参数 ---
        self.tau = 5.0
        self.sim_frames = 1000

        # --- 7. 计算与能耗 (BS & Sat) ---
        self.E_max_BS = 120.0
        self.f_max_BS = 4e9
        self.phi = 100
        self.kappa1 = 2e-26

        self.E_max_Sat = 60.0
        self.f_max_Sat = 2e9
        self.kappa2 = 1e-26

        # --- 8. 优化参数 ---
        self.w = 2.0
        self.K_p = 3e3
        self.L_mean = 15e6
        self.L_std = 3e6
        self.newton_iter = 10

        #----UE的参数
        self.f_max_UE=1e8       #这是可以调整的


    def _calculate_noise_power(self, bandwidth, nf_db, t_antenna):
        """
        根据带宽和噪声参数计算热噪声功率 (Watts)
        Formula: N = k * T_sys * B
        T_sys = T_antenna + T_effective
        T_effective = T0 * (10^(NF/10) - 1)
        """
        # 1. 将 NF(dB) 转换为线性噪声因子 F
        noise_factor = 10 ** (nf_db / 10.0)

        # 2. 计算接收机等效噪声温度 T_e
        t_effective = self.T0 * (noise_factor - 1)

        # 3. 计算系统总噪声温度 T_sys
        t_sys = t_antenna + t_effective

        # 4. 计算总噪声功率 kTB
        noise_power = self.k_B * t_sys * bandwidth

        return noise_power