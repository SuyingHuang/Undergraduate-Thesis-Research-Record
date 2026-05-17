import numpy as np


class SystemConfig:
    def __init__(self):
        # --- 1. 拓扑与环境参数 ---
        self.I = 3  # BS数量 (论文 Table II 为 3)
        self.J = 10  # 每个BS的用户数 (论文 Table II 为 10)
        self.d_min = 500.0
        self.d_max = 1500.0

        # --- 2. 物理常数 (新增) ---
        self.k_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)
        self.T0 = 290.0  # 标准参考温度 (K)

        # --- 3. 信道与噪声模型 (C-band BS) ---
        self.f_c = 6e9
        self.c = 3e8
        self.wl_c = self.c / self.f_c
        self.B_c = 500e6  # 每个BS的C-band带宽 500 MHz (单BS服务J个用户，每用户 B_c/J)
        self.p_tx = 0.2  # UE 发射功率 (Watts)

        # [新增] BS 噪声参数 (由用户指定)
        self.NF_BS_dB = 5.0  # BS 接收机噪声系数 (dB) - 典型值
        self.T_ant_BS = 290.0  # BS 天线温度 (K) - 视向地面环境温度
        self.G_rx_bs =316     #约等于25dBi
        # --- 4. 信道与噪声模型 (UAV air-to-ground sub-6 GHz) ---
        self.f_c_uav = 4e9  # C 频段 (UE ↔ UAVr)
        self.NF_UAV_dB = 3.0  # UAV 接收机噪声系数 (dB)
        self.T_ant_UAV = 290.0  # UAV 天线温度 (K)

        # --- 5. 信道与噪声模型 (Ka-band Satellite) ---
        self.H_sat = 600e3
        self.f_c_sat = 30e9
        self.B_sat = 800e6  # 卫星总带宽

        # [新增] Satellite 噪声参数 (由用户指定)
        self.NF_Sat_dB = 1.0  # 卫星接收机噪声系数 (dB) - Ka波段LNA典型值
        self.T_ant_Sat = 290.0  # 卫星天线温度 (K) - 天线指向地球

        self.p_tx_ue_sat_dbm = 23.0  # UE卫星发射功率 (dBm) = 0.2W
        self.G_tx_ue_dbi = 20.0  # 移动终端定向卫星天线增益 (dBi)
        self.G_rx_sat_dbi = 44.0
        self.beta_uavr_leo = 2.0  # UAVr-LEO 路径损耗指数

        # --- 6. 噪声功率计算 (初步阶段：平均分配) ---
        self._update_bandwidth_params()

        # --- 7. 时间与仿真参数 ---
        self.tau = 5.0
        self.sim_frames = 4096
        self.use_uav_relay = True   # 是否启用 UAV 中继增强星地链路

        # --- 8. 计算与能耗 (BS & Sat) ---
        self.E_max_BS = 180.0
        self.f_max_BS = 4e9
        self.phi = 100
        self.kappa1 = 2e-26

        self.E_max_Sat = 80.0
        self.f_max_Sat = 2e9
        self.kappa2 = 1e-26

        # --- 9. 优化参数 ---
        self.w = 2.0
        self.K_p = 0.1  # 归一化权重法中固定为1，让lambda_p独立控制PAoI权重
        self.L_mean = 12e6
        self.L_std = 3e6
        self.newton_iter = 10

        # G_1 量纲均衡参考尺度 (5种子 × 200帧探索期, 跨种子中位数之中位数)
        self.Q_ref = 1.28775e6     # 队列项参考尺度
        self.PAoI_ref = 21.0241    # PAoI项参考尺度
        self.E_ref = 1.87961e4     # 能量项参考尺度

        #----UE的参数
        self.f_max_UE=1e8       #这是可以调整的

        # --- 10. DNN与训练参数 (新增) ---
        self.hidden_dim = 512  # 神经网络隐藏层维度
        self.lr = 1e-3  # 学习率
        self.batch_size = 64  # 训练批次大小
        self.memory_capacity =1024   # 经验回放池容量
        self.train_interval = 10  # 每多少帧训练一次
        self.focal_alpha = 0.5  # Focal Loss 参数 alpha
        self.focal_gamma = 0.0  # Focal Loss 参数 gamma

        # --- 10.5. 探索窗口自适应参数 ---
        self.delta_init = 0.5       # 初始探索窗口
        self.delta_min = 0.08       # 最小探索窗口 (越大保留越多候选)
        self.delta_max = 0.5        # 最大探索窗口
        self.delta_ema_fast = 0.9   # 快速EMA系数 (跟踪近期loss, 越小越灵敏)
        self.delta_ema_slow = 0.99  # 慢速EMA系数 (长期基线, 越接近1越稳定)
        self.delta_decay = 0.985    # 衰减因子 (越接近1收敛越慢)
        self.delta_grow = 1.008     # 增长因子 (略大于衰减，保持对称)
        self.delta_ratio_lo = 0.95  # 低于此比值触发收缩 (越小越不敏感)

        # --- 11. Multi-seed 实验参数 (新增) ---
        self.seeds = [42, 123, 456, 789]  # 默认实验种子列表

    def _update_bandwidth_params(self):
        """当 B_c 或 B_sat 改变时重新计算带宽相关参数"""
        self.bw_per_user_bs = self.B_c / self.J
        self.sigma1 = self._calculate_noise_power(
            self.bw_per_user_bs, self.NF_BS_dB, self.T_ant_BS
        )
        self.bw_per_user_sat = self.B_sat / (self.I * self.J)
        self.sigma2 = self._calculate_noise_power(
            self.bw_per_user_sat, self.NF_Sat_dB, self.T_ant_Sat
        )
        self.sigma_uavr = self._calculate_noise_power(
            self.bw_per_user_bs, self.NF_UAV_dB, self.T_ant_UAV
        )

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
