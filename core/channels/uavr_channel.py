import numpy as np
from scipy.special import hyp1f1


class SimplifiedUAVRelayChannel:
    """
    简化的 UAV 中继通信信道模型
    协同分集协议: AF (Amplify-and-Forward) + MRC (Maximum Ratio Combining)

    链路构成:
    1. UE ↔ UAVr: 纯视距 (Pure LoS), 仅考虑自由空间路径损耗 (FSPL)
    2. UAVr ↔ LEO: 复用 Shadowed-Rician 衰落模型 (轻度阴影衰落)
    3. 协同分集: AF-MRC 合并
    """

    # UAV 高度 (m) - 典型低轨无人机高度 200 m
    H_UAV = 200

    # UAV 发射功率 (Watts) - 微基站级别
    P_TX_UAV_W = 0.1  # 20 dBm

    def __init__(self, cfg):
        self.cfg = cfg

        # UE ↔ UAVr 链路参数
        self.f_c_a2g = 2e9  # 2 GHz
        self.c = 3e8  # 光速 m/s
        self.noise_psd_dbm = -174  # 高斯白噪声功率谱密度 dBm/Hz

        # Shadowed-Rician 参数 (轻度阴影衰落 / 频繁视距)
        # 相比地面 UE↔LEO，使用更轻的阴影衰落参数
        self.b_k_uav = 0.1  # 减小 (原地面: 0.158)
        self.Omega_k_uav = 1.5  # 增大 (原地面: 1.29)
        self.m_k_uav = 25  # 增大 (原地面: 20)

        # 预计算 FSPL 常数项
        self.fspl_const = 20 * np.log10(4 * np.pi * self.f_c_a2g / self.c)

        # 预计算采样用的最大包络
        self.x_max_scan = 8.0
        self.f_max_envelope = self._find_envelope_max_uav()

    def _target_pdf_uav(self, x):
        """UAV 信道: |h_k|^2 的理论概率密度函数 (Shadowed-Rician)"""
        two_b = 2 * self.b_k_uav
        term1 = ((two_b * self.m_k_uav) / (two_b * self.m_k_uav + self.Omega_k_uav)) ** self.m_k_uav
        term2 = (1 / two_b) * np.exp(-x / two_b)
        hyp_arg = (self.Omega_k_uav * x) / (two_b * (two_b * self.m_k_uav + self.Omega_k_uav))
        term3 = hyp1f1(self.m_k_uav, 1, hyp_arg)
        return term1 * term2 * term3

    def _find_envelope_max_uav(self):
        """辅助函数: 寻找 UAV 信道 PDF 最大值用于采样"""
        x_test = np.linspace(0, self.x_max_scan, 1000)
        vals = [self._target_pdf_uav(i) for i in x_test]
        return np.max(vals) * 1.1

    def generate_uav_leo_channel_samples(self, n_samples):
        """
        生成 n 个符合轻度 Shadowed-Rician 分布的 UAV↔LEO 信道增益样本
        使用接受-拒绝采样
        """
        samples = []
        while len(samples) < n_samples:
            batch_size = n_samples * 2
            x_c = np.random.uniform(0, self.x_max_scan, batch_size)
            u = np.random.rand(batch_size)

            for i in range(batch_size):
                if u[i] < self._target_pdf_uav(x_c[i]) / self.f_max_envelope:
                    samples.append(x_c[i])
                    if len(samples) == n_samples:
                        break
        return np.array(samples)

    def calculate_ue_uavr_snr(self, d_3d, p_tx_ue_w, bw_hz):
        """
        计算 UE ↔ UAVr 空地链路的信噪比 (Pure LoS + FSPL)

        参数:
            d_3d: 三维距离 (m), shape=(I, J) 或标量
            p_tx_ue_w: UE 发射功率 (Watts)
            bw_hz: 分配带宽 (Hz)

        返回:
            gamma_ue_uavr: 信噪比 (线性值)
        """
        # FSPL (dB) = 20 * log10(4πf_c * d / c)
        fspl_db = self.fspl_const + 20 * np.log10(d_3d)

        # 接收功率 (dBm) = 发射功率 (dBm) - 路径损耗 (dB)
        # 注意: 不考虑收发端天线增益 (与现有基站模型一致)
        p_tx_dbm = 10 * np.log10(p_tx_ue_w * 1000)  # Watts -> dBm
        p_rx_dbm = p_tx_dbm - fspl_db

        # 噪声功率 (dBm)
        p_noise_dbm = self.noise_psd_dbm + 10 * np.log10(bw_hz)

        # 信噪比 (dB) -> 线性值
        snr_db = p_rx_dbm - p_noise_dbm
        gamma = 10 ** (snr_db / 10)

        return gamma

    def calculate_uavr_leo_snr(self, dist_m, h_sq_samples, bw_hz):
        """
        计算 UAVr ↔ LEO 星地链路的信噪比 (Shadowed-Rician 衰落)

        参数:
            dist_m: UAV 到卫星的距离 (m)
            h_sq_samples: 信道增益样本 |h|^2
            bw_hz: 分配带宽 (Hz)

        返回:
            gamma_uavr_leo: 信噪比 (线性值)
        """
        # 物理参数
        lambda_ka = self.c / self.cfg.f_c_sat  # Ka 波段波长
        beta = 2.2  # 路径损耗指数

        # 路径损耗常数
        pl_const_db = 20 * np.log10(4 * np.pi / lambda_ka)
        pl_dist_db = 10 * beta * np.log10(dist_m)

        # 衰落增益
        fading_db = 10 * np.log10(h_sq_samples)

        # UAV 发射功率 (dBm)
        p_tx_uavr_dbm = 10 * np.log10(self.P_TX_UAV_W * 1000)

        # 接收功率 (dBm)
        # 注意: UAV 具有高度优势，假设更好的天线配置
        # 使用与地面卫星链路相同的接收天线增益
        pr_dbm = (p_tx_uavr_dbm +
                  self.cfg.G_tx_ue_dbi +  # UAV 发射增益
                  self.cfg.G_rx_sat_dbi +  # 卫星接收增益
                  fading_db -
                  pl_const_db - pl_dist_db)

        # 线性值转换 (Watts)
        pr_linear = 10 ** ((pr_dbm - 30) / 10)

        # 噪声功率
        bw_per_user = bw_hz
        sigma2 = 1.37e-20 * bw_per_user  # 使用与卫星信道相同的噪声基底

        # 信噪比
        gamma = pr_linear / sigma2

        return gamma

    def calculate_cooperative_diversity_rate(self, gamma_ue_leo, gamma_ue_uavr, gamma_uavr_leo, bw_hz):
        """
        计算协同分集增强后的有效吞吐量 (AF + MRC)

        参数:
            gamma_ue_leo: 直连 UE↔LEO 信噪比 (线性值)
            gamma_ue_uavr: 空地 UE↔UAVr 信噪比 (线性值)
            gamma_uavr_leo: 星地 UAVr↔LEO 信噪比 (线性值)
            bw_hz: 分配带宽 (Hz)

        返回:
            R_cd: 有效吞吐量 (bps)
        """
        # AF-MRC 总信噪比
        # gamma_CD = gamma_ue_leo + (gamma_ue_uavr * gamma_uavr_leo) / (gamma_uavr_leo + zeta)
        # 固定惩罚因子 zeta = 1.0
        zeta = 1.0
        gamma_cd = gamma_ue_leo + (gamma_ue_uavr * gamma_uavr_leo) / (gamma_uavr_leo + zeta)

        # 有效吞吐量 (时分双工折半)
        # R = (B / 2) * log2(1 + gamma_CD)
        R_cd = (bw_hz / 2) * np.log2(1 + gamma_cd)

        return R_cd

    def calculate_enhanced_sat_rate(self, d_bs_mat, d_ue_leo_mat, h_sq_samples_mat, p_tx_ue_w, bw_hz):
        """
        计算增强后的卫星速率 (通过 UAV 中继协同)

        参数:
            d_bs_mat: UE 到基站的距离矩阵 (I, J), 单位米
            d_ue_leo_mat: UE 到卫星的距离矩阵 (I, J), 单位米
            h_sq_samples_mat: Shadowed-Rician 信道增益样本矩阵 (I, J)
            p_tx_ue_w: UE 发射功率 (Watts)
            bw_hz: 分配带宽 (Hz)

        返回:
            R_enhanced: 增强后的卫星速率矩阵 (I, J), bps
        """
        I, J = d_bs_mat.shape

        # 计算 UE ↔ UAVr 三维距离
        # 假设 UAV 固定在基站正上方，高度为 H_UAV
        d_ue_uavr = np.sqrt(d_bs_mat ** 2 + self.H_UAV ** 2)

        # 计算 UAVr ↔ LEO 距离 (垂直距离)
        d_uavr_leo = self.cfg.H_sat - self.H_UAV

        # 1. 计算 gamma_ue_uavr (Pure LoS)
        gamma_ue_uavr = self.calculate_ue_uavr_snr(d_ue_uavr, p_tx_ue_w, bw_hz)

        # 2. 计算 gamma_uavr_leo (Shadowed-Rician, 轻度阴影)
        # UAV↔LEO 距离对所有 UE 是相同的 (垂直链路)
        gamma_uavr_leo = self.calculate_uavr_leo_snr(d_uavr_leo, h_sq_samples_mat, bw_hz)

        # 3. 计算直连 gamma_ue_leo (用于对比/合并)
        # 使用与原卫星信道相同的方法计算 gamma_ue_leo
        # 这里简化处理: gamma_ue_leo = (H_sat / d_ue_leo)^2 * some_constant
        # 实际上应该调用 sat_channel 的计算逻辑
        # 为了保持一致，我们使用类似的路径损耗模型
        lambda_ka = self.c / self.cfg.f_c_sat
        beta = 2.2
        pl_const_db = 20 * np.log10(4 * np.pi / lambda_ka)
        pl_dist_db_leo = 10 * beta * np.log10(d_ue_leo_mat)

        # UE 发射功率
        p_tx_dbm_ue = 10 * np.log10(p_tx_ue_w * 1000)  # Watts -> dBm

        # 接收功率 (dBm) - 地面 UE 到卫星
        fading_db_ue_leo = 10 * np.log10(h_sq_samples_mat)
        pr_dbm_ue_leo = (p_tx_dbm_ue +
                         self.cfg.G_tx_ue_dbi +
                         self.cfg.G_rx_sat_dbi +
                         fading_db_ue_leo -
                         pl_const_db - pl_dist_db_leo)

        # 线性值
        pr_linear_ue_leo = 10 ** ((pr_dbm_ue_leo - 30) / 10)
        sigma2 = 1.37e-20 * bw_hz
        gamma_ue_leo = pr_linear_ue_leo / sigma2

        # 4. 计算协同分集增强速率
        R_enhanced = self.calculate_cooperative_diversity_rate(
            gamma_ue_leo, gamma_ue_uavr, gamma_uavr_leo, bw_hz
        )

        return R_enhanced