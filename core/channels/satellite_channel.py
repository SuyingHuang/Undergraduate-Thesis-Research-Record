import numpy as np
from scipy.special import hyp1f1


class SatelliteChannel:
    def __init__(self, cfg):
        self.cfg = cfg
        # Shadowed-Rician 参数 (来自 plot_shadowed_rician.py)
        self.b_k = 0.158
        self.Omega_k = 1.29
        self.m_k = 20

        # 预先计算采样用的最大包络 (用于接受-拒绝采样)
        self.x_max_scan = 8.0
        self.f_max_envelope = self._find_envelope_max()

        # 预生成样本池 (初始化时一次性生成10万个样本)
        self._sample_pool = None
        self._pregenerate_sample_pool(pool_size=100000)

    def _target_pdf(self, x):
        """ 公式 (5): |h_k|^2 的理论概率密度函数 [cite: 167] """
        two_b = 2 * self.b_k
        term1 = ((two_b * self.m_k) / (two_b * self.m_k + self.Omega_k)) ** self.m_k
        term2 = (1 / two_b) * np.exp(-x / two_b)
        hyp_arg = (self.Omega_k * x) / (two_b * (two_b * self.m_k + self.Omega_k))
        term3 = hyp1f1(self.m_k, 1, hyp_arg)
        return term1 * term2 * term3

    def _find_envelope_max(self):
        """网格搜索 PDF 最大值 + 解析 f(0) 作为下限，取 max 并加安全系数"""
        x_test = np.linspace(0, self.x_max_scan, 2000)
        vals = [self._target_pdf(x) for x in x_test]
        f0 = self._target_pdf(0)
        return max(f0, np.max(vals)) * 1.5

    def _pregenerate_sample_pool(self, pool_size):
        """ 预生成样本池 (使用接受-拒绝采样，仅在初始化时调用一次) """
        samples = []
        batch_size = 1000
        max_attempts = pool_size * 10

        attempts = 0
        while len(samples) < pool_size and attempts < max_attempts:
            x_c = np.random.uniform(0, self.x_max_scan, batch_size)
            u = np.random.rand(batch_size)

            for i in range(batch_size):
                if u[i] < self._target_pdf(x_c[i]) / self.f_max_envelope:
                    samples.append(x_c[i])
                    if len(samples) == pool_size:
                        break
            attempts += 1

        self._sample_pool = np.array(samples)

    def generate_channel_gain_samples(self, n_samples):
        """
        生成 n 个符合 Shadowed-Rician 分布的 |h_k|^2 样本
        从预生成的样本池中随机采样 (O(1) 复杂度)
        """
        if self._sample_pool is None or len(self._sample_pool) < n_samples:
            self._pregenerate_sample_pool(max(n_samples, 100000))

        # 从样本池中随机抽取 n 个样本
        indices = np.random.choice(len(self._sample_pool), size=n_samples, replace=True)
        return self._sample_pool[indices]

    def calculate_snr(self, dist_m, h_sq_samples):
        """
        计算 UE ↔ LEO 直连链路信噪比 (线性值)

        参数:
            dist_m: UE 到卫星的距离 (m)
            h_sq_samples: Shadowed-Rician 衰落系数 |h|^2

        返回:
            snr: 信噪比 (线性值)
        """
        lambda_ka = self.cfg.c / self.cfg.f_c_sat
        beta = 2.0

        PL_const_dB = 20 * np.log10(4 * np.pi / lambda_ka)
        PL_dist_dB = 10 * beta * np.log10(dist_m)
        fading_dB = 10 * np.log10(h_sq_samples)

        Pr_dBm = (self.cfg.p_tx_ue_sat_dbm +
                  self.cfg.G_tx_ue_dbi +
                  self.cfg.G_rx_sat_dbi +
                  fading_dB -
                  PL_const_dB - PL_dist_dB)

        Pr_linear = 10 ** ((Pr_dBm - 30) / 10)

        return Pr_linear / self.cfg.sigma2

    def calculate_uplink_rate(self, dist_m, h_sq_samples):
        """
        计算上行链路速率
        :param dist_m: UE 到卫星的距离 (米)
        :param h_sq_samples: Shadowed-Rician 衰落系数 |h|^2
        :return: 速率 (bps), 传播时延 (s)
        """
        snr = self.calculate_snr(dist_m, h_sq_samples)
        R_sat = self.cfg.bw_per_user_sat * np.log2(1 + snr)
        t_prop = dist_m / self.cfg.c

        return R_sat, t_prop