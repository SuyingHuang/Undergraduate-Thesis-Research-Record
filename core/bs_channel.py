import numpy as np

class BSChannel:
    """
    基站 (BS) 信道模型
    负责计算 UE 到 BS 的上行链路速率
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def calculate_uplink_rate(self, dist_m):
        """
        计算上行链路速率 (Shannon Formula)
        :param dist_m: UE 到 BS 的距离 (米), shape=(J,)
        :return: 速率 (bps), shape=(J,)
        """
        # 1. 自由空间路径损耗 (FSPL) 对应的信道增益
        # Gain = (lambda / 4 * pi * d)^2
        # 注意: 这里假设是 LOS 视距传播，若需 Shadowing 可后续在此扩展
        path_loss_gain = (self.cfg.wl_c / (4 * np.pi * dist_m)) ** 2

        # 2. 接收信噪比 (SNR)
        # Pr = P_tx * Gain * G_rx
        # SNR = Pr / Noise_Power
        rx_power = self.cfg.p_tx * path_loss_gain * self.cfg.G_rx_bs
        snr = rx_power / self.cfg.sigma1

        # 3. 香农公式计算速率
        # Bandwidth per User = B_c / J (OFDMA 平均分配)
        bw_per_user = self.cfg.B_c / self.cfg.J
        rate = bw_per_user * np.log2(1 + snr)

        return rate