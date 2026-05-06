import numpy as np


class UAVRelayOptimizer:
    """
    基于延迟驱动的启发式功率解耦优化机制

    参考 Bhola 2024 Table 1 参数设定:
    - varsigma: 1.0 (AF中继常量)
    - 噪声功率使用 config.py 中统一计算的 sigma2
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # UAV最大发射功率: 40 dBm = 10 W
        self.p_max_w = 10.0

        # UAV最小发射功率: 2 W
        self.p_min_w = 2

        # AF中继常量
        self.varsigma = 1.0

    def optimize_power(self, D, T_max, bw_hz, snr_ue_leo, snr_ue_uavr,
                       h_gain_uavr_leo_linear, noise_power):
        """
        计算UAV中继的最优发射功率

        参数:
            D: 卸载数据量 (bits)
            T_max: 最大容忍延迟 (s)
            bw_hz: 分配带宽 (Hz)
            snr_ue_leo: UE-LEO直连信噪比 (线性值)
            snr_ue_uavr: UE-UAVr信噪比 (线性值)
            h_gain_uavr_leo_linear: UAVr-LEO纯信道增益 (线性值)
            noise_power: 噪声功率 (W)

        返回:
            optimal_power_w: 最优发射功率 (W)
        """
        # 1. 计算目标数据速率
        r_target = D / T_max if T_max > 1e-9 else 1e15

        # 2. 计算目标协同信噪比 (时分双工折半)
        # R = (B/2) * log2(1 + snr_cd) => snr_cd = 2^(2*R/B) - 1
        snr_cd_target = 2 ** (2 * r_target / bw_hz) - 1

        # 3. 计算增量信噪比
        delta_snr = snr_cd_target - snr_ue_leo

        # 4. 分支判断
        if delta_snr <= 0:
            # 信道条件已满足，无需UAV中继辅助
            return self.p_min_w

        if delta_snr >= snr_ue_uavr:
            # 目标不可达，返回最大功率
            return self.p_max_w

        # 5. 计算需求的UAVr-LEO信噪比 (AF-MRC反推)
        # gamma_cd = gamma_ue_leo + (gamma_ue_uavr * gamma_uavr_leo) / (gamma_uavr_leo + varsigma)
        # delta_snr = (gamma_ue_uavr * gamma_uavr_leo) / (gamma_uavr_leo + varsigma)
        # 求解 gamma_uavr_leo:
        req_snr_uavr_leo = (delta_snr * self.varsigma) / (snr_ue_uavr - delta_snr)

        # 6. 计算需求发射功率
        # snr = p_tx * h_gain / noise => p_tx = snr * noise / h_gain
        p_req = (req_snr_uavr_leo * noise_power) / h_gain_uavr_leo_linear

        # 7. 截断到物理可行范围
        optimal_power_w = np.clip(p_req, self.p_min_w, self.p_max_w)

        return optimal_power_w

    def calculate_snr_from_power(self, power_w, h_gain_uavr_leo_linear, noise_power):
        """
        根据发射功率计算UAVr-LEO信噪比

        参数:
            power_w: 发射功率 (W)
            h_gain_uavr_leo_linear: 信道增益 (线性值)
            noise_power: 噪声功率 (W)

        返回:
            snr_uavr_leo: 信噪比 (线性值)
        """
        return (power_w * h_gain_uavr_leo_linear) / noise_power
