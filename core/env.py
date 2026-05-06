import numpy as np
from core.channels.bs_channel import BSChannel
from core.channels.satellite_channel import SatelliteChannel
from core.channels.uavr_channel import SimplifiedUAVRelayChannel
from core.optimizers.uavr_optimizer import UAVRelayOptimizer


class SAGINEnvironment:
    """
    物理环境模拟器：负责维护队列、能量、信道状态以及记录历史。
    核心创新：引入“基于矩阵延迟账本的跨帧状态摊销机制”，严防跨回合数据污染。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.bs_channel = BSChannel(cfg)
        self.sat_channel = SatelliteChannel(cfg)
        self.uavr_channel = SimplifiedUAVRelayChannel(cfg)
        self.uavr_opt = UAVRelayOptimizer(cfg)

        # 记录每帧的UAVr最优发射功率和能耗
        self.current_uavr_power = 0.0
        self.current_uavr_energy = 0.0

        # 存储当前帧的信道状态供step()使用
        self.current_d_bs = None
        self.current_h_sq = None
        self.current_snr_ue_leo = None

        # 记录天空中所有正在飞离的“旧卫星”的剩余任务量
        # 列表中存储的是形状为 (I, J) 的 numpy 矩阵，确保每个用户的积压被精准追踪
        self.sat_ledger = []

        # 记录当前帧自然流逝所清算的旧卫星总能耗和各用户队列减少量，供 Agent 读取
        self.current_e_sat_old = 0.0
        self.current_q_sat_reduction_mat = np.zeros((cfg.I, cfg.J))

        self.reset()

    @property
    def Q_sat_pending(self):
        """所有旧卫星账本中仍未完成的任务量之和, shape (I, J)"""
        total = np.zeros((self.cfg.I, self.cfg.J))
        for mat in self.sat_ledger:
            total += mat
        return total

    @property
    def Q_sat_total(self):
        """卫星总积压 = 当前卫星队列 + 旧卫星账本, shape (I, J)"""
        return self.Q_sat + self.Q_sat_pending

    def reset(self):
        """重置环境状态，确立物理世界的“大爆炸”奇点"""
        I, J = self.cfg.I, self.cfg.J
        self.Q_bs = np.zeros((I, J))
        self.Q_sat = np.zeros((I, J))
        self.Q_total = np.zeros((I, J))

        self.E_BS = np.zeros(I)

        self.T_BS_left_prev = np.zeros(I)
        self.L_BS_left_prev_vec = np.zeros((I, J))
        self.L_Sat_left_prev_vec = np.zeros((I, J))

        self.l_proc_total_prev = np.zeros((I, J))
        self.f_sat_prev = np.zeros((I, J))

        self.history = {
            'Q_total': [], 'Q_bs': [], 'Q_sat': [],
            'PAoI': [], 'Cost': [], 'E_virt_bs': [], 'E_virt_sat': [],
            'E_queue_bs_max': [], 'Loss': [], 'Drift': [], 'Reward': [],
            'R_bs_max': [], 'R_bs_min': [], 'R_sat_max': [], 'R_sat_min': [],
            'uavr_energy': []
        }
        self.frame_count = 0

        # 🚨【关键修复：清空飞行账本】🚨
        # 严防上一局积压在天上的残余卫星变成“幽灵”带入新一局
        self.sat_ledger = []
        self.current_e_sat_old = 0.0
        self.current_q_sat_reduction_mat = np.zeros((I, J))

    def generate_channel_states(self):
        """生成本帧的信道状态"""
        I, J = self.cfg.I, self.cfg.J
        d_bs = np.random.uniform(self.cfg.d_min, self.cfg.d_max, (I, J))
        R_bs = self.bs_channel.calculate_uplink_rate(d_bs)

        d_ue_leo = np.full((I, J), self.cfg.H_sat)
        # 衰落样本独立生成：直连 UE-LEO 与中继 UAV-LEO 为独立信道
        # UAV-LEO 使用专用的轻度 Shadowed-Rician 参数 (b=0.1, Omega=1.5, m=25)
        h_sq_direct = self.sat_channel.generate_channel_gain_samples(I * J).reshape(I, J)
        h_sq_relay = self.uavr_channel.generate_uav_leo_channel_samples(I * J).reshape(I, J)

        # ===== 动态UAV发射功率优化 =====
        # 计算平均数据量和最大容忍延迟
        D_avg = self.cfg.L_mean
        T_prop_avg = self.cfg.H_sat / self.cfg.c
        T_max = self.cfg.tau - T_prop_avg

        # 计算UE-UAVr距离 (Pure LoS)
        d_ue_uavr = np.sqrt(d_bs ** 2 + self.uavr_channel.H_UAV ** 2)

        # 计算各链路SNR

        # UE-UAVr SNR (Pure LoS, sub-6 GHz)
        snr_ue_uavr = self.uavr_channel.calculate_ue_uavr_snr(d_ue_uavr, self.cfg.p_tx, self.cfg.bw_per_user_bs)

        # UE-LEO SNR (Ka-band, Shadowed-Rician)
        noise_power_sat = self.cfg.sigma2
        lambda_ka = self.uavr_channel.c / self.cfg.f_c_sat
        beta = 2.2
        pl_const_db = 20 * np.log10(4 * np.pi / lambda_ka)
        pl_dist_db_leo = 10 * beta * np.log10(d_ue_leo)
        p_tx_dbm_ue = 10 * np.log10(self.cfg.p_tx * 1000)
        fading_db = 10 * np.log10(h_sq_direct)
        pr_dbm_ue_leo = (p_tx_dbm_ue + self.cfg.G_tx_ue_dbi + self.cfg.G_rx_sat_dbi +
                         fading_db - pl_const_db - pl_dist_db_leo)
        pr_linear_ue_leo = 10 ** ((pr_dbm_ue_leo - 30) / 10)
        snr_ue_leo = pr_linear_ue_leo / noise_power_sat

        # UAVr-LEO 信道增益 (Shadowed-Rician衰落增益 h_sq_relay)
        h_gain_uavr_leo_linear = h_sq_relay  # 这是|h|^2衰落增益

        # 调用优化器计算最优UAV发射功率
        optimal_uavr_power = self.uavr_opt.optimize_power(
            D_avg, T_max, self.cfg.bw_per_user_sat,
            np.mean(snr_ue_leo), np.mean(snr_ue_uavr),
            np.mean(h_gain_uavr_leo_linear), noise_power_sat
        )

        # 保存最优功率供后续能量计算使用
        self.current_uavr_power = optimal_uavr_power

        # 存储信道样本供step()中计算UAV能耗使用
        self.current_d_bs = d_bs.copy()
        self.current_h_sq = h_sq_relay.copy()
        self.current_snr_ue_leo = snr_ue_leo.copy()

        # 使用动态功率计算增强卫星速率
        R_sat = self.uavr_channel.calculate_enhanced_sat_rate(
            d_bs, h_sq_relay, snr_ue_leo,
            self.cfg.p_tx, self.cfg.bw_per_user_sat,
            p_tx_uavr_w=optimal_uavr_power
        )

        # 传播时延 (UE -> LEO 直接链路, 用于传输时间计算)
        T_prop = d_ue_leo / self.cfg.c

        self.history['R_bs_max'].append(np.max(R_bs))
        self.history['R_bs_min'].append(np.min(R_bs))
        self.history['R_sat_max'].append(np.max(R_sat))
        self.history['R_sat_min'].append(np.min(R_sat))

        return R_bs, R_sat, T_prop

    def step(self, action, L_t):
        self.frame_count += 1
        phi = self.cfg.phi
        tau = self.cfg.tau
        kappa2 = self.cfg.kappa2

        # 🚨【核心修复：引入基于电池功率的真实物理短板】🚨
        f_max_sat = self.cfg.f_max_Sat
        E_max_bat = self.cfg.E_max_Sat

        # 卫星不仅受限于 CPU 频率，更受限于每帧的最大供电能力 E_max_Sat
        # E = kappa2 * tau * f^3  => 逆推电池能支撑的最高频率
        f_limit_energy = (E_max_bat / (kappa2 * tau)) ** (1 / 3)

        # 真实的物理天花板 (取 CPU限制 和 电池限制 的最小值)
        f_effective_max = min(f_max_sat, f_limit_energy)

        # ==========================================================
        # 1. 物理时间的自然流逝：旧卫星账本的延期摊销清算
        # ==========================================================
        # 基于真实的木桶短板，计算单帧最大处理量和耗电
        max_process_per_frame = (f_effective_max * tau) / phi
        max_energy_per_frame = kappa2 * phi * (f_effective_max ** 2) * max_process_per_frame

        current_e_sat_old = 0.0

        current_q_sat_reduction_mat = np.zeros((self.cfg.I, self.cfg.J))
        new_ledger = []

        # 遍历天空中所有正在飞离的旧卫星 (每个 leftover_mat 是一个 I x J 的矩阵)
        for leftover_mat in self.sat_ledger:
            total_leftover = np.sum(leftover_mat)
            if total_leftover > 1e-9:
                # 这颗旧卫星在当前帧能物理处理掉的总数据量
                processed_total = min(total_leftover, max_process_per_frame)

                # 按照任务比例，完美映射回各个用户的矩阵中（同步完成策略）
                ratio = processed_total / total_leftover
                processed_mat = leftover_mat * ratio
                current_q_sat_reduction_mat += processed_mat

                # 计算这颗旧卫星在当前帧消耗的真实能量
                if total_leftover >= max_process_per_frame:
                    current_e_sat_old += max_energy_per_frame
                else:
                    f_tail = f_max_sat * (processed_total / max_process_per_frame)
                    current_e_sat_old += kappa2 * phi * (f_tail ** 2) * processed_total

                # 结算剩余量并判断是否需要带入下一帧
                remain_mat = leftover_mat - processed_mat
                if np.sum(remain_mat) > 1e-9:
                    new_ledger.append(remain_mat)

        # 更新账本与暴露给 Agent 的数据
        self.sat_ledger = new_ledger
        self.current_e_sat_old = current_e_sat_old
        self.current_q_sat_reduction_mat = current_q_sat_reduction_mat

        # ==========================================================
        # 2. 解析动作与队列更新
        # ==========================================================
        details = action['details']
        l_vec = action['l']
        b_vec = action['b']

        mask_bs = (l_vec == 0) & (b_vec == 1)
        mask_sat = (l_vec == 0) & (b_vec == 0)

        # BS Queue: Q(t+1) = max(0, Q(t) + Arrival - Service)
        service_bs_total = details['l_proc_bs'] + details['l_proc_old_bs']
        arrival_bs = np.where(mask_bs, L_t, 0.0)
        self.Q_bs = np.maximum(0, self.Q_bs + arrival_bs - service_bs_total)

        # Sat Queue: 扣除自然流逝处理掉的积压
        # 服务量 = 新卫星处理掉的 + 账本自然流逝处理掉的矩阵
        service_sat_total = details['l_proc_sat'] + self.current_q_sat_reduction_mat
        arrival_sat = np.where(mask_sat, L_t, 0.0)
        self.Q_sat = np.maximum(0, self.Q_sat + arrival_sat - service_sat_total)

        self.Q_total = self.Q_bs + self.Q_sat

        # ==========================================================
        # 3. UAV中继能耗计算 (LEO卸载分支)
        # ==========================================================
        L_to_sat = np.where(mask_sat, L_t, 0.0)

        # 使用generate_channel_states中存储的信道样本和最优功率
        bw_hz = self.cfg.bw_per_user_sat

        # 用存储的最优UAV功率和信道样本计算实际传输速率
        actual_R_sat = self.uavr_channel.calculate_enhanced_sat_rate(
            self.current_d_bs, self.current_h_sq, self.current_snr_ue_leo,
            self.cfg.p_tx, bw_hz,
            p_tx_uavr_w=self.current_uavr_power
        )

        # 计算实际传输延迟 (bits / bps = seconds)
        # 避免除零
        actual_T_tran_sat = np.where(mask_sat & (actual_R_sat > 1e-9),
                                     L_to_sat / actual_R_sat, 0.0)

        # UAV通信能耗 = 发射功率 * 传输时间
        uavr_energy = self.current_uavr_power * np.sum(actual_T_tran_sat)
        self.current_uavr_energy = uavr_energy

        # ==========================================================
        # 4. 更新虚拟能量队列 E (Power Constraint)
        # ==========================================================
        e_bs_total = details['e_bs_total']
        self.E_BS = np.maximum(0.0, self.E_BS + e_bs_total - self.cfg.E_max_BS)
        self.history['E_queue_bs_max'].append(float(np.max(self.E_BS)))

        # ==========================================================
        # 5. 更新物理状态并记账
        # ==========================================================
        l_left_bs_next = details['l_left_bs']
        self.T_BS_left_prev = details.get('t_next_left_bs_scalar', np.zeros(self.cfg.I))

        if np.sum(l_left_bs_next) > 1e-9:
            self.L_BS_left_prev_vec = l_left_bs_next.copy()
        else:
            self.L_BS_left_prev_vec = np.zeros((self.cfg.I, self.cfg.J))

        self.l_proc_total_prev = details['l_proc_total'].copy()

        # 【核心记账】：将本帧产生的新卫星残留矩阵，正式加入飞行账本
        l_left_sat_next = details['l_left_sat']
        if np.sum(l_left_sat_next) > 1e-9:
            self.sat_ledger.append(l_left_sat_next.copy())

        # ==========================================================
        # 5. 记录历史流水账 (用于画图)
        # ==========================================================
        self.history['Q_total'].append(np.mean(self.Q_total))
        self.history['Q_bs'].append(np.mean(self.Q_bs))
        self.history['Q_sat'].append(np.mean(self.Q_sat))
        self.history['Cost'].append(np.mean(details['paoi']))

        # 【修正 1：基站平均能耗】
        # 记录 1 个基站的平均瞬时能耗 (总能耗 / 基站数量 I)
        avg_e_bs_per_node = np.mean(details['e_bs_total'])
        self.history['E_virt_bs'].append(avg_e_bs_per_node)

        # 【修正 2：卫星平均能耗 (动态分母)】
        # 活跃卫星数量 = 1 (当前接收新任务的新卫星) + 账本里尚未坠毁的旧卫星数量
        active_sat_count = 1 + len(self.sat_ledger)
        # 记录 1 颗活跃卫星的平均瞬时能耗 (当帧卫星总耗电 / 动态活跃卫星数)
        avg_e_sat_per_node = details['e_sat'] / active_sat_count
        self.history['E_virt_sat'].append(avg_e_sat_per_node)

        self.history['Drift'].append(details['real_drift'])

        # UAV中继通信能耗记录 (可用于后续Penalty计算)
        self.history['uavr_energy'].append(self.current_uavr_energy)

        if 'G1' in action:
            self.history['Reward'].append(-action['G1'])