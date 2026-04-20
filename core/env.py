import numpy as np
from core.channels.bs_channel import BSChannel
from core.channels.satellite_channel import SatelliteChannel
from core.channels.uavr_channel import SimplifiedUAVRelayChannel


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

        # 记录天空中所有正在飞离的“旧卫星”的剩余任务量
        # 列表中存储的是形状为 (I, J) 的 numpy 矩阵，确保每个用户的积压被精准追踪
        self.sat_ledger = []

        # 记录当前帧自然流逝所清算的旧卫星总能耗和各用户队列减少量，供 Agent 读取
        self.current_e_sat_old = 0.0
        self.current_q_sat_reduction_mat = np.zeros((cfg.I, cfg.J))

        self.reset()

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
            'Loss': [], 'Drift': [], 'Reward': [],
            'R_bs_max': [], 'R_bs_min': [], 'R_sat_max': [], 'R_sat_min': []
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
        h_sq = self.sat_channel.generate_channel_gain_samples(I * J).reshape(I, J)

        # 使用 UAV 中继协同分集计算增强的卫星速率
        # UE 卸载到 LEO 必然经过绑定的 UAVr 协同
        bw_hz = self.cfg.bw_per_user_sat  # 卫星每用户带宽
        R_sat = self.uavr_channel.calculate_enhanced_sat_rate(
            d_bs, d_ue_leo, h_sq,
            self.cfg.p_tx, bw_hz
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
        # 3. 更新虚拟能量队列 E (Power Constraint)
        # ==========================================================
        e_bs_total = details['e_bs_total']
        self.E_BS = np.maximum(0.0, self.E_BS + e_bs_total - self.cfg.E_max_BS)

        # ==========================================================
        # 4. 更新物理状态并记账
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

        if 'G1' in action:
            self.history['Reward'].append(-action['G1'])