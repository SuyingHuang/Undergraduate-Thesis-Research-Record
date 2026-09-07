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
        self._sat_frame_plan = None

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
        """兼容旧调用：卫星积压只来自账本，不再重复叠加。"""
        return self.Q_sat_pending

    @property
    def Q_sat(self):
        """卫星物理队列，严格由逐颗卫星账本派生。"""
        return self.Q_sat_pending

    @property
    def Q_total(self):
        return self.Q_bs + self.Q_sat

    def reset(self):
        """重置环境状态，确立物理世界的“大爆炸”奇点"""
        I, J = self.cfg.I, self.cfg.J
        self.Q_bs = np.zeros((I, J))

        self.E_BS = np.zeros(I)

        self.T_BS_left_prev = np.zeros(I)
        self.L_BS_left_prev_vec = np.zeros((I, J))

        self.history = {
            'Q_total': [], 'Q_bs': [], 'Q_sat': [],
            'PAoI': [], 'Cost': [], 'E_virt_bs': [], 'E_virt_sat': [],
            'E_queue_bs_max': [], 'Loss': [], 'Drift': [], 'Reward': [],
            'R_bs_max': [], 'R_bs_min': [], 'R_sat_max': [], 'R_sat_min': [],
            'uavr_energy': [], 'lyapunov_value': [], 'drift_bound_quadratic': [],
            'E_sat_total': [], 'E_sat_node_max': [], 'active_sat_count': []
        }
        self.frame_count = 0

        # 🚨【关键修复：清空飞行账本】🚨
        # 严防上一局积压在天上的残余卫星变成“幽灵”带入新一局
        self.sat_ledger = []
        self.current_e_sat_old = 0.0
        self.current_q_sat_reduction_mat = np.zeros((I, J))
        self._sat_frame_plan = None

    def prepare_frame(self):
        """Plan old-satellite service for this frame without mutating queues.

        Old satellites are independent of the current offloading decision, so
        this plan is available to every candidate at the same decision epoch.
        """
        if self._sat_frame_plan is not None:
            return self._sat_frame_plan
        phi, tau = self.cfg.phi, self.cfg.tau
        f_limit_energy = (self.cfg.E_max_Sat / (self.cfg.kappa2 * tau)) ** (1 / 3)
        f_effective = min(self.cfg.f_max_Sat, f_limit_energy)
        max_process = f_effective * tau / phi
        service = np.zeros((self.cfg.I, self.cfg.J))
        remaining, energies = [], []
        for leftover in self.sat_ledger:
            total = float(np.sum(leftover))
            if total <= 1e-9:
                continue
            processed = min(total, max_process)
            processed_mat = leftover * (processed / total)
            service += processed_mat
            # Use exactly the frequency required over one frame. It cannot
            # exceed f_effective, so every old satellite respects E_max_Sat.
            frequency = phi * processed / tau
            energy = self.cfg.kappa2 * phi * frequency ** 2 * processed
            if energy > self.cfg.E_max_Sat * (1 + 1e-9):
                raise RuntimeError("Old-satellite energy constraint violated")
            energies.append(float(energy))
            remain = leftover - processed_mat
            if np.sum(remain) > 1e-9:
                remaining.append(remain)
        self._sat_frame_plan = {
            'service': service,
            'remaining': remaining,
            'energies': energies,
            'energy_total': float(np.sum(energies)),
            'ledger_loads': [float(np.sum(x)) for x in self.sat_ledger],
        }
        self.current_q_sat_reduction_mat = service
        self.current_e_sat_old = self._sat_frame_plan['energy_total']
        return self._sat_frame_plan

    def satellite_state_context(self):
        plan = self.prepare_frame()
        return {
            'service': plan['service'],
            'ledger_loads': plan['ledger_loads'],
            'old_energy': plan['energy_total'],
        }

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
        snr_ue_uavr = self.uavr_channel.calculate_ue_uavr_snr(d_ue_uavr, self.cfg.p_tx)

        # UE-LEO SNR (Ka-band, Shadowed-Rician) — 统一由 SatelliteChannel 计算
        snr_ue_leo = self.sat_channel.calculate_snr(d_ue_leo, h_sq_direct)

        # UAVr-LEO 完整信道增益 (包含 Ka 路径损耗 + 天线增益 + 小尺度衰落)
        d_uavr_leo = self.cfg.H_sat - self.uavr_channel.H_UAV
        h_gain_uavr_leo_complete = self.uavr_channel.calculate_uavr_leo_channel_gain(
            d_uavr_leo, h_sq_relay
        )

        # 保存直连UE-LEO信噪比供step()使用
        self.current_snr_ue_leo = snr_ue_leo.copy()

        if self.cfg.use_uav_relay:
            # 调用优化器计算最优UAV发射功率
            optimal_uavr_power = self.uavr_opt.optimize_power(
                D_avg, T_max, self.cfg.bw_per_user_sat,
                np.mean(snr_ue_leo), np.mean(snr_ue_uavr),
                np.mean(h_gain_uavr_leo_complete), self.cfg.sigma2
            )
            self.current_uavr_power = optimal_uavr_power
            self.current_d_bs = d_bs.copy()
            self.current_h_sq = h_sq_relay.copy()

            # UAV中继增强卫星速率
            R_sat = self.uavr_channel.calculate_enhanced_sat_rate(
                d_bs, h_sq_relay, snr_ue_leo,
                self.cfg.p_tx, self.cfg.bw_per_user_sat,
                p_tx_uavr_w=optimal_uavr_power
            )
        else:
            # 无UAV中继：仅使用直连UE-LEO链路
            self.current_uavr_power = 0.0
            self.current_d_bs = d_bs.copy()
            self.current_h_sq = np.zeros_like(h_sq_relay)
            R_sat = self.cfg.bw_per_user_sat * np.log2(1.0 + snr_ue_leo)

        # 传播时延 (UE -> LEO 直接链路, 用于传输时间计算)
        T_prop = d_ue_leo / self.cfg.c

        self.history['R_bs_max'].append(np.max(R_bs))
        self.history['R_bs_min'].append(np.min(R_bs))
        self.history['R_sat_max'].append(np.max(R_sat))
        self.history['R_sat_min'].append(np.min(R_sat))

        return R_bs, R_sat, T_prop

    def step(self, action, L_t):
        self.frame_count += 1
        sat_plan = self.prepare_frame()
        q_before = self.Q_total.copy()
        q_bs_before = self.Q_bs.copy()
        q_sat_before = self.Q_sat.copy()
        e_before = self.E_BS.copy()

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

        # ==========================================================
        # 3. UAV中继能耗计算 (LEO卸载分支)
        # ==========================================================
        L_to_sat = np.where(mask_sat, L_t, 0.0)

        if self.cfg.use_uav_relay:
            # 用存储的最优UAV功率和信道样本计算实际传输速率
            bw_hz = self.cfg.bw_per_user_sat
            actual_R_sat = self.uavr_channel.calculate_enhanced_sat_rate(
                self.current_d_bs, self.current_h_sq, self.current_snr_ue_leo,
                self.cfg.p_tx, bw_hz,
                p_tx_uavr_w=self.current_uavr_power
            )
            actual_T_tran_sat = np.where(mask_sat & (actual_R_sat > 1e-9),
                                         L_to_sat / actual_R_sat, 0.0)
            uavr_energy = self.current_uavr_power * np.sum(actual_T_tran_sat)
        else:
            # 无UAV中继：使用直连速率
            actual_R_sat = self.cfg.bw_per_user_sat * np.log2(1.0 + self.current_snr_ue_leo)
            actual_T_tran_sat = np.where(mask_sat & (actual_R_sat > 1e-9),
                                         L_to_sat / actual_R_sat, 0.0)
            uavr_energy = 0.0
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
        l_left_bs_next = details['l_left_bs_total']
        self.T_BS_left_prev = details.get('t_next_left_bs_scalar', np.zeros(self.cfg.I))

        if np.sum(l_left_bs_next) > 1e-9:
            self.L_BS_left_prev_vec = l_left_bs_next.copy()
        else:
            self.L_BS_left_prev_vec = np.zeros((self.cfg.I, self.cfg.J))

        # 提交旧卫星服务计划，再加入本帧当前卫星的新残余。
        self.sat_ledger = [x.copy() for x in sat_plan['remaining']]
        l_left_sat_next = details['l_left_sat']
        if np.sum(l_left_sat_next) > 1e-9:
            self.sat_ledger.append(l_left_sat_next.copy())

        # 运行时守恒检查：物理队列只能由“旧队列 + 新到达 - 本帧服务”得到。
        expected_q_bs = np.maximum(
            0.0, q_bs_before + arrival_bs - service_bs_total
        )
        expected_q_sat = np.maximum(
            0.0,
            q_sat_before + L_to_sat
            - sat_plan['service'] - details['l_proc_sat']
        )
        if not np.allclose(self.Q_bs, expected_q_bs, rtol=1e-9, atol=1e-5):
            raise RuntimeError("BS queue conservation violated")
        if not np.allclose(self.L_BS_left_prev_vec, self.Q_bs,
                           rtol=1e-9, atol=1e-5):
            raise RuntimeError("BS physical ledger and queue diverged")
        if not np.allclose(self.Q_sat, expected_q_sat, rtol=1e-9, atol=1e-5):
            raise RuntimeError("Satellite queue conservation violated")

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

        old_active = sum(e > 1e-12 for e in sat_plan['energies'])
        new_active = int(details['e_sat_new'] > 1e-12)
        active_sat_count = old_active + new_active
        avg_e_sat_per_node = (details['e_sat'] / active_sat_count
                              if active_sat_count else 0.0)
        self.history['E_virt_sat'].append(avg_e_sat_per_node)
        self.history['E_sat_total'].append(details['e_sat'])
        node_energies = list(sat_plan['energies'])
        if new_active:
            node_energies.append(float(details['e_sat_new']))
        max_sat_energy = max(node_energies, default=0.0)
        if max_sat_energy > self.cfg.E_max_Sat * (1 + 1e-6):
            raise RuntimeError("Per-satellite energy constraint violated")
        self.history['E_sat_node_max'].append(max_sat_energy)
        self.history['active_sat_count'].append(active_sat_count)

        lyapunov_before = 0.5 * (np.sum(q_before ** 2) + np.sum(e_before ** 2))
        lyapunov_after = 0.5 * (np.sum(self.Q_total ** 2) + np.sum(self.E_BS ** 2))
        drift = float(lyapunov_after - lyapunov_before)
        self.history['lyapunov_value'].append(float(lyapunov_after))
        self.history['Drift'].append(drift)
        self.history['drift_bound_quadratic'].append(details['drift_bound_quadratic'])

        # UAV中继通信能耗记录 (可用于后续Penalty计算)
        self.history['uavr_energy'].append(self.current_uavr_energy)

        if 'G1' in action:
            self.history['Reward'].append(-action['G1'])
        self._sat_frame_plan = None
