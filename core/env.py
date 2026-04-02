import numpy as np
from core.channels.bs_channel import BSChannel
from core.channels.satellite_channel import SatelliteChannel

class SAGINEnvironment:
    """
        物理环境模拟器：负责维护队列、能量、信道状态以及记录历史。
        所有算法 (Agents) 都必须通过此环境更新状态，以保证对比的公平性。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.bs_channel = BSChannel(cfg)
        self.sat_channel = SatelliteChannel(cfg)
        self.reset()

    def reset(self):
        """重置环境状态"""
        I, J = self.cfg.I, self.cfg.J
        self.Q_bs = np.zeros((I, J))
        self.Q_sat = np.zeros((I, J))
        self.Q_total = np.zeros((I, J))

        self.E_BS = np.zeros(I)  # [修改] 每个基站独立的能量队列

        self.T_BS_left_prev = np.zeros(I)
        self.L_BS_left_prev_vec = np.zeros((I, J))
        self.L_Sat_left_prev_vec = np.zeros((I, J))

        self.l_proc_total_prev = np.zeros((I, J))
        #记录上一帧卫星的频率决策
        self.f_sat_prev = np.zeros((I,J))

        self.history = {
            'Q_total': [],
            'Q_bs': [],
            'Q_sat': [],
            'PAoI': [],
            'Cost': [],
            'E_virt_bs': [],
            'E_virt_sat': [],
            'Loss': [],
            'Drift': [],
            'Reward':[],
            'R_bs_max': [],
            'R_bs_min': [],
            'R_sat_max': [],
            'R_sat_min': []
        }
        self.frame_count = 0

    def generate_channel_states(self):
        """生成本帧的信道状态"""
        I, J = self.cfg.I, self.cfg.J
        d_bs = np.random.uniform(self.cfg.d_min, self.cfg.d_max, (I, J))
        R_bs = self.bs_channel.calculate_uplink_rate(d_bs)

        d_sat = np.full((I, J), self.cfg.H_sat)
        h_sq = self.sat_channel.generate_channel_gain_samples(I * J).reshape(I, J)
        R_sat, T_prop = self.sat_channel.calculate_uplink_rate(d_sat, h_sq)

        # 记录速率供画图
        self.history['R_bs_max'].append(np.max(R_bs))
        self.history['R_bs_min'].append(np.min(R_bs))
        self.history['R_sat_max'].append(np.max(R_sat))
        self.history['R_sat_min'].append(np.min(R_sat))

        return R_bs, R_sat, T_prop

    def step(self, action, L_t):
        """
        执行环境步进：根据算法做出的动作 (action)，更新物理世界队列和虚拟能量队列，并记录指标。

        :param action: 算法给出的决策字典，包含 'l', 'b', 以及优化细节 'details'
        :param L_t: 当前帧新到达的任务量 (bits), shape=(I, J)
        """
        self.frame_count += 1

        # 1. 解析动作与底层运算细节
        details = action['details']
        l_vec = action['l']
        b_vec = action['b']

        # 提取卸载掩码 (I, J)
        mask_bs = (l_vec == 0) & (b_vec == 1)
        mask_sat = (l_vec == 0) & (b_vec == 0)

        # ==========================================================
        # 2. 更新虚拟任务队列 Q (Lyapunov Queue)
        # ==========================================================
        # BS Queue: Q(t+1) = max(0, Q(t) + Arrival - Service)
        service_bs_total = details['l_proc_bs'] + self.L_BS_left_prev_vec
        arrival_bs = np.where(mask_bs, L_t, 0.0)
        self.Q_bs = np.maximum(0, self.Q_bs + arrival_bs - service_bs_total)

        # Sat Queue: 新卫星处理的新任务 + 旧卫星在后台处理的积压任务
        service_sat_total = details['l_proc_sat'] + details['l_proc_old_sat']
        arrival_sat = np.where(mask_sat, L_t, 0.0)
        self.Q_sat = np.maximum(0, self.Q_sat + arrival_sat - service_sat_total)

        # 汇总总积压，用于下一帧的状态观测
        self.Q_total = self.Q_bs + self.Q_sat

        # ==========================================================
        # 3. 更新虚拟能量队列 E (Power Constraint)
        # ==========================================================
        e_bs_total = details['e_bs_total']  # 形状为 (I,) 的数组
        # 支持 (I,) 数组的逐元素更新，确保不超过最大能量预算
        self.E_BS = np.maximum(0.0, self.E_BS + e_bs_total - self.cfg.E_max_BS)

        # ==========================================================
        # 4. 更新物理状态 (为下一帧的继承做准备)
        # ==========================================================
        # --- 基站 (BS) 状态更新 ---
        l_left_bs_next = details['l_left_bs']  # 本帧产生的新残留任务
        self.T_BS_left_prev = details.get('t_next_left_bs_scalar', np.zeros(self.cfg.I))

        # 维护基站残余任务队列 (I, J 维度)
        if np.sum(l_left_bs_next) > 1e-9:
            self.L_BS_left_prev_vec = l_left_bs_next.copy()
        else:
            self.L_BS_left_prev_vec = np.zeros((self.cfg.I, self.cfg.J))

        self.l_proc_total_prev = details['l_proc_total'].copy()

        # --- 卫星 (Sat) 状态更新 ---
        l_left_sat_next = details['l_left_sat']

        # 维护卫星残余任务队列 (I, J 维度)
        if np.sum(l_left_sat_next) > 1e-9:
            self.L_Sat_left_prev_vec = l_left_sat_next.copy()
        else:
            self.L_Sat_left_prev_vec = np.zeros((self.cfg.I, self.cfg.J))

        # ==========================================================
        # 5. 记录历史流水账 (用于画图)
        # ==========================================================
        self.history['Q_total'].append(np.mean(self.Q_total))
        self.history['Q_bs'].append(np.mean(self.Q_bs))
        self.history['Q_sat'].append(np.mean(self.Q_sat))
        self.history['Cost'].append(np.mean(details['paoi']))
        self.history['E_virt_bs'].append(np.mean(self.E_BS))
        self.history['E_virt_sat'].append(details['e_sat'])
        self.history['Drift'].append(details['real_drift'])

        # 如果 Agent 传来了目标函数 G1，将其取反作为强化学习的 Reward 记录
        if 'G1' in action:
            self.history['Reward'].append(-action['G1'])
