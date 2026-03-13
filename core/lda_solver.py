import numpy as np
import torch
import torch.optim as optim
import random
from collections import deque # [新增] 用于经验回放

from config import SystemConfig
from core.dnn_model import OffloadingActor, get_input_vector,FocalLoss
from core.bs_optimizer import BS_Optimizer
from core.leo_optimizer import LEO_Optimizer
from core.bs_channel import BSChannel
from core.satellite_channel import SatelliteChannel
from core.tcopq import generate_candidates, check_local_feasibility

class LDASolver:
    """
        LDA 算法总控器 (The Solver)
        实现论文 Algorithm 1 的主流程。
        """

    def __init__(self, cfg):
        self.cfg = cfg

        # 1. 初始化各组件
        self.actor = OffloadingActor(cfg.J)
        self.bs_opt = BS_Optimizer(cfg)
        self.leo_opt = LEO_Optimizer(cfg)
        self.bs_channel = BSChannel(cfg)
        self.sat_channel = SatelliteChannel(cfg)

        # 2. 训练组件初始化
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.criterion = FocalLoss(alpha=self.cfg.focal_alpha, gamma=self.cfg.focal_gamma)

        self.memory = deque(maxlen=self.cfg.memory_capacity)
        self.batch_size = self.cfg.batch_size
        self.train_interval = self.cfg.train_interval

        # 3. 初始化环境状态
        self.reset()

        # 4. 优化参数
        self.delta_t = 0.5
        self.gamma = 0.95
        self.frame_count = 0  # 计数器

    def reset(self):
        """重置环境状态"""
        self.Q_bs = np.zeros(self.cfg.J)
        self.Q_sat = np.zeros(self.cfg.J)
        self.Q_total =np.zeros(self.cfg.J)
        self.E_BS = 0.0
        self.E_Sat = 0.0  # [新增] 卫星虚拟能量队列

        self.T_BS_left_prev = 0.0
        self.L_BS_left_prev_vec = np.zeros(self.cfg.J)
        self.L_Sat_left_prev_vec = np.zeros(self.cfg.J)

        self.l_proc_total_prev = np.zeros(self.cfg.J)
        #记录上一帧卫星的频率决策
        self.f_sat_prev = np.zeros(self.cfg.J)

        self.history = {
            'Q_total': [],
            'Q_bs': [],
            'Q_sat': [],
            'PAoI': [],
            'Cost': [],
            'E_virt_bs': [],
            'E_virt_sat': [],
            'Loss': [],
            'Drift': []
        }
        self.frame_count = 0

    def step(self,L_t):
        """
        执行一帧的完整调度 (Algorithm 1)
        :param L_t: 当前帧新到达的任务量 (bits), shape=(J,)
        """
        self.frame_count += 1

        # --- Step 1: 环境感知 (Channel & Rates) ---
        # 1. 基站信道：依然是随机分布 (用户在基站覆盖范围内移动)
        d_bs = np.random.uniform(self.cfg.d_min, self.cfg.d_max, self.cfg.J)
        R_bs = self.bs_channel.calculate_uplink_rate(d_bs)

        # 2. 卫星信道：[修改点]
        # 假设：每一帧都接入一个新的、位于固定轨道高度的卫星
        # 距离 d 固定为 H_sat (忽略仰角带来的斜距变化，做最简处理)
        d_sat = np.full(self.cfg.J, self.cfg.H_sat)

        # 衰落 (Shadowing/Multipath) 依然随机
        # 虽然距离不变，但大气环境和多径效应依然是动态的
        h_sq = self.sat_channel.generate_channel_gain_samples(self.cfg.J)
        R_sat, T_prop = self.sat_channel.calculate_uplink_rate(d_sat, h_sq)

        # --- Step 2: 状态构建与 Actor 预测 ---
        # 构造状态向量 X_t (归一化交给 DNN 内部 BN 层)
        self.Q_total = self.Q_bs + self.Q_sat
        state_tensor = get_input_vector(self.Q_total, self.E_BS, self.T_BS_left_prev, R_bs, R_sat)

        # Actor 前向传播
        self.actor.eval()
        with torch.no_grad():
            prob_b = self.actor(state_tensor).numpy().flatten()

        # --- Step 3: TCOPQ 候选生成 ---
        # 确定本地计算约束 (l=1 表示本地)
        f_local = np.ones(self.cfg.J) * self.cfg.f_max_UE
        l_decisions = check_local_feasibility(L_t, f_local)

        # 生成候选集
        candidates = generate_candidates(prob_b, self.delta_t, l_decisions)

        # --- Step 4: 候选评估 (Parallel Solving) ---
        best_G1 = float('inf')
        best_sol = None

        for l_vec, b_vec in candidates:
            # 4.1 任务分流 logic
            # l=0 (Offload), b=1 (BS) -> BS
            mask_bs = (l_vec == 0) & (b_vec == 1)
            L_to_bs = np.where(mask_bs, L_t, 0.0)

            # l=0 (Offload), b=0 (Sat) -> Sat
            mask_sat = (l_vec == 0) & (b_vec == 0)
            L_to_sat = np.where(mask_sat, L_t, 0.0)

            # 4.2 计算时间窗口
            T_tran_bs = np.zeros_like(L_t)
            T_tran_bs[mask_bs] = L_to_bs[mask_bs] / R_bs[mask_bs]

            T_tran_sat = np.zeros_like(L_t)
            T_tran_sat[mask_sat] = L_to_sat[mask_sat] / R_sat[mask_sat]
            # Sat 可用时间 = tau - 传输 - 传播
            T_avail_sat = np.maximum(0, self.cfg.tau - T_tran_sat - T_prop)

            # 4.3 调用底层优化器
            f_bs = self.bs_opt.optimize(L_to_bs, self.Q_total, self.E_BS, T_tran_bs, self.T_BS_left_prev)
            f_sat = self.leo_opt.optimize(L_to_sat, self.Q_total, T_avail_sat)

            # 4.4 计算目标函数 G1
            G1, details = self.calculate_objective(
                L_t, self.Q_bs,self.Q_sat, self.E_BS, self.E_Sat,
                l_vec, mask_bs, mask_sat,
                f_bs, f_sat, f_local,
                T_tran_bs, T_avail_sat,
                self.T_BS_left_prev, self.L_BS_left_prev_vec
            )

            if G1 < best_G1:
                best_G1 = G1
                best_sol = {
                    'l': l_vec, 'b': b_vec,
                    'f_bs': f_bs, 'f_sat': f_sat,
                    'details': details
                }
        # --- [新增] Step 4.5: 存储经验 ---
        # 我们要让神经网络学习的是：在当前 state 下，best_sol['b'] 是最好的选择
        # 注意：只存 b (Offloading Decision)，因为 l (Local) 是由 feasibility 决定的硬约束
        best_action_b = best_sol['b']
        self.store_experience(state_tensor, best_action_b)

        # Step 4.6: 触发训练 ---
        # 1. 获取当前回放池的样本数量
        current_memory_size = len(self.memory)
        # 2. 获取最大容量的一半
        half_capacity = self.cfg.memory_capacity / 2.0
        if (current_memory_size >= half_capacity) and ( self.frame_count% self.cfg.train_interval == 0):
            loss_val = self.train_network()
            self.history['Loss'].append((self.frame_count, loss_val))
        # --- Step 5: 环境更新 ---
        self.update_env(best_sol, L_t)

        # --- Step 6: 动态阈值衰减 (Eq. 44) ---
        # 简单实现：每帧衰减，或者每隔 Delta T 衰减
        self.delta_t = max(0.1, self.delta_t * self.gamma)
        # ================= [新增调试信息] =================
        # 统计本帧的决策分布
        # l_vec: 1=Local, 0=Offload
        # b_vec: 1=BS, 0=Sat
        l = best_sol['l']
        b = best_sol['b']

        cnt_local = np.sum(l == 1)
        cnt_bs = np.sum((l == 0) & (b == 1))
        cnt_sat = np.sum((l == 0) & (b == 0))

        # 统计资源利用率
        # f_bs shape (J,), f_max_BS 是标量
        util_bs = np.sum(best_sol['f_bs']) / (self.cfg.f_max_BS + 1e-9)
        util_sat = np.sum(best_sol['f_sat']) / (self.cfg.f_max_Sat + 1e-9)

        # 统计吞吐量供需关系
        arrival_mb = np.sum(L_t) / 1e6
        served_mb = np.sum(best_sol['details']['l_proc_total']) / 1e6

        debug_info = {
            'dist': (cnt_local, cnt_bs, cnt_sat),  # 决策分布
            'util': (util_bs, util_sat),  # 频率占用率 (0~1)
            'flow': (arrival_mb, served_mb),  # 流量 (Mb)
            'q_trend': served_mb - arrival_mb,  # 净流量 (>0 表示在清理积压)
            'prob_mean': np.mean(prob_b)  # 神经网络平均输出值
        }
        best_sol['debug'] = debug_info
        # =================================================

        return best_sol

    def calculate_objective(self, L_t, Q_bs,Q_sat, E_BS, E_Sat,
                            l_vec, mask_bs, mask_sat,
                            f_bs, f_sat, f_local,
                            T_tran_bs, T_avail_sat, T_left_prev, L_left_prev_vec):
        """
        计算单帧目标函数 G1(t) (Eq. 39)
        修正版：处理量包含历史积压 Q，确保队列可以被消除。
        """
        phi = self.cfg.phi
        kappa1 = self.cfg.kappa1
        kappa2 = self.cfg.kappa2

        # --- A. 计算处理量与残留量 (修正：基于 Total Load) ---
        # 1. BS 处理情况 (只针对新任务 L_t)
        # 物理负载仅为当前帧卸载到 BS 的新任务
        load_bs_new = np.where(mask_bs, L_t, 0.0)

        # BS 物理可用时间 = 总帧长 - max(传输时间, 旧任务占用的时间)
        t_bs_avail_phys = np.maximum(0, self.cfg.tau - np.maximum(T_tran_bs, T_left_prev))

        # 理论最大处理能力 (Capacity)
        cap_bs = f_bs * t_bs_avail_phys / phi

        # 实际处理的新任务量
        l_proc_bs_new = np.minimum(load_bs_new, cap_bs)

        # 本帧新任务的残留量 (将进入下一帧成为旧任务)
        l_left_bs_new = np.maximum(0, load_bs_new - l_proc_bs_new)

        # 2. Sat 处理情况 (只针对新任务)
        load_sat_new = np.where(mask_sat, L_t, 0.0)
        cap_sat = f_sat * T_avail_sat / phi
        l_proc_sat_new = np.minimum(load_sat_new, cap_sat)
        l_left_sat_new = np.maximum(0, load_sat_new - l_proc_sat_new)

        # 3. Local 处理情况
        l_proc_loc_new = np.where(l_vec == 1, L_t, 0.0)

        # [新增修正] 4. 旧任务的处理量 (Old Tasks Processing)
        # BS 旧任务: 假设 T_left_prev 分配的时间足够处理 L_left_prev_vec (除非 T_left > tau，但此处做简化假设)
        l_proc_old_bs = np.zeros_like(L_t)
        total_l_prev_bs = np.sum(L_left_prev_vec)

        if total_l_prev_bs > 1e-9:
            # 1. 计算按残留比例分配给旧任务的频率
            f_old_bs_vec = self.cfg.f_max_BS * (L_left_prev_vec / total_l_prev_bs)
            # 2. 计算在 \tau 时间内物理上最多能处理的数据量上限
            cap_old_bs = (f_old_bs_vec * self.cfg.tau) / phi
            # 3. 真实处理量：取 [实际残留量] 和 [算力天花板] 的最小值
            l_proc_old_bs = np.minimum(L_left_prev_vec, cap_old_bs)

        # Sat 旧任务: 假设交给旧卫星处理，本帧内被移出当前用户队列
        l_proc_old_sat = np.zeros_like(L_t)
        total_l_sat_prev = np.sum(self.L_Sat_left_prev_vec)

        if total_l_sat_prev > 1e-9:
            # 旧卫星将算力按残留数据比例分配给各个任务
            f_old_sat_vec = self.cfg.f_max_Sat * (self.L_Sat_left_prev_vec / total_l_sat_prev)
            # 旧卫星在当前 \tau 时间内能处理的最大物理量
            cap_old_sat = (f_old_sat_vec * self.cfg.tau) / phi
            # 真实处理量取最小值
            l_proc_old_sat = np.minimum(self.L_Sat_left_prev_vec, cap_old_sat)

        # 总处理量 (用于更新虚拟队列 Q 的 Drift)
        # 包含：所有新任务的处理量 + 所有旧任务(积压)的清除量
        l_proc_total = l_proc_bs_new + l_proc_sat_new + l_proc_loc_new + l_proc_old_bs + l_proc_old_sat

        # --- B. 计算能耗 ---

        # 1. BS 能耗
        # Part 1: 处理新任务的能耗
        e_bs_new = np.sum(kappa1 * phi * (f_bs ** 2) * l_proc_bs_new)

        # Part 2: 处理旧任务(上帧残留)的能耗 (E_old)
        e_old = 0.0
        total_l_prev = np.sum(L_left_prev_vec)
        if total_l_prev > 1e-9:
            # 根据 Eq. 50: 按比例分配频率
            ratio = L_left_prev_vec / total_l_prev
            f_old_vec = self.cfg.f_max_BS * ratio
            e_old = np.sum(kappa1 * phi * (f_old_vec ** 2) * L_left_prev_vec)

        e_bs_total = e_bs_new + e_old

        # 2. Sat Energy (只计算本帧新分配卫星的能耗，旧卫星能耗由旧卫星负责/或忽略)
        e_sat = np.sum(kappa2 * phi * (f_sat ** 2) * l_proc_sat_new)

        # --- C. 计算 PAoI ---

        # 1. 基站：估算下一帧处理本帧残留所需的标量时间
        total_left_bs_new = np.sum(l_left_bs_new)
        t_next_left_bs_scalar = 0.0
        t_next_left_bs_est = np.zeros_like(L_t)
        if total_left_bs_new > 1e-9:
            # 化简：所有残留任务共享算力，等待时间等于处理总残留的时间
            t_next_left_bs_scalar = (phi * total_left_bs_new) / self.cfg.f_max_BS
            t_next_left_bs_est[l_left_bs_new > 1e-9] = t_next_left_bs_scalar

        # 2. 卫星：估算旧卫星在下一帧处理本帧残留所需的标量时间
        total_left_sat_new = np.sum(l_left_sat_new)
        t_next_left_sat_scalar = 0.0
        t_next_left_sat_est = np.zeros_like(L_t)
        if total_left_sat_new > 1e-9:
            # 化简：同样适用总残留量除以卫星总算力
            t_next_left_sat_scalar = (phi * total_left_sat_new) / self.cfg.f_max_Sat
            t_next_left_sat_est[l_left_sat_new > 1e-9] = t_next_left_sat_scalar

        # 3. 本地 PAoI
        paoi_loc = np.where(l_vec == 1, (phi * L_t) / f_local, 0.0)

        # 4. BS PAoI
        time_finish_bs = np.maximum(T_tran_bs, T_left_prev) + (l_proc_bs_new * phi / (f_bs + 1e-9))
        paoi_bs = np.where(l_left_bs_new > 1e-9,
                           self.cfg.tau + self.cfg.w * t_next_left_bs_est,
                           time_finish_bs)
        paoi_bs = np.where(mask_bs, paoi_bs, 0.0)

        # 5. Sat PAoI
        paoi_sat = np.where(l_left_sat_new > 1e-9,
                            self.cfg.tau + self.cfg.w * t_next_left_sat_est,
                            (self.cfg.tau - T_avail_sat) + l_proc_sat_new * phi / (f_sat + 1e-9))
        paoi_sat = np.where(mask_sat, paoi_sat, 0.0)

        # 6. 合并总 PAoI
        paoi_total = paoi_bs + paoi_sat + paoi_loc

        # --- D. 组合 G1 ---
        # Drift Term: Q * (Arrival - Service)
        term_q = np.sum(((Q_bs + Q_sat) / 1e7) * ((self.l_proc_total_prev - l_proc_total) / 1e7))
        term_p = self.cfg.K_p * np.sum(paoi_total)
        term_e_bs = E_BS * (e_bs_total - self.cfg.E_max_BS)

        G1 = term_q + term_p + term_e_bs

        real_drift = 0.5 * np.sum((L_t - l_proc_total) ** 2) + 0.5 * ((e_bs_total - self.cfg.E_max_BS) ** 2)

        # 更新 details 字典，将基站和卫星的残留量都导出去，并加入标量时间
        details = {
            'l_proc_total': l_proc_total,
            'l_proc_bs': l_proc_bs_new,
            'l_proc_sat': l_proc_sat_new,
            'l_proc_old_sat': l_proc_old_sat,
            'l_left_bs': l_left_bs_new,
            'l_left_sat': l_left_sat_new,  # [补充] 卫星的新残留量
            'e_bs_total': e_bs_total,
            'e_sat': e_sat,
            'paoi': paoi_total,
            'real_drift': real_drift,
            't_next_left_bs_scalar': t_next_left_bs_scalar  # 只传 BS 的标量给 update_env 即可
        }

        return G1, details

    def update_env(self, sol, L_t):
        """
        根据最佳策略更新系统状态 (Algorithm 1)
        """
        details = sol['details']
        l_vec = sol['l']
        b_vec = sol['b']

        # 提取卸载掩码
        mask_bs = (l_vec == 0) & (b_vec == 1)
        mask_sat = (l_vec == 0) & (b_vec == 0)

        # ==========================================================
        # 1. 更新虚拟任务队列 Q (Lyapunov Queue)
        # ==========================================================
        # BS Queue: Q(t+1) = max(0, Q(t) + Arrival - Service)
        # Service 包含处理的新任务和被处理掉的旧任务(上帧残留)
        service_bs_total = details['l_proc_bs'] + self.L_BS_left_prev_vec
        arrival_bs = np.where(mask_bs, L_t, 0.0)
        self.Q_bs = np.maximum(0, self.Q_bs + arrival_bs - service_bs_total)

        # Sat Queue:
        # Service 包含新卫星处理的新任务，以及旧卫星在后台处理掉的积压任务
        service_sat_total = details['l_proc_sat'] + details['l_proc_old_sat']
        arrival_sat = np.where(mask_sat, L_t, 0.0)
        self.Q_sat = np.maximum(0, self.Q_sat + arrival_sat - service_sat_total)

        # 汇总总积压，用于下一帧 DNN 的状态输入
        self.Q_total = self.Q_bs + self.Q_sat

        # ==========================================================
        # 2. 更新虚拟能量队列 E
        # ==========================================================
        # 基站 (固定不动，有长期的平均能耗约束，必须跨帧累加)
        e_bs_total = details['e_bs_total']
        self.E_BS = max(0.0, self.E_BS + e_bs_total - self.cfg.E_max_BS)

        # 卫星 (高速移动，单帧过境，只受瞬时硬约束限制，无需跨帧维护 E_Sat)
        # 因此彻底删除了 self.E_Sat 的队列更新逻辑

        # ==========================================================
        # 3. 更新物理状态 (为下一帧做准备)
        # ==========================================================
        # --- 基站 (BS) 状态更新 ---
        l_left_bs_next = details['l_left_bs']  # 提取本帧产生的新残留任务

        # 直接从 details 读取已在 calculate_objective 中算好的标量阻塞时间，彻底消除冗余
        self.T_BS_left_prev = details.get('t_next_left_bs_scalar', 0.0)

        # 保存本帧的残留任务向量，它将在下一帧变为旧任务 L_BS_left_prev_vec
        if np.sum(l_left_bs_next) > 1e-9:
            self.L_BS_left_prev_vec = l_left_bs_next.copy()
        else:
            self.L_BS_left_prev_vec = np.zeros(self.cfg.J)
        self.l_proc_total_prev = details['l_proc_total'].copy()
        # --- 卫星 (Sat) 状态说明 ---
        # 下一帧接入的是全新的卫星，拥有完整的 tau，无需扣除旧卫星的残留时间。
        # 旧卫星带走的未完成任务，其延迟 PAoI 惩罚已经在 calculate_objective 中结算完毕。
        # 所以这里不需要维护任何卫星的物理残留时间。
        l_left_sat_next = details['l_left_sat']
        if np.sum(l_left_sat_next) > 1e-9:
            self.L_Sat_left_prev_vec = l_left_sat_next.copy()
        else:
            self.L_Sat_left_prev_vec = np.zeros(self.cfg.J)
        # ==========================================================
        # 4. 记录历史 (用于 matplotlib 画图)
        # ==========================================================
        self.history['Q_total'].append(np.mean(self.Q_total))
        self.history['Q_bs'].append(np.mean(self.Q_bs))
        self.history['Q_sat'].append(np.mean(self.Q_sat))
        self.history['Cost'].append(np.mean(details['paoi']))

        self.history['E_virt_bs'].append(self.E_BS)
        # 卫星能量不作为队列维护，但依然可以记录单帧真实消耗 e_sat 用于画图观察
        self.history['E_virt_sat'].append(details['e_sat'])

        self.history['Drift'].append(details['real_drift'])

    def store_experience(self, state_tensor, best_action_b):
        """将 (State, Best_Action) 存入 Replay Buffer"""
        # state_tensor: (1, input_dim) -> squeeze -> (input_dim,)
        # best_action_b: (J,) 0/1 vector
        s = state_tensor.detach().numpy().flatten()
        a = best_action_b.copy()
        self.memory.append((s, a))

    def train_network(self):
        """从 Replay Buffer 采样并更新 DNN 参数"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # 1. 随机采样
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch = zip(*batch)

        # 2. 转为 Tensor
        # states: (Batch, Input_Dim)
        states = torch.FloatTensor(np.array(state_batch))
        # targets: (Batch, J) - 这些是 TCOPQ 筛选出的最佳 b
        targets = torch.FloatTensor(np.array(action_batch))

        # 3. 前向传播
        self.actor.train()  # 切换到训练模式
        self.optimizer.zero_grad()

        # probs: (Batch, J)
        probs = self.actor(states)

        # 4. 计算损失 (Focal Loss)
        # 使得网络输出的 probs 尽可能接近 targets (0 或 1)
        loss = self.criterion(probs, targets)

        # 5. 反向传播
        loss.backward()
        self.optimizer.step()

        return loss.item()