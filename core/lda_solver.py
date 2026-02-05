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

        # [新增] 2. 训练组件初始化
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
        self.frame_count = 0  # [新增] 计数器

    def reset(self):
        """重置环境状态"""
        self.Q_bs = np.zeros(self.cfg.J)
        self.Q_sat = np.zeros(self.cfg.J)
        self.Q_total =np.zeros(self.cfg.J)
        self.E_BS = 0.0
        self.E_Sat = 0.0  # [新增] 卫星虚拟能量队列

        self.T_BS_left_prev = 0.0
        self.L_BS_left_prev_vec = np.zeros(self.cfg.J)

        self.history = {
            'Q_total': [],
            'Q_bs': [],
            'Q_sat': [],
            'PAoI': [],
            'Cost': [],
            'E_virt_bs': [],
            'E_virt_sat': [],
            'Loss': []
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

        # --- [新增] Step 4.6: 触发训练 ---
        if self.frame_count % self.train_interval == 0:
            loss_val = self.train_network()
            self.history['Loss'].append(loss_val)
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

        # 1. BS 处理情况
        # [关键修正] 只有卸载到 BS (mask_bs=True) 的用户，其新任务 L_t 和积压 Q 才会进入 BS 队列
        load_bs_total = np.where(mask_bs, L_t + Q_bs, 0.0)

        # BS 物理可用时间 (扣除传输和旧任务阻塞)
        t_bs_avail_phys = np.maximum(0, self.cfg.tau - np.maximum(T_tran_bs, T_left_prev))
        # 理论最大处理量
        cap_bs = f_bs * t_bs_avail_phys / phi

        # 实际处理量 (取 总负载 和 容量 的最小值)
        # 此时 l_proc_bs 可能大于 L_t，从而消除 Q
        l_proc_bs = np.minimum(load_bs_total, cap_bs)

        # 残留量 (没做完的部分)
        l_left_bs = np.maximum(0, load_bs_total - l_proc_bs)

        # 2. Sat 处理情况
        # [关键修正] 同理，Sat 也要处理 Q
        load_sat_total = np.where(mask_sat, L_t, 0.0)

        cap_sat = f_sat * T_avail_sat / phi
        l_proc_sat = np.minimum(load_sat_total, cap_sat)
        l_left_sat = np.maximum(0, load_sat_total - l_proc_sat)
        l_proc_old_sat = Q_sat #这里直接默认可以直接处理掉
        # 3. Local 处理情况
        load_to_loc = np.where(l_vec == 1, L_t, 0.0)
        l_proc_loc = load_to_loc

        # 总处理量 (用于更新 Q)
        l_proc_total = l_proc_bs + l_proc_sat + l_proc_loc + l_proc_old_sat

        # --- B. 计算能耗 ---

        # 1. BS 能耗 (含 E_old)
        e_bs_new = np.sum(kappa1 * phi * (f_bs ** 2) * l_proc_bs)
        e_old = 0.0
        total_l_prev = np.sum(L_left_prev_vec)
        if total_l_prev > 1e-9:
            f_old_vec = self.cfg.f_max_BS * (L_left_prev_vec / total_l_prev)
            e_old = np.sum(kappa1 * phi * (f_old_vec ** 2) * L_left_prev_vec)
        e_bs_total = e_bs_new + e_old

        # 2. Sat Energy
        e_sat = np.sum(kappa2 * phi * (f_sat ** 2) * l_proc_sat)

        # --- C. 计算 PAoI ---
        # 逻辑：如果做完了(残留很少)，延迟是执行时间；如果没做完，延迟是 tau + 罚项

        # BS PAoI
        if np.sum(l_left_bs) > 1e-9:
            f_next_est = self.cfg.f_max_BS * (l_left_bs / (np.sum(l_left_bs) + 1e-9))
            t_next_left = phi * l_left_bs / (f_next_est + 1e-9)
        else:
            t_next_left = np.zeros_like(L_t)

        paoi_bs = np.where(l_left_bs > 1e-9,
                           self.cfg.tau + self.cfg.w * t_next_left,
                           T_tran_bs + l_proc_bs * phi / (f_bs + 1e-9))
        paoi_bs = np.where(mask_bs, paoi_bs, 0.0)

        # Sat PAoI
        paoi_sat = np.where(l_left_sat > 1e-9,
                            self.cfg.tau + self.cfg.w * (phi * l_left_sat / self.cfg.f_max_Sat),
                            (self.cfg.tau - T_avail_sat) + l_proc_sat * phi / (f_sat + 1e-9))
        paoi_sat = np.where(mask_sat, paoi_sat, 0.0)

        paoi_total = paoi_bs + paoi_sat

        # --- D. 组合 G1 ---

        # Drift Minimize term: Q * (L_t - l_proc_total)
        # 如果 l_proc_total > L_t (因为处理了 Q)，这这一项就会变成负数，从而降低 Cost
        term_q = np.sum((Q_bs+Q_sat) * (L_t - l_proc_total))

        term_p = self.cfg.K_p * np.sum(paoi_total)

        # 能量漂移
        term_e_bs = E_BS * (e_bs_total - self.cfg.E_max_BS)
        #term_e_sat = E_Sat * (e_sat - self.cfg.E_max_Sat)

        G1 = term_q + term_p + term_e_bs

        # --- 调试与记录 ---
        # 真实的 Drift
        quad_q = 0.5 * np.sum((L_t - l_proc_total) ** 2)
        quad_e = 0.5 * ((e_bs_total - self.cfg.E_max_BS) ** 2)
        real_drift = (term_q + term_e_bs) + (quad_q + quad_e)

        details = {
            'l_proc_total': l_proc_total,
            'l_proc_bs': l_proc_bs,  # 新增：用于更新 Q_bs
            'l_proc_sat': l_proc_sat,  # 新增：用于更新 Q_sat
            'l_proc_old_sat': l_proc_old_sat,  # 新增
            'l_left_bs': l_left_bs,
            'e_bs_total': e_bs_total,
            'e_sat': e_sat,
            'paoi': paoi_total,
            'real_drift': real_drift
        }

        return G1, details

    def update_env(self, sol, L_t):
        """
        根据最佳策略更新系统状态
        对应公式 (26), (27) 以及物理状态的时间演进
        """
        details = sol['details']

        # 1. 获取当前帧决策掩码 (从 sol 中重建) [cite: 438]
        l_vec = sol['l']
        b_vec = sol['b']
        mask_bs = (l_vec == 0) & (b_vec == 1)
        mask_sat = (l_vec == 0) & (b_vec == 0)

        # 2. 分别更新物理任务队列 (Eq. 26 变体) [cite: 890]
        # 更新基站队列：旧积压 + 新到BS任务 - 基站处理量
        self.Q_bs = np.maximum(0, self.Q_bs + np.where(mask_bs, L_t, 0.0) - details['l_proc_bs'])

        # 更新卫星队列：旧积压 - 旧星处理量 + 新到Sat任务 - 新星处理量 [cite: 436, 659]
        l_proc_old_sat = details.get('l_proc_old_sat', self.Q_sat)
        self.Q_sat = np.maximum(0, self.Q_sat - l_proc_old_sat +
                                np.where(mask_sat, L_t, 0.0) - details['l_proc_sat'])

        self.Q_total = self.Q_bs + self.Q_sat

        # 2. Update E_BS
        e_bs_total = details['e_bs_total']
        self.E_BS = max(0.0, self.E_BS + e_bs_total - self.cfg.E_max_BS)

        # 3. [新增] Update E_Sat
        e_sat = details['e_sat']
        self.E_Sat = max(0.0, self.E_Sat + e_sat - self.cfg.E_max_Sat)

        # 4. 更新物理状态 (为下一帧做准备)
        # 也就是计算 T^{left}(t) 和 L^{left}(t)，这将成为下一帧的 T_left_prev
        l_left_bs = details['l_left_bs']
        total_l_left = np.sum(l_left_bs)

        if total_l_left > 1e-9:
            # 对应论文 Eq. 51: 所有残留任务共享资源
            # 计算这些残留任务会占用下一帧多少时间
            # T_left = (phi * Sum(L_left)) / f_max
            self.T_BS_left_prev = (self.cfg.phi * total_l_left) / self.cfg.f_max_BS

            # [关键] 记录具体的残留量向量
            # 下一帧计算 E_old 时，需要知道每个用户具体留了多少，以便按比例分配频率
            self.L_BS_left_prev_vec = l_left_bs
        else:
            self.T_BS_left_prev = 0.0
            self.L_BS_left_prev_vec = np.zeros(self.cfg.J)

        # 5. 记录历史 (Logging)
        self.history['Q_total'].append(np.mean(self.Q_total))
        self.history['Q_bs'].append(np.mean(self.Q_bs))  # 记录 BS 积压均值
        self.history['Q_sat'].append(np.mean(self.Q_sat))  # 记录 Sat 积压均值
        self.history['Cost'].append(np.mean(details['paoi']))
        self.history['E_virt_bs'].append(self.E_BS)
        self.history['E_virt_sat'].append(self.E_Sat)  # [新增]
        if 'real_drift' in details:
            if 'Drift' not in self.history: self.history['Drift'] = []
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