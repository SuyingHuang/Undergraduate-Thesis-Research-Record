import numpy as np
import matplotlib.pyplot as plt
from config import SystemConfig
from core import BS_Optimizer


def run_simulation():
    cfg = SystemConfig()
    optimizer = BS_Optimizer(cfg)

    Q_current = np.zeros(cfg.J)
    E_backlog = 0.0
    T_left_prev = 0.0

    # 历史记录
    history_Q_avg = []
    history_E = []
    history_Energy_Phys = []
    history_Freq_Usage = []
    history_T_left = []

    # [新增] PAoI 记录
    history_PAoI_avg = []

    print(f"=== 开始仿真: {cfg.sim_frames} 帧, {cfg.J} 用户 ===")
    print(f"模式: 自由空间路径损耗 + 牛顿迭代 + PAoI计算")

    for t in range(cfg.sim_frames):
        # ===========================
        # A. 环境生成: 任务 & 信道
        # ===========================
        # 1. 生成任务
        new_tasks = np.maximum(np.random.normal(cfg.L_mean, cfg.L_std, cfg.J), 0)

        # 2. [新增] 动态信道模型 (Free Space Path Loss) [Eq. 4 & 7]
        # (1) 随机生成距离 (每帧变化，符合论文描述)
        distances = np.random.uniform(cfg.d_min, cfg.d_max, cfg.J)

        # (2) 计算信道增益 |h|^2 = (lambda / 4*pi*d)^2
        # 注意: 论文 Eq. 4 h = lambda / 4*pi*d (这是幅值或路径损耗因子)
        # 实际接收功率项通常是 p * |h|^2
        path_loss_factor = cfg.wl_c / (4 * np.pi * distances)
        channel_gain_sq = path_loss_factor ** 2

        # (3) 计算香农速率 R = B/J * log2(1 + SNR)
        # SNR = p * |h|^2 / sigma^2
        snr = (cfg.p_tx * channel_gain_sq) / cfg.sigma2
        # 带宽由 J 个用户均分
        bw_per_user = cfg.B_c / cfg.J
        R_uplink = bw_per_user * np.log2(1 + snr)

        # (4) 计算传输延迟
        T_tran_current = new_tasks / R_uplink

        # ===========================
        # B. 算法决策 (牛顿法优化)
        # ===========================
        f_alloc = optimizer.optimize(new_tasks, Q_current, E_backlog,
                                     T_tran_current, T_left_prev)

        # ===========================
        # C. 物理过程 & PAoI 计算
        # ===========================
        # 1. 可用时间
        delay_occupancy = np.maximum(T_tran_current, T_left_prev)
        t_avail = np.maximum(0, cfg.tau - delay_occupancy)

        # 2. 处理能力与实际处理量
        # 避免除以0
        safe_f = np.maximum(f_alloc, 1e-5)
        capacity_bits = safe_f * t_avail / cfg.phi

        total_load = Q_current + new_tasks
        processed_bits = np.minimum(total_load, capacity_bits)

        # 3. 计算物理能耗
        energy_phys = np.sum(cfg.kappa1 * cfg.phi * (f_alloc ** 2) * processed_bits)

        # 4. [新增] 计算核心指标 PAoI [Eq. 24]
        # 公式分解:
        # Part 1: 等待时间 (tau - t_avail) = delay_occupancy (在帧内的起始时刻)
        # Part 2: 执行时间 (phi * processed / f)
        # Part 3: 惩罚项 (w * T_next_left) -> 这需要算出每个用户的 T_next_left

        # 先算出每个用户剩余的任务量
        q_next_per_user = np.maximum(total_load - processed_bits, 0.0)

        # 计算 T_next_left (下一帧为了处理这个用户的剩余量需要的时间)
        # 论文逻辑：下一帧全速 f_max 处理，按比例分配。
        # 等效于：T_left_user = phi * q_next / (f_max * (q_next/sum_q)) = phi * sum_q / f_max
        # 这意味着所有 Type B 用户的 T_left 是相同的。
        if np.sum(q_next_per_user) > 1e-9:
            T_next_left_system = (cfg.phi * np.sum(q_next_per_user)) / cfg.f_max_BS
            # 只有没做完的用户(Type B)才有这个惩罚，做完的用户(Type A)惩罚为0
            # 实际上 Eq 24 中，对于 Type A，T_left 项本身就是 0
            # 对于 Type B，T_left 项是存在的
            # 我们生成一个 mask
            is_type_B = q_next_per_user > 1e-6
            penalty_term = np.zeros(cfg.J)
            penalty_term[is_type_B] = cfg.w * T_next_left_system
        else:
            penalty_term = np.zeros(cfg.J)

        # 组装 PAoI
        # 注意: 这里的 PAoI 是瞬时值
        process_time = cfg.phi * processed_bits / safe_f
        # 修正: 根据 Eq 24, 第一项是 (tau - T_proc)，即 delay_occupancy
        p_aoi_users = delay_occupancy + process_time + penalty_term

        avg_paoi = np.mean(p_aoi_users)

        # ===========================
        # D. 状态更新
        # ===========================
        Q_next = q_next_per_user
        E_next = max(0.0, E_backlog + energy_phys - cfg.E_max_BS)

        # 更新系统级 T_left_prev (用于下一帧的 Optimize 输入)
        if np.sum(Q_next) > 1e-9:
            T_left_prev = (cfg.phi * np.sum(Q_next)) / cfg.f_max_BS
        else:
            T_left_prev = 0.0

        # ===========================
        # E. 记录
        # ===========================
        history_Q_avg.append(np.mean(Q_current) / 1e6)
        history_E.append(E_backlog)
        history_Energy_Phys.append(energy_phys)
        history_Freq_Usage.append(np.sum(f_alloc) / 1e9)
        history_T_left.append(T_left_prev)
        history_PAoI_avg.append(avg_paoi)

        Q_current = Q_next
        E_backlog = E_next

        if t % 50 == 0:
            print(f"Frame {t:3d} | Q: {np.mean(Q_current) / 1e6:5.2f} Mb | "
                  f"PAoI: {avg_paoi:5.3f} s | "
                  f"E_virt: {E_backlog:6.1f}")

    return history_Q_avg, history_E, history_Energy_Phys, history_Freq_Usage, history_T_left, history_PAoI_avg, cfg


def plot_results(h_Q, h_E, h_Phys, h_Freq, h_T_left, h_PAoI, cfg):
    """绘图增加 PAoI"""
    plt.figure(figsize=(18, 10))
    plt.rcParams.update({'font.size': 10})

    # ... 前 5 个图保持不变 (代码复用上一轮的，略) ...
    # 为了节省篇幅，这里简写，你需要把之前的 1-5 复制过来

    # 子图 1-5 ...
    ax1 = plt.subplot(2, 3, 1);
    ax1.plot(h_Q);
    ax1.set_title('Queue')
    ax2 = plt.subplot(2, 3, 2);
    ax2.plot(h_E, color='orange');
    ax2.set_title('E(t)')
    ax3 = plt.subplot(2, 3, 3);
    ax3.plot(h_Phys, color='green');
    ax3.axhline(cfg.E_max_BS, color='red', ls='--');
    ax3.set_title('Phys Energy')
    ax4 = plt.subplot(2, 3, 4);
    ax4.plot(h_Freq, color='purple');
    ax4.axhline(cfg.f_max_BS / 1e9, color='red', ls='--');
    ax4.set_title('Frequency')
    ax5 = plt.subplot(2, 3, 5);
    ax5.plot(np.array(h_T_left) * 1000, color='brown');
    ax5.set_title('Blocking Time (ms)')

    # 6. [新增] PAoI
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(h_PAoI, color='blue', label='Avg PAoI', linewidth=1.2)
    ax6.set_title('Average Peak Age of Information (PAoI)')
    ax6.set_xlabel('Frame')
    ax6.set_ylabel('PAoI (s)')
    ax6.grid(True, linestyle='--', alpha=0.6)
    ax6.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_simulation()
    plot_results(*results)