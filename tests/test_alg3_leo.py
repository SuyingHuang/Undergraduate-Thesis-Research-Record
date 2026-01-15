import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 路径 hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig
from core.leo_optimizer import LEO_Optimizer
from core.satellite_channel import SatelliteChannel


def run_integrated_test():
    # 初始化
    cfg = SystemConfig()
    optimizer = LEO_Optimizer(cfg)
    channel_model = SatelliteChannel(cfg)

    total_users = cfg.I * cfg.J
    print(f"=== 启动 P4 (LEOS) 集成测试 ===")
    print(f"模式: Shadowed-Rician 信道 + Algorithm 3")
    print(f"用户数: {total_users}, 卫星 E_max: {cfg.E_max_Sat}J, F_max: {cfg.f_max_Sat / 1e9}GHz")

    # 状态变量
    Q_current = np.zeros(total_users)
    E_virt_sat = 0.0  # 虚拟能量队列 (Lyapunov)
    T_left_prev = 0.0  # 上一帧残留任务带来的阻塞时间

    # 记录字典 (与 test_alg2_bs 对齐)
    records = {
        'Q': [],  # 平均队列积压
        'E': [],  # 虚拟能量队列
        'E_phys': [],  # 实际物理能耗
        'Freq': [],  # 总频率使用
        'PAoI': [],  # 平均 PAoI
        'Rate': []  # 平均上行速率
    }

    for t in range(cfg.sim_frames):
        # --- 1. 环境与信道 ---
        # 生成任务
        L_new = np.maximum(np.random.normal(cfg.L_mean, cfg.L_std, total_users), 0)

        # 生成动态距离 (模拟卫星运动)
        d_t = cfg.H_sat + np.random.uniform(0, 500e3, total_users)

        # 生成信道衰落与速率
        h_sq = channel_model.generate_channel_gain_samples(total_users)
        R_sat, T_prop = channel_model.calculate_uplink_rate(d_t, h_sq)

        # 计算传输时延
        T_tran = np.zeros(total_users)
        mask_r = R_sat > 1e-9
        T_tran[mask_r] = L_new[mask_r] / R_sat[mask_r]
        T_tran[~mask_r] = 10.0  # 无法传输

        # 计算可用计算时间 T_avail (考虑传输、传播和上一帧残留)
        # T_occupancy = T_tran + T_prop + T_left_prev ?
        # 简化模型：T_avail = [tau - (T_tran + T_prop)]+
        # 这里假设 T_left_prev 是在卫星端的计算排队，不影响传输，但挤占计算时间
        T_total_delay = T_tran + T_prop
        T_avail = np.maximum(0, cfg.tau - T_total_delay)

        # --- 2. 优化 (Algorithm 3) ---
        # 注意：这里 Q_current 传入的是当前积压
        f_alloc = optimizer.optimize(L_new, Q_current, T_avail)

        # --- 3. 物理状态更新 ---
        # 计算处理能力与实际处理量
        capacity = np.maximum(f_alloc, 1e-9) * T_avail / cfg.phi
        L_load = Q_current + L_new
        L_proc = np.minimum(L_load, capacity)

        # 计算物理能耗 E_phys
        e_phys = np.sum(cfg.kappa2 * cfg.phi * (f_alloc ** 2) * L_proc)

        # 更新队列 Q(t+1)
        Q_next = np.maximum(L_load - L_proc, 0)

        # 更新虚拟能量队列 E(t+1)
        E_virt_sat = max(0.0, E_virt_sat + e_phys - cfg.E_max_Sat)

        # 估算 PAoI
        # PAoI ~ T_tran + T_prop + T_proc + Penalty
        proc_time = np.zeros(total_users)
        mask_proc = L_proc > 0
        proc_time[mask_proc] = cfg.phi * L_proc[mask_proc] / np.maximum(f_alloc[mask_proc], 1e-9)

        # 估算下一帧残留时间 (用于 PAoI 惩罚)
        if np.sum(Q_next) > 1e-9:
            T_next_left = (cfg.phi * np.sum(Q_next)) / cfg.f_max_Sat
            penalty = np.where(Q_next > 1e-6, cfg.w * T_next_left, 0.0)
        else:
            T_next_left = 0.0
            penalty = np.zeros(total_users)

        paoi_val = np.mean(T_total_delay + proc_time + penalty)

        # 状态迭代
        Q_current = Q_next
        T_left_prev = T_next_left

        # 记录数据
        records['Q'].append(np.mean(Q_current) / 1e6)
        records['E'].append(E_virt_sat)
        records['E_phys'].append(e_phys)
        records['Freq'].append(np.sum(f_alloc) / 1e9)
        records['PAoI'].append(paoi_val)
        records['Rate'].append(np.mean(R_sat) / 1e6)

        if t % 50 == 0:
            print(f"Frame {t:4d} | Q: {records['Q'][-1]:5.2f}Mb | "
                  f"Freq: {records['Freq'][-1]:4.2f}G | "
                  f"E_phys: {records['E_phys'][-1]:5.1f}J")

    # --- 4. 绘图 (与 test_alg2_bs 对齐) ---
    plot_comparison(records, cfg)


def plot_comparison(rec, cfg):
    plt.figure(figsize=(15, 8))

    # 1. 平均队列 Q
    plt.subplot(2, 3, 1)
    plt.plot(rec['Q'])
    plt.title('Avg Queue (Mb)')
    plt.grid(True, alpha=0.3)

    # 2. 虚拟能量队列 E (Lyapunov)
    plt.subplot(2, 3, 2)
    plt.plot(rec['E'], color='orange')
    plt.title('Virtual Energy Queue (Sat)')
    plt.grid(True, alpha=0.3)

    # 3. 物理能耗 vs 预算
    plt.subplot(2, 3, 3)
    plt.plot(rec['E_phys'], color='green', label='Used')
    plt.axhline(cfg.E_max_Sat, color='red', ls='--', label='Limit')
    plt.title('Phys Energy vs Budget (J)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 频率使用 vs 上限
    plt.subplot(2, 3, 4)
    plt.plot(rec['Freq'], color='purple', label='Used')
    plt.axhline(cfg.f_max_Sat / 1e9, color='red', ls='--', label='Limit')
    plt.title('Freq Usage (GHz)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. 平均 PAoI
    plt.subplot(2, 3, 5)
    plt.plot(rec['PAoI'], color='blue')
    plt.title('Avg PAoI (s)')
    plt.grid(True, alpha=0.3)

    # 6. 平均上行速率
    plt.subplot(2, 3, 6)
    plt.plot(rec['Rate'], color='brown')
    plt.title('Avg Uplink Rate (Mbps)')
    plt.ylabel('Mbps')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_integrated_test()