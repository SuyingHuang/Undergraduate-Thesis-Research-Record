import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# [重要] 将项目根目录加入路径，确保能 import core 和 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig
from core.bs_optimizer import BS_Optimizer


def run_bs_verification():
    """
    专门验证 Algorithm 2 (基站资源分配) 的独立测试脚本。
    不包含卫星，只模拟所有任务强制卸载到基站的场景。
    """
    cfg = SystemConfig()
    optimizer = BS_Optimizer(cfg)

    # 初始化状态
    Q_current = np.zeros(cfg.J)
    E_backlog = 0.0
    T_left_prev = 0.0

    # 记录
    records = {
        'Q': [], 'E': [], 'E_phys': [], 'Freq': [], 'PAoI': [], 'R_uplink':[]
    }

    print(f"=== 启动基准测试: Algorithm 2 Verification ===")
    print(f"参数检查: Kappa1={cfg.kappa1} (未修改), E_max={cfg.E_max_BS}")

    for t in range(cfg.sim_frames):
        # 1. 生成任务
        new_tasks = np.maximum(np.random.normal(cfg.L_mean, cfg.L_std, cfg.J), 0)

        # 2. 生成信道 (动态距离)
        d = np.random.uniform(cfg.d_min, cfg.d_max, cfg.J)
        gain = (cfg.wl_c / (4 * np.pi * d)) ** 2
        snr = (cfg.p_tx * gain*cfg.G_rx_bs) / cfg.sigma1
        R_uplink = (cfg.B_c / cfg.J) * np.log2(1 + snr)

        T_tran = new_tasks / R_uplink
        # 这里在原来的基础上记录上
        records['R_uplink'].append(np.mean(R_uplink)/1e6)


        # 3. 执行优化 (Algorithm 2)
        f_alloc = optimizer.optimize(new_tasks, Q_current, E_backlog, T_tran, T_left_prev)

        # 4. 物理状态更新 (Environment Step)
        delay_occ = np.maximum(T_tran, T_left_prev)
        t_avail = np.maximum(0, cfg.tau - delay_occ)

        # 计算处理量
        capacity = np.maximum(f_alloc, 1e-6) * t_avail / cfg.phi
        total_load = Q_current + new_tasks
        processed = np.minimum(total_load, capacity)

        # 计算物理能耗
        e_phys = np.sum(cfg.kappa1 * cfg.phi * (f_alloc ** 2) * processed)

        # 计算 PAoI
        q_next = np.maximum(total_load - processed, 0.0)
        # 估算处理时间 (Type B 用满 t_avail, Type A 用实际时间)
        proc_time = np.zeros(cfg.J)
        mask = processed > 0
        proc_time[mask] = cfg.phi * processed[mask] / np.maximum(f_alloc[mask], 1e-6)

        # 估算下一帧拥堵惩罚 (基于论文 Eq.50 比例分配)
        if np.sum(q_next) > 1e-9:
            T_next_left = (cfg.phi * np.sum(q_next)) / cfg.f_max_BS
            penalty = np.where(q_next > 1e-6, cfg.w * T_next_left, 0.0)
        else:
            T_next_left = 0.0
            penalty = np.zeros(cfg.J)

        paoi = np.mean(delay_occ + proc_time + penalty)

        # 状态迭代
        Q_current = q_next
        E_backlog = max(0.0, E_backlog + e_phys - cfg.E_max_BS)
        T_left_prev = T_next_left

        # 记录
        records['Q'].append(np.mean(Q_current) / 1e6)
        records['E'].append(E_backlog)
        records['E_phys'].append(e_phys)
        records['Freq'].append(np.sum(f_alloc) / 1e9)
        records['PAoI'].append(paoi)

        if t % 50 == 0:
            print(f"Frame {t:3d} | Q: {records['Q'][-1]:5.2f}Mb | E_virt: {E_backlog:5.1f}")

    # 绘图验证
    plot_verification(records, cfg)


def plot_verification(rec, cfg):
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.plot(rec['Q'])
    plt.title('Avg Queue (Mb)')

    plt.subplot(2, 3, 2)
    plt.plot(rec['E'], color='orange')
    plt.title('Virtual Energy Queue')

    plt.subplot(2, 3, 3)
    plt.plot(rec['E_phys'], color='green')
    plt.axhline(cfg.E_max_BS, color='red', ls='--')
    plt.title('Phys Energy vs Budget')

    plt.subplot(2, 3, 4)
    plt.plot(rec['Freq'], color='purple')
    plt.axhline(cfg.f_max_BS / 1e9, color='red', ls='--')
    plt.title('Freq Usage (GHz)')

    plt.subplot(2, 3, 5)
    plt.plot(rec['PAoI'], color='blue')
    plt.title('Avg PAoI (s)')

    plt.tight_layout()
    plt.show()

    plt.subplot(2, 3, 6)
    plt.plot(rec['R_uplink'], color='brown')
    plt.ylabel('Rate (Mbps)')
    plt.title('Avg Uplink Transmission')

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    run_bs_verification()