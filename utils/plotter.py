# utils/plotter.py
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth_curve(data, window_size=50):
    """
    滑动平均滤波辅助函数
    用于平滑强化学习由于探索（Exploration）带来的高频震荡，凸显长期的收敛趋势
    """
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_results(history, cfg, save_path='simulation_results.png'):
    """
    绘制关键性能指标曲线 (适配独立 Env 环境)
    :param history: 环境记录的数据字典
    :param cfg: 系统配置
    :param save_path: 图片保存路径
    """
    frames = range(len(history['Q_total']))

    # 创建一个 3x2 的图表布局
    plt.figure(figsize=(16, 12))
    plt.suptitle(f"LDA Distributed System Simulation (Frames: {len(frames)})", fontsize=16, fontweight='bold', y=0.98)

    # ==========================================
    # --- 子图 1: 强化学习 Reward 曲线 ---
    # ==========================================
    plt.subplot(3, 2, 1)
    if 'Reward' in history and len(history['Reward']) > 0:
        raw_reward = history['Reward']
        smoothed_reward = smooth_curve(raw_reward, window_size=50)

        plt.plot(frames, raw_reward, color='lightgray', alpha=0.6, label='Raw Reward ($-G_1(t)$)')
        if len(raw_reward) >= 50:
            x_smoothed = np.arange(50 // 2 - 1, 50 // 2 - 1 + len(smoothed_reward))
            plt.plot(x_smoothed, smoothed_reward, color='#d62728', linewidth=2.5, label='Smoothed Trend')
        else:
            plt.plot(frames, smoothed_reward, color='#d62728', linewidth=2.5, label='Smoothed Trend')

    plt.title("RL Reward Curve (Convergence)")
    plt.xlabel("Time Frame")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    # ==========================================
    # --- 子图 2: DNN 训练损失 Loss ---
    # ==========================================
    plt.subplot(3, 2, 2)
    if 'Loss' in history and len(history['Loss']) > 0:
        frames_x = [item[0] for item in history['Loss']]
        loss_y = [item[1] for item in history['Loss']]
        plt.plot(frames_x, loss_y, linewidth=1.5, color='#9467bd', marker='.', markersize=4)

        plt.title("Distributed DNN Training Loss (Avg Focal Loss)")
        plt.ylabel("Loss Value")
        plt.xlabel("Time Frame")
        plt.yscale('log')
    else:
        plt.text(0.5, 0.5, "No Training Data", ha='center', va='center', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)

    # ==========================================
    # --- 子图 3: 任务积压 Q(t) ---
    # ==========================================
    plt.subplot(3, 2, 3)
    plt.plot(frames, history['Q_total'], linewidth=2, color='#1f77b4', label='Global Avg Workload')
    if 'Q_bs' in history:
        plt.plot(frames, history['Q_bs'], '--', color='#2ca02c', label='Avg BS Queue', alpha=0.8)
    if 'Q_sat' in history:
        plt.plot(frames, history['Q_sat'], ':', color='#ff7f0e', label='Avg Sat Queue', alpha=0.8)

    plt.title('Task Queue Backlog (Averaged over I*J Users)')
    plt.xlabel('Time Frame')
    plt.ylabel('Queue Length (bits)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    # ==========================================
    # --- 子图 4: 虚拟能量队列 E(t) ---
    # ==========================================
    plt.subplot(3, 2, 4)
    plt.plot(frames, history['E_virt_bs'], linewidth=2, color='#ff7f0e', label='Avg BS Energy Queue ($E_{virt}$)')

    plt.title("Virtual Energy Queues (Power Constraint)")
    plt.ylabel("Energy Deficit Level")
    plt.xlabel("Time Frame")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    # ==========================================
    # --- 子图 5: 系统性能 Cost (PAoI) ---
    # ==========================================
    plt.subplot(3, 2, 5)
    window_size = 20
    cost_data = np.array(history['Cost'])

    plt.plot(frames, cost_data, color='lightblue', alpha=0.5, label='Raw PAoI')
    if len(cost_data) >= window_size:
        cost_smooth = np.convolve(cost_data, np.ones(window_size) / window_size, mode='valid')
        x_smooth = np.arange(window_size // 2 - 1, window_size // 2 - 1 + len(cost_smooth))
        plt.plot(x_smooth, cost_smooth, linewidth=2.5, color='#17becf', label=f'Moving Avg ({window_size})')

    plt.title(f"Global System Cost (Average PAoI)")
    plt.ylabel("Time Penalty (Seconds)")
    plt.xlabel("Time Frame")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    # ==========================================
    # --- 子图 6: 传输速率动态范围 ---
    # ==========================================
    plt.subplot(3, 2, 6)
    if 'R_bs_max' in history and len(history['R_bs_max']) > 0:
        r_bs_max = np.array(history['R_bs_max'][:len(frames)]) / 1e6
        r_bs_min = np.array(history['R_bs_min'][:len(frames)]) / 1e6
        r_sat_max = np.array(history['R_sat_max'][:len(frames)]) / 1e6
        r_sat_min = np.array(history['R_sat_min'][:len(frames)]) / 1e6

        plt.plot(frames, r_bs_max, color='#2ca02c', linestyle='-', linewidth=1.5, label='BS Max')
        plt.plot(frames, r_bs_min, color='#98df8a', linestyle='--', linewidth=1.5, label='BS Min')
        plt.fill_between(frames, r_bs_min, r_bs_max, color='#2ca02c', alpha=0.15)

        plt.plot(frames, r_sat_max, color='#ff7f0e', linestyle='-', linewidth=1.5, label='Sat Max')
        plt.plot(frames, r_sat_min, color='#ffbb78', linestyle='--', linewidth=1.5, label='Sat Min')
        plt.fill_between(frames, r_sat_min, r_sat_max, color='#ff7f0e', alpha=0.15)

    plt.title("Channel Transmission Rates (Extrema)")
    plt.ylabel("Data Rate (Mbps)")
    plt.xlabel("Time Frame")
    plt.legend(loc='best', ncol=2, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # [修复核心] 这里使用传入的 save_path 变量进行保存
    plt.savefig(save_path, dpi=150)
    print(f">>> 完美！图表已生成并保存至: {os.path.abspath(save_path)}")
    plt.show()