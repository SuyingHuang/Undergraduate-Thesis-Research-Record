import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import sys
import os

# [系统路径设置] 确保能导入 core 和 config 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig
from core.lda_solver import LDASolver


def set_seed(seed=42):
    """
    设置全局随机种子，确保实验结果可复现 (Reproducibility)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random_seed = seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_main_simulation():
    # --- 1. 初始化配置与环境 ---
    set_seed(42)  # 设定种子
    cfg = SystemConfig()

    # [调试建议] 初次运行时，可以适当减少帧数以快速看结果
    # cfg.sim_frames = 500

    print(f"==================================================")
    print(f"   LDA Simulation Start (Multi-BS Architecture)")
    print(f"==================================================")
    print(f"Configuration Summary:")
    print(f"  - Base Stations (I): {cfg.I}")  # [新增] 打印基站数
    print(f"  - Users per BS (J):  {cfg.J}")  # [修改] 打印单基站用户数
    print(f"  - Time Frames:       {cfg.sim_frames}")
    print(f"  - DNN Hidden Dim:    {cfg.hidden_dim}")
    print(f"  - Learning Rate:     {cfg.lr}")
    print(f"  - Train Interval:    Every {cfg.train_interval} frames")
    print(f"==================================================\n")

    # 实例化 Solver (内部会自动加载 config 中的 DNN 参数)
    solver = LDASolver(cfg)

    # --- 2. 主仿真循环 (Time-Slotted Execution) ---
    print(">>> 开始执行时隙仿真...")

    for t in range(cfg.sim_frames):
        # A. 流量生成 (Traffic Generation)
        # [核心修改]: 模拟随机任务到达 -> 维度扩展为 (I, J)
        # L_t shape: (I, J) 单位: bits
        noise = np.random.normal(0, cfg.L_std, (cfg.I, cfg.J))
        L_t = np.maximum(0, cfg.L_mean + noise)

        # B. 算法步进 (Solver Step)
        # 这一步包含了：环境感知 -> 3个DNN并行推理 -> 候选对齐评估 -> 资源优化 -> 训练(反向传播) -> 状态更新
        sol = solver.step(L_t)

        # C. 进度监控 (Logging)
        if t % 10 == 0:  # 提高打印频率以便观察
            info = sol['debug']

            # 解包数据
            n_loc, n_bs, n_sat = info['dist']
            util_bs, util_sat = info['util']
            arr, srv = info['flow']
            net_flow = info['q_trend']

            # 取全网平均任务积压
            q_mb = np.mean(solver.Q_total) / 1e6

            # [核心修改]: E_BS 现在是长度为 I 的数组，取最大违约情况监控系统瓶颈
            max_e_virt = np.max(solver.E_BS)

            # 动态颜色/符号标记
            trend_symbol = "🟢" if net_flow > 0 else "🔴"  # 红灯代表积压在变多

            print(f"[Fr {t:03d}] "
                  f"Q:{q_mb:6.1f}Mb {trend_symbol} | "
                  f"In/Out: {arr:4.1f}/{srv:4.1f} | "
                  f"Dec(L/B/S): {n_loc}/{n_bs}/{n_sat} | "
                  f"Util(B/S): {util_bs:.1%}/{util_sat:.1%}")

            # 极度危险预警
            if t > 50 and q_mb > 500:  # 假设 500Mb 是个阈值
                print("   !!! WARNING: Queue Explosion Detected !!!")
                # [核心修改]: 打印最大能量队列情况
                print(f"   Diagnosis: Max E_virt={max_e_virt:.1f} (Power Limited?), Prob_Net={info['prob_mean']:.2f}")
                # 可以在这里 break 停止仿真

    print("\n>>> 仿真结束。正在生成分析图表...")

    # --- 3. 结果可视化 ---
    plot_results(solver, cfg)


def smooth_curve(data, window_size=50):
    """
    滑动平均滤波辅助函数
    用于平滑强化学习由于探索（Exploration）带来的高频震荡，凸显长期的收敛趋势
    """
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_results(solver, cfg):
    """
    绘制关键性能指标曲线 (升级版：包含 Reward 收敛曲线)
    """
    history = solver.history
    frames = range(len(history['Q_total']))

    # 创建一个 3x2 的图表布局 (宽度稍宽，高度拉长以容纳 5 张图)
    plt.figure(figsize=(16, 12))
    plt.suptitle(f"LDA Distributed System Simulation (Frames: {len(frames)})", fontsize=16, fontweight='bold', y=0.98)

    # ==========================================
    # --- 子图 1: 强化学习 Reward 曲线 (新增：最核心的收敛指标) ---
    # ==========================================
    plt.subplot(3, 2, 1)
    if 'Reward' in history and len(history['Reward']) > 0:
        raw_reward = history['Reward']
        smoothed_reward = smooth_curve(raw_reward, window_size=50)

        # 画出原始的浅色震荡背景
        plt.plot(frames, raw_reward, color='lightgray', alpha=0.6, label='Raw Reward ($-G_1(t)$)')

        # 画出平滑后的红色主干收敛线
        if len(raw_reward) >= 50:
            # 校准 x 轴偏移，使平滑曲线与原曲线对齐
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
    # --- 子图 2: DNN 训练损失 Loss [学习曲线] ---
    # ==========================================
    plt.subplot(3, 2, 2)
    if len(history['Loss']) > 0:
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
    # --- 子图 3: 任务积压 Q(t) [稳定性指标] ---
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
    # --- 子图 4: 虚拟能量队列 E(t) [约束指标] ---
    # ==========================================
    plt.subplot(3, 2, 4)
    plt.plot(frames, history['E_virt_bs'], linewidth=2, color='#ff7f0e', label='Avg BS Energy Queue ($E_{virt}$)')

    plt.title("Virtual Energy Queues (Power Constraint)")
    plt.ylabel("Energy Deficit Level")
    plt.xlabel("Time Frame")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    # ==========================================
    # --- 子图 5: 系统性能 Cost (PAoI) [优化目标] ---
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
    # --- 子图 6: 传输速率动态范围 (Max & Min Rates) ---
    # ==========================================
    plt.subplot(3, 2, 6)
    if 'R_bs_max' in history and len(history['R_bs_max']) > 0:
        # 截取与 frames 等长的数据，防止 off-by-one 报错
        r_bs_max = np.array(history['R_bs_max'][:len(frames)]) / 1e6  # 转为 Mbps
        r_bs_min = np.array(history['R_bs_min'][:len(frames)]) / 1e6
        r_sat_max = np.array(history['R_sat_max'][:len(frames)]) / 1e6
        r_sat_min = np.array(history['R_sat_min'][:len(frames)]) / 1e6

        # 绘制基站速率 (绿色系)
        plt.plot(frames, r_bs_max, color='#2ca02c', linestyle='-', linewidth=1.5, label='BS Max')
        plt.plot(frames, r_bs_min, color='#98df8a', linestyle='--', linewidth=1.5, label='BS Min')
        # 填充 BS 速率的波动区间
        plt.fill_between(frames, r_bs_min, r_bs_max, color='#2ca02c', alpha=0.15)

        # 绘制卫星速率 (橙色系)
        plt.plot(frames, r_sat_max, color='#ff7f0e', linestyle='-', linewidth=1.5, label='Sat Max')
        plt.plot(frames, r_sat_min, color='#ffbb78', linestyle='--', linewidth=1.5, label='Sat Min')
        # 填充卫星速率的波动区间
        plt.fill_between(frames, r_sat_min, r_sat_max, color='#ff7f0e', alpha=0.15)

    plt.title("Channel Transmission Rates (Extrema)")
    plt.ylabel("Data Rate (Mbps)")
    plt.xlabel("Time Frame")

    # 将图例放在最佳位置，两列排列，避免遮挡曲线
    plt.legend(loc='best', ncol=2, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 留出 suptitle 的空间

    # 保存图片
    save_path = 'simulation_results.png'
    plt.savefig(save_path, dpi=150)
    print(f">>> 完美！图表已生成并保存至: {os.path.abspath(save_path)}")
    plt.show()


if __name__ == "__main__":
    run_main_simulation()