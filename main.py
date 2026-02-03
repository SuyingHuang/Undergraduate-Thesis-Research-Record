import numpy as np
import matplotlib.pyplot as plt
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
    set_seed(2025)  # 设定种子
    cfg = SystemConfig()

    # [调试建议] 初次运行时，可以适当减少帧数以快速看结果
    # cfg.sim_frames = 500

    print(f"==================================================")
    print(f"   LDA Simulation Start ")
    print(f"==================================================")
    print(f"Configuration Summary:")
    print(f"  - Users (J):      {cfg.J}")
    print(f"  - Time Frames:    {cfg.sim_frames}")
    print(f"  - DNN Hidden Dim: {cfg.hidden_dim}")
    print(f"  - Learning Rate:  {cfg.lr}")
    print(f"  - Train Interval: Every {cfg.train_interval} frames")
    print(f"==================================================\n")

    # 实例化 Solver (内部会自动加载 config 中的 DNN 参数)
    solver = LDASolver(cfg)

    # --- 2. 主仿真循环 (Time-Slotted Execution) ---
    print(">>> 开始执行时隙仿真...")

    for t in range(cfg.sim_frames):
        # A. 流量生成 (Traffic Generation)
        # 模拟随机任务到达：正态分布截断 (Mean, Std) -> 保证非负
        # L_t shape: (J,) 单位: bits
        noise = np.random.normal(0, cfg.L_std, cfg.J)
        L_t = np.maximum(0, cfg.L_mean + noise)

        # B. 算法步进 (Solver Step)
        # 这一步包含了：环境感知 -> DNN推理 -> TCOPQ -> 资源优化 -> 训练(反向传播) -> 状态更新
        sol = solver.step(L_t)

        # C. 进度监控 (Logging) - [修改版]
        if t % 10 == 0:  # 提高打印频率以便观察
            info = sol['debug']

            # 解包数据
            n_loc, n_bs, n_sat = info['dist']
            util_bs, util_sat = info['util']
            arr, srv = info['flow']
            net_flow = info['q_trend']
            q_mb = np.mean(solver.Q) / 1e6
            e_virt = solver.E_BS

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
                print(f"   Diagnosis: E_virt={e_virt:.1f} (Power Limited?), Prob_Net={info['prob_mean']:.2f}")
                # 可以在这里 break 停止仿真

    print("\n>>> 仿真结束。正在生成分析图表...")

    # --- 3. 结果可视化 ---
    plot_results(solver, cfg)


def plot_results(solver, cfg):
    """
    绘制关键性能指标曲线
    """
    history = solver.history
    frames = range(len(history['Q']))

    # 创建一个 2x2 的图表布局
    plt.figure(figsize=(14, 10))

    # --- 子图 1: 任务积压 Q(t) [稳定性指标] ---
    plt.subplot(2, 2, 1)
    plt.plot(frames, np.array(history['Q']) / 1e6, linewidth=1.5, color='#1f77b4')
    plt.title("Avg Task Queue Backlog Q(t)")
    plt.ylabel("Queue Size (Mb)")
    plt.xlabel("Time Frame")
    plt.grid(True, alpha=0.3)
    # 说明: 只要它在一个水平线附近震荡，不无限上升，系统就是稳定的。

    # --- [修正点 2] 子图 2: 虚拟能量队列 E(t) [约束指标] ---
    plt.subplot(2, 2, 2)
    # 画 BS 能量
    plt.plot(frames, history['E_virt_bs'], linewidth=1.5, color='#ff7f0e', label='BS Energy Queue')
    # 画 Sat 能量
    plt.plot(frames, history['E_virt_sat'], linewidth=1.5, color='#9467bd', label='Sat Energy Queue', linestyle='--')
    plt.title("Virtual Energy Queues E(t)")
    plt.ylabel("Virtual Level")
    plt.xlabel("Time Frame")
    plt.legend()  # 显示图例区分 BS 和 Sat
    plt.grid(True, alpha=0.3)

    # --- 子图 3: DNN 训练损失 Loss [学习曲线] ---
    plt.subplot(2, 2, 3)
    if len(history['Loss']) > 0:
        # Loss 是稀疏记录的 (每 train_interval 一次)，需要对齐 x 轴
        train_steps = np.arange(len(history['Loss'])) * cfg.train_interval
        plt.plot(train_steps, history['Loss'], linewidth=1.5, color='#d62728', marker='.')
        plt.title("DNN Training Loss (Focal Loss)")
        plt.ylabel("Loss Value")
        plt.xlabel("Time Frame")
        plt.yscale('log')  # Loss 可能会跨度很大，用对数坐标看下降趋势
    else:
        plt.text(0.5, 0.5, "No Training Data", ha='center')
    plt.grid(True, alpha=0.3)
    # 说明: 曲线应呈现震荡下降趋势，表明 DNN 正在从 Solver 的行为中学习。

    # --- 子图 4: 系统性能 Cost (PAoI) [优化目标] ---
    plt.subplot(2, 2, 4)
    # 对 Cost 做一个简单的移动平均，让曲线更平滑易读
    window_size = 20
    cost_data = np.array(history['Cost'])
    if len(cost_data) >= window_size:
        cost_smooth = np.convolve(cost_data, np.ones(window_size) / window_size, mode='valid')
        plt.plot(frames[window_size - 1:], cost_smooth, linewidth=1.5, color='#2ca02c')
    else:
        plt.plot(frames, cost_data, color='#2ca02c', alpha=0.5)

    plt.title(f"System Cost (Avg PAoI) - Moving Avg {window_size}")
    plt.ylabel("Time (s)")
    plt.xlabel("Time Frame")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    save_path = 'simulation_results.png'
    plt.savefig(save_path, dpi=120)
    print(f">>> 图表已保存至: {os.path.abspath(save_path)}")
    plt.show()


if __name__ == "__main__":
    run_main_simulation()