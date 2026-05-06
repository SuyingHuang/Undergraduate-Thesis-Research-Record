# utils/plotter.py
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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


def plot_results_with_ci(all_aggregated, cfg, save_path='results/multi_seed/comparison_with_ci.png'):
    """
    绘制带置信区间的多算法性能对比图

    :param all_aggregated: dict of {algo_name: aggregated_result}
                          aggregated_result = {metric_name: {'mean': array, 'std': array}}
    :param cfg: SystemConfig 实例
    :param save_path: 图片保存路径
    """
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(3, 2, figsize=(14, 15))

    metrics = [
        ('Cost', 'Average PAoI [s]', axs[0, 0]),
        ('Q_total', 'Average Data Queue Length [Mbit]', axs[0, 1]),
        ('E_virt_bs', 'Average BS Energy [J]', axs[1, 0]),
        ('E_virt_sat', 'Average LEO Energy [J]', axs[1, 1]),
        ('R_sat_max', 'Enhanced UE->LEO Rate [Mbps]', axs[2, 0]),
    ]

    colors = {'LDA': '#1f77b4', 'AC': '#ff7f0e', 'COB': '#2ca02c', 'MTD': '#d62728'}
    markers = {'LDA': 'x', 'AC': '^', 'COB': 'o', 'MTD': 's'}
    markevery_ratio = 0.1

    x_frames = None

    for metric_key, ylabel, ax in metrics:
        for algo_name, agg_result in all_aggregated.items():
            if metric_key not in agg_result:
                continue

            mean_data = agg_result[metric_key]['mean']
            std_data = agg_result[metric_key]['std']

            if x_frames is None:
                x_frames = np.arange(len(mean_data))

            if 'R_sat' in metric_key or 'R_bs' in metric_key:
                mean_data = mean_data / 1e6
                std_data = std_data / 1e6

            smoothed_mean = smooth_curve(mean_data, window_size=50)
            smoothed_std = smooth_curve(std_data, window_size=50)

            n_smooth = len(smoothed_mean)
            x_smooth = np.arange(n_smooth)

            mark_step = max(1, int(n_smooth * markevery_ratio))

            ax.plot(
                x_smooth, smoothed_mean,
                label=algo_name,
                color=colors.get(algo_name, '#000000'),
                marker=markers.get(algo_name, ''),
                markevery=mark_step,
                linewidth=1.5
            )

            ax.fill_between(
                x_smooth,
                smoothed_mean - smoothed_std,
                smoothed_mean + smoothed_std,
                color=colors.get(algo_name, '#000000'),
                alpha=0.2
            )

        ax.set_xlabel('Time Frames')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    ax_loss = axs[2, 1]

    if 'LDA' in all_aggregated and 'Loss' in all_aggregated['LDA']:
        loss_data = all_aggregated['LDA']['Loss']
        if 'mean' in loss_data and 'std' in loss_data:
            mean_loss = np.array(loss_data['mean'])
            std_loss = np.array(loss_data['std'])

            if mean_loss.ndim == 2 and mean_loss.shape[1] == 2:
                mean_loss = mean_loss[:, 1]
                std_loss = std_loss[:, 1] if std_loss.ndim == 2 else std_loss

            valid_mask = np.isfinite(mean_loss) & np.isfinite(std_loss)
            if np.any(valid_mask):
                mean_loss = mean_loss[valid_mask]
                std_loss = std_loss[valid_mask]

            if len(mean_loss) >= 50:
                smoothed_mean = smooth_curve(mean_loss, window_size=50)
                smoothed_std = smooth_curve(std_loss, window_size=50)
            else:
                smoothed_mean = mean_loss
                smoothed_std = std_loss

            n_smooth = len(smoothed_mean)
            x_smooth = np.arange(n_smooth)

            ax_loss.plot(
                x_smooth, smoothed_mean,
                color='#1f77b4', linewidth=1.5, label='Mean Loss'
            )
            ax_loss.fill_between(
                x_smooth,
                np.clip(smoothed_mean - smoothed_std, 0, None),
                smoothed_mean + smoothed_std,
                color='#1f77b4', alpha=0.2, label='±1 Std'
            )

    ax_loss.set_xlabel('Time Frames')
    ax_loss.set_ylabel('DNN Loss')
    ax_loss.set_title('LDA Training Loss (Multi-seed Mean ± Std)')
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    ax_loss.legend()
    ax_loss.set_yscale('log')

    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f">>> 带置信区间的对比图已保存: {os.path.abspath(save_path)}")
    plt.show()


def plot_results_comparison(all_histories, save_path="results/final_comparison_plot.png"):
    """
    绘制四个算法的对比图，输出一张 3x2 的大图
    包含: PAoI, 数据队列长度, BS 能耗, LEO 能耗, 以及增强后的 UE->LEO 传输速率
    """
    # 设置全局字体大小，方便放在论文里阅读
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(3, 2, figsize=(14, 15))

    # 定义你需要对比的指标。格式: (history字典里的key, Y轴标签, 子图位置)
    # 注意：这里的 'PAoI', 'Q_length' 等需要替换为你代码中实际记录在 env.history 里的 key 名字
    # 这里的第一个元素必须和你 env.py 中 history 字典的 Key 严格一致
    metrics = [
        ('Cost', 'Average PAoI [s]', axs[0, 0]),
        ('Q_total', 'Average Data Queue Length [Mbit]', axs[0, 1]),
        ('E_virt_bs', 'Average BS Energy [J]', axs[1, 0]),
        ('E_virt_sat', 'Average LEO Energy [J]', axs[1, 1]),
        ('R_sat_max', 'Enhanced UE->LEO Rate [Mbps]', axs[2, 0]),
    ]

    # 定义论文常用的颜色和标记符号
    colors = {'LDA': '#1f77b4', 'AC': '#ff7f0e', 'COB': '#2ca02c', 'MTD': '#d62728'}
    markers = {'LDA': 'x', 'AC': '^', 'COB': 'o', 'MTD': 's'}

    # 为了防止 marker 密集导致看不清，设置 marker 的采样间隔
    markevery_ratio = 0.1

    for metric_key, ylabel, ax in metrics:
        for algo_name, history in all_histories.items():
            # 兼容性检查：确保指标存在于该算法的记录中
            if metric_key in history:
                raw_data = history[metric_key]
                # 速率指标需要从 bps 转换为 Mbps
                if 'R_sat' in metric_key or 'R_bs' in metric_key:
                    raw_data = np.array(raw_data) / 1e6
                # 使用滑动平均使曲线平滑 (论文里的指标通常也是窗口平均)
                smoothed_data = smooth_curve(raw_data, window_size=50)

                # 计算 marker 步长
                mark_step = max(1, int(len(smoothed_data) * markevery_ratio))

                ax.plot(
                    smoothed_data,
                    label=algo_name,
                    color=colors.get(algo_name, '#000000'),
                    marker=markers.get(algo_name, ''),
                    markevery=mark_step,
                    linewidth=1.5
                )

        ax.set_xlabel('Time Frames')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    # --- 子图 6 (axs[2,1]): LDA 三个 BS 的 DNN 损失曲线 ---
    ax_loss = axs[2, 1]
    bs_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for algo_name, history in all_histories.items():
        if 'Loss_per_BS' in history:
            for bs_idx, bs_loss_history in enumerate(history['Loss_per_BS']):
                if len(bs_loss_history) > 0:
                    frames_x = [item[0] for item in bs_loss_history]
                    loss_y = [item[1] for item in bs_loss_history]
                    smoothed_loss = smooth_curve(loss_y, window_size=50)
                    smoothed_frames = frames_x[:len(smoothed_loss)]
                    mark_step = max(1, int(len(smoothed_loss) * markevery_ratio))
                    ax_loss.plot(
                        smoothed_frames, smoothed_loss,
                        label=f'BS {bs_idx}',
                        color=bs_colors[bs_idx],
                        linewidth=1.5,
                        markevery=mark_step
                    )
    ax_loss.set_xlabel('Time Frames')
    ax_loss.set_ylabel('DNN Loss')
    ax_loss.set_title('LDA Per-BS Training Loss')
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    ax_loss.legend()
    ax_loss.set_yscale('log')

    plt.tight_layout()

    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"综合对比图已成功保存为: {save_path}")


    plt.show()