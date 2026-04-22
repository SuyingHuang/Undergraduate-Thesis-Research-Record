import os
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch

from config import SystemConfig
from core.agents.lda_agent import LDAAgent
from core.agents.baselines import ACAgent, COBAgent, MTDAgent
from main import run_simulation


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_metrics(history):
    """从 history 字典中提取我们需要的平均指标"""
    # 丢弃前20%的帧，等待系统度过冷启动，计算稳态平均值
    start_idx = int(len(history['Cost']) * 0.2)

    avg_paoi = np.mean(history['Cost'][start_idx:])
    avg_q = np.mean(history['Q_total'][start_idx:])

    avg_e_bs = np.mean(history.get('E_actual_bs', history['E_virt_bs'])[start_idx:])
    avg_e_sat = np.mean(history['E_virt_sat'][start_idx:])

    return avg_paoi, avg_e_bs, avg_e_sat, avg_q


def run_experiment_sweep(sweep_name, param_name, param_values, algos, cfg):
    print(f"\n{'=' * 50}\n🚀 启动实验组: {sweep_name}\n{'=' * 50}")

    results = {algo_name: {'PAoI': [], 'E_BS': [], 'E_LEO': [], 'Q': []} for algo_name, _ in algos}

    for val in param_values:
        print(f"\n--- 测试参数 {param_name} = {val} ---")

        for algo_name, AgentClass in algos:
            test_cfg = copy.deepcopy(cfg)
            setattr(test_cfg, param_name, val)

            # 仿真帧数（出图建议设为 2000 左右，测试时可以设 500）
            test_cfg.sim_frames = 1000

            # 固定随机种子，保证公平对比
            set_seed(42)

            env, _ = run_simulation(test_cfg, AgentClass, algorithm_name=f"{algo_name} ({param_name}={val})")

            paoi, e_bs, e_sat, q = extract_metrics(env.history)
            results[algo_name]['PAoI'].append(paoi)
            results[algo_name]['E_BS'].append(e_bs)
            results[algo_name]['E_LEO'].append(e_sat)
            results[algo_name]['Q'].append(q)

    return results


def plot_sweep_results(param_values, results, x_label, title_prefix, filename,
                       metrics_to_plot=['PAoI', 'E_BS', 'E_LEO']):
    """自动化绘制对比折线图，并在保存后立即释放内存出图"""
    num_metrics = len(metrics_to_plot)
    fig, axs = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    if num_metrics == 1: axs = [axs]

    ylabels = {
        'PAoI': 'Average PAoI [s]',
        'E_BS': 'Average BS Energy [J]',
        'E_LEO': 'Average LEO Energy [J]',
        'Q': 'Average Data Queue [Mbit]'
    }
    markers = {'LDA': 'x', 'AC': '^', 'COB': 'o', 'MTD': 's'}

    for i, metric in enumerate(metrics_to_plot):
        ax = axs[i]
        for algo_name, data in results.items():
            y_data = np.array(data[metric])
            if metric == 'Q': y_data = y_data / 1e6

            ax.plot(param_values, y_data, marker=markers.get(algo_name, 'o'), label=algo_name)

        if "K" in x_label: ax.set_xscale('log')

        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabels[metric])
        ax.grid(True, linestyle='--')
        ax.legend()

    plt.suptitle(f"{title_prefix}")
    plt.tight_layout()

    if not os.path.exists("results"): os.makedirs("results")
    save_path = f"results/{filename}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')

    print(f"\n✅✅✅ 【生成成功】 {title_prefix} 实验组图表已保存至: {save_path} ✅✅✅\n")

    # 🚨【关键修改】：清空画布并关闭 Figure，强制刷新写入硬盘并释放内存
    plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    base_cfg = SystemConfig()

    all_four_algos = [("LDA", LDAAgent), ("AC", ACAgent), ("COB", COBAgent), ("MTD", MTDAgent)]
    three_algos = [("LDA", LDAAgent), ("COB", COBAgent), ("MTD", MTDAgent)]

    # ========================================================
    # 实验 1: 用户数 J 从 6 到 14
    # ========================================================
    j_values = [6, 8, 10, 12, 14]
    res_exp1 = run_experiment_sweep("Exp 1: Number of UEs (J)", "J", j_values, all_four_algos, base_cfg)
    plot_sweep_results(j_values, res_exp1, "Number of UEs", "Impact of UE Quantity", "exp1_J_sweep")

    # ========================================================
    # 实验 2: 任务均值 L_mean 从 5M 到 25M
    # ========================================================
    l_values = [6e6, 9e6, 12e6, 15e6, 18e6]
    l_labels = [6, 9, 12, 15, 18]
    res_exp2 = run_experiment_sweep("Exp 2: Task Mean (L_mean)", "L_mean", l_values, all_four_algos, base_cfg)
    plot_sweep_results(l_labels, res_exp2, "Number of tasks [Mbit]", "Impact of Task Volume", "exp2_L_sweep")

    # ========================================================
    # 实验 3: 本地计算能力 f_max_UE
    # ========================================================
    f_values = [1e8, 2e8, 3e8, 5e8, 10e8]
    f_labels = [1, 2, 3, 5, 10]
    res_exp3 = run_experiment_sweep("Exp 3: UE Compute Capacity", "f_max_UE", f_values, all_four_algos, base_cfg)
    plot_sweep_results(f_labels, res_exp3, r"Computing capability of users ($\times 10^8$ CPU cycles/s)",
                       "Impact of Local Computing", "exp3_FUE_sweep")

    # ========================================================
    # 实验 4: 惩罚系数 K (仅 LDA, COB, MTD)
    # ========================================================
    k_values = [30, 300, 3000, 30000, 300000]
    res_exp4 = run_experiment_sweep("Exp 4: Penalty Parameter K", "K_p", k_values, three_algos, base_cfg)
    plot_sweep_results(k_values, res_exp4, "Lyapunov control parameter K", "Impact of Parameter K", "exp4_K_sweep",
                       metrics_to_plot=['Q', 'E_BS', 'E_LEO'])

    print("🎉 所有实验已执行完毕！")