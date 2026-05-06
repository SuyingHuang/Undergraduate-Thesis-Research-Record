import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import multiprocessing
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from config import SystemConfig
from core.agents.lda_agent import LDAAgent
from core.agents.baselines import ACAgent, COBAgent, MTDAgent
from main import run_simulation

E_MAX_BS = 160.0
E_ANOMALY_THRESHOLD = E_MAX_BS * 10


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_metrics(history):
    start_idx = int(len(history['Cost']) * 0.2)
    avg_paoi = np.mean(history['Cost'][start_idx:])
    avg_q = np.mean(history['Q_total'][start_idx:])
    avg_e_bs = np.mean(history.get('E_actual_bs', history['E_virt_bs'])[start_idx:])
    avg_e_sat = np.mean(history['E_virt_sat'][start_idx:])
    return avg_paoi, avg_e_bs, avg_e_sat, avg_q


def _worker_sweep(args):
    """
    每个任务独立运行，输出重定向到专属日志文件。
    返回: (param_val, algo_name, seed, paoi, e_bs, e_sat, q, failed,
            max_e_queue, final_e_queue, first_anomaly_frame, log_path)
    """
    cfg, param_name, param_val, algo_name, AgentClass, sim_frames, seed, log_path = args

    # 限制 PyTorch 内部线程数，避免多进程互相抢占 CPU
    torch.set_num_threads(1)

    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write(f"任务: {algo_name}  {param_name}={param_val}  seed={seed}\n")
        log_f.write(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"{'='*60}\n")
        log_f.flush()

        old_stdout = sys.stdout
        sys.stdout = log_f
        try:
            set_seed(seed)
            test_cfg = copy.deepcopy(cfg)
            test_cfg.sim_frames = sim_frames

            agent_kwargs = None
            if hasattr(test_cfg, param_name):
                setattr(test_cfg, param_name, param_val)
            else:
                agent_kwargs = {param_name: param_val}

            env, _ = run_simulation(test_cfg, AgentClass,
                                    algorithm_name=f"{algo_name} ({param_name}={param_val}, seed={seed})",
                                    agent_kwargs=agent_kwargs)
        except (ValueError, RuntimeError) as e:
            sys.stdout = old_stdout
            log_f.write(f"\n[FAILED] {e}\n")
            return (param_val, algo_name, seed, np.nan, np.nan, np.nan, np.nan, True,
                    np.nan, np.nan, -1, log_path)
        finally:
            sys.stdout = old_stdout

        # 恢复 stdout 后提取指标和能量数据
        paoi, e_bs, e_sat, q = extract_metrics(env.history)

        e_queue_traj = np.array(env.history.get('E_queue_bs_max', [0.0]))
        max_e_queue = float(np.max(e_queue_traj)) if len(e_queue_traj) > 0 else 0.0
        final_e_queue = float(e_queue_traj[-1]) if len(e_queue_traj) > 0 else 0.0
        anomaly_frames = np.where(e_queue_traj > E_ANOMALY_THRESHOLD)[0]
        first_anomaly_frame = int(anomaly_frames[0]) if len(anomaly_frames) > 0 else -1

        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Max E_queue_BS: {max_e_queue:.1f}\n")
        log_f.write(f"Final E_queue_BS: {final_e_queue:.1f}\n")
        if first_anomaly_frame >= 0:
            log_f.write(f"首次越界帧 (>{E_ANOMALY_THRESHOLD:.0f}): {first_anomaly_frame}\n")
        log_f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        return (param_val, algo_name, seed, paoi, e_bs, e_sat, q, False,
                max_e_queue, final_e_queue, first_anomaly_frame, log_path)


def _print_anomaly_report(anomalies, sweep_name, param_name):
    if not anomalies:
        print(f"\n  [能量检查] 未发现异常 (所有 Max E_BS_queue < {E_ANOMALY_THRESHOLD:.0f})")
        return

    print(f"\n{'!' * 70}")
    print(f"  ⚠️  能量异常报告: {sweep_name}")
    print(f"  ⚠️  阈值 = {E_ANOMALY_THRESHOLD:.0f} (E_max_BS = {E_MAX_BS})")
    print(f"  ⚠️  共 {len(anomalies)} 个异常任务:")
    print(f"{'!' * 70}")
    header = (f"  {'算法':<6s} | {param_name:<10s} | {'种子':>5s} | "
              f"{'Max E_q':>12s} | {'Final E_q':>12s} | {'首次越界':>10s} | 日志")
    print(header)
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*20}")

    algo_counts = defaultdict(int)
    for a in anomalies:
        algo_counts[a['algo']] += 1

    for a in anomalies:
        first_str = f"{a['first_frame']}" if a['first_frame'] >= 0 else "N/A"
        log_name = os.path.basename(a['log_path'])
        print(f"  {a['algo']:<6s} | {str(a['param']):>10s} | {a['seed']:5d} | "
              f"{a['max_e']:12.1f} | {a['final_e']:12.1f} | {first_str:>10s} | {log_name}")

    print(f"\n  各算法异常次数: {dict(algo_counts)}")
    print(f"  日志目录: {os.path.dirname(anomalies[0]['log_path'])}")
    print(f"{'!' * 70}\n")


def run_experiment_sweep(sweep_name, param_name, param_values, algos, cfg,
                         n_workers=None, seeds=None, sim_frames=4096):
    if seeds is None:
        seeds = getattr(cfg, 'seeds', [42, 123, 456, 789, 1000])

    n_params = len(param_values)
    n_algos = len(algos)
    n_seeds = len(seeds)
    n_tasks = n_params * n_algos * n_seeds

    if n_workers is None:
        env_workers = os.environ.get('LDA_WORKERS')
        if env_workers:
            n_workers = int(env_workers)
        else:
            n_workers = min(max(1, multiprocessing.cpu_count() // 2+4), n_tasks)

    # 为本次实验组创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'sweep')
    exp_slug = sweep_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
    log_dir = os.path.join(log_base, f"{timestamp}_{exp_slug}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"  实验组: {sweep_name}")
    print(f"  参数: {param_name} in {param_values}")
    print(f"  算法: {[name for name, _ in algos]}")
    print(f"  种子: {seeds}")
    print(f"  总任务数: {n_tasks} | 并行: {n_workers}")
    print(f"  日志目录: {log_dir}")
    print(f"{'=' * 50}")

    # 每个任务带独立日志路径
    tasks = []
    for val in param_values:
        for algo_name, AgentClass in algos:
            for seed in seeds:
                log_name = f"{algo_name}_{param_name}{val}_s{seed}.log"
                log_path = os.path.join(log_dir, log_name)
                tasks.append((cfg, param_name, val, algo_name, AgentClass, sim_frames, seed, log_path))

    if n_workers <= 1:
        raw_results = []
        for task in tasks:
            raw_results.append(_worker_sweep(task))
    else:
        raw_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_worker_sweep, task) for task in tasks]
            completed = 0
            anomaly_count = 0
            for future in as_completed(futures):
                result = future.result()
                raw_results.append(result)
                completed += 1

                param_val, algo_name, seed, _, _, _, _, failed, max_e_q, _, first_frame, _ = result
                status = "❌" if failed else "✅"
                flag = ""
                if not failed and not np.isnan(max_e_q) and max_e_q > E_ANOMALY_THRESHOLD:
                    anomaly_count += 1
                    flag = f" ⚠️ E_QUEUE={max_e_q:.0f} @Fr{first_frame}"
                print(f"  [{completed}/{n_tasks}] {algo_name} "
                      f"({param_name}={param_val}, s={seed}) {status}{flag}")

            if anomaly_count > 0:
                print(f"  ⚠️  本组已发现 {anomaly_count} 个能量异常 (详见日志目录)")

    # 解析结果并收集异常
    anomalies = []
    seed_metrics = defaultdict(list)

    for r in raw_results:
        (param_val, algo_name, seed, paoi, e_bs, e_sat, q,
         failed, max_e_queue, final_e_queue, first_frame, log_path) = r
        seed_metrics[(param_val, algo_name)].append((paoi, e_bs, e_sat, q, failed))

        if not failed and not np.isnan(max_e_queue) and max_e_queue > E_ANOMALY_THRESHOLD:
            anomalies.append({
                'algo': algo_name, 'param': param_val, 'seed': seed,
                'max_e': max_e_queue, 'final_e': final_e_queue,
                'first_frame': first_frame, 'log_path': log_path,
            })

    # 聚合多种子结果
    result_map = {}
    for (param_val, algo_name), metrics_list in seed_metrics.items():
        failed_flags = [m[4] for m in metrics_list]
        valid = [m for m in metrics_list if not m[4]]
        if len(valid) == 0:
            avg = (np.nan, np.nan, np.nan, np.nan, True)
        else:
            avg = (np.mean([m[0] for m in valid]), np.mean([m[1] for m in valid]),
                   np.mean([m[2] for m in valid]), np.mean([m[3] for m in valid]), False)
        result_map[(param_val, algo_name)] = avg

    # 按顺序填充结果
    results = {algo_name: {'PAoI': [], 'E_BS': [], 'E_LEO': [], 'Q': [], 'failed': []}
               for algo_name, _ in algos}
    for algo_name, _ in algos:
        for val in param_values:
            m = result_map.get((val, algo_name), (np.nan, np.nan, np.nan, np.nan, True))
            results[algo_name]['PAoI'].append(m[0])
            results[algo_name]['E_BS'].append(m[1])
            results[algo_name]['E_LEO'].append(m[2])
            results[algo_name]['Q'].append(m[3])
            results[algo_name]['failed'].append(m[4])

    _print_anomaly_report(anomalies, sweep_name, param_name)
    return results


def plot_sweep_results(param_values, results, x_label, title_prefix, filename,
                       metrics_to_plot=['PAoI', 'E_BS', 'E_LEO']):
    num_metrics = len(metrics_to_plot)
    fig, axs = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    if num_metrics == 1: axs = [axs]

    ylabels = {
        'PAoI': 'Average PAoI [s]', 'E_BS': 'Average BS Energy [J]',
        'E_LEO': 'Average LEO Energy [J]', 'Q': 'Average Data Queue [Mbit]'
    }
    markers = {'LDA': 'x', 'AC': '^', 'COB': 'o', 'MTD': 's'}

    for i, metric in enumerate(metrics_to_plot):
        ax = axs[i]
        for algo_name, data in results.items():
            y_data = np.array(data[metric], dtype=float)
            failed = np.array(data['failed'])
            valid_mask = ~failed & ~np.isnan(y_data)
            failed_mask = failed | np.isnan(y_data)
            if np.any(valid_mask):
                ax.plot(np.array(param_values)[valid_mask], y_data[valid_mask],
                        marker=markers.get(algo_name, 'o'), label=algo_name,
                        color=None, linestyle='-')
            if np.any(failed_mask):
                ax.scatter(np.array(param_values)[failed_mask], [0] * np.sum(failed_mask),
                           marker='X', color='red', s=150, label=f'{algo_name} (Failed)', zorder=10)
        if "K" in x_label: ax.set_xscale('log')
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabels[metric])
        ax.grid(True, linestyle='--')
        ax.legend()

    fig.text(0.5, 0.02, '[X] = Physical constraint violation, simulation failed',
             ha='center', fontsize=10, style='italic', color='red')
    plt.suptitle(f"{title_prefix}")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if not os.path.exists("results"): os.makedirs("results")
    save_path = f"results/{filename}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n  【生成成功】 {title_prefix} 实验组图表已保存至: {save_path}")
    plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    base_cfg = SystemConfig()
    sweep_seeds = [42, 123, 456, 789]

    print(f"\n>>> 可用 CPU 核心数: {multiprocessing.cpu_count()}")
    print(f">>> 每种参数组合运行 {len(sweep_seeds)} 个种子取均值")
    print(f">>> 每个任务有独立日志文件 (logs/sweep/)")
    print(f">>> 能量异常阈值: {E_ANOMALY_THRESHOLD:.0f} (E_max_BS={E_MAX_BS})")

    all_four_algos = [("LDA", LDAAgent), ("AC", ACAgent), ("COB", COBAgent), ("MTD", MTDAgent)]

    # Exp 1: J
    j_values = [4, 6, 8, 10, 12, 14]
    res_exp1 = run_experiment_sweep("Exp1_J", "J", j_values, all_four_algos, base_cfg, seeds=sweep_seeds)
    plot_sweep_results(j_values, res_exp1, "Number of UEs", "Impact of UE Quantity", "exp1_J_sweep")

    # Exp 2: L_mean
    l_values = [6e6, 9e6, 12e6, 15e6,18e6]
    l_labels = [6, 9, 12, 15,18]
    res_exp2 = run_experiment_sweep("Exp2_L", "L_mean", l_values, all_four_algos, base_cfg, seeds=sweep_seeds)
    plot_sweep_results(l_labels, res_exp2, "Number of tasks [Mbit]", "Impact of Task Volume", "exp2_L_sweep")

    # Exp 3: f_max_UE
    f_values = [1e8, 2e8, 3e8, 5e8, 10e8]
    f_labels = [1, 2, 3, 5, 10]
    res_exp3 = run_experiment_sweep("Exp3_fUE", "f_max_UE", f_values, all_four_algos, base_cfg, seeds=sweep_seeds)
    plot_sweep_results(f_labels, res_exp3, r"Computing capability of users ($\times 10^8$ CPU cycles/s)",
                       "Impact of Local Computing", "exp3_FUE_sweep")

    # Exp 4: K_p
    k_values = [0.1, 1, 10, 100, 1000]
    four_algos_with_ac = [("LDA", LDAAgent), ("AC", ACAgent), ("COB", COBAgent), ("MTD", MTDAgent)]
    res_exp4 = run_experiment_sweep("Exp4_K", "K_p", k_values, four_algos_with_ac, base_cfg, seeds=sweep_seeds)
    plot_sweep_results(k_values, res_exp4, "Lyapunov control parameter K", "Impact of Parameter K", "exp4_K_sweep",
                       metrics_to_plot=['PAoI', 'Q', 'E_BS'])

    print("  所有实验已执行完毕！")
