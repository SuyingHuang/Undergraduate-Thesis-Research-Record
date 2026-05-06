# run_multi_seed_experiment.py
"""
Multi-seed 实验框架
对每个算法在多个随机种子下独立重复实验，收集结果并绘制带置信区间的性能曲线

支持多进程并行 + 独立日志文件 + 能量异常检测
"""

import numpy as np
import random
import torch
import sys
import os
import pickle
import multiprocessing
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig
from core.env import SAGINEnvironment
from core.agents.lda_agent import LDAAgent
from core.agents.baselines import COBAgent, MTDAgent, ACAgent
from utils.plotter import plot_results_with_ci

E_MAX_BS = 160.0
E_ANOMALY_THRESHOLD = E_MAX_BS * 10


def set_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single_experiment(cfg, agent_class, seed, algorithm_name):
    """单次独立实验（单个种子）—— 输出到 stdout (由调用方重定向到日志文件)"""
    from main import run_simulation
    set_seed(seed)
    env, agent = run_simulation(cfg, agent_class, algorithm_name=algorithm_name)
    if hasattr(agent, 'loss_history'):
        env.history['Loss'] = agent.loss_history
    if hasattr(agent, 'loss_history_per_bs'):
        env.history['Loss_per_BS'] = agent.loss_history_per_bs
    return env.history


def _worker_flat(args):
    """
    Worker函数：运行单次实验，输出重定向到日志文件。
    返回: (algo_name, seed, history, max_e_queue, final_e_queue, first_anomaly_frame, log_path)
    """
    cfg, agent_class, seed, algo_name, log_path = args

    # 限制 PyTorch 内部线程数，避免多进程互相抢占 CPU
    torch.set_num_threads(1)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write(f"任务: {algo_name}  seed={seed}\n")
        log_f.write(f"启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"{'='*60}\n")
        log_f.flush()

        old_stdout = sys.stdout
        sys.stdout = log_f
        try:
            history = run_single_experiment(cfg, agent_class, seed, algo_name)
        finally:
            sys.stdout = old_stdout

        e_queue_traj = np.array(history.get('E_queue_bs_max', [0.0]))
        max_e_queue = float(np.max(e_queue_traj)) if len(e_queue_traj) > 0 else 0.0
        final_e_queue = float(e_queue_traj[-1]) if len(e_queue_traj) > 0 else 0.0
        anomaly_frames = np.where(e_queue_traj > E_ANOMALY_THRESHOLD)[0]
        first_anomaly_frame = int(anomaly_frames[0]) if len(anomaly_frames) > 0 else -1

        q_final = np.mean(history['Q_total'][-100:]) / 1e6
        cost_final = np.mean(history['Cost'][-100:])

        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Q_avg(last100): {float(q_final):.2f} Mb\n")
        log_f.write(f"PAoI_avg(last100): {float(cost_final):.2f} s\n")
        log_f.write(f"Max E_queue_BS: {max_e_queue:.1f}\n")
        log_f.write(f"Final E_queue_BS: {final_e_queue:.1f}\n")
        if first_anomaly_frame >= 0:
            log_f.write(f"首次越界帧 (>{E_ANOMALY_THRESHOLD:.0f}): {first_anomaly_frame}\n")
        log_f.write(f"完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        return (algo_name, seed, history, max_e_queue, final_e_queue, first_anomaly_frame, log_path)


def aggregate_results(all_results):
    """将多个种子的结果聚合成带统计信息的形式"""
    if not all_results:
        return {}

    aggregated = {}
    keys = all_results[0].keys()

    for key in keys:
        raw_arrays = []
        for result in all_results:
            if key in result and isinstance(result[key], list):
                raw_arrays.append(np.array(result[key]))
        if not raw_arrays:
            continue
        min_len = min(len(arr) for arr in raw_arrays)
        trimmed = [arr[:min_len] for arr in raw_arrays]
        stacked = np.stack(trimmed, axis=0)
        aggregated[key] = {
            'mean': np.mean(stacked, axis=0),
            'std': np.std(stacked, axis=0),
            'raw': trimmed
        }
    return aggregated


def save_aggregated_results(aggregated, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(aggregated, f)
    print(f">>> 聚合结果已保存: {filepath}")


def _print_anomaly_report(anomalies):
    """打印能量异常汇总"""
    if not anomalies:
        print(f"\n  [能量检查] 未发现异常 (所有 Max E_queue_BS < {E_ANOMALY_THRESHOLD:.0f})")
        return

    print(f"\n{'!' * 70}")
    print(f"  ⚠️  能量异常报告 ({len(anomalies)} 个任务)")
    print(f"  ⚠️  阈值 = {E_ANOMALY_THRESHOLD:.0f} (E_max_BS = {E_MAX_BS})")
    print(f"{'!' * 70}")
    print(f"  {'算法':<6s} | {'种子':>5s} | {'Max E_q':>12s} | {'Final E_q':>12s} | {'首次越界':>10s} | 日志")
    print(f"  {'-'*6}-+-{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*20}")

    algo_counts = defaultdict(int)
    for a in anomalies:
        algo_counts[a['algo']] += 1

    for a in anomalies:
        first_str = f"{a['first_frame']}" if a['first_frame'] >= 0 else "N/A"
        log_name = os.path.basename(a['log_path'])
        print(f"  {a['algo']:<6s} | {a['seed']:5d} | {a['max_e']:12.1f} | "
              f"{a['final_e']:12.1f} | {first_str:>10s} | {log_name}")

    print(f"\n  各算法异常次数: {dict(algo_counts)}")
    print(f"  日志目录: {os.path.dirname(anomalies[0]['log_path'])}")
    print(f"{'!' * 70}\n")


def run_full_experiment(cfg, seeds=None, algorithms=None, n_workers=None):
    """
    完整的多算法、多种子实验流程（多进程并行 + 独立日志 + 能量异常检测）

    :param cfg: SystemConfig 实例
    :param seeds: list of int
    :param algorithms: dict of {name: agent_class}
    :param n_workers: 并行进程数
    :return: dict of {algo_name: aggregated_result}, timestamp
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1000]

    if algorithms is None:
        algorithms = {
            'LDA': LDAAgent,
            'AC': ACAgent,
            'COB': COBAgent,
            'MTD': MTDAgent
        }

    n_tasks = len(seeds) * len(algorithms)
    if n_workers is None:
        env_workers = os.environ.get('LDA_WORKERS')
        if env_workers:
            n_workers = int(env_workers)
        else:
            n_workers = min(max(1, multiprocessing.cpu_count() // 2+2), n_tasks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'logs', 'multi_seed', timestamp)
    os.makedirs(log_dir, exist_ok=True)

    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'multi_seed')
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  完整实验启动")
    print(f"  算法: {list(algorithms.keys())}")
    print(f"  种子: {seeds}")
    print(f"  总任务数: {n_tasks} | 并行: {n_workers}")
    print(f"  日志目录: {log_dir}")
    print(f"{'='*60}")

    # 构建所有任务 (带日志路径)
    tasks = []
    for algo_name, agent_class in algorithms.items():
        for seed in seeds:
            log_path = os.path.join(log_dir, f"{algo_name}_s{seed}.log")
            tasks.append((cfg, agent_class, seed, algo_name, log_path))

    results_by_algo = {name: [] for name in algorithms}
    anomalies = []

    if n_workers <= 1:
        for task in tasks:
            algo_name, seed, history, max_e_q, final_e_q, first_frame, log_path = _worker_flat(task)
            results_by_algo[algo_name].append(history)
            if not np.isnan(max_e_q) and max_e_q > E_ANOMALY_THRESHOLD:
                anomalies.append({
                    'algo': algo_name, 'seed': seed, 'max_e': max_e_q,
                    'final_e': final_e_q, 'first_frame': first_frame, 'log_path': log_path,
                })
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_worker_flat, task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                algo_name, seed, history, max_e_q, final_e_q, first_frame, log_path = future.result()
                results_by_algo[algo_name].append(history)
                completed += 1

                q_final = float(np.mean(history['Q_total'][-100:]) / 1e6)
                cost_final = float(np.mean(history['Cost'][-100:]))
                flag = ""
                if not np.isnan(max_e_q) and max_e_q > E_ANOMALY_THRESHOLD:
                    anomalies.append({
                        'algo': algo_name, 'seed': seed, 'max_e': max_e_q,
                        'final_e': final_e_q, 'first_frame': first_frame, 'log_path': log_path,
                    })
                    flag = f" ⚠️ E_QUEUE={max_e_q:.0f}"
                print(f"  [{completed}/{n_tasks}] {algo_name} (seed={seed}) | "
                      f"Q_avg={q_final:.2f}Mb | PAoI_avg={cost_final:.2f}s{flag}")

    # 聚合并保存
    all_aggregated = {}
    for algo_name in algorithms:
        aggregated = aggregate_results(results_by_algo[algo_name])
        all_aggregated[algo_name] = aggregated
        save_path = os.path.join(results_dir, f'{algo_name}_aggregated_{timestamp}.pkl')
        save_aggregated_results(aggregated, save_path)

    print(f"\n{'='*60}")
    print("  所有实验完成!")
    print(f"{'='*60}")

    _print_anomaly_report(anomalies)

    return all_aggregated, timestamp


def plot_multi_seed_results(aggregated_results, cfg, timestamp):
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'multi_seed')
    save_path = os.path.join(results_dir, f'comparison_with_ci_{timestamp}.png')
    plot_results_with_ci(aggregated_results, cfg, save_path=save_path)
    return save_path


if __name__ == "__main__":
    cfg = SystemConfig()

    print("=" * 60)
    print("  Multi-seed DRL 算法评估实验")
    print("=" * 60)

    seeds = [42, 123, 456, 789, 1000]
    algorithms = {
        'LDA': LDAAgent,
        'AC': ACAgent,
        'COB': COBAgent,
        'MTD': MTDAgent
    }

    print(f"\n>>> 可用 CPU 核心数: {multiprocessing.cpu_count()}")
    print(f">>> 每个任务有独立日志文件 (logs/multi_seed/)")
    print(f">>> 能量异常阈值: {E_ANOMALY_THRESHOLD:.0f}")

    aggregated, timestamp = run_full_experiment(cfg, seeds=seeds, algorithms=algorithms)
    plot_path = plot_multi_seed_results(aggregated, cfg, timestamp)
    print(f"\n>>> 置信区间对比图已保存: {plot_path}")
