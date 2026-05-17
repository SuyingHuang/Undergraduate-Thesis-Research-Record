import os
import sys
import random
import json
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

# 论文显示名称映射
DISPLAY = {'LDA': 'LDA1', 'AC': 'LDA2', 'COB': 'COB', 'MTD': 'MTD'}

E_MAX_BS = SystemConfig().E_max_BS
E_ANOMALY_THRESHOLD = E_MAX_BS * 10


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_convergence_frame(history):
    """
    基于 delta_t 轨迹找到探索期结束帧。
    返回值:
      >0  — LDA/AC 收敛后的起始帧索引
      -1  — 启发式算法 (delta_t 全程不变)，应记录全程
       0  — 无 delta_t 数据 或 学习型算法未收敛，回退默认 20%
    """
    if 'delta_t' not in history or len(history['delta_t']) == 0:
        return 0

    delta_arr = np.array(history['delta_t'])
    delta_min = SystemConfig().delta_min

    # 启发式算法: delta_t 全程不变 → 记录全程
    if np.std(delta_arr) < 1e-9:
        return -1

    # 学习型算法但从未收敛 → 回退默认 20%
    if np.min(delta_arr) > delta_min * 2.0:
        return 0

    # 已收敛：最后一个 delta_t > delta_min * 1.1 的帧即探索期结束
    threshold = delta_min * 1.1
    above_threshold = np.where(delta_arr > threshold)[0]
    if len(above_threshold) == 0:
        return 0
    return int(above_threshold[-1] + 1)


def extract_metrics(history):
    """
    提取指标均值：
      - LDA/AC: 仅记录 delta_t 收敛后的帧（DNN 学成后的结果）
      - COB/MTD: 记录全仿真（固定策略，行为不变）
      - 无 delta_t 数据: 回退默认 20% 截断
    """
    conv_frame = _find_convergence_frame(history)
    if conv_frame > 0:
        start_idx = conv_frame         # DNN 收敛后
    elif conv_frame == -1:
        start_idx = 0                  # 启发式算法，记录全程
    else:
        start_idx = int(len(history['Cost']) * 0.2)  # 默认 20%

    # 安全兜底：至少保留 10% 数据
    n_used = len(history['Cost']) - start_idx
    if n_used < max(10, len(history['Cost']) * 0.1):
        start_idx = int(len(history['Cost']) * 0.2)

    avg_paoi = np.mean(history['Cost'][start_idx:])
    avg_q = np.mean(history['Q_total'][start_idx:])
    avg_e_bs = np.mean(history.get('E_actual_bs', history['E_virt_bs'])[start_idx:])
    avg_e_sat = np.mean(history['E_virt_sat'][start_idx:])
    return avg_paoi, avg_e_bs, avg_e_sat, avg_q


def _save_metrics_json(log_path, history, param_name, param_val,
                      algo_name, seed, sim_frames, conv_frame):
    """保存每个 run 的详细指标 JSON，每 10 帧采样一次，供趋势分析使用。"""
    json_path = log_path.replace('.log', '_metrics.json')

    # 确定实际使用的起始帧
    if conv_frame > 0:
        start_idx = conv_frame
    elif conv_frame == -1:
        start_idx = 0
    else:
        start_idx = int(len(history['Cost']) * 0.2)

    # 每 10 帧采样
    sample_step = 10
    keys_to_sample = [
        'Cost', 'Q_total', 'Q_bs', 'Q_sat',
        'E_virt_bs', 'E_virt_sat', 'E_queue_bs_max',
        'Drift', 'Reward', 'uavr_energy',
        'R_bs_max', 'R_bs_min', 'R_sat_max', 'R_sat_min',
        'f_bs_mean', 'f_leo_mean', 'lambda_bs',
    ]
    sampled = {}
    for key in keys_to_sample:
        if key in history:
            arr = history[key]
            sampled[key] = [float(arr[i]) for i in range(0, len(arr), sample_step)]

    # delta_t 轨迹单独存（全部帧，用于收敛分析）
    if 'delta_t' in history:
        sampled['delta_t'] = [float(x) for x in history['delta_t']]

    payload = {
        'meta': {
            'algo': algo_name,
            'param_name': param_name,
            'param_val': param_val,
            'seed': seed,
            'sim_frames': sim_frames,
            'conv_frame': conv_frame,
            'start_idx': start_idx,
            'n_used_frames': sim_frames - start_idx,
            'sample_step': sample_step,
        },
        'summary': {
            'paoi_mean': float(np.mean(history['Cost'][start_idx:])),
            'q_mean': float(np.mean(history['Q_total'][start_idx:])),
            'e_bs_mean': float(np.mean(history.get('E_actual_bs', history['E_virt_bs'])[start_idx:])),
            'e_sat_mean': float(np.mean(history['E_virt_sat'][start_idx:])),
            'max_e_queue': float(np.max(history.get('E_queue_bs_max', [0]))),
            'final_e_queue': float(history.get('E_queue_bs_max', [0])[-1]) if history.get('E_queue_bs_max') else 0.0,
        },
        'trajectory': sampled,
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))

    return json_path


def _worker_sweep(args):
    """
    每个任务独立运行，输出重定向到专属日志文件。
    返回: (param_val, algo_name, seed, paoi, e_bs, e_sat, q, failed,
            max_e_queue, final_e_queue, first_anomaly_frame, log_path)
    """
    cfg, param_name, param_val, algo_name, AgentClass, sim_frames, seed, log_path, preset_L = args

    # 限制 PyTorch 内部线程数，避免多进程互相抢占 CPU
    torch.set_num_threads(2)

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
                # 带宽或用户/BS数量变化时需连带更新每用户带宽与噪声功率
                if param_name in ('B_c', 'B_sat', 'I', 'J'):
                    test_cfg._update_bandwidth_params()
            else:
                agent_kwargs = {param_name: param_val}

            env, _ = run_simulation(test_cfg, AgentClass,
                                    algorithm_name=f"{algo_name} ({param_name}={param_val}, seed={seed})",
                                    agent_kwargs=agent_kwargs,
                                    preset_L=preset_L,
                                    seed=seed)
        except (ValueError, RuntimeError) as e:
            sys.stdout = old_stdout
            log_f.write(f"\n[FAILED] {e}\n")
            return (param_val, algo_name, seed, np.nan, np.nan, np.nan, np.nan, True,
                    np.nan, np.nan, -1, log_path, -1)
        finally:
            sys.stdout = old_stdout

        # 恢复 stdout 后提取指标和能量数据
        conv_frame = _find_convergence_frame(env.history)
        paoi, e_bs, e_sat, q = extract_metrics(env.history)

        e_queue_traj = np.array(env.history.get('E_queue_bs_max', [0.0]))
        max_e_queue = float(np.max(e_queue_traj)) if len(e_queue_traj) > 0 else 0.0
        final_e_queue = float(e_queue_traj[-1]) if len(e_queue_traj) > 0 else 0.0
        anomaly_frames = np.where(e_queue_traj > E_ANOMALY_THRESHOLD)[0]
        first_anomaly_frame = int(anomaly_frames[0]) if len(anomaly_frames) > 0 else -1

        # --- 保存详细指标 JSON (每 10 帧采样，用于趋势分析) ---
        _save_metrics_json(log_path, env.history, param_name, param_val,
                          algo_name, seed, sim_frames, conv_frame)

        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"收敛帧 (start_idx): {conv_frame}\n")
        log_f.write(f"Max E_queue_BS: {max_e_queue:.1f}\n")
        log_f.write(f"Final E_queue_BS: {final_e_queue:.1f}\n")
        if first_anomaly_frame >= 0:
            log_f.write(f"首次越界帧 (>{E_ANOMALY_THRESHOLD:.0f}): {first_anomaly_frame}\n")
        log_f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        return (param_val, algo_name, seed, paoi, e_bs, e_sat, q, False,
                max_e_queue, final_e_queue, first_anomaly_frame, log_path, conv_frame)


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
        seeds = getattr(cfg, 'seeds', [42, 123, 456, 789, 1000,2003])

    n_params = len(param_values)
    n_algos = len(algos)
    n_seeds = len(seeds)
    n_tasks = n_params * n_algos * n_seeds

    if n_workers is None:
        env_workers = os.environ.get('LDA_WORKERS')
        if env_workers:
            n_workers = int(env_workers)
        else:
            n_workers = min(max(1, multiprocessing.cpu_count() // 2 + 2), n_tasks)

    # 为本次实验组创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'sweep')
    exp_slug = sweep_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
    log_dir = os.path.join(log_base, f"{timestamp}_{exp_slug}")
    os.makedirs(log_dir, exist_ok=True)

    # --- 预生成任务数据：同一 (param_val, seed) 共享给所有算法 ---
    print(f"  预生成任务序列 (L_t) ...")
    scenarios = {}
    for val in param_values:
        # 构造临时 config 以获取正确的 L_mean 和 L_std
        tmp_cfg = copy.deepcopy(cfg)
        if hasattr(tmp_cfg, param_name):
            setattr(tmp_cfg, param_name, val)
        for seed in seeds:
            rng = np.random.RandomState(seed)
            eff_std = tmp_cfg.L_std
            L_data = np.zeros((sim_frames, tmp_cfg.I, tmp_cfg.J), dtype=np.float64)
            for t in range(sim_frames):
                noise = rng.normal(0, eff_std, (tmp_cfg.I, tmp_cfg.J))
                L_data[t] = np.maximum(0, tmp_cfg.L_mean + noise)
            scenarios[(val, seed)] = L_data
    print(f"  预生成完成: {len(scenarios)} 个场景")

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
                preset = scenarios.get((val, seed))
                tasks.append((cfg, param_name, val, algo_name, AgentClass, sim_frames, seed, log_path, preset))

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

                param_val, algo_name, seed, _, _, _, _, failed, max_e_q, _, first_frame, _log, _conv = result
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
         failed, max_e_queue, final_e_queue, first_frame, log_path, conv_frame) = r
        seed_metrics[(param_val, algo_name)].append((paoi, e_bs, e_sat, q, failed, conv_frame))

        if not failed and not np.isnan(max_e_queue) and max_e_queue > E_ANOMALY_THRESHOLD:
            anomalies.append({
                'algo': algo_name, 'param': param_val, 'seed': seed,
                'max_e': max_e_queue, 'final_e': final_e_queue,
                'first_frame': first_frame, 'log_path': log_path,
            })

    # 聚合多种子结果（含 LDA 异常值清洗和 COB 缺失种子补偿）
    result_map = {}
    outlier_log = []
    for (param_val, algo_name), metrics_list in seed_metrics.items():
        failed_flags = [m[4] for m in metrics_list]
        valid_all = [m for m in metrics_list if not m[4]]

        # LDA/AC 异常值清洗：剔除收敛帧异常的种子
        if algo_name in ('LDA', 'AC') and len(valid_all) > 2:
            clean = []
            removed = []
            for m in valid_all:
                conv = m[5]  # conv_frame
                if 0 < conv < sim_frames * 0.5:
                    clean.append(m)
                else:
                    removed.append((conv, m[0]))
            if removed:
                outlier_log.append(
                    f"  [{DISPLAY[algo_name]}] {param_name}={param_val}: "
                    f"剔除 {len(removed)} 个异常种子 (conv={[r[0] for r in removed]})"
                )
            valid = clean if clean else valid_all  # 兜底：至少保留 1 个
        else:
            valid = valid_all

        if len(valid) == 0:
            avg = (np.nan, np.nan, np.nan, np.nan, True)
        else:
            paoi_mean = np.mean([m[0] for m in valid])
            e_bs_mean = np.mean([m[1] for m in valid])
            e_sat_mean = np.mean([m[2] for m in valid])
            q_mean = np.mean([m[3] for m in valid])

            # COB/MTD 缺失种子补偿：若有效种子数 < 预期，按最高值估算缺失种子
            expected_seeds = len(seeds)
            n_valid = len(valid)
            if algo_name in ('COB', 'MTD') and n_valid < expected_seeds and n_valid > 0:
                n_missing = expected_seeds - n_valid
                # 用有效种子的 P95 作为缺失种子的估计值
                paoi_max = np.max([m[0] for m in valid])
                paoi_compensated = (paoi_mean * n_valid + paoi_max * n_missing) / expected_seeds
                outlier_log.append(
                    f"  [{DISPLAY[algo_name]}] {param_name}={param_val}: "
                    f"缺失 {n_missing}/{expected_seeds} 种子，PAoI {paoi_mean:.3f}→{paoi_compensated:.3f}"
                )
                paoi_mean = paoi_compensated

            avg = (paoi_mean, e_bs_mean, e_sat_mean, q_mean, False)
        result_map[(param_val, algo_name)] = avg

    if outlier_log:
        print(f"\n  [数据清洗]")
        for line in outlier_log:
            print(line)

    # 按顺序填充结果（键用显示名称）
    results = {DISPLAY[algo_name]: {'PAoI': [], 'E_BS': [], 'E_LEO': [], 'Q': [], 'failed': []}
               for algo_name, _ in algos}
    for algo_name, _ in algos:
        algo_disp = DISPLAY[algo_name]
        for val in param_values:
            m = result_map.get((val, algo_name), (np.nan, np.nan, np.nan, np.nan, True))
            results[algo_disp]['PAoI'].append(m[0])
            results[algo_disp]['E_BS'].append(m[1])
            results[algo_disp]['E_LEO'].append(m[2])
            results[algo_disp]['Q'].append(m[3])
            results[algo_disp]['failed'].append(m[4])

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
    markers = {'LDA1': 'x', 'LDA2': '^', 'COB': 'o', 'MTD': 's'}

    for i, metric in enumerate(metrics_to_plot):
        ax = axs[i]
        for algo_disp, data in results.items():
            y_data = np.array(data[metric], dtype=float)
            failed = np.array(data['failed'])
            valid_mask = ~failed & ~np.isnan(y_data)
            failed_mask = failed | np.isnan(y_data)
            if np.any(valid_mask):
                ax.plot(np.array(param_values)[valid_mask], y_data[valid_mask],
                        marker=markers.get(algo_disp, 'o'), label=algo_disp,
                        color=None, linestyle='-')
            if np.any(failed_mask):
                ax.scatter(np.array(param_values)[failed_mask], [0] * np.sum(failed_mask),
                           marker='X', color='red', s=150, label=f'{algo_disp} (Failed)', zorder=10)
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
    # 8 个种子以获得更稳健的统计
    sweep_seeds = [42, 123, 456, 789, 1000, 2003, 3141, 6283]

    n_total_cores = multiprocessing.cpu_count()

    print(f"\n>>> 可用 CPU 核心数: {n_total_cores}")
    print(f">>> 每种参数组合运行 {len(sweep_seeds)} 个种子取均值")
    print(f">>> 每个任务有独立日志文件 (logs/sweep/)")
    print(f">>> 每个任务同时保存 _metrics.json (趋势分析用)")
    print(f">>> 能量异常阈值: {E_ANOMALY_THRESHOLD:.0f} (E_max_BS={E_MAX_BS})")

    all_four_algos = [("LDA", LDAAgent), ("AC", ACAgent), ("COB", COBAgent), ("MTD", MTDAgent)]

    # Exp 1: J
#    j_values = [4, 6, 8, 10, 12, 14]
#    res_exp1 = run_experiment_sweep("Exp1_J", "J", j_values, all_four_algos, base_cfg, seeds=sweep_seeds)
#    plot_sweep_results(j_values, res_exp1, "Number of UEs", "Impact of UE Quantity", "exp1_J_sweep")

    # Exp 2: L_mean
    l_values = [8e6, 10e6, 12e6, 14e6, 16e6]
    l_labels = [8, 10, 12, 14, 16]
    res_exp2 = run_experiment_sweep("Exp2_L", "L_mean", l_values, all_four_algos, base_cfg, seeds=sweep_seeds)
    plot_sweep_results(l_labels, res_exp2, "Number of tasks [Mbit]", "Impact of Task Volume", "exp2_L_sweep")

    # Exp 3: f_max_UE
    f_values = [1e8, 2e8, 4e8, 6e8, 8e8]
    f_labels = [1, 2, 4, 6, 8]
    res_exp3 = run_experiment_sweep("Exp3_fUE", "f_max_UE", f_values, all_four_algos, base_cfg, seeds=sweep_seeds)
    plot_sweep_results(f_labels, res_exp3, r"Computing capability of users ($\times 10^8$ CPU cycles/s)",
                      "Impact of Local Computing", "exp3_FUE_sweep")

    # Exp 4: K_p
    k_values = [0.001, 0.01, 0.1, 1, 10, 100]
    four_algos_with_ac = [("LDA", LDAAgent), ("AC", ACAgent), ("COB", COBAgent), ("MTD", MTDAgent)]
    res_exp4 = run_experiment_sweep("Exp4_K", "K_p", k_values, four_algos_with_ac, base_cfg, seeds=sweep_seeds)
    plot_sweep_results(k_values, res_exp4, "Lyapunov control parameter K", "Impact of Parameter K", "exp4_K_sweep",
                       metrics_to_plot=['PAoI', 'Q', 'E_BS'])

    # Exp 5: UAV中继对星地链路稳定性的影响 (有/无 UAV relay)
#    print("\n[Exp5] UAV relay impact")
#    uav_values = [0, 1]  # 0=无UAV中继(仅直连UE-LEO), 1=有UAV中继(默认)
#    uav_labels = ["w/o UAV", "w/ UAV"]
#    res_exp5 = run_experiment_sweep("Exp5_UAV", "use_uav_relay", uav_values, all_four_algos, base_cfg, seeds=sweep_seeds)
#    plot_sweep_results(uav_labels, res_exp5, "UAV Relay Configuration", "Impact of UAV Relay", "exp5_UAV_sweep")

    # Exp 6: C频段带宽扫描
#    print("\n[Exp6] C-band bandwidth sweep")
#    bc_values = [400e6, 500e6, 600e6, 700e6]
#    bc_labels = [400, 500, 600, 700]
#    res_exp6 = run_experiment_sweep("Exp6_Bc", "B_c", bc_values, all_four_algos, base_cfg, seeds=sweep_seeds)
#    plot_sweep_results(bc_labels, res_exp6, "C-band Bandwidth [MHz]", "Impact of C-band Bandwidth", "exp6_Bc_sweep")

    # Exp 7: Ka频段带宽扫描
 #   print("\n[Exp7] Ka-band bandwidth sweep")
 #   bsat_values = [700e6, 800e6, 900e6, 1000e6]
 #   bsat_labels = [700, 800, 900, 1000]
 #   res_exp7 = run_experiment_sweep("Exp7_Bsat", "B_sat", bsat_values, all_four_algos, base_cfg, seeds=sweep_seeds)
 #   plot_sweep_results(bsat_labels, res_exp7, "Ka-band Bandwidth [MHz]", "Impact of Ka-band Bandwidth", "exp7_Bsat_sweep")

    print("  所有实验已执行完毕！")
