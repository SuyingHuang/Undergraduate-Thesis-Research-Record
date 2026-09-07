import os
import sys
import argparse
import json
import csv
import hashlib
import platform
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import multiprocessing
import subprocess
import traceback
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
METRICS = ('PAoI', 'E_BS', 'E_LEO', 'Q')


from utils.reproducibility import set_seed


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, type):
        return obj.__name__
    return str(obj)


def _config_snapshot(cfg):
    snapshot = {}
    for key, val in vars(cfg).items():
        if key.startswith('_') or callable(val):
            continue
        try:
            json.dumps(val, default=_json_default)
            snapshot[key] = val
        except TypeError:
            snapshot[key] = str(val)
    return snapshot


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _scenario_hash(arr):
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode('ascii'))
    h.update(str(arr.dtype).encode('ascii'))
    h.update(arr.view(np.uint8).tobytes())
    return h.hexdigest()


def _git_worktree_dirty():
    """A commit hash alone cannot identify a run made with uncommitted edits."""
    try:
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=os.path.dirname(os.path.abspath(__file__)), text=True,
            stderr=subprocess.DEVNULL)
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _ci95(values):
    vals = np.array([v for v in values if np.isfinite(v)], dtype=float)
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else np.nan
    ci = float(1.96 * std / np.sqrt(n)) if n > 1 else np.nan
    return mean, std, ci, n


def _metric_tuple_to_dict(paoi, e_bs, e_sat, q):
    return {
        'PAoI': float(paoi) if not np.isnan(paoi) else np.nan,
        'E_BS': float(e_bs) if not np.isnan(e_bs) else np.nan,
        'E_LEO': float(e_sat) if not np.isnan(e_sat) else np.nan,
        'Q': float(q) if not np.isnan(q) else np.nan,
    }


def _extract_metrics_from_start(history, start_idx):
    avg_paoi = np.mean(history['Cost'][start_idx:])
    avg_q = np.mean(history['Q_total'][start_idx:])
    avg_e_bs = np.mean(history.get('E_actual_bs', history['E_virt_bs'])[start_idx:])
    avg_e_sat = np.mean(history['E_virt_sat'][start_idx:])
    return _metric_tuple_to_dict(avg_paoi, avg_e_bs, avg_e_sat, avg_q)


def _find_convergence_frame(history, delta_min=None):
    """Legacy field name: window-shrink diagnostic, NOT proof of convergence.

    -1: no exploration window (heuristic); 0: no sustained low window;
    >0: first frame after the last above-threshold observation.
    """
    values = history.get('delta_t', [])
    if not values:
        return -1
    delta = np.asarray(values, dtype=float)
    threshold = (SystemConfig().delta_min if delta_min is None else delta_min) * 1.1
    if not np.all(np.isfinite(delta)) or delta[-1] > threshold:
        return 0
    above = np.flatnonzero(delta > threshold)
    return int(above[-1] + 1) if len(above) else 1


def extract_metric_bundle(history, delta_min=None):
    conv_frame = _find_convergence_frame(history, delta_min)
    if conv_frame > 0:
        adaptive_start = conv_frame
    elif conv_frame == -1:
        adaptive_start = 0
    else:
        adaptive_start = int(len(history['Cost']) * 0.2)

    n_used = len(history['Cost']) - adaptive_start
    if n_used < max(10, len(history['Cost']) * 0.1):
        adaptive_start = int(len(history['Cost']) * 0.2)

    fixed_half_start = int(len(history['Cost']) * 0.5)
    fixed_last_1000_start = max(0, len(history['Cost']) - 1000)
    return {
        'conv_frame': conv_frame,
        'adaptive_start': adaptive_start,
        'fixed_half_start': fixed_half_start,
        'fixed_last_1000_start': fixed_last_1000_start,
        'adaptive': _extract_metrics_from_start(history, adaptive_start),
        'fixed_half': _extract_metrics_from_start(history, fixed_half_start),
        'fixed_last_1000': _extract_metrics_from_start(history, fixed_last_1000_start),
    }


def _save_metrics_json(log_path, history, param_name, param_val,
                      algo_name, seed, sim_frames, metric_bundle):
    """保存每个 run 的详细指标 JSON，每 10 帧采样一次，供趋势分析使用。"""
    json_path = log_path.replace('.log', '_metrics.json')

    conv_frame = metric_bundle['conv_frame']
    start_idx = metric_bundle['adaptive_start']

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
    sampled['Loss'] = history.get('Loss', [])
    sampled['Loss_per_BS'] = history.get('Loss_per_BS', [])

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
            'fixed_half_start': metric_bundle['fixed_half_start'],
            'fixed_last_1000_start': metric_bundle['fixed_last_1000_start'],
            'sample_step': sample_step,
        },
        'summary': {
            'adaptive': metric_bundle['adaptive'],
            'fixed_half': metric_bundle['fixed_half'],
            'fixed_last_1000': metric_bundle['fixed_last_1000'],
            'max_e_queue': float(np.max(history.get('E_queue_bs_max', [0]))),
            'final_e_queue': float(history.get('E_queue_bs_max', [0])[-1]) if history.get('E_queue_bs_max') else 0.0,
        },
        'trajectory': sampled,
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'), default=_json_default)

    return json_path


def _worker_sweep(args):
    """
    每个任务独立运行，输出重定向到专属日志文件。
    返回: (param_val, algo_name, seed, paoi, e_bs, e_sat, q, failed,
            max_e_queue, final_e_queue, first_anomaly_frame, log_path)
    """
    cfg, param_name, param_val, algo_name, AgentClass, sim_frames, seed, log_path, preset_L, scenario_hash = args

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

            anomaly_threshold = test_cfg.E_max_BS * 10

            env, _ = run_simulation(test_cfg, AgentClass,
                                    algorithm_name=f"{algo_name} ({param_name}={param_val}, seed={seed})",
                                    agent_kwargs=agent_kwargs,
                                    preset_L=preset_L,
                                    seed=seed)
        except Exception as e:
            sys.stdout = old_stdout
            failure_reason = ''.join(traceback.format_exception_only(type(e), e)).strip()
            log_f.write(f"\n[FAILED] {failure_reason}\n")
            log_f.write(traceback.format_exc())
            return (param_val, algo_name, seed, np.nan, np.nan, np.nan, np.nan, True,
                    np.nan, np.nan, -1, log_path, -1, failure_reason, scenario_hash,
                    np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan)
        finally:
            sys.stdout = old_stdout

        # 恢复 stdout 后提取指标和能量数据
        metric_bundle = extract_metric_bundle(env.history, test_cfg.delta_min)
        conv_frame = metric_bundle['conv_frame']
        adaptive = metric_bundle['adaptive']
        fixed_half = metric_bundle['fixed_half']
        fixed_last_1000 = metric_bundle['fixed_last_1000']
        paoi, e_bs, e_sat, q = adaptive['PAoI'], adaptive['E_BS'], adaptive['E_LEO'], adaptive['Q']

        e_queue_traj = np.array(env.history.get('E_queue_bs_max', [0.0]))
        max_e_queue = float(np.max(e_queue_traj)) if len(e_queue_traj) > 0 else 0.0
        final_e_queue = float(e_queue_traj[-1]) if len(e_queue_traj) > 0 else 0.0
        anomaly_frames = np.where(e_queue_traj > anomaly_threshold)[0]
        first_anomaly_frame = int(anomaly_frames[0]) if len(anomaly_frames) > 0 else -1

        # --- 保存详细指标 JSON (每 10 帧采样，用于趋势分析) ---
        _save_metrics_json(log_path, env.history, param_name, param_val,
                          algo_name, seed, sim_frames, metric_bundle)

        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"探索窗口诊断帧 (不等于已证明收敛): {conv_frame}\n")
        log_f.write(f"Max E_queue_BS: {max_e_queue:.1f}\n")
        log_f.write(f"Final E_queue_BS: {final_e_queue:.1f}\n")
        if first_anomaly_frame >= 0:
            log_f.write(f"首次越界帧 (>{anomaly_threshold:.0f}): {first_anomaly_frame}\n")
        log_f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        return (param_val, algo_name, seed, paoi, e_bs, e_sat, q, False,
                max_e_queue, final_e_queue, first_anomaly_frame, log_path, conv_frame,
                '', scenario_hash,
                fixed_half['PAoI'], fixed_half['E_BS'], fixed_half['E_LEO'], fixed_half['Q'],
                fixed_last_1000['PAoI'], fixed_last_1000['E_BS'],
                fixed_last_1000['E_LEO'], fixed_last_1000['Q'])


def _print_anomaly_report(anomalies, sweep_name, param_name, anomaly_threshold=E_ANOMALY_THRESHOLD):
    if not anomalies:
        print(f"\n  [能量检查] 未发现异常 (所有 Max E_BS_queue < {anomaly_threshold:.0f})")
        return

    print(f"\n{'!' * 70}")
    print(f"  ⚠️  能量异常报告: {sweep_name}")
    print(f"  ⚠️  阈值 = {anomaly_threshold:.0f}")
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


def _save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_metric_rows(rows, algos, param_values, cleaned=False):
    results = {DISPLAY[algo_name]: {m: [] for m in METRICS} for algo_name, _ in algos}
    for data in results.values():
        for metric in METRICS:
            data[f'{metric}_std'] = []
            data[f'{metric}_ci95'] = []
        data['failed'] = []
        data['n_valid'] = []
        data['n_failed'] = []

    summary_rows = []
    for algo_name, _ in algos:
        algo_disp = DISPLAY[algo_name]
        for param_val in param_values:
            group = [r for r in rows if r['algo'] == algo_name and r['param_val'] == param_val]
            if cleaned and algo_name in ('LDA', 'AC') and len([r for r in group if not r['failed']]) > 2:
                keep = []
                for row in group:
                    row = dict(row)
                    if (not row['failed']) and not (0 < row['conv_frame'] < row['sim_frames'] * 0.5):
                        row['cleaned_out'] = True
                        row['clean_reason'] = 'conv_frame_outside_first_half'
                    keep.append(row)
                group = keep

            if cleaned:
                usable = [r for r in group if not r['failed'] and not r.get('cleaned_out')]
                n_cleaned = sum(1 for r in group if r.get('cleaned_out'))
            else:
                usable = [r for r in group if not r['failed']]
                n_cleaned = 0
            n_failed = sum(1 for r in group if r['failed'])
            failed_combo = len(usable) == 0

            stats_by_metric = {}
            for metric in METRICS:
                mean, std, ci, n = _ci95([r[metric] for r in usable])
                stats_by_metric[metric] = (mean, std, ci, n)
                results[algo_disp][metric].append(mean)
                results[algo_disp][f'{metric}_std'].append(std)
                results[algo_disp][f'{metric}_ci95'].append(ci)

            results[algo_disp]['failed'].append(failed_combo)
            results[algo_disp]['n_valid'].append(len(usable))
            results[algo_disp]['n_failed'].append(n_failed)

            base = {
                'view': 'cleaned' if cleaned else 'raw',
                'param_val': param_val,
                'algo': algo_name,
                'algo_display': algo_disp,
                'n_total': len(group),
                'n_valid': len(usable),
                'n_failed': n_failed,
                'n_cleaned': n_cleaned,
                'failed_combo': failed_combo,
            }
            for metric, (mean, std, ci, n) in stats_by_metric.items():
                base[f'{metric}_mean'] = mean
                base[f'{metric}_std'] = std
                base[f'{metric}_ci95'] = ci
                base[f'{metric}_n'] = n
            summary_rows.append(base)

    return results, summary_rows


def run_experiment_sweep(sweep_name, param_name, param_values, algos, cfg,
                         n_workers=None, seeds=None, sim_frames=None,
                         metric_view='fixed_half'):
    """Run a sweep. Primary view: all successful seeds, common last-half window.

    raw/cleaned retain historical adaptive-window summaries for comparison only.
    """
    if sim_frames is None:
        sim_frames = cfg.sim_frames
    if sim_frames < 1 or not param_values or not algos:
        raise ValueError('Positive frames and non-empty parameters/algorithms are required')
    if metric_view not in ('fixed_half', 'raw', 'cleaned'):
        raise ValueError('metric_view must be fixed_half, raw, or cleaned')
    if seeds is None:
        seeds = getattr(cfg, 'seeds', [42, 123, 456, 789, 1000,2003])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be non-empty and unique')

    n_params = len(param_values)
    n_algos = len(algos)
    n_seeds = len(seeds)
    n_tasks = n_params * n_algos * n_seeds
    anomaly_threshold = cfg.E_max_BS * 10

    if n_workers is None:
        env_workers = os.environ.get('LDA_WORKERS')
        if env_workers:
            n_workers = int(env_workers)
        else:
            n_workers = min(max(1, multiprocessing.cpu_count() // 2 + 2), n_tasks)

    # 为本次实验组创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'sweep')
    exp_slug = sweep_name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
    log_dir = os.path.join(log_base, f"{timestamp}_{exp_slug}")
    os.makedirs(log_dir, exist_ok=True)
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'results', 'sweep', f"{timestamp}_{exp_slug}")
    os.makedirs(results_dir, exist_ok=True)

    # --- 预生成任务数据：同一 (param_val, seed) 共享给所有算法 ---
    print(f"  预生成任务序列 (L_t) ...")
    scenarios = {}
    scenario_hashes = {}
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
            scenario_hashes[(val, seed)] = _scenario_hash(L_data)
    print(f"  预生成完成: {len(scenarios)} 个场景")

    manifest = {
        'sweep_name': sweep_name,
        'metric_view': metric_view,
        'primary_window': 'last half of frames, shared by every algorithm',
        'ci_method': 'normal approximation: 1.96 * sample_std / sqrt(n)',
        'convergence_note': 'delta_t shrinkage is a diagnostic, not convergence proof',
        'param_name': param_name,
        'param_values': list(param_values),
        'algorithms': [name for name, _ in algos],
        'seeds': list(seeds),
        'sim_frames': sim_frames,
        'n_workers': n_workers,
        'timestamp': timestamp,
        'log_dir': log_dir,
        'results_dir': results_dir,
        'anomaly_threshold': anomaly_threshold,
        'git_commit': _git_commit(),
        'git_worktree_dirty': _git_worktree_dirty(),
        'runtime': {
            'python': sys.version,
            'platform': platform.platform(),
            'numpy': np.__version__,
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        },
        'config': _config_snapshot(cfg),
        'scenario_hashes': [
            {'param_val': val, 'seed': seed, 'sha256': scenario_hashes[(val, seed)]}
            for val in param_values for seed in seeds
        ],
    }
    _save_json(os.path.join(results_dir, 'manifest.json'), manifest)

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
                tasks.append((cfg, param_name, val, algo_name, AgentClass, sim_frames, seed,
                              log_path, preset, scenario_hashes[(val, seed)]))

    if n_workers <= 1:
        raw_results = []
        for task in tasks:
            raw_results.append(_worker_sweep(task))
    else:
        raw_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_sweep, task): task for task in tasks}
            completed = 0
            anomaly_count = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    failure_reason = ''.join(traceback.format_exception_only(type(e), e)).strip()
                    print(f"  [worker failed outside task wrapper] {failure_reason}")
                    task = futures[future]
                    _, _, val, name, _, _, seed, log, _, scenario_hash = task
                    raw_results.append((val, name, seed, np.nan, np.nan, np.nan, np.nan, True,
                                        np.nan, np.nan, -1, log, -1, failure_reason, scenario_hash,
                                        np.nan, np.nan, np.nan, np.nan,
                                        np.nan, np.nan, np.nan, np.nan))
                    completed += 1
                    continue
                raw_results.append(result)
                completed += 1

                param_val, algo_name, seed, _, _, _, _, failed, max_e_q, _, first_frame, _log, _conv = result[:13]
                status = "❌" if failed else "✅"
                flag = ""
                if not failed and not np.isnan(max_e_q) and max_e_q > anomaly_threshold:
                    anomaly_count += 1
                    flag = f" ⚠️ E_QUEUE={max_e_q:.0f} @Fr{first_frame}"
                print(f"  [{completed}/{n_tasks}] {algo_name} "
                      f"({param_name}={param_val}, s={seed}) {status}{flag}")

            if anomaly_count > 0:
                print(f"  ⚠️  本组已发现 {anomaly_count} 个能量异常 (详见日志目录)")

    # 解析结果并收集异常；保留失败任务，不填补未观测的数值。
    anomalies = []
    run_rows = []
    for r in raw_results:
        (param_val, algo_name, seed, paoi, e_bs, e_sat, q, failed,
         max_e_queue, final_e_queue, first_frame, log_path, conv_frame,
         failure_reason, scenario_hash, fixed_paoi, fixed_e_bs,
         fixed_e_sat, fixed_q, last1000_paoi, last1000_e_bs,
         last1000_e_sat, last1000_q) = r

        row = {
            'sweep_name': sweep_name, 'param_name': param_name, 'param_val': param_val,
            'algo': algo_name, 'algo_display': DISPLAY[algo_name], 'seed': seed,
            'sim_frames': sim_frames, 'scenario_hash': scenario_hash,
            'failed': bool(failed), 'failure_reason': failure_reason,
            'conv_frame': conv_frame, 'log_path': log_path,
            'PAoI': paoi, 'E_BS': e_bs, 'E_LEO': e_sat, 'Q': q,
            'fixed_half_PAoI': fixed_paoi, 'fixed_half_E_BS': fixed_e_bs,
            'fixed_half_E_LEO': fixed_e_sat, 'fixed_half_Q': fixed_q,
            'last1000_PAoI': last1000_paoi, 'last1000_E_BS': last1000_e_bs,
            'last1000_E_LEO': last1000_e_sat, 'last1000_Q': last1000_q,
            'max_e_queue': max_e_queue, 'final_e_queue': final_e_queue,
            'first_anomaly_frame': first_frame,
            'cleaned_out': False, 'clean_reason': '',
        }
        if algo_name in ('LDA', 'AC') and (not failed) and not (0 < conv_frame < sim_frames * 0.5):
            row['cleaned_out'] = True
            row['clean_reason'] = 'conv_frame_outside_first_half'
        run_rows.append(row)

        if not failed and not np.isnan(max_e_queue) and max_e_queue > anomaly_threshold:
            anomalies.append({
                'algo': algo_name, 'param': param_val, 'seed': seed,
                'max_e': max_e_queue, 'final_e': final_e_queue,
                'first_frame': first_frame, 'log_path': log_path,
            })

    raw_results_plot, raw_summary = _aggregate_metric_rows(run_rows, algos, param_values, cleaned=False)
    cleaned_results_plot, cleaned_summary = _aggregate_metric_rows(run_rows, algos, param_values, cleaned=True)
    fixed_rows = [dict(row, **{m: row[f'fixed_half_{m}'] for m in METRICS}) for row in run_rows]
    fixed_results_plot, fixed_summary = _aggregate_metric_rows(fixed_rows, algos, param_values)
    for row in fixed_summary:
        row['view'] = 'fixed_half'
    detail_fields = [
        'sweep_name', 'param_name', 'param_val', 'algo', 'algo_display', 'seed',
        'sim_frames', 'scenario_hash', 'failed', 'failure_reason', 'conv_frame',
        'cleaned_out', 'clean_reason', 'PAoI', 'E_BS', 'E_LEO', 'Q',
        'fixed_half_PAoI', 'fixed_half_E_BS', 'fixed_half_E_LEO', 'fixed_half_Q',
        'last1000_PAoI', 'last1000_E_BS', 'last1000_E_LEO', 'last1000_Q',
        'max_e_queue', 'final_e_queue', 'first_anomaly_frame', 'log_path',
    ]
    summary_fields = list(raw_summary[0].keys()) if raw_summary else []
    _write_csv(os.path.join(results_dir, 'per_seed_metrics.csv'), run_rows, detail_fields)
    _write_csv(os.path.join(results_dir, 'summary_raw.csv'), raw_summary, summary_fields)
    _write_csv(os.path.join(results_dir, 'summary_cleaned.csv'), cleaned_summary, summary_fields)
    _save_json(os.path.join(results_dir, 'summary_raw.json'), raw_summary)
    _save_json(os.path.join(results_dir, 'summary_cleaned.json'), cleaned_summary)
    _write_csv(os.path.join(results_dir, 'summary_fixed_half.csv'), fixed_summary, summary_fields)
    _save_json(os.path.join(results_dir, 'summary_fixed_half.json'), fixed_summary)

    cleaned_removed = sum(1 for row in run_rows if row['cleaned_out'])
    if cleaned_removed:
        print(f"\n  [数据清洗] cleaned 视图标记剔除 {cleaned_removed} 个 LDA/AC 种子；raw 视图完整保留。")

    _print_anomaly_report(anomalies, sweep_name, param_name, anomaly_threshold)
    print(f"  结果明细: {results_dir}")
    selected = {'fixed_half': fixed_results_plot, 'raw': raw_results_plot,
                'cleaned': cleaned_results_plot}[metric_view]
    plot_sweep_results(param_values, selected, param_name,
                       f'{sweep_name} ({metric_view})', f'{sweep_name}_{metric_view}',
                       metrics_to_plot=['PAoI', 'Q', 'E_BS', 'E_LEO'],
                       output_dir=results_dir)
    return selected


def plot_sweep_results(param_values, results, x_label, title_prefix, filename,
                       metrics_to_plot=None, output_dir='results'):
    if metrics_to_plot is None:
        metrics_to_plot = ['PAoI', 'E_BS', 'E_LEO']
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
            unit_scale = 1e6 if metric == 'Q' else 1.0
            y_data = y_data / unit_scale
            failed = np.array(data['failed'])
            valid_mask = ~failed & ~np.isnan(y_data)
            failed_mask = failed | np.isnan(y_data)
            if np.any(valid_mask):
                yerr = np.array(data.get(f'{metric}_ci95', [0] * len(y_data)), dtype=float)
                yerr = yerr / unit_scale
                ax.errorbar(np.array(param_values)[valid_mask], y_data[valid_mask],
                            yerr=yerr[valid_mask],
                            marker=markers.get(algo_disp, 'o'), label=algo_disp,
                            linestyle='-', capsize=3, linewidth=1.6)
            if np.any(failed_mask):
                ax.scatter(np.array(param_values)[failed_mask], [0] * np.sum(failed_mask),
                           marker='X', color='red', s=150, label=f'{algo_disp} (Failed)', zorder=10)
        if "K" in x_label: ax.set_xscale('log')
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabels[metric])
        ax.grid(True, linestyle='--')
        ax.legend()

    fig.text(0.5, 0.02, '[X] = No usable observations; see failure details',
             ha='center', fontsize=10, style='italic', color='red')
    plt.suptitle(f"{title_prefix}")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n  【生成成功】 {title_prefix} 实验组图表已保存至: {save_path}")
    plt.clf()
    plt.close(fig)


EXPERIMENTS = {
    'Exp1_J': ('J', [4, 6, 8, 10, 12, 14]),
    'Exp2_L': ('L_mean', [8e6, 10e6, 12e6, 14e6, 16e6]),
    'Exp3_fUE': ('f_max_UE', [1e8, 2e8, 4e8, 6e8, 8e8]),
    'Exp4_K': ('K_p', [0.001, 0.01, 0.1, 1, 10, 100]),
    'Exp5_UAV': ('use_uav_relay', [0, 1]),
    'Exp6_Bc': ('B_c', [400e6, 500e6, 600e6, 700e6]),
    'Exp7_Bsat': ('B_sat', [700e6, 800e6, 900e6, 1000e6]),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='LDA parameter sweeps (fresh agents per run)')
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS),
                        default=['Exp1_J', 'Exp2_L'])
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[42, 123, 456, 789, 1000, 2003, 3141, 6283])
    parser.add_argument('--frames', type=int, default=SystemConfig().sim_frames)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--view', choices=['fixed_half', 'raw', 'cleaned'], default='fixed_half',
                        help='fixed_half: primary unfiltered view; others: legacy diagnostics')
    parser.add_argument('--smoke', action='store_true',
                        help='2 BS x 3 UEs, small DNN, 20 frames, 2 seeds, one worker; not thesis results')
    args = parser.parse_args(argv)
    if args.frames < 1 or (args.workers is not None and args.workers < 1):
        parser.error('frames and workers must be positive')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be unique')
    cfg = SystemConfig()
    cfg.sim_frames = args.frames
    if args.smoke:
        cfg.I, cfg.J, cfg.hidden_dim = 2, 3, 32
        cfg.batch_size, cfg.memory_capacity, cfg.train_interval = 4, 16, 1
        cfg.sim_frames = 20
        cfg._update_bandwidth_params()
        args.seeds, args.workers = [42, 123], 1
        runs = [('Smoke_K', 'K_p', [cfg.K_p])]
    else:
        runs = [(name, *EXPERIMENTS[name]) for name in args.experiments]
    algorithms = [('LDA', LDAAgent), ('AC', ACAgent), ('COB', COBAgent), ('MTD', MTDAgent)]
    for name, parameter, values in runs:
        run_experiment_sweep(name, parameter, values, algorithms, cfg,
                             n_workers=args.workers, seeds=args.seeds, metric_view=args.view)


if __name__ == '__main__':
    main()
