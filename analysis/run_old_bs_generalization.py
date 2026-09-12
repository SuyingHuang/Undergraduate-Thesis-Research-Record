"""Test old-BS energy policies across workloads and environments.

The independent replication unit is an environment seed. Policy seeds are
nested within each environment seed and are not counted as independent
workload replications. Every treatment in a matched block uses the same task
tensor and channel seed.

Smoke example:
    python analysis/run_old_bs_generalization.py --frames 8 \
        --loads-mbit 12 --scenario-seeds 104729 --policy-seeds 42 \
        --budget-fractions 0.25 0.5 --workers 2 --device cpu
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lda-old-bs-generalization-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

from analysis.run_l16_old_bs_ablation import (
    ENERGY_QUEUE_SLOPE_TOLERANCE,
    _tail_slope,
)
from analysis.run_seed_sensitivity import (
    TRACE_KEYS,
    array_hash,
    make_workloads,
    summarize_history,
    worker_device,
)
from config import SystemConfig


def treatment_name(policy, fraction=None, joint_grid_points=9):
    if policy == 'legacy':
        return 'legacy'
    if policy == 'joint_dpp':
        return f'joint_dpp_g{int(joint_grid_points)}'
    digits = f'{float(fraction):.6f}'.rstrip('0').rstrip('.').replace('.', 'p')
    return f'budgeted_{digits}'


def configuration(load_mbit, task_std_mbit, users_per_bs, frames, device,
                  policy, fraction=None, joint_grid_points=9):
    cfg = SystemConfig()
    cfg.L_mean = float(load_mbit) * 1e6
    cfg.L_std = float(task_std_mbit) * 1e6
    cfg.J = int(users_per_bs)
    cfg.sim_frames = int(frames)
    cfg.dnn_device = device
    cfg.resource_solver = 'coupled'
    cfg.include_baseline_candidates = True
    cfg.audit_baseline_candidates = True
    cfg.paoi_ablation = 'none'
    cfg.old_bs_policy = policy
    if policy == 'budgeted':
        cfg.old_bs_energy_budget_fraction = float(fraction)
    if policy == 'joint_dpp':
        cfg.joint_dpp_old_frequency_grid_points = int(joint_grid_points)
    cfg.progress_log_interval = max(1, min(500, frames // 4))
    cfg.agent_log_interval = max(1, min(500, frames // 4))
    cfg.objective_log_interval = max(1, min(1000, frames // 2))
    cfg.anomaly_snapshot_interval = max(1, min(1000, frames // 2))
    cfg._update_bandwidth_params()
    return cfg


def case_name(load_mbit, task_std_mbit, users_per_bs):
    load = f'{float(load_mbit):g}'.replace('.', 'p')
    std = f'{float(task_std_mbit):g}'.replace('.', 'p')
    return f'L{load}_std{std}_J{int(users_per_bs)}'


def task_stem(directory, case, treatment, scenario_seed, policy_seed):
    return Path(directory) / (
        f'{case}_{treatment}_env{scenario_seed}_policy{policy_seed}')


def workload_capacity_envelope(cfg, workloads):
    """Finite-sample old-work budget needed to clear all raw arrivals.

    For a workload vector w completed in one frame under proportional
    allocation, E = kappa*phi^3*sum(w_j^3)/tau^2. Any BS-offloaded subset has
    no larger cube sum or total workload, so the all-arrival calculation is a
    conservative envelope for that frame (not a stochastic tail guarantee).
    """
    workloads = np.asarray(workloads, dtype=float)
    beta_required = (
        cfg.kappa1 * cfg.phi ** 3 * np.sum(workloads ** 3, axis=-1)
        / (cfg.tau ** 2 * cfg.E_max_BS)
    )
    frequency_required = cfg.phi * np.sum(workloads, axis=-1) / cfg.tau
    return {
        'beta_required_all_arrivals_mean': float(np.mean(beta_required)),
        'beta_required_all_arrivals_p95': float(np.quantile(
            beta_required, 0.95)),
        'beta_required_all_arrivals_p99': float(np.quantile(
            beta_required, 0.99)),
        'beta_required_all_arrivals_max': float(np.max(beta_required)),
        'frequency_required_all_arrivals_p99_Hz': float(np.quantile(
            frequency_required, 0.99)),
        'frequency_required_all_arrivals_max_Hz': float(np.max(
            frequency_required)),
        'all_arrivals_frequency_feasible_fraction': float(np.mean(
            frequency_required <= cfg.f_max_BS)),
    }


def _safe_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def feedback_metrics(history, frames):
    """Return observable implications of the cross-frame model.

    Correlations and the pooled AR(1) Jacobian are diagnostics, not causal
    estimates: workload and policy decisions remain possible confounders.
    """
    required = (
        'energy_old_bs_by_bs', 'energy_new_bs_by_bs',
        'service_old_bs_by_bs', 'service_new_bs_by_bs',
        'old_bs_occupied_by_bs', 'energy_queue_by_bs',
        'bs_residual_by_bs',
    )
    if any(key not in history or not len(history[key]) for key in required):
        return {}
    start = frames // 2
    old_e = np.asarray(history['energy_old_bs_by_bs'], float)
    new_e = np.asarray(history['energy_new_bs_by_bs'], float)
    old_s = np.asarray(history['service_old_bs_by_bs'], float)
    new_s = np.asarray(history['service_new_bs_by_bs'], float)
    occupied = np.asarray(history['old_bs_occupied_by_bs'], float)
    z_after = np.asarray(history['energy_queue_by_bs'], float)
    q_after = np.asarray(history['bs_residual_by_bs'], float)
    z_before = np.vstack([np.zeros((1, z_after.shape[1])), z_after[:-1]])
    delta_z = z_after - z_before

    output = {
        'E_old_BS_J_per_node': float(np.mean(old_e[start:])),
        'E_new_BS_J_per_node': float(np.mean(new_e[start:])),
        'old_service_Mbit_per_node': float(np.mean(old_s[start:]) / 1e6),
        'new_service_Mbit_per_node': float(np.mean(new_s[start:]) / 1e6),
        'old_occupied_s_per_node': float(np.mean(occupied[start:])),
        'corr_old_energy_delta_energy_queue': _safe_corr(
            old_e[start:], delta_z[start:]),
        'corr_energy_queue_next_new_service': _safe_corr(
            z_after[start:-1], new_s[start + 1:]),
        'corr_old_occupancy_new_service': _safe_corr(
            occupied[start:], new_s[start:]),
    }
    old_frequency = np.asarray(history.get(
        'old_bs_aggregate_frequency_by_bs', []), float)
    if old_frequency.size and np.any(np.isfinite(old_frequency[start:])):
        output['old_bs_aggregate_frequency_Hz_per_node'] = float(
            np.nanmean(old_frequency[start:]))

    # This standardized pooled linearization is only a screening proxy because
    # it does not condition on every exogenous state.
    if len(q_after) - start >= 4:
        previous = np.column_stack((
            q_after[start:-1].reshape(-1),
            z_after[start:-1].reshape(-1),
        ))
        current = np.column_stack((
            q_after[start + 1:].reshape(-1),
            z_after[start + 1:].reshape(-1),
        ))
        combined = np.vstack((previous, current))
        scale = np.std(combined, axis=0)
        if np.all(scale > 1e-12):
            center = np.mean(combined, axis=0)
            x = (previous - center) / scale
            y = (current - center) / scale
            design = np.column_stack((np.ones(len(x)), x))
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0][1:]
            jacobian = coefficients.T
            output['feedback_jacobian_ols'] = jacobian.tolist()
            output['feedback_spectral_radius_ols'] = float(
                np.max(np.abs(np.linalg.eigvals(jacobian))))
    return output


def worker(task):
    import torch
    from core.agents.lda_agent import LDAAgent
    from main import run_simulation

    (load_mbit, task_std_mbit, users_per_bs, policy, fraction,
     policy_seed, scenario_seed, frames, directory, device,
     joint_grid_points) = task
    torch.set_num_threads(1)
    cfg = configuration(
        load_mbit, task_std_mbit, users_per_bs, frames,
        worker_device(device, torch), policy, fraction, joint_grid_points)
    workloads = make_workloads(cfg, frames, scenario_seed)
    case = case_name(load_mbit, task_std_mbit, users_per_bs)
    treatment = treatment_name(policy, fraction, joint_grid_points)
    stem = task_stem(directory, case, treatment, scenario_seed, policy_seed)
    started = time.monotonic()

    with stem.with_suffix('.log').open('w', buffering=1) as output, \
            redirect_stdout(output), redirect_stderr(output):
        env, agent = run_simulation(
            cfg, LDAAgent, f'LDA1 old-BS generalization ({case}/{treatment})',
            preset_L=workloads, seed=scenario_seed, policy_seed=policy_seed,
        )

    arrays = {
        key: np.asarray(env.history[key])
        for key in TRACE_KEYS if key in env.history and len(env.history[key])
    }
    arrays['workload'] = workloads
    np.savez_compressed(stem.with_suffix('.npz'), **arrays)

    metrics = summarize_history(env.history, frames)
    metrics.update(feedback_metrics(env.history, frames))
    metrics.update(workload_capacity_envelope(cfg, workloads))
    metrics['Q_slope_Mbit_per_user_per_frame'] = _tail_slope(
        env.history['Q_total'], frames, 1e6)
    energy_by_bs = np.asarray(metrics['mean_actual_energy_J_by_bs'])
    metrics['all_bs_tail_mean_within_budget'] = bool(
        np.all(energy_by_bs <= cfg.E_max_BS))
    metrics['energy_queue_tail_stable'] = bool(
        metrics['max_energy_queue_slope_J_per_frame']
        <= ENERGY_QUEUE_SLOPE_TOLERANCE)
    metrics['physical_queue_tail_nonincreasing'] = bool(
        metrics['Q_slope_Mbit_per_user_per_frame'] <= 0.0)
    metrics['screen_pass'] = bool(
        metrics['all_bs_tail_mean_within_budget']
        and metrics['energy_queue_tail_stable']
        and metrics['physical_queue_tail_nonincreasing'])

    payload = {
        'case': case,
        'load_mbit': load_mbit,
        'task_std_mbit': task_std_mbit,
        'users_per_bs': users_per_bs,
        'treatment': treatment,
        'policy': policy,
        'budget_fraction': fraction,
        'joint_dpp_old_frequency_grid_points': (
            joint_grid_points if policy == 'joint_dpp' else None),
        'policy_seed': policy_seed,
        'scenario_seed': scenario_seed,
        'scenario_hash': array_hash(workloads),
        'frames': frames,
        'seconds': time.monotonic() - started,
        'training_updates': len(getattr(agent, 'loss_history', [])),
        'config': vars(cfg),
        'metrics': metrics,
    }
    stem.with_suffix('.json').write_text(json.dumps(payload, indent=2))
    return payload


def analyze(rows):
    """Aggregate policy seeds within environments, then environments."""
    output = {'groups': {}, 'paired_vs_legacy': {}}
    group_keys = sorted({(row['case'], row['treatment']) for row in rows})
    for case, treatment in group_keys:
        group = [row for row in rows
                 if row['case'] == case and row['treatment'] == treatment]
        environment_means = []
        for scenario_seed in sorted({row['scenario_seed'] for row in group}):
            env_rows = [row for row in group
                        if row['scenario_seed'] == scenario_seed]
            environment_means.append({
                'scenario_seed': scenario_seed,
                'n_policy_seeds': len(env_rows),
                'PAoI': float(np.mean([
                    row['metrics']['PAoI'] for row in env_rows])),
                'Q_Mbit_per_user': float(np.mean([
                    row['metrics']['Q_Mbit_per_user'] for row in env_rows])),
                'E_BS_J_per_node': float(np.mean([
                    row['metrics']['E_BS_J_per_node'] for row in env_rows])),
                'all_runs_pass': all(
                    row['metrics']['screen_pass'] for row in env_rows),
            })
        output['groups'][f'{case}/{treatment}'] = {
            'n_environment_seeds': len(environment_means),
            'n_runs': len(group),
            'PAoI_environment_mean': float(np.mean([
                item['PAoI'] for item in environment_means])),
            'Q_environment_mean_Mbit_per_user': float(np.mean([
                item['Q_Mbit_per_user'] for item in environment_means])),
            'E_BS_environment_mean_J_per_node': float(np.mean([
                item['E_BS_J_per_node'] for item in environment_means])),
            'passing_environment_count': int(sum(
                item['all_runs_pass'] for item in environment_means)),
            'environments': environment_means,
        }

    by_key = {
        (row['case'], row['treatment'], row['scenario_seed'],
         row['policy_seed']): row for row in rows
    }
    for row in rows:
        if row['treatment'] == 'legacy':
            continue
        legacy = by_key.get((
            row['case'], 'legacy', row['scenario_seed'], row['policy_seed']))
        if legacy is None:
            continue
        comparison = output['paired_vs_legacy'].setdefault(
            f"{row['case']}/{row['treatment']}", [])
        comparison.append({
            'scenario_seed': row['scenario_seed'],
            'policy_seed': row['policy_seed'],
            'PAoI_difference': (
                row['metrics']['PAoI'] - legacy['metrics']['PAoI']),
            'Q_difference_Mbit_per_user': (
                row['metrics']['Q_Mbit_per_user']
                - legacy['metrics']['Q_Mbit_per_user']),
            'E_BS_difference_J_per_node': (
                row['metrics']['E_BS_J_per_node']
                - legacy['metrics']['E_BS_J_per_node']),
        })
    return output


def write_summary(directory, rows, failures):
    rows = sorted(rows, key=lambda row: (
        row['case'], row['treatment'], row['scenario_seed'],
        row['policy_seed']))
    result = analyze(rows)
    (directory / 'summary.json').write_text(json.dumps({
        'runs': rows, 'failures': failures, 'analysis': result,
    }, indent=2))
    fields = [
        'case', 'treatment', 'budget_fraction',
        'joint_dpp_old_frequency_grid_points', 'scenario_seed',
        'policy_seed', 'frames', 'seconds', 'PAoI', 'Q_Mbit_per_user',
        'Q_slope_Mbit_per_user_per_frame', 'E_BS_J_per_node',
        'E_old_BS_J_per_node', 'E_new_BS_J_per_node',
        'old_service_Mbit_per_node', 'new_service_Mbit_per_node',
        'old_occupied_s_per_node',
        'old_bs_aggregate_frequency_Hz_per_node',
        'max_energy_queue_slope_J_per_frame',
        'feedback_spectral_radius_ols',
        'beta_required_all_arrivals_p95',
        'beta_required_all_arrivals_p99',
        'beta_required_all_arrivals_max',
        'all_arrivals_frequency_feasible_fraction', 'screen_pass',
    ]
    with (directory / 'summary.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row.get(key, row['metrics'].get(key)) for key in fields
            })


def source_hashes():
    sources = [ROOT / 'config.py', ROOT / 'main.py', Path(__file__),
               ROOT / 'analysis' / 'run_seed_sensitivity.py',
               ROOT / 'analysis' / 'run_l16_old_bs_ablation.py']
    sources += list((ROOT / 'core').rglob('*.py'))
    sources += list((ROOT / 'utils').rglob('*.py'))
    sources = sorted(set(sources))
    return sources, {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--loads-mbit', nargs='+', type=float, required=True)
    parser.add_argument('--task-std-mbit', type=float, default=3.0)
    parser.add_argument('--users-per-bs', nargs='+', type=int, default=[10])
    parser.add_argument('--scenario-seeds', nargs='+', type=int, required=True)
    parser.add_argument('--policy-seeds', nargs='+', type=int, required=True)
    parser.add_argument('--budget-fractions', nargs='+', type=float,
                        required=True)
    parser.add_argument('--include-joint-dpp', action='store_true')
    parser.add_argument('--joint-dpp-grid-points', type=int, default=9)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    if args.frames < 4 or args.workers < 1:
        parser.error('frames must be >= 4 and workers must be >= 1')
    for name in ('loads_mbit', 'users_per_bs', 'scenario_seeds',
                 'policy_seeds', 'budget_fractions'):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f'{name.replace("_", "-")} must be unique')
    if any(value <= 0 for value in args.loads_mbit):
        parser.error('loads-mbit values must be positive')
    if args.task_std_mbit < 0:
        parser.error('task-std-mbit must be nonnegative')
    if any(value < 1 for value in args.users_per_bs):
        parser.error('users-per-bs values must be positive')
    if any(not 0 < value <= 1 for value in args.budget_fractions):
        parser.error('budget-fractions must lie in (0, 1]')
    if args.joint_dpp_grid_points < 2:
        parser.error('joint-dpp-grid-points must be >= 2')

    sources, hashes = source_hashes()
    arguments = {key: value for key, value in vars(args).items()
                 if key != 'resume'}
    directory = (args.resume.resolve() if args.resume else
                 ROOT / 'results' / 'old_bs_generalization' /
                 datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    manifest = {
        'arguments': arguments,
        'source_hashes': hashes,
        'independent_replication_unit': 'scenario_seed',
        'nested_randomness': 'policy_seed within scenario_seed',
        'selection_rule': (
            'constraint feasibility first; among treatments passing every '
            'screening environment, minimize mean environment-level PAoI'),
    }
    if args.resume:
        previous = json.loads((directory / 'manifest.json').read_text())
        if previous != manifest:
            parser.error('resume requires identical arguments and source files')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        with zipfile.ZipFile(directory / 'source.zip', 'w',
                            zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(path, str(path.relative_to(ROOT)))

    policies = [('legacy', None)] + [
        ('budgeted', fraction) for fraction in args.budget_fractions]
    if args.include_joint_dpp:
        policies.append(('joint_dpp', None))
    tasks = [
        (load, args.task_std_mbit, users, policy, fraction, policy_seed,
         scenario_seed, args.frames, str(directory), args.device,
         args.joint_dpp_grid_points)
        for load in args.loads_mbit
        for users in args.users_per_bs
        for scenario_seed in args.scenario_seeds
        for policy_seed in args.policy_seeds
        for policy, fraction in policies
    ]
    rows, failures, remaining = [], [], []
    for task in tasks:
        (load, std, users, policy, fraction, policy_seed,
         scenario_seed) = task[:7]
        stem = task_stem(
            directory, case_name(load, std, users),
            treatment_name(policy, fraction, args.joint_dpp_grid_points),
            scenario_seed, policy_seed)
        if (args.resume and stem.with_suffix('.json').exists()
                and stem.with_suffix('.npz').exists()):
            rows.append(json.loads(stem.with_suffix('.json').read_text()))
        else:
            remaining.append(task)

    print(directory, flush=True)
    print(f'Reusing {len(rows)} runs; scheduling {len(remaining)}', flush=True)
    write_summary(directory, rows, failures)
    with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(worker, task): task for task in remaining}
        for future in as_completed(pending):
            try:
                row = future.result()
                rows.append(row)
                print(
                    f"{len(rows) + len(failures)}/{len(tasks)} "
                    f"{row['case']} {row['treatment']} "
                    f"env={row['scenario_seed']} policy={row['policy_seed']} "
                    f"PAoI={row['metrics']['PAoI']:.4f} "
                    f"pass={row['metrics']['screen_pass']} "
                    f"seconds={row['seconds']:.1f}", flush=True)
            except Exception as exc:
                failures.append({'task': pending[future][:-3],
                                 'error': repr(exc)})
                print('FAILED', failures[-1], flush=True)
            write_summary(directory, rows, failures)
    if failures:
        raise SystemExit(1)

    hashes_by_block = {}
    for row in rows:
        key = (row['case'], row['scenario_seed'], row['policy_seed'])
        hashes_by_block.setdefault(key, set()).add(row['scenario_hash'])
    if any(len(values) != 1 for values in hashes_by_block.values()):
        raise RuntimeError('matched treatments did not use identical workloads')
    return directory


if __name__ == '__main__':
    main()
