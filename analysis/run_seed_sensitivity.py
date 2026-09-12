"""Separate LDA1 policy randomness from a fixed environment scenario.

The workload tensor and channel stream are fixed by ``--scenario-seed``.
Only DNN initialization and replay-buffer sampling vary with ``--policy-seeds``.
Runs are checkpointed individually and can be resumed safely.

Example smoke run:
    python analysis/run_seed_sensitivity.py --frames 32 --cases j4 l16 \
        --policy-seeds 42 3141 --workers 2

Example full run:
    python analysis/run_seed_sensitivity.py --frames 4096 --workers 4
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
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lda-seed-sensitivity-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

from config import SystemConfig


CASES = {
    'j4': {'J': 4},
    'j8': {'J': 8},
    'l10': {'L_mean': 10e6},
    'l16': {'L_mean': 16e6},
}
DEFAULT_POLICY_SEEDS = [
    42, 123, 456, 789, 1000, 2003, 3141, 6283,
    17, 29, 61, 97, 211, 307, 401, 503, 601, 701, 809, 907,
]
TRACE_KEYS = (
    'Cost', 'Q_total', 'Q_bs', 'Q_sat', 'E_virt_bs', 'E_sat_total',
    'E_queue_bs_max', 'delta_t', 'decision_counts_by_bs',
    'bs_residual_by_bs', 'energy_queue_by_bs', 'energy_actual_by_bs',
    'energy_old_bs_by_bs', 'energy_new_bs_by_bs',
    'service_old_bs_by_bs', 'service_new_bs_by_bs',
    'old_bs_occupied_by_bs', 'old_bs_aggregate_frequency_by_bs',
    'policy_prob_mean_by_bs', 'candidate_baseline_improvement',
    'candidate_baseline_selected',
)


def configuration(case, frames, device):
    cfg = SystemConfig()
    for key, value in CASES[case].items():
        setattr(cfg, key, value)
    cfg.sim_frames = frames
    cfg.dnn_device = device
    cfg.resource_solver = 'coupled'
    cfg.include_baseline_candidates = True
    cfg.audit_baseline_candidates = True
    cfg.paoi_ablation = 'none'
    cfg.old_bs_policy = 'legacy'
    # Long runs retain enough heartbeats for diagnosis without spending time
    # formatting hundreds of repetitive progress lines per worker.
    cfg.progress_log_interval = 500
    cfg.agent_log_interval = 500
    cfg.objective_log_interval = 1000
    cfg.anomaly_snapshot_interval = 1000
    cfg._update_bandwidth_params()
    return cfg


def make_workloads(cfg, frames, scenario_seed):
    # RandomState keeps this stream independent of policy and worker ordering.
    rng = np.random.RandomState(scenario_seed)
    return np.maximum(
        0.0,
        cfg.L_mean + rng.normal(0.0, cfg.L_std, (frames, cfg.I, cfg.J)),
    )


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode('ascii'))
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def worker_device(requested, torch_module):
    requested = str(requested).strip().lower()
    if requested != 'auto' or not torch_module.cuda.is_available():
        return requested
    count = torch_module.cuda.device_count()
    identity = multiprocessing.current_process()._identity
    worker_number = identity[0] - 1 if identity else 0
    return f'cuda:{worker_number % count}'


def _tail_mean(values, start):
    array = np.asarray(values, dtype=float)
    return float(np.mean(array[start:]))


def _slope(values, start):
    array = np.asarray(values, dtype=float)[start:]
    if len(array) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(array)), array, 1)[0])


def summarize_history(history, frames):
    start = frames // 2
    counts = np.asarray(history['decision_counts_by_bs'], dtype=float)[start:]
    residual = np.asarray(history['bs_residual_by_bs'], dtype=float)[start:] / 1e6
    energy_queue = np.asarray(history['energy_queue_by_bs'], dtype=float)
    actual_energy = np.asarray(history['energy_actual_by_bs'], dtype=float)[start:]
    probabilities = np.asarray(history['policy_prob_mean_by_bs'], dtype=float)[start:]
    delta = np.asarray(history['delta_t'], dtype=float)

    metrics = {
        'PAoI': _tail_mean(history['Cost'], start),
        'Q_Mbit_per_user': _tail_mean(history['Q_total'], start) / 1e6,
        'E_BS_J_per_node': _tail_mean(history['E_virt_bs'], start),
        'final_delta': float(delta[-1]),
        'min_delta': float(np.min(delta)),
        'final_max_energy_queue_J': float(np.max(energy_queue[-1])),
        'max_energy_queue_J': float(np.max(energy_queue)),
        'max_energy_queue_slope_J_per_frame': _slope(
            np.max(energy_queue, axis=1), start),
        'mean_local_count': float(np.mean(counts[:, :, 0])),
        'mean_bs_count': float(np.mean(counts[:, :, 1])),
        'mean_sat_count': float(np.mean(counts[:, :, 2])),
        'mean_bs_residual_Mbit': float(np.mean(residual)),
        'mean_policy_probability': float(np.nanmean(probabilities)),
        'mean_actual_energy_J_by_bs': np.mean(actual_energy, axis=0).tolist(),
        'final_energy_queue_J_by_bs': energy_queue[-1].tolist(),
        'mean_decision_counts_by_bs': np.mean(counts, axis=0).tolist(),
        'mean_residual_Mbit_by_bs': np.mean(residual, axis=0).tolist(),
    }
    for key in ('candidate_baseline_improvement', 'candidate_baseline_selected'):
        if key in history and history[key]:
            values = np.asarray(history[key], dtype=float)[start:]
            metrics[key + '_mean'] = float(np.mean(values))
            metrics[key + '_positive_fraction'] = float(np.mean(values > 1e-12))
    return metrics


def worker(task):
    import torch
    from core.agents.lda_agent import LDAAgent
    from main import run_simulation

    case, policy_seed, scenario_seed, frames, directory, device = task
    torch.set_num_threads(1)
    cfg = configuration(case, frames, worker_device(device, torch))
    workloads = make_workloads(cfg, frames, scenario_seed)
    scenario_hash = array_hash(workloads)
    stem = Path(directory) / f'{case}_env{scenario_seed}_policy{policy_seed}'
    started = time.monotonic()

    with stem.with_suffix('.log').open('w', buffering=1) as output, \
            redirect_stdout(output), redirect_stderr(output):
        env, agent = run_simulation(
            cfg, LDAAgent, f'LDA1 seed sensitivity ({case})',
            preset_L=workloads, seed=scenario_seed, policy_seed=policy_seed,
        )

    arrays = {
        key: np.asarray(env.history[key])
        for key in TRACE_KEYS if key in env.history and env.history[key]
    }
    arrays['workload'] = workloads
    for index, losses in enumerate(env.history.get('Loss_per_BS', [])):
        arrays[f'Loss_BS_{index}'] = np.asarray(losses)
    np.savez_compressed(stem.with_suffix('.npz'), **arrays)

    payload = {
        'case': case,
        'policy_seed': policy_seed,
        'scenario_seed': scenario_seed,
        'scenario_hash': scenario_hash,
        'frames': frames,
        'seconds': time.monotonic() - started,
        'training_updates': len(getattr(agent, 'loss_history', [])),
        'config': vars(cfg),
        'metrics': summarize_history(env.history, frames),
    }
    stem.with_suffix('.json').write_text(json.dumps(payload, indent=2))
    return payload


def aggregate(rows):
    output = {}
    for case in sorted({row['case'] for row in rows}):
        group = sorted(
            (row for row in rows if row['case'] == case),
            key=lambda row: row['policy_seed'],
        )
        paoi = np.asarray([row['metrics']['PAoI'] for row in group])
        energy = np.asarray([
            row['metrics']['final_max_energy_queue_J'] for row in group
        ])
        correlation = None
        if len(group) > 1 and np.std(paoi) > 0 and np.std(energy) > 0:
            correlation = float(np.corrcoef(paoi, energy)[0, 1])
        output[case] = {
            'n': len(group),
            'PAoI_mean': float(np.mean(paoi)),
            'PAoI_sample_std': float(np.std(paoi, ddof=1)) if len(group) > 1 else None,
            'PAoI_min': float(np.min(paoi)),
            'PAoI_max': float(np.max(paoi)),
            'PAoI_energy_queue_correlation': correlation,
            'runs': [
                {
                    'policy_seed': row['policy_seed'],
                    'PAoI': row['metrics']['PAoI'],
                    'final_max_energy_queue_J': row['metrics']['final_max_energy_queue_J'],
                    'final_delta': row['metrics']['final_delta'],
                    'mean_bs_count': row['metrics']['mean_bs_count'],
                    'mean_sat_count': row['metrics']['mean_sat_count'],
                }
                for row in group
            ],
        }
    return output


def write_summary(directory, rows, failures):
    rows = sorted(rows, key=lambda row: (row['case'], row['policy_seed']))
    payload = {'runs': rows, 'failures': failures, 'aggregate': aggregate(rows)}
    (directory / 'summary.json').write_text(json.dumps(payload, indent=2))
    fields = [
        'case', 'scenario_seed', 'policy_seed', 'frames', 'seconds',
        'training_updates', 'PAoI', 'Q_Mbit_per_user', 'E_BS_J_per_node',
        'final_delta', 'mean_bs_count', 'mean_sat_count',
        'mean_bs_residual_Mbit', 'final_max_energy_queue_J',
        'max_energy_queue_slope_J_per_frame',
    ]
    with (directory / 'summary.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key, row['metrics'].get(key)) for key in fields}
            writer.writerow(flat)


def source_hashes():
    sources = [ROOT / 'config.py', ROOT / 'main.py', Path(__file__)]
    sources += list((ROOT / 'core').rglob('*.py'))
    sources += list((ROOT / 'utils').rglob('*.py'))
    return sources, {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=4096)
    parser.add_argument('--scenario-seed', type=int, default=42)
    parser.add_argument('--policy-seeds', nargs='+', type=int,
                        default=DEFAULT_POLICY_SEEDS)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cpu',
                        help='cpu, auto, cuda, or cuda:N; use one worker per GPU')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.frames < 4 or args.workers < 1:
        parser.error('frames must be >= 4 and workers must be >= 1')
    if len(set(args.policy_seeds)) != len(args.policy_seeds):
        parser.error('policy seeds must be unique')

    sources, hashes = source_hashes()
    arguments = {key: value for key, value in vars(args).items() if key != 'resume'}
    directory = (args.resume.resolve() if args.resume else
                 ROOT / 'results' / 'seed_sensitivity' /
                 datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    manifest = {
        'arguments': arguments,
        'source_hashes': hashes,
        'note': ('Workloads and channels share one fixed scenario seed; only '
                 'DNN initialization and replay sampling use policy_seed.'),
    }
    if args.resume:
        previous = json.loads((directory / 'manifest.json').read_text())
        if previous != manifest:
            parser.error('resume requires identical arguments and source files')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        with zipfile.ZipFile(directory / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(path, str(path.relative_to(ROOT)))

    # Interleave cases so an interrupted first pass still contains evidence
    # from every stress point instead of only the first case.
    tasks = [
        (case, seed, args.scenario_seed, args.frames, str(directory), args.device)
        for seed in args.policy_seeds for case in args.cases
    ]
    rows, failures, remaining = [], [], []
    for task in tasks:
        case, policy_seed, scenario_seed, _, _, _ = task
        stem = directory / f'{case}_env{scenario_seed}_policy{policy_seed}'
        if args.resume and stem.with_suffix('.json').exists() and stem.with_suffix('.npz').exists():
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
                    f"{len(rows) + len(failures)}/{len(tasks)} {row['case']} "
                    f"policy={row['policy_seed']} PAoI={row['metrics']['PAoI']:.4f} "
                    f"Emax={row['metrics']['final_max_energy_queue_J']:.1f} "
                    f"seconds={row['seconds']:.1f}", flush=True,
                )
            except Exception as exc:
                failures.append({'task': pending[future][:-2], 'error': repr(exc)})
                print('FAILED', failures[-1], flush=True)
            write_summary(directory, rows, failures)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
