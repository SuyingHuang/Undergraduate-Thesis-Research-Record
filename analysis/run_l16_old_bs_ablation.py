"""Diagnose old-BS carry-over service at L_mean=16 Mbit.

This is a matched, fixed-environment experiment.  ``scenario_seed`` fixes the
workload tensor and channel stream; ``policy_seed`` changes only Actor
initialization and replay sampling.  Both variants retain the full LDA1 PAoI
objective.  The sole treatment is the old-BS service policy.

Example smoke run:
    python analysis/run_l16_old_bs_ablation.py --frames 32 \
        --policy-seeds 42 123 --workers 2 --device cpu

Planned diagnostic:
    python analysis/run_l16_old_bs_ablation.py --frames 2048 \
        --policy-seeds 42 123 456 6283 --workers 8 --device auto
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
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lda-l16-old-bs-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

from analysis.run_seed_sensitivity import (
    TRACE_KEYS,
    array_hash,
    make_workloads,
    summarize_history,
    worker_device,
)
from config import SystemConfig


VARIANTS = {
    'legacy': 'legacy',
    'energy_aware': 'energy_aware',
    'budgeted': 'budgeted',
}
DEFAULT_POLICY_SEEDS = [42, 123, 456, 6283]
ENERGY_BUDGET_J = SystemConfig().E_max_BS
OLD_BS_BUDGET_FRACTION = SystemConfig().old_bs_energy_budget_fraction
ENERGY_QUEUE_SLOPE_TOLERANCE = 0.01


def configuration(variant, frames, device):
    """Create two configs that differ only in old-BS service policy."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant: {variant}')
    cfg = SystemConfig()
    cfg.L_mean = 16e6
    cfg.sim_frames = frames
    cfg.dnn_device = device
    cfg.resource_solver = 'coupled'
    cfg.include_baseline_candidates = True
    cfg.audit_baseline_candidates = True
    cfg.paoi_ablation = 'none'
    cfg.old_bs_policy = VARIANTS[variant]
    cfg.progress_log_interval = max(1, min(500, frames // 4))
    cfg.agent_log_interval = max(1, min(500, frames // 4))
    cfg.objective_log_interval = max(1, min(1000, frames // 2))
    cfg.anomaly_snapshot_interval = max(1, min(1000, frames // 2))
    cfg._update_bandwidth_params()
    return cfg


def _tail_slope(values, frames, scale=1.0):
    values = np.asarray(values, dtype=float)[frames // 2:] / scale
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def _task_stem(directory, variant, scenario_seed, policy_seed):
    return Path(directory) / (
        f'{variant}_l16_env{scenario_seed}_policy{policy_seed}')


def worker(task):
    import torch
    from core.agents.lda_agent import LDAAgent
    from main import run_simulation

    variant, policy_seed, scenario_seed, frames, directory, device = task
    torch.set_num_threads(1)
    cfg = configuration(variant, frames, worker_device(device, torch))
    workloads = make_workloads(cfg, frames, scenario_seed)
    stem = _task_stem(directory, variant, scenario_seed, policy_seed)
    started = time.monotonic()

    with stem.with_suffix('.log').open('w', buffering=1) as output, \
            redirect_stdout(output), redirect_stderr(output):
        env, agent = run_simulation(
            cfg, LDAAgent, f'LDA1 old-BS {variant} (L=16 Mbit)',
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

    metrics = summarize_history(env.history, frames)
    metrics['Q_slope_Mbit_per_user_per_frame'] = _tail_slope(
        env.history['Q_total'], frames, 1e6)
    mean_energy_by_bs = np.asarray(
        metrics['mean_actual_energy_J_by_bs'], dtype=float)
    metrics['max_mean_actual_energy_J_by_bs'] = float(
        np.max(mean_energy_by_bs))
    metrics['all_bs_tail_mean_within_budget'] = bool(
        np.all(mean_energy_by_bs <= cfg.E_max_BS))
    metrics['energy_queue_tail_nonincreasing'] = bool(
        metrics['max_energy_queue_slope_J_per_frame'] <= 0.0)
    metrics['energy_queue_tail_stable'] = bool(
        metrics['max_energy_queue_slope_J_per_frame']
        <= ENERGY_QUEUE_SLOPE_TOLERANCE)
    metrics['physical_queue_tail_nonincreasing'] = bool(
        metrics['Q_slope_Mbit_per_user_per_frame'] <= 0.0)

    payload = {
        'variant': variant,
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
    """Return per-variant summaries and same-policy-seed differences."""
    result = {'variants': {}, 'paired_vs_legacy': {}}
    for variant in VARIANTS:
        group = [row for row in rows if row['variant'] == variant]
        if not group:
            continue
        result['variants'][variant] = {
            'n': len(group),
            'PAoI_mean': float(np.mean([
                row['metrics']['PAoI'] for row in group])),
            'Q_Mbit_per_user_mean': float(np.mean([
                row['metrics']['Q_Mbit_per_user'] for row in group])),
            'E_BS_J_per_node_mean': float(np.mean([
                row['metrics']['E_BS_J_per_node'] for row in group])),
            'all_bs_tail_mean_within_budget_count': int(sum(
                row['metrics']['all_bs_tail_mean_within_budget']
                for row in group)),
            'energy_queue_tail_stable_count': int(sum(
                row['metrics'].get(
                    'energy_queue_tail_stable',
                    row['metrics']['max_energy_queue_slope_J_per_frame']
                    <= ENERGY_QUEUE_SLOPE_TOLERANCE)
                for row in group)),
            'physical_queue_tail_nonincreasing_count': int(sum(
                row['metrics']['physical_queue_tail_nonincreasing']
                for row in group)),
        }

    by_variant = {
        variant: {row['policy_seed']: row for row in rows
                  if row['variant'] == variant}
        for variant in VARIANTS
    }
    present_variants = {row['variant'] for row in rows}
    for variant in VARIANTS:
        if variant == 'legacy':
            continue
        if variant not in present_variants:
            continue
        common = sorted(set(by_variant['legacy']) & set(by_variant[variant]))
        result['paired_vs_legacy'][variant] = {}
        for metric in (
                'PAoI', 'Q_Mbit_per_user', 'E_BS_J_per_node',
                'final_max_energy_queue_J',
                'max_energy_queue_slope_J_per_frame',
                'Q_slope_Mbit_per_user_per_frame'):
            differences = [
                by_variant[variant][seed]['metrics'][metric]
                - by_variant['legacy'][seed]['metrics'][metric]
                for seed in common
            ]
            result['paired_vs_legacy'][variant][metric] = {
                'n': len(differences),
                'mean': float(np.mean(differences)) if differences else None,
                'improved_count': int(sum(value < 0 for value in differences)),
                'differences_by_seed': dict(zip(map(str, common), differences)),
            }
    return result


def write_report(directory, result):
    variants = result['variants']
    lines = [
        '# L=16 Mbit old-BS service ablation', '',
        '- All reported variants retain the complete LDA1 PAoI objective.',
        '- The workload and channel stream are matched across variants.',
        '- The environment/workload scenario seed is fixed at 42; uncertainty here is policy-seed uncertainty only.',
        f'- The BS long-term energy budget is {ENERGY_BUDGET_J:.0f} J per node per frame.',
        f'- `budgeted` reserves at most {100 * OLD_BS_BUDGET_FRACTION:.0f}% of that budget for carry-over BS work.',
        '- This is a finite-horizon mechanism diagnostic, not a convergence proof.',
        '',
        f'| Variant | n | PAoI (s) | Q (Mbit/user) | E_BS (J/node) | all BS means <= budget | E-queue slope <= {ENERGY_QUEUE_SLOPE_TOLERANCE:g} J/frame | Q slope <= 0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for variant in VARIANTS:
        item = variants.get(variant)
        if not item:
            continue
        lines.append(
            f"| `{variant}` | {item['n']} | {item['PAoI_mean']:.3f} | "
            f"{item['Q_Mbit_per_user_mean']:.3f} | "
            f"{item['E_BS_J_per_node_mean']:.1f} | "
            f"{item['all_bs_tail_mean_within_budget_count']}/{item['n']} | "
            f"{item['energy_queue_tail_stable_count']}/{item['n']} | "
            f"{item['physical_queue_tail_nonincreasing_count']}/{item['n']} |"
        )
    lines += ['', 'Paired treatment-minus-legacy differences:', '']
    for variant, paired in result['paired_vs_legacy'].items():
        lines += [f'`{variant}`:', '']
        for metric, item in paired.items():
            mean = ('—' if item['mean'] is None else f"{item['mean']:.6g}")
            lines.append(
                f"- `{metric}`: mean {mean}; "
                f"lower in {item['improved_count']}/{item['n']} seeds.")
        lines.append('')
    lines += [
        '',
        'Decision rule:', '',
        '- A treatment passes this finite-horizon screen only when it controls every per-BS energy mean and avoids persistent energy-queue and physical-queue growth.',
        '- A passing treatment still requires longer runs and independent environment seeds before it can replace the production policy.',
        '',
    ]
    (directory / 'analysis.md').write_text('\n'.join(lines))


def write_summary(directory, rows, failures):
    rows = sorted(rows, key=lambda row: (
        row['variant'], row['policy_seed']))
    result = analyze(rows)
    payload = {'runs': rows, 'failures': failures, 'analysis': result}
    (directory / 'summary.json').write_text(json.dumps(payload, indent=2))
    fields = [
        'variant', 'scenario_seed', 'policy_seed', 'frames', 'seconds',
        'training_updates', 'PAoI', 'Q_Mbit_per_user',
        'Q_slope_Mbit_per_user_per_frame', 'E_BS_J_per_node',
        'max_mean_actual_energy_J_by_bs', 'final_max_energy_queue_J',
        'max_energy_queue_slope_J_per_frame',
        'all_bs_tail_mean_within_budget',
        'energy_queue_tail_nonincreasing',
        'energy_queue_tail_stable',
        'physical_queue_tail_nonincreasing',
    ]
    with (directory / 'summary.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row.get(key, row['metrics'].get(key)) for key in fields
            })
    write_report(directory, result)


def source_hashes():
    sources = [ROOT / 'config.py', ROOT / 'main.py', Path(__file__),
               ROOT / 'analysis' / 'run_seed_sensitivity.py']
    sources += list((ROOT / 'core').rglob('*.py'))
    sources += list((ROOT / 'utils').rglob('*.py'))
    sources = sorted(set(sources))
    return sources, {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=2048)
    parser.add_argument('--scenario-seed', type=int, default=42)
    parser.add_argument('--policy-seeds', nargs='+', type=int,
                        default=DEFAULT_POLICY_SEEDS)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS,
                        default=list(VARIANTS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    if args.frames < 4 or args.workers < 1:
        parser.error('frames must be >= 4 and workers must be >= 1')
    if len(set(args.policy_seeds)) != len(args.policy_seeds):
        parser.error('policy seeds must be unique')
    if len(set(args.variants)) != len(args.variants):
        parser.error('variants must be unique')

    sources, hashes = source_hashes()
    arguments = {key: value for key, value in vars(args).items()
                 if key != 'resume'}
    directory = (args.resume.resolve() if args.resume else
                 ROOT / 'results' / 'l16_old_bs_ablation' /
                 datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    manifest = {
        'arguments': arguments,
        'source_hashes': hashes,
        'treatment': ('old_bs_policy only; full LDA1 PAoI retained; '
                      'budgeted uses the configured old-work energy share'),
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

    tasks = [
        (variant, policy_seed, args.scenario_seed, args.frames,
         str(directory), args.device)
        for policy_seed in args.policy_seeds
        for variant in args.variants
    ]
    rows, failures, remaining = [], [], []
    for task in tasks:
        variant, policy_seed, scenario_seed = task[:3]
        stem = _task_stem(directory, variant, scenario_seed, policy_seed)
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
                    f"{row['variant']} policy={row['policy_seed']} "
                    f"PAoI={row['metrics']['PAoI']:.4f} "
                    f"Emax={row['metrics']['final_max_energy_queue_J']:.1f} "
                    f"seconds={row['seconds']:.1f}", flush=True)
            except Exception as exc:
                failures.append({
                    'task': pending[future][:-2], 'error': repr(exc)})
                print('FAILED', failures[-1], flush=True)
            write_summary(directory, rows, failures)
    if failures:
        raise SystemExit(1)
    hashes_by_policy = {}
    for row in rows:
        hashes_by_policy.setdefault(row['policy_seed'], set()).add(
            row['scenario_hash'])
    if any(len(values) != 1 for values in hashes_by_policy.values()):
        raise RuntimeError('paired variants did not use identical workloads')
    return directory


if __name__ == '__main__':
    main()
