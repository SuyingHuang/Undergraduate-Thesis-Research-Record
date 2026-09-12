"""Run the first three LDA1 instability ablations under a fixed scenario.

The matrix intentionally contains five cells:

* no_train: J=4 and L_mean=16 Mbit, online replay training disabled;
* fixed_delta: J=4 and L_mean=16 Mbit, training enabled, delta=0.5;
* exhaustive_j4: J=4 only, all 2**J per-BS candidates enumerated.

Workloads and channel randomness use one scenario seed.  Policy seeds affect
only actor initialization and replay sampling.  Every run is checkpointed as
JSON + NPZ and ``--resume`` validates both arguments and source hashes.
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
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lda-instability-ablation-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

from analysis.run_seed_sensitivity import (
    DEFAULT_POLICY_SEEDS, TRACE_KEYS, array_hash, configuration,
    make_workloads, summarize_history, worker_device,
)


VARIANT_CASES = {
    'no_train': ('j4', 'l16'),
    'fixed_delta': ('j4', 'l16'),
    'exhaustive_j4': ('j4',),
}


def ablation_configuration(variant, case, frames, device):
    if case not in VARIANT_CASES[variant]:
        raise ValueError(f'{variant} does not support case {case}')
    cfg = configuration(case, frames, device)
    if variant == 'no_train':
        cfg.online_training_enabled = False
        cfg.delta_init = cfg.delta_min = cfg.delta_max = 0.5
    elif variant == 'fixed_delta':
        cfg.online_training_enabled = True
        cfg.delta_init = cfg.delta_min = cfg.delta_max = 0.5
    elif variant == 'exhaustive_j4':
        cfg.online_training_enabled = True
        cfg.candidate_mode = 'exhaustive_per_bs'
        cfg.max_exhaustive_candidate_bits = 4
    cfg.ablation_variant = variant
    return cfg


def task_stem(directory, variant, case, scenario_seed, policy_seed):
    return Path(directory) / (
        f'{variant}_{case}_env{scenario_seed}_policy{policy_seed}')


def worker(task):
    import torch
    from core.agents.lda_agent import LDAAgent
    from main import run_simulation

    variant, case, policy_seed, scenario_seed, frames, directory, device = task
    torch.set_num_threads(1)
    cfg = ablation_configuration(
        variant, case, frames, worker_device(device, torch))
    workloads = make_workloads(cfg, frames, scenario_seed)
    stem = task_stem(directory, variant, case, scenario_seed, policy_seed)
    started = time.monotonic()

    with stem.with_suffix('.log').open('w', buffering=1) as output, \
            redirect_stdout(output), redirect_stderr(output):
        env, agent = run_simulation(
            cfg, LDAAgent, f'LDA1 {variant} ({case})',
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
        'variant': variant,
        'case': case,
        'policy_seed': policy_seed,
        'scenario_seed': scenario_seed,
        'scenario_hash': array_hash(workloads),
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
    cells = sorted({(row['variant'], row['case']) for row in rows})
    for variant, case in cells:
        group = [row for row in rows
                 if row['variant'] == variant and row['case'] == case]
        paoi = np.asarray([row['metrics']['PAoI'] for row in group])
        energy = np.asarray([
            row['metrics']['final_max_energy_queue_J'] for row in group])
        output[f'{variant}/{case}'] = {
            'n': len(group),
            'PAoI_mean': float(np.mean(paoi)),
            'PAoI_sample_std': (
                float(np.std(paoi, ddof=1)) if len(group) > 1 else None),
            'PAoI_min': float(np.min(paoi)),
            'PAoI_max': float(np.max(paoi)),
            'energy_runaway_count': int(np.sum(energy > 1800.0)),
        }
    return output


def write_summary(directory, rows, failures):
    rows = sorted(rows, key=lambda row: (
        row['variant'], row['case'], row['policy_seed']))
    payload = {'runs': rows, 'failures': failures, 'aggregate': aggregate(rows)}
    (directory / 'summary.json').write_text(json.dumps(payload, indent=2))
    fields = [
        'variant', 'case', 'scenario_seed', 'policy_seed', 'frames',
        'seconds', 'training_updates', 'PAoI', 'Q_Mbit_per_user',
        'E_BS_J_per_node', 'final_delta', 'mean_bs_count',
        'mean_sat_count', 'mean_bs_residual_Mbit',
        'final_max_energy_queue_J', 'max_energy_queue_slope_J_per_frame',
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
               ROOT / 'analysis' / 'run_seed_sensitivity.py']
    sources += list((ROOT / 'core').rglob('*.py'))
    sources += list((ROOT / 'utils').rglob('*.py'))
    sources = sorted(set(sources))
    return sources, {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def tasks_for(args, directory):
    return [
        (variant, case, seed, args.scenario_seed, args.frames,
         str(directory), args.device)
        for seed in args.policy_seeds
        for variant in args.variants
        for case in VARIANT_CASES[variant]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=4096)
    parser.add_argument('--scenario-seed', type=int, default=42)
    parser.add_argument('--policy-seeds', nargs='+', type=int,
                        default=DEFAULT_POLICY_SEEDS[:10])
    parser.add_argument('--variants', nargs='+', choices=VARIANT_CASES,
                        default=list(VARIANT_CASES))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.frames < 4 or args.workers < 1:
        parser.error('frames must be >= 4 and workers must be >= 1')
    if len(set(args.policy_seeds)) != len(args.policy_seeds):
        parser.error('policy seeds must be unique')

    sources, hashes = source_hashes()
    arguments = {key: value for key, value in vars(args).items()
                 if key != 'resume'}
    directory = (args.resume.resolve() if args.resume else
                 ROOT / 'results' / 'instability_ablations' /
                 datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    manifest = {
        'arguments': arguments,
        'source_hashes': hashes,
        'note': ('Fixed scenario; policy seed controls actor initialization '
                 'and replay sampling. exhaustive_j4 is per-BS 2**J '
                 'enumeration followed by the unchanged coordinate search.'),
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

    tasks = tasks_for(args, directory)
    rows, failures, remaining = [], [], []
    for task in tasks:
        variant, case, policy_seed, scenario_seed = task[:4]
        stem = task_stem(
            directory, variant, case, scenario_seed, policy_seed)
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
                    f"{row['variant']}/{row['case']} "
                    f"policy={row['policy_seed']} "
                    f"PAoI={row['metrics']['PAoI']:.4f} "
                    f"Emax={row['metrics']['final_max_energy_queue_J']:.1f} "
                    f"seconds={row['seconds']:.1f}", flush=True,
                )
            except Exception as exc:
                failures.append({'task': pending[future][:-2],
                                 'error': repr(exc)})
                print('FAILED', failures[-1], flush=True)
            write_summary(directory, rows, failures)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
