"""Paired diagnostic runs, with immutable config/source metadata and trajectories.

Example: python analysis/run_diagnostic_ablation.py --frames 512 --seeds 42 123
These runs retain the existing reference scales to isolate algorithm changes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lda-diagnostic-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from config import SystemConfig

CASES = {'load10': {'L_mean': 10e6}, 'users8': {'J': 8}, 'default': {}}
VARIANTS = {
    'legacy_lda': ('LDA', 'legacy', False, 'none', 'legacy'),
    'coupled_lda': ('LDA', 'coupled', False, 'none', 'legacy'),
    'guarded_lda': ('LDA', 'coupled', True, 'none', 'legacy'),
    'upper_only': ('LDA', 'coupled', True, 'upper', 'legacy'),
    'lower_only': ('LDA', 'coupled', True, 'lower', 'legacy'),
    'both': ('AC', 'coupled', True, 'both', 'legacy'),
    'both_oldaware': ('AC', 'coupled', True, 'both', 'energy_aware'),
    'coupled_cob': ('COB', 'coupled', False, 'none', 'legacy'),
    'coupled_mtd': ('MTD', 'coupled', False, 'none', 'legacy'),
    'legacy_mtd': ('MTD', 'legacy', False, 'none', 'legacy'),
    'legacy_both': ('AC', 'legacy', False, 'both', 'legacy'),
}


def configuration(case, variant, frames):
    cfg = SystemConfig()
    for key, value in CASES[case].items():
        setattr(cfg, key, value)
    _, cfg.resource_solver, cfg.include_baseline_candidates, cfg.paoi_ablation, cfg.old_bs_policy = VARIANTS[variant]
    cfg.audit_baseline_candidates = True
    cfg.sim_frames, cfg.dnn_device = frames, 'cpu'
    cfg._update_bandwidth_params()
    return cfg


def worker(task):
    import torch
    from core.agents.lda_agent import LDAAgent
    from core.agents.baselines import ACAgent, COBAgent, MTDAgent
    from main import run_simulation
    from utils.reproducibility import set_seed
    case, variant, seed, frames, directory = task
    torch.set_num_threads(1)
    cfg = configuration(case, variant, frames)
    stem = Path(directory) / f'{case}_{variant}_s{seed}'
    rng = np.random.RandomState(seed)
    workloads = np.maximum(0, cfg.L_mean+rng.normal(0, cfg.L_std, (frames, cfg.I, cfg.J)))
    scenario_hash = hashlib.sha256(workloads.tobytes()).hexdigest()
    cls = {'LDA': LDAAgent, 'AC': ACAgent, 'COB': COBAgent, 'MTD': MTDAgent}[VARIANTS[variant][0]]
    started = time.monotonic()
    with stem.with_suffix('.log').open('w', buffering=1) as output, redirect_stdout(output), redirect_stderr(output):
        set_seed(seed)
        env, agent = run_simulation(cfg, cls, variant, preset_L=workloads, seed=seed)
    arrays = {key: np.asarray(value) for key, value in env.history.items()
              if value and key != 'Loss_per_BS'}
    for i, losses in enumerate(env.history.get('Loss_per_BS', [])):
        arrays[f'Loss_BS_{i}'] = np.asarray(losses)
    np.savez_compressed(stem.with_suffix('.npz'), **arrays)
    start = frames//2
    metrics = {label: float(np.mean(env.history[key][start:])) for label, key in
               [('PAoI', 'Cost'), ('Q', 'Q_total'), ('E_BS', 'E_virt_bs'), ('E_sat_total', 'E_sat_total')]}
    queue = np.asarray(env.history['E_queue_bs_max'])
    metrics.update(final_energy_queue=float(queue[-1]),
                   energy_queue_slope=float(np.polyfit(np.arange(frames-start), queue[start:], 1)[0]))
    for key in ('candidate_baseline_improvement', 'candidate_baseline_selected'):
        if key in env.history:
            values = np.asarray(env.history[key][start:])
            metrics[key+'_mean'] = float(np.mean(values))
            metrics[key+'_positive_fraction'] = float(np.mean(values > 1e-12))
    payload = dict(case=case, variant=variant, seed=seed, frames=frames, config=vars(cfg),
                   scenario_hash=scenario_hash, seconds=time.monotonic()-started,
                   training_updates=len(getattr(agent, 'loss_history', [])),
                   window_start=start, metrics=metrics)
    stem.with_suffix('.json').write_text(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=512)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123])
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', type=Path, help='Resume an existing directory with identical sources and experiment settings')
    args = parser.parse_args()
    if args.frames < 4 or args.workers < 1:
        parser.error('frames must be >=4 and workers >=1')
    directory = args.resume.resolve() if args.resume else ROOT/'results'/'diagnostic'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    sources = [ROOT/'config.py', ROOT/'main.py', Path(__file__)]
    sources += list((ROOT/'core').rglob('*.py')) + list((ROOT/'utils').rglob('*.py'))
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    arguments = {key: value for key, value in vars(args).items() if key != 'resume'}
    manifest = dict(arguments=arguments, source_hashes=source_hashes,
                    note='Paired finite-horizon diagnostic, not a convergence or recalibration claim.')
    if args.resume:
        previous = json.loads((directory/'manifest.json').read_text())
        settings = ('frames', 'seeds', 'cases', 'variants')
        if any(previous['arguments'][key] != arguments[key] for key in settings):
            parser.error('Resume requires the original frames, seeds, cases and variants')
        if previous['source_hashes'] != source_hashes:
            parser.error('Resume requires identical source files')
    else:
        directory.mkdir(parents=True, exist_ok=False)
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2))
    # Preserve the executed sources, including uncommitted implementation.
    import zipfile
    with zipfile.ZipFile(directory/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(ROOT)))
    tasks = [(case, variant, seed, args.frames, str(directory))
             for case in args.cases for variant in args.variants for seed in args.seeds]
    print(directory, flush=True)
    rows, failures = [], []
    remaining = []
    for task in tasks:
        case, variant, seed, _, _ = task
        stem = directory/f'{case}_{variant}_s{seed}'
        if args.resume and stem.with_suffix('.json').exists() and stem.with_suffix('.npz').exists():
            rows.append(json.loads(stem.with_suffix('.json').read_text()))
        else:
            remaining.append(task)
    print(f'Reusing {len(rows)} completed runs; scheduling {len(remaining)}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(worker, task): task for task in remaining}
        for future in as_completed(pending):
            try:
                row = future.result()
                rows.append(row)
                print(f"{len(rows)+len(failures)}/{len(tasks)} {row['case']} {row['variant']} "
                      f"s{row['seed']} P={row['metrics']['PAoI']:.4f} "
                      f"seconds={row['seconds']:.1f}", flush=True)
            except Exception as exc:
                failures.append(dict(task=pending[future][:-1], error=repr(exc)))
                print('FAILED', failures[-1], flush=True)
            (directory/'summary.json').write_text(json.dumps(dict(runs=rows, failures=failures), indent=2))
    (directory/'summary.json').write_text(json.dumps(dict(runs=rows, failures=failures), indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
