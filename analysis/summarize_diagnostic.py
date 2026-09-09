"""Export a diagnostic comparison figure and seed-level metric table."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/lda-diagnostic-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--allow-partial', action='store_true', help='Explicitly export incomplete diagnostic runs')
    args = parser.parse_args()
    data = json.loads((args.directory/'summary.json').read_text())
    runs = data['runs']
    manifest = json.loads((args.directory/'manifest.json').read_text())
    settings = manifest['arguments']
    expected = {(case, variant, seed) for case in settings['cases']
                for variant in settings['variants'] for seed in settings['seeds']}
    actual = [(r['case'], r['variant'], r['seed']) for r in runs]
    if len(actual) != len(set(actual)) or set(actual)-expected:
        parser.error('Duplicate or unexpected run identifiers')
    missing = expected-set(actual)
    if (missing or data['failures']) and not args.allow_partial:
        parser.error(f'{len(missing)} missing runs and {len(data["failures"])} failures; use --allow-partial for a preliminary export')
    if not runs:
        parser.error('No completed runs')
    for case in settings['cases']:
        for seed in settings['seeds']:
            paired = [r for r in runs if r['case'] == case and r['seed'] == seed]
            if len({r['scenario_hash'] for r in paired}) > 1:
                parser.error(f'Workload mismatch for {case}, seed {seed}')
    for r in runs:
        if r['frames'] != settings['frames'] or r['window_start'] != r['frames']//2:
            parser.error('Inconsistent run length or averaging window')
        if not all(np.isfinite(value) for value in r['metrics'].values()):
            parser.error('Nonfinite metrics')
    flat = [dict(case=r['case'], variant=r['variant'], seed=r['seed'],
                 frames=r['frames'], training_updates=r['training_updates'],
                 seconds=r['seconds'], **r['metrics']) for r in runs]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with (args.directory/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(flat, key=lambda r: (r['case'], r['variant'], r['seed'])))
    cases = list(dict.fromkeys(r['case'] for r in runs))
    order = ['legacy_lda', 'coupled_lda', 'guarded_lda', 'legacy_mtd',
             'coupled_mtd', 'coupled_cob', 'upper_only', 'lower_only',
             'legacy_both', 'both', 'both_oldaware']
    metrics = [('PAoI', 'Reported PAoI proxy (s)', 1),
               ('E_BS', 'BS energy (J / node / frame)', 1),
               ('Q', 'Queue (Mb / user)', 1e6)]
    fig, axes = plt.subplots(len(cases), 3, figsize=(15, 4.7*len(cases)), squeeze=False)
    colors = {name: ('#237a57' if name == 'guarded_lda' else
                     '#cb7046' if name in ('coupled_mtd', 'legacy_mtd') else '#547ea7') for name in order}
    for i, case in enumerate(cases):
        names = [name for name in order if any(r['case'] == case and r['variant'] == name for r in runs)]
        for j, (key, label, scale) in enumerate(metrics):
            ax = axes[i, j]
            values = [np.mean([r['metrics'][key]/scale for r in runs
                               if r['case'] == case and r['variant'] == name]) for name in names]
            ax.barh(names, values, color=[colors[name] for name in names])
            for index, name in enumerate(names):
                samples = [r['metrics'][key]/scale for r in runs
                           if r['case'] == case and r['variant'] == name]
                ax.scatter(samples, [index]*len(samples), s=15, color='black', zorder=3)
            ax.invert_yaxis()
            ax.set_xlabel(label)
            ax.set_xlim(left=0)
            ax.set_title(case)
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
            if j > 0:
                ax.tick_params(axis='y', labelleft=False)
    frames = sorted(set(r['frames'] for r in runs))
    seeds = sorted(set(r['seed'] for r in runs))
    status = 'PARTIAL' if missing or data['failures'] else 'Complete matrix'
    fig.suptitle(f'{status}: frames={frames}, seeds={seeds}; shared last-half window', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(args.directory/'comparison.png', dpi=160)
    plt.close(fig)
    # Paired seed differences isolate one intervention at a time.
    comparisons = [('solver', 'legacy_lda', 'coupled_lda'),
                   ('candidate_coverage', 'coupled_lda', 'guarded_lda'),
                   ('remove_upper_paoi', 'guarded_lda', 'upper_only'),
                   ('remove_lower_paoi', 'guarded_lda', 'lower_only'),
                   ('remove_both_paoi', 'guarded_lda', 'both'),
                   ('old_bs_energy_policy', 'both', 'both_oldaware'),
                   ('mtd_solver', 'legacy_mtd', 'coupled_mtd'),
                   ('ac_solver_and_candidates', 'legacy_both', 'both'),
                   ('versus_mtd', 'coupled_mtd', 'guarded_lda')]
    lookup = {(r['case'], r['variant'], r['seed']): r for r in runs}
    paired_rows = []
    for case in cases:
        for label, before, after in comparisons:
            for seed in seeds:
                a, b = lookup.get((case, before, seed)), lookup.get((case, after, seed))
                if a is None or b is None:
                    continue
                for metric in ('PAoI', 'E_BS', 'Q', 'final_energy_queue', 'energy_queue_slope'):
                    old, new = a['metrics'][metric], b['metrics'][metric]
                    paired_rows.append(dict(case=case, comparison=label, seed=seed,
                                            before=before, after=after, metric=metric,
                                            before_value=old, after_value=new, delta=new-old,
                                            percent_change=100*(new-old)/abs(old) if old else ''))
    if paired_rows:
        with (args.directory/'paired_differences.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)
    # Show queue growth over time; a terminal maximum alone is insufficient.
    selected = ['legacy_lda', 'guarded_lda', 'coupled_mtd', 'legacy_both', 'both', 'both_oldaware']
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 4*len(cases)), squeeze=False)
    for i, case in enumerate(cases):
        for variant in selected:
            paths = [args.directory/f'{case}_{variant}_s{seed}.npz' for seed in seeds
                     if (case, variant, seed) in lookup]
            if not paths:
                continue
            trajectories = []
            for path in paths:
                with np.load(path) as history:
                    trajectories.append((history['Cost'].copy(), history['E_queue_bs_max'].copy()))
            for j, label in enumerate(('PAoI proxy, 32-frame mean (s)', 'Maximum BS energy queue (J)')):
                mean = np.mean([t[j] for t in trajectories], axis=0)
                if j == 0:
                    width = min(32, len(mean))
                    values = np.convolve(mean, np.ones(width)/width, mode='valid')
                    x = np.arange(width-1, len(mean))
                else:
                    values, x = mean, np.arange(len(mean))
                axes[i, j].plot(x, values, label=variant)
                axes[i, j].set(title=case, xlabel='Frame', ylabel=label)
                axes[i, j].grid(alpha=.2)
        axes[i, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.directory/'trajectories.png', dpi=160)
    plt.close(fig)
    lines = [f'# Diagnostic results ({"PARTIAL" if missing or data["failures"] else "complete matrix"})', '',
             f'{len(runs)}/{len(expected)} runs; {len(data["failures"])} failures; frames={frames}; seeds={seeds}.', '',
             'Means use the last half of each run. Dots show individual seeds, not confidence intervals.', '',
             '| Case | Variant | PAoI proxy (s) | BS energy (J/node/frame) | Queue (Mb/user) | Final max energy queue (J) | Energy queue slope (J/frame) | Training updates |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for case in cases:
        for variant in order:
            group = [r for r in runs if r['case'] == case and r['variant'] == variant]
            if not group:
                continue
            means = [np.mean([r['metrics'][key] for r in group])/scale for key, scale in
                     [('PAoI', 1), ('E_BS', 1), ('Q', 1e6), ('final_energy_queue', 1), ('energy_queue_slope', 1)]]
            updates = ','.join(str(r['training_updates']) for r in sorted(group, key=lambda r: r['seed']))
            lines.append(f'| {case} | {variant} | '+ ' | '.join(f'{value:.4f}' for value in means)+f' | {updates} |')
    lines += ['', 'Finite-horizon diagnosis only: the recorded Cost is a completion-delay proxy, not independently reconstructed age peaks.',
              'Reference scales are held fixed to isolate changes; no recalibration or convergence claim is made.',
              'The large-node completion-set search is approximate. Baseline candidates guarantee a nonworse same-state score only.',
              'BS energy is averaged across nodes; it does not establish compliance of every BS. The queue slope is fitted to the per-frame maximum.',
              'The AC solver comparison also changes candidate inclusion and therefore is not a pure solver ablation.', '']
    (args.directory/'summary.md').write_text('\n'.join(lines))
    print(args.directory/'comparison.png')
    print(f'{len(runs)} successful runs; {len(data["failures"])} failed runs')


if __name__ == '__main__':
    main()
