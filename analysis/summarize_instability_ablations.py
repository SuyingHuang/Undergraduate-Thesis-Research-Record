"""Compare completed instability ablations with the fixed-scenario baseline."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lda-instability-summary-mpl')
from utils.matplotlib_backend import configure_matplotlib

configure_matplotlib()
import matplotlib.pyplot as plt

from analysis.run_instability_ablations import VARIANT_CASES


RUNAWAY_THRESHOLD_J = 1800.0
CELL_ORDER = [
    ('baseline', 'j4'), ('no_train', 'j4'), ('fixed_delta', 'j4'),
    ('exhaustive_j4', 'j4'), ('baseline', 'l16'),
    ('no_train', 'l16'), ('fixed_delta', 'l16'),
]


def metric(row, name):
    return float(row['metrics'][name])


def cell_stats(rows):
    paoi = np.asarray([metric(row, 'PAoI') for row in rows])
    energy = np.asarray([
        metric(row, 'final_max_energy_queue_J') for row in rows])
    return {
        'n': len(rows),
        'PAoI_mean': float(np.mean(paoi)),
        'PAoI_sample_std': (
            float(np.std(paoi, ddof=1)) if len(paoi) > 1 else None),
        'PAoI_min': float(np.min(paoi)),
        'PAoI_max': float(np.max(paoi)),
        'energy_runaway_count': int(np.sum(energy > RUNAWAY_THRESHOLD_J)),
        'energy_runaway_fraction': float(np.mean(energy > RUNAWAY_THRESHOLD_J)),
    }


def analyze(ablation_rows, baseline_rows, policy_seeds):
    by_cell = {}
    for case in ('j4', 'l16'):
        by_cell[('baseline', case)] = [
            row for row in baseline_rows
            if row['case'] == case and row['policy_seed'] in policy_seeds
        ]
    for variant, case in CELL_ORDER:
        if variant == 'baseline':
            continue
        by_cell[(variant, case)] = [
            row for row in ablation_rows
            if row['variant'] == variant and row['case'] == case
        ]

    output = {'cells': {}, 'paired_vs_baseline': {}}
    for cell, rows in by_cell.items():
        output['cells']['/'.join(cell)] = cell_stats(rows)

    for (variant, case), rows in by_cell.items():
        if variant == 'baseline':
            continue
        baseline_by_seed = {
            row['policy_seed']: row for row in by_cell[('baseline', case)]}
        current_by_seed = {row['policy_seed']: row for row in rows}
        seeds = sorted(set(baseline_by_seed) & set(current_by_seed))
        differences = np.asarray([
            metric(current_by_seed[seed], 'PAoI')
            - metric(baseline_by_seed[seed], 'PAoI') for seed in seeds
        ])
        output['paired_vs_baseline'][f'{variant}/{case}'] = {
            'n': len(seeds),
            'mean_PAoI_difference': float(np.mean(differences)),
            'median_PAoI_difference': float(np.median(differences)),
            'improved_seed_count': int(np.sum(differences < 0)),
            'worsened_seed_count': int(np.sum(differences > 0)),
        }
    return output, by_cell


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def write_report(directory, result):
    lines = [
        '# LDA1 instability ablations', '',
        '- Same workload and channel stream for every run (scenario seed 42).',
        '- Ten matched policy seeds per cell.',
        f'- Runaway: final maximum BS virtual-energy queue > {RUNAWAY_THRESHOLD_J:.0f} J.',
        '- `exhaustive_j4` enumerates all per-BS 2^J candidates; the unchanged joint coordinate search follows.',
        '',
        '| Cell | n | PAoI mean ± sample std | Range | Runaway | Paired ΔPAoI vs baseline | Better seeds |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for variant, case in CELL_ORDER:
        key = f'{variant}/{case}'
        item = result['cells'][key]
        paired = result['paired_vs_baseline'].get(key)
        lines.append(
            f"| `{key}` | {item['n']} | {fmt(item['PAoI_mean'])} ± "
            f"{fmt(item['PAoI_sample_std'])} | {item['PAoI_min']:.3f}–"
            f"{item['PAoI_max']:.3f} | {item['energy_runaway_count']}/{item['n']} | "
            f"{fmt(paired['mean_PAoI_difference']) if paired else '—'} | "
            f"{paired['improved_seed_count'] if paired else '—'} |"
        )
    lines += [
        '',
        'Interpretation guide:', '',
        '- At `j4`, `no_train` removes the observed split, so online self-training participates in that instability.',
        '- At `j4`, `fixed_delta` removes the observed split, so adaptive candidate-window pruning is a key amplifier there.',
        '- At `j4`, `exhaustive_j4` removes the observed split, so dependence of the search set on Actor outputs is a key cause there.',
        '- At `l16`, neither `no_train` nor `fixed_delta` reliably controls the energy queue; the J=4 interpretation must not be generalized to this load.',
        '- More than one successful ablation means the mechanisms interact; it does not make the tests contradictory.',
        '',
    ]
    (directory / 'analysis.md').write_text('\n'.join(lines))


def plot_cells(directory, by_cell):
    cells = [cell for cell in CELL_ORDER if cell in by_cell]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, case in zip(axes, ('j4', 'l16')):
        case_cells = [cell for cell in cells if cell[1] == case]
        values = [[metric(row, 'PAoI') for row in by_cell[cell]]
                  for cell in case_cells]
        labels = [cell[0] for cell in case_cells]
        axis.boxplot(values, tick_labels=labels, showmeans=True)
        for x, rows in enumerate(values, 1):
            axis.scatter(np.full(len(rows), x), rows, s=15, alpha=.65)
        axis.set_title(case)
        axis.set_ylabel('Last-half PAoI (s)')
        axis.grid(True, axis='y', linestyle='--', alpha=.4)
        axis.tick_params(axis='x', rotation=20)
    fig.suptitle('LDA1 instability ablations under matched fixed scenarios')
    fig.tight_layout()
    fig.savefig(directory / 'instability_ablations.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    ablation = json.loads((directory / 'summary.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    baseline = json.loads(
        (args.baseline_dir.resolve() / 'summary.json').read_text())
    expected = (sum(len(VARIANT_CASES[variant])
                    for variant in manifest['arguments']['variants'])
                * len(manifest['arguments']['policy_seeds']))
    if len(ablation['runs']) != expected or ablation['failures']:
        raise SystemExit(
            f"Incomplete matrix: {len(ablation['runs'])}/{expected}, "
            f"failures={len(ablation['failures'])}")

    result, by_cell = analyze(
        ablation['runs'], baseline['runs'],
        set(manifest['arguments']['policy_seeds']))
    for case in ('j4', 'l16'):
        baseline_rows = by_cell[('baseline', case)]
        if len(baseline_rows) != len(manifest['arguments']['policy_seeds']):
            raise SystemExit(f'Baseline {case} does not contain all matched seeds')
        hashes = {row['scenario_hash'] for row in baseline_rows}
        hashes.update(row['scenario_hash'] for row in ablation['runs']
                      if row['case'] == case)
        if len(hashes) != 1:
            raise SystemExit(f'Scenario hash mismatch for {case}')

    (directory / 'analysis.json').write_text(json.dumps(result, indent=2))
    write_report(directory, result)
    plot_cells(directory, by_cell)
    print(directory / 'analysis.md')


if __name__ == '__main__':
    main()
