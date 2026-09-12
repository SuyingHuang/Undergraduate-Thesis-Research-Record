"""Summarize a completed fixed-scenario policy-seed sensitivity matrix."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lda-seed-summary-mpl')
from utils.matplotlib_backend import configure_matplotlib

configure_matplotlib()
import matplotlib.pyplot as plt


RUNAWAY_THRESHOLD_J = 1800.0


def correlation(left, right, log_right=False):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if log_right:
        right = np.log1p(np.maximum(0.0, right))
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def mean_std_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, None, None
    std = float(np.std(values, ddof=1))
    ci = float(student_t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
    return mean, std, ci


def analyze(rows):
    output = {}
    for case in sorted({row['case'] for row in rows}):
        group = sorted((row for row in rows if row['case'] == case),
                       key=lambda row: row['policy_seed'])
        paoi = np.asarray([row['metrics']['PAoI'] for row in group])
        energy = np.asarray([
            row['metrics']['final_max_energy_queue_J'] for row in group
        ])
        residual = np.asarray([
            row['metrics']['mean_bs_residual_Mbit'] for row in group
        ])
        sat_count = np.asarray([row['metrics']['mean_sat_count'] for row in group])
        delta = np.asarray([row['metrics']['final_delta'] for row in group])
        runaway = energy > RUNAWAY_THRESHOLD_J
        mean, std, ci = mean_std_ci(paoi)
        output[case] = {
            'n': len(group),
            'PAoI_mean': mean,
            'PAoI_sample_std': std,
            'PAoI_ci95_half_width': ci,
            'PAoI_min': float(np.min(paoi)),
            'PAoI_max': float(np.max(paoi)),
            'energy_runaway_threshold_J': RUNAWAY_THRESHOLD_J,
            'energy_runaway_count': int(np.sum(runaway)),
            'energy_runaway_fraction': float(np.mean(runaway)),
            'stable_PAoI_mean': float(np.mean(paoi[~runaway])) if np.any(~runaway) else None,
            'runaway_PAoI_mean': float(np.mean(paoi[runaway])) if np.any(runaway) else None,
            'correlation_PAoI_log_energy_queue': correlation(paoi, energy, True),
            'correlation_PAoI_bs_residual': correlation(paoi, residual),
            'correlation_PAoI_sat_count': correlation(paoi, sat_count),
            'correlation_PAoI_final_delta': correlation(paoi, delta),
        }
    return output


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def write_report(directory, analysis, expected, failures, scenario_seed):
    lines = [
        '# LDA1 fixed-scenario policy-seed sensitivity', '',
        f'- Complete runs: {expected}',
        f'- Environment/workload scenario seed: {scenario_seed}',
        f'- Energy-runaway definition: final maximum BS virtual energy queue > {RUNAWAY_THRESHOLD_J:.0f} J',
        '- The intervals quantify policy-seed variation for one fixed environment scenario; they do not include between-scenario uncertainty.',
        '',
        '| Case | n | PAoI mean ± sample std (s) | 95% CI half-width | Range (s) | Energy runaway | Stable mean | Runaway mean | r(PAoI, log Equeue) | r(PAoI, BS residual) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for case, item in analysis.items():
        lines.append(
            f"| `{case}` | {item['n']} | {fmt(item['PAoI_mean'])} ± "
            f"{fmt(item['PAoI_sample_std'])} | {fmt(item['PAoI_ci95_half_width'])} | "
            f"{item['PAoI_min']:.3f}–{item['PAoI_max']:.3f} | "
            f"{item['energy_runaway_count']}/{item['n']} | "
            f"{fmt(item['stable_PAoI_mean'])} | {fmt(item['runaway_PAoI_mean'])} | "
            f"{fmt(item['correlation_PAoI_log_energy_queue'])} | "
            f"{fmt(item['correlation_PAoI_bs_residual'])} |"
        )
    lines += ['', f'Failures: {len(failures)}', '']
    (directory / 'analysis.md').write_text('\n'.join(lines))


def plot_rows(directory, rows):
    cases = sorted({row['case'] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for axis, case in zip(axes.ravel(), cases):
        group = [row for row in rows if row['case'] == case]
        paoi = np.asarray([row['metrics']['PAoI'] for row in group])
        energy = np.asarray([
            row['metrics']['final_max_energy_queue_J'] for row in group
        ])
        runaway = energy > RUNAWAY_THRESHOLD_J
        axis.scatter(np.log10(1.0 + energy[~runaway]), paoi[~runaway],
                     label='stable energy queue', marker='o')
        axis.scatter(np.log10(1.0 + energy[runaway]), paoi[runaway],
                     label='energy runaway', marker='x', color='tab:red')
        for row, x, y in zip(group, np.log10(1.0 + energy), paoi):
            axis.annotate(str(row['policy_seed']), (x, y), fontsize=6,
                          xytext=(2, 2), textcoords='offset points')
        axis.set_title(case)
        axis.set_xlabel('log10(1 + final max BS energy queue [J])')
        axis.set_ylabel('Last-half PAoI [s]')
        axis.grid(True, linestyle='--', alpha=0.5)
        axis.legend(fontsize=8)
    fig.suptitle('LDA1 policy-seed sensitivity under a fixed scenario')
    fig.tight_layout()
    fig.savefig(directory / 'seed_sensitivity.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    summary = json.loads((directory / 'summary.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    expected = (len(manifest['arguments']['cases']) *
                len(manifest['arguments']['policy_seeds']))
    rows, failures = summary['runs'], summary['failures']
    if len(rows) != expected or failures:
        print(f'Incomplete matrix: {len(rows)}/{expected}, failures={len(failures)}')
        raise SystemExit(2)
    result = analyze(rows)
    (directory / 'analysis.json').write_text(json.dumps(result, indent=2))
    write_report(directory, result, expected, failures,
                 manifest['arguments']['scenario_seed'])
    plot_rows(directory, rows)
    print(directory / 'analysis.md')


if __name__ == '__main__':
    main()
