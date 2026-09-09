"""Independent two-user dense primal checks; no upper-level scorer is reused."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import SystemConfig
from core.optimizers.bs_optimizer import BS_Optimizer
from core.optimizers.leo_optimizer import LEO_Optimizer


def main():
    cfg = SystemConfig()
    cfg.I, cfg.J = 1, 2
    rng = np.random.RandomState(260908)
    rows = []
    for kind in ('BS', 'LEO'):
        opt = BS_Optimizer(cfg) if kind == 'BS' else LEO_Optimizer(cfg)
        capacity = cfg.f_max_BS if kind == 'BS' else cfg.f_max_Sat
        kappa = cfg.kappa1 if kind == 'BS' else cfg.kappa2
        future = capacity if kind == 'BS' else opt.paoi_future_frequency
        axis = np.linspace(0, capacity, 301)
        xx, yy = np.meshgrid(axis, axis)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        grid = grid[grid.sum(axis=1) <= capacity]
        for trial in range(20):
            L, Q = rng.uniform(1e6, 25e6, 2), rng.uniform(0, 8e7, 2)
            available, E = rng.uniform(.3, 5, 2), float(rng.uniform(0, 2000))
            allocation = (opt.optimize(L, Q, E, cfg.tau-available, 0) if kind == 'BS'
                          else opt.optimize(L, Q, available))

            def evaluate(f):
                left = np.maximum(0, L-f*available/cfg.phi)
                partial = left > 1e-6
                finish = cfg.tau-available + np.divide(cfg.phi*L, f,
                                     out=np.zeros_like(f), where=f > 0)
                age = np.where(partial, cfg.tau+cfg.w*cfg.phi*left.sum(axis=1, keepdims=True)/future, finish)
                energy = (kappa*cfg.phi*f**2*(L-left)).sum(axis=1)
                value = opt.queue_weight*(Q*left).sum(axis=1)+opt.paoi_weight*age.sum(axis=1)
                if kind == 'BS':
                    value += E*opt.energy_weight*energy
                else:
                    value = np.where(energy <= cfg.E_max_Sat*(1+1e-10), value, np.inf)
                return value

            actual, oracle = float(evaluate(allocation[None, :])[0]), float(evaluate(grid).min())
            rows.append(dict(kind=kind, trial=trial, score=actual, grid_score=oracle,
                             positive_gap=max(0, actual-oracle)))
    payload = dict(grid_points_per_axis=301, seed=260908, trials=rows)
    (ROOT/'analysis'/'coupled_oracle_checks.json').write_text(json.dumps(payload, indent=2))
    maximum = max(row['positive_gap'] for row in rows)
    print(f'{len(rows)} instances; maximum positive gap against feasible dense grid: {maximum:.6g}')
    if maximum > 1e-4:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
