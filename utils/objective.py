"""Canonical coefficients and piecewise scoring shared by resource optimizers."""

import numpy as np


def objective_coefficients(cfg, include_paoi=True):
    """Return coefficients for queue, PAoI and BS virtual-energy terms.

    The queue implementation first divides Q by 1e5 and workload by 1e4,
    hence its physical coefficient is 1 / (1e9 * Q_ref).
    """
    if cfg.Q_ref <= 0 or cfg.PAoI_ref <= 0 or cfg.E_ref <= 0:
        raise ValueError("Objective reference scales must be positive")
    return {
        'queue': 1.0 / (1e9 * cfg.Q_ref),
        'paoi': (cfg.K_p / cfg.PAoI_ref) if include_paoi else 0.0,
        'energy': 1.0 / cfg.E_ref,
    }


def select_piecewise_frequency(L, Q, t_avail, f_complete, f_partial,
                               f_threshold, resource_dual, energy_dual,
                               cfg, queue_weight, paoi_weight, kappa,
                               frequency_max, paoi_future_frequency=None):
    """Choose the lower-Lagrangian feasible branch of a piecewise task.

    ``f_complete`` is the stationary point derived under full completion and
    must lie at or above ``f_threshold``. ``f_partial`` is derived under a
    remaining workload and must lie below that threshold. Merely checking one
    unconstrained stationary point can select it outside its derivation domain;
    clipping both candidates to their domains and comparing their Lagrangian
    values also handles optima at the boundary.
    """
    L = np.asarray(L, dtype=float)
    Q = np.asarray(Q, dtype=float)
    t_avail = np.asarray(t_avail, dtype=float)
    f_threshold = np.asarray(f_threshold, dtype=float)
    resource_dual = np.asarray(resource_dual, dtype=float)
    energy_dual = np.asarray(energy_dual, dtype=float)
    if paoi_future_frequency is None:
        paoi_future_frequency = frequency_max

    valid = (L > 1e-6) & (t_avail > 1e-6)
    partial = np.clip(np.minimum(f_partial, f_threshold), 0.0, frequency_max)
    complete = np.clip(np.maximum(f_complete, f_threshold), 0.0, frequency_max)

    leftover = np.maximum(0.0, L - partial * t_avail / cfg.phi)
    score_partial = (
        queue_weight * Q * leftover
        + paoi_weight * (
            cfg.tau + cfg.w * cfg.phi * leftover / paoi_future_frequency
        )
        + energy_dual * kappa * partial ** 3 * t_avail
        + resource_dual * partial
    )

    completion_time = np.divide(
        cfg.phi * L, complete,
        out=np.full(np.broadcast(L, complete).shape, np.inf, dtype=float),
        where=complete > 1e-20,
    )
    score_complete = (
        paoi_weight * (cfg.tau - t_avail + completion_time)
        + energy_dual * kappa * cfg.phi * complete ** 2 * L
        + resource_dual * complete
    )
    complete_feasible = valid & (f_threshold <= frequency_max)
    score_complete = np.where(complete_feasible, score_complete, np.inf)
    chosen = np.where(score_complete < score_partial, complete, partial)
    return np.where(valid, chosen, 0.0)
