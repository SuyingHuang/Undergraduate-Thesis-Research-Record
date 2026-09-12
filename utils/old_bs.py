"""Deterministic old-BS service, frozen before evaluating new-task actions."""
import numpy as np
from utils.objective import objective_coefficients


def old_bs_service(cfg, workload, energy_queue):
    workload = np.asarray(workload, float)
    total = workload.sum(axis=-1, keepdims=True)
    ratio = np.divide(workload, total, out=np.zeros_like(workload), where=total > 0)
    frequency = np.full(total.shape, cfg.f_max_BS)
    if cfg.old_bs_policy == 'energy_aware':
        weights = objective_coefficients(cfg)
        # Fixed proportional shares: minimize old-work queue drift + energy.
        benefit = weights['queue'] * (workload*ratio).sum(axis=-1, keepdims=True) / cfg.phi
        penalty = (3*weights['energy']*np.asarray(energy_queue)[..., None]
                   *cfg.kappa1*(ratio**3).sum(axis=-1, keepdims=True))
        stationary = np.sqrt(np.divide(benefit, penalty,
                             out=np.full_like(benefit, np.inf), where=penalty > 0))
        frequency = np.minimum(frequency, np.minimum(cfg.phi*total/cfg.tau, stationary))
    elif cfg.old_bs_policy == 'budgeted':
        fraction = float(cfg.old_bs_energy_budget_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError('old_bs_energy_budget_fraction must be in (0, 1]')
        # Energy is monotone in the aggregate proportional-share frequency.
        # Below the completion threshold E=kappa*tau*f^3*sum(r^3); after all
        # old work completes E=kappa*phi*total*f^2*sum(r^3).  Invert the
        # applicable branch to use the largest frequency within the reserved
        # old-work energy budget, preserving as much time as possible for new
        # work without the legacy max-frequency energy spike.
        shape = (ratio ** 3).sum(axis=-1, keepdims=True)
        budget = fraction * cfg.E_max_BS
        f_complete = cfg.phi * total / cfg.tau
        e_complete = cfg.kappa1 * cfg.tau * f_complete ** 3 * shape
        f_partial = np.cbrt(np.divide(
            budget, cfg.kappa1 * cfg.tau * shape,
            out=np.full_like(shape, np.inf), where=shape > 0))
        f_after_complete = np.sqrt(np.divide(
            budget, cfg.kappa1 * cfg.phi * total * shape,
            out=np.full_like(shape, np.inf), where=(total * shape) > 0))
        f_budget = np.where(budget < e_complete,
                            f_partial, f_after_complete)
        frequency = np.minimum(frequency, f_budget)
        frequency = np.where(total > 0, frequency, 0.0)
    elif cfg.old_bs_policy != 'legacy':
        raise ValueError(
            'old_bs_policy must be legacy, energy_aware, or budgeted')
    f = frequency * ratio
    processed = np.minimum(workload, f*cfg.tau/cfg.phi)
    energy = (cfg.kappa1*cfg.phi*f**2*processed).sum(axis=-1)
    occupied = np.minimum(cfg.tau, np.divide(cfg.phi*total, frequency,
                          out=np.full_like(total, cfg.tau), where=frequency > 0))
    occupied = np.where(total > 0, occupied, 0)[..., 0]
    return processed, energy, occupied
