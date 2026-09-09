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
    elif cfg.old_bs_policy != 'legacy':
        raise ValueError('old_bs_policy must be legacy or energy_aware')
    f = frequency * ratio
    processed = np.minimum(workload, f*cfg.tau/cfg.phi)
    energy = (cfg.kappa1*cfg.phi*f**2*processed).sum(axis=-1)
    occupied = np.minimum(cfg.tau, np.divide(cfg.phi*total, frequency,
                          out=np.full_like(total, cfg.tau), where=frequency > 0))
    occupied = np.where(total > 0, occupied, 0)[..., 0]
    return processed, energy, occupied
