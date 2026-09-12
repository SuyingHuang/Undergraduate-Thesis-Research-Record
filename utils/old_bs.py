"""Old-BS service helpers for fixed and candidate-coupled policies."""
import numpy as np
from utils.objective import objective_coefficients


def old_bs_service_at_frequency(cfg, workload, aggregate_frequency):
    """Serve proportional old work at an explicit aggregate BS frequency.

    ``aggregate_frequency`` has one value per BS.  Old users receive
    proportional shares, so their completion times are equal.  This primitive
    is policy-free and is used by the joint DPP search to score candidate old
    service together with the current-task allocation.
    """
    workload = np.asarray(workload, float)
    total = workload.sum(axis=-1, keepdims=True)
    ratio = np.divide(
        workload, total, out=np.zeros_like(workload), where=total > 0)
    frequency = np.asarray(aggregate_frequency, float)
    if frequency.ndim == 0:
        frequency = np.full(total.shape, float(frequency))
    else:
        frequency = np.broadcast_to(frequency.reshape(-1, 1), total.shape)
    if np.any(~np.isfinite(frequency)) or np.any(frequency < 0):
        raise ValueError('aggregate_frequency must be finite and nonnegative')
    if np.any(frequency > cfg.f_max_BS * (1 + 1e-12)):
        raise ValueError('aggregate_frequency exceeds f_max_BS')
    frequency = np.minimum(frequency, cfg.f_max_BS)

    f = frequency * ratio
    processed = np.minimum(workload, f * cfg.tau / cfg.phi)
    energy = (cfg.kappa1 * cfg.phi * f ** 2 * processed).sum(axis=-1)
    occupied = np.minimum(
        cfg.tau,
        np.divide(
            cfg.phi * total, frequency,
            out=np.full_like(total, cfg.tau), where=frequency > 0,
        ),
    )
    occupied = np.where(total > 0, occupied, 0.0)[..., 0]
    return processed, energy, occupied


def budget_limited_old_frequency(cfg, workload, fraction):
    """Invert the proportional-service energy curve for a fixed budget."""
    workload = np.asarray(workload, float)
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError('budget fraction must be in (0, 1]')
    total = workload.sum(axis=-1, keepdims=True)
    ratio = np.divide(
        workload, total, out=np.zeros_like(workload), where=total > 0)
    shape = (ratio ** 3).sum(axis=-1, keepdims=True)
    budget = float(fraction) * cfg.E_max_BS
    f_complete = cfg.phi * total / cfg.tau
    e_complete = cfg.kappa1 * cfg.tau * f_complete ** 3 * shape
    f_partial = np.cbrt(np.divide(
        budget, cfg.kappa1 * cfg.tau * shape,
        out=np.full_like(shape, np.inf), where=shape > 0))
    f_after_complete = np.sqrt(np.divide(
        budget, cfg.kappa1 * cfg.phi * total * shape,
        out=np.full_like(shape, np.inf), where=(total * shape) > 0))
    frequency = np.where(
        budget < e_complete, f_partial, f_after_complete)
    frequency = np.minimum(cfg.f_max_BS, frequency)
    return np.where(total > 0, frequency, 0.0)[..., 0]


def joint_dpp_frequency_candidates(cfg, workload, energy_queue,
                                   transition_times=()):
    """Return deterministic old-frequency candidates for joint DPP scoring.

    The grid includes zero, full frequency, the old-work completion threshold,
    the old-only DPP stationary point, and frequencies whose completion time
    coincides with a current-task transmission time.  It is a bounded search,
    not a continuous global-optimality claim.
    """
    workload = np.asarray(workload, float)
    if workload.ndim != 1:
        raise ValueError('workload must be one-dimensional for one BS')
    total = float(workload.sum())
    if total <= 0:
        return np.array([0.0])
    points = int(getattr(cfg, 'joint_dpp_old_frequency_grid_points', 9))
    if points < 2:
        raise ValueError('joint_dpp_old_frequency_grid_points must be >= 2')

    values = list(np.linspace(0.0, cfg.f_max_BS, points))
    completion_frequency = cfg.phi * total / cfg.tau
    values.append(completion_frequency)
    if getattr(cfg, 'joint_dpp_include_budgeted_witness', True):
        witness_fraction = float(getattr(
            cfg, 'joint_dpp_budgeted_witness_fraction', 0.75))
        values.append(float(budget_limited_old_frequency(
            cfg, workload[None, :], witness_fraction)[0]))

    ratio = workload / total
    weights = objective_coefficients(cfg)
    benefit = weights['queue'] * float(np.sum(workload * ratio)) / cfg.phi
    penalty = (3.0 * weights['energy'] * max(0.0, float(energy_queue))
               * cfg.kappa1 * float(np.sum(ratio ** 3)))
    if penalty > 0:
        values.append(np.sqrt(benefit / penalty))

    for transition_time in np.asarray(transition_times, float).ravel():
        if 0 < transition_time <= cfg.tau:
            values.append(cfg.phi * total / transition_time)

    clipped = np.clip(np.asarray(values, float), 0.0, cfg.f_max_BS)
    return np.unique(clipped)


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
        frequency = budget_limited_old_frequency(
            cfg, workload, fraction)[:, None]
    elif cfg.old_bs_policy == 'joint_dpp':
        raise RuntimeError(
            'joint_dpp old service must be selected jointly with the current '
            'candidate; use old_bs_service_at_frequency')
    elif cfg.old_bs_policy != 'legacy':
        raise ValueError(
            'old_bs_policy must be legacy, energy_aware, budgeted, or joint_dpp')
    return old_bs_service_at_frequency(cfg, workload, frequency[..., 0])
