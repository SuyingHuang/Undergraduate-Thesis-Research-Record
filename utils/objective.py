"""Canonical coefficients shared by candidate scoring and resource optimizers."""


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
