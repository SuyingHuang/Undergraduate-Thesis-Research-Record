"""Lyapunov quantities for independent BS and satellite task queues."""

import numpy as np


def lyapunov_value(q_bs, q_sat, e_bs,
                   queue_weight=1.0, energy_weight=1.0):
    """Return L(t) for two independent physical task queues.

    L(t) = 1/2 * [w_q sum(Q_bs^2 + Q_sat^2) + w_e sum(E_bs^2)].
    Both weights default to one for the raw physical Lyapunov value.
    The BS and satellite backlogs must not be added before squaring because
    they are separate queues with separate arrivals and services.
    """
    if queue_weight < 0 or energy_weight < 0:
        raise ValueError("Lyapunov weights must be non-negative")
    return 0.5 * float(
        queue_weight * (
            np.sum(np.asarray(q_bs, dtype=float) ** 2)
            + np.sum(np.asarray(q_sat, dtype=float) ** 2)
        )
        + energy_weight * np.sum(np.asarray(e_bs, dtype=float) ** 2)
    )


def drift_decomposition(q_bs, q_sat, e_bs,
                        delta_bs, delta_sat, delta_energy,
                        queue_weight=1.0, energy_weight=1.0):
    """Return the one-step quadratic drift upper-bound components.

    For each non-negative queue updated by Q' = max(0, Q + delta),

        1/2 * (Q'^2 - Q^2) <= Q * delta + 1/2 * delta^2.

    Applying the inequality independently to Q_bs, Q_sat and E_bs gives the
    queue-linear term, energy-linear term, quadratic remainder and their sum.
    Optional weights define the normalized Lyapunov function used by the
    decision objective. PAoI is a penalty, not part of L(t).
    """
    q_bs = np.asarray(q_bs, dtype=float)
    q_sat = np.asarray(q_sat, dtype=float)
    e_bs = np.asarray(e_bs, dtype=float)
    delta_bs = np.asarray(delta_bs, dtype=float)
    delta_sat = np.asarray(delta_sat, dtype=float)
    delta_energy = np.asarray(delta_energy, dtype=float)
    if queue_weight < 0 or energy_weight < 0:
        raise ValueError("Lyapunov weights must be non-negative")

    queue_linear = queue_weight * float(
        np.sum(q_bs * delta_bs) + np.sum(q_sat * delta_sat)
    )
    energy_linear = energy_weight * float(np.sum(e_bs * delta_energy))
    quadratic = 0.5 * float(
        queue_weight * (
            np.sum(delta_bs ** 2) + np.sum(delta_sat ** 2)
        )
        + energy_weight * np.sum(delta_energy ** 2)
    )
    return {
        'queue_linear': queue_linear,
        'energy_linear': energy_linear,
        'quadratic': quadratic,
        'upper_bound': queue_linear + energy_linear + quadratic,
    }
