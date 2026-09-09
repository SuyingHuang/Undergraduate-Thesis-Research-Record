"""Feasible primal recovery for the shared-residual PAoI objective.

For a fixed set of completed tasks the remaining optimization is convex.
Small nodes enumerate all sets; larger nodes use bounded, deterministic sets.
The latter is an approximation, not a claim of global optimality. Every result
is checked and compared using the same physical score as the agent.
"""
from itertools import product

import numpy as np
from scipy.optimize import minimize


def node_metrics(L, f, available, cfg, kappa, future_frequency, old_left=0.0):
    """Shared by primal optimization and the reported upper-level score."""
    L, f = np.asarray(L), np.asarray(f)
    available = np.maximum(0.0, available)
    processed = np.minimum(L, f * available / cfg.phi)
    leftover = np.maximum(0.0, L - processed)
    # Roundoff at a completion boundary (less than one millionth of a bit).
    leftover = np.where(leftover <= 1e-6, 0.0, leftover)
    processed = L - leftover
    complete_time = cfg.tau - available + np.divide(
        cfg.phi * processed, f, out=np.zeros_like(f), where=f > 0)
    future = cfg.tau + cfg.w * cfg.phi * (old_left + leftover.sum()) / future_frequency
    paoi = np.where(L > 0, np.where(leftover > 0, future, complete_time), 0.0)
    energy = kappa * cfg.phi * f ** 2 * processed
    return leftover, paoi, energy


def recover_primal(L, Q, available, seed, cfg, *, capacity, kappa,
                   queue_weight, paoi_weight, energy_weight=0.0,
                   energy_limit=None, future_frequency=None, old_left=0.0):
    """Return the best feasible incumbent, including the legacy allocation."""
    L, Q = np.asarray(L, float), np.asarray(Q, float)
    available = np.maximum(0.0, np.asarray(available, float))
    future_frequency = capacity if future_frequency is None else future_frequency
    active = np.flatnonzero((L > 0) & (available > 1e-6))
    if not len(active):
        return np.zeros_like(L)

    def metrics(f):
        return node_metrics(L, f, available, cfg, kappa, future_frequency, old_left)

    def score(f):
        left, age, energy = metrics(f)
        return queue_weight * np.dot(Q, left) + paoi_weight * age.sum() + energy_weight * energy.sum()

    def feasible(f):
        return (np.isfinite(f).all() and np.min(f) >= 0 and
                f.sum() <= capacity * (1 + 1e-10) and
                (energy_limit is None or metrics(f)[2].sum() <= energy_limit * (1 + 1e-10)))

    best = np.zeros_like(L)
    seed = np.maximum(0, np.asarray(seed, float))
    seed = np.where((L > 0) & (available > 1e-6), seed, 0.0)
    if feasible(seed) and score(seed) < score(best):
        best = seed.copy()
    best_score = score(best)
    l, q, t = L[active], Q[active], available[active]
    threshold = cfg.phi * l / (capacity * t)
    n = len(active)
    masks = []
    seen = set()

    def add(mask):
        key = tuple(bool(v) for v in mask)
        if key not in seen:
            seen.add(key)
            masks.append(np.array(key))

    if paoi_weight == 0:
        # Without PAoI, completion does not change the objective's derivative
        # below threshold; frequencies above it can only waste energy.
        add(np.zeros(n, bool))
    elif n <= cfg.completion_enum_limit:
        for mask in product((False, True), repeat=n):
            add(mask)
    else:
        add(seed[active] / capacity >= threshold * (1 - 1e-12))
        add(np.zeros(n, bool))
        add(np.ones(n, bool))
        for order in (np.argsort(threshold, kind='stable'),
                      np.argsort(-q * l, kind='stable')):
            mask = np.zeros(n, bool)
            for j in order:
                trial = mask.copy()
                trial[j] = True
                f_min = np.zeros_like(L)
                f_min[active] = np.where(trial, threshold * capacity, 0)
                if feasible(f_min):
                    mask = trial
            add(mask)
            # A second completion count helps when completing everyone
            # consumes too much of the resource needed by queued tasks.
            selected = [j for j in order if mask[j]]
            half = mask.copy()
            half[selected[len(selected)//2:]] = False
            add(half)

    fixed_left = L.sum() - l.sum() + old_left
    fixed_incomplete = int(np.count_nonzero((L > 0) & (available <= 1e-6)))
    for complete in masks:
        lo = np.where(complete, threshold * (1 + 1e-12), 0.0)
        hi = np.where(complete, 1.0, np.minimum(threshold, 1.0))
        if np.any(lo > hi) or lo.sum() > 1 + 1e-12:
            continue
        m = fixed_incomplete + int(np.count_nonzero(~complete))
        # Safe lower bound for this completion set.  Queue and energy terms
        # are non-negative; completed tasks cannot run above capacity, and
        # future work cannot be smaller than the fixed residual.  If even
        # this optimistic value cannot beat the incumbent, an SLSQP call is
        # unnecessary.
        if (queue_weight >= 0 and paoi_weight >= 0 and energy_weight >= 0
                and fixed_left >= 0 and np.all(q >= 0)):
            completion_age_lb = np.sum(
                (cfg.tau - t)[complete]
                + cfg.phi * l[complete] / capacity)
            future_age_lb = m * (
                cfg.tau
                + cfg.w * cfg.phi * fixed_left / future_frequency)
            objective_lb = paoi_weight * (
                completion_age_lb + future_age_lb)
            if objective_lb >= best_score:
                continue
        # Physical energy and its derivative in normalized frequency x=f/F.
        def energy(x):
            return kappa * np.where(complete, cfg.phi * l * capacity**2 * x**2,
                                    capacity**3 * t * x**3)

        def energy_grad(x):
            return kappa * np.where(complete, 2 * cfg.phi * l * capacity**2 * x,
                                    3 * capacity**3 * t * x**2)

        if energy_limit is not None and energy(lo).sum() > energy_limit * (1 + 1e-12):
            continue
        # Constants from old work and unserviceable tasks matter for comparing
        # completion sets; final comparisons below use the exact node score.
        def objective(x):
            left = np.where(complete, 0, l - capacity * t * x / cfg.phi)
            delay = np.divide(cfg.phi * l, capacity * x,
                              out=np.zeros(n), where=complete & (x > 0))
            value = (queue_weight * np.dot(q, left) + energy_weight * energy(x).sum()
                     + paoi_weight * (delay.sum() + np.sum((cfg.tau-t)[complete])
                       + m * (cfg.tau + cfg.w * cfg.phi * (fixed_left + left.sum()) / future_frequency)))
            gradient = np.where(complete,
                -paoi_weight * np.divide(cfg.phi*l, capacity*x*x,
                                         out=np.zeros(n), where=complete & (x > 0)),
                -(queue_weight*q/cfg.phi + paoi_weight*m*cfg.w/future_frequency)*capacity*t)
            gradient += energy_weight * energy_grad(x)
            return value, gradient

        constraints = [dict(type='ineq', fun=lambda x: 1-x.sum(),
                            jac=lambda x: -np.ones(n))]
        if energy_limit is not None:
            constraints.append(dict(type='ineq',
                fun=lambda x: 1-energy(x).sum()/energy_limit,
                jac=lambda x: -energy_grad(x)/energy_limit))
        x0 = np.maximum(lo, np.minimum(hi, seed[active]/capacity))
        # Interpolate towards a known feasible lower-bound point.
        for _ in range(50):
            if x0.sum() <= 1 and (energy_limit is None or energy(x0).sum() <= energy_limit):
                break
            x0 = (x0 + lo) / 2
        result = minimize(objective, x0, jac=True, bounds=list(zip(lo, hi)),
                          constraints=constraints, method='SLSQP',
                          options={'ftol': 1e-10, 'maxiter': 100})
        # Never trust solver status alone: keep feasible improvements only.
        for x in (lo, result.x):
            f = np.zeros_like(L)
            f[active] = np.maximum(0, x) * capacity
            if feasible(f):
                value = score(f)
                if value < best_score:
                    best, best_score = f, value
    return best
