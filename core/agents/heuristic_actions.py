"""The exact same complete actions are used by baselines and candidate audit."""
import numpy as np


def baseline_actions(local, workload, satellite_rate, k_sat=2):
    cob = np.ones(local.shape, dtype=int)
    mtd = cob.copy()
    delay = workload / (satellite_rate + 1e-9)
    for i in range(local.shape[0]):
        order = np.argsort(np.where(local[i] == 0, delay[i], np.inf))
        selected = [j for j in order if local[i, j] == 0][:k_sat]
        mtd[i, selected] = 0
    return cob, mtd
