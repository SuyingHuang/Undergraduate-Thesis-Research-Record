"""Read-only, deterministic probes; run from the repository root."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SystemConfig
from core.optimizers.bs_optimizer import BS_Optimizer
from core.optimizers.leo_optimizer import LEO_Optimizer


def main():
    cfg = SystemConfig()
    n = cfg.I * cfg.J
    records = []
    for equal in (True, False):
        workloads = np.full(n, 12e6) if equal else np.linspace(9e6, 15e6, n)
        for include_paoi in (True, False):
            bs, sat = BS_Optimizer(cfg), LEO_Optimizer(cfg)
            if not include_paoi:
                bs.paoi_weight = sat.paoi_weight = 0.0
            args = (workloads, np.zeros(n), np.zeros(cfg.I),
                    np.full(n, 0.1), np.zeros(cfg.I))
            frequencies = bs.optimize_batched(*args)
            candidate_frequencies = bs.optimize_multi_candidate(
                workloads[None, :], args[1], args[2], args[3][None, :], args[4])
            satellite_frequencies = sat.optimize_vectorized(
                workloads, np.zeros(n), np.full(n, 4.9))
            records.append(dict(equal_workloads=equal, include_paoi=include_paoi,
                                bs_total_GHz=float(frequencies.sum()/1e9),
                                bs_candidate_total_GHz=float(candidate_frequencies.sum()/1e9),
                                satellite_total_GHz=float(satellite_frequencies.sum()/1e9)))
    # All-BS action, empty physical/virtual queues, no old work.
    # This feasible allocation even respects the 180 J per-BS average budget
    # in this single frame, although that budget is not a hard frame cap.
    f = 250e6
    witness = dict(per_user_frequency_Hz=f,
                   per_bs_frequency_Hz=cfg.J*f,
                   per_bs_energy_J=cfg.J*cfg.kappa1*cfg.phi*f*f*12e6,
                   feasible_delay_seconds=0.1+cfg.phi*12e6/f,
                   returned_zero_delay_seconds=cfg.tau+cfg.w*cfg.phi*(cfg.J*12e6)/cfg.f_max_BS)
    print(json.dumps(dict(probes=records, feasible_bs_witness=witness), indent=2))


if __name__ == '__main__':
    main()
