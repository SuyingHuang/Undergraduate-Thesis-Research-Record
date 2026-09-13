"""Shared RNG setup for simulation entry points."""
import random
import numpy as np
import torch


def set_seed(seed=42, use_cuda=True):
    """Seed Python/NumPy/PyTorch; hash randomization must be set before startup.

    ``use_cuda=False`` is for CPU-only agents (COB/MTD).  ``torch.manual_seed``
    forwards to ``torch.cuda.manual_seed_all`` for *every* caller, so seeding a
    heuristic worker used to create a CUDA context anyway.  In a multiprocess
    sweep one unhealthy GPU then aborted the whole pool, including the CPU-only
    algorithms, instead of a single learning run.  This branch seeds only the
    CPU generator, which leaves the Python/NumPy/CPU-torch streams identical.
    """
    random.seed(seed)
    np.random.seed(seed)
    if not use_cuda:
        torch.default_generator.manual_seed(seed)
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
