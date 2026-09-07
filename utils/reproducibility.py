"""Shared RNG setup for simulation entry points."""
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Seed Python/NumPy/PyTorch; hash randomization must be set before startup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
