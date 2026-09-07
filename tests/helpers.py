import numpy as np
from config import SystemConfig
from core.env import SAGINEnvironment
from core.agents.lda_agent import LDAAgent


def small_config():
    cfg = SystemConfig()
    cfg.I, cfg.J, cfg.hidden_dim = 2, 3, 32
    cfg.batch_size, cfg.memory_capacity, cfg.train_interval = 4, 16, 1
    cfg.sim_frames = 20
    cfg._update_bandwidth_params()
    return cfg


def bookkeeping_fixture():
    """No channel sampling: directly exercise the real accounting implementation."""
    cfg = small_config()
    cfg.I, cfg.J, cfg.use_uav_relay = 1, 1, False
    cfg._update_bandwidth_params()
    env = SAGINEnvironment.__new__(SAGINEnvironment)
    env.cfg = cfg
    env.reset()
    env.current_snr_ue_leo = np.ones((1, 1))
    agent = LDAAgent.__new__(LDAAgent)
    agent.cfg = cfg
    return cfg, env, agent


def fixed_action(env, agent, workload=0.0, sat_frequency=0.0):
    L = np.array([[workload]])
    zero, one = np.zeros((1, 1)), np.ones((1, 1))
    score, details = agent.calculate_objective(
        env, L, zero, zero.astype(bool), one.astype(bool), zero,
        np.full((1, 1), sat_frequency), one * env.cfg.f_max_UE,
        zero, one * env.cfg.tau)
    return {'l': zero, 'b': zero, 'G1': score, 'details': details}, L
