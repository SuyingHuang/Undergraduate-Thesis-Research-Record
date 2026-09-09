import unittest
from unittest.mock import patch

import numpy as np

from config import SystemConfig
from core.agents.baselines import COBAgent, MTDAgent
from core.agents.heuristic_actions import baseline_actions
from core.agents.lda_agent import LDAAgent
from core.optimizers.bs_optimizer import BS_Optimizer, LegacyBS_Optimizer
from core.optimizers.leo_optimizer import LEO_Optimizer
from core.optimizers.coupled import node_metrics, recover_primal
from tests.helpers import bookkeeping_fixture, small_config
from utils.old_bs import old_bs_service


class CoupledTests(unittest.TestCase):
    def test_empty_queue_equal_tasks_beat_known_feasible_witness(self):
        cfg = SystemConfig()
        L = np.full(cfg.J, 12e6)
        Q = np.zeros(cfg.J)
        f = BS_Optimizer(cfg).optimize(L, Q, 0, np.full(cfg.J, .1), 0)
        left, age, energy = node_metrics(L, f, np.full(cfg.J, 4.9), cfg,
                                         cfg.kappa1, cfg.f_max_BS)
        self.assertLessEqual(f.sum(), cfg.f_max_BS*(1+1e-10))
        self.assertLessEqual(age.mean(), 4.9)
        self.assertEqual(left.sum(), 0)

    def test_shared_residual_includes_unserviceable_tasks_and_old_work(self):
        cfg = SystemConfig()
        left, age, _ = node_metrics(np.array([12e6, 10e6]), np.array([1e8, 0.]),
                                    np.array([5., 0.]), cfg, cfg.kappa1,
                                    cfg.f_max_BS, old_left=3e6)
        np.testing.assert_allclose(left, [7e6, 10e6])
        np.testing.assert_allclose(age, np.full(2, 5+2*100*20e6/4e9))

    def test_two_user_bs_matches_independent_dense_shared_objective(self):
        cfg = SystemConfig()
        cfg.I, cfg.J = 1, 2
        L, Q = np.array([16e6, 23e6]), np.array([3e7, 6e7])
        t = np.array([1., 3.])
        E = 600.
        solver = BS_Optimizer(cfg)
        f = solver.optimize(L, Q, E, cfg.tau-t, 0)
        # Independent primal grid includes interior energy-saving solutions.
        x = np.linspace(0, cfg.f_max_BS, 701)
        a, b = np.meshgrid(x, x)
        grid = np.column_stack((a.ravel(), b.ravel()))
        grid = grid[grid.sum(axis=1) <= cfg.f_max_BS]

        def score(fs):
            remaining = np.maximum(0, L - fs*t/cfg.phi)
            partial = remaining > 1e-6
            time = cfg.tau-t + np.divide(cfg.phi*L, fs,
                                        out=np.zeros_like(fs), where=fs > 0)
            age = np.where(partial, cfg.tau+cfg.w*cfg.phi*remaining.sum(axis=-1, keepdims=True)/cfg.f_max_BS, time)
            energy = cfg.kappa1*cfg.phi*fs**2*(L-remaining)
            return (solver.queue_weight*(remaining*Q).sum(axis=-1)
                    + solver.paoi_weight*age.sum(axis=-1)
                    + E*solver.energy_weight*energy.sum(axis=-1))
        self.assertLessEqual(float(score(f[None, :])[0]), float(score(grid).min())+1e-5)

    def test_upper_score_uses_same_node_delay(self):
        cfg, env, agent = bookkeeping_fixture()
        cfg.I, cfg.J = 1, 2
        env.reset()
        L = np.array([[12e6, 15e6]])
        f = np.array([[1e8, 2e8]])
        zero = np.zeros_like(L)
        _, details = agent.calculate_objective(env, L, zero, np.ones_like(L, bool),
            np.zeros_like(L, bool), f, zero, zero+cfg.f_max_UE,
            zero+.1, zero+4.9)
        _, age, _ = node_metrics(L[0], f[0], np.full(2, 4.9), cfg, cfg.kappa1, cfg.f_max_BS)
        np.testing.assert_allclose(details['paoi'][0], age)

    def test_ablation_switches_are_independent(self):
        for mode, upper, lower in [('none', True, True), ('upper', False, True),
                                   ('lower', True, False), ('both', False, False)]:
            cfg = small_config()
            cfg.paoi_ablation = mode
            agent = COBAgent(cfg)
            self.assertEqual(agent.upper_paoi_enabled, upper)
            self.assertEqual(agent.bs_opt.paoi_weight > 0, lower)
            self.assertEqual(agent.leo_opt.paoi_weight > 0, lower)

    def test_mtd_audit_action_matches_baseline_including_local_mask(self):
        cfg = small_config()
        agent = MTDAgent(cfg)
        L = np.array([[1e6, 12e6, 15e6], [13e6, 12e6, 1e6]])
        rate = np.array([[1., 3., 2.], [1., 2., 3.]])*1e7
        local = (cfg.phi*L/cfg.f_max_UE <= cfg.tau).astype(int)
        _, expected = baseline_actions(local, L, rate)
        with patch.object(agent, '_evaluate_fixed_action', side_effect=lambda *args: args[-1]):
            actual = agent.select_action(None, L, rate, rate, np.zeros_like(L))
        np.testing.assert_array_equal(actual, expected)

    def test_energy_aware_old_service_reduces_frequency_under_energy_pressure(self):
        cfg = SystemConfig()
        cfg.old_bs_policy = 'energy_aware'
        workload = np.full((1, cfg.J), 12e6)
        served_free, energy_free, occupied_free = old_bs_service(cfg, workload, np.array([0.]))
        served_costly, energy_costly, occupied_costly = old_bs_service(cfg, workload, np.array([1e6]))
        self.assertLess(energy_costly[0], energy_free[0])
        self.assertLess(served_costly.sum(), served_free.sum())
        self.assertLessEqual(occupied_costly[0], cfg.tau)
        np.testing.assert_allclose(served_free, workload)
        self.assertGreaterEqual(np.min(served_costly), 0)

    def test_guarded_action_dominates_same_state_baseline_scores(self):
        cfg = small_config()
        from core.env import SAGINEnvironment
        env = SAGINEnvironment.__new__(SAGINEnvironment)
        env.cfg = cfg
        env.reset()
        agent = LDAAgent(cfg)
        L = np.full((cfg.I, cfg.J), 12e6)
        rates = np.full_like(L, 2e7)
        action = agent.select_action(env, L, rates, rates, np.full_like(L, .01), t=1)
        for cls in (COBAgent, MTDAgent):
            baseline = cls(cfg).select_action(env, L, rates, rates, np.full_like(L, .01), t=1)
            self.assertLessEqual(action['G1'], baseline['G1']+1e-9)
        self.assertIn('baseline_improvement', action['candidate_audit'])

    def test_joint_action_cache_skips_repeated_resource_solves(self):
        cfg, env, _ = bookkeeping_fixture()
        agent = LDAAgent(cfg)
        L = np.full((cfg.I, cfg.J), 12e6)
        rates = np.full_like(L, 2e7)
        prop = np.full_like(L, .01)
        local = np.zeros_like(L, dtype=int)
        candidate = np.zeros_like(L, dtype=int)
        cache = {}

        with patch.object(
                agent.leo_opt, 'optimize_multi_candidate',
                wraps=agent.leo_opt.optimize_multi_candidate) as leo_solve:
            first = agent._evaluate_joint_candidates(
                env, L, rates, rates, prop, local, [candidate],
                solution_cache=cache)
            second = agent._evaluate_joint_candidates(
                env, L, rates, rates, prop, local, [candidate],
                solution_cache=cache)

        self.assertIs(first, second)
        self.assertEqual(leo_solve.call_count, 1)

    def test_no_paoi_bs_coupled_solution_is_legacy_kkt_solution(self):
        cfg = small_config()
        optimizer = BS_Optimizer(cfg)
        optimizer.paoi_weight = 0.0
        rng = np.random.RandomState(20260909)
        L = rng.uniform(0, 25e6, cfg.I * cfg.J)
        Q = rng.uniform(0, 1e8, cfg.I * cfg.J)
        E = rng.uniform(0, 5000, cfg.I)
        transfer = rng.uniform(0, 2, cfg.I * cfg.J)
        left = rng.uniform(0, 1, cfg.I)

        coupled = optimizer.optimize_multi_candidate(
            L[None, :], Q, E, transfer[None, :], left)[0]
        legacy = LegacyBS_Optimizer.optimize_batched(
            optimizer, L, Q, E, transfer, left)
        np.testing.assert_allclose(coupled, legacy, rtol=1e-8, atol=1e-6)

    def test_primal_lower_bound_prunes_when_incumbent_is_zero(self):
        cfg = small_config()
        zero = np.zeros(cfg.J)
        with patch('core.optimizers.coupled.minimize') as minimize_mock:
            actual = recover_primal(
                np.full(cfg.J, 12e6), zero, np.full(cfg.J, cfg.tau),
                zero, cfg, capacity=cfg.f_max_BS, kappa=cfg.kappa1,
                queue_weight=1.0, paoi_weight=0.0)
        np.testing.assert_array_equal(actual, zero)
        minimize_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
