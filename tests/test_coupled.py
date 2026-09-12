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
from utils.old_bs import (
    budget_limited_old_frequency,
    joint_dpp_frequency_candidates,
    old_bs_service,
    old_bs_service_at_frequency,
)


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

    def test_budgeted_old_service_uses_reserved_energy_without_exceeding_it(self):
        cfg = SystemConfig()
        cfg.old_bs_policy = 'budgeted'
        cfg.old_bs_energy_budget_fraction = 0.5
        workload = np.full((1, cfg.J), 12e6)
        served, energy, occupied = old_bs_service(
            cfg, workload, np.array([1e6]))
        self.assertLessEqual(
            energy[0], cfg.E_max_BS * cfg.old_bs_energy_budget_fraction
            * (1 + 1e-9))
        self.assertGreater(served.sum(), 0.0)
        self.assertLess(occupied[0], cfg.tau)

    def test_budgeted_old_service_rejects_invalid_fraction(self):
        cfg = SystemConfig()
        cfg.old_bs_policy = 'budgeted'
        cfg.old_bs_energy_budget_fraction = 0.0
        with self.assertRaises(ValueError):
            old_bs_service(
                cfg, np.full((1, cfg.J), 12e6), np.array([0.0]))

    def test_budgeted_energy_inversion_holds_across_workloads_and_shares(self):
        cfg = SystemConfig()
        cfg.old_bs_policy = 'budgeted'
        rng = np.random.RandomState(20260911)
        for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
            cfg.old_bs_energy_budget_fraction = fraction
            for scale in (1e5, 1e6, 10e6, 100e6):
                workload = rng.uniform(0.1, 2.0, (cfg.I, cfg.J)) * scale
                _, energy, _ = old_bs_service(
                    cfg, workload, np.zeros(cfg.I))
                self.assertTrue(np.all(
                    energy <= fraction * cfg.E_max_BS * (1 + 1e-9)))

    def test_explicit_old_frequency_preserves_proportions_and_accounting(self):
        cfg = SystemConfig()
        workload = np.array([[4e6, 8e6, 12e6] + [0.0] * (cfg.J - 3)])
        frequency = np.array([1.5e9])
        processed, energy, occupied = old_bs_service_at_frequency(
            cfg, workload, frequency)
        expected = np.minimum(
            workload,
            frequency[:, None] * (workload / workload.sum(axis=1)[:, None])
            * cfg.tau / cfg.phi,
        )
        np.testing.assert_allclose(processed, expected)
        expected_energy = np.sum(
            cfg.kappa1 * cfg.phi
            * (frequency[:, None]
               * workload / workload.sum(axis=1)[:, None]) ** 2
            * processed,
            axis=1,
        )
        np.testing.assert_allclose(energy, expected_energy)
        np.testing.assert_allclose(
            occupied,
            np.minimum(cfg.tau, cfg.phi * workload.sum(axis=1) / frequency),
        )

    def test_joint_dpp_candidates_include_boundaries_and_transitions(self):
        cfg = SystemConfig()
        workload = np.full(cfg.J, 10e6)
        candidates = joint_dpp_frequency_candidates(
            cfg, workload, 1000.0, transition_times=[2.5])
        self.assertIn(0.0, candidates)
        self.assertIn(cfg.f_max_BS, candidates)
        self.assertIn(cfg.phi * workload.sum() / cfg.tau, candidates)
        self.assertIn(
            min(cfg.f_max_BS, cfg.phi * workload.sum() / 2.5), candidates)
        witness = budget_limited_old_frequency(
            cfg, workload[None, :],
            cfg.joint_dpp_budgeted_witness_fraction)[0]
        self.assertIn(witness, candidates)

    def test_budgeted_witness_exactly_reproduces_budgeted_service(self):
        cfg = SystemConfig()
        cfg.old_bs_policy = 'budgeted'
        cfg.old_bs_energy_budget_fraction = 0.75
        workload = np.linspace(2e6, 20e6, cfg.J)[None, :]
        expected = old_bs_service(cfg, workload, np.array([123.0]))
        frequency = budget_limited_old_frequency(cfg, workload, 0.75)
        actual = old_bs_service_at_frequency(cfg, workload, frequency)
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(left, right)

    def test_joint_dpp_score_dominates_full_frequency_witness(self):
        cfg, env, agent = bookkeeping_fixture()
        cfg.old_bs_policy = 'joint_dpp'
        cfg.joint_dpp_old_frequency_grid_points = 5
        env.Q_bs[:] = 40e6
        env.L_BS_left_prev_vec[:] = 40e6
        env.E_BS[:] = 1e5
        env.prepare_frame()
        agent.bs_opt = BS_Optimizer(cfg)
        agent.leo_opt = LEO_Optimizer(cfg)
        agent.upper_paoi_enabled = True

        L = np.array([[20e6]])
        l_mat = np.zeros((1, 1), dtype=int)
        b_mat = np.ones((1, 1), dtype=int)
        rate = np.full((1, 1), 1e8)
        prop = np.zeros((1, 1))
        selected = agent._evaluate_joint_candidates(
            env, L, rate, rate, prop, l_mat, [b_mat])
        misses = agent._joint_dpp_cache_misses
        repeated = agent._evaluate_joint_candidates(
            env, L, rate, rate, prop, l_mat, [b_mat])
        self.assertEqual(agent._joint_dpp_cache_misses, misses)
        self.assertGreater(agent._joint_dpp_cache_hits, 0)
        self.assertEqual(repeated['G1'], selected['G1'])

        old_processed, old_energy, old_occupied = (
            old_bs_service_at_frequency(
                cfg, env.L_BS_left_prev_vec,
                np.array([cfg.f_max_BS])))
        old_left = float(np.sum(
            env.L_BS_left_prev_vec - old_processed))
        transfer = L / rate
        current_frequency = agent.bs_opt.optimize(
            L[0], env.Q_bs[0], env.E_BS[0], transfer[0],
            float(old_occupied[0]), old_left=old_left)[None, :]
        zero = np.zeros((1, 1))
        full_score, _ = agent.calculate_objective(
            env, L, l_mat, np.ones((1, 1), dtype=bool),
            np.zeros((1, 1), dtype=bool), current_frequency, zero,
            np.full((1, 1), cfg.f_max_UE), transfer,
            np.full((1, 1), cfg.tau),
            old_bs_plan={
                'processed': old_processed,
                'energy': old_energy,
                'occupied': old_occupied,
                'aggregate_frequency': np.array([cfg.f_max_BS]),
            },
        )
        self.assertLessEqual(selected['G1'], full_score + 1e-12)
        self.assertLess(
            selected['details']['old_bs_aggregate_frequency'][0],
            cfg.f_max_BS,
        )

        witness_frequency = budget_limited_old_frequency(
            cfg, env.L_BS_left_prev_vec,
            cfg.joint_dpp_budgeted_witness_fraction)
        witness_processed, witness_energy, witness_occupied = (
            old_bs_service_at_frequency(
                cfg, env.L_BS_left_prev_vec, witness_frequency))
        witness_left = float(np.sum(
            env.L_BS_left_prev_vec - witness_processed))
        witness_current = agent.bs_opt.optimize(
            L[0], env.Q_bs[0], env.E_BS[0], transfer[0],
            float(witness_occupied[0]), old_left=witness_left)[None, :]
        witness_score, _ = agent.calculate_objective(
            env, L, l_mat, np.ones((1, 1), dtype=bool),
            np.zeros((1, 1), dtype=bool), witness_current, zero,
            np.full((1, 1), cfg.f_max_UE), transfer,
            np.full((1, 1), cfg.tau),
            old_bs_plan={
                'processed': witness_processed,
                'energy': witness_energy,
                'occupied': witness_occupied,
                'aggregate_frequency': witness_frequency,
            },
        )
        self.assertLessEqual(selected['G1'], witness_score + 1e-12)

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
