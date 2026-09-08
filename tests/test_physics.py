import unittest
import numpy as np
from tests.helpers import bookkeeping_fixture, fixed_action
from utils.lyapunov import lyapunov_value, drift_decomposition
from utils.objective import objective_coefficients


class PhysicalAccountingTests(unittest.TestCase):
    def test_satellite_backlog_has_single_source(self):
        _,env,_ = bookkeeping_fixture()
        env.sat_ledger = [np.array([[10e6]])]
        np.testing.assert_allclose(env.Q_sat_total,env.Q_sat_pending)
        np.testing.assert_allclose(env.Q_sat,np.array([[10e6]]))

    def test_old_satellite_tail_respects_per_satellite_energy_budget(self):
        cfg,env,_ = bookkeeping_fixture()
        env.sat_ledger = [np.array([[50e6]])]
        plan = env.prepare_frame()
        self.assertLessEqual(max(plan['energies']),cfg.E_max_Sat*(1+1e-9))

    def test_scored_old_satellite_energy_belongs_to_current_frame(self):
        _,env,agent = bookkeeping_fixture()
        env.sat_ledger = [np.array([[80e6]])]
        action,L = fixed_action(env,agent)
        scored_energy = action['details']['e_sat']
        env.step(action,L)
        self.assertAlmostEqual(scored_energy,env.current_e_sat_old)

    def test_new_satellite_is_not_duplicated_in_energy_denominator(self):
        cfg,env,agent = bookkeeping_fixture()
        action,L = fixed_action(env,agent,workload=100e6,sat_frequency=1e9)
        env.step(action,L)
        self.assertEqual(env.history['active_sat_count'][-1],1)
        self.assertAlmostEqual(env.history['E_virt_sat'][-1],action['details']['e_sat'])
        self.assertLessEqual(env.history['E_sat_node_max'][-1],cfg.E_max_Sat*(1+1e-6))

    def test_uncleared_bs_debt_remains_in_physical_ledger(self):
        _,env,agent = bookkeeping_fixture()
        env.Q_bs[:] = 205e6
        env.L_BS_left_prev_vec[:] = 205e6
        env.T_BS_left_prev[:] = env.cfg.tau
        action,L = fixed_action(env,agent)
        env.step(action,L)
        np.testing.assert_allclose(env.L_BS_left_prev_vec,env.Q_bs)

    def test_reported_drift_equals_lyapunov_difference(self):
        _,env,agent = bookkeeping_fixture()
        env.Q_bs[:] = 5e6
        env.L_BS_left_prev_vec[:] = 5e6
        env.sat_ledger = [np.array([[8e6]])]
        before = lyapunov_value(env.Q_bs, env.Q_sat, env.E_BS)
        combined_queue_value = 0.5 * np.sum(env.Q_total ** 2)
        independent_queue_value = 0.5 * (
            np.sum(env.Q_bs ** 2) + np.sum(env.Q_sat ** 2)
        )
        self.assertNotAlmostEqual(combined_queue_value, independent_queue_value)
        action,L = fixed_action(env,agent)
        env.step(action,L)
        after = lyapunov_value(env.Q_bs, env.Q_sat, env.E_BS)
        self.assertAlmostEqual(env.history['Drift'][-1],after-before)

    def test_independent_queue_drift_decomposition_and_upper_bound(self):
        _,env,agent = bookkeeping_fixture()
        env.Q_bs[:] = 5e6
        env.L_BS_left_prev_vec[:] = 5e6
        env.sat_ledger = [np.array([[8e6]])]
        action,L = fixed_action(env,agent,workload=10e6,sat_frequency=1e8)
        details = action['details']

        terms = drift_decomposition(
            env.Q_bs, env.Q_sat, env.E_BS,
            details['queue_delta_bs'], details['queue_delta_sat'],
            details['energy_delta_bs'],
        )
        self.assertAlmostEqual(
            terms['queue_linear'], details['objective_terms']['queue_raw']
        )
        self.assertAlmostEqual(
            terms['energy_linear'], details['objective_terms']['energy_raw']
        )
        self.assertAlmostEqual(
            terms['quadratic'], details['drift_bound_quadratic']
        )
        self.assertAlmostEqual(
            terms['upper_bound'], details['lyapunov_drift_upper_bound']
        )
        weights = objective_coefficients(env.cfg)
        normalized_terms = drift_decomposition(
            env.Q_bs, env.Q_sat, env.E_BS,
            details['queue_delta_bs'], details['queue_delta_sat'],
            details['energy_delta_bs'],
            queue_weight=weights['queue'], energy_weight=weights['energy'],
        )
        self.assertAlmostEqual(
            normalized_terms['queue_linear'],
            details['objective_terms']['queue_weighted'],
        )
        self.assertAlmostEqual(
            normalized_terms['energy_linear'],
            details['objective_terms']['energy_weighted'],
        )
        self.assertAlmostEqual(
            normalized_terms['quadratic'],
            details['normalized_drift_bound_quadratic'],
        )

        env.step(action,L)
        drift = env.history['Drift'][-1]
        upper = env.history['lyapunov_drift_upper_bound'][-1]
        self.assertLessEqual(drift, upper + 1e-9 * max(1.0, abs(upper)))
        normalized_drift = env.history['normalized_Drift'][-1]
        normalized_upper = env.history['normalized_lyapunov_drift_upper_bound'][-1]
        self.assertLessEqual(
            normalized_drift,
            normalized_upper + 1e-9 * max(1.0, abs(normalized_upper)),
        )

    def test_random_independent_queue_updates_satisfy_drift_inequality(self):
        rng = np.random.RandomState(20260908)
        for _ in range(1000):
            q_bs = rng.uniform(0.0, 20e6, (3, 10))
            q_sat = rng.uniform(0.0, 20e6, (3, 10))
            e_bs = rng.uniform(0.0, 5000.0, 3)
            delta_bs = rng.uniform(-25e6, 25e6, (3, 10))
            delta_sat = rng.uniform(-25e6, 25e6, (3, 10))
            delta_energy = rng.uniform(-500.0, 500.0, 3)

            before = lyapunov_value(q_bs, q_sat, e_bs)
            after = lyapunov_value(
                np.maximum(0.0, q_bs + delta_bs),
                np.maximum(0.0, q_sat + delta_sat),
                np.maximum(0.0, e_bs + delta_energy),
            )
            upper = drift_decomposition(
                q_bs, q_sat, e_bs,
                delta_bs, delta_sat, delta_energy,
            )['upper_bound']
            tolerance = 1e-12 * max(1.0, abs(after - before), abs(upper))
            self.assertLessEqual(after - before, upper + tolerance)

    def test_queue_conservation_for_bs_and_satellite(self):
        _,env,agent = bookkeeping_fixture()
        env.Q_bs[:] = 5e6
        env.L_BS_left_prev_vec[:] = 5e6
        env.sat_ledger = [np.array([[8e6]])]
        q_before = env.Q_total.copy()
        action,L = fixed_action(env,agent,workload=10e6,sat_frequency=1e8)
        arrival = L.copy()
        service = action['details']['l_proc_old_bs'] + action['details']['l_proc_sat']
        service += env.current_q_sat_reduction_mat
        env.step(action,L)
        np.testing.assert_allclose(env.Q_total,q_before+arrival-service,atol=1e-6)


if __name__ == '__main__':
    unittest.main()
