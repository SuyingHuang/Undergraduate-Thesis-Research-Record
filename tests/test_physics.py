import unittest
import numpy as np
from tests.helpers import bookkeeping_fixture, fixed_action


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
        before = .5*(np.sum(env.Q_total**2)+np.sum(env.E_BS**2))
        action,L = fixed_action(env,agent)
        env.step(action,L)
        after = .5*(np.sum(env.Q_total**2)+np.sum(env.E_BS**2))
        self.assertAlmostEqual(env.history['Drift'][-1],after-before)

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
