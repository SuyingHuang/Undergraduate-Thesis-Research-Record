"""Unresolved model defects, NOT regression successes.

Run explicitly: python -m unittest tests.known_model_issues -v
These assertions describe desired physical invariants and currently FAIL.
Do not mark them expectedFailure or weaken them to make the model appear validated.
"""
import contextlib
import io
import unittest
import numpy as np
from tests.helpers import bookkeeping_fixture, fixed_action


class KnownModelIssues(unittest.TestCase):
    def test_satellite_backlog_is_not_counted_twice(self):
        _,env,_ = bookkeeping_fixture()
        env.Q_sat[:] = 10e6
        env.sat_ledger = [np.array([[10e6]])]
        np.testing.assert_allclose(env.Q_sat_total,env.Q_sat_pending,
                                   err_msg='Q_sat already includes the same ledger leftovers')

    def test_old_satellite_tail_respects_per_satellite_energy_budget(self):
        cfg,env,agent = bookkeeping_fixture()
        env.Q_sat[:] = 50e6
        env.sat_ledger = [np.array([[50e6]])]
        action,L = fixed_action(env,agent)
        env.step(action,L)
        self.assertLessEqual(env.current_e_sat_old,cfg.E_max_Sat*(1+1e-9))

    def test_scored_old_satellite_energy_belongs_to_current_frame(self):
        _,env,agent = bookkeeping_fixture()
        env.Q_sat[:] = 80e6
        env.sat_ledger = [np.array([[80e6]])]
        action,L = fixed_action(env,agent)
        scored_energy = action['details']['e_sat']
        env.step(action,L)
        self.assertAlmostEqual(scored_energy,env.current_e_sat_old)

    def test_new_satellite_is_not_duplicated_in_energy_denominator(self):
        _,env,agent = bookkeeping_fixture()
        action,L = fixed_action(env,agent,workload=100e6,sat_frequency=1e9)
        env.step(action,L)
        self.assertAlmostEqual(env.history['E_virt_sat'][-1],action['details']['e_sat'])

    def test_small_uncleared_bs_debt_remains_in_physical_ledger(self):
        _,env,agent = bookkeeping_fixture()
        env.Q_bs[:] = 205e6
        env.L_BS_left_prev_vec[:] = 205e6
        with contextlib.redirect_stdout(io.StringIO()):
            action,L = fixed_action(env,agent)
        env.step(action,L)
        np.testing.assert_allclose(env.L_BS_left_prev_vec,env.Q_bs,
                                   err_msg='5 Mbit remains in Q_bs but disappears from service ledger')

    def test_reported_drift_equals_lyapunov_difference(self):
        _,env,agent = bookkeeping_fixture()
        before = .5*(np.sum(env.Q_total**2)+np.sum(env.E_BS**2))
        action,L = fixed_action(env,agent)
        env.step(action,L)
        after = .5*(np.sum(env.Q_total**2)+np.sum(env.E_BS**2))
        self.assertAlmostEqual(env.history['Drift'][-1],after-before)


if __name__ == '__main__':
    unittest.main()
