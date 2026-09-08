import unittest
import numpy as np
from core.optimizers.bs_optimizer import BS_Optimizer
from core.optimizers.leo_optimizer import LEO_Optimizer
from core.optimizers.uavr_optimizer import UAVRelayOptimizer
from tests.helpers import small_config
from tests.helpers import bookkeeping_fixture
from core.agents.lda_agent import LDAAgent
from config import SystemConfig
from utils.math_utils import solve_cubic_newton, solve_cubic_newton_vectorized


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = small_config()
        self.bs = BS_Optimizer(self.cfg)
        self.leo = LEO_Optimizer(self.cfg)
        self.rng = np.random.RandomState(20260907)

    def test_bs_scalar_vector_batched_and_candidate_agree(self):
        cfg, rng = self.cfg, self.rng
        for trial in range(5):
            with self.subTest(trial=trial):
                K, I, J = 3, cfg.I, cfg.J
                L = rng.uniform(0,25e6,(K,I*J))
                L[:,0] = 0
                Q = rng.uniform(0,5e7,I*J)
                E = rng.uniform(0,5000,I)
                T = rng.uniform(0,2,(K,I*J))
                left = rng.uniform(0,1,I)
                multiple = self.bs.optimize_multi_candidate(L,Q,E,T,left)
                for k in range(K):
                    batch = self.bs.optimize_batched(L[k],Q,E,T[k],left)
                    np.testing.assert_allclose(multiple[k],batch,rtol=1e-8,atol=1e-6)
                    for i in range(I):
                        sl = slice(i*J,(i+1)*J)
                        args = (L[k,sl],Q[sl],E[i],T[k,sl],left[i])
                        scalar = self.bs.optimize(*args)
                        vector = self.bs.optimize_vectorized(*args)
                        np.testing.assert_allclose(scalar,vector,rtol=1e-8,atol=1e-6)
                        np.testing.assert_allclose(batch[sl],scalar,rtol=1e-8,atol=1e-6)
                        self.assertTrue(np.isfinite(scalar).all())
                        self.assertTrue((scalar>=0).all())
                        self.assertLessEqual(scalar.sum(),cfg.f_max_BS*(1+1e-8))

    def test_leo_scalar_vector_and_candidate_agree_and_obey_limits(self):
        cfg, rng = self.cfg, self.rng
        for trial in range(5):
            with self.subTest(trial=trial):
                N = cfg.I*cfg.J
                L = rng.uniform(0,25e6,(3,N))
                L[:,0] = 0
                Q = rng.uniform(0,1e8,N)
                available = rng.uniform(2,5,(3,N))
                multiple = self.leo.optimize_multi_candidate(L,Q,available)
                for k in range(3):
                    scalar = self.leo.optimize(L[k],Q,available[k])
                    vector = self.leo.optimize_vectorized(L[k],Q,available[k])
                    np.testing.assert_allclose(scalar,vector,rtol=1e-8,atol=1e-6)
                    np.testing.assert_allclose(multiple[k],vector,rtol=1e-8,atol=1e-6)
                    self.assertTrue(np.isfinite(vector).all())
                    self.assertTrue((vector>=0).all())
                    self.assertLessEqual(vector.sum(),cfg.f_max_Sat*(1+1e-8))
                    processed = np.minimum(L[k],vector*available[k]/cfg.phi)
                    energy = np.sum(cfg.kappa2*cfg.phi*vector**2*processed)
                    self.assertLessEqual(energy,cfg.E_max_Sat*(1+1e-6))

    def test_default_scale_high_load_bs_does_not_collapse_to_zero(self):
        cfg = SystemConfig()
        optimizer = BS_Optimizer(cfg)
        L = np.full(cfg.J, 12e6)
        Q = np.full(cfg.J, 10e6)
        transfer = np.full(cfg.J, 0.1)

        scalar = optimizer.optimize(L, Q, 100.0, transfer, 0.0)
        vector = optimizer.optimize_vectorized(L, Q, 100.0, transfer, 0.0)
        np.testing.assert_allclose(scalar, vector, rtol=1e-8, atol=1e-6)
        self.assertGreater(vector.sum(), 0.5 * cfg.f_max_BS)
        self.assertLessEqual(vector.sum(), cfg.f_max_BS * (1 + 1e-8))

        L_all = np.tile(L, cfg.I)
        Q_all = np.tile(Q, cfg.I)
        transfer_all = np.tile(transfer, cfg.I)
        batched = optimizer.optimize_batched(
            L_all, Q_all, np.full(cfg.I, 100.0), transfer_all, np.zeros(cfg.I)
        )
        candidate = optimizer.optimize_multi_candidate(
            L_all[None, :], Q_all, np.full(cfg.I, 100.0),
            transfer_all[None, :], np.zeros(cfg.I)
        )[0]
        np.testing.assert_allclose(batched, candidate, rtol=1e-8, atol=1e-6)
        for i in range(cfg.I):
            group = batched[i * cfg.J:(i + 1) * cfg.J]
            self.assertGreater(group.sum(), 0.5 * cfg.f_max_BS)
            self.assertLessEqual(group.sum(), cfg.f_max_BS * (1 + 1e-8))

    def test_default_scale_high_load_leo_and_lda2_do_not_collapse_to_zero(self):
        cfg = SystemConfig()
        N = cfg.I * cfg.J
        L = np.full(N, 12e6)
        Q = np.full(N, 10e6)
        available = np.full(N, 4.9)

        for include_paoi in (True, False):
            with self.subTest(include_paoi=include_paoi):
                optimizer = LEO_Optimizer(cfg)
                if not include_paoi:
                    optimizer.paoi_weight = 0.0
                scalar = optimizer.optimize(L, Q, available)
                vector = optimizer.optimize_vectorized(L, Q, available)
                candidate = optimizer.optimize_multi_candidate(
                    L[None, :], Q, available[None, :]
                )[0]
                np.testing.assert_allclose(scalar, vector, rtol=1e-8, atol=1e-6)
                np.testing.assert_allclose(candidate, vector, rtol=1e-8, atol=1e-6)
                self.assertGreater(vector.sum(), 0.5 * cfg.f_max_Sat)
                self.assertLessEqual(vector.sum(), cfg.f_max_Sat * (1 + 1e-8))
                processed = np.minimum(L, vector * available / cfg.phi)
                energy = np.sum(cfg.kappa2 * cfg.phi * vector ** 2 * processed)
                self.assertLessEqual(energy, cfg.E_max_Sat * (1 + 1e-6))

    def test_cubic_degenerate_negative_slope_points_to_upper_boundary(self):
        scalar = solve_cubic_newton(1e-30, 0.0, -1.0)
        vector = solve_cubic_newton_vectorized(
            np.array([1e-30]), 0.0, np.array([-1.0])
        )[0]
        self.assertTrue(np.isposinf(scalar))
        self.assertTrue(np.isposinf(vector))

    def test_no_work_or_no_time_allocates_zero(self):
        N = self.cfg.I*self.cfg.J
        zero = np.zeros(N)
        np.testing.assert_array_equal(self.leo.optimize_vectorized(zero,zero,np.ones(N)),zero)
        np.testing.assert_array_equal(self.leo.optimize_vectorized(np.ones(N)*1e6,zero,zero),zero)
        np.testing.assert_array_equal(self.bs.optimize_batched(zero,zero,np.ones(self.cfg.I),zero,
                                                             np.zeros(self.cfg.I)),zero)

    def test_zero_time_does_not_evaluate_masked_divisions(self):
        cfg = self.cfg
        N = cfg.I*cfg.J
        L, Q, zero = np.ones(N)*1e6, np.ones(N)*1e6, np.zeros(N)
        with np.errstate(divide='raise', invalid='raise'):
            np.testing.assert_array_equal(self.leo.optimize_vectorized(L,Q,zero),zero)
            np.testing.assert_array_equal(self.leo.optimize_multi_candidate(L[None,:],Q,zero[None,:]),zero[None,:])
            args = (L,Q,np.ones(cfg.I),np.ones(N)*cfg.tau,np.zeros(cfg.I))
            np.testing.assert_array_equal(self.bs.optimize_batched(*args),zero)
            np.testing.assert_array_equal(self.bs.optimize_multi_candidate(L[None,:],Q,np.ones(cfg.I),
                                         np.ones((1,N))*cfg.tau,np.zeros(cfg.I)),zero[None,:])

    def test_uav_power_stays_in_configured_range(self):
        opt = UAVRelayOptimizer(self.cfg)
        for D in (0.,1e6,12e6,100e6):
            power = opt.optimize_power(D,4.99,self.cfg.bw_per_user_sat,
                                       2.,10.,1e-12,self.cfg.sigma2)
            self.assertTrue(np.isfinite(power))
            self.assertGreaterEqual(power,opt.p_min_w)
            self.assertLessEqual(power,opt.p_max_w)

    def test_bs_allocator_matches_dense_objective_oracle(self):
        cfg,env,agent = bookkeeping_fixture()
        env.Q_bs[:] = 20e6
        env.E_BS[:] = 100.0
        L = np.array([[12e6]])
        transfer = np.array([[0.5]])
        allocated = BS_Optimizer(cfg).optimize_vectorized(
            L.ravel(),env.Q_bs.ravel(),env.E_BS[0],transfer.ravel(),0.0)[0]

        def score(frequency):
            value,_ = agent.calculate_objective(
                env,L,np.zeros((1,1),dtype=int),np.ones((1,1),dtype=bool),
                np.zeros((1,1),dtype=bool),np.array([[frequency]]),
                np.zeros((1,1)),np.ones((1,1))*cfg.f_max_UE,
                transfer,np.ones((1,1))*cfg.tau)
            return value
        grid = np.linspace(0,cfg.f_max_BS,10001)
        grid_scores = np.array([score(f) for f in grid])
        self.assertLessEqual(score(allocated),float(grid_scores.min())+2e-3)

    def test_leo_allocator_matches_dense_feasible_objective_oracle(self):
        cfg,env,agent = bookkeeping_fixture()
        env.sat_ledger = [np.array([[20e6]])]
        env.prepare_frame()
        L = np.array([[12e6]])
        available = np.array([[4.0]])
        allocated = LEO_Optimizer(cfg).optimize_vectorized(
            L.ravel(),env.Q_sat.ravel(),available.ravel())[0]

        def score(frequency):
            value,details = agent.calculate_objective(
                env,L,np.zeros((1,1),dtype=int),np.zeros((1,1),dtype=bool),
                np.ones((1,1),dtype=bool),np.zeros((1,1)),
                np.array([[frequency]]),np.ones((1,1))*cfg.f_max_UE,
                np.zeros((1,1)),available)
            return value,details['e_sat_new']
        grid = np.linspace(0,cfg.f_max_Sat,10001)
        evaluated = [score(f) for f in grid]
        feasible_scores = [value for value,energy in evaluated
                           if energy <= cfg.E_max_Sat*(1+1e-9)]
        allocated_score,allocated_energy = score(allocated)
        self.assertLessEqual(allocated_energy,cfg.E_max_Sat*(1+1e-6))
        self.assertLessEqual(allocated_score,min(feasible_scores)+2e-3)


if __name__ == '__main__':
    unittest.main()
