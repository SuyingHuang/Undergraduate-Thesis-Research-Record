import unittest
import numpy as np
from core.optimizers.bs_optimizer import BS_Optimizer
from core.optimizers.leo_optimizer import LEO_Optimizer
from core.optimizers.uavr_optimizer import UAVRelayOptimizer
from tests.helpers import small_config


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


if __name__ == '__main__':
    unittest.main()
