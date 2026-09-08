import unittest
import numpy as np
import torch
from core.models.dnn_model import OffloadingActor, get_input_vector, FocalLoss
from core.models.tcopq import check_local_feasibility, generate_candidates
from core.agents.lda_agent import LDAAgent
from core.agents.baselines import COBAgent, MTDAgent, ACAgent
from tests.helpers import small_config, bookkeeping_fixture, fixed_action
from utils.reproducibility import set_seed


class StateAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        set_seed(42)
        self.cfg = small_config()

    def test_state_layout_and_scaling(self):
        one = np.ones((2, 3))
        state = get_input_vector(one*12e6, one*2e6, one*3e6,
                                 np.ones(2)*20, np.ones(2), one*2e7, one*1e7)
        self.assertEqual(tuple(state.shape), (2, 35))
        local = [12.,12.,12.,2.,2.,2.,3.,3.,3.,2.,1.,1.,1.,1.,1.,1.,1.]
        shared = [12.]*6 + [3.]*6 + [.76]*6
        expected = torch.tensor([local+shared]*2)
        torch.testing.assert_close(state, expected)

    def test_current_task_updates_local_and_global_competition_entries(self):
        one = np.ones((2, 3))
        L = one*12e6
        kwargs = dict(
            Q_bs=one, Q_sat=one, E=np.ones(2), T_left=np.ones(2),
            R_BS=one*2e7, R_LEOS=one*1e7,
            T_prop=np.zeros_like(one), offload_mask=np.ones_like(one,dtype=bool),
            tau=5.0,
        )
        before = get_input_vector(L, **kwargs)
        L[0, 1] += 1e6
        difference = get_input_vector(L, **kwargs) - before
        expected = torch.zeros_like(difference)
        expected[0, 1] = 1
        expected[:, 18] = 1
        expected[:, 30] = -.02
        torch.testing.assert_close(difference, expected)

    def test_actor_dimensions_and_layernorm(self):
        for J in (3, 10, 14):
            actor = OffloadingActor(J, num_bs=2, hidden_dim=32)
            self.assertEqual(actor.input_dim, 5*J+2+6*J)
            state = torch.randn(2, actor.input_dim)
            actor.train()
            train = actor(state)
            actor.eval()
            torch.testing.assert_close(train, actor(state))
            self.assertEqual(tuple(train.shape), (2, J))

    def test_default_actor_grows_with_shared_satellite_context(self):
        cfg = __import__('config').SystemConfig()
        actor = OffloadingActor(cfg.J,num_bs=cfg.I,
                                hidden_dim=cfg.hidden_dim)
        # 重构前结构：input=5J+2=52，hidden=512，共 689,930 个参数。
        old_parameter_count = 689_930
        self.assertEqual(actor.input_dim,142)
        self.assertEqual(sum(p.numel() for p in actor.parameters()),1_124_810)
        self.assertGreater(sum(p.numel() for p in actor.parameters()),
                           old_parameter_count)

    def test_current_satellite_competition_changes_every_actor_state(self):
        one = np.ones((2,3))
        kwargs = dict(L_t=one*12e6,Q_bs=one,Q_sat=one,E=np.ones(2),
                      T_left=np.ones(2),R_BS=one,R_LEOS=one*1e7,
                      T_prop=np.zeros_like(one),
                      offload_mask=np.ones_like(one,dtype=bool),tau=5.0)
        base = get_input_vector(**kwargs)
        changed_rates = kwargs['R_LEOS'].copy()
        changed_rates[1,2] = 2e7
        changed = get_input_vector(**{**kwargs,'R_LEOS':changed_rates})
        self.assertFalse(torch.equal(base[0],changed[0]))
        self.assertFalse(torch.equal(base[1],changed[1]))

    def test_local_only_task_is_excluded_from_shared_satellite_competition(self):
        one = np.ones((2,3))
        offload = np.ones((2,3),dtype=bool)
        offload[1,2] = False
        kwargs = dict(L_t=one*12e6,Q_bs=one,Q_sat=one*3e6,E=np.ones(2),
                      T_left=np.ones(2),R_BS=one,R_LEOS=one*1e7,
                      T_prop=np.zeros_like(one),offload_mask=offload,tau=5.0)
        base = get_input_vector(**kwargs)
        changed_L = kwargs['L_t'].copy()
        changed_L[1,2] += 10e6
        changed = get_input_vector(**{**kwargs,'L_t':changed_L})
        # Actor 0 has no local entry for this user, and the shared mask removes it.
        torch.testing.assert_close(base[0],changed[0])
        self.assertFalse(torch.equal(base[1],changed[1]))

    def test_old_checkpoint_dimension_is_rejected(self):
        actor = OffloadingActor(3, num_bs=2, hidden_dim=32)
        old = actor.state_dict()
        old['input_proj.weight'] = torch.zeros((32, 5*3+2))
        with self.assertRaises(RuntimeError):
            actor.load_state_dict(old)

    def test_default_focal_loss_is_half_mean_bce(self):
        logits = torch.tensor([[0., 2., -2.]], requires_grad=True)
        targets = torch.tensor([[0., 1., 0.]])
        loss = FocalLoss()(logits, targets)
        torch.testing.assert_close(loss, .5*torch.nn.functional.binary_cross_entropy_with_logits(logits, targets))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_replay_warmup_then_real_gradient_update(self):
        for cls in (LDAAgent, ACAgent):
            with self.subTest(agent=cls.__name__):
                agent = cls(self.cfg)
                before = [p.detach().clone() for p in agent.actors.parameters()]
                state = torch.ones((self.cfg.I, agent.actors[0].input_dim))
                labels = np.zeros((self.cfg.I, self.cfg.J))
                for t in range(15):
                    agent.store_experience(state, labels)
                    agent.train(t)
                self.assertEqual(agent.loss_history, [])
                agent.store_experience(state, labels)
                agent.train(15)
                self.assertEqual(len(agent.loss_history), 1)
                self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,agent.actors.parameters())))
                for _ in range(4):
                    agent.store_experience(state, labels)
                self.assertTrue(all(len(m)==16 for m in agent.memories))

    def test_heuristics_have_no_unused_learning_state(self):
        for cls in (COBAgent, MTDAgent):
            agent = cls(self.cfg)
            self.assertFalse(hasattr(agent, 'actors'))
            self.assertFalse(hasattr(agent, 'memories'))
            self.assertFalse(hasattr(agent, 'delta_t'))
            agent.train(0)


class QuantizationTests(unittest.TestCase):
    def test_local_boundary(self):
        cfg = small_config()
        boundary = cfg.tau*cfg.f_max_UE/cfg.phi
        actual = check_local_feasibility(np.array([0., boundary, boundary+1]),
                                         np.full(3,cfg.f_max_UE), cfg)
        np.testing.assert_array_equal(actual, [1,1,0])

    def test_candidates_unique_local_mask_and_cumulative_flips(self):
        p, local = np.array([.51,.48,.7,.49]), np.array([0,0,0,1])
        candidates = generate_candidates(p, .5, local)
        bits = [tuple(b) for _,b in candidates]
        self.assertEqual(len(bits), len(set(bits)))
        self.assertEqual(bits[0], (1,0,1,0))
        self.assertIn((0,1,0,0), bits)
        for l,b in candidates:
            np.testing.assert_array_equal(l, local)
            self.assertEqual(b[3],0)

    def test_all_local_or_empty_window_keeps_base_candidate(self):
        self.assertEqual(len(generate_candidates(np.array([.1,.8]), .5, np.ones(2))),1)
        self.assertEqual(len(generate_candidates(np.array([.1,.8]), 0., np.zeros(2))),1)

    def test_one_all_local_bs_does_not_truncate_other_bs_candidates(self):
        local_only = [(np.ones(3,dtype=int),np.zeros(3,dtype=int))]
        exploring = generate_candidates(np.array([.49,.51,.8]),.5,np.zeros(3,dtype=int))
        current = np.array([local_only[0][1],exploring[0][1]])
        proposals = LDAAgent._coordinate_proposals(current,[local_only,exploring])
        self.assertEqual(len(proposals),len(exploring))
        self.assertTrue(all(np.array_equal(p[0],current[0]) for p in proposals))

    def test_environment_reset_clears_cross_run_state(self):
        _, env, _ = bookkeeping_fixture()
        env.sat_ledger.append(np.ones((1,1)))
        env.Q_bs[:] = 3
        env.E_BS[:] = 5
        env.reset()
        self.assertEqual(env.sat_ledger, [])
        self.assertEqual(float(env.Q_total.sum()),0.)
        self.assertEqual(float(env.E_BS.sum()),0.)


if __name__ == '__main__':
    unittest.main()
