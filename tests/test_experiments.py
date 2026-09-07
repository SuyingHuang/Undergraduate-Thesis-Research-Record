import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
from tests.helpers import small_config, bookkeeping_fixture, fixed_action
from run_sweeps import (_find_convergence_frame, extract_metric_bundle, _ci95,
                        _aggregate_metric_rows, _scenario_hash, _paired_comparisons,
                        run_experiment_sweep, main)
from collect_calibration import compute_raw_terms, run_single_seed
from core.agents.lda_agent import LDAAgent
from core.agents.baselines import ACAgent


class ExperimentTests(unittest.TestCase):
    def test_constant_learning_window_is_not_heuristic(self):
        self.assertEqual(_find_convergence_frame({'delta_t':[.5]*20}),0)
        self.assertEqual(_find_convergence_frame({}),-1)
        self.assertEqual(_find_convergence_frame({'delta_t':[.5,.2,.1]}),0)
        self.assertEqual(_find_convergence_frame({'delta_t':[.5,.2,.1]},.1),2)

    def test_fixed_window_is_same_for_both_algorithm_types(self):
        base = {key:list(range(20)) for key in ('Cost','Q_total','E_virt_bs','E_virt_sat')}
        for history in (base,dict(base,delta_t=[.5]*20)):
            bundle = extract_metric_bundle(history)
            self.assertEqual(bundle['fixed_half_start'],10)
            self.assertEqual(bundle['fixed_half']['PAoI'],14.5)

    def test_primary_aggregation_keeps_all_successful_seeds(self):
        rows = []
        for i,value in enumerate((1.,2.,30.)):
            rows.append(dict(algo='LDA',param_val=1,failed=False,sim_frames=100,
                             conv_frame=0,cleaned_out=True,PAoI=value,E_BS=value,E_LEO=value,Q=value))
        rows.append(dict(rows[0],failed=True,PAoI=999.))
        result, summary = _aggregate_metric_rows(rows,[('LDA',LDAAgent)],[1])
        self.assertEqual(result['LDA1']['n_valid'],[3])
        self.assertEqual(result['LDA1']['n_failed'],[1])
        self.assertEqual(result['LDA1']['PAoI'],[11.])
        self.assertEqual(summary[0]['n_cleaned'],0)

    def test_confidence_interval_requires_repeated_observations(self):
        mean,std,ci,n = _ci95([1.,np.nan,np.inf])
        self.assertEqual((mean,n),(1.,1))
        self.assertTrue(np.isnan(std) and np.isnan(ci))
        self.assertEqual(_ci95([1.,3.])[0],2.)
        self.assertAlmostEqual(_ci95([1.,3.])[2],12.706204736,places=6)
        self.assertEqual(_ci95([])[3],0)

    def test_paired_comparisons_use_only_matching_successful_seeds(self):
        rows = []
        for algo,values in [('LDA',{1:10.,2:20.}),('AC',{1:12.,2:19.,3:100.})]:
            for seed,value in values.items():
                rows.append(dict(algo=algo,param_val=.1,seed=seed,failed=False,
                                 **{metric:value for metric in ('PAoI','E_BS','E_LEO','Q')}))
        comparison = _paired_comparisons(rows,[.1])[0]
        self.assertEqual(comparison['paired_seeds'],[1,2])
        self.assertEqual(comparison['n_pairs'],2)
        self.assertEqual(comparison['PAoI_difference_mean'],.5)

    def test_scenario_hash_detects_value_and_shape_changes(self):
        a = np.ones((2,3))
        self.assertEqual(_scenario_hash(a),_scenario_hash(a.copy()))
        self.assertNotEqual(_scenario_hash(a),_scenario_hash(a.reshape(3,2)))
        self.assertNotEqual(_scenario_hash(a),_scenario_hash(a+1))

    def test_invalid_sweep_inputs_fail_before_writing(self):
        cfg = small_config()
        for kwargs in ({'sim_frames':0},{'seeds':[]},{'seeds':[42,42]},{'metric_view':'invalid'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_experiment_sweep('invalid','K_p',[.1],[('LDA',LDAAgent)],cfg,**kwargs)

    def test_cli_help_does_not_start_simulation(self):
        with patch('run_sweeps.run_experiment_sweep') as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(['--help'])
            self.assertEqual(error.exception.code,0)
            run.assert_not_called()

    def test_calibration_terms_match_decision_score(self):
        cfg,env,agent = bookkeeping_fixture()
        env.sat_ledger = [np.ones((1,1))*1e6]
        env.E_BS[:] = 20
        action,_ = fixed_action(env,agent,workload=10e6,sat_frequency=1e8)
        q,p,e = compute_raw_terms(env,action['details'],cfg)
        self.assertAlmostEqual(action['G1'],q/cfg.Q_ref+p/cfg.PAoI_ref+e/cfg.E_ref)

    def test_ac_removes_paoi_from_scoring_and_both_allocators(self):
        cfg = small_config()
        agent = ACAgent(cfg)
        self.assertEqual(agent.bs_opt.paoi_weight,0.0)
        self.assertEqual(agent.leo_opt.paoi_weight,0.0)

    def test_calibration_collects_before_step_and_never_trains(self):
        events = []
        cfg = small_config()
        with patch('collect_calibration.SystemConfig',return_value=cfg), \
             patch('collect_calibration.SAGINEnvironment') as Env, \
             patch('collect_calibration.LDAAgent') as Agent, \
             patch('collect_calibration.compute_raw_terms') as terms:
            Env.return_value.generate_channel_states.return_value = (1.,1.,1.)
            Agent.return_value.select_action.return_value = {'details':{}}
            terms.side_effect = lambda *args: (events.append('terms') or (1.,2.,3.))
            Env.return_value.step.side_effect = lambda *args: events.append('step')
            run_single_seed(20,42,verbose=False)
            self.assertEqual(events,['terms','step']*20)
            Agent.return_value.train.assert_not_called()


if __name__ == '__main__':
    unittest.main()
