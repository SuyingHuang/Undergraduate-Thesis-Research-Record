import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
from tests.helpers import small_config, bookkeeping_fixture, fixed_action
from run_sweeps import (_find_convergence_frame, extract_metric_bundle, _ci95,
                        _aggregate_metric_rows, _scenario_hash, _paired_comparisons,
                        _apply_formal_candidate_window_policy,
                        run_experiment_sweep, main)
from config import SystemConfig
from collect_calibration import (compute_raw_terms, log_damped_scale,
                                 reference_scale, run_single_seed)
from core.agents.lda_agent import LDAAgent
from core.agents.baselines import ACAgent
from analysis.run_l16_old_bs_ablation import (
    analyze as analyze_l16_old_bs,
    configuration as l16_old_bs_configuration,
    write_report as write_l16_old_bs_report,
)
from analysis.run_old_bs_generalization import (
    analyze as analyze_old_bs_generalization,
    configuration as old_bs_generalization_configuration,
    feedback_metrics,
    treatment_name,
    workload_capacity_envelope,
)


class ExperimentTests(unittest.TestCase):
    def test_generalization_configuration_is_not_tied_to_l16(self):
        cfg = old_bs_generalization_configuration(
            10, 2, 8, 32, 'cpu', 'budgeted', 0.25)
        self.assertEqual((cfg.L_mean, cfg.L_std, cfg.J), (10e6, 2e6, 8))
        self.assertEqual(cfg.old_bs_policy, 'budgeted')
        self.assertEqual(cfg.old_bs_energy_budget_fraction, 0.25)
        self.assertEqual(treatment_name('budgeted', 0.25), 'budgeted_0p25')

    def test_generalization_aggregates_policy_seeds_within_environment(self):
        def row(treatment, environment, policy, paoi):
            return {
                'case': 'L12_std3_J10', 'treatment': treatment,
                'scenario_seed': environment, 'policy_seed': policy,
                'metrics': {
                    'PAoI': paoi, 'Q_Mbit_per_user': paoi,
                    'E_BS_J_per_node': paoi, 'screen_pass': True,
                },
            }
        rows = [
            row('legacy', 1, 10, 4), row('legacy', 1, 20, 6),
            row('budgeted_0p5', 1, 10, 3),
            row('budgeted_0p5', 1, 20, 5),
            row('legacy', 2, 10, 10), row('legacy', 2, 20, 14),
            row('budgeted_0p5', 2, 10, 8),
            row('budgeted_0p5', 2, 20, 10),
        ]
        result = analyze_old_bs_generalization(rows)
        group = result['groups']['L12_std3_J10/budgeted_0p5']
        self.assertEqual(group['n_environment_seeds'], 2)
        self.assertEqual(group['n_runs'], 4)
        self.assertEqual(group['PAoI_environment_mean'], 6.5)
        paired = result['paired_vs_legacy']['L12_std3_J10/budgeted_0p5']
        self.assertEqual(len(paired), 4)

    def test_feedback_metrics_keep_statistical_proxy_separate(self):
        frames, nodes = 8, 2
        ramp = np.arange(frames * nodes, dtype=float).reshape(frames, nodes)
        history = {
            'energy_old_bs_by_bs': ramp + 1,
            'energy_new_bs_by_bs': ramp + 2,
            'service_old_bs_by_bs': ramp + 3,
            'service_new_bs_by_bs': 100 - ramp,
            'old_bs_occupied_by_bs': ramp / 10,
            'energy_queue_by_bs': 2 * ramp,
            'bs_residual_by_bs': 3 * ramp,
        }
        metrics = feedback_metrics(history, frames)
        self.assertIn('feedback_jacobian_ols', metrics)
        self.assertIn('feedback_spectral_radius_ols', metrics)
        self.assertGreater(metrics['E_old_BS_J_per_node'], 0)

    def test_workload_capacity_envelope_matches_completion_energy(self):
        cfg = SystemConfig()
        cfg.I, cfg.J = 1, 2
        workloads = np.array([[[10e6, 20e6]], [[15e6, 5e6]]])
        result = workload_capacity_envelope(cfg, workloads)
        direct = (cfg.kappa1 * cfg.phi ** 3
                  * np.sum(workloads ** 3, axis=-1)
                  / (cfg.tau ** 2 * cfg.E_max_BS))
        self.assertAlmostEqual(
            result['beta_required_all_arrivals_max'], float(np.max(direct)))
        self.assertEqual(result['all_arrivals_frequency_feasible_fraction'], 1)

    def test_l16_old_bs_ablation_changes_only_requested_policy(self):
        legacy = l16_old_bs_configuration('legacy', 32, 'cpu')
        aware = l16_old_bs_configuration('energy_aware', 32, 'cpu')
        self.assertEqual(legacy.L_mean, 16e6)
        self.assertEqual(aware.L_mean, 16e6)
        self.assertEqual(legacy.paoi_ablation, 'none')
        self.assertEqual(aware.paoi_ablation, 'none')
        self.assertEqual(legacy.old_bs_policy, 'legacy')
        self.assertEqual(aware.old_bs_policy, 'energy_aware')
        ignored = {'old_bs_policy'}
        self.assertEqual(
            {k: v for k, v in vars(legacy).items() if k not in ignored},
            {k: v for k, v in vars(aware).items() if k not in ignored},
        )

    def test_l16_old_bs_analysis_pairs_policy_seeds(self):
        def row(variant, seed, value):
            metrics = {
                'PAoI': value,
                'Q_Mbit_per_user': value,
                'E_BS_J_per_node': value,
                'final_max_energy_queue_J': value,
                'max_energy_queue_slope_J_per_frame': value,
                'Q_slope_Mbit_per_user_per_frame': value,
                'all_bs_tail_mean_within_budget': True,
                'energy_queue_tail_nonincreasing': value <= 0,
                'physical_queue_tail_nonincreasing': value <= 0,
            }
            return {'variant': variant, 'policy_seed': seed, 'metrics': metrics}

        rows = [row('legacy', 42, 3.0), row('energy_aware', 42, 1.0),
                row('legacy', 123, 4.0), row('energy_aware', 123, 5.0)]
        result = analyze_l16_old_bs(rows)
        paired = result['paired_vs_legacy']['energy_aware']['PAoI']
        self.assertEqual(paired['n'], 2)
        self.assertEqual(paired['mean'], -0.5)
        self.assertEqual(paired['improved_count'], 1)

    def test_l16_old_bs_partial_report_accepts_no_pairs(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            result = analyze_l16_old_bs([])
            write_l16_old_bs_report(Path(directory), result)
            report = (Path(directory) / 'analysis.md').read_text()
            self.assertIn('finite-horizon', report)
            self.assertNotIn('`energy_aware`:', report)

    def test_constant_learning_window_is_not_heuristic(self):
        self.assertEqual(_find_convergence_frame({'delta_t':[.5]*20}),0)
        self.assertEqual(_find_convergence_frame({}),-1)
        self.assertEqual(_find_convergence_frame({'delta_t':[.5,.2,.1]}),0)
        self.assertEqual(_find_convergence_frame({'delta_t':[.5,.2,.1]},.1),2)
        self.assertEqual(
            _find_convergence_frame({'delta_t':[.5]*20},.5,.5),0)

    def test_formal_j4_learning_runs_use_fixed_wide_window(self):
        for algo in ('LDA', 'AC'):
            cfg = SystemConfig()
            cfg.J = 4
            policy = _apply_formal_candidate_window_policy(cfg, algo)
            self.assertEqual(policy, 'fixed_j4:0.5')
            self.assertEqual(
                (cfg.delta_init, cfg.delta_min, cfg.delta_max),
                (0.5, 0.5, 0.5),
            )

    def test_formal_j4_policy_does_not_change_other_runs(self):
        cfg = SystemConfig()
        before = (cfg.delta_init, cfg.delta_min, cfg.delta_max)
        self.assertEqual(
            _apply_formal_candidate_window_policy(cfg, 'LDA'), 'adaptive')
        self.assertEqual(
            _apply_formal_candidate_window_policy(cfg, 'COB'),
            'not_applicable',
        )
        self.assertEqual(
            (cfg.delta_init, cfg.delta_min, cfg.delta_max), before)

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

    def test_legacy_cleaning_keeps_deliberately_fixed_windows(self):
        rows = [
            dict(
                algo='LDA', param_val=4, failed=False, sim_frames=100,
                conv_frame=0, cleaned_out=False, clean_reason='',
                candidate_window_policy='fixed_j4:0.5',
                **{metric: float(seed) for metric in
                   ('PAoI', 'E_BS', 'E_LEO', 'Q')},
            )
            for seed in (1, 2, 3)
        ]
        result, summary = _aggregate_metric_rows(
            rows, [('LDA', LDAAgent)], [4], cleaned=True)
        self.assertEqual(result['LDA1']['n_valid'], [3])
        self.assertEqual(summary[0]['n_cleaned'], 0)

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

    def test_sparse_calibration_scale_ignores_structural_zeros(self):
        self.assertEqual(reference_scale([0.,0.,2.,4.],nonzero_only=True),3.)
        self.assertEqual(reference_scale([0.,0.,2.,4.]),1.)
        self.assertTrue(np.isnan(reference_scale([0.,0.],nonzero_only=True)))
        self.assertAlmostEqual(log_damped_scale(4.,9.),6.)
        with self.assertRaises(ValueError):
            log_damped_scale(0.,9.)

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
