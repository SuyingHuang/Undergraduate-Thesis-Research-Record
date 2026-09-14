"""Regression tests for the CUDA-containment and retry behaviour of sweeps.

Background: ``torch.manual_seed`` forwards to ``torch.cuda.manual_seed_all``
for every caller, so seeding a CPU-only heuristic worker still created a CUDA
context.  A single unhealthy GPU then failed 94/128 tasks of one sweep,
including COB/MTD, and stalled the remainder for hours.  These tests pin the
containment contract so the blast radius stays limited to learning runs.
"""
import os
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.agents.baselines import ACAgent, COBAgent, MTDAgent
from core.agents.lda_agent import LDAAgent
from run_sweeps import (_cuda_health_check, _failed_result, _host_uptime_seconds,
                        _run_serial_tasks, run_experiment_sweep)
from tests.helpers import small_config
from utils.reproducibility import set_seed


def _cpu_draws():
    """Draw one value from each seeded CPU stream."""
    return (random.random(), float(np.random.rand()), float(torch.rand(1).item()))


def _result(failed, log_path='unused.log'):
    """Build a per-task result tuple with the exact shape sweeps parse."""
    marker = np.nan
    return (10, 'LDA', 42, marker, marker, marker, marker, failed,
            marker, marker, -1, log_path, -1,
            'boom' if failed else '', 'hash', 'adaptive',
            marker, marker, marker, marker,
            marker, marker, marker, marker)


def _task(log_path='unused.log'):
    """Build a task tuple with the exact shape ``_worker_sweep`` unpacks."""
    return (small_config(), 'K_p', 10, 'LDA', LDAAgent, 8, 42, log_path, None, 'hash')


class CudaContainmentTests(unittest.TestCase):
    def test_cpu_only_seed_never_touches_cuda_seeding_paths(self):
        def explode(*args, **kwargs):
            raise AssertionError('CPU-only seeding must not reach CUDA')

        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.manual_seed', side_effect=explode), \
                patch('torch.cuda.manual_seed_all', side_effect=explode):
            set_seed(7, use_cuda=False)

    def test_cpu_only_seed_leaves_cpu_streams_identical(self):
        set_seed(11, use_cuda=False)
        cpu_only = _cpu_draws()
        set_seed(11)
        default = _cpu_draws()
        set_seed(11, use_cuda=False)
        repeated = _cpu_draws()
        self.assertEqual(cpu_only, default)
        self.assertEqual(cpu_only, repeated)

    def test_heuristic_agents_declare_no_dnn_and_build_no_network(self):
        cfg = small_config()
        self.assertTrue(LDAAgent.uses_dnn)
        self.assertTrue(ACAgent.uses_dnn)
        self.assertFalse(COBAgent.uses_dnn)
        self.assertFalse(MTDAgent.uses_dnn)
        self.assertTrue(hasattr(LDAAgent(cfg), 'actors'))
        self.assertFalse(hasattr(COBAgent(cfg), 'actors'))
        self.assertFalse(hasattr(MTDAgent(cfg), 'actors'))


class CudaHealthCheckTests(unittest.TestCase):
    def test_host_uptime_is_a_finite_non_negative_reading(self):
        uptime = _host_uptime_seconds()
        self.assertIsInstance(uptime, float)
        self.assertGreaterEqual(uptime, 0.0)

    def test_host_uptime_is_none_when_unreadable(self):
        with patch('builtins.open', side_effect=OSError('no /proc/uptime')):
            self.assertIsNone(_host_uptime_seconds())

    def test_unavailable_cuda_is_reported_not_raised(self):
        with patch('torch.cuda.is_available', return_value=False):
            health = _cuda_health_check()
        self.assertEqual(health, {'available': False, 'ok': None,
                                  'device_count': 0, 'error': None})

    def test_unhealthy_device_is_reported_not_raised(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=2), \
                patch('torch.zeros', side_effect=RuntimeError('CUDA error: unknown error')):
            health = _cuda_health_check()
        self.assertTrue(health['available'])
        self.assertFalse(health['ok'])
        self.assertIn('CUDA error: unknown error', health['error'])


class TaskRetryTests(unittest.TestCase):
    def test_failed_result_keeps_the_result_tuple_shape(self):
        task = _task()
        result = _failed_result(task, 'boom')
        self.assertEqual(len(result), 24)
        self.assertEqual(result[:3], (10, 'LDA', 42))
        self.assertTrue(result[7])
        self.assertEqual(result[13], 'boom')
        self.assertEqual(result[11], 'unused.log')
        # The sweep parser unpacks exactly this many fields.
        (param_val, algo_name, seed, paoi, e_bs, e_sat, q, failed,
         max_e, final_e, first_frame, log_path, conv_frame, reason,
         scenario_hash, window, *rest) = result
        self.assertEqual(len(rest), 8)
        self.assertEqual((param_val, algo_name, seed, failed), (10, 'LDA', 42, True))

    def test_retry_reruns_a_failed_task_and_archives_the_failed_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, 'LDA_K_p10_s42.log')
            with open(log_path, 'w', encoding='utf-8') as handle:
                handle.write('[FAILED] boom\n')
            task = _task(log_path)
            with patch('run_sweeps._worker_sweep',
                       side_effect=[_result(True, log_path), _result(False, log_path)]) as worker:
                results = _run_serial_tasks([task], 1, 'K_p', printer=lambda *_: None)
            self.assertEqual(worker.call_count, 2)
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0][7])
            self.assertTrue(os.path.exists(f'{log_path}.attempt1'))

    def test_retry_attempts_are_bounded(self):
        task = _task()
        with patch('run_sweeps._worker_sweep',
                   side_effect=[_result(True)] * 5) as worker:
            results = _run_serial_tasks([task], 2, 'K_p', printer=lambda *_: None)
        self.assertEqual(worker.call_count, 3)
        self.assertTrue(results[0][7])
        self.assertEqual(results[0][13], 'boom')

    def test_zero_retries_preserves_single_attempt_behaviour(self):
        task = _task()
        with patch('run_sweeps._worker_sweep',
                   side_effect=[_result(True)]) as worker:
            results = _run_serial_tasks([task], 0, 'K_p', printer=lambda *_: None)
        self.assertEqual(worker.call_count, 1)
        self.assertTrue(results[0][7])

    def test_negative_retries_are_rejected_before_writing(self):
        cfg = small_config()
        with self.assertRaises(ValueError):
            run_experiment_sweep('Exp1_J', 'J', [4], [('LDA', LDAAgent)], cfg,
                                 n_workers=1, seeds=[42], sim_frames=8,
                                 task_retries=-1)


if __name__ == '__main__':
    unittest.main()
