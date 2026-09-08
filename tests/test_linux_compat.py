import os
import unittest
from unittest.mock import patch

from run_sweeps import (
    DEFAULT_MAX_WORKERS, _assign_worker_dnn_device, _process_pool_context,
)
from tests.helpers import small_config
from utils.matplotlib_backend import is_headless


class LinuxCompatibilityTests(unittest.TestCase):
    def test_headless_environment_override(self):
        with patch.dict(os.environ, {'LDA_HEADLESS':'1'}):
            self.assertTrue(is_headless())
        with patch.dict(os.environ, {'LDA_HEADLESS':'0'}):
            self.assertFalse(is_headless())

    def test_sweep_pool_uses_spawn(self):
        self.assertEqual(_process_pool_context().get_start_method(), 'spawn')

    def test_automatic_worker_limit_is_conservative(self):
        self.assertEqual(DEFAULT_MAX_WORKERS, 8)

    def test_heuristic_worker_does_not_initialize_cuda(self):
        cfg = small_config()
        cfg.dnn_device = 'auto'
        with patch('run_sweeps.torch.cuda.is_available') as available:
            _assign_worker_dnn_device(cfg, 'COB')
        available.assert_not_called()
        self.assertEqual(cfg.dnn_device, 'auto')

    def test_cpu_request_is_not_overridden(self):
        cfg = small_config()
        with patch('run_sweeps.torch.cuda.is_available') as available:
            _assign_worker_dnn_device(cfg, 'LDA')
        available.assert_not_called()
        self.assertEqual(cfg.dnn_device, 'cpu')

    def test_auto_learning_workers_are_distributed_across_gpus(self):
        cfg = small_config()
        cfg.dnn_device = 'auto'
        fake_process = type('Process', (), {'_identity': (4,)})()
        with patch('run_sweeps.torch.cuda.is_available', return_value=True), \
             patch('run_sweeps.torch.cuda.device_count', return_value=2), \
             patch('run_sweeps.multiprocessing.current_process',
                   return_value=fake_process):
            _assign_worker_dnn_device(cfg, 'AC')
        self.assertEqual(cfg.dnn_device, 'cuda:1')

    def test_legacy_running_config_stays_on_cpu(self):
        cfg = small_config()
        del cfg.dnn_device
        with patch('run_sweeps.torch.cuda.is_available') as available:
            _assign_worker_dnn_device(cfg, 'LDA')
        available.assert_not_called()
        self.assertEqual(cfg.dnn_device, 'cpu')


if __name__ == '__main__':
    unittest.main()
