import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from run_sweeps import (
    DEFAULT_MAX_WORKERS, _assign_worker_dnn_device, _process_pool_context,
)
from tests.helpers import small_config
from utils.matplotlib_backend import is_headless


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_GPU_UUID = 'GPU-9bd37703-4d1f-1565-6e12-6630229223e5'


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

    def test_formal_service_enforces_shared_uuid_gate(self):
        service = (REPO_ROOT / 'systemd/lda-experiments.service').read_text()
        gpu_env = (REPO_ROOT / 'systemd/formal-gpu.env').read_text()
        self.assertIn(
            'EnvironmentFile=/home/hp/projects/LDA/Undergraduate-Thesis-Research-Record/'
            'systemd/formal-gpu.env', service)
        self.assertIn(
            'ExecStartPre=/home/hp/projects/LDA/Undergraduate-Thesis-Research-Record/'
            'scripts/formal_gpu_gate', service)
        self.assertIn(f'CUDA_VISIBLE_DEVICES={FORMAL_GPU_UUID}', gpu_env)
        self.assertIn(f'LDA_EXPECTED_GPU_UUID={FORMAL_GPU_UUID}', gpu_env)

    def test_formal_gpu_gate_rejects_uuid_mismatch_before_gpu_access(self):
        env = os.environ.copy()
        env.update({
            'LDA_EXPECTED_GPU_UUID': FORMAL_GPU_UUID,
            'CUDA_VISIBLE_DEVICES': 'GPU-wrong-device',
            'PYTHON_BIN': sys.executable,
        })
        completed = subprocess.run(
            [str(REPO_ROOT / 'scripts/formal_gpu_gate')],
            cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 1)
        self.assertIn('CUDA_VISIBLE_DEVICES must equal', completed.stderr)


if __name__ == '__main__':
    unittest.main()
