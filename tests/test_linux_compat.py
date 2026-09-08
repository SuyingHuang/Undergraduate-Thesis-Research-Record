import os
import unittest
from unittest.mock import patch

from run_sweeps import DEFAULT_MAX_WORKERS, _process_pool_context
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


if __name__ == '__main__':
    unittest.main()
