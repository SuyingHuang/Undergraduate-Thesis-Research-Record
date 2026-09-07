import os
import unittest
from unittest.mock import patch

from utils.matplotlib_backend import is_headless


class LinuxCompatibilityTests(unittest.TestCase):
    def test_headless_environment_override(self):
        with patch.dict(os.environ, {'LDA_HEADLESS':'1'}):
            self.assertTrue(is_headless())
        with patch.dict(os.environ, {'LDA_HEADLESS':'0'}):
            self.assertFalse(is_headless())


if __name__ == '__main__':
    unittest.main()
