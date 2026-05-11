"""
Regression test to add to test/test_unary_ufuncs.py for issue #181806.

Usage:
    cp test_unary_ops.py <pytorch-root>/test/test_unary_ops_signbit.py
    python -m pytest test_unary_ops_signbit.py -v
"""

import numpy as np
import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)


class TestSignbitCUDA(TestCase):

    @onlyCUDA
    def test_signbit_negative_nan_float16(self, device):
        """CUDA signbit must return True for negative float16 NaN (issue #181806)."""
        src = torch.from_numpy(
            np.array([0xFE00, 0xFE00, 0xFE00], dtype=np.uint16).view(np.float16)
        )
        cpu_result = torch.signbit(src)
        cuda_result = torch.signbit(src.to(device)).cpu()
        self.assertEqual(cpu_result, cuda_result)
        self.assertTrue(cuda_result.all())

    @onlyCUDA
    def test_signbit_positive_nan_float16(self, device):
        """CUDA signbit must return False for positive float16 NaN."""
        src = torch.from_numpy(
            np.array([0x7E00, 0x7E00, 0x7E00], dtype=np.uint16).view(np.float16)
        )
        cpu_result = torch.signbit(src)
        cuda_result = torch.signbit(src.to(device)).cpu()
        self.assertEqual(cpu_result, cuda_result)
        self.assertFalse(cuda_result.any())

    @onlyCUDA
    def test_signbit_various_negative_nan_float16(self, device):
        """CUDA signbit must return True for all negative NaN bit patterns."""
        nan_patterns = [0xFE00, 0xFC01, 0xFF00, 0xFFFF]
        src = torch.from_numpy(
            np.array(nan_patterns, dtype=np.uint16).view(np.float16)
        )
        cpu_result = torch.signbit(src)
        cuda_result = torch.signbit(src.to(device)).cpu()
        self.assertEqual(cpu_result, cuda_result)
        self.assertTrue(cuda_result.all())

    @onlyCUDA
    def test_signbit_negative_nan_bfloat16(self, device):
        """CUDA signbit must return True for negative bfloat16 NaN."""
        neg_nan_bf16 = torch.tensor(float("nan"), dtype=torch.bfloat16).neg()
        neg_nan_bf16 = neg_nan_bf16.expand(3)
        cpu_result = torch.signbit(neg_nan_bf16)
        cuda_result = torch.signbit(neg_nan_bf16.to(device)).cpu()
        self.assertEqual(cpu_result, cuda_result)


instantiate_device_type_tests(TestSignbitCUDA, globals())

if __name__ == "__main__":
    run_tests()
