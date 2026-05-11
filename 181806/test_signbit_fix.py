"""
Reproduction and validation script for PyTorch issue #181806:
  CUDA torch.signbit returns false for negative float16 NaN

https://github.com/pytorch/pytorch/issues/181806

Run:
    python test_signbit_fix.py
"""

import sys
import numpy as np
import torch

def make_negative_nan_f16():
    """Create a float16 tensor whose raw bits are 0xFE00 (negative NaN)."""
    return torch.from_numpy(
        np.array([0xFE00, 0xFE00, 0xFE00], dtype=np.uint16).view(np.float16)
    )

def make_positive_nan_f16():
    """Create a float16 tensor whose raw bits are 0x7E00 (positive NaN)."""
    return torch.from_numpy(
        np.array([0x7E00, 0x7E00, 0x7E00], dtype=np.uint16).view(np.float16)
    )

def make_negative_inf_f16():
    """float16 -Inf: 0xFC00."""
    return torch.from_numpy(
        np.array([0xFC00], dtype=np.uint16).view(np.float16)
    )

def make_positive_inf_f16():
    """float16 +Inf: 0x7C00."""
    return torch.from_numpy(
        np.array([0x7C00], dtype=np.uint16).view(np.float16)
    )

def make_neg_zero_f16():
    """float16 -0.0: 0x8000."""
    return torch.from_numpy(
        np.array([0x8000], dtype=np.uint16).view(np.float16)
    )


def run_tests():
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device     : {torch.cuda.get_device_name(0)}")
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Cannot reproduce the bug.")
        sys.exit(1)

    all_pass = True

    # ── Test 1: Negative NaN (the primary bug) ──────────────────────────
    src = make_negative_nan_f16()
    cpu_result = torch.signbit(src)
    cuda_result = torch.signbit(src.cuda()).cpu()
    match = torch.equal(cpu_result, cuda_result)
    expected = torch.tensor([True, True, True])

    print("Test 1: signbit on negative float16 NaN (0xFE00)")
    print(f"  CPU  result: {cpu_result.tolist()}  (expected [True, True, True])")
    print(f"  CUDA result: {cuda_result.tolist()}  (expected [True, True, True])")
    print(f"  CPU == CUDA: {match}")
    print(f"  PASS: {match and torch.equal(cpu_result, expected)}")
    if not match:
        all_pass = False
    print()

    # ── Test 2: Positive NaN (should be False) ──────────────────────────
    src = make_positive_nan_f16()
    cpu_result = torch.signbit(src)
    cuda_result = torch.signbit(src.cuda()).cpu()
    match = torch.equal(cpu_result, cuda_result)
    expected = torch.tensor([False, False, False])

    print("Test 2: signbit on positive float16 NaN (0x7E00)")
    print(f"  CPU  result: {cpu_result.tolist()}  (expected [False, False, False])")
    print(f"  CUDA result: {cuda_result.tolist()}  (expected [False, False, False])")
    print(f"  CPU == CUDA: {match}")
    print(f"  PASS: {match and torch.equal(cpu_result, expected)}")
    if not match:
        all_pass = False
    print()

    # ── Test 3: Negative infinity ────────────────────────────────────────
    src = make_negative_inf_f16()
    cpu_result = torch.signbit(src)
    cuda_result = torch.signbit(src.cuda()).cpu()
    match = torch.equal(cpu_result, cuda_result)

    print("Test 3: signbit on -Inf float16 (0xFC00)")
    print(f"  CPU  result: {cpu_result.tolist()}")
    print(f"  CUDA result: {cuda_result.tolist()}")
    print(f"  CPU == CUDA: {match}")
    print(f"  PASS: {match}")
    if not match:
        all_pass = False
    print()

    # ── Test 4: Positive infinity ────────────────────────────────────────
    src = make_positive_inf_f16()
    cpu_result = torch.signbit(src)
    cuda_result = torch.signbit(src.cuda()).cpu()
    match = torch.equal(cpu_result, cuda_result)

    print("Test 4: signbit on +Inf float16 (0x7C00)")
    print(f"  CPU  result: {cpu_result.tolist()}")
    print(f"  CUDA result: {cuda_result.tolist()}")
    print(f"  CPU == CUDA: {match}")
    print(f"  PASS: {match}")
    if not match:
        all_pass = False
    print()

    # ── Test 5: Negative zero ────────────────────────────────────────────
    src = make_neg_zero_f16()
    cpu_result = torch.signbit(src)
    cuda_result = torch.signbit(src.cuda()).cpu()
    match = torch.equal(cpu_result, cuda_result)

    print("Test 5: signbit on -0.0 float16 (0x8000)")
    print(f"  CPU  result: {cpu_result.tolist()}")
    print(f"  CUDA result: {cuda_result.tolist()}")
    print(f"  CPU == CUDA: {match}")
    print(f"  PASS: {match}")
    if not match:
        all_pass = False
    print()

    # ── Test 6: Normal negative values ───────────────────────────────────
    src = torch.tensor([-1.0, -0.5, -100.0], dtype=torch.float16)
    cpu_result = torch.signbit(src)
    cuda_result = torch.signbit(src.cuda()).cpu()
    match = torch.equal(cpu_result, cuda_result)

    print("Test 6: signbit on normal negative float16 values")
    print(f"  CPU  result: {cpu_result.tolist()}")
    print(f"  CUDA result: {cuda_result.tolist()}")
    print(f"  CPU == CUDA: {match}")
    print(f"  PASS: {match}")
    if not match:
        all_pass = False
    print()

    # ── Test 7: Multiple NaN variants ────────────────────────────────────
    nan_patterns = [0xFE00, 0xFC01, 0xFF00, 0xFFFF]
    src = torch.from_numpy(
        np.array(nan_patterns, dtype=np.uint16).view(np.float16)
    )
    cpu_result = torch.signbit(src)
    cuda_result = torch.signbit(src.cuda()).cpu()
    match = torch.equal(cpu_result, cuda_result)

    print("Test 7: signbit on various negative float16 NaN bit patterns")
    print(f"  Patterns : {[hex(p) for p in nan_patterns]}")
    print(f"  CPU  result: {cpu_result.tolist()}")
    print(f"  CUDA result: {cuda_result.tolist()}")
    print(f"  CPU == CUDA: {match}")
    print(f"  PASS: {match}")
    if not match:
        all_pass = False
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 50)
    if all_pass:
        print("ALL TESTS PASSED — signbit is consistent on CPU and CUDA.")
    else:
        print("SOME TESTS FAILED — CPU/CUDA mismatch detected (bug present).")
    print("=" * 50)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    run_tests()
