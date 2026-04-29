"""
Test: torch.linspace CPU/CUDA Integer Precision Fix
Issue: https://github.com/pytorch/pytorch/issues/181807

Usage:
    python test_linspace_fix.py
"""

import torch
import sys


def test_linspace_cpu_cuda_match(start, end, steps, dtype):
    cpu = torch.linspace(start, end, steps, dtype=dtype, device="cpu")
    gpu = torch.linspace(start, end, steps, dtype=dtype, device="cuda").cpu()

    match = torch.equal(cpu, gpu)
    status = "PASS" if match else "FAIL"

    print(f"[{status}] linspace({start}, {end}, {steps}, dtype={dtype})")
    if not match:
        diff_idx = (cpu != gpu).nonzero(as_tuple=True)[0].tolist()
        print(f"  CPU: {cpu.tolist()}")
        print(f"  GPU: {gpu.tolist()}")
        print(f"  Differ at indices: {diff_idx}")
    return match


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This test requires a GPU.")
        sys.exit(1)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    print("=" * 60)
    print("Test: Original reproducer from issue #181807")
    print("=" * 60)
    test_linspace_cpu_cuda_match(3.7, -3, 10, torch.int64)
    print()

    print("=" * 60)
    print("Test: Additional edge cases")
    print("=" * 60)

    all_passed = True
    test_cases = [
        (3.7, -3, 10, torch.int64),
        (0, 100, 51, torch.int64),
        (-50, 50, 101, torch.int64),
        (0, 1000000, 999, torch.int64),
        (1.5, 10.5, 20, torch.int32),
        (-100.7, 100.3, 200, torch.int64),
        (0, 255, 256, torch.int16),
        (10, -10, 5, torch.int64),
        (0.1, 0.9, 5, torch.int64),
        (1e6, -1e6, 100, torch.int64),
    ]

    for start, end, steps, dtype in test_cases:
        if not test_linspace_cpu_cuda_match(start, end, steps, dtype):
            all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED - Fix is working!")
    else:
        print("SOME TESTS FAILED - Fix needs more work.")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
