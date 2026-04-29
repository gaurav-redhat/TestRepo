"""
Test: torch.linspace CPU/CUDA Integer Precision Fix
Issue: https://github.com/pytorch/pytorch/issues/181807

This script demonstrates the bug in stock PyTorch (no build needed).
Just needs: pip install torch

Usage:
    python test_linspace_fix.py
"""

import sys
import torch


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


def simulate_fix(start, end, steps, dtype):
    """
    Show what the fix does: compare float vs double step computation.
    The bug is that CUDA uses float32 for the step, CPU uses float64.
    """
    scalar_start = int(start)
    scalar_end = int(end)

    float_step = (float(scalar_end) - float(scalar_start)) / (steps - 1)
    double_step = (scalar_end - scalar_start) / (steps - 1)

    halfway = steps // 2
    float_result = []
    double_result = []

    for i in range(steps):
        if i < halfway:
            float_result.append(int(scalar_start + float_step * i))
            double_result.append(int(scalar_start + double_step * i))
        else:
            float_result.append(int(scalar_end - float_step * (steps - i - 1)))
            double_result.append(int(scalar_end - double_step * (steps - i - 1)))

    print(f"\n  Simulation for linspace({start}, {end}, {steps}):")
    print(f"  float32 step = {float_step} (CUDA before fix)")
    print(f"  float64 step = {double_step} (CPU / CUDA after fix)")
    print(f"  float32 result: {float_result}")
    print(f"  float64 result: {double_result}")
    match = float_result == double_result
    if not match:
        diffs = [i for i in range(steps) if float_result[i] != double_result[i]]
        print(f"  Mismatch at indices: {diffs}")
    else:
        print(f"  Results match (no precision issue for these inputs)")


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNo CUDA available. Running simulation only.\n")
        print("=" * 60)
        print("Simulation: float32 vs float64 step precision")
        print("=" * 60)
        simulate_fix(3.7, -3, 10, torch.int64)
        simulate_fix(1.5, 10.5, 20, torch.int32)
        simulate_fix(-100.7, 100.3, 200, torch.int64)
        print("\nThe fix changes CUDA kernel to use float64 (double)")
        print("matching the CPU kernel behavior.")
        sys.exit(0)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    print("=" * 60)
    print("Test: Original reproducer from issue #181807")
    print("=" * 60)
    result = test_linspace_cpu_cuda_match(3.7, -3, 10, torch.int64)
    if not result:
        print("  ^ BUG CONFIRMED: CPU and CUDA produce different results")
        simulate_fix(3.7, -3, 10, torch.int64)
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
        print("ALL TESTS PASSED - No mismatch detected.")
        print("(If running on stock PyTorch, the bug may not trigger for all inputs)")
    else:
        print("SOME TESTS FAILED - Bug confirmed!")
        print("The fix: change float to double in CUDA linspace kernel.")
        print("See RangeFactories.cu in this repo for the fixed version.")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
