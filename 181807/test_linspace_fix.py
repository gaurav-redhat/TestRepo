"""
Test: torch.linspace CPU/CUDA Integer Precision Fix
Issue: https://github.com/pytorch/pytorch/issues/181807

Usage (on Google Colab or any machine with CUDA):
    python test_linspace_fix.py
"""

import os
import subprocess
import sys


def run(cmd):
    print(f">>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def setup():
    """Clone repos, apply fix, and build PyTorch from source."""
    if os.path.exists("pytorch-fix"):
        print("pytorch-fix already exists, skipping setup.")
        return

    run("git clone https://github.com/gaurav-redhat/TestRepo.git")
    run("git clone --depth 1 https://github.com/pytorch/pytorch.git pytorch-fix")
    os.chdir("pytorch-fix")
    run("git submodule update --init --recursive --depth 1")

    run("cp ../TestRepo/181807/RangeFactories.cu aten/src/ATen/native/cuda/RangeFactories.cu")
    run("cp ../TestRepo/181807/test_tensor_creation_ops.py test/test_tensor_creation_ops.py")
    print("Fixed files applied successfully!")

    run("pip install -r requirements.txt")
    run("python setup.py develop")


def test_linspace_cpu_cuda_match(start, end, steps, dtype):
    import torch

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
    setup()

    import torch

    print(f"\nPyTorch version: {torch.__version__}")
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
