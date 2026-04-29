# Fix: CPU/CUDA Precision Mismatch in `linspace` and `logspace` for Integer Dtypes

**Issue:** [pytorch/pytorch#181807](https://github.com/pytorch/pytorch/issues/181807)

## Problem

`torch.linspace` (and `torch.logspace`) produce different integer results on CPU vs CUDA for the same inputs.

```python
cpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device="cpu")
gpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device="cuda").cpu()
# cpu: [3, 2, 1, 1, 0, 0, -1, -1, -2, -3]
# gpu: [3, 2, 1, 0, 0, 0,  0, -1, -2, -3]   <-- different at indices 3 and 6
```

## Root Cause

When computing the step for integer dtypes:

- **CPU kernel** uses `double` (64-bit, ~15 digits of precision)
- **CUDA kernel** uses `float` (32-bit, ~7 digits of precision)

The lower precision of `float` causes intermediate values to round differently before truncation to integer, producing wrong results.

## Files to Modify

```
aten/src/ATen/native/cuda/RangeFactories.cu   # CUDA kernels
test/test_tensor_creation_ops.py               # regression tests
```

## Fix for `linspace` (Done)

In `linspace_cuda_out`, change the integral-type step from `float` to `double`:

```diff
-      float step = (static_cast<float>(scalar_end) - static_cast<float>(scalar_start)) / (steps - 1);
+      double step = (static_cast<double>(scalar_end) - static_cast<double>(scalar_start)) / (steps - 1);
```

## Fix for `logspace` (Same pattern, still open)

In `logspace_cuda_out`, apply the same change to both `step` and `scalar_base`:

```diff
-      float scalar_base = static_cast<float>(base);
+      double scalar_base = static_cast<double>(base);
       scalar_t scalar_start = start.to<scalar_t>();
       scalar_t scalar_end = end.to<scalar_t>();
-      float step = static_cast<float>(scalar_end - scalar_start) / (steps - 1);
+      double step = static_cast<double>(scalar_end - scalar_start) / (steps - 1);
```

## Why only `linspace` and `logspace`?

| Function   | CPU step type (integers) | CUDA step type (integers) | Needs fix? |
|------------|--------------------------|---------------------------|------------|
| `linspace` | `double`                 | `float`                   | Yes        |
| `logspace` | `double`                 | `float`                   | Yes        |
| `arange`   | `int64_t`                | `int64_t`                 | No         |
| `range`    | `int64_t`                | `int64_t`                 | No         |

`arange` and `range` use `acc_type` which maps all integer types to `int64_t` on both backends, so they are already consistent.

## Regression Test

Add to `test/test_tensor_creation_ops.py` inside `class TestTensorCreation`:

```python
@onlyCUDA
def test_linspace_integral_dtype_matches_cpu(self, device):
    cpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device="cpu")
    gpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device=device)
    self.assertEqual(cpu, gpu.cpu())
```

## Test on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/TestRepo/blob/main/181807/test_linspace_fix.ipynb)

Click the badge above or open the notebook directly: [test_linspace_fix.ipynb](https://colab.research.google.com/github/gaurav-redhat/TestRepo/blob/main/181807/test_linspace_fix.ipynb)

## Test on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/TestRepo/blob/main/181807/test_linspace_fix.ipynb)

## Reference

- CPU linspace kernel: `aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp` (line 48)
- CPU logspace: `aten/src/ATen/native/RangeFactories.cpp` (line 129-133)
- CUDA kernels: `aten/src/ATen/native/cuda/RangeFactories.cu` (lines 86-101, 143-156)
