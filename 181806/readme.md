# Fix: CUDA `torch.signbit` Returns False for Negative Float16 NaN

**Issue:** [pytorch/pytorch#181806](https://github.com/pytorch/pytorch/issues/181806)

## Problem

`torch.signbit` produces inconsistent results between CPU and CUDA for negative `float16` NaN values.

```python
import numpy as np
import torch

src = torch.from_numpy(
    np.array([0xfe00, 0xfe00, 0xfe00], dtype=np.uint16).view(np.float16)
)

cpu  = torch.signbit(src)
cuda = torch.signbit(src.cuda()).cpu()

# cpu:  tensor([True,  True,  True])   ← correct
# cuda: tensor([False, False, False])  ← wrong
```

The raw half-precision bit pattern `0xFE00` is a NaN with the sign bit set (`sign=1, exponent=11111, mantissa≠0`).
Per IEEE 754, `signbit` must return `True` for any value whose sign bit is 1, including NaN.

## Root Cause

When computing `signbit` for `Half` / `BFloat16` on CUDA, the kernel promotes the 16-bit value to `float` (32-bit) **before** checking the sign bit:

```cpp
// aten/src/ATen/native/cuda/UnarySignKernels.cu  (lines 83-85, original)
AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, ScalarType::Half, iter.input_dtype(), "signbit_cuda", [&]() {
  using opmath_t = at::opmath_type<scalar_t>;   // Half → float
  gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> bool { return signbit(opmath_t{a}); });
});
```

The promotion `opmath_t{a}` (i.e. `float{half_value}`) passes through CUDA's `__half2float` intrinsic.
**NVIDIA GPUs canonicalize NaN during half→float conversion**, replacing any NaN with the canonical positive quiet NaN `0x7FC00000` — whose sign bit is **0**.

| Stage | Value (hex) | Sign bit |
|---|---|---|
| Input `Half` | `0xFE00` | 1 (negative NaN) |
| After `__half2float` on GPU | `0x7FC00000` | **0** (canonical +qNaN) |
| `signbit(float)` result | | `false` ← **wrong** |

On CPU, the same promotion (`std::half → float`) preserves the sign bit (`0xFE00` → `0xFFC00000`), so `std::signbit` correctly returns `true`.

## Files to Modify

```
aten/src/ATen/native/cuda/UnarySignKernels.cu   # CUDA kernel (the fix)
test/test_unary_ufuncs.py                        # regression tests
```

## Fix

In `signbit_kernel_cuda`, bypass the `float` promotion for `Half` and `BFloat16`.
Instead, extract the sign bit directly from the raw 16-bit representation:

```diff
     AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, ScalarType::Half, iter.input_dtype(), "signbit_cuda", [&]() {
-      using opmath_t = at::opmath_type<scalar_t>;
-      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> bool { return signbit(opmath_t{a}); });
+      if constexpr (std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>) {
+        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> bool {
+          uint16_t bits;
+          memcpy(&bits, &a, sizeof(uint16_t));
+          return (bits >> 15) & 1;
+        });
+      } else {
+        using opmath_t = at::opmath_type<scalar_t>;
+        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> bool { return signbit(opmath_t{a}); });
+      }
     });
```

## Why This Fix Works

For `float` and `double`, the existing `signbit(opmath_t{a})` path is correct — no lossy promotion occurs (`opmath_type<float>` is `float`, `opmath_type<double>` is `double`).

The problem is specific to **16-bit** types where promotion to `float32` triggers CUDA's NaN canonicalization. By reading the raw `uint16_t` bits and shifting right by 15, we directly inspect the sign bit **without any float conversion**, matching the IEEE 754 definition of `signbit`.

The `memcpy` idiom is the standard type-punning approach in C++ (no strict-aliasing violation) and compiles to a single register operation on GPU.

## Why Only `Half` and `BFloat16`?

| Dtype | `opmath_type` | Promotion lossy for NaN sign? | Needs fix? |
|---|---|---|---|
| `float16` (Half) | `float` | **Yes** — CUDA canonicalizes NaN | **Yes** |
| `bfloat16` | `float` | **Yes** — same canonicalization | **Yes** |
| `float32` | `float` | No promotion (identity) | No |
| `float64` | `double` | No promotion (identity) | No |
| Integral types | N/A | Uses `is_negative()`, no float | No |

## Build PyTorch from Source (commands that worked on this server)

CUDA **12.1** or later, **Fedora/RHEL**, system **GCC**, **conda** env with **Python 3.12**.

### 1. Environment

```bash
conda create -n pytorch-build python=3.12 -y
conda activate pytorch-build
pip install cmake ninja pyyaml setuptools
```

### 2. Sources and fix files

```bash
cd ~
git clone https://github.com/gaurav-redhat/TestRepo.git
git clone --depth 1 https://github.com/pytorch/pytorch.git pytorch-fix
cd pytorch-fix
git submodule update --init --recursive --depth 1

# Apply the fix
cp ../TestRepo/181806/UnarySignKernels.cu aten/src/ATen/native/cuda/UnarySignKernels.cu
```

### 3. Build env and compile

```bash
conda activate pytorch-build
cd ~/pytorch-fix

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$CONDA_PREFIX/bin:/usr/bin:/bin

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOST_COMPILER=/usr/bin/gcc
export CUDACXX=$CUDA_HOME/bin/nvcc
export CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc

export CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"

export USE_CUDA=1
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export USE_CUDNN=0
export USE_KINETO=0
export BUILD_TEST=0

pip install -r requirements.txt
rm -rf build
python setup.py develop
```

### 4. Run tests

```bash
conda activate pytorch-build
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python ~/TestRepo/181806/test_signbit_fix.py
```

## Regression Test

Add to `test/test_unary_ufuncs.py` (or a separate test file):

```python
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
```

## Reference

* **CUDA signbit kernel:** `aten/src/ATen/native/cuda/UnarySignKernels.cu` (line 76–88)
* **CPU signbit kernel:** `aten/src/ATen/native/cpu/UnaryOpsKernel.cpp` (line 317–328)
* **Structured dispatch:** `aten/src/ATen/native/UnaryOps.cpp` (line 389–395)
* **OpMathType (Half→float):** `aten/src/ATen/OpMathType.h` (line 14–26)
