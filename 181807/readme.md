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

## Why this change works

For integral output dtypes, each sample is still computed in **floating point** (using `step` and the start/end scalars), then **converted to the integer dtype**. The kernel uses two numerically equivalent formulas (from the start toward the middle, and from the end toward the middle). With a **`float`** step, rounding error is large enough that those two sides can disagree slightly **before** truncation. Truncation to `int64` then snaps values to different integers at some indices (e.g. `0.999…` vs `1.000…` across a boundary).

A **`double`** step matches the **CPU** path: about **15** decimal digits of precision instead of **~7** for `float`, so the intermediate values stay consistent where it matters, and the truncated integers **match CPU**.

The math of `linspace` is unchanged; only the **precision of the scratch computation** for `step` and the chain `scalar ± step * index` is aligned with `RangeFactoriesKernel.cpp` on CPU.

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

## Build PyTorch from source (commands that worked on this server)

CUDA **12.8** at `/usr/local/cuda-12.8`, **Fedora**, system **GCC 11.5** at `/usr/bin/gcc`, **conda** env `pytorch-build` with **Python 3.12**, GPU **H200**.

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

cp ../TestRepo/181807/RangeFactories.cu aten/src/ATen/native/cuda/RangeFactories.cu
cp ../TestRepo/181807/test_tensor_creation_ops.py test/test_tensor_creation_ops.py
```

### 3. Build env and compile

```bash
conda activate pytorch-build
cd ~/pytorch-fix

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$CONDA_PREFIX/bin:/usr/bin:/bin

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOST_COMPILER=/usr/bin/gcc
export CUDACXX=$CUDA_HOME/bin/nvcc
export CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc

export CMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler -idirafter -Xcompiler /usr/include"

export USE_CUDA=1
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export USE_CUDNN=0
export USE_KINETO=0
export BUILD_TEST=0

unset CPATH CPLUS_INCLUDE_PATH

pip install -r requirements.txt
rm -rf build
$CONDA_PREFIX/bin/python setup.py develop
```

### 4. Run (same shell or new login)

```bash
conda activate pytorch-build
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python ~/TestRepo/181807/test_linspace_fix.py
```

---

## Regression Test

Add to `test/test_tensor_creation_ops.py` inside `class TestTensorCreation`:

```python
@onlyCUDA
def test_linspace_integral_dtype_matches_cpu(self, device):
    cpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device="cpu")
    gpu = torch.linspace(3.7, -3, 10, dtype=torch.int64, device=device)
    self.assertEqual(cpu, gpu.cpu())
```

## Reference

- CPU linspace kernel: `aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp` (line 48)
- CPU logspace: `aten/src/ATen/native/RangeFactories.cpp` (line 129-133)
- CUDA kernels: `aten/src/ATen/native/cuda/RangeFactories.cu` (lines 86-101, 143-156)
