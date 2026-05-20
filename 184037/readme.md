# Fix: `LPPool{1d,2d,3d}` Returns NaN for Non-Integer `norm_type` on Signed Input

**Issue:** [pytorch/pytorch#184037](https://github.com/pytorch/pytorch/issues/184037)

## Problem

`torch.nn.LPPool2d(norm_type=p)` computes `x**p` instead of `|x|**p`. For non-integer `p` with negative input values, `x**p` is mathematically complex and IEEE-evaluates to `NaN`.

```python
import torch

torch.manual_seed(0)
m = torch.nn.LPPool2d(norm_type=2.3273891720497133, kernel_size=(1, 1), stride=(1, 1))
inp = torch.randn(2, 8, 8, 8)   # contains negatives
out = m(inp)
print("any NaN:", torch.isnan(out).any().item())
# any NaN: True
```

The docstring also states `At p = ∞, one gets Max Pooling`, which only holds for the `|x|^p` formulation, not `x^p`. The implementation matched the wrong formula.

## Root Cause

In `torch/nn/functional.py`, the LP pooling functions compute:

```python
out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
```

For negative `input` values and non-integer `norm_type`, `input.pow(norm_type)` produces NaN because raising a negative number to a non-integer power yields a complex result in IEEE 754 arithmetic.

| `input` value | `norm_type` | `input.pow(norm_type)` | Expected (`input.abs().pow(norm_type)`) |
|---|---|---|---|
| -0.5 | 2.33 | NaN | 0.293 |
| -1.2 | 2.33 | NaN | 1.507 |
| 0.8 | 2.33 | 0.617 | 0.617 |

For integer `p` (like `p=2`), this was never a problem because `x^2 >= 0` for all real `x`.

## Files Modified

```
torch/nn/functional.py                                    # Python implementation (the fix)
torch/nn/modules/pooling.py                               # Docstring correction
torch/csrc/api/include/torch/nn/functional/pooling.h      # C++ implementation (the fix)
test/nn/test_pooling.py                                   # Regression test
```

## Fix

Change `input.pow(norm_type)` to `input.abs().pow(norm_type)` in all three LP pooling functions (1d, 2d, 3d), in both Python and C++.

### Python (`torch/nn/functional.py`)

```diff
 def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
     kw, kh = _pair(kernel_size)
     if stride is not None:
-        out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
+        out = avg_pool2d(input.abs().pow(norm_type), kernel_size, stride, 0, ceil_mode)
     else:
-        out = avg_pool2d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)
+        out = avg_pool2d(input.abs().pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)
```

Same change in `lp_pool1d` and `lp_pool3d`.

### C++ (`torch/csrc/api/include/torch/nn/functional/pooling.h`)

```diff
 inline Tensor lp_pool2d(...) {
   Tensor out = detail::avg_pool2d(
-      input.pow(norm_type),
+      input.abs().pow(norm_type),
       ...);
```

Same change in `lp_pool1d` and `lp_pool3d`.

### Docstrings (`torch/nn/modules/pooling.py`)

```diff
     .. math::
-        f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}
+        f(X) = \sqrt[p]{\sum_{x \in X} |x|^{p}}
```

Updated for `LPPool1d`, `LPPool2d`, and `LPPool3d`.

## Why This Fix Is Correct

The Lp-norm is defined as `||x||_p = (Σ|x_i|^p)^(1/p)`. The absolute value is part of the mathematical definition. Without it:

- `p = ∞` does NOT give Max Pooling (the docstring claim is wrong without `|x|`)
- Non-integer `p` produces NaN for negative inputs
- `p = 1` gives signed sum instead of L1 norm

With `|x|^p`:
- `p = ∞` correctly gives `max(|x|)` (Max Pooling)
- `p = 1` correctly gives `sum(|x|)` (L1 norm / Sum Pooling)
- Non-integer `p` is always well-defined for real inputs

## Backward Compatibility

For even integer `p` (the most common case, `p=2`): `|x|^2 = x^2`, so results are identical.

For odd integer `p` (e.g., `p=1`, `p=3`): behavior changes — the new code computes the proper Lp-norm instead of a signed sum. This matches the documented formula and the `p→∞ = MaxPooling` claim.

## Build and Test

```bash
cd ~/pytorch
conda activate torch-dev
python setup.py develop

# Run the regression test
python -m pytest test/nn/test_pooling.py -k "test_LPPool_nan_on_non_integer_norm" -v
```

Or run the standalone test:

```bash
python TestRepo/184037/test_lppool_fix.py
```

## Reference

* **Python LP pooling:** `torch/nn/functional.py` (lines 1087–1203)
* **C++ LP pooling:** `torch/csrc/api/include/torch/nn/functional/pooling.h` (lines 1006–1122)
* **Module classes:** `torch/nn/modules/pooling.py` (lines 1125–1280)
* **Existing tests:** `test/nn/test_pooling.py`, `torch/testing/_internal/common_modules.py`
