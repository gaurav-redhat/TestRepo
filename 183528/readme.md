# Fix: ONNX Export Mismatch for `avg_pool2d` with `ceil_mode=True` and `count_include_pad=True`

**Issue:** [pytorch/pytorch#183528](https://github.com/pytorch/pytorch/issues/183528)

## Problem

When exporting `torch.nn.functional.avg_pool2d` to ONNX with both `ceil_mode=True` and `count_include_pad=True`, the exported model produces different results from PyTorch eager execution. The mismatch appears at the right/bottom boundary windows introduced by `ceil_mode=True`.

```python
import torch
import torch.nn.functional as F

x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)

out = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1,
                   ceil_mode=True, count_include_pad=True)

# PyTorch eager:    [1.556, 3.333, 2.0,   6.333, 11.0, 6.0,   4.5, 7.5, 4.0]
# ONNX (ORT):      [1.556, 3.333, 1.333, 6.333, 11.0, 4.0,   3.0, 5.0, 1.778]
#                                  ^^^^                  ^^^   ^^^  ^^^  ^^^^^
# max_abs_diff: 2.5
```

The `ceil_mode=False` case matches perfectly between PyTorch and ONNXRuntime.

## Root Cause

There are **two different ONNX export paths** for `avg_pool2d`, and they handle `ceil_mode + count_include_pad` differently:

### Path 1: `symbolic_opset9._avg_pool` (opset < 10)

This path **manually** handles `ceil_mode` by computing extra padding via `get_pool_ceil_padding()`. When `count_include_pad=True`, it first wraps the input with a `Pad` op (zero-padding), then adjusts padding — but the interaction between the explicit `Pad` and the `ceil_mode` padding arithmetic is incorrect.

The issue is at lines 1582-1597 of `symbolic_opset9.py`:

```python
if count_include_pad:
    input = ... "Pad" ...  # zero-pads the input
    adjusted_padding = (0,) * len(padding)  # resets padding to 0
if ceil_mode:
    padding_ceil = get_pool_ceil_padding(input, kernel_size, stride, padding)
    # BUG: 'padding' here is the ORIGINAL padding, but input was already padded
    # and adjusted_padding was set to 0. The ceil padding calc uses stale values.
    adjusted_padding = adjusted_padding + tuple(
        a + b for (a, b) in zip(padding_ceil, adjusted_padding)
    )
```

The `get_pool_ceil_padding` calculation uses the original `padding` parameter, but after the `Pad` op, the effective padding is different. This causes the ONNX `AveragePool` node to get incorrect asymmetric padding, leading to wrong boundary window sizes.

### Path 2: `symbolic_opset10._avg_pool` (opset >= 10)

This path directly passes `ceil_mode_i` and `count_include_pad_i` as ONNX attributes, relying on the ONNX runtime to handle them. However, **PyTorch and ONNX spec disagree** on how `count_include_pad` interacts with `ceil_mode` boundary windows:

- **PyTorch:** When `ceil_mode=True` creates a boundary window that extends beyond the input, and `count_include_pad=True`, PyTorch counts the padding pixels in the denominator but correctly clips the window to the padded input extent.
- **ONNX AveragePool:** The ONNX spec's `count_include_pad` counts all positions in the kernel window (including out-of-bounds positions), producing a different denominator for boundary windows.

The exported ONNX node attributes are:

```json
{
  "auto_pad": "NOTSET",
  "ceil_mode": 1,
  "count_include_pad": 1,
  "kernel_shape": [3, 3],
  "pads": [1, 1, 1, 1],
  "strides": [2, 2]
}
```

This is a faithful representation of the PyTorch parameters, but the ONNX runtime interprets them differently at the boundary.

## Files Involved

```
torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py    # opset 9 avg_pool (manual ceil_mode handling)
torch/onnx/_internal/torchscript_exporter/symbolic_opset10.py   # opset 10+ avg_pool (native ceil_mode/count_include_pad)
torch/onnx/_internal/torchscript_exporter/symbolic_helper.py    # _avgpool_helper
```

## Potential Fix Approaches

### Approach A: Fix the opset 10+ path to match opset 9 strategy

When both `ceil_mode=True` and `count_include_pad=True`, use the same manual padding strategy as opset 9 (explicit `Pad` op + adjusted padding) instead of relying on the ONNX runtime's interpretation:

```python
def _avg_pool(name, expand_size):
    def symbolic_fn(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
        kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
            expand_size, kernel_size, stride, padding
        )

        if ceil_mode and count_include_pad:
            # Handle manually: pad input, disable count_include_pad in ONNX node,
            # compute ceil_mode padding correctly
            ...
        else:
            result = g.op("AveragePool", input,
                          ceil_mode_i=ceil_mode,
                          count_include_pad_i=count_include_pad,
                          kernel_shape_i=kernel_shape,
                          pads_i=pads, strides_i=strides)
        return result
    return symbolic_fn
```

### Approach B: Compute correct explicit padding for the combined case

Compute the exact padding needed so that the ONNX `AveragePool` (without `ceil_mode` and `count_include_pad`) produces the same output as PyTorch. This involves:

1. Computing the output shape PyTorch would produce
2. Adding explicit `Pad` zeros around the input
3. Using `AveragePool` without `ceil_mode` or `count_include_pad`

## Reproducer

See `test_avgpool_onnx_export.py` in this folder.

## Reference

* **PyTorch avg_pool2d:** `aten/src/ATen/native/AveragePool2d.cpp`
* **ONNX opset 9 export:** `torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py` (line 1556–1609)
* **ONNX opset 10 export:** `torch/onnx/_internal/torchscript_exporter/symbolic_opset10.py` (line 401–430)
* **ONNX AveragePool spec:** https://onnx.ai/onnx/operators/onnx__AveragePool.html
* **Related issue:** [pytorch/pytorch#57178](https://github.com/pytorch/pytorch/issues/57178) (original ceil_mode accommodation)
