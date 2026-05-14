"""
Extracted from: torch/onnx/_internal/torchscript_exporter/symbolic_opset10.py
Lines 354-430

This is the opset 10+ avg_pool ONNX export path.
It passes ceil_mode and count_include_pad directly as ONNX attributes,
but ONNX runtime interprets them differently from PyTorch at boundary windows.
"""


# === _adjust_attributes_of_avg_pool (lines 354-386) ===

# For AvgPool
def _adjust_attributes_of_avg_pool(
    expand_size: int,
    kernel_size: Sequence[int] | int,
    stride: Sequence[int] | int,
    padding: Sequence[int] | int,
) -> tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """Adjust attributes of avg_pool to match ONNX specification."""

    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size

    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2
    elif len(padding) == 2:
        pads = padding * expand_size
    else:
        pads = padding * 2

    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride

    return (kernel_shape, strides, pads)


# === _avg_pool (lines 401-430) ===

def _avg_pool(name, expand_size):
    @symbolic_helper.quantized_args(True, False, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
    def symbolic_fn(
        g,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: int | Sequence[int],
        ceil_mode: int,
        count_include_pad: int,
        divisor_override=None,
    ):
        kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
            expand_size, kernel_size, stride, padding
        )

        # BUG: When ceil_mode=1 and count_include_pad=1, PyTorch and ONNX
        # disagree on how to count elements in boundary windows.
        # PyTorch clips the window to the actual padded input extent,
        # while ONNX counts all kernel positions (including fully out-of-bounds).
        result = g.op(
            "AveragePool",
            input,
            ceil_mode_i=ceil_mode,
            count_include_pad_i=count_include_pad,
            kernel_shape_i=kernel_shape,
            pads_i=pads,
            strides_i=strides,
        )

        return result

    return symbolic_fn
