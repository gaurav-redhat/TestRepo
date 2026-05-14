"""
Extracted from: torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py
Lines 1556-1609, 1371-1421

This is the opset 9 avg_pool ONNX export path (used for opset < 10).
The bug is in the interaction between count_include_pad and ceil_mode.
"""


# === _avg_pool (lines 1556-1609) ===

def _avg_pool(name, tuple_fn):
    @symbolic_helper.quantized_args(True)
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
        if not stride:
            stride = kernel_size
        padding = symbolic_helper._avgpool_helper(
            tuple_fn, padding, kernel_size, stride, divisor_override, name
        )
        if not isinstance(padding, tuple):
            raise AssertionError(f"Expected padding to be tuple, got {type(padding)}")
        adjusted_padding = padding
        # Although onnx::AvgPool provides count_include_pad,
        # The corner case of Average Pooling with ceil_mode on
        # PyTorch allows sliding window go off bound, which leads to
        # this accommodation.
        # More detail on https://github.com/pytorch/pytorch/issues/57178
        if count_include_pad:
            input = symbolic_helper._op_with_optional_float_cast(
                g,
                "Pad",
                input,
                pads_i=((0,) * 2 + padding) * 2,
                mode_s="constant",
                value_f=0.0,
                opset_before=11,
            )
            adjusted_padding = (0,) * len(padding)
        if ceil_mode:
            # BUG: get_pool_ceil_padding uses original 'padding' but after the
            # Pad op above, the input is already padded and adjusted_padding is 0.
            # The ceil padding arithmetic is computed against stale values.
            padding_ceil = get_pool_ceil_padding(input, kernel_size, stride, padding)
            adjusted_padding = adjusted_padding + tuple(
                a + b for (a, b) in zip(padding_ceil, adjusted_padding)
            )
        else:
            adjusted_padding = adjusted_padding * 2
        output = g.op(
            "AveragePool",
            input,
            kernel_shape_i=tuple_fn(kernel_size),
            strides_i=tuple_fn(stride),
            pads_i=adjusted_padding,
        )
        return output

    return symbolic_fn


# === get_pool_ceil_padding (lines 1371-1421) ===

@_onnx_symbolic("aten::get_pool_ceil_padding")
def get_pool_ceil_padding(input, kernel_size, stride, padding):
    sizes = symbolic_helper._get_tensor_sizes(input)
    dim = sizes[-len(padding) :] if sizes is not None else None
    if dim is None or any(i is None for i in dim):
        return symbolic_helper._unimplemented(
            "get_pool_ceil_padding", "input size not accessible", input
        )
    ceiled_output_dim = [
        math.ceil((dim[i] + 2 * padding[i] - kernel_size[i]) / float(stride[i])) + 1
        for i in range(len(padding))
    ]
    # ensure last pooling starts inside
    ceiled_output_dim = [
        (
            ceiled_output_dim[i] - 1
            if (((ceiled_output_dim[i] - 1) * stride[i]) >= (dim[i] + padding[i]))
            else ceiled_output_dim[i]
        )
        for i in range(len(ceiled_output_dim))
    ]
    padding_ceil = [
        (
            0
            if (stride[i] == 1)
            else (
                kernel_size[i]
                - (
                    dim[i]
                    + 2 * padding[i]
                    - ((ceiled_output_dim[i] - 1) * stride[i] + 1)
                )
            )
        )
        for i in range(len(padding))
    ]
    # ensure padding is not > kernel_size
    padding_ceil = [
        (
            (
                int(padding_ceil[i])
                if padding_ceil[i] < kernel_size[i] - 1
                else int(kernel_size[i] - 1)
            )
            if ((padding_ceil[i] + 2 * padding[i]) >= (kernel_size[i]))
            else int(padding_ceil[i])
        )
        for i in range(len(padding_ceil))
    ]
    return padding_ceil
