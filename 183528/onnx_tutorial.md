# ONNX Export in PyTorch — Quick Tutorial

## What is ONNX?

**ONNX** (Open Neural Network Exchange) is an open format for representing machine learning models. It allows you to train a model in PyTorch and run inference in other runtimes like:

- **ONNXRuntime** (Microsoft) — optimized CPU/GPU inference
- **TensorRT** (NVIDIA) — GPU-accelerated inference
- **OpenVINO** (Intel) — CPU-optimized inference
- **CoreML** (Apple) — iOS/macOS inference

## Basic Export

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = MyModel().eval()
dummy_input = torch.randn(1, 3, 32, 32)

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_input,),
    "model.onnx",
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
)
```

## Running with ONNXRuntime

```bash
pip install onnxruntime
```

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

result = sess.run(None, {input_name: np.random.randn(1, 3, 32, 32).astype(np.float32)})
print("Output shape:", result[0].shape)
```

## Key Concepts

### Opset Version

ONNX has **opset versions** (like API versions). Each opset adds or changes operators. Common versions:

| Opset | Notes |
|-------|-------|
| 9 | Baseline for most PyTorch ops |
| 10 | Added `ceil_mode` and `count_include_pad` as native attributes for `AveragePool` |
| 11 | Improved `Pad`, `Resize`, dynamic shapes |
| 13 | Better quantization support |
| 17-18 | Latest stable, recommended for new exports |

Use the latest opset unless your runtime requires an older one:

```python
torch.onnx.export(model, (x,), "model.onnx", opset_version=18)
```

### Symbolic Functions

PyTorch maps each ATen operator to an ONNX graph via **symbolic functions**. These live in:

```
torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py    # base ops
torch/onnx/_internal/torchscript_exporter/symbolic_opset10.py   # opset 10 overrides
torch/onnx/_internal/torchscript_exporter/symbolic_opset11.py   # opset 11 overrides
...
```

When you export at opset 18, PyTorch uses the **highest matching** symbolic function. If opset 10 defines `avg_pool2d`, that overrides the opset 9 version.

### Dynamic Shapes

By default, exported models have **fixed** input shapes. For variable batch sizes:

```python
torch.onnx.export(
    model,
    (dummy_input,),
    "model.onnx",
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
```

## Validating Exports

Always verify your exported model matches PyTorch:

```python
import torch
import numpy as np
import onnxruntime as ort

model.eval()
x = torch.randn(1, 3, 32, 32)

# PyTorch output
with torch.no_grad():
    torch_out = model(x).numpy()

# ONNX output
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]

# Compare
print("Max diff:", np.max(np.abs(torch_out - ort_out)))
assert np.allclose(torch_out, ort_out, atol=1e-5), "Mismatch!"
```

## Common Pitfalls

### 1. Operator Not Supported

```
torch.onnx.errors.UnsupportedOperatorError: ...
```

Fix: Use a different opset, or register a custom symbolic:

```python
from torch.onnx import register_custom_op_symbolic

def my_custom_op(g, input):
    return g.op("CustomDomain::MyOp", input)

register_custom_op_symbolic("mylib::my_op", my_custom_op, opset_version=9)
```

### 2. Dynamic Control Flow

ONNX traces a **single execution path**. Data-dependent `if/else` or loops may not export correctly. Use `torch.jit.script` or restructure the model.

### 3. Numerical Mismatches

Some ops behave slightly differently in ONNX runtimes:
- **Pooling with `ceil_mode`** — boundary window counting can differ (this is issue #183528)
- **Padding modes** — `reflect` and `replicate` edge cases
- **Reduction ops** — floating-point accumulation order

### 4. In-place Operations

In-place ops (e.g., `x.add_(1)`) can cause issues during tracing. Prefer out-of-place versions.

## Useful Tools

### Inspect ONNX Model

```python
import onnx

model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# Print graph
print(onnx.helper.printable_graph(model.graph))
```

### Visualize with Netron

```bash
pip install netron
netron model.onnx
```

Opens a browser with an interactive graph visualization.

### ONNX Shape Inference

```python
import onnx
from onnx import shape_inference

model = onnx.load("model.onnx")
inferred = shape_inference.infer_shapes(model)
onnx.save(inferred, "model_with_shapes.onnx")
```

## Contributing ONNX Fixes to PyTorch

When fixing ONNX export bugs (like #183528):

1. **Identify the opset path** — Check which `symbolic_opsetN.py` file handles the op
2. **Write a reproducer** — Export + compare PyTorch vs ONNXRuntime output
3. **Fix the symbolic function** — Modify the graph construction in the symbolic
4. **Add tests** — Add to `test/onnx/test_pytorch_onnx_onnxruntime.py`
5. **Test across opsets** — Verify the fix works for opset 9, 10, and latest

Key test file:

```
test/onnx/test_pytorch_onnx_onnxruntime.py
```

Run ONNX tests:

```bash
python test/onnx/test_pytorch_onnx_onnxruntime.py -k "avg_pool"
```

## References

- [PyTorch ONNX Export Docs](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Operator Spec](https://onnx.ai/onnx/operators/)
- [ONNX AveragePool Spec](https://onnx.ai/onnx/operators/onnx__AveragePool.html)
- [ONNXRuntime](https://onnxruntime.ai/)
- [Netron Model Viewer](https://netron.app/)
