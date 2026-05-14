# ONNX Deep Dive — Architecture, Internals & PyTorch Integration

## 1. What is ONNX?

ONNX (Open Neural Network Exchange) is a **serialization format** for computational graphs. It defines:

- A **protobuf schema** (`onnx.proto3`) describing models as directed acyclic graphs (DAGs)
- A **standard operator set** (opset) — ~180 operators with strict mathematical semantics
- A **type system** for tensors (dtype, shape, symbolic dimensions)
- A **versioning scheme** (IR version + opset version) for backward compatibility

ONNX is **not** a runtime. It is a specification. Runtimes (ONNXRuntime, TensorRT, OpenVINO) implement the spec independently — which is why semantic mismatches like issue #183528 happen.

## 2. The ONNX Model Structure

An ONNX model is a protobuf message with this hierarchy:

```
ModelProto
├── ir_version: int           # IR format version (e.g., 9)
├── opset_import[]            # which opset versions are used
│   └── OpsetImportProto
│       ├── domain: string    # "" = default ONNX domain
│       └── version: int      # e.g., 18
├── graph: GraphProto
│   ├── name: string
│   ├── input[]: ValueInfoProto       # model inputs (name + type)
│   ├── output[]: ValueInfoProto      # model outputs
│   ├── node[]: NodeProto             # the computation nodes
│   │   ├── op_type: string           # e.g., "Conv", "AveragePool"
│   │   ├── input[]: string           # input tensor names
│   │   ├── output[]: string          # output tensor names
│   │   └── attribute[]: AttributeProto  # op-specific params
│   ├── initializer[]: TensorProto    # constant tensors (weights)
│   └── value_info[]: ValueInfoProto  # intermediate tensor shapes
└── metadata_props[]
```

### Example: What a single `AveragePool` node looks like

```protobuf
node {
  op_type: "AveragePool"
  input: "conv1_output"
  output: "pool1_output"
  attribute { name: "kernel_shape"     ints: [3, 3]       }
  attribute { name: "strides"          ints: [2, 2]       }
  attribute { name: "pads"             ints: [1, 1, 1, 1] }
  attribute { name: "ceil_mode"        i: 1               }
  attribute { name: "count_include_pad" i: 1              }
}
```

There are no "layers" — ONNX is a **flat list of nodes** with data-flow edges defined by matching tensor names.

## 3. The Opset System

### What is an opset?

An opset is a **versioned collection of operator definitions**. Each operator has:

- A **name** (e.g., `AveragePool`)
- A **since_version** (the opset that introduced/last changed it)
- A **spec** defining inputs, outputs, attributes, and mathematical behavior

### Opset versioning rules

| Concept | Meaning |
|---------|---------|
| `opset_import.version = 18` | Use operator definitions from opset 18 |
| Operator `AveragePool` since_version = 10 | This op's spec was last changed in opset 10 |
| Backward compatibility | opset 18 uses the opset-10 definition of `AveragePool` (unchanged since) |

When a runtime sees `opset_import = 18` and encounters `AveragePool`, it looks up the **latest version of AveragePool that is ≤ 18**, which is version 10.

### Key opset milestones for pooling

| Opset | Change |
|-------|--------|
| 1 | `AveragePool` introduced (no `ceil_mode`, no `count_include_pad`) |
| 7 | Added `count_include_pad` attribute |
| 10 | Added `ceil_mode` attribute |
| 19 | Added `dilations` attribute |

Before opset 10, PyTorch had to **emulate** `ceil_mode` by manually computing extra padding — this is why `symbolic_opset9.py` has the complex `get_pool_ceil_padding` logic.

## 4. ONNX Type System

### Tensor types

```
TensorProto.DataType:
    FLOAT    = 1     # float32
    UINT8    = 2
    INT8     = 3
    UINT16   = 4
    INT16    = 5
    INT32    = 6
    INT64    = 7
    STRING   = 8
    BOOL     = 9
    FLOAT16  = 10
    DOUBLE   = 11
    UINT32   = 12
    UINT64   = 13
    BFLOAT16 = 16
    FLOAT8E4M3FN  = 17
    FLOAT8E5M2    = 19
```

### Shape representation

Shapes can mix concrete and symbolic dimensions:

```protobuf
# Fixed shape: [1, 3, 224, 224]
type {
  tensor_type {
    elem_type: 1  # FLOAT
    shape {
      dim { dim_value: 1 }
      dim { dim_value: 3 }
      dim { dim_value: 224 }
      dim { dim_value: 224 }
    }
  }
}

# Dynamic batch: [batch, 3, 224, 224]
type {
  tensor_type {
    elem_type: 1
    shape {
      dim { dim_param: "batch" }    # symbolic
      dim { dim_value: 3 }
      dim { dim_value: 224 }
      dim { dim_value: 224 }
    }
  }
}
```

## 5. How PyTorch Exports to ONNX

### The export pipeline

```
Python model
    │
    ▼
torch.jit.trace() / torch.jit.script()
    │  Produces a TorchScript IR (JIT graph)
    ▼
JIT Graph (aten ops)
    │  e.g., aten::avg_pool2d(input, kernel_size, stride, padding, ceil_mode, ...)
    │
    ▼
Symbolic function dispatch
    │  Looks up symbolic_opsetN.py for the highest N ≤ target opset
    │  e.g., target=18 → uses symbolic_opset10._avg_pool (since opset10 defines it)
    │
    ▼
ONNX Graph construction
    │  Builds NodeProto objects via g.op("AveragePool", ...)
    │
    ▼
Optimization passes
    │  Constant folding, dead code elimination, shape inference
    │
    ▼
Serialization to protobuf (.onnx file)
```

### Symbolic function resolution

When exporting at opset 18, PyTorch resolves each ATen op like this:

```
aten::avg_pool2d
  → Check symbolic_opset18.py — not defined
  → Check symbolic_opset17.py — not defined
  → ...
  → Check symbolic_opset10.py — FOUND: _avg_pool(name, expand_size=2)
  → Use this function to build the ONNX subgraph
```

The symbolic function receives the JIT graph builder `g` and the op's arguments, and returns ONNX `Value` objects:

```python
def symbolic_fn(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, ...):
    # g.op() creates an ONNX NodeProto and returns its output Value
    result = g.op(
        "AveragePool",          # ONNX op type
        input,                  # input tensor (Value)
        kernel_shape_i=[3, 3],  # attribute: _i suffix = int list
        strides_i=[2, 2],
        pads_i=[1, 1, 1, 1],
        ceil_mode_i=1,          # attribute: _i suffix = int
        count_include_pad_i=1,
    )
    return result
```

The `g.op()` naming convention:

| Suffix | Type | Example |
|--------|------|---------|
| `_i` | int or int list | `ceil_mode_i=1`, `pads_i=[1,1,1,1]` |
| `_f` | float | `value_f=0.0` |
| `_s` | string | `mode_s="constant"` |
| `_t` | tensor | `value_t=torch.tensor(...)` |

### The `g` object (Graph builder)

`g` is a `torch._C.Graph` object that accumulates ONNX nodes:

```python
# Create an ONNX node
output_value = g.op("OpType", input1, input2, attr_i=42)

# Chain operations
padded = g.op("Pad", input, pads_i=[0,0,1,1,0,0,1,1], mode_s="constant")
pooled = g.op("AveragePool", padded, kernel_shape_i=[3,3], strides_i=[2,2])

# Constants
const = g.op("Constant", value_t=torch.tensor([1.0, 2.0]))
```

## 6. AveragePool — ONNX Spec vs PyTorch Semantics

### The ONNX AveragePool spec (opset 10+)

```
Inputs:
    X: input tensor (N, C, H, W)

Attributes:
    kernel_shape: list[int]       # e.g., [3, 3]
    strides: list[int]            # e.g., [2, 2]  (default: [1, 1])
    pads: list[int]               # [top, left, bottom, right]  (default: all 0)
    ceil_mode: int                # 0 or 1 (default: 0)
    count_include_pad: int        # 0 or 1 (default: 0)
    auto_pad: string              # "NOTSET" (default)

Output shape (ceil_mode=1):
    out_h = ceil((H + pad_top + pad_bottom - kernel_h) / stride_h) + 1
    out_w = ceil((W + pad_left + pad_right - kernel_w) / stride_w) + 1
```

### Where PyTorch and ONNX disagree

For a **boundary window** created by `ceil_mode=True` that extends beyond the input:

```
Input (4x4) with padding=1, kernel=3, stride=2:

  0  0  0  0  0  0       ← padding row
  0  1  2  3  4  0
  0  5  6  7  8  0
  0  9 10 11 12  0
  0 13 14 15 16  0
  0  0  0  0  0  0       ← padding row

Window at position (4,4) with ceil_mode — extends to column 6:

  col:  3   4   5   (6)
       12   0   ·    ·      ← · = beyond padded input
       16   0   ·    ·
        0   0   ·    ·
```

**PyTorch behavior:** The window is `[12, 0, 16, 0, 0, 0]` — 6 real elements (including pad zeros), average = `(12+0+16+0+0+0)/6 ≈ 4.67`. But with `count_include_pad=True`, PyTorch includes padding positions in the count but still clips to the valid padded region.

**ONNX runtime behavior:** Counts all `3×3 = 9` kernel positions including those fully outside, giving a different denominator.

This difference is the core of issue #183528.

## 7. ONNX Operator Semantics — AveragePool Pseudocode

From the ONNX spec, the `AveragePool` computation is:

```python
for each output position (oh, ow):
    # Input window coordinates
    h_start = oh * stride_h - pad_top
    w_start = ow * stride_w - pad_left
    h_end = h_start + kernel_h
    w_end = w_start + kernel_w

    # Clip to input bounds
    h_start_clipped = max(h_start, 0)
    w_start_clipped = max(w_start, 0)
    h_end_clipped = min(h_end, H)
    w_end_clipped = min(w_end, W)

    # Sum elements in the clipped window
    total = sum(X[n, c, h_start_clipped:h_end_clipped, w_start_clipped:w_end_clipped])

    if count_include_pad:
        # Divide by full kernel size (includes all padding positions)
        count = kernel_h * kernel_w
    else:
        # Divide by number of valid (non-padding) elements
        count = (h_end_clipped - h_start_clipped) * (w_end_clipped - w_start_clipped)

    output[n, c, oh, ow] = total / count
```

PyTorch's `count_include_pad` instead uses the number of positions that fall within the **padded** input (not the full kernel size), which gives a different count at `ceil_mode` boundary windows.

## 8. ONNX Graph Optimization Passes

After symbolic functions build the raw ONNX graph, PyTorch runs optimization passes:

### Constant folding

```
Before:  Shape(X) → Gather(0) → Unsqueeze → Concat
After:   Constant([1])    (if X has known batch=1)
```

### Dead code elimination

Removes nodes whose outputs are not consumed by any other node.

### Shape inference

Propagates known shapes through the graph so downstream ops and runtimes can optimize memory allocation.

### Enabling/disabling

```python
torch.onnx.export(
    model, (x,), "model.onnx",
    opset_version=18,
    do_constant_folding=True,   # default: True
)
```

## 9. ONNX Runtime Internals

When ONNXRuntime loads a model:

```
Load .onnx protobuf
    │
    ▼
Graph partitioning
    │  Split graph into subgraphs assigned to execution providers
    │  (CPU, CUDA, TensorRT, etc.)
    │
    ▼
Kernel selection
    │  For each node, select the best kernel implementation
    │  e.g., AveragePool → oneDNN pool kernel (CPU) or cuDNN pool kernel (CUDA)
    │
    ▼
Memory planning
    │  Pre-allocate buffers, reuse memory where possible
    │
    ▼
Execution
    │  Run kernels in topological order
    │
    ▼
Output
```

Each runtime implements operators independently from the spec. This means:

- **ONNXRuntime** may interpret `count_include_pad + ceil_mode` one way
- **TensorRT** may interpret it another way
- **PyTorch** has its own interpretation

The ONNX spec tries to be unambiguous, but edge cases (like ceil_mode boundary windows) are sometimes underspecified.

## 10. Debugging ONNX Export Issues

### Step 1: Check the exported graph

```python
import onnx

model = onnx.load("model.onnx")
for node in model.graph.node:
    print(f"{node.op_type}: {list(node.input)} → {list(node.output)}")
    for attr in node.attribute:
        print(f"  {attr.name} = {onnx.helper.get_attribute_value(attr)}")
```

### Step 2: Compare outputs element-wise

```python
import numpy as np

diff = np.abs(torch_out - ort_out)
print("Max diff:", diff.max())
print("Diff locations:", np.argwhere(diff > 1e-5))
```

### Step 3: Isolate the problematic op

Export a minimal model containing only the suspicious op:

```python
class MinimalModel(torch.nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=3, stride=2, padding=1,
                            ceil_mode=True, count_include_pad=True)
```

### Step 4: Check which symbolic function is used

```python
# Print the ONNX graph before serialization
torch.onnx.export(model, (x,), "model.onnx",
                  opset_version=18,
                  verbose=True)   # prints the graph
```

### Step 5: Test across opsets

```python
for opset in [9, 10, 11, 13, 18]:
    try:
        torch.onnx.export(model, (x,), f"model_opset{opset}.onnx",
                          opset_version=opset)
        # compare outputs ...
    except Exception as e:
        print(f"opset {opset}: {e}")
```

## 11. File Map — Where Things Live in PyTorch

```
torch/onnx/
├── __init__.py                          # torch.onnx.export entry point
├── _internal/
│   ├── torchscript_exporter/
│   │   ├── symbolic_opset9.py           # base symbolic functions
│   │   ├── symbolic_opset10.py          # opset 10 overrides
│   │   ├── symbolic_opset11.py          # opset 11 overrides
│   │   ├── ...
│   │   ├── symbolic_helper.py           # shared utilities
│   │   └── _exporter_legacy.py          # legacy export logic
│   └── exporter/                        # new TorchDynamo-based exporter
│       ├── _onnx_program.py
│       └── ...
├── utils.py                             # helper utilities
└── errors.py                            # ONNX-specific exceptions

test/onnx/
├── test_pytorch_onnx_onnxruntime.py     # main ONNX export test suite
├── test_pytorch_onnx_onnxruntime_cuda.py
├── test_op_consistency.py
└── ...
```

## 12. References

- [ONNX Specification](https://onnx.ai/onnx/)
- [ONNX Operator Schemas](https://onnx.ai/onnx/operators/)
- [ONNX AveragePool](https://onnx.ai/onnx/operators/onnx__AveragePool.html)
- [ONNX IR Spec (protobuf)](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3)
- [PyTorch ONNX Export Docs](https://pytorch.org/docs/stable/onnx.html)
- [ONNXRuntime Architecture](https://onnxruntime.ai/docs/reference/high-level-design.html)
- [ONNX Opset Changelog](https://github.com/onnx/onnx/blob/main/docs/Changelog.md)
- [Issue #57178 — original ceil_mode accommodation](https://github.com/pytorch/pytorch/issues/57178)
- [Issue #183528 — avg_pool2d ceil_mode+count_include_pad mismatch](https://github.com/pytorch/pytorch/issues/183528)
