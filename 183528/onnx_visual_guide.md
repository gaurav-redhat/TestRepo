# ONNX Visual Guide — Understanding Computational Graphs

## 1. A Simple Neural Network as a Graph

Consider this PyTorch model:

```python
class TinyNet(nn.Module):
    def forward(self, x):
        x = torch.relu(x @ self.weight + self.bias)
        return x.mean()
```

### The ONNX graph looks like this:

```
  ┌─────────┐     ┌──────────┐
  │  input   │     │  weight   │
  │ (1, 784) │     │ (784,10)  │
  └────┬─────┘     └─────┬─────┘
       │                  │
       └──────┬───────────┘
              │
         ┌────▼────┐
         │  MatMul  │
         │(1, 784)×(784,10)
         └────┬────┘
              │ (1, 10)
         ┌────▼────┐    ┌──────┐
         │   Add    │◄───│ bias │
         │          │    │(10,) │
         └────┬────┘    └──────┘
              │ (1, 10)
         ┌────▼────┐
         │   Relu   │
         └────┬────┘
              │ (1, 10)
         ┌────▼────────┐
         │ ReduceMean   │
         │ axes=[0,1]   │
         └────┬────────┘
              │ scalar
         ┌────▼────┐
         │  output  │
         └─────────┘
```

**Key idea:** Every box is a **NodeProto**. Every arrow is a **tensor** (named string). Weights are stored as **initializers**.

## 2. Data Flow — How Tensors Move Through the Graph

```
                    ONNX Graph Data Flow
    ┌──────────────────────────────────────────────┐
    │                                              │
    │   inputs[]          initializers[]           │
    │   ┌──────┐          ┌────────┐               │
    │   │"img" │          │"conv.W"│               │
    │   │      │          │"conv.b"│               │
    │   └──┬───┘          │"fc.W"  │               │
    │      │              │"fc.b"  │               │
    │      │              └───┬────┘               │
    │      │                  │                    │
    │      ▼                  ▼                    │
    │   ┌──────────────────────────┐               │
    │   │     node[0]: Conv        │               │
    │   │     inputs:  "img","conv.W","conv.b"     │
    │   │     outputs: "conv_out"  │               │
    │   └──────────┬───────────────┘               │
    │              │"conv_out"                     │
    │              ▼                               │
    │   ┌──────────────────────────┐               │
    │   │     node[1]: Relu        │               │
    │   │     inputs:  "conv_out"  │               │
    │   │     outputs: "relu_out"  │               │
    │   └──────────┬───────────────┘               │
    │              │"relu_out"                     │
    │              ▼                               │
    │   ┌──────────────────────────┐               │
    │   │     node[2]: Flatten     │               │
    │   │     inputs:  "relu_out"  │               │
    │   │     outputs: "flat_out"  │               │
    │   └──────────┬───────────────┘               │
    │              │"flat_out"                     │
    │              ▼                               │
    │   ┌──────────────────────────┐               │
    │   │     node[3]: Gemm        │               │
    │   │     inputs:  "flat_out","fc.W","fc.b"    │
    │   │     outputs: "output"    │               │
    │   └──────────┬───────────────┘               │
    │              │                               │
    │              ▼                               │
    │   outputs[]                                  │
    │   ┌────────┐                                 │
    │   │"output"│                                 │
    │   └────────┘                                 │
    │                                              │
    └──────────────────────────────────────────────┘
```

Tensors are **never stored in nodes**. Nodes only reference tensor names as strings. The actual data for weights lives in `initializer[]`.

## 3. The Protobuf Structure — Visualized

```
ModelProto ─────────────────────────────────────────────────────────
│
├── ir_version: 9
│
├── opset_import ──────────────────────────┐
│   │  domain: ""  (default ONNX)          │
│   │  version: 18                         │
│   └──────────────────────────────────────┘
│
├── graph: GraphProto ─────────────────────────────────────────────
│   │
│   ├── name: "main_graph"
│   │
│   ├── input[] ──────────────────────────────────┐
│   │   │                                         │
│   │   │  ValueInfoProto                         │
│   │   │  ┌─────────────────────────────┐        │
│   │   │  │ name: "input"               │        │
│   │   │  │ type:                        │        │
│   │   │  │   tensor_type:              │        │
│   │   │  │     elem_type: FLOAT (1)    │        │
│   │   │  │     shape:                  │        │
│   │   │  │       dim: {value: 1}       │        │
│   │   │  │       dim: {value: 1}       │        │
│   │   │  │       dim: {value: 4}       │        │
│   │   │  │       dim: {value: 4}       │        │
│   │   │  └─────────────────────────────┘        │
│   │   └─────────────────────────────────────────┘
│   │
│   ├── node[] ───────────────────────────────────┐
│   │   │                                         │
│   │   │  NodeProto [0]                          │
│   │   │  ┌─────────────────────────────┐        │
│   │   │  │ op_type: "AveragePool"      │        │
│   │   │  │ input:  ["input"]           │        │
│   │   │  │ output: ["output"]          │        │
│   │   │  │ attribute:                  │        │
│   │   │  │   kernel_shape: [3, 3]      │        │
│   │   │  │   strides:      [2, 2]      │        │
│   │   │  │   pads:    [1, 1, 1, 1]     │        │
│   │   │  │   ceil_mode: 1              │        │
│   │   │  │   count_include_pad: 1      │        │
│   │   │  └─────────────────────────────┘        │
│   │   └─────────────────────────────────────────┘
│   │
│   ├── output[] ─────────────────────────────────┐
│   │   │  ValueInfoProto                         │
│   │   │  ┌─────────────────────────────┐        │
│   │   │  │ name: "output"              │        │
│   │   │  │ type: tensor(float, [1,1,3,3])      │
│   │   │  └─────────────────────────────┘        │
│   │   └─────────────────────────────────────────┘
│   │
│   └── initializer[] (empty — no weights for pooling)
│
└───────────────────────────────────────────────────────────────────
```

## 4. AveragePool — The Sliding Window Visualized

### Input (4×4) with padding=1

```
Padded input (6×6):

    col: 0    1    2    3    4    5
row 0:  [0]  [0]  [0]  [0]  [0]  [0]    ← pad
row 1:  [0]   1    2    3    4   [0]
row 2:  [0]   5    6    7    8   [0]
row 3:  [0]   9   10   11   12   [0]
row 4:  [0]  13   14   15   16   [0]
row 5:  [0]  [0]  [0]  [0]  [0]  [0]    ← pad

  [0] = padding zero
```

### Without ceil_mode — Output 2×2

```
Window 1 (row 0-2, col 0-2):        Window 2 (row 0-2, col 2-4):
  ┌─────────────┐                      ┌─────────────┐
  │ [0] [0] [0] │                      │ [0] [0] [0] │
  │ [0]  1   2  │                      │  2   3   4  │
  │ [0]  5   6  │                      │  6   7   8  │
  └─────────────┘                      └─────────────┘
  sum=14, count=9                      sum=30, count=9
  avg=1.556                            avg=3.333

Window 3 (row 2-4, col 0-2):        Window 4 (row 2-4, col 2-4):
  ┌─────────────┐                      ┌─────────────┐
  │ [0]  5   6  │                      │  6   7   8  │
  │ [0]  9  10  │                      │ 10  11  12  │
  │ [0] 13  14  │                      │ 14  15  16  │
  └─────────────┘                      └─────────────┘
  sum=57, count=9                      sum=99, count=9
  avg=6.333                            avg=11.0
```

### With ceil_mode=True — Output 3×3 (extra row + column)

The formula `ceil((4 + 2*1 - 3) / 2) + 1 = ceil(1.5) + 1 = 3` gives 3 output positions per dimension.

The **extra boundary windows** (shaded with `░`) are where the bug appears:

```
  Normal windows:                    Boundary windows:

  ┌───┬───┬───┐                      ┌───┬───┬░░░┐
  │1.6│3.3│   │                      │1.6│3.3│2.0│ ← PyTorch
  ├───┼───┤   │                      ├───┼───┤░░░│    vs
  │6.3│11 │   │                      │6.3│11 │6.0│    ORT: 1.3, 4.0
  ├───┼───┤   │                      ├───┼───┤░░░│
  │   │   │   │                      │4.5│7.5│4.0│ ← PyTorch
  └───┴───┴───┘                      └░░░┴░░░┴░░░┘    vs
     ✓ match                           ✗ MISMATCH      ORT: 3.0, 5.0, 1.8
```

### The boundary window at position (0, 2) — where the mismatch happens

```
Kernel starts at row=0, col=4 (in padded coords):

    col: 4    5    6
row 0:  [0]  [0]   ·     ← · = beyond input entirely
row 1:   4   [0]   ·
row 2:   8   [0]   ·

  Actual values in window: [0, 0, 4, 0, 8, 0]
```

**PyTorch (count_include_pad=True):**
- Counts positions that are within the padded input: 6 positions (3×2)
- `sum = 0 + 0 + 4 + 0 + 8 + 0 = 12`
- `avg = 12 / 6 = 2.0` ✓

**ONNX Runtime (count_include_pad=True):**
- Counts ALL kernel positions: 9 positions (3×3)
- `sum = 0 + 0 + 4 + 0 + 8 + 0 = 12`
- `avg = 12 / 9 = 1.333` ✗

```
  PyTorch window (6 cells counted):    ONNX window (9 cells counted):

    ┌────┬────┬ ─ ─ ┐                   ┌────┬────┬────┐
    │ 0  │ 0  │     │                   │ 0  │ 0  │ 0  │
    ├────┼────┤     │                   ├────┼────┼────┤
    │ 4  │ 0  │     │                   │ 4  │ 0  │ 0  │
    ├────┼────┤     │                   ├────┼────┼────┤
    │ 8  │ 0  │     │                   │ 8  │ 0  │ 0  │
    └────┴────┴ ─ ─ ┘                   └────┴────┴────┘
    count = 6                            count = 9
    avg = 12/6 = 2.0                     avg = 12/9 = 1.333
         ▲                                    ▲
      CORRECT                              WRONG
```

## 5. The Export Pipeline — Visual Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                     PyTorch Model (Python)                       │
│                                                                  │
│   class AvgPoolModel(nn.Module):                                │
│       def forward(self, x):                                     │
│           return F.avg_pool2d(x, 3, 2, 1, ceil_mode=True, ...) │
│                                                                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       │  torch.onnx.export()
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   TorchScript Trace (JIT IR)                     │
│                                                                  │
│   %input : Tensor = aten::avg_pool2d(                           │
│       %x,                                                        │
│       kernel_size=[3, 3],                                        │
│       stride=[2, 2],                                             │
│       padding=[1, 1],                                            │
│       ceil_mode=True,                                            │
│       count_include_pad=True                                     │
│   )                                                              │
│                                                                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       │  Symbolic function lookup
                       │  opset=18 → find highest override ≤ 18
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              Symbolic Function Resolution                        │
│                                                                  │
│   aten::avg_pool2d                                               │
│       │                                                          │
│       ├── symbolic_opset18.py?  → NOT FOUND                    │
│       ├── symbolic_opset17.py?  → NOT FOUND                    │
│       ├── ...                                                    │
│       ├── symbolic_opset10.py?  → FOUND: _avg_pool(name, 2)    │
│       │                                                          │
│       └── Uses opset10._avg_pool                                │
│                                                                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       │  Build ONNX graph nodes
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                 ONNX Graph (in memory)                           │
│                                                                  │
│   node {                                                         │
│     op_type: "AveragePool"                                       │
│     input: "input"                                               │
│     output: "output"                                             │
│     attribute: kernel_shape=[3,3]                                │
│     attribute: strides=[2,2]                                     │
│     attribute: pads=[1,1,1,1]                                    │
│     attribute: ceil_mode=1           ◄── passed directly         │
│     attribute: count_include_pad=1   ◄── passed directly         │
│   }                                                              │
│                                                                  │
│   ⚠ PROBLEM: ONNX spec interprets these attributes              │
│     differently from PyTorch at boundary windows                 │
│                                                                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       │  Serialize to protobuf
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    model.onnx (protobuf file)                    │
│                                                                  │
│   Loaded by ONNXRuntime → different boundary behavior → BUG      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 6. Opset Version Resolution — Visual Lookup

```
Target opset: 18

    ┌─────────────────────────────────────────────────────┐
    │              Operator: avg_pool2d                    │
    │                                                     │
    │  opset 18 ─── no override ──────────────┐           │
    │  opset 17 ─── no override               │           │
    │  opset 16 ─── no override               │ search    │
    │  opset 15 ─── no override               │ upward    │
    │  opset 14 ─── no override               │           │
    │  opset 13 ─── no override               │           │
    │  opset 12 ─── no override               │           │
    │  opset 11 ─── no override               │           │
    │  opset 10 ─── ★ FOUND ◄────────────────┘           │
    │                 _avg_pool(name, expand_size=2)       │
    │                 Uses native ceil_mode_i attr         │
    │                                                     │
    │  opset 9  ─── also defined (but overridden by 10)   │
    │               _avg_pool(name, tuple_fn)              │
    │               Manual ceil padding logic              │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    If you export at opset 9 specifically:
    → Uses opset9._avg_pool (manual Pad + padding math)
    → ALSO has the bug (stale padding values)
```

## 7. The Fix Strategy — Visual Comparison

### Current (buggy) opset 10+ path:

```
    input (1,1,4,4)
         │
         ▼
  ┌──────────────────┐
  │   AveragePool     │
  │   ceil_mode=1     │  ← ONNX runtime counts boundary
  │   count_incl_pad=1│     windows differently
  │   pads=[1,1,1,1]  │
  └────────┬─────────┘
           │
           ▼
    output (1,1,3,3)  ← WRONG at boundary positions
```

### Proposed fix — explicit padding approach:

```
    input (1,1,4,4)
         │
         ▼
  ┌──────────────────┐
  │      Pad           │  ← explicitly add zeros
  │   pads=[0,0,1,1,   │     to match PyTorch behavior
  │         0,0,1,1]   │
  │   mode="constant"  │
  │   value=0.0        │
  └────────┬─────────┘
           │
           ▼
    padded (1,1,6,6)
           │
           ▼
  ┌──────────────────┐
  │   AveragePool     │
  │   ceil_mode=1     │  ← now boundary windows see the
  │   count_incl_pad=0│     correct number of elements
  │   pads=[0,0,0,0]  │
  └────────┬─────────┘
           │
           ▼
    output (1,1,3,3)  ← CORRECT — matches PyTorch
```

## 8. Netron — Visual ONNX Model Inspector

[Netron](https://netron.app/) renders ONNX graphs as interactive diagrams.

### How to use:

```bash
pip install netron
python -c "import netron; netron.start('model.onnx')"
```

Or visit https://netron.app/ and drag-drop your `.onnx` file.

### What Netron shows:

```
┌──────────────────────────────────────────┐
│              Netron View                  │
│                                          │
│    ┌────────┐                            │
│    │ input  │  float32[1,1,4,4]          │
│    └───┬────┘                            │
│        │                                 │
│    ┌───▼────────────────┐                │
│    │   AveragePool      │                │
│    │                    │  ◄─ click to   │
│    │ kernel: [3,3]      │     see all    │
│    │ stride: [2,2]      │     attributes │
│    │ pads: [1,1,1,1]    │                │
│    │ ceil_mode: 1       │                │
│    │ count_incl_pad: 1  │                │
│    └───┬────────────────┘                │
│        │                                 │
│    ┌───▼────┐                            │
│    │ output │  float32[1,1,3,3]          │
│    └────────┘                            │
│                                          │
└──────────────────────────────────────────┘
```

### Generating an ONNX file to inspect:

```python
import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 3, 2, 1, ceil_mode=True, count_include_pad=True)

x = torch.randn(1, 1, 4, 4)
torch.onnx.export(Model().eval(), (x,), "avgpool_debug.onnx",
                  opset_version=18, input_names=["input"], output_names=["output"])

# View it
import netron
netron.start("avgpool_debug.onnx")
```

## 9. Full Picture — PyTorch vs ONNX Output Grid

```
Input 4×4 (values 1-16):

    ┌────┬────┬────┬────┐
    │  1 │  2 │  3 │  4 │
    ├────┼────┼────┼────┤
    │  5 │  6 │  7 │  8 │
    ├────┼────┼────┼────┤
    │  9 │ 10 │ 11 │ 12 │
    ├────┼────┼────┼────┤
    │ 13 │ 14 │ 15 │ 16 │
    └────┴────┴────┴────┘

avg_pool2d(kernel=3, stride=2, pad=1, ceil_mode=True, count_include_pad=True)

PyTorch output 3×3:              ONNX Runtime output 3×3:

    ┌───────┬───────┬──────┐       ┌───────┬───────┬──────┐
    │ 1.556 │ 3.333 │ 2.0  │       │ 1.556 │ 3.333 │1.333 │
    ├───────┼───────┼──────┤       ├───────┼───────┼──────┤
    │ 6.333 │ 11.0  │ 6.0  │       │ 6.333 │ 11.0  │ 4.0  │
    ├───────┼───────┼──────┤       ├───────┼───────┼──────┤
    │ 4.5   │ 7.5   │ 4.0  │       │ 3.0   │ 5.0   │1.778 │
    └───────┴───────┴──────┘       └───────┴───────┴──────┘
         ▲                              ▲
      CORRECT                        WRONG (boundary cells)

    Difference grid:

    ┌───────┬───────┬──────┐
    │  0.0  │  0.0  │ 0.67 │
    ├───────┼───────┼──────┤
    │  0.0  │  0.0  │ 2.0  │
    ├───────┼───────┼──────┤
    │  1.5  │  2.5  │ 2.22 │
    └───────┴───────┴──────┘
                        ▲
                   max_diff = 2.5
```

## 10. Summary Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   PyTorch Model                                             │
│       │                                                     │
│       │ torch.onnx.export(opset=18)                        │
│       │                                                     │
│       ▼                                                     │
│   symbolic_opset10._avg_pool                                │
│       │                                                     │
│       │ Passes ceil_mode=1, count_include_pad=1 directly   │
│       │                                                     │
│       ▼                                                     │
│   ONNX AveragePool node                                     │
│       │                                                     │
│       ├──► PyTorch:   boundary count = valid padded cells  │
│       │                (clips to input extent)              │
│       │                                                     │
│       └──► ORT:       boundary count = full kernel size    │
│                        (includes out-of-bounds)             │
│                                                             │
│   FIX: Don't rely on runtime interpretation.                │
│        Use explicit Pad + adjusted AveragePool              │
│        so both runtimes produce the same result.            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
