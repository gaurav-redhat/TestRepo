# ONNX Quantization — Visual Deep Dive

## 1. What is Quantization?

Quantization converts **float32** weights and activations to **lower-precision** types (int8, uint8, int4) to reduce model size and speed up inference.

```
                     Float32                          INT8
              ┌─────────────────┐            ┌─────────────────┐
              │ 32 bits per value│            │  8 bits per value│
              │                 │            │                 │
              │  3.14159265...  │  ──────►   │      25         │
              │ -0.00723844...  │  quantize  │      -1         │
              │  1.41421356...  │            │      11         │
              │                 │            │                 │
              │ Range: ±3.4e38  │            │ Range: -128~127 │
              │ Size: 400 MB    │            │ Size: 100 MB    │
              │ Speed: 1x       │            │ Speed: 2-4x     │
              └─────────────────┘            └─────────────────┘
```

## 2. The Quantization Math

### Affine (Asymmetric) Quantization

Maps a float range `[min, max]` to integer range `[0, 255]` (uint8):

```
    float value
         │
         ▼
  ┌──────────────────────────────────────┐
  │                                      │
  │   q = round(x / scale) + zero_point  │
  │                                      │
  │   x = (q - zero_point) × scale       │  (dequantize)
  │                                      │
  └──────────────────────────────────────┘

  where:
    scale      = (max - min) / 255
    zero_point = round(-min / scale)
```

**Visual example:**

```
  Float range:  [-1.0 ─────────── 0.0 ─────────── 2.0]
                  │                 │                │
                  ▼                 ▼                ▼
  INT8 range:   [ 0 ─────────── 85 ──────────── 255 ]
                  │                 │                │
                 min            zero_point          max

  scale = (2.0 - (-1.0)) / 255 = 0.01176
  zero_point = round(1.0 / 0.01176) = 85

  Quantize:   1.5  → round(1.5 / 0.01176) + 85 = 213
  Dequantize: 213  → (213 - 85) × 0.01176 = 1.506  (≈ 1.5)
                                                  ▲
                                            small error
```

### Symmetric Quantization

Maps float range `[-abs_max, +abs_max]` to `[-128, 127]` (int8):

```
  Float range:  [-2.0 ─────────── 0.0 ─────────── 2.0]
                  │                 │                │
                  ▼                 ▼                ▼
  INT8 range:  [-128 ──────────── 0 ──────────── 127 ]

  scale = 2.0 / 127 = 0.01575
  zero_point = 0  (always zero for symmetric)

  Quantize:   1.5  → round(1.5 / 0.01575) = 95
  Dequantize: 95   → 95 × 0.01575 = 1.496  (≈ 1.5)
```

## 3. Quantization Granularity

```
Per-Tensor (1 scale for entire tensor):

  ┌──────────────────────────┐
  │  W[0,0]  W[0,1]  W[0,2] │
  │  W[1,0]  W[1,1]  W[1,2] │  ← all use scale=0.015, zp=0
  │  W[2,0]  W[2,1]  W[2,2] │
  └──────────────────────────┘
  1 scale, 1 zero_point


Per-Channel (1 scale per output channel):

  ┌──────────────────────────┐
  │  W[0,0]  W[0,1]  W[0,2] │ ← scale[0]=0.012, zp[0]=0
  │  W[1,0]  W[1,1]  W[1,2] │ ← scale[1]=0.019, zp[1]=0
  │  W[2,0]  W[2,1]  W[2,2] │ ← scale[2]=0.008, zp[2]=0
  └──────────────────────────┘
  3 scales, 3 zero_points


Per-Group (1 scale per group of values):

  ┌──────────────────────────────────────┐
  │  W[0,0]  W[0,1] │ W[0,2]  W[0,3]   │
  │  group 0         │ group 1           │
  │  scale=0.010     │ scale=0.015       │
  └──────────────────────────────────────┘
  Finer granularity → better accuracy
```

## 4. Quantization Approaches

### Post-Training Quantization (PTQ)

Quantize a **pre-trained** float model without retraining:

```
  ┌──────────────┐     calibration      ┌──────────────┐
  │  Float32      │     dataset          │   INT8        │
  │  Model        │ ──────────────────►  │   Model       │
  │  (400 MB)     │  collect min/max     │  (100 MB)     │
  └──────────────┘   per layer           └──────────────┘
        │                                       │
        │ accuracy: 95.2%                       │ accuracy: 94.8%
        │ latency:  50ms                        │ latency:  15ms
        │                                       │
```

### Quantization-Aware Training (QAT)

Insert **fake quantization** nodes during training:

```
  Forward pass with fake quant:

  ┌───────┐    ┌──────────┐    ┌───────┐    ┌──────────┐    ┌───────┐
  │ input │───►│ FakeQuant │───►│ Conv  │───►│ FakeQuant │───►│ ReLU  │
  │ fp32  │    │ (simulate │    │ fp32  │    │ (simulate │    │ fp32  │
  │       │    │  int8     │    │       │    │  int8     │    │       │
  │       │    │  rounding)│    │       │    │  rounding)│    │       │
  └───────┘    └──────────┘    └───────┘    └──────────┘    └───────┘
                    │                            │
                    │   Learns to be robust      │
                    │   to quantization error    │
                    ▼                            ▼
              scale, zp learned            scale, zp learned
              via backprop                 via backprop
```

### Dynamic Quantization

Quantize **weights** statically, **activations** dynamically at runtime:

```
                    Weights: quantized offline
                    ┌────────────────────┐
                    │  INT8 weights      │
                    │  (stored on disk)  │
                    └─────────┬──────────┘
                              │
  ┌───────────┐               │
  │ input     │ ──► compute ──┤
  │ (fp32)    │     min/max   │
  └───────────┘     at        │
       │            runtime   │
       ▼                      ▼
  ┌──────────┐         ┌──────────┐
  │ Quantize │         │ INT8     │
  │ on the   │────────►│ MatMul   │──► Dequantize ──► output (fp32)
  │ fly      │         │          │
  └──────────┘         └──────────┘
```

## 5. ONNX Quantization Operators

### QuantizeLinear

```
  ┌──────────────┐    ┌───────┐    ┌──────────┐
  │ input (fp32) │    │ scale │    │zero_point│
  └──────┬───────┘    └───┬───┘    └────┬─────┘
         │                │              │
         └────────┬───────┴──────────────┘
                  │
             ┌────▼──────────┐
             │QuantizeLinear │
             │               │
             │ q = clamp(    │
             │   round(x/s)  │
             │   + zp,       │
             │   0, 255)     │
             └────┬──────────┘
                  │
             ┌────▼──────────┐
             │ output (uint8)│
             └───────────────┘
```

### DequantizeLinear

```
  ┌──────────────┐    ┌───────┐    ┌──────────┐
  │ input (uint8)│    │ scale │    │zero_point│
  └──────┬───────┘    └───┬───┘    └────┬─────┘
         │                │              │
         └────────┬───────┴──────────────┘
                  │
             ┌────▼────────────┐
             │DequantizeLinear │
             │                 │
             │ x = (q - zp) × s│
             │                 │
             └────┬────────────┘
                  │
             ┌────▼──────────┐
             │ output (fp32) │
             └───────────────┘
```

### QLinearConv (Quantized Convolution)

Performs convolution entirely in INT8:

```
  ┌────────┐ ┌────────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌──────┐ ┌────────┐ ┌──────┐
  │x (uint8)│ │x_scale │ │x_zp  │ │w (int8)│ │w_scale│ │w_zp  │ │y_scale │ │y_zp  │
  └───┬────┘ └───┬────┘ └──┬───┘ └───┬────┘ └──┬───┘ └──┬───┘ └───┬────┘ └──┬───┘
      │          │         │         │         │        │         │         │
      └──────────┴─────────┴─────────┴─────────┴────────┴─────────┴─────────┘
                                      │
                              ┌───────▼────────┐
                              │  QLinearConv    │
                              │                │
                              │  INT8 × INT8   │
                              │  = INT32 accum │
                              │  → rescale     │
                              │  → uint8 out   │
                              └───────┬────────┘
                                      │
                                ┌─────▼──────┐
                                │ y (uint8)  │
                                └────────────┘
```

## 6. Quantized ONNX Graph — Before vs After

### Float32 model graph:

```
  ┌───────┐
  │ input │ fp32
  └───┬───┘
      │
  ┌───▼───┐    ┌────────┐
  │ Conv  │◄───│weight  │ fp32
  │ fp32  │    │(fp32)  │
  └───┬───┘    └────────┘
      │
  ┌───▼───┐
  │ Relu  │
  │ fp32  │
  └───┬───┘
      │
  ┌───▼───┐    ┌────────┐
  │ Conv  │◄───│weight  │ fp32
  │ fp32  │    │(fp32)  │
  └───┬───┘    └────────┘
      │
  ┌───▼───┐
  │output │ fp32
  └───────┘

  Model size: ~400 MB
  All ops in fp32
```

### Quantized model graph (QDQ format):

```
  ┌───────┐
  │ input │ fp32
  └───┬───┘
      │
  ┌───▼──────────────┐
  │ QuantizeLinear   │ fp32 → uint8
  │ scale=0.02, zp=128│
  └───┬──────────────┘
      │ uint8
  ┌───▼──────────────┐
  │DequantizeLinear  │ uint8 → fp32 (approximate)
  └───┬──────────────┘
      │ fp32 (quantized)
  ┌───▼───┐    ┌────────────────────┐
  │ Conv  │◄───│weight              │
  │       │    │ QuantizeLinear     │ fp32 → int8
  │       │    │ DequantizeLinear   │ int8 → fp32
  └───┬───┘    └────────────────────┘
      │
  ┌───▼──────────────┐
  │ QuantizeLinear   │
  └───┬──────────────┘
      │
  ┌───▼──────────────┐
  │DequantizeLinear  │
  └───┬──────────────┘
      │
  ┌───▼───┐
  │ Relu  │
  └───┬───┘
      │
   ... (repeat pattern)
      │
  ┌───▼───┐
  │output │ fp32
  └───────┘

  Model size: ~100 MB
  Runtime fuses Q→DQ→Conv into INT8 Conv
```

### Operator fusion by runtime:

```
  Before fusion:                    After fusion (by ONNXRuntime):

  QuantizeLinear                    ┌──────────────────┐
       │                            │                  │
  DequantizeLinear                  │   QLinearConv    │
       │                 ────►      │   (INT8 × INT8)  │
     Conv                           │                  │
       │                            └────────┬─────────┘
  QuantizeLinear                             │
       │                               (uint8 output)
  DequantizeLinear

  The QDQ nodes are "hints" to the runtime
  about where to do integer math.
```

## 7. PyTorch Quantization → ONNX Export Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                    PyTorch Model (fp32)                        │
└───────────────────────┬────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────────┐              ┌───────────────────┐
│ Post-Training     │              │ Quantization-     │
│ Quantization      │              │ Aware Training    │
│                   │              │                   │
│ torch.ao.         │              │ torch.ao.         │
│  quantization.    │              │  quantization.    │
│  quantize_dynamic │              │  prepare_qat     │
│  quantize_static  │              │  convert         │
└────────┬──────────┘              └────────┬──────────┘
         │                                  │
         └──────────┬───────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────┐
│               Quantized PyTorch Model                          │
│                                                                │
│   QuantizedConv2d, QuantizedLinear, etc.                      │
│   Weights stored as qint8/quint8                               │
│   Scale and zero_point per tensor/channel                     │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        │  torch.onnx.export()
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│                  ONNX Quantized Model                          │
│                                                                │
│   QuantizeLinear / DequantizeLinear nodes                      │
│   OR QLinearConv / QLinearMatMul nodes                         │
│   Weights as int8 TensorProto                                  │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        │  Load in runtime
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│              ONNXRuntime / TensorRT                             │
│                                                                │
│   Fuses QDQ patterns into native INT8 kernels                 │
│   2-4x faster inference                                        │
└────────────────────────────────────────────────────────────────┘
```

## 8. Quantization Formats in ONNX

### Format 1: QDQ (QuantizeLinear/DequantizeLinear)

The **recommended** format. Inserts Q/DQ nodes around float ops:

```
  input(fp32) → QuantizeLinear → DequantizeLinear → Conv(fp32) → QuantizeLinear → ...
```

Runtimes fuse these into integer ops. If a runtime doesn't support fusion, it still produces correct (float) results.

### Format 2: QOperator (QLinearConv, QLinearMatMul)

Directly uses integer operators:

```
  input(uint8) → QLinearConv(int8 weights) → output(uint8)
```

More explicit but less portable — requires runtime to support every Q-operator.

### Comparison:

```
┌────────────────────┬──────────────────┬─────────────────────┐
│                    │      QDQ         │     QOperator       │
├────────────────────┼──────────────────┼─────────────────────┤
│ Graph readability  │ Easy (float ops  │ Harder (new op      │
│                    │  with Q/DQ hints)│  types everywhere)  │
├────────────────────┼──────────────────┼─────────────────────┤
│ Fallback if no     │ Yes (just runs   │ No (fails if        │
│ INT8 support       │  as float)       │  op not supported)  │
├────────────────────┼──────────────────┼─────────────────────┤
│ Runtime fusion     │ Required for     │ Already fused       │
│                    │  speedup         │                     │
├────────────────────┼──────────────────┼─────────────────────┤
│ ONNX recommended   │ ✓ Yes           │ Legacy              │
├────────────────────┼──────────────────┼─────────────────────┤
│ TensorRT support   │ ✓ Excellent     │ Limited             │
├────────────────────┼──────────────────┼─────────────────────┤
│ ORT support        │ ✓ Excellent     │ ✓ Good             │
└────────────────────┴──────────────────┴─────────────────────┘
```

## 9. Calibration — Choosing Scale and Zero Point

```
  Run N batches of real data through the model:

  Batch 1:  layer3_output = [0.12, -0.85, 1.23, 0.44, -0.02, ...]
  Batch 2:  layer3_output = [0.55, -0.31, 0.98, 0.77, -1.10, ...]
  Batch 3:  layer3_output = [0.03, -0.66, 1.45, 0.21, -0.89, ...]
                 ...
  Batch N:  layer3_output = [0.34, -0.72, 1.11, 0.59, -0.55, ...]

  ┌─────────────────────────────────────────────────────────┐
  │              Calibration Methods                         │
  │                                                         │
  │  MinMax:    min = -1.10,  max = 1.45                   │
  │             scale = (1.45 - (-1.10)) / 255 = 0.01      │
  │             Simple but sensitive to outliers             │
  │                                                         │
  │  Percentile: Use 99.99th percentile instead of min/max │
  │              Clips outliers for better overall accuracy  │
  │                                                         │
  │  Entropy (KL divergence):                               │
  │              Find scale that minimizes information loss  │
  │              between float and quantized distributions   │
  │                                                         │
  │  MSE:       Find scale that minimizes mean squared      │
  │              error between float and dequantized values  │
  └─────────────────────────────────────────────────────────┘
```

### Visual: MinMax vs Percentile

```
  Distribution of activation values:

  count
    │
    │    ╭──╮
    │   ╭╯  ╰╮
    │  ╭╯    ╰╮
    │ ╭╯      ╰╮
    │╭╯        ╰──╮
    ├╯            ╰──╮                              ★ outlier
    │                ╰──────╮                   ★
    └──────────────────────────────────────────────── value
   -1.5  -1.0  -0.5   0.0   0.5   1.0   1.5   3.0

  MinMax range:       [-1.5 ──────────────────────── 3.0]
                      scale = 4.5/255 = 0.0176
                      ⚠ Outliers waste most of the int8 range

  Percentile range:   [-1.5 ─────────────── 1.5]
                      scale = 3.0/255 = 0.0118
                      ✓ Better resolution for the common values
                      ✓ Clips rare outliers (small accuracy loss)
```

## 10. Quantization-Aware Training — Fake Quantization

```
  Forward pass:

  float_weight ──► FakeQuantize ──► "quantized" weight (still fp32)
                       │
                       │  round(w / scale) * scale
                       │  (simulates quantization error)
                       ▼
                   ┌──────────────────────────┐
                   │                          │
                   │  w_fq = 0.34, 0.67, ...  │  (fp32 but restricted to
                   │                          │   values representable
                   │                          │   as int8 × scale)
                   └──────────────────────────┘

  Backward pass:

  Straight-Through Estimator (STE):

                ┌─────────────────────────┐
                │    FakeQuantize          │
                │                         │
  Forward:      │  y = round(x / s) * s   │  (non-differentiable)
                │                         │
  Backward:     │  grad_x = grad_y        │  (pretend it's identity)
                │  (if x is within range) │
                │  grad_x = 0             │
                │  (if x is clipped)      │
                │                         │
                └─────────────────────────┘

  Gradient flow:

  loss
    │
    ▼ grad
  ┌──────┐     ┌───────────┐     ┌──────┐     ┌───────────┐     ┌──────┐
  │output│◄────│FakeQuant  │◄────│ Conv │◄────│FakeQuant  │◄────│input │
  │      │ grad│pass thru  │ grad│      │ grad│pass thru  │     │      │
  └──────┘     └───────────┘     └──────┘     └───────────┘     └──────┘
                    │                              │
                    │ grad also flows to           │
                    │ scale/zero_point             │
                    ▼ (learnable!)                 ▼
```

## 11. INT4 and Weight-Only Quantization (LLMs)

For large language models, **weight-only** quantization is popular:

```
  ┌──────────────────────────────────────────────────┐
  │           LLM Weight-Only Quantization            │
  │                                                  │
  │   Weights: INT4 (4 bits per value)               │
  │   Activations: FP16 (kept in float)              │
  │                                                  │
  │   ┌────────────────────────────────────┐         │
  │   │  Original weight: 0.0234375       │         │
  │   │                                    │         │
  │   │  FP32: 00111100 10000000 ... (32b) │         │
  │   │  FP16: 0 01100 0000000000   (16b) │         │
  │   │  INT8: 00000010              (8b)  │         │
  │   │  INT4: 0010                  (4b)  │  ◄──── │
  │   └────────────────────────────────────┘         │
  │                                                  │
  │   Model size reduction:                          │
  │     7B param model:                              │
  │     FP32: 28 GB                                  │
  │     FP16: 14 GB                                  │
  │     INT8:  7 GB                                  │
  │     INT4:  3.5 GB  ◄── fits on consumer GPU     │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### ONNX representation for INT4:

```
  ┌─────────────┐
  │ weight_int4  │  packed: 2 values per byte
  │ (uint8 blob) │
  └──────┬──────┘
         │
  ┌──────▼───────────────┐    ┌───────┐    ┌──────┐
  │  DequantizeLinear    │◄───│ scale │◄───│ zp   │
  │  (int4 → fp16)      │    │(fp16) │    │(int4)│
  └──────┬───────────────┘    └───────┘    └──────┘
         │ fp16
  ┌──────▼───────┐    ┌──────────────┐
  │   MatMul     │◄───│ input (fp16) │
  │  fp16 × fp16 │    └──────────────┘
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ output (fp16)│
  └──────────────┘
```

## 12. Full Quantization Workflow — End to End

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Step 1: Train in FP32                                             │
│  ┌──────────────┐                                                  │
│  │ model.pth    │  accuracy: 95.2%                                 │
│  │ (400 MB)     │                                                  │
│  └──────┬───────┘                                                  │
│         │                                                          │
│  Step 2: Export to ONNX                                            │
│  ┌──────▼───────┐                                                  │
│  │ model.onnx   │  still fp32, same accuracy                      │
│  │ (400 MB)     │                                                  │
│  └──────┬───────┘                                                  │
│         │                                                          │
│  Step 3: Quantize (using ONNX quantization tools)                  │
│         │                                                          │
│         ├─── Option A: onnxruntime.quantization                    │
│         │    from onnxruntime.quantization import quantize_static  │
│         │    quantize_static("model.onnx", "model_q.onnx",        │
│         │                     calibration_data)                    │
│         │                                                          │
│         ├─── Option B: pytorch → quantize → export                 │
│         │    model_q = torch.ao.quantization.quantize_dynamic(     │
│         │        model, {nn.Linear}, dtype=torch.qint8)            │
│         │    torch.onnx.export(model_q, ...)                       │
│         │                                                          │
│  ┌──────▼───────┐                                                  │
│  │model_q.onnx  │  accuracy: 94.9%  (-0.3%)                      │
│  │ (100 MB)     │  latency:  15ms   (3.3x faster)                │
│  └──────┬───────┘                                                  │
│         │                                                          │
│  Step 4: Deploy                                                    │
│  ┌──────▼───────────────────────┐                                  │
│  │ ONNXRuntime / TensorRT       │                                  │
│  │                              │                                  │
│  │  sess = ort.InferenceSession(│                                  │
│  │    "model_q.onnx",          │                                  │
│  │    providers=["CUDA..."])    │                                  │
│  │                              │                                  │
│  │  result = sess.run(None,     │                                  │
│  │    {"input": data})          │                                  │
│  └──────────────────────────────┘                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 13. Accuracy vs Speed Tradeoff

```
  accuracy
    │
 96%├─── ★ FP32 (baseline)
    │
 95%├─── ★ FP16
    │        ★ INT8 (per-channel, QAT)
 94%├───
    │        ★ INT8 (per-channel, PTQ)
 93%├───
    │    ★ INT8 (per-tensor, PTQ)
 92%├───
    │
 91%├───
    │            ★ INT4 (weight-only)
 90%├───
    │
    └──────┬──────┬──────┬──────┬──── speed (relative)
          1x    2x     3x    4x

  Per-channel > Per-tensor (accuracy)
  QAT > PTQ (accuracy, but needs retraining)
  INT4 = most compression, some accuracy loss
```

## 14. References

- [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
- [ONNX DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html)
- [ONNXRuntime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Quantization Spec](https://github.com/onnx/onnx/blob/main/docs/QuantizationSpec.md)
- [TensorRT INT8 Calibration](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#int8-calibration)
- [Netron — ONNX Model Viewer](https://netron.app/)
