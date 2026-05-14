"""
Reproducer for pytorch/pytorch#183528

ONNX export mismatch for avg_pool2d with ceil_mode=True and count_include_pad=True.

Usage:
    pip install onnx onnxruntime
    python test_avgpool_onnx_export.py
"""

import json
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

try:
    import onnx
    import onnxruntime as ort
    from onnx import helper

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class AvgPoolModel(torch.nn.Module):
    def __init__(self, ceil_mode: bool):
        super().__init__()
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return F.avg_pool2d(
            x,
            kernel_size=3,
            stride=2,
            padding=1,
            ceil_mode=self.ceil_mode,
            count_include_pad=True,
        )


def get_averagepool_attrs(model):
    out = []
    for node in model.graph.node:
        if node.op_type == "AveragePool":
            attrs = {}
            for a in node.attribute:
                v = helper.get_attribute_value(a)
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                attrs[a.name] = v
            out.append(attrs)
    return out


def run_ort(model_path, x):
    sess = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    return sess.run(None, {sess.get_inputs()[0].name: x.detach().cpu().numpy()})[0]


def export_and_run(ceil_mode, x, opset=18):
    model = AvgPoolModel(ceil_mode=ceil_mode).eval()

    with torch.no_grad():
        torch_out = model(x).detach().cpu().numpy()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = f"{tmp}/model.onnx"

        torch.onnx.export(
            model,
            (x,),
            onnx_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )

        exported = onnx.load(onnx_path)
        onnx.checker.check_model(exported)
        attrs = get_averagepool_attrs(exported)

        ort_out = run_ort(onnx_path, x)

    return torch_out, ort_out, attrs


def test_pytorch_only():
    """Test PyTorch eager behavior (no ONNX deps needed)."""
    print("=" * 60)
    print("TEST: PyTorch eager avg_pool2d behavior")
    print("=" * 60)

    x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
    print(f"Input shape: {x.shape}")
    print(f"Input:\n{x.squeeze()}\n")

    for ceil_mode in [False, True]:
        out = F.avg_pool2d(
            x, kernel_size=3, stride=2, padding=1,
            ceil_mode=ceil_mode, count_include_pad=True,
        )
        print(f"ceil_mode={ceil_mode}: shape={out.shape}")
        print(f"  values: {out.squeeze().flatten().tolist()}")

    # Additional edge cases
    print("\nEdge case: count_include_pad=False + ceil_mode=True")
    out_no_count = F.avg_pool2d(
        x, kernel_size=3, stride=2, padding=1,
        ceil_mode=True, count_include_pad=False,
    )
    print(f"  values: {out_no_count.squeeze().flatten().tolist()}")

    print("\nEdge case: different kernel/stride combos")
    for ks, st, pad in [(2, 2, 0), (3, 1, 1), (4, 3, 1)]:
        try:
            out = F.avg_pool2d(
                x, kernel_size=ks, stride=st, padding=pad,
                ceil_mode=True, count_include_pad=True,
            )
            print(f"  ks={ks}, stride={st}, pad={pad}: shape={out.shape} values={out.squeeze().flatten().tolist()}")
        except Exception as e:
            print(f"  ks={ks}, stride={st}, pad={pad}: ERROR {e}")

    print()


def test_onnx_export():
    """Test ONNX export mismatch (requires onnx + onnxruntime)."""
    if not HAS_ONNX:
        print("SKIP: onnx/onnxruntime not installed")
        return

    print("=" * 60)
    print("TEST: ONNX export mismatch")
    print("=" * 60)

    x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
    all_pass = True

    for opset in [9, 10, 13, 18]:
        print(f"\n--- opset {opset} ---")
        for ceil_mode in [False, True]:
            try:
                torch_out, ort_out, attrs = export_and_run(ceil_mode, x, opset=opset)
                match = np.allclose(torch_out, ort_out)
                max_diff = float(np.max(np.abs(torch_out - ort_out)))
                status = "PASS" if match else "FAIL"
                if not match:
                    all_pass = False
                print(f"  ceil_mode={ceil_mode}: {status} (max_diff={max_diff:.6f})")
                if not match:
                    print(f"    PyTorch: {torch_out.reshape(-1).tolist()}")
                    print(f"    ORT:     {ort_out.reshape(-1).tolist()}")
                    print(f"    Attrs:   {json.dumps(attrs, sort_keys=True)}")
            except Exception as e:
                print(f"  ceil_mode={ceil_mode}: ERROR {e}")

    print()
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — mismatch between PyTorch and ONNX export")

    return all_pass


def main():
    print(f"PyTorch version: {torch.__version__}")
    if HAS_ONNX:
        print(f"ONNX version: {onnx.__version__}")
        print(f"ONNXRuntime version: {ort.__version__}")
    print()

    test_pytorch_only()
    test_onnx_export()


if __name__ == "__main__":
    main()
