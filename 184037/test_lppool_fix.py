"""
Regression test for https://github.com/pytorch/pytorch/issues/184037

LPPool{1d,2d,3d} returns NaN when norm_type is non-integer and the
input contains negative values (x**p is complex for negative x and
non-integer p).

Fix: use |x|**p instead of x**p.
"""
import torch


def test_lppool_nan_non_integer_norm():
    torch.manual_seed(0)
    p = 2.3273891720497133

    # LPPool2d
    x = torch.randn(2, 8, 8, 8)
    out = torch.nn.LPPool2d(p, kernel_size=3, stride=2)(x)
    assert not out.isnan().any(), f"LPPool2d produced NaN with norm_type={p}"

    # LPPool1d
    x = torch.randn(2, 8, 16)
    out = torch.nn.LPPool1d(p, kernel_size=3, stride=2)(x)
    assert not out.isnan().any(), f"LPPool1d produced NaN with norm_type={p}"

    # LPPool3d
    x = torch.randn(2, 4, 4, 4, 4)
    out = torch.nn.LPPool3d(p, kernel_size=3, stride=2)(x)
    assert not out.isnan().any(), f"LPPool3d produced NaN with norm_type={p}"

    print("All tests PASSED")


def test_backward_compat_integer_norm():
    """Integer norm_type (p=2) should produce identical results since |x|^2 = x^2."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, 8, 8)
    out = torch.nn.LPPool2d(2, kernel_size=2, stride=2)(x)
    assert not out.isnan().any(), "LPPool2d with integer norm produced NaN"
    print("Backward compat test PASSED")


if __name__ == "__main__":
    test_lppool_nan_non_integer_norm()
    test_backward_compat_integer_norm()
