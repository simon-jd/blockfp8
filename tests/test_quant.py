import pytest
import torch

from blockfp8 import FP8_MAX, dequantize_activations, dequantize_weights
from blockfp8 import quantize_activations, quantize_weights


def _rand(m, k, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(m, k, generator=g) * 3).to(dtype)


def _over_fp8_max(t):
    return t / torch.full_like(t, FP8_MAX)  # a true division, not t * (1 / 448)


def test_scales_are_group_amax_over_fp8_max():
    x = _rand(5, 384)
    _, s = quantize_activations(x)
    expect = _over_fp8_max(torch.stack(
        [x[:, i * 128 : (i + 1) * 128].abs().amax(dim=1) for i in range(3)], dim=1
    ))
    assert s.shape == (5, 3)
    torch.testing.assert_close(s, expect, rtol=0, atol=0)


def test_group_max_maps_to_fp8_max():
    x = _rand(7, 256, seed=1)
    q, _ = quantize_activations(x)
    qa = q.float().abs().reshape(7, 2, 128).amax(dim=2)
    assert torch.all(qa == FP8_MAX)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_round_trip_error_is_within_half_ulp(dtype):
    x = _rand(9, 512, dtype=dtype, seed=2)
    q, s = quantize_activations(x)
    err = (dequantize_activations(q, s) - x.double()).abs()
    # e4m3 keeps 3 fraction bits: relative error <= 2^-4 for normals, and an
    # absolute error <= 2^-10 * scale in the subnormal range.
    s_full = s.double().repeat_interleave(128, dim=1)
    bound = torch.maximum(x.double().abs() * 2.0**-4, s_full * 2.0**-10)
    assert torch.all(err <= bound * (1 + 1e-6))


def test_zero_group_gets_eps_scale_and_zero_values():
    x = _rand(3, 256, seed=3)
    x[1, 128:] = 0
    q, s = quantize_activations(x, eps=1e-10)
    assert s[1, 1].item() == pytest.approx(1e-10 / FP8_MAX)
    assert torch.all(q[1, 128:].float() == 0)


def test_k_tail_group_is_narrower():
    x = _rand(4, 200, seed=4)
    q, s = quantize_activations(x)
    assert q.shape == (4, 200) and s.shape == (4, 2)
    torch.testing.assert_close(s[:, 1], _over_fp8_max(x[:, 128:].abs().amax(dim=1)), rtol=0, atol=0)


def test_scale_is_a_true_division():
    # amax = 5.03125: amax * (1 / 448) is one ulp away from amax / 448.
    x = torch.zeros(1, 128)
    x[0, 0] = 5.03125
    _, s = quantize_activations(x)
    amax = torch.tensor([5.03125])
    assert torch.equal(s.flatten(), _over_fp8_max(amax))
    assert not torch.equal(_over_fp8_max(amax), amax * (1.0 / FP8_MAX))


def test_scale_layouts():
    x = _rand(6, 256, seed=5)
    _, s_col = quantize_activations(x)
    _, s_row = quantize_activations(x, column_major_scales=False)
    assert s_col.stride() == (1, 6)
    assert s_row.stride() == (2, 1)
    assert torch.equal(s_col, s_row)


def test_weight_blocks():
    w = _rand(200, 272, seed=6)
    q, s = quantize_weights(w)
    assert s.shape == (2, 3)
    assert s[1, 2].item() == pytest.approx(w[128:, 256:].abs().max().item() / FP8_MAX)
    err = (dequantize_weights(q, s) - w.double()).abs()
    s_full = s.double().repeat_interleave(128, 0)[:200].repeat_interleave(128, 1)[:, :272]
    assert torch.all(err <= torch.maximum(w.double().abs() * 2.0**-4, s_full * 2.0**-10) * (1 + 1e-6))


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        quantize_activations(torch.zeros(128))
    with pytest.raises(ValueError):
        quantize_activations(torch.zeros(2, 128), group_size=64)
