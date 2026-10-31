import torch

from blockfp8 import FP8_DTYPE, dequantize_activations, dequantize_weights, reference_gemm
from blockfp8 import quantize_activations, quantize_weights


def _operands(m, n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    a_q, a_s = quantize_activations(torch.randn(m, k, generator=g))
    b_q, b_s = quantize_weights(torch.randn(n, k, generator=g) * 0.02)
    return a_q, a_s, b_q, b_s


def test_matches_dequantized_matmul():
    a_q, a_s, b_q, b_s = _operands(5, 264, 400)
    c = reference_gemm(a_q, a_s, b_q, b_s)
    expect = dequantize_activations(a_q, a_s) @ dequantize_weights(b_q, b_s).t()
    torch.testing.assert_close(c, expect, rtol=1e-12, atol=1e-12)


def test_exact_on_integers_with_power_of_two_scales():
    g = torch.Generator().manual_seed(1)
    m, n, k = 4, 136, 272  # N and K tails: 2 weight row blocks, 3 K blocks
    ai = torch.randint(-16, 17, (m, k), generator=g)
    bi = torch.randint(-16, 17, (n, k), generator=g)
    a_q, b_q = ai.float().to(FP8_DTYPE), bi.float().to(FP8_DTYPE)
    assert torch.equal(a_q.float(), ai.float()) and torch.equal(b_q.float(), bi.float())
    ea = torch.randint(-6, 7, (m, 3), generator=g)
    eb = torch.randint(-6, 7, (2, 3), generator=g)
    a_s, b_s = torch.exp2(ea.float()), torch.exp2(eb.float())
    c = reference_gemm(a_q, a_s, b_q, b_s)
    # Exact answer in rationals: integer block dots times 2^(ea+eb).
    exact = torch.zeros(m, n, dtype=torch.float64)
    for kb in range(3):
        sl = slice(kb * 128, min(k, kb * 128 + 128))
        dot = ai[:, sl] @ bi[:, sl].t()  # int64, exact
        e = ea[:, kb][:, None] + eb[torch.arange(n) // 128, kb][None, :]
        exact += dot.double() * torch.exp2(e.double())
    assert torch.equal(c, exact)


def test_extreme_fp8_values_are_exact_per_block():
    # Largest and smallest-subnormal products in one block: the block sum needs
    # 2^17.6 down to 2^-18, which FP64 still holds exactly.
    a_q = torch.zeros(1, 128).to(FP8_DTYPE)
    b_q = torch.zeros(8, 128).to(FP8_DTYPE)
    a_q[0, 0], b_q[0, 0] = 448.0, 448.0
    a_q[0, 1], b_q[0, 1] = 2.0**-9, 2.0**-9
    one = torch.ones(1, 1)
    c = reference_gemm(a_q, one, b_q, one)
    assert c[0, 0].item() == 448.0 * 448.0 + 2.0**-18


def test_absolute_bounds_the_result():
    a_q, a_s, b_q, b_s = _operands(3, 128, 256, seed=2)
    c = reference_gemm(a_q, a_s, b_q, b_s)
    c_abs = reference_gemm(a_q, a_s, b_q, b_s, absolute=True)
    assert torch.all(c.abs() <= c_abs)
