"""FP64 reference for the block-scaled GEMM.

The reference works block by block along K. Within one 128-wide K block the
products of two e4m3 values are multiples of 2^-18 (the smallest subnormal is
2^-9) and below 2^18 in magnitude, so a 128-term block sum is a multiple of
2^-18 below 2^25: it needs at most 43 significant bits and is exact in FP64
in any summation order. Only the per-block scaling and the sum over blocks
round, each by at most 2^-53 relative, which is negligible next to FP32.
"""

import torch

from .layout import BLOCK_N, GROUP_K, cdiv


def _expand_b_scales(b_scales: torch.Tensor, n: int) -> torch.Tensor:
    return b_scales.repeat_interleave(BLOCK_N, dim=0)[:n]  # [N, KB]


def reference_gemm(
    a_q: torch.Tensor,
    a_scales: torch.Tensor,
    b_q: torch.Tensor,
    b_scales: torch.Tensor,
    absolute: bool = False,
) -> torch.Tensor:
    """C[M, N] in float64; layouts as in blockfp8.layout.

    With absolute=True it returns sum_kb |sa||sb| sum_k |a||b| instead, the
    magnitude that accumulation-error bounds scale with.
    """
    m, k = a_q.shape
    n = b_q.shape[0]
    kb_count = cdiv(k, GROUP_K)
    a = a_q.double()
    b = b_q.double()
    sa = a_scales.double()
    sb = _expand_b_scales(b_scales.double(), n)
    if absolute:
        a, b, sa, sb = a.abs(), b.abs(), sa.abs(), sb.abs()
    c = torch.zeros(m, n, dtype=torch.float64, device=a_q.device)
    for kb in range(kb_count):
        k0, k1 = kb * GROUP_K, min(k, (kb + 1) * GROUP_K)
        block = a[:, k0:k1] @ b[:, k0:k1].t()  # exact, see module docstring
        c += block * (sa[:, kb : kb + 1] * sb[:, kb][None, :])
    return c


def dequantize_activations(x_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """x_q [M, K] e4m3 and scales [M, ceil(K/128)] -> float64 [M, K]."""
    k = x_q.shape[1]
    return x_q.double() * scales.double().repeat_interleave(GROUP_K, dim=1)[:, :k]


def dequantize_weights(w_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """w_q [N, K] e4m3 and scales [ceil(N/128), ceil(K/128)] -> float64 [N, K]."""
    n, k = w_q.shape
    s = _expand_b_scales(scales.double(), n).repeat_interleave(GROUP_K, dim=1)[:, :k]
    return w_q.double() * s


# Error model of the kernel (see docs/design.md, "Precision").
# The FP8 MMA aligns its 32 products and the accumulator input to the largest
# of them and keeps MMA_FRACTION_BITS bits below that leading bit; the rest is
# dropped. One mma therefore errs by at most 33 * 2^-F * max|addend|, and the
# largest addend is bounded by the block's sum of |products| so far. Four mmas
# per 128-wide block, then FP32 scaling and accumulation.
MMA_FRACTION_BITS = 13  # measured: results/micro/mma_precision.json, "summary"
BF16_U = 2.0**-8  # unit roundoff of bfloat16 (8 significant bits)
FP32_U = 2.0**-24


def acc_gamma(k: int) -> float:
    """Worst-case accumulation error relative to sum_kb |sa sb| sum_k |a b|."""
    kb = -(-k // 128)
    return 4 * 33 * 2.0**-MMA_FRACTION_BITS + (4 + 3 * kb) * FP32_U


def error_bound(c_exact: torch.Tensor, c_abs: torch.Tensor, k: int) -> torch.Tensor:
    """Elementwise worst-case bound on |C_kernel - C_exact| for a bf16 output."""
    acc = acc_gamma(k) * c_abs
    return BF16_U * (c_exact.abs() + acc) + acc + torch.finfo(torch.float32).tiny
