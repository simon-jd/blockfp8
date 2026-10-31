"""Tensor layouts shared by the quantizer, the reference and the GEMM.

These are the layouts used by DeepSeek-V3 / Qwen3-FP8 style checkpoints and by
vLLM's block-FP8 linear path:

  activations  A_q      [M, K]            float8_e4m3fn, row-major (K contiguous)
               A_scales [M, ceil(K/128)]  float32, one scale per row per 128-wide
                                          K group (any strides are accepted)
  weights      B_q      [N, K]            float8_e4m3fn, row-major, as stored in
                                          the checkpoint ("weight")
               B_scales [ceil(N/128), ceil(K/128)]  float32, one scale per 128x128
                                          weight block ("weight_scale_inv")
  output       C        [M, N]            bfloat16, row-major

  C[m, n] = sum_kb  A_scales[m, kb] * B_scales[n // 128, kb]
                    * sum_{k in block kb} A_q[m, k] * B_q[n, k]

Despite its checkpoint name, weight_scale_inv is the dequantization multiplier:
w = w_q * scale. The same holds for the activation scales.
"""

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0  # largest finite float8_e4m3fn
GROUP_K = 128  # activation group width and weight block width along K
BLOCK_N = 128  # weight block height along N

# The GEMM loads 16-byte chunks along K and stores bf16 pairs along N.
K_ALIGN = 16
N_ALIGN = 8


def cdiv(a: int, b: int) -> int:
    return -(-a // b)
