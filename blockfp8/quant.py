"""Block quantization to float8_e4m3fn (pure PyTorch; runs on CPU and GPU).

Activations: one scale per row per 128-wide K group, computed on the fly
(dynamic quantization), with the same arithmetic as vLLM's
per_token_group_quant_fp8:

    scale = max(amax(|x_group|), eps) / 448
    x_q   = clamp(x / scale, -448, 448) rounded to e4m3 (round to nearest even)

Weights: one scale per 128x128 block, the layout of DeepSeek-V3 / Qwen3-FP8
checkpoints. Checkpoints already ship quantized weights; quantize_weights is
for building synthetic test and benchmark operands.
"""

import torch
import torch.nn.functional as F

from .layout import BLOCK_N, FP8_DTYPE, FP8_MAX, GROUP_K, cdiv


def _group_amax(x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """amax over [rows x cols] blocks of a 2-D tensor; ragged edge blocks allowed."""
    m, k = x.shape
    xp = F.pad(x.abs(), (0, cdiv(k, cols) * cols - k, 0, cdiv(m, rows) * rows - m))
    mb, kb = xp.shape[0] // rows, xp.shape[1] // cols
    return xp.reshape(mb, rows, kb, cols).amax(dim=(1, 3))


def quantize_activations(
    x: torch.Tensor,
    group_size: int = GROUP_K,
    eps: float = 1e-10,
    column_major_scales: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x [M, K] to e4m3 with one float32 scale per row per K group.

    Returns (x_q [M, K] float8_e4m3fn row-major, scales [M, ceil(K/group)]
    float32). With column_major_scales=True (the default) the scales are stored
    M-contiguous (strides (1, M)), the layout DeepGEMM and vLLM's CUTLASS path
    use; the GEMM accepts either layout. K need not be a multiple of the group
    size: the last group is then narrower.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D [M, K], got shape {tuple(x.shape)}")
    if group_size != GROUP_K:
        raise ValueError(f"only group_size={GROUP_K} is supported, got {group_size}")
    m, k = x.shape
    xf = x.float()
    scales = _group_amax(xf, 1, group_size).clamp_min(eps) / FP8_MAX  # [M, KB]
    s_full = scales.repeat_interleave(group_size, dim=1)[:, :k]
    x_q = (xf / s_full).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    if column_major_scales:
        scales = scales.t().contiguous().t()
    return x_q, scales


def quantize_weights(w: torch.Tensor, eps: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize w [N, K] to e4m3 with one float32 scale per 128x128 block.

    Returns (w_q [N, K] float8_e4m3fn, scales [ceil(N/128), ceil(K/128)] float32).
    """
    if w.dim() != 2:
        raise ValueError(f"w must be 2-D [N, K], got shape {tuple(w.shape)}")
    n, k = w.shape
    wf = w.float()
    scales = _group_amax(wf, BLOCK_N, GROUP_K).clamp_min(eps) / FP8_MAX
    s_full = scales.repeat_interleave(BLOCK_N, 0)[:n].repeat_interleave(GROUP_K, 1)[:, :k]
    w_q = (wf / s_full).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return w_q, scales.contiguous()
