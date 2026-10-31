"""Python entry point for the block-scaled FP8 GEMM (sm_89 CUDA kernel)."""

import os

import torch

from .layout import BLOCK_N, FP8_DTYPE, GROUP_K, K_ALIGN, N_ALIGN, cdiv

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_C.so")
_loaded = False


def native_available() -> bool:
    """True once the CUDA extension (blockfp8/_C.so) is built and loaded."""
    global _loaded
    if not _loaded and os.path.exists(_LIB):
        torch.ops.load_library(_LIB)
        _loaded = True
    return _loaded


def check_gemm_args(
    a_q: torch.Tensor,
    a_scales: torch.Tensor,
    b_q: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[int, int, int]:
    """Validate shapes, dtypes and alignment; returns (M, N, K).

    Raises ValueError with the reason for anything the kernel cannot handle.
    """
    for name, t in (("a_q", a_q), ("b_q", b_q)):
        if t.dtype != FP8_DTYPE:
            raise ValueError(f"{name} must be float8_e4m3fn, got {t.dtype}")
        if t.dim() != 2:
            raise ValueError(f"{name} must be 2-D, got shape {tuple(t.shape)}")
    m, k = a_q.shape
    n, kb_ = b_q.shape
    if k != kb_:
        raise ValueError(f"K mismatch: a_q is [{m}, {k}], b_q is [{n}, {kb_}]")
    if k == 0 or k % K_ALIGN != 0:
        raise ValueError(f"K must be a positive multiple of {K_ALIGN}, got {k}")
    if n == 0 or n % N_ALIGN != 0:
        raise ValueError(f"N must be a positive multiple of {N_ALIGN}, got {n}")
    for name, t in (("a_q", a_q), ("b_q", b_q)):
        if t.stride(1) != 1 or t.stride(0) % K_ALIGN != 0:
            raise ValueError(
                f"{name} must be row-major with a row stride that is a multiple of "
                f"{K_ALIGN} elements, got strides {t.stride()}"
            )
        if t.data_ptr() % 16 != 0:
            raise ValueError(f"{name} must be 16-byte aligned")
    kb = cdiv(k, GROUP_K)
    if a_scales.dtype != torch.float32 or tuple(a_scales.shape) != (m, kb):
        raise ValueError(
            f"a_scales must be float32 [{m}, {kb}], got {a_scales.dtype} {tuple(a_scales.shape)}"
        )
    nb = cdiv(n, BLOCK_N)
    if b_scales.dtype != torch.float32 or tuple(b_scales.shape) != (nb, kb):
        raise ValueError(
            f"b_scales must be float32 [{nb}, {kb}], got {b_scales.dtype} {tuple(b_scales.shape)}"
        )
    devices = {t.device for t in (a_q, a_scales, b_q, b_scales)}
    if len(devices) != 1:
        raise ValueError(f"all operands must be on one device, got {sorted(map(str, devices))}")
    if out_dtype != torch.bfloat16:
        raise ValueError(f"out_dtype must be torch.bfloat16, got {out_dtype}")
    return m, n, k


def gemm(
    a_q: torch.Tensor,
    a_scales: torch.Tensor,
    b_q: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    config: int = -1,
    split_k: int = 0,
) -> torch.Tensor:
    """C [M, N] = block-scaled A_q [M, K] @ B_q [N, K]^T; layouts in blockfp8.layout.

    config / split_k override the built-in kernel choice (-1 / 0 = automatic);
    blockfp8.configs() lists the compiled kernel configurations.
    """
    m, n, _ = check_gemm_args(a_q, a_scales, b_q, b_scales, out_dtype)
    if not a_q.is_cuda:
        raise ValueError("gemm runs on CUDA tensors only; use blockfp8.reference_gemm on CPU")
    if not native_available():
        raise RuntimeError(f"native extension not built ({_LIB} missing); see README: Build")
    if m == 0:
        return torch.empty(0, n, dtype=out_dtype, device=a_q.device)
    return torch.ops.blockfp8.gemm(a_q, a_scales, b_q, b_scales, config, split_k)


def configs() -> list[str]:
    """Names of the compiled kernel configurations, indexed by config id."""
    if not native_available():
        raise RuntimeError(f"native extension not built ({_LIB} missing)")
    return torch.ops.blockfp8.config_names().split(";")
