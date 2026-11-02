"""Python entry points for the block-scaled FP8 GEMM (sm_89 CUDA kernel)."""

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
        _register_fakes()
        _loaded = True
    return _loaded


def _register_fakes() -> None:
    """Shape functions, so torch.compile (and vLLM) can trace through the ops."""

    @torch.library.register_fake("blockfp8::gemm")
    def _(a, a_scales, b, b_scales, config=-1, split_k=0):
        return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)

    @torch.library.register_fake("blockfp8::gemm_bf16")
    def _(x, b, b_scales, config=-1, split_k=0, eps=1e-10):
        return x.new_empty((x.shape[0], b.shape[0]), dtype=torch.bfloat16)

    @torch.library.register_fake("blockfp8::linear")
    def _(x, b, b_scales):
        return x.new_empty((x.shape[0], b.shape[0]), dtype=torch.bfloat16)

    @torch.library.register_fake("blockfp8::quantize")
    def _(x, eps=1e-10):
        m, k = x.shape
        q = x.new_empty((m, k), dtype=FP8_DTYPE)
        s = x.new_empty((cdiv(k, GROUP_K), m), dtype=torch.float32).t()
        return q, s


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


def gemm_bf16(
    x: torch.Tensor,
    b_q: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    eps: float = 1e-10,
    config: int = -1,
    split_k: int = 0,
) -> torch.Tensor:
    """C [M, N] bf16 = x [M, K] bf16 @ B^T, quantizing x inside the GEMM.

    x is quantized per row per 128-wide K group exactly as
    quantize_activations(x, eps=eps) (and vLLM's per_token_group_quant_fp8)
    does, so the result is bitwise gemm(*quantize_activations(x), b_q,
    b_scales). config / split_k index blockfp8.fused_configs().
    """
    if x.dtype != torch.bfloat16 or x.dim() != 2:
        raise ValueError(f"x must be 2-D bfloat16, got {x.dtype} {tuple(x.shape)}")
    if x.stride(1) != 1 or x.stride(0) % 8 != 0 or x.data_ptr() % 16 != 0:
        raise ValueError(f"x must be row-major, 16-byte aligned rows; got strides {x.stride()}")
    m, k = x.shape
    if b_q.dtype != FP8_DTYPE or b_q.dim() != 2 or b_q.shape[1] != k:
        raise ValueError(f"b_q must be float8_e4m3fn [N, {k}], got {b_q.dtype} {tuple(b_q.shape)}")
    if k % K_ALIGN != 0 or b_q.shape[0] % N_ALIGN != 0:
        raise ValueError(f"K must be a multiple of {K_ALIGN} and N of {N_ALIGN}")
    nb, kb = cdiv(b_q.shape[0], BLOCK_N), cdiv(k, GROUP_K)
    if b_scales.dtype != torch.float32 or tuple(b_scales.shape) != (nb, kb):
        raise ValueError(f"b_scales must be float32 [{nb}, {kb}]")
    if not x.is_cuda:
        raise ValueError("gemm_bf16 runs on CUDA tensors only")
    if not native_available():
        raise RuntimeError(f"native extension not built ({_LIB} missing); see README: Build")
    return torch.ops.blockfp8.gemm_bf16(x, b_q, b_scales, config, split_k, eps)


def prefers_fused(m: int, n: int, k: int) -> bool:
    """Whether the plan table runs this shape with fused quantization."""
    if not native_available():
        raise RuntimeError(f"native extension not built ({_LIB} missing)")
    return torch.ops.blockfp8.prefers_fused(m, n, k)


def _check_bf16(x: torch.Tensor) -> None:
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_cuda:
        raise ValueError(f"x must be a 2-D bfloat16 CUDA tensor, got {x.dtype} {tuple(x.shape)}")
    if x.stride(1) != 1 or x.stride(0) % 8 != 0 or x.data_ptr() % 16 != 0:
        raise ValueError(f"x must be row-major, 16-byte aligned rows; got strides {x.stride()}")
    if x.shape[1] == 0 or x.shape[1] % K_ALIGN != 0:
        raise ValueError(f"K must be a positive multiple of {K_ALIGN}, got {x.shape[1]}")
    if not native_available():
        raise RuntimeError(f"native extension not built ({_LIB} missing); see README: Build")


def quantize_cuda(x: torch.Tensor, eps: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor]:
    """The CUDA version of quantize_activations for bf16 x (bitwise equal to
    it and to vLLM's per_token_group_quant_fp8); scales are M-contiguous."""
    _check_bf16(x)
    return torch.ops.blockfp8.quantize(x, eps)


def linear(x: torch.Tensor, b_q: torch.Tensor, b_scales: torch.Tensor) -> torch.Tensor:
    """bf16 x [M, K] @ block-FP8 weight^T as one call: fused quantization
    where the plan table says it wins (decode batches), else quantize_cuda
    followed by gemm. This is what the vLLM plugin runs.
    """
    _check_bf16(x)
    return torch.ops.blockfp8.linear(x, b_q, b_scales)


def configs() -> list[str]:
    """Names of the compiled kernel configurations, indexed by config id."""
    if not native_available():
        raise RuntimeError(f"native extension not built ({_LIB} missing)")
    return torch.ops.blockfp8.config_names().split(";")


def fused_configs() -> list[str]:
    """Names of the compiled fused-quantization configurations (gemm_bf16)."""
    if not native_available():
        raise RuntimeError(f"native extension not built ({_LIB} missing)")
    return torch.ops.blockfp8.fused_config_names().split(";")
