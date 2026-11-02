"""GPU tests of the CUDA kernel against the FP64 reference.

Four kinds of check:
  * exact: small-integer operands with power-of-two scales make every
    partial sum exact in the MMA and in FP32, so the kernel must return the
    exact result rounded to bf16, bit for bit, for every configuration,
    tail shape and split count;
  * bounded: every result must stay within blockfp8.error_bound, the
    worst case of the measured MMA accumulation model;
  * differential: on the same operands the error beyond bf16 rounding must
    be no worse than that of vLLM's Triton block-FP8 kernel (same MMA
    instruction, same promotion interval);
  * adversarial: cancellation, extreme scales, outliers, zeros.
"""

import glob
import os

import pytest
import torch

import blockfp8
from blockfp8 import FP8_DTYPE, error_bound, reference_gemm
from blockfp8.reference import BF16_U

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif(not blockfp8.native_available(), reason="extension not built"),
]

DEV = "cuda"
CONFIGS = list(range(len(blockfp8.configs()))) if blockfp8.native_available() else []


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def integer_operands(m, n, k, seed=0):
    """|a|, |b| <= 4 integers, scales in 2^[-1, 1]: every partial sum is exact."""
    g = _gen(seed)
    kb, nb = -(-k // 128), -(-n // 128)
    a = torch.randint(-4, 5, (m, k), generator=g, device=DEV).float().to(FP8_DTYPE)
    b = torch.randint(-4, 5, (n, k), generator=g, device=DEV).float().to(FP8_DTYPE)
    sa = torch.exp2(torch.randint(-1, 2, (m, kb), generator=g, device=DEV).float())
    sb = torch.exp2(torch.randint(-1, 2, (nb, kb), generator=g, device=DEV).float())
    return a, sa, b, sb


def realistic_operands(m, n, k, seed=0, col_major=True):
    g = _gen(seed)
    x = torch.randn(m, k, generator=g, device=DEV)
    w = torch.randn(n, k, generator=g, device=DEV) * 0.02
    a, sa = blockfp8.quantize_activations(x, column_major_scales=col_major)
    b, sb = blockfp8.quantize_weights(w)
    return a, sa, b, sb


def _triton():
    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            w8a8_triton_block_scaled_mm,
        )
    except Exception:  # vLLM not installed
        return None
    return w8a8_triton_block_scaled_mm


def excess_error(c, ref, ref_abs):
    """Error beyond the bf16 rounding of the exact result, relative to sum |terms|."""
    err = (c.double() - ref).abs()
    return ((err - BF16_U * ref.abs()).clamp_min(0) / ref_abs.clamp_min(1e-300))


def check_bounded(c, a, sa, b, sb, vs_triton=True):
    ref = reference_gemm(a, sa, b, sb)
    ref_abs = reference_gemm(a, sa, b, sb, absolute=True)
    err = (c.double() - ref).abs()
    bound = error_bound(ref, ref_abs, a.shape[1])
    worst = (err / bound).max().item()
    assert worst <= 1.0, f"error exceeds the bound by {worst:.3g}x"
    mm = _triton() if vs_triton else None
    if mm is not None:
        ours = excess_error(c, ref, ref_abs)
        theirs = excess_error(mm(a, b, sa.contiguous(), sb, [128, 128], torch.bfloat16), ref, ref_abs)
        assert ours.max() <= 2 * theirs.max() + 2.0**-22, (ours.max().item(), theirs.max().item())
        assert ours.mean() <= 2 * theirs.mean() + 2.0**-24, (ours.mean().item(), theirs.mean().item())
    return err


SHAPES = [
    (1, 128, 128),
    (5, 256, 384),
    (16, 512, 1024),
    (33, 264, 272),  # tails in M, N and K
    (64, 384, 2048),
    (130, 136, 4096),
    (257, 1024, 512),
    (300, 520, 144),
]


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("shape", SHAPES)
def test_exact_integer_case_all_configs(config, shape):
    a, sa, b, sb = integer_operands(*shape, seed=hash(shape) % 1000)
    c = blockfp8.gemm(a, sa, b, sb, config=config)
    expect = reference_gemm(a, sa, b, sb).to(torch.bfloat16)
    assert torch.equal(c, expect)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("splits", [2, 3, 7])
def test_exact_integer_case_split_k(config, splits):
    a, sa, b, sb = integer_operands(37, 392, 2048 + 16, seed=splits)
    c = blockfp8.gemm(a, sa, b, sb, config=config, split_k=splits)
    assert torch.equal(c, reference_gemm(a, sa, b, sb).to(torch.bfloat16))


@pytest.mark.parametrize("shape", SHAPES + [(1, 4096, 4096), (64, 6144, 4096), (512, 4096, 12288)])
def test_realistic_within_bound_auto_plan(shape):
    a, sa, b, sb = realistic_operands(*shape)
    check_bounded(blockfp8.gemm(a, sa, b, sb), a, sa, b, sb)


@pytest.mark.parametrize("col_major", [True, False])
def test_both_activation_scale_layouts(col_major):
    a, sa, b, sb = realistic_operands(70, 256, 640, col_major=col_major)
    c = blockfp8.gemm(a, sa, b, sb)
    c2 = blockfp8.gemm(a, sa.contiguous() if col_major else sa.t().contiguous().t(), b, sb)
    assert torch.equal(c, c2)
    check_bounded(c, a, sa, b, sb)


@pytest.mark.parametrize("config, splits", [(0, 1), (0, 4), (3, 1), (3, 8), (2, 5)])
def test_bitwise_deterministic(config, splits):
    a, sa, b, sb = realistic_operands(48, 1024, 4096, seed=3)
    outs = [blockfp8.gemm(a, sa, b, sb, config=config, split_k=splits) for _ in range(5)]
    assert all(torch.equal(outs[0], o) for o in outs[1:])


def test_split_k_inside_cuda_graph():
    a, sa, b, sb = realistic_operands(4, 4096, 4096, seed=4)
    eager = blockfp8.gemm(a, sa, b, sb, config=3, split_k=8)
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        blockfp8.gemm(a, sa, b, sb, config=3, split_k=8)  # warm-up allocations
        with torch.cuda.graph(graph, stream=s):
            out = blockfp8.gemm(a, sa, b, sb, config=3, split_k=8)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)


def test_cancellation_across_blocks():
    # Block 1 is block 0 negated: the exact result is zero, so everything
    # left is accumulation error and must stay within the bound.
    a, sa, b, sb = realistic_operands(40, 256, 128, seed=5)
    a2 = torch.cat([a, a], dim=1)
    sa2 = torch.cat([sa, sa], dim=1)
    b2 = torch.cat([b, (-b.float()).to(FP8_DTYPE)], dim=1)
    sb2 = torch.cat([sb, sb], dim=1)
    for config in CONFIGS:
        c = blockfp8.gemm(a2, sa2, b2, sb2, config=config)
        check_bounded(c, a2, sa2, b2, sb2)


def test_cancellation_leaves_small_residual():
    # Large +X and -X blocks plus a block 2^-12 smaller: the exact result is
    # the small block alone. Because every block is promoted on its own, the
    # large blocks can leave at most their MMA-level error (13 bits) behind,
    # and the result must stay within the bound and match Triton's accuracy.
    a, sa, b, sb = realistic_operands(24, 384, 128, seed=6)
    a3 = torch.cat([a, a, a], dim=1)
    b3 = torch.cat([b, (-b.float()).to(FP8_DTYPE), b], dim=1)
    sa3 = torch.cat([sa, sa, sa * 2.0**-12], dim=1)
    sb3 = torch.cat([sb, sb, sb], dim=1)
    c = blockfp8.gemm(a3, sa3, b3, sb3)
    ref = reference_gemm(a3, sa3, b3, sb3)
    big = reference_gemm(a, sa, b, sb, absolute=True)
    err = (c.double() - ref).abs()
    assert torch.all(err <= BF16_U * ref.abs() + 2.0**-13 * big)
    check_bounded(c, a3, sa3, b3, sb3)


@pytest.mark.parametrize("log2_scale", [-60, -20, 20, 50])
def test_extreme_scales(log2_scale):
    a, sa, b, sb = realistic_operands(17, 256, 512, seed=7)
    sa = sa * 2.0**log2_scale
    sb = sb * 2.0 ** (-log2_scale // 2)
    c = blockfp8.gemm(a, sa, b, sb)
    check_bounded(c, a, sa, b, sb)


def test_extreme_fp8_values():
    g = _gen(8)
    m, n, k = 32, 128, 512
    # Mix of the largest normal, subnormals and zeros.
    choices = torch.tensor([448.0, -448.0, 2.0**-9, -(2.0**-9), 2.0**-6, 0.0], device=DEV)
    a = choices[torch.randint(0, 6, (m, k), generator=g, device=DEV)].to(FP8_DTYPE)
    b = choices[torch.randint(0, 6, (n, k), generator=g, device=DEV)].to(FP8_DTYPE)
    sa = torch.full((m, 4), 2.0**-10, device=DEV)
    sb = torch.full((1, 4), 2.0**-8, device=DEV)
    check_bounded(blockfp8.gemm(a, sa, b, sb), a, sa, b, sb)


def test_outlier_channel_and_zero_rows():
    g = _gen(9)
    x = torch.randn(20, 1024, generator=g, device=DEV)
    x[:, 77] *= 300.0  # an outlier channel, common in LLM activations
    x[3] = 0.0  # an all-zero row
    a, sa = blockfp8.quantize_activations(x)
    b, sb = blockfp8.quantize_weights(torch.randn(512, 1024, generator=g, device=DEV) * 0.02)
    c = blockfp8.gemm(a, sa, b, sb)
    assert torch.all(c[3] == 0)
    check_bounded(c, a, sa, b, sb)


def test_m_zero_returns_empty():
    a, sa, b, sb = realistic_operands(1, 256, 256)
    c = blockfp8.gemm(a[:0], sa[:0], b, sb)
    assert c.shape == (0, 256) and c.dtype == torch.bfloat16


def test_op_rejects_bad_shapes_itself():
    a, sa, b, sb = realistic_operands(4, 256, 256)
    with pytest.raises(RuntimeError, match="b_scales"):
        torch.ops.blockfp8.gemm(a, sa, b, sb[:1], -1, 0)


def _qwen_snapshot():
    root = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots")
    files = sorted(glob.glob(os.path.join(root, "*", "*.safetensors")))
    return files


@pytest.mark.skipif(not _qwen_snapshot(), reason="Qwen3-8B-FP8 not in the local HF cache")
def test_real_qwen3_8b_fp8_down_proj():
    from safetensors import safe_open

    name = "model.layers.0.mlp.down_proj"
    for path in _qwen_snapshot():
        with safe_open(path, "pt", device=DEV) as f:
            if name + ".weight" in f.keys():
                b = f.get_tensor(name + ".weight")
                sb = f.get_tensor(name + ".weight_scale_inv").float()
                break
    assert b.dtype == FP8_DTYPE and tuple(b.shape) == (4096, 12288)
    g = _gen(10)
    a, sa = blockfp8.quantize_activations(torch.randn(16, 12288, generator=g, device=DEV))
    check_bounded(blockfp8.gemm(a, sa, b, sb), a, sa, b, sb)


# ---- fused activation quantization (gemm_bf16)

FUSED = list(range(len(blockfp8.fused_configs()))) if blockfp8.native_available() else []


def _vllm_quant():
    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
    except Exception:  # vLLM not installed
        return None
    return lambda x: per_token_group_quant_fp8(x, 128, column_major_scales=True, use_ue8m0=False)


def bf16_activations(m, k, seed=0, ties=False):
    """Random bf16 activations. ties=True gives every 128-wide group an amax
    of 448 * 2^j, so the scale is a power of two, x / scale is exact and
    bf16 values often fall exactly halfway between two e4m3 values."""
    g = _gen(seed)
    x = torch.randn(m, k, generator=g, device=DEV) * 2
    if ties:
        kb = -(-k // 128)
        xp = torch.nn.functional.pad(x, (0, kb * 128 - k)).view(m, kb, 128)
        top = 448.0 * torch.exp2(torch.randint(-12, 12, (m, kb, 1), generator=g, device=DEV).float())
        xp = xp / xp.abs().amax(dim=2, keepdim=True) * top
        xp[:, :, :1] = top * torch.sign(torch.randn(m, kb, 1, generator=g, device=DEV))
        x = xp.reshape(m, kb * 128)[:, :k]
    return x.to(torch.bfloat16).contiguous()


def _weights(n, k, seed):
    g = _gen(seed)
    return blockfp8.quantize_weights(torch.randn(n, k, generator=g, device=DEV) * 0.02)


@pytest.mark.parametrize("config", FUSED)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("ties", [False, True])
def test_fused_equals_quantize_then_gemm(config, shape, ties):
    m, n, k = shape
    x = bf16_activations(m, k, seed=hash(shape) % 1000, ties=ties)
    b, sb = _weights(n, k, seed=1)
    a, sa = blockfp8.quantize_activations(x)
    expect = blockfp8.gemm(a, sa, b, sb, config=0, split_k=1)
    c = blockfp8.gemm_bf16(x, b, sb, config=config, split_k=1)
    assert torch.equal(c, expect)


@pytest.mark.parametrize("config", FUSED)
@pytest.mark.parametrize("splits", [2, 3, 7])
def test_fused_split_k(config, splits):
    x = bf16_activations(37, 2048 + 16, seed=splits, ties=True)
    b, sb = _weights(392, 2048 + 16, seed=2)
    a, sa = blockfp8.quantize_activations(x)
    expect = blockfp8.gemm(a, sa, b, sb, config=0, split_k=splits)
    assert torch.equal(blockfp8.gemm_bf16(x, b, sb, config=config, split_k=splits), expect)


@pytest.mark.skipif(_vllm_quant() is None, reason="vLLM not installed")
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("shape", [(1, 4096, 4096), (16, 6144, 4096), (64, 4096, 12288)])
def test_fused_equals_vllm_quant_then_gemm(shape, ties):
    m, n, k = shape
    x = bf16_activations(m, k, seed=11, ties=ties)
    b, sb = _weights(n, k, seed=3)
    q, s = _vllm_quant()(x)
    for q2, s2 in (blockfp8.quantize_activations(x), blockfp8.quantize_cuda(x)):
        assert torch.equal(q.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(s, s2)
    for config in FUSED:
        assert torch.equal(blockfp8.gemm_bf16(x, b, sb, config=config, split_k=1),
                           blockfp8.gemm(q, s, b, sb, config=0, split_k=1))


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("shape", [(1, 128), (7, 272), (33, 4096), (300, 1040)])
def test_quantize_cuda_equals_reference(shape, ties):
    x = bf16_activations(*shape, seed=15, ties=ties)
    x[0, :16] = 0.0
    for eps in (1e-10, 1e-2):
        q, s = blockfp8.quantize_cuda(x, eps=eps)
        q2, s2 = blockfp8.quantize_activations(x, eps=eps)
        assert s.stride() == s2.stride() == (1, shape[0])
        assert torch.equal(q.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(s, s2)


def test_fused_zero_rows_outliers_and_eps():
    x = bf16_activations(20, 1024, seed=12)
    x[:, 77] *= 300.0
    x[3] = 0.0
    x[5, 128:256] = 1e-38  # below eps * 448: the scale is floored at eps / 448
    b, sb = _weights(512, 1024, seed=4)
    for eps in (1e-10, 1e-4):
        a, sa = blockfp8.quantize_activations(x, eps=eps)
        for config in FUSED:
            c = blockfp8.gemm_bf16(x, b, sb, eps=eps, config=config, split_k=1)
            assert torch.equal(c, blockfp8.gemm(a, sa, b, sb, config=0, split_k=1))
            assert torch.all(c[3] == 0)


@pytest.mark.parametrize("m", [1, 4, 16, 64, 256])
def test_fused_auto_plan_and_linear(m):
    x = bf16_activations(m, 4096, seed=m)
    b, sb = _weights(6144, 4096, seed=5)
    a, sa = blockfp8.quantize_activations(x)
    check_bounded(blockfp8.gemm_bf16(x, b, sb), a, sa, b, sb, vs_triton=False)
    expect = blockfp8.gemm_bf16(x, b, sb) if blockfp8.prefers_fused(m, 6144, 4096) else \
        blockfp8.gemm(a, sa, b, sb)
    assert torch.equal(blockfp8.linear(x, b, sb), expect)
    assert blockfp8.linear(x[:0], b, sb).shape == (0, 6144)


def test_fused_inside_cuda_graph():
    x = bf16_activations(4, 4096, seed=13)
    b, sb = _weights(4096, 4096, seed=6)
    eager = blockfp8.gemm_bf16(x, b, sb, config=0, split_k=8)
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        blockfp8.gemm_bf16(x, b, sb, config=0, split_k=8)
        with torch.cuda.graph(graph, stream=s):
            out = blockfp8.gemm_bf16(x, b, sb, config=0, split_k=8)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)


def test_fused_rejects_bad_input():
    b, sb = _weights(256, 256, seed=7)
    with pytest.raises(ValueError):
        blockfp8.gemm_bf16(torch.zeros(2, 256, device=DEV), b, sb)  # float32
    with pytest.raises(ValueError):
        blockfp8.gemm_bf16(torch.zeros(2, 512, device=DEV, dtype=torch.bfloat16)[:, ::2], b, sb)
    with pytest.raises(RuntimeError, match="b_scales"):
        torch.ops.blockfp8.gemm_bf16(torch.zeros(2, 256, device=DEV, dtype=torch.bfloat16),
                                     b, sb[:, :1], -1, 0, 1e-10)


def test_ops_trace_under_torch_compile():
    x = bf16_activations(8, 256, seed=14)
    b, sb = _weights(256, 256, seed=8)
    a, sa = blockfp8.quantize_activations(x)

    def f(x, a, sa):
        q, s = torch.ops.blockfp8.quantize(x, 1e-10)
        return torch.ops.blockfp8.gemm_bf16(x, b, sb, -1, 0, 1e-10) + \
            torch.ops.blockfp8.gemm(a, sa, b, sb, -1, 0) + torch.ops.blockfp8.linear(x, b, sb) + \
            torch.ops.blockfp8.gemm(q, s, b, sb, -1, 0)

    out = torch.compile(f, fullgraph=True)(x, a, sa)
    assert torch.equal(out, f(x, a, sa))
