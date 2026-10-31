import pytest
import torch

from blockfp8 import FP8_DTYPE, check_gemm_args, gemm


def _args(m=4, n=256, k=512, col_major=True):
    a_q = torch.zeros(m, k).to(FP8_DTYPE)
    b_q = torch.zeros(n, k).to(FP8_DTYPE)
    kb, nb = -(-k // 128), -(-n // 128)
    a_s = torch.ones(kb, m).t() if col_major else torch.ones(m, kb)
    b_s = torch.ones(nb, kb)
    return a_q, a_s, b_q, b_s


@pytest.mark.parametrize("col_major", [True, False])
def test_valid_args(col_major):
    assert check_gemm_args(*_args(col_major=col_major), torch.bfloat16) == (4, 256, 512)


def test_tails_are_accepted():
    assert check_gemm_args(*_args(m=3, n=136, k=144), torch.bfloat16) == (3, 136, 144)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda a: (a[0].float(), *a[1:]), "float8_e4m3fn"),
        (lambda a: (a[0][:, :500], a[1], a[2], a[3]), "K mismatch"),
        (lambda a: (a[0], a[1], a[2].to(torch.float8_e5m2), a[3]), "float8_e4m3fn"),
        (lambda a: (a[0], a[1].double(), a[2], a[3]), "a_scales"),
        (lambda a: (a[0], a[1][:, :3], a[2], a[3]), "a_scales"),
        (lambda a: (a[0], a[1], a[2], a[3][:1]), "b_scales"),
        (lambda a: (a[0].t().contiguous().t(), *a[1:]), "row-major"),
    ],
)
def test_rejects(mutate, match):
    with pytest.raises(ValueError, match=match):
        check_gemm_args(*mutate(_args()), torch.bfloat16)


def test_rejects_unaligned_k_and_n():
    with pytest.raises(ValueError, match="K must be"):
        check_gemm_args(*_args(k=200), torch.bfloat16)
    with pytest.raises(ValueError, match="N must be"):
        check_gemm_args(*_args(n=100), torch.bfloat16)


def test_rejects_other_output_dtypes():
    with pytest.raises(ValueError, match="out_dtype"):
        check_gemm_args(*_args(), torch.float16)


def test_cpu_tensors_point_to_the_reference():
    with pytest.raises(ValueError, match="reference_gemm"):
        gemm(*_args())
