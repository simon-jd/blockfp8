"""BlockFp8LinearKernel: vLLM's block-FP8 linear-kernel interface over blockfp8."""

import torch
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.platforms import current_platform

import blockfp8


class BlockFp8LinearKernel(Fp8BlockScaledMMLinearKernel):
    """W8A8 block-FP8 linear: blockfp8.linear (activation quantization fused
    into the GEMM for decode batches, a separate quantization kernel
    otherwise) on weights left as the checkpoint stores them ([N, K] e4m3,
    one float32 scale per 128x128 block)."""

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "needs CUDA"
        if compute_capability is None:
            cc = current_platform.get_device_capability()
            compute_capability = None if cc is None else cc[0] * 10 + cc[1]
        if compute_capability != 89:
            return False, f"is built for sm_89 only, not sm_{compute_capability}"
        if not blockfp8.native_available():
            return False, "needs the built blockfp8 extension (blockfp8/_C.so)"
        return True, None

    @classmethod
    def can_implement(cls, config) -> tuple[bool, str | None]:
        ok, reason = super().can_implement(config)  # dynamic activation scales only
        if not ok:
            return ok, reason
        act, wgt = config.activation_quant_key.scale, config.weight_quant_key.scale
        if tuple(act.group_shape) != (1, blockfp8.GROUP_K):
            return False, f"needs 1x128 activation groups, got {tuple(act.group_shape)}"
        if tuple(wgt.group_shape) != (blockfp8.BLOCK_N, blockfp8.GROUP_K):
            return False, f"needs 128x128 weight blocks, got {tuple(wgt.group_shape)}"
        if config.weight_quant_key.dtype != blockfp8.FP8_DTYPE or wgt.dtype != torch.float32:
            return False, "needs float8_e4m3fn weights with float32 block scales"
        if config.input_dtype != torch.bfloat16 or config.out_dtype != torch.bfloat16:
            return False, "needs bf16 activations and output"
        n, k = config.weight_shape
        if n % 8 or k % 16:
            return False, f"needs N % 8 == 0 and K % 16 == 0, got N={n} K={k}"
        return True, None

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        params = self._get_layer_params(layer)
        weight = params.weight
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.ops.blockfp8.linear(x2, weight, params.block_scale)
        if bias is not None:
            out = out + bias
        return out.view(*x.shape[:-1], weight.shape[0])

    def apply_block_scaled_mm(self, A: torch.Tensor, B: torch.Tensor, As: torch.Tensor,
                              Bs: torch.Tensor) -> torch.Tensor:
        return torch.ops.blockfp8.gemm(A, As, B, Bs, -1, 0)
