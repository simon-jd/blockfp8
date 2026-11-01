"""The vLLM 0.31 kernels this project is compared against, called the way vLLM
calls them.

On CUDA, vLLM picks the block-FP8 linear kernel by walking
_POSSIBLE_FP8_BLOCK_KERNELS in order and taking the first one that reports
itself supported; selected_block_fp8_kernel() asks vLLM which one that is on
the current GPU. Both candidates that matter on sm_89 are wrapped here:
  * MarlinFP8ScaledMMLinearKernel: weight-only FP8 (W8A16). Weights are
    repacked once; activations stay bf16 and the MMAs run in bf16.
  * w8a8_triton_block_scaled_mm: the Triton block-FP8 W8A8 kernel, used when
    Marlin is disabled (VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel).
"""

import torch


def selected_block_fp8_kernel(n: int = 4096, k: int = 4096) -> str:
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        create_fp8_quant_key,
    )

    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        kernel = init_fp8_linear_kernel(
            activation_quant_key=create_fp8_quant_key(static=False, group_shape=GroupShape(1, 128)),
            weight_quant_key=create_fp8_quant_key(static=True, group_shape=GroupShape(128, 128)),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(n, k),
        )
    return type(kernel).__name__


class MarlinFp8Linear:
    """vLLM's FP8 Marlin path for a block-quantized [N, K] weight."""

    def __init__(self, b_q: torch.Tensor, b_s: torch.Tensor):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            process_fp8_weight_block_strategy,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_fp8_layer_for_marlin,
        )

        n, k = b_q.shape
        layer = torch.nn.Module()
        w, s = process_fp8_weight_block_strategy(b_q.clone(), b_s.clone())
        layer.weight = torch.nn.Parameter(w, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(s, requires_grad=False)
        layer.output_size_per_partition = n
        layer.input_size_per_partition = k
        layer.weight_block_size = [128, 128]
        layer.orig_dtype = torch.bfloat16
        layer.logical_widths = [n]
        layer.input_scale = None
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            prepare_fp8_layer_for_marlin(layer, size_k_first=False)
        self.layer, self.n, self.k = layer, n, k

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
        )

        return apply_fp8_marlin_linear(
            input=x, weight=self.layer.weight, weight_scale=self.layer.weight_scale_inv,
            workspace=None, size_n=self.n, size_k=self.k, bias=None,
        )


def vllm_group_quant(x: torch.Tensor, column_major: bool):
    """vLLM's CUDA per-token-group FP8 quantization (what its W8A8 paths run)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    return per_token_group_quant_fp8(x, 128, column_major_scales=column_major, use_ue8m0=False)
