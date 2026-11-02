"""The vLLM plugin: registration order, vLLM's own selection, and the layer math."""

import pytest
import torch

pytest.importorskip("vllm")

import blockfp8  # noqa: E402
import blockfp8_vllm  # noqa: E402


def _candidates():
    from vllm.model_executor.kernels.linear import _POSSIBLE_FP8_BLOCK_KERNELS
    from vllm.platforms import PlatformEnum

    return [k.__name__ for k in _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.CUDA]]


def test_register_is_idempotent_and_goes_right_before_marlin():
    blockfp8_vllm.register()
    blockfp8_vllm.register()
    names = _candidates()
    assert names.count("BlockFp8LinearKernel") == 1
    assert names.index("BlockFp8LinearKernel") + 1 == names.index("MarlinFP8ScaledMMLinearKernel")


def test_entry_point_is_declared():
    from importlib.metadata import entry_points

    eps = {e.name: e.value for e in entry_points(group="vllm.general_plugins")}
    if "blockfp8" not in eps:
        pytest.skip("package not installed (pip install -e .)")
    assert eps["blockfp8"] == "blockfp8_vllm:register"


def _kernel(n, k):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        create_fp8_quant_key,
    )

    blockfp8_vllm.register()
    with set_current_vllm_config(VllmConfig()):
        return init_fp8_linear_kernel(
            activation_quant_key=create_fp8_quant_key(static=False, group_shape=GroupShape(1, 128)),
            weight_quant_key=create_fp8_quant_key(static=True, group_shape=GroupShape(128, 128)),
            input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, weight_shape=(n, k))


@pytest.mark.gpu
@pytest.mark.parametrize("m", [1, 16, 300])
def test_vllm_selects_it_and_the_layer_equals_blockfp8_linear(m):
    if not torch.cuda.is_available() or not blockfp8.native_available():
        pytest.skip("needs a CUDA GPU and the built extension")
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("the plugin only takes over on sm_89")
    from vllm.config import VllmConfig, set_current_vllm_config

    n, k = 1024, 2048
    kernel = _kernel(n, k)
    assert type(kernel).__name__ == "BlockFp8LinearKernel"
    g = torch.Generator(device="cuda").manual_seed(m)
    w, ws = blockfp8.quantize_weights(torch.randn(n, k, generator=g, device="cuda") * 0.02)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(ws.clone(), requires_grad=False)
    layer.input_scale = None
    with set_current_vllm_config(VllmConfig()):
        kernel.process_weights_after_loading(layer)
        x = torch.randn(2, m, k, generator=g, device="cuda").to(torch.bfloat16)
        bias = torch.randn(n, generator=g, device="cuda").to(torch.bfloat16)
        out = kernel.apply_weights(layer, x, bias)
    expect = blockfp8.linear(x.reshape(-1, k), w, ws) + bias
    assert out.shape == (2, m, n) and torch.equal(out.reshape(-1, n), expect)
