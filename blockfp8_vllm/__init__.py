"""vLLM plugin: run block-FP8 linear layers on sm_89 with blockfp8.linear.

On Ada, vLLM 0.31 runs 128x128-block FP8 checkpoints (DeepSeek-V3 style,
e.g. Qwen3-8B-FP8) with MarlinFP8ScaledMMLinearKernel: weight-only FP8,
bf16 activations and bf16 MMAs. register(), the vllm.general_plugins entry
point that vLLM calls in each of its processes, inserts BlockFp8LinearKernel
into vLLM's block-FP8 kernel list right ahead of Marlin. vLLM still picks
the first candidate that reports itself supported and able to implement a
layer, so the new kernel takes over exactly where it applies (sm_89, bf16,
128x128 weight blocks, dynamic 1x128 activation groups) and every other GPU
or layer keeps vLLM's own choice. Nothing in the installed vLLM is edited.

To run stock vLLM with this package installed, leave "blockfp8" out of
VLLM_PLUGINS (e.g. VLLM_PLUGINS=lora_filesystem_resolver).
"""

import logging

logger = logging.getLogger(__name__)

_registered = False


def register() -> None:
    """Entry point; safe to call more than once per process."""
    global _registered
    if _registered:
        return
    from vllm.model_executor.kernels import linear as linear_kernels
    from vllm.platforms import PlatformEnum

    from .kernel import BlockFp8LinearKernel

    candidates = linear_kernels._POSSIBLE_FP8_BLOCK_KERNELS.setdefault(PlatformEnum.CUDA, [])
    if BlockFp8LinearKernel not in candidates:
        names = [k.__name__ for k in candidates]
        pos = names.index("MarlinFP8ScaledMMLinearKernel") if "MarlinFP8ScaledMMLinearKernel" \
            in names else 0
        candidates.insert(pos, BlockFp8LinearKernel)
        logger.info("blockfp8: BlockFp8LinearKernel inserted ahead of %s",
                    names[pos] if pos < len(names) else "nothing")
    _registered = True
