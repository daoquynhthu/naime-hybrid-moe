import torch


def prepare_attention_mask_for_device(
    attention_mask: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor | None, bool | None]:
    """Move a padding mask to device, or drop full masks for causal SDPA.

    HFDiskCausalDataset emits an all-true mask for fixed-length blocks. Passing
    that mask into attention disables the efficient ``is_causal=True`` path, so
    we explicitly mark the batch as padding-free and let attention run causal-only.
    """
    if attention_mask is None:
        return None, None
    mask_bool = attention_mask.to(torch.bool)
    if bool(mask_bool.all().item()):
        return None, False
    return mask_bool.to(device, non_blocking=True), True
