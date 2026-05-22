from __future__ import annotations

import os

import torch


def force_fp32_state_attention() -> bool:
    return os.environ.get("NAIME_FORCE_FP32_STATE_ATTENTION", "0") == "1"


def state_softmax(
    scores: torch.Tensor,
    *,
    dim: int = -1,
    mask: torch.Tensor | None = None,
    zero_invalid: bool = False,
) -> torch.Tensor:
    """Softmax for state read/write paths with an fp32 escape hatch.

    Stress tests on RTX 4090 showed bf16/fp16 native softmax+matmul stays finite
    for our state-slot shapes, while the old unconditional fp32 path creates a
    large amount of cast/copy traffic.  Keep ``NAIME_FORCE_FP32_STATE_ATTENTION``
    for emergency rollback if a future shape proves more hostile.
    """

    use_fp32 = force_fp32_state_attention() or not scores.is_cuda
    work = scores.float() if use_fp32 else scores
    if mask is not None:
        work = work.masked_fill(mask, torch.finfo(work.dtype).min)
    weights = torch.softmax(work, dim=dim)
    if zero_invalid and mask is not None:
        valid = (~mask).any(dim=dim, keepdim=True)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
    return weights


def state_softmax_matmul(
    scores: torch.Tensor,
    values: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    zero_invalid: bool = False,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = state_softmax(scores, mask=mask, zero_invalid=zero_invalid)
    if weights.dtype != values.dtype:
        context = torch.matmul(weights, values.float())
    else:
        context = torch.matmul(weights, values)
    if out_dtype is not None and context.dtype != out_dtype:
        context = context.to(dtype=out_dtype)
    return context, weights
