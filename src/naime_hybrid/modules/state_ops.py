from __future__ import annotations

import os
import warnings

import torch


def force_fp32_state_attention() -> bool:
    return os.environ.get("NAIME_FORCE_FP32_STATE_ATTENTION", "0") == "1"


def use_fused_state_attention() -> bool:
    # The native forward/backward prototype is kept as an opt-in experiment.
    # Benchmarks show it only helps tiny masked slot banks; for larger V6
    # history banks PyTorch's bmm/softmax kernels are materially faster.
    return os.environ.get("NAIME_USE_FUSED_STATE_ATTENTION", "0") == "1"


_fused_disabled_after_error = False
_warned_fused_error = False


class _FusedStateSoftmaxMatmulFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None,
        zero_invalid: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from naime_hybrid.kernels.cuda_ext import load_cuda_extension

        ext = load_cuda_extension()
        mask_arg = (
            torch.empty(0, device=scores.device, dtype=torch.bool)
            if mask is None
            else mask.expand(scores.size(0), -1, -1) if mask.size(0) == 1 else mask
        )
        context, weights = ext.fused_state_softmax_matmul_forward(scores, values, mask_arg, bool(zero_invalid))
        ctx.save_for_backward(weights, values, mask_arg)
        ctx.has_mask = mask is not None
        return context, weights

    @staticmethod
    def backward(
        ctx,
        grad_context: torch.Tensor,
        grad_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        from naime_hybrid.kernels.cuda_ext import load_cuda_extension

        weights, values, mask = ctx.saved_tensors
        ext = load_cuda_extension()
        grad_weights_arg = (
            torch.empty(0, device=weights.device, dtype=weights.dtype)
            if grad_weights is None
            else grad_weights.to(dtype=weights.dtype)
        )
        mask_arg = mask if ctx.has_mask else torch.empty(0, device=weights.device, dtype=torch.bool)
        grad_scores, grad_values = ext.fused_state_softmax_matmul_backward(
            grad_context.to(dtype=weights.dtype),
            grad_weights_arg,
            weights,
            values,
            mask_arg,
        )
        return grad_scores, grad_values, None, None


def _can_use_fused_state_attention(scores: torch.Tensor, values: torch.Tensor, mask: torch.Tensor | None) -> bool:
    if force_fp32_state_attention() or not use_fused_state_attention():
        return False
    if _fused_disabled_after_error:
        return False
    if not scores.is_cuda or not values.is_cuda:
        return False
    if scores.dtype != values.dtype or scores.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return False
    if scores.ndim != 3 or values.ndim != 3:
        return False
    if scores.size(0) != values.size(0) or scores.size(2) != values.size(1):
        return False
    if scores.size(2) > 1024:
        return False
    if mask is not None and (not mask.is_cuda or mask.dtype != torch.bool or mask.ndim != 3):
        return False
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and getattr(compiler, "is_compiling", lambda: False)():
        return False
    return True


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
    global _fused_disabled_after_error, _warned_fused_error
    if out_dtype is None:
        out_dtype = scores.dtype
    if _can_use_fused_state_attention(scores, values, mask):
        try:
            context, weights = _FusedStateSoftmaxMatmulFn.apply(scores, values, mask, zero_invalid)
            if context.dtype != out_dtype:
                context = context.to(dtype=out_dtype)
            return context, weights
        except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain.
            _fused_disabled_after_error = True
            if not _warned_fused_error:
                warnings.warn(
                    f"Fused state attention unavailable; falling back to torch implementation: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _warned_fused_error = True

    weights = state_softmax(scores, mask=mask, zero_invalid=zero_invalid)
    if weights.dtype != values.dtype:
        context = torch.matmul(weights, values.float())
    else:
        context = torch.matmul(weights, values)
    if out_dtype is not None and context.dtype != out_dtype:
        context = context.to(dtype=out_dtype)
    return context, weights
