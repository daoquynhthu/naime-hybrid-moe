from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on optional runtime package
    triton = None
    tl = None


_MAX_TRITON_VOCAB = 65536


def _torch_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def _can_use_triton(logits: torch.Tensor, labels: torch.Tensor) -> bool:
    if triton is None or tl is None:
        return False
    if not logits.is_cuda or not labels.is_cuda:
        return False
    if logits.ndim < 2 or logits.size(-1) <= 1:
        return False
    if logits.size(-1) > _MAX_TRITON_VOCAB:
        return False
    return logits.is_floating_point() and labels.dtype == torch.long


if triton is not None and tl is not None:

    @triton.jit
    def _ce_forward_kernel(
        logits,
        labels,
        losses,
        n_cols: tl.constexpr,
        ignore_index: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_V)
        mask = offsets < n_cols
        row_logits = tl.load(logits + row * n_cols + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        label = tl.load(labels + row)
        valid = label != ignore_index
        max_logit = tl.max(row_logits, axis=0)
        shifted = row_logits - max_logit
        exp_shifted = tl.exp(shifted)
        denom = tl.sum(exp_shifted, axis=0)
        target = tl.load(logits + row * n_cols + label, mask=valid, other=0.0).to(tl.float32)
        loss = tl.log(denom) + max_logit - target
        tl.store(losses + row, tl.where(valid, loss, 0.0))

    @triton.jit
    def _ce_backward_kernel(
        logits,
        labels,
        grad_logits,
        grad_scale,
        n_cols: tl.constexpr,
        ignore_index: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_V)
        mask = offsets < n_cols
        label = tl.load(labels + row)
        valid = label != ignore_index
        row_logits = tl.load(logits + row * n_cols + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        max_logit = tl.max(row_logits, axis=0)
        exp_shifted = tl.exp(row_logits - max_logit)
        denom = tl.sum(exp_shifted, axis=0)
        probs = exp_shifted / denom
        target = offsets == label
        scale = tl.load(grad_scale)
        grad = (probs - target.to(tl.float32)) * scale
        grad = tl.where(valid & mask, grad, 0.0)
        tl.store(grad_logits + row * n_cols + offsets, grad, mask=mask)


class _TritonCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1)).contiguous()
        flat_labels = labels.reshape(-1).contiguous()
        n_rows, n_cols = flat_logits.shape
        block_v = triton.next_power_of_2(n_cols)
        losses = torch.empty((n_rows,), device=flat_logits.device, dtype=torch.float32)
        _ce_forward_kernel[(n_rows,)](
            flat_logits,
            flat_labels,
            losses,
            n_cols,
            ignore_index,
            BLOCK_V=block_v,
        )
        valid_count = flat_labels.ne(ignore_index).sum()
        ctx.save_for_backward(flat_logits, flat_labels, valid_count)
        ctx.ignore_index = ignore_index
        ctx.input_shape = tuple(logits.shape)
        ctx.n_cols = n_cols
        ctx.block_v = block_v
        return losses.sum() / valid_count.clamp_min(1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        flat_logits, flat_labels, valid_count = ctx.saved_tensors
        grad_logits = torch.empty_like(flat_logits)
        grad_scale = (grad_output.float() / valid_count.clamp_min(1).float()).contiguous()
        _ce_backward_kernel[(flat_logits.size(0),)](
            flat_logits,
            flat_labels,
            grad_logits,
            grad_scale,
            ctx.n_cols,
            ctx.ignore_index,
            BLOCK_V=ctx.block_v,
        )
        return grad_logits.reshape(ctx.input_shape), None, None


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
    backend: str = "auto",
) -> torch.Tensor:
    """Compute causal LM cross entropy with an optional Triton backend.

    The Triton backend is intentionally conservative: it only handles CUDA
    logits with vocab size up to 65,536. Larger multilingual vocabularies fall
    back to PyTorch until the more important fused ``lm_head + CE`` kernel is
    implemented.
    """

    if backend not in {"auto", "torch", "triton_ce"}:
        raise ValueError("lm loss backend must be one of: auto, torch, triton_ce")
    if backend in {"auto", "torch"}:
        return _torch_cross_entropy(logits, labels, ignore_index)
    if _can_use_triton(logits, labels):
        return _TritonCrossEntropy.apply(logits, labels, ignore_index)
    raise RuntimeError(
        "triton_ce backend is unavailable for this input. "
        "Use backend='auto' for PyTorch fallback or reduce vocab size to <= 65536 on CUDA."
    )
