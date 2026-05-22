from __future__ import annotations

import torch
import torch.nn.functional as F

from .cuda_ext import load_cuda_extension


class _CudaExtFusedLmCE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        flat_hidden = hidden.reshape(-1, hidden.size(-1)).contiguous()
        flat_labels = labels.reshape(-1).contiguous()
        kernel_weight = weight.to(dtype=flat_hidden.dtype).contiguous()
        ext = load_cuda_extension()
        loss, valid_count = ext.fused_lm_ce_forward(flat_hidden, kernel_weight, flat_labels, ignore_index)
        ctx.save_for_backward(flat_hidden, weight.contiguous(), flat_labels, valid_count)
        ctx.ignore_index = ignore_index
        ctx.hidden_shape = tuple(hidden.shape)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        flat_hidden, weight, flat_labels, valid_count = ctx.saved_tensors
        logits = flat_hidden.float().matmul(weight.float().t())
        probs = torch.softmax(logits, dim=-1)
        valid_mask = flat_labels.ne(ctx.ignore_index)
        safe_labels = flat_labels.clamp_min(0)
        one_hot = F.one_hot(safe_labels, num_classes=weight.size(0)).to(dtype=probs.dtype)
        grad_logits = probs - one_hot
        grad_logits = grad_logits * valid_mask.unsqueeze(-1)
        grad_logits = grad_logits * (grad_output.float() / valid_count.clamp_min(1).float())
        grad_hidden = grad_logits.matmul(weight.float()).to(dtype=flat_hidden.dtype).reshape(ctx.hidden_shape)
        grad_weight = grad_logits.t().matmul(flat_hidden.float()).to(dtype=weight.dtype)
        return grad_hidden, grad_weight, None, None


def fused_lm_cross_entropy_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
    backend: str = "auto",
) -> torch.Tensor:
    if backend not in {"auto", "torch", "cuda_ext_fused_ce"}:
        raise ValueError("fused LM CE backend must be one of: auto, torch, cuda_ext_fused_ce")
    if backend in {"auto", "torch"}:
        logits = hidden_states.float().matmul(lm_head_weight.float().t())
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=ignore_index)
    if not hidden_states.is_cuda or not lm_head_weight.is_cuda or not labels.is_cuda:
        raise RuntimeError("cuda_ext_fused_ce requires CUDA hidden states, weights, and labels")
    return _CudaExtFusedLmCE.apply(hidden_states, lm_head_weight, labels, ignore_index)
