from __future__ import annotations

import os
import warnings

import torch
from torch import nn


class _FusedRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        from naime_hybrid.kernels.cuda_ext import load_cuda_extension

        ext = load_cuda_extension()
        y, inv_rms = ext.fused_rms_norm_forward(x, weight, float(eps))
        ctx.save_for_backward(x, weight, inv_rms)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        from naime_hybrid.kernels.cuda_ext import load_cuda_extension

        x, weight, inv_rms = ctx.saved_tensors
        ext = load_cuda_extension()
        grad_x, grad_weight = ext.fused_rms_norm_backward(grad_output, x, weight, inv_rms)
        return grad_x, grad_weight, None


class RMSNorm(nn.Module):
    _fused_disabled_after_error = False
    _warned_fused_error = False

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._should_use_fused(x):
            try:
                return _FusedRMSNormFn.apply(x, self.weight, self.eps)
            except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain.
                type(self)._fused_disabled_after_error = True
                if not type(self)._warned_fused_error:
                    warnings.warn(
                        f"Fused RMSNorm unavailable; falling back to torch implementation: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    type(self)._warned_fused_error = True

        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).type_as(x)

    def _should_use_fused(self, x: torch.Tensor) -> bool:
        if os.environ.get("NAIME_DISABLE_FUSED_RMSNORM", "0") == "1":
            return False
        if type(self)._fused_disabled_after_error:
            return False
        if not x.is_cuda or x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            return False
        if not self.weight.is_cuda or self.weight.dtype != torch.float32:
            return False
        compiler = getattr(torch, "compiler", None)
        if compiler is not None and getattr(compiler, "is_compiling", lambda: False)():
            return False
        return True
