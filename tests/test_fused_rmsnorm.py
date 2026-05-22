import os

import pytest
import torch

from naime_hybrid.modules.norm import RMSNorm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fused RMSNorm")
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.bfloat16, 6e-3, 6e-3),
        (torch.float16, 3e-3, 3e-3),
        (torch.float32, 2e-5, 2e-5),
    ],
)
def test_fused_rmsnorm_matches_torch_reference(dtype: torch.dtype, atol: float, rtol: float):
    previous_disable = os.environ.pop("NAIME_DISABLE_FUSED_RMSNORM", None)
    try:
        torch.manual_seed(7)
        x = torch.randn(4, 9, 128, device="cuda", dtype=dtype, requires_grad=True)
        weight = torch.randn(128, device="cuda", dtype=torch.float32, requires_grad=True)
        grad = torch.randn_like(x)

        ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight).to(dtype)
        ref.backward(grad, retain_graph=True)
        ref_grad_x = x.grad.detach().clone()
        ref_grad_w = weight.grad.detach().clone()
        x.grad = None
        weight.grad = None

        mod = RMSNorm(128).cuda()
        mod.weight.data.copy_(weight.detach())
        out = mod(x)
        out.backward(grad)

        assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
        assert torch.allclose(x.grad.float(), ref_grad_x.float(), atol=atol * 2, rtol=rtol * 2)
        assert torch.allclose(mod.weight.grad.float(), ref_grad_w.float(), atol=atol * 20, rtol=rtol * 20)
    finally:
        if previous_disable is not None:
            os.environ["NAIME_DISABLE_FUSED_RMSNORM"] = previous_disable
