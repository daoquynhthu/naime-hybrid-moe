import pytest
import torch

from naime_hybrid.modules.state_ops import state_softmax_matmul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native state attention")
def test_state_softmax_matmul_zeroes_all_masked_rows():
    scores = torch.randn(2, 4, 6, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    values = torch.randn(2, 6, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.zeros(1, 4, 6, device="cuda", dtype=torch.bool)
    mask[:, 0, :] = True

    context, weights = state_softmax_matmul(scores, values, mask=mask, zero_invalid=True, out_dtype=scores.dtype)
    assert torch.isfinite(context).all()
    assert torch.isfinite(weights).all()
    assert torch.equal(context[:, 0, :], torch.zeros_like(context[:, 0, :]))
    assert torch.equal(weights[:, 0, :], torch.zeros_like(weights[:, 0, :]))

    context.square().float().mean().backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.isfinite(values.grad).all()
