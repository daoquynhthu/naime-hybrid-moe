import os

import pytest
import torch

from naime_hybrid.modules.state_ops import state_softmax_matmul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fused state attention")
@pytest.mark.parametrize("masked", [False, True])
def test_fused_state_attention_matches_torch_reference(masked: bool):
    previous_disable = os.environ.pop("NAIME_USE_FUSED_STATE_ATTENTION", None)
    previous_force = os.environ.pop("NAIME_FORCE_FP32_STATE_ATTENTION", None)
    try:
        torch.manual_seed(11)
        scores = torch.randn(2, 13, 9, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        values = torch.randn(2, 9, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        mask = None
        if masked:
            mask = torch.zeros(1, 13, 9, device="cuda", dtype=torch.bool)
            mask[:, 0, :] = True
            mask[:, 3::4, 5:] = True
        grad_context = torch.randn(2, 13, 32, device="cuda", dtype=torch.bfloat16)
        grad_weights = torch.randn(2, 13, 9, device="cuda", dtype=torch.bfloat16) * 0.01

        os.environ["NAIME_USE_FUSED_STATE_ATTENTION"] = "0"
        ref_context, ref_weights = state_softmax_matmul(scores, values, mask=mask, zero_invalid=True)
        ref_loss = (ref_context.float() * grad_context.float()).mean() + (ref_weights.float() * grad_weights.float()).mean()
        ref_loss.backward(retain_graph=True)
        ref_grad_scores = scores.grad.detach().clone()
        ref_grad_values = values.grad.detach().clone()
        scores.grad = None
        values.grad = None

        os.environ["NAIME_USE_FUSED_STATE_ATTENTION"] = "1"
        context, weights = state_softmax_matmul(scores, values, mask=mask, zero_invalid=True)
        loss = (context.float() * grad_context.float()).mean() + (weights.float() * grad_weights.float()).mean()
        loss.backward()

        assert torch.allclose(context.float(), ref_context.float(), atol=7e-3, rtol=7e-3)
        assert torch.allclose(weights.float(), ref_weights.float(), atol=7e-3, rtol=7e-3)
        assert torch.allclose(scores.grad.float(), ref_grad_scores.float(), atol=8e-3, rtol=8e-3)
        assert torch.allclose(values.grad.float(), ref_grad_values.float(), atol=8e-3, rtol=8e-3)
    finally:
        if previous_disable is not None:
            os.environ["NAIME_USE_FUSED_STATE_ATTENTION"] = previous_disable
        if previous_force is not None:
            os.environ["NAIME_FORCE_FP32_STATE_ATTENTION"] = previous_force
