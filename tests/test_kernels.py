import pytest
import torch
import torch.nn.functional as F

from naime_hybrid.training.losses import IGNORE_INDEX, fused_lm_loss, lm_loss


def test_lm_loss_torch_backend_matches_cross_entropy():
    logits = torch.randn(3, 5, 11, requires_grad=True)
    labels = torch.randint(0, 11, (3, 5))
    labels[0, 0] = IGNORE_INDEX

    actual = lm_loss(logits, labels, backend="torch")
    expected = F.cross_entropy(logits.reshape(-1, 11).float(), labels.reshape(-1), ignore_index=IGNORE_INDEX)

    assert torch.allclose(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel validation")
def test_triton_cross_entropy_matches_torch_forward_and_backward():
    torch.manual_seed(1234)
    logits = torch.randn(4, 7, 257, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 257, (4, 7), device="cuda")
    labels[1, 2] = IGNORE_INDEX

    ref_logits = logits.detach().clone().requires_grad_(True)
    actual = lm_loss(logits, labels, backend="triton_ce")
    expected = lm_loss(ref_logits, labels, backend="torch")

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    actual.backward()
    expected.backward()

    assert torch.allclose(logits.grad, ref_logits.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for backend validation")
def test_auto_cross_entropy_falls_back_for_large_vocab():
    logits = torch.randn(2, 3, 70000, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 70000, (2, 3), device="cuda")

    loss = lm_loss(logits, labels, backend="auto")
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_fused_lm_loss_torch_backend_matches_explicit_logits():
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    weight = torch.randn(17, 8, requires_grad=True)
    labels = torch.randint(0, 17, (2, 3))
    labels[0, 1] = IGNORE_INDEX

    actual = fused_lm_loss(hidden, weight, labels, backend="torch")
    expected = lm_loss(hidden.matmul(weight.t()), labels, backend="torch")

    assert torch.allclose(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native CUDA extension validation")
def test_cuda_ext_fused_lm_ce_matches_torch_forward_and_backward():
    torch.manual_seed(5678)
    hidden = torch.randn(3, 4, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(31, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 31, (3, 4), device="cuda")
    labels[2, 1] = IGNORE_INDEX

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)

    actual = fused_lm_loss(hidden, weight, labels, backend="cuda_ext_fused_ce")
    expected = fused_lm_loss(ref_hidden, ref_weight, labels, backend="torch")

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)

    actual.backward()
    expected.backward()

    assert torch.allclose(hidden.grad, ref_hidden.grad, atol=5e-5, rtol=5e-5)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=5e-5, rtol=5e-5)
