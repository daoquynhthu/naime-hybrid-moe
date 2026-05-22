from __future__ import annotations

import os

import torch

from naime_hybrid.config import NAIMEStateMoEConfig
from naime_hybrid.models.decoder import NAIMEV6RecursiveSelfMoEDecoder
from naime_hybrid.training.losses import lm_loss


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    config = NAIMEStateMoEConfig(
        vocab_size=4096,
        max_seq_len=256,
        d_model=256,
        n_heads=8,
        n_kv_heads=2,
        d_ff=1024,
        n_layers=2,
        n_dense_layers=0,
        n_experts=4,
        top_k=2,
        expert_hidden_dim=512,
        stride=16,
        window=24,
        z_dim=64,
        world_state_slots=6,
        self_state_slots=6,
        causal_state_stride=128,
        latent_thought_steps=1,
        latent_thought_write_mode="state_only",
        latent_thought_hidden_scale=0.0,
        latent_field_coupling=True,
        moe_dispatch_mode="auto",
    )
    model = NAIMEV6RecursiveSelfMoEDecoder(config).cuda().train()
    input_ids = torch.randint(0, config.vocab_size, (3, 256), device="cuda")
    labels = torch.randint(0, config.vocab_size, (3, 256), device="cuda")
    mask = torch.ones_like(input_ids, dtype=torch.bool)

    def run(force_fp32: bool) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
        os.environ["NAIME_FORCE_FP32_STATE_ATTENTION"] = "1" if force_fp32 else "0"
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids, attention_mask=mask, return_logits=True, return_state=True)
            loss = lm_loss(out["logits"], labels, backend="torch")
        loss.backward()
        max_grad = max((p.grad.float().abs().max().item() for p in model.parameters() if p.grad is not None), default=0.0)
        finite = torch.isfinite(loss).item() and all(
            torch.isfinite(value).all().item() for value in (out["hidden_states"], out["world_state"], out["self_state"])
        )
        return loss.detach(), out["hidden_states"].detach(), max_grad, finite

    loss_fp32, hidden_fp32, grad_fp32, finite_fp32 = run(True)
    loss_native, hidden_native, grad_native, finite_native = run(False)
    hidden_max_abs = (hidden_native.float() - hidden_fp32.float()).abs().max().item()
    print(
        {
            "loss_fp32": float(loss_fp32),
            "loss_native": float(loss_native),
            "hidden_max_abs": hidden_max_abs,
            "grad_fp32": grad_fp32,
            "grad_native": grad_native,
            "finite_fp32": finite_fp32,
            "finite_native": finite_native,
        }
    )
    assert finite_fp32 and finite_native
    assert abs(float(loss_native - loss_fp32)) < 0.05

    os.environ["NAIME_FORCE_FP32_STATE_ATTENTION"] = "0"
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(20):
        opt.zero_grad(set_to_none=True)
        ids = torch.randint(0, config.vocab_size, (3, 256), device="cuda")
        targets = torch.randint(0, config.vocab_size, (3, 256), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool), return_logits=True, return_state=True)
            loss = lm_loss(out["logits"], targets, backend="torch")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        assert torch.isfinite(loss).item(), step
    print("NATIVE_V6_MINI_TRAIN_OK")


if __name__ == "__main__":
    main()
