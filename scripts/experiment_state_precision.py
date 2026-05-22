from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _finite(*tensors: torch.Tensor | None) -> bool:
    return all(t is None or torch.isfinite(t).all().item() for t in tensors)


def _masked_softmax_matmul(
    scores: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    fp32_path: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fp32_path:
        scores_work = scores.float()
        if mask is not None:
            scores_work = scores_work.masked_fill(mask, torch.finfo(torch.float32).min)
        weights = torch.softmax(scores_work, dim=-1)
        if mask is not None:
            valid = (~mask).any(dim=-1, keepdim=True)
            weights = torch.where(valid, weights, torch.zeros_like(weights))
        context = torch.matmul(weights, values.float()).to(dtype=scores.dtype)
        return context, weights

    scores_work = scores
    if mask is not None:
        scores_work = scores_work.masked_fill(mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores_work, dim=-1)
    if mask is not None:
        valid = (~mask).any(dim=-1, keepdim=True)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
    context = torch.matmul(weights, values)
    return context, weights


def _one_case(
    *,
    name: str,
    dtype: torch.dtype,
    batch: int,
    tokens: int,
    slots: int,
    dim: int,
    scale: float,
    masked: bool,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(abs(hash((name, str(dtype), scale, masked))) % (2**31))
    query = torch.randn(batch, tokens, dim, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(batch, slots, dim, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(batch, slots, dim, device=device, dtype=dtype, requires_grad=True)
    scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(dim)
    scores = scores * scale
    mask = None
    if masked:
        token_idx = torch.arange(tokens, device=device)
        limits = ((token_idx // max(1, tokens // 16)) * max(1, slots // 16)).clamp(max=slots)
        slot_idx = torch.arange(slots, device=device)
        mask = slot_idx.view(1, 1, slots) >= limits.view(1, tokens, 1)

    ctx_ref, weights_ref = _masked_softmax_matmul(scores, value, mask, fp32_path=True)
    grad = torch.randn_like(ctx_ref)
    ref_loss = (ctx_ref.float() * grad.float()).mean() + weights_ref.float().square().mean() * 1e-4
    ref_loss.backward(retain_graph=True)
    ref_grads = [query.grad.detach().clone(), key.grad.detach().clone(), value.grad.detach().clone()]
    ref_finite = _finite(ctx_ref, weights_ref, ref_loss, *ref_grads)

    query.grad = None
    key.grad = None
    value.grad = None
    ctx_native, weights_native = _masked_softmax_matmul(scores, value, mask, fp32_path=False)
    native_loss = (ctx_native.float() * grad.float()).mean() + weights_native.float().square().mean() * 1e-4
    native_loss.backward()
    native_grads = [query.grad, key.grad, value.grad]
    native_finite = _finite(ctx_native, weights_native, native_loss, *native_grads)

    ctx_diff = (ctx_native.float() - ctx_ref.float()).abs()
    w_diff = (weights_native.float() - weights_ref.float()).abs()
    grad_diffs = [(ng.float() - rg.float()).abs().max().item() for ng, rg in zip(native_grads, ref_grads, strict=True)]
    return {
        "name": name,
        "dtype": str(dtype).replace("torch.", ""),
        "batch": batch,
        "tokens": tokens,
        "slots": slots,
        "dim": dim,
        "scale": scale,
        "masked": masked,
        "ref_finite": ref_finite,
        "native_finite": native_finite,
        "ctx_max_abs": ctx_diff.max().item(),
        "ctx_mean_abs": ctx_diff.mean().item(),
        "weights_max_abs": w_diff.max().item(),
        "weights_mean_abs": w_diff.mean().item(),
        "grad_max_abs": max(grad_diffs),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stress-test fp32 vs native precision state softmax paths.")
    p.add_argument("--out", type=Path, default=Path("experiments/profiles/state_precision.json"))
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--dim", type=int, default=640)
    p.add_argument("--world-slots", type=int, default=6)
    p.add_argument("--history-slots", type=int, default=192)
    p.add_argument("--self-slots", type=int, default=6)
    p.add_argument("--latent-slots", type=int, default=204)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    rows: list[dict[str, object]] = []
    specs = [
        ("world_read", args.world_slots, False),
        ("world_history", args.history_slots, True),
        ("self_slot_context", args.tokens // 16, False),
        ("latent_field", args.latent_slots, True),
    ]
    for dtype in (torch.bfloat16, torch.float16):
        for name, slots, masked in specs:
            for scale in (1.0, 4.0, 16.0, 64.0):
                rows.append(
                    _one_case(
                        name=name,
                        dtype=dtype,
                        batch=args.batch,
                        tokens=args.tokens,
                        slots=slots,
                        dim=args.dim,
                        scale=scale,
                        masked=masked,
                        device=device,
                    )
                )
                torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    failures = [r for r in rows if not r["native_finite"]]
    unstable = [r for r in rows if r["ctx_max_abs"] > 0.25 or r["weights_max_abs"] > 0.1 or r["grad_max_abs"] > 0.5]
    print(json.dumps({"cases": len(rows), "native_nan_cases": len(failures), "large_diff_cases": len(unstable)}, indent=2))
    for row in failures + unstable[:10]:
        print(json.dumps(row, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
