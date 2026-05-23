from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from naime_hybrid.modules.state_ops import state_softmax_matmul


def _bench_case(
    *,
    fused: bool,
    batch: int,
    tokens: int,
    slots: int,
    dim: int,
    dtype: torch.dtype,
    masked: bool,
    iters: int,
    warmup: int,
) -> dict[str, float | int | str | bool]:
    os.environ["NAIME_USE_FUSED_STATE_ATTENTION"] = "1" if fused else "0"
    device = torch.device("cuda")
    torch.manual_seed(1234 + slots + dim + int(fused))
    scores = torch.randn(batch, tokens, slots, device=device, dtype=dtype, requires_grad=True)
    values = torch.randn(batch, slots, dim, device=device, dtype=dtype, requires_grad=True)
    grad_context = torch.randn(batch, tokens, dim, device=device, dtype=dtype)
    grad_weights = torch.randn(batch, tokens, slots, device=device, dtype=dtype) * 0.01
    mask = None
    if masked:
        mask = torch.zeros(1, tokens, slots, device=device, dtype=torch.bool)
        mask[:, 0, :] = True
        mask[:, 3::4, slots // 2 :] = True

    def step() -> None:
        scores.grad = None
        values.grad = None
        context, weights = state_softmax_matmul(scores, values, mask=mask, zero_invalid=True)
        loss = (context.float() * grad_context.float()).mean() + (weights.float() * grad_weights.float()).mean()
        loss.backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "fused": fused,
        "batch": batch,
        "tokens": tokens,
        "slots": slots,
        "dim": dim,
        "dtype": str(dtype).replace("torch.", ""),
        "masked": masked,
        "iters": iters,
        "ms": elapsed * 1000.0 / iters,
        "peak_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark state softmax-matmul forward+backward.")
    p.add_argument("--out", type=Path, default=Path("experiments/profiles/state_attention_bench.json"))
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--dim", type=int, default=640)
    p.add_argument("--slots", type=int, nargs="+", default=[6, 64, 192, 204])
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    rows = []
    for slots in args.slots:
        for masked in (False, True):
            for fused in (False, True):
                rows.append(
                    _bench_case(
                        fused=fused,
                        batch=args.batch,
                        tokens=args.tokens,
                        slots=slots,
                        dim=args.dim,
                        dtype=torch.bfloat16,
                        masked=masked,
                        iters=args.iters,
                        warmup=args.warmup,
                    )
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
