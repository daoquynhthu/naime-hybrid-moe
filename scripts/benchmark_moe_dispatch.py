from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from naime_hybrid.modules.moe import TopKMoE  # noqa: E402


@dataclass
class SweepResult:
    experts: int
    top_k: int
    mode: str
    batch: int
    seq_len: int
    d_model: int
    expert_hidden: int
    backward: bool
    ok: bool
    mean_ms: float | None = None
    tokens_per_sec: float | None = None
    peak_memory_mb: float | None = None
    error: str | None = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _run_one(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    seq_len: int,
    d_model: int,
    experts: int,
    top_k: int,
    expert_hidden: int,
    mode: str,
    warmup: int,
    iters: int,
    backward: bool,
) -> SweepResult:
    hidden = torch.randn(batch, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True)
    semantic = torch.randn(batch, seq_len, d_model, device=device, dtype=torch.float32)
    model = TopKMoE(
        d_model=d_model,
        semantic_dim=d_model,
        n_experts=experts,
        top_k=top_k,
        expert_hidden_dim=expert_hidden,
        semantic_router_mode="hybrid",
        dispatch_mode=mode,
    ).to(device)
    model.train()
    amp = device.type == "cuda" and dtype != torch.float32

    def step() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        if hidden.grad is not None:
            hidden.grad = None
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            out, aux = model(hidden, semantic)
            loss = out.float().square().mean() + aux["load_balance"].float() * 0.01
        if backward:
            loss.backward()
        return loss

    try:
        _clear(device)
        for _ in range(warmup):
            step()
        _sync(device)
        start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        start_time = time.perf_counter()
        if start_event is not None:
            start_event.record()
        for _ in range(iters):
            step()
        if end_event is not None:
            end_event.record()
            _sync(device)
            elapsed_ms = float(start_event.elapsed_time(end_event))
        else:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        mean_ms = elapsed_ms / max(1, iters)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
        return SweepResult(
            experts=experts,
            top_k=top_k,
            mode=mode,
            batch=batch,
            seq_len=seq_len,
            d_model=d_model,
            expert_hidden=expert_hidden,
            backward=backward,
            ok=True,
            mean_ms=mean_ms,
            tokens_per_sec=(batch * seq_len * 1000.0) / mean_ms if mean_ms > 0 else math.inf,
            peak_memory_mb=peak_memory,
        )
    except Exception as exc:  # noqa: BLE001 - sweep should continue across failed points.
        return SweepResult(
            experts=experts,
            top_k=top_k,
            mode=mode,
            batch=batch,
            seq_len=seq_len,
            d_model=d_model,
            expert_hidden=expert_hidden,
            backward=backward,
            ok=False,
            error=repr(exc),
        )
    finally:
        _clear(device)


def _print(results: list[SweepResult]) -> None:
    headers = ["E", "top_k", "mode", "ms", "tok/s", "peak_mb", "ok", "error"]
    rows = []
    for result in results:
        rows.append(
            [
                result.experts,
                result.top_k,
                result.mode,
                "" if result.mean_ms is None else f"{result.mean_ms:.3f}",
                "" if result.tokens_per_sec is None else f"{result.tokens_per_sec:.0f}",
                "" if result.peak_memory_mb is None else f"{result.peak_memory_mb:.1f}",
                result.ok,
                result.error or "",
            ]
        )
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows, strict=False)]
    print(" | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep dense vs sparse TopKMoE dispatch performance.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--batch", type=int, default=29)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--expert-hidden", type=int, default=768)
    parser.add_argument("--experts", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--top-k", type=int, nargs="+", default=[2])
    parser.add_argument("--modes", nargs="+", default=["dense", "sparse"], choices=["dense", "sparse", "auto"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)
        torch.set_float32_matmul_precision("high")

    results: list[SweepResult] = []
    for experts in args.experts:
        for top_k in args.top_k:
            if top_k > experts:
                continue
            for mode in args.modes:
                results.append(
                    _run_one(
                        device=device,
                        dtype=dtype,
                        batch=args.batch,
                        seq_len=args.seq_len,
                        d_model=args.d_model,
                        experts=experts,
                        top_k=top_k,
                        expert_hidden=args.expert_hidden,
                        mode=mode,
                        warmup=args.warmup,
                        iters=args.iters,
                        backward=not args.forward_only,
                    )
                )
    _print(results)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))


if __name__ == "__main__":
    main()
