from __future__ import annotations

import argparse
import gc
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

from naime_hybrid.config import NAIMEStateMoEConfig
from naime_hybrid.models.decoder import NAIMEV6RecursiveSelfMoEDecoder
from naime_hybrid.modules.moe import TopKMoE
from naime_hybrid.training.losses import fused_lm_loss, lm_loss


@dataclass
class BenchResult:
    name: str
    ok: bool
    mean_ms: float | None = None
    tokens_per_sec: float | None = None
    max_memory_mb: float | None = None
    error: str | None = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _amp_enabled(args: argparse.Namespace, device: torch.device) -> bool:
    return bool(device.type == "cuda" and args.dtype_torch != torch.float32)


def _timed(
    name: str,
    fn,
    *,
    device: torch.device,
    tokens: int,
    warmup: int,
    iters: int,
    backward: bool,
) -> BenchResult:
    try:
        _clear(device)
        for _ in range(warmup):
            loss = fn()
            if backward:
                loss.backward()
        _sync(device)

        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        start_time = time.perf_counter()
        if start is not None:
            start.record()
        for _ in range(iters):
            loss = fn()
            if backward:
                loss.backward()
        if end is not None:
            end.record()
            _sync(device)
            elapsed_ms = float(start.elapsed_time(end))
        else:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        mean_ms = elapsed_ms / max(1, iters)
        peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
        return BenchResult(
            name=name,
            ok=True,
            mean_ms=mean_ms,
            tokens_per_sec=(tokens * 1000.0) / mean_ms if mean_ms > 0 else math.inf,
            max_memory_mb=peak,
        )
    except Exception as exc:  # noqa: BLE001 - profiler should keep going across optional backends.
        return BenchResult(name=name, ok=False, error=repr(exc))
    finally:
        _clear(device)


def _bench_moe(args: argparse.Namespace, device: torch.device) -> list[BenchResult]:
    results: list[BenchResult] = []
    batch, seq, d_model = args.batch, args.seq_len, args.d_model
    semantic_dim = d_model
    hidden = torch.randn(batch, seq, d_model, device=device, dtype=torch.float32, requires_grad=True)
    semantic = torch.randn(batch, seq, semantic_dim, device=device, dtype=torch.float32)

    for mode in args.moe_modes:
        moe = TopKMoE(
            d_model=d_model,
            semantic_dim=semantic_dim,
            n_experts=args.experts,
            top_k=args.top_k,
            expert_hidden_dim=args.expert_hidden,
            semantic_router_mode="hybrid",
            dispatch_mode=mode,
        ).to(device=device)
        moe.train()

        def step() -> torch.Tensor:
            moe.zero_grad(set_to_none=True)
            if hidden.grad is not None:
                hidden.grad = None
            with torch.autocast(device_type=device.type, dtype=args.dtype_torch, enabled=_amp_enabled(args, device)):
                out, aux = moe(hidden, semantic)
            return out.float().square().mean() + aux["load_balance"].float() * 0.01

        results.append(
            _timed(
                f"moe_{mode}",
                step,
                device=device,
                tokens=batch * seq,
                warmup=args.warmup,
                iters=args.iters,
                backward=not args.forward_only,
            )
        )
    return results


def _bench_lm(args: argparse.Namespace, device: torch.device) -> list[BenchResult]:
    results: list[BenchResult] = []
    hidden = torch.randn(args.batch, args.seq_len, args.d_model, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(args.vocab, args.d_model, device=device, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, args.vocab, (args.batch, args.seq_len), device=device)
    if labels.numel() > 0:
        labels.reshape(-1)[0] = -100

    def torch_logits_step() -> torch.Tensor:
        if hidden.grad is not None:
            hidden.grad = None
        if weight.grad is not None:
            weight.grad = None
        with torch.autocast(device_type=device.type, dtype=args.dtype_torch, enabled=_amp_enabled(args, device)):
            logits = hidden.matmul(weight.t())
        return lm_loss(logits, labels, backend="torch")

    results.append(
        _timed(
            "lm_torch_logits",
            torch_logits_step,
            device=device,
            tokens=args.batch * args.seq_len,
            warmup=args.warmup,
            iters=args.iters,
            backward=not args.forward_only,
        )
    )

    if device.type == "cuda":
        def cuda_ext_step() -> torch.Tensor:
            if hidden.grad is not None:
                hidden.grad = None
            if weight.grad is not None:
                weight.grad = None
            with torch.autocast(device_type=device.type, dtype=args.dtype_torch, enabled=_amp_enabled(args, device)):
                return fused_lm_loss(hidden, weight, labels, backend="cuda_ext_fused_ce")

        results.append(
            _timed(
                "lm_cuda_ext_fused_ce",
                cuda_ext_step,
                device=device,
                tokens=args.batch * args.seq_len,
                warmup=max(0, min(args.warmup, 1)),
                iters=max(1, min(args.iters, 3)),
                backward=not args.forward_only,
            )
        )
    return results


def _bench_v6_decoder(args: argparse.Namespace, device: torch.device) -> list[BenchResult]:
    config = NAIMEStateMoEConfig(
        vocab_size=args.vocab,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.heads,
        n_kv_heads=max(1, min(args.kv_heads, args.heads)),
        d_ff=args.d_ff,
        n_layers=1,
        n_dense_layers=0,
        n_experts=args.experts,
        top_k=args.top_k,
        expert_hidden_dim=args.expert_hidden,
        stride=args.stride,
        window=args.window,
        z_dim=args.z_dim,
        world_state_slots=args.world_slots,
        self_state_slots=args.self_slots,
        causal_state_stride=args.causal_state_stride,
        moe_dispatch_mode=args.block_moe_dispatch,
        latent_thought_steps=args.latent_thought_steps,
        latent_thought_write_mode="state_only",
        latent_thought_hidden_scale=0.0,
    )
    model = NAIMEV6RecursiveSelfMoEDecoder(config).to(device=device)
    model.train()
    input_ids = torch.randint(0, args.vocab, (args.batch, args.seq_len), device=device)
    mask = torch.ones(args.batch, args.seq_len, device=device, dtype=torch.bool)

    def step() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=args.dtype_torch, enabled=_amp_enabled(args, device)):
            out = model(
                input_ids,
                attention_mask=mask,
                return_logits=False,
                return_state=True,
            )
        loss = out["hidden_states"].float().square().mean()
        for aux in out.get("aux", []):
            for group in aux.values():
                if isinstance(group, dict):
                    for value in group.values():
                        if isinstance(value, torch.Tensor) and value.ndim == 0:
                            loss = loss + value.float() * 0.0
        for name in ("world_state", "self_state", "memory"):
            value = out.get(name)
            if isinstance(value, torch.Tensor):
                loss = loss + value.float().square().mean() * 0.0
        return loss

    return [
        _timed(
            f"v6_decoder_{args.block_moe_dispatch}",
            step,
            device=device,
            tokens=args.batch * args.seq_len,
            warmup=args.warmup,
            iters=args.iters,
            backward=not args.forward_only,
        )
    ]


def _print_table(results: list[BenchResult]) -> None:
    rows = []
    for r in results:
        rows.append(
            [
                r.name,
                "ok" if r.ok else "fail",
                "" if r.mean_ms is None else f"{r.mean_ms:.3f}",
                "" if r.tokens_per_sec is None else f"{r.tokens_per_sec:.0f}",
                "" if r.max_memory_mb is None else f"{r.max_memory_mb:.1f}",
                r.error or "",
            ]
        )
    headers = ["name", "status", "ms/iter", "tok/s", "peak_mb", "error"]
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows, strict=False)]
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile NAIME CUDA hotspots with compact output.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=640)
    p.add_argument("--heads", type=int, default=10)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=2560)
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--experts", type=int, default=6)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--expert-hidden", type=int, default=1280)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--z-dim", type=int, default=160)
    p.add_argument("--world-slots", type=int, default=6)
    p.add_argument("--self-slots", type=int, default=6)
    p.add_argument("--causal-state-stride", type=int, default=512)
    p.add_argument("--latent-thought-steps", type=int, default=1)
    p.add_argument("--moe-modes", nargs="+", default=["dense", "sparse", "auto"], choices=["dense", "sparse", "auto"])
    p.add_argument("--block-moe-dispatch", default="dense", choices=["dense", "sparse", "auto"])
    p.add_argument("--skip-moe", action="store_true")
    p.add_argument("--skip-lm", action="store_true")
    p.add_argument(
        "--include-block",
        action="store_true",
        help="Also profile a one-layer V6 decoder. This is heavier and disabled by default.",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--forward-only", action="store_true", help="Profile forward only; useful for quick smoke checks.")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()
    args.dtype_torch = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    return args


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)
        torch.set_float32_matmul_precision("high")

    results: list[BenchResult] = []
    if not args.skip_moe:
        results.extend(_bench_moe(args, device))
    if not args.skip_lm:
        results.extend(_bench_lm(args, device))
    if args.include_block:
        results.extend(_bench_v6_decoder(args, device))

    _print_table(results)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
