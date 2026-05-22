from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from naime_hybrid.config import NAIMEStateMoEConfig  # noqa: E402
from naime_hybrid.models.decoder import NAIMEV6RecursiveSelfMoEDecoder  # noqa: E402
from naime_hybrid.training.losses import lm_loss  # noqa: E402


def _make_config(args: argparse.Namespace) -> NAIMEStateMoEConfig:
    return NAIMEStateMoEConfig(
        vocab_size=args.vocab,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        d_ff=args.d_ff,
        n_layers=args.layers,
        n_dense_layers=args.dense_layers,
        n_experts=args.experts,
        top_k=args.top_k,
        expert_hidden_dim=args.expert_hidden,
        stride=args.stride,
        window=args.window,
        z_dim=args.z_dim,
        world_state_slots=args.world_slots,
        self_state_slots=args.self_slots,
        causal_state_stride=args.causal_state_stride,
        moe_dispatch_mode=args.moe_dispatch,
        latent_thought_steps=args.latent_thought_steps,
        latent_thought_write_mode="state_only",
        latent_thought_hidden_scale=0.0,
        latent_field_coupling=False,
    )


def _loss_from_output(out: dict, labels: torch.Tensor | None, include_lm: bool) -> torch.Tensor:
    if include_lm:
        if labels is None or "logits" not in out:
            raise RuntimeError("include_lm requires labels and logits")
        return lm_loss(out["logits"], labels, backend="torch")

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


def _one_step(
    model: NAIMEV6RecursiveSelfMoEDecoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    include_lm: bool,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
        with record_function("naime_v6_forward"):
            out = model(
                input_ids,
                attention_mask=attention_mask,
                return_logits=include_lm,
                return_state=True,
            )
        with record_function("naime_v6_loss"):
            loss = _loss_from_output(out, labels, include_lm)
    with record_function("naime_v6_backward"):
        loss.backward()
    return loss


def _summarize_key_averages(prof, *, row_limit: int) -> list[dict[str, object]]:
    rows = []
    for evt in prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit).splitlines():
        rows.append(evt)
    return [{"table": "\n".join(rows)}]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Torch profiler for V6 CUDA op hotspots.")
    p.add_argument("--out-dir", type=Path, default=Path("experiments/profiles/v6_torch_ops"))
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--d-model", type=int, default=640)
    p.add_argument("--heads", type=int, default=10)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=2560)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dense-layers", type=int, default=0)
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
    p.add_argument("--moe-dispatch", choices=["auto", "dense", "sparse"], default="auto")
    p.add_argument("--include-lm", action="store_true")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--row-limit", type=int, default=40)
    p.add_argument("--trace", action="store_true", help="Export a Chrome trace. This can be large.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device("cuda")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision("high")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = _make_config(args)
    model = NAIMEV6RecursiveSelfMoEDecoder(config).to(device)
    model.train()
    input_ids = torch.randint(0, args.vocab, (args.batch, args.seq_len), device=device)
    attention_mask = torch.ones(args.batch, args.seq_len, dtype=torch.bool, device=device)
    labels = torch.randint(0, args.vocab, (args.batch, args.seq_len), device=device) if args.include_lm else None

    for _ in range(args.warmup):
        _one_step(model, input_ids, attention_mask, labels, dtype=dtype, include_lm=args.include_lm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.steps):
            _one_step(model, input_ids, attention_mask, labels, dtype=dtype, include_lm=args.include_lm)
            prof.step()
    torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.row_limit)
    (args.out_dir / "torch_ops_table.txt").write_text(table, encoding="utf-8")
    summary = {
        "args": vars(args) | {"out_dir": str(args.out_dir)},
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "table_path": str(args.out_dir / "torch_ops_table.txt"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.trace:
        prof.export_chrome_trace(str(args.out_dir / "trace.json"))

    print(table)
    print(f"PROFILE_OUT={args.out_dir}")
    print(f"PEAK_MEMORY_MB={summary['peak_memory_mb']:.1f}")


if __name__ == "__main__":
    main()
