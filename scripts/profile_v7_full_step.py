from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from naime_hybrid.config import NAIMEStateMoEConfig  # noqa: E402
from naime_hybrid.models.factory import build_model  # noqa: E402
from naime_hybrid.training.losses import lm_loss  # noqa: E402


def _template_params(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    params = payload.get("params", payload)
    return dict(params)


def _get(params: dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key, default)


def _make_config(params: dict[str, Any], args: argparse.Namespace) -> NAIMEStateMoEConfig:
    return NAIMEStateMoEConfig(
        vocab_size=args.vocab,
        max_seq_len=args.seq_len,
        d_model=int(_get(params, "DModel", 384)),
        n_layers=int(_get(params, "Layers", 8)),
        n_dense_layers=int(_get(params, "DenseLayers", 2)),
        n_heads=int(_get(params, "Heads", 6)),
        n_kv_heads=int(_get(params, "KvHeads", 2)),
        d_ff=int(_get(params, "Dff", 1536)),
        dropout=float(_get(params, "Dropout", 0.0)),
        stride=int(_get(params, "Stride", 16)),
        window=int(_get(params, "Window", 24)),
        z_dim=int(_get(params, "ZDim", 96)),
        causal_state_stride=int(_get(params, "CausalStateStride", 512)),
        n_experts=int(_get(params, "Experts", 4)),
        top_k=int(_get(params, "TopK", 2)),
        expert_hidden_dim=int(_get(params, "ExpertHidden", 768)),
        moe_dispatch_mode=str(_get(params, "MoeDispatchMode", "auto")),
        world_state_slots=int(_get(params, "WorldStateSlots", 4)),
        self_state_slots=int(_get(params, "SelfStateSlots", 4)),
        latent_thought_steps=int(_get(params, "LatentThoughtSteps", 1)),
        latent_thought_write_mode=str(_get(params, "LatentThoughtWriteMode", "final_hidden")),
        latent_thought_hidden_scale=float(_get(params, "LatentThoughtHiddenScale", 0.02)),
        latent_field_coupling=bool(_get(params, "LatentFieldCoupling", True)),
        latent_field_token_scale=float(_get(params, "LatentFieldTokenScale", 0.02)),
        latent_field_max_ratio=float(_get(params, "LatentFieldMaxRatio", 0.05)),
        v7_dynamics_steps=int(_get(params, "V7DynamicsSteps", 1)),
        v7_latent_slots=int(_get(params, "V7LatentSlots", 4)),
        v7_latent_write_scale=float(_get(params, "V7LatentWriteScale", 0.03)),
        v7_hidden_write_scale=float(_get(params, "V7HiddenWriteScale", 0.05)),
        v7_max_hidden_write_ratio=float(_get(params, "V7MaxHiddenWriteRatio", 0.10)),
        v7_state_write_scale=float(_get(params, "V7StateWriteScale", 0.02)),
        semantic_state_write_scale=float(_get(params, "SemanticStateWriteScale", 0.075)),
        semantic_memory_hidden_scale=float(_get(params, "SemanticMemoryHiddenScale", 0.035)),
        world_router_max_ratio=float(_get(params, "WorldRouterMaxRatio", 0.08)),
        semantic_gate_mixer_max_state_weight=float(_get(params, "SemanticGateMixerMaxStateWeight", 0.35)),
        self_state_world_gate_min=float(_get(params, "SelfStateWorldGateMin", 0.10)),
        self_state_world_gate_scale=float(_get(params, "SelfStateWorldGateScale", 1.0)),
    )


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _event_metric(evt: Any, *names: str, default: float = 0.0) -> float:
    for name in names:
        value = getattr(evt, name, None)
        if value is not None:
            return float(value)
    return default


def _top_events(prof: torch.profiler.profile, *, sort_by: str, limit: int) -> list[dict[str, Any]]:
    rows = []
    for evt in prof.key_averages():
        self_device_us = _event_metric(evt, "self_device_time_total", "self_cuda_time_total")
        device_us = _event_metric(evt, "device_time_total", "cuda_time_total")
        rows.append(
            {
                "key": evt.key,
                "count": evt.count,
                "self_cpu_time_total_us": evt.self_cpu_time_total,
                "cpu_time_total_us": evt.cpu_time_total,
                "self_cuda_time_total_us": self_device_us,
                "cuda_time_total_us": device_us,
                "self_cuda_memory_usage": getattr(evt, "self_cuda_memory_usage", 0),
                "cuda_memory_usage": getattr(evt, "cuda_memory_usage", 0),
            }
        )
    return sorted(rows, key=lambda row: float(row.get(sort_by, 0.0)), reverse=True)[:limit]


def _make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(1, args.vocab, (args.batch, args.seq_len), device=device)
    attention_mask = torch.ones(args.batch, args.seq_len, dtype=torch.bool, device=device)
    labels = input_ids.roll(shifts=-1, dims=1)
    labels[:, -1] = -100
    return input_ids, attention_mask, labels


def _train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    amp_dtype: torch.dtype,
    lm_loss_backend: str,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
        with record_function("naime_v7_forward"):
            out = model(input_ids, attention_mask=attention_mask, return_logits=True, return_state=False)
        with record_function("naime_v7_lm_loss"):
            loss = lm_loss(out["logits"], labels, backend=lm_loss_backend)
    with record_function("naime_v7_backward"):
        loss.backward()
    with record_function("naime_v7_optimizer_step"):
        optimizer.step()
    return loss.detach()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile a complete V7 training step with torch.profiler.")
    p.add_argument(
        "--template", type=Path, default=ROOT / "configs" / "training_templates" / "v7_remote_64m_probe.json"
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "profiles" / "v7_full_step")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--row-limit", type=int, default=60)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-backend", default="inductor")
    p.add_argument("--lm-loss-backend", choices=["torch", "triton_ce", "cuda_ext_ce"], default="torch")
    p.add_argument("--use-fused-state-attention", action="store_true")
    p.add_argument("--quiet", action="store_true", help="Write profiler tables to disk but only print the summary.")
    p.add_argument("--trace", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this profiler.")
    device = torch.device("cuda")
    amp_dtype = _dtype(args.dtype)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision("high")
    if args.use_fused_state_attention:
        os.environ["NAIME_USE_FUSED_STATE_ATTENTION"] = "1"

    params = _template_params(args.template)
    config = _make_config(params, args)
    model = build_model("naime_v7_typed_dynamics", config).to(device)
    model.train()
    if args.compile:
        model = torch.compile(model, backend=args.compile_backend)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    input_ids, attention_mask, labels = _make_batch(args, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(args.warmup):
        _train_step(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            amp_dtype=amp_dtype,
            lm_loss_backend=args.lm_loss_backend,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
    ) as prof:
        for _ in range(args.steps):
            _train_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                amp_dtype=amp_dtype,
                lm_loss_backend=args.lm_loss_backend,
            )
            prof.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    cuda_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.row_limit)
    self_cuda_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.row_limit)
    cpu_table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.row_limit)
    (args.out_dir / "cuda_time_total.txt").write_text(cuda_table, encoding="utf-8")
    (args.out_dir / "self_cuda_time_total.txt").write_text(self_cuda_table, encoding="utf-8")
    (args.out_dir / "self_cpu_time_total.txt").write_text(cpu_table, encoding="utf-8")
    if args.trace:
        prof.export_chrome_trace(str(args.out_dir / "trace.json"))

    summary = {
        "args": {**vars(args), "template": str(args.template), "out_dir": str(args.out_dir)},
        "config": config.__dict__,
        "tokens_per_step": args.batch * args.seq_len,
        "profiled_steps": args.steps,
        "elapsed_s": elapsed,
        "profiled_tokens_per_second": (args.batch * args.seq_len * args.steps) / elapsed,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "top_cuda_time_total": _top_events(prof, sort_by="cuda_time_total_us", limit=25),
        "top_self_cuda_time_total": _top_events(prof, sort_by="self_cuda_time_total_us", limit=25),
        "top_self_cpu_time_total": _top_events(prof, sort_by="self_cpu_time_total_us", limit=25),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if not args.quiet:
        print(cuda_table)
    print(f"PROFILE_OUT={args.out_dir}")
    print(f"PROFILED_TOKENS_PER_SECOND={summary['profiled_tokens_per_second']:.0f}")
    print(f"PEAK_MEMORY_MB={summary['peak_memory_mb']:.1f}")


if __name__ == "__main__":
    main()
