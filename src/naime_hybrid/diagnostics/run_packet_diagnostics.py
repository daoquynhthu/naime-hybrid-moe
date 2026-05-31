from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from naime_hybrid.config import NAIMEStateMoEConfig
from naime_hybrid.data import ByteTextDataset, HFDiskCausalDataset
from naime_hybrid.models import build_model
from naime_hybrid.training.train import resolve_device

from .packet_diagnostics import run_state_packet_diagnostics
from .trace_config import TraceConfig
from .trace_context import TraceContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline NAIME state-packet data-flow diagnostics.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing config.json.")
    parser.add_argument("--checkpoint", default="models/model_best.pt")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-format", default=None, choices=[None, "byte", "hf_disk", "auto"])
    parser.add_argument("--data-split", default="validation")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--chunk-len", type=int, default=None)
    parser.add_argument("--boundary-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--record-full-tensors", action="store_true")
    return parser.parse_args()


def _resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.is_absolute():
        return path
    return run_dir / path


def _build_dataset(data_path: Path, data_format: str, split: str, seq_len: int):
    if data_format == "auto":
        data_format = "hf_disk" if data_path.is_dir() and (data_path / "dataset_dict.json").exists() else "byte"
    if data_format == "hf_disk":
        return HFDiskCausalDataset(data_path, split=split, seq_len=seq_len)
    return ByteTextDataset.from_file(data_path, seq_len=seq_len)


def _load_state_dict(checkpoint: dict) -> dict:
    if "model" in checkpoint:
        return checkpoint["model"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json in {run_dir}")

    train_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = NAIMEStateMoEConfig(**train_config["model"])
    data_path = Path(args.data_path or train_config["data_path"])
    data_format = args.data_format or train_config.get("data_format", "auto")
    chunk_len = args.chunk_len or max(1, model_config.max_seq_len // 2)

    device = resolve_device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"
    model = build_model(train_config["architecture"], model_config).to(device)
    checkpoint_path = _resolve_checkpoint(run_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_load_state_dict(checkpoint), strict=True)
    model.eval()

    dataset = _build_dataset(data_path, data_format, args.data_split, model_config.max_seq_len)
    collate_fn = None
    if isinstance(dataset, HFDiskCausalDataset):
        collate_fn = partial(HFDiskCausalDataset.causal_collate, seq_len=model_config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    selected = None
    for batch_idx, batch in enumerate(loader):
        if batch_idx == args.batch_index:
            selected = batch
            break
    if selected is None:
        raise RuntimeError(f"batch-index {args.batch_index} not available")

    input_ids = selected["input_ids"].to(device, non_blocking=True)
    labels = selected["labels"].to(device, non_blocking=True)
    attention_mask = selected.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device, non_blocking=True)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "diagnostics" / "packet_flow"
    trace_context = TraceContext(
        config=TraceConfig(
            enabled=True,
            output_dir=str(output_dir),
            record_full_tensors=args.record_full_tensors,
            tags={
                "architecture": str(train_config["architecture"]),
                "checkpoint": str(checkpoint_path),
                "split": str(args.data_split),
            },
        )
    )
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        report = run_state_packet_diagnostics(
            model,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            chunk_len=chunk_len,
            boundary_tokens=args.boundary_tokens,
            trace_context=trace_context,
            output_dir=str(output_dir),
        )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
        "split": args.data_split,
        "batch_index": args.batch_index,
        "chunk_len": chunk_len,
        "boundary_tokens": args.boundary_tokens,
        "output_dir": str(output_dir),
        "metrics": report["metrics"],
        "summary": report["summary"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
