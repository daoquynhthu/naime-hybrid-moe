import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from naime_hybrid.config import NAIMEStateMoEConfig
from naime_hybrid.data import ByteTextDataset, HFDiskCausalDataset
from naime_hybrid.models import build_model

from .losses import collect_aux_losses, lm_loss
from .train import effective_kl_lambda, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a NAIME Hybrid training run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-format", default=None, choices=[None, "byte", "hf_disk", "auto"])
    parser.add_argument("--data-split", default="validation")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means evaluate the full split.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def build_eval_dataset(data_path: Path, data_format: str, split: str, seq_len: int):
    if data_format == "auto":
        data_format = "hf_disk" if data_path.is_dir() and (data_path / "dataset_dict.json").exists() else "byte"
    if data_format == "hf_disk":
        return HFDiskCausalDataset(data_path, split=split, seq_len=seq_len)
    return ByteTextDataset.from_file(data_path, seq_len=seq_len)


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
    batch_size = args.batch_size or train_config["batch_size"]

    device = resolve_device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"

    model = build_model(train_config["architecture"], model_config).to(device)
    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    checkpoint_step = int(checkpoint.get("step", 0))
    checkpoint_metrics = checkpoint.get("metrics", {})
    model.eval()

    dataset = build_eval_dataset(data_path, data_format, args.data_split, model_config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    totals = {
        key: 0.0
        for key in [
            "lm",
            "alpha",
            "alpha_raw",
            "alpha_prob",
            "alpha_clean_prob",
            "alpha_capped",
            "alpha_downstream",
            "entropy",
            "prior_entropy",
            "kl",
            "load",
            "sparse",
            "semantic_pred",
            "fusion_mid",
            "fusion_global",
        ]
    }
    total_tokens = 0
    batch_count = 0
    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if start_event is not None:
        torch.cuda.synchronize(device)
        start_event.record()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(input_ids)
                loss = lm_loss(out["logits"], labels)
                aux = collect_aux_losses(
                    out.get("aux", []),
                    model_config.target_sparsity,
                    sparse_alpha=model_config.semantic_sparse_alpha,
                    alpha_cap=model_config.semantic_router_alpha_cap,
                )

            totals["lm"] += float(loss.detach().cpu())
            totals["alpha"] += float(aux["alpha_mean"].detach().cpu())
            totals["alpha_raw"] += float(aux["alpha_raw_mean"].detach().cpu())
            totals["alpha_prob"] += float(aux["alpha_prob_mean"].detach().cpu())
            totals["alpha_clean_prob"] += float(aux["alpha_clean_prob_mean"].detach().cpu())
            totals["alpha_capped"] += float(aux["alpha_capped_mean"].detach().cpu())
            totals["alpha_downstream"] += float(aux["alpha_downstream_mean"].detach().cpu())
            totals["entropy"] += float(aux["router_entropy"].detach().cpu())
            totals["prior_entropy"] += float(aux["semantic_prior_entropy"].detach().cpu())
            totals["kl"] += float(aux["kl"].detach().cpu())
            totals["load"] += float(aux["load"].detach().cpu())
            totals["sparse"] += float(aux["sparse"].detach().cpu())
            totals["semantic_pred"] += float(aux["semantic_pred"].detach().cpu())
            totals["fusion_mid"] += float(aux["fusion_mid_weight"].detach().cpu())
            totals["fusion_global"] += float(aux["fusion_global_weight"].detach().cpu())
            total_tokens += int(input_ids.numel())
            batch_count += 1

    elapsed_ms = None
    if end_event is not None and start_event is not None:
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start_event.elapsed_time(end_event))

    if batch_count == 0:
        raise RuntimeError("no evaluation batches were processed")

    val_loss = totals["lm"] / batch_count
    load = totals["load"] / batch_count
    sparse = totals["sparse"] / batch_count
    kl = totals["kl"] / batch_count
    semantic_pred = totals["semantic_pred"] / batch_count
    lambda_load = float(train_config.get("lambda_load", 0.0))
    lambda_sparse = float(checkpoint_metrics.get("lambda_sparse_effective", train_config.get("lambda_sparse", 0.0)))
    lambda_kl = effective_kl_lambda(
        float(train_config.get("lambda_kl", 0.0)),
        checkpoint_step,
        int(train_config.get("kl_warmup_steps", 0)),
    )
    lambda_semantic_pred = float(train_config.get("lambda_semantic_pred", 0.0))
    load_contrib = lambda_load * load
    sparse_contrib = lambda_sparse * sparse
    kl_contrib = lambda_kl * kl
    semantic_pred_contrib = lambda_semantic_pred * semantic_pred
    val_total_loss = val_loss + load_contrib + sparse_contrib + kl_contrib + semantic_pred_contrib
    val_aux_loss = val_total_loss - val_loss
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "architecture": train_config["architecture"],
        "split": args.data_split,
        "val_batches": batch_count,
        "val_tokens": total_tokens,
        "val_total_loss": val_total_loss,
        "val_aux_loss": val_aux_loss,
        "val_lm_loss": val_loss,
        "val_ppl": math.exp(min(20.0, val_loss)),
        "alpha_mean": totals["alpha"] / batch_count,
        "alpha_raw_mean": totals["alpha_raw"] / batch_count,
        "alpha_prob_mean": totals["alpha_prob"] / batch_count,
        "alpha_clean_prob_mean": totals["alpha_clean_prob"] / batch_count,
        "alpha_capped_mean": totals["alpha_capped"] / batch_count,
        "alpha_downstream_mean": totals["alpha_downstream"] / batch_count,
        "router_entropy": totals["entropy"] / batch_count,
        "semantic_prior_entropy": totals["prior_entropy"] / batch_count,
        "kl": kl,
        "load": load,
        "sparse": sparse,
        "semantic_pred": semantic_pred,
        "lambda_load": lambda_load,
        "lambda_sparse": lambda_sparse,
        "lambda_kl": lambda_kl,
        "lambda_semantic_pred": lambda_semantic_pred,
        "load_contrib": load_contrib,
        "sparse_contrib": sparse_contrib,
        "kl_contrib": kl_contrib,
        "semantic_pred_contrib": semantic_pred_contrib,
        "fusion_mid_weight": totals["fusion_mid"] / batch_count,
        "fusion_global_weight": totals["fusion_global"] / batch_count,
    }
    if elapsed_ms is not None and elapsed_ms > 0:
        summary["eval_tokens_per_sec"] = total_tokens / (elapsed_ms / 1000.0)

    output_path = Path(args.output) if args.output else run_dir / f"{args.data_split}_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
