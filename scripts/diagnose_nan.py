"""Diagnose NaN source: load clean checkpoint, run steps, report unstable params."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from naime_hybrid.models import build_model
from naime_hybrid.training.losses import collect_aux_losses, lm_loss


def cycle_loader(loader):
    while True:
        yield from loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-path", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "latest.pt"
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}")
        return

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config_dict = payload["config"]

    from naime_hybrid.config import NAIMEStateMoEConfig

    model_config = NAIMEStateMoEConfig(**config_dict["model"])
    model = build_model("naime_v5_world_state_moe", model_config)

    state_dict = payload["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    device = torch.device("cuda")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.08)
    if payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])

    scaler = torch.amp.GradScaler("cuda")
    if payload.get("scaler"):
        try:
            scaler.load_state_dict(payload["scaler"])
        except Exception:
            pass

    print(f"loaded step {payload['step']}")
    print("compiling model...")
    import logging
    import warnings

    warnings.filterwarnings("ignore", message="online softmax")
    torch._logging.set_logs(dynamo=logging.ERROR, inductor=logging.ERROR)
    model = torch.compile(model)
    print("building dataset...")
    from functools import partial

    from torch.utils.data import DataLoader

    from naime_hybrid.data import HFDiskCausalDataset

    ds = HFDiskCausalDataset(
        path=args.data_path,
        split="train",
        seq_len=model_config.max_seq_len,
    )
    collate_fn = partial(HFDiskCausalDataset.causal_collate, seq_len=model_config.max_seq_len)
    train_loader = DataLoader(
        ds, batch_size=3, shuffle=True, pin_memory=True, num_workers=0, drop_last=True, collate_fn=collate_fn
    )
    data_iter = cycle_loader(train_loader)

    print("running batches with grad inspection...")
    for i in range(800):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids)
            loss = lm_loss(out["logits"], labels)
            aux = collect_aux_losses(out.get("aux", []), 0.45, "downstream", 0.90)
            total_loss = (
                loss
                + 0.01 * aux["sparse"]
                + 0.003 * aux["kl"]
                + 0.02 * aux["v5_state_pred"]
                + 0.01 * aux["v5_slot_diversity"]
            )

        if not torch.isfinite(total_loss.detach()):
            print(f"\n🔴 loss inf/NaN at step {payload['step'] + i + 1}")
            break

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        g = 0.0
        max_pg = 0.0
        max_name = ""
        for name, p in model.named_parameters():
            if p.grad is not None:
                pg = float(p.grad.norm().item())
                g += pg * pg
                if pg > max_pg:
                    max_pg = pg
                    max_name = name

        g = g**0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

        if i % 100 == 0:
            suffix = (
                f" max={max_pg:.0f}({max_name.split('.')[-2] + '.' + max_name.split('.')[-1] if '.' in max_name else max_name})"
                if max_name
                else ""
            )
            print(f"step {payload['step'] + i + 1:5d}  lm={float(loss):.3f}  grad={g:.1f}{suffix}")

        if g > 15:
            print(f"\n🔴 grad_norm={g:.1f} at step {payload['step'] + i + 1}")
            top_params = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    pg = float(p.grad.norm().item())
                    top_params.append((pg, name))
            top_params.sort(reverse=True)
            print("top 10 largest parameter gradients:")
            for pg, name in top_params[:10]:
                short = ".".join(name.split(".")[-3:])
                print(f"  {pg:12.3f}  {short}")
            break

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    else:
        print("\n✅ no grad explosion detected in 500 steps")


if __name__ == "__main__":
    main()
