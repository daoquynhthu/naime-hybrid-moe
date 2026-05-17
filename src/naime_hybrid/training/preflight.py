import importlib.util
import shutil
import tempfile
from pathlib import Path

import torch

from naime_hybrid import NAIMEStateMoEConfig, build_model
from naime_hybrid.data import HFDiskCausalDataset

from .checkpoint import load_checkpoint, save_checkpoint, save_model_weights
from .scheduler import cosine_with_warmup


def check_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    print("== NAIME Hybrid Preflight ==")
    print(f"python torch={torch.__version__}")
    print(f"cuda available={torch.cuda.is_available()} version={torch.version.cuda}")
    print(f"pytest installed={check_package('pytest')}")
    print(f"datasets installed={check_package('datasets')}")

    data_path = Path("data/naime/wikitext_processed")
    if data_path.exists():
        ds = HFDiskCausalDataset(data_path, split="train", seq_len=64, max_samples=4)
        sample = ds[0]
        print(f"data ok path={data_path} rows={len(ds)} sample={tuple(sample['input_ids'].shape)}")
    else:
        print(f"data missing path={data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = NAIMEStateMoEConfig(
        vocab_size=257,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=2,
        top_k=1,
        expert_hidden_dim=48,
    )

    input_ids = torch.randint(1, config.vocab_size, (2, 17), device=device)
    for architecture in [
        "dense",
        "token_moe",
        "naime_state_moe",
        "naime_v4_state_moe",
        "naime_v41_state_moe",
        "naime_v42_state_moe",
    ]:
        model = build_model(architecture, config).to(device)
        out = model(input_ids)
        loss = out["logits"].float().mean()
        loss.backward()
        print(f"architecture ok {architecture} logits={tuple(out['logits'].shape)}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="naime_preflight_"))
    try:
        model = build_model("naime_state_moe", config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = cosine_with_warmup(optimizer, warmup_steps=1, max_steps=2, min_lr_ratio=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        ckpt_path = tmp_dir / "latest.pt"
        model_path = tmp_dir / "models" / "model_latest.pt"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, 1, {"preflight": True}, {"loss": 0.0})
        save_model_weights(model_path, model, 1, {"preflight": True}, {"loss": 0.0})
        restored_step = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler)
        if restored_step != 1:
            raise RuntimeError(f"checkpoint restored unexpected step {restored_step}")
        if not model_path.exists():
            raise RuntimeError("model-only weights were not saved")
        print(f"checkpoint ok full={ckpt_path.name} model_only={model_path.name}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("preflight complete")


if __name__ == "__main__":
    main()
