from pathlib import Path

import torch

from .checkpoint import (
    AsyncCheckpointWriter,
    build_checkpoint_payload,
    build_model_payload,
    save_checkpoint,
    save_model_weights,
    save_payload,
)
from .config import TrainConfig


def save_checkpoint_pair(
    writer: AsyncCheckpointWriter | None,
    checkpoint_path: Path,
    model_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    step: int,
    config: TrainConfig,
    metrics: dict,
) -> None:
    config_dict = config.to_dict()
    if writer is None:
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, step, config_dict, metrics)
        save_model_weights(model_path, model, step, config_dict, metrics)
        return

    checkpoint_payload = build_checkpoint_payload(model, optimizer, scheduler, scaler, step, config_dict, metrics)
    writer.submit(checkpoint_path, checkpoint_payload)
    writer.wait()
    del checkpoint_payload

    model_payload = build_model_payload(model, step, config_dict, metrics)
    writer.submit(model_path, model_payload)
    writer.wait()
    del model_payload


def save_checkpoint_bundle(
    writer: AsyncCheckpointWriter | None,
    checkpoint_paths: list[Path],
    model_paths: list[Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    step: int,
    config: TrainConfig,
    metrics: dict,
) -> None:
    """Write full and model-only checkpoints without keeping both payloads alive.

    Full checkpoints are already large because they include optimizer state.
    Keeping a full payload and a model-only payload in memory at the same time
    can push Windows commit charge over the limit during long GPU runs.
    """
    config_dict = config.to_dict()
    checkpoint_payload = build_checkpoint_payload(model, optimizer, scheduler, scaler, step, config_dict, metrics)
    if writer is None:
        for path in checkpoint_paths:
            save_payload(path, checkpoint_payload)
    else:
        for path in checkpoint_paths:
            writer.submit(path, checkpoint_payload)
        writer.wait()
    del checkpoint_payload

    if not model_paths:
        return

    model_payload = build_model_payload(model, step, config_dict, metrics)
    if writer is None:
        for path in model_paths:
            save_payload(path, model_payload)
    else:
        for path in model_paths:
            writer.submit(path, model_payload)
        writer.wait()
    del model_payload
