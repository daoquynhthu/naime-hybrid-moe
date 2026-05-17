import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from naime_hybrid.config import NAIMEStateMoEConfig
from naime_hybrid.data import ByteTextDataset, HFDiskCausalDataset, RandomTokenDataset
from naime_hybrid.models import build_model

from .config import TrainConfig
from .losses import collect_aux_losses, lm_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_dataset(config: TrainConfig, split: str | None = None):
    if config.random_data:
        return RandomTokenDataset(
            vocab_size=config.model.vocab_size,
            seq_len=config.model.max_seq_len,
            num_samples=config.max_samples or max(config.batch_size * config.max_steps, 128),
            seed=config.seed,
        )
    if config.data_path is None:
        raise ValueError("provide --data-path or use --random-data for smoke training")
    data_path = Path(config.data_path)
    data_format = config.data_format
    if data_format == "auto":
        data_format = "hf_disk" if data_path.is_dir() and (data_path / "dataset_dict.json").exists() else "byte"
    if data_format == "hf_disk":
        return HFDiskCausalDataset(
            data_path,
            split=split or config.data_split,
            seq_len=config.model.max_seq_len,
            max_samples=config.max_samples,
        )
    return ByteTextDataset.from_file(data_path, seq_len=config.model.max_seq_len, max_samples=config.max_samples)


def cycle_loader(loader: DataLoader):
    while True:
        yield from loader


def _cuda_cache_clear() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def probe_auto_batch_size(
    architecture: str,
    model_config: NAIMEStateMoEConfig,
    requested_batch: int,
    max_batch: int,
    vram_fraction: float,
    use_amp: bool,
    device: torch.device,
    logger,
) -> int:
    if device.type != "cuda":
        logger.info("auto-batch skipped: device is not cuda")
        return requested_batch

    if not 0.1 <= vram_fraction <= 0.98:
        raise ValueError("--vram-fraction must be between 0.1 and 0.98")

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    budget_bytes = int(free_bytes * vram_fraction)
    selection_budget_bytes = int(budget_bytes * 0.95)
    upper = max(1, max_batch)
    best = 0
    tried: dict[int, tuple[bool, int]] = {}

    logger.info(
        "auto-batch probing: gpu=%s total_vram=%.2fGiB free_vram=%.2fGiB budget=%.2fGiB select_budget=%.2fGiB max_batch=%d",
        torch.cuda.get_device_name(device),
        total_bytes / 1024**3,
        free_bytes / 1024**3,
        budget_bytes / 1024**3,
        selection_budget_bytes / 1024**3,
        upper,
    )

    def predicted_peak_bytes(batch_size: int) -> int | None:
        successful = sorted((b, peak) for b, (ok, peak) in tried.items() if ok and peak > 0)
        if len(successful) < 2:
            return None
        (b1, p1), (b2, p2) = successful[-2], successful[-1]
        if b2 <= b1:
            return None
        slope = max(0, (p2 - p1) / (b2 - b1))
        return int(p2 + slope * (batch_size - b2))

    def try_batch(batch_size: int) -> tuple[bool, int]:
        if batch_size in tried:
            return tried[batch_size]

        _cuda_cache_clear()
        ok = False
        peak = 0
        probe_model = None
        probe_optimizer = None
        input_ids = None
        labels = None
        out = None
        loss = None
        total_loss = None
        try:
            probe_model = build_model(architecture, model_config).to(device)
            probe_model.train()
            probe_optimizer = torch.optim.AdamW(probe_model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            input_ids = torch.randint(
                1,
                model_config.vocab_size,
                (batch_size, model_config.max_seq_len),
                device=device,
                dtype=torch.long,
            )
            labels = torch.randint(
                1,
                model_config.vocab_size,
                (batch_size, model_config.max_seq_len),
                device=device,
                dtype=torch.long,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = probe_model(input_ids)
                loss = lm_loss(out["logits"], labels)
                aux = collect_aux_losses(
                    out.get("aux", []),
                    model_config.target_sparsity,
                    sparse_alpha=model_config.semantic_sparse_alpha,
                    alpha_cap=model_config.semantic_router_alpha_cap,
                )
                total_loss = (
                    loss
                    + 0.01 * aux["load"]
                    + 0.01 * aux["sparse"]
                    + 0.001 * aux["kl"]
                    + 0.01 * aux["semantic_pred"]
                    + 0.01 * aux["v5_state_pred"]
                    + 0.01 * aux["v5_slot_diversity"]
                    + 0.01 * aux["v5_slot_stability"]
                )
            scaler.scale(total_loss).backward()
            scaler.step(probe_optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            ok = peak <= budget_bytes
        except torch.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated(device)
            ok = False
        finally:
            del probe_model, probe_optimizer, input_ids, labels, out, loss, total_loss
            _cuda_cache_clear()

        tried[batch_size] = (ok, peak)
        logger.info("auto-batch probe batch=%d ok=%s peak=%.2fGiB", batch_size, ok, peak / 1024**3)
        return tried[batch_size]

    candidate = min(max(1, requested_batch), upper)
    while candidate <= upper:
        predicted_peak = predicted_peak_bytes(candidate)
        if predicted_peak is not None and predicted_peak > budget_bytes:
            logger.info(
                "auto-batch skip probe batch=%d predicted_peak=%.2fGiB exceeds budget=%.2fGiB",
                candidate,
                predicted_peak / 1024**3,
                budget_bytes / 1024**3,
            )
            break

        ok, peak = try_batch(candidate)
        if not ok:
            break
        if peak <= selection_budget_bytes:
            best = candidate
        else:
            logger.info(
                "auto-batch stop growth at batch=%d peak=%.2fGiB exceeds select_budget=%.2fGiB",
                candidate,
                peak / 1024**3,
                selection_budget_bytes / 1024**3,
            )
            break
        candidate *= 2

    low = best + 1
    high = min(upper, candidate - 1)
    while low <= high:
        mid = (low + high) // 2
        predicted_peak = predicted_peak_bytes(mid)
        if predicted_peak is not None and predicted_peak > budget_bytes:
            logger.info(
                "auto-batch skip probe batch=%d predicted_peak=%.2fGiB exceeds budget=%.2fGiB",
                mid,
                predicted_peak / 1024**3,
                budget_bytes / 1024**3,
            )
            high = mid - 1
            continue
        ok, peak = try_batch(mid)
        if ok and peak <= selection_budget_bytes:
            best = mid
            low = mid + 1
        else:
            if ok:
                logger.info(
                    "auto-batch reject batch=%d peak=%.2fGiB exceeds select_budget=%.2fGiB",
                    mid,
                    peak / 1024**3,
                    selection_budget_bytes / 1024**3,
                )
            high = mid - 1

    if best <= 0:
        logger.warning("auto-batch could not fit requested batch=%d; falling back to batch=1", requested_batch)
        best = 1
    logger.info("auto-batch selected batch_size=%d", best)
    return best
