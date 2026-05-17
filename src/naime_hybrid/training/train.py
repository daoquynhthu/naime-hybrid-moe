import json
import logging
import math
import os
import sys
import time
import warnings
from collections import deque
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from naime_hybrid.data import HFDiskCausalDataset
from naime_hybrid.models import build_model

from .checkpoint import (
    AsyncCheckpointWriter,
    build_checkpoint_payload,
    build_model_payload,
    load_checkpoint,
    save_checkpoint,
    save_payload,
    save_payloads_in_subprocess,
)
from .checkpoint_policy import save_checkpoint_pair
from .cli import build_train_config, parse_args
from .control import effective_kl_lambda, load_reference_curve, reference_value_at_step
from .logging_utils import JsonlMetricLogger, metrics_jsonl_to_csv, setup_logger
from .losses import collect_aux_losses, lm_loss
from .prefetch import AsyncPrefetcher
from .progress import TrainingProgress
from .runtime import build_dataset, cycle_loader, probe_auto_batch_size, resolve_device, set_seed
from .scheduler import cosine_with_restarts, cosine_with_warmup
from .signals import StopSignalMonitor
from .validation import evaluate_model


class ResumableRandomSampler(Sampler[int]):
    """Deterministic shuffled stream that resumes at a consumed-sample offset."""

    def __init__(self, data_len: int, seed: int, consumed_samples: int = 0):
        if data_len <= 0:
            raise ValueError("data_len must be positive")
        self.data_len = int(data_len)
        self.seed = int(seed)
        self.start_epoch = int(consumed_samples // self.data_len)
        self.start_offset = int(consumed_samples % self.data_len)
        self._epoch = self.start_epoch
        self._first_iter = True

    def __iter__(self):
        epoch = self._epoch
        offset = self.start_offset if self._first_iter else 0
        self._first_iter = False
        self._epoch += 1

        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        order = torch.randperm(self.data_len, generator=generator).tolist()
        yield from order[offset:]

    def __len__(self) -> int:
        return self.data_len - self.start_offset if self._first_iter else self.data_len


def _build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool,
    *,
    collate_fn=None,
    seed: int = 1234,
    resume_step: int = 0,
    grad_accum_steps: int = 1,
) -> DataLoader:
    sampler = None
    if shuffle:
        epoch_samples = len(dataset)
        if drop_last:
            epoch_samples = (epoch_samples // batch_size) * batch_size
        consumed_samples = max(0, resume_step) * max(1, grad_accum_steps) * batch_size
        sampler = ResumableRandomSampler(epoch_samples, seed=seed, consumed_samples=consumed_samples)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        # Keep workers alive and prefetch multiple batches so the GPU is less
        # likely to stall on Python-side dataset work between optimizer steps.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)


def _extract_max_steps(path: Path) -> int:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return int(ckpt.get("config", {}).get("max_steps", 0))
    except Exception:
        return 0


def _extract_step(path: Path) -> int:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return int(ckpt.get("step", 0))
    except Exception:
        return 0


def _resolve_resume_path(
    resume: str,
    run_dir: Path,
    model_dir: Path,
    allow_failed: bool,
) -> Path | None:
    if resume in {"none", ""}:
        return None
    if resume != "auto":
        return Path(resume)

    candidates = [
        run_dir / "latest.pt",
        run_dir / "interrupted.pt",
        run_dir / "best.pt",
        model_dir / "model_latest.pt",
        model_dir / "model_interrupted.pt",
        model_dir / "model_best.pt",
    ]
    if allow_failed:
        candidates.extend([run_dir / "failed.pt", model_dir / "model_failed.pt"])
    for path in candidates:
        if path.exists():
            return path
    return None


def _set_scheduler_to_step(scheduler, step: int) -> None:
    step = max(0, step)
    scheduler.last_epoch = step - 1
    scheduler._step_count = max(getattr(scheduler, "_step_count", 0), step)
    if not hasattr(scheduler, "lr_lambdas"):
        return
    for group, base_lr, lr_lambda in zip(
        scheduler.optimizer.param_groups,
        scheduler.base_lrs,
        scheduler.lr_lambdas,
        strict=False,
    ):
        group["lr"] = base_lr * lr_lambda(step)
    scheduler._last_lr = [group["lr"] for group in scheduler.optimizer.param_groups]


def _apply_lr_safety_factor(optimizer, scheduler, factor: float) -> float:
    """Apply a runtime LR backoff without changing the scheduler's shape."""

    factor = max(0.05, min(1.0, float(factor)))
    if hasattr(scheduler, "lr_lambdas"):
        step = max(0, int(getattr(scheduler, "last_epoch", 0)))
        scheduled_lrs = [
            float(base_lr) * float(lr_lambda(step))
            for base_lr, lr_lambda in zip(scheduler.base_lrs, scheduler.lr_lambdas, strict=False)
        ]
    else:
        scheduled_lrs = [float(group["lr"]) / factor for group in optimizer.param_groups]
    applied_lrs: list[float] = []
    for group, scheduled_lr in zip(optimizer.param_groups, scheduled_lrs, strict=False):
        lr = float(scheduled_lr) * factor
        group["lr"] = lr
        applied_lrs.append(lr)
    scheduler._last_lr = applied_lrs
    return applied_lrs[0] if applied_lrs else 0.0


def _save_shutdown_checkpoint(
    *,
    reason_name: str,
    checkpoint_writer: AsyncCheckpointWriter | None,
    run_dir: Path,
    model_dir: Path,
    latest_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    step: int,
    config,
    metrics: dict,
    logger: logging.Logger,
) -> None:
    if checkpoint_writer is not None:
        try:
            checkpoint_writer.wait()
        except Exception:
            logger.exception("checkpoint writer failed while flushing before %s checkpoint", reason_name)

    config_dict = config.to_dict()
    full_payload = build_checkpoint_payload(model, optimizer, scheduler, scaler, step, config_dict, metrics)
    model_payload = build_model_payload(model, step, config_dict, metrics)
    save_payloads_in_subprocess(
        [
            (latest_path, full_payload),
            (run_dir / f"{reason_name}.pt", full_payload),
            (model_dir / "model_latest.pt", model_payload),
            (model_dir / f"model_{reason_name}.pt", model_payload),
        ]
    )


def main() -> None:
    args = parse_args()
    config = build_train_config(args)
    set_seed(config.seed)

    run_dir = Path(config.output_dir) / config.run_name
    model_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)
    metrics_logger = JsonlMetricLogger(run_dir / "metrics.jsonl")

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

    device = resolve_device(config.device)
    logger.info("run_dir=%s", run_dir)
    logger.info("device=%s cuda=%s", device, torch.cuda.is_available())
    logger.info("architecture=%s", config.architecture)

    if config.auto_batch:
        config.batch_size = probe_auto_batch_size(
            config.architecture,
            config.model,
            config.batch_size,
            config.auto_batch_max,
            config.vram_fraction,
            config.amp and device.type == "cuda",
            device,
            logger,
        )
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

    resume_probe_path = _resolve_resume_path(config.resume, run_dir, model_dir, config.resume_allow_failed)
    resume_probe_step = _extract_step(resume_probe_path) if resume_probe_path is not None else 0

    if config.target_tokens is not None:
        tokens_per_step = config.batch_size * config.model.max_seq_len * config.grad_accum_steps
        budget_steps = max(1, math.ceil(config.target_tokens / tokens_per_step))
        if config.target_tokens_mode == "additional":
            config.max_steps = max(resume_probe_step + 1, resume_probe_step + budget_steps)
        else:
            config.max_steps = max(1, budget_steps)
        logger.info(
            "target_tokens=%d mode=%s resume_step=%d tokens_per_step=%d adjusted_max_steps=%d effective_tokens=%d",
            config.target_tokens,
            config.target_tokens_mode,
            resume_probe_step,
            tokens_per_step,
            config.max_steps,
            tokens_per_step * config.max_steps,
        )
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

    dataset = build_dataset(config)
    collate_fn = None
    if isinstance(dataset, HFDiskCausalDataset):
        collate_fn = partial(HFDiskCausalDataset.causal_collate, seq_len=config.model.max_seq_len)
    loader = _build_loader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        seed=config.seed,
        resume_step=resume_probe_step,
        grad_accum_steps=config.grad_accum_steps,
    )
    data_iter = cycle_loader(loader)
    if device.type == "cuda":
        data_iter = AsyncPrefetcher(data_iter, device)
    logger.info("dataset_size=%s batch_size=%s seq_len=%s", len(dataset), config.batch_size, config.model.max_seq_len)
    if resume_probe_step > 0:
        epoch_batches = max(1, len(dataset) // config.batch_size)
        offset_batches = (resume_probe_step * config.grad_accum_steps) % epoch_batches
        logger.info(
            "train sampler resumed stream seed=%d resume_step=%d offset_batches=%d/%d",
            config.seed,
            resume_probe_step,
            offset_batches,
            epoch_batches,
        )
    logger.info(
        "loader config train_workers=%d persistent=%s prefetch=%s pin_memory=%s",
        config.num_workers,
        config.num_workers > 0,
        4 if config.num_workers > 0 else 0,
        device.type == "cuda",
    )

    eval_loader = None
    if config.eval_every > 0 and not config.random_data and config.data_path is not None:
        eval_dataset = build_dataset(config, split=config.eval_split)
        eval_collate_fn = None
        if isinstance(eval_dataset, HFDiskCausalDataset):
            eval_collate_fn = partial(HFDiskCausalDataset.causal_collate, seq_len=config.model.max_seq_len)
        eval_workers = min(config.num_workers, 2)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=eval_workers,
            drop_last=False,
            pin_memory=device.type == "cuda",
            collate_fn=eval_collate_fn,
        )
        logger.info(
            "eval enabled split=%s dataset_size=%s eval_every=%s max_batches=%s",
            config.eval_split,
            len(eval_dataset),
            config.eval_every,
            config.eval_max_batches,
        )
    elif config.eval_every > 0:
        logger.warning("eval requested but skipped because random data or missing data path is in use")

    model = build_model(config.architecture, config.model).to(device)
    model_eval = model
    if config.compile_model:
        warnings.filterwarnings("ignore", message="online softmax")
        torch._logging.set_logs(dynamo=logging.ERROR, inductor=logging.ERROR)
        logger.info("compiling model with torch.compile (first call triggers JIT; this may take 1-2 minutes)")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    scheduler = cosine_with_warmup(
        optimizer,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        min_lr_ratio=config.min_lr_ratio,
    )
    if config.lr_cycle_length > 0:
        scheduler = cosine_with_restarts(
            optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr_ratio=config.min_lr_ratio,
            cycle_length=config.lr_cycle_length,
            restart_ratio=config.lr_restart_ratio,
            restart_warmup=config.lr_restart_warmup,
        )
        logger.info(
            "lr schedule warm_restarts cycle=%d restart_ratio=%.2f restart_warmup=%d",
            config.lr_cycle_length,
            config.lr_restart_ratio,
            config.lr_restart_warmup,
        )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_writer = (
        AsyncCheckpointWriter(max_queue=config.async_checkpoint_queue) if config.async_checkpoint else None
    )
    if checkpoint_writer is not None:
        logger.info(
            "async checkpoint writer enabled queue=%d save_every=%d latest_every=%d latest_sync=%s",
            config.async_checkpoint_queue,
            config.save_every,
            config.latest_every,
            config.latest_sync,
        )
    else:
        logger.info(
            "async checkpoint writer disabled save_every=%d latest_every=%d latest_sync=%s",
            config.save_every,
            config.latest_every,
            config.latest_sync,
        )

    start_step = 0
    latest_path = run_dir / "latest.pt"
    failed_path = run_dir / "failed.pt"
    best_path = run_dir / "best.pt"
    stop_path = Path(config.stop_file) if config.stop_file else run_dir / "STOP"
    logger.info("graceful stop file=%s check_every=%d", stop_path, config.stop_check_every)
    (run_dir / "trainer.pid").write_text(str(os.getpid()), encoding="utf-8")
    checkpoint_max_steps = 0
    resume_path = _resolve_resume_path(config.resume, run_dir, model_dir, config.resume_allow_failed)
    if resume_path is not None:
        logger.info("resuming from %s", resume_path)
        start_step = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, strict=config.strict_resume)
        checkpoint_max_steps = _extract_max_steps(resume_path)
    else:
        logger.info("starting from scratch")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if config.resume_lr_policy == "reset" and start_step > 0:
        _set_scheduler_to_step(scheduler, 0)
        logger.info("resume lr policy reset: scheduler restarted after loading step=%d", start_step)
    elif config.resume_lr_policy == "absolute" and start_step > 0:
        _set_scheduler_to_step(scheduler, start_step)
        logger.info("resume lr policy absolute: scheduler aligned to resumed step=%d", start_step)
    elif (
        config.resume_lr_policy == "progress"
        and checkpoint_max_steps > 0
        and checkpoint_max_steps != config.max_steps
        and start_step > config.warmup_steps
    ):
        old_progress = (start_step - config.warmup_steps) / max(1, checkpoint_max_steps - config.warmup_steps)
        old_progress = min(1.0, old_progress)
        new_step = int(old_progress * max(1, config.max_steps - config.warmup_steps)) + config.warmup_steps
        _set_scheduler_to_step(scheduler, new_step)
        logger.info(
            "scheduler lr continuity: old_max=%d -> new_max=%d, progress=%.3f, remapped last_epoch %d->%d",
            checkpoint_max_steps,
            config.max_steps,
            old_progress,
            start_step,
            new_step,
        )
    elif start_step > 0:
        logger.info(
            "resume lr policy checkpoint: keeping loaded scheduler state at step=%d old_max=%d new_max=%d",
            start_step,
            checkpoint_max_steps,
            config.max_steps,
        )
    if start_step > 0:
        logger.info("saving resume baseline checkpoint at step=%d", start_step)
        save_checkpoint_pair(
            checkpoint_writer,
            latest_path,
            model_dir / "model_latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            start_step,
            config,
            {"resume_baseline": 1.0, "resume_path": str(resume_path) if resume_path is not None else ""},
        )
    rolling_loss = 0.0
    rolling_lm_loss = 0.0
    rolling_tokens = 0
    rolling_steps = 0
    last_log_time = time.time()
    best_val_loss = math.inf
    eval_count = 0
    stale_eval_count = 0
    nan_streak = 0
    grad_explosion_streak = 0
    bad_grad_window: deque[int] = deque()
    lr_safety_factor = 1.0
    last_lr_backoff_step = -10**9
    bad_grad_window_steps = 500
    bad_grad_window_threshold = 15
    bad_grad_backoff_cooldown = 500
    structural_gap_previous: float | None = None
    structural_widen_count = 0
    reference_curve = load_reference_curve(config.reference_metrics_path) if config.structural_stop else []
    if config.structural_stop and reference_curve:
        logger.info(
            "structural stop enabled reference=%s points=%d min_gap=%.3f widen_delta=%.3f patience=%d warmup_steps=%d",
            config.reference_metrics_path,
            len(reference_curve),
            config.structural_stop_min_gap,
            config.structural_stop_widen_delta,
            config.structural_stop_patience,
            config.structural_stop_warmup_steps,
        )
    elif config.structural_stop:
        logger.warning(
            "structural stop requested but reference metrics were not found/readable: %s", config.reference_metrics_path
        )
    best_metric_path = best_path if best_path.exists() else model_dir / "model_best.pt"
    if best_metric_path.exists():
        try:
            existing_best = torch.load(best_metric_path, map_location="cpu", weights_only=False)
            existing_metrics = existing_best.get("metrics", {})
            existing_best_loss = existing_metrics.get("val_lm_loss")
            if isinstance(existing_best_loss, (int, float)) and math.isfinite(existing_best_loss):
                best_val_loss = float(existing_best_loss)
                logger.info("loaded existing best val_loss=%.4f from %s", best_val_loss, best_metric_path)
        except Exception as exc:
            logger.warning("could not read existing best checkpoint metrics from %s: %s", best_metric_path, exc)

    step = start_step
    console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout]
    for h in console_handlers:
        logger.removeHandler(h)
    progress = TrainingProgress(config.max_steps, config.architecture)
    stop_signals = StopSignalMonitor()
    stop_signals.install(stop_file=stop_path)
    try:
        for step in range(start_step + 1, config.max_steps + 1):
            if stop_signals.requested:
                save_step = max(start_step, step - 1)
                logger.warning("stop signal requested before next step (%s); saving at step=%d", stop_signals.reason, save_step)
                _save_shutdown_checkpoint(
                    reason_name="interrupted",
                    checkpoint_writer=checkpoint_writer,
                    run_dir=run_dir,
                    model_dir=model_dir,
                    latest_path=latest_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=save_step,
                    config=config,
                    metrics={"stop_reason": stop_signals.reason},
                    logger=logger,
                )
                logger.info("signal checkpoint saved by subprocess; exiting cleanly")
                break
            batch_loss = 0.0
            batch_lm_loss = 0.0
            metrics: dict[str, float] = {}
            for _micro_step in range(config.grad_accum_steps):
                batch = next(data_iter)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = model(input_ids)
                    main_loss = lm_loss(out["logits"], labels)
                    aux = collect_aux_losses(
                        out.get("aux", []),
                        config.model.target_sparsity,
                        sparse_alpha=config.model.semantic_sparse_alpha,
                        alpha_cap=config.model.semantic_router_alpha_cap,
                    )

                    # Fixed auxiliary weights keep the objective explicit. The
                    # previous learned balancer masked gradients in V5 ablations.
                    w_sparse = torch.tensor(config.lambda_sparse, device=device)
                    w_kl = torch.tensor(config.lambda_kl, device=device)
                    w_state_pred = torch.tensor(config.lambda_state_pred, device=device)
                    w_slot_div = torch.tensor(config.lambda_slot_diversity, device=device)
                    w_self_pred = torch.tensor(config.lambda_self_pred, device=device)
                    w_self_slot_div = torch.tensor(config.lambda_self_slot_diversity, device=device)

                    kl_warmup = effective_kl_lambda(1.0, step, config.kl_warmup_steps)
                    load_contrib = config.lambda_load * aux["load"]
                    sparse_contrib = w_sparse * aux["sparse"]
                    kl_contrib = w_kl * kl_warmup * aux["kl"]
                    semantic_pred_contrib = config.lambda_semantic_pred * aux["semantic_pred"]
                    state_pred_contrib = w_state_pred * aux["v5_state_pred"]
                    slot_diversity_contrib = w_slot_div * aux["v5_slot_diversity"]
                    slot_stability_contrib = config.lambda_slot_stability * aux["v5_slot_stability"]
                    self_pred_contrib = w_self_pred * aux["v6_self_pred"]
                    self_slot_diversity_contrib = w_self_slot_div * aux["v6_slot_diversity"]
                    total_loss = (
                        main_loss
                        + load_contrib
                        + sparse_contrib
                        + kl_contrib
                        + semantic_pred_contrib
                        + state_pred_contrib
                        + slot_diversity_contrib
                        + slot_stability_contrib
                        + self_pred_contrib
                        + self_slot_diversity_contrib
                    )
                    total_loss = total_loss / config.grad_accum_steps

                if not torch.isfinite(total_loss.detach()):
                    raise FloatingPointError(f"non-finite loss at step {step}: {float(total_loss.detach())}")

                scaler.scale(total_loss).backward()
                batch_loss += float(total_loss.detach()) * config.grad_accum_steps
                batch_lm_loss += float(main_loss.detach())
                metrics = {
                    "loss_total": float(total_loss.detach()) * config.grad_accum_steps,
                    "loss_lm": float(main_loss.detach()),
                    "loss_load": float(aux["load"].detach()),
                    "loss_sparse": float(aux["sparse"].detach()),
                    "loss_kl": float(aux["kl"].detach()),
                    "loss_semantic_pred": float(aux["semantic_pred"].detach()),
                    "loss_v5_state_pred": float(aux["v5_state_pred"].detach()),
                    "loss_v5_slot_diversity": float(aux["v5_slot_diversity"].detach()),
                    "loss_v5_slot_stability": float(aux["v5_slot_stability"].detach()),
                    "loss_v6_self_pred": float(aux["v6_self_pred"].detach()),
                    "loss_v6_slot_diversity": float(aux["v6_slot_diversity"].detach()),
                    "loss_load_contrib": float(load_contrib.detach()),
                    "loss_sparse_contrib": float(sparse_contrib.detach()),
                    "loss_kl_contrib": float(kl_contrib.detach()),
                    "loss_semantic_pred_contrib": float(semantic_pred_contrib.detach()),
                    "loss_v5_state_pred_contrib": float(state_pred_contrib.detach()),
                    "loss_v5_slot_diversity_contrib": float(slot_diversity_contrib.detach()),
                    "loss_v5_slot_stability_contrib": float(slot_stability_contrib.detach()),
                    "loss_v6_self_pred_contrib": float(self_pred_contrib.detach()),
                    "loss_v6_slot_diversity_contrib": float(self_slot_diversity_contrib.detach()),
                    "router_entropy": float(aux["router_entropy"].detach()),
                    "semantic_prior_entropy": float(aux["semantic_prior_entropy"].detach()),
                    "fusion_mid_weight": float(aux["fusion_mid_weight"].detach()),
                    "fusion_global_weight": float(aux["fusion_global_weight"].detach()),
                    "alpha_mean": float(aux["alpha_mean"].detach()),
                    "alpha_raw_mean": float(aux["alpha_raw_mean"].detach()),
                    "alpha_prob_mean": float(aux["alpha_prob_mean"].detach()),
                    "alpha_clean_prob_mean": float(aux["alpha_clean_prob_mean"].detach()),
                    "alpha_capped_mean": float(aux["alpha_capped_mean"].detach()),
                    "alpha_downstream_mean": float(aux["alpha_downstream_mean"].detach()),
                    "v4_layer_scale": float(aux["v4_layer_scale"].detach()),
                    "v4_state_norm": float(aux["v4_state_norm"].detach()),
                    "v4_memory_norm": float(aux["v4_memory_norm"].detach()),
                    "v4_memory_gate": float(aux["v4_memory_gate"].detach()),
                    "v4_memory_attention_entropy": float(aux["v4_memory_attention_entropy"].detach()),
                    "v4_memory_read_strength": float(aux["v4_memory_read_strength"].detach()),
                    "v4_memory_novelty": float(aux["v4_memory_novelty"].detach()),
                    "v4_state_gate": float(aux["v4_state_gate"].detach()),
                    "v4_state_confidence": float(aux["v4_state_confidence"].detach()),
                    "v4_state_delta": float(aux["v4_state_delta"].detach()),
                    "v4_state_agreement": float(aux["v4_state_agreement"].detach()),
                    "gate_mix_alpha_weight": float(aux["gate_mix_alpha_weight"].detach()),
                    "gate_mix_clean_weight": float(aux["gate_mix_clean_weight"].detach()),
                    "gate_mix_state_weight": float(aux["gate_mix_state_weight"].detach()),
                    "v5_state_pred": float(aux["v5_state_pred"].detach()),
                    "v5_slot_diversity": float(aux["v5_slot_diversity"].detach()),
                    "v5_slot_stability": float(aux["v5_slot_stability"].detach()),
                    "v5_slot_update_gate": float(aux["v5_slot_update_gate"].detach()),
                    "v5_slot_write_max": float(aux["v5_slot_write_max"].detach()),
                    "v5_slot_write_entropy": float(aux["v5_slot_write_entropy"].detach()),
                    "v5_slot_write_min": float(aux["v5_slot_write_min"].detach()),
                    "v5_slot_write_active": float(aux["v5_slot_write_active"].detach()),
                    "v5_slot_confidence": float(aux["v5_slot_confidence"].detach()),
                    "v5_slot_confidence_std": float(aux["v5_slot_confidence_std"].detach()),
                    "v5_slot_delta": float(aux["v5_slot_delta"].detach()),
                    "v5_slot_cosine": float(aux["v5_slot_cosine"].detach()),
                    "v5_slot_read_entropy": float(aux["v5_slot_read_entropy"].detach()),
                    "v5_slot_read_max": float(aux["v5_slot_read_max"].detach()),
                    "v6_self_pred": float(aux["v6_self_pred"].detach()),
                    "v6_slot_diversity": float(aux["v6_slot_diversity"].detach()),
                    "v6_slot_cosine": float(aux["v6_slot_cosine"].detach()),
                    "v6_slot_context_cosine": float(aux["v6_slot_context_cosine"].detach()),
                    "v6_state_delta": float(aux["v6_state_delta"].detach()),
                    "v6_state_norm": float(aux["v6_state_norm"].detach()),
                    "v6_reflection_norm": float(aux["v6_reflection_norm"].detach()),
                    "v6_boundary_entropy": float(aux["v6_boundary_entropy"].detach()),
                    "v6_boundary_self": float(aux["v6_boundary_self"].detach()),
                    "v6_boundary_world": float(aux["v6_boundary_world"].detach()),
                    "v6_boundary_other": float(aux["v6_boundary_other"].detach()),
                    "v6_boundary_unknown": float(aux["v6_boundary_unknown"].detach()),
                    "dispatch_dense": float(aux["dispatch_dense"].detach()),
                    "lambda_sparse_effective": float(w_sparse),
                    "lambda_kl_effective": float(w_kl * kl_warmup),
                    "lambda_state_pred_effective": float(w_state_pred),
                    "lambda_slot_diversity_effective": float(w_slot_div),
                    "lambda_self_pred_effective": float(w_self_pred),
                    "lambda_self_slot_diversity_effective": float(w_self_slot_div),
                    "lr_safety_factor": float(lr_safety_factor),
                    "bad_grad_window_count": float(len(bad_grad_window)),
                    "balancer_conf": 0.0,
                }

            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            else:
                grad_norm = torch.tensor(0.0)

            if not torch.isfinite(grad_norm) or float(grad_norm) > 20.0:
                bad_grad_window.append(step)
                while bad_grad_window and step - bad_grad_window[0] > bad_grad_window_steps:
                    bad_grad_window.popleft()
                if not torch.isfinite(grad_norm):
                    nan_streak += 1
                    tag = f"NaN streak={nan_streak}"
                else:
                    grad_explosion_streak += 1
                    tag = f"explosion streak={grad_explosion_streak} grad={float(grad_norm):.1f}"
                logger.warning("bad grad at step %d %s; skipping step", step, tag)
                if (
                    len(bad_grad_window) >= bad_grad_window_threshold
                    and step - last_lr_backoff_step >= bad_grad_backoff_cooldown
                ):
                    old_factor = lr_safety_factor
                    lr_safety_factor = max(0.25, lr_safety_factor * 0.7)
                    last_lr_backoff_step = step
                    applied_lr = _apply_lr_safety_factor(optimizer, scheduler, lr_safety_factor)
                    logger.warning(
                        "bad grad window %d/%d in %d steps; lr safety %.3f -> %.3f applied_lr=%.8g",
                        len(bad_grad_window),
                        bad_grad_window_threshold,
                        bad_grad_window_steps,
                        old_factor,
                        lr_safety_factor,
                        applied_lr,
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                total_streak = nan_streak + grad_explosion_streak
                if total_streak >= 3:
                    old_factor = lr_safety_factor
                    lr_safety_factor = max(0.25, lr_safety_factor * 0.5)
                    last_lr_backoff_step = step
                    reload_path = None
                    recovery_candidates = [
                        latest_path,
                        best_path,
                        run_dir / "interrupted.pt",
                        model_dir / "model_latest.pt",
                        model_dir / "model_best.pt",
                        model_dir / "model_interrupted.pt",
                    ]
                    if resume_path is not None:
                        recovery_candidates.append(resume_path)
                    if config.resume_allow_failed:
                        recovery_candidates.extend([failed_path, model_dir / "model_failed.pt"])
                    for candidate in recovery_candidates:
                        if candidate.exists():
                            reload_path = candidate
                            break
                    if reload_path is not None:
                        logger.warning("bad grad streak %d >= 3; reloading from %s", total_streak, reload_path)
                        load_checkpoint(reload_path, model, optimizer, scheduler, scaler, strict=True)
                        if config.resume_lr_policy == "absolute":
                            _set_scheduler_to_step(scheduler, step)
                        applied_lr = _apply_lr_safety_factor(optimizer, scheduler, lr_safety_factor)
                        logger.warning(
                            "bad grad reload lr safety %.3f -> %.3f applied_lr=%.8g",
                            old_factor,
                            lr_safety_factor,
                            applied_lr,
                        )
                        nan_streak = 0
                        grad_explosion_streak = 0
                        logger.info("reloaded; resuming at step %d", step)
                    else:
                        logger.error(
                            "bad grad streak %d >= 3 but NO checkpoint available to reload; stopping", total_streak
                        )
                        break
                continue

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            _apply_lr_safety_factor(optimizer, scheduler, lr_safety_factor)
            optimizer.zero_grad(set_to_none=True)
            nan_streak = 0
            grad_explosion_streak = 0

            lr = scheduler.get_last_lr()[0]
            tokens = config.batch_size * config.model.max_seq_len * config.grad_accum_steps
            # Report the effective-step loss as the mean of micro-batches.
            # Backward already uses loss / grad_accum_steps above; keeping logs
            # on the same scale avoids inflated loss/ppl when accumulation > 1.
            batch_loss_avg = batch_loss / max(1, config.grad_accum_steps)
            batch_lm_loss_avg = batch_lm_loss / max(1, config.grad_accum_steps)
            metrics["loss_total"] = batch_loss_avg
            metrics["loss_lm"] = batch_lm_loss_avg
            rolling_loss += batch_loss_avg
            rolling_lm_loss += batch_lm_loss_avg
            rolling_tokens += tokens
            rolling_steps += 1

            log_payload = {
                "record_type": "train",
                "step": step,
                "loss": batch_loss_avg,
                "loss_aux": batch_loss_avg - batch_lm_loss_avg,
                "ppl": math.exp(min(20.0, batch_lm_loss_avg)),
                "ppl_lm": math.exp(min(20.0, batch_lm_loss_avg)),
                "ppl_total": math.exp(min(20.0, batch_loss_avg)),
                "lr": lr,
                "tokens": tokens,
                "grad_norm": float(grad_norm),
                **metrics,
            }

            if stop_signals.requested:
                save_metrics = {k: float(v) if isinstance(v, (int, float)) else v for k, v in log_payload.items()}
                save_metrics["stop_reason"] = stop_signals.reason
                logger.warning("stop signal requested (%s); handing checkpoint to save subprocess", stop_signals.reason)
                _save_shutdown_checkpoint(
                    reason_name="interrupted",
                    checkpoint_writer=checkpoint_writer,
                    run_dir=run_dir,
                    model_dir=model_dir,
                    latest_path=latest_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=step,
                    config=config,
                    metrics=save_metrics,
                    logger=logger,
                )
                logger.info("signal checkpoint saved by subprocess; exiting cleanly")
                break

            if eval_loader is not None and (step % config.eval_every == 0 or step == config.max_steps):
                eval_metrics = evaluate_model(
                    model_eval,
                    eval_loader,
                    config.model,
                    device,
                    use_amp,
                    config.eval_max_batches,
                    config.lambda_load,
                    float(w_sparse),
                    float(w_kl * kl_warmup),
                    config.lambda_semantic_pred,
                    float(w_state_pred),
                    float(w_slot_div),
                    config.lambda_slot_stability,
                    float(w_self_pred),
                    float(w_self_slot_div),
                )
                log_payload.update(eval_metrics)
                log_payload["record_type"] = "train_eval"
                log_payload["is_eval_step"] = 1.0
                eval_count += 1
                structural_should_stop = False
                if config.structural_stop and reference_curve:
                    reference = reference_value_at_step(reference_curve, step)
                    if reference is not None:
                        reference_step, reference_val_lm = reference
                        structural_gap = eval_metrics["val_lm_loss"] - reference_val_lm
                        structural_gap_delta = (
                            0.0 if structural_gap_previous is None else structural_gap - structural_gap_previous
                        )
                        is_widening = (
                            step >= config.structural_stop_warmup_steps
                            and eval_count >= config.structural_stop_min_evals
                            and structural_gap >= config.structural_stop_min_gap
                            and structural_gap_delta >= config.structural_stop_widen_delta
                        )
                        structural_widen_count = structural_widen_count + 1 if is_widening else 0
                        structural_gap_previous = structural_gap
                        structural_should_stop = (
                            config.structural_stop_patience > 0
                            and structural_widen_count >= config.structural_stop_patience
                            and step < config.max_steps
                        )
                        log_payload["reference_step"] = float(reference_step)
                        log_payload["reference_val_lm_loss"] = float(reference_val_lm)
                        log_payload["structural_gap"] = float(structural_gap)
                        log_payload["structural_gap_delta"] = float(structural_gap_delta)
                        log_payload["structural_widen_count"] = float(structural_widen_count)
                eval_payload = {
                    "step": step,
                    "val_lm_loss": eval_metrics["val_lm_loss"],
                    "val_ppl_val": eval_metrics["val_ppl"],
                    "val_alpha_downstream_mean": eval_metrics["val_alpha_downstream_mean"],
                    "val_router_entropy": eval_metrics["val_router_entropy"],
                    "best_val_lm_loss": min(best_val_loss, eval_metrics["val_lm_loss"]),
                    "structural_gap": log_payload.get("structural_gap", 0.0),
                    "tokens_per_step": config.batch_size * config.model.max_seq_len * config.grad_accum_steps,
                }
                progress.render_eval(eval_payload)
                logger.info(
                    "val %d | lm %.4f | ppl %.1f | alpha %.3f | ent %.3f | best %.4f | gap %.3f",
                    step,
                    eval_metrics["val_lm_loss"],
                    eval_metrics["val_ppl"],
                    eval_metrics["val_alpha_downstream_mean"],
                    eval_metrics["val_router_entropy"],
                    min(best_val_loss, eval_metrics["val_lm_loss"]),
                    log_payload.get("structural_gap", 0.0),
                )
                improved = eval_metrics["val_lm_loss"] < (best_val_loss - config.early_stop_min_delta)
                if improved:
                    best_val_loss = eval_metrics["val_lm_loss"]
                    stale_eval_count = 0
                    best_metrics = {k: float(v) if isinstance(v, (int, float)) else v for k, v in log_payload.items()}
                    if config.best_checkpoint_mode == "model":
                        best_model_payload = build_model_payload(model, step, config.to_dict(), best_metrics)
                        if checkpoint_writer is None:
                            save_payload(model_dir / "model_best.pt", best_model_payload)
                        else:
                            checkpoint_writer.submit(model_dir / "model_best.pt", best_model_payload)
                            checkpoint_writer.wait()
                    else:
                        save_checkpoint_pair(
                            checkpoint_writer,
                            run_dir / "best.pt",
                            model_dir / "model_best.pt",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            step,
                            config,
                            best_metrics,
                        )
                    logger.info("save best | step %d | val_lm %.4f", step, best_val_loss)
                    progress.render_save(step, is_best=True)
                else:
                    stale_eval_count += 1
                    logger.info(
                        "no validation improvement: best_val_loss=%.4f stale_evals=%d/%d min_delta=%.4g",
                        best_val_loss,
                        stale_eval_count,
                        config.early_stop_patience,
                        config.early_stop_min_delta,
                    )
                    progress.render_save(step, is_best=False)
                should_early_stop = (
                    config.early_stop_patience > 0
                    and eval_count >= config.early_stop_min_evals
                    and stale_eval_count >= config.early_stop_patience
                    and step < config.max_steps
                )
                log_payload["eval_count"] = float(eval_count)
                log_payload["stale_eval_count"] = float(stale_eval_count)
                log_payload["best_val_lm_loss"] = float(best_val_loss)
                if structural_should_stop or should_early_stop:
                    save_metrics = {k: float(v) if isinstance(v, (int, float)) else v for k, v in log_payload.items()}
                    stop_name = "structural_stopped" if structural_should_stop else "early_stopped"
                    full_payload = build_checkpoint_payload(
                        model, optimizer, scheduler, scaler, step, config.to_dict(), save_metrics
                    )
                    model_payload = build_model_payload(model, step, config.to_dict(), save_metrics)
                    if checkpoint_writer is None:
                        save_payload(latest_path, full_payload)
                        save_payload(run_dir / f"{stop_name}.pt", full_payload)
                        save_payload(model_dir / "model_latest.pt", model_payload)
                        save_payload(model_dir / f"model_{stop_name}.pt", model_payload)
                    else:
                        for p in [latest_path, run_dir / f"{stop_name}.pt"]:
                            checkpoint_writer.submit(p, full_payload)
                        checkpoint_writer.wait()
                        for p in [model_dir / "model_latest.pt", model_dir / f"model_{stop_name}.pt"]:
                            checkpoint_writer.submit(p, model_payload)
                        checkpoint_writer.wait()
                    if structural_should_stop:
                        logger.warning(
                            "structural stop triggered at step=%d gap=%.4f widen_count=%d/%d reference=%s",
                            step,
                            log_payload.get("structural_gap", 0.0),
                            structural_widen_count,
                            config.structural_stop_patience,
                            config.reference_metrics_path,
                        )
                    else:
                        logger.info(
                            "early stopping triggered at step=%d best_val_loss=%.4f stale_evals=%d",
                            step,
                            best_val_loss,
                            stale_eval_count,
                        )
                    metrics_logger.write(log_payload)
                    break
            metrics_logger.write(log_payload)

            if step % config.log_every == 0 or step == 1:
                now = time.time()
                elapsed = max(1e-6, now - last_log_time)
                tok_s = rolling_tokens / elapsed
                avg_lm_loss = rolling_lm_loss / max(1, rolling_steps)
                tokens_per_step = config.batch_size * config.model.max_seq_len * config.grad_accum_steps
                progress_payload = {
                    "step": step,
                    "lm": avg_lm_loss,
                    "ppl": math.exp(min(20.0, avg_lm_loss)),
                    "alpha_downstream_mean": metrics.get("alpha_downstream_mean", 0.0),
                    "router_entropy": metrics.get("router_entropy", 0.0),
                    "grad_norm": float(grad_norm),
                    "lr": lr,
                    "lambda_sparse_effective": metrics.get("lambda_sparse_effective", 0.0),
                    "v5_slot_confidence": metrics.get("v5_slot_confidence", 0.0),
                    "v5_slot_cosine": metrics.get("v5_slot_cosine", 0.0),
                    "v5_slot_read_entropy": metrics.get("v5_slot_read_entropy", 0.0),
                    "v5_state_pred": metrics.get("v5_state_pred", 0.0),
                    "v5_slot_diversity": metrics.get("v5_slot_diversity", 0.0),
                    "v5_slot_stability": metrics.get("v5_slot_stability", 0.0),
                    "v5_slot_delta": metrics.get("v5_slot_delta", 0.0),
                    "v5_slot_update_gate": metrics.get("v5_slot_update_gate", 0.0),
                    "v5_slot_write_entropy": metrics.get("v5_slot_write_entropy", 0.0),
                    "v5_slot_write_max": metrics.get("v5_slot_write_max", 0.0),
                    "v5_slot_write_min": metrics.get("v5_slot_write_min", 0.0),
                    "v5_slot_write_active": metrics.get("v5_slot_write_active", 0.0),
                    "v6_self_pred": metrics.get("v6_self_pred", 0.0),
                    "v6_slot_diversity": metrics.get("v6_slot_diversity", 0.0),
                    "v6_slot_cosine": metrics.get("v6_slot_cosine", 0.0),
                    "v6_slot_context_cosine": metrics.get("v6_slot_context_cosine", 0.0),
                    "v6_state_delta": metrics.get("v6_state_delta", 0.0),
                    "v6_state_norm": metrics.get("v6_state_norm", 0.0),
                    "v6_reflection_norm": metrics.get("v6_reflection_norm", 0.0),
                    "v6_boundary_entropy": metrics.get("v6_boundary_entropy", 0.0),
                    "v6_boundary_self": metrics.get("v6_boundary_self", 0.0),
                    "v6_boundary_world": metrics.get("v6_boundary_world", 0.0),
                    "v6_boundary_other": metrics.get("v6_boundary_other", 0.0),
                    "v6_boundary_unknown": metrics.get("v6_boundary_unknown", 0.0),
                    "gate_mix_alpha_weight": metrics.get("gate_mix_alpha_weight", 0.0),
                    "gate_mix_clean_weight": metrics.get("gate_mix_clean_weight", 0.0),
                    "gate_mix_state_weight": metrics.get("gate_mix_state_weight", 0.0),
                    "v4_state_confidence": metrics.get("v4_state_confidence", 0.0),
                    "v4_state_delta": metrics.get("v4_state_delta", 0.0),
                    "v4_memory_norm": metrics.get("v4_memory_norm", 0.0),
                    "v4_memory_read_strength": metrics.get("v4_memory_read_strength", 0.0),
                    "v4_memory_novelty": metrics.get("v4_memory_novelty", 0.0),
                    "tok_s": tok_s,
                    "tokens_per_step": tokens_per_step,
                }
                progress.render_step(progress_payload)
                extra_metrics = (
                    " | v6 pred %.4f cos %.3f ctx %.3f dlt %.4f refl %.3f bnd %.3f/%.3f/%.3f/%.3f"
                    if "v6" in config.architecture
                    else ""
                )
                logger.info(
                    (
                        "tr %d/%d | lm %.4f | ppl %.1f | alpha %.3f | ent %.3f | tok %.0f/s | grad %.2f | "
                        "wt_ent %.3f wt_m %.3f wt_n %.3f wt_a %.1f conf %.3f cos %.3f "
                        "mix_a %d%% mix_c %d%% mix_s %d%%" + extra_metrics
                    ),
                    step,
                    config.max_steps,
                    avg_lm_loss,
                    math.exp(min(20.0, avg_lm_loss)),
                    metrics.get("alpha_downstream_mean", 0.0),
                    metrics.get("router_entropy", 0.0),
                    tok_s,
                    float(grad_norm),
                    metrics.get("v5_slot_write_entropy", 0.0),
                    metrics.get("v5_slot_write_max", 0.0),
                    metrics.get("v5_slot_write_min", 0.0),
                    metrics.get("v5_slot_write_active", 0.0),
                    metrics.get("v5_slot_confidence", 0.0),
                    metrics.get("v5_slot_cosine", 0.0),
                    int(metrics.get("gate_mix_alpha_weight", 0.0) * 100),
                    int(metrics.get("gate_mix_clean_weight", 0.0) * 100),
                    int(metrics.get("gate_mix_state_weight", 0.0) * 100),
                    *(
                        (
                            metrics.get("v6_self_pred", 0.0),
                            metrics.get("v6_slot_cosine", 0.0),
                            metrics.get("v6_slot_context_cosine", 0.0),
                            metrics.get("v6_state_delta", 0.0),
                            metrics.get("v6_reflection_norm", 0.0),
                            metrics.get("v6_boundary_self", 0.0),
                            metrics.get("v6_boundary_world", 0.0),
                            metrics.get("v6_boundary_other", 0.0),
                            metrics.get("v6_boundary_unknown", 0.0),
                        )
                        if "v6" in config.architecture
                        else ()
                    ),
                )
                rolling_loss = 0.0
                rolling_lm_loss = 0.0
                rolling_tokens = 0
                rolling_steps = 0
                last_log_time = now

            should_save_step = (config.save_every > 0 and step % config.save_every == 0) or step == config.max_steps
            should_save_latest = (
                (config.latest_every > 0 and step % config.latest_every == 0)
                or should_save_step
                or step == config.max_steps
            )
            if should_save_step or should_save_latest:
                save_metrics = {k: float(v) if isinstance(v, (int, float)) else v for k, v in log_payload.items()}
                full_payload = build_checkpoint_payload(
                    model, optimizer, scheduler, scaler, step, config.to_dict(), save_metrics
                )
                model_payload = build_model_payload(model, step, config.to_dict(), save_metrics)
                if checkpoint_writer is None:
                    save_payload(latest_path, full_payload)
                    save_payload(model_dir / "model_latest.pt", model_payload)
                else:
                    checkpoint_writer.submit(latest_path, full_payload)
                    checkpoint_writer.submit(model_dir / "model_latest.pt", model_payload)
                    if config.latest_sync:
                        checkpoint_writer.wait()
                logger.info("save ckpt | step %d | %s", step, latest_path.name)
                progress.render_save(step)

            should_check_stop = config.stop_check_every > 0 and step % config.stop_check_every == 0
            if should_check_stop and stop_path.exists():
                save_metrics = {k: float(v) if isinstance(v, (int, float)) else v for k, v in log_payload.items()}
                full_payload = build_checkpoint_payload(
                    model, optimizer, scheduler, scaler, step, config.to_dict(), save_metrics
                )
                model_payload = build_model_payload(model, step, config.to_dict(), save_metrics)
                if checkpoint_writer is None:
                    save_payload(latest_path, full_payload)
                    save_payload(run_dir / "stopped.pt", full_payload)
                    save_payload(model_dir / "model_latest.pt", model_payload)
                    save_payload(model_dir / "model_stopped.pt", model_payload)
                else:
                    for p in [latest_path, run_dir / "stopped.pt"]:
                        checkpoint_writer.submit(p, full_payload)
                    checkpoint_writer.wait()
                    for p in [model_dir / "model_latest.pt", model_dir / "model_stopped.pt"]:
                        checkpoint_writer.submit(p, model_payload)
                    checkpoint_writer.wait()
                logger.warning("graceful stop requested by %s at step=%d; checkpoint saved", stop_path, step)
                progress.render_save(step)
                break

    except KeyboardInterrupt:
        progress.finalize()
        logger.warning("interrupted; handing checkpoint to save subprocess")
        _save_shutdown_checkpoint(
            reason_name="interrupted",
            checkpoint_writer=checkpoint_writer,
            run_dir=run_dir,
            model_dir=model_dir,
            latest_path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            config=config,
            metrics={"stop_reason": "KeyboardInterrupt"},
            logger=logger,
        )
        logger.info("interrupt checkpoint saved by subprocess; exiting cleanly")
        return
    except Exception:
        progress.finalize()
        logger.exception("training failed; saving failure checkpoint")
        if checkpoint_writer is not None:
            try:
                checkpoint_writer.wait()
            except Exception:
                logger.exception("checkpoint writer failed while flushing before failure checkpoint")
        failed_step = locals().get("step", start_step)
        save_checkpoint(run_dir / "failed.pt", model, optimizer, scheduler, scaler, failed_step, config.to_dict(), {})
        raise
    finally:
        stop_signals.restore()
        if checkpoint_writer is not None:
            active_exception = sys.exc_info()[0] is not None
            try:
                checkpoint_writer.close()
            except Exception:
                logger.exception("checkpoint writer close failed")
                if not active_exception:
                    raise
        csv_path = metrics_jsonl_to_csv(run_dir / "metrics.jsonl")
        if csv_path is not None:
            logger.info("metrics csv saved | %s", csv_path.name)

    progress.finalize()
    logger.info("training complete")


if __name__ == "__main__":
    main()
