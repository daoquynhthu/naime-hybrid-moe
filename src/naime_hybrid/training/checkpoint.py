import queue
import random
import signal
import threading
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _snapshot_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _snapshot_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_snapshot_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_value(v) for v in value)
    return value


def _save_payload(path: Path, payload: dict[str, Any], retries: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        tmp = path.with_name(f"{path.name}.{threading.get_ident()}.{attempt}.tmp")
        try:
            torch.save(payload, tmp)
            tmp.replace(path)
            return
        except BaseException as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt >= retries:
                break
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed to save checkpoint payload to {path}") from last_error


def _ignore_process_stop_signals() -> None:
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            signal.signal(signum, signal.SIG_IGN)
        except (OSError, ValueError):
            continue


def _save_payloads_worker(jobs: list[tuple[str, dict[str, Any]]]) -> None:
    _ignore_process_stop_signals()
    for path_str, payload in jobs:
        _save_payload(Path(path_str), payload)


def save_payloads_in_subprocess(jobs: list[tuple[Path, dict[str, Any]]]) -> None:
    """Persist payloads in an isolated process for emergency shutdown.

    The parent process still builds a CPU snapshot, because only the training
    process owns live model and optimizer state. Once the child starts, disk IO
    is isolated from the parent and the child ignores ordinary console stop
    signals, so a second Ctrl+C is much less likely to corrupt checkpoint files.
    """
    if not jobs:
        return
    serializable_jobs = [(str(path), payload) for path, payload in jobs]
    ctx = get_context("spawn")
    process = ctx.Process(target=_save_payloads_worker, args=(serializable_jobs,))
    process.daemon = False
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"checkpoint subprocess failed with exit code {process.exitcode}")


def build_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    step: int,
    config: dict,
    metrics: dict,
) -> dict[str, Any]:
    return {
        "step": step,
        "model": _snapshot_value(model.state_dict()),
        "optimizer": _snapshot_value(optimizer.state_dict()),
        "scheduler": _snapshot_value(scheduler.state_dict() if scheduler is not None else None),
        "scaler": _snapshot_value(scaler.state_dict() if scaler is not None else None),
        "config": config,
        "metrics": metrics,
        "rng": _snapshot_value(rng_state()),
    }


def build_model_payload(
    model: torch.nn.Module,
    step: int,
    config: dict,
    metrics: dict,
) -> dict[str, Any]:
    return {
        "step": step,
        "model": _snapshot_value(model.state_dict()),
        "config": config,
        "metrics": metrics,
    }


class AsyncCheckpointWriter:
    """Single-worker async disk writer for checkpoint payloads.

    The training loop still snapshots model state on the main thread to avoid
    racing with parameter updates, but slow disk writes run in the background.
    """

    def __init__(self, max_queue: int = 2, submit_timeout: float = 120.0):
        self._queue: queue.Queue[tuple[Path, dict[str, Any]] | None] = queue.Queue(maxsize=max_queue)
        self._error: BaseException | None = None
        self._submit_timeout = submit_timeout
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="checkpoint-writer", daemon=False)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, payload = item
                _save_payload(path, payload)
            except BaseException as exc:
                self._error = exc
            finally:
                self._queue.task_done()

    def submit(self, path: Path, payload: dict[str, Any]) -> None:
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError("async checkpoint writer is closed")
        try:
            self._queue.put((path, payload), timeout=self._submit_timeout)
        except queue.Full as exc:
            raise RuntimeError(f"async checkpoint writer queue stayed full for {self._submit_timeout:.0f}s") from exc
        self.raise_if_failed()

    def wait(self) -> None:
        self._queue.join()
        self.raise_if_failed()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.wait()
        finally:
            self._queue.put(None)
            self._thread.join()
            self._closed = True
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("async checkpoint writer failed") from self._error


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    step: int,
    config: dict,
    metrics: dict,
) -> None:
    _save_payload(path, build_checkpoint_payload(model, optimizer, scheduler, scaler, step, config, metrics))


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    _save_payload(path, payload)


def save_model_weights(
    path: Path,
    model: torch.nn.Module,
    step: int,
    config: dict,
    metrics: dict,
) -> None:
    _save_payload(path, build_model_payload(model, step, config, metrics))


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    strict: bool = True,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=strict)
    step = int(checkpoint.get("step", 0))
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    elif scheduler is not None and step > 0 and hasattr(scheduler, "lr_lambdas"):
        # Model-only checkpoints intentionally omit optimizer/scheduler state.
        # Keep LR on the same cosine schedule rather than restarting warmup.
        scheduler.last_epoch = step - 1
        scheduler._step_count = max(getattr(scheduler, "_step_count", 0), step)
        for group, base_lr, lr_lambda in zip(
            scheduler.optimizer.param_groups,
            scheduler.base_lrs,
            scheduler.lr_lambdas,
            strict=False,
        ):
            group["lr"] = base_lr * lr_lambda(step)
        scheduler._last_lr = [group["lr"] for group in scheduler.optimizer.param_groups]
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if checkpoint.get("rng") is not None:
        load_rng_state(checkpoint["rng"])
    return step


def prune_checkpoints(run_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    checkpoints = sorted(run_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    excess = len(checkpoints) - keep_last_n
    for path in checkpoints[: max(0, excess)]:
        path.unlink(missing_ok=True)


def prune_model_weights(model_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    weights = sorted(model_dir.glob("model_step_*.pt"), key=lambda p: p.stat().st_mtime)
    excess = len(weights) - keep_last_n
    for path in weights[: max(0, excess)]:
        path.unlink(missing_ok=True)
