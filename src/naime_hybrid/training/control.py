import json
import math
from pathlib import Path


def update_sparse_lambda(
    current_lambda: float,
    alpha_ema: float | None,
    alpha_mean: float,
    target_sparsity: float,
    ema_decay: float = 0.95,
    gain: float = 0.1,
    min_value: float = 1e-4,
    max_value: float = 1.0,
    deadband: float = 0.02,
) -> tuple[float, float]:
    """Adapt sparse regularization strength from the alpha-target gap.

    This controller is kept for compatibility with older experiments and
    tests. Current V5 training uses fixed auxiliary weights unless explicitly
    re-enabled by a future trainer.
    """
    alpha_ema = alpha_mean if alpha_ema is None else ema_decay * alpha_ema + (1.0 - ema_decay) * alpha_mean
    gap = abs(alpha_ema - target_sparsity)
    if gap <= deadband:
        next_lambda = current_lambda * (1.0 - gain)
    else:
        next_lambda = current_lambda * (1.0 + gain * gap)
    next_lambda = min(max(next_lambda, min_value), max_value)
    return next_lambda, alpha_ema


def effective_kl_lambda(base_lambda: float, step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lambda
    return base_lambda * min(1.0, step / max(1, warmup_steps))


def load_reference_curve(path: str | Path | None, metric_name: str = "val_lm_loss") -> list[tuple[int, float]]:
    if path is None:
        return []
    metrics_path = Path(path)
    if not metrics_path.exists():
        return []

    curve: list[tuple[int, float]] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if metric_name not in payload or "step" not in payload:
                continue
            value = payload[metric_name]
            step_value = payload["step"]
            if isinstance(value, (int, float)) and isinstance(step_value, (int, float)):
                value = float(value)
                if math.isfinite(value):
                    curve.append((int(step_value), value))
    return sorted(curve)


def reference_value_at_step(curve: list[tuple[int, float]], step: int) -> tuple[int, float] | None:
    if not curve:
        return None
    selected = curve[0]
    for ref_step, value in curve:
        if ref_step > step:
            break
        selected = (ref_step, value)
    return selected
