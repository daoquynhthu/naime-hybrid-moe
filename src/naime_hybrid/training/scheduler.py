import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    warmup_steps = max(1, warmup_steps)
    max_steps = max(warmup_steps + 1, max_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def cosine_with_restarts(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float,
    cycle_length: int,
    restart_ratio: float,
    restart_warmup: int,
) -> LambdaLR:
    warmup_steps = max(1, warmup_steps)
    max_steps = max(warmup_steps + 1, max_steps)
    cycle_length = max(warmup_steps + 1, cycle_length)
    restart_ratio = max(min_lr_ratio, min(1.0, restart_ratio))
    restart_warmup = max(1, min(restart_warmup, cycle_length - 1))

    def _decay(progress: float, peak: float) -> float:
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (peak - min_lr_ratio) * cosine

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        effective = step - warmup_steps
        cycle = effective // cycle_length
        position = effective % cycle_length

        if cycle == 0:
            return _decay(position / cycle_length, 1.0)

        if position < restart_warmup:
            frac = float(position) / float(restart_warmup)
            return min_lr_ratio + (restart_ratio - min_lr_ratio) * frac

        decay_progress = (position - restart_warmup) / max(1, cycle_length - restart_warmup)
        return _decay(decay_progress, restart_ratio)

    return LambdaLR(optimizer, lr_lambda)
