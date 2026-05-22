from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NAIMEStatePacket:
    """Portable latent state packet for stateful NAIME decoding.

    The packet deliberately stores only compact model-owned latent state. It is
    not a KV cache, not hidden activations, and not a replayable computation
    graph. ``world_state`` and ``self_state`` may be either final slot banks
    ``[batch, slots, dim]`` or compact causal slot traces
    ``[batch, blocks, slots, dim]``.
    """

    world_state: torch.Tensor | None = None
    self_state: torch.Tensor | None = None
    memory: torch.Tensor | None = None
    state_version: int = 1
    architecture_id: str = "naime"
    causal_integrity_version: int = 2
    created_step: int | None = None
    confidence: float | None = None

    def detach(self) -> "NAIMEStatePacket":
        return NAIMEStatePacket(
            world_state=_detach_optional(self.world_state),
            self_state=_detach_optional(self.self_state),
            memory=_detach_optional(self.memory),
            state_version=self.state_version,
            architecture_id=self.architecture_id,
            causal_integrity_version=self.causal_integrity_version,
            created_step=self.created_step,
            confidence=self.confidence,
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "NAIMEStatePacket":
        return NAIMEStatePacket(
            world_state=_to_optional(self.world_state, device=device, dtype=dtype),
            self_state=_to_optional(self.self_state, device=device, dtype=dtype),
            memory=_to_optional(self.memory, device=device, dtype=dtype),
            state_version=self.state_version,
            architecture_id=self.architecture_id,
            causal_integrity_version=self.causal_integrity_version,
            created_step=self.created_step,
            confidence=self.confidence,
        )

    def validate_batch(self, batch_size: int) -> None:
        for name, value in (
            ("world_state", self.world_state),
            ("self_state", self.self_state),
            ("memory", self.memory),
        ):
            if value is not None and value.size(0) != batch_size:
                raise ValueError(f"{name} batch mismatch: expected {batch_size}, got {value.size(0)}")


def _detach_optional(value: torch.Tensor | None) -> torch.Tensor | None:
    return value.detach() if value is not None else None


def _to_optional(
    value: torch.Tensor | None,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> torch.Tensor | None:
    if value is None:
        return None
    kwargs: dict[str, torch.device | str | torch.dtype] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    return value.to(**kwargs) if kwargs else value
