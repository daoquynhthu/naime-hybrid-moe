from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NAIMEStatePacket:
    """Portable latent state packet for stateful NAIME decoding.

    The packet deliberately stores only compact model-owned latent state. It is
    not a KV cache, not hidden activations, and not a replayable computation
    graph. Public packet tensors must use the compact final-slot form
    ``[batch, slots, dim]`` so carried state has a single stable ingress
    protocol across V5/V6/V7 paths.
    """

    world_state: torch.Tensor | None = None
    self_state: torch.Tensor | None = None
    latent_field: torch.Tensor | None = None
    memory: torch.Tensor | None = None
    controller_state: torch.Tensor | None = None
    state_version: int = 1
    protocol_version: str = "state-protocol-v1"
    architecture_id: str = "naime"
    causal_integrity_version: int = 2
    tokenizer_hash: str | None = None
    created_step: int | None = None
    confidence: float | None = None

    def detach(self) -> NAIMEStatePacket:
        return NAIMEStatePacket(
            world_state=_detach_optional(self.world_state),
            self_state=_detach_optional(self.self_state),
            latent_field=_detach_optional(self.latent_field),
            memory=_detach_optional(self.memory),
            controller_state=_detach_optional(self.controller_state),
            state_version=self.state_version,
            protocol_version=self.protocol_version,
            architecture_id=self.architecture_id,
            causal_integrity_version=self.causal_integrity_version,
            tokenizer_hash=self.tokenizer_hash,
            created_step=self.created_step,
            confidence=self.confidence,
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> NAIMEStatePacket:
        return NAIMEStatePacket(
            world_state=_to_optional(self.world_state, device=device, dtype=dtype),
            self_state=_to_optional(self.self_state, device=device, dtype=dtype),
            latent_field=_to_optional(self.latent_field, device=device, dtype=dtype),
            memory=_to_optional(self.memory, device=device, dtype=dtype),
            controller_state=_to_optional(self.controller_state, device=device, dtype=dtype),
            state_version=self.state_version,
            protocol_version=self.protocol_version,
            architecture_id=self.architecture_id,
            causal_integrity_version=self.causal_integrity_version,
            tokenizer_hash=self.tokenizer_hash,
            created_step=self.created_step,
            confidence=self.confidence,
        )

    def validate_batch(self, batch_size: int) -> None:
        for name, value in (
            ("world_state", self.world_state),
            ("self_state", self.self_state),
            ("latent_field", self.latent_field),
            ("memory", self.memory),
            ("controller_state", self.controller_state),
        ):
            if value is None:
                continue
            if value.ndim != 3:
                raise ValueError(
                    f"{name} must use compact final-slot shape [batch, slots, dim]; got ndim={value.ndim}"
                )
            if value.size(0) != batch_size:
                raise ValueError(f"{name} batch mismatch: expected {batch_size}, got {value.size(0)}")


@dataclass(frozen=True)
class ObservationPacket:
    """Typed causal observation container for future multimodal NAIME work.

    This is an interface contract, not a new model path. Encoders for text,
    image, video, audio, tools, or sensors should produce observation packets
    before writing into NAIME state. Observations are allowed to update outgoing
    state; they must not bypass the state protocol with independent hidden
    authorities.
    """

    modality: str
    embeddings: torch.Tensor
    attention_mask: torch.Tensor | None = None
    time_index: torch.Tensor | None = None
    spatial_anchors: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    provenance: str = "unknown"
    causal_segment_id: str | None = None

    def detach(self) -> ObservationPacket:
        return ObservationPacket(
            modality=self.modality,
            embeddings=self.embeddings.detach(),
            attention_mask=_detach_optional(self.attention_mask),
            time_index=_detach_optional(self.time_index),
            spatial_anchors=_detach_optional(self.spatial_anchors),
            confidence=_detach_optional(self.confidence),
            provenance=self.provenance,
            causal_segment_id=self.causal_segment_id,
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ObservationPacket:
        return ObservationPacket(
            modality=self.modality,
            embeddings=_to_optional(self.embeddings, device=device, dtype=dtype),  # type: ignore[arg-type]
            attention_mask=_to_optional(self.attention_mask, device=device, dtype=None),
            time_index=_to_optional(self.time_index, device=device, dtype=None),
            spatial_anchors=_to_optional(self.spatial_anchors, device=device, dtype=dtype),
            confidence=_to_optional(self.confidence, device=device, dtype=dtype),
            provenance=self.provenance,
            causal_segment_id=self.causal_segment_id,
        )

    def validate_batch(self, batch_size: int) -> None:
        for name, value in (
            ("embeddings", self.embeddings),
            ("attention_mask", self.attention_mask),
            ("time_index", self.time_index),
            ("spatial_anchors", self.spatial_anchors),
            ("confidence", self.confidence),
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
