import torch
from torch import nn

from naime_hybrid.config import NAIMEStateMoEConfig
from naime_hybrid.modules.blocks import (
    DenseTransformerBlock,
    NAIMEStateMoEBlock,
    NAIMEV4StateMoEBlock,
    NAIMEV5WorldStateMoEBlock,
    TokenMoEBlock,
)
from naime_hybrid.modules.latent_field import LatentFieldCoupler
from naime_hybrid.modules.norm import RMSNorm
from naime_hybrid.modules.self_state import RecursiveSelfState
from naime_hybrid.modules.state import CrossLayerSemanticState
from naime_hybrid.modules.world_state import WorldStateSlots
from naime_hybrid.models.state_packet import NAIMEStatePacket


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def _resolve_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: NAIMEStateMoEConfig,
    infer_pad_mask: bool | None,
) -> torch.Tensor | None:
    if attention_mask is not None:
        return attention_mask
    if infer_pad_mask is False:
        return None
    return input_ids.ne(config.pad_token_id)


def _public_state(state: torch.Tensor | None) -> torch.Tensor | None:
    return state[:, -1, :, :] if state is not None and state.ndim == 4 else state


def _resolve_state_packet(
    past_state: NAIMEStatePacket | None,
    *,
    past_world_state: torch.Tensor | None,
    past_self_state: torch.Tensor | None,
    past_memory: torch.Tensor | None,
    detach_past_state: bool,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if past_state is not None:
        state = past_state.detach() if detach_past_state else past_state
        state = state.to(device=device, dtype=dtype)
        state.validate_batch(batch_size)
        world_state = state.world_state
        self_state = state.self_state
        memory = state.memory
    else:
        world_state = past_world_state
        self_state = past_self_state
        memory = past_memory
        if detach_past_state:
            world_state = world_state.detach() if world_state is not None else None
            self_state = self_state.detach() if self_state is not None else None
            memory = memory.detach() if memory is not None else None
        world_state = world_state.to(device=device, dtype=dtype) if world_state is not None else None
        self_state = self_state.to(device=device, dtype=dtype) if self_state is not None else None
        memory = memory.to(device=device, dtype=dtype) if memory is not None else None
        NAIMEStatePacket(world_state=world_state, self_state=self_state, memory=memory).validate_batch(batch_size)
    return world_state, self_state, memory


class NAIMEStateMoEDecoder(nn.Module):
    def __init__(self, config: NAIMEStateMoEConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            if layer_idx < config.n_dense_layers:
                self.blocks.append(DenseTransformerBlock(config))
            else:
                self.blocks.append(NAIMEStateMoEBlock(config))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(_init_weights)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tau: float | None = None,
        return_aux: bool = True,
        return_logits: bool = True,
        infer_pad_mask: bool | None = None,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        attention_mask = _resolve_attention_mask(input_ids, attention_mask, self.config, infer_pad_mask)

        hidden_states = self.embed_tokens(input_ids)
        aux_by_layer = []
        for block in self.blocks:
            if isinstance(block, NAIMEStateMoEBlock):
                hidden_states, aux = block(hidden_states, attention_mask=attention_mask, tau=tau)
            else:
                hidden_states, aux = block(hidden_states, attention_mask=attention_mask)
            if return_aux:
                aux_by_layer.append(aux)

        hidden_states = self.norm(hidden_states)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "hidden_states": hidden_states,
        }
        if return_logits:
            output["logits"] = self.lm_head(hidden_states)
        if return_aux:
            output["aux"] = aux_by_layer
        return output


class NAIMEV4StateMoEDecoder(nn.Module):
    """V4 decoder with recurrent semantic state and per-forward working memory."""

    def __init__(self, config: NAIMEStateMoEConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            if layer_idx < config.n_dense_layers:
                self.blocks.append(DenseTransformerBlock(config))
            else:
                self.blocks.append(NAIMEV4StateMoEBlock(config, layer_idx=layer_idx))
        self.semantic_state = CrossLayerSemanticState(
            config.d_model,
            confidence_mode=config.semantic_state_confidence_mode,
            confidence_temperature=config.semantic_state_confidence_temperature,
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(_init_weights)
        self.lm_head.weight = self.embed_tokens.weight

    def _initial_memory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.config.semantic_memory_slots <= 0:
            return None
        for block in self.blocks:
            if isinstance(block, NAIMEV4StateMoEBlock) and block.memory is not None:
                return block.memory.initial_memory(batch_size, device, dtype)
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tau: float | None = None,
        return_aux: bool = True,
        return_logits: bool = True,
        infer_pad_mask: bool | None = None,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        attention_mask = _resolve_attention_mask(input_ids, attention_mask, self.config, infer_pad_mask)

        hidden_states = self.embed_tokens(input_ids)
        batch_size = hidden_states.size(0)
        semantic_state = self.semantic_state.initial_state(batch_size, hidden_states.device, hidden_states.dtype)
        semantic_state_confidence = torch.ones(batch_size, 1, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        memory = self._initial_memory(batch_size, hidden_states.device, hidden_states.dtype)

        aux_by_layer = []
        for block in self.blocks:
            if isinstance(block, NAIMEV4StateMoEBlock):
                hidden_states, aux, semantic_summary, memory = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    tau=tau,
                    semantic_state=semantic_state,
                    semantic_state_confidence=semantic_state_confidence,
                    memory=memory,
                )
                if semantic_summary is not None:
                    semantic_state, state_gate, semantic_state_confidence, state_delta, state_agreement = (
                        self.semantic_state(
                            semantic_state,
                            semantic_summary,
                        )
                    )
                    aux["v4"]["state_gate"] = state_gate.mean()
                    aux["v4"]["state_confidence"] = semantic_state_confidence.mean()
                    aux["v4"]["state_delta"] = state_delta.mean()
                    aux["v4"]["state_agreement"] = state_agreement.mean()
            else:
                hidden_states, aux = block(hidden_states, attention_mask=attention_mask)
            if return_aux:
                aux_by_layer.append(aux)

        hidden_states = self.norm(hidden_states)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "hidden_states": hidden_states,
        }
        if return_logits:
            output["logits"] = self.lm_head(hidden_states)
        if return_aux:
            output["aux"] = aux_by_layer
        return output


class NAIMEV5WorldStateMoEDecoder(NAIMEV4StateMoEDecoder):
    """V5 decoder with structured world-state slots shared across semantic layers."""

    def __init__(self, config: NAIMEStateMoEConfig):
        super().__init__(config)
        slots = config.world_state_slots or max(1, config.semantic_memory_slots)
        shared_world_state = WorldStateSlots(
            config.d_model,
            slots=slots,
            diversity_margin=config.world_state_diversity_margin,
            stability_threshold=config.world_state_stability_threshold,
            write_top_k=config.world_state_write_top_k,
            pred_detach_target=config.world_state_pred_detach_target,
        )
        self.blocks = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            if layer_idx < config.n_dense_layers:
                self.blocks.append(DenseTransformerBlock(config))
            else:
                self.blocks.append(
                    NAIMEV5WorldStateMoEBlock(config, layer_idx=layer_idx, world_state_slots=shared_world_state)
                )
        self.world_state_slots = shared_world_state
        self.apply(_init_weights)
        self.lm_head.weight = self.embed_tokens.weight

    def _initial_world_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        return self.world_state_slots.initial_state(batch_size, device, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tau: float | None = None,
        return_aux: bool = True,
        return_logits: bool = True,
        infer_pad_mask: bool | None = None,
        past_state: NAIMEStatePacket | None = None,
        past_world_state: torch.Tensor | None = None,
        past_memory: torch.Tensor | None = None,
        detach_past_state: bool = True,
        return_state: bool = False,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        attention_mask = _resolve_attention_mask(input_ids, attention_mask, self.config, infer_pad_mask)

        hidden_states = self.embed_tokens(input_ids)
        batch_size = hidden_states.size(0)
        world_state, _, memory = _resolve_state_packet(
            past_state,
            past_world_state=past_world_state,
            past_self_state=None,
            past_memory=past_memory,
            detach_past_state=detach_past_state,
            batch_size=batch_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if world_state is None:
            world_state = self._initial_world_state(batch_size, hidden_states.device, hidden_states.dtype)
        if memory is None:
            memory = self._initial_memory(batch_size, hidden_states.device, hidden_states.dtype)

        aux_by_layer = []
        for block in self.blocks:
            if isinstance(block, NAIMEV5WorldStateMoEBlock):
                hidden_states, aux, world_state, memory = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    tau=tau,
                    world_state=world_state,
                    memory=memory,
                )
            else:
                hidden_states, aux = block(hidden_states, attention_mask=attention_mask)
            if return_aux:
                aux_by_layer.append(aux)

        hidden_states = self.norm(hidden_states)
        public_world_state = _public_state(world_state)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "hidden_states": hidden_states,
            "world_state": public_world_state,
        }
        if memory is not None:
            output["memory"] = memory
        if return_state:
            output["state_packet"] = NAIMEStatePacket(
                world_state=world_state,
                memory=memory,
                architecture_id="naime_v5_world_state_moe",
            )
        if return_logits:
            output["logits"] = self.lm_head(hidden_states)
        if return_aux:
            output["aux"] = aux_by_layer
        return output


class NAIMEV6RecursiveSelfMoEDecoder(NAIMEV5WorldStateMoEDecoder):
    """V6 decoder with recursive self-state slots over the V5 world model."""

    def __init__(self, config: NAIMEStateMoEConfig):
        super().__init__(config)
        slots = config.self_state_slots or max(1, config.world_state_slots or config.semantic_memory_slots or 4)
        self.self_state_slots = RecursiveSelfState(
            config.d_model,
            slots=slots,
            recursion_depth=config.self_state_recursion_depth,
            write_scale=config.self_state_write_scale,
            hidden_scale=config.self_state_hidden_scale,
            boundary_temperature=config.self_state_boundary_temperature,
            diversity_margin=config.self_state_diversity_margin,
            identity_scale=config.self_state_identity_scale,
            context_score_scale=config.self_state_context_score_scale,
            pred_detach_target=config.self_state_pred_detach_target,
            world_gate=config.self_state_world_gate,
            world_gate_min=config.self_state_world_gate_min,
            world_gate_scale=config.self_state_world_gate_scale,
            latent_thought_steps=config.latent_thought_steps,
            latent_thought_write_mode=config.latent_thought_write_mode,
            latent_thought_hidden_scale=config.latent_thought_hidden_scale,
        )
        self.latent_field = (
            LatentFieldCoupler(
                config.d_model,
                token_scale=config.latent_field_token_scale,
                max_ratio=config.latent_field_max_ratio,
            )
            if config.latent_field_coupling
            else None
        )
        self.self_state_slots.apply(_init_weights)
        if self.latent_field is not None:
            self.latent_field.apply(_init_weights)

    def _initial_self_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.self_state_slots.initial_state(batch_size, device, dtype)

    def _first_memory_module(self):
        for block in self.blocks:
            if isinstance(block, NAIMEV5WorldStateMoEBlock) and block.memory is not None:
                return block.memory
        return None

    def _final_state_slots(self, state: torch.Tensor | None) -> torch.Tensor | None:
        return state[:, -1, :, :] if state is not None and state.ndim == 4 else state

    def _append_state_trace(
        self,
        state: torch.Tensor | None,
        next_slots: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if next_slots is None:
            return state
        if state is None:
            return next_slots.unsqueeze(1)
        if state.ndim == 4:
            return torch.cat([state, next_slots.unsqueeze(1)], dim=1)
        return torch.stack([state, next_slots], dim=1)

    def _evolve_internal_state(
        self,
        *,
        hidden_states: torch.Tensor,
        world_state: torch.Tensor | None,
        self_state: torch.Tensor | None,
        memory: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:
        steps = int(self.config.state_evolution_steps)
        zero = hidden_states.new_tensor(0.0)
        metrics = {
            "state_evolution_delta": zero,
            "state_evolution_world_delta": zero,
            "state_evolution_self_delta": zero,
            "state_evolution_memory_delta": zero,
            "state_evolution_steps": zero,
        }
        if steps <= 0:
            return world_state, self_state, memory, metrics

        current_world = self._final_state_slots(world_state)
        current_self = self._final_state_slots(self_state)
        current_memory = memory
        if current_world is None or current_self is None:
            return world_state, self_state, memory, metrics

        memory_module = self._first_memory_module() if self.config.state_evolution_memory else None
        total_world_delta = zero
        total_self_delta = zero
        total_memory_delta = zero
        for _ in range(steps):
            world_summary = self.world_state_slots.slot_norm(current_world).mean(dim=1)
            self_summary = self.self_state_slots.self_norm(current_self).mean(dim=1)
            if current_memory is not None:
                memory_summary = current_memory.mean(dim=1)
            else:
                memory_summary = torch.zeros_like(world_summary)
            evolution_summary = (world_summary + self_summary + memory_summary.to(dtype=world_summary.dtype)) / 3.0

            next_world, _world_metrics = self.world_state_slots.update_slots(current_world, evolution_summary)
            total_world_delta = total_world_delta + (next_world - current_world).float().pow(2).mean()

            hidden_seed = evolution_summary.unsqueeze(1)
            _hidden_seed, next_self, self_metrics = self.self_state_slots._apply_latent_thought(
                hidden_seed,
                world_state=next_world,
                self_state=current_self,
                causal_safe=True,
                steps=1,
                metric_prefix="state_evolution_self",
            )
            total_self_delta = total_self_delta + (next_self - current_self).float().pow(2).mean()

            next_memory = current_memory
            if memory_module is not None and current_memory is not None:
                next_memory, _memory_gate, _memory_novelty = memory_module.write(current_memory, evolution_summary)
                total_memory_delta = total_memory_delta + (next_memory - current_memory).float().pow(2).mean()

            current_world = next_world
            current_self = next_self
            current_memory = next_memory
            # Keep the self transition observable without treating it as a separate thought object.
            metrics["state_evolution_self_velocity"] = self_metrics["state_evolution_self_velocity"]

        steps_t = hidden_states.new_tensor(float(steps))
        metrics["state_evolution_world_delta"] = total_world_delta / steps_t.clamp_min(1.0)
        metrics["state_evolution_self_delta"] = total_self_delta / steps_t.clamp_min(1.0)
        metrics["state_evolution_memory_delta"] = total_memory_delta / steps_t.clamp_min(1.0)
        metrics["state_evolution_delta"] = (
            metrics["state_evolution_world_delta"]
            + metrics["state_evolution_self_delta"]
            + metrics["state_evolution_memory_delta"]
        ) / 3.0
        metrics["state_evolution_steps"] = steps_t
        world_state = self._append_state_trace(world_state, current_world)
        self_state = self._append_state_trace(self_state, current_self)
        return world_state, self_state, current_memory, metrics

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tau: float | None = None,
        return_aux: bool = True,
        return_logits: bool = True,
        infer_pad_mask: bool | None = None,
        past_state: NAIMEStatePacket | None = None,
        past_world_state: torch.Tensor | None = None,
        past_self_state: torch.Tensor | None = None,
        past_memory: torch.Tensor | None = None,
        detach_past_state: bool = True,
        return_state: bool = False,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        attention_mask = _resolve_attention_mask(input_ids, attention_mask, self.config, infer_pad_mask)

        hidden_states = self.embed_tokens(input_ids)
        batch_size = hidden_states.size(0)
        world_state, self_state, memory = _resolve_state_packet(
            past_state,
            past_world_state=past_world_state,
            past_self_state=past_self_state,
            past_memory=past_memory,
            detach_past_state=detach_past_state,
            batch_size=batch_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if world_state is None:
            world_state = self._initial_world_state(batch_size, hidden_states.device, hidden_states.dtype)
        if self_state is None:
            self_state = self._initial_self_state(batch_size, hidden_states.device, hidden_states.dtype)
        if memory is None:
            memory = self._initial_memory(batch_size, hidden_states.device, hidden_states.dtype)

        aux_by_layer = []
        field_trace_is_past = (
            past_state is not None
            or past_world_state is not None
            or past_self_state is not None
            or past_memory is not None
        )
        for block in self.blocks:
            if isinstance(block, NAIMEV5WorldStateMoEBlock):
                field_aux = None
                if self.latent_field is not None:
                    field_memory = memory if field_trace_is_past else None
                    hidden_states, field_aux = self.latent_field(
                        hidden_states,
                        world_state=world_state,
                        self_state=self_state,
                        memory=field_memory,
                        attention_mask=attention_mask,
                        block_size=max(self.config.stride, self.config.causal_state_stride),
                        treat_trace_as_past=field_trace_is_past,
                    )
                    field_trace_is_past = False
                hidden_states, aux, world_state, memory = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    tau=tau,
                    world_state=world_state,
                    memory=memory,
                )
                hidden_states, self_state, v6_aux = self.self_state_slots(
                    hidden_states,
                    attention_mask=attention_mask,
                    world_state=world_state,
                    self_state=self_state,
                    causal_safe=self.config.semantic_causal,
                    block_size=max(self.config.stride, self.config.causal_state_stride),
                )
                aux["v6"] = v6_aux
                if field_aux is not None:
                    aux["v6"].update(field_aux)
            else:
                hidden_states, aux = block(hidden_states, attention_mask=attention_mask)
            if return_aux:
                aux_by_layer.append(aux)

        world_state, self_state, memory, evolution_metrics = self._evolve_internal_state(
            hidden_states=hidden_states,
            world_state=world_state,
            self_state=self_state,
            memory=memory,
        )
        if return_aux and aux_by_layer:
            aux_by_layer[-1].setdefault("v6", {}).update(evolution_metrics)

        hidden_states = self.norm(hidden_states)
        public_world_state = _public_state(world_state)
        public_self_state = _public_state(self_state)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "hidden_states": hidden_states,
            "world_state": public_world_state,
            "self_state": public_self_state,
        }
        if memory is not None:
            output["memory"] = memory
        if return_state:
            output["state_packet"] = NAIMEStatePacket(
                world_state=world_state,
                self_state=self_state,
                memory=memory,
                architecture_id="naime_v6_recursive_self_moe",
            )
        if return_logits:
            output["logits"] = self.lm_head(hidden_states)
        if return_aux:
            output["aux"] = aux_by_layer
        return output


class DenseDecoder(nn.Module):
    """Plain decoder-only Transformer baseline."""

    def __init__(self, config: NAIMEStateMoEConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([DenseTransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(_init_weights)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_aux: bool = True,
        return_logits: bool = True,
        infer_pad_mask: bool | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        attention_mask = _resolve_attention_mask(input_ids, attention_mask, self.config, infer_pad_mask)

        hidden_states = self.embed_tokens(input_ids)
        aux_by_layer = []
        for block in self.blocks:
            hidden_states, aux = block(hidden_states, attention_mask=attention_mask)
            if return_aux:
                aux_by_layer.append(aux)

        hidden_states = self.norm(hidden_states)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "hidden_states": hidden_states,
        }
        if return_logits:
            output["logits"] = self.lm_head(hidden_states)
        if return_aux:
            output["aux"] = aux_by_layer
        return output


class TokenMoEDecoder(nn.Module):
    """Token-only MoE baseline with the semantic router disabled."""

    def __init__(self, config: NAIMEStateMoEConfig):
        super().__init__()
        self.config = NAIMEStateMoEConfig(**{**config.__dict__, "use_semantic_router": False})
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList()
        for layer_idx in range(self.config.n_layers):
            if layer_idx < self.config.n_dense_layers:
                self.blocks.append(DenseTransformerBlock(self.config))
            else:
                self.blocks.append(TokenMoEBlock(self.config))
        self.norm = RMSNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.apply(_init_weights)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_aux: bool = True,
        return_logits: bool = True,
        infer_pad_mask: bool | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        attention_mask = _resolve_attention_mask(input_ids, attention_mask, self.config, infer_pad_mask)

        hidden_states = self.embed_tokens(input_ids)
        aux_by_layer = []
        for block in self.blocks:
            hidden_states, aux = block(hidden_states, attention_mask=attention_mask)
            if return_aux:
                aux_by_layer.append(aux)

        hidden_states = self.norm(hidden_states)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "hidden_states": hidden_states,
        }
        if return_logits:
            output["logits"] = self.lm_head(hidden_states)
        if return_aux:
            output["aux"] = aux_by_layer
        return output
