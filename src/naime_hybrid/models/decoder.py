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
from naime_hybrid.modules.norm import RMSNorm
from naime_hybrid.modules.self_state import RecursiveSelfState
from naime_hybrid.modules.state import CrossLayerSemanticState
from naime_hybrid.modules.world_state import WorldStateSlots


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


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
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

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
        logits = self.lm_head(hidden_states)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
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
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

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
        logits = self.lm_head(hidden_states)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
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
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        hidden_states = self.embed_tokens(input_ids)
        batch_size = hidden_states.size(0)
        world_state = self._initial_world_state(batch_size, hidden_states.device, hidden_states.dtype)
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
        logits = self.lm_head(hidden_states)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "logits": logits,
            "hidden_states": hidden_states,
            "world_state": world_state,
        }
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
        )
        self.self_state_slots.apply(_init_weights)

    def _initial_self_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.self_state_slots.initial_state(batch_size, device, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tau: float | None = None,
        return_aux: bool = True,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        hidden_states = self.embed_tokens(input_ids)
        batch_size = hidden_states.size(0)
        world_state = self._initial_world_state(batch_size, hidden_states.device, hidden_states.dtype)
        self_state = self._initial_self_state(batch_size, hidden_states.device, hidden_states.dtype)
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
                hidden_states, self_state, v6_aux = self.self_state_slots(
                    hidden_states,
                    attention_mask=attention_mask,
                    world_state=world_state,
                    self_state=self_state,
                )
                aux["v6"] = v6_aux
            else:
                hidden_states, aux = block(hidden_states, attention_mask=attention_mask)
            if return_aux:
                aux_by_layer.append(aux)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "logits": logits,
            "hidden_states": hidden_states,
            "world_state": world_state,
            "self_state": self_state,
        }
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
        **_: object,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        hidden_states = self.embed_tokens(input_ids)
        aux_by_layer = []
        for block in self.blocks:
            hidden_states, aux = block(hidden_states, attention_mask=attention_mask)
            if return_aux:
                aux_by_layer.append(aux)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
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
        **_: object,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        hidden_states = self.embed_tokens(input_ids)
        aux_by_layer = []
        for block in self.blocks:
            hidden_states, aux = block(hidden_states, attention_mask=attention_mask)
            if return_aux:
                aux_by_layer.append(aux)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        if return_aux:
            output["aux"] = aux_by_layer
        return output
