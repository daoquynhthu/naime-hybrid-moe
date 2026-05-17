import torch
from torch import nn

from naime_hybrid.config import NAIMEStateMoEConfig

from .attention import GQAAttention, MLAAttention
from .moe import SwiGLUExpert, TopKMoE
from .norm import RMSNorm
from .semantic_compressor import SemanticCompressor
from .state import SemanticGateMixer, SemanticMemory


def _build_attention(config: NAIMEStateMoEConfig):
    if config.attention_type == "mla":
        return MLAAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.max_seq_len,
            d_latent=config.mla_latent_dim,
            d_rope_per_head=config.mla_rope_per_head,
            dropout=config.dropout,
            qk_norm=config.qk_norm,
            rope_theta=config.rope_theta,
        )
    return GQAAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        qk_norm=config.qk_norm,
        rope_theta=config.rope_theta,
    )


class DenseTransformerBlock(nn.Module):
    def __init__(self, config: NAIMEStateMoEConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = _build_attention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUExpert(config.d_model, config.d_ff)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), attention_mask)
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states, {}


class NAIMEStateMoEBlock(nn.Module):
    def __init__(self, config: NAIMEStateMoEConfig):
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = _build_attention(config)
        self.compressor = SemanticCompressor(
            d_model=config.d_model,
            z_dim=config.z_dim,
            stride=config.stride,
            window=config.window,
            target_sparsity=config.target_sparsity,
            gate_eval_mode=config.gate_eval_mode,
            logvar_clip=config.logvar_clip,
            semantic_scales=config.semantic_scales,
            mid_stride=config.mid_stride,
            mid_window=config.mid_window,
            use_global_semantic=config.use_global_semantic,
            semantic_fusion=config.semantic_fusion,
            semantic_pred_horizon=config.semantic_pred_horizon,
            downstream_deterministic=config.semantic_downstream_deterministic,
            downstream_detach_latent=config.semantic_router_detach,
        )
        self.moe_norm = RMSNorm(config.d_model)
        self.moe = TopKMoE(
            d_model=config.d_model,
            semantic_dim=config.d_model,
            n_experts=config.n_experts,
            top_k=config.top_k,
            expert_hidden_dim=config.expert_hidden_dim,
            use_semantic_router=config.use_semantic_router,
            semantic_router_mode=config.semantic_router_mode,
            semantic_router_prior_scale=config.semantic_router_prior_scale,
            semantic_router_prior_clip=config.semantic_router_prior_clip,
            router_jitter=config.router_jitter,
            dispatch_mode=config.moe_dispatch_mode,
            use_prior_gate=config.semantic_router_prior_gate,
        )
        self.semantic_write = nn.Linear(config.d_model, config.d_model, bias=False)

    def _apply_alpha_cap(self, token_alpha: torch.Tensor) -> torch.Tensor:
        cap = self.config.semantic_router_alpha_cap
        if cap <= 0:
            return token_alpha
        mode = self.config.semantic_alpha_cap_mode
        if mode == "clamp":
            return token_alpha.clamp(max=cap)
        if mode == "scale":
            return token_alpha * cap
        raise ValueError("semantic_alpha_cap_mode must be clamp or scale")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tau: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), attention_mask)

        semantic = self.compressor(
            hidden_states,
            attention_mask=attention_mask,
            tau=self.config.gumbel_tau if tau is None else tau,
        )
        gate_source = self.config.semantic_gate_downstream
        if gate_source == "alpha":
            token_alpha = semantic["token_alpha"]
        elif gate_source == "prob":
            token_alpha = semantic["token_gate_prob"]
        elif gate_source == "clean_prob":
            token_alpha = semantic["token_gate_clean_prob"]
        elif gate_source == "none":
            token_alpha = torch.ones_like(semantic["token_alpha"])
        else:
            raise ValueError("semantic_gate_downstream must be alpha, prob, clean_prob, or none")
        token_alpha = self._apply_alpha_cap(token_alpha)
        semantic["token_alpha_downstream"] = token_alpha
        token_semantic = semantic["token_semantic_downstream"] * token_alpha
        router_semantic = token_semantic

        if self.config.use_semantic_residual_write:
            hidden_states = hidden_states + self.config.semantic_write_scale * self.semantic_write(router_semantic)

        moe_input = self.moe_norm(hidden_states)
        moe_output, moe_aux = self.moe(moe_input, router_semantic)
        hidden_states = hidden_states + moe_output

        aux = {
            "semantic": semantic,
            "moe": moe_aux,
        }
        return hidden_states, aux


class NAIMEV4StateMoEBlock(NAIMEStateMoEBlock):
    """State-centric V4 block with cross-layer state and working memory."""

    def __init__(self, config: NAIMEStateMoEConfig, layer_idx: int = 0):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.gate_mixer = (
            SemanticGateMixer(
                config.d_model,
                temperature=config.semantic_gate_mixer_temperature,
                min_weight=config.semantic_gate_mixer_min_weight,
                max_clean_weight=config.semantic_gate_mixer_max_clean_weight,
                max_state_weight=config.semantic_gate_mixer_max_state_weight,
            )
            if config.semantic_gate_mixer
            else None
        )
        self.memory = (
            SemanticMemory(config.d_model, config.semantic_memory_slots) if config.semantic_memory_slots > 0 else None
        )
        self.state_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.memory_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def _layer_scale(self) -> float:
        if not self.config.layerwise_semantic_schedule:
            return 1.0
        semantic_layers = max(1, self.config.n_layers - self.config.n_dense_layers)
        depth = max(0, self.layer_idx - self.config.n_dense_layers)
        return 0.5 + 0.75 * (depth / max(1, semantic_layers - 1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tau: float | None = None,
        semantic_state: torch.Tensor | None = None,
        semantic_state_confidence: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), attention_mask)

        semantic = self.compressor(
            hidden_states,
            attention_mask=attention_mask,
            tau=self.config.gumbel_tau if tau is None else tau,
        )
        batch, seq_len, _ = hidden_states.shape
        if attention_mask is None:
            mask = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden_states.device)
        else:
            mask = attention_mask.to(torch.bool)

        gate_source = self.config.semantic_gate_downstream
        if gate_source == "alpha":
            token_alpha = semantic["token_alpha"]
        elif gate_source == "prob":
            token_alpha = semantic["token_gate_prob"]
        elif gate_source == "clean_prob":
            token_alpha = semantic["token_gate_clean_prob"]
        elif gate_source == "none":
            token_alpha = torch.ones_like(semantic["token_alpha"])
        else:
            raise ValueError("semantic_gate_downstream must be alpha, prob, clean_prob, or none")

        state_confidence = torch.ones(batch, 1, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        if semantic_state_confidence is not None:
            state_confidence = semantic_state_confidence.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if state_confidence.ndim == 2:
                state_confidence = state_confidence.unsqueeze(1)
        if self.gate_mixer is not None:
            token_alpha, gate_mix_weights = self.gate_mixer(
                hidden_states,
                semantic["token_alpha"].type_as(hidden_states),
                semantic["token_gate_clean_prob"].type_as(hidden_states),
                state_confidence,
            )
            semantic["gate_mix_weights"] = gate_mix_weights

        layer_scale = self._layer_scale()
        token_alpha = self._apply_alpha_cap(token_alpha) * layer_scale
        token_alpha = token_alpha.clamp(0.0, 1.0)
        semantic["token_alpha_downstream"] = token_alpha

        token_semantic = semantic["token_semantic_downstream"] * token_alpha
        router_semantic = token_semantic
        state_context = torch.zeros_like(router_semantic)
        if semantic_state is not None:
            state_context = self.state_proj(semantic_state).unsqueeze(1).expand_as(router_semantic)
            state_scale = self.config.semantic_state_write_scale
            if self.config.semantic_state_confidence_gate:
                state_scale = state_scale * state_confidence
            router_semantic = router_semantic + state_scale * state_context

        memory_context = torch.zeros_like(router_semantic)
        memory_weights = torch.zeros(batch, seq_len, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        memory_read_strength = torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
        memory_novelty = torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
        if self.memory is not None and memory is not None:
            memory_context, memory_weights = self.memory.read(hidden_states, memory)
            memory_token_strength = memory_weights.max(dim=-1).values.unsqueeze(-1)
            memory_read_strength = memory_token_strength.mean()
            memory_scale = self.config.semantic_memory_write_scale
            if self.config.semantic_memory_read_gate:
                memory_scale = memory_scale * memory_token_strength
            router_semantic = router_semantic + memory_scale * self.memory_proj(memory_context)

        if self.config.use_semantic_residual_write:
            hidden_states = hidden_states + self.config.semantic_write_scale * self.semantic_write(router_semantic)

        moe_input = self.moe_norm(hidden_states)
        moe_output, moe_aux = self.moe(moe_input, router_semantic)
        hidden_states = hidden_states + moe_output

        mask_f = mask.unsqueeze(-1).type_as(router_semantic)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        semantic_summary = (router_semantic * mask_f).sum(dim=1) / denom
        if self.memory is not None and memory is not None:
            memory, memory_gate, memory_novelty_values = self.memory.write(memory, semantic_summary)
            memory_novelty = memory_novelty_values.mean()
        else:
            memory_gate = torch.zeros(
                batch, 1, self.config.d_model, device=hidden_states.device, dtype=hidden_states.dtype
            )

        aux = {
            "semantic": semantic,
            "moe": moe_aux,
            "v4": {
                "layer_scale": torch.tensor(layer_scale, device=hidden_states.device, dtype=hidden_states.dtype),
                "state_norm": semantic_state.norm(dim=-1).mean()
                if semantic_state is not None
                else torch.zeros((), device=hidden_states.device),
                "memory_norm": memory.norm(dim=-1).mean()
                if memory is not None
                else torch.zeros((), device=hidden_states.device),
                "memory_gate": memory_gate.mean(),
                "memory_attention_entropy": -(
                    memory_weights.clamp_min(1e-6).float() * memory_weights.clamp_min(1e-6).float().log()
                )
                .sum(dim=-1)
                .mean(),
                "memory_read_strength": memory_read_strength,
                "memory_novelty": memory_novelty,
            },
        }
        return hidden_states, aux, semantic_summary, memory


class NAIMEV5WorldStateMoEBlock(NAIMEV4StateMoEBlock):
    """V5 block that replaces the single recurrent state with structured state slots."""

    def __init__(self, config: NAIMEStateMoEConfig, layer_idx: int = 0, world_state_slots: nn.Module | None = None):
        super().__init__(config, layer_idx=layer_idx)
        self.world_state_slots = world_state_slots

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tau: float | None = None,
        world_state: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), attention_mask)

        semantic = self.compressor(
            hidden_states,
            attention_mask=attention_mask,
            tau=self.config.gumbel_tau if tau is None else tau,
        )
        batch, seq_len, _ = hidden_states.shape
        if attention_mask is None:
            mask = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden_states.device)
        else:
            mask = attention_mask.to(torch.bool)

        if self.world_state_slots is not None and world_state is None:
            world_state = self.world_state_slots.initial_state(batch, hidden_states.device, hidden_states.dtype)
        if self.world_state_slots is not None and world_state is not None:
            state_context, state_weights, slot_confidence = self.world_state_slots.read(hidden_states, world_state)
            read_focus = state_weights.max(dim=-1, keepdim=True).values
            state_confidence = slot_confidence * read_focus.type_as(slot_confidence)
        else:
            state_context = torch.zeros(
                batch, seq_len, self.config.d_model, device=hidden_states.device, dtype=hidden_states.dtype
            )
            state_confidence = torch.ones(batch, seq_len, 1, device=hidden_states.device, dtype=hidden_states.dtype)

        gate_source = self.config.semantic_gate_downstream
        if gate_source == "alpha":
            token_alpha = semantic["token_alpha"]
        elif gate_source == "prob":
            token_alpha = semantic["token_gate_prob"]
        elif gate_source == "clean_prob":
            token_alpha = semantic["token_gate_clean_prob"]
        elif gate_source == "none":
            token_alpha = torch.ones_like(semantic["token_alpha"])
        else:
            raise ValueError("semantic_gate_downstream must be alpha, prob, clean_prob, or none")

        if self.gate_mixer is not None:
            token_alpha, gate_mix_weights = self.gate_mixer(
                hidden_states,
                semantic["token_alpha"].type_as(hidden_states),
                semantic["token_gate_clean_prob"].type_as(hidden_states),
                state_confidence,
            )
            semantic["gate_mix_weights"] = gate_mix_weights

        layer_scale = self._layer_scale()
        token_alpha = self._apply_alpha_cap(token_alpha) * layer_scale
        token_alpha = token_alpha.clamp(0.0, 1.0)
        semantic["token_alpha_downstream"] = token_alpha

        token_semantic = semantic["token_semantic_downstream"] * token_alpha
        router_semantic = token_semantic
        state_scale = self.config.semantic_state_write_scale
        if self.config.semantic_state_confidence_gate:
            state_scale = state_scale * state_confidence
        router_semantic = router_semantic + state_scale * self.state_proj(state_context)

        memory_context = torch.zeros_like(router_semantic)
        memory_weights = torch.zeros(batch, seq_len, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        memory_read_strength = torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
        memory_novelty = torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
        if self.memory is not None and memory is not None:
            memory_context, memory_weights = self.memory.read(hidden_states, memory)
            memory_token_strength = memory_weights.max(dim=-1).values.unsqueeze(-1)
            memory_read_strength = memory_token_strength.mean()
            memory_scale = self.config.semantic_memory_hidden_scale
            if self.config.semantic_memory_read_gate:
                memory_scale = memory_scale * memory_token_strength
            hidden_states = hidden_states + memory_scale * self.memory_proj(memory_context)

        if self.config.use_semantic_residual_write:
            hidden_states = hidden_states + self.config.semantic_write_scale * self.semantic_write(router_semantic)

        moe_input = self.moe_norm(hidden_states)
        moe_output, moe_aux = self.moe(moe_input, router_semantic)
        hidden_states = hidden_states + moe_output

        mask_f = mask.unsqueeze(-1).type_as(router_semantic)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        semantic_summary = (router_semantic * mask_f).sum(dim=1) / denom
        if self.world_state_slots is not None and world_state is not None:
            world_state, v5_metrics = self.world_state_slots.update_slots(world_state, semantic_summary)
        else:
            v5_metrics = {
                "slot_update_gate": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
                "slot_write_max": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
                "slot_write_min": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
                "slot_write_active": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
                "slot_write_entropy": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
                "slot_confidence": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
                "slot_confidence_std": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
                "slot_delta": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
                "slot_cosine": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
                "slot_diversity": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
                "state_pred": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
            }

        if self.memory is not None and memory is not None:
            memory, memory_gate, memory_novelty_values = self.memory.write(memory, semantic_summary)
            memory_novelty = memory_novelty_values.mean()
        else:
            memory_gate = torch.zeros(
                batch, 1, self.config.d_model, device=hidden_states.device, dtype=hidden_states.dtype
            )

        slot_read_entropy = (
            -(state_weights.clamp_min(1e-6).float() * state_weights.clamp_min(1e-6).float().log()).sum(dim=-1).mean()
            if world_state is not None
            else torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
        )
        slot_read_max = (
            state_weights.max(dim=-1).values.mean()
            if world_state is not None
            else torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
        )
        slot_count = self.world_state_slots.slots if self.world_state_slots is not None else 0
        state_norm = (
            world_state.norm(dim=-1).mean()
            if world_state is not None
            else torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
        )
        aux = {
            "semantic": semantic,
            "moe": moe_aux,
            "v4": {
                "layer_scale": torch.tensor(layer_scale, device=hidden_states.device, dtype=hidden_states.dtype),
                "state_norm": state_norm,
                "memory_norm": memory.norm(dim=-1).mean()
                if memory is not None
                else torch.zeros((), device=hidden_states.device),
                "memory_gate": memory_gate.mean(),
                "memory_attention_entropy": -(
                    memory_weights.clamp_min(1e-6).float() * memory_weights.clamp_min(1e-6).float().log()
                )
                .sum(dim=-1)
                .mean(),
                "memory_read_strength": memory_read_strength,
                "memory_novelty": memory_novelty,
                "state_gate": v5_metrics["slot_update_gate"],
                "state_confidence": v5_metrics["slot_confidence"],
                "state_delta": v5_metrics["slot_delta"],
                "state_agreement": torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
            },
            "v5": {
                **v5_metrics,
                "slot_read_entropy": slot_read_entropy.type_as(hidden_states),
                "slot_read_max": slot_read_max,
                "slot_count": torch.tensor(slot_count, device=hidden_states.device, dtype=hidden_states.dtype),
            },
        }
        return hidden_states, aux, world_state, memory


class TokenMoEBlock(nn.Module):
    """Standard token-only MoE baseline block."""

    def __init__(self, config: NAIMEStateMoEConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = _build_attention(config)
        self.moe_norm = RMSNorm(config.d_model)
        self.moe = TopKMoE(
            d_model=config.d_model,
            semantic_dim=config.d_model,
            n_experts=config.n_experts,
            top_k=config.top_k,
            expert_hidden_dim=config.expert_hidden_dim,
            use_semantic_router=False,
            semantic_router_mode=config.semantic_router_mode,
            semantic_router_prior_scale=config.semantic_router_prior_scale,
            semantic_router_prior_clip=config.semantic_router_prior_clip,
            router_jitter=config.router_jitter,
            dispatch_mode=config.moe_dispatch_mode,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), attention_mask)
        moe_output, moe_aux = self.moe(self.moe_norm(hidden_states), semantic_states=None)
        hidden_states = hidden_states + moe_output
        return hidden_states, {"moe": moe_aux}
