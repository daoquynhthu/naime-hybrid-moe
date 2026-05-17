from dataclasses import dataclass


@dataclass
class NAIMEStateMoEConfig:
    vocab_size: int = 32000
    max_seq_len: int = 1024
    d_model: int = 256
    n_layers: int = 6
    n_dense_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 1024
    dropout: float = 0.0

    # NAIME semantic compressor.
    stride: int = 8
    window: int = 12
    z_dim: int = 64
    target_sparsity: float = 0.2
    gumbel_tau: float = 1.0
    gate_eval_mode: str = "prob"
    logvar_clip: float = 10.0
    semantic_scales: str = "local"
    mid_stride: int = 32
    mid_window: int = 64
    use_global_semantic: bool = False
    semantic_fusion: str = "local"
    semantic_pred_horizon: int = 0

    # MoE.
    n_experts: int = 4
    top_k: int = 2
    expert_hidden_dim: int = 512
    moe_dispatch_mode: str = "auto"
    router_jitter: float = 0.0
    use_semantic_router: bool = True
    semantic_router_mode: str = "concat"
    semantic_router_prior_scale: float = 1.0
    semantic_router_prior_clip: float = 0.0
    semantic_router_detach: bool = False
    semantic_router_alpha_cap: float = 0.0
    semantic_alpha_cap_mode: str = "clamp"
    semantic_gate_downstream: str = "alpha"
    semantic_sparse_alpha: str = "alpha"
    semantic_downstream_deterministic: bool = False
    semantic_router_prior_gate: bool = False
    use_semantic_residual_write: bool = False
    semantic_write_scale: float = 1.0

    # V4 state-centric semantic system.
    semantic_memory_slots: int = 0
    semantic_memory_write_scale: float = 0.05
    semantic_state_write_scale: float = 0.05
    semantic_gate_mixer: bool = False
    semantic_gate_mixer_temperature: float = 1.0
    semantic_gate_mixer_min_weight: float = 0.0
    semantic_gate_mixer_max_clean_weight: float = 0.0
    semantic_gate_mixer_max_state_weight: float = 0.35
    semantic_state_confidence_mode: str = "learned"
    semantic_state_confidence_temperature: float = 2.0
    semantic_state_confidence_gate: bool = False
    semantic_memory_read_gate: bool = False
    semantic_memory_hidden_scale: float = 0.035
    layerwise_semantic_schedule: bool = False

    # V5 structured world-state system.
    world_state_slots: int = 0
    world_state_diversity_margin: float = 0.85
    world_state_pred_detach_target: bool = True
    world_state_stability_threshold: float = 1e-3
    world_state_write_top_k: int = 2

    # V6 recursive self-state system.
    self_state_slots: int = 0
    self_state_recursion_depth: int = 1
    self_state_write_scale: float = 0.03
    self_state_hidden_scale: float = 0.02
    self_state_boundary_temperature: float = 1.0
    self_state_diversity_margin: float = 0.85
    self_state_identity_scale: float = 0.02
    self_state_context_score_scale: float = 4.0
    self_state_pred_detach_target: bool = True

    # Stability / architecture toggles.
    qk_norm: bool = True
    rope_theta: float = 10000.0
    pad_token_id: int = 0
    attention_type: str = "gqa"
    mla_latent_dim: int = 128
    mla_rope_per_head: int = 32


@dataclass
class BaselineConfig(NAIMEStateMoEConfig):
    """Config used by dense and token-only MoE baselines."""

    architecture: str = "naime_state_moe"
