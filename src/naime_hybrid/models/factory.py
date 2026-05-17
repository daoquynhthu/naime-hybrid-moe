from naime_hybrid.config import NAIMEStateMoEConfig

from .decoder import (
    DenseDecoder,
    NAIMEStateMoEDecoder,
    NAIMEV4StateMoEDecoder,
    NAIMEV5WorldStateMoEDecoder,
    NAIMEV6RecursiveSelfMoEDecoder,
    TokenMoEDecoder,
)


def build_model(architecture: str, config: NAIMEStateMoEConfig):
    architecture = architecture.lower().replace("-", "_")
    if architecture in {"dense", "dense_transformer", "transformer"}:
        return DenseDecoder(config)
    if architecture in {"token_moe", "token_only_moe", "moe"}:
        return TokenMoEDecoder(config)
    if architecture in {"naime", "naime_state_moe", "naime_hybrid_moe"}:
        return NAIMEStateMoEDecoder(config)
    if architecture in {
        "naime_v4",
        "naime_v4_state_moe",
        "state_moe_v4",
        "naime_v41",
        "naime_v41_state_moe",
        "state_moe_v41",
        "naime_v42",
        "naime_v42_state_moe",
        "state_moe_v42",
    }:
        return NAIMEV4StateMoEDecoder(config)
    if architecture in {"naime_v5", "naime_v5_world_state_moe", "world_state_moe_v5"}:
        return NAIMEV5WorldStateMoEDecoder(config)
    if architecture in {"naime_v6", "naime_v6_recursive_self_moe", "recursive_self_moe_v6"}:
        return NAIMEV6RecursiveSelfMoEDecoder(config)
    raise ValueError(f"unknown architecture: {architecture}")
