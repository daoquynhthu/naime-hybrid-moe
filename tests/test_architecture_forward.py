import warnings

import pytest
import torch

from naime_hybrid import (
    NAIMEStateMoEConfig,
    NAIMEStateMoEDecoder,
    NAIMEV4StateMoEDecoder,
    NAIMEV5WorldStateMoEDecoder,
    NAIMEV6RecursiveSelfMoEDecoder,
    build_model,
)
from naime_hybrid.data import HFDiskCausalDataset
from naime_hybrid.modules.gate import GumbelBlockGate
from naime_hybrid.modules.moe import TopKMoE
from naime_hybrid.modules.self_state import RecursiveSelfState
from naime_hybrid.modules.world_state import WorldStateSlots
from naime_hybrid.training.checkpoint import (
    build_model_payload,
    normalize_state_dict_for_model,
    save_payloads_in_subprocess,
)
from naime_hybrid.training.cli import build_train_config, parse_args
from naime_hybrid.training.control import reference_value_at_step, update_sparse_lambda
from naime_hybrid.training.losses import lm_loss
from naime_hybrid.training.masks import prepare_attention_mask_for_device


class _FakeCompiledWrapper(torch.nn.Module):
    """Small stand-in for the torch.compile wrapper shape used in checkpoints."""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self._orig_mod = module


def _require_torch_compile_support():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable in this PyTorch build")
    try:
        probe = torch.compile(lambda x: x + 1, backend="eager")
        probe(torch.ones(1))
    except Exception as exc:
        pytest.skip(f"torch.compile unavailable in this environment: {exc}")


def test_checkpoint_subprocess_saves_payload(tmp_path):
    path = tmp_path / "subprocess.pt"
    payload = {"step": 7, "model": {"w": torch.arange(4)}}

    save_payloads_in_subprocess([(path, payload)])

    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert restored["step"] == 7
    assert torch.equal(restored["model"]["w"], torch.arange(4))


def test_checkpoint_normalizes_compiled_prefix():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    model = build_model("dense", config)
    eager_state = model.state_dict()
    compiled_like_state = {f"_orig_mod.{key}": value.clone() for key, value in eager_state.items()}

    normalized = normalize_state_dict_for_model(model, compiled_like_state)

    assert set(normalized) == set(eager_state)


def test_model_payload_unwraps_compiled_prefix():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    model = build_model("dense", config)
    compiled = _FakeCompiledWrapper(model)

    payload = build_model_payload(compiled, step=1, config={}, metrics={})

    assert payload["model"]
    assert all(not key.startswith("_orig_mod.") for key in payload["model"])


def test_eval_sampling_cli_defaults_to_random(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--architecture", "dense", "--eval-every", "10", "--eval-max-batches", "4"],
    )

    config = build_train_config(parse_args())

    assert config.eval_sampling == "random"
    assert config.eval_seed == 4321
    assert config.model.semantic_causal is True
    assert config.model.causal_state_stride == 512


def test_lm_loss_trains_token_zero_and_ignores_only_negative_sentinel():
    logits = torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]])
    labels = torch.tensor([[0, -100]])

    loss = lm_loss(logits, labels)

    assert loss < 1e-4


def test_hf_collate_keeps_token_zero_visible():
    batch = [{"input_ids": torch.tensor([0, 5, 6, 7, 8])}, {"input_ids": torch.tensor([9, 0, 10, 11, 12])}]

    collated = HFDiskCausalDataset.causal_collate(batch, seq_len=4)

    assert collated["input_ids"][0, 0].item() == 0
    assert collated["input_ids"][1, 1].item() == 0
    assert collated["attention_mask"].all()
    assert (collated["labels"] == -100).sum().item() == 0


def test_tiny_decoder_forward_and_backward():
    config = NAIMEStateMoEConfig(
        vocab_size=128,
        max_seq_len=64,
        d_model=64,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        stride=4,
        window=8,
        z_dim=16,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=96,
    )
    model = NAIMEStateMoEDecoder(config)
    input_ids = torch.randint(1, config.vocab_size, (2, 17))

    out = model(input_ids)
    assert out["logits"].shape == (2, 17, config.vocab_size)
    assert len(out["aux"]) == config.n_layers

    moe_aux = out["aux"][-1]["moe"]
    semantic_aux = out["aux"][-1]["semantic"]
    assert moe_aux["topk_indices"].shape == (2, 17, config.top_k)
    assert semantic_aux["z"].shape == (2, 5, config.z_dim)
    assert semantic_aux["token_semantic"].shape == (2, 17, config.d_model)

    loss = out["logits"].float().mean() + moe_aux["load_balance"] + semantic_aux["kl"].mean() * 0.0
    loss.backward()


def test_v6_semantic_path_is_causal_by_default():
    torch.manual_seed(1234)
    config = NAIMEStateMoEConfig(
        vocab_size=96,
        max_seq_len=64,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=48,
        semantic_router_mode="hybrid",
        semantic_scales="local_mid_global",
        mid_stride=8,
        mid_window=16,
        use_global_semantic=True,
        semantic_fusion="concat",
        use_semantic_residual_write=True,
        semantic_write_scale=0.03,
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_gate_mixer=True,
        world_state_slots=4,
        self_state_slots=3,
        self_state_recursion_depth=2,
    )
    model = build_model("naime_v6_recursive_self_moe", config).eval()
    input_ids = torch.randint(1, config.vocab_size, (2, 31))
    changed = input_ids.clone()
    cutoff = 12
    changed[:, cutoff:] = torch.randint(1, config.vocab_size, changed[:, cutoff:].shape)

    with torch.no_grad():
        original_logits = model(input_ids)["logits"]
        changed_logits = model(changed)["logits"]

    assert torch.allclose(original_logits[:, :cutoff, :], changed_logits[:, :cutoff, :], atol=1e-5, rtol=1e-5)


def test_v6_does_not_leak_within_state_blocks():
    torch.manual_seed(2026)
    config = NAIMEStateMoEConfig(
        vocab_size=96,
        max_seq_len=64,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=48,
        semantic_router_mode="hybrid",
        semantic_scales="local_mid_global",
        mid_stride=8,
        mid_window=16,
        use_global_semantic=True,
        semantic_fusion="concat",
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_gate_mixer=True,
        world_state_slots=4,
        self_state_slots=3,
        self_state_recursion_depth=2,
    )
    model = build_model("naime_v6_recursive_self_moe", config).eval()
    input_ids = torch.randint(1, config.vocab_size, (2, 31))
    changed = input_ids.clone()
    cutoff = 1
    changed[:, cutoff:] = torch.randint(1, config.vocab_size, changed[:, cutoff:].shape)

    with torch.no_grad():
        original_logits = model(input_ids)["logits"]
        changed_logits = model(changed)["logits"]

    assert torch.allclose(original_logits[:, :cutoff, :], changed_logits[:, :cutoff, :], atol=1e-5, rtol=1e-5)


def test_world_state_history_reads_only_past_blocks():
    torch.manual_seed(2027)
    slots = WorldStateSlots(d_model=16, slots=3)
    hidden = torch.randn(2, 12, 16)
    semantic = torch.randn(2, 12, 16)
    trace = torch.randn(2, 3, 3, 16)
    changed_future_trace = trace.clone()
    changed_future_trace[:, 2, :, :] = torch.randn_like(changed_future_trace[:, 2, :, :])

    context, _, _, _, _ = slots.read_update_sequence(hidden, semantic, trace, stride=4)
    changed_context, _, _, _, _ = slots.read_update_sequence(hidden, semantic, changed_future_trace, stride=4)

    assert torch.allclose(context[:, :8, :], changed_context[:, :8, :], atol=1e-5, rtol=1e-5)


def test_world_state_history_first_block_has_finite_backward():
    torch.manual_seed(2028)
    slots = WorldStateSlots(d_model=16, slots=3)
    hidden = torch.randn(2, 12, 16, requires_grad=True)
    semantic = torch.randn(2, 12, 16, requires_grad=True)
    trace = torch.randn(2, 3, 3, 16, requires_grad=True)

    context, _, _, traced_state, _ = slots.read_update_sequence(hidden, semantic, trace, stride=4)
    loss = context[:, 4:, :].float().square().mean() + traced_state.float().square().mean()
    loss.backward()

    assert torch.isfinite(context).all()
    assert torch.isfinite(traced_state).all()
    assert trace.grad is not None
    assert torch.isfinite(trace.grad).all()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 CUDA backward stability test requires CUDA")
def test_world_state_history_bf16_context_backward_is_finite():
    torch.manual_seed(2029)
    slots = WorldStateSlots(d_model=32, slots=4).cuda()
    hidden = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    semantic = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    trace = torch.randn(2, 4, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        context, _, _, traced_state, metrics = slots.read_update_sequence(hidden, semantic, trace, stride=4)
    loss = (
        context.float().square().mean()
        + traced_state.float().square().mean()
        + metrics["history_read_entropy"].float()
    )
    loss.backward()

    assert torch.isfinite(context).all()
    assert torch.isfinite(traced_state).all()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert trace.grad is not None
    assert torch.isfinite(trace.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 CUDA backward stability test requires CUDA")
def test_recursive_self_state_bf16_slot_context_backward_is_finite():
    torch.manual_seed(2030)
    module = RecursiveSelfState(d_model=32, slots=4, hidden_scale=0.02).cuda()
    hidden = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    world_trace = torch.randn(2, 4, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    self_trace = torch.randn(2, 4, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    attention_mask = torch.ones(2, 16, device="cuda", dtype=torch.bool)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, next_state, metrics = module(
            hidden,
            attention_mask=attention_mask,
            world_state=world_trace,
            self_state=self_trace,
            causal_safe=True,
            block_size=4,
        )
    loss = (
        output.float().square().mean()
        + next_state.float().square().mean()
        + metrics["self_pred"].float()
        + metrics["slot_context_cosine"].float()
    )
    loss.backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(next_state).all()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert self_trace.grad is not None
    assert torch.isfinite(self_trace.grad).all()


def test_multiscale_semantic_architectures_do_not_leak_future_tokens():
    torch.manual_seed(4321)
    base_config = NAIMEStateMoEConfig(
        vocab_size=96,
        max_seq_len=64,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=48,
        semantic_router_mode="hybrid",
        semantic_scales="local_mid_global",
        mid_stride=8,
        mid_window=16,
        use_global_semantic=True,
        semantic_fusion="concat",
        use_semantic_residual_write=True,
        semantic_write_scale=0.03,
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_gate_mixer=True,
        world_state_slots=4,
        self_state_slots=3,
    )
    input_ids = torch.randint(1, base_config.vocab_size, (2, 31))
    changed = input_ids.clone()
    cutoff = 12
    changed[:, cutoff:] = torch.randint(1, base_config.vocab_size, changed[:, cutoff:].shape)

    for architecture in [
        "naime_state_moe",
        "naime_v4_state_moe",
        "naime_v5_world_state_moe",
        "naime_v6_recursive_self_moe",
    ]:
        model = build_model(architecture, base_config).eval()
        with torch.no_grad():
            original_logits = model(input_ids)["logits"]
            changed_logits = model(changed)["logits"]
        assert torch.allclose(
            original_logits[:, :cutoff, :],
            changed_logits[:, :cutoff, :],
            atol=1e-5,
            rtol=1e-5,
        ), architecture


def test_baseline_factory_forward():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=2,
        top_k=1,
        expert_hidden_dim=48,
    )
    input_ids = torch.randint(1, config.vocab_size, (2, 13))
    for architecture in [
        "dense",
        "token_moe",
        "naime_state_moe",
        "naime_v4_state_moe",
        "naime_v41_state_moe",
        "naime_v42_state_moe",
        "naime_v5_world_state_moe",
        "naime_v6_recursive_self_moe",
    ]:
        model = build_model(architecture, config)
        out = model(input_ids)
        assert out["logits"].shape == (2, 13, config.vocab_size)


def test_semantic_router_prior_mode_forward_and_backward():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=48,
        semantic_router_mode="prior",
        semantic_router_prior_scale=0.75,
    )
    model = build_model("naime_state_moe", config)
    input_ids = torch.randint(1, config.vocab_size, (2, 13))

    out = model(input_ids)
    moe_aux = out["aux"][-1]["moe"]
    assert out["logits"].shape == (2, 13, config.vocab_size)
    assert moe_aux["semantic_bias"].shape == (2, 13, config.n_experts)
    assert moe_aux["semantic_prior_entropy"].ndim == 0

    loss = out["logits"].float().mean() + moe_aux["semantic_prior_entropy"] * 0.0
    loss.backward()


def test_v3_multiscale_semantic_forward_and_backward():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=64,
        d_model=32,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=48,
        semantic_router_mode="hybrid",
        semantic_scales="local_mid_global",
        mid_stride=8,
        mid_window=16,
        use_global_semantic=True,
        semantic_fusion="gated_sum",
        use_semantic_residual_write=True,
        semantic_write_scale=0.05,
        semantic_pred_horizon=1,
    )
    model = build_model("naime_state_moe", config)
    input_ids = torch.randint(1, config.vocab_size, (2, 31))

    out = model(input_ids)
    semantic_aux = out["aux"][-1]["semantic"]
    moe_aux = out["aux"][-1]["moe"]

    assert out["logits"].shape == (2, 31, config.vocab_size)
    assert semantic_aux["fusion_weights"].shape[-1] == 3
    assert semantic_aux["semantic_pred_loss"].ndim == 0
    assert moe_aux["semantic_prior_entropy"].ndim == 0

    loss = out["logits"].float().mean() + semantic_aux["semantic_pred_loss"]
    loss.backward()


def test_v4_state_memory_forward_and_backward():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=64,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=48,
        semantic_router_mode="hybrid",
        semantic_scales="local_mid_global",
        mid_stride=8,
        mid_window=16,
        use_global_semantic=True,
        semantic_fusion="concat",
        use_semantic_residual_write=True,
        semantic_write_scale=0.03,
        semantic_pred_horizon=1,
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_memory_write_scale=0.03,
        semantic_state_write_scale=0.03,
        semantic_gate_mixer=True,
        layerwise_semantic_schedule=True,
    )
    model = build_model("naime_v4_state_moe", config)
    assert isinstance(model, NAIMEV4StateMoEDecoder)
    input_ids = torch.randint(1, config.vocab_size, (2, 31))

    out = model(input_ids)
    v4_aux = out["aux"][-1]["v4"]

    assert out["logits"].shape == (2, 31, config.vocab_size)
    assert v4_aux["state_norm"].ndim == 0
    assert v4_aux["memory_norm"].ndim == 0
    assert v4_aux["state_confidence"].ndim == 0

    loss = out["logits"].float().mean() + v4_aux["memory_gate"] * 0.0
    loss.backward()


def test_v41_calibrated_state_and_mixer_metrics_forward_and_backward():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=64,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=48,
        semantic_router_mode="hybrid",
        semantic_scales="local_mid_global",
        mid_stride=8,
        mid_window=16,
        use_global_semantic=True,
        semantic_fusion="concat",
        use_semantic_residual_write=True,
        semantic_write_scale=0.03,
        semantic_pred_horizon=1,
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_memory_write_scale=0.025,
        semantic_state_write_scale=0.035,
        semantic_gate_mixer=True,
        semantic_gate_mixer_temperature=1.35,
        semantic_gate_mixer_min_weight=0.08,
        semantic_state_confidence_mode="hybrid",
        semantic_state_confidence_temperature=3.0,
        layerwise_semantic_schedule=True,
    )
    model = build_model("naime_v41_state_moe", config)
    input_ids = torch.randint(1, config.vocab_size, (2, 31))

    out = model(input_ids)
    v4_aux = out["aux"][-1]["v4"]
    semantic_aux = out["aux"][-1]["semantic"]

    assert out["logits"].shape == (2, 31, config.vocab_size)
    assert v4_aux["state_delta"].ndim == 0
    assert v4_aux["state_agreement"].ndim == 0
    assert v4_aux["memory_read_strength"].ndim == 0
    assert v4_aux["memory_novelty"].ndim == 0
    assert semantic_aux["gate_mix_weights"][..., 0].min() >= config.semantic_gate_mixer_min_weight - 1e-6

    loss = out["logits"].float().mean() + v4_aux["state_delta"] * 0.0 + v4_aux["memory_novelty"] * 0.0
    loss.backward()


def test_v42_accountable_state_memory_gates_forward_and_backward():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=64,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=48,
        semantic_router_mode="hybrid",
        semantic_scales="local_mid_global",
        mid_stride=8,
        mid_window=16,
        use_global_semantic=True,
        semantic_fusion="concat",
        use_semantic_residual_write=True,
        semantic_write_scale=0.03,
        semantic_pred_horizon=1,
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_memory_write_scale=0.035,
        semantic_state_write_scale=0.045,
        semantic_gate_mixer=True,
        semantic_gate_mixer_temperature=1.60,
        semantic_gate_mixer_min_weight=0.08,
        semantic_gate_mixer_max_clean_weight=0.58,
        semantic_state_confidence_mode="hybrid",
        semantic_state_confidence_temperature=3.0,
        semantic_state_confidence_gate=True,
        semantic_memory_read_gate=True,
        layerwise_semantic_schedule=True,
    )
    model = build_model("naime_v42_state_moe", config)
    input_ids = torch.randint(1, config.vocab_size, (2, 31))

    out = model(input_ids)
    v4_aux = out["aux"][-1]["v4"]
    semantic_aux = out["aux"][-1]["semantic"]

    assert out["logits"].shape == (2, 31, config.vocab_size)
    assert semantic_aux["gate_mix_weights"][..., 1].max() <= config.semantic_gate_mixer_max_clean_weight + 1e-6
    assert v4_aux["memory_read_strength"].ndim == 0
    assert v4_aux["state_confidence"].ndim == 0

    loss = out["logits"].float().mean() + v4_aux["memory_read_strength"] * 0.0
    loss.backward()


def test_v5_world_state_slots_forward_and_backward():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=64,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=48,
        semantic_router_mode="hybrid",
        semantic_scales="local_mid_global",
        mid_stride=8,
        mid_window=16,
        use_global_semantic=True,
        semantic_fusion="concat",
        use_semantic_residual_write=True,
        semantic_write_scale=0.03,
        semantic_pred_horizon=1,
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_memory_write_scale=0.035,
        semantic_state_write_scale=0.045,
        semantic_gate_mixer=True,
        semantic_gate_mixer_temperature=1.60,
        semantic_gate_mixer_min_weight=0.08,
        semantic_gate_mixer_max_clean_weight=0.58,
        semantic_state_confidence_mode="hybrid",
        semantic_state_confidence_temperature=3.0,
        semantic_state_confidence_gate=True,
        semantic_memory_read_gate=True,
        layerwise_semantic_schedule=True,
        world_state_slots=4,
    )
    model = build_model("naime_v5_world_state_moe", config)
    assert isinstance(model, NAIMEV5WorldStateMoEDecoder)
    input_ids = torch.randint(1, config.vocab_size, (2, 31))

    out = model(input_ids)
    v5_aux = out["aux"][-1]["v5"]

    assert out["logits"].shape == (2, 31, config.vocab_size)
    assert v5_aux["state_pred"].ndim == 0
    assert v5_aux["slot_cosine"].ndim == 0
    assert v5_aux["slot_read_entropy"].ndim == 0

    loss = out["logits"].float().mean() + v5_aux["state_pred"] + v5_aux["slot_diversity"] * 0.0
    loss.backward()


def test_v6_recursive_self_state_forward_and_backward():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=64,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=48,
        semantic_router_mode="hybrid",
        semantic_scales="local_mid_global",
        mid_stride=8,
        mid_window=16,
        use_global_semantic=True,
        semantic_fusion="concat",
        use_semantic_residual_write=True,
        semantic_write_scale=0.03,
        semantic_pred_horizon=1,
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_gate_mixer=True,
        world_state_slots=4,
        self_state_slots=3,
        self_state_recursion_depth=2,
    )
    model = build_model("naime_v6_recursive_self_moe", config)
    assert isinstance(model, NAIMEV6RecursiveSelfMoEDecoder)
    input_ids = torch.randint(1, config.vocab_size, (2, 31))

    out = model(input_ids)
    v6_aux = out["aux"][-1]["v6"]

    assert out["logits"].shape == (2, 31, config.vocab_size)
    assert out["self_state"].shape == (2, config.self_state_slots, config.d_model)
    assert v6_aux["self_pred"].ndim == 0
    assert v6_aux["boundary_entropy"].ndim == 0
    assert v6_aux["reflection_norm"].ndim == 0
    assert v6_aux["slot_context_cosine"].ndim == 0

    loss = out["logits"].float().mean() + v6_aux["self_pred"] + v6_aux["slot_diversity"] * 0.0
    loss.backward()


def test_topk_moe_sparse_dispatch_matches_dense_dispatch():
    torch.manual_seed(1234)
    dense = TopKMoE(
        d_model=16,
        semantic_dim=16,
        n_experts=4,
        top_k=2,
        expert_hidden_dim=32,
        use_semantic_router=False,
        dispatch_mode="dense",
    )
    sparse = TopKMoE(
        d_model=16,
        semantic_dim=16,
        n_experts=4,
        top_k=2,
        expert_hidden_dim=32,
        use_semantic_router=False,
        dispatch_mode="sparse",
    )
    sparse.load_state_dict(dense.state_dict())
    x_dense = torch.randn(2, 7, 16, requires_grad=True)
    x_sparse = x_dense.detach().clone().requires_grad_()

    y_dense, aux_dense = dense(x_dense)
    y_sparse, aux_sparse = sparse(x_sparse)

    assert torch.allclose(y_sparse, y_dense, atol=1e-6)
    assert torch.equal(aux_sparse["topk_indices"], aux_dense["topk_indices"])
    assert torch.allclose(aux_sparse["topk_weights"], aux_dense["topk_weights"], atol=1e-6)
    assert torch.allclose(aux_sparse["token_load"], aux_dense["token_load"], atol=1e-6)

    dense_loss = y_dense.float().pow(2).mean()
    sparse_loss = y_sparse.float().pow(2).mean()
    dense_loss.backward()
    sparse_loss.backward()
    assert torch.allclose(x_sparse.grad, x_dense.grad, atol=1e-6)


def test_topk_moe_auto_dispatch_matches_dense_dispatch_for_small_expert_cuda_heuristic():
    torch.manual_seed(1234)
    dense = TopKMoE(
        d_model=16,
        semantic_dim=16,
        n_experts=4,
        top_k=2,
        expert_hidden_dim=32,
        use_semantic_router=False,
        dispatch_mode="dense",
    )
    auto = TopKMoE(
        d_model=16,
        semantic_dim=16,
        n_experts=4,
        top_k=2,
        expert_hidden_dim=32,
        use_semantic_router=False,
        dispatch_mode="auto",
    )
    auto.load_state_dict(dense.state_dict())
    x_dense = torch.randn(2, 128, 16, requires_grad=True)
    x_auto = x_dense.detach().clone().requires_grad_()

    dense_dispatch = auto._resolve_dispatch_mode(x_auto)
    assert dense_dispatch == "dense"

    y_dense, aux_dense = dense(x_dense)
    y_auto, aux_auto = auto(x_auto)

    assert torch.allclose(y_auto, y_dense, atol=1e-6)
    assert torch.equal(aux_auto["topk_indices"], aux_dense["topk_indices"])
    assert torch.allclose(aux_auto["topk_weights"], aux_dense["topk_weights"], atol=1e-6)
    assert float(aux_auto["dispatch_dense"]) == 1.0

    dense_loss = y_dense.float().pow(2).mean()
    auto_loss = y_auto.float().pow(2).mean()
    dense_loss.backward()
    auto_loss.backward()
    assert torch.allclose(x_auto.grad, x_dense.grad, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA dispatch heuristic only applies on GPU")
def test_topk_moe_auto_dispatch_uses_dense_for_six_experts_on_cuda():
    moe = TopKMoE(
        d_model=16,
        semantic_dim=16,
        n_experts=6,
        top_k=2,
        expert_hidden_dim=32,
        use_semantic_router=False,
        dispatch_mode="auto",
    ).cuda()
    x = torch.randn(2, 512, 16, device="cuda")

    assert moe._resolve_dispatch_mode(x) == "dense"


def test_full_attention_mask_is_dropped_for_causal_fast_path():
    mask = torch.ones(2, 8, dtype=torch.bool)

    prepared, infer_pad_mask = prepare_attention_mask_for_device(mask, torch.device("cpu"))

    assert prepared is None
    assert infer_pad_mask is False


def test_decoder_full_mask_matches_causal_fast_path_without_pad_inference():
    torch.manual_seed(1234)
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_dense_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
    )
    model = NAIMEStateMoEDecoder(config).eval()
    input_ids = torch.randint(1, config.vocab_size, (2, 16))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        masked = model(input_ids, attention_mask=attention_mask, return_aux=False)["logits"]
        fast = model(input_ids, attention_mask=None, infer_pad_mask=False, return_aux=False)["logits"]

    assert torch.allclose(fast, masked, atol=1e-5)


def test_mla_attention_forward_and_backward():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=32,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=2,
        top_k=1,
        expert_hidden_dim=48,
        attention_type="mla",
        mla_latent_dim=32,
        mla_rope_per_head=4,
    )
    model = build_model("naime_v5_world_state_moe", config)
    assert isinstance(model, NAIMEV5WorldStateMoEDecoder)
    input_ids = torch.randint(1, config.vocab_size, (2, 17))

    out = model(input_ids)
    assert out["logits"].shape == (2, 17, config.vocab_size)

    loss = out["logits"].float().mean()
    loss.backward()

    for name, param in model.named_parameters():
        if "kv_compress" in name:
            assert param.grad is not None, f"MLA parameter {name} has no gradient"

    assert True


def test_sparse_controller_strengthens_when_alpha_is_off_target_on_either_side():
    higher, ema = update_sparse_lambda(
        current_lambda=0.01,
        alpha_ema=None,
        alpha_mean=0.6,
        target_sparsity=0.2,
        ema_decay=0.95,
        gain=0.1,
        min_value=1e-4,
        max_value=1.0,
    )
    lower, _ = update_sparse_lambda(
        current_lambda=0.01,
        alpha_ema=ema,
        alpha_mean=0.05,
        target_sparsity=0.8,
        ema_decay=0.0,
        gain=0.1,
        min_value=1e-4,
        max_value=1.0,
    )

    assert higher > 0.01
    assert lower > 0.01


def test_sparse_controller_relaxes_inside_deadband():
    relaxed, _ = update_sparse_lambda(
        current_lambda=0.01,
        alpha_ema=None,
        alpha_mean=0.51,
        target_sparsity=0.5,
        ema_decay=0.0,
        gain=0.1,
        min_value=1e-4,
        max_value=1.0,
        deadband=0.03,
    )

    assert relaxed < 0.01


def test_gate_eval_prob_mode_uses_soft_probability():
    gate = GumbelBlockGate(d_model=4, target_sparsity=0.2)
    gate.eval()
    x = torch.zeros(3, 4)

    alpha, _, prob, clean_prob = gate(x, eval_mode="prob")
    hard_alpha, _, _, _ = gate(x, eval_mode="hard")

    assert torch.allclose(alpha, prob)
    assert torch.allclose(prob, clean_prob)
    assert set(hard_alpha.tolist()).issubset({0.0, 1.0})


def test_reference_value_uses_latest_available_step():
    curve = [(500, 5.0), (1000, 4.5), (1500, 4.0)]

    assert reference_value_at_step(curve, 200) == (500, 5.0)
    assert reference_value_at_step(curve, 1200) == (1000, 4.5)
    assert reference_value_at_step(curve, 2000) == (1500, 4.0)


def _compile_v6_config():
    return NAIMEStateMoEConfig(
        vocab_size=96,
        max_seq_len=64,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        window=8,
        z_dim=8,
        n_experts=3,
        top_k=2,
        expert_hidden_dim=48,
        semantic_router_mode="hybrid",
        semantic_scales="local_mid_global",
        mid_stride=8,
        mid_window=16,
        use_global_semantic=True,
        semantic_fusion="concat",
        use_semantic_residual_write=True,
        semantic_write_scale=0.03,
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_gate_mixer=True,
        world_state_slots=4,
        self_state_slots=3,
        self_state_recursion_depth=2,
    )


def test_compile_v6_forward_no_crash():
    _require_torch_compile_support()
    warnings.filterwarnings("ignore")
    torch._logging.set_logs(dynamo=40, inductor=40)

    torch.manual_seed(1)
    config = _compile_v6_config()
    model = build_model("naime_v6_recursive_self_moe", config)
    input_ids = torch.randint(1, config.vocab_size, (2, 31))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "inductor" if device.type == "cuda" else "eager"
    model = model.to(device)
    input_ids = input_ids.to(device)

    compiled = torch.compile(model, mode="reduce-overhead", backend=backend)
    out = compiled(input_ids)
    assert out["logits"].shape == (2, 31, config.vocab_size)


def test_compile_v6_backward_no_crash():
    _require_torch_compile_support()
    warnings.filterwarnings("ignore")
    torch._logging.set_logs(dynamo=40, inductor=40)

    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "inductor" if device.type == "cuda" else "eager"
    config = _compile_v6_config()
    model = build_model("naime_v6_recursive_self_moe", config).to(device)
    input_ids = torch.randint(1, config.vocab_size, (2, 31), device=device)

    compiled = torch.compile(model, mode="reduce-overhead", backend=backend)
    out = compiled(input_ids)
    loss = out["logits"].float().mean()
    loss.backward()
    grads = []
    for name, p in compiled.named_parameters():
        if p.grad is not None:
            grads.append((name, p.grad))
    assert len(grads) > 0, "no parameters received gradients"


def test_compile_v6_preserves_causality():
    _require_torch_compile_support()
    warnings.filterwarnings("ignore")
    torch._logging.set_logs(dynamo=40, inductor=40)

    torch.manual_seed(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "inductor" if device.type == "cuda" else "eager"
    config = _compile_v6_config()
    model = build_model("naime_v6_recursive_self_moe", config).eval().to(device)
    input_ids = torch.randint(1, config.vocab_size, (2, 31), device=device)
    changed = input_ids.clone()
    cutoff = 12
    changed[:, cutoff:] = torch.randint(1, config.vocab_size, changed[:, cutoff:].shape, device=device)

    compiled = torch.compile(model, mode="reduce-overhead", backend=backend)
    with torch.no_grad():
        original_logits = compiled(input_ids)["logits"]
        changed_logits = compiled(changed)["logits"]

    assert torch.allclose(
        original_logits[:, :cutoff, :],
        changed_logits[:, :cutoff, :],
        atol=1e-5,
        rtol=1e-5,
    )
