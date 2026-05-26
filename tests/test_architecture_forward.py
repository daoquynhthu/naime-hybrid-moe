import math
import warnings

import pytest
import torch

from naime_hybrid import (
    NAIMEStateMoEConfig,
    NAIMEStateMoEDecoder,
    NAIMEStatePacket,
    NAIMEV4StateMoEDecoder,
    NAIMEV5WorldStateMoEDecoder,
    NAIMEV6RecursiveSelfMoEDecoder,
    NAIMEV7TypedDynamicsDecoder,
    ObservationPacket,
    build_model,
)
from naime_hybrid.data import HFDiskCausalDataset
from naime_hybrid.modules.gate import GumbelBlockGate
from naime_hybrid.modules.moe import TopKMoE
from naime_hybrid.modules.self_state import RecursiveSelfState
from naime_hybrid.modules.world_state import WorldStateSlots
from naime_hybrid.training.checkpoint import (
    build_checkpoint_payload,
    build_model_payload,
    load_checkpoint,
    normalize_state_dict_for_model,
    save_checkpoint,
    save_payloads_in_subprocess,
)
from naime_hybrid.training.cli import build_train_config, parse_args
from naime_hybrid.training.config import TrainConfig
from naime_hybrid.training.control import reference_value_at_step, update_sparse_lambda
from naime_hybrid.training.losses import lm_loss
from naime_hybrid.training.masks import prepare_attention_mask_for_device
from naime_hybrid.training.runtime import split_stateful_chunks
from naime_hybrid.training.train import _apply_self_state_warmup
from naime_hybrid.training.validation import evaluate_model


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


def test_checkpoint_payload_falls_back_to_model_only_when_optimizer_snapshot_fails():
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    model = build_model("dense", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def broken_state_dict():
        raise RuntimeError("simulated optimizer CPU snapshot failure")

    optimizer.state_dict = broken_state_dict

    payload = build_checkpoint_payload(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        step=12,
        config={},
        metrics={"loss_lm": 1.25},
        fallback_to_model_only=True,
    )

    assert payload["step"] == 12
    assert payload["checkpoint_kind"] == "model_only_fallback"
    assert "simulated optimizer" in payload["checkpoint_fallback_reason"]
    assert payload["optimizer"] is None
    assert payload["scheduler"] is None
    assert payload["model"]


def test_save_checkpoint_fallback_is_loadable_without_optimizer_state(tmp_path):
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    model = build_model("dense", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def broken_state_dict():
        raise RuntimeError("simulated optimizer CPU snapshot failure")

    optimizer.state_dict = broken_state_dict
    path = tmp_path / "fallback_latest.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        step=21,
        config={},
        metrics={},
        fallback_to_model_only=True,
    )

    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert restored["checkpoint_kind"] == "model_only_fallback"

    fresh = build_model("dense", config)
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    step = load_checkpoint(path, fresh, fresh_optimizer, scheduler=None, scaler=None)

    assert step == 21


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


def test_split_stateful_chunks_preserves_contiguous_causal_alignment():
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
    labels = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    chunks = split_stateful_chunks(input_ids, labels, attention_mask, chunk_len=3, target_chunks=2)

    assert len(chunks) == 2
    assert torch.equal(chunks[0]["input_ids"], torch.tensor([[10, 11, 12]]))
    assert torch.equal(chunks[0]["labels"], torch.tensor([[11, 12, 13]]))
    assert torch.equal(chunks[1]["input_ids"], torch.tensor([[13, 14, 15]]))
    assert torch.equal(chunks[1]["labels"], torch.tensor([[14, 15, 16]]))
    assert chunks[0]["attention_mask"].all()
    assert chunks[1]["attention_mask"].all()


def test_cli_maps_stateful_carry_and_doc_continuity_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--architecture",
            "naime_v7_typed_dynamics",
            "--eval-doc-continuity",
            "--eval-doc-continuity-docs",
            "7",
            "--eval-doc-continuity-chunks",
            "3",
            "--stateful-batch-ratio",
            "0.1",
            "--stateful-chunk-len",
            "256",
            "--lambda-stateful-carry",
            "0.0005",
            "--stateful-carry-margin",
            "0.001",
            "--self-state-hidden-scale-warmup-steps",
            "2000",
            "--self-state-context-score-warmup-steps",
            "1500",
            "--self-state-context-score-start",
            "0.5",
        ],
    )

    config = build_train_config(parse_args())

    assert config.eval_doc_continuity is True
    assert config.eval_doc_continuity_docs == 7
    assert config.eval_doc_continuity_chunks == 3
    assert config.stateful_batch_ratio == pytest.approx(0.1)
    assert config.stateful_chunk_len == 256
    assert config.lambda_stateful_carry == pytest.approx(5e-4)
    assert config.stateful_carry_margin == pytest.approx(1e-3)
    assert config.self_state_hidden_scale_warmup_steps == 2000
    assert config.self_state_context_score_warmup_steps == 1500
    assert config.self_state_context_score_start == pytest.approx(0.5)


def test_apply_self_state_warmup_updates_effective_scales():
    model_config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        world_state_slots=2,
        self_state_slots=2,
        self_state_hidden_scale=0.02,
        self_state_context_score_scale=4.0,
    )
    model = build_model("naime_v6_recursive_self_moe", model_config)
    train_config = TrainConfig(
        architecture="naime_v6_recursive_self_moe",
        self_state_hidden_scale_warmup_steps=2000,
        self_state_context_score_warmup_steps=2000,
        self_state_context_score_start=1.0,
        model=model_config,
    )

    hidden_scale, context_scale = _apply_self_state_warmup(model, train_config, step=1000)

    assert hidden_scale == pytest.approx(0.01)
    assert context_scale == pytest.approx(2.5)
    assert model.self_state_slots.hidden_scale == pytest.approx(0.01)
    assert model.self_state_slots.context_score_scale == pytest.approx(2.5)


def test_evaluate_model_reports_doc_continuity_metrics_for_v7():
    torch.manual_seed(1234)
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=32,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        world_state_slots=2,
        self_state_slots=2,
        v7_dynamics_steps=1,
        v7_latent_slots=2,
        n_experts=2,
        top_k=1,
        expert_hidden_dim=48,
        stride=4,
        window=8,
        z_dim=8,
    )
    model = build_model("naime_v7_typed_dynamics", config)
    input_ids = torch.randint(1, config.vocab_size, (2, 12))
    labels = torch.randint(1, config.vocab_size, (2, 12))
    loader = [{"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids, dtype=torch.bool)}]

    metrics = evaluate_model(
        model,
        loader,
        config,
        device=torch.device("cpu"),
        use_amp=False,
        max_batches=1,
        doc_continuity=True,
        doc_continuity_docs=1,
        doc_continuity_chunks=3,
        stateful_chunk_len=4,
    )

    assert "val_doc_carry_gain_mean" in metrics
    assert "val_doc_carry_gain_cumulative" in metrics
    assert "val_doc_carry_gain_slope" in metrics
    assert metrics["val_doc_continuity_batches"] == 1.0
    assert math.isfinite(metrics["val_doc_carry_gain_mean"])


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
        context.float().square().mean() + traced_state.float().square().mean() + metrics["history_read_entropy"].float()
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
        "naime_v7_typed_dynamics",
    ]:
        model = build_model(architecture, config)
        out = model(input_ids)
        assert out["logits"].shape == (2, 13, config.vocab_size)


def test_v7_typed_dynamics_forward_state_packet_and_backward():
    torch.manual_seed(2040)
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
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_gate_mixer=True,
        world_state_slots=4,
        self_state_slots=3,
        self_state_recursion_depth=1,
        v7_dynamics_steps=1,
        v7_latent_slots=5,
        v7_hidden_write_scale=0.01,
        v7_max_hidden_write_ratio=0.05,
        v7_controller_slots=2,
        v7_controller_write_scale=0.015,
        v7_latent_timescale=0.5,
        v7_world_timescale=0.75,
        v7_self_timescale=0.25,
    )
    model = build_model("naime_v7_typed_dynamics", config)
    assert isinstance(model, NAIMEV7TypedDynamicsDecoder)
    input_ids = torch.randint(1, config.vocab_size, (2, 31))

    out = model(input_ids, return_state=True)
    packet = out["state_packet"]
    v7_aux = out["aux"][-1]["v7"]

    assert isinstance(packet, NAIMEStatePacket)
    assert packet.world_state is not None
    assert packet.self_state is not None
    assert packet.latent_field is not None
    assert packet.controller_state is not None
    assert packet.world_state.shape == (2, config.world_state_slots, config.d_model)
    assert packet.self_state.shape == (2, config.self_state_slots, config.d_model)
    assert packet.latent_field.shape == (2, 5, config.d_model)
    assert packet.controller_state.shape == (2, 2, config.d_model)
    assert packet.architecture_id == "naime_v7_typed_dynamics"
    assert out["logits"].shape == (2, 31, config.vocab_size)
    assert v7_aux["v7_thought_steps"].item() == pytest.approx(1.0)
    assert torch.isfinite(v7_aux["v7_latent_delta"])
    assert torch.isfinite(v7_aux["v7_hidden_delta"])
    assert torch.isfinite(v7_aux["v7_world_delta"])
    assert torch.isfinite(v7_aux["v7_self_delta"])
    assert torch.isfinite(v7_aux["v7_controller_delta"])
    assert torch.isfinite(v7_aux["v7_world_write_gate"])
    assert torch.isfinite(v7_aux["v7_self_write_gate"])
    assert torch.isfinite(v7_aux["v7_controller_write_gate"])
    assert v7_aux["v7_past_latent_read_suppressed"].item() == pytest.approx(1.0)
    assert v7_aux["v7_hidden_write_ratio"].item() == pytest.approx(0.0)
    assert v7_aux["v7_latent_timescale"].item() == pytest.approx(0.5)
    assert v7_aux["v7_world_timescale"].item() == pytest.approx(0.75)
    assert v7_aux["v7_self_timescale"].item() == pytest.approx(0.25)
    assert v7_aux["v7_controller_fixed"].item() == pytest.approx(1.0)
    assert v7_aux["v7_effective_controller_write_scale"].item() == pytest.approx(0.015)

    second = model(input_ids[:, :17], past_state=packet, return_state=True)
    second_v7_aux = second["aux"][-1]["v7"]
    assert second["state_packet"].world_state is not None
    assert second["state_packet"].self_state is not None
    assert second["state_packet"].latent_field is not None
    assert second["state_packet"].controller_state is not None
    assert second["logits"].shape == (2, 17, config.vocab_size)
    assert second_v7_aux["v7_past_latent_adapt_steps"].item() == pytest.approx(1.0)
    assert second_v7_aux["v7_past_latent_read_suppressed"].item() == pytest.approx(1.0)
    assert second_v7_aux["v7_hidden_write_ratio"].item() == pytest.approx(0.0)

    loss = out["logits"].float().mean() + v7_aux["v7_latent_delta"] * 0.0
    loss.backward()


def test_state_packet_rejects_traced_public_state():
    packet = NAIMEStatePacket(
        world_state=torch.randn(2, 3, 4, 32),
        self_state=torch.randn(2, 3, 3, 32),
        latent_field=torch.randn(2, 5, 32),
        controller_state=torch.randn(2, 2, 32),
    )

    with pytest.raises(ValueError, match="compact final-slot shape"):
        packet.validate_batch(2)


def test_v7_typed_dynamics_preserves_causal_prefix_with_multiple_steps():
    torch.manual_seed(2041)
    config = NAIMEStateMoEConfig(
        vocab_size=80,
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
        self_state_recursion_depth=1,
        v7_dynamics_steps=2,
        v7_latent_slots=5,
        v7_hidden_write_scale=0.02,
        v7_max_hidden_write_ratio=0.08,
    )
    model = build_model("naime_v7_typed_dynamics", config).eval()
    input_ids = torch.randint(1, config.vocab_size, (2, 31))
    changed = input_ids.clone()
    cutoff = 13
    changed[:, cutoff:] = torch.randint(1, config.vocab_size, changed[:, cutoff:].shape)

    with torch.no_grad():
        original = model(input_ids, return_state=True)
        changed_out = model(changed, return_state=True)

    assert torch.allclose(original["logits"][:, :cutoff, :], changed_out["logits"][:, :cutoff, :], atol=1e-5, rtol=1e-5)
    assert original["aux"][-1]["v7"]["v7_hidden_write_ratio"].item() == pytest.approx(0.0)
    assert original["aux"][-1]["v7"]["v7_past_latent_read_suppressed"].item() == pytest.approx(2.0)

    prefix_state = model(input_ids[:, :cutoff], return_state=True)["state_packet"]
    with torch.no_grad():
        original_with_past = model(input_ids, past_state=prefix_state, return_state=True)
        changed_with_past = model(changed, past_state=prefix_state, return_state=True)

    assert torch.allclose(
        original_with_past["logits"][:, :cutoff, :],
        changed_with_past["logits"][:, :cutoff, :],
        atol=1e-5,
        rtol=1e-5,
    )
    assert original_with_past["aux"][-1]["v7"]["v7_past_latent_read_suppressed"].item() == pytest.approx(1.0)


def test_v7_dynamic_depth_halts_after_minimum_step():
    torch.manual_seed(2042)
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=16,
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
        semantic_memory_slots=2,
        world_state_slots=3,
        self_state_slots=3,
        v7_dynamics_steps=3,
        v7_latent_slots=3,
        v7_dynamic_depth=True,
        v7_min_dynamics_steps=1,
        v7_dynamic_convergence_threshold=1e9,
    )
    model = build_model("naime_v7_typed_dynamics", config)
    input_ids = torch.randint(1, config.vocab_size, (2, config.max_seq_len))
    out = model(input_ids)
    v7_aux = out["aux"][-1]["v7"]

    assert v7_aux["v7_dynamic_depth_enabled"].item() == pytest.approx(1.0)
    assert v7_aux["v7_thought_steps"].item() == pytest.approx(1.0)
    assert v7_aux["v7_dynamic_halt_fraction"].item() == pytest.approx(1.0)
    assert torch.isfinite(v7_aux["v7_dynamic_continue_score"])


def test_v7_homeostatic_controller_reports_bounded_rate_scales():
    torch.manual_seed(2044)
    config = NAIMEStateMoEConfig(
        vocab_size=64,
        max_seq_len=24,
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
        semantic_memory_slots=2,
        world_state_slots=3,
        self_state_slots=3,
        v7_dynamics_steps=3,
        v7_latent_slots=3,
        v7_homeostatic_control=True,
        v7_homeostatic_strength=0.4,
        v7_homeostatic_min_scale=0.6,
        v7_homeostatic_max_scale=1.4,
    )
    model = build_model("naime_v7_typed_dynamics", config)
    input_ids = torch.randint(1, config.vocab_size, (2, config.max_seq_len))

    out = model(input_ids)
    v7_aux = out["aux"][-1]["v7"]

    assert v7_aux["v7_homeostatic_control_enabled"].item() == pytest.approx(1.0)
    assert torch.isfinite(v7_aux["v7_homeostatic_dhi"])
    assert torch.isfinite(v7_aux["v7_homeostatic_balance_pressure"])
    assert torch.isfinite(v7_aux["v7_homeostatic_accel_pressure"])
    for key in (
        "v7_latent_rate_scale",
        "v7_world_rate_scale",
        "v7_self_rate_scale",
        "v7_hidden_read_rate_scale",
    ):
        value = v7_aux[key].item()
        assert 0.6 <= value <= 1.4


def test_v7_refactored_dynamics_reports_geometry_carry_and_tau_metrics():
    torch.manual_seed(2045)
    config = NAIMEStateMoEConfig(
        vocab_size=72,
        max_seq_len=24,
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
        semantic_memory_slots=2,
        world_state_slots=3,
        self_state_slots=3,
        v7_dynamics_steps=2,
        v7_latent_slots=3,
        v7_state_compatibility_gate=True,
        v7_state_compatibility_min=0.2,
        v7_adaptive_tau=True,
        v7_adaptive_tau_min=0.6,
        v7_adaptive_tau_max=1.3,
        v7_hyperspherical_state=True,
        v7_causal_summary=True,
        v7_causal_summary_decay=0.97,
    )
    model = build_model("naime_v7_typed_dynamics", config)
    input_ids = torch.randint(1, config.vocab_size, (2, config.max_seq_len))

    first = model(input_ids, return_state=True)
    second = model(input_ids, past_state=first["state_packet"], return_state=True)
    v7_aux = second["aux"][-1]["v7"]

    assert v7_aux["v7_hyperspherical_state_enabled"].item() == pytest.approx(1.0)
    assert v7_aux["v7_causal_summary_enabled"].item() == pytest.approx(1.0)
    assert v7_aux["v7_state_compatibility_enabled"].item() == pytest.approx(1.0)
    assert v7_aux["v7_adaptive_tau_enabled"].item() == pytest.approx(1.0)
    assert 0.2 <= v7_aux["v7_carry_latent_gate"].item() <= 1.0
    assert torch.isfinite(v7_aux["v7_carry_blend_delta"])
    assert v7_aux["v7_ingress_compatibility_enabled"].item() == pytest.approx(1.0)
    for key in (
        "v7_ingress_latent_gate",
        "v7_ingress_world_gate",
        "v7_ingress_self_gate",
        "v7_ingress_controller_gate",
        "v7_ingress_memory_gate",
    ):
        assert 0.2 <= v7_aux[key].item() <= 1.0
    assert second["state_packet"].world_state is not None
    assert second["state_packet"].self_state is not None
    assert second["state_packet"].world_state.ndim == 3
    assert second["state_packet"].self_state.ndim == 3
    for key in ("v7_latent_tau", "v7_world_tau", "v7_self_tau", "v7_controller_tau"):
        value = v7_aux[key].item()
        assert 0.6 <= value <= 1.3


def test_world_router_modulation_path_is_observable():
    torch.manual_seed(2046)
    config = NAIMEStateMoEConfig(
        vocab_size=72,
        max_seq_len=24,
        d_model=32,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        stride=4,
        causal_state_stride=8,
        window=8,
        z_dim=8,
        n_experts=2,
        top_k=1,
        expert_hidden_dim=48,
        world_state_slots=3,
        world_router_mode="modulate",
        world_router_modulation_scale=0.25,
        world_router_max_ratio=0.10,
    )
    model = build_model("naime_v5_world_state_moe", config)
    input_ids = torch.randint(1, config.vocab_size, (2, config.max_seq_len))

    out = model(input_ids)
    v5_aux = out["aux"][-1]["v5"]

    assert v5_aux["router_world_modulation"].item() == pytest.approx(1.0)
    assert torch.isfinite(v5_aux["router_world_ratio"])


def test_v7_causal_state_chunks_enable_later_hidden_read_without_prefix_leakage():
    torch.manual_seed(2043)
    config = NAIMEStateMoEConfig(
        vocab_size=80,
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
        semantic_memory_slots=2,
        world_state_slots=3,
        self_state_slots=3,
        v7_dynamics_steps=1,
        v7_latent_slots=3,
        v7_hidden_write_scale=0.03,
        v7_max_hidden_write_ratio=0.08,
        v7_state_chunk_size=8,
        v7_internal_latent_adapt_steps=0,
    )
    model = build_model("naime_v7_typed_dynamics", config).eval()
    input_ids = torch.randint(1, config.vocab_size, (2, 32))
    changed = input_ids.clone()
    cutoff = 8
    changed[:, cutoff:] = torch.randint(1, config.vocab_size, changed[:, cutoff:].shape)

    with torch.no_grad():
        original = model(input_ids, return_state=True)
        changed_out = model(changed, return_state=True)

    v7_aux = original["aux"][-1]["v7"]
    assert v7_aux["v7_causal_segments"].item() == pytest.approx(4.0)
    assert v7_aux["v7_hidden_write_ratio"].item() > 0.0
    assert v7_aux["v7_past_latent_read_suppressed"].item() == pytest.approx(1.0)
    assert torch.allclose(
        original["logits"][:, :cutoff, :],
        changed_out["logits"][:, :cutoff, :],
        atol=1e-5,
        rtol=1e-5,
    )


def test_state_and_observation_packets_preserve_batch_and_metadata():
    world_state = torch.randn(2, 3, 8)
    controller_state = torch.randn(2, 2, 8, requires_grad=True)
    packet = NAIMEStatePacket(
        world_state=world_state,
        controller_state=controller_state,
        architecture_id="naime_v7_typed_dynamics",
        protocol_version="state-protocol-v1",
    )

    packet.validate_batch(2)
    detached = packet.detach()

    assert detached.controller_state is not None
    assert detached.controller_state.requires_grad is False
    assert detached.architecture_id == "naime_v7_typed_dynamics"

    obs = ObservationPacket(
        modality="image",
        embeddings=torch.randn(2, 5, 8, requires_grad=True),
        attention_mask=torch.ones(2, 5, dtype=torch.bool),
        time_index=torch.arange(2),
        spatial_anchors=torch.randn(2, 5, 4),
        confidence=torch.ones(2, 5),
        provenance="unit-test",
        causal_segment_id="frame-0",
    )
    obs.validate_batch(2)
    obs_detached = obs.detach()

    assert obs_detached.embeddings.requires_grad is False
    assert obs_detached.modality == "image"
    assert obs_detached.provenance == "unit-test"
    assert obs_detached.causal_segment_id == "frame-0"


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


def test_v6_state_packet_carries_compact_latent_state_across_chunks():
    torch.manual_seed(2031)
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
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_gate_mixer=True,
        world_state_slots=4,
        self_state_slots=3,
        self_state_recursion_depth=2,
    )
    model = build_model("naime_v6_recursive_self_moe", config)
    first = torch.randint(1, config.vocab_size, (2, 17))
    second = torch.randint(1, config.vocab_size, (2, 19))

    out_first = model(first, return_state=True)
    packet = out_first["state_packet"]

    assert isinstance(packet, NAIMEStatePacket)
    assert packet.world_state is not None
    assert packet.self_state is not None
    assert packet.memory is not None
    assert packet.world_state.ndim == 3
    assert packet.self_state.ndim == 3
    assert out_first["world_state"].shape == (2, config.world_state_slots, config.d_model)
    assert out_first["self_state"].shape == (2, config.self_state_slots, config.d_model)

    out_second = model(second, past_state=packet, return_state=True)
    next_packet = out_second["state_packet"]

    assert isinstance(next_packet, NAIMEStatePacket)
    assert out_second["logits"].shape == (2, 19, config.vocab_size)
    assert next_packet.world_state is not None
    assert next_packet.self_state is not None
    assert next_packet.memory is not None

    loss = out_second["logits"].float().mean()
    loss.backward()


def test_state_packet_rejects_batch_mismatch():
    packet = NAIMEStatePacket(world_state=torch.randn(3, 4, 8))

    with pytest.raises(ValueError, match="world_state batch mismatch"):
        packet.validate_batch(2)


def test_v6_latent_thought_state_only_metrics_and_backward():
    torch.manual_seed(2032)
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
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_gate_mixer=True,
        world_state_slots=4,
        self_state_slots=3,
        self_state_recursion_depth=1,
        latent_thought_steps=2,
        latent_thought_write_mode="state_only",
    )
    model = build_model("naime_v6_recursive_self_moe", config)
    input_ids = torch.randint(1, config.vocab_size, (2, 31))

    out = model(input_ids, return_state=True)
    v6_aux = out["aux"][-1]["v6"]

    assert v6_aux["latent_thought_steps"].item() == pytest.approx(2.0)
    assert torch.isfinite(v6_aux["latent_thought_delta"])
    assert torch.isfinite(v6_aux["latent_thought_velocity"])
    assert v6_aux["latent_thought_write_norm"].item() == pytest.approx(0.0)

    loss = out["logits"].float().mean() + v6_aux["latent_thought_delta"] * 0.0
    loss.backward()


def test_v6_latent_thought_final_hidden_is_causal_and_observable():
    torch.manual_seed(2037)
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
        self_state_recursion_depth=1,
        latent_thought_steps=2,
        latent_thought_write_mode="final_hidden",
        latent_thought_hidden_scale=0.01,
    )
    model = build_model("naime_v6_recursive_self_moe", config).eval()
    input_ids = torch.randint(1, config.vocab_size, (2, 31))
    changed = input_ids.clone()
    cutoff = 12
    changed[:, cutoff:] = torch.randint(1, config.vocab_size, changed[:, cutoff:].shape)

    with torch.no_grad():
        original = model(input_ids)
        changed_out = model(changed)

    v6_aux = original["aux"][-1]["v6"]
    assert v6_aux["latent_thought_steps"].item() == pytest.approx(2.0)
    assert v6_aux["latent_thought_write_norm"].item() > 0.0
    assert torch.allclose(original["logits"][:, :cutoff, :], changed_out["logits"][:, :cutoff, :], atol=1e-5, rtol=1e-5)

    train_out = model(input_ids)
    loss = train_out["logits"].float().mean()
    loss.backward()


def test_v6_state_evolution_updates_persistent_packet_without_hidden_write():
    torch.manual_seed(2034)
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
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_gate_mixer=True,
        world_state_slots=4,
        self_state_slots=3,
        self_state_recursion_depth=1,
        state_evolution_steps=2,
    )
    model = build_model("naime_v6_recursive_self_moe", config)
    input_ids = torch.randint(1, config.vocab_size, (2, 31))

    out = model(input_ids, return_state=True)
    packet = out["state_packet"]
    v6_aux = out["aux"][-1]["v6"]

    assert isinstance(packet, NAIMEStatePacket)
    assert packet.world_state is not None
    assert packet.self_state is not None
    assert packet.world_state.ndim == 3
    assert packet.self_state.ndim == 3
    assert packet.world_state.size(1) == config.world_state_slots
    assert packet.self_state.size(1) == config.self_state_slots
    assert v6_aux["state_evolution_steps"].item() == pytest.approx(2.0)
    assert torch.isfinite(v6_aux["state_evolution_delta"])
    assert torch.isfinite(v6_aux["state_evolution_world_delta"])
    assert torch.isfinite(v6_aux["state_evolution_self_delta"])
    assert torch.isfinite(v6_aux["state_evolution_memory_delta"])

    loss = out["logits"].float().mean() + v6_aux["state_evolution_delta"] * 0.0
    loss.backward()


def test_v6_latent_field_coupling_is_bounded_and_trainable():
    torch.manual_seed(2035)
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
        semantic_gate_downstream="clean_prob",
        semantic_sparse_alpha="downstream",
        semantic_memory_slots=2,
        semantic_gate_mixer=True,
        world_state_slots=4,
        self_state_slots=3,
        self_state_recursion_depth=1,
        latent_field_coupling=True,
        latent_field_token_scale=0.02,
        latent_field_max_ratio=0.05,
    )
    model = build_model("naime_v6_recursive_self_moe", config)
    first = torch.randint(1, config.vocab_size, (2, 17))
    second = torch.randint(1, config.vocab_size, (2, 19))

    first_out = model(first, return_state=True)
    second_out = model(second, past_state=first_out["state_packet"], return_state=True)
    v6_aux = second_out["aux"][-1]["v6"]

    assert second_out["logits"].shape == (2, 19, config.vocab_size)
    assert torch.isfinite(v6_aux["latent_field_token_delta_norm"])
    assert torch.isfinite(v6_aux["latent_field_token_delta_ratio"])
    assert torch.isfinite(v6_aux["latent_field_read_entropy"])
    assert v6_aux["latent_field_token_delta_ratio"].item() <= config.latent_field_max_ratio + 1e-4
    assert v6_aux["latent_field_gate"].item() >= 0.0

    loss = second_out["logits"].float().mean() + v6_aux["latent_field_token_delta_norm"] * 0.0
    loss.backward()


def test_v6_latent_field_coupling_preserves_causal_prefix():
    torch.manual_seed(2036)
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
        self_state_recursion_depth=1,
        latent_field_coupling=True,
        latent_field_token_scale=0.02,
        latent_field_max_ratio=0.05,
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


def test_evaluate_model_reports_state_carry_gain_metric():
    torch.manual_seed(2033)
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
        semantic_memory_slots=2,
        world_state_slots=3,
        self_state_slots=3,
    )
    model = build_model("naime_v6_recursive_self_moe", config)
    samples = []
    for _ in range(2):
        input_ids = torch.randint(1, config.vocab_size, (config.max_seq_len,))
        samples.append(
            {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "attention_mask": torch.ones(config.max_seq_len, dtype=torch.bool),
            }
        )
    loader = torch.utils.data.DataLoader(samples, batch_size=2)

    metrics = evaluate_model(
        model,
        loader,
        config,
        torch.device("cpu"),
        use_amp=False,
        max_batches=1,
        state_carry=True,
    )

    assert metrics["val_state_carry_batches"] == 1.0
    assert math.isfinite(metrics["val_state_carry_gain_lm"])
    assert math.isfinite(metrics["val_state_carry_stateful_lm"])
    assert math.isfinite(metrics["val_state_carry_fresh_lm"])


def test_evaluate_model_reports_latent_thought_gain_metric():
    torch.manual_seed(2038)
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
        semantic_memory_slots=2,
        world_state_slots=3,
        self_state_slots=3,
        latent_thought_steps=1,
        latent_thought_write_mode="final_hidden",
        latent_thought_hidden_scale=0.01,
    )
    model = build_model("naime_v6_recursive_self_moe", config)
    samples = []
    for _ in range(2):
        input_ids = torch.randint(1, config.vocab_size, (config.max_seq_len,))
        samples.append(
            {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "attention_mask": torch.ones(config.max_seq_len, dtype=torch.bool),
            }
        )
    loader = torch.utils.data.DataLoader(samples, batch_size=2)

    metrics = evaluate_model(
        model,
        loader,
        config,
        torch.device("cpu"),
        use_amp=False,
        max_batches=1,
        latent_thought_gain=True,
    )

    assert metrics["val_latent_thought_gain_batches"] == 1.0
    assert math.isfinite(metrics["val_latent_thought_gain_lm"])
    assert math.isfinite(metrics["val_latent_thought_lm"])
    assert math.isfinite(metrics["val_latent_thought_disabled_lm"])


def test_evaluate_model_reports_v7_probe_metrics():
    torch.manual_seed(2041)
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
        semantic_memory_slots=2,
        world_state_slots=3,
        self_state_slots=3,
        v7_dynamics_steps=1,
        v7_latent_slots=3,
    )
    model = build_model("naime_v7_typed_dynamics", config)
    samples = []
    for _ in range(2):
        input_ids = torch.randint(1, config.vocab_size, (config.max_seq_len,))
        samples.append(
            {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "attention_mask": torch.ones(config.max_seq_len, dtype=torch.bool),
            }
        )
    loader = torch.utils.data.DataLoader(samples, batch_size=2)

    metrics = evaluate_model(
        model,
        loader,
        config,
        torch.device("cpu"),
        use_amp=False,
        max_batches=1,
        v7_dynamics_gain=True,
        v7_state_swap=True,
        v7_state_erase=True,
    )

    assert metrics["val_v7_dynamics_gain_batches"] == 1.0
    assert metrics["val_v7_state_swap_batches"] == 1.0
    assert metrics["val_v7_state_erase_batches"] == 1.0
    assert math.isfinite(metrics["val_v7_dynamics_gain_lm"])
    assert math.isfinite(metrics["val_v7_state_swap_delta_lm"])
    assert math.isfinite(metrics["val_v7_latent_erase_delta_lm"])


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


def test_topk_moe_auto_dispatch_matches_dense_dispatch_for_small_expert_cpu_heuristic():
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
def test_topk_moe_auto_dispatch_uses_sparse_for_six_experts_on_cuda():
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

    assert moe._resolve_dispatch_mode(x) == "sparse"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA dispatch heuristic only applies on GPU")
def test_topk_moe_auto_dispatch_uses_sparse_for_four_experts_on_cuda():
    moe = TopKMoE(
        d_model=16,
        semantic_dim=16,
        n_experts=4,
        top_k=2,
        expert_hidden_dim=32,
        use_semantic_router=False,
        dispatch_mode="auto",
    ).cuda()
    x = torch.randn(2, 512, 16, device="cuda")

    assert moe._resolve_dispatch_mode(x) == "sparse"


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
