# NAIME V5 Architecture Plan

Status: **implemented and validated.**

V5 turned the single recurrent semantic state into a structured, trainable, and measurable world-state system. A 200M-token pre-training run confirmed all submodules work without degradation.

## 1. Validation Results (200M tokens, 67M params, ctx512)

| Metric | 50M tokens | 200M tokens | Trend |
|--------|-----------|------------|-------|
| val_lm | 1.963 | 1.263 | ↓ monotonic |
| ppl | 7.1 | 3.5 | ↓ |
| slot_cosine | 0.52 | 0.47 | ↓ slots diverging |
| state_pred_loss | 0.008 | 0.005 | ↓ prediction improving |
| state_norm | 18.85 | 18.03 | ↓ compressing |
| router_entropy | 1.17 | 1.08 | ↓ leaving uniform |
| slot_confidence | 0.92 | 0.91 | → stable |
| gate mix (α/c/s) | 0.14/0.52/0.34 | 0.14/0.52/0.34 | → stable |
| dispatch_dense | — | 0.0 | auto→sparse at ctx512 |

### Architecture Verdict

| Component | Result |
|-----------|--------|
| Block-level VAE compressor | Alpha tracks target, no KL collapse. ✓ |
| World state slots | Learned differentiable representations. ✓ |
| Cross-layer memory read/write gates | No degeneration. ✓ |
| Semantic gate mixer | Converged early; weights stable. ✓ |
| Semantic prior → router influence | Present but mild (14% weight). ⚠ |

The main architectural gap: **semantic prior contributes 14% vs clean_prob's 52% to routing.** V5 implemented the mechanism, but the semantic signal-to-noise ratio in routing needs improvement. Next round targets: larger z_dim, higher prior_scale, lower lambda_kl, cosine warm restarts.

## 2. Design — Implemented Features

### 2.1 Structured State Slots

4 world-state slots of dimension `d_model`. The model learns slot roles through diversity pressure:

```text
world_state_slots = 4
slot_dim = d_model
```

- Attribution-pooled semantic summary → `semantic_summary`
- Slot router: multi-head attention over query/slot pairs → `slot_write_weights`
- Slot update: `candidate = layer(semantic_summary)`, gated linear interpolation
- Slot confidence: sigmoid(self.confidence(normalized_slots))
- Diversity loss: pairwise cosine similarity (target: orthogonal)
- Stability loss: confidence-gated slot delta (activated when diversity below threshold)
- Transition predictor: next-layer semantic summary from current slots → state_pred_loss

### 2.2 Semantic Gate Mixer

Three-way weighted combination at each MoE block:

```text
α_input = alpha(stochastic gate output)
clean_input = gate_clean_prob (downstream deterministic)
state_input = memory output (cross-layer semantic state)

mix_weights = softmax(gate_mixer.proj([α, clean, state]) / temperature)
gate_signal = weighted_sum(α, clean, state) × mix_weights
```

The `max_clean_weight` cap prevents clean_prob from dominating, leaving room for semantic state.

### 2.3 Cross-Layer Semantic Memory

State passing between layers:
- layer-wise gate schedule controls per-block read intensity.
- memory is a small vector bank (memory_slots × d_model).
- read gate + attention over memory slots → `memory_output`.
- confidence-gated write from current semantic state.

### 2.4 Routing Architecture

```text
router_input = concat(hidden_states, token_semantic_downstream)
logits = proj(router_input) + prior_gate * prior_scale * semantic_prior(semantic)
```

- `semantic_router_mode=hybrid`: both concat and prior paths active.
- `semantic_gate_downstream=clean_prob`: downstream deterministic path uses clean probability, not stochastic alpha.
- `dispatch_mode=auto`: resolves to dense or sparse based on model scale, expert count, and sequence length.

### 2.5 Compressor Configuration

```text
scales: local_mid_global (stride 16, mid_stride 32)
z_dim: 96 → 128 (planned upgrade)
fusion: concat over 3 scales
pred_horizon: 1 step ahead
target_sparsity: 0.55 (→ 0.45 planned)
```

## 3. Next Phase (V5.1)

Target: 100M params, 1B tokens, ctx1024.

LR: cosine warm restarts (cycle_length ≈ 250M tokens worth of steps, restart_ratio=0.5).

Semantic routing enhancements:

| Parameter | V5 | V5.1 |
|-----------|-----|------|
| prior_scale | 0.5 | 1.5 |
| lambda_kl | 0.005 | 0.003 |
| target_sparsity | 0.55 | 0.45 |
| gate_mixer_max_clean | 0.58 | 0.45 |
| gate_mixer_temperature | 1.6 | 2.5 |
| lambda_state_pred | 0.01 | 0.02 |
| z_dim | 96 | 128 |
