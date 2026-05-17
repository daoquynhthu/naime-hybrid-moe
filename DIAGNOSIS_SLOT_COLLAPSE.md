# V5 World-State Slot Collapse: Complete Diagnosis

**Date**: 2026-05-14
**Affected Architecture**: `naime_v5_world_state_moe`
**Symptom**: Slot write gates collapse to winner-take-all, min gate = 0, confidence → 0

---

## 1. What is the slot system supposed to do?

V5 introduced a **structured world-state** bank — a set of N learnable latent slot vectors shared
across all semantic layers. At each layer:

```
                                     read path
                                     ─────────
  hidden_states ──→ cross-attn(slots) ──→ context, conf
       │                                     │
       │                              gate_mixer(α, clean, conf)
       │                                     │
       │                              router_semantic (goes into MoE)
       │
       │  write path
       │  ──────────
       └──→ compressor ──→ semantic_summary
                                   │
                          update_slots(summary):  slots ← slots + gate * (candidate − slots)
```

The read path uses **softmax attention** over slot keys (competition *is* legitimate here).
The write path uses a **slot_router** to decide how much each slot absorbs the current summary.
This is the part that collapses.

---

## 2. Collapse timeline (proven across 3 independent runs)

All logs came from local V5 run directories under the configured run root.

### v7 run 2 (non-normalized sigmoid — current code)

```
Step    gate_mean   gate_min   conf    cos     key file
────    ─────────   ────────   ────    ───     ─────────
  10    0.37        0.37       0.49   0.06    ```
  40    0.42        0.42       0.92   0.42
  90    0.55        0.11       0.97   0.14
 120    0.65        0.20       0.52   0.01    ← start of collapse
 200    0.68        0.18       0.72   -0.06
 400    0.77        0.04       0.04   0.31    ← min gate ≈ 0, conf → 0
1500    0.92        0.00       0.08   0.40    ← stuck
5000    0.93        0.00       0.03   0.39    ← permanent collapse
```

The collapse is **not a one-time event** — it happens in two phases:
1. **Phase 1 (step 0-150)**: gate_mean rises from 0.33 to 0.68, gate_min drops from 0.33 to 0.18, cos drops from 0.23 to -0.07, conf rises to 0.97.  Slots begin differentiating.
2. **Phase 2 (step 150-400)**: gate_mean continues to 0.92, gate_min drops to 0, conf drops to 0.

### Failed intervention history

| Version | Write mechanism | Collapse characteristic |
|---------|----------------|----------------------|
| v6 (balancer) | softmax | w_max 0.91, conf 0.00 |
| v7-r1 (T=2.5) | softmax(logits/T) | w_max 0.30 → 0.62 within 600 steps |
| v8-r1 (per-slot pred) | softmax, weighted per-slot loss | w_max 0.71 → 0.97 within 700 steps |
| v8-r2 (sigmoid + norm) | sigmoid / Σgates | w_max 0.93 within 200 steps |
| v8-r3 (raw sigmoid) | sigmoid (no norm) | gate_min → 0, conf → 0 within 400 steps |

**No variation of the write gate mechanism has prevented collapse past step 400-700.**

---

## 3. Root cause

### 3.1 The write gate has no counter-pressure

```python
# world_state.py update_slots()
slot_gate_raw = sigmoid(slot_router(norm(summary)))   # [B, N_slots]
slot_write = slot_gate_raw.unsqueeze(-1)
next_slots = slots + slot_write * update_gate * (candidate - slots)

# Then:
pred_pooled = transition(Σ_i slot_write_i * next_slot_i)
state_pred_loss = smooth_l1(pred_pooled, semantic_summary.detach())
```

**Why gate_mean always rises to ~0.9**:

The update equation `slots + gate * (candidate - slots)` is a mixing gate.
When `gate = 1`, the slot overwrites to `candidate`. When `gate = 0`, the slot stays put.
`candidate` is computed from `[slots, summary, identity]` — it contains the latest summary.
Setting `gate = high` means "update more aggressively", which makes `pred_pooled`
better match `semantic_summary`. There is **no penalty term opposing gate growth**.

The model learns: *"the more aggressively I overwrite, the lower my prediction error."*
This is a ratchet — gate only goes up.

### 3.2 Why gate_min drops to zero

The slot_router is a `Linear(d_model → N_slots)`. Initialized with small random weights.
Once one slot's router weight grows slightly larger, two things happen:

1. That slot's gate is higher → it gets more gradient from state_pred_loss → its router weight grows faster.
2. Other slots' gates drift toward zero because:
   - Their router weights receive *less* gradient (they contribute less to pred_pooled).
   - The only source of gradient for underused slots is state_pred_loss through a *vanishingly small* gate weight.
   - Result: rich get richer, poor get poorer.

**Even sigmoid (non-competitive) doesn't fix this** because the asymmetry comes from
*gradient flow inequality*, not from normalization. Softmax accelerates it, sigmoid slows it,
but neither stops it — because the root is:

> **All slots are trained through a single pooled prediction target.**

### 3.3 Why confidence drops to zero

```python
# read()
confidence = sigmoid(confidence(key)) · sigmoid(max_score)
```

`confidence` is computed over `key = slot_norm(slots)`. When slots become dominated
by one slot (the other 3 are never significantly updated), `confidence(key)` averages
over 3 near-initial vectors → ~0.5 · ~0.5 = ~0.25, decaying toward 0 as the dead
slots drift into random noise.

### 3.4 The self-reinforcing death spiral

```
gate_mean ↑ → underused slots starved of gradient → dead slots
                                                          │
                            confidence ↓ ← ───────────────┘
                                │
                    gate_mixer: state_confidence ↓
                                │
                α 8% / clean 42% / state 50%
                    ↑                            ↑
          compressor untrusted    trusting a dead world state
                    │
              lm stuck at 4.2-5.0
```

Once confidence → 0, the gate_mixer still gives 50% weight to state_confidence
(the mixer itself uses softmax and `min_weight=0.08`), but the state_confidence
signal is now noise → MoE routing is partially corrupted → lm plateaus.

---

## 4. Design problem

### 4.1 Shared `self.transition` causes cross-slot gradient coupling

```python
# All slots share the same Linear(512→512)
pred_pooled = self.transition((slot_write * next_slots).sum(dim=1))
```

The transition function sees a **weighted mixture of 4 slot vectors**. From its
perspective, all that matters is the mixture quality — not slot composition.
If 1 strong slot + 3 zero-weight slots gives the same mixture as 4 quarters,
the loss is identical. But the **gradient is not identical**: with 1 strong slot,
only that slot's router receives meaningful gradient.

### 4.2 The read side is fine — the write side is the problem

Read uses **softmax cross-attention** — this *should* be competitive because
it's query-key matching. A token should attend to the *most relevant* slot(s).

Write uses **single pooling target** — this *should not* be competitive because
all slots are being asked to predict the same thing. Competition serves no purpose.

### 4.3 The state_pred_loss architecture is the root

```
pred = transition(weighted_sum_of_updated_slots)
loss = |pred - summary|²
```

This is a **decomposable target with non-decomposable penalty**. The model can
satisfy it with 1 slot, so it does. To force multi-slot usage, one of these must change:

**Option A: Decomposable penalty (per-slot targets)**
Make each slot predict *something different* so they specialize.
Requires defining what each slot should predict — semantically non-trivial.

**Option B: Diversity-gated target**
`pred = transition(Σ max(gate_i, θ) * slot_i)` — guaranteed minimum injection from each slot.
This prevents zero-gradient starvation but is an architectural hack.

**Option C: Separate per-slot transitions**
Give each slot its own `transition_i` function. Each slot independently predicts `summary`.
Then the gate decides "how much does slot i contribute to the final prediction."
This creates natural specialization: different `transition_i` functions will be
differently good at predicting different aspects of the summary.

**Option D: Predictive coding / contrastive slot objective**
Instead of predicting the summary, use the summary as query and the slots as memory:
train with a contrastive loss where "correct slot" = the one that was most recently written-to.
This makes write distribution matter for a different reason.

---

## 5. Concrete architecture issue

The full gradient path for `slot_router`:

```
∂L/∂router = ∂L/∂pred · ∂pred/∂mixture · ∂mixture/∂gate · ∂gate/∂router
                                    ↑              ↑
                            same scalar     sigmoid'(logit) · router_input
                            for all slots

∂mixture/∂gate_i = transition(next_slot_i)  ← different vector per slot
```

So `∂L/∂router_i = scalar · sigmoid'(logit_i) · transition(next_slot_i) · input`.

- If `gate_i ≈ 0`, then `next_slot_i ≈ old_slot_i` (uninformative, near-initial).
- `transition(initial_slot)` is almost random → no useful gradient direction.
- If `gate_j ≈ 1`, then `next_slot_j` carries information.
- `transition(informed_slot_j)` gives a meaningful gradient → slot_j's router weight grows.

**This creates an absorbing state** at gate = [1, 0, 0, 0]. Once entered, no gradient
signal can pull the zero-gate slots out because their `next_slot` vectors contain
zero information about the current summary.

---

## 6. Summary for expert review

1. The slot write gate uses sigmoid (no normalization, non-competitive) yet collapses within 400 steps.
2. The collapse is inevitable under the current state_pred_loss formulation: one strong slot suffices.
3. The read path (cross-attention) is architecturally sound; the problem is isolated to the write path.
4. Dead slots produce `confidence → 0`, which cascades into `gate_mixer` giving 50% weight to a zero signal.
5. The ladder of failed interventions (softmax → temperature → per-slot-pred → sigmoid → raw sigmoid)
   demonstrates that the collapse is not about the gate mechanism but about the training objective.

**Key code locations**:
- Write gate + prediction loss: [src/naime_hybrid/modules/world_state.py](file:///g:/Program/naime-hybrid-moe/src/naime_hybrid/modules/world_state.py#L53-L75)
- Read path (fine): [src/naime_hybrid/modules/world_state.py](file:///g:/Program/naime-hybrid-moe/src/naime_hybrid/modules/world_state.py#L37-L50)
- Block integration: [src/naime_hybrid/modules/blocks.py](file:///g:/Program/naime-hybrid-moe/src/naime_hybrid/modules/blocks.py#L317-L400)
- Gate mixer (affected by dead slots): [src/naime_hybrid/modules/state.py](file:///g:/Program/naime-hybrid-moe/src/naime_hybrid/modules/state.py#L1-L56)
- Training loss construction: [src/naime_hybrid/training/train.py](file:///g:/Program/naime-hybrid-moe/src/naime_hybrid/training/train.py#L348-L370)
