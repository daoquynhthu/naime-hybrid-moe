# NAIME V7 Status and Key Limitations

Status: **engineering implementation active, mechanism partially validated, not final**

Last updated: 2026-05-26

This document summarizes the current V7 architecture state after the 100M-class
FineWeb-Edu 200M-token validation runs:

- `v7_100m_fineweb_200m_v1`
- `v7_100m_fineweb_200m_v2_statecarry`

It is intentionally critical. V7 is not considered complete merely because
validation loss improved. The central question is whether typed internal
dynamics, persistent state, and state-conditioned prediction are becoming real
mechanisms rather than decorative residual capacity.

## 1. Current Architecture Snapshot

V7 extends the V6.5 stateful MoE line with a typed internal dynamics layer.
The current state families are:

```text
world_state       text-world / discourse / entity-like latent structure
self_state        internal processing boundary and reflection state
latent_field      compact evolving internal latent condition
controller_state  small fixed controller slot bank
memory            inherited semantic memory path
```

The important causal rule is:

```text
incoming state may affect the current segment
current segment may update outgoing state
outgoing state must not rewrite the same segment as if it were past context
```

This is the key philosophical distinction from ordinary hidden residual
modules. V7 state is meant to be the endpoint of previous internal computation
and the starting condition of future computation, not a same-forward shortcut.

## 2. Implemented V7 Mechanisms

The current implementation includes:

- typed latent dynamics with separate latent/world/self/controller updates;
- causal chunked state flow through `v7_state_chunk_size`;
- incoming latent read with bounded hidden write;
- state packet input/output for continuation;
- V7 mechanism probes during validation;
- state swap, state erase, and dynamics-disabled diagnostics;
- optional homeostatic rate modulation;
- fused state attention path for small state-slot operations;
- template-based remote launch path.

The latest optional controller is the homeostatic rate modulator:

```text
v7_homeostatic_control
v7_homeostatic_dhi
v7_latent_rate_scale
v7_world_rate_scale
v7_self_rate_scale
v7_hidden_read_rate_scale
```

It is not a learned deliberation controller. It is a bounded, detached
stability controller based on relative typed-state motion and acceleration.

## 3. What Has Been Validated

The strongest positive evidence so far is that V7 dynamics gain increases
during training.

For `v7_100m_fineweb_200m_v1`:

```text
val_lm_loss:              5.7760 -> 4.4072
val_ppl:                  322.5  -> 82.0
val_v7_dynamics_gain_lm:  0.0009 -> 0.0061
val_v7_state_swap_delta:  0.0007 -> 0.0016
bad_grad_window_count:    0
```

Interpretation:

- V7 dynamics are not inert.
- Disabling V7 dynamics increasingly hurts validation loss.
- Swapping the state packet with another sample's state hurts loss, so the
  state carries some sample-specific information.
- Training is numerically stable at this scale.

The V6/V5 inherited state behavior also looks healthier than earlier V6 runs:

```text
v6_boundary_self   decreased toward ~0.22
v6_boundary_world  increased toward ~0.28
v5_router_world_ratio stayed around ~0.05
```

This means self-state is not obviously swallowing the world-state path, and
world influence remains bounded.

## 4. V2 State-Carry Micro-Adjustment Result

The follow-up run `v7_100m_fineweb_200m_v2_statecarry` changed:

```text
V7PastLatentAdaptSteps: 1 -> 2
V7HomeostaticControl: enabled
V7HomeostaticMinScale: 0.70
V7HomeostaticMaxScale: 1.20
```

Final comparison:

```text
v1 final val_lm: 4.4072
v2 final val_lm: 4.4448

v1 final val_ppl: 82.0
v2 final val_ppl: 85.2
```

The final loss is slightly worse in V2, but the comparison is not perfectly
step-equivalent:

```text
v1: batch=12, max_steps=16277
v2: batch=13, max_steps=15025
```

Both consume about 200M tokens, but V2 has fewer optimizer updates.

The meaningful mechanism change is:

```text
state_carry_gain_lm:
v1 final ~= -0.00137
v2 final ~= -0.00067

latent_erase_delta_lm:
v1 final ~= -0.00103
v2 final ~= -0.00025
```

Interpretation:

- The micro-adjustment reduced the negative effect of carried latent state.
- It did not turn carried state into a positive contributor.
- Homeostatic control was stable, but it did not solve state usefulness.

## 5. Current Key Weakness: Carry State Is Not Yet Useful Enough

The biggest unresolved issue is:

```text
state_carry_gain_lm remains negative or near zero
```

This means that passing the state packet from the first half of a sequence into
the second half does not yet reliably improve next-token prediction.

This is a serious mechanism gap. The core V7 philosophy is that internal state
should persist as a meaningful continuation condition. If carried state does
not help future prediction, then V7 is still closer to:

```text
current-segment internal dynamics
```

than to:

```text
persistent internal continuity
```

The current evidence says V7 dynamics help within the evaluated forward path,
but persistent carry remains weak.

## 6. Why Carry May Be Weak

Current hypotheses:

### 6.1 Stale State Mismatch

A state packet produced from one segment may not align cleanly with the next
segment. If it writes into hidden too early, it can behave like stale context.
Increasing `V7PastLatentAdaptSteps` reduced the damage but did not create a
positive gain.

### 6.2 State Is Useful But Too Weak

State swap penalty is positive. That means the state contains some useful
sample-specific information. But the magnitude is small:

```text
val_v7_state_swap_delta_lm ~= 0.001
```

So the model notices wrong state, but not strongly.

### 6.3 Latent Field Is Still More Like Local Computation Than Durable Memory

`val_v7_dynamics_gain_lm` grows clearly, but `state_carry_gain_lm` does not.
That suggests the latent field currently helps as an internal computation path
more than as a durable cross-segment condition.

### 6.4 Erase Metrics Are Ambiguous

Several erase deltas are near zero or negative. This implies that removing some
carried state components does not reliably hurt validation loss.

The important point:

```text
state motion is not the same as state usefulness
```

World/self/latent slots can move, while still failing to become necessary for
prediction.

## 7. Secondary Weakness: World-State Contribution Is Bounded but Small

The world path is stable and not exploding:

```text
v5_router_world_ratio ~= 0.05
```

This is safe, but also conservative. It means world-state influence on routing
is still a small fraction of the semantic routing field.

This may be correct at current scale, but it limits how much world-state can
shape expert selection. If the long-term goal is a model with an internal world
that actively structures prediction, the current world influence is probably
too weak.

The risk is not instability. The risk is underuse.

## 8. Secondary Weakness: Hidden Write Works, But It Is Still a Shortcut Risk

V7 hidden write ratio is bounded and stable:

```text
v7_hidden_write_ratio ~= 0.02
```

This is safe. It is not currently exploding or destabilizing training.

However, hidden write remains the easiest way for V7 to improve loss without
proving durable state continuity. If validation improves mostly because latent
read writes a useful residual into hidden, then V7 may still be a strong
state-conditioned residual system rather than a full internal continuity
system.

Therefore hidden-write success should not be overinterpreted.

## 9. Homeostatic Controller Status

The homeostatic controller ran stably in V2:

```text
v7_homeostatic_control_enabled = 1
v7_homeostatic_dhi ~= 0.70-0.75
rate scales stayed bounded
bad_grad_window_count = 0
```

It appears safe at the tested scale.

What it did:

- reduced some negative carried latent effects;
- bounded typed state motion;
- did not introduce gradient spikes.

What it did not do:

- did not make state carry positive;
- did not improve final validation loss versus V1;
- did not prove autonomous deliberation.

So homeostatic control should remain available, but it should not be treated as
the final solution.

## 10. What V7 Currently Is

The honest current description:

```text
V7 is a stable typed internal dynamics architecture whose current-segment
dynamics are useful and measurable.
```

It is not yet:

```text
a fully validated persistent inner continuity architecture
```

The architecture has crossed an important line beyond ordinary Transformer +
adapter stacking, because prediction is now measurably affected by typed
internal dynamics. But the harder goal, durable carried-state usefulness, is
not solved.

## 11. What Should Not Be Done Next

Do not simply:

- increase `V7PastLatentAdaptSteps` again;
- increase hidden write scale;
- add another parallel state bank;
- rely on lower validation loss as proof;
- move to huge-scale training before carry usefulness improves;
- let `world/self/latent` collapse into one mixed state vector.

Those would either mask the problem or make attribution worse.

## 12. Recommended Next Engineering Direction

The next architectural step should target controlled state carry.

Recommended direction:

```text
gated state carry / residual state blending
```

Instead of only delaying carried-state read, the model should learn or compute a
bounded compatibility gate:

```text
incoming_state
current_input_summary
state_confidence
homeostatic pressure
        -> carry_gate
        -> blended readable state
```

The goal is:

```text
read carried state when it is compatible
suppress it when stale
preserve it when useful
avoid immediate hidden pollution
```

This should be implemented as a named, bounded, observable path, not as an
opaque residual shortcut.

Required metrics:

```text
v7_carry_gate_mean
v7_carry_gate_min/max
v7_carry_compatibility
v7_carry_blend_delta
val_state_carry_gain_lm
val_v7_state_swap_delta_lm
val_v7_world/self/latent_erase_delta_lm
```

Success criteria:

```text
val_state_carry_gain_lm > 0 over multiple validation points
state_swap_delta remains positive
erase deltas become non-negative and interpretable
hidden_write_ratio remains bounded
bad_grad_window_count remains 0
validation loss does not regress materially
```

## 13. Bottom Line

V7 is worth continuing.

The core dynamics path is real enough to measure:

```text
v7_dynamics_gain_lm grows during training
state swap penalty is positive
hidden write is bounded
training is stable
```

But the central philosophical goal is not fully achieved yet:

```text
carried internal state is not reliably useful for future prediction
```

The next milestone is not a larger run. It is making persistent state carry
cross from weak/negative to consistently positive while preserving the V7 state
protocol.
