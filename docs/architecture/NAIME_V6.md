# NAIME V6: Recursive Self-State MoE

V6 extends the V5 world-state model with a recursive self-state subsystem. The goal is not just lower language-model loss; the architecture is also meant to test whether a compact model can maintain measurable internal state boundaries while learning from large-scale text.

## Core Mechanism

- `naime_v6_recursive_self_moe` keeps V5 world-state slots and adds recursive self-state slots.
- The self-state module predicts and updates internal state from hidden activations, previous self slots, and world context.
- Boundary metrics split state signal into `self`, `world`, `other`, and `unknown` components.
- V6 auxiliary objectives include self prediction and self-slot diversity.
- Training logs expose LM metrics and structure metrics such as `v6_self_pred`, `v6_slot_cosine`, `v6_slot_context_cosine`, `v6_boundary_self`, `v6_boundary_world`, and `v6_reflection_norm`.

## Validation Snapshot

Latest completed validated remote run:

```text
naime_v6_100m_1b_conservative_logfix_add40m_20260517_1645
```

Configuration summary:

```text
architecture = naime_v6_recursive_self_moe
dataset      = fineweb_edu_1b_ctx1024
ctx          = 1024
d_model      = 768
layers       = 12
experts      = 6
top_k        = 2
world slots  = 6
self slots   = 6
```

Validation trajectory:

| Step | val_lm | val_ppl | alpha | router_ent | Notes |
|------|--------|---------|-------|------------|-------|
| 10000 | 2.0106 | 7.47 | ~0.72 | ~0.67 | First credible V6 validation from the formal 1B run. |
| 27500 | 0.9483 | 2.581 | 0.657 | 1.117 | Stable continuation checkpoint. |
| 30000 | 0.8281 | 2.289 | 0.648 | 1.204 | Continued improvement without eval collapse. |
| 32500 | 0.7183 | 2.051 | 0.644 | 1.266 | Best pre-final eval. |
| 32958 | 0.7033 | 2.020 | 0.645 | 1.260 | Best validated checkpoint so far. |

Tail-100 training metrics in the latest completed run:

```text
loss_lm mean       ~= 0.7955
ppl mean           ~= 2.225
grad_norm mean     ~= 3.30
alpha mean         ~= 0.654
router entropy     ~= 1.25
```

## Structure Metrics

The early formal run showed self-state dominance:

```text
v6_boundary_self  ~= 0.79
v6_boundary_world ~= 0.06
```

The later continuation is healthier:

```text
v6_self_pred              ~= 0.0048
v6_slot_context_cosine    ~= 0.56
v6_boundary_self          ~= 0.71
v6_boundary_world         ~= 0.07
v6_boundary_other         ~= 0.12
v6_boundary_unknown       ~= 0.10
```

Interpretation:

- The recursive self-state path is active and no longer just noise.
- Self remains the strongest boundary component, but it is less extreme than in the first V6 run.
- World-state utilization is still weaker than desired. Future architecture work should strengthen world coupling rather than merely amplifying self-recursion.
- Router entropy around `1.2-1.3` is currently healthier than the early collapse toward very low entropy.

## Known Risks

- Bad-gradient spikes remain the main reliability problem. The latest completed segment had 153 skipped bad-gradient steps, but no checkpoint reloads.
- Long schedules can enter unstable late phases. Prefer segmented continuation with explicit LR strategy.
- Model-only checkpoint resumes are stable for continuation, but LR and warmup must be set deliberately.
- Checkpoint I/O can hurt throughput and, on shared/unstable Windows environments, increase native crash risk.
- Visible remote windows or SSH-tied foreground launches can be interrupted by other desktop activity. Use hidden/background launch helpers.

## Current Training Policy

The project target remains feeding approximately 1B tokens. Recommended continuation policy:

```text
resume          previous validated model_best.pt
target mode     additional
segment size    100M tokens when GPU is available
vram fraction   0.80
learning rate   2.5e-5
warmup steps    500
min lr ratio    0.03
grad clip       0.8
eval every      5000
eval batches    40
save every      10000
latest every    5000
best mode       model
```

When the server is shared, first sample other GPU usage over a short window. If another training job occupies most VRAM, use conservative fixed batch and gradient accumulation. If the GPU is free, use auto-batch with predictive skip/headroom.

Segmented continuation must preserve data-stream continuity. Confirm the sampler log:

```text
train sampler resumed stream seed=1234 resume_step=<step> offset_batches=<offset>/<epoch_batches>
```

## Current Verdict

V6 is validated as the current best architecture direction. It has surpassed the earlier V5 validation metrics on the current 1B-corpus training path, and its internal metrics show genuine structure rather than a purely decorative module.

It is not yet a finished large-scale model. The next work should focus on:

- completing the 1B-token curriculum without replaying batches;
- reducing bad-gradient frequency with adaptive LR and safer late-run scheduling;
- strengthening world-state utilization so recursive self-state does not monopolize structure;
- evaluating generation quality from the best completed and later continuation checkpoints.
