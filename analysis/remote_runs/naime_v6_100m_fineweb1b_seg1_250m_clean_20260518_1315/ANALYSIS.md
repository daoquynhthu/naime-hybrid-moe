# V6 Remote 100M FineWeb-Edu Segment Analysis

Run: `naime_v6_100m_fineweb1b_seg1_250m_clean_20260518_1315`

Status: complete.

Local log copy:

- `train.log`
- `metrics.jsonl`
- `metrics.csv`
- `config.json`

## Run Configuration

- Architecture: `naime_v6_recursive_self_moe`
- Data: `fineweb_edu_1b_ctx1024`
- Resume: `none`
- Causal integrity: `2`
- Model scale: 12 layers, `d_model=768`, 6 experts, top-2 MoE, ctx1024
- Effective steps: `40691`
- Tokens per step: `6144`
- Effective token budget: about `250.0M`
- LR: `1e-5`, cosine to min ratio `0.05`
- Final LR: about `5e-7`
- Batch: 6
- AMP: enabled
- Async checkpoint: enabled

## Validation Curve

| Step | val_lm | val_ppl | alpha | router_ent | b_self | b_world | v6_cos | v6_ctx |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 6.5915 | 728.87 | 0.431 | 1.423 | 0.323 | 0.145 | 0.350 | 0.446 |
| 5000 | 6.1230 | 456.25 | 0.431 | 1.260 | 0.299 | 0.155 | 0.290 | 0.372 |
| 7500 | 5.9098 | 368.65 | 0.438 | 1.167 | 0.287 | 0.160 | 0.265 | 0.365 |
| 10000 | 5.7398 | 310.99 | 0.428 | 1.117 | 0.281 | 0.156 | 0.246 | 0.366 |
| 12500 | 5.5967 | 269.53 | 0.434 | 1.085 | 0.266 | 0.168 | 0.225 | 0.353 |
| 15000 | 5.4950 | 243.48 | 0.434 | 1.052 | 0.267 | 0.158 | 0.221 | 0.372 |
| 17500 | 5.4387 | 230.14 | 0.444 | 1.033 | 0.257 | 0.164 | 0.219 | 0.369 |
| 20000 | 5.3612 | 212.98 | 0.445 | 1.022 | 0.254 | 0.162 | 0.200 | 0.350 |
| 22500 | 5.2793 | 196.24 | 0.445 | 1.015 | 0.241 | 0.167 | 0.215 | 0.384 |
| 25000 | 5.2710 | 194.62 | 0.449 | 1.006 | 0.238 | 0.165 | 0.209 | 0.378 |
| 27500 | 5.2079 | 182.71 | 0.452 | 1.002 | 0.238 | 0.161 | 0.208 | 0.378 |
| 30000 | 5.2327 | 187.30 | 0.453 | 0.995 | 0.236 | 0.164 | 0.211 | 0.381 |
| 32500 | 5.1437 | 171.34 | 0.453 | 0.990 | 0.235 | 0.167 | 0.207 | 0.374 |
| 35000 | 5.1263 | 168.40 | 0.456 | 0.989 | 0.231 | 0.165 | 0.209 | 0.377 |
| 37500 | 5.1629 | 174.67 | 0.456 | 0.990 | 0.229 | 0.165 | 0.214 | 0.386 |
| 40000 | 5.0957 | 163.32 | 0.456 | 0.993 | 0.228 | 0.166 | 0.209 | 0.377 |
| 40691 | 5.1321 | 169.37 | 0.456 | 0.988 | 0.229 | 0.166 | 0.210 | 0.376 |

Best validation:

- Step: `40000`
- `val_lm_loss`: `5.0957`
- `val_ppl`: `163.32`

Final validation:

- Step: `40691`
- `val_lm_loss`: `5.1321`
- `val_ppl`: `169.37`
- Final is slightly worse than best, so `model_best.pt` is the preferred checkpoint.

## Tail Metrics

Tail 200 training records:

- `loss_lm`: `5.1483`
- `ppl_lm`: `174.37`
- `grad_norm`: `2.6697`
- `grad_norm max`: `4.4952`
- `alpha_downstream`: `0.4625`
- `router_entropy`: `0.9958`
- `bad_grad_window_count`: `0.0`
- `load_loss`: `1.0330`
- `sparse_loss`: `0.000873`
- `kl_loss`: `0.053542`
- `semantic_pred`: `0.13177`
- `v5_state_pred`: `0.005673`
- `v6_self_pred`: `0.038566`

Tail 1000 and tail 5000 are very close to tail 200, indicating a stable late
training regime rather than an unstable collapse.

## Throughput

Parsed from console log:

- Average: `12198 tok/s`
- P10: `9143 tok/s`
- Median: `12889 tok/s`
- P90: `13215 tok/s`
- Min: `2724 tok/s`
- Max: `13352 tok/s`

The low minimum is attributable to checkpoint/eval or startup overhead. Stable
training throughput is around `12.8k-13.2k tok/s`.

## Objective Components

The final training objective remains dominated by LM loss.

Late-stage auxiliary contribution is small:

- load contribution: about `0.0103`
- sparse contribution: around `1e-5`
- KL contribution: around `0.00016`
- semantic prediction contribution: about `0.0020`
- V5 state prediction contribution: about `0.00011`
- V6 self prediction contribution: about `0.00039`

This means `loss_total` and `loss_lm` are close enough for monitoring, while
`val_lm_loss` remains the primary model-quality metric.

## Router And Alpha

Targets:

- Alpha should stay near target sparsity `0.45`.
- Router entropy should not collapse.
- Sparse regularization should not dominate loss.

Observed:

- Late alpha: `0.4625`
- Final validation alpha: `0.4560`
- Late router entropy: `0.9958`
- Final validation router entropy: `0.9882`
- Sparse loss is tiny.

Assessment: pass. Alpha is slightly above target but healthy. Router is active
without collapse.

## V5 World-State Metrics

Targets:

- World state should be active but not unstable.
- Slot writes should remain top-k sparse and non-collapsed.
- Slot confidence should be stable.

Observed late-stage values:

- `v5_slot_confidence`: `0.5487`
- `v5_slot_delta`: `0.0499`
- `v5_slot_cosine`: about `-0.006`
- `v5_slot_read_entropy`: `1.3469`
- `v5_slot_write_entropy`: `0.5999`
- `v5_slot_write_max/min`: `0.669 / 0.331`
- `v5_slot_write_active`: `2.0`

Assessment: pass for stability. World slots are active and top-2 writing works.
However, this run used conservative `semantic_state_write_scale=0.045`, so it
should be treated as a stable baseline, not proof that world-state influence is
maximized.

## V6 Self-State Metrics

Targets:

- Self-state should not dominate world-state.
- Reflection should be active but bounded.
- Slot/context cosine should show structured participation rather than collapse.

Observed late-stage values:

- `v6_boundary_self`: `0.2296`
- `v6_boundary_world`: `0.1646`
- `v6_boundary_other`: `0.3304`
- `v6_boundary_unknown`: `0.2754`
- `v6_slot_cosine`: `0.2091`
- `v6_slot_context_cosine`: `0.3799`
- `v6_reflection_norm`: `6.3549`
- `v6_state_delta`: `0.000025`
- `v6_self_pred`: `0.0386`

Assessment: pass. Self-state is active and measurable, but not runaway. The
boundary distribution is much healthier than earlier self-dominant variants.

## Stability And Engineering

Targets:

- Complete training without crash.
- Preserve checkpoint integrity.
- Produce persisted full logs and CSV.
- Avoid bad-gradient spikes.

Observed:

- `training complete`: yes
- `metrics csv saved`: yes
- `save ckpt`: 17 log entries
- `save best`: 14 log entries
- `bad grad`: 0
- `non-finite`: 0
- `model_best.pt`: present
- `model_latest.pt`: present
- `latest.pt`: present

Assessment: pass.

## Target Alignment

| Target | Result | Status |
|---|---|---|
| Clean from-scratch run | `resume=none`, causal integrity `2` | pass |
| 250M-token segment | about `250M` tokens | pass |
| Stable training | complete, no bad gradients | pass |
| Meaningful validation improvement | `val_ppl 728.9 -> 163.3` best | pass |
| Alpha near target | `0.456` validation | pass |
| Router not collapsed | entropy about `0.99` late | pass |
| V6 self not dominant | self boundary about `0.23` | pass |
| World participation | stable but conservative | partial |
| Best checkpoint retained | `model_best.pt` | pass |

## Conclusion

This run validates the conservative V6 protocol path at the 100M-parameter,
250M-token segment scale. The best checkpoint is at step `40000`, not the final
step. Training did not collapse, did not show bad gradients, and did not show
self-state dominance.

The remaining architectural question is not stability. It is whether world-state
participation is strong enough. The next comparable run should use the newer
`semantic_state_write_scale=0.075` template to test whether world coupling
improves without destabilizing alpha, router entropy, or V6 boundary metrics.
