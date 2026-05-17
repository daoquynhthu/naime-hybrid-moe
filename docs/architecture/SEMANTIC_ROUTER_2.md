# Semantic Router 2.0

## Motivation

The first NAIME-State MoE router used direct concatenation:

```text
router_input = concat(token_hidden, semantic_state)
```

This proved useful, but it treats semantic state as just another feature vector. Router 2.0 makes semantic state act as an expert prior.

## Modes

### concat

The original mode.

```text
logits = W[token_hidden, semantic_state]
```

### prior

Token state produces token-local routing logits. Semantic state produces expert bias.

```text
token_logits = W_token token_hidden
semantic_bias = W_sem semantic_state
logits = token_logits + scale * semantic_bias
```

This is the cleanest test of semantic-state-driven expert scheduling.

### hybrid

Uses both concatenation and semantic prior.

```text
concat_logits = W[token_hidden, semantic_state]
semantic_bias = W_sem semantic_state
logits = concat_logits + scale * semantic_bias
```

This is more expressive but less clean for attribution.

## Token MoE Reference

The generic model launcher can still run token MoE if a reference is needed:

```powershell
.\scripts\train_model.ps1 -Model token_moe -RunName router2_token_moe_reference
```

## Direct 1.0 vs 2.0 Experiment

After the base architecture has already beaten dense and token-only MoE, use the direct NAIME comparison:

```powershell
.\scripts\train_model.ps1 -Model naime_v1 -RunName router1_naime_concat_router_compare_v1
.\scripts\train_model.ps1 -Model naime_v2 -RunName router2_naime_prior_router_compare_v1
```

It runs:

- `naime_v1`: concat router
- `naime_v2`: semantic prior router

This is the cleanest test of whether Router 2.0 improves over Router 1.0.
