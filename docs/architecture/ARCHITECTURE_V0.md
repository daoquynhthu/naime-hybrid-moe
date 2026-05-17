# Architecture V0: Selective Semantic Memory MoE

Date: 2026-05-09

## Thesis

The model should not spend equal computation on every token and every block. It should first derive a compact semantic state, then use that state to decide which expensive paths are worth activating.

## V0 Block

```text
H
-> Norm
-> Attention branch: GQA first, MLA later
-> NAIME semantic compressor
-> Context-state-aware router
-> Shared expert + sparse MoE experts
-> Residual output
```

## Key Innovation

Context-state-aware expert routing:

```text
z_block = SemanticCompressor(H_window)
router_input = concat(h_token, z_block)
expert_ids, expert_weights = Router(router_input)
```

This makes expert selection depend on both local token representation and block-level semantic state.

## NAIME Adaptation

The original NAIME idea is adapted from replacement to residual writing:

```text
H_out = H + alpha * DeltaH(z_block)
```

This reduces the risk of damaging the backbone representation while preserving selective semantic intervention.

## First Prototype Scope

V0 intentionally excludes the hardest pieces:

- no full MLA yet;
- no SSM branch yet;
- no MTP head yet;
- no large-scale distributed MoE.

V0 includes:

- Transformer-compatible block;
- block/window semantic compressor;
- sparse gate;
- small top-k MoE;
- load-balancing loss;
- routing diagnostics.
