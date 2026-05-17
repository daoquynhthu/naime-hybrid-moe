# Experiment Plan V0

## Goal

Prove the semantic compressor is useful rather than ornamental.

## Baselines

- `dense`: small decoder-only Transformer.
- `token_moe`: same backbone with standard token-only MoE router.
- `naime_state_moe`: NAIME semantic-state-aware MoE.

## Metrics

- validation perplexity;
- routing entropy;
- expert load balance;
- correlation between semantic state and expert choice;
- long-context retrieval or synthetic needle accuracy;
- throughput and memory footprint where feasible.

## Minimal Datasets

- TinyStories for quick language modeling;
- WikiText for conventional perplexity;
- synthetic retrieval/needle tasks for long-context behavior.

## Success Criteria

- Hybrid model matches or improves baseline perplexity at similar compute.
- Semantic latent state measurably changes routing decisions.
- Expert collapse does not occur under normal training settings.
