# Proposal V1: NAIME-State MoE

Date: 2026-05-09 (updated 2026-05-12)

Status: **implemented and validated through 200M-token pre-training.**

## Position

NAIME-State MoE is a practical decoder-only language-model architecture that keeps the Transformer training ecosystem, but adds a NAIME-style semantic state pathway to control sparse expert routing.

The core bet is simple:

```text
Do not route tokens using only local token hidden states.
First summarize block-level semantic state, then route tokens and optional memory writes using that state.
```

This makes the architecture distinct from standard MoE while staying buildable with PyTorch first and optimized kernels later.

## Validation

A 200M-token pre-training run on FineWeb-Edu (67M params, ctx512) confirmed:

- val_lm: 6.25 → 1.26 (monotonic)
- Semantic memory and world state are stable (no degradation).
- Semantic gate mixer converges early and stays fixed thereafter (α/c/s = 0.14/0.52/0.34).
- Router entropy drifts from 1.17 toward 1.08, indicating emerging expert specialisation.
- `torch.compile` delivers 2.4x throughput.

Current gap: semantic prior contributes 14% to routing vs clean_prob's 52%. Next phase targets higher signal-to-noise ratio.

## Research Sources

Local sources:

- local NAIME instruction/design notes
- local NAIME model implementation notes
- local NAIME loss implementation notes

Open architecture sources:

- `G:\llm_architecture\docs\architectures\source-audit-2026-05-09.md`
- `G:\llm_architecture\docs\architectures\architecture-pros-cons-2026-05-09.md`

## What We Keep

From NAIME:

- Read-window / write-stride block view.
- Attention pooling over local windows.
- Gumbel-style sparse gate.
- Latent semantic variable `z_block`.
- KL / sparsity control and collapse prevention.
- InfoNCE-style semantic anchor as a later training objective.

From DeepSeek / GLM / Kimi:

- Dense warmup layers before sparse blocks.
- Sparse MoE with routed experts plus shared expert.
- Router load-balancing diagnostics.
- Later MLA-style attention to reduce KV pressure.

From Qwen3:

- Conservative decoder-only Transformer base.
- QK normalization as a low-risk stability improvement.
- GQA as the first attention implementation before MLA.

From Mamba:

- Optional SSM memory branch as a later long-context extension.
- The first version will not depend on SSM kernels.

From GLM / vLLM / SGLang:

- No-op expert dispatch API that accepts dynamic routing in a look-ahead-compatible way.
- Shared expert forward path that is structurally compatible with speculative decoding.

## Architecture

### Token-Level

```text
h_t = RMSNorm(embed_t)
h_t = h_t + Attn(h_{t-ctx:t})
h_t = h_t + FFN_MoE(h_t, alpha_block * latent_block(⌊t/stride⌋))
```

### Semantic Compressor

```text
For block b spanning tokens [b*stride, b*stride+window]:

pooled = AttentionPool(hidden[block_range])
mu_b, sigma_b = Encoder(pooled)       # VAE bottleneck
z_b = mu_b + sigma_b * N(0,1)         # train-time; mu_b at eval
alpha_b = GumbelGate(pooled)          # ∈ [0,1], controlled by target_sparsity
kl_b = -0.5 * Σ(1 + log(sigma²) - mu² - sigma²)

local_semantic  = Linear(z_b)
mid_semantic    = MidScalePool(local_semantic)   # optional, stride 32
global_semantic = GlobalPool(hidden)              # optional
semantic = Fusion(local, mid, global)            # gated_sum or concat
```

### Semantic-MoE Routing

```text
router_input_t = concat(h_t, alpha_block * z_for_token_t)
expert_logits_t = Router(router_input_t) + semantic_prior_gate * prior_scale * SemanticPrior(semantic_t)
experts_t = top_k(expert_logits_t)
```

This is the main innovation. Routing is conditioned on both local hidden state and block-level semantic state.

### Sparse Semantics Boundary

The research claim depends on sparse expert routing semantics, not on any one
PyTorch dispatch implementation.

Required architectural properties:

- routing still produces `top_k` expert selection per token;
- routed output still comes only from the selected experts plus the shared expert;
- semantic state must still be able to change expert choice in a measurable way.

Allowed implementation changes:

- replace Python sparse dispatch with vectorized dispatch;
- replace vectorized dispatch with Triton/CUDA kernels;
- use a denser execution backend internally if the final routed computation is
  still equivalent to `top_k` sparse routing.

Not allowed under the same architecture ID:

- changing routed experts into an all-expert weighted mixture;
- removing `top_k` conditional activation while still calling the model
  NAIME-State MoE;
- redefining the block as a dense FFN with semantic conditioning only.

If those semantics change, the project should introduce a new architecture ID
and treat it as an ablation or successor, not as the same NAIME-State MoE.

## Shared Expert

Every token passes through a shared expert:

```text
y_shared = SharedSwiGLU(h_t)
```

Routed experts add conditional capacity:

```text
y_routed = sum_i router_weight_i * Expert_i(h_t)
```

Block FFN output:

```text
h_t = y_shared + y_routed
```

## Implementation Status

| Component | V1 | V4 | V5 |
|-----------|:--:|:--:|:--:|
| Dense baseline | ✓ | ✓ | ✓ |
| Token MoE baseline | ✓ | ✓ | ✓ |
| NAIME VAE compressor | ✓ | ✓ | ✓ |
| Semantic-MoE routing (concat) | ✓ | ✓ | ✓ |
| Semantic-MoE routing (prior) | ✓ | ✓ | ✓ |
| Semantic-MoE routing (hybrid) | ✓ | ✓ | ✓ |
| 3-scale semantic (local+mid+global) | — | ✓ | ✓ |
| Confidence-gated cross-layer state | — | ✓ | ✓ |
| Semantic memory bank | — | ✓ | ✓ |
| Semantic gate mixer | — | ✓ | ✓ |
| Gate mixer max_clean_weight | — | ✓ | ✓ |
| World state slots (4) | — | — | ✓ |
| Slot diversity/stability loss | — | — | ✓ |
| State transition predictor | — | — | ✓ |
| `torch.compile` support | — | — | ✓ |
| FlashAttention (SDPA backend) | ✓ | ✓ | ✓ |
| `auto` MoE dispatch mode | — | — | ✓ |
| `dispatch_dense` metric | — | — | ✓ |
| Async prefetch pipeline | — | — | ✓ |
| Collate-based causal shift | — | — | ✓ |
| Cosine warm restarts | — | — | ✓ |
| LR continuity on resume | — | — | ✓ |
