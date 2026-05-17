# NAIME and Frontier Architecture Research Notes

Date: 2026-05-09

## Summary

The usable path is not to copy any single frontier architecture. The strongest direction is to combine NAIME's semantic compression with the sparse-capacity pattern used by DeepSeek/GLM/Kimi-style MoE models.

The result should be a Transformer-compatible model whose experts are routed by both token state and compressed semantic state.

## NAIME: What Matters

NAIME contributes an original local mechanism:

```text
hidden sequence -> window summary -> sparse gate -> latent semantic state -> controlled write/use
```

The strongest ideas are:

- read-window / write-stride processing;
- block-level semantic summaries;
- differentiable sparse activation;
- VIB latent bottleneck;
- explicit anti-collapse controls;
- semantic anchor loss.

## NAIME: What We Should Change

The original implementation tries to produce `H_prime` and can replace hidden-state blocks. That is too invasive for our first model.

Change:

```text
from: H_prime = (1 - alpha) * H + alpha * H_stoch
to:   use z_block to condition routing first
```

Optional later:

```text
H = H + alpha * DeltaH(z_block)
```

This makes NAIME an internal routing and memory mechanism instead of a risky hidden-state replacement module.

## DeepSeek-V3 Lesson

DeepSeek-V3 suggests the following practical pattern:

```text
dense early layers
-> sparse MoE later layers
-> shared experts
-> MLA for cache economics
```

The important lesson is not just MoE. It is sparse capacity with a dense shared fallback.

Our use:

- start with dense warmup layers;
- every MoE layer has shared expert plus routed experts;
- save MLA for V2 after the router mechanism is proven.

## Qwen3 Lesson

Qwen3 is a reminder that conservative Transformer evolution wins when tooling matters.

Useful parts:

- decoder-only base;
- RoPE;
- GQA;
- QK RMSNorm;
- broad PyTorch/vLLM/SGLang compatibility.

Our use:

- use GQA/SDPA first;
- add QK norm as a simple stability feature;
- avoid exotic dependencies in V1.

## GLM Lesson

GLM-style architectures show that MTP/speculative decoding is becoming architectural, not merely a serving trick.

Our use:

- keep MTP in the roadmap;
- do not implement it before the base training loop and router diagnostics are reliable.

## Mamba Lesson

Mamba's SSM branch is a promising way to avoid full KV-cache dependence, but it is a risky first dependency because performance depends on specialized kernels and state compression may hurt retrieval-like tasks.

Our use:

- defer SSM to V2/V3;
- treat it as an optional memory branch, not a replacement for attention.

## Kimi-K2.5 Lesson

Kimi-K2.5 validates the direction of huge sparse capacity, MLA, shared experts, and multimodal/agentic ambition. But the local evidence is mostly release-level, not source-level.

Our use:

- borrow the strategic pattern, not undocumented implementation details.

## Runtime Lesson

vLLM and SGLang show that architecture and serving are now intertwined:

- KV layout matters;
- expert dispatch matters;
- batching matters;
- speculative decoding matters;
- parser/tool-call runtime matters for agentic models.

Our use:

- keep V1 PyTorch-native;
- design module boundaries so routing and expert dispatch can later be replaced by Triton/CUDA kernels.

## Final Research Claim

If the project becomes publishable, the claim should not be:

```text
We combined MoE, MLA, Mamba, and NAIME.
```

The claim should be:

```text
Block-level semantic latent states improve sparse expert routing by giving the router access to compressed context beyond local token embeddings.
```

This is precise, testable, and connected to both NAIME and frontier MoE systems.
