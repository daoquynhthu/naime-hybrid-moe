# Frontier Architecture Synthesis

## Useful Lessons

- DeepSeek-V3: sparse MoE plus shared experts plus MLA is a strong capability/efficiency pattern.
- DeepSeek-R1: reasoning behavior is mostly post-training on top of a strong base architecture.
- Qwen3: conservative Transformer evolution remains valuable because ecosystem compatibility matters.
- Kimi-K2.5: large sparse multimodal MoE is strategically important, but source-level auditability is limited.
- Mamba/Mamba2: SSMs are promising as long-context memory branches, but replacing attention entirely is risky.
- GLM-4.x: MTP/speculative decoding is becoming architecture-level, not just an inference trick.
- vLLM/SGLang: serving runtime is part of the effective architecture.

## Local Project Lessons

- NAIME contributes selective semantic compression, sparse activation, latent semantic state, and control dynamics.
- SNN contributes prediction error, surprise/modulation, functional regions, and structural plasticity as inspiration, but is deferred due to compute and controllability constraints.

## Working Direction

Use NAIME as the bridge between original ideas and practical LLM architecture:

```text
Transformer backbone
+ semantic compressor
+ context-state router
+ sparse experts
+ future MLA/SSM/MTP extensions
```
