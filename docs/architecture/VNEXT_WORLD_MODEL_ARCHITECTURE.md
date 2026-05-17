# NAIME VNext World-Model Architecture Plan

Status: **VNext architecture is unverified.**

This document is a planning blueprint. It is not a claim that the proposed architecture has already surpassed existing frontier systems. The current V4/V4.1/V4.2 code validates parts of the NAIME direction at small scale, but the VNext architecture described here still requires implementation, ablation, scaling, and independent evaluation.

## 1. North Star

The final goal is not merely a better MoE and not merely a multimodal demo. The goal is a language model with an operational **inner world**:

- It maintains persistent latent state beyond the visible token window.
- It builds and revises internal models of entities, goals, causes, time, and uncertainty.
- It can simulate possible futures before answering or acting.
- It can separate observed facts, inferred beliefs, plans, memories, and generated hypotheses.
- It can decide when to use fast response, extended reasoning, retrieval, memory write, tool use, or self-checking.
- It can ground language in perception and action without letting multimodal engineering distract from the core intelligence architecture.

In short: the model should not only predict the next token. It should learn a compact, updateable, self-auditing world state from which token prediction, reasoning, dialogue, perception, and action become different readout modes.

## 2. Frontier Signals Considered

The plan considers, but does not blindly copy, several frontier directions:

- Closed-source frontier systems emphasize multimodality, reasoning-time compute, long context, tool use, safety layers, and strong post-training.
- GPT-4o-like systems show the value of native multimodal training across text, vision, and audio.
- Claude-style hybrid reasoning suggests a single model should support both immediate answers and deeper deliberation.
- Gemini-style long-context multimodality suggests memory and retrieval must be first-class, not bolted on.
- DeepSeek-style MoE shows that efficient active parameters, latent attention/KV compression, MTP objectives, and load balancing are core scaling levers.
- Mamba/SSM and Titans-style memory work suggests attention alone is not enough for persistent world state.
- MiniMind-O is useful only as a small Thinker-Talker engineering reference; it is not a central architecture target.

Our conclusion: the winning architecture is likely not a pure Transformer, pure SSM, pure MoE, pure retriever, or pure multimodal stack. It should be a **stateful predictive architecture** with multiple memory timescales, accountable semantic routing, and explicit internal simulation.

## 3. Core Hypothesis

Current LLMs mostly compress the world into weights and a temporary context window. This gives broad knowledge, but weak continuity:

- no stable private world state;
- brittle long-horizon planning;
- expensive long context;
- weak separation between memory and hallucination;
- limited ability to update beliefs during inference;
- limited introspection into why modules are used.

NAIME VNext should add a structured latent world state between tokens and weights:

```text
tokens / modalities
    -> perception encoders
    -> semantic state builder
    -> persistent world state
    -> simulator / reasoner / router
    -> language, action, speech, or tool outputs
```

The model should learn not only `P(next token | context)`, but also:

```text
P(next state | current state, observation, action)
P(observation | latent state)
P(outcome | plan, state)
P(confidence | state, evidence)
P(memory write usefulness | state transition)
```

## 4. Proposed Architecture: NAIME VNext

### 4.1 Foundation Backbone

Use a Transformer-compatible backbone, but treat it as the local working cortex, not the entire mind.

Required upgrades:

- Efficient attention: GQA now, MLA-style latent KV compression later.
- Optional SSM/linear-memory layers for long-range sequential continuity.
- RoPE/Yarn-style context scaling only as a bridge, not as the final memory solution.
- Multi-token prediction objective for better planning and faster decoding.
- Separate inference paths for fast mode and deliberation mode.

Design rule: do not replace the Transformer prematurely. Wrap it with persistent state and memory first; only then decide where SSM or MLA gives real benefit.

### 4.2 Semantic State System

V4/V4.2 already introduced semantic compression, cross-layer state, memory, and gate mixing. VNext should turn these into a real state system.

Components:

- **Observation encoder:** converts token spans or modality embeddings into structured semantic events.
- **State updater:** updates latent world state with gated, confidence-weighted transitions.
- **State slots:** represent entities, tasks, speaker intent, constraints, uncertainty, and temporal context.
- **State confidence:** calibrated confidence, not just learned scalar output.
- **State delta:** measures how much the world state changed after each observation.
- **State provenance:** tracks whether a state came from prompt evidence, memory, retrieval, inference, or model imagination.

State must not be a decorative residual. It must be used by router decisions, memory writes, reasoning mode selection, and output calibration.

### 4.3 Memory Hierarchy

The architecture needs at least four memory timescales:

- **Working memory:** active context tokens and local attention.
- **Recurrent semantic state:** per-sequence latent state passed across layers and chunks.
- **Episodic memory:** compressed records of interactions, observations, decisions, and outcomes.
- **Semantic memory:** abstracted stable knowledge distilled from repeated episodes or external corpora.

Memory writes must be accountable:

- write only when novelty and utility exceed thresholds;
- store provenance and confidence;
- decay or revise stale memories;
- allow contradiction detection;
- evaluate whether reading memory improves prediction or reasoning.

The model should learn memory policy, not merely append vectors.

### 4.4 Accountable MoE

MoE should become expert specialization, not just parameter expansion.

Required upgrades:

- Hierarchical router: semantic domain router -> task router -> token expert router.
- Expert identity metrics: each expert should develop measurable specialization.
- Router counterfactuals: periodically compare with and without semantic state/memory.
- Auxiliary-loss-free or low-interference balancing where possible.
- Expert dropout / expert swap tests to detect fake specialization.
- Persistent expert memory: experts may own small specialized memory banks.

The router must answer: "Why this expert?" Metrics should expose whether the answer is meaningful.

### 4.5 Internal Simulator

To move toward an inner world, VNext needs a simulator branch.

The simulator predicts:

- next latent state;
- likely observations;
- consequences of candidate actions;
- contradictions between planned answer and known state;
- uncertainty of predicted outcomes.

During normal LM training, the simulator can be trained with self-supervised future-state prediction. During reasoning/post-training, it can be used for:

- plan ranking;
- answer verification;
- tool-call anticipation;
- hallucination suppression;
- multi-step reasoning search.

This should start small: one latent transition head and one outcome verifier before attempting full tree search.

### 4.6 Deliberation Controller

Closed-source systems increasingly separate fast answering from extended thinking. VNext should make this explicit.

Controller outputs:

- fast answer;
- deliberate answer;
- retrieve memory;
- write memory;
- call tool;
- run simulator;
- ask clarification;
- refuse/abstain;
- verify answer.

The controller observes:

- task difficulty;
- state uncertainty;
- conflict between memory and prompt;
- router entropy;
- verifier confidence;
- expected cost.

This can be implemented first as a small learned policy head with heuristic guardrails.

### 4.7 Multimodal Path

MiniMind-O is not a frontier target, but it offers a useful engineering lesson: multimodal I/O can be staged with frozen encoders and projectors before native end-to-end training.

VNext multimodal roadmap:

1. Frozen image/audio encoders + NAIME projectors.
2. Unified semantic event representation for text, image, and audio.
3. State-grounded perception: visual/audio features update world state, not just prompt tokens.
4. Optional Talker branch for speech output if needed.
5. Native multimodal token training only after text-world-state architecture is stable.

Multimodality is important, but it should not hijack the core LM world-model work.

## 5. Training Plan

### Phase A: Stabilize V4.2

Goal: verify accountable semantic influence.

Success criteria:

- validation loss improves versus V4/V4.1 at matched token budget;
- gate mixer does not collapse to clean gate;
- state confidence remains informative;
- memory read/write metrics move dynamically;
- semantic ablation shows nonzero usefulness.

### Phase B: VNext State Pretraining

Add self-supervised objectives:

- next-token LM loss;
- next-latent-state prediction;
- masked semantic event reconstruction;
- memory usefulness prediction;
- contradiction detection on synthetic perturbations;
- multi-token prediction.

Train on high-quality text first. Do not start with full multimodality.

### Phase C: Expert Specialization

Add:

- semantic-domain routing;
- expert coherence metrics;
- expert specialization regularizers;
- counterfactual expert ablations;
- low-interference load balancing.

Success is not lower loss alone. Experts must become interpretable and robust under ablation.

### Phase D: Deliberation and Verification

Add:

- verifier head;
- uncertainty calibration;
- answer-vs-state consistency loss;
- tool-use/action traces if available;
- fast vs deliberate controller.

Measure:

- reasoning benchmark improvement;
- hallucination reduction;
- calibration;
- compute-adjusted performance.

### Phase E: Persistent Memory

Add chunk-to-chunk and session-to-session memory.

Measure:

- long-context recall without full context replay;
- contradiction handling;
- memory precision/recall;
- degradation under stale or adversarial memories.

### Phase F: Multimodal Grounding

Add frozen encoders first.

Measure:

- visual/audio grounding;
- state updates from modalities;
- cross-modal contradiction detection;
- text-only ability retention.

## 6. Evaluation Matrix

VNext cannot be judged by perplexity alone.

Required groups:

- LM: validation loss, perplexity, tokenizer-normalized loss.
- Routing: entropy, expert load, expert specialization, ablation damage.
- State: confidence calibration, state delta, state agreement, contradiction rate.
- Memory: read usefulness, write precision, stale-memory resistance.
- Reasoning: math/code/planning benchmarks, compute-adjusted gains.
- Long context: recall, synthesis, timeline tracking, entity consistency.
- Multimodal: grounding accuracy, contradiction detection, retention.
- Safety/robustness: hallucination, prompt injection, memory poisoning.
- Systems: throughput, VRAM, checkpoint pressure, crash rate.

Each new module must justify itself with at least one direct metric and one ablation.

## 7. Implementation Roadmap

### V4.2 Current

Already implemented:

- clean-gate cap;
- state-confidence-gated semantic injection;
- memory-read-gated semantic injection;
- conservative checkpoint defaults;
- V4.2 architecture ID.

Status: **implemented but not yet trained/validated**.

### V4.3

Next concrete target:

- semantic ablation metrics;
- expert specialization metrics;
- state/memory counterfactual logging;
- first memory usefulness loss;
- optional MTP head for text.

### V5

World-state architecture:

- structured state slots;
- provenance tags;
- next-state prediction head;
- verifier head;
- deliberation controller.

### V6

Persistent memory and long-context system:

- session memory;
- memory compaction;
- contradiction/revision policy;
- retrieval-free long-context tests.

### V7

Grounded multimodal extension:

- frozen encoders + projectors;
- modality-to-state events;
- optional speech Talker;
- native multimodal training only after text-state stability.

## 8. Risks

- Semantic state may become a decorative residual.
- Memory may store noise and reinforce hallucinations.
- MoE experts may not specialize meaningfully.
- Simulator may learn shallow next-token proxies instead of causal state transitions.
- Extra objectives may improve diagnostics but hurt generation.
- Hardware limits may prevent meaningful scale.
- Too many modules can create debugging fog.

Countermeasure: every module requires an ablation and a failure-mode metric.

## 9. What "Surpass Existing Architectures" Means

We should not claim superiority because of novelty. The target is to surpass current architectures on axes where current LLMs are structurally weak:

- persistent world state;
- controllable memory;
- calibrated uncertainty;
- long-horizon coherence;
- expert interpretability;
- reasoning-time compute efficiency;
- state-grounded multimodal understanding;
- lower hallucination under memory/retrieval pressure.

If NAIME VNext only matches Transformer/MoE loss while adding reliable state, memory, and self-verification, that is already meaningful. True superiority requires scaling and external evaluation.

## 10. References

- OpenAI GPT-4o System Card: https://openai.com/index/gpt-4o-system-card/
- Anthropic Claude 3.7 Sonnet announcement: https://www.anthropic.com/news/claude-3-7-sonnet
- Google Gemini 1.5 technical report: https://arxiv.org/abs/2403.05530
- Google Gemini 2.5 model docs: https://ai.google.dev/gemini-api/docs/models/gemini-v2
- DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437
- Mamba-2 / Structured State Space Duality: https://arxiv.org/abs/2405.21060
- Titans test-time memory: https://arxiv.org/abs/2501.00663
