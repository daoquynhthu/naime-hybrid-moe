# Implementation Roadmap

Date: 2026-05-09 (updated 2026-05-17)

## Milestone 0: Project Base - Done

- Project directory and repository.
- Architecture proposal and research synthesis.
- Experiment plan and working documentation.

## Milestone 1: Minimal Modules - Done

- `semantic_compressor.py`: VAE compressor with multi-scale pooling.
- `gate.py`: Gumbel block gate.
- `moe.py`: TopKMoE with auto/dense/sparse dispatch and hybrid router.
- `attention.py`: GQA and MLA attention variants.

## Milestone 2: Decoder Models - Done

- `decoder.py`: dense, TokenMoE, NAIME state MoE, V4, V5, and V6 blocks.
- `factory.py`: build-model dispatcher.
- Architecture smoke tests cover the current model families.

## Milestone 3: Training Loop - Done

- CLI with broad parameter coverage.
- Automatic VRAM batch-size probing.
- Resume with scheduler/LR policy options.
- Model-only and full checkpoint save.
- Structural stop and early stop.
- Async checkpoint writer.
- Compact console logging plus persistent full logs.
- `torch.compile` integration where supported.

## Milestone 4: V4 Semantic State - Done

- Cross-layer semantic memory.
- Confidence-gated read/write.
- Semantic gate mixer.
- V4.2 validation: `val_lm 3.43` on 50M-token FineWeb-Edu.

## Milestone 5: V5 World State - Done

- Structured state slots with diversity/stability objectives.
- Slot attention router.
- State transition predictor.
- Hybrid confidence mode.
- Validation: 200M-token run, `val_lm 1.26`, no major degradation.

## Milestone 6: Data Pipeline - Done

- `HFDiskCausalDataset` with `set_format(type="torch")`.
- Collate-based causal shift.
- Async prefetch through CUDA stream overlap.
- `persistent_workers + prefetch_factor`.
- 1B-token FineWeb-Edu ctx1024 corpus preparation.
- Resumable shuffled sampler for segmented non-replaying continuation.

## Milestone 7: V6 Recursive Self-State - Active

Target: 100M-param class model, FineWeb-Edu 1B ctx1024, segmented continuation.

Completed:

- [x] V6 recursive self-state architecture.
- [x] Self/world/other/unknown boundary metrics.
- [x] Remote 4090 training workflow.
- [x] First credible 1B-corpus validation at step 10000 (`val_lm 2.0106`).
- [x] Latest completed continuation reaches step 32958 (`val_lm 0.7033`, `val_ppl 2.020`).
- [x] Bad-gradient logging and adaptive LR safety factor.
- [x] Auto-batch probing with safer skip/headroom behavior.

In progress:

- [ ] Complete the 1B-token curriculum without data replay.
- [ ] Reduce bad-gradient spike rate in late continuation.
- [ ] Strengthen world-state utilization relative to self-state.
- [ ] Validate generation quality, not only validation loss.
- [ ] Convert more MoE paths to efficient sparse dispatch where it materially improves throughput.

## Milestone 8: Post-V6 Scaling Plan - Planned

The next stage should not be a blind parameter increase. Scaling should happen only with:

- stable segmented continuation;
- verified checkpoint/STOP behavior;
- clear generation evaluation;
- architecture metrics showing that recursive self-state and world-state remain useful;
- throughput that justifies larger expert/state capacity.
