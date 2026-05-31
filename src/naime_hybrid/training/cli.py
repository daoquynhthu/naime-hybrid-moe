import argparse

from naime_hybrid.config import NAIMEStateMoEConfig

from .config import TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NAIME Hybrid architectures.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--architecture",
        default="naime_state_moe",
        choices=[
            "dense",
            "token_moe",
            "naime_state_moe",
            "naime_v4_state_moe",
            "naime_v41_state_moe",
            "naime_v42_state_moe",
            "naime_v5_world_state_moe",
            "naime_v6_recursive_self_moe",
            "naime_v7_typed_dynamics",
        ],
    )
    parser.add_argument("--run-name", default="debug")
    parser.add_argument("--output-dir", default="experiments/runs")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-format", default="auto", choices=["auto", "byte", "hf_disk"])
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--random-data", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--auto-batch", action="store_true", help="Probe GPU memory and raise batch size up to the VRAM budget."
    )
    parser.add_argument(
        "--vram-fraction",
        type=float,
        default=0.9,
        help="Fraction of currently free VRAM allowed during auto-batch probing.",
    )
    parser.add_argument("--auto-batch-max", type=int, default=128, help="Upper bound for auto-selected batch size.")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="If set, derive max_steps after auto-batch so runs consume a comparable token budget.",
    )
    parser.add_argument(
        "--target-tokens-mode",
        default="total",
        choices=["total", "additional"],
        help="Interpret --target-tokens as total run budget or extra budget after the resumed step.",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--lm-loss-backend",
        default="auto",
        choices=["auto", "torch", "triton_ce", "cuda_ext_ce", "cuda_ext_fused_ce"],
        help="LM cross-entropy backend. auto uses safe accelerated kernels when available.",
    )
    parser.add_argument(
        "--use-fused-state-attention",
        action="store_true",
        help="Use the native CUDA state softmax+matmul path for small slot banks.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument(
        "--lr-cycle-length", type=int, default=0, help="Steps per cosine warm restart cycle. 0 = single cosine."
    )
    parser.add_argument("--lr-restart-ratio", type=float, default=0.5, help="Peak LR multiplier at each restart.")
    parser.add_argument("--lr-restart-warmup", type=int, default=200, help="Warmup steps at each restart.")

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument(
        "--latest-every",
        type=int,
        default=1000,
        help="Refresh latest.pt this often. Heavy step_*.pt archives still follow --save-every.",
    )
    parser.add_argument(
        "--async-latest",
        action="store_true",
        help="Deprecated compatibility flag. latest.pt is asynchronous by default unless --sync-latest is set.",
    )
    parser.add_argument(
        "--sync-latest",
        action="store_true",
        help="Wait for latest.pt writes to finish before continuing. Safer, but can visibly stall training.",
    )
    parser.add_argument("--no-async-checkpoint", action="store_true")
    parser.add_argument("--async-checkpoint-queue", type=int, default=2)
    parser.add_argument(
        "--metrics-flush-every",
        type=int,
        default=50,
        help="Flush metrics.jsonl to the OS page cache every N writes.",
    )
    parser.add_argument(
        "--metrics-fsync-every",
        type=int,
        default=1000,
        help="Force metrics.jsonl to disk every N writes. 0 disables periodic fsync; final close still fsyncs.",
    )
    parser.add_argument(
        "--best-checkpoint-mode",
        default="model",
        choices=["full", "model"],
        help="Use model to save only model_best.pt on validation improvement; full also saves best.pt with optimizer state.",
    )
    parser.add_argument(
        "--eval-every", type=int, default=0, help="Evaluate periodically and save best checkpoint when > 0."
    )
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-max-batches", type=int, default=10, help="0 means full eval split.")
    parser.add_argument(
        "--eval-sampling",
        default="random",
        choices=["random", "sequential"],
        help="random avoids repeatedly validating only the prefix of the validation split; sequential keeps legacy order.",
    )
    parser.add_argument(
        "--eval-seed", type=int, default=4321, help="Seed for deterministic random validation sampling."
    )
    parser.add_argument(
        "--eval-state-carry",
        action="store_true",
        help="During validation, estimate whether StatePacket carryover improves the next chunk LM loss.",
    )
    parser.add_argument(
        "--eval-latent-thought-gain",
        action="store_true",
        help="During validation, estimate whether latent thought steps improve LM loss versus steps=0.",
    )
    parser.add_argument(
        "--eval-v7-dynamics-gain",
        action="store_true",
        help="During validation, estimate whether V7 typed dynamics steps improve LM loss versus steps=0.",
    )
    parser.add_argument(
        "--eval-v7-state-swap",
        action="store_true",
        help="During validation, swap StatePacket batch entries and measure whether wrong state hurts next chunk loss.",
    )
    parser.add_argument(
        "--eval-v7-state-erase",
        action="store_true",
        help="During validation, erase world/self/latent packet fields and measure next chunk sensitivity.",
    )
    parser.add_argument(
        "--eval-doc-continuity",
        action="store_true",
        help="During validation, run multi-chunk continuity probes over sequential chunks from the same sample.",
    )
    parser.add_argument(
        "--eval-doc-continuity-docs",
        type=int,
        default=32,
        help="How many sequential-sample continuity traces to evaluate when --eval-doc-continuity is enabled.",
    )
    parser.add_argument(
        "--eval-doc-continuity-chunks",
        type=int,
        default=4,
        help="How many consecutive chunks to follow inside each continuity trace.",
    )
    parser.add_argument(
        "--diagnostics-mode",
        action="store_true",
        help="Enable training-time data-flow diagnostics. Disabled by default for real training.",
    )
    parser.add_argument(
        "--diagnostics-every",
        type=int,
        default=0,
        help="Run training-time packet diagnostics every N optimizer steps when --diagnostics-mode is set.",
    )
    parser.add_argument(
        "--diagnostics-output-dir",
        default=None,
        help="Optional diagnostics artifact root. Defaults to <run_dir>/training_diagnostics.",
    )
    parser.add_argument(
        "--diagnostics-chunk-len",
        type=int,
        default=None,
        help="Chunk length for training-time packet diagnostics. Defaults to --stateful-chunk-len or half sequence.",
    )
    parser.add_argument(
        "--diagnostics-boundary-tokens",
        type=int,
        default=64,
        help="Boundary window used by training-time state carry diagnostics.",
    )
    parser.add_argument(
        "--diagnostics-max-batch",
        type=int,
        default=2,
        help="Maximum current-batch samples used by diagnostics to keep explicit mode bounded.",
    )
    parser.add_argument(
        "--diagnostics-no-tensor-stats",
        action="store_true",
        help="Record event topology only; skip tensor statistics in training-time diagnostics.",
    )
    parser.add_argument(
        "--diagnostics-no-grad-components",
        action="store_true",
        help="Skip per-module gradient norm grouping in diagnostics mode.",
    )
    parser.add_argument(
        "--diagnostics-window-size",
        type=int,
        default=16,
        help="Recent dynamics events retained in bad-gradient window snapshots.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many evals without validation improvement. 0 disables.",
    )
    parser.add_argument(
        "--early-stop-min-delta", type=float, default=0.0, help="Required val loss decrease to count as improvement."
    )
    parser.add_argument(
        "--early-stop-min-evals", type=int, default=0, help="Minimum eval count before early stopping can trigger."
    )
    parser.add_argument(
        "--reference-metrics-path", default=None, help="Historical metrics.jsonl used by structural-stop comparison."
    )
    parser.add_argument(
        "--structural-stop",
        action="store_true",
        help="Abort when validation falls structurally behind the reference curve.",
    )
    parser.add_argument(
        "--structural-stop-min-gap",
        type=float,
        default=0.30,
        help="Minimum current-reference val_lm gap before structural stop can trigger.",
    )
    parser.add_argument(
        "--structural-stop-widen-delta",
        type=float,
        default=0.05,
        help="Required gap increase versus previous eval to count as widening.",
    )
    parser.add_argument(
        "--structural-stop-patience", type=int, default=2, help="Consecutive widening evals before structural stop."
    )
    parser.add_argument(
        "--structural-stop-min-evals",
        type=int,
        default=3,
        help="Minimum eval count before structural stop can trigger.",
    )
    parser.add_argument(
        "--structural-stop-warmup-steps", type=int, default=1000, help="Ignore structural stop before this step."
    )
    parser.add_argument("--keep-last-n", type=int, default=2)
    parser.add_argument(
        "--resume",
        default="none",
        help="Checkpoint to resume from. Defaults to none; pass auto or an explicit path only for clean checkpoints.",
    )
    parser.add_argument(
        "--resume-lr-policy",
        default="checkpoint",
        choices=["checkpoint", "absolute", "progress", "reset"],
        help=(
            "LR handling after resume. checkpoint keeps loaded scheduler state; absolute uses the resumed step on the "
            "current schedule; progress remaps old max_steps to new max_steps; reset restarts LR warmup."
        ),
    )
    parser.add_argument(
        "--resume-allow-failed",
        action="store_true",
        help="Allow --resume auto and bad-gradient recovery to use failed.pt. Disabled by default.",
    )
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help=(
            "Allow checkpoints without the current causal-integrity marker. "
            "Use only for forensic inspection or intentionally contaminated baselines, never for clean training."
        ),
    )
    parser.add_argument(
        "--stop-file",
        default=None,
        help="Gracefully stop after a step when this file exists. Defaults to STOP in the run directory.",
    )
    parser.add_argument(
        "--stop-check-every",
        type=int,
        default=1,
        help="Check the stop file every N optimizer steps. 1 is safest for shared machines.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--compile-scope",
        default="full",
        choices=["full", "dense"],
        help=(
            "torch.compile scope. full compiles the whole decoder; dense only compiles ordinary dense "
            "Transformer blocks and leaves state/MoE/self-recursion eager."
        ),
    )
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        choices=["inductor", "eager", "aot_eager"],
        help="torch.compile backend. Use eager only for plumbing diagnostics; inductor is the performance path.",
    )
    parser.add_argument(
        "--disable-flash-sdp",
        action="store_true",
        help=(
            "Disable CUDA flash and memory-efficient SDPA kernels. Useful when torch.compile "
            "or driver builds crash in flash-attention backward; keeps training on the safer math backend."
        ),
    )
    parser.add_argument("--device", default="auto")

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-dense-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-type", default="gqa", choices=["gqa", "mla"])
    parser.add_argument("--mla-latent-dim", type=int, default=128)
    parser.add_argument("--mla-rope-per-head", type=int, default=32)

    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--target-sparsity", type=float, default=0.2)
    parser.add_argument("--gumbel-tau", type=float, default=1.0)
    parser.add_argument("--gate-eval-mode", default="prob", choices=["prob", "hard"])
    parser.add_argument("--logvar-clip", type=float, default=10.0)
    parser.add_argument("--semantic-router-mode", default="concat", choices=["concat", "prior", "hybrid"])
    parser.add_argument("--semantic-router-prior-scale", type=float, default=1.0)
    parser.add_argument("--semantic-router-prior-clip", type=float, default=0.0)
    parser.add_argument("--semantic-router-prior-gate", action="store_true")
    parser.add_argument("--semantic-router-detach", action="store_true")
    parser.add_argument("--semantic-router-alpha-cap", type=float, default=0.0)
    parser.add_argument("--semantic-alpha-cap-mode", default="clamp", choices=["clamp", "scale"])
    parser.add_argument("--semantic-gate-downstream", default="alpha", choices=["alpha", "prob", "clean_prob", "none"])
    parser.add_argument(
        "--semantic-sparse-alpha",
        default="alpha",
        choices=["alpha", "prob", "clean_prob", "capped_alpha", "downstream"],
    )
    parser.add_argument("--semantic-downstream-deterministic", action="store_true")
    parser.add_argument("--semantic-scales", default="local", choices=["local", "local_mid", "local_mid_global"])
    parser.add_argument("--mid-stride", type=int, default=32)
    parser.add_argument("--mid-window", type=int, default=64)
    parser.add_argument("--global-semantic", action="store_true")
    parser.add_argument("--semantic-fusion", default="local", choices=["local", "gated_sum", "concat"])
    parser.add_argument("--semantic-pred-horizon", type=int, default=0)
    parser.add_argument(
        "--semantic-noncausal",
        action="store_true",
        help="Allow bidirectional semantic summaries. Research-only; unsafe for autoregressive LM training.",
    )
    parser.add_argument(
        "--research-unsafe",
        action="store_true",
        help="Permit research-only unsafe options such as --semantic-noncausal.",
    )
    parser.add_argument(
        "--causal-state-stride",
        type=int,
        default=512,
        help="Prefix-causal update stride for V5/V6 world/self state loops. Larger values reduce small CUDA kernels.",
    )
    parser.add_argument("--semantic-residual-write", action="store_true")
    parser.add_argument("--semantic-write-scale", type=float, default=1.0)
    parser.add_argument("--semantic-memory-slots", type=int, default=0)
    parser.add_argument("--semantic-memory-write-scale", type=float, default=0.05)
    parser.add_argument("--semantic-state-write-scale", type=float, default=0.05)
    parser.add_argument("--semantic-gate-mixer", action="store_true")
    parser.add_argument("--semantic-gate-mixer-temperature", type=float, default=1.0)
    parser.add_argument("--semantic-gate-mixer-min-weight", type=float, default=0.0)
    parser.add_argument("--semantic-gate-mixer-max-clean-weight", type=float, default=0.0)
    parser.add_argument("--semantic-gate-mixer-max-state-weight", type=float, default=0.35)
    parser.add_argument(
        "--semantic-state-confidence-mode", default="learned", choices=["learned", "calibrated", "hybrid"]
    )
    parser.add_argument("--semantic-state-confidence-temperature", type=float, default=2.0)
    parser.add_argument("--semantic-state-confidence-gate", action="store_true")
    parser.add_argument("--semantic-memory-read-gate", action="store_true")
    parser.add_argument("--semantic-memory-hidden-scale", type=float, default=0.035)
    parser.add_argument("--layerwise-semantic-schedule", action="store_true")
    parser.add_argument("--world-state-slots", type=int, default=0)
    parser.add_argument("--world-state-diversity-margin", type=float, default=0.85)
    parser.add_argument("--world-state-stability-threshold", type=float, default=1e-3)
    parser.add_argument("--world-state-write-top-k", type=int, default=2)
    parser.add_argument("--no-world-router-normalize", action="store_true")
    parser.add_argument("--no-world-router-confidence-gate", action="store_true")
    parser.add_argument("--world-router-max-ratio", type=float, default=0.08)
    parser.add_argument("--world-router-mode", default="add", choices=["add", "modulate"])
    parser.add_argument("--world-router-modulation-scale", type=float, default=0.35)
    parser.add_argument("--self-state-slots", type=int, default=0)
    parser.add_argument("--self-state-recursion-depth", type=int, default=1)
    parser.add_argument("--self-state-write-scale", type=float, default=0.03)
    parser.add_argument("--self-state-hidden-scale", type=float, default=0.02)
    parser.add_argument("--self-state-boundary-temperature", type=float, default=1.0)
    parser.add_argument("--self-state-diversity-margin", type=float, default=0.85)
    parser.add_argument("--self-state-identity-scale", type=float, default=0.02)
    parser.add_argument("--self-state-context-score-scale", type=float, default=4.0)
    parser.add_argument("--self-state-hidden-scale-warmup-steps", type=int, default=0)
    parser.add_argument("--self-state-context-score-warmup-steps", type=int, default=0)
    parser.add_argument("--self-state-context-score-start", type=float, default=1.0)
    parser.add_argument("--stateful-boundary-tokens", type=int, default=64)
    parser.add_argument("--stateful-boundary-decay", type=float, default=0.97)
    parser.add_argument("--lambda-stateful-boundary", type=float, default=0.0)
    parser.add_argument("--lambda-stateful-full", type=float, default=0.0)
    parser.add_argument("--stateful-target-margin", type=float, default=0.0)
    parser.add_argument("--no-self-state-world-gate", action="store_true")
    parser.add_argument("--self-state-world-gate-min", type=float, default=0.10)
    parser.add_argument("--self-state-world-gate-scale", type=float, default=1.0)
    parser.add_argument("--latent-thought-steps", type=int, default=0)
    parser.add_argument(
        "--latent-thought-write-mode",
        default="state_only",
        choices=["state_only", "final_hidden"],
        help=(
            "Implicit latent thought write permission. state_only is the canonical "
            "continuous-state path; final_hidden is an experimental side-channel probe."
        ),
    )
    parser.add_argument("--latent-thought-hidden-scale", type=float, default=0.0)
    parser.add_argument("--state-evolution-steps", type=int, default=0)
    parser.add_argument("--no-state-evolution-memory", action="store_true")
    parser.add_argument(
        "--latent-field-coupling",
        action="store_true",
        help="Enable experimental token/slot field coupling side-channel. Disabled in canonical V6.5.",
    )
    parser.add_argument("--latent-field-token-scale", type=float, default=0.02)
    parser.add_argument("--latent-field-max-ratio", type=float, default=0.05)
    parser.add_argument("--v7-dynamics-steps", type=int, default=0)
    parser.add_argument("--v7-latent-slots", type=int, default=0)
    parser.add_argument("--v7-latent-write-scale", type=float, default=0.03)
    parser.add_argument("--v7-hidden-write-scale", type=float, default=0.01)
    parser.add_argument("--v7-max-hidden-write-ratio", type=float, default=0.05)
    parser.add_argument("--v7-state-write-scale", type=float, default=0.02)
    parser.add_argument("--v7-world-state-write-scale", type=float, default=-1.0)
    parser.add_argument("--v7-self-state-write-scale", type=float, default=-1.0)
    parser.add_argument("--v7-latent-timescale", type=float, default=1.0)
    parser.add_argument("--v7-world-timescale", type=float, default=1.0)
    parser.add_argument("--v7-self-timescale", type=float, default=1.0)
    parser.add_argument("--v7-controller-slots", type=int, default=1)
    parser.add_argument("--v7-controller-write-scale", type=float, default=0.02)
    parser.add_argument(
        "--v7-controller-mode",
        default="fixed",
        choices=["fixed"],
        help="Internal dynamics controller mode. fixed is the only implemented protocol-safe mode.",
    )
    parser.add_argument(
        "--v7-past-latent-adapt-steps",
        type=int,
        default=1,
        help="Suppress hidden reads from a carried V7 latent field for this many dynamics steps.",
    )
    parser.add_argument(
        "--v7-state-chunk-size",
        type=int,
        default=0,
        help="If >0, run V7 typed dynamics causally over sequence chunks so prior chunks can affect later chunks.",
    )
    parser.add_argument(
        "--v7-internal-latent-adapt-steps",
        type=int,
        default=0,
        help="Suppression steps for latent reads carried from an earlier chunk in the same forward pass.",
    )
    parser.add_argument("--v7-dynamic-depth", action="store_true")
    parser.add_argument("--v7-min-dynamics-steps", type=int, default=1)
    parser.add_argument(
        "--v7-max-dynamics-steps",
        type=int,
        default=0,
        help="Maximum V7 dynamics steps when dynamic depth is enabled. 0 reuses --v7-dynamics-steps.",
    )
    parser.add_argument("--v7-dynamic-convergence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--v7-homeostatic-control",
        action="store_true",
        help="Enable protocol-safe relative homeostatic rate modulation inside V7 typed dynamics.",
    )
    parser.add_argument("--v7-homeostatic-strength", type=float, default=0.25)
    parser.add_argument("--v7-homeostatic-min-scale", type=float, default=0.5)
    parser.add_argument("--v7-homeostatic-max-scale", type=float, default=1.5)
    parser.add_argument(
        "--v7-state-compatibility-gate",
        action="store_true",
        help="Gate carried V7 latent/controller state against the current segment before hidden reads.",
    )
    parser.add_argument("--v7-state-compatibility-strength", type=float, default=1.0)
    parser.add_argument("--v7-state-compatibility-min", type=float, default=0.0)
    parser.add_argument(
        "--v7-adaptive-tau",
        action="store_true",
        help="Enable learned bounded per-update tau modulation for V7 typed state updates.",
    )
    parser.add_argument("--v7-adaptive-tau-min", type=float, default=0.5)
    parser.add_argument("--v7-adaptive-tau-max", type=float, default=1.5)
    parser.add_argument("--no-v7-hyperspherical-state", action="store_true")
    parser.add_argument("--no-v7-causal-summary", action="store_true")
    parser.add_argument("--v7-causal-summary-decay", type=float, default=0.98)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-hidden-dim", type=int, default=512)
    parser.add_argument("--moe-dispatch-mode", default="sparse", choices=["auto", "dense", "sparse"])

    parser.add_argument("--lambda-load", type=float, default=0.01)
    parser.add_argument("--lambda-sparse", type=float, default=0.01)
    parser.add_argument("--lambda-kl", type=float, default=0.001)
    parser.add_argument("--kl-warmup-steps", type=int, default=0)
    parser.add_argument("--lambda-semantic-pred", type=float, default=0.0)
    parser.add_argument("--lambda-state-pred", type=float, default=0.0)
    parser.add_argument("--lambda-slot-diversity", type=float, default=0.0)
    parser.add_argument("--lambda-slot-stability", type=float, default=0.0)
    parser.add_argument("--lambda-self-pred", type=float, default=0.0)
    parser.add_argument("--lambda-self-slot-diversity", type=float, default=0.0)
    parser.add_argument(
        "--stateful-batch-ratio",
        type=float,
        default=0.0,
        help="Fraction of training micro-batches that should run continuation carry training.",
    )
    parser.add_argument(
        "--stateful-chunk-len",
        type=int,
        default=None,
        help="Chunk length for continuation training/eval. Defaults to seq_len // 2 when unset.",
    )
    parser.add_argument(
        "--lambda-stateful-carry",
        type=float,
        default=0.0,
        help="Weight for the continuation carry hinge loss that penalizes stateful chunk-2 loss being worse than fresh.",
    )
    parser.add_argument(
        "--stateful-carry-margin",
        type=float,
        default=0.0,
        help="Optional positive margin inside the continuation carry hinge loss.",
    )
    return parser.parse_args()


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    if args.semantic_noncausal and not args.research_unsafe:
        raise ValueError("--semantic-noncausal requires --research-unsafe because it is invalid for clean LM training")

    model_config = NAIMEStateMoEConfig(
        vocab_size=args.vocab_size or (50257 if args.data_format == "hf_disk" else 257),
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_dense_layers=args.n_dense_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attention_type=args.attention_type,
        mla_latent_dim=args.mla_latent_dim,
        mla_rope_per_head=args.mla_rope_per_head,
        stride=args.stride,
        window=args.window,
        z_dim=args.z_dim,
        target_sparsity=args.target_sparsity,
        gumbel_tau=args.gumbel_tau,
        gate_eval_mode=args.gate_eval_mode,
        logvar_clip=args.logvar_clip,
        semantic_scales=args.semantic_scales,
        mid_stride=args.mid_stride,
        mid_window=args.mid_window,
        use_global_semantic=args.global_semantic,
        semantic_fusion=args.semantic_fusion,
        semantic_pred_horizon=args.semantic_pred_horizon,
        semantic_causal=not args.semantic_noncausal,
        causal_state_stride=args.causal_state_stride,
        semantic_router_mode=args.semantic_router_mode,
        semantic_router_prior_scale=args.semantic_router_prior_scale,
        semantic_router_prior_clip=args.semantic_router_prior_clip,
        semantic_router_prior_gate=args.semantic_router_prior_gate,
        semantic_router_detach=args.semantic_router_detach,
        semantic_router_alpha_cap=args.semantic_router_alpha_cap,
        semantic_alpha_cap_mode=args.semantic_alpha_cap_mode,
        semantic_gate_downstream=args.semantic_gate_downstream,
        semantic_sparse_alpha=args.semantic_sparse_alpha,
        semantic_downstream_deterministic=args.semantic_downstream_deterministic,
        use_semantic_residual_write=args.semantic_residual_write,
        semantic_write_scale=args.semantic_write_scale,
        semantic_memory_slots=args.semantic_memory_slots,
        semantic_memory_write_scale=args.semantic_memory_write_scale,
        semantic_state_write_scale=args.semantic_state_write_scale,
        semantic_gate_mixer=args.semantic_gate_mixer,
        semantic_gate_mixer_temperature=args.semantic_gate_mixer_temperature,
        semantic_gate_mixer_min_weight=args.semantic_gate_mixer_min_weight,
        semantic_gate_mixer_max_clean_weight=args.semantic_gate_mixer_max_clean_weight,
        semantic_gate_mixer_max_state_weight=args.semantic_gate_mixer_max_state_weight,
        semantic_state_confidence_mode=args.semantic_state_confidence_mode,
        semantic_state_confidence_temperature=args.semantic_state_confidence_temperature,
        semantic_state_confidence_gate=args.semantic_state_confidence_gate,
        semantic_memory_read_gate=args.semantic_memory_read_gate,
        semantic_memory_hidden_scale=args.semantic_memory_hidden_scale,
        layerwise_semantic_schedule=args.layerwise_semantic_schedule,
        world_state_slots=args.world_state_slots,
        world_state_diversity_margin=args.world_state_diversity_margin,
        world_state_stability_threshold=args.world_state_stability_threshold,
        world_state_write_top_k=args.world_state_write_top_k,
        world_router_normalize=not args.no_world_router_normalize,
        world_router_confidence_gate=not args.no_world_router_confidence_gate,
        world_router_max_ratio=args.world_router_max_ratio,
        world_router_mode=args.world_router_mode,
        world_router_modulation_scale=args.world_router_modulation_scale,
        self_state_slots=args.self_state_slots,
        self_state_recursion_depth=args.self_state_recursion_depth,
        self_state_write_scale=args.self_state_write_scale,
        self_state_hidden_scale=args.self_state_hidden_scale,
        self_state_boundary_temperature=args.self_state_boundary_temperature,
        self_state_diversity_margin=args.self_state_diversity_margin,
        self_state_identity_scale=args.self_state_identity_scale,
        self_state_context_score_scale=args.self_state_context_score_scale,
        self_state_world_gate=not args.no_self_state_world_gate,
        self_state_world_gate_min=args.self_state_world_gate_min,
        self_state_world_gate_scale=args.self_state_world_gate_scale,
        latent_thought_steps=args.latent_thought_steps,
        latent_thought_write_mode=args.latent_thought_write_mode,
        latent_thought_hidden_scale=args.latent_thought_hidden_scale,
        state_evolution_steps=args.state_evolution_steps,
        state_evolution_memory=not args.no_state_evolution_memory,
        latent_field_coupling=args.latent_field_coupling,
        latent_field_token_scale=args.latent_field_token_scale,
        latent_field_max_ratio=args.latent_field_max_ratio,
        v7_dynamics_steps=args.v7_dynamics_steps,
        v7_latent_slots=args.v7_latent_slots,
        v7_latent_write_scale=args.v7_latent_write_scale,
        v7_hidden_write_scale=args.v7_hidden_write_scale,
        v7_max_hidden_write_ratio=args.v7_max_hidden_write_ratio,
        v7_state_write_scale=args.v7_state_write_scale,
        v7_world_state_write_scale=args.v7_world_state_write_scale,
        v7_self_state_write_scale=args.v7_self_state_write_scale,
        v7_latent_timescale=args.v7_latent_timescale,
        v7_world_timescale=args.v7_world_timescale,
        v7_self_timescale=args.v7_self_timescale,
        v7_controller_slots=args.v7_controller_slots,
        v7_controller_write_scale=args.v7_controller_write_scale,
        v7_controller_mode=args.v7_controller_mode,
        v7_past_latent_adapt_steps=args.v7_past_latent_adapt_steps,
        v7_state_chunk_size=args.v7_state_chunk_size,
        v7_internal_latent_adapt_steps=args.v7_internal_latent_adapt_steps,
        v7_dynamic_depth=args.v7_dynamic_depth,
        v7_min_dynamics_steps=args.v7_min_dynamics_steps,
        v7_max_dynamics_steps=args.v7_max_dynamics_steps,
        v7_dynamic_convergence_threshold=args.v7_dynamic_convergence_threshold,
        v7_homeostatic_control=args.v7_homeostatic_control,
        v7_homeostatic_strength=args.v7_homeostatic_strength,
        v7_homeostatic_min_scale=args.v7_homeostatic_min_scale,
        v7_homeostatic_max_scale=args.v7_homeostatic_max_scale,
        v7_state_compatibility_gate=args.v7_state_compatibility_gate,
        v7_state_compatibility_strength=args.v7_state_compatibility_strength,
        v7_state_compatibility_min=args.v7_state_compatibility_min,
        v7_adaptive_tau=args.v7_adaptive_tau,
        v7_adaptive_tau_min=args.v7_adaptive_tau_min,
        v7_adaptive_tau_max=args.v7_adaptive_tau_max,
        v7_hyperspherical_state=not args.no_v7_hyperspherical_state,
        v7_causal_summary=not args.no_v7_causal_summary,
        v7_causal_summary_decay=args.v7_causal_summary_decay,
        n_experts=args.n_experts,
        top_k=args.top_k,
        expert_hidden_dim=args.expert_hidden_dim,
        moe_dispatch_mode=args.moe_dispatch_mode,
        pad_token_id=0,
    )
    return TrainConfig(
        architecture=args.architecture,
        run_name=args.run_name,
        output_dir=args.output_dir,
        data_path=args.data_path,
        data_format=args.data_format,
        data_split=args.data_split,
        random_data=args.random_data,
        max_samples=args.max_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        auto_batch=args.auto_batch,
        vram_fraction=args.vram_fraction,
        auto_batch_max=args.auto_batch_max,
        target_tokens=args.target_tokens,
        target_tokens_mode=args.target_tokens_mode,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        latest_every=args.latest_every,
        latest_sync=bool(args.sync_latest and not args.async_latest),
        async_checkpoint=not args.no_async_checkpoint,
        async_checkpoint_queue=args.async_checkpoint_queue,
        metrics_flush_every=args.metrics_flush_every,
        metrics_fsync_every=args.metrics_fsync_every,
        best_checkpoint_mode=args.best_checkpoint_mode,
        eval_every=args.eval_every,
        eval_split=args.eval_split,
        eval_max_batches=args.eval_max_batches,
        eval_sampling=args.eval_sampling,
        eval_seed=args.eval_seed,
        eval_state_carry=args.eval_state_carry,
        eval_latent_thought_gain=args.eval_latent_thought_gain,
        eval_v7_dynamics_gain=args.eval_v7_dynamics_gain,
        eval_v7_state_swap=args.eval_v7_state_swap,
        eval_v7_state_erase=args.eval_v7_state_erase,
        eval_doc_continuity=args.eval_doc_continuity,
        eval_doc_continuity_docs=args.eval_doc_continuity_docs,
        eval_doc_continuity_chunks=args.eval_doc_continuity_chunks,
        diagnostics_mode=args.diagnostics_mode,
        diagnostics_every=args.diagnostics_every,
        diagnostics_output_dir=args.diagnostics_output_dir,
        diagnostics_chunk_len=args.diagnostics_chunk_len,
        diagnostics_boundary_tokens=args.diagnostics_boundary_tokens,
        diagnostics_max_batch=args.diagnostics_max_batch,
        diagnostics_record_tensor_stats=not args.diagnostics_no_tensor_stats,
        diagnostics_grad_components=not args.diagnostics_no_grad_components,
        diagnostics_window_size=args.diagnostics_window_size,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_min_evals=args.early_stop_min_evals,
        reference_metrics_path=args.reference_metrics_path,
        structural_stop=args.structural_stop,
        structural_stop_min_gap=args.structural_stop_min_gap,
        structural_stop_widen_delta=args.structural_stop_widen_delta,
        structural_stop_patience=args.structural_stop_patience,
        structural_stop_min_evals=args.structural_stop_min_evals,
        structural_stop_warmup_steps=args.structural_stop_warmup_steps,
        keep_last_n=args.keep_last_n,
        grad_accum_steps=args.grad_accum_steps,
        lm_loss_backend=args.lm_loss_backend,
        use_fused_state_attention=args.use_fused_state_attention,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        lr_cycle_length=args.lr_cycle_length,
        lr_restart_ratio=args.lr_restart_ratio,
        lr_restart_warmup=args.lr_restart_warmup,
        amp=not args.no_amp,
        compile_model=args.compile_model,
        compile_scope=args.compile_scope,
        compile_backend=args.compile_backend,
        disable_flash_sdp=args.disable_flash_sdp,
        device=args.device,
        resume=args.resume,
        resume_lr_policy=args.resume_lr_policy,
        resume_allow_failed=args.resume_allow_failed,
        allow_legacy_resume=args.allow_legacy_resume,
        stop_file=args.stop_file,
        stop_check_every=args.stop_check_every,
        lambda_load=args.lambda_load,
        lambda_sparse=args.lambda_sparse,
        lambda_kl=args.lambda_kl,
        kl_warmup_steps=args.kl_warmup_steps,
        lambda_semantic_pred=args.lambda_semantic_pred,
        lambda_state_pred=args.lambda_state_pred,
        lambda_slot_diversity=args.lambda_slot_diversity,
        lambda_slot_stability=args.lambda_slot_stability,
        lambda_self_pred=args.lambda_self_pred,
        lambda_self_slot_diversity=args.lambda_self_slot_diversity,
        stateful_batch_ratio=args.stateful_batch_ratio,
        stateful_chunk_len=args.stateful_chunk_len,
        lambda_stateful_carry=args.lambda_stateful_carry,
        stateful_carry_margin=args.stateful_carry_margin,
        stateful_boundary_tokens=args.stateful_boundary_tokens,
        stateful_boundary_decay=args.stateful_boundary_decay,
        lambda_stateful_boundary=args.lambda_stateful_boundary,
        lambda_stateful_full=args.lambda_stateful_full,
        stateful_target_margin=args.stateful_target_margin,
        self_state_hidden_scale_warmup_steps=args.self_state_hidden_scale_warmup_steps,
        self_state_context_score_warmup_steps=args.self_state_context_score_warmup_steps,
        self_state_context_score_start=args.self_state_context_score_start,
        model=model_config,
    )
