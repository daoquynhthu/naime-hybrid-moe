import argparse

from naime_hybrid.config import NAIMEStateMoEConfig

from .config import TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NAIME Hybrid architectures.")
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
        help="Do not wait for latest.pt to finish writing. Faster, but a hard crash may lose the newest alias.",
    )
    parser.add_argument("--no-async-checkpoint", action="store_true")
    parser.add_argument("--async-checkpoint-queue", type=int, default=2)
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
    parser.add_argument("--resume", default="auto")
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
    parser.add_argument("--self-state-slots", type=int, default=0)
    parser.add_argument("--self-state-recursion-depth", type=int, default=1)
    parser.add_argument("--self-state-write-scale", type=float, default=0.03)
    parser.add_argument("--self-state-hidden-scale", type=float, default=0.02)
    parser.add_argument("--self-state-boundary-temperature", type=float, default=1.0)
    parser.add_argument("--self-state-diversity-margin", type=float, default=0.85)
    parser.add_argument("--self-state-identity-scale", type=float, default=0.02)
    parser.add_argument("--self-state-context-score-scale", type=float, default=4.0)
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
    return parser.parse_args()


def build_train_config(args: argparse.Namespace) -> TrainConfig:
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
        self_state_slots=args.self_state_slots,
        self_state_recursion_depth=args.self_state_recursion_depth,
        self_state_write_scale=args.self_state_write_scale,
        self_state_hidden_scale=args.self_state_hidden_scale,
        self_state_boundary_temperature=args.self_state_boundary_temperature,
        self_state_diversity_margin=args.self_state_diversity_margin,
        self_state_identity_scale=args.self_state_identity_scale,
        self_state_context_score_scale=args.self_state_context_score_scale,
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
        latest_sync=not args.async_latest,
        async_checkpoint=not args.no_async_checkpoint,
        async_checkpoint_queue=args.async_checkpoint_queue,
        best_checkpoint_mode=args.best_checkpoint_mode,
        eval_every=args.eval_every,
        eval_split=args.eval_split,
        eval_max_batches=args.eval_max_batches,
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
        device=args.device,
        resume=args.resume,
        resume_lr_policy=args.resume_lr_policy,
        resume_allow_failed=args.resume_allow_failed,
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
        model=model_config,
    )
