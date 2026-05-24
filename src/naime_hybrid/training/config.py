from dataclasses import asdict, dataclass, field

from naime_hybrid.config import NAIMEStateMoEConfig


@dataclass
class TrainConfig:
    architecture: str = "naime_state_moe"
    run_name: str = "debug"
    output_dir: str = "experiments/runs"

    data_path: str | None = None
    data_format: str = "auto"
    data_split: str = "train"
    random_data: bool = False
    max_samples: int | None = None
    seed: int = 1234

    batch_size: int = 4
    auto_batch: bool = False
    vram_fraction: float = 0.9
    auto_batch_max: int = 128
    target_tokens: int | None = None
    target_tokens_mode: str = "total"
    num_workers: int = 0
    max_steps: int = 1000
    log_every: int = 10
    save_every: int = 2000
    latest_every: int = 1000
    latest_sync: bool = False
    async_checkpoint: bool = True
    async_checkpoint_queue: int = 2
    metrics_flush_every: int = 50
    metrics_fsync_every: int = 1000
    best_checkpoint_mode: str = "model"
    eval_every: int = 0
    eval_split: str = "validation"
    eval_max_batches: int = 10
    eval_sampling: str = "random"
    eval_seed: int = 4321
    eval_state_carry: bool = False
    eval_latent_thought_gain: bool = False
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    early_stop_min_evals: int = 0
    reference_metrics_path: str | None = None
    structural_stop: bool = False
    structural_stop_min_gap: float = 0.30
    structural_stop_widen_delta: float = 0.05
    structural_stop_patience: int = 2
    structural_stop_min_evals: int = 3
    structural_stop_warmup_steps: int = 1000
    keep_last_n: int = 2
    grad_accum_steps: int = 1
    lm_loss_backend: str = "auto"

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    lr_cycle_length: int = 0
    lr_restart_ratio: float = 0.5
    lr_restart_warmup: int = 200

    amp: bool = True
    compile_model: bool = False
    compile_scope: str = "full"
    compile_backend: str = "inductor"
    disable_flash_sdp: bool = False
    device: str = "auto"
    resume: str = "none"
    resume_lr_policy: str = "checkpoint"
    resume_allow_failed: bool = False
    allow_legacy_resume: bool = False
    causal_integrity_version: int = 2
    strict_resume: bool = True
    stop_file: str | None = None
    stop_check_every: int = 1

    lambda_load: float = 0.01
    lambda_sparse: float = 0.01
    lambda_kl: float = 0.001
    kl_warmup_steps: int = 0
    lambda_semantic_pred: float = 0.0
    lambda_state_pred: float = 0.0
    lambda_slot_diversity: float = 0.0
    lambda_slot_stability: float = 0.0
    lambda_self_pred: float = 0.0
    lambda_self_slot_diversity: float = 0.0

    model: NAIMEStateMoEConfig = field(default_factory=NAIMEStateMoEConfig)

    def to_dict(self) -> dict:
        data = asdict(self)
        return data
