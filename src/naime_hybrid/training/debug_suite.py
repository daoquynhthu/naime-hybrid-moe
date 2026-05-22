"""Integrated component-level debug suite for NAIME training.

This module is deliberately more penetrative than a smoke training run.  It
checks each training subsystem in isolation first, then runs a tiny end-to-end
step only after data, masks, router dispatch, state modules, losses, eval,
checkpointing, and metric persistence have already been exercised.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import platform
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from naime_hybrid.config import NAIMEStateMoEConfig
from naime_hybrid.data import HFDiskCausalDataset, RandomTokenDataset
from naime_hybrid.models import build_model
from naime_hybrid.modules.moe import TopKMoE
from naime_hybrid.modules.self_state import RecursiveSelfState
from naime_hybrid.modules.world_state import WorldStateSlots

from .checkpoint import AsyncCheckpointWriter, load_checkpoint
from .checkpoint_policy import save_checkpoint_pair
from .config import TrainConfig
from .logging_utils import JsonlMetricLogger, metrics_jsonl_to_csv
from .losses import IGNORE_INDEX, collect_aux_losses, lm_loss
from .masks import prepare_attention_mask_for_device
from .runtime import probe_auto_batch_size, resolve_device, set_seed
from .scheduler import cosine_with_warmup
from .train import _assert_torch_compile_available, _compile_model
from .validation import evaluate_model


PhaseFn = Callable[[], dict[str, Any] | None]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "finite": bool(torch.isfinite(value.float()).all().item()) if value.is_floating_point() else True,
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    return str(value)


def _assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.is_floating_point() and not torch.isfinite(tensor.float()).all():
        raise AssertionError(f"{name} contains non-finite values")


def _assert_finite_mapping(prefix: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if torch.is_tensor(value):
            _assert_finite_tensor(f"{prefix}.{key}", value)


def _nonfinite_grad_report(model: torch.nn.Module, limit: int = 12) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None or not grad.is_floating_point():
            continue
        finite = torch.isfinite(grad.float())
        if bool(finite.all().item()):
            continue
        bad = ~finite
        report.append(
            {
                "name": name,
                "shape": list(grad.shape),
                "dtype": str(grad.dtype),
                "nonfinite": int(bad.sum().item()),
                "total": grad.numel(),
                "nan": int(torch.isnan(grad.float()).sum().item()),
                "inf": int(torch.isinf(grad.float()).sum().item()),
            }
        )
        if len(report) >= limit:
            break
    return report


def _setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("naime_hybrid.debug_suite")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(console_formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(run_dir / "debug_suite.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class ComponentDebugSuite:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path.cwd()
        self.run_dir = Path(args.run_dir) if args.run_dir else self.repo_root / "experiments" / "debug_suite" / _timestamp()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._clear_previous_artifacts()
        self.logger = _setup_logger(self.run_dir)
        self.results = JsonlMetricLogger(self.run_dir / "component_results.jsonl")
        self.metrics = JsonlMetricLogger(self.run_dir / "debug_metrics.jsonl")
        self.summary: dict[str, Any] = {
            "status": "running",
            "run_dir": str(self.run_dir),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "phases": [],
        }
        self.device = resolve_device(args.device)
        self.use_amp = bool(args.amp and self.device.type == "cuda")
        self.config = self._tiny_v6_config()

    def _clear_previous_artifacts(self) -> None:
        for name in [
            "component_results.jsonl",
            "component_results.csv",
            "debug_metrics.jsonl",
            "debug_metrics.csv",
            "summary.json",
            "resolved_model_config.json",
        ]:
            (self.run_dir / name).unlink(missing_ok=True)

    def _tiny_v6_config(self) -> NAIMEStateMoEConfig:
        return NAIMEStateMoEConfig(
            vocab_size=self.args.vocab_size,
            max_seq_len=self.args.seq_len,
            d_model=self.args.d_model,
            n_layers=3,
            n_dense_layers=1,
            n_heads=4,
            n_kv_heads=2,
            d_ff=self.args.d_model * 2,
            stride=4,
            window=8,
            z_dim=max(8, self.args.d_model // 4),
            target_sparsity=0.45,
            semantic_scales="local_mid_global",
            mid_stride=8,
            mid_window=16,
            use_global_semantic=True,
            semantic_fusion="concat",
            semantic_pred_horizon=1,
            semantic_causal=True,
            causal_state_stride=16,
            n_experts=4,
            top_k=2,
            expert_hidden_dim=self.args.d_model * 2,
            moe_dispatch_mode="auto",
            semantic_router_mode="hybrid",
            semantic_router_prior_scale=0.5,
            semantic_router_detach=True,
            semantic_router_alpha_cap=0.9,
            semantic_gate_downstream="clean_prob",
            semantic_sparse_alpha="downstream",
            semantic_downstream_deterministic=True,
            use_semantic_residual_write=True,
            semantic_write_scale=0.03,
            semantic_memory_slots=2,
            semantic_memory_write_scale=0.035,
            semantic_state_write_scale=0.045,
            semantic_gate_mixer=True,
            semantic_gate_mixer_temperature=1.6,
            semantic_gate_mixer_min_weight=0.08,
            semantic_gate_mixer_max_clean_weight=0.58,
            semantic_state_confidence_mode="hybrid",
            semantic_state_confidence_temperature=3.0,
            semantic_state_confidence_gate=True,
            semantic_memory_read_gate=True,
            layerwise_semantic_schedule=True,
            world_state_slots=4,
            world_state_write_top_k=2,
            self_state_slots=3,
            self_state_recursion_depth=2,
            self_state_world_gate=True,
            self_state_world_gate_min=0.1,
            attention_type="gqa",
            qk_norm=True,
        )

    def run(self) -> int:
        set_seed(self.args.seed)
        (self.run_dir / "resolved_model_config.json").write_text(
            json.dumps(asdict(self.config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        phases: list[tuple[str, PhaseFn]] = [
            ("environment", self.phase_environment),
            ("data_and_masks", self.phase_data_and_masks),
            ("moe_router_dispatch", self.phase_moe_router_dispatch),
            ("world_state_slots", self.phase_world_state_slots),
            ("recursive_self_state", self.phase_recursive_self_state),
            ("causal_model_contract", self.phase_causal_model_contract),
            ("loss_and_gradients", self.phase_loss_and_gradients),
            ("validation_eval", self.phase_validation_eval),
            ("checkpoint_roundtrip", self.phase_checkpoint_roundtrip),
            ("metrics_persistence", self.phase_metrics_persistence),
        ]
        if self.args.include_auto_batch:
            phases.insert(2, ("auto_batch_probe", self.phase_auto_batch_probe))
        if self.args.compile_smoke:
            phases.insert(-3, ("torch_compile_penetration", self.phase_torch_compile_penetration))

        failed = False
        for name, fn in phases:
            started = time.perf_counter()
            self.logger.info("phase start | %s", name)
            try:
                details = fn() or {}
                elapsed = time.perf_counter() - started
                record = {
                    "phase": name,
                    "status": "pass",
                    "elapsed_sec": round(elapsed, 4),
                    "details": _jsonable(details),
                }
                self.logger.info("phase pass  | %s | %.3fs", name, elapsed)
            except BaseException as exc:
                failed = True
                elapsed = time.perf_counter() - started
                record = {
                    "phase": name,
                    "status": "fail",
                    "elapsed_sec": round(elapsed, 4),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                self.logger.exception("phase fail  | %s | %.3fs", name, elapsed)
            self.results.write(record)
            self.summary["phases"].append(record)
            if failed and self.args.stop_on_first_failure:
                break

        csv_path = metrics_jsonl_to_csv(self.metrics.path, self.run_dir / "debug_metrics.csv")
        results_csv = metrics_jsonl_to_csv(self.results.path, self.run_dir / "component_results.csv")
        self.summary.update(
            {
                "status": "failed" if failed else "passed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "artifacts": {
                    "debug_suite_log": str(self.run_dir / "debug_suite.log"),
                    "component_results_jsonl": str(self.results.path),
                    "component_results_csv": str(results_csv) if results_csv else None,
                    "debug_metrics_jsonl": str(self.metrics.path),
                    "debug_metrics_csv": str(csv_path) if csv_path else None,
                    "summary_json": str(self.run_dir / "summary.json"),
                },
            }
        )
        (self.run_dir / "summary.json").write_text(
            json.dumps(_jsonable(self.summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.logger.info("debug suite %s | run_dir=%s", self.summary["status"], self.run_dir)
        return 1 if failed else 0

    def phase_environment(self) -> dict[str, Any]:
        disk = shutil.disk_usage(self.run_dir.anchor or ".")
        packages = {
            name: importlib.util.find_spec(name) is not None
            for name in ["torch", "datasets", "numpy", "pytest"]
        }
        cuda: dict[str, Any] = {"available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            free, total = torch.cuda.mem_get_info(device_index)
            cuda.update(
                {
                    "device": torch.cuda.get_device_name(device_index),
                    "capability": list(torch.cuda.get_device_capability(device_index)),
                    "total_vram_gib": total / 1024**3,
                    "free_vram_gib": free / 1024**3,
                    "property_total_vram_gib": props.total_memory / 1024**3,
                }
            )
        details = {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(self.device),
            "amp": self.use_amp,
            "packages": packages,
            "disk_free_gib": disk.free / 1024**3,
        }
        details["cuda"] = cuda
        if not packages["torch"]:
            raise RuntimeError("torch is not importable")
        return details

    def phase_data_and_masks(self) -> dict[str, Any]:
        random_dataset = RandomTokenDataset(
            vocab_size=self.config.vocab_size,
            seq_len=self.config.max_seq_len,
            num_samples=max(8, self.args.batch_size * 4),
            seed=self.args.seed,
        )
        sample = random_dataset[0]
        if sample["input_ids"].shape != (self.config.max_seq_len,):
            raise AssertionError("RandomTokenDataset returned wrong input shape")

        full_mask = torch.ones(self.args.batch_size, self.config.max_seq_len, dtype=torch.bool)
        prepared_full, infer_full = prepare_attention_mask_for_device(full_mask, self.device)
        if prepared_full is not None or infer_full:
            raise AssertionError("all-true attention mask should be dropped for causal fast path")

        collated = HFDiskCausalDataset.causal_collate(
            [
                {"input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)},
                {"input_ids": torch.tensor([6, 7, 8], dtype=torch.long)},
            ],
            seq_len=4,
            pad_token_id=self.config.pad_token_id,
        )
        if collated["input_ids"].shape[1] != 2:
            raise AssertionError("HF causal collate should crop to the shortest safe causal length")

        manual_partial = torch.tensor(
            [[True, True, True, False], [True, True, False, False]],
            dtype=torch.bool,
        )
        partial_mask, infer_partial = prepare_attention_mask_for_device(manual_partial, self.device)
        if partial_mask is None or not infer_partial:
            raise AssertionError("partial padding mask must be preserved")

        loader = DataLoader(random_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        return {
            "random_len": len(random_dataset),
            "batch_input_shape": list(batch["input_ids"].shape),
            "full_mask_dropped": prepared_full is None and not infer_full,
            "hf_collate_shape": list(collated["input_ids"].shape),
            "partial_mask_shape": list(partial_mask.shape),
            "manual_partial_valid_tokens": int(manual_partial.sum().item()),
        }

    def phase_auto_batch_probe(self) -> dict[str, Any]:
        selected = probe_auto_batch_size(
            architecture=self.args.architecture,
            model_config=self.config,
            requested_batch=max(1, min(self.args.batch_size, self.args.auto_batch_max)),
            max_batch=max(1, self.args.auto_batch_max),
            vram_fraction=min(0.5, self.args.vram_fraction),
            use_amp=self.use_amp,
            device=self.device,
            logger=self.logger,
        )
        return {"selected_batch_size": selected}

    def phase_moe_router_dispatch(self) -> dict[str, Any]:
        torch.manual_seed(self.args.seed)
        dense = TopKMoE(16, 16, 4, 2, 32, use_semantic_router=False, dispatch_mode="dense")
        sparse = TopKMoE(16, 16, 4, 2, 32, use_semantic_router=False, dispatch_mode="sparse")
        auto = TopKMoE(16, 16, 4, 2, 32, use_semantic_router=False, dispatch_mode="auto")
        sparse.load_state_dict(dense.state_dict())
        auto.load_state_dict(dense.state_dict())
        x_dense = torch.randn(2, 128, 16, requires_grad=True)
        x_sparse = x_dense.detach().clone().requires_grad_()
        x_auto = x_dense.detach().clone().requires_grad_()

        y_dense, aux_dense = dense(x_dense)
        y_sparse, aux_sparse = sparse(x_sparse)
        y_auto, aux_auto = auto(x_auto)
        if not torch.allclose(y_sparse, y_dense, atol=1e-6):
            raise AssertionError("sparse MoE dispatch does not match dense dispatch")
        if not torch.allclose(y_auto, y_dense, atol=1e-6):
            raise AssertionError("auto MoE dispatch does not match dense dispatch in dense heuristic case")

        y_dense.float().pow(2).mean().backward()
        y_sparse.float().pow(2).mean().backward()
        y_auto.float().pow(2).mean().backward()
        _assert_finite_tensor("moe_sparse_grad", x_sparse.grad)
        _assert_finite_tensor("moe_auto_grad", x_auto.grad)
        return {
            "auto_dispatch": auto._resolve_dispatch_mode(x_auto.detach()),
            "router_entropy": aux_dense["router_entropy"],
            "sparse_dispatch_dense_flag": aux_sparse["dispatch_dense"],
            "auto_dispatch_dense_flag": aux_auto["dispatch_dense"],
        }

    def phase_world_state_slots(self) -> dict[str, Any]:
        module = WorldStateSlots(d_model=32, slots=4)
        hidden = torch.randn(2, 16, 32, requires_grad=True)
        semantic = torch.randn(2, 16, 32, requires_grad=True)
        trace = torch.randn(2, 4, 4, 32, requires_grad=True)
        context, weights, confidence, traced_state, metrics = module.read_update_sequence(
            hidden,
            semantic,
            trace,
            stride=4,
        )
        loss = context.float().square().mean() + traced_state.float().square().mean()
        loss = loss + metrics["history_read_entropy"].float() + metrics["state_velocity"].float()
        loss.backward()
        _assert_finite_tensor("world_context", context)
        _assert_finite_tensor("world_trace_grad", trace.grad)
        _assert_finite_mapping("world_metrics", metrics)

        bf16_checked = False
        if self.device.type == "cuda":
            module_cuda = WorldStateSlots(d_model=32, slots=4).to(self.device)
            hidden_cuda = torch.randn(2, 16, 32, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            semantic_cuda = torch.randn(2, 16, 32, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            trace_cuda = torch.randn(2, 4, 4, 32, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                out_cuda = module_cuda.read_update_sequence(hidden_cuda, semantic_cuda, trace_cuda, stride=4)
            context_cuda, _, _, traced_cuda, metrics_cuda = out_cuda
            cuda_loss = context_cuda.float().square().mean() + traced_cuda.float().square().mean()
            cuda_loss = cuda_loss + metrics_cuda["history_read_entropy"].float()
            cuda_loss.backward()
            _assert_finite_tensor("world_bf16_context", context_cuda)
            _assert_finite_tensor("world_bf16_trace_grad", trace_cuda.grad)
            bf16_checked = True

        return {
            "context_shape": list(context.shape),
            "weights_shape": list(weights.shape),
            "confidence_mean": confidence.float().mean(),
            "history_read_entropy": metrics["history_read_entropy"],
            "bf16_cuda_backward_checked": bf16_checked,
        }

    def phase_recursive_self_state(self) -> dict[str, Any]:
        module = RecursiveSelfState(d_model=32, slots=4, recursion_depth=2)
        hidden = torch.randn(2, 16, 32, requires_grad=True)
        world_trace = torch.randn(2, 4, 4, 32, requires_grad=True)
        self_trace = torch.randn(2, 4, 4, 32, requires_grad=True)
        mask = torch.ones(2, 16, dtype=torch.bool)
        output, next_state, metrics = module(
            hidden,
            attention_mask=mask,
            world_state=world_trace,
            self_state=self_trace,
            causal_safe=True,
            block_size=4,
        )
        loss = output.float().square().mean() + next_state.float().square().mean()
        loss = loss + metrics["self_pred"].float() + metrics["slot_context_cosine"].float() * 0.0
        loss.backward()
        _assert_finite_tensor("self_output", output)
        _assert_finite_tensor("self_trace_grad", self_trace.grad)
        _assert_finite_mapping("self_metrics", metrics)

        bf16_checked = False
        if self.device.type == "cuda":
            module_cuda = RecursiveSelfState(d_model=32, slots=4, recursion_depth=2).to(self.device)
            hidden_cuda = torch.randn(2, 16, 32, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            world_cuda = torch.randn(2, 4, 4, 32, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            self_cuda = torch.randn(2, 4, 4, 32, device=self.device, dtype=torch.bfloat16, requires_grad=True)
            mask_cuda = torch.ones(2, 16, device=self.device, dtype=torch.bool)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                out_cuda, state_cuda, metrics_cuda = module_cuda(
                    hidden_cuda,
                    attention_mask=mask_cuda,
                    world_state=world_cuda,
                    self_state=self_cuda,
                    causal_safe=True,
                    block_size=4,
                )
            cuda_loss = out_cuda.float().square().mean() + state_cuda.float().square().mean()
            cuda_loss = cuda_loss + metrics_cuda["self_pred"].float()
            cuda_loss.backward()
            _assert_finite_tensor("self_bf16_output", out_cuda)
            _assert_finite_tensor("self_bf16_self_grad", self_cuda.grad)
            bf16_checked = True

        return {
            "output_shape": list(output.shape),
            "next_state_shape": list(next_state.shape),
            "reflection_norm": metrics["reflection_norm"],
            "world_residual_ratio": metrics["world_residual_ratio"],
            "bf16_cuda_backward_checked": bf16_checked,
        }

    def phase_causal_model_contract(self) -> dict[str, Any]:
        torch.manual_seed(self.args.seed)
        model = build_model(self.args.architecture, self.config).to(self.device)
        model.eval()
        input_ids = torch.randint(
            1,
            self.config.vocab_size,
            (self.args.batch_size, min(31, self.config.max_seq_len)),
            device=self.device,
        )
        changed = input_ids.clone()
        cutoff = min(8, input_ids.size(1) - 1)
        changed[:, cutoff:] = torch.randint(1, self.config.vocab_size, changed[:, cutoff:].shape, device=self.device)
        with torch.no_grad():
            original = model(input_ids)["logits"]
            perturbed = model(changed)["logits"]
        if not torch.allclose(original[:, :cutoff, :], perturbed[:, :cutoff, :], atol=1e-5, rtol=1e-5):
            max_diff = (original[:, :cutoff, :] - perturbed[:, :cutoff, :]).abs().max().item()
            raise AssertionError(f"future token perturbation leaked into prefix logits: max_diff={max_diff:.6g}")

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        prepared, infer_pad = prepare_attention_mask_for_device(attention_mask, self.device)
        with torch.no_grad():
            masked = model(input_ids, attention_mask=attention_mask, return_aux=False)["logits"]
            fast = model(input_ids, attention_mask=prepared, infer_pad_mask=infer_pad, return_aux=False)["logits"]
        if not torch.allclose(masked, fast, atol=1e-5, rtol=1e-5):
            raise AssertionError("full attention mask path differs from causal fast path")
        return {"prefix_cutoff": cutoff, "mask_fast_path": prepared is None and not infer_pad}

    def _make_loader(self, *, batches: int = 4) -> DataLoader:
        dataset = RandomTokenDataset(
            vocab_size=self.config.vocab_size,
            seq_len=self.config.max_seq_len,
            num_samples=max(self.args.batch_size * batches, self.args.batch_size),
            seed=self.args.seed + 17,
        )
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0)

    def _build_train_objects(self) -> tuple[torch.nn.Module, torch.optim.Optimizer, Any, Any]:
        model = build_model(self.args.architecture, self.config).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        scheduler = cosine_with_warmup(
            optimizer,
            warmup_steps=1,
            max_steps=max(2, self.args.steps + 1),
            min_lr_ratio=0.1,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        return model, optimizer, scheduler, scaler

    def _training_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        batch: dict[str, torch.Tensor],
        step: int,
    ) -> dict[str, float]:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        attention_mask, infer_pad = prepare_attention_mask_for_device(batch.get("attention_mask"), self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            out = model(input_ids, attention_mask=attention_mask, infer_pad_mask=infer_pad)
            base_loss = lm_loss(out["logits"], labels)
            aux = collect_aux_losses(
                out.get("aux", []),
                self.config.target_sparsity,
                sparse_alpha=self.config.semantic_sparse_alpha,
                alpha_cap=self.config.semantic_router_alpha_cap,
            )
            total_loss = (
                base_loss
                + 0.01 * aux["load"]
                + 0.01 * aux["sparse"]
                + 0.001 * aux["kl"]
                + 0.01 * aux["semantic_pred"]
                + 0.01 * aux["v5_state_pred"]
                + 0.01 * aux["v5_slot_diversity"]
                + 0.01 * aux["v5_slot_stability"]
                + 0.01 * aux["v6_self_pred"]
                + 0.01 * aux["v6_slot_diversity"]
            )
        if not torch.isfinite(total_loss.detach().float()):
            raise AssertionError("total_loss is non-finite before backward")
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm.detach().float()):
            report = _nonfinite_grad_report(model)
            self.metrics.write(
                {
                    "kind": "nonfinite_gradient",
                    "step": step,
                    "grad_norm": _jsonable(grad_norm),
                    "bad_parameters": report,
                }
            )
            raise AssertionError(f"grad_norm is non-finite; bad_parameters={report[:3]}")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        metrics = {
            "kind": "train_debug_step",
            "step": step,
            "lm_loss": float(base_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "lr": float(scheduler.get_last_lr()[0]),
            "alpha": float(aux["alpha_mean"].detach().cpu()),
            "router_entropy": float(aux["router_entropy"].detach().cpu()),
            "dispatch_dense": float(aux["dispatch_dense"].detach().cpu()),
            "v5_state_pred": float(aux["v5_state_pred"].detach().cpu()),
            "v6_self_pred": float(aux["v6_self_pred"].detach().cpu()),
            "v6_reflection_norm": float(aux["v6_reflection_norm"].detach().cpu()),
        }
        self.metrics.write(metrics)
        return metrics

    def phase_loss_and_gradients(self) -> dict[str, Any]:
        model, optimizer, scheduler, scaler = self._build_train_objects()
        loader = self._make_loader(batches=max(2, self.args.steps))
        last_metrics = {}
        for step, batch in enumerate(loader, start=1):
            if step > self.args.steps:
                break
            last_metrics = self._training_step(model, optimizer, scheduler, scaler, batch, step)
        return last_metrics

    def phase_validation_eval(self) -> dict[str, Any]:
        model, _, _, _ = self._build_train_objects()
        loader = self._make_loader(batches=2)
        result = evaluate_model(
            model,
            loader,
            self.config,
            self.device,
            self.use_amp,
            max_batches=1,
            lambda_load=0.01,
            lambda_sparse=0.01,
            lambda_kl=0.001,
            lambda_semantic_pred=0.01,
            lambda_state_pred=0.01,
            lambda_slot_diversity=0.01,
            lambda_slot_stability=0.01,
            lambda_self_pred=0.01,
            lambda_self_slot_diversity=0.01,
        )
        if not math.isfinite(float(result["val_total_loss"])):
            raise AssertionError("validation total loss is non-finite")
        self.metrics.write({"kind": "validation_debug", **result})
        return result

    def phase_torch_compile_penetration(self) -> dict[str, Any]:
        _assert_torch_compile_available()
        eager_model = build_model(self.args.architecture, self.config).to(self.device).eval()
        compiled_model = build_model(self.args.architecture, self.config).to(self.device)
        compiled_model.load_state_dict(eager_model.state_dict())

        compile_config = TrainConfig(
            architecture=self.args.architecture,
            compile_model=True,
            compile_scope=self.args.compile_scope,
            compile_backend=self.args.compile_backend,
            model=self.config,
        )
        compiled_model = _compile_model(compiled_model, compile_config, self.logger)
        compiled_model.eval()

        input_ids = torch.randint(
            1,
            self.config.vocab_size,
            (self.args.batch_size, min(24, self.config.max_seq_len)),
            device=self.device,
        )
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                eager_logits = eager_model(input_ids, return_aux=False)["logits"]
                compiled_logits = compiled_model(input_ids, return_aux=False)["logits"]
        _assert_finite_tensor("compiled_logits", compiled_logits)
        max_diff = (eager_logits.float() - compiled_logits.float()).abs().max().item()
        if max_diff > self.args.compile_atol:
            raise AssertionError(
                f"compiled forward drift exceeds tolerance: max_diff={max_diff:.6g} "
                f"tolerance={self.args.compile_atol:.6g}"
            )

        optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=self.args.learning_rate)
        scheduler = cosine_with_warmup(optimizer, warmup_steps=1, max_steps=2, min_lr_ratio=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        batch = next(iter(self._make_loader(batches=1)))
        compiled_model.train()
        metrics = self._training_step(compiled_model, optimizer, scheduler, scaler, batch, 9001)

        from .checkpoint import build_model_payload

        payload = build_model_payload(compiled_model, step=9001, config=compile_config.to_dict(), metrics=metrics)
        bad_keys = [key for key in payload["model"] if "_orig_mod" in key]
        if bad_keys:
            raise AssertionError(f"compiled checkpoint payload contains _orig_mod keys, first={bad_keys[0]}")

        return {
            "backend": self.args.compile_backend,
            "scope": self.args.compile_scope,
            "max_forward_diff": max_diff,
            "compiled_backward_lm": metrics["lm_loss"],
            "compiled_checkpoint_key_count": len(payload["model"]),
        }

    def phase_checkpoint_roundtrip(self) -> dict[str, Any]:
        model, optimizer, scheduler, scaler = self._build_train_objects()
        loader = self._make_loader(batches=2)
        first_batch = next(iter(loader))
        metrics = self._training_step(model, optimizer, scheduler, scaler, first_batch, 1001)
        train_config = TrainConfig(
            architecture=self.args.architecture,
            run_name="debug_suite",
            output_dir=str(self.run_dir),
            random_data=True,
            batch_size=self.args.batch_size,
            max_steps=self.args.steps,
            learning_rate=self.args.learning_rate,
            amp=self.use_amp,
            model=self.config,
        )
        checkpoint_dir = self.run_dir / "checkpoints"
        model_dir = self.run_dir / "models"
        writer = AsyncCheckpointWriter(max_queue=2)
        try:
            save_checkpoint_pair(
                writer,
                checkpoint_dir / "latest.pt",
                model_dir / "model_latest.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                step=1001,
                config=train_config,
                metrics=metrics,
            )
        finally:
            writer.close()

        restored_model, restored_optimizer, restored_scheduler, restored_scaler = self._build_train_objects()
        restored_step = load_checkpoint(
            checkpoint_dir / "latest.pt",
            restored_model,
            restored_optimizer,
            restored_scheduler,
            restored_scaler,
            strict=True,
        )
        if restored_step != 1001:
            raise AssertionError(f"restored step mismatch: {restored_step}")
        second_batch = next(iter(loader))
        resume_metrics = self._training_step(
            restored_model,
            restored_optimizer,
            restored_scheduler,
            restored_scaler,
            second_batch,
            1002,
        )
        return {
            "checkpoint": checkpoint_dir / "latest.pt",
            "model_only": model_dir / "model_latest.pt",
            "restored_step": restored_step,
            "resume_lm_loss": resume_metrics["lm_loss"],
        }

    def phase_metrics_persistence(self) -> dict[str, Any]:
        csv_path = metrics_jsonl_to_csv(self.metrics.path, self.run_dir / "debug_metrics.csv")
        results_csv = metrics_jsonl_to_csv(self.results.path, self.run_dir / "component_results.csv")
        if csv_path is None or not csv_path.exists():
            raise AssertionError("debug_metrics.csv was not created")
        if results_csv is None or not results_csv.exists():
            raise AssertionError("component_results.csv was not created")
        return {
            "debug_metrics_jsonl_size": self.metrics.path.stat().st_size,
            "debug_metrics_csv": csv_path,
            "component_results_csv": results_csv,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run integrated component-level NAIME debug suite.")
    parser.add_argument("--run-dir", default=None, help="Directory for persisted debug logs and artifacts.")
    parser.add_argument("--architecture", default="naime_v6_recursive_self_moe")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--steps", type=int, default=2, help="Tiny end-to-end train steps after component tests.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-auto-batch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-batch-max", type=int, default=4)
    parser.add_argument("--vram-fraction", type=float, default=0.5)
    parser.add_argument("--compile-smoke", action="store_true")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-scope", choices=["full", "dense"], default="dense")
    parser.add_argument("--compile-atol", type=float, default=3e-2)
    parser.add_argument("--stop-on-first-failure", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite = ComponentDebugSuite(args)
    return suite.run()


if __name__ == "__main__":
    raise SystemExit(main())
