"""Remote training/data probe for NAIME runs.

This script is intentionally dependency-light so it can run inside the remote
training venv without importing the full training stack. It inspects datasets,
runs, metrics, logs, checkpoints, GPU state, processes, and disk pressure, then
emits one JSON report for local tooling to archive and summarize.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "loss_lm",
    "loss_total",
    "ppl_lm",
    "grad_norm",
    "lr",
    "tokens_per_second",
    "alpha_mean",
    "router_entropy",
    "bad_grad_window_count",
    "v5_router_world_ratio",
    "v5_router_world_gate",
    "v5_state_velocity",
    "v5_state_acceleration",
    "v6_boundary_self",
    "v6_boundary_world",
    "v6_state_delta",
    "v6_latent_thought_delta",
    "v6_latent_thought_write_norm",
    "v7_dynamics_gain_lm",
    "v7_state_swap_delta_lm",
    "v7_state_erase_delta_lm",
    "v7_latent_write_norm",
    "v7_hidden_write_norm",
)


def _json_load(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def _tail_lines(path: Path, n: int) -> list[str]:
    if n <= 0 or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return list(deque(f, maxlen=n))
    except Exception:
        return []


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    return sum(values) / len(values)


def _summarize_numeric(records: list[dict[str, Any]], keys: tuple[str, ...] = METRIC_KEYS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        vals = [_safe_float(record.get(key)) for record in records if key in record]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        out[key] = {
            "latest": vals[-1],
            "avg": _mean(vals),
            "min": min(vals),
            "max": max(vals),
            "delta": vals[-1] - vals[0] if len(vals) >= 2 else 0.0,
        }
    return out


def _parse_metrics(path: Path, recent_window: int, scan_all: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "recent_train": [],
        "recent_val": [],
        "summary": {},
        "latest_train": None,
        "latest_val": None,
        "best_val": None,
        "line_count": None,
    }
    if not path.exists():
        return result

    recent_records: deque[dict[str, Any]] = deque(maxlen=max(1, recent_window))
    best_val: dict[str, Any] | None = None
    latest_train: dict[str, Any] | None = None
    latest_val: dict[str, Any] | None = None
    line_count = 0

    if scan_all:
        source_iter = path.open("r", encoding="utf-8", errors="replace")
    else:
        source_iter = _tail_lines(path, recent_window * 4)

    try:
        for line in source_iter:
            line_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_type = record.get("record_type", "train")
            if record_type == "train":
                latest_train = record
            elif record_type in {"validation", "val"}:
                latest_val = record
                val_loss = _safe_float(record.get("val_lm_loss", record.get("loss_lm", record.get("loss"))))
                best_loss = None
                if best_val is not None:
                    best_loss = _safe_float(best_val.get("val_lm_loss", best_val.get("loss_lm", best_val.get("loss"))))
                if val_loss is not None and (best_loss is None or val_loss < best_loss):
                    best_val = record
            recent_records.append(record)
    finally:
        if scan_all and hasattr(source_iter, "close"):
            source_iter.close()

    recent_train = [r for r in recent_records if r.get("record_type", "train") == "train"]
    recent_val = [r for r in recent_records if r.get("record_type") in {"validation", "val"}]
    result.update(
        {
            "recent_train": recent_train[-recent_window:],
            "recent_val": recent_val[-max(1, min(recent_window, 20)) :],
            "summary": _summarize_numeric(recent_train),
            "latest_train": latest_train,
            "latest_val": latest_val,
            "best_val": best_val,
            "line_count": line_count if scan_all else None,
        }
    )
    return result


def _dataset_split_info(split_dir: Path) -> dict[str, Any]:
    info = _json_load(split_dir / "dataset_info.json") or {}
    split_info = {}
    try:
        split_info = next(iter(info.get("splits", {}).values()))
    except Exception:
        split_info = {}
    feature_types = {}
    for name, spec in (info.get("features") or {}).items():
        feature_types[name] = spec.get("_type") if isinstance(spec, dict) else str(spec)
    return {
        "path": str(split_dir),
        "exists": split_dir.exists(),
        "num_examples": split_info.get("num_examples"),
        "num_bytes": split_info.get("num_bytes"),
        "size_in_bytes": info.get("size_in_bytes"),
        "shard_count": len(split_info.get("shard_lengths") or []),
        "shard_lengths": split_info.get("shard_lengths"),
        "features": feature_types,
    }


def _dataset_report(dataset_path: Path, seq_len: int | None, auto_batch_max: int) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(dataset_path), "exists": dataset_path.exists(), "splits": {}}
    if not dataset_path.exists():
        return report
    for split in ("train", "validation", "test"):
        split_dir = dataset_path / split
        if split_dir.exists():
            report["splits"][split] = _dataset_split_info(split_dir)
    train_examples = report["splits"].get("train", {}).get("num_examples")
    if train_examples and seq_len:
        safe_examples = max(1, int(train_examples) - max(0, auto_batch_max - 1))
        report["token_estimate"] = int(train_examples) * int(seq_len)
        report["safe_no_repeat_target_tokens"] = safe_examples * int(seq_len)
        report["safe_no_repeat_note"] = (
            "safe target subtracts auto_batch_max-1 sequences so ceil(target/(batch*seq_len)) "
            "does not exceed one shuffled epoch for any batch <= auto_batch_max"
        )
    return report


def _latest_run(runs_root: Path) -> str | None:
    candidates = []
    if not runs_root.exists():
        return None
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        if (path / "metrics.jsonl").exists() or (path / "train.log").exists():
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).name


def _checkpoint_report(run_dir: Path) -> list[dict[str, Any]]:
    files = []
    for root in (run_dir, run_dir / "models"):
        if not root.exists():
            continue
        for path in root.glob("*.pt"):
            files.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "gb": round(path.stat().st_size / (1024**3), 3),
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
    return sorted(files, key=lambda item: item["gb"], reverse=True)


def _disk_report(path: Path) -> dict[str, Any]:
    target = path if path.exists() else Path(path.anchor or ".")
    usage = shutil.disk_usage(target)
    return {
        "path": str(target),
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
    }


def _run_command(args: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except Exception as exc:
        return 1, str(exc)


def _gpu_samples(sample_count: int, interval: float) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    fields = [
        "index",
        "name",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "temperature.gpu",
        "power.draw",
    ]
    query = "--query-gpu=" + ",".join(fields)
    for i in range(max(1, sample_count)):
        code, text = _run_command(["nvidia-smi", query, "--format=csv,noheader,nounits"], timeout=10)
        rows = []
        if code == 0:
            for line in text.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= len(fields):
                    rows.append(dict(zip(fields, parts, strict=False)))
        samples.append({"sample": i + 1, "timestamp": datetime.now().isoformat(timespec="seconds"), "gpus": rows})
        if i + 1 < sample_count:
            time.sleep(max(0.1, interval))
    return samples


def _process_report() -> str:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match 'naime_hybrid|python.*train|train_model|train_template' } | "
        "Select-Object ProcessId,Name,CommandLine | Format-Table -AutoSize | Out-String -Width 240"
    )
    _, text = _run_command(["powershell", "-NoProfile", "-Command", command], timeout=20)
    return text


def _run_report(run_dir: Path, recent_window: int, tail_log: int, scan_all_metrics: bool) -> dict[str, Any]:
    config = _json_load(run_dir / "config.json") or {}
    metrics = _parse_metrics(run_dir / "metrics.jsonl", recent_window, scan_all_metrics)
    latest_train = metrics.get("latest_train") or {}
    max_steps = config.get("max_steps")
    step = latest_train.get("step")
    progress = None
    if isinstance(step, int) and isinstance(max_steps, int) and max_steps > 0:
        progress = {
            "step": step,
            "max_steps": max_steps,
            "percent": round(step / max_steps * 100.0, 3),
            "remaining_steps": max(0, max_steps - step),
        }
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "exists": run_dir.exists(),
        "config": config,
        "progress": progress,
        "metrics": metrics,
        "checkpoints": _checkpoint_report(run_dir),
        "stop_file": (run_dir / "STOP").exists(),
        "failed_checkpoint": (run_dir / "failed.pt").exists(),
        "interrupted_checkpoint": (run_dir / "interrupted.pt").exists(),
        "log_tail": [line.rstrip("\n") for line in _tail_lines(run_dir / "train.log", tail_log)],
        "stderr_tail": [line.rstrip("\n") for line in _tail_lines(run_dir / "trainer.stderr.log", min(tail_log, 80))],
    }


def _warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    run = report.get("run") or {}
    progress = run.get("progress") or {}
    latest = (((run.get("metrics") or {}).get("latest_train")) or {})
    summary = ((run.get("metrics") or {}).get("summary")) or {}
    processes = report.get("processes") or ""
    if run.get("exists") and progress and progress.get("remaining_steps", 0) > 0 and "python" not in processes.lower():
        warnings.append("run appears incomplete but no matching training process is visible")
    grad = _safe_float(latest.get("grad_norm"))
    if grad is not None and grad > 20:
        warnings.append(f"latest grad_norm is high: {grad:.3f}")
    bad_grad = summary.get("bad_grad_window_count", {}).get("max")
    if bad_grad is not None and bad_grad > 0:
        warnings.append(f"recent bad_grad_window_count reached {bad_grad}")
    disk = report.get("disk") or {}
    if disk.get("free_gb") is not None and disk["free_gb"] < 50:
        warnings.append(f"low disk space: {disk['free_gb']}GB free")
    dataset = report.get("dataset") or {}
    config = run.get("config") or {}
    safe_target = dataset.get("safe_no_repeat_target_tokens")
    target = config.get("target_tokens")
    if safe_target and target and target > safe_target:
        warnings.append(f"target_tokens={target} exceeds safe one-epoch target={safe_target}")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--auto-batch-max", type=int, default=64)
    parser.add_argument("--recent-window", type=int, default=200)
    parser.add_argument("--tail-log", type=int, default=80)
    parser.add_argument("--gpu-samples", type=int, default=3)
    parser.add_argument("--gpu-interval", type=float, default=1.0)
    parser.add_argument("--scan-all-metrics", action="store_true")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_name = args.run_name or _latest_run(runs_root)
    run_dir = runs_root / run_name if run_name else runs_root / "__missing__"

    config = _json_load(run_dir / "config.json") or {}
    model_config = config.get("model") or {}
    seq_len = args.seq_len or int(model_config.get("max_seq_len") or 0)
    auto_batch_max = args.auto_batch_max or int(config.get("auto_batch_max") or 64)
    dataset_path = Path(args.dataset_path or config.get("data_path") or "")

    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "runs_root": str(runs_root),
        "run_name": run_name,
        "disk": _disk_report(runs_root),
        "gpu_samples": _gpu_samples(args.gpu_samples, args.gpu_interval),
        "processes": _process_report(),
        "dataset": _dataset_report(dataset_path, seq_len if seq_len > 0 else None, auto_batch_max),
        "run": _run_report(run_dir, args.recent_window, args.tail_log, args.scan_all_metrics),
    }
    report["warnings"] = _warnings(report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
