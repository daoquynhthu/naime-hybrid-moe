import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any


class CompactColorFormatter(logging.Formatter):
    COLORS = {
        "step": "\033[96m",
        "eval": "\033[95m",
        "lm": "\033[92m",
        "total": "\033[93m",
        "ppl": "\033[94m",
        "aux": "\033[33m",
        "alpha": "\033[36m",
        "tr": "\033[96m",
        "val": "\033[95m",
        "best": "\033[92m",
        "gap": "\033[91m",
        "ent": "\033[35m",
        "v4": "\033[36m",
        "tok": "\033[90m",
        "grad": "\033[90m",
        "sp": "\033[35m",
        "kl": "\033[34m",
        "save": "\033[90m",
        "warn": "\033[91m",
    }
    RESET = "\033[0m"

    def __init__(self, *args: Any, use_color: bool = True, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.use_color = use_color and os.environ.get("NO_COLOR") is None

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if not self.use_color:
            return rendered
        if record.levelno >= logging.WARNING:
            return f"{self.COLORS['warn']}{rendered}{self.RESET}"
        for key in ["tr", "val", "best", "gap", "lm", "ppl", "alpha", "ent", "v4", "tok", "grad", "save"]:
            rendered = re.sub(
                rf"\b({key})\b",
                f"{self.COLORS[key]}\\1{self.RESET}",
                rendered,
            )
        return rendered


def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("naime_hybrid.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = CompactColorFormatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(console_formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    return logger


class JsonlMetricLogger:
    def __init__(self, path: Path, *, flush_every: int = 1, fsync_every: int = 100):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self.fsync_every = max(0, int(fsync_every))
        self._writes = 0
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any], *, force_sync: bool = False) -> None:
        self._file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self._writes += 1
        should_flush = force_sync or self._writes % self.flush_every == 0
        should_fsync = force_sync or (self.fsync_every > 0 and self._writes % self.fsync_every == 0)
        if should_flush or should_fsync:
            self._file.flush()
        if should_fsync:
            os.fsync(self._file.fileno())

    def flush(self, *, force_sync: bool = False) -> None:
        self._file.flush()
        if force_sync:
            os.fsync(self._file.fileno())

    def close(self) -> None:
        if self._file.closed:
            return
        self.flush(force_sync=True)
        self._file.close()


def metrics_jsonl_to_csv(jsonl_path: Path, csv_path: Path | None = None) -> Path | None:
    """Convert full JSONL metrics to a flat CSV for analysis."""
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return None
    csv_path = csv_path or jsonl_path.with_suffix(".csv")

    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    seen: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    if not rows:
        return None

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path
