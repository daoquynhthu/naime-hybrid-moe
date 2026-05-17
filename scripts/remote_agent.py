from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


def _workspace_value(section: str, key: str, env_name: str, fallback: str | None = None) -> str:
    if os.environ.get(env_name):
        return os.environ[env_name]
    config_path = Path(os.environ.get("NAIME_WORKSPACE_CONFIG", "configs/workspace.local.json"))
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        value = config.get(section, {}).get(key)
        if value:
            return str(value)
    if fallback is not None:
        return fallback
    raise RuntimeError(
        f"Missing {section}.{key}. Set {env_name} or create configs/workspace.local.json from workspace.example.json."
    )


DEFAULT_ROOT = Path(_workspace_value("remote", "root", "NAIME_REMOTE_ROOT"))
DEFAULT_REPO = Path(_workspace_value("remote", "repo", "NAIME_REMOTE_REPO", str(DEFAULT_ROOT / "naime-hybrid-moe")))
DEFAULT_RUN_ROOT = Path(_workspace_value("remote", "runs", "NAIME_REMOTE_RUNS", str(DEFAULT_ROOT / "runs")))
DEFAULT_PYTHON = Path(_workspace_value("remote", "python", "NAIME_REMOTE_PYTHON"))


class AgentState:
    def __init__(self, token: str, repo: Path, run_root: Path, python: Path) -> None:
        self.token = token
        self.repo = repo
        self.run_root = run_root
        self.python = python


def read_tail(path: Path, max_lines: int = 80, max_bytes: int = 256 * 1024) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as file:
        file.seek(max(0, size - max_bytes))
        data = file.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def latest_run(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    runs = [p for p in run_root.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def parse_jsonl_tail(path: Path, max_lines: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in read_tail(path, max_lines=max_lines * 4):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-max_lines:]


def read_incremental(path: Path, offset: int) -> tuple[str, int, bool]:
    if not path.exists():
        return "", 0, False
    size = path.stat().st_size
    reset = offset > size
    if reset:
        offset = 0
    with path.open("rb") as file:
        file.seek(max(0, offset))
        data = file.read()
    return data.decode("utf-8", errors="replace"), size, reset


def compact_metric_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        row = {"step": raw.get("step"), "record_type": raw.get("record_type")}
        aliases = {
            "loss_lm": ("loss_lm", "lm_loss"),
            "val_lm_loss": ("val_lm_loss", "val_loss_lm"),
            "val_ppl": ("val_ppl", "val_ppl_val"),
        }
        for target, names in aliases.items():
            for name in names:
                if name in raw:
                    row[target] = raw.get(name)
                    break
        for key in (
            "tokens_per_second",
            "alpha_downstream_mean",
            "val_alpha_downstream_mean",
            "router_entropy",
            "val_router_entropy",
            "v6_slot_cosine",
            "v6_slot_context_cosine",
            "val_v6_slot_cosine",
            "val_v6_slot_context_cosine",
            "v6_boundary_self",
            "v6_boundary_world",
            "v6_boundary_other",
            "v6_boundary_unknown",
        ):
            if key in raw:
                row[key] = raw.get(key)
        rows.append({key: value for key, value in row.items() if value is not None})
    return rows


def run_nvidia_smi() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
            check=False,
        )
        return result.stdout
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"nvidia-smi failed: {exc}"


def build_training_args(state: AgentState, payload: dict[str, Any]) -> tuple[list[str], Path]:
    run_name = payload.get("run_name") or time.strftime("naime_remote_%Y%m%d_%H%M%S")
    run_dir = Path(payload.get("run_dir") or state.run_root / run_name)
    data_path = payload.get(
        "data_path",
        _workspace_value("local", "fineweb_edu_50m", "NAIME_DEFAULT_DATASET", str(DEFAULT_ROOT / "datasets" / "fineweb_edu_50m")),
    )

    if payload.get("model"):
        wrapper_args = [
            "-Model",
            str(payload["model"]),
            "-RunName",
            run_name,
            "-OutputDir",
            str(state.run_root),
            "-DataPath",
            str(data_path),
        ]
        for key, value in payload.get("script_args", {}).items():
            param = "-" + key
            if isinstance(value, bool):
                if value:
                    wrapper_args.append(param)
            elif value is not None:
                wrapper_args.extend([param, str(value)])
        args = [
            str(state.python),
            str(state.repo / "scripts" / "launch_train_detached.py"),
            "--repo",
            str(state.repo),
            "--python",
            str(state.python),
            "--run-dir",
            str(run_dir),
            "--",
            *wrapper_args,
        ]
        return args, run_dir

    trainer_flags = [
        "--architecture",
        payload.get("architecture", "naime_state_moe_v6"),
        "--dataset-format",
        payload.get("dataset_format", "hf_disk"),
        "--data-path",
        str(data_path),
        "--output-dir",
        str(run_dir),
        "--batch-size",
        str(payload.get("batch_size", 24)),
        "--seq-len",
        str(payload.get("seq_len", 512)),
        "--max-steps",
        str(payload.get("max_steps", 2000)),
        "--eval-every",
        str(payload.get("eval_every", 500)),
        "--save-every",
        str(payload.get("save_every", 2000)),
        "--latest-every",
        str(payload.get("latest_every", 500)),
        "--keep-last",
        str(payload.get("keep_last", 2)),
    ]

    skip_extra = {"eval_every", "save_every", "latest_every", "keep_last"}
    for key, value in payload.get("extra_args", {}).items():
        if key in skip_extra:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                trainer_flags.append(flag)
        elif value is not None:
            trainer_flags.extend([flag, str(value)])

    guardian_exe = state.repo / "scripts" / "naime_guardian.exe"
    args = [
        str(guardian_exe),
        "--repo",
        str(state.repo),
        "--trainer-python",
        str(state.python),
        "--run-dir",
        str(run_dir),
        "--max-restarts", "0",
        "--",
        *trainer_flags,
    ]

    return args, run_dir


def json_response(handler: BaseHTTPRequestHandler, status: int, data: dict[str, Any]) -> None:
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class RemoteAgentHandler(BaseHTTPRequestHandler):
    state: AgentState

    def log_message(self, fmt: str, *args: Any) -> None:
        message = fmt % args
        sys.stderr.write(f"{self.log_date_time_string()} - {message}\n")

    def authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {self.state.token}"

    def read_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def do_GET(self) -> None:  # noqa: N802
        try:
            if not self.authorized():
                json_response(self, 401, {"ok": False, "error": "unauthorized"})
                return

            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            if parsed.path == "/health":
                json_response(self, 200, {"ok": True, "repo": str(self.state.repo), "python": str(self.state.python)})
                return

            if parsed.path == "/status":
                run_name = query.get("run", [""])[0]
                run_dir = self.state.run_root / run_name if run_name else latest_run(self.state.run_root)
                if run_dir is None:
                    json_response(self, 200, {"ok": True, "gpu": run_nvidia_smi(), "run": None})
                    return
                data = {
                    "ok": True,
                    "gpu": run_nvidia_smi(),
                    "run": str(run_dir),
                    "metrics": parse_jsonl_tail(run_dir / "metrics.jsonl"),
                    "train_log": read_tail(run_dir / "train.log", max_lines=30),
                    "stderr": read_tail(run_dir / "launcher.stderr.log", max_lines=30),
                }
                json_response(self, 200, data)
                return

            if parsed.path == "/stream":
                run_name = query.get("run", [""])[0]
                run_dir = self.state.run_root / run_name if run_name else latest_run(self.state.run_root)
                if run_dir is None:
                    json_response(self, 200, {"ok": True, "run": None})
                    return
                log_text, log_offset, log_reset = read_incremental(
                    run_dir / "train.log", int(query.get("log_offset", ["0"])[0] or 0)
                )
                metrics_text, metrics_offset, metrics_reset = read_incremental(
                    run_dir / "metrics.jsonl", int(query.get("metrics_offset", ["0"])[0] or 0)
                )
                stderr_text, stderr_offset, stderr_reset = read_incremental(
                    run_dir / "launcher.stderr.log", int(query.get("stderr_offset", ["0"])[0] or 0)
                )
                data = {
                    "ok": True,
                    "run": str(run_dir),
                    "offsets": {
                        "log": log_offset,
                        "metrics": metrics_offset,
                        "stderr": stderr_offset,
                    },
                    "resets": {
                        "log": log_reset,
                        "metrics": metrics_reset,
                        "stderr": stderr_reset,
                    },
                    "train_log": log_text.splitlines(),
                    "metrics": compact_metric_rows(metrics_text),
                    "stderr": stderr_text.splitlines(),
                }
                json_response(self, 200, data)
                return

            json_response(self, 404, {"ok": False, "error": "not found"})
        except Exception as exc:  # pragma: no cover - operational safety net
            sys.stderr.write(f"{self.log_date_time_string()} - GET failed: {exc!r}\n")
            json_response(self, 500, {"ok": False, "error": repr(exc)})

    def do_POST(self) -> None:  # noqa: N802
        if not self.authorized():
            json_response(self, 401, {"ok": False, "error": "unauthorized"})
            return

        try:
            payload = self.read_payload()
            if self.path.startswith("/run"):
                args, run_dir = build_training_args(self.state, payload)
                run_dir.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(self.state.repo / "src")
                env["NAIME_HYBRID_PYTHON"] = str(self.state.python)
                env.setdefault("PYTHONUTF8", "1")
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                stdout = open(run_dir / "launcher.stdout.log", "ab")
                stderr = open(run_dir / "launcher.stderr.log", "ab")
                creationflags = 0
                if os.name == "nt":
                    creationflags = (
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NO_WINDOW
                    )
                proc = subprocess.Popen(
                    args,
                    cwd=self.state.repo,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    stdin=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                json_response(self, 200, {"ok": True, "pid": proc.pid, "run_dir": str(run_dir), "args": args})
                return

            if self.path.startswith("/stop"):
                run_name = payload.get("run_name")
                run_dir = self.state.run_root / run_name if run_name else latest_run(self.state.run_root)
                if run_dir is None:
                    json_response(self, 404, {"ok": False, "error": "no run found"})
                    return
                (run_dir / "STOP").write_text("requested by remote_agent\n", encoding="utf-8")
                json_response(self, 200, {"ok": True, "stop_file": str(run_dir / "STOP")})
                return

            json_response(self, 404, {"ok": False, "error": "not found"})
        except Exception as exc:  # pragma: no cover - operational safety net
            json_response(self, 500, {"ok": False, "error": repr(exc)})


def _ensure_no_window():
    if os.name != "nt":
        return
    if "pythonw" in sys.executable.lower():
        return
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        return
    subprocess.Popen(
        [str(pythonw)] + sys.argv,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
    )
    sys.exit(0)


def main() -> None:
    _ensure_no_window()

    parser = argparse.ArgumentParser(description="Small HTTP control plane for NAIME remote training.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    args = parser.parse_args()

    RemoteAgentHandler.state = AgentState(
        token=args.token,
        repo=args.repo,
        run_root=args.run_root,
        python=args.python,
    )
    server = ThreadingHTTPServer((args.host, args.port), RemoteAgentHandler)
    print(f"NAIME remote agent listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
