"""Monitor Windows training processes and nearby system events.

The monitor is intentionally passive: it never kills, pauses, or restarts
training. It records enough context to diagnose silent exits on shared Windows
GPU hosts where Python/CUDA failures may not produce a traceback.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_text(command: list[str], timeout: int = 20) -> tuple[int, str]:
    flags = 0
    if os.name == "nt":
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            creationflags=flags,
            check=False,
        )
        return result.returncode, result.stdout.strip()
    except Exception as exc:  # noqa: BLE001 - diagnostics should not crash.
        return -1, f"{type(exc).__name__}: {exc}"


def powershell(script: str, timeout: int = 30) -> tuple[int, str]:
    return run_text(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        timeout=timeout,
    )


def emit(path: Path, record: dict[str, Any]) -> None:
    record.setdefault("ts_utc", utc_now())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def get_processes(match: str) -> list[dict[str, Any]]:
    pattern = match.replace("'", "''")
    script = rf"""
$items = Get-CimInstance Win32_Process |
  Where-Object {{ $_.CommandLine -like '*{pattern}*' -or $_.CommandLine -like '*naime_hybrid.training.train*' -or $_.CommandLine -like '*naime_guardian*' }} |
  Select-Object ProcessId,ParentProcessId,Name,CreationDate,ExecutablePath,CommandLine
$items | ConvertTo-Json -Compress -Depth 4
"""
    code, output = powershell(script, timeout=20)
    if code != 0 or not output:
        return []
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return [{"parse_error": output}]
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def get_events(minutes: int, match: str) -> list[dict[str, Any]]:
    safe = match.replace("'", "''")
    script = rf"""
$since = (Get-Date).AddMinutes(-{minutes})
$patterns = @('{safe}', 'python', 'python.exe', 'naime', 'torch', 'cuda', 'cudnn', 'nvcuda', 'nvlddmkm', 'NVIDIA', 'Application Error', 'Windows Error Reporting')
$events = foreach ($log in @('Application','System')) {{
  Get-WinEvent -FilterHashtable @{{LogName=$log; StartTime=$since}} -ErrorAction SilentlyContinue |
    Where-Object {{
      $msg = $_.Message
      $provider = $_.ProviderName
      foreach ($p in $patterns) {{
        if ($msg -like "*$p*" -or $provider -like "*$p*") {{ return $true }}
      }}
      return $false
    }} |
    Select-Object @{{Name='LogName';Expression={{$log}}}}, TimeCreated, ProviderName, Id, LevelDisplayName, Message
}}
$events | Sort-Object TimeCreated | Select-Object -Last 80 | ConvertTo-Json -Compress -Depth 4
"""
    code, output = powershell(script, timeout=40)
    if code != 0 or not output:
        return []
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return [{"parse_error": output[-4000:]}]
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def get_gpu() -> dict[str, Any]:
    code, output = run_text(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    return {"returncode": code, "output": output}


def file_snapshot(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*")):
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append(
            {
                "name": path.name,
                "is_dir": path.is_dir(),
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            }
        )
    model_dir = run_dir / "models"
    if model_dir.exists():
        for path in sorted(model_dir.glob("*")):
            try:
                stat = path.stat()
            except OSError:
                continue
            rows.append(
                {
                    "name": f"models/{path.name}",
                    "is_dir": path.is_dir(),
                    "size": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                }
            )
    return rows


def tail_text(path: Path, max_bytes: int) -> str:
    if not path.exists() or max_bytes <= 0:
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--event-window-minutes", type=int, default=20)
    parser.add_argument("--tail-bytes", type=int, default=12000)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (run_dir / "event_monitor.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "event_monitor.pid").write_text(str(os.getpid()), encoding="utf-8")

    emit(
        output,
        {
            "type": "monitor_start",
            "run_name": args.run_name,
            "run_dir": str(run_dir),
            "interval": args.interval,
            "duration": args.duration,
            "argv": sys.argv,
        },
    )

    deadline = time.monotonic() + args.duration
    last_process_ids: set[int] = set()
    had_process = False
    while time.monotonic() < deadline:
        processes = get_processes(args.run_name)
        process_ids = {int(p["ProcessId"]) for p in processes if isinstance(p.get("ProcessId"), int)}
        if process_ids:
            had_process = True
        disappeared = sorted(last_process_ids - process_ids)
        last_process_ids = process_ids

        emit(
            output,
            {
                "type": "sample",
                "processes": processes,
                "disappeared_pids": disappeared,
                "gpu": get_gpu(),
                "files": file_snapshot(run_dir),
                "train_log_tail": tail_text(run_dir / "train.log", args.tail_bytes),
                "launcher_stdout_tail": tail_text(run_dir / "launcher.stdout.log", args.tail_bytes),
                "launcher_stderr_tail": tail_text(run_dir / "launcher.stderr.log", args.tail_bytes),
            },
        )

        if disappeared or (had_process and not process_ids):
            emit(
                output,
                {
                    "type": "process_disappeared",
                    "last_pids": sorted(last_process_ids),
                    "events": get_events(args.event_window_minutes, args.run_name),
                },
            )
        time.sleep(max(1.0, args.interval))

    emit(
        output,
        {
            "type": "monitor_stop",
            "processes": get_processes(args.run_name),
            "events": get_events(args.event_window_minutes, args.run_name),
            "gpu": get_gpu(),
            "files": file_snapshot(run_dir),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
