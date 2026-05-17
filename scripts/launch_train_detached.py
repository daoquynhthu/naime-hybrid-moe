"""Launch NAIME training without creating a visible Windows console window.

When available, the native naime_guardian.exe is used as a shutdown
coordinator. It receives ordinary console/logoff/shutdown events, writes the
trainer STOP file, and waits for the Python trainer to checkpoint and exit.
The trainer is launched with CREATE_NO_WINDOW, so python.exe is preferred over
pythonw.exe for reliable stderr/stdout redirection on Windows. It is not a
process anti-kill mechanism.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _creationflags() -> int:
    flags = 0
    for name in ("CREATE_NEW_PROCESS_GROUP", "DETACHED_PROCESS", "CREATE_NO_WINDOW"):
        flags |= getattr(subprocess, name, 0)
    return flags


def _parse_printed_args(output: str) -> list[str]:
    args: list[str] = []
    started = False
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Resolved training arguments") or line.startswith("=== Resolved training arguments"):
            started = True
            continue
        if started:
            args.append(line)
    if not args:
        raise RuntimeError(f"could not parse train_model.ps1 -PrintArgs output:\n{output}")
    return args


def resolve_training_args(repo: Path, wrapper_args: list[str], env: dict[str, str]) -> list[str]:
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(repo / "scripts" / "train_model.ps1"),
        *wrapper_args,
        "-PrintArgs",
    ]
    result = subprocess.run(
        command,
        cwd=str(repo),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(f"train_model.ps1 -PrintArgs failed ({result.returncode}):\n{result.stdout}")
    return _parse_printed_args(result.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-guardian", action="store_true")
    parser.add_argument("wrapper_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["NAIME_HYBRID_PYTHON"] = str(args.python)
    env["PYTHONPATH"] = str(repo / "src")

    wrapper_args = list(args.wrapper_args)
    if wrapper_args and wrapper_args[0] == "--":
        wrapper_args = wrapper_args[1:]
    train_args = resolve_training_args(repo, wrapper_args, env)

    train_python = args.python
    command = [
        str(train_python),
        "-m",
        "naime_hybrid.training.train",
        *train_args,
    ]
    guardian_exe = repo / "scripts" / "naime_guardian.exe"
    if not args.no_guardian and os.name == "nt" and guardian_exe.exists():
        command = [
            str(guardian_exe),
            "--repo",
            str(repo),
            "--trainer-python",
            str(train_python),
            "--run-dir",
            str(run_dir),
            "--max-restarts",
            "0",
            "--",
            *train_args,
        ]
    (run_dir / "launch_cmd.txt").write_text(" ".join(command), encoding="utf-8")

    stdout = open(run_dir / "launcher.stdout.log", "ab", buffering=0)
    stderr = open(run_dir / "launcher.stderr.log", "ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=str(repo),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        creationflags=_creationflags(),
        close_fds=True,
    )
    (run_dir / "daemon.pid").write_text(str(process.pid), encoding="utf-8")
    print(process.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
