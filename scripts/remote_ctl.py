from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def workspace_value(section: str, key: str, env_name: str, fallback: str = "") -> str:
    if os.environ.get(env_name):
        return os.environ[env_name]
    config_path = os.environ.get("NAIME_WORKSPACE_CONFIG", "configs/workspace.local.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        value = config.get(section, {}).get(key)
        if value:
            return str(value)
    return fallback


def emit(text: object = "") -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(str(text).encode(encoding, errors="replace").decode(encoding, errors="replace"))


def request(base_url: str, token: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {text}") from exc


def print_status(data: dict[str, Any]) -> None:
    emit("\n[gpu]")
    emit(data.get("gpu", "").rstrip())
    emit("\n[run]")
    emit(data.get("run"))
    emit("\n[metrics]")
    for row in data.get("metrics") or []:
        compact = {
            key: row.get(key)
            for key in (
                "step",
                "record_type",
                "loss_lm",
                "val_lm_loss",
                "val_ppl",
                "tokens_per_second",
                "v6_slot_cosine",
                "v6_slot_context_cosine",
                "v6_boundary_self",
            )
            if key in row
        }
        emit(json.dumps(compact, ensure_ascii=False))
    emit("\n[train.log]")
    for line in data.get("train_log") or []:
        emit(line)
    if data.get("stderr"):
        emit("\n[stderr]")
        for line in data["stderr"]:
            emit(line)


def stream_path(run_name: str, offsets: dict[str, int]) -> str:
    params = {
        "log_offset": str(offsets.get("log", 0)),
        "metrics_offset": str(offsets.get("metrics", 0)),
        "stderr_offset": str(offsets.get("stderr", 0)),
    }
    if run_name:
        params["run"] = run_name
    return "/stream?" + urllib.parse.urlencode(params)


def print_stream(data: dict[str, Any]) -> dict[str, int]:
    run = data.get("run")
    if run:
        emit(f"\n[run] {run}")
    for row in data.get("metrics") or []:
        emit("[metric] " + json.dumps(row, ensure_ascii=False))
    for line in data.get("train_log") or []:
        emit(line)
    if data.get("stderr"):
        emit("\n[stderr]")
        for line in data["stderr"]:
            emit(line)
    return {
        "log": int(data.get("offsets", {}).get("log", 0)),
        "metrics": int(data.get("offsets", {}).get("metrics", 0)),
        "stderr": int(data.get("offsets", {}).get("stderr", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Client for the NAIME remote training agent.")
    parser.add_argument("command", choices=["health", "status", "watch", "stream", "run", "stop"])
    parser.add_argument("--url", default=os.environ.get("NAIME_AGENT_URL", "http://127.0.0.1:8766"))
    parser.add_argument("--token", default=os.environ.get("NAIME_AGENT_TOKEN"))
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--data-path", default=workspace_value("local", "fineweb_edu_50m", "NAIME_DEFAULT_DATASET"))
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Missing token. Set NAIME_AGENT_TOKEN or pass --token.")

    if args.command == "health":
        print(json.dumps(request(args.url, args.token, "GET", "/health"), ensure_ascii=False, indent=2))
    elif args.command == "status":
        print_status(request(args.url, args.token, "GET", "/status"))
    elif args.command == "watch":
        print_status(request(args.url, args.token, "GET", "/status"))
        offsets = {"log": 0, "metrics": 0, "stderr": 0}
        first = request(args.url, args.token, "GET", stream_path(args.run_name, offsets))
        offsets = {
            "log": int(first.get("offsets", {}).get("log", 0)),
            "metrics": int(first.get("offsets", {}).get("metrics", 0)),
            "stderr": int(first.get("offsets", {}).get("stderr", 0)),
        }
        while True:
            time.sleep(args.interval)
            offsets = print_stream(request(args.url, args.token, "GET", stream_path(args.run_name, offsets)))
    elif args.command == "stream":
        offsets = {"log": 0, "metrics": 0, "stderr": 0}
        print_stream(request(args.url, args.token, "GET", stream_path(args.run_name, offsets)))
    elif args.command == "stop":
        payload = {"run_name": args.run_name} if args.run_name else {}
        print(json.dumps(request(args.url, args.token, "POST", "/stop", payload), ensure_ascii=False, indent=2))
    elif args.command == "run":
        payload = {
            "run_name": args.run_name or time.strftime("naime_v6_remote_%Y%m%d_%H%M%S"),
            "model": "naime_v6_recursive_self_moe",
            "data_path": args.data_path,
            "script_args": {
                "TargetTokens": args.max_steps * args.batch_size * args.seq_len,
                "SeqLen": args.seq_len,
                "EvalEvery": 1000,
                "EvalMaxBatches": 0,
                "SaveEvery": 5000,
                "LatestEvery": 0,
                "AutoBatchMax": 64,
                "LambdaSelfSlotDiversity": 0.02,
                "SelfStateIdentityScale": 0.02,
            },
        }
        print(json.dumps(request(args.url, args.token, "POST", "/run", payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
