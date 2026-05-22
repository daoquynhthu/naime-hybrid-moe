"""AT Protocol / Jetstream data pipeline for NAIME corpora.

The pipeline follows docs/DATA_PIPELINE_SPEC.md:

- raw event capture is separated from normalized text documents;
- every stage writes manifests, logs, and completion markers;
- final training data is a prebuilt Hugging Face disk dataset;
- private/social identifiers are not preserved in processed documents.

Jetstream is used for the default intake path because it is JSON based and can
filter by collection. For verified archival mirroring, use a full ATProto sync
or Tap-based backfill outside this script and feed the resulting JSONL shards to
the normalize/filter stages.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import socket
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


PIPELINE_VERSION = "atproto-data-pipeline-v1"
DEFAULT_COLLECTIONS = ["app.bsky.feed.post"]
DEFAULT_ENDPOINT = "wss://jetstream2.us-east.bsky.network/subscribe"


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
SECRET_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|ssh-rsa|ghp_[A-Za-z0-9_]+|hf_[A-Za-z0-9_]+)\b"
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w[\w.-]{1,64}")
TAG_RE = re.compile(r"#\w[\w-]{1,80}")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REPEATED_CHAR_RE = re.compile(r"(.)\1{8,}")


@dataclass
class StageStats:
    stage: str
    records_in: int = 0
    records_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    dropped: dict[str, int] | None = None
    started_at_utc: str = ""
    finished_at_utc: str = ""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_stage_marker(output_dir: Path, stats: StageStats, extra: dict[str, Any] | None = None) -> None:
    payload = asdict(stats)
    payload["status"] = "complete"
    payload["pipeline_version"] = PIPELINE_VERSION
    if extra:
        payload.update(extra)
    atomic_write_json(output_dir / "_stage_complete.json", payload)


def open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", errors="replace", newline="\n")
    return open(path, mode, encoding="utf-8", errors="replace", newline="\n")


def jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    patterns = ["*.jsonl", "*.jsonl.gz"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.rglob(pattern))
    return sorted(
        file
        for file in files
        if not file.name.startswith("_")
        and file.name != "document-manifest.jsonl"
        and not file.name.endswith("-manifest.jsonl")
    )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for file in jsonl_files(path):
        with open_text(file, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with gzip.open(tmp, "wt", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp.replace(path)
    return count, path.stat().st_size


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalized_text(text: str) -> str:
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    text = text.replace("\ufffd", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = text.strip()
    return text


def redact_private_patterns(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text


def get_nested(row: dict[str, Any], path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def first_text_value(row: dict[str, Any], candidates: list[str]) -> str | None:
    for field in candidates:
        value = get_nested(row, field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def longest_string_value(row: dict[str, Any]) -> str | None:
    strings: list[str] = []
    for value in row.values():
        if isinstance(value, str) and value.strip():
            strings.append(value)
    if not strings:
        return None
    return max(strings, key=len)


def doc_from_text(
    text: str,
    *,
    source_family: str,
    source_name: str,
    source_ref: str,
    row_id: str,
    license_label: str,
    language_hint: str,
) -> dict[str, Any] | None:
    text = redact_private_patterns(normalized_text(text))
    if not text:
        return None
    doc_hash = sha256_text(text)
    source_doc_id = sha256_text(f"{source_name}:{source_ref}:{row_id}")
    return {
        "doc_id": f"{source_family}:{doc_hash[:24]}",
        "source_doc_id_hash": source_doc_id,
        "source_family": source_family,
        "source_collection": source_name,
        "text": text,
        "text_chars": len(text),
        "doc_hash": doc_hash,
        "near_hash": f"{simhash64(text):016x}",
        "created_at": "",
        "language_hint": language_hint,
        "license": license_label,
        "has_facets": False,
        "reply_like": False,
        "embed_like": False,
    }


def extract_post_text(event: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    commit = event.get("commit") or {}
    record = commit.get("record") or {}
    collection = commit.get("collection")
    operation = commit.get("operation")
    text = record.get("text")
    if operation not in {"create", "update"}:
        return None, {"drop_reason": "not_create_or_update"}
    if collection != "app.bsky.feed.post":
        return None, {"drop_reason": "unsupported_collection"}
    if not isinstance(text, str):
        return None, {"drop_reason": "missing_text"}
    facets = record.get("facets")
    langs = record.get("langs")
    meta = {
        "source_collection": collection,
        "created_at": record.get("createdAt"),
        "language_hint": langs[0] if isinstance(langs, list) and langs else None,
        "has_facets": isinstance(facets, list) and bool(facets),
        "reply_like": record.get("reply") is not None,
        "embed_like": record.get("embed") is not None,
    }
    return text, meta


def quality_scores(text: str) -> dict[str, Any]:
    length = len(text)
    if length == 0:
        return {"quality_score": 0.0, "drop_reason": "empty"}
    url_chars = sum(len(m.group(0)) for m in URL_RE.finditer(text))
    ascii_letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    nonspace = sum(not ch.isspace() for ch in text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    unique_ratio = len(set(text)) / max(1, min(length, 256))
    repeated = bool(REPEATED_CHAR_RE.search(text))
    url_ratio = url_chars / max(1, length)
    punct_ratio = punct / max(1, nonspace)
    digit_ratio = digits / max(1, nonspace)
    alpha_ratio = ascii_letters / max(1, nonspace)
    dialogue_ratio = 1.0 if length < 500 else 0.5
    code_ratio = min(1.0, (text.count("{") + text.count("}") + text.count(";")) / max(1, length / 100))

    score = 0.35
    if length >= 80:
        score += 0.18
    if length >= 180:
        score += 0.12
    if len(lines) >= 2:
        score += 0.06
    if alpha_ratio > 0.45:
        score += 0.08
    if unique_ratio > 0.25:
        score += 0.08
    if url_ratio > 0.35:
        score -= 0.25
    if punct_ratio > 0.55:
        score -= 0.20
    if digit_ratio > 0.55:
        score -= 0.15
    if repeated:
        score -= 0.20
    if SECRET_RE.search(text):
        score = 0.0

    score = max(0.0, min(1.0, score))
    if score < 0.25:
        tier = "q0_reject"
    elif score < 0.45:
        tier = "q1_low"
    elif score < 0.72:
        tier = "q2_base"
    else:
        tier = "q3_high"
    return {
        "quality_score": score,
        "quality_tier": tier,
        "language_score": 0.5,
        "dialogue_ratio": dialogue_ratio,
        "code_ratio": code_ratio,
        "url_ratio": url_ratio,
        "punct_ratio": punct_ratio,
        "digit_ratio": digit_ratio,
        "alpha_ratio": alpha_ratio,
        "repeated_char": repeated,
        "drop_reason": "low_quality" if tier == "q0_reject" else "",
    }


def simhash64(text: str) -> int:
    words = re.findall(r"[\w\u4e00-\u9fff]{2,}", text.lower())
    if not words:
        return 0
    acc = [0] * 64
    for word in words[:2048]:
        h = int(hashlib.blake2b(word.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for bit in range(64):
            acc[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit, value in enumerate(acc):
        if value >= 0:
            out |= 1 << bit
    return out


def stable_split(doc_hash: str, seed: str, validation_ratio: float, test_ratio: float) -> str:
    h = hashlib.sha256(f"{seed}:{doc_hash}".encode("utf-8")).hexdigest()
    value = int(h[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if value < test_ratio:
        return "test"
    if value < test_ratio + validation_ratio:
        return "validation"
    return "train"


def command_fetch_jetstream(args: argparse.Namespace) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit("fetch-jetstream requires package 'websockets'. Install it in the data-processing venv.") from exc

    output = Path(args.output)
    shard_dir = output / "shards"
    manifest_dir = output / "manifests"
    log_dir = output / "logs"
    ensure_dir(shard_dir)
    ensure_dir(manifest_dir)
    ensure_dir(log_dir)

    collections = args.collections or DEFAULT_COLLECTIONS
    query: list[tuple[str, str]] = []
    for collection in collections:
        query.append(("wantedCollections", collection))
    if args.cursor:
        query.append(("cursor", str(args.cursor)))
    if args.max_message_size_bytes:
        query.append(("maxMessageSizeBytes", str(args.max_message_size_bytes)))
    uri = args.endpoint + ("?" + urlencode(query) if query else "")

    stats = StageStats(stage="download", started_at_utc=utc_now())
    cursor_path = output / "manifests" / "cursor.json"
    source_manifest = output / "manifests" / "source-manifest.jsonl"
    progress_log = log_dir / "fetch-progress.jsonl"
    status_path = log_dir / "fetch-status.json"
    started = time.time()
    max_output_bytes = int(args.max_output_gib * 1024 * 1024 * 1024) if args.max_output_gib else 0
    stop_file = Path(args.stop_file) if args.stop_file else None
    shard_idx = int(time.time())
    shard_rows: list[dict[str, Any]] = []
    last_progress_time = 0.0
    last_cursor_time = 0.0
    last_cursor_record = 0
    latest_cursor: int | None = None

    def reached_output_budget() -> bool:
        return bool(max_output_bytes and stats.bytes_out >= max_output_bytes)

    def write_progress(reason: str) -> None:
        elapsed = max(time.time() - started, 1e-6)
        row = {
            "time_utc": utc_now(),
            "reason": reason,
            "records": stats.records_out,
            "input_gib": round(stats.bytes_in / (1024**3), 4),
            "output_gib": round(stats.bytes_out / (1024**3), 4),
            "target_output_gib": args.max_output_gib,
            "events_per_sec": round(stats.records_out / elapsed, 2),
            "elapsed_sec": round(elapsed, 1),
            "pending_shard_events": len(shard_rows),
        }
        append_jsonl(progress_log, row)
        atomic_write_json(status_path, row)

    def write_cursor(force: bool = False) -> None:
        nonlocal last_cursor_time, last_cursor_record
        if latest_cursor is None:
            return
        now = time.time()
        record_gap = stats.records_out - last_cursor_record
        time_gap = now - last_cursor_time
        if not force and args.cursor_events and record_gap < args.cursor_events and time_gap < args.cursor_seconds:
            return
        atomic_write_json(cursor_path, {"cursor": latest_cursor, "updated_at_utc": utc_now()})
        last_cursor_time = now
        last_cursor_record = stats.records_out

    async def run() -> None:
        nonlocal shard_idx, shard_rows, last_progress_time, latest_cursor
        async with websockets.connect(uri, max_size=args.websocket_max_size) as websocket:
            while True:
                if args.max_events and stats.records_out >= args.max_events:
                    break
                if args.max_seconds and time.time() - started >= args.max_seconds:
                    break
                if reached_output_budget():
                    break
                if stop_file and stop_file.exists():
                    write_progress("stop_file")
                    break
                raw = await websocket.recv()
                stats.bytes_in += len(raw) if isinstance(raw, (bytes, bytearray)) else len(raw.encode("utf-8"))
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                stats.records_in += 1
                shard_rows.append(event)
                stats.records_out += 1
                if event.get("time_us"):
                    latest_cursor = int(event["time_us"])
                    write_cursor()
                if len(shard_rows) >= args.shard_events:
                    flush_shard()
                    if reached_output_budget():
                        break
                now = time.time()
                if args.progress_events and stats.records_out % args.progress_events == 0:
                    write_progress("event_interval")
                    last_progress_time = now
                elif args.progress_seconds and now - last_progress_time >= args.progress_seconds:
                    write_progress("time_interval")
                    last_progress_time = now

    def flush_shard() -> None:
        nonlocal shard_idx, shard_rows
        if not shard_rows:
            return
        shard = shard_dir / f"jetstream_{shard_idx:014d}.jsonl.gz"
        rows = shard_rows
        shard_rows = []
        count, size = write_jsonl_gz(shard, rows)
        digest = sha256_file(shard)
        stats.bytes_out += size
        append_jsonl(
            source_manifest,
            {
                "source_id": "atproto/jetstream",
                "source_name": "ATProto Jetstream",
                "source_url": uri,
                "download_time_utc": utc_now(),
                "raw_path": str(shard),
                "raw_bytes": size,
                "raw_records": count,
                "sha256": digest,
                "license": args.license,
                "language_hint": args.language_hint,
                "split_policy": "can-split",
                "notes": "Raw public Jetstream events; processed outputs hash or omit user identifiers.",
            },
        )
        shard_idx += 1
        write_cursor(force=True)
        write_progress("shard_flush")

    try:
        write_progress("start")
        asyncio.run(run())
    finally:
        flush_shard()
        write_cursor(force=True)
        stats.finished_at_utc = utc_now()
        write_progress("finish")
        write_stage_marker(
            output,
            stats,
            {
                "endpoint": args.endpoint,
                "collections": collections,
                "max_output_gib": args.max_output_gib,
                "progress_log": str(progress_log),
            },
        )


def command_normalize(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output = Path(args.output)
    ensure_dir(output)
    rows_out: list[dict[str, Any]] = []
    stats = StageStats(stage="normalize", started_at_utc=utc_now(), dropped={})
    dropped = Counter()

    for event in iter_jsonl(input_path):
        stats.records_in += 1
        text, meta = extract_post_text(event)
        if text is None:
            dropped[meta.get("drop_reason", "unsupported")] += 1
            continue
        text = redact_private_patterns(normalized_text(text))
        if not text:
            dropped["empty_after_normalize"] += 1
            continue
        commit = event.get("commit") or {}
        doc_hash = sha256_text(text)
        source_doc_id = sha256_text(f"{event.get('did','')}:{commit.get('rkey','')}:{event.get('time_us','')}")
        rows_out.append(
            {
                "doc_id": f"atproto:{doc_hash[:24]}",
                "source_doc_id_hash": source_doc_id,
                "source_family": "atproto",
                "source_collection": meta.get("source_collection"),
                "text": text,
                "text_chars": len(text),
                "doc_hash": doc_hash,
                "near_hash": f"{simhash64(text):016x}",
                "created_at": meta.get("created_at"),
                "language_hint": meta.get("language_hint") or args.language_hint,
                "license": args.license,
                "privacy_filter": "email_phone_redacted;user_ids_hashed",
                "has_facets": meta.get("has_facets"),
                "reply_like": meta.get("reply_like"),
                "embed_like": meta.get("embed_like"),
            }
        )
        stats.records_out += 1
        if len(rows_out) >= args.output_shard_docs:
            shard_no = len(list(output.glob("normalized_*.jsonl.gz")))
            _, size = write_jsonl_gz(output / f"normalized_{shard_no:06d}.jsonl.gz", rows_out)
            stats.bytes_out += size
            rows_out = []

    if rows_out:
        shard_no = len(list(output.glob("normalized_*.jsonl.gz")))
        _, size = write_jsonl_gz(output / f"normalized_{shard_no:06d}.jsonl.gz", rows_out)
        stats.bytes_out += size
    stats.dropped = dict(dropped)
    stats.finished_at_utc = utc_now()
    write_stage_marker(output, stats)


def command_import_jsonl(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output = Path(args.output)
    ensure_dir(output)
    log_dir = output / "logs"
    ensure_dir(log_dir)
    status_path = log_dir / "import-status.json"
    progress_log = log_dir / "import-progress.jsonl"
    rows_out: list[dict[str, Any]] = []
    stats = StageStats(stage="import_jsonl", started_at_utc=utc_now(), dropped={})
    dropped = Counter()
    text_fields = [field.strip() for field in args.text_fields.split(",") if field.strip()]
    max_text_bytes = int(args.max_text_gib * 1024 * 1024 * 1024) if args.max_text_gib else 0
    stop_file = Path(args.stop_file) if args.stop_file else None
    text_bytes = 0
    started = time.time()
    last_progress_time = 0.0

    def write_import_progress(reason: str) -> None:
        elapsed = max(time.time() - started, 1e-6)
        row = {
            "time_utc": utc_now(),
            "reason": reason,
            "records_in": stats.records_in,
            "records_out": stats.records_out,
            "text_gib": round(text_bytes / (1024**3), 4),
            "target_text_gib": args.max_text_gib,
            "records_per_sec": round(stats.records_in / elapsed, 2),
            "elapsed_sec": round(elapsed, 1),
            "pending_docs": len(rows_out),
        }
        append_jsonl(progress_log, row)
        atomic_write_json(status_path, row)

    write_import_progress("start")
    for idx, row in enumerate(iter_jsonl(input_path)):
        now = time.time()
        if args.progress_rows and stats.records_in and stats.records_in % args.progress_rows == 0:
            write_import_progress("row_interval")
            last_progress_time = now
        elif args.progress_seconds and now - last_progress_time >= args.progress_seconds:
            write_import_progress("time_interval")
            last_progress_time = now
        if args.max_rows and stats.records_in >= args.max_rows:
            break
        if max_text_bytes and text_bytes >= max_text_bytes:
            break
        if stop_file and stop_file.exists():
            write_import_progress("stop_file")
            break
        stats.records_in += 1
        text = first_text_value(row, text_fields)
        if text is None and args.auto_text_field:
            text = longest_string_value(row)
        if text is None:
            dropped["missing_text"] += 1
            continue
        row_id_value = first_text_value(row, [field.strip() for field in args.id_fields.split(",") if field.strip()])
        doc = doc_from_text(
            text,
            source_family=args.source_family,
            source_name=args.source_name,
            source_ref=str(input_path),
            row_id=row_id_value or str(idx),
            license_label=args.license,
            language_hint=args.language_hint,
        )
        if doc is None:
            dropped["empty_after_normalize"] += 1
            continue
        rows_out.append(doc)
        text_bytes += len(doc["text"].encode("utf-8"))
        stats.records_out += 1
        if len(rows_out) >= args.output_shard_docs:
            shard_no = len(list(output.glob("normalized_*.jsonl.gz")))
            _, size = write_jsonl_gz(output / f"normalized_{shard_no:06d}.jsonl.gz", rows_out)
            stats.bytes_out += size
            rows_out = []

    if rows_out:
        shard_no = len(list(output.glob("normalized_*.jsonl.gz")))
        _, size = write_jsonl_gz(output / f"normalized_{shard_no:06d}.jsonl.gz", rows_out)
        stats.bytes_out += size
    stats.bytes_in = text_bytes
    stats.dropped = dict(dropped)
    stats.finished_at_utc = utc_now()
    write_import_progress("finish")
    write_stage_marker(
        output,
        stats,
        {
            "input": str(input_path),
            "text_fields": text_fields,
            "source_family": args.source_family,
            "source_name": args.source_name,
            "max_text_gib": args.max_text_gib,
        },
    )


def command_import_hf_dataset(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("import-hf-dataset requires package 'datasets'.") from exc

    output = Path(args.output)
    ensure_dir(output)
    log_dir = output / "logs"
    ensure_dir(log_dir)
    status_path = log_dir / "import-status.json"
    progress_log = log_dir / "import-progress.jsonl"
    rows_out: list[dict[str, Any]] = []
    stats = StageStats(stage="import_hf_dataset", started_at_utc=utc_now(), dropped={})
    dropped = Counter()
    text_fields = [field.strip() for field in args.text_fields.split(",") if field.strip()]
    id_fields = [field.strip() for field in args.id_fields.split(",") if field.strip()]
    max_text_bytes = int(args.max_text_gib * 1024 * 1024 * 1024) if args.max_text_gib else 0
    stop_file = Path(args.stop_file) if args.stop_file else None
    text_bytes = 0
    started = time.time()
    last_progress_time = 0.0

    def write_import_progress(reason: str) -> None:
        elapsed = max(time.time() - started, 1e-6)
        row = {
            "time_utc": utc_now(),
            "reason": reason,
            "records_in": stats.records_in,
            "records_out": stats.records_out,
            "text_gib": round(text_bytes / (1024**3), 4),
            "target_text_gib": args.max_text_gib,
            "records_per_sec": round(stats.records_in / elapsed, 2),
            "elapsed_sec": round(elapsed, 1),
            "pending_docs": len(rows_out),
        }
        append_jsonl(progress_log, row)
        atomic_write_json(status_path, row)

    load_kwargs: dict[str, Any] = {
        "path": args.dataset,
        "name": args.config or None,
        "split": args.split,
        "streaming": args.streaming,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.data_files:
        load_kwargs["data_files"] = args.data_files
    if args.revision:
        load_kwargs["revision"] = args.revision
    if args.token:
        load_kwargs["token"] = args.token
    load_kwargs = {k: v for k, v in load_kwargs.items() if v not in {None, ""}}
    dataset = load_dataset(**load_kwargs)

    write_import_progress("start")
    for idx, row in enumerate(dataset):
        now = time.time()
        if args.progress_rows and stats.records_in and stats.records_in % args.progress_rows == 0:
            write_import_progress("row_interval")
            last_progress_time = now
        elif args.progress_seconds and now - last_progress_time >= args.progress_seconds:
            write_import_progress("time_interval")
            last_progress_time = now
        if args.max_rows and stats.records_in >= args.max_rows:
            break
        if max_text_bytes and text_bytes >= max_text_bytes:
            break
        if stop_file and stop_file.exists():
            write_import_progress("stop_file")
            break
        stats.records_in += 1
        if not isinstance(row, dict):
            dropped["non_dict_row"] += 1
            continue
        text = first_text_value(row, text_fields)
        if text is None and args.auto_text_field:
            text = longest_string_value(row)
        if text is None:
            dropped["missing_text"] += 1
            continue
        row_id_value = first_text_value(row, id_fields)
        doc = doc_from_text(
            text,
            source_family=args.source_family,
            source_name=args.dataset,
            source_ref=args.split,
            row_id=row_id_value or str(idx),
            license_label=args.license,
            language_hint=args.language_hint,
        )
        if doc is None:
            dropped["empty_after_normalize"] += 1
            continue
        rows_out.append(doc)
        text_bytes += len(doc["text"].encode("utf-8"))
        stats.records_out += 1
        if len(rows_out) >= args.output_shard_docs:
            shard_no = len(list(output.glob("normalized_*.jsonl.gz")))
            _, size = write_jsonl_gz(output / f"normalized_{shard_no:06d}.jsonl.gz", rows_out)
            stats.bytes_out += size
            rows_out = []

    if rows_out:
        shard_no = len(list(output.glob("normalized_*.jsonl.gz")))
        _, size = write_jsonl_gz(output / f"normalized_{shard_no:06d}.jsonl.gz", rows_out)
        stats.bytes_out += size
    stats.bytes_in = text_bytes
    stats.dropped = dict(dropped)
    stats.finished_at_utc = utc_now()
    write_import_progress("finish")
    write_stage_marker(
        output,
        stats,
        {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "text_fields": text_fields,
            "source_family": args.source_family,
            "max_text_gib": args.max_text_gib,
            "streaming": args.streaming,
        },
    )


def command_filter(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output = Path(args.output)
    ensure_dir(output)
    manifest = Path(args.manifest) if args.manifest else output / "document-manifest.jsonl"
    manifest.unlink(missing_ok=True)
    stats = StageStats(stage="filter", started_at_utc=utc_now(), dropped={})
    dropped = Counter()
    rows_out: list[dict[str, Any]] = []

    for doc in iter_jsonl(input_path):
        stats.records_in += 1
        text = doc.get("text") or ""
        scores = quality_scores(text)
        reason = ""
        if len(text) < args.min_text_chars:
            reason = "too_short"
        elif len(text) > args.max_text_chars:
            text = text[: args.max_text_chars]
            doc["text"] = text
            doc["text_chars"] = len(text)
        elif SECRET_RE.search(text):
            reason = "secret_like"
        elif scores["quality_score"] < args.min_quality_score:
            reason = scores["drop_reason"] or "low_quality"

        manifest_row = {
            "doc_id": doc.get("doc_id"),
            "doc_hash": doc.get("doc_hash"),
            "near_hash": doc.get("near_hash"),
            "source_family": doc.get("source_family"),
            "license": doc.get("license"),
            "language_hint": doc.get("language_hint"),
            **scores,
            "accepted": not reason,
            "drop_reason": reason,
        }
        append_jsonl(manifest, manifest_row)
        if reason:
            dropped[reason] += 1
            continue
        doc.update(scores)
        rows_out.append(doc)
        stats.records_out += 1
        if len(rows_out) >= args.output_shard_docs:
            shard_no = len(list(output.glob("filtered_*.jsonl.gz")))
            _, size = write_jsonl_gz(output / f"filtered_{shard_no:06d}.jsonl.gz", rows_out)
            stats.bytes_out += size
            rows_out = []

    if rows_out:
        shard_no = len(list(output.glob("filtered_*.jsonl.gz")))
        _, size = write_jsonl_gz(output / f"filtered_{shard_no:06d}.jsonl.gz", rows_out)
        stats.bytes_out += size
    stats.dropped = dict(dropped)
    stats.finished_at_utc = utc_now()
    write_stage_marker(output, stats, {"document_manifest": str(manifest)})


def command_dedup(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output = Path(args.output)
    ensure_dir(output)
    stats = StageStats(stage="dedup", started_at_utc=utc_now(), dropped={})
    dropped = Counter()
    exact_seen: dict[str, dict[str, Any]] = {}
    near_seen: dict[str, dict[str, Any]] = {}

    for doc in iter_jsonl(input_path):
        stats.records_in += 1
        doc_hash = doc["doc_hash"]
        near_hash = str(doc.get("near_hash") or "")
        if doc_hash in exact_seen:
            dropped["exact_duplicate"] += 1
            continue
        near_bucket = near_hash[:8]
        if args.near_dedup and near_bucket in near_seen:
            dropped["near_duplicate_bucket"] += 1
            continue
        exact_seen[doc_hash] = doc
        near_seen[near_bucket] = doc

    rows = list(exact_seen.values())
    if not rows:
        raise ValueError("dedup produced no documents; inspect filter thresholds or input shards")
    rows.sort(key=lambda r: (str(r.get("source_family", "")), str(r.get("doc_hash", ""))))
    for idx, doc in enumerate(rows):
        doc["cluster_id"] = f"atproto_cluster_{idx:012d}"
        doc["cluster_size"] = 1
    count, size = write_jsonl_gz(output / "dedup_000000.jsonl.gz", rows)
    stats.records_out = count
    stats.bytes_out = size
    stats.dropped = dict(dropped)
    stats.finished_at_utc = utc_now()
    write_stage_marker(output, stats, {"near_dedup": args.near_dedup})


def command_split(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output = Path(args.output)
    ensure_dir(output)
    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = StageStats(stage="split", started_at_utc=utc_now())
    split_counts = Counter()

    for doc in iter_jsonl(input_path):
        stats.records_in += 1
        split = stable_split(doc["doc_hash"], args.split_seed, args.validation_ratio, args.test_ratio)
        if doc.get("quality_tier") == "q0_reject":
            split = "reject"
        if split == "test" and not args.write_test:
            split = "validation"
        doc["split"] = split
        buffers[split].append(doc)
        split_counts[split] += 1
        stats.records_out += 1

    for split, rows in buffers.items():
        if not rows:
            continue
        rows.sort(key=lambda r: r["doc_hash"])
        _, size = write_jsonl_gz(output / f"{split}.jsonl.gz", rows)
        stats.bytes_out += size
    if split_counts.get("train", 0) == 0:
        raise ValueError("split produced no train documents")
    if args.validation_ratio > 0 and split_counts.get("validation", 0) == 0:
        raise ValueError("split produced no validation documents; lower ratios need more input documents")
    stats.finished_at_utc = utc_now()
    write_stage_marker(output, stats, {"split_seed": args.split_seed, "split_counts": dict(split_counts)})


def token_chunks(
    docs: Iterable[dict[str, Any]],
    tokenizer: Any,
    block_size: int,
    token_budget: int,
) -> Iterator[dict[str, Any]]:
    eos = tokenizer.eos_token_id
    buffer: list[int] = []
    emitted = 0
    doc_count = 0
    quality_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    for doc in docs:
        text = doc.get("text")
        if not isinstance(text, str) or not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        buffer.extend(int(x) for x in ids)
        if eos is not None:
            buffer.append(int(eos))
        doc_count += 1
        quality_counter[str(doc.get("quality_tier", "unknown"))] += 1
        source_counter[str(doc.get("source_family", "unknown"))] += 1
        while len(buffer) >= block_size and emitted + block_size <= token_budget:
            chunk = buffer[:block_size]
            del buffer[:block_size]
            emitted += block_size
            yield {
                "input_ids": chunk,
                "source_mix_id": "atproto",
                "quality_tier": "mixed",
                "doc_count": doc_count,
                "token_count": block_size,
            }
        if emitted + block_size > token_budget:
            break


def command_tokenize_pack(args: argparse.Namespace) -> None:
    try:
        from datasets import Dataset, DatasetDict, Features, Sequence, Value
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("tokenize-pack requires 'datasets' and 'transformers'.") from exc

    input_path = Path(args.input)
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "source_mix_id": Value("string"),
            "quality_tier": Value("string"),
            "doc_count": Value("int32"),
            "token_count": Value("int32"),
        }
    )

    train_file = input_path / "train.jsonl.gz"
    val_file = input_path / "validation.jsonl.gz"
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError("split directory must contain train.jsonl.gz and validation.jsonl.gz")

    stats = StageStats(stage="tokenize_pack", started_at_utc=utc_now())
    train = Dataset.from_generator(
        lambda: token_chunks(iter_jsonl(train_file), tokenizer, args.block_size, args.train_tokens),
        features=features,
    )
    validation = Dataset.from_generator(
        lambda: token_chunks(iter_jsonl(val_file), tokenizer, args.block_size, args.validation_tokens),
        features=features,
    )
    dataset = DatasetDict({"train": train, "validation": validation})
    dataset.save_to_disk(str(output))
    stats.records_out = len(train) + len(validation)
    stats.finished_at_utc = utc_now()

    card = build_data_card(args, output, tokenizer, train, validation)
    card_path = output.with_suffix(".dataset-card.json")
    atomic_write_json(card_path, card)
    write_stage_marker(output, stats, {"dataset_card": str(card_path)})


def build_data_card(args: argparse.Namespace, output: Path, tokenizer: Any, train: Any, validation: Any) -> dict[str, Any]:
    build_command = " ".join(sys.argv)
    return {
        "name": output.name,
        "created_at_utc": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": os.environ.get("NAIME_GIT_COMMIT", "unknown"),
        "tokenizer": {
            "name": str(args.tokenizer_path),
            "path": str(args.tokenizer_path),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer)),
            "add_special_tokens": False,
            "eos_token_id": int(tokenizer.eos_token_id),
        },
        "block_size": args.block_size,
        "train_tokens": int(len(train) * args.block_size),
        "validation_tokens": int(len(validation) * args.block_size),
        "splits": {
            "train": {"rows": len(train), "tokens": int(len(train) * args.block_size)},
            "validation": {"rows": len(validation), "tokens": int(len(validation) * args.block_size)},
        },
        "source_mix": [{"source_id": "atproto/jetstream", "license": args.license, "ratio": 1.0}],
        "filters": {
            "min_text_chars": args.min_text_chars,
            "dedup": "exact+near_hash_bucket",
            "language_policy": args.language_policy,
            "privacy_filter": "email_phone_redacted;user_ids_hashed",
        },
        "split_seed": args.split_seed,
        "build_command": build_command,
        "build_host": socket.gethostname(),
        "notes": "ATProto public social text. Keep separate from release-grade corpora unless license/privacy review passes.",
    }


def command_validate(args: argparse.Namespace) -> None:
    try:
        from datasets import load_from_disk
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("validate requires 'datasets' and 'transformers'.") from exc
    dataset_path = Path(args.dataset)
    ds = load_from_disk(str(dataset_path))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    output = Path(args.output) if args.output else dataset_path.with_suffix(".validation-report.json")
    report: dict[str, Any] = {
        "dataset": str(dataset_path),
        "tokenizer": str(args.tokenizer_path),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer)),
        "block_size": args.block_size,
        "splits": {},
        "bad_token_rows": 0,
        "decode_sample_pass": True,
    }
    for split in ["train", "validation"]:
        if split not in ds:
            raise AssertionError(f"missing split: {split}")
        split_ds = ds[split]
        rows = len(split_ds)
        report["splits"][split] = {"rows": rows, "tokens": rows * args.block_size}
        if rows == 0:
            report["bad_token_rows"] += 1
            continue
        for idx in random.Random(args.sample_seed).sample(range(rows), min(args.sample_rows, rows)):
            ids = split_ds[int(idx)]["input_ids"]
            if len(ids) != args.block_size:
                report["bad_token_rows"] += 1
                continue
            if min(ids) < 0 or max(ids) >= report["vocab_size"]:
                report["bad_token_rows"] += 1
            text = tokenizer.decode(ids[: min(128, len(ids))])
            if not text.strip():
                report["decode_sample_pass"] = False
    report["accepted"] = report["bad_token_rows"] == 0 and report["decode_sample_pass"]
    atomic_write_json(output, report)
    if not report["accepted"]:
        raise SystemExit(f"dataset validation failed; see {output}")


def add_common_io(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NAIME ATProto data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-jetstream", help="Capture public Jetstream events into raw JSONL.GZ shards.")
    p.add_argument("--output", required=True)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--collections", nargs="*", default=DEFAULT_COLLECTIONS)
    p.add_argument("--cursor", type=int, default=0)
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument("--max-seconds", type=int, default=0)
    p.add_argument("--max-output-gib", type=float, default=0.0)
    p.add_argument("--stop-file", default="")
    p.add_argument("--shard-events", type=int, default=50_000)
    p.add_argument("--progress-events", type=int, default=10_000)
    p.add_argument("--progress-seconds", type=int, default=60)
    p.add_argument("--cursor-events", type=int, default=10_000)
    p.add_argument("--cursor-seconds", type=int, default=60)
    p.add_argument("--max-message-size-bytes", type=int, default=1_000_000)
    p.add_argument("--websocket-max-size", type=int, default=8_000_000)
    p.add_argument("--license", default="research-only")
    p.add_argument("--language-hint", default="multi")
    p.set_defaults(func=command_fetch_jetstream)

    p = sub.add_parser("normalize", help="Extract and normalize public text documents from raw Jetstream shards.")
    add_common_io(p)
    p.add_argument("--license", default="research-only")
    p.add_argument("--language-hint", default="multi")
    p.add_argument("--output-shard-docs", type=int, default=100_000)
    p.set_defaults(func=command_normalize)

    p = sub.add_parser("import-jsonl", help="Import local JSONL/JSONL.GZ text rows into normalized NAIME documents.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--text-fields", default="text,content,body,post.text,record.text,commit.record.text")
    p.add_argument("--id-fields", default="id,uri,cid,did,rkey,doc_id")
    p.add_argument("--source-family", default="atproto")
    p.add_argument("--source-name", default="local-jsonl")
    p.add_argument("--license", default="research-only")
    p.add_argument("--language-hint", default="multi")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--max-text-gib", type=float, default=0.0)
    p.add_argument("--output-shard-docs", type=int, default=100_000)
    p.add_argument("--auto-text-field", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress-rows", type=int, default=100_000)
    p.add_argument("--progress-seconds", type=int, default=60)
    p.add_argument("--stop-file", default="")
    p.set_defaults(func=command_import_jsonl)

    p = sub.add_parser("import-hf-dataset", help="Stream a Hugging Face dataset into normalized NAIME documents.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--text-fields", default="text,content,body,post.text,record.text,commit.record.text")
    p.add_argument("--id-fields", default="id,uri,cid,did,rkey,doc_id")
    p.add_argument("--source-family", default="atproto")
    p.add_argument("--license", default="research-only")
    p.add_argument("--language-hint", default="multi")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--max-text-gib", type=float, default=0.0)
    p.add_argument("--output-shard-docs", type=int, default=100_000)
    p.add_argument("--auto-text-field", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress-rows", type=int, default=100_000)
    p.add_argument("--progress-seconds", type=int, default=60)
    p.add_argument("--stop-file", default="")
    p.add_argument("--data-files", default="")
    p.add_argument("--revision", default="")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    p.set_defaults(func=command_import_hf_dataset)

    p = sub.add_parser("filter", help="Apply privacy, quality, and length filters.")
    add_common_io(p)
    p.add_argument("--min-text-chars", type=int, default=40)
    p.add_argument("--max-text-chars", type=int, default=4096)
    p.add_argument("--min-quality-score", type=float, default=0.25)
    p.add_argument("--output-shard-docs", type=int, default=100_000)
    p.set_defaults(func=command_filter)

    p = sub.add_parser("dedup", help="Exact and near-bucket deduplicate filtered documents.")
    add_common_io(p)
    p.add_argument("--near-dedup", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=command_dedup)

    p = sub.add_parser("split", help="Stable hash train/validation/test split.")
    add_common_io(p)
    p.add_argument("--split-seed", "--seed", dest="split_seed", default="4321")
    p.add_argument("--validation-ratio", type=float, default=0.01)
    p.add_argument("--test-ratio", type=float, default=0.0)
    p.add_argument("--write-test", action="store_true")
    p.set_defaults(func=command_split)

    p = sub.add_parser("tokenize-pack", help="Tokenize and pack split docs into HF disk DatasetDict.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer-path", default="data/naime/gpt2")
    p.add_argument("--block-size", type=int, default=1025)
    p.add_argument("--train-tokens", type=int, default=50_000_000)
    p.add_argument("--validation-tokens", type=int, default=2_000_000)
    p.add_argument("--min-text-chars", type=int, default=40)
    p.add_argument("--language-policy", default="multi")
    p.add_argument("--license", default="research-only")
    p.add_argument("--split-seed", default="4321")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=command_tokenize_pack)

    p = sub.add_parser("validate", help="Validate a packed HF disk dataset.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="")
    p.add_argument("--tokenizer-path", default="data/naime/gpt2")
    p.add_argument("--block-size", type=int, default=1025)
    p.add_argument("--sample-rows", type=int, default=100)
    p.add_argument("--sample-seed", type=int, default=2026)
    p.set_defaults(func=command_validate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
