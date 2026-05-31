from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    return None


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    number = _as_float(value)
    if number is not None:
        out[prefix] = number
        return
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(next_prefix, item, out)


def _read_events(path: Path) -> list[dict[str, Any]]:
    event_path = path / "dynamics_events.jsonl" if path.is_dir() else path
    if not event_path.exists():
        raise FileNotFoundError(f"training dynamics events not found: {event_path}")
    events: list[dict[str, Any]] = []
    with event_path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _series_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "first": values[0],
        "latest": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "delta": values[-1] - values[0],
    }


def _summarize_series(events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    series: dict[str, list[float]] = defaultdict(list)
    for event in events:
        flat: dict[str, float] = {}
        for section in ("core", "router", "state", "packet"):
            _flatten_numeric(section, event.get(section, {}), flat)
        for key, value in flat.items():
            series[key].append(value)
    return {key: _series_summary(values) for key, values in sorted(series.items())}


def _summarize_gradients(events: list[dict[str, Any]]) -> dict[str, Any]:
    component_peaks: dict[str, float] = defaultdict(float)
    component_latest: dict[str, float] = {}
    loss_component_peaks: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    low_cosines: list[dict[str, Any]] = []
    for event in events:
        gradients = event.get("gradients", {})
        total_norm = gradients.get("grad_component_total_norm", {})
        if isinstance(total_norm, dict):
            component_latest = {str(k): float(v) for k, v in total_norm.items() if _as_float(v) is not None}
            for group, value in component_latest.items():
                component_peaks[group] = max(component_peaks[group], abs(value))
        loss_norms = gradients.get("loss_grad_component_norm", {})
        if isinstance(loss_norms, dict):
            for loss_name, group_values in loss_norms.items():
                if not isinstance(group_values, dict):
                    continue
                for group, value in group_values.items():
                    number = _as_float(value)
                    if number is not None:
                        loss_component_peaks[str(loss_name)][str(group)] = max(
                            loss_component_peaks[str(loss_name)][str(group)],
                            abs(number),
                        )
        loss_cosines = gradients.get("loss_grad_component_cosine", {})
        if isinstance(loss_cosines, dict):
            for loss_name, group_values in loss_cosines.items():
                if not isinstance(group_values, dict):
                    continue
                for group, value in group_values.items():
                    cosine = _as_float(value)
                    if cosine is not None and cosine < 0.25:
                        low_cosines.append(
                            {
                                "step": event.get("step"),
                                "loss": loss_name,
                                "component": group,
                                "cosine": cosine,
                            }
                        )
    return {
        "component_latest_total_norm": component_latest,
        "component_peak_total_norm": dict(sorted(component_peaks.items(), key=lambda item: item[1], reverse=True)),
        "loss_component_peak_norm": {
            loss: dict(sorted(groups.items(), key=lambda item: item[1], reverse=True))
            for loss, groups in sorted(loss_component_peaks.items())
        },
        "low_alignment": sorted(low_cosines, key=lambda item: item["cosine"])[:20],
    }


def _build_warnings(events: list[dict[str, Any]], series: dict[str, dict[str, float]], gradients: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    bad_grad_events = [event for event in events if event.get("phase") == "bad_grad_skip"]
    if bad_grad_events:
        warnings.append(f"bad_grad_skip events detected: {len(bad_grad_events)}")
    grad_max = series.get("core.grad_norm", {}).get("max")
    if grad_max is not None and grad_max > 20.0:
        warnings.append(f"grad_norm exceeded bad-gradient threshold: max={grad_max:.4g}")
    packet_gain = series.get("packet.diagnostics_boundary_gain", {}).get("latest")
    if packet_gain is not None and packet_gain < 0.0:
        warnings.append(f"latest boundary packet carry gain is negative: {packet_gain:.6g}")
    low_alignment = gradients.get("low_alignment", [])
    if low_alignment:
        worst = low_alignment[0]
        warnings.append(
            "low loss-gradient alignment detected: "
            f"{worst['loss']}->{worst['component']} cosine={worst['cosine']:.4g} step={worst['step']}"
        )
    return warnings


def build_training_diagnostics_report(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    events = _read_events(root)
    phase_counts = Counter(str(event.get("phase", "unknown")) for event in events)
    steps = [int(event["step"]) for event in events if isinstance(event.get("step"), int)]
    series = _summarize_series(events)
    gradients = _summarize_gradients(events)
    report = {
        "event_count": len(events),
        "first_step": min(steps) if steps else None,
        "latest_step": max(steps) if steps else None,
        "phase_counts": dict(sorted(phase_counts.items())),
        "series": series,
        "gradients": gradients,
    }
    report["warnings"] = _build_warnings(events, series, gradients)
    return report


def _top_items(mapping: dict[str, float], limit: int = 8) -> list[str]:
    return [f"{key}={value:.4g}" for key, value in list(mapping.items())[:limit]]


def render_training_diagnostics_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Training Diagnostics Report",
        "",
        f"- events: {report['event_count']}",
        f"- steps: {report['first_step']} -> {report['latest_step']}",
        f"- phases: {report['phase_counts']}",
        "",
        "## Warnings",
    ]
    warnings = report.get("warnings") or []
    lines.extend([f"- {item}" for item in warnings] if warnings else ["- none"])
    lines.extend(["", "## Key Series"])
    for key in (
        "core.loss_lm",
        "core.grad_norm",
        "router.router_entropy",
        "state.v7_latent_delta",
        "state.v7_world_delta",
        "state.v7_self_delta",
        "packet.diagnostics_boundary_gain",
    ):
        item = report.get("series", {}).get(key)
        if item:
            lines.append(
                f"- {key}: latest={item['latest']:.6g}, min={item['min']:.6g}, "
                f"max={item['max']:.6g}, delta={item['delta']:.6g}"
            )
    gradients = report.get("gradients", {})
    lines.extend(["", "## Gradient Components"])
    peaks = gradients.get("component_peak_total_norm", {})
    lines.append("- peak total norm: " + (", ".join(_top_items(peaks)) if peaks else "none"))
    lines.extend(["", "## Loss Component Peaks"])
    loss_peaks = gradients.get("loss_component_peak_norm", {})
    if loss_peaks:
        for loss_name, groups in loss_peaks.items():
            lines.append(f"- {loss_name}: " + ", ".join(_top_items(groups, limit=6)))
    else:
        lines.append("- none")
    low_alignment = gradients.get("low_alignment", [])
    lines.extend(["", "## Low Alignment"])
    if low_alignment:
        for item in low_alignment[:10]:
            lines.append(
                f"- step {item['step']}: {item['loss']}->{item['component']} cosine={item['cosine']:.6g}"
            )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_training_diagnostics_report(path: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(path)
    target = Path(output_dir) if output_dir else (root if root.is_dir() else root.parent)
    target.mkdir(parents=True, exist_ok=True)
    report = build_training_diagnostics_report(root)
    json_path = target / "diagnostics_report.json"
    md_path = target / "diagnostics_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_training_diagnostics_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NAIME training-time diagnostics events.")
    parser.add_argument("path", help="Diagnostics directory or dynamics_events.jsonl path.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    outputs = write_training_diagnostics_report(args.path, args.output_dir)
    print(f"summary_json={outputs['json']}")
    print(f"summary_markdown={outputs['markdown']}")


if __name__ == "__main__":
    main()
