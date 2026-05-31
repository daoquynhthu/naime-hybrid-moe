from .interventions import InterventionSpec
from .report_builder import build_trace_summary, write_trace_artifacts
from .trace_config import TraceConfig
from .trace_context import TraceContext
from .training_dynamics import (
    append_training_dynamics_event,
    build_training_dynamics_event,
    collect_gradient_component_stats,
    diagnostics_root,
    flatten_gradient_component_stats,
)


def run_state_packet_diagnostics(*args, **kwargs):
    from .packet_diagnostics import run_state_packet_diagnostics as _run_state_packet_diagnostics

    return _run_state_packet_diagnostics(*args, **kwargs)


__all__ = [
    "InterventionSpec",
    "TraceConfig",
    "TraceContext",
    "append_training_dynamics_event",
    "build_trace_summary",
    "build_training_dynamics_event",
    "collect_gradient_component_stats",
    "diagnostics_root",
    "flatten_gradient_component_stats",
    "run_state_packet_diagnostics",
    "write_trace_artifacts",
]
