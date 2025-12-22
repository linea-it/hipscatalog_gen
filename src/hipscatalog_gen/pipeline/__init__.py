"""Pipeline coordination, validation, and logging utilities.

Imports are kept lazy here to avoid circular imports during documentation builds.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "MODE_REGISTRY",
    "PipelineContext",
    "PipelineStage",
    "SelectionMode",
    "get_selection_mode",
    "run_pipeline",
    "run_stages",
    "setup_structured_logger",
    "validate_common_cfg",
    "validate_mag_global_cfg",
    "validate_score_density_hybrid_cfg",
    "validate_score_global_cfg",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy import shim
    if name == "run_pipeline":
        return import_module(".main", __name__).run_pipeline
    if name in {"MODE_REGISTRY", "SelectionMode", "get_selection_mode"}:
        mod = import_module(".modes", __name__)
        return getattr(mod, name)
    if name in {"PipelineContext", "PipelineStage", "run_stages"}:
        mod = import_module(".structure", __name__)
        return getattr(mod, name)
    if name == "setup_structured_logger":
        return import_module(".logging_utils", __name__).setup_structured_logger
    if name in {
        "validate_common_cfg",
        "validate_mag_global_cfg",
        "validate_score_density_hybrid_cfg",
        "validate_score_global_cfg",
    }:
        mod = import_module(".validation", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
