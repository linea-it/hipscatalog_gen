"""Registry of selection modes and helpers to resolve them by name."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from ..mag_global.pipeline import (
    normalize_mag_global,
    prepare_mag_global,
    run_mag_global_selection,
)
from ..score_density_hybrid.pipeline import (
    normalize_score_density_hybrid,
    prepare_score_density_hybrid,
    run_score_density_hybrid_selection,
)
from ..score_global.pipeline import normalize_score_global, prepare_score_global, run_score_global_selection
from .validation import (
    validate_mag_global_cfg,
    validate_score_density_hybrid_cfg,
    validate_score_global_cfg,
)

PrepareFn = Callable[..., Any]
RunFn = Callable[..., None]
NormalizeFn = Callable[..., Tuple[Any, Any]]
ValidateFn = Callable[[Any], None]


@dataclass(frozen=True)
class SelectionMode:
    """Registry entry for a selection mode."""

    name: str
    validate_fn: ValidateFn
    normalize_fn: NormalizeFn
    prepare_fn: PrepareFn
    run_fn: RunFn
    description: str


MODE_REGISTRY: Dict[str, SelectionMode] = {
    "mag_global": SelectionMode(
        name="mag_global",
        validate_fn=validate_mag_global_cfg,
        normalize_fn=normalize_mag_global,
        prepare_fn=prepare_mag_global,
        run_fn=run_mag_global_selection,
        description="Magnitude-complete slices across all depths.",
    ),
    "score_global": SelectionMode(
        name="score_global",
        validate_fn=validate_score_global_cfg,
        normalize_fn=normalize_score_global,
        prepare_fn=prepare_score_global,
        run_fn=run_score_global_selection,
        description="Score-based slices across all depths.",
    ),
    "score_density_hybrid": SelectionMode(
        name="score_density_hybrid",
        validate_fn=validate_score_density_hybrid_cfg,
        normalize_fn=normalize_score_density_hybrid,
        prepare_fn=prepare_score_density_hybrid,
        run_fn=run_score_density_hybrid_selection,
        description="Density-driven depths 1–3, score-based distribution afterwards.",
    ),
}


def get_selection_mode(name: str) -> SelectionMode:
    """Return a selection mode entry, raising on unsupported names.

    Args:
        name: Selection mode identifier (e.g., ``mag_global``).

    Returns:
        ``SelectionMode`` instance with validators and handlers.

    Raises:
        ValueError: If the name is not registered.
    """
    key = (name or "").lower()
    if key not in MODE_REGISTRY:
        opts = ", ".join(f"{k} ({v.description})" for k, v in sorted(MODE_REGISTRY.items()))
        raise ValueError(f"Unsupported selection_mode '{name}'. Available modes: {opts}.")
    return MODE_REGISTRY[key]
