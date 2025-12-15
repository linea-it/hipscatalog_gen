from __future__ import annotations

from typing import Any

from ..mag_global.utils import _quantile_from_histogram, compute_mag_histogram_ddf

__all__ = [
    "compute_score_histogram_ddf",
    "_quantile_from_histogram",
]


def compute_score_histogram_ddf(
    ddf_like: Any,
    score_col: str,
    score_min: float,
    score_max: float,
    nbins: int,
):
    """Thin wrapper around the mag_global histogram helper for generic scores."""
    return compute_mag_histogram_ddf(ddf_like, score_col, score_min, score_max, nbins)
