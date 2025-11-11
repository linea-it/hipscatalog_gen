from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
from dask import compute as dask_compute
from dask import delayed as _delayed


__all__ = [
    "compute_mag_histogram_ddf",
    "_quantile_from_histogram",
]


# =============================================================================
# Magnitude-based global selection helpers
# =============================================================================


def compute_mag_histogram_ddf(
    ddf_like: Any,
    mag_col: str,
    mag_min: float,
    mag_max: float,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute a 1D magnitude histogram in a Dask-friendly way.

    Uses map_partitions plus a single compute() call. Works with Dask
    DataFrames and LSDB catalogs.

    Args:
        ddf_like: Dask DataFrame or LSDB catalog with the magnitude column.
        mag_col: Magnitude column name.
        mag_min: Lower magnitude bound.
        mag_max: Upper magnitude bound.
        nbins: Number of histogram bins.

    Returns:
        Tuple (hist, bin_edges, n_total) where:
            hist: Counts per magnitude bin.
            bin_edges: Bin edges of length nbins + 1.
            n_total: Total rows within [mag_min, mag_max].
    """
    edges = np.linspace(mag_min, mag_max, nbins + 1, dtype="float64")

    def _part_hist(pdf: pd.DataFrame) -> tuple[np.ndarray, int]:
        if pdf is None or len(pdf) == 0:
            return np.zeros(nbins, dtype="int64"), 0

        vals = pd.to_numeric(pdf[mag_col], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.zeros(nbins, dtype="int64"), 0

        mask = (vals >= mag_min) & (vals <= mag_max)
        vals = vals[mask]
        if vals.size == 0:
            return np.zeros(nbins, dtype="int64"), 0

        h, _ = np.histogram(vals, bins=edges)
        return h.astype("int64"), int(vals.size)

    # Use only the magnitude column to keep partitions small.
    parts = ddf_like[[mag_col]].to_delayed()
    delayed_results = [_delayed(_part_hist)(p) for p in parts]

    def _sum_results(seq: List[tuple[np.ndarray, int]]) -> tuple[np.ndarray, int]:
        h_total = np.zeros(nbins, dtype="int64")
        n_total = 0
        for h, n in seq:
            h_total += h
            n_total += int(n)
        return h_total, n_total

    total = _delayed(_sum_results)(delayed_results)
    hist, n_total = dask_compute(total)[0]
    return hist, edges, int(n_total)


def _quantile_from_histogram(
    cdf: np.ndarray,
    bin_edges: np.ndarray,
    q: float,
) -> float:
    """Invert a 1D histogram CDF into a magnitude threshold.

    Args:
        cdf: Monotonic cumulative distribution in [0, 1].
        bin_edges: Histogram bin edges (len = len(cdf) + 1).
        q: Target quantile in [0, 1].

    Returns:
        Magnitude threshold such that CDF(mag_thresh) ~= q.
    """
    if not len(cdf):
        return float(bin_edges[0])

    q = float(np.clip(q, 0.0, 1.0))

    if q <= 0.0:
        return float(bin_edges[0])
    if q >= 1.0:
        return float(bin_edges[-1])

    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = max(0, min(idx, len(cdf) - 1))
    # Return the left edge of the bin where the CDF crosses q.
    return float(bin_edges[idx])
