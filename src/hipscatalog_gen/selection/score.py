"""Score computations, histograms, and sentinel handling for selection modes."""

from __future__ import annotations

import math
from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd
from dask import compute as dask_compute
from dask import delayed as _delayed

from ..utils import _get_meta_df

# Type alias for histogram function used by range resolution.
HistFn = Callable[[Any, str, float, float, int], Tuple[np.ndarray, np.ndarray, int]]

__all__ = [
    "add_score_column",
    "compute_score_histogram_ddf",
    "compute_histogram_ddf",
    "_finite_min_max",
    "_sentinel_for_order",
    "_map_invalid_to_sentinel",
    "resolve_value_range",
    "_quantile_from_histogram",
]


def compute_score_histogram_ddf(
    ddf_like: Any,
    score_col: str,
    score_min: float,
    score_max: float,
    nbins: int,
    *,
    keep_invalid: bool = False,
    sentinel: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute a 1D histogram for score-like columns (Dask/LSDB friendly)."""
    return compute_histogram_ddf(
        ddf_like=ddf_like,
        value_col=score_col,
        value_min=score_min,
        value_max=score_max,
        nbins=nbins,
        keep_invalid=keep_invalid,
        sentinel=sentinel,
    )


def compute_histogram_ddf(
    ddf_like: Any,
    value_col: str,
    value_min: float,
    value_max: float,
    nbins: int,
    *,
    keep_invalid: bool = False,
    sentinel: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generic 1D histogram computation for Dask DataFrames or LSDB catalogs.

    Args:
        ddf_like: Dask-like collection or LSDB catalog with the target column.
        value_col: Column name to histogram.
        value_min: Lower bound (inclusive).
        value_max: Upper bound (inclusive).
        nbins: Number of bins.
        keep_invalid: When True, replace NaN/Inf with ``sentinel`` instead of dropping.
        sentinel: Sentinel value for invalid entries (used only when ``keep_invalid`` is True).

    Returns:
        Tuple of ``(hist, edges, n_total)`` where:
            - hist: numpy array with bin counts.
            - edges: numpy array with bin edges.
            - n_total: total number of rows inspected (including invalid rows).
    """
    edges = np.linspace(value_min, value_max, nbins + 1, dtype="float64")

    def _part_hist(pdf: pd.DataFrame) -> tuple[np.ndarray, int]:
        """Build histogram counts and totals for one partition."""
        if pdf is None or len(pdf) == 0:
            return np.zeros(nbins, dtype="int64"), 0

        vals = pd.to_numeric(pdf[value_col], errors="coerce").to_numpy()
        if vals.size == 0:
            return np.zeros(nbins, dtype="int64"), 0

        n_total = int(len(vals))
        finite_mask = np.isfinite(vals)
        vals_finite = vals[finite_mask]

        if keep_invalid and sentinel is not None:
            vals = vals.copy()
            vals[~finite_mask] = sentinel
        else:
            vals = vals_finite

        mask = (vals >= value_min) & (vals <= value_max)
        vals = vals[mask]
        if vals.size == 0:
            return np.zeros(nbins, dtype="int64"), n_total

        h, _ = np.histogram(vals, bins=edges)
        return h.astype("int64"), n_total

    parts = ddf_like[[value_col]].to_delayed()
    delayed_results = [_delayed(_part_hist)(p) for p in parts]

    def _sum_results(seq: List[tuple[np.ndarray, int]]) -> tuple[np.ndarray, int]:
        """Sum partition histograms into a total histogram and row count."""
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
    """Invert a 1D histogram CDF into a threshold with intra-bin interpolation."""
    if not len(cdf):
        return float(bin_edges[0])

    q = float(np.clip(q, 0.0, 1.0))

    if q <= 0.0:
        return float(bin_edges[0])
    if q >= 1.0:
        return float(bin_edges[-1])

    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = max(0, min(idx, len(cdf) - 1))

    cdf_left = float(cdf[idx - 1]) if idx > 0 else 0.0
    cdf_right = float(cdf[idx])
    edge_left = float(bin_edges[idx])
    edge_right = float(bin_edges[idx + 1])

    if cdf_right <= cdf_left:
        # Flat region (empty bin) — advance to the next increase if present.
        j = idx + 1
        while j < len(cdf) and float(cdf[j]) <= cdf_left:
            j += 1
        if j >= len(cdf):
            return float(bin_edges[-1])
        cdf_right = float(cdf[j])
        edge_right = float(bin_edges[j + 1])

    if cdf_right <= cdf_left:
        return edge_right  # pragma: no cover

    frac = (q - cdf_left) / (cdf_right - cdf_left)
    frac = float(np.clip(frac, 0.0, 1.0))
    return edge_left + frac * (edge_right - edge_left)


def _finite_min_max(ddf_like: Any, value_col: str) -> tuple[float | None, float | None]:
    """Return finite (min, max) ignoring NaN/Inf; (None, None) if no finite values."""

    def _part(pdf: pd.DataFrame) -> tuple[float | None, float | None]:
        """Return finite min/max for one partition."""
        vals = pd.to_numeric(pdf[value_col], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            return None, None
        return float(vals.min()), float(vals.max())

    parts = ddf_like[[value_col]].to_delayed()
    delayed_results = [_delayed(_part)(p) for p in parts]

    mins: List[float] = []
    maxs: List[float] = []
    for mn, mx in dask_compute(*delayed_results):
        if mn is not None and mx is not None:
            mins.append(float(mn))
            maxs.append(float(mx))

    if not mins or not maxs:
        return None, None
    return float(np.min(mins)), float(np.max(maxs))


def _sentinel_for_order(min_val: float, max_val: float, order_desc: bool) -> float:
    """Return a sentinel just outside the finite range, aligned to an integer."""
    if order_desc:
        sentinel = math.floor(min_val)
        if sentinel >= min_val:
            sentinel -= 1
    else:
        sentinel = math.ceil(max_val)
        if sentinel <= max_val:
            sentinel += 1
    return float(sentinel)


def _map_invalid_to_sentinel(
    ddf: Any, col: str, sentinel: float, extra_mask_fn=None, meta: pd.DataFrame | None = None
):
    """Replace non-finite (and optional extra mask) values with a sentinel."""
    meta_out = meta if meta is not None else _get_meta_df(ddf).copy()
    meta_out[col] = pd.Series([], dtype="float64")

    def _map(pdf: pd.DataFrame) -> pd.DataFrame:
        """Replace invalid values (and optional mask) with a sentinel."""
        if pdf.empty:
            pdf[col] = pd.Series([], dtype="float64")
            return pdf
        pdf = pdf.copy()
        vals = pd.to_numeric(pdf[col], errors="coerce")
        mask = ~np.isfinite(vals)
        if extra_mask_fn is not None:
            mask |= extra_mask_fn(vals)
        if mask.any():
            vals = vals.astype("float64")
            vals[mask.to_numpy()] = float(sentinel)
        pdf[col] = vals
        return pdf

    return ddf.map_partitions(_map, meta=meta_out)


def add_score_column(ddf: Any, score_expr: str, output_col: str = "__score__") -> Any:
    """Attach a numeric score column derived from a column or expression."""
    score_expr = str(score_expr).strip()
    base_meta = _get_meta_df(ddf)
    meta_with_score = base_meta.copy()
    meta_with_score[output_col] = pd.Series([], dtype="float64")

    def _add(pdf: pd.DataFrame, expr: str) -> pd.DataFrame:
        """Attach a numeric score column to one partition."""
        if pdf.empty:
            pdf[output_col] = pd.Series([], dtype="float64")
            return pdf

        pdf = pdf.copy()
        if expr in pdf.columns:
            sc = pd.to_numeric(pdf[expr], errors="coerce")
        else:
            # Evaluate the expression using pandas to avoid builtin eval; force Python engine
            # so we don't rely on optional numexpr being installed.
            env = {"np": np, "numpy": np}
            env.update({col: pdf[col] for col in pdf.columns})
            out = pd.eval(expr, local_dict=env, global_dict={}, engine="python", parser="python")
            sc = pd.to_numeric(out, errors="coerce")

        sc = sc.replace([np.inf, -np.inf], np.nan)
        pdf[output_col] = sc
        return pdf

    return ddf.map_partitions(_add, score_expr, meta=meta_with_score)


def resolve_value_range(
    ddf: Any,
    value_col: str,
    range_mode: str,
    min_cfg: float | None,
    max_cfg: float | None,
    hist_nbins: int,
    compute_hist_fn: HistFn,
    diag_ctx,
    log_fn,
    label: str,
) -> tuple[float, float]:
    """Resolve [min, max] for score-like columns with optional histogram peak.

    Args:
        ddf: Dask-like collection with the target column.
        value_col: Column name to inspect.
        range_mode: Either ``\"complete\"`` or ``\"hist_peak\"``.
        min_cfg: Optional configured minimum.
        max_cfg: Optional configured maximum.
        hist_nbins: Number of bins for histogram estimation.
        compute_hist_fn: Callable to compute histograms (signature-compatible with ``compute_histogram_ddf``).
        diag_ctx: Diagnostics context factory.
        log_fn: Logging callback.
        label: Human-readable label for logging and error messages.

    Returns:
        Tuple ``(min_value, max_value)`` resolved according to the mode.

    Raises:
        ValueError: When ranges are invalid, non-finite, or histogram estimation fails.
        RuntimeError: When required bounds are missing for the chosen mode.
    """
    if range_mode not in ("complete", "hist_peak"):
        raise ValueError(f"{label}: range_mode must be 'complete' or 'hist_peak'.")

    if hist_nbins <= 0:
        raise ValueError(f"{label}: hist_nbins must be a positive integer.")

    with diag_ctx(f"dask_{label}_minmax"):
        val_min_raw, val_max_raw = dask_compute(ddf[value_col].min(), ddf[value_col].max())

    if val_min_raw is None or val_max_raw is None:
        raise ValueError(f"{label}: unable to determine global range (min/max returned None).")

    val_min_raw = float(val_min_raw)
    val_max_raw = float(val_max_raw)

    if not (np.isfinite(val_min_raw) and np.isfinite(val_max_raw)):
        raise ValueError(f"{label}: global min/max are not finite.")

    if val_min_raw >= val_max_raw:
        raise ValueError(f"{label}: invalid global range [{val_min_raw}, {val_max_raw}].")

    def _hist_peak(lo: float, hi: float, ctx_name: str) -> tuple[float, float, float]:
        """Estimate histogram peak center within provided bounds."""
        with diag_ctx(ctx_name):
            hist, edges, n_tot = compute_hist_fn(ddf, value_col, lo, hi, hist_nbins)

        if n_tot == 0:
            raise ValueError(f"{label}: no objects found when estimating histogram peak.")

        peak_idx = int(np.argmax(hist))
        bin_left = float(edges[peak_idx])
        bin_right = float(edges[peak_idx + 1])
        peak_center = float(np.round(0.5 * (bin_left + bin_right), 6))
        return peak_center, bin_left, bin_right

    val_min: float | None = float(min_cfg) if min_cfg is not None else None
    val_max: float | None = float(max_cfg) if max_cfg is not None else None

    if val_min is None and val_max is None:
        if range_mode == "complete":
            val_min = val_min_raw
            val_max = val_max_raw
            log_fn(
                f"[{label}] min/max not provided; using global range [{val_min:.6f}, {val_max:.6f}].",
                always=True,
            )
        else:
            peak_center, bin_left, bin_right = _hist_peak(
                val_min_raw, val_max_raw, f"dask_{label}_hist_peak_auto"
            )
            val_min = val_min_raw
            val_max = peak_center
            log_fn(
                f"[{label}] min/max not provided; using global minimum {val_min:.6f} and histogram peak at "
                f"{val_max:.6f} (bin center from [{bin_left:.6f}, {bin_right:.6f}]).",
                always=True,
            )
    elif val_min is None:
        if range_mode == "complete":
            val_min = val_min_raw
            log_fn(
                f"[{label}] min not provided; using global minimum {val_min:.6f} (max={val_max}).",
                always=True,
            )
        else:
            peak_center, bin_left, bin_right = _hist_peak(
                val_min_raw,
                val_max if val_max is not None else val_max_raw,
                f"dask_{label}_hist_peak_min",
            )
            val_min = peak_center
            if val_max is None:
                raise RuntimeError(
                    f"{label}: max must be provided when using hist_peak for min."
                )  # pragma: no cover
            if val_min > float(val_max):
                raise ValueError(
                    f"{label}: histogram peak used as min ({val_min:.6f}) is greater than "
                    f"provided max ({val_max})."
                )
            log_fn(
                f"[{label}] min not provided; using histogram peak at {val_min:.6f} as minimum "
                f"(bin center from [{bin_left:.6f}, {bin_right:.6f}]).",
                always=True,
            )
    elif val_max is None:
        if range_mode == "complete":
            val_max = val_max_raw
            log_fn(
                f"[{label}] max not provided; using global maximum {val_max:.6f} (min={val_min}).",
                always=True,
            )
        else:
            peak_center, bin_left, bin_right = _hist_peak(
                val_min if val_min is not None else val_min_raw,
                val_max_raw,
                f"dask_{label}_hist_peak_max",
            )
            val_max = peak_center
            if val_min is None:
                raise RuntimeError(
                    f"{label}: min must be provided when using hist_peak for max."
                )  # pragma: no cover
            if float(val_min) > val_max:
                raise ValueError(
                    f"{label}: histogram peak used as max ({val_max:.6f}) is smaller than "
                    f"provided min ({val_min})."
                )
            log_fn(
                f"[{label}] max not provided; using histogram peak at {val_max:.6f} as maximum "
                f"(bin center from [{bin_left:.6f}, {bin_right:.6f}]).",
                always=True,
            )

    if val_min is None or val_max is None:
        raise RuntimeError(f"{label}: min/max resolution failed.")  # pragma: no cover

    if not (np.isfinite(val_min) and np.isfinite(val_max)):
        raise ValueError(f"{label}: resolved min/max are not finite.")

    if val_min >= val_max:
        raise ValueError(f"{label}: min ({val_min}) must be strictly smaller than max ({val_max}).")

    return float(val_min), float(val_max)
