from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd
from dask import compute as dask_compute

from ..utils import _get_meta_df

# Type alias for histogram function used by range resolution.
HistFn = Callable[[Any, str, float, float, int], Tuple[np.ndarray, np.ndarray, int]]


def add_score_column(ddf: Any, score_expr: str, output_col: str = "__score__") -> Any:
    """Attach a numeric score column derived from a column or expression."""
    score_expr = str(score_expr)
    base_meta = _get_meta_df(ddf)
    meta_with_score = base_meta.copy()
    meta_with_score[output_col] = pd.Series([], dtype="float64")

    code = compile(score_expr, "<score_expr>", "eval")

    def _add(pdf: pd.DataFrame, expr: str, compiled_expr) -> pd.DataFrame:
        if pdf.empty:
            pdf[output_col] = pd.Series([], dtype="float64")
            return pdf

        pdf = pdf.copy()
        if expr in pdf.columns:
            sc = pd.to_numeric(pdf[expr], errors="coerce")
        else:
            env = {"__builtins__": {}, "np": np, "numpy": np}
            env.update({col: pdf[col] for col in pdf.columns})
            out = eval(compiled_expr, env, {})
            sc = pd.to_numeric(out, errors="coerce")

        sc = sc.replace([np.inf, -np.inf], np.nan)
        pdf[output_col] = sc
        return pdf

    return ddf.map_partitions(_add, score_expr, code, meta=meta_with_score)


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
    """Resolve [min, max] for score-like columns with optional histogram peak."""
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
                raise RuntimeError(f"{label}: max must be provided when using hist_peak for min.")
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
                raise RuntimeError(f"{label}: min must be provided when using hist_peak for max.")
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
        raise RuntimeError(f"{label}: min/max resolution failed.")

    if not (np.isfinite(val_min) and np.isfinite(val_max)):
        raise ValueError(f"{label}: resolved min/max are not finite.")

    if val_min >= val_max:
        raise ValueError(f"{label}: min ({val_min}) must be strictly smaller than max ({val_max}).")

    return float(val_min), float(val_max)
