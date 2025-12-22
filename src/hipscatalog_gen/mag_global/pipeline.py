"""Pipeline steps for magnitude-complete selection."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dask import compute as dask_compute

from ..pipeline.common import maybe_persist_ddf
from ..pipeline.params import MagGlobalParams
from ..selection.levels import assign_level_edges
from ..selection.score import (
    _finite_min_max,
    _map_invalid_to_sentinel,
    _sentinel_for_order,
    compute_histogram_ddf,
)
from ..selection.slicing import select_by_value_slices
from ..utils import _get_meta_df

__all__ = ["normalize_mag_global", "prepare_mag_global", "run_mag_global_selection"]

# Use math.log to avoid numpy dependency during module import (Sphinx mocks may lack it).
MAG_CONV = math.log(10.0) * 0.4


def normalize_mag_global(
    ddf: Any,
    cfg: Any,
    diag_ctx,
    log_fn,
    persist_ddfs: bool = False,
    avoid_computes: bool = True,
) -> tuple[Any, MagGlobalParams]:
    """Add ``__mag__`` column and compute selection window without mutating cfg.

    Args:
        ddf: Dask-like collection with magnitude or flux columns.
        cfg: Parsed configuration object.
        diag_ctx: Diagnostics context factory.
        log_fn: Logging callback.
        persist_ddfs: Whether to persist intermediate DDFs.
        avoid_computes: Whether to avoid explicit ``compute()`` calls when possible.

    Returns:
        Tuple ``(ddf_with_mag, MagGlobalParams)``.

    Raises:
        ValueError: When magnitude/flux configuration is invalid or missing.
        KeyError: If configured magnitude/flux columns are not present.
    """
    algo = cfg.algorithm
    mag_col_cfg = getattr(algo, "mag_column", None)
    flux_col_cfg = getattr(algo, "flux_column", None)

    if mag_col_cfg and flux_col_cfg:
        raise ValueError("mag_column and flux_column are mutually exclusive for mag_global mode.")

    if flux_col_cfg:
        if flux_col_cfg not in ddf.columns:
            raise KeyError(f"Configured flux_column '{flux_col_cfg}' not found in input columns.")
        mag_offset = getattr(algo, "mag_offset", None)
        if mag_offset is None:
            raise ValueError("mag_global selection with flux_column requires algorithm.mag_offset to be set.")
    elif mag_col_cfg:
        if mag_col_cfg not in ddf.columns:
            raise KeyError(f"Configured mag_column '{mag_col_cfg}' not found in input columns.")
    else:
        raise ValueError(
            "mag_global selection requires either algorithm.mag_column or algorithm.flux_column."
        )

    base_meta_mag = _get_meta_df(ddf)
    meta_with_mag = base_meta_mag.copy()
    meta_with_mag["__mag__"] = pd.Series([], dtype="float64")

    def _add_mag_column(pdf: pd.DataFrame, mag_col_name: str) -> pd.DataFrame:
        """Copy a magnitude column into the internal __mag__ field."""
        if pdf.empty:
            pdf["__mag__"] = pd.Series([], dtype="float64")
            return pdf
        pdf = pdf.copy()
        pdf["__mag__"] = pd.to_numeric(pdf[mag_col_name], errors="coerce")
        return pdf

    def _add_mag_from_flux(pdf: pd.DataFrame, flux_col_name: str, mag_offset_val: float) -> pd.DataFrame:
        """Derive magnitude from flux and store in the internal __mag__ field."""
        if pdf.empty:
            pdf["__mag__"] = pd.Series([], dtype="float64")
            return pdf
        pdf = pdf.copy()
        flux = pd.to_numeric(pdf[flux_col_name], errors="coerce")
        mag_arr = np.full(len(flux), 99.0, dtype="float64")
        valid = flux > 0
        if valid.any():
            mag_arr[valid.to_numpy()] = -2.5 * np.log10(flux[valid]) + float(mag_offset_val)
        pdf["__mag__"] = mag_arr
        return pdf

    if flux_col_cfg:
        ddf = ddf.map_partitions(
            _add_mag_from_flux,
            flux_col_cfg,
            float(algo.mag_offset),
            meta=meta_with_mag,
        )
    else:
        ddf = ddf.map_partitions(
            _add_mag_column,
            mag_col_cfg,
            meta=meta_with_mag,
        )

    mag_col_internal = "__mag__"
    mag_min_cfg = getattr(algo, "mag_min", None)
    mag_max_cfg = getattr(algo, "mag_max", None)

    with diag_ctx("dask_mag_minmax"):
        mag_min_global_raw, mag_max_global_raw = dask_compute(
            ddf[mag_col_internal].min(),
            ddf[mag_col_internal].max(),
        )

    if mag_min_global_raw is None or mag_max_global_raw is None:
        raise ValueError(
            "mag_global selection: unable to determine global magnitude "
            "range (min/max returned None). Check the magnitude column."
        )

    mag_min_global_raw = float(mag_min_global_raw)
    mag_max_global_raw = float(mag_max_global_raw)

    keep_invalid = bool(getattr(algo, "mag_keep_invalid_values", False))
    sentinel_mag: float | None = None
    range_mode = str(getattr(algo, "mag_adaptive_range", "complete")).lower()
    order_desc = bool(getattr(algo, "mg_order_desc", getattr(algo, "order_desc", False)))
    if (not keep_invalid) and (not np.isfinite(mag_min_global_raw) or not np.isfinite(mag_max_global_raw)):
        raise ValueError(
            "mag_global selection: global magnitude min/max are not finite. "
            "Check the magnitude column values."
        )

    if mag_min_global_raw >= mag_max_global_raw:
        raise ValueError(
            f"mag_global selection: invalid global magnitude range "
            f"[{mag_min_global_raw}, {mag_max_global_raw}]."
        )
    if range_mode not in {"complete", "hist_peak"}:
        raise ValueError(
            f"mag_global selection: invalid mag_adaptive_range '{range_mode}'. "
            "Allowed values are: 'complete', 'hist_peak'."
        )
    if keep_invalid and range_mode != "complete":
        raise ValueError(
            "mag_global: keep_invalid_values=True is only supported with mag_adaptive_range=complete."
        )

    def _histogram_peak(
        lower: float,
        upper: float,
        ctx_name: str,
    ) -> tuple[float, float, float]:
        """Return (peak_center, bin_left, bin_right) for the given range."""
        if upper <= lower:
            raise ValueError(
                f"mag_global selection: invalid histogram bounds [{lower}, {upper}]. "
                "Upper bound must be larger than lower bound."
            )

        with diag_ctx(ctx_name):
            hist, edges, n_tot = compute_histogram_ddf(
                ddf_like=ddf,
                value_col=mag_col_internal,
                value_min=lower,
                value_max=upper,
                nbins=algo.mag_hist_nbins,
            )

        if n_tot == 0:
            raise ValueError(
                "mag_global selection: no objects found when estimating "
                "histogram peak. Check the magnitude column and configured bounds."
            )

        peak_idx = int(np.argmax(hist))
        bin_left = float(edges[peak_idx])
        bin_right = float(edges[peak_idx + 1])
        peak_center = 0.5 * (bin_left + bin_right)
        return float(np.round(peak_center, 2)), bin_left, bin_right

    if keep_invalid:
        # Complete-mode only: map NaN/Inf to an integer sentinel just outside the finite range.
        fin_min, fin_max = _finite_min_max(ddf, "__mag__")
        if fin_min is None or fin_max is None:
            raise ValueError("mag_global: all magnitude values are NaN/Inf; nothing to select.")
        sentinel_mag = _sentinel_for_order(fin_min, fin_max, order_desc)
        mag_min = float(mag_min_cfg) if mag_min_cfg is not None else float(fin_min)
        mag_max = float(mag_max_cfg) if mag_max_cfg is not None else float(fin_max)
        if not order_desc:
            mag_max = max(mag_max, sentinel_mag)
        else:
            mag_min = min(mag_min, sentinel_mag)

        ddf = _map_invalid_to_sentinel(
            ddf,
            "__mag__",
            sentinel=sentinel_mag,
            meta=meta_with_mag,
        )
        log_fn(
            f"[mag_global] keep_invalid_values=True → mapping NaN/Inf to sentinel {sentinel_mag} "
            f"and using range [{mag_min}, {mag_max}].",
            always=True,
        )
    elif range_mode == "complete":
        if mag_min_cfg is not None and mag_max_cfg is not None:
            mag_min = float(mag_min_cfg)
            mag_max = float(mag_max_cfg)
            log_fn(
                "[mag_global] mag_adaptive_range=complete with explicit mag_min/mag_max "
                f"→ using [{mag_min}, {mag_max}].",
                always=True,
            )
        elif mag_min_cfg is not None:
            mag_min = float(mag_min_cfg)
            mag_max = mag_max_global_raw
            log_fn(
                "[mag_global] mag_adaptive_range=complete and mag_min provided "
                f"→ mag_min={mag_min}, mag_max={mag_max_global_raw} (global maximum).",
                always=True,
            )
        elif mag_max_cfg is not None:
            mag_min = mag_min_global_raw
            mag_max = float(mag_max_cfg)
            log_fn(
                "[mag_global] mag_adaptive_range=complete and mag_max provided "
                f"→ mag_min={mag_min_global_raw} (global minimum), mag_max={mag_max}.",
                always=True,
            )
        else:
            mag_min = mag_min_global_raw
            mag_max = mag_max_global_raw
            log_fn(
                "[mag_global] mag_adaptive_range=complete with no bounds provided "
                f"→ using full global range [{mag_min}, {mag_max}].",
                always=True,
            )
    else:  # hist_peak
        if mag_min_cfg is not None and mag_max_cfg is not None:
            mag_min = float(mag_min_cfg)
            mag_max = float(mag_max_cfg)
            log_fn(
                "[mag_global] mag_adaptive_range=hist_peak with explicit mag_min/mag_max "
                f"→ using [{mag_min}, {mag_max}] (skipping histogram fill).",
                always=True,
            )
        elif mag_min_cfg is not None:
            mag_min = float(mag_min_cfg)
            hist_upper = min(mag_max_global_raw, 40.0)
            mag_max, bin_left, bin_right = _histogram_peak(
                mag_min,
                hist_upper,
                "dask_mag_hist_peak_from_min",
            )
            log_fn(
                "[mag_global] mag_adaptive_range=hist_peak and mag_min provided "
                f"→ mag_min={mag_min}, mag_max from histogram peak={mag_max} "
                f"(bin center from [{bin_left:.4f}, {bin_right:.4f}], clipped at < 40).",
                always=True,
            )
        elif mag_max_cfg is not None:
            mag_max = float(mag_max_cfg)
            hist_lower = -2.0
            mag_min, bin_left, bin_right = _histogram_peak(
                hist_lower,
                mag_max,
                "dask_mag_hist_peak_from_max",
            )
            log_fn(
                "[mag_global] mag_adaptive_range=hist_peak and mag_max provided "
                f"→ mag_min from histogram peak={mag_min} "
                f"(bin center from [{bin_left:.4f}, {bin_right:.4f}], clipped at > -2), "
                f"mag_max={mag_max}.",
                always=True,
            )
        else:
            raw_min = mag_min_global_raw
            mag_min = max(raw_min, -2.0)
            hist_upper = min(mag_max_global_raw, 40.0)
            mag_max, bin_left, bin_right = _histogram_peak(
                -2.0,
                hist_upper,
                "dask_mag_hist_peak_from_none",
            )
            log_fn(
                "[mag_global] mag_adaptive_range=hist_peak with no bounds provided "
                f"→ mag_min={mag_min} (global minimum clipped to >= -2), "
                f"mag_max from histogram peak={mag_max} "
                f"(bin center from [{bin_left:.4f}, {bin_right:.4f}], clipped to [-2, 40]).",
                always=True,
            )

    if mag_min >= mag_max:
        raise ValueError(
            f"algorithm.mag_min ({mag_min}) must be strictly smaller than "
            f"algorithm.mag_max ({mag_max}) for mag_global selection."
        )

    sentinel_val = sentinel_mag if keep_invalid else None
    params = MagGlobalParams(mag_min=mag_min, mag_max=mag_max, sentinel=sentinel_val)
    return ddf, params


def prepare_mag_global(
    ddf: Any,
    cfg: Any,
    diag_ctx,
    log_fn,
    params: MagGlobalParams,
    persist_ddfs: bool = False,
    avoid_computes: bool = True,
):
    """Restrict to the configured magnitude window using pre-computed params.

    Args:
        ddf: Dask-like collection with ``__mag__`` already attached.
        cfg: Parsed configuration object.
        diag_ctx: Diagnostics context factory.
        log_fn: Logging callback.
        params: Resolved magnitude parameters.
        persist_ddfs: Whether to persist the filtered DDF.
        avoid_computes: Whether to avoid explicit ``compute()`` calls when possible.

    Returns:
        Dask-like collection filtered to the magnitude window.
    """
    mag_col_internal = "__mag__"
    meta_sel = _get_meta_df(ddf).copy()
    meta_sel["__mag__"] = pd.Series([], dtype="float64")

    def _filter_mag_window(
        pdf: pd.DataFrame,
        mag_min_val: float,
        mag_max_val: float,
    ) -> pd.DataFrame:
        """Keep rows within the resolved magnitude window for one partition."""
        if pdf.empty:
            return pdf
        m = pd.to_numeric(pdf[mag_col_internal], errors="coerce")
        mask = (m >= mag_min_val) & (m <= mag_max_val)
        return pdf.loc[mask]

    ddf_sel = ddf.map_partitions(
        _filter_mag_window,
        params.mag_min,
        params.mag_max,
        meta=meta_sel,
    )

    should_persist = persist_ddfs or (not avoid_computes)
    reason = "cluster.persist_ddfs=True" if persist_ddfs else "avoid_computes_wherever_possible=False"
    ddf_sel = maybe_persist_ddf(
        ddf_sel,
        should_persist=should_persist,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        log_prefix="mag_global",
        diag_label="dask_mag_persist_filtered",
        reason=reason,
    )

    return ddf_sel


def run_mag_global_selection(
    remainder_ddf: Any,
    densmaps: Dict[int, np.ndarray],
    keep_cols: List[str],
    ra_col: str,
    dec_col: str,
    cfg: Any,
    out_dir,
    diag_ctx,
    log_fn,
    avoid_computes: bool = True,
    params: MagGlobalParams | None = None,
) -> None:
    """Execute the mag_global selection path and write tiles.

    Args:
        remainder_ddf: Dask-like collection after pre-filtering.
        densmaps: Mapping depth -> densmap counts.
        keep_cols: Ordered list of columns to keep in tiles.
        ra_col: Name of the RA column.
        dec_col: Name of the DEC column.
        cfg: Parsed configuration object.
        out_dir: Output directory for HiPS tiles.
        diag_ctx: Diagnostics context factory.
        log_fn: Logging callback.
        avoid_computes: Whether to avoid explicit ``compute()`` calls when possible.
        params: Optional resolved magnitude parameters (auto-resolved when None).
    """
    algo = cfg.algorithm
    mag_col_internal = "__mag__"
    if params is None:
        if getattr(algo, "mag_min", None) is None or getattr(algo, "mag_max", None) is None:
            raise RuntimeError("mag_global: selection parameters were not provided.")
        params = MagGlobalParams(mag_min=float(algo.mag_min), mag_max=float(algo.mag_max))

    mag_min = float(params.mag_min)
    mag_max = float(params.mag_max)
    depths_sel = list(range(1, cfg.algorithm.level_limit + 1))
    tie_col = getattr(cfg.algorithm, "mag_tie_column", None) or getattr(cfg.algorithm, "tie_column", None)

    with diag_ctx("dask_mag_hist"):
        hist, mag_edges_hist, n_tot_mag = compute_histogram_ddf(
            ddf_like=remainder_ddf,
            value_col=mag_col_internal,
            value_min=mag_min,
            value_max=mag_max,
            nbins=algo.mag_hist_nbins,
        )

    if n_tot_mag == 0:
        log_fn(
            "[selection] mag_global: no objects found in the magnitude range "
            f"[{mag_min}, {mag_max}] → nothing to select.",
            always=True,
        )
        return

    cdf_hist = hist.cumsum().astype("float64")
    if cdf_hist[-1] > 0:
        cdf_hist /= float(cdf_hist[-1])
    else:
        cdf_hist[:] = 0.0

    fixed_targets: Dict[int, float] = {}
    for d, n_val, k_val in (
        (1, getattr(algo, "n_1", None), getattr(algo, "k_1", None)),
        (2, getattr(algo, "n_2", None), getattr(algo, "k_2", None)),
        (3, getattr(algo, "n_3", None), getattr(algo, "k_3", None)),
    ):
        if (n_val is not None) and (k_val is not None):
            raise ValueError(f"mag_global: both n_{d} and k_{d} are set; choose one.")
        if k_val is not None:
            active_tiles = int(np.count_nonzero(densmaps.get(d, [])))
            n_val = int(round(float(k_val) * active_tiles))
        if (d in depths_sel) and (n_val is not None):
            fixed_targets[d] = float(n_val)

    level_edges, _ = assign_level_edges(
        densmaps=densmaps,
        depths_sel=depths_sel,
        fixed_targets=fixed_targets,
        cdf_hist=cdf_hist,
        score_edges_hist=mag_edges_hist,
        score_min=mag_min,
        score_max=mag_max,
        n_tot_score=float(n_tot_mag),
        log_fn=log_fn,
        label="mag_global",
    )

    order_desc = bool(getattr(cfg.algorithm, "mg_order_desc", getattr(cfg.algorithm, "order_desc", False)))

    select_by_value_slices(
        remainder_ddf=remainder_ddf,
        densmaps=densmaps,
        depths_sel=depths_sel,
        keep_cols=keep_cols,
        ra_col=ra_col,
        dec_col=dec_col,
        value_col=mag_col_internal,
        order_desc=order_desc,
        label="mag_global",
        out_dir=out_dir,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        level_edges=level_edges,
        depth_diag_prefix="dask_depth_mag",
        tie_col=tie_col,
    )
