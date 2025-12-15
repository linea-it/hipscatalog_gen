from __future__ import annotations

import time
from typing import Any, Dict, List

import healpy as hp
import numpy as np
import pandas as pd
from dask import compute as dask_compute

from .io_output import build_header_line_from_keep
from .mag_global import _quantile_from_histogram, compute_mag_histogram_ddf
from .pipeline_common import write_tiles_with_allsky
from .utils import _fmt_dur, _get_meta_df, _log_depth_stats

__all__ = ["prepare_mag_global", "run_mag_global_selection"]


def prepare_mag_global(ddf: Any, cfg: Any, diag_ctx, log_fn):
    """Add __mag__ column and restrict to the configured magnitude window."""
    algo = cfg.algorithm
    mag_col_cfg = getattr(algo, "mag_column", None)
    if not mag_col_cfg:
        raise ValueError(
            "algorithm.selection_mode='mag_global' requires "
            "algorithm.mag_column to be set to the name of a magnitude column."
        )

    if mag_col_cfg not in ddf.columns:
        raise KeyError(f"Configured mag_column '{mag_col_cfg}' not found in input columns.")

    base_meta_mag = _get_meta_df(ddf)
    meta_with_mag = base_meta_mag.copy()
    meta_with_mag["__mag__"] = pd.Series([], dtype="float64")

    def _add_mag_column(pdf: pd.DataFrame, mag_col_name: str) -> pd.DataFrame:
        if pdf.empty:
            pdf["__mag__"] = pd.Series([], dtype="float64")
            return pdf
        pdf = pdf.copy()
        pdf["__mag__"] = pd.to_numeric(pdf[mag_col_name], errors="coerce")
        return pdf

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

    if not np.isfinite(mag_min_global_raw) or not np.isfinite(mag_max_global_raw):
        raise ValueError(
            "mag_global selection: global magnitude min/max are not finite. "
            "Check the magnitude column values."
        )

    if mag_min_global_raw >= mag_max_global_raw:
        raise ValueError(
            f"mag_global selection: invalid global magnitude range "
            f"[{mag_min_global_raw}, {mag_max_global_raw}]."
        )

    mag_completeness = bool(getattr(algo, "mag_completeness", False))
    if mag_completeness:
        mag_min = mag_min_global_raw
        mag_max = mag_max_global_raw
        mag_upper_for_hist = mag_max_global_raw

        log_fn(
            f"[mag_global] mag_completeness=True → using full magnitude range: "
            f"[{mag_min}, {mag_max}] (raw catalog bounds).",
            always=True,
        )

        algo.mag_min = mag_min
        algo.mag_max = mag_max
    else:
        if mag_min_cfg is None:
            raw = mag_min_global_raw
            mag_min = max(raw, -2.0)
            if mag_min != raw:
                log_fn(
                    "[mag_global] mag_min not provided; using global minimum "
                    f"{raw:.4f} clipped to {mag_min:.4f} (>= -2).",
                    always=True,
                )
            else:
                log_fn(
                    "[mag_global] mag_min not provided; using global minimum " f"magnitude {mag_min:.4f}.",
                    always=True,
                )
        else:
            mag_min = float(mag_min_cfg)

        mag_upper_for_hist = min(mag_max_global_raw, 40.0)
        if mag_upper_for_hist <= mag_min:
            raise ValueError(
                "mag_global selection: after applying automatic bounds, "
                f"mag_min={mag_min:.4f} is not smaller than the upper histogram "
                f"limit={mag_upper_for_hist:.4f}. Check magnitude values or "
                "configured mag_min."
            )

        if mag_max_cfg is None:
            with diag_ctx("dask_mag_hist_auto_max"):
                hist_auto, edges_auto, n_tot_auto = compute_mag_histogram_ddf(
                    ddf_like=ddf,
                    mag_col=mag_col_internal,
                    mag_min=mag_min,
                    mag_max=mag_upper_for_hist,
                    nbins=algo.mag_hist_nbins,
                )

            if n_tot_auto == 0:
                raise ValueError(
                    "mag_global selection: no objects found when estimating "
                    "automatic mag_max. Check the magnitude column and range."
                )

            peak_idx = int(np.argmax(hist_auto))
            bin_left = float(edges_auto[peak_idx])
            bin_right = float(edges_auto[peak_idx + 1])
            peak_center = 0.5 * (bin_left + bin_right)

            peak_center = min(peak_center, mag_upper_for_hist)
            mag_max = float(np.round(peak_center, 2))

            log_fn(
                "[mag_global] mag_max not provided; using histogram peak at "
                f"{mag_max:.2f} (bin center from [{bin_left:.4f}, "
                f"{bin_right:.4f}], with mag < 40).",
                always=True,
            )
        else:
            mag_max = float(mag_max_cfg)

        if mag_min >= mag_max:
            raise ValueError(
                f"algorithm.mag_min ({mag_min}) must be strictly smaller than "
                f"algorithm.mag_max ({mag_max}) for mag_global selection."
            )

        algo.mag_min = mag_min
        algo.mag_max = mag_max

    meta_sel = meta_with_mag.copy()

    def _filter_mag_window(
        pdf: pd.DataFrame,
        mag_min_val: float,
        mag_max_val: float,
    ) -> pd.DataFrame:
        if pdf.empty:
            return pdf
        m = pd.to_numeric(pdf[mag_col_internal], errors="coerce")
        mask = (m >= mag_min_val) & (m <= mag_max_val)
        return pdf.loc[mask]

    ddf_sel = ddf.map_partitions(
        _filter_mag_window,
        mag_min,
        mag_max,
        meta=meta_sel,
    )
    return ddf_sel


def _assign_targets_per_depth(
    densmaps: Dict[int, np.ndarray],
    depths_sel: List[int],
    algo: Any,
    cdf_hist: np.ndarray,
    mag_edges_hist: np.ndarray,
    mag_min: float,
    mag_max: float,
    n_tot_mag: float,
    log_fn,
) -> np.ndarray:
    weights_list: List[float] = []
    for d in depths_sel:
        counts_d = densmaps[d]
        tiles_active = int((counts_d > 0).sum())
        weights_list.append(max(1, tiles_active))

    weights = np.asarray(weights_list, dtype="float64")
    T = np.zeros_like(weights, dtype="float64")

    fixed_targets: Dict[int, float] = {}
    for d, n_val in (
        (1, getattr(algo, "n_1", None)),
        (2, getattr(algo, "n_2", None)),
        (3, getattr(algo, "n_3", None)),
    ):
        if (d in depths_sel) and (n_val is not None):
            if int(n_val) < 0:
                raise ValueError(f"algorithm.n_{d} must be non-negative if provided (got {n_val}).")
            fixed_targets[d] = float(int(n_val))

    sum_fixed = float(sum(fixed_targets.values()))
    if sum_fixed > n_tot_mag and sum_fixed > 0.0:
        scale = float(n_tot_mag) / sum_fixed if sum_fixed > 0 else 0.0
        log_fn(
            "[mag_global] Sum of fixed targets n_1/n_2/n_3 "
            f"({int(sum_fixed)}) exceeds the total number of objects "
            f"in the magnitude range ({int(n_tot_mag)}). "
            f"Rescaling n_1/n_2/n_3 by a factor {scale:.3f}.",
            always=True,
        )
        for d in list(fixed_targets.keys()):
            fixed_targets[d] *= scale
        sum_fixed = float(n_tot_mag)

    for d, val in fixed_targets.items():
        idx = depths_sel.index(d)
        T[idx] = val

    N_rem = max(0.0, float(n_tot_mag) - sum_fixed)
    if N_rem > 0.0:
        free_mask = np.ones_like(weights, dtype=bool)
        for d in fixed_targets:
            idx = depths_sel.index(d)
            free_mask[idx] = False

        W_free = float(weights[free_mask].sum())
        if W_free <= 0.0:
            n_free = int(free_mask.sum())
            if n_free > 0:
                T[free_mask] += N_rem / float(n_free)
        else:
            T[free_mask] += weights[free_mask] / W_free * N_rem

    T_cum = np.cumsum(T)
    Q = T_cum / float(n_tot_mag) if n_tot_mag > 0.0 else np.zeros_like(T_cum, dtype="float64")

    level_edges: np.ndarray = np.empty(len(depths_sel) + 1, dtype="float64")
    level_edges[0] = mag_min
    for i, q in enumerate(Q, start=1):
        level_edges[i] = _quantile_from_histogram(cdf_hist, mag_edges_hist, q)

    level_edges = np.maximum.accumulate(level_edges)
    level_edges[0] = mag_min
    level_edges[-1] = mag_max
    return level_edges


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
) -> None:
    """Execute the mag_global selection path and write tiles."""
    algo = cfg.algorithm
    mag_col_internal = "__mag__"
    if algo.mag_min is None or algo.mag_max is None:
        raise RuntimeError("mag_global: internal error — mag_min/mag_max should have been set earlier.")

    mag_min = float(algo.mag_min)
    mag_max = float(algo.mag_max)
    depths_sel = list(range(1, cfg.algorithm.level_limit + 1))

    with diag_ctx("dask_mag_hist"):
        hist, mag_edges_hist, n_tot_mag = compute_mag_histogram_ddf(
            remainder_ddf,
            mag_col=mag_col_internal,
            mag_min=mag_min,
            mag_max=mag_max,
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

    level_edges = _assign_targets_per_depth(
        densmaps=densmaps,
        depths_sel=depths_sel,
        algo=algo,
        cdf_hist=cdf_hist,
        mag_edges_hist=mag_edges_hist,
        mag_min=mag_min,
        mag_max=mag_max,
        n_tot_mag=float(n_tot_mag),
        log_fn=log_fn,
    )

    log_fn(
        "[selection] mag_global mode: per-depth magnitude slices:\n"
        + "\n".join(
            f"  depth {d}: [{level_edges[i]:.6f}, {level_edges[i+1]:.6f}"
            f"{')' if d != depths_sel[-1] else ']'}"
            for i, d in enumerate(depths_sel)
        ),
        always=True,
    )

    header_line = build_header_line_from_keep(keep_cols)

    for i, depth in enumerate(depths_sel):
        depth_t0 = time.time()
        m_lo = level_edges[i]
        m_hi = level_edges[i + 1]

        with diag_ctx(f"dask_depth_mag_{depth:02d}"):
            if depth != depths_sel[-1]:
                mag_mask = (remainder_ddf[mag_col_internal] >= m_lo) & (
                    remainder_ddf[mag_col_internal] < m_hi
                )
            else:
                mag_mask = (remainder_ddf[mag_col_internal] >= m_lo) & (
                    remainder_ddf[mag_col_internal] <= m_hi
                )

            depth_ddf = remainder_ddf[mag_mask]
            selected_pdf = depth_ddf.compute()
            _log_depth_stats(
                log_fn,
                depth,
                "selected",
                counts=densmaps[depth],
                selected_len=len(selected_pdf),
            )

            if len(selected_pdf) == 0:
                log_fn(
                    f"[DEPTH {depth}] mag_global: no rows in "
                    f"magnitude slice [{m_lo:.6f}, {m_hi:.6f}] → skipping.",
                    always=True,
                )
                log_fn(
                    f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
                    always=True,
                )
                continue

            ra_vals = pd.to_numeric(selected_pdf[ra_col], errors="coerce").to_numpy()
            dec_vals = pd.to_numeric(selected_pdf[dec_col], errors="coerce").to_numpy()

            theta = np.deg2rad(90.0 - dec_vals)
            phi = np.deg2rad(ra_vals % 360.0)

            NSIDE_L = 1 << depth
            ipixL = hp.ang2pix(NSIDE_L, theta, phi, nest=True).astype(np.int64)
            selected_pdf["__ipix__"] = ipixL

            counts = densmaps[depth]
            allsky_needed = depth in (1, 2)

            written_per_ipix, _ = write_tiles_with_allsky(
                out_dir=out_dir,
                depth=depth,
                header_line=header_line,
                ra_col=ra_col,
                dec_col=dec_col,
                counts=counts,
                selected=selected_pdf,
                order_desc=cfg.algorithm.order_desc,
                allsky_needed=allsky_needed,
                log_fn=log_fn,
            )
            _log_depth_stats(log_fn, depth, "written", counts=densmaps[depth], written=written_per_ipix)

        log_fn(
            f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
            always=True,
        )
