"""Slice selections by value or score with HEALPix-aware ordering."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import healpy as hp
import numpy as np
import pandas as pd

from ..io.output import build_header_line_from_keep
from ..pipeline.common import write_tiles_with_allsky
from ..utils import _fmt_dur, _log_depth_stats
from .levels import assign_level_edges
from .score import compute_score_histogram_ddf

__all__ = ["select_by_value_slices", "select_by_score_slices"]


def select_by_value_slices(
    remainder_ddf: Any,
    densmaps: Dict[int, np.ndarray],
    depths_sel: Sequence[int],
    keep_cols: List[str],
    ra_col: str,
    dec_col: str,
    value_col: str,
    order_desc: bool,
    label: str,
    out_dir,
    diag_ctx,
    log_fn,
    *,
    level_edges: np.ndarray | None = None,
    tie_col: str | None = None,
    compute_hist_fn=None,
    value_min: float | None = None,
    value_max: float | None = None,
    hist_nbins: int | None = None,
    fixed_targets: Dict[int, float] | None = None,
    hist_diag_ctx_name: str | None = None,
    depth_diag_prefix: str | None = None,
) -> None:
    """Slice by per-depth value ranges and write tiles; expects value_col and densmaps populated."""
    if level_edges is None:
        if compute_hist_fn is None or hist_nbins is None or value_min is None or value_max is None:
            raise ValueError(
                f"{label}: missing histogram parameters (compute_hist_fn, hist_nbins, value_min, value_max)."
            )

        hist_ctx = hist_diag_ctx_name or f"dask_{label}_hist"
        with diag_ctx(hist_ctx):
            hist, edges_hist, n_tot = compute_hist_fn(
                remainder_ddf,
                value_col,
                value_min,
                value_max,
                hist_nbins,
            )

        if n_tot == 0:
            log_fn(
                f"[selection] {label}: no objects found in the range "
                f"[{value_min}, {value_max}] → nothing to select.",
                always=True,
            )
            return

        cdf_hist = hist.cumsum().astype("float64")
        if cdf_hist[-1] > 0:
            cdf_hist /= float(cdf_hist[-1])
        else:
            cdf_hist[:] = 0.0

        level_edges, _ = assign_level_edges(
            densmaps=densmaps,
            depths_sel=list(depths_sel),
            fixed_targets=fixed_targets or {},
            cdf_hist=cdf_hist,
            score_edges_hist=edges_hist,
            score_min=value_min,
            score_max=value_max,
            n_tot_score=float(n_tot),
            log_fn=log_fn,
            label=label,
        )
    else:
        level_edges = np.asarray(level_edges, dtype="float64")

    depths_list = list(depths_sel)
    log_fn(
        f"[selection] {label} mode: per-depth slices:\n"
        + "\n".join(
            f"  depth {d}: [{level_edges[i]:.6f}, {level_edges[i + 1]:.6f}"
            f"{')' if d != depths_list[-1] else ']'}"
            for i, d in enumerate(depths_list)
        ),
        always=True,
    )

    header_line = build_header_line_from_keep(keep_cols)
    depth_ctx = depth_diag_prefix or f"dask_{label}_depth"

    for i, depth in enumerate(depths_list):
        depth_t0 = time.time()
        v_lo = level_edges[i]
        v_hi = level_edges[i + 1]

        with diag_ctx(f"{depth_ctx}_{depth:02d}"):
            if depth != depths_list[-1]:
                val_mask = (remainder_ddf[value_col] >= v_lo) & (remainder_ddf[value_col] < v_hi)
            else:
                val_mask = (remainder_ddf[value_col] >= v_lo) & (remainder_ddf[value_col] <= v_hi)

            depth_ddf = remainder_ddf[val_mask]
            selected_pdf = depth_ddf.compute()
            if tie_col and tie_col not in selected_pdf.columns:
                raise KeyError(f"{label}: tie_column '{tie_col}' not found in selected data.")
            _log_depth_stats(
                log_fn,
                depth,
                "selected",
                counts=densmaps[depth],
                selected_len=len(selected_pdf),
            )

            if len(selected_pdf) == 0:
                log_fn(
                    f"[DEPTH {depth}] {label}: no rows in slice [{v_lo:.6f}, {v_hi:.6f}] → skipping.",
                    always=True,
                    depth=depth,
                )
                log_fn(
                    f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
                    always=True,
                    depth=depth,
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

            sort_cols = [value_col]
            ascending = [not order_desc]
            if tie_col and tie_col in selected_pdf.columns:
                sort_cols.append(tie_col)
                ascending.append(True)
            if ra_col in selected_pdf.columns:
                sort_cols.append(ra_col)
                ascending.append(True)
            if dec_col in selected_pdf.columns:
                sort_cols.append(dec_col)
                ascending.append(True)
            selected_pdf = selected_pdf.sort_values(sort_cols, ascending=ascending, kind="mergesort")

            written_per_ipix, _ = write_tiles_with_allsky(
                out_dir=out_dir,
                depth=depth,
                header_line=header_line,
                ra_col=ra_col,
                dec_col=dec_col,
                counts=counts,
                selected=selected_pdf,
                order_desc=order_desc,
                allsky_needed=allsky_needed,
                log_fn=log_fn,
            )
            _log_depth_stats(log_fn, depth, "written", counts=densmaps[depth], written=written_per_ipix)

        log_fn(
            f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
            always=True,
            depth=depth,
        )


def select_by_score_slices(
    remainder_ddf: Any,
    densmaps: Dict[int, np.ndarray],
    depths_sel: Sequence[int],
    keep_cols: List[str],
    ra_col: str,
    dec_col: str,
    score_col: str,
    score_min: float,
    score_max: float,
    hist_nbins: int,
    out_dir,
    diag_ctx,
    log_fn,
    label: str,
    order_desc: bool,
    fixed_targets: Dict[int, float] | None = None,
    hist_diag_ctx_name: str | None = None,
    depth_diag_prefix: str | None = None,
    tie_col: str | None = None,
) -> None:
    """Score-specialized wrapper around select_by_value_slices."""
    select_by_value_slices(
        remainder_ddf=remainder_ddf,
        densmaps=densmaps,
        depths_sel=depths_sel,
        keep_cols=keep_cols,
        ra_col=ra_col,
        dec_col=dec_col,
        value_col=score_col,
        order_desc=order_desc,
        label=label,
        out_dir=out_dir,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        compute_hist_fn=compute_score_histogram_ddf,
        value_min=score_min,
        value_max=score_max,
        hist_nbins=hist_nbins,
        fixed_targets=fixed_targets,
        hist_diag_ctx_name=hist_diag_ctx_name,
        depth_diag_prefix=depth_diag_prefix,
        tie_col=tie_col,
    )
