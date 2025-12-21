from __future__ import annotations

import time
from typing import Any, Dict, List

import healpy as hp
import numpy as np
import pandas as pd

from ..io.output import build_header_line_from_keep
from ..pipeline.common import write_tiles_with_allsky
from ..selection.common import assign_level_edges
from ..selection.score import add_score_column, resolve_value_range
from ..utils import _fmt_dur, _get_meta_df, _log_depth_stats
from .utils import compute_score_histogram_ddf

__all__ = ["prepare_score_global", "run_score_global_selection"]


def prepare_score_global(
    ddf: Any,
    cfg: Any,
    diag_ctx,
    log_fn,
    persist_ddfs: bool = False,
    avoid_computes: bool = True,
):
    """Add __score__ column (from an expression) and restrict to a score window."""
    algo = cfg.algorithm
    score_expr = getattr(algo, "score_column", None)
    if not score_expr:
        raise ValueError("score_global selection requires algorithm.score_column to be set.")

    score_expr = str(score_expr)
    score_range_mode = str(getattr(algo, "score_adaptive_range", "complete") or "complete").lower()
    if score_range_mode not in ("complete", "hist_peak"):
        raise ValueError("algorithm.score_adaptive_range must be 'complete' or 'hist_peak'.")

    score_hist_nbins = int(getattr(algo, "score_hist_nbins", getattr(algo, "mag_hist_nbins", 512)))
    if score_hist_nbins <= 0:
        raise ValueError("algorithm.score_hist_nbins must be a positive integer.")

    ddf = add_score_column(ddf, score_expr, output_col="__score__")

    score_col_internal = "__score__"
    score_min_cfg = getattr(algo, "score_min", None)
    score_max_cfg = getattr(algo, "score_max", None)
    score_min, score_max = resolve_value_range(
        ddf=ddf,
        value_col=score_col_internal,
        range_mode=score_range_mode,
        min_cfg=score_min_cfg,
        max_cfg=score_max_cfg,
        hist_nbins=score_hist_nbins,
        compute_hist_fn=compute_score_histogram_ddf,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        label="score_global",
    )

    algo.score_min = score_min
    algo.score_max = score_max

    meta_sel = _get_meta_df(ddf).copy()
    meta_sel["__score__"] = pd.Series([], dtype="float64")

    def _filter_score_window(
        pdf: pd.DataFrame,
        score_min_val: float,
        score_max_val: float,
    ) -> pd.DataFrame:
        if pdf.empty:
            return pdf
        s = pd.to_numeric(pdf[score_col_internal], errors="coerce")
        mask = (s >= score_min_val) & (s <= score_max_val)
        return pdf.loc[mask]

    ddf_sel = ddf.map_partitions(
        _filter_score_window,
        score_min,
        score_max,
        meta=meta_sel,
    )

    should_persist = persist_ddfs or (not avoid_computes)
    if should_persist and hasattr(ddf_sel, "persist"):
        reason = "cluster.persist_ddfs=True" if persist_ddfs else "avoid_computes_wherever_possible=False"
        log_fn(f"[score_global] Persisting filtered DDF in memory ({reason}).", always=True)
        with diag_ctx("dask_score_persist_filtered"):
            ddf_sel = ddf_sel.persist()
            try:
                from dask.distributed import wait
            except Exception:
                wait = None  # type: ignore[assignment]
            if wait is not None:
                wait(ddf_sel)
    return ddf_sel


def run_score_global_selection(
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
) -> None:
    """Execute the score_global selection path and write tiles."""
    algo = cfg.algorithm
    score_col_internal = "__score__"
    if algo.score_min is None or algo.score_max is None:
        raise RuntimeError("score_global: internal error — score_min/score_max should have been set earlier.")

    score_min = float(algo.score_min)
    score_max = float(algo.score_max)
    depths_sel = list(range(1, cfg.algorithm.level_limit + 1))

    with diag_ctx("dask_score_hist"):
        hist, score_edges_hist, n_tot_score = compute_score_histogram_ddf(
            remainder_ddf,
            score_col=score_col_internal,
            score_min=score_min,
            score_max=score_max,
            nbins=algo.score_hist_nbins,
        )

    if n_tot_score == 0:
        log_fn(
            "[selection] score_global: no objects found in the score range "
            f"[{score_min}, {score_max}] → nothing to select.",
            always=True,
        )
        return

    cdf_hist = hist.cumsum().astype("float64")
    if cdf_hist[-1] > 0:
        cdf_hist /= float(cdf_hist[-1])
    else:
        cdf_hist[:] = 0.0

    fixed_targets: Dict[int, float] = {}
    for d, n_val in (
        (1, getattr(algo, "score_n_1", None)),
        (2, getattr(algo, "score_n_2", None)),
        (3, getattr(algo, "score_n_3", None)),
    ):
        if (d in depths_sel) and (n_val is not None):
            fixed_targets[d] = float(n_val)

    level_edges, _ = assign_level_edges(
        densmaps=densmaps,
        depths_sel=depths_sel,
        fixed_targets=fixed_targets,
        cdf_hist=cdf_hist,
        score_edges_hist=score_edges_hist,
        score_min=score_min,
        score_max=score_max,
        n_tot_score=float(n_tot_score),
        log_fn=log_fn,
        label="score_global",
    )

    log_fn(
        "[selection] score_global mode: per-depth score slices:\n"
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
        s_lo = level_edges[i]
        s_hi = level_edges[i + 1]

        with diag_ctx(f"dask_depth_score_{depth:02d}"):
            if depth != depths_sel[-1]:
                score_mask = (remainder_ddf[score_col_internal] >= s_lo) & (
                    remainder_ddf[score_col_internal] < s_hi
                )
            else:
                score_mask = (remainder_ddf[score_col_internal] >= s_lo) & (
                    remainder_ddf[score_col_internal] <= s_hi
                )

            depth_ddf = remainder_ddf[score_mask]
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
                    f"[DEPTH {depth}] score_global: no rows in "
                    f"score slice [{s_lo:.6f}, {s_hi:.6f}] → skipping.",
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
                order_desc=cfg.algorithm.sg_order_desc,
                allsky_needed=allsky_needed,
                log_fn=log_fn,
            )
            _log_depth_stats(log_fn, depth, "written", counts=densmaps[depth], written=written_per_ipix)

        log_fn(
            f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
            always=True,
        )
