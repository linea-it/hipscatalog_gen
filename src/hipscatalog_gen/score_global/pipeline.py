"""Pipeline steps for score-based selection across all depths."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..pipeline.common import maybe_persist_ddf
from ..pipeline.params import ScoreGlobalParams
from ..selection.score import (
    _finite_min_max,
    _map_invalid_to_sentinel,
    _sentinel_for_order,
    add_score_column,
    compute_score_histogram_ddf,
    resolve_value_range,
)
from ..selection.slicing import select_by_score_slices
from ..utils import _get_meta_df

__all__ = ["normalize_score_global", "prepare_score_global", "run_score_global_selection"]


def normalize_score_global(
    ddf: Any,
    cfg: Any,
    diag_ctx,
    log_fn,
    persist_ddfs: bool = False,
    avoid_computes: bool = True,
) -> tuple[Any, ScoreGlobalParams]:
    """Add ``__score__`` column and compute score window without mutating cfg.

    Args:
        ddf: Dask-like collection with score columns or expressions.
        cfg: Parsed configuration object.
        diag_ctx: Diagnostics context factory.
        log_fn: Logging callback.
        persist_ddfs: Whether to persist intermediate DDFs.
        avoid_computes: Whether to avoid explicit ``compute()`` calls when possible.

    Returns:
        Tuple ``(ddf_with_score, ScoreGlobalParams)``.

    Raises:
        ValueError: If the score expression/configuration is invalid.
    """
    algo = cfg.algorithm
    score_expr = getattr(algo, "score_column", None)
    if not score_expr:
        raise ValueError("score_global selection requires algorithm.score_column to be set.")

    score_expr = str(score_expr)
    score_range_mode = str(getattr(algo, "score_adaptive_range", "complete") or "complete").lower()
    if score_range_mode not in ("complete", "hist_peak"):
        raise ValueError("algorithm.score_adaptive_range must be 'complete' or 'hist_peak'.")

    score_hist_nbins = int(getattr(algo, "score_hist_nbins", getattr(algo, "mag_hist_nbins", 2048)))
    if score_hist_nbins <= 0:
        raise ValueError("algorithm.score_hist_nbins must be a positive integer.")

    ddf = add_score_column(ddf, score_expr, output_col="__score__")

    score_col_internal = "__score__"
    score_min_cfg = getattr(algo, "score_min", None)
    score_max_cfg = getattr(algo, "score_max", None)
    keep_invalid = bool(getattr(algo, "score_keep_invalid_values", False))
    if keep_invalid and score_range_mode != "complete":
        raise ValueError(
            "score_global: keep_invalid_values=True is only supported with score_adaptive_range=complete."
        )
    order_desc = bool(getattr(cfg.algorithm, "sg_order_desc", getattr(cfg.algorithm, "order_desc", False)))

    sentinel: float | None = None
    if keep_invalid:
        fin_min, fin_max = _finite_min_max(ddf, score_col_internal)
        if fin_min is None or fin_max is None:
            raise ValueError("score_global: all score values are NaN/Inf; nothing to select.")
        sentinel = _sentinel_for_order(fin_min, fin_max, order_desc)
        ddf = _map_invalid_to_sentinel(
            ddf,
            score_col_internal,
            sentinel=sentinel,
            meta=_get_meta_df(ddf),
        )
        score_min = float(score_min_cfg) if score_min_cfg is not None else float(fin_min)
        score_max = float(score_max_cfg) if score_max_cfg is not None else float(fin_max)
        if not order_desc:
            score_max = max(score_max, sentinel)
        else:
            score_min = min(score_min, sentinel)
        log_fn(
            f"[score_global] keep_invalid_values=True → mapping NaN/Inf to sentinel {sentinel} "
            f"and using range [{score_min}, {score_max}] (order_desc={order_desc}).",
            always=True,
        )
    else:
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

    params = ScoreGlobalParams(score_min=score_min, score_max=score_max, sentinel=sentinel)
    return ddf, params


def prepare_score_global(
    ddf: Any,
    cfg: Any,
    diag_ctx,
    log_fn,
    params: ScoreGlobalParams,
    persist_ddfs: bool = False,
    avoid_computes: bool = True,
):
    """Restrict to a score window using pre-computed params.

    Args:
        ddf: Dask-like collection with ``__score__`` already attached.
        cfg: Parsed configuration object.
        diag_ctx: Diagnostics context factory.
        log_fn: Logging callback.
        params: Resolved score parameters.
        persist_ddfs: Whether to persist the filtered DDF.
        avoid_computes: Whether to avoid explicit ``compute()`` calls when possible.

    Returns:
        Dask-like collection filtered to the score window.
    """
    score_col_internal = "__score__"

    meta_sel = _get_meta_df(ddf).copy()
    meta_sel["__score__"] = pd.Series([], dtype="float64")

    def _filter_score_window(
        pdf: pd.DataFrame,
        score_min_val: float,
        score_max_val: float,
    ) -> pd.DataFrame:
        """Keep rows whose score falls within the resolved window."""
        if pdf.empty:
            return pdf
        s = pd.to_numeric(pdf[score_col_internal], errors="coerce")
        mask = (s >= score_min_val) & (s <= score_max_val)
        return pdf.loc[mask]

    ddf_sel = ddf.map_partitions(
        _filter_score_window,
        params.score_min,
        params.score_max,
        meta=meta_sel,
    )

    should_persist = persist_ddfs or (not avoid_computes)
    reason = "cluster.persist_ddfs=True" if persist_ddfs else "avoid_computes_wherever_possible=False"
    ddf_sel = maybe_persist_ddf(
        ddf_sel,
        should_persist=should_persist,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        log_prefix="score_global",
        diag_label="dask_score_persist_filtered",
        reason=reason,
    )
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
    params: ScoreGlobalParams | None = None,
) -> None:
    """Execute the score_global selection path and write tiles.

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
        params: Optional resolved score parameters (auto-resolved when None).
    """
    algo = cfg.algorithm
    score_col_internal = "__score__"
    if params is None:
        raise RuntimeError("score_global: selection parameters were not provided.")

    score_min = float(params.score_min)
    score_max = float(params.score_max)
    depths_sel = list(range(1, cfg.algorithm.level_limit + 1))
    order_desc = bool(getattr(cfg.algorithm, "sg_order_desc", getattr(cfg.algorithm, "order_desc", False)))
    tie_col = getattr(cfg.algorithm, "score_tie_column", None) or getattr(cfg.algorithm, "tie_column", None)

    fixed_targets: Dict[int, float] = {}
    for d, n_val, k_val in (
        (1, getattr(algo, "score_n_1", None), getattr(algo, "score_k_1", None)),
        (2, getattr(algo, "score_n_2", None), getattr(algo, "score_k_2", None)),
        (3, getattr(algo, "score_n_3", None), getattr(algo, "score_k_3", None)),
    ):
        if (n_val is not None) and (k_val is not None):
            raise ValueError(f"score_global: both n_{d} and k_{d} are set; choose one.")
        if k_val is not None:
            active_tiles = int(np.count_nonzero(densmaps.get(d, [])))
            n_val = int(round(float(k_val) * active_tiles))
        if (d in depths_sel) and (n_val is not None):
            fixed_targets[d] = float(n_val)

    select_by_score_slices(
        remainder_ddf=remainder_ddf,
        densmaps=densmaps,
        depths_sel=depths_sel,
        keep_cols=keep_cols,
        ra_col=ra_col,
        dec_col=dec_col,
        score_col=score_col_internal,
        score_min=score_min,
        score_max=score_max,
        hist_nbins=algo.score_hist_nbins,
        out_dir=out_dir,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        label="score_global",
        order_desc=order_desc,
        fixed_targets=fixed_targets,
        hist_diag_ctx_name="dask_score_hist",
        depth_diag_prefix="dask_depth_score",
        tie_col=tie_col,
    )
