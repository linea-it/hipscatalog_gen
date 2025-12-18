from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import healpy as hp
import numpy as np
import pandas as pd
from dask import compute as dask_compute

from ..io.output import build_header_line_from_keep
from ..pipeline.common import write_tiles_with_allsky
from ..utils import _fmt_dur, _get_meta_df, _log_depth_stats
from .utils import _quantile_from_histogram, compute_score_histogram_ddf

__all__ = ["prepare_score_global", "run_score_global_selection"]


def prepare_score_global(ddf: Any, cfg: Any, diag_ctx, log_fn):
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

    base_meta_score = _get_meta_df(ddf)
    meta_with_score = base_meta_score.copy()
    meta_with_score["__score__"] = pd.Series([], dtype="float64")

    score_code = compile(score_expr, "<score_global>", "eval")

    def _add_score_column(pdf: pd.DataFrame, score_col_expr: str, compiled_expr) -> pd.DataFrame:
        if pdf.empty:
            pdf["__score__"] = pd.Series([], dtype="float64")
            return pdf

        pdf = pdf.copy()
        if score_col_expr in pdf.columns:
            sc = pd.to_numeric(pdf[score_col_expr], errors="coerce")
        else:
            env = {"__builtins__": {}, "np": np, "numpy": np}
            env.update({col: pdf[col] for col in pdf.columns})
            out = eval(compiled_expr, env, {})
            sc = pd.to_numeric(out, errors="coerce")

        sc = sc.replace([np.inf, -np.inf], np.nan)
        pdf["__score__"] = sc
        return pdf

    ddf = ddf.map_partitions(
        _add_score_column,
        score_expr,
        score_code,
        meta=meta_with_score,
    )

    score_col_internal = "__score__"
    score_min_cfg = getattr(algo, "score_min", None)
    score_max_cfg = getattr(algo, "score_max", None)

    with diag_ctx("dask_score_minmax"):
        score_min_global_raw, score_max_global_raw = dask_compute(
            ddf[score_col_internal].min(),
            ddf[score_col_internal].max(),
        )

    if score_min_global_raw is None or score_max_global_raw is None:
        raise ValueError(
            "score_global selection: unable to determine global score range "
            "(min/max returned None). Check the score expression/column."
        )

    score_min_global_raw = float(score_min_global_raw)
    score_max_global_raw = float(score_max_global_raw)

    if not np.isfinite(score_min_global_raw) or not np.isfinite(score_max_global_raw):
        raise ValueError(
            "score_global selection: global score min/max are not finite. "
            "Check the score expression/column values."
        )

    if score_min_global_raw >= score_max_global_raw:
        raise ValueError(
            f"score_global selection: invalid global score range "
            f"[{score_min_global_raw}, {score_max_global_raw}]."
        )

    def _compute_hist_peak() -> Tuple[float, float, float]:
        with diag_ctx("dask_score_hist_peak"):
            hist_auto, edges_auto, n_tot_auto = compute_score_histogram_ddf(
                ddf_like=ddf,
                score_col=score_col_internal,
                score_min=score_min_global_raw,
                score_max=score_max_global_raw,
                nbins=score_hist_nbins,
            )

        if n_tot_auto == 0:
            raise ValueError(
                "score_global selection: no objects found when estimating the "
                "histogram peak. Check the score expression/column."
            )

        peak_idx = int(np.argmax(hist_auto))
        bin_left = float(edges_auto[peak_idx])
        bin_right = float(edges_auto[peak_idx + 1])
        peak_center = float(np.round(0.5 * (bin_left + bin_right), 6))
        return peak_center, bin_left, bin_right

    peak_info: Tuple[float, float, float] | None = None

    def _get_peak_info() -> Tuple[float, float, float]:
        nonlocal peak_info
        if peak_info is None:
            peak_info = _compute_hist_peak()
        return peak_info

    score_min: float | None = float(score_min_cfg) if score_min_cfg is not None else None
    score_max: float | None = float(score_max_cfg) if score_max_cfg is not None else None

    if score_min is None and score_max is None:
        if score_range_mode == "complete":
            score_min = score_min_global_raw
            score_max = score_max_global_raw
            log_fn(
                "[score_global] score_min/score_max not provided; using global "
                f"range [{score_min:.6f}, {score_max:.6f}].",
                always=True,
            )
        else:
            peak_center, bin_left, bin_right = _get_peak_info()
            score_min = score_min_global_raw
            score_max = peak_center
            log_fn(
                "[score_global] score_min/score_max not provided; using global minimum "
                f"{score_min:.6f} and histogram peak at {score_max:.6f} "
                f"(bin center from [{bin_left:.6f}, {bin_right:.6f}]).",
                always=True,
            )
    elif score_min is None:
        if score_range_mode == "complete":
            score_min = score_min_global_raw
            log_fn(
                "[score_global] score_min not provided; using global minimum "
                f"{score_min:.6f} (score_max={score_max}).",
                always=True,
            )
        else:
            peak_center, bin_left, bin_right = _get_peak_info()
            score_min = peak_center
            if score_max is None:
                raise RuntimeError(
                    "score_global: score_max must be provided when using hist_peak for score_min."
                )
            if score_min > float(score_max):
                raise ValueError(
                    "score_global selection: histogram peak used as score_min "
                    f"({score_min:.6f}) is greater than the provided score_max "
                    f"({score_max})."
                )
            log_fn(
                "[score_global] score_min not provided; using histogram peak at "
                f"{score_min:.6f} as minimum (bin center from "
                f"[{bin_left:.6f}, {bin_right:.6f}]).",
                always=True,
            )
    elif score_max is None:
        if score_range_mode == "complete":
            score_max = score_max_global_raw
            log_fn(
                "[score_global] score_max not provided; using global maximum "
                f"{score_max:.6f} (score_min={score_min}).",
                always=True,
            )
        else:
            peak_center, bin_left, bin_right = _get_peak_info()
            score_max = peak_center
            if score_min is None:
                raise RuntimeError(
                    "score_global: score_min must be provided when using hist_peak for score_max."
                )
            if float(score_min) > score_max:
                raise ValueError(
                    "score_global selection: histogram peak used as score_max "
                    f"({score_max:.6f}) is smaller than the provided score_min "
                    f"({score_min})."
                )
            log_fn(
                "[score_global] score_max not provided; using histogram peak at "
                f"{score_max:.6f} as maximum (bin center from "
                f"[{bin_left:.6f}, {bin_right:.6f}]).",
                always=True,
            )

    if score_min is None or score_max is None:
        raise RuntimeError("score_global: internal error — score_min/score_max resolution failed.")

    if not (np.isfinite(score_min) and np.isfinite(score_max)):
        raise ValueError("score_global selection: resolved score_min/score_max are not finite.")

    if score_min >= score_max:
        raise ValueError(
            f"algorithm.score_min ({score_min}) must be strictly smaller than "
            f"algorithm.score_max ({score_max}) for score_global selection."
        )

    algo.score_min = score_min
    algo.score_max = score_max

    meta_sel = meta_with_score.copy()

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
    return ddf_sel


def _assign_targets_per_depth(
    densmaps: Dict[int, np.ndarray],
    depths_sel: List[int],
    algo: Any,
    cdf_hist: np.ndarray,
    score_edges_hist: np.ndarray,
    score_min: float,
    score_max: float,
    n_tot_score: float,
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
        (1, getattr(algo, "score_n_1", None)),
        (2, getattr(algo, "score_n_2", None)),
        (3, getattr(algo, "score_n_3", None)),
    ):
        if (d in depths_sel) and (n_val is not None):
            if int(n_val) < 0:
                raise ValueError(f"algorithm.score_n_{d} must be non-negative if provided (got {n_val}).")
            fixed_targets[d] = float(int(n_val))

    sum_fixed = float(sum(fixed_targets.values()))
    if sum_fixed > n_tot_score and sum_fixed > 0.0:
        scale = float(n_tot_score) / sum_fixed if sum_fixed > 0 else 0.0
        log_fn(
            "[score_global] Sum of fixed targets score_n_1/score_n_2/score_n_3 "
            f"({int(sum_fixed)}) exceeds the total number of objects "
            f"in the score range ({int(n_tot_score)}). "
            f"Rescaling score_n_1/score_n_2/score_n_3 by a factor {scale:.3f}.",
            always=True,
        )
        for d in list(fixed_targets.keys()):
            fixed_targets[d] *= scale
        sum_fixed = float(n_tot_score)

    for d, val in fixed_targets.items():
        idx = depths_sel.index(d)
        T[idx] = val

    N_rem = max(0.0, float(n_tot_score) - sum_fixed)
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
    Q = T_cum / float(n_tot_score) if n_tot_score > 0.0 else np.zeros_like(T_cum, dtype="float64")

    level_edges: np.ndarray = np.empty(len(depths_sel) + 1, dtype="float64")
    level_edges[0] = score_min
    for i, q in enumerate(Q, start=1):
        level_edges[i] = _quantile_from_histogram(cdf_hist, score_edges_hist, q)

    level_edges = np.maximum.accumulate(level_edges)
    level_edges[0] = score_min
    level_edges[-1] = score_max
    return level_edges


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
    id_col: str | None = None,
    id_sink: set[int] | None = None,
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

    level_edges = _assign_targets_per_depth(
        densmaps=densmaps,
        depths_sel=depths_sel,
        algo=algo,
        cdf_hist=cdf_hist,
        score_edges_hist=score_edges_hist,
        score_min=score_min,
        score_max=score_max,
        n_tot_score=float(n_tot_score),
        log_fn=log_fn,
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
            if id_col and id_col in selected_pdf and id_sink is not None:
                id_sink.update(selected_pdf[id_col].astype(int).tolist())
            if id_col and id_col in selected_pdf:
                selected_pdf = selected_pdf.drop(columns=[id_col])
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
                order_desc=cfg.algorithm.order_desc,
                allsky_needed=allsky_needed,
                log_fn=log_fn,
            )
            _log_depth_stats(log_fn, depth, "written", counts=densmaps[depth], written=written_per_ipix)

        log_fn(
            f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
            always=True,
        )
