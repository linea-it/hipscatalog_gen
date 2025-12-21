from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List

import healpy as hp
import numpy as np
import pandas as pd
from lsdb.catalog import Catalog as LsdbCatalog

from ..io.output import build_header_line_from_keep
from ..pipeline.common import write_tiles_with_allsky
from ..score_global.utils import compute_score_histogram_ddf
from ..selection.common import (
    add_ipix_column,
    assign_level_edges,
    reduce_topk_by_group_dask,
    targets_per_tile,
)
from ..selection.score import add_score_column, resolve_value_range
from ..utils import _fmt_dur, _get_meta_df, _log_depth_stats

__all__ = ["prepare_score_density_hybrid", "run_score_density_hybrid_selection"]


# =============================================================================
# Helpers
# =============================================================================


def _filter_score_window(pdf: pd.DataFrame, score_min_val: float, score_max_val: float) -> pd.DataFrame:
    if pdf.empty:
        return pdf
    s = pd.to_numeric(pdf["__score__"], errors="coerce")
    mask = (s >= score_min_val) & (s <= score_max_val)
    return pdf.loc[mask]


def _attach_unique_id(pdf: pd.DataFrame, partition_info=None) -> pd.DataFrame:
    """Attach a unique integer id per row (partition-aware)."""
    if pdf.empty:
        pdf["__sdh_id__"] = pd.Series([], dtype="int64")
        return pdf

    part_no = int(partition_info["number"]) if partition_info and "number" in partition_info else 0
    pdf = pdf.copy()
    local = np.arange(len(pdf), dtype="int64")
    pdf["__sdh_id__"] = local + (np.int64(part_no) << 32)
    return pdf


def _distribute_by_weights(total: int, weights: Dict[int, int]) -> Dict[int, int]:
    """Distribute an integer total following relative weights with rounding."""
    if total <= 0 or not weights:
        return {k: 0 for k in weights}

    idx = list(weights.keys())
    w_vec = np.asarray([max(0, int(weights[k])) for k in idx], dtype="float64")
    w_sum = float(w_vec.sum())
    if w_sum <= 0.0:
        base = np.zeros_like(w_vec, dtype=int)
    else:
        raw = w_vec / w_sum * float(total)
        base = np.floor(raw).astype(int)
        remainder = int(total - base.sum())
        if remainder > 0:
            frac = raw - base
            order = np.argsort(-frac, kind="mergesort")
            for j in order[:remainder]:
                base[j] += 1
    return {int(k): int(v) for k, v in zip(idx, base, strict=False)}


def _targets_stage1_by_depth(
    densmaps: Dict[int, np.ndarray],
    base_targets: Dict[int, float],
    n_tot_score: float,
    provided: Dict[int, float],
    log_fn,
) -> Dict[int, int]:
    """Redistribute stage-1 totals across depths 1–3 using active tiles."""
    depths_stage1 = sorted([d for d in base_targets if d <= 3])
    if not depths_stage1:
        return {}

    active_per_depth = {d: int((densmaps[d] > 0).sum()) for d in depths_stage1}
    base_total = int(round(sum(base_targets.get(d, 0.0) for d in depths_stage1)))
    provided_sum = int(round(sum(provided.get(d, 0.0) for d in depths_stage1)))
    total_cap = int(n_tot_score)
    total_target = max(base_total, provided_sum)
    total_target = min(total_target, total_cap)

    remaining_total = max(0, total_target - provided_sum)
    remaining_weights = {d: active_per_depth.get(d, 0) for d in depths_stage1 if d not in provided}
    distributed = _distribute_by_weights(remaining_total, remaining_weights)

    totals: Dict[int, int] = {}
    for d in depths_stage1:
        if d in provided:
            totals[d] = int(provided[d])
        else:
            totals[d] = int(distributed.get(d, 0))

        avail = int(densmaps[d].sum())
        if totals[d] > avail:
            log_fn(
                f"[score_density_hybrid] Requested {totals[d]} objects for depth {d} "
                f"but only {avail} available → clamping.",
                always=True,
            )
            totals[d] = avail

    diff = total_target - sum(totals.values())
    if diff != 0 and depths_stage1:
        order = sorted(depths_stage1, key=lambda x: active_per_depth.get(x, 0), reverse=True)
        idx = 0
        while diff != 0 and order:
            d = order[idx % len(order)]
            adj = 1 if diff > 0 else -1
            new_val = max(0, totals.get(d, 0) + adj)
            totals[d] = new_val
            diff -= adj
            idx += 1

    return totals


def _reduce_topk_by_group_dask(
    ddf_like: Any,
    group_col: str,
    score_col: str,
    order_desc: bool,
    k_per_group: Dict[int, int],
    ra_col: str,
    dec_col: str,
):
    """Keep up to k_per_group rows per group, sorted by score then RA/DEC."""
    if not k_per_group:
        empty_meta = _get_meta_df(ddf_like)
        return ddf_like.map_partitions(lambda pdf: pdf.iloc[0:0], meta=empty_meta)

    asc = not order_desc
    k_map = {int(k): int(v) for k, v in k_per_group.items()}

    def _take_topk(group: pd.DataFrame) -> pd.DataFrame:
        if group.empty:
            return group
        g_id = int(group[group_col].iloc[0])
        k = int(k_map.get(g_id, 0))
        if k <= 0:
            return group.iloc[0:0]
        sort_cols = [score_col]
        ascending = [asc]
        if ra_col in group.columns:
            sort_cols.append(ra_col)
            ascending.append(True)
        if dec_col in group.columns:
            sort_cols.append(dec_col)
            ascending.append(True)
        group_sorted = group.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        return group_sorted.head(k)

    meta = _get_meta_df(ddf_like)
    cols_all = list(meta.columns)
    # LSDB Catalog: fall back to underlying Dask DataFrame when groupby is absent.
    if isinstance(ddf_like, LsdbCatalog) and hasattr(ddf_like, "_ddf"):
        base_ddf = ddf_like._ddf  # type: ignore[attr-defined]
        cols_all = list(base_ddf.columns)
        return base_ddf.groupby(group_col, group_keys=False)[cols_all].apply(_take_topk, meta=meta)

    if hasattr(ddf_like, "groupby"):
        return ddf_like.groupby(group_col, group_keys=False)[cols_all].apply(_take_topk, meta=meta)

    return ddf_like


def _drop_selected_ids(pdf: pd.DataFrame, ids: Iterable[int]) -> pd.DataFrame:
    if pdf.empty:
        return pdf
    ids_set = set(int(x) for x in ids)
    if not ids_set:
        return pdf
    return pdf.loc[~pdf["__sdh_id__"].isin(ids_set)]


# =============================================================================
# Public API
# =============================================================================


def prepare_score_density_hybrid(
    ddf: Any,
    cfg: Any,
    diag_ctx,
    log_fn,
    persist_ddfs: bool = False,
    avoid_computes: bool = True,
):
    """Add __score__ column (from an expression) and restrict to a score window."""
    algo = cfg.algorithm
    score_expr = getattr(algo, "sdh_score_column", None)
    if not score_expr:
        raise ValueError("score_density_hybrid selection requires algorithm.sdh_score_column to be set.")

    score_expr = str(score_expr)
    score_range_mode = str(getattr(algo, "sdh_score_adaptive_range", "complete") or "complete").lower()
    if score_range_mode not in ("complete", "hist_peak"):
        raise ValueError("algorithm.sdh_score_adaptive_range must be 'complete' or 'hist_peak'.")

    score_hist_nbins = int(getattr(algo, "sdh_score_hist_nbins", getattr(algo, "score_hist_nbins", 512)))
    if score_hist_nbins <= 0:
        raise ValueError("algorithm.sdh_score_hist_nbins must be a positive integer.")

    ddf = add_score_column(ddf, score_expr, output_col="__score__")

    score_col_internal = "__score__"
    score_min_cfg = getattr(algo, "sdh_score_min", None)
    score_max_cfg = getattr(algo, "sdh_score_max", None)
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
        label="score_density_hybrid",
    )

    algo.sdh_score_min = score_min
    algo.sdh_score_max = score_max

    meta_sel = _get_meta_df(ddf).copy()
    meta_sel["__score__"] = pd.Series([], dtype="float64")
    ddf_sel = ddf.map_partitions(
        _filter_score_window,
        score_min,
        score_max,
        meta=meta_sel,
    )

    meta_with_id = meta_sel.copy()
    meta_with_id["__sdh_id__"] = pd.Series([], dtype="int64")
    ddf_sel = ddf_sel.map_partitions(_attach_unique_id, meta=meta_with_id, partition_info=True)

    should_persist = persist_ddfs or (not avoid_computes)
    if should_persist and hasattr(ddf_sel, "persist"):
        reason = "cluster.persist_ddfs=True" if persist_ddfs else "avoid_computes_wherever_possible=False"
        log_fn(f"[score_density_hybrid] Persisting filtered DDF in memory ({reason}).", always=True)
        with diag_ctx("dask_sdh_persist_filtered"):
            ddf_sel = ddf_sel.persist()
            try:
                from dask.distributed import wait
            except Exception:
                wait = None  # type: ignore[assignment]
            if wait is not None:
                wait(ddf_sel)

    return ddf_sel


def run_score_density_hybrid_selection(
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
    """Execute the score_density_hybrid selection."""
    algo = cfg.algorithm
    score_col_internal = "__score__"
    if algo.sdh_score_min is None or algo.sdh_score_max is None:
        raise RuntimeError(
            "score_density_hybrid: internal error — sdh_score_min/sdh_score_max should have been set earlier."
        )

    score_min = float(algo.sdh_score_min)
    score_max = float(algo.sdh_score_max)
    depths_sel = list(range(1, cfg.algorithm.level_limit + 1))

    header_line = build_header_line_from_keep(keep_cols)

    # ------------------------------------------------------------------
    # Stage 1: density-driven for depths 1–3
    # ------------------------------------------------------------------
    with diag_ctx("dask_sdh_score_hist_initial"):
        hist, score_edges_hist, n_tot_score = compute_score_histogram_ddf(
            remainder_ddf,
            score_col=score_col_internal,
            score_min=score_min,
            score_max=score_max,
            nbins=algo.sdh_score_hist_nbins,
        )

    if n_tot_score == 0:
        log_fn(
            "[selection] score_density_hybrid: no objects found in the score range "
            f"[{score_min}, {score_max}] → nothing to select.",
            always=True,
        )
        return

    cdf_hist = hist.cumsum().astype("float64")
    if cdf_hist[-1] > 0:
        cdf_hist /= float(cdf_hist[-1])
    else:
        cdf_hist[:] = 0.0

    fixed_targets_map: Dict[int, Any] = {}
    for d in (1, 2, 3):
        val = getattr(algo, f"sdh_n_{d}", None)
        if val is not None:
            fixed_targets_map[d] = val
    fixed_targets_clean: Dict[int, float] = {}
    for k, v in fixed_targets_map.items():
        if v is None:
            continue
        fixed_targets_clean[int(k)] = float(v)

    level_edges_initial, targets_per_depth_raw = assign_level_edges(
        densmaps=densmaps,
        depths_sel=depths_sel,
        fixed_targets=fixed_targets_clean,
        cdf_hist=cdf_hist,
        score_edges_hist=score_edges_hist,
        score_min=score_min,
        score_max=score_max,
        n_tot_score=float(n_tot_score),
        log_fn=log_fn,
        label="score_density_hybrid",
    )

    base_targets = {depths_sel[i]: float(targets_per_depth_raw[i]) for i in range(len(depths_sel))}
    stage1_totals = _targets_stage1_by_depth(
        densmaps=densmaps,
        base_targets=base_targets,
        n_tot_score=float(n_tot_score),
        provided=fixed_targets_clean,
        log_fn=log_fn,
    )
    log_fn(
        "[selection] score_density_hybrid stage 1 targets (depth 1–3): "
        + ", ".join(f"{d}: {stage1_totals.get(d, 0)}" for d in sorted(stage1_totals)),
        always=True,
    )

    available_ddf = remainder_ddf
    order_desc = bool(getattr(algo, "sdh_order_desc", False))

    for depth in [d for d in depths_sel if d <= 3]:
        depth_t0 = time.time()
        depth_total = int(stage1_totals.get(depth, 0))
        if depth_total <= 0:
            log_fn(f"[DEPTH {depth}] score_density_hybrid: target is 0 → skipping.", always=True)
            continue

        counts = densmaps[depth]
        bias = float(getattr(algo, f"sdh_density_bias_n{depth}", 0.0))
        targets_per_tile_map = targets_per_tile(counts, depth_total, bias)
        if not targets_per_tile_map:
            log_fn(
                f"[DEPTH {depth}] score_density_hybrid: no active tiles or zero targets → skipping.",
                always=True,
            )
            continue

        with diag_ctx(f"dask_sdh_depth_{depth:02d}_candidates"):
            meta_ipix = _get_meta_df(available_ddf).copy()
            meta_ipix["__ipix__"] = pd.Series([], dtype="int64")
            ddf_with_ipix = available_ddf.map_partitions(
                add_ipix_column,
                depth,
                ra_col,
                dec_col,
                meta=meta_ipix,
            )

            target_tiles = list(targets_per_tile_map.keys())
            cand_ddf = ddf_with_ipix[ddf_with_ipix["__ipix__"].isin(target_tiles)]
            selected_ddf = reduce_topk_by_group_dask(
                cand_ddf,
                group_col="__ipix__",
                score_col=score_col_internal,
                order_desc=order_desc,
                k_per_group=targets_per_tile_map,
                ra_col=ra_col,
                dec_col=dec_col,
            )
            selected_pdf = selected_ddf.compute()

        _log_depth_stats(log_fn, depth, "selected", counts=counts, selected_len=len(selected_pdf))

        if len(selected_pdf) == 0:
            log_fn(
                f"[DEPTH {depth}] score_density_hybrid: no rows selected for this depth.",
                always=True,
            )
            continue

        allsky_needed = depth in (1, 2)
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
        _log_depth_stats(log_fn, depth, "written", counts=counts, written=written_per_ipix)

        ids_used = selected_pdf["__sdh_id__"].dropna().astype("int64").tolist()
        if ids_used:
            meta_avail = _get_meta_df(available_ddf)
            available_ddf = available_ddf.map_partitions(_drop_selected_ids, ids_used, meta=meta_avail)

        log_fn(f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}", always=True)

    # ------------------------------------------------------------------
    # Stage 2: remaining depths via score_global logic on the remainder
    # ------------------------------------------------------------------
    remaining_depths = [d for d in depths_sel if d > 3]
    if not remaining_depths:
        return

    with diag_ctx("dask_sdh_score_hist_remaining"):
        hist_rem, edges_rem, n_tot_rem = compute_score_histogram_ddf(
            available_ddf,
            score_col=score_col_internal,
            score_min=score_min,
            score_max=score_max,
            nbins=algo.sdh_score_hist_nbins,
        )

    if n_tot_rem == 0:
        log_fn(
            "[selection] score_density_hybrid: no remaining objects after depths 1–3 "
            "→ nothing else to select.",
            always=True,
        )
        return

    cdf_rem = hist_rem.cumsum().astype("float64")
    if cdf_rem[-1] > 0:
        cdf_rem /= float(cdf_rem[-1])
    else:
        cdf_rem[:] = 0.0

    level_edges_rem, _ = assign_level_edges(
        densmaps=densmaps,
        depths_sel=remaining_depths,
        fixed_targets={},
        cdf_hist=cdf_rem,
        score_edges_hist=edges_rem,
        score_min=score_min,
        score_max=score_max,
        n_tot_score=float(n_tot_rem),
        log_fn=log_fn,
        label="score_density_hybrid",
    )

    log_fn(
        "[selection] score_density_hybrid stage 2 (score slices):\n"
        + "\n".join(
            f"  depth {d}: [{level_edges_rem[i]:.6f}, {level_edges_rem[i+1]:.6f}"
            f"{')' if d != remaining_depths[-1] else ']'}"
            for i, d in enumerate(remaining_depths)
        ),
        always=True,
    )

    for i, depth in enumerate(remaining_depths):
        depth_t0 = time.time()
        s_lo = level_edges_rem[i]
        s_hi = level_edges_rem[i + 1]

        with diag_ctx(f"dask_sdh_depth_score_{depth:02d}"):
            if depth != remaining_depths[-1]:
                score_mask = (available_ddf[score_col_internal] >= s_lo) & (
                    available_ddf[score_col_internal] < s_hi
                )
            else:
                score_mask = (available_ddf[score_col_internal] >= s_lo) & (
                    available_ddf[score_col_internal] <= s_hi
                )

            depth_ddf = available_ddf[score_mask]
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
                    f"[DEPTH {depth}] score_density_hybrid: no rows in "
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
                order_desc=order_desc,
                allsky_needed=allsky_needed,
                log_fn=log_fn,
            )
            _log_depth_stats(log_fn, depth, "written", counts=densmaps[depth], written=written_per_ipix)

        log_fn(
            f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
            always=True,
        )
