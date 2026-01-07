"""Hybrid density-first then score-based selection pipeline."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from lsdb.catalog import Catalog as LsdbCatalog
else:  # Optional LSDB import for HATS catalogs.
    LsdbCatalog: type[Any] | None
    try:
        from lsdb.catalog import Catalog as LsdbCatalog
    except Exception:  # pragma: no cover - optional dependency
        LsdbCatalog = None

from ..io.output import build_header_line_from_keep
from ..pipeline.common import maybe_persist_ddf, write_tiles_with_allsky
from ..pipeline.params import ScoreDensityHybridParams
from ..selection.common import add_ipix_column, reduce_topk_by_group_dask, targets_per_tile
from ..selection.levels import assign_level_edges
from ..selection.score import (
    _finite_min_max,
    _map_invalid_to_sentinel,
    _sentinel_for_order,
    add_score_column,
    compute_score_histogram_ddf,
    resolve_value_range,
)
from ..selection.slicing import select_by_score_slices
from ..utils import _fmt_dur, _get_meta_df, _log_depth_stats

__all__ = [
    "normalize_score_density_hybrid",
    "prepare_score_density_hybrid",
    "run_score_density_hybrid_selection",
]


# =============================================================================
# Helpers
# =============================================================================


def normalize_score_density_hybrid(
    ddf: Any,
    cfg: Any,
    diag_ctx,
    log_fn,
    persist_ddfs: bool = False,
    avoid_computes: bool = True,
) -> tuple[Any, ScoreDensityHybridParams]:
    """Add ``__score__`` column and compute window without mutating cfg.

    Args:
        ddf: Dask-like collection with score columns or expressions.
        cfg: Parsed configuration object.
        diag_ctx: Diagnostics context factory.
        log_fn: Logging callback.
        persist_ddfs: Whether to persist intermediate DDFs.
        avoid_computes: Whether to avoid explicit ``compute()`` calls when possible.

    Returns:
        Tuple ``(ddf_with_score, ScoreDensityHybridParams)``.

    Raises:
        ValueError: If the score expression/configuration is invalid.
    """
    algo = cfg.algorithm
    score_expr = getattr(algo, "sdh_score_column", getattr(algo, "score_column", None))
    if not score_expr:
        raise ValueError("score_density_hybrid selection requires algorithm.sdh_score_column/score_column.")

    score_expr = str(score_expr)
    score_range_mode = str(getattr(algo, "sdh_score_adaptive_range", "complete") or "complete").lower()
    if score_range_mode not in ("complete", "hist_peak"):
        raise ValueError("algorithm.sdh_score_adaptive_range must be 'complete' or 'hist_peak'.")

    score_hist_nbins = int(getattr(algo, "sdh_score_hist_nbins", getattr(algo, "score_hist_nbins", 2048)))
    if score_hist_nbins <= 0:
        raise ValueError("algorithm.sdh_score_hist_nbins must be a positive integer.")

    ddf = add_score_column(ddf, score_expr, output_col="__score__")

    score_col_internal = "__score__"
    score_min_cfg = getattr(algo, "sdh_score_min", getattr(algo, "score_min", None))
    score_max_cfg = getattr(algo, "sdh_score_max", getattr(algo, "score_max", None))
    keep_invalid = bool(getattr(algo, "sdh_keep_invalid_values", False))
    if keep_invalid and score_range_mode != "complete":
        raise ValueError(
            "score_density_hybrid: keep_invalid_values=True is only supported with "
            "score_adaptive_range=complete."
        )
    order_desc = bool(getattr(cfg.algorithm, "sdh_order_desc", getattr(cfg.algorithm, "order_desc", False)))

    sentinel: float | None = None
    if keep_invalid:
        fin_min, fin_max = _finite_min_max(ddf, score_col_internal)
        if fin_min is None or fin_max is None:
            raise ValueError("score_density_hybrid: all score values are NaN/Inf; nothing to select.")
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
            f"[score_density_hybrid] keep_invalid_values=True → mapping NaN/Inf to sentinel {sentinel} "
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
            label="score_density_hybrid",
        )

    params = ScoreDensityHybridParams(score_min=score_min, score_max=score_max, sentinel=sentinel)
    return ddf, params


def _filter_score_window(pdf: pd.DataFrame, score_min_val: float, score_max_val: float) -> pd.DataFrame:
    """Keep rows within the resolved score window for one partition."""
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


def _drop_selected_ids(pdf: pd.DataFrame, ids: Iterable[int]) -> pd.DataFrame:
    """Remove already-selected IDs from a partition."""
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
    params: ScoreDensityHybridParams,
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
        Dask-like collection filtered to the score window with unique IDs attached.
    """

    meta_sel = _get_meta_df(ddf).copy()
    meta_sel["__score__"] = pd.Series([], dtype="float64")
    ddf_sel = ddf.map_partitions(
        _filter_score_window,
        params.score_min,
        params.score_max,
        meta=meta_sel,
    )

    meta_with_id = meta_sel.copy()
    meta_with_id["__sdh_id__"] = pd.Series([], dtype="int64")

    def _attach_id_partition(pdf: pd.DataFrame, partition_info=None, **kwargs) -> pd.DataFrame:
        """Attach __sdh_id__ deterministically using partition number."""
        if not isinstance(partition_info, dict) or "number" not in partition_info:
            if pdf.empty:
                pdf["__sdh_id__"] = pd.Series([], dtype="int64")
                return pdf[meta_with_id.columns]
            raise RuntimeError("score_density_hybrid: missing partition_info; cannot build unique IDs.")
        part_no = int(partition_info["number"])
        base = np.int64(part_no) << 32
        seq = np.arange(len(pdf), dtype="int64")
        pdf = pdf.copy()
        pdf["__sdh_id__"] = base + seq
        return pdf[meta_with_id.columns]

    def _parse_pixel_order_pixel(pixel_obj) -> tuple[int | None, int | None]:
        """Extract (order, pixel) from an LSDB pixel object or its string form."""
        order = getattr(pixel_obj, "order", None)
        pix = getattr(pixel_obj, "pixel", None)
        if order is not None and pix is not None:
            return int(order), int(pix)
        try:
            text = str(pixel_obj)
            parts = text.replace(",", "").split()
            if "Order:" in parts and "Pixel:" in parts:
                order = int(parts[parts.index("Order:") + 1])
                pix = int(parts[parts.index("Pixel:") + 1])
                return order, pix
        except Exception:
            return None, None
        return None, None

    def _attach_id_pixel(pdf: pd.DataFrame, pixel=None, **kwargs) -> pd.DataFrame:
        """Attach __sdh_id__ using (order, pixel, row_index)."""
        if pdf.empty:
            pdf["__sdh_id__"] = pd.Series([], dtype="int64")
            return pdf[meta_with_id.columns]
        order, pix = _parse_pixel_order_pixel(pixel)
        if order is None or pix is None:
            if pdf.empty:
                pdf["__sdh_id__"] = pd.Series([], dtype="int64")
                return pdf[meta_with_id.columns]
            raise RuntimeError("score_density_hybrid: missing LSDB pixel metadata; cannot build unique IDs.")
        base = (np.int64(order) << np.int64(58)) | (np.int64(pix) << np.int64(32))
        seq = np.arange(len(pdf), dtype="int64")
        pdf = pdf.copy()
        pdf["__sdh_id__"] = base + seq
        return pdf[meta_with_id.columns]

    is_lsdb = LsdbCatalog is not None and isinstance(ddf, LsdbCatalog)
    if is_lsdb:
        log_fn(
            "[score_density_hybrid] Using LSDB pixel-based IDs for stage-1 de-duplication.",
            always=True,
        )
        ddf_sel = ddf_sel.map_partitions(
            _attach_id_pixel, meta=meta_with_id, include_pixel=True, enforce_metadata=True
        )
    else:
        log_fn(
            "[score_density_hybrid] Using partition_info-based IDs for stage-1 de-duplication.",
            always=True,
        )
        ddf_sel = ddf_sel.map_partitions(
            _attach_id_partition, meta=meta_with_id, partition_info=True, enforce_metadata=True
        )

    should_persist = persist_ddfs or (not avoid_computes)
    reason = "cluster.persist_ddfs=True" if persist_ddfs else "avoid_computes_wherever_possible=False"
    ddf_sel = maybe_persist_ddf(
        ddf_sel,
        should_persist=should_persist,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        log_prefix="score_density_hybrid",
        diag_label="dask_sdh_persist_filtered",
        reason=reason,
    )

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
    params: ScoreDensityHybridParams | None = None,
) -> None:
    """Execute the score_density_hybrid selection.

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
        params: Resolved score parameters (required).
    """
    algo = cfg.algorithm
    score_col_internal = "__score__"
    if params is None:
        raise RuntimeError("score_density_hybrid: selection parameters were not provided.")

    score_min = float(params.score_min)
    score_max = float(params.score_max)
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
        n_val = getattr(algo, f"sdh_n_{d}", None)
        k_val = getattr(algo, f"sdh_k_{d}", None)
        if (n_val is not None) and (k_val is not None):
            raise ValueError(f"score_density_hybrid: both n_{d} and k_{d} are set; choose one.")
        if k_val is not None:
            active_tiles = int(np.count_nonzero(densmaps.get(d, [])))
            n_val = int(round(float(k_val) * active_tiles))
        if n_val is not None:
            fixed_targets_map[d] = n_val
    fixed_targets_clean: Dict[int, float] = {}
    for k, v in fixed_targets_map.items():
        if v is None:
            continue  # pragma: no cover
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
    order_desc = bool(getattr(algo, "sdh_order_desc", getattr(algo, "order_desc", False)))
    tie_col = getattr(algo, "sdh_tie_column", None) or getattr(algo, "tie_column", None)

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
                tie_col=tie_col,
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

    select_by_score_slices(
        remainder_ddf=available_ddf,
        densmaps=densmaps,
        depths_sel=remaining_depths,
        keep_cols=keep_cols,
        ra_col=ra_col,
        dec_col=dec_col,
        score_col=score_col_internal,
        score_min=score_min,
        score_max=score_max,
        hist_nbins=algo.sdh_score_hist_nbins,
        out_dir=out_dir,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        label="score_density_hybrid",
        order_desc=order_desc,
        fixed_targets={},
        hist_diag_ctx_name="dask_sdh_score_hist_remaining",
        depth_diag_prefix="dask_sdh_depth_score",
        tie_col=tie_col,
    )
