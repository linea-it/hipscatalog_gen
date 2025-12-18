from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import healpy as hp
import numpy as np
import pandas as pd
from dask import compute as dask_compute

from ..io.output import build_header_line_from_keep
from ..pipeline.common import write_tiles_with_allsky
from ..score_global.utils import _quantile_from_histogram, compute_score_histogram_ddf
from ..utils import _fmt_dur, _get_meta_df, _log_depth_stats

__all__ = ["prepare_score_density_hybrid", "run_score_density_hybrid_selection"]


def _project_counts_to_order(counts_src: np.ndarray, order_src: int, order_tgt: int) -> np.ndarray:
    """Aggregate a HEALPix count vector from order_src down to order_tgt."""
    if order_src == order_tgt:
        return counts_src.astype(np.int64)

    if order_src < order_tgt:
        # Cannot safely upsample; caller should avoid this situation.
        return counts_src.astype(np.int64)

    delta = order_src - order_tgt
    parent_size = 12 * (4**order_tgt)
    parents = np.zeros(parent_size, dtype=np.int64)

    idx = np.arange(len(counts_src), dtype=np.int64)
    parent_idx = idx >> (2 * delta)
    np.add.at(parents, parent_idx, counts_src.astype(np.int64))
    return parents


def _resolve_weight_for_level(algo: Any, depth: int) -> float:
    """Return density weight for a given level using the per-level override if present."""
    levels_cfg = getattr(algo, "sdh_density_weight_levels", None)
    if isinstance(levels_cfg, int | float):
        return float(levels_cfg)
    if isinstance(levels_cfg, list | tuple):
        idx = depth - 1
        if 0 <= idx < len(levels_cfg):
            return float(levels_cfg[idx])
    return float(getattr(algo, "sdh_density_weight", 0.0))


def _compute_density_weights(counts: np.ndarray, weight: float, eps: float = 1e-6) -> np.ndarray:
    """Convert counts into normalized weights with a smoothing exponent."""
    if weight <= 0.0:
        base = np.ones_like(counts, dtype=np.float64)
    else:
        counts_f = counts.astype(np.float64)
        active = counts_f > 0
        mean_val = float(np.mean(counts_f[active])) if np.any(active) else 0.0
        if not np.isfinite(mean_val) or mean_val <= 0.0:
            base = np.ones_like(counts_f)
        else:
            base = ((counts_f + eps) / mean_val) ** float(weight)
        base = base * active
    mask = base > 0
    if mask.any():
        base = base * mask
        norm = float(base.sum())
        if norm > 0.0:
            base /= norm
        else:
            base[:] = 0.0
    else:
        base[:] = 0.0
    return base


def _assign_targets_per_depth(
    densmaps: Dict[int, np.ndarray],
    depths_sel: List[int],
    fixed_targets: Dict[int, Optional[int]],
    cdf_hist: np.ndarray,
    score_edges_hist: np.ndarray,
    score_min: float,
    score_max: float,
    n_tot_score: float,
    log_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reuse score_global target allocation with fixed targets map.

    Returns
        level_edges: score edges per depth (len = depths_sel + 1)
        targets_per_depth: expected total rows per depth (float)
    """
    weights_list: List[float] = []
    for d in depths_sel:
        counts_d = densmaps[d]
        tiles_active = int((counts_d > 0).sum())
        weights_list.append(max(1, tiles_active))

    weights = np.asarray(weights_list, dtype="float64")
    T = np.zeros_like(weights, dtype="float64")

    fixed_map: Dict[int, float] = {}
    for depth, n_val in fixed_targets.items():
        if n_val is None:
            continue
        if int(n_val) < 0:
            raise ValueError(f"algorithm.sdh_n_{depth} must be non-negative if provided (got {n_val}).")
        fixed_map[depth] = float(int(n_val))

    sum_fixed = float(sum(fixed_map.values()))
    if sum_fixed > n_tot_score and sum_fixed > 0.0:
        scale = float(n_tot_score) / sum_fixed if sum_fixed > 0 else 0.0
        log_fn(
            "[score_density_hybrid] Sum of fixed targets sdh_n_1/sdh_n_2/sdh_n_3 "
            f"({int(sum_fixed)}) exceeds the total number of objects "
            f"in the score range ({int(n_tot_score)}). "
            f"Rescaling by a factor {scale:.3f}.",
            always=True,
        )
        for d in list(fixed_map.keys()):
            fixed_map[d] *= scale
        sum_fixed = float(n_tot_score)

    for d, val in fixed_map.items():
        idx = depths_sel.index(d)
        T[idx] = val

    N_rem = max(0.0, float(n_tot_score) - sum_fixed)
    if N_rem > 0.0:
        free_mask = np.ones_like(weights, dtype=bool)
        for d in fixed_map:
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
    targets_per_depth = np.empty(len(depths_sel), dtype="float64")
    targets_per_depth[:] = T
    return level_edges, targets_per_depth


def _add_rowid_partition(pdf: pd.DataFrame, start: int) -> pd.DataFrame:
    """Attach a sequential __sdh_id__ starting from a given offset."""
    if pdf is None or len(pdf) == 0:
        pdf = pdf.copy()
        pdf["__sdh_id__"] = pd.Series([], dtype="int64")
        return pdf
    pdf = pdf.copy()
    pdf["__sdh_id__"] = np.arange(start, start + len(pdf), dtype=np.int64)
    return pdf


def _redistribute_by_density(
    pdf: pd.DataFrame,
    counts_ref: np.ndarray,
    weight: float,
    order_desc: bool,
    seed: Optional[int],
    log_fn,
) -> pd.DataFrame:
    """Bias selection per ipix according to density weights while keeping top scores."""
    if pdf.empty or weight <= 0.0:
        return pdf.sort_values("__score__", ascending=not order_desc, kind="mergesort")

    before_counts = pdf["__ipix__"].value_counts().sort_index().to_dict()

    ipix_present = np.asarray(sorted(pdf["__ipix__"].unique()), dtype=np.int64)
    if ipix_present.size == 0:
        return pdf.sort_values("__score__", ascending=not order_desc, kind="mergesort")

    if ipix_present.max(initial=-1) >= len(counts_ref) or ipix_present.min(initial=0) < 0:
        log_fn(
            "[score_density_hybrid] ipix out of bounds for provided density map; skipping redistribution.",
            always=True,
        )
        return pdf.sort_values("__score__", ascending=not order_desc, kind="mergesort")

    counts_subset = counts_ref[ipix_present]
    weights_subset = _compute_density_weights(counts_subset, weight)
    if weights_subset.sum() <= 0.0:
        return pdf.sort_values("__score__", ascending=not order_desc, kind="mergesort")

    n_total = len(pdf)
    desired = weights_subset * float(n_total)
    quotas = np.floor(desired).astype(np.int64)
    remainder = n_total - int(quotas.sum())

    if remainder > 0:
        frac = desired - quotas.astype(np.float64)
        if seed is not None:
            rng = np.random.default_rng(int(seed))
            frac = frac + rng.random(len(frac)) * 1e-6
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            quotas[idx] += 1

    quotas_map = {int(ip): int(q) for ip, q in zip(ipix_present, quotas, strict=False) if q > 0}

    pdf = pdf.copy()
    pdf["__ipix__"] = pdf["__ipix__"].astype(np.int64)
    pdf = pdf.sort_values(
        ["__ipix__", "__score__"],
        ascending=[True, not order_desc],
        kind="mergesort",
    )

    selected_parts: List[pd.DataFrame] = []
    extras: List[pd.DataFrame] = []

    for ipix, group in pdf.groupby("__ipix__", sort=False):
        quota = quotas_map.get(int(ipix), 0)
        if quota <= 0:
            extras.append(group)
            continue
        if quota >= len(group):
            selected_parts.append(group)
            if quota > len(group):
                # deficit will be covered by extras from other cells
                pass
            continue

        selected_parts.append(group.iloc[:quota])
        extras.append(group.iloc[quota:])

    selected_count = sum(len(g) for g in selected_parts)
    remaining_needed = n_total - selected_count
    if remaining_needed > 0:
        extras_pool = pd.concat(extras, ignore_index=False) if extras else pd.DataFrame(columns=pdf.columns)
        extras_pool = extras_pool.sort_values(
            "__score__",
            ascending=not order_desc,
            kind="mergesort",
        )
        selected_parts.append(extras_pool.iloc[:remaining_needed])

    out = pd.concat(selected_parts, ignore_index=False)
    if len(out) != n_total:
        log_fn(
            f"[score_density_hybrid] density redistribution size mismatch "
            f"(expected {n_total}, got {len(out)}); falling back to original order.",
            always=True,
        )
        return pdf

    after_counts = out["__ipix__"].value_counts().sort_index().to_dict()
    if log_fn is not None:
        log_fn(
            "[sdh] density redistribution stats: "
            f"weight={weight:.3f}, "
            f"ipix={len(ipix_present)}, "
            f"before(min={min(before_counts.values(), default=0)}, "
            f"max={max(before_counts.values(), default=0)}), "
            f"after(min={min(after_counts.values(), default=0)}, "
            f"max={max(after_counts.values(), default=0)})",
            always=True,
        )
    return out


def prepare_score_density_hybrid(ddf: Any, cfg: Any, diag_ctx, log_fn):
    """Add __score__ column (from an expression) and restrict to a score window."""
    algo = cfg.algorithm
    score_expr = getattr(algo, "sdh_score_column", None)
    if not score_expr:
        raise ValueError("score_density_hybrid selection requires algorithm.sdh_score_column to be set.")

    score_expr = str(score_expr)
    score_range_mode = str(getattr(algo, "sdh_score_adaptive_range", "complete") or "complete").lower()
    if score_range_mode not in ("complete", "hist_peak"):
        raise ValueError("algorithm.sdh_score_adaptive_range must be 'complete' or 'hist_peak'.")

    score_hist_nbins = int(getattr(algo, "sdh_score_hist_nbins", getattr(algo, "sdh_score_hist_nbins", 512)))
    if score_hist_nbins <= 0:
        raise ValueError("algorithm.sdh_score_hist_nbins must be a positive integer.")

    base_meta_score = _get_meta_df(ddf)
    meta_with_score = base_meta_score.copy()
    meta_with_score["__score__"] = pd.Series([], dtype="float64")

    score_code = compile(score_expr, "<score_density_hybrid>", "eval")

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
    score_min_cfg = getattr(algo, "sdh_score_min", None)
    score_max_cfg = getattr(algo, "sdh_score_max", None)

    with diag_ctx("dask_score_minmax"):
        score_min_global_raw, score_max_global_raw = dask_compute(
            ddf[score_col_internal].min(),
            ddf[score_col_internal].max(),
        )

    if score_min_global_raw is None or score_max_global_raw is None:
        raise ValueError(
            "score_density_hybrid selection: unable to determine global score range "
            "(min/max returned None). Check the score expression/column."
        )

    score_min_global_raw = float(score_min_global_raw)
    score_max_global_raw = float(score_max_global_raw)

    if not np.isfinite(score_min_global_raw) or not np.isfinite(score_max_global_raw):
        raise ValueError(
            "score_density_hybrid selection: global score min/max are not finite. "
            "Check the score expression/column values."
        )

    if score_min_global_raw >= score_max_global_raw:
        raise ValueError(
            f"score_density_hybrid selection: invalid global score range "
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
                "score_density_hybrid selection: no objects found when estimating the "
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
                "[score_density_hybrid] score_min/score_max not provided; using global "
                f"range [{score_min:.6f}, {score_max:.6f}].",
                always=True,
            )
        else:
            peak_center, bin_left, bin_right = _get_peak_info()
            score_min = score_min_global_raw
            score_max = peak_center
            log_fn(
                "[score_density_hybrid] score_min/score_max not provided; using global minimum "
                f"{score_min:.6f} and histogram peak at {score_max:.6f} "
                f"(bin center from [{bin_left:.6f}, {bin_right:.6f}]).",
                always=True,
            )
    elif score_min is None:
        if score_range_mode == "complete":
            score_min = score_min_global_raw
            log_fn(
                "[score_density_hybrid] score_min not provided; using global minimum "
                f"{score_min:.6f} (score_max={score_max}).",
                always=True,
            )
        else:
            peak_center, bin_left, bin_right = _get_peak_info()
            score_min = peak_center
            if score_max is None:
                raise RuntimeError(
                    "score_density_hybrid: score_max must be provided when using hist_peak for score_min."
                )
            if score_min > float(score_max):
                raise ValueError(
                    "score_density_hybrid selection: histogram peak used as score_min "
                    f"({score_min:.6f}) is greater than the provided score_max "
                    f"({score_max})."
                )
            log_fn(
                "[score_density_hybrid] score_min not provided; using histogram peak at "
                f"{score_min:.6f} as minimum (bin center from "
                f"[{bin_left:.6f}, {bin_right:.6f}]).",
                always=True,
            )
    elif score_max is None:
        if score_range_mode == "complete":
            score_max = score_max_global_raw
            log_fn(
                "[score_density_hybrid] score_max not provided; using global maximum "
                f"{score_max:.6f} (score_min={score_min}).",
                always=True,
            )
        else:
            peak_center, bin_left, bin_right = _get_peak_info()
            score_max = peak_center
            if score_min is None:
                raise RuntimeError(
                    "score_density_hybrid: score_min must be provided when using hist_peak for score_max."
                )
            if float(score_min) > score_max:
                raise ValueError(
                    "score_density_hybrid selection: histogram peak used as score_max "
                    f"({score_max:.6f}) is smaller than the provided score_min "
                    f"({score_min})."
                )
            log_fn(
                "[score_density_hybrid] score_max not provided; using histogram peak at "
                f"{score_max:.6f} as maximum (bin center from "
                f"[{bin_left:.6f}, {bin_right:.6f}]).",
                always=True,
            )

    if score_min is None or score_max is None:
        raise RuntimeError("score_density_hybrid: internal error — score_min/score_max resolution failed.")

    if not (np.isfinite(score_min) and np.isfinite(score_max)):
        raise ValueError("score_density_hybrid selection: resolved score_min/score_max are not finite.")

    if score_min >= score_max:
        raise ValueError(
            f"algorithm.sdh_score_min ({score_min}) must be strictly smaller than "
            f"algorithm.sdh_score_max ({score_max}) for score_density_hybrid selection."
        )

    algo.sdh_score_min = score_min
    algo.sdh_score_max = score_max

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
    densmap_ref_base: Optional[np.ndarray] = None,
    densmap_ref_order: Optional[int] = None,
    id_col: Optional[str] = None,
    id_sink: Optional[set[int]] = None,
) -> None:
    """Execute the score_density_hybrid selection path and write tiles."""
    algo = cfg.algorithm
    score_col_internal = "__score__"
    if id_col is None:
        raise ValueError("id_col must be provided for score_density_hybrid selection.")
    if algo.sdh_score_min is None or algo.sdh_score_max is None:
        raise RuntimeError(
            "score_density_hybrid: internal error — score_min/score_max should have been set earlier."
        )

    score_min = float(algo.sdh_score_min)
    score_max = float(algo.sdh_score_max)
    depths_sel = list(range(1, cfg.algorithm.level_limit + 1))

    with diag_ctx("dask_score_hist"):
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

    fixed_targets = {
        1: getattr(algo, "sdh_n_1", None),
        2: getattr(algo, "sdh_n_2", None),
        3: getattr(algo, "sdh_n_3", None),
    }
    level_edges, targets_per_depth = _assign_targets_per_depth(
        densmaps=densmaps,
        depths_sel=depths_sel,
        fixed_targets=fixed_targets,
        cdf_hist=cdf_hist,
        score_edges_hist=score_edges_hist,
        score_min=score_min,
        score_max=score_max,
        n_tot_score=float(n_tot_score),
        log_fn=log_fn,
    )

    def _filter_selected(pdf: pd.DataFrame, idx_sel: np.ndarray, id_column: str) -> pd.DataFrame:
        if pdf is None or len(pdf) == 0:
            return pdf
        if id_column not in pdf.columns:
            return pdf
        return pdf.loc[~pdf[id_column].isin(idx_sel)]

    log_fn(
        "[selection] score_density_hybrid mode: per-depth score slices:\n"
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
        n_target = int(round(targets_per_depth[i])) if i < len(targets_per_depth) else None

        with diag_ctx(f"dask_depth_sdh_{depth:02d}"):
            if depth <= 3:
                depth_pdf = remainder_ddf.compute()
                if len(depth_pdf) == 0 or (n_target is not None and n_target <= 0):
                    log_fn(
                        f"[DEPTH {depth}] score_density_hybrid: nothing to select for this depth.",
                        always=True,
                    )
                    log_fn(
                        f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
                        always=True,
                    )
                    continue

                # Compute ipix for this depth
                ra_vals = pd.to_numeric(depth_pdf[ra_col], errors="coerce").to_numpy()
                dec_vals = pd.to_numeric(depth_pdf[dec_col], errors="coerce").to_numpy()
                theta = np.deg2rad(90.0 - dec_vals)
                phi = np.deg2rad(ra_vals % 360.0)
                NSIDE_L = 1 << depth
                ipixL = hp.ang2pix(NSIDE_L, theta, phi, nest=True).astype(np.int64)
                depth_pdf = depth_pdf.copy()
                depth_pdf["__ipix__"] = ipixL

                target_total = n_target if n_target is not None else len(depth_pdf)
                target_total = min(target_total, len(depth_pdf))

                base_order = (
                    int(densmap_ref_order)
                    if densmap_ref_order is not None
                    else int(getattr(algo, "sdh_coverage_order", cfg.algorithm.level_coverage))
                )
                if densmap_ref_base is not None:
                    counts_ref = np.asarray(densmap_ref_base, dtype=np.int64)
                else:
                    counts_ref_opt = densmaps.get(base_order, densmaps.get(depth))
                    counts_ref = densmaps[depth] if counts_ref_opt is None else counts_ref_opt
                if base_order >= depth:
                    counts_ref = _project_counts_to_order(counts_ref, base_order, depth)
                elif len(counts_ref) != len(densmaps[depth]):
                    counts_ref = densmaps[depth]

                # Subset to ipix present
                ipix_present = np.asarray(sorted(depth_pdf["__ipix__"].unique()), dtype=np.int64)
                counts_subset = counts_ref[ipix_present]
                weights_subset = _compute_density_weights(
                    counts_subset, _resolve_weight_for_level(algo, depth)
                )
                if weights_subset.sum() <= 0.0:
                    quotas = np.zeros_like(ipix_present, dtype=np.int64)
                else:
                    desired = weights_subset * float(target_total)
                    quotas = np.floor(desired).astype(np.int64)
                    remainder_quota = target_total - int(quotas.sum())
                    if remainder_quota > 0:
                        frac = desired - quotas.astype(np.float64)
                        rng = np.random.default_rng(int(getattr(algo, "sdh_shuffle_seed", 0) or 0))
                        frac = frac + rng.random(len(frac)) * 1e-6
                        order_q = np.argsort(-frac)
                        for idx_q in order_q[:remainder_quota]:
                            quotas[idx_q] += 1

                quota_map = {int(ip): int(q) for ip, q in zip(ipix_present, quotas, strict=False) if q > 0}

                # Select best per ipix up to quota
                depth_pdf = depth_pdf.sort_values(
                    "__score__", ascending=not cfg.algorithm.order_desc, kind="mergesort"
                )
                selected_parts: List[pd.DataFrame] = []
                leftovers: List[pd.DataFrame] = []
                for ipix_val, g in depth_pdf.groupby("__ipix__", sort=False):
                    quota = quota_map.get(int(ipix_val), 0)
                    if quota <= 0:
                        leftovers.append(g)
                        continue
                    if quota >= len(g):
                        selected_parts.append(g)
                    else:
                        selected_parts.append(g.iloc[:quota])
                        leftovers.append(g.iloc[quota:])

                selected_pdf = (
                    pd.concat(selected_parts, ignore_index=False)
                    if selected_parts
                    else pd.DataFrame(columns=depth_pdf.columns)
                )

                remaining_needed = target_total - len(selected_pdf)
                if remaining_needed > 0 and leftovers:
                    extras_pool = pd.concat(leftovers, ignore_index=False)
                    extras_pool = extras_pool.sort_values(
                        "__score__", ascending=not cfg.algorithm.order_desc, kind="mergesort"
                    )
                    selected_pdf = pd.concat(
                        [selected_pdf, extras_pool.iloc[:remaining_needed]], ignore_index=False
                    )

                _log_depth_stats(
                    log_fn,
                    depth,
                    "selected",
                    counts=densmaps[depth],
                    selected_len=len(selected_pdf),
                )

                if len(selected_pdf) == 0:
                    log_fn(
                        f"[DEPTH {depth}] score_density_hybrid: no rows selected for this depth.",
                        always=True,
                    )
                    log_fn(
                        f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
                        always=True,
                    )
                    continue

                # Remove selected from remainder for next depths
                remainder_meta = _get_meta_df(remainder_ddf)
                remainder_ddf = remainder_ddf.map_partitions(
                    _filter_selected,
                    selected_pdf[id_col].to_numpy(),
                    id_col,
                    meta=remainder_meta,
                )
                if id_col and id_sink is not None and id_col in selected_pdf:
                    id_sink.update(selected_pdf[id_col].astype(int).tolist())
                if id_col and id_col in selected_pdf:
                    selected_pdf = selected_pdf.drop(columns=[id_col])
            else:
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
                ipixL = hp.ang2pix(
                    NSIDE_L,
                    theta,
                    phi,
                    nest=True,
                ).astype(np.int64)
                selected_pdf["__ipix__"] = ipixL

                remainder_meta = _get_meta_df(remainder_ddf)
                remainder_ddf = remainder_ddf.map_partitions(
                    _filter_selected,
                    selected_pdf[id_col].to_numpy(),
                    id_col,
                    meta=remainder_meta,
                )
                if id_col and id_sink is not None and id_col in selected_pdf:
                    id_sink.update(selected_pdf[id_col].astype(int).tolist())
                if id_col and id_col in selected_pdf:
                    selected_pdf = selected_pdf.drop(columns=[id_col])

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
