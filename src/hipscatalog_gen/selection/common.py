from __future__ import annotations

from typing import Any, Dict, List, Tuple

import healpy as hp
import numpy as np
import pandas as pd
from lsdb.catalog import Catalog as LsdbCatalog

from ..score_global.utils import _quantile_from_histogram
from ..utils import _get_meta_df

__all__ = [
    "assign_level_edges",
    "targets_per_tile",
    "reduce_topk_by_group_dask",
    "add_ipix_column",
]


def assign_level_edges(
    densmaps: Dict[int, np.ndarray],
    depths_sel: List[int],
    fixed_targets: Dict[int, float],
    cdf_hist: np.ndarray,
    score_edges_hist: np.ndarray,
    score_min: float,
    score_max: float,
    n_tot_score: float,
    log_fn,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cumulative targets per depth and corresponding score edges."""
    weights_list: List[float] = []
    for d in depths_sel:
        counts_d = densmaps[d]
        tiles_active = int((counts_d > 0).sum())
        weights_list.append(max(1, tiles_active))

    weights = np.asarray(weights_list, dtype="float64")
    T = np.zeros_like(weights, dtype="float64")

    fixed_norm: Dict[int, float] = {}
    for d, val in fixed_targets.items():
        if d not in depths_sel:
            continue
        if int(val) < 0:
            raise ValueError(f"{label}: fixed target for depth {d} must be non-negative (got {val}).")
        fixed_norm[int(d)] = float(int(val))

    sum_fixed = float(sum(fixed_norm.values()))
    if sum_fixed > n_tot_score and sum_fixed > 0.0:
        scale = float(n_tot_score) / sum_fixed if sum_fixed > 0 else 0.0
        log_fn(
            f"[{label}] Sum of fixed targets ({int(sum_fixed)}) exceeds total objects "
            f"in range ({int(n_tot_score)}). Rescaling by factor {scale:.3f}.",
            always=True,
        )
        for d in list(fixed_norm.keys()):
            fixed_norm[d] *= scale
        sum_fixed = float(n_tot_score)

    for d, val in fixed_norm.items():
        idx = depths_sel.index(d)
        T[idx] = val

    N_rem = max(0.0, float(n_tot_score) - sum_fixed)
    if N_rem > 0.0:
        free_mask = np.ones_like(weights, dtype=bool)
        for d in fixed_norm:
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
    return level_edges, T


def targets_per_tile(counts_depth: np.ndarray, depth_total: int, bias: float) -> Dict[int, int]:
    """Distribute depth_total across active tiles with optional density bias."""
    if depth_total <= 0:
        return {}

    active_idx = np.nonzero(counts_depth > 0)[0]
    if len(active_idx) == 0:
        return {}

    weights_uniform = np.ones(len(active_idx), dtype="float64")
    weights_uniform /= float(weights_uniform.sum())

    dens_vals = counts_depth[active_idx].astype("float64")
    dens_weights = dens_vals / float(dens_vals.sum()) if dens_vals.sum() > 0 else weights_uniform.copy()

    bias = max(0.0, min(1.0, float(bias)))
    weights = (1.0 - bias) * weights_uniform + bias * dens_weights
    weights = weights / float(weights.sum()) if weights.sum() > 0 else weights_uniform

    raw = weights * float(depth_total)
    base = np.floor(raw).astype(int)
    remainder = int(depth_total - base.sum())
    if remainder > 0:
        frac = raw - base
        order = np.argsort(-frac, kind="mergesort")
        for idx in order[:remainder]:
            base[idx] += 1

    return {int(ipix): int(val) for ipix, val in zip(active_idx, base, strict=False) if val > 0}


def reduce_topk_by_group_dask(
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


def add_ipix_column(pdf: pd.DataFrame, depth: int, ra_col: str, dec_col: str) -> pd.DataFrame:
    """Attach __ipix__ for a given depth."""
    if pdf.empty:
        pdf["__ipix__"] = pd.Series([], dtype="int64")
        return pdf

    ra_vals = pd.to_numeric(pdf[ra_col], errors="coerce").to_numpy()
    dec_vals = pd.to_numeric(pdf[dec_col], errors="coerce").to_numpy()
    theta = np.deg2rad(90.0 - dec_vals)
    phi = np.deg2rad(ra_vals % 360.0)
    nside = 1 << depth
    ipix = hp.ang2pix(nside, theta, phi, nest=True).astype(np.int64)

    pdf = pdf.copy()
    pdf["__ipix__"] = ipix
    return pdf
