"""Shared selection utilities for HEALPix-aware slicing."""

from __future__ import annotations

from typing import Any, Dict

import healpy as hp
import numpy as np
import pandas as pd

from ..utils import _get_dask_base, _get_meta_df

__all__ = ["targets_per_tile", "reduce_topk_by_group_dask", "add_ipix_column"]


def _as_plain_pandas_frame(pdf: pd.DataFrame) -> pd.DataFrame:
    """Return a plain pandas DataFrame, avoiding backend sort/groupby overrides."""
    if type(pdf) is pd.DataFrame:
        return pdf
    return pd.DataFrame(pdf)


def _sort_by_plain_keys(pdf: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> pd.DataFrame:
    """Sort using a plain helper frame with only key columns, then select rows by position."""
    if pdf.empty:
        return pdf

    key_df = pd.DataFrame(index=pd.RangeIndex(len(pdf)))
    for col in sort_cols:
        values = pd.Series(pdf[col].to_numpy(), index=key_df.index)
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().any() or values.isna().all():
            key_df[col] = numeric
        else:
            key_df[col] = values.astype("string")

    order = key_df.sort_values(sort_cols, ascending=ascending, kind="mergesort").index.to_numpy()
    return pdf.iloc[order]


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
    tie_col: str | None = None,
):
    """Keep up to k_per_group rows per group, sorted by score then RA/DEC.

    Uses a two-stage exact strategy:
    1) per-partition local top-k pruning (no global shuffle),
    2) global top-k by group on the pruned collection.
    """
    if not k_per_group:
        empty_meta = _get_meta_df(ddf_like)
        return ddf_like.map_partitions(lambda pdf: pdf.iloc[0:0], meta=empty_meta)

    asc = not order_desc
    k_map = {int(k): int(v) for k, v in k_per_group.items()}
    groups_allowed = set(k_map.keys())

    def _take_topk(group: pd.DataFrame) -> pd.DataFrame:
        """Select the top-k rows for a single group."""
        group = _as_plain_pandas_frame(group)
        if group.empty:
            return group
        g_id = int(group[group_col].iloc[0])
        k = int(k_map.get(g_id, 0))
        if k <= 0:
            return group.iloc[0:0]
        sort_cols = [score_col]
        ascending = [asc]
        if tie_col and tie_col in group.columns:
            sort_cols.append(tie_col)
            ascending.append(True)
        if ra_col in group.columns:
            sort_cols.append(ra_col)
            ascending.append(True)
        if dec_col in group.columns:
            sort_cols.append(dec_col)
            ascending.append(True)
        group_sorted = _sort_by_plain_keys(group, sort_cols, ascending)
        return group_sorted.head(k)

    def _local_topk_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        """Exact local pruning: keep only per-group top-k within this partition."""
        pdf = _as_plain_pandas_frame(pdf)
        if pdf.empty:
            return pdf.iloc[0:0]

        # Drop groups that are not requested for this depth.
        pdf = pdf[pdf[group_col].isin(groups_allowed)]
        if pdf.empty:
            return pdf.iloc[0:0]
        sort_cols = [group_col, score_col]
        ascending = [True, asc]
        if tie_col and tie_col in pdf.columns:
            sort_cols.append(tie_col)
            ascending.append(True)
        if ra_col in pdf.columns:
            sort_cols.append(ra_col)
            ascending.append(True)
        if dec_col in pdf.columns:
            sort_cols.append(dec_col)
            ascending.append(True)

        pdf_sorted = _sort_by_plain_keys(pdf, sort_cols, ascending)
        rank_in_group = pdf_sorted.groupby(group_col, sort=False).cumcount()
        k_by_row = pdf_sorted[group_col].map(k_map).fillna(0).astype("int64")
        keep_mask = rank_in_group.to_numpy() < k_by_row.to_numpy()
        return pdf_sorted.loc[keep_mask]

    meta = _get_meta_df(ddf_like)
    cols_all = list(meta.columns)

    base = _get_dask_base(ddf_like, require_groupby=True)
    if hasattr(base, "groupby"):
        cols_all = list(meta.columns)
        if hasattr(base, "map_partitions"):
            pruned = base.map_partitions(_local_topk_partition, meta=meta)
            return pruned.groupby(group_col, group_keys=False)[cols_all].apply(_take_topk, meta=meta)
        # Compatibility fallback for test doubles / non-Dask backends that
        # implement groupby-apply but not map_partitions.
        return base.groupby(group_col, group_keys=False)[cols_all].apply(_take_topk, meta=meta)

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
