from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lsdb.catalog import Catalog as LsdbCatalog

from .utils import _get_meta_df


__all__ = [
    "build_cov_thresholds",
    "filter_remainder_by_coverage_partition",
    "_candidates_by_coverage_partition",
    "_reduce_coverage_exact",
    "_reduce_coverage_exact_dask",
    "apply_fractional_k_per_cov",
]


# =============================================================================
# Coverage-based selection helpers
# =============================================================================


def build_cov_thresholds(
    selected: pd.DataFrame,
    score_col: str,
    order_desc: bool,
) -> Dict[int, float]:
    """Compute per-coverage score thresholds from already kept rows.

    For each coverage cell (__icov__), the worst kept score is recorded:
        * ascending score → max(score)
        * descending score → min(score)

    Args:
        selected: DataFrame with already kept rows (must have __icov__ and score_col).
        score_col: Score column name.
        order_desc: If True, scores are sorted in descending order.

    Returns:
        Mapping {icov: threshold_score}.
    """
    if len(selected) == 0:
        return {}

    if order_desc:
        s = selected.groupby("__icov__")[score_col].min()
    else:
        s = selected.groupby("__icov__")[score_col].max()

    return {int(k): float(v) for k, v in s.items()}


def filter_remainder_by_coverage_partition(
    pdf: pd.DataFrame,
    score_expr: str,
    order_desc: bool,
    thr_cov: Dict[int, float],
    ra_col: str,
    dec_col: str,
) -> pd.DataFrame:
    """Filter remainder rows so kept rows never reappear.

    Rows from pixels present in thr_cov are kept only if they are strictly
    worse than the threshold for that pixel. Rows from pixels not in thr_cov
    are kept as they are.

    Args:
        pdf: Input pandas DataFrame partition.
        score_expr: Score column or expression.
        order_desc: If True, higher scores are better.
        thr_cov: Mapping {icov: threshold_score}.
        ra_col: RA column name (unused, kept for interface consistency).
        dec_col: DEC column name (unused, kept for interface consistency).

    Returns:
        Filtered pandas DataFrame.
    """
    if len(pdf) == 0:
        return pdf

    # Score computation: direct column or expression.
    if score_expr in pdf.columns:
        sc = pd.to_numeric(pdf[score_expr], errors="coerce")
    else:
        code = compile(score_expr, "<score>", "eval")
        env = {"__builtins__": {}, "np": np, "numpy": np}
        env.update({col: pdf[col] for col in pdf.columns})
        out = eval(code, env, {})
        sc = pd.to_numeric(out, errors="coerce")

    sc = sc.replace([np.inf, -np.inf], np.nan)
    sc = sc.fillna(-np.inf if order_desc else np.inf)

    if "__icov__" not in pdf.columns:
        return pdf

    icov = pdf["__icov__"].to_numpy()
    thr = np.array([thr_cov.get(int(c), None) for c in icov], dtype=object)

    if order_desc:
        mask = np.array([(t is None) or (s < t) for s, t in zip(sc.values, thr)], dtype=bool)
    else:
        mask = np.array([(t is None) or (s > t) for s, t in zip(sc.values, thr)], dtype=bool)

    return pdf.loc[mask]


def _candidates_by_coverage_partition(
    pdf: pd.DataFrame,
    score_col: str,
    order_desc: bool,
    k_per_cov: int | Dict[int, float],
    tie_buffer: int,
    ra_col: str,
    dec_col: str,
) -> pd.DataFrame:
    """Select candidate rows per coverage cell (__icov__) with a tie buffer.

    Keeps up to (k + tie_buffer) rows per coverage cell:
        * If k_per_cov is scalar → same k for all cells.
        * If dict → k_per_cov[icov] for each cell.

    Args:
        pdf: Input pandas DataFrame.
        score_col: Column used for ranking within each coverage cell.
        order_desc: If True, higher scores are better.
        k_per_cov: Scalar or mapping {icov: k}.
        tie_buffer: Extra rows kept near the cut to avoid hard score edges.
        ra_col: RA column name, used as tie-breaker.
        dec_col: DEC column name, used as tie-breaker.

    Returns:
        DataFrame with candidate rows, including at most (k + tie_buffer)
        rows per coverage cell.
    """
    if len(pdf) == 0:
        return pdf.iloc[0:0]

    asc = not order_desc
    pdf = pdf.sort_values(
        ["__icov__", score_col, ra_col, dec_col],
        ascending=[True, asc, True, True],
        kind="mergesort",
    )
    pdf["__rk__"] = pdf.groupby("__icov__").cumcount()

    if isinstance(k_per_cov, dict):
        def _keep_group(g: pd.DataFrame) -> pd.DataFrame:
            ic = int(g["__icov__"].iloc[0])
            k = int(k_per_cov.get(ic, 0))
            need = max(0, k + int(tie_buffer))
            return g.iloc[:need]

        out = pdf.groupby("__icov__", group_keys=False).apply(_keep_group)
    else:
        need = int(k_per_cov) + int(tie_buffer)
        out = pdf[pdf["__rk__"] < need]

    return out.drop(columns=["__rk__"])


def _reduce_coverage_exact(
    pdf: pd.DataFrame,
    score_col: str,
    order_desc: bool,
    k_per_cov: int | Dict[int, float],
    ra_col: str,
    dec_col: str,
) -> pd.DataFrame:
    """Keep up to k rows per coverage cell (__icov__), per pandas DataFrame.

    If a cell has fewer than k rows, all rows are kept.

    Args:
        pdf: Input pandas DataFrame.
        score_col: Column used for ranking within each coverage cell.
        order_desc: If True, higher scores are better.
        k_per_cov: Scalar or mapping {icov: k}.
        ra_col: RA column name (tie-breaker).
        dec_col: DEC column name (tie-breaker).

    Returns:
        DataFrame with up to k rows per coverage cell.
    """
    if len(pdf) == 0:
        return pdf.iloc[0:0]

    asc = not order_desc
    out: List[pd.DataFrame] = []

    for icov, g in pdf.groupby("__icov__"):
        k = int(k_per_cov.get(int(icov), 0)) if isinstance(k_per_cov, dict) else int(k_per_cov)
        if k <= 0 or len(g) == 0:
            continue
        gg = g.sort_values(
            [score_col, ra_col, dec_col],
            ascending=[asc, True, True],
            kind="mergesort",
        ).head(k)
        out.append(gg)

    return pd.concat(out, ignore_index=True) if out else pdf.iloc[0:0]


def _reduce_coverage_exact_dask(
    ddf_like: Any,
    score_col: str,
    order_desc: bool,
    k_per_cov: int | Dict[int, float],
    ra_col: str,
    dec_col: str,
) -> Any:
    """Dask-based equivalent of `_reduce_coverage_exact`.

    Keeps up to k rows per coverage cell (__icov__), with the same ordering
    and tie-breaking rules as the pandas version:
        * sort by score_col, then RA, then DEC
        * if a cell has fewer rows than k, keep them all

    Args:
        ddf_like: Dask DataFrame or LSDB Catalog.
        score_col: Column used for ranking within each coverage cell.
        order_desc: If True, higher scores are better.
        k_per_cov: Scalar or mapping {icov: k}.
        ra_col: RA column name.
        dec_col: DEC column name.

    Returns:
        Dask collection with up to k rows per coverage cell.
    """
    asc = not order_desc
    is_dict = isinstance(k_per_cov, dict)

    # Scalar and non-positive → empty result of same schema.
    if (not is_dict) and int(k_per_cov) <= 0:
        empty_meta = _get_meta_df(ddf_like)
        return ddf_like.map_partitions(lambda pdf: pdf.iloc[0:0], meta=empty_meta)

    def _take_topk(group: pd.DataFrame) -> pd.DataFrame:
        """Select top-k rows within a single __icov__ group."""
        if group.empty:
            return group

        if is_dict:
            icov_val = int(group["__icov__"].iloc[0])
            k = int(k_per_cov.get(icov_val, 0))
        else:
            k = int(k_per_cov)

        if k <= 0:
            return group.iloc[0:0]

        group_sorted = group.sort_values(
            [score_col, ra_col, dec_col],
            ascending=[asc, True, True],
            kind="mergesort",
        )
        return group_sorted.head(k)

    meta = _get_meta_df(ddf_like)

    # LSDB Catalog: operate on internal Dask DataFrame.
    if isinstance(ddf_like, LsdbCatalog) and hasattr(ddf_like, "_ddf"):
        base_ddf = ddf_like._ddf  # type: ignore[attr-defined]
        return base_ddf.groupby("__icov__", group_keys=False).apply(_take_topk, meta=meta)

    # Generic Dask DataFrame path.
    if hasattr(ddf_like, "groupby"):
        return ddf_like.groupby("__icov__", group_keys=False).apply(_take_topk, meta=meta)

    # Fallback: unsupported type → return input unchanged.
    return ddf_like


def apply_fractional_k_per_cov(
    selected: pd.DataFrame,
    k_desired: float,
    score_col: str,
    order_desc: bool,
    mode: str = "random",
    mode_logic: str = "auto",
    ra_col: str | None = None,
    dec_col: str | None = None,
    k_per_cov_desired_map: Optional[Dict[int, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply fractional-k adjustment on top of per-coverage selection.

    This function adjusts the integer per-coverage selection using:
        * local or global logic, and
        * random or score-based selection.

    Args:
        selected: DataFrame limited to <= k_int rows per __icov__.
        k_desired: Base expected number of rows per coverage cell (can be fractional).
        score_col: Score column name for ranking.
        order_desc: If True, higher scores are better.
        mode: "random" or "score".
        mode_logic: "auto", "local" or "global".
        ra_col: RA column name (used as tie-breaker when needed).
        dec_col: DEC column name (used as tie-breaker when needed).
        k_per_cov_desired_map: Optional mapping {icov: k_desired_icov}.

    Returns:
        Tuple (kept_df, dropped_df), where:
            kept_df: Rows actually kept at this depth.
            dropped_df: Rows not written at this depth (available for deeper levels).

    Raises:
        ValueError: If the mode or logic is unknown, or if `selected` is None.
    """
    if selected is None:
        raise ValueError("apply_fractional_k_per_cov expects a pandas.DataFrame, got None.")

    if len(selected) == 0 or k_desired <= 0.0:
        return selected, selected.iloc[0:0]

    mode = (mode or "random").lower()
    logic_raw = (mode_logic or "auto").lower()

    if mode not in ("random", "score"):
        raise ValueError(f"Unknown fractional_mode: {mode!r}")

    # "auto" preserves backward-compatible behavior.
    if logic_raw == "auto":
        logic = "global" if mode == "score" else "local"
    else:
        if logic_raw not in ("local", "global"):
            raise ValueError(f"Unknown fractional_mode_logic: {mode_logic!r}")
        logic = logic_raw

    if "__icov__" not in selected.columns:
        return selected, selected.iloc[0:0]

    def _k_desired_for_icov(icov: int, n_avail: int) -> float:
        if k_per_cov_desired_map is not None:
            v = k_per_cov_desired_map.get(int(icov), k_desired)
        else:
            v = k_desired
        return min(float(v), float(n_avail))

    # ------------------------------------------------------------------
    # mode == "score" and logic == "global"
    # ------------------------------------------------------------------
    if mode == "score" and logic == "global":
        n_cov = int(selected["__icov__"].nunique())
        if n_cov == 0:
            return selected.iloc[0:0], selected

        k_total_desired = float(k_desired) * float(n_cov)
        n_sel = len(selected)
        n_keep = int(round(k_total_desired))
        n_keep = max(0, min(n_keep, n_sel))

        if n_keep == 0:
            return selected.iloc[0:0], selected

        asc = not order_desc
        sort_cols = [score_col]
        ascending = [asc]

        if ra_col is not None and ra_col in selected.columns:
            sort_cols.append(ra_col)
            ascending.append(True)
        if dec_col is not None and dec_col in selected.columns:
            sort_cols.append(dec_col)
            ascending.append(True)

        selected_sorted = selected.sort_values(
            sort_cols,
            ascending=ascending,
            kind="mergesort",
        )

        kept_df = selected_sorted.iloc[:n_keep]
        dropped_df = selected_sorted.iloc[n_keep:]
        return kept_df, dropped_df

    # ------------------------------------------------------------------
    # mode == "score" and logic == "local"
    # ------------------------------------------------------------------
    if mode == "score" and logic == "local":
        asc = not order_desc
        rng = np.random.default_rng()

        kept_parts: List[pd.DataFrame] = []
        dropped_parts: List[pd.DataFrame] = []

        for icov, g in selected.groupby("__icov__"):
            if g.empty:
                continue

            sort_cols = [score_col]
            ascending = [asc]
            if ra_col is not None and ra_col in g.columns:
                sort_cols.append(ra_col)
                ascending.append(True)
            if dec_col is not None and dec_col in g.columns:
                sort_cols.append(dec_col)
                ascending.append(True)

            g_sorted = g.sort_values(
                sort_cols,
                ascending=ascending,
                kind="mergesort",
            )

            n_avail = len(g_sorted)
            k_local_desired = _k_desired_for_icov(icov, n_avail)

            k_floor = int(math.floor(k_local_desired))
            k_floor = max(0, k_floor)

            if k_floor > 0:
                base_keep = g_sorted.iloc[:k_floor]
            else:
                base_keep = g_sorted.iloc[0:0]

            remaining = g_sorted.iloc[k_floor:]

            extra_keep = 0
            frac_local = max(0.0, min(1.0, k_local_desired - float(k_floor)))
            if len(remaining) > 0 and frac_local > 0.0:
                if rng.random() < frac_local:
                    extra_keep = 1

            if extra_keep == 1:
                extra_row = remaining.iloc[:1]
                final_keep = pd.concat([base_keep, extra_row], ignore_index=False)
                final_drop = remaining.iloc[1:]
            else:
                final_keep = base_keep
                final_drop = remaining

            if len(final_keep) > 0:
                kept_parts.append(final_keep)
            if len(final_drop) > 0:
                dropped_parts.append(final_drop)

        kept_df = (
            pd.concat(kept_parts, ignore_index=False)
            if kept_parts
            else selected.iloc[0:0]
        )
        dropped_df = (
            pd.concat(dropped_parts, ignore_index=False)
            if dropped_parts
            else selected.iloc[0:0]
        )
        return kept_df, dropped_df

    # ------------------------------------------------------------------
    # mode == "random" and logic == "global"
    # ------------------------------------------------------------------
    if mode == "random" and logic == "global":
        n_cov = int(selected["__icov__"].nunique())
        if n_cov == 0:
            return selected.iloc[0:0], selected

        k_total_desired = float(k_desired) * float(n_cov)
        n_sel = len(selected)
        n_keep = int(round(k_total_desired))
        n_keep = max(0, min(n_keep, n_sel))

        if n_keep == 0:
            return selected.iloc[0:0], selected

        shuffled = selected.sample(frac=1.0, replace=False)
        kept_df = shuffled.iloc[:n_keep]
        dropped_df = shuffled.iloc[n_keep:]
        return kept_df, dropped_df

    # ------------------------------------------------------------------
    # mode == "random" and logic == "local"
    # ------------------------------------------------------------------
    k_floor_global = math.floor(k_desired)
    frac_global = k_desired - k_floor_global

    if frac_global <= 1e-9 and k_per_cov_desired_map is None:
        return selected, selected.iloc[0:0]

    rng = np.random.default_rng()
    kept_parts: List[pd.DataFrame] = []
    dropped_parts: List[pd.DataFrame] = []

    for icov, g in selected.groupby("__icov__"):
        n_avail = len(g)
        if n_avail == 0:
            continue

        k_local_desired = _k_desired_for_icov(icov, n_avail)

        k_floor = int(math.floor(k_local_desired))
        k_floor = max(0, k_floor)

        if k_floor > 0:
            perm = rng.permutation(n_avail)
            base_idx = perm[:k_floor]
            remaining_idx = perm[k_floor:]

            base_keep = g.iloc[base_idx]
            remaining = g.iloc[remaining_idx]
        else:
            base_keep = g.iloc[0:0]
            remaining = g

        extra_keep = 0
        frac_local = max(0.0, min(1.0, k_local_desired - float(k_floor)))

        if len(remaining) > 0 and frac_local > 0.0:
            if rng.random() < frac_local:
                extra_keep = 1

        if extra_keep == 1:
            j = rng.integers(len(remaining))
            extra_row = remaining.iloc[[j]]
            remaining_without_extra = remaining.drop(remaining.index[j])

            final_keep = pd.concat([base_keep, extra_row], ignore_index=False)
            final_drop = remaining_without_extra
        else:
            final_keep = base_keep
            final_drop = remaining

        if len(final_keep) > 0:
            kept_parts.append(final_keep)
        if len(final_drop) > 0:
            dropped_parts.append(final_drop)

    kept_df = (
        pd.concat(kept_parts, ignore_index=False)
        if kept_parts
        else selected.iloc[0:0]
    )
    dropped_df = (
        pd.concat(dropped_parts, ignore_index=False)
        if dropped_parts
        else selected.iloc[0:0]
    )
    return kept_df, dropped_df
