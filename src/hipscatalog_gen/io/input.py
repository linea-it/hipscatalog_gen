"""Input readers for Parquet/CSV/TSV and HATS/LSDB catalogs."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, cast

import dask.dataframe as dd
import lsdb
import numpy as np
import pandas as pd
from dask import compute as dask_compute
from lsdb.catalog import Catalog as LsdbCatalog

from ..config import Config
from ..utils import _ID_RE, _get_dask_base, _get_meta_df, _resolve_col_name, _score_deps

__all__ = [
    "_build_input_ddf",
    "compute_column_report_sample",
    "compute_column_report_global",
]


# =============================================================================
# Build Dask / LSDB input collection
# =============================================================================


def _unique_available(cols: List[Any], available_cols: List[Any]) -> List[Any]:
    """Return unique columns in input order, filtered by availability."""
    avail = set(available_cols)
    out: List[Any] = []
    seen: set[Any] = set()
    for c in cols:
        if c in avail and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _resolve_keep_columns_order(
    *,
    available_cols: List[Any],
    ra_name: Any,
    dec_name: Any,
    must_keep: List[Any],
    requested_keep_cfg: List[Any] | None,
) -> List[Any]:
    """Resolve final output column order based on columns.keep semantics."""
    must_keep_unique = _unique_available(must_keep, available_cols)

    # keep omitted/null -> preserve input catalog order.
    if requested_keep_cfg is None:
        return list(available_cols)

    requested_keep_unique = _unique_available(list(requested_keep_cfg), available_cols)

    # keep provided but empty -> RA/DEC first, then remaining required dependencies.
    if len(requested_keep_cfg) == 0:
        lead = _unique_available([ra_name, dec_name], available_cols)
        tail = [c for c in must_keep_unique if c not in lead]
        return [*lead, *tail]

    # keep provided and non-empty:
    # - if keep already contains all required columns, keep order wins;
    # - otherwise required missing columns go first, then keep order.
    missing_required = [c for c in must_keep_unique if c not in requested_keep_unique]
    if not missing_required:
        return requested_keep_unique

    # Special case: if RA/DEC are missing from keep, they lead the missing block.
    lead_missing = [c for c in (ra_name, dec_name) if c in missing_required]
    rest_missing = [c for c in missing_required if c not in lead_missing]
    return [*lead_missing, *rest_missing, *requested_keep_unique]


def _build_input_ddf(paths: List[str], cfg: Config) -> tuple[Any, str, str, List[str]]:
    """Build the main input collection for the pipeline.

    Supports Parquet/CSV/TSV and HATS/LSDB catalogs.

    Args:
        paths: List of resolved input file paths (after globbing).
        cfg: Parsed configuration object.

    Returns:
        Tuple (ddf_like, ra_name, dec_name, keep_cols) where:
            ddf_like: Dask-like collection (dd.DataFrame or LSDB Catalog).
            ra_name: Resolved RA column name.
            dec_name: Resolved DEC column name.
            keep_cols: Final ordered list of columns to keep (tile header order).
    """
    if not paths:
        raise ValueError("No input files matched.")

    fmt = cfg.input.format.lower()
    # Single declaration for the whole function (avoid no-redef)
    mag_col_cfg: str | None = None
    flux_col_cfg: str | None = None
    score_global_expr = getattr(cfg.algorithm, "score_column", None) or ""
    sdh_score_expr = getattr(cfg.algorithm, "sdh_score_column", None) or ""
    selection_mode = getattr(cfg.algorithm, "selection_mode", "mag_global").lower()
    if selection_mode == "score_global":
        active_score_expr = score_global_expr
    elif selection_mode == "score_density_hybrid":
        active_score_expr = sdh_score_expr
    else:
        active_score_expr = ""
    if selection_mode == "mag_global":
        mag_col_cfg = cfg.algorithm.mag_column
        flux_col_cfg = getattr(cfg.algorithm, "flux_column", None)

    # If columns.keep is None, preserve all input columns.
    keep_all_columns = cfg.columns.keep is None

    # ------------------------------------------------------------------
    # HATS / LSDB input: keep LSDB structure
    # ------------------------------------------------------------------
    if fmt == "hats":
        if len(paths) != 1:
            raise ValueError(
                "For input.format='hats', please specify exactly one HATS catalog path in input.paths."
            )

        hats_path = paths[0]

        # Columns explicitly requested by the user in the YAML.
        requested_keep_cfg = cfg.columns.keep
        requested_keep = requested_keep_cfg or []

        # Extract potential score dependencies from the score expression.
        score_tokens = set(_ID_RE.findall(str(active_score_expr))) if active_score_expr else set()

        # Always request RA, DEC and score dependencies; mag/flux if applicable.
        must_keep = [cfg.columns.ra, cfg.columns.dec, *score_tokens]
        if mag_col_cfg:
            must_keep.append(mag_col_cfg)
        if flux_col_cfg:
            must_keep.append(flux_col_cfg)

        needed_cols: List[str] = []
        seen_needed: set[str] = set()
        for c in [*must_keep, *requested_keep]:
            if c and (c not in seen_needed):
                needed_cols.append(c)
                seen_needed.add(c)

        # If columns.keep is None → open all columns.
        if needed_cols and not keep_all_columns:
            cat0 = cast(LsdbCatalog, lsdb.open_catalog(hats_path, columns=needed_cols))
        else:
            cat0 = cast(LsdbCatalog, lsdb.open_catalog(hats_path))

        available_cols = list(cat0.columns)

        # HATS catalog always has named columns → header=True.
        ra_col = _resolve_col_name(
            cfg.columns.ra,
            cat0,  # type: ignore[arg-type]
            header=True,
        )
        dec_col = _resolve_col_name(
            cfg.columns.dec,
            cat0,  # type: ignore[arg-type]
            header=True,
        )
        RA_NAME = ra_col
        DEC_NAME = dec_col

        score_dependencies = [c for c in score_tokens if c in available_cols]

        must_keep_resolved = [RA_NAME, DEC_NAME, *score_dependencies]
        if mag_col_cfg and mag_col_cfg in available_cols:
            must_keep_resolved.append(mag_col_cfg)
        if flux_col_cfg and flux_col_cfg in available_cols:
            must_keep_resolved.append(flux_col_cfg)

        keep_cols_out = _resolve_keep_columns_order(
            available_cols=available_cols,
            ra_name=RA_NAME,
            dec_name=DEC_NAME,
            must_keep=must_keep_resolved,
            requested_keep_cfg=requested_keep_cfg,
        )

        # Sub-select via LSDB API; returns a new Catalog. Convert to a Dask DF-friendly
        # object to keep meta valid for future Dask releases.
        ddf_sel = cast(Any, cat0)[keep_cols_out]
        ddf_base = _get_dask_base(ddf_sel, require_map_partitions=True)
        meta = _get_meta_df(ddf_base)
        ddf = ddf_base.map_partitions(lambda pdf: pdf, meta=meta)
        return ddf, RA_NAME, DEC_NAME, keep_cols_out

    # ------------------------------------------------------------------
    # Standard Parquet / CSV / TSV input
    # ------------------------------------------------------------------

    # 1) Base read to discover columns and resolve RA/DEC.
    if fmt == "parquet":
        ddf0 = dd.read_parquet(paths, engine="pyarrow")
    elif fmt in ("csv", "tsv"):
        ascii_fmt = (cfg.input.ascii_format or "").upper().strip()
        if ascii_fmt in ("CSV", ""):
            sep = ","
        elif ascii_fmt == "TSV":
            sep = "\t"
        else:
            sep = "," if fmt == "csv" else "\t"

        if cfg.input.header:
            ddf0 = dd.read_csv(paths, sep=sep, assume_missing=True)
        else:
            ddf0 = dd.read_csv(paths, sep=sep, header=None, assume_missing=True)
    else:
        raise ValueError("Unsupported input.format; use 'parquet', 'csv', 'tsv', or 'hats'.")

    # Resolve RA/DEC.
    ra_col = _resolve_col_name(
        cfg.columns.ra,
        ddf0,
        header=(fmt == "parquet" or cfg.input.header),
    )
    dec_col = _resolve_col_name(
        cfg.columns.dec,
        ddf0,
        header=(fmt == "parquet" or cfg.input.header),
    )
    RA_NAME = ra_col
    DEC_NAME = dec_col

    # 2) Column selection (preserve order; ensure score deps).
    available_cols = list(ddf0.columns)
    score_dependencies = _score_deps(active_score_expr, available_cols)

    requested_keep_cfg = cfg.columns.keep

    flux_col_cfg = getattr(cfg.algorithm, "flux_column", None)

    must_keep = [RA_NAME, DEC_NAME, *score_dependencies]
    if mag_col_cfg and mag_col_cfg in available_cols:
        must_keep.append(mag_col_cfg)
    if flux_col_cfg and flux_col_cfg in available_cols:
        must_keep.append(flux_col_cfg)

    keep_cols_out_2 = _resolve_keep_columns_order(
        available_cols=available_cols,
        ra_name=RA_NAME,
        dec_name=DEC_NAME,
        must_keep=must_keep,
        requested_keep_cfg=requested_keep_cfg,
    )

    ddf = ddf0[keep_cols_out_2]
    return ddf, RA_NAME, DEC_NAME, keep_cols_out_2


# =============================================================================
# Column report helpers
# =============================================================================


def compute_column_report_sample(ddf_like: Any, sample_rows: int = 200_000) -> Dict:
    """Build a small column summary from a sample.

    Uses sampling to keep the computation fast and scalable. Works with
    Dask DataFrames and LSDB catalogs.

    Args:
        ddf_like: Dask-like collection or LSDB catalog.
        sample_rows: Approximate maximum number of rows to materialize.

    Returns:
        Nested dict with basic column statistics and examples.
    """
    # Try to use the native .sample(...) API whenever it exists.
    if hasattr(ddf_like, "sample"):
        # Heuristic for sampling fraction based on number of columns.
        try:
            ncols = len(getattr(ddf_like, "columns", []))
        except Exception:
            ncols = 0

        frac = min(1.0, sample_rows / max(1, ncols * 10_000)) if ncols > 0 else 1.0

        # First try Dask/pandas-style signature (frac, replace).
        try:
            sample = ddf_like.sample(frac=frac, replace=False)
        except TypeError:
            # Some implementations may support only "n=".
            try:
                sample = ddf_like.sample(n=int(sample_rows))
            except Exception:
                sample = ddf_like
    else:
        sample = ddf_like

    # Materialize up to `sample_rows` as a pandas.DataFrame.
    try:
        pdf = sample.head(sample_rows, compute=True)
    except TypeError:
        pdf = sample.head(sample_rows)

    report: Dict[str, Dict[str, Any]] = {}
    for c in pdf.columns:
        s = pdf[c]
        col_info: Dict[str, Any] = {
            "dtype": str(s.dtype),
            "n_null": int(s.isna().sum()),
        }

        if pd.api.types.is_numeric_dtype(s):
            if len(s):
                col_info.update(
                    {
                        "min": float(np.nanmin(s.values)),
                        "max": float(np.nanmax(s.values)),
                        "mean": float(np.nanmean(s.values)),
                    }
                )
            else:
                col_info.update({"min": np.nan, "max": np.nan, "mean": np.nan})
        else:
            example = next((x for x in s.values if pd.notna(x)), "")
            col_info["example"] = str(example)

        report[c] = col_info

    return {"columns": report}


def compute_column_report_global(ddf_like: Any) -> Dict:
    """Build a column summary using global Dask-based statistics.

    Computes min, max, mean and null counts using a single Dask graph.

    Args:
        ddf_like: Dask-like collection or LSDB catalog.

    Returns:
        Nested dict with global column statistics and examples.
    """
    report: Dict[str, Dict[str, Any]] = {}

    dtypes = ddf_like.dtypes.to_dict()

    tasks: List[Any] = []
    task_keys: List[tuple[str, str]] = []

    for col, dt in dtypes.items():
        s = ddf_like[col]

        # Always compute n_null.
        tasks.append(s.isna().sum())
        task_keys.append((col, "n_null"))

        # Numeric → global min/max/mean.
        if np.issubdtype(dt, np.number):
            tasks.append(s.min())
            task_keys.append((col, "min"))

            tasks.append(s.max())
            task_keys.append((col, "max"))

            tasks.append(s.mean())
            task_keys.append((col, "mean"))
        else:
            # For non-numeric, get one non-null example if available.
            tasks.append(s.dropna().head(1))
            task_keys.append((col, "example"))

    # Execute all aggregations in a single Dask compute.
    results: Tuple[Any, ...] = dask_compute(*tasks)

    tmp: Dict[str, Dict[str, Any]] = {}
    for (col, field), value in zip(task_keys, results, strict=False):
        if col not in tmp:
            tmp[col] = {"dtype": str(dtypes[col])}

        if field == "example":
            # At runtime this is usually a pandas Series; keep typing lenient.
            try:
                iloc = getattr(value, "iloc", None)
                v = iloc[0] if iloc is not None else ""
            except Exception:
                v = ""
            tmp[col]["example"] = str(v)
        elif field in ("min", "max", "mean"):
            tmp[col][field] = float(value) if value is not None else np.nan
        elif field == "n_null":
            tmp[col]["n_null"] = int(value)

    report["columns"] = tmp
    return report
