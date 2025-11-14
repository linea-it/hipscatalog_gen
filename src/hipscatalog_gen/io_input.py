from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import dask.dataframe as dd
import lsdb
import numpy as np
import pandas as pd
from dask import compute as dask_compute
from lsdb.catalog import Catalog as LsdbCatalog

from .config import Config
from .utils import _ID_RE, _resolve_col_name, _score_deps

__all__ = [
    "_build_input_ddf",
    "compute_column_report_sample",
    "compute_column_report_global",
]


# =============================================================================
# Build Dask / LSDB input collection
# =============================================================================


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
    assert len(paths) > 0, "No input files matched."

    fmt = cfg.input.format.lower()
    # Single declaration for the whole function (avoid no-redef)
    mag_col_cfg: Optional[str] = None
    if getattr(cfg.algorithm, "selection_mode", "coverage").lower() == "mag_global":
        mag_col_cfg = cfg.algorithm.mag_column

    # If columns.keep is None, preserve all input columns.
    keep_all_columns = cfg.columns.keep is None

    # ------------------------------------------------------------------
    # HATS / LSDB input: keep LSDB structure
    # ------------------------------------------------------------------
    if fmt == "hats":
        if len(paths) != 1:
            raise ValueError(
                "For input.format='hats', please specify exactly one HATS catalog " "path in input.paths."
            )

        hats_path = paths[0]

        # Columns explicitly requested by the user in the YAML.
        requested_keep_cfg = cfg.columns.keep
        requested_keep = requested_keep_cfg or []

        # Extract potential score dependencies from the score expression.
        score_expr = cfg.columns.score or ""
        score_tokens = set(_ID_RE.findall(str(score_expr)))

        # Always request RA, DEC and score dependencies; mag_col if applicable.
        must_keep = [cfg.columns.ra, cfg.columns.dec, *score_tokens]
        if mag_col_cfg:
            must_keep.append(mag_col_cfg)

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

        # Build keep_cols in a deterministic order:
        #   RA, DEC, score dependencies, then requested extras (if any).
        score_dependencies = [c for c in score_tokens if c in available_cols]

        must_keep_resolved = [RA_NAME, DEC_NAME, *score_dependencies]
        if mag_col_cfg and mag_col_cfg in available_cols:
            must_keep_resolved.append(mag_col_cfg)

        # When keep_all_columns is True, we still put RA/DEC (and deps) first,
        # but preserve all remaining columns from the catalog.
        if keep_all_columns:
            candidate = [c for c in available_cols if c not in must_keep_resolved]
        else:
            candidate = requested_keep  # only include if explicitly provided

        seen = set()
        keep_cols_out: List[str] = []
        for c in [*must_keep_resolved, *candidate]:
            if c in available_cols and c not in seen:
                keep_cols_out.append(c)
                seen.add(c)

        # Sub-select via LSDB API; returns a new Catalog.
        ddf = cast(Any, cat0)[keep_cols_out]
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
    score_dependencies = _score_deps(cfg.columns.score, available_cols)

    requested_keep_cfg = cfg.columns.keep
    requested_keep = requested_keep_cfg or []

    must_keep = [RA_NAME, DEC_NAME, *score_dependencies]
    if mag_col_cfg and mag_col_cfg in available_cols:
        must_keep.append(mag_col_cfg)

    # If columns.keep is None, preserve all columns:
    # RA/DEC and score deps first, then all remaining columns.
    if keep_all_columns:
        candidate = [c for c in available_cols if c not in must_keep]
    else:
        candidate = requested_keep

    seen = set()
    keep_cols_out_2: List[str] = []
    for c in [*must_keep, *candidate]:
        if c in available_cols and c not in seen:
            keep_cols_out_2.append(c)
            seen.add(c)

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

        if ncols > 0:
            frac = min(1.0, sample_rows / max(1, ncols * 10_000))
        else:
            frac = 1.0

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
                v = value.iloc[0]  # type: ignore[attr-defined]
            except Exception:
                v = ""
            tmp[col]["example"] = str(v)
        elif field in ("min", "max", "mean"):
            tmp[col][field] = float(value) if value is not None else np.nan
        elif field == "n_null":
            tmp[col]["n_null"] = int(value)

    report["columns"] = tmp
    return report
