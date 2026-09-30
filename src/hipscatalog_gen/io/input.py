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
from ..photometry import PhotometryPlan, build_photometry_plan
from ..utils import _ID_RE, _resolve_col_name, _score_deps

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


def _active_selection_dependencies(cfg: Config, available_columns: List[str]) -> List[str]:
    """Resolve physical or virtual columns needed by the active selection mode."""
    algo = cfg.algorithm
    mode = str(algo.selection_mode).lower()
    dependencies: List[str] = []

    if mode == "score_global":
        dependencies.extend(_score_deps(str(algo.score_column or ""), available_columns))
        tie_column = algo.score_tie_column or algo.tie_column
    elif mode == "score_density_hybrid":
        dependencies.extend(_score_deps(str(algo.sdh_score_column or ""), available_columns))
        tie_column = algo.sdh_tie_column or algo.tie_column
    else:
        for name in (algo.mag_column, algo.flux_column):
            if name and name in available_columns:
                dependencies.append(name)
        tie_column = algo.mag_tie_column or algo.tie_column

    if tie_column and tie_column in available_columns:
        dependencies.append(tie_column)
    return list(dict.fromkeys(dependencies))


def _plan_input_columns(
    *,
    available_columns: List[str],
    ra_name: str,
    dec_name: str,
    cfg: Config,
) -> tuple[List[str], List[str], PhotometryPlan | None]:
    """Separate physical read dependencies from final output columns."""
    photometry_plan = build_photometry_plan(cfg, available_columns)
    derived = list(photometry_plan.derived_columns) if photometry_plan else []
    logical_columns = [*available_columns, *derived]
    selection_dependencies = _active_selection_dependencies(cfg, logical_columns)

    if photometry_plan is None:
        output_available = list(available_columns)
        output_required = [ra_name, dec_name, *selection_dependencies]
    else:
        replaced = (
            set(photometry_plan.replaced_columns)
            if cfg.photometry is not None and cfg.photometry.replace_fluxes
            else set()
        )
        output_available = [name for name in available_columns if name not in replaced]
        output_available.extend(photometry_plan.output_columns)
        selection_output_dependencies = [
            name
            for name in selection_dependencies
            if name not in replaced and (name in available_columns or name in photometry_plan.output_columns)
        ]
        output_required = [
            ra_name,
            dec_name,
            *photometry_plan.output_columns,
            *selection_output_dependencies,
        ]

    output_columns = _resolve_keep_columns_order(
        available_cols=output_available,
        ra_name=ra_name,
        dec_name=dec_name,
        must_keep=output_required,
        requested_keep_cfg=cfg.columns.keep,
    )

    read_required = [name for name in output_columns if name in available_columns]
    read_required.extend(name for name in selection_dependencies if name in available_columns)
    if photometry_plan is not None:
        read_required.extend(photometry_plan.source_columns)

    if cfg.columns.keep is None:
        read_columns = list(available_columns)
    else:
        read_columns = _unique_available(read_required, available_columns)

    return read_columns, output_columns, photometry_plan


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

    if fmt == "hats":
        if len(paths) != 1:
            raise ValueError(
                "For input.format='hats', please specify exactly one HATS catalog path in input.paths."
            )

        # Preserve the established narrow LSDB open when no transformation
        # needs schema-wide collision checks. Projection remains a native
        # Catalog operation in both branches.
        if cfg.photometry is None and cfg.columns.keep is not None:
            mode = str(cfg.algorithm.selection_mode).lower()
            if mode == "score_global":
                expression = str(cfg.algorithm.score_column or "")
            elif mode == "score_density_hybrid":
                expression = str(cfg.algorithm.sdh_score_column or "")
            else:
                expression = ""
            requested = [
                cfg.columns.ra,
                cfg.columns.dec,
                *_ID_RE.findall(expression),
            ]
            if mode == "mag_global":
                requested.extend(
                    name for name in (cfg.algorithm.mag_column, cfg.algorithm.flux_column) if name
                )
            requested.extend(cfg.columns.keep)
            requested = list(dict.fromkeys(requested))
            cat0 = cast(LsdbCatalog, lsdb.open_catalog(paths[0], columns=requested))
        else:
            cat0 = cast(LsdbCatalog, lsdb.open_catalog(paths[0], columns="all"))
        available_cols = list(cat0.columns)
        RA_NAME = _resolve_col_name(
            cfg.columns.ra,
            cat0,  # type: ignore[arg-type]
            header=True,
        )
        DEC_NAME = _resolve_col_name(
            cfg.columns.dec,
            cat0,  # type: ignore[arg-type]
            header=True,
        )
        read_cols, keep_cols, _ = _plan_input_columns(
            available_columns=available_cols,
            ra_name=RA_NAME,
            dec_name=DEC_NAME,
            cfg=cfg,
        )
        projected = cast(Any, cat0)[read_cols]
        return projected, RA_NAME, DEC_NAME, keep_cols

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

    RA_NAME = _resolve_col_name(
        cfg.columns.ra,
        ddf0,
        header=(fmt == "parquet" or cfg.input.header),
    )
    DEC_NAME = _resolve_col_name(
        cfg.columns.dec,
        ddf0,
        header=(fmt == "parquet" or cfg.input.header),
    )
    available_cols = list(ddf0.columns)
    read_cols, keep_cols, _ = _plan_input_columns(
        available_columns=available_cols,
        ra_name=RA_NAME,
        dec_name=DEC_NAME,
        cfg=cfg,
    )

    # Dask's Parquet optimizer pushes this projection into the Arrow read.
    return ddf0[read_cols], RA_NAME, DEC_NAME, keep_cols


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
