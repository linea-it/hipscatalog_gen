"""Shared utilities reused across the hipscatalog-gen pipeline."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, cast

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import compute as dask_compute

__all__ = [
    "_mkdirs",
    "_write_text",
    "_detect_hats_catalog_root",
    "_now_str",
    "_ts",
    "_fmt_dur",
    "_stats_counts",
    "_log_depth_stats",
    "_ID_RE",
    "_HEALPIX_INDEX_RE",
    "_score_deps",
    "_resolve_col_name",
    "_get_dask_base",
    "_get_meta_df",
    "_validate_and_normalize_radec",
]


# =============================================================================
# Filesystem and simple logging helpers
# =============================================================================


def _mkdirs(p: Path) -> None:
    """Create directory and parents if they do not exist."""
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text to file."""
    path.write_text(content, encoding="utf-8")


def _detect_hats_catalog_root(paths: List[str]) -> Path | None:
    """Best-effort detection of a HATS catalog root directory.

    Strategy:
        For each path, walk up its parents looking for
        'collection.properties' or 'hats.properties'. The first match is
        returned as the catalog root.

    Args:
        paths: List of input paths (files or directories).

    Returns:
        Catalog root path if found, otherwise None.
    """
    marker_names = ("collection.properties", "hats.properties")

    for p in paths:
        cur = Path(p).resolve()

        # If this is a file, scan its parents; if it's a directory, include it.
        candidates = list(cur.parents) if cur.is_file() else [cur] + list(cur.parents)

        for root in candidates:
            for name in marker_names:
                candidate = root / name
                if candidate.exists():
                    return root

    return None


def _now_str() -> str:
    """Return current UTC time in HiPS-friendly ISO 8601 format.

    Format is YYYY-mm-ddTHH:MMZ, always in UTC.
    """
    return time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())


def _ts() -> str:
    """Return local timestamp string for logging."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _fmt_dur(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS.mmm."""
    s = int(seconds)
    ms = int(round((seconds - s) * 1000))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# =============================================================================
# Small numeric and logging utilities
# =============================================================================


def _stats_counts(counts: np.ndarray) -> tuple[int, int]:
    """Return (total_rows, non_empty_pixels) for a densmap vector."""
    total = int(counts.sum())
    nonempty = int((counts > 0).sum())
    return total, nonempty


def _log_depth_stats(
    _log_fn: Callable[..., None],
    depth: int,
    phase: str,
    counts: np.ndarray | None = None,
    candidates_len: int | None = None,
    selected_len: int | None = None,
    written: Dict[int, int] | None = None,
    remainder_len: int | None = None,
) -> None:
    """Log a compact one-line summary for a depth and pipeline phase.

    Args:
        _log_fn: Logger callable receiving (message, always_flag).
        depth: HiPS depth (order).
        phase: Phase label ("start", "candidates", "selected", etc.).
        counts: Optional densmap counts array.
        candidates_len: Optional number of candidate rows.
        selected_len: Optional number of selected rows.
        written: Optional mapping tile_index -> rows_written.
        remainder_len: Optional number of remainder rows.
    """
    parts: List[str] = []
    if counts is not None:
        tot, nz = _stats_counts(counts)
        parts.append(f"input_rows={tot}")
        parts.append(f"non_empty_pixels={nz}")
    if candidates_len is not None:
        parts.append(f"candidates={candidates_len}")
    if selected_len is not None:
        parts.append(f"selected={selected_len}")
    if written is not None:
        rows_written = int(sum(written.values())) if written else 0
        tiles_written = int(len(written)) if written else 0
        parts.append(f"tiles_written={tiles_written}")
        parts.append(f"rows_written={rows_written}")
    if remainder_len is not None:
        parts.append(f"remainder={remainder_len}")

    _log_fn(f"[DEPTH {depth}] {phase}: " + "; ".join(parts), True, depth=depth)


# =============================================================================
# Dask/LSDB base resolution
# =============================================================================


def _get_dask_base(
    ddf_like: Any,
    require_groupby: bool = False,
    require_map_partitions: bool = False,
    require_to_delayed: bool = False,
) -> Any:
    """Prefer public Dask-like interfaces; fall back to LSDB ._ddf only when needed.

    Args:
        ddf_like: Dask-like object or LSDB catalog.
        require_groupby: Require a ``groupby`` attribute.
        require_map_partitions: Require a ``map_partitions`` attribute.
        require_to_delayed: Require a ``to_delayed`` attribute.

    Returns:
        The same object if it already satisfies the required interface, otherwise
        its underlying Dask DataFrame when dealing with LSDB catalogs.

    Raises:
        TypeError: If the object (or its LSDB base) does not expose the required methods.
        TypeError: If an LSDB catalog lacks the ``_ddf`` attribute.
    """

    def _has_required(obj: Any) -> bool:
        """Check that obj exposes all required dask-like methods."""
        if require_groupby and not hasattr(obj, "groupby"):
            return False
        if require_map_partitions and not hasattr(obj, "map_partitions"):
            return False
        if require_to_delayed and not hasattr(obj, "to_delayed"):
            return False
        # If nothing explicitly required, accept any of the common dask-like methods.
        if not (require_groupby or require_map_partitions or require_to_delayed):
            return hasattr(obj, "groupby") or hasattr(obj, "map_partitions") or hasattr(obj, "to_delayed")
        return True

    if TYPE_CHECKING:
        from lsdb.catalog import Catalog as LsdbCatalogType

    LsdbCatalog: type[Any] | None

    try:
        from lsdb.catalog import Catalog as _LsdbCatalog

        LsdbCatalog = cast("type[LsdbCatalogType]", _LsdbCatalog)
    except Exception:  # pragma: no cover - lsdb optional
        LsdbCatalog = None

    if _has_required(ddf_like):
        return ddf_like

    # For LSDB catalogs without public Dask-like methods, fall back to the underlying Dask DataFrame.
    if LsdbCatalog is not None and isinstance(ddf_like, LsdbCatalog):
        base = getattr(ddf_like, "_ddf", None)
        if base is None:
            raise TypeError("LSDB catalog missing _ddf attribute; cannot extract Dask base.")
        if _has_required(base):
            return base
        raise TypeError("LSDB catalog _ddf does not expose required dask-like methods.")

    raise TypeError("Object is not Dask-like (missing groupby/map_partitions/to_delayed).")


# =============================================================================
# Score dependency extraction and column resolution
# =============================================================================

_ID_RE = re.compile(r"[A-Za-z_]\w*")
_HEALPIX_INDEX_RE = re.compile(r"_healpix_(\d+)$")


def _score_deps(score_expr: str, available: List[str]) -> List[str]:
    """Return column names referenced in a score expression.

    Only identifiers that exist in `available` are kept.

    Args:
        score_expr: Score expression string.
        available: List of available column names.

    Returns:
        List of column names used in the expression.
    """
    if not score_expr:
        return []
    tokens = set(_ID_RE.findall(str(score_expr)))
    return [c for c in available if c in tokens]


def _resolve_col_name(spec: str, ddf: dd.DataFrame, header: bool) -> str:
    """Resolve a column spec that can be a name or 1-based index.

    Args:
        spec: Column name or 1-based index as string.
        ddf: Dask DataFrame used to resolve columns.
        header: Whether input has a header row.

    Returns:
        Resolved column name.

    Raises:
        KeyError: If the column is not found.
        IndexError: If a numeric index is out of range.
    """
    if header:
        if spec not in ddf.columns:
            raise KeyError(f"Column '{spec}' not found in input.")
        return spec

    # header == False → allow numeric indices (1-based)
    try:
        idx_1based = int(spec)
        idx = idx_1based - 1
        cols = list(ddf.columns)
        if not (0 <= idx < len(cols)):
            raise IndexError(f"Column index {idx_1based} out of range (1..{len(cols)})")
        return cols[idx]
    except ValueError as err:
        if spec not in ddf.columns:
            raise KeyError(f"Column '{spec}' not found (header=False).") from err
        return spec


# =============================================================================
# Metadata extraction for Dask/LSDB collections
# =============================================================================


def _get_meta_df(ddf_like: Any) -> pd.DataFrame:
    """Return an empty DataFrame with same columns/dtypes as a collection.

    Supports plain Dask DataFrames and LSDB catalogs.

    Args:
        ddf_like: Dask-like collection or LSDB catalog.

    Returns:
        Empty pandas.DataFrame with the same schema.
    """
    base = _get_dask_base(ddf_like)

    def _as_pandas_df(obj: Any) -> pd.DataFrame | None:
        """Convert meta-like objects (including NestedFrame) to pandas."""
        if isinstance(obj, pd.DataFrame):
            return obj
        if hasattr(obj, "to_pandas"):
            try:
                candidate = obj.to_pandas()
                if isinstance(candidate, pd.DataFrame):
                    return candidate
            except Exception:
                return None
        return None

    if hasattr(base, "_meta"):
        meta = getattr(base, "_meta", None)
        meta_pdf = _as_pandas_df(meta)
        if meta_pdf is not None:
            return meta_pdf

    # Fallback: try .head(0)
    try:
        head0 = base.head(0)
        meta_pdf = _as_pandas_df(head0)
        if meta_pdf is not None:
            return meta_pdf
    except Exception:
        return pd.DataFrame()

    # Last resort: empty DataFrame
    return pd.DataFrame()


# =============================================================================
# RA/DEC validation and normalization
# =============================================================================


def _validate_and_normalize_radec(
    ddf_like: Any,
    ra_col: str,
    dec_col: str,
    log_fn: Callable[[str, bool], None],
) -> Any:
    """Validate RA/DEC ranges and normalize RA into [0, 360] if needed.

    Supports both plain Dask DataFrames and LSDB catalogs.

    Rules:
        * DEC must be within [-90, +90] degrees (up to a small epsilon).
        * RA must be either:
            - [0, 360] degrees      → kept as is, or
            - [-180, 180] degrees   → shifted to [0, 360] via (RA + 360) % 360.
        * Any other range raises ValueError.

    Args:
        ddf_like: Dask-like collection or LSDB catalog.
        ra_col: RA column name.
        dec_col: DEC column name.
        log_fn: Logger callable receiving (message, always_flag).

    Returns:
        The same collection, possibly with RA normalized.

    Raises:
        ValueError: If RA/DEC ranges are unsupported or non-finite.
    """
    # Convert to numeric with coercion so non-numeric values become NaN.
    ra_num = dd.to_numeric(ddf_like[ra_col], errors="coerce")
    dec_num = dd.to_numeric(ddf_like[dec_col], errors="coerce")

    ra_min, ra_max, dec_min, dec_max = dask_compute(
        ra_num.min(),
        ra_num.max(),
        dec_num.min(),
        dec_num.max(),
    )

    msg = (
        f"[RA/DEC check] RA in [{ra_min:.6f}, {ra_max:.6f}], "
        f"DEC in [{dec_min:.6f}, {dec_max:.6f}] (assuming degrees)"
    )
    log_fn(msg, True)

    # Require finite values
    if not (np.isfinite(ra_min) and np.isfinite(ra_max) and np.isfinite(dec_min) and np.isfinite(dec_max)):
        raise ValueError(
            "RA/DEC contain non-finite values or could not be converted to numeric. "
            "This pipeline currently supports only RA/DEC in degrees."
        )

    eps = 1e-6

    # DEC in [-90, +90]
    if dec_min < -90.0 - eps or dec_max > 90.0 + eps:
        raise ValueError(
            f"Unsupported DEC range: [{dec_min}, {dec_max}]. "
            "This pipeline currently supports only DEC in [-90, +90] degrees."
        )

    # RA logic:
    #   * [0, 360]   → keep as is
    #   * [-180,180] → convert to [0, 360]
    if (0.0 - eps) <= ra_min <= (360.0 + eps) and (0.0 - eps) <= ra_max <= (360.0 + eps):
        log_fn("[RA/DEC check] Detected RA in [0, 360] degrees; keeping values as they are.", True)
        return ddf_like

    if (-180.0 - eps) <= ra_min <= (180.0 + eps) and (-180.0 - eps) <= ra_max <= (180.0 + eps):
        log_fn("[RA/DEC check] Detected RA in [-180, 180] degrees; converting to [0, 360].", True)

        def _shift_ra_partition(pdf: pd.DataFrame) -> pd.DataFrame:
            """Shift RA from [-180, 180] to [0, 360] for one partition."""
            if pdf.empty:
                return pdf
            vals = pd.to_numeric(pdf[ra_col], errors="coerce")
            vals = (vals + 360.0) % 360.0
            pdf = pdf.copy()
            pdf[ra_col] = vals
            return pdf

        meta = _get_meta_df(ddf_like)
        ddf_like = ddf_like.map_partitions(_shift_ra_partition, meta=meta)
        return ddf_like

    # Any other RA range is considered unsupported (likely wrong units)
    raise ValueError(
        f"Unsupported RA range: [{ra_min}, {ra_max}]. "
        "This pipeline currently supports only RA in degrees, with RA either in "
        "[0, 360] or [-180, 180] and DEC in [-90, +90]. "
        "Please convert your coordinates to degrees before running the pipeline."
    )
