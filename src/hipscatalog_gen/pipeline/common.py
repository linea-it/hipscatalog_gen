"""Shared pipeline steps for input handling, densmaps, and outputs."""

from __future__ import annotations

import glob
import json
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dask import compute as dask_compute
from dask import delayed as dask_delayed
from lsdb.catalog import Catalog as LsdbCatalog

from ..healpix.densmap import densmap_for_depth_delayed
from ..io.input import _build_input_ddf
from ..io.output import (
    finalize_write_tiles,
    write_arguments,
    write_densmap_fits,
    write_index_html,
    write_metadata_xml,
    write_moc,
)
from ..utils import _detect_hats_catalog_root, _fmt_dur, _get_dask_base, _validate_and_normalize_radec

__all__ = [
    "build_and_prepare_input",
    "compute_and_write_densmaps",
    "compute_input_total",
    "write_counts_summaries",
    "write_common_static_products",
    "log_epilogue",
    "log_prologue",
    "write_tiles_with_allsky",
    "maybe_persist_ddf",
]


def log_prologue(cfg: Any, out_dir: Path, log_fn) -> None:
    """Emit the initial pipeline log lines."""
    log_fn(
        f"START HiPS catalog pipeline: cat_name={cfg.output.cat_name} out_dir={out_dir}",
        always=True,
    )
    sel_mode = (getattr(cfg.algorithm, "selection_mode", "") or "").lower()
    base = f"Config -> lM={cfg.algorithm.level_limit} selection_mode={sel_mode}"
    log_fn(base, always=True)


def log_epilogue(
    out_dir: Path, log_lines: List[str], t0: float, log_fn, write_process_log: bool = True
) -> None:
    """Emit closing log lines and optionally persist process.log."""
    import time

    elapsed_raw = time.time() - t0
    elapsed = _fmt_dur(elapsed_raw)

    log_fn(
        f"END HiPS catalog pipeline. Elapsed {elapsed} ({elapsed_raw:.3f} s)",
        always=True,
    )

    if write_process_log:
        try:
            with (out_dir / "process.log").open("a", encoding="utf-8") as f:
                f.write("\n".join(log_lines) + "\n")
        except Exception as e:  # pragma: no cover - defensive logging
            log_fn(f"ERROR writing process.log: {type(e).__name__}: {e}", always=True)


def maybe_persist_ddf(
    ddf_like: Any,
    should_persist: bool,
    diag_ctx,
    log_fn,
    *,
    log_prefix: str,
    diag_label: str | None = None,
    reason: str | None = None,
):
    """Persist a Dask collection when requested, logging and awaiting completion."""
    if (not should_persist) or (not hasattr(ddf_like, "persist")):
        return ddf_like

    diag_name = diag_label or f"dask_{log_prefix}_persist"
    reason_text = reason or "persisting intermediate"
    log_fn(f"[{log_prefix}] Persisting DDF in memory ({reason_text}).", always=True)

    with diag_ctx(diag_name):
        persisted = ddf_like.persist()
        try:
            from dask.distributed import wait
        except Exception:  # pragma: no cover - optional dask.distributed dependency
            wait = None  # type: ignore[assignment]
        if wait is not None:
            wait(persisted)
    return persisted


def _collect_input_paths(cfg: Any, log_fn) -> List[str]:
    """Expand glob patterns from the config and log a preview."""
    paths: List[str] = []
    for p in cfg.input.paths:
        paths.extend(glob.glob(p))
    if len(paths) == 0:  # pragma: no cover - validated via calling code
        raise AssertionError("No input files matched.")

    log_fn(f"Matched {len(paths)} input files", always=True)
    log_fn(
        "Some input files: " + ", ".join(paths[:3]) + (" ..." if len(paths) > 3 else ""),
        always=True,
    )
    return paths


def _warn_if_hats_mismatch(paths: List[str], cfg: Any, log_fn) -> None:
    """Warn when paths look like HATS but the format is not 'hats'."""
    hats_root = _detect_hats_catalog_root(paths)
    if hats_root is not None and cfg.input.format.lower() != "hats":
        log_fn(
            "[input] Detected a HATS catalog layout "
            f"(found 'collection.properties' or 'hats.properties' under: {hats_root}). "
            f"You requested input.format='{cfg.input.format}'. "
            "The pipeline will proceed, but consider using input.format='hats' to "
            "enable HATS/LSDB-specific features (e.g. LSDB partitions).",
            always=True,
        )


def build_and_prepare_input(
    cfg: Any,
    diag_ctx,
    log_fn,
    persist_ddfs: bool,
) -> Tuple[Any, str, str, List[str], bool, List[str]]:
    """Load inputs, validate RA/DEC, repartition, and persist when needed.

    Args:
        cfg: Parsed configuration object.
        diag_ctx: Diagnostics context factory (label -> context manager).
        log_fn: Logging callback.
        persist_ddfs: Whether to persist the input collection in memory.

    Returns:
        Tuple containing ``(ddf, RA_NAME, DEC_NAME, keep_cols, is_hats, paths)`` where:
            - ddf: Dask-like collection ready for downstream stages.
            - RA_NAME / DEC_NAME: Resolved column names for coordinates.
            - keep_cols: Ordered list of columns to keep.
            - is_hats: True when the input is an LSDB/HATS catalog.
            - paths: List of resolved input paths.
    """
    paths = _collect_input_paths(cfg, log_fn)
    _warn_if_hats_mismatch(paths, cfg, log_fn)

    ddf, RA_NAME, DEC_NAME, keep_cols = _build_input_ddf(paths, cfg)
    is_hats = isinstance(ddf, LsdbCatalog)

    with diag_ctx("dask_radec"):
        ddf_local = _validate_and_normalize_radec(
            ddf_like=ddf,
            ra_col=RA_NAME,
            dec_col=DEC_NAME,
            log_fn=log_fn,
        )
    ddf = ddf_local

    if not is_hats:
        ddf = ddf.repartition(partition_size="256MB")

    if persist_ddfs and hasattr(ddf, "persist"):
        ddf = ddf.persist()
        from dask.distributed import wait

        wait(ddf)

    return ddf, RA_NAME, DEC_NAME, keep_cols, is_hats, paths


def compute_and_write_densmaps(
    ddf_sel: Any,
    ra_col: str,
    dec_col: str,
    level_limit: int,
    out_dir: Path,
    diag_ctx,
    log_fn=None,
) -> Dict[int, np.ndarray]:
    """Compute density maps for all depths and write them to disk.

    Args:
        ddf_sel: Dask-like collection with RA/DEC columns.
        ra_col: Name of the RA column (degrees).
        dec_col: Name of the DEC column (degrees).
        level_limit: Maximum HiPS order to compute.
        out_dir: Output directory where FITS files are written.
        diag_ctx: Diagnostics context factory (label -> context manager).
        log_fn: Optional logging callback for progress updates.

    Returns:
        Mapping of depth -> numpy array with counts per HEALPix pixel.
    """

    def _npix_for_depth(depth: int) -> int:
        return int(12 * (4 ** int(depth)))

    depths = list(range(0, level_limit + 1))
    densmaps: Dict[int, np.ndarray] = {}
    finest = int(level_limit)

    with diag_ctx("dask_densmaps"):
        # Single source pass on the finest depth.
        if log_fn is not None:
            log_fn(
                f"[densmaps] Computing densmap_o{finest}.fits (single source pass)...",
                always=True,
            )
        t_finest = time.time()
        dens_finest = dask_compute(densmap_for_depth_delayed(ddf_sel, ra_col, dec_col, depth=finest))[0]
        densmaps[finest] = dens_finest
        if log_fn is not None:
            log_fn(
                f"[densmaps] Computed densmap_o{finest}.fits in {_fmt_dur(time.time() - t_finest)}",
                always=True,
            )

        # If the finest vector shape matches HEALPix expectations, derive lower
        # depths by aggregating 4 children per parent in NESTED indexing.
        expected_finest_npix = _npix_for_depth(finest)
        can_derive = int(getattr(dens_finest, "size", -1)) == expected_finest_npix

        if can_derive:
            child_counts = dens_finest
            for depth in range(finest - 1, -1, -1):
                if log_fn is not None:
                    log_fn(
                        f"[densmaps] Deriving densmap_o{depth}.fits from densmap_o{depth + 1}.fits...",
                        always=True,
                    )
                t_der = time.time()
                parent_counts = (
                    np.asarray(child_counts, dtype=np.int64).reshape(-1, 4).sum(axis=1, dtype=np.int64)
                )
                densmaps[depth] = parent_counts
                child_counts = parent_counts
                if log_fn is not None:
                    log_fn(
                        f"[densmaps] Derived densmap_o{depth}.fits in {_fmt_dur(time.time() - t_der)}",
                        always=True,
                    )
        else:
            # Defensive fallback for non-standard testing doubles.
            if log_fn is not None:
                log_fn(
                    "[densmaps] Finest densmap size does not match expected HEALPix npix; "
                    "falling back to per-depth source computation.",
                    always=True,
                )
            for depth in depths:
                if depth == finest:
                    continue
                if log_fn is not None:
                    log_fn(
                        f"[densmaps] Computing densmap_o{depth}.fits (fallback source pass)...",
                        always=True,
                    )
                t_depth = time.time()
                densmaps[depth] = dask_compute(
                    densmap_for_depth_delayed(ddf_sel, ra_col, dec_col, depth=depth)
                )[0]
                if log_fn is not None:
                    log_fn(
                        f"[densmaps] Computed densmap_o{depth}.fits in {_fmt_dur(time.time() - t_depth)}",
                        always=True,
                    )

        # Write outputs in increasing depth order for deterministic layout/logs.
        for depth in depths:
            t_write = time.time()
            write_densmap_fits(out_dir, depth, densmaps[depth])
            if log_fn is not None:
                log_fn(
                    f"[densmaps] Wrote densmap_o{depth}.fits in {_fmt_dur(time.time() - t_write)}",
                    always=True,
                )

    return densmaps


def compute_input_total(ddf: Any, diag_ctx, log_fn, avoid_computes: bool) -> int:
    """Compute total number of input rows (post RA/DEC validation).

    Args:
        ddf: Dask-like collection with validated RA/DEC.
        diag_ctx: Diagnostics context factory (label -> context manager).
        log_fn: Logging callback.
        avoid_computes: Whether to avoid explicit ``compute()`` when possible.

    Returns:
        Total number of rows as an integer.
    """
    log_fn(
        f"[input] Counting total number of rows (avoid_computes={avoid_computes}).",
        always=True,
    )

    with diag_ctx("dask_input_total"):
        base_ddf = _get_dask_base(ddf, require_to_delayed=True)

        if hasattr(base_ddf, "to_delayed"):
            parts = base_ddf.to_delayed()
            delayed_lengths = [dask_delayed(lambda pdf: len(pdf) if pdf is not None else 0)(p) for p in parts]
            total = dask_compute(dask_delayed(sum)(delayed_lengths))[0]
        elif hasattr(base_ddf, "map_partitions"):
            meta_len = pd.Series([], dtype="int64")
            total = dask_compute(
                base_ddf.map_partitions(lambda pdf: pd.Series([len(pdf)], dtype="int64"), meta=meta_len).sum()
            )[0]
        elif hasattr(base_ddf, "__len__"):
            total = len(base_ddf)
        else:
            raise TypeError("Unable to determine input length for counting.")

    total_int = int(total)
    log_fn(f"[input] Total rows: {total_int}", always=True)
    return total_int


def _format_argument_value(value: Any) -> str:
    """Render argument values consistently (null for unset)."""
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=True, default=str)


def write_common_static_products(
    out_dir: Path,
    cfg: Any,
    densmaps: Dict[int, np.ndarray],
    keep_cols: List[str],
    ra_col: str,
    dec_col: str,
    paths: List[str],
    ddf: Any,
) -> None:
    """Write MOC, metadata.xml, and arguments echo.

    Args:
        out_dir: Destination HiPS root directory.
        cfg: Parsed configuration object.
        densmaps: Mapping depth -> densmap counts.
        keep_cols: Ordered list of columns retained in outputs.
        ra_col: Name of the RA column.
        dec_col: Name of the DEC column.
        paths: Resolved input paths.
        ddf: Dask-like collection used to infer column dtypes.
    """
    moc_order = getattr(cfg.algorithm, "moc_order", cfg.algorithm.level_limit)
    dens_lc = densmaps[moc_order]
    write_moc(out_dir, moc_order, dens_lc)

    dtypes_map = ddf.dtypes.to_dict()
    cols: List[Tuple[str, str, str | None]] = [(c, str(dtypes_map.get(c, "object")), None) for c in keep_cols]
    ra_idx = keep_cols.index(ra_col)
    dec_idx = keep_cols.index(dec_col)
    write_metadata_xml(out_dir, cols, ra_idx, dec_idx)
    write_index_html(out_dir, cfg.output)

    arg_entries: List[tuple[str, Any]] = [
        ("# input", None),
        ("input.paths", paths),
        ("input.format", getattr(cfg.input, "format", None)),
        ("input.header", getattr(cfg.input, "header", None)),
        ("input.ascii_format", getattr(cfg.input, "ascii_format", None)),
        ("# columns", None),
        ("columns.ra", ra_col),
        ("columns.dec", dec_col),
        ("columns.keep", getattr(cfg.columns, "keep", None)),
        ("# algorithm.common", None),
        ("algorithm.selection_mode", getattr(cfg.algorithm, "selection_mode", None)),
        ("algorithm.level_limit", getattr(cfg.algorithm, "level_limit", None)),
        ("algorithm.moc_order", moc_order),
        ("algorithm.selection_defaults.hist_nbins", None),
        ("algorithm.selection_defaults.adaptive_range", None),
        ("algorithm.selection_defaults.order_desc", getattr(cfg.algorithm, "order_desc", None)),
        ("algorithm.selection_defaults.tie_column", getattr(cfg.algorithm, "tie_column", None)),
        ("algorithm.selection_defaults.keep_invalid_values", None),
        ("algorithm.selection_defaults.density_bias_n1", None),
        ("algorithm.selection_defaults.density_bias_n2", None),
        ("algorithm.selection_defaults.density_bias_n3", None),
        ("# algorithm.mag_global", None),
        ("mag_global.mag_column", getattr(cfg.algorithm, "mag_column", None)),
        ("mag_global.flux_column", getattr(cfg.algorithm, "flux_column", None)),
        ("mag_global.mag_offset", getattr(cfg.algorithm, "mag_offset", None)),
        ("mag_global.mag_min", getattr(cfg.algorithm, "mag_min", None)),
        ("mag_global.mag_max", getattr(cfg.algorithm, "mag_max", None)),
        ("mag_global.adaptive_range", getattr(cfg.algorithm, "mag_adaptive_range", None)),
        ("mag_global.hist_nbins", getattr(cfg.algorithm, "mag_hist_nbins", None)),
        ("mag_global.keep_invalid_values", getattr(cfg.algorithm, "mag_keep_invalid_values", None)),
        ("mag_global.tie_column", getattr(cfg.algorithm, "mag_tie_column", None)),
        ("mag_global.order_desc", getattr(cfg.algorithm, "mg_order_desc", None)),
        ("mag_global.n_1", getattr(cfg.algorithm, "n_1", None)),
        ("mag_global.n_2", getattr(cfg.algorithm, "n_2", None)),
        ("mag_global.n_3", getattr(cfg.algorithm, "n_3", None)),
        ("mag_global.k_1", getattr(cfg.algorithm, "k_1", None)),
        ("mag_global.k_2", getattr(cfg.algorithm, "k_2", None)),
        ("mag_global.k_3", getattr(cfg.algorithm, "k_3", None)),
        ("# algorithm.score_global", None),
        ("score_global.score_column", getattr(cfg.algorithm, "score_column", None)),
        ("score_global.score_min", getattr(cfg.algorithm, "score_min", None)),
        ("score_global.score_max", getattr(cfg.algorithm, "score_max", None)),
        ("score_global.adaptive_range", getattr(cfg.algorithm, "score_adaptive_range", None)),
        ("score_global.hist_nbins", getattr(cfg.algorithm, "score_hist_nbins", None)),
        ("score_global.keep_invalid_values", getattr(cfg.algorithm, "score_keep_invalid_values", None)),
        ("score_global.tie_column", getattr(cfg.algorithm, "score_tie_column", None)),
        ("score_global.order_desc", getattr(cfg.algorithm, "sg_order_desc", None)),
        ("score_global.n_1", getattr(cfg.algorithm, "score_n_1", None)),
        ("score_global.n_2", getattr(cfg.algorithm, "score_n_2", None)),
        ("score_global.n_3", getattr(cfg.algorithm, "score_n_3", None)),
        ("score_global.k_1", getattr(cfg.algorithm, "score_k_1", None)),
        ("score_global.k_2", getattr(cfg.algorithm, "score_k_2", None)),
        ("score_global.k_3", getattr(cfg.algorithm, "score_k_3", None)),
        ("# algorithm.score_density_hybrid", None),
        ("score_density_hybrid.score_column", getattr(cfg.algorithm, "sdh_score_column", None)),
        ("score_density_hybrid.score_min", getattr(cfg.algorithm, "sdh_score_min", None)),
        ("score_density_hybrid.score_max", getattr(cfg.algorithm, "sdh_score_max", None)),
        ("score_density_hybrid.adaptive_range", getattr(cfg.algorithm, "sdh_score_adaptive_range", None)),
        ("score_density_hybrid.hist_nbins", getattr(cfg.algorithm, "sdh_score_hist_nbins", None)),
        ("score_density_hybrid.keep_invalid_values", getattr(cfg.algorithm, "sdh_keep_invalid_values", None)),
        ("score_density_hybrid.tie_column", getattr(cfg.algorithm, "sdh_tie_column", None)),
        ("score_density_hybrid.order_desc", getattr(cfg.algorithm, "sdh_order_desc", None)),
        ("score_density_hybrid.n_1", getattr(cfg.algorithm, "sdh_n_1", None)),
        ("score_density_hybrid.n_2", getattr(cfg.algorithm, "sdh_n_2", None)),
        ("score_density_hybrid.n_3", getattr(cfg.algorithm, "sdh_n_3", None)),
        ("score_density_hybrid.k_1", getattr(cfg.algorithm, "sdh_k_1", None)),
        ("score_density_hybrid.k_2", getattr(cfg.algorithm, "sdh_k_2", None)),
        ("score_density_hybrid.k_3", getattr(cfg.algorithm, "sdh_k_3", None)),
        ("score_density_hybrid.density_bias_n1", getattr(cfg.algorithm, "sdh_density_bias_n1", None)),
        ("score_density_hybrid.density_bias_n2", getattr(cfg.algorithm, "sdh_density_bias_n2", None)),
        ("score_density_hybrid.density_bias_n3", getattr(cfg.algorithm, "sdh_density_bias_n3", None)),
        ("# cluster", None),
        ("cluster.mode", getattr(cfg.cluster, "mode", None)),
        ("cluster.n_workers", getattr(cfg.cluster, "n_workers", None)),
        ("cluster.threads_per_worker", getattr(cfg.cluster, "threads_per_worker", None)),
        ("cluster.memory_per_worker", getattr(cfg.cluster, "memory_per_worker", None)),
        ("cluster.low_memory_mode", getattr(cfg.cluster, "low_memory_mode", None)),
        ("cluster.persist_ddfs", getattr(cfg.cluster, "persist_ddfs", None)),
        (
            "cluster.avoid_computes_wherever_possible",
            getattr(cfg.cluster, "avoid_computes_wherever_possible", None),
        ),
        ("cluster.diagnostics_mode", getattr(cfg.cluster, "diagnostics_mode", None)),
        ("cluster.slurm", getattr(cfg.cluster, "slurm", None)),
        ("# output", None),
        ("output.out_dir", str(out_dir)),
        ("output.cat_name", getattr(cfg.output, "cat_name", None)),
        ("output.target", getattr(cfg.output, "target", None)),
        ("output.creator_did", getattr(cfg.output, "creator_did", None)),
        ("output.obs_title", getattr(cfg.output, "obs_title", None)),
        ("output.overwrite", getattr(cfg.output, "overwrite", None)),
    ]

    arg_lines: List[str] = []
    for key, val in arg_entries:
        if key.startswith("# "):
            arg_lines.append(key)
            continue
        arg_lines.append(f"{key}: {_format_argument_value(val)}")

    arg_text = "\n".join(arg_lines)
    write_arguments(out_dir, arg_text + "\n")


def write_allsky(
    out_dir: Path,
    depth: int,
    header_line: str,
    counts: np.ndarray,
    allsky_df: pd.DataFrame,
    nwritten_tot: int,
) -> None:
    """Write the Allsky.tsv file for a depth, if provided."""
    norder_dir = out_dir / f"Norder{depth}"
    norder_dir.mkdir(parents=True, exist_ok=True)
    tmp_allsky = norder_dir / ".Allsky.tsv.tmp"
    final_allsky = norder_dir / "Allsky.tsv"

    nsrc_tot = int(counts.sum())
    nremaining_tot = max(0, nsrc_tot - nwritten_tot)
    completeness_header_allsky = f"# Completeness = {nremaining_tot} / {nsrc_tot}\n"

    header_cols = header_line.strip("\n").split("\t")
    allsky_cols = [c for c in header_cols if c in allsky_df.columns]
    df_as = allsky_df[allsky_cols].copy()

    with tmp_allsky.open("w", encoding="utf-8", newline="") as f:
        f.write(completeness_header_allsky)
        f.write(header_line)

    obj_cols = df_as.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) > 0:
        df_as[obj_cols] = df_as[obj_cols].replace(
            {r"[\t\r\n]": " "},
            regex=True,
        )

    df_as.to_csv(
        tmp_allsky,
        sep="\t",
        index=False,
        header=False,
        mode="a",
        encoding="utf-8",
        lineterminator="\n",
    )
    import os

    os.replace(tmp_allsky, final_allsky)


def write_tiles_with_allsky(
    out_dir: Path,
    depth: int,
    header_line: str,
    ra_col: str,
    dec_col: str,
    counts: np.ndarray,
    selected: pd.DataFrame,
    order_desc: bool,
    allsky_needed: bool,
    log_fn,
) -> tuple[dict[int, int] | None, pd.DataFrame | None]:
    """Finalize tiles and write optional Allsky.tsv."""
    written_per_ipix, allsky_df = finalize_write_tiles(
        out_dir=out_dir,
        depth=depth,
        header_line=header_line,
        ra_col=ra_col,
        dec_col=dec_col,
        counts=counts,
        selected=selected,
        order_desc=order_desc,
        allsky_collect=allsky_needed,
    )

    if allsky_needed and allsky_df is not None and len(allsky_df) > 0:
        nwritten_tot = int(sum(written_per_ipix.values())) if written_per_ipix else 0
        write_allsky(out_dir, depth, header_line, counts, allsky_df, nwritten_tot)

    return written_per_ipix, allsky_df


def write_counts_summaries(
    out_dir: Path,
    level_limit: int,
    input_total: int,
    log_fn,
    precomputed_depth_totals: Dict[str, int],
) -> tuple[int, dict]:
    """Build output counts from selection-stage precomputed depth totals."""
    if not isinstance(precomputed_depth_totals, dict):
        raise RuntimeError(
            "Missing precomputed selection write stats; final TSV recount fallback is disabled."
        )

    depth_totals: Dict[str, int] = {}
    for depth_key, depth_total in precomputed_depth_totals.items():
        d: int | None = None
        with suppress(TypeError, ValueError):
            d = int(depth_key)
        if d is None:
            raise RuntimeError(f"Invalid depth key in precomputed stats: {depth_key!r}")

        v: int | None = None
        with suppress(TypeError, ValueError):
            v = int(depth_total)
        if v is None:
            raise RuntimeError(f"Invalid depth total in precomputed stats: depth={d!r} value={depth_total!r}")
        if v < 0:
            raise RuntimeError(f"Negative depth total in precomputed stats: depth={d!r} value={v!r}")
        if d < 0 or d > int(level_limit):
            raise RuntimeError(
                f"Depth out of bounds in precomputed stats: depth={d!r}, level_limit={level_limit!r}"
            )

        depth_totals[str(d)] = int(v)

    total_all_depths = int(sum(depth_totals.values()))
    output_counts = {
        "total": int(total_all_depths),
        "depth_totals": depth_totals,
        "depths": {},
    }
    input_counts = {"total": int(input_total)}

    log_fn("[counts] Using precomputed output counts from selection write stage.", always=True)
    log_fn(f"[output] Total rows written: {total_all_depths}", always=True)

    return int(total_all_depths), {"output": output_counts, "input": input_counts}
