"""Shared pipeline steps for input handling, densmaps, and outputs."""

from __future__ import annotations

import glob
import textwrap
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
) -> Dict[int, np.ndarray]:
    """Compute density maps for all depths and write them to disk.

    Args:
        ddf_sel: Dask-like collection with RA/DEC columns.
        ra_col: Name of the RA column (degrees).
        dec_col: Name of the DEC column (degrees).
        level_limit: Maximum HiPS order to compute.
        out_dir: Output directory where FITS files are written.
        diag_ctx: Diagnostics context factory (label -> context manager).

    Returns:
        Mapping of depth -> numpy array with counts per HEALPix pixel.
    """
    depths = list(range(0, level_limit + 1))
    densmaps: Dict[int, np.ndarray] = {}

    delayed_maps = {d: densmap_for_depth_delayed(ddf_sel, ra_col, dec_col, depth=d) for d in depths}

    with diag_ctx("dask_densmaps"):
        computed = dask_compute(*delayed_maps.values())

    for d, dens in zip(delayed_maps.keys(), computed, strict=False):
        densmaps[d] = dens
        write_densmap_fits(out_dir, d, dens)

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

    arg_text = textwrap.dedent(
        f"""
        # input
        input.paths: {paths}
        input.format: {cfg.input.format}
        input.header: {getattr(cfg.input, "header", None)}
        input.ascii_format: {getattr(cfg.input, "ascii_format", None)}
        # columns
        columns.ra: {ra_col}
        columns.dec: {dec_col}
        columns.keep: {cfg.columns.keep}
        # algorithm.common
        algorithm.selection_mode: {cfg.algorithm.selection_mode}
        algorithm.level_limit: {cfg.algorithm.level_limit}
        algorithm.moc_order: {moc_order}
        algorithm.mag_global.order_desc: {getattr(cfg.algorithm, "mg_order_desc", False)}
        algorithm.score_global.order_desc: {getattr(cfg.algorithm, "sg_order_desc", False)}
        algorithm.score_density_hybrid.order_desc: {getattr(cfg.algorithm, "sdh_order_desc", False)}
        # algorithm.mag_global
        mag_global.mag_column: {cfg.algorithm.mag_column}
        mag_global.flux_column: {cfg.algorithm.flux_column}
        mag_global.mag_offset: {cfg.algorithm.mag_offset}
        mag_global.mag_min: {cfg.algorithm.mag_min}
        mag_global.mag_max: {cfg.algorithm.mag_max}
        mag_global.adaptive_range: {cfg.algorithm.mag_adaptive_range}
        mag_global.hist_nbins: {cfg.algorithm.mag_hist_nbins}
        mag_global.n_1: {cfg.algorithm.n_1}
        mag_global.n_2: {cfg.algorithm.n_2}
        mag_global.n_3: {cfg.algorithm.n_3}
        # algorithm.score_global
        score_global.score_column: {cfg.algorithm.score_column}
        score_global.score_min: {cfg.algorithm.score_min}
        score_global.score_max: {cfg.algorithm.score_max}
        score_global.adaptive_range: {cfg.algorithm.score_adaptive_range}
        score_global.hist_nbins: {cfg.algorithm.score_hist_nbins}
        score_global.n_1: {cfg.algorithm.score_n_1}
        score_global.n_2: {cfg.algorithm.score_n_2}
        score_global.n_3: {cfg.algorithm.score_n_3}
        # algorithm.score_density_hybrid
        score_density_hybrid.score_column: {getattr(cfg.algorithm, "sdh_score_column", None)}
        score_density_hybrid.score_min: {getattr(cfg.algorithm, "sdh_score_min", None)}
        score_density_hybrid.score_max: {getattr(cfg.algorithm, "sdh_score_max", None)}
        score_density_hybrid.adaptive_range: {getattr(cfg.algorithm, "sdh_score_adaptive_range", None)}
        score_density_hybrid.hist_nbins: {getattr(cfg.algorithm, "sdh_score_hist_nbins", None)}
        score_density_hybrid.n_1: {getattr(cfg.algorithm, "sdh_n_1", None)}
        score_density_hybrid.n_2: {getattr(cfg.algorithm, "sdh_n_2", None)}
        score_density_hybrid.n_3: {getattr(cfg.algorithm, "sdh_n_3", None)}
        score_density_hybrid.density_bias_n1: {getattr(cfg.algorithm, "sdh_density_bias_n1", None)}
        score_density_hybrid.density_bias_n2: {getattr(cfg.algorithm, "sdh_density_bias_n2", None)}
        score_density_hybrid.density_bias_n3: {getattr(cfg.algorithm, "sdh_density_bias_n3", None)}
        # cluster
        cluster.mode: {cfg.cluster.mode}
        cluster.n_workers: {cfg.cluster.n_workers}
        cluster.threads_per_worker: {cfg.cluster.threads_per_worker}
        cluster.memory_per_worker: {cfg.cluster.memory_per_worker}
        cluster.persist_ddfs: {cfg.cluster.persist_ddfs}
        cluster.avoid_computes_wherever_possible: {cfg.cluster.avoid_computes_wherever_possible}
        cluster.diagnostics_mode: {cfg.cluster.diagnostics_mode}
        cluster.slurm: {cfg.cluster.slurm}
        # output
        output.out_dir: {out_dir}
        output.cat_name: {cfg.output.cat_name}
        output.target: {cfg.output.target}
        output.creator_did: {cfg.output.creator_did}
        output.obs_title: {cfg.output.obs_title}
        """
    ).strip("\n")
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


def write_counts_summaries(out_dir: Path, level_limit: int, input_total: int, log_fn) -> tuple[int, dict]:
    """Compute counts for later cross-checks and return (total written, counts dict)."""

    def _count_rows(tile_path: Path) -> int:
        """Count rows for one tile file (ignoring header lines)."""
        with tile_path.open("r", encoding="utf-8") as f:
            # Skip completeness + header lines.
            next(f, None)
            next(f, None)
            return sum(1 for _ in f)

    depth_totals: Dict[str, int] = {}
    depth_counts: Dict[str, Dict[str, int]] = {}
    total_all_depths = 0

    for depth in range(0, level_limit + 1):
        norder_dir = out_dir / f"Norder{depth}"
        if not norder_dir.exists():
            continue

        counts_depth: Dict[str, int] = {}
        for tile_path in norder_dir.rglob("Npix*.tsv"):
            name = tile_path.name
            if not name.startswith("Npix") or not name.endswith(".tsv"):  # pragma: no cover - rglob filters
                continue
            try:
                ipix = int(name[len("Npix") : -len(".tsv")])
            except ValueError:
                continue
            counts_depth[str(ipix)] = _count_rows(tile_path)

        if counts_depth:
            depth_total = int(sum(counts_depth.values()))
            depth_totals[str(depth)] = depth_total
            depth_counts[str(depth)] = counts_depth
            total_all_depths += depth_total

    output_counts = {
        "total": int(total_all_depths),
        "depth_totals": depth_totals,
        "depths": depth_counts,
    }
    input_counts = {"total": int(input_total)}

    log_fn("[counts] Computed output/input counts.", always=True)

    return int(total_all_depths), {"output": output_counts, "input": input_counts}
