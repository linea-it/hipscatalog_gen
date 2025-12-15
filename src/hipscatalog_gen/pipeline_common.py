from __future__ import annotations

import glob
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dask import compute as dask_compute
from lsdb.catalog import Catalog as LsdbCatalog

from .healpix import densmap_for_depth_delayed
from .io_input import _build_input_ddf
from .io_output import (
    finalize_write_tiles,
    write_arguments,
    write_densmap_fits,
    write_metadata_xml,
    write_moc,
    write_properties,
)
from .utils import _detect_hats_catalog_root, _fmt_dur, _validate_and_normalize_radec

__all__ = [
    "build_and_prepare_input",
    "compute_and_write_densmaps",
    "write_common_static_products",
    "log_epilogue",
    "log_prologue",
    "write_tiles_with_allsky",
]


def log_prologue(cfg: Any, out_dir: Path, log_fn) -> None:
    """Emit the initial pipeline log lines."""
    log_fn(
        f"START HiPS catalog pipeline: " f"cat_name={cfg.output.cat_name} out_dir={out_dir}",
        always=True,
    )
    log_fn(
        f"Config -> lM={cfg.algorithm.level_limit} "
        f"lC={cfg.algorithm.level_coverage} "
        f"Oc={cfg.algorithm.coverage_order} "
        f"order_desc={cfg.algorithm.order_desc}",
        always=True,
    )


def log_epilogue(out_dir: Path, log_lines: List[str], t0: float, log_fn) -> None:
    """Emit closing log lines and persist process.log."""
    import time

    elapsed_raw = time.time() - t0
    elapsed = _fmt_dur(elapsed_raw)

    log_fn(
        f"END HiPS catalog pipeline. Elapsed {elapsed} " f"({elapsed_raw:.3f} s)",
        always=True,
    )

    try:
        with (out_dir / "process.log").open("a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
    except Exception as e:
        log_fn(f"ERROR writing process.log: {type(e).__name__}: {e}", always=True)


def _collect_input_paths(cfg: Any, log_fn) -> List[str]:
    """Expand glob patterns from the config and log a preview."""
    paths: List[str] = []
    for p in cfg.input.paths:
        paths.extend(glob.glob(p))
    if len(paths) == 0:
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
            "enable HATS/LSDB-specific features (e.g. LSDB partitions, "
            "algorithm.use_hats_as_coverage).",
            always=True,
        )


def build_and_prepare_input(
    cfg: Any,
    diag_ctx,
    log_fn,
    persist_ddfs: bool,
) -> Tuple[Any, str, str, List[str], bool, List[str]]:
    """Load inputs, validate RA/DEC, repartition, and persist when needed."""
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
    """Compute density maps for all depths and write them to disk."""
    depths = list(range(0, level_limit + 1))
    densmaps: Dict[int, np.ndarray] = {}

    delayed_maps = {d: densmap_for_depth_delayed(ddf_sel, ra_col, dec_col, depth=d) for d in depths}

    with diag_ctx("dask_densmaps"):
        computed = dask_compute(*delayed_maps.values())

    for d, dens in zip(delayed_maps.keys(), computed, strict=False):
        densmaps[d] = dens
        write_densmap_fits(out_dir, d, dens)

    return densmaps


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
    """Write MOC, metadata.xml, properties, and arguments echo."""
    dens_lc = densmaps[cfg.algorithm.level_coverage]
    write_moc(out_dir, cfg.algorithm.level_coverage, dens_lc)

    dtypes_map = ddf.dtypes.to_dict()
    cols: List[Tuple[str, str, Optional[str]]] = [
        (c, str(dtypes_map.get(c, "object")), None) for c in keep_cols
    ]
    ra_idx = keep_cols.index(ra_col)
    dec_idx = keep_cols.index(dec_col)
    write_metadata_xml(out_dir, cols, ra_idx, dec_idx)

    n_src_total = int(densmaps[0].sum())
    write_properties(
        out_dir,
        cfg.output,
        cfg.algorithm.level_limit,
        n_src_total,
        tile_format="tsv",
    )

    arg_text = textwrap.dedent(
        f"""
        # Input/output
        Input files: {paths}
        Input type: {cfg.input.format}
        Output dir: {out_dir}
        # Input data parameters
        Catalogue name: {cfg.output.cat_name}
        RA column name: {ra_col}
        DE column name: {dec_col}
        # Selection parameters
        level_limit(lM): {cfg.algorithm.level_limit}
        level_coverage(lC): {cfg.algorithm.level_coverage}
        coverage_order(Oc): {cfg.algorithm.coverage_order}
        order_desc: {cfg.algorithm.order_desc}
        selection_mode: {cfg.algorithm.selection_mode}
        mag_column: {cfg.algorithm.mag_column}
        mag_min: {cfg.algorithm.mag_min}
        mag_max: {cfg.algorithm.mag_max}
        mag_completeness: {cfg.algorithm.mag_completeness}
        mag_hist_nbins: {cfg.algorithm.mag_hist_nbins}
        n_1: {cfg.algorithm.n_1}
        n_2: {cfg.algorithm.n_2}
        n_3: {cfg.algorithm.n_3}
        k_per_cov_per_level: {cfg.algorithm.k_per_cov_per_level}
        targets_total_per_level: {cfg.algorithm.targets_total_per_level}
        tie_buffer: {cfg.algorithm.tie_buffer}
        density_mode: {cfg.algorithm.density_mode}
        k_per_cov_initial: {cfg.algorithm.k_per_cov_initial}
        targets_total_initial: {cfg.algorithm.targets_total_initial}
        density_exp_base: {cfg.algorithm.density_exp_base}
        density_bias_mode: {cfg.algorithm.density_bias_mode}
        density_bias_exponent: {cfg.algorithm.density_bias_exponent}
        fractional_mode: {cfg.algorithm.fractional_mode}
        fractional_mode_logic: {cfg.algorithm.fractional_mode_logic}
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
