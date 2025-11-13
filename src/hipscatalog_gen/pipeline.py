#!/usr/bin/env python3
"""Central orchestration for the HiPS catalog pipeline.

This module wires configuration, cluster setup, input reading, densmap
computation, and selection logic implemented in the submodules.

Typical usage (library):
    from hipscatalog_gen import load_config, run_pipeline
    cfg = load_config("config.yaml")
    run_pipeline(cfg)

Command-line interface:
    python -m hipscatalog_gen.cli --config config.yaml
"""

from __future__ import annotations

# =============================================================================
# Standard library
# =============================================================================
import glob
import math
import os
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Third-party libraries
# =============================================================================
import dask.dataframe as dd
import healpy as hp
import numpy as np
import pandas as pd
from dask import compute as dask_compute
from lsdb.catalog import Catalog as LsdbCatalog

# =============================================================================
# Internal modules
# =============================================================================
from .cluster import setup_cluster, shutdown_cluster
from .config import Config
from .healpix import densmap_for_depth_delayed
from .io_input import _build_input_ddf
from .io_output import (
    build_header_line_from_keep,
    finalize_write_tiles,
    write_arguments,
    write_densmap_fits,
    write_metadata_xml,
    write_moc,
    write_properties,
)
from .mag_global import _quantile_from_histogram, compute_mag_histogram_ddf
from .selection import (
    _candidates_by_coverage_partition,
    _reduce_coverage_exact,
    _reduce_coverage_exact_dask,
    apply_fractional_k_per_cov,
    build_cov_thresholds,
    filter_remainder_by_coverage_partition,
)
from .utils import (
    _HEALPIX_INDEX_RE,
    _detect_hats_catalog_root,
    _fmt_dur,
    _get_meta_df,
    _log_depth_stats,
    _mkdirs,
    _ts,
    _validate_and_normalize_radec,
)

__all__ = ["run_pipeline"]


# =============================================================================
# Pipeline (per_cov-only)
# =============================================================================


def run_pipeline(cfg: Config) -> None:
    """Run the full HiPS catalog generation pipeline.

    Args:
        cfg: Parsed configuration object with input, algorithm, cluster,
            and output options.
    """
    out_dir = Path(cfg.output.out_dir)
    _mkdirs(out_dir)

    # Directory for Dask performance reports.
    report_dir = out_dir / "dask_reports"
    _mkdirs(report_dir)

    t0 = time.time()
    log_lines: List[str] = []

    def _log(msg: str, always: bool = False) -> None:
        """Log message with timestamp to stdout and the in-memory buffer.

        Args:
            msg: Message to log.
            always: Kept for backward compatibility; all messages are logged.
        """
        line = f"{_ts()} | {msg}"
        print(line)
        log_lines.append(line)

    _log(
        f"START HiPS catalog pipeline: " f"cat_name={cfg.output.cat_name} out_dir={out_dir}",
        always=True,
    )
    _log(
        f"Config -> lM={cfg.algorithm.level_limit} "
        f"lC={cfg.algorithm.level_coverage} "
        f"Oc={cfg.algorithm.coverage_order} "
        f"order_desc={cfg.algorithm.order_desc}",
        always=True,
    )

    # CDS-like validation for level_limit
    if not (4 <= int(cfg.algorithm.level_limit) <= 11):
        raise ValueError("level_limit (lM) must be within [4, 11] to mirror the CDS tool.")

    if cfg.algorithm.level_coverage > cfg.algorithm.level_limit:
        cfg.algorithm.level_coverage = cfg.algorithm.level_limit
        _log("WARNING: level_coverage was > level_limit; set lC = lM", always=True)

    # ------------------------------------------------------------------
    # Configuration sanity / warning layer (HATS-related behaviour)
    # ------------------------------------------------------------------
    fmt_lower = str(cfg.input.format).lower()

    # Warn when use_hats_as_coverage is requested but format != 'hats'.
    if fmt_lower != "hats" and getattr(cfg.algorithm, "use_hats_as_coverage", False):
        _log(
            "[config] algorithm.use_hats_as_coverage=True was requested, but "
            "input.format is not 'hats'. This option is only meaningful for "
            "HATS/LSDB catalogs and will be ignored for this run.",
            always=True,
        )

    # Warn and effectively ignore density_bias_mode when using format='hats'
    # + use_hats_as_coverage=True.
    if (
        fmt_lower == "hats"
        and getattr(cfg.algorithm, "use_hats_as_coverage", False)
        and str(getattr(cfg.algorithm, "density_bias_mode", "none")).lower() != "none"
    ):
        _log(
            "[config] density_bias_mode is not supported when using format='hats' "
            "with algorithm.use_hats_as_coverage=True. "
            "The density bias will be ignored for this run (forcing density_bias_mode='none').",
            always=True,
        )
        cfg.algorithm.density_bias_mode = "none"

    # ------------------------------------------------------------------
    # Cluster setup (local or SLURM) + diagnostics
    # ------------------------------------------------------------------
    runtime, diag_ctx = setup_cluster(cfg.cluster, report_dir, _log)
    client = runtime.client
    persist_ddfs = runtime.persist_ddfs
    avoid_computes = runtime.avoid_computes
    diagnostics_mode = runtime.diagnostics_mode

    def _run_core_pipeline() -> None:
        """Main pipeline body.

        This function contains the logic that builds the HiPS catalog:
        input reading, RA/DEC validation, densmaps, MOC, metadata, and
        per-depth selection.
        """
        # ------------------------------------------------------------------
        # Input
        # ------------------------------------------------------------------
        paths: List[str] = []
        for p in cfg.input.paths:
            paths.extend(glob.glob(p))
        assert len(paths) > 0, "No input files matched."

        _log(f"Matched {len(paths)} input files", always=True)
        _log(
            "Some input files: " + ", ".join(paths[:3]) + (" ..." if len(paths) > 3 else ""),
            always=True,
        )

        # Warn if the input paths look like a HATS catalog but format != 'hats'.
        hats_root = _detect_hats_catalog_root(paths)
        if hats_root is not None and cfg.input.format.lower() != "hats":
            _log(
                "[input] Detected a HATS catalog layout "
                f"(found 'collection.properties' or 'hats.properties' under: {hats_root}). "
                f"You requested input.format='{cfg.input.format}'. "
                "The pipeline will proceed, but consider using input.format='hats' to "
                "enable HATS/LSDB-specific features (e.g. LSDB partitions, "
                "algorithm.use_hats_as_coverage).",
                always=True,
            )

        # Build input collection (Dask DataFrame or LSDB Catalog).
        ddf, RA_NAME, DEC_NAME, keep_cols = _build_input_ddf(paths, cfg)

        is_hats = isinstance(ddf, LsdbCatalog)

        # ------------------------------------------------------------------
        # RA/DEC sanity check + normalization for all inputs
        # ------------------------------------------------------------------
        # Supports:
        #   * plain Dask DataFrames (parquet/csv/tsv)
        #   * HATS / LSDB catalogs (lsdb.catalog.Catalog).
        with diag_ctx("dask_radec"):
            ddf_local = _validate_and_normalize_radec(
                ddf_like=ddf,
                ra_col=RA_NAME,
                dec_col=DEC_NAME,
                log_fn=_log,
            )
        ddf = ddf_local

        # For HATS catalogs we keep the native spatial partitioning and do not repartition.
        if not is_hats:
            ddf = ddf.repartition(partition_size="256MB")

        # In high-throughput mode we keep the repartitioned dataframe in memory.
        if persist_ddfs and hasattr(ddf, "persist"):
            ddf = ddf.persist()
            from dask.distributed import wait

            wait(ddf)

        algo = cfg.algorithm
        selection_mode = (getattr(algo, "selection_mode", "coverage") or "coverage").lower()

        # ==================================================================
        # Selection pre-processing
        # ==================================================================
        if selection_mode == "mag_global":
            # --------------------------------------------------------------
            # Global magnitude-selection path:
            #   * No coverage (__icov__) needed for the selection itself.
            #   * Keep numeric magnitude column in __mag__.
            #   * If mag_min is not provided, use the global minimum magnitude,
            #     but not smaller than -2 (to avoid extreme outliers).
            #   * If mag_max is not provided, estimate it from the histogram
            #     peak (bin center of the highest-count bin, rounded to 0.01)
            #     using a histogram restricted to mag < 40.
            #   * Restrict working catalog to [mag_min, mag_max].
            # --------------------------------------------------------------
            mag_col_cfg = getattr(algo, "mag_column", None)
            if not mag_col_cfg:
                raise ValueError(
                    "algorithm.selection_mode='mag_global' requires "
                    "algorithm.mag_column to be set to the name of a magnitude column."
                )

            if mag_col_cfg not in ddf.columns:
                raise KeyError(f"Configured mag_column '{mag_col_cfg}' not found in input columns.")

            # Numeric magnitude column kept in internal __mag__ field.
            base_meta_mag = _get_meta_df(ddf)
            meta_with_mag = base_meta_mag.copy()
            meta_with_mag["__mag__"] = pd.Series([], dtype="float64")

            def _add_mag_column(pdf: pd.DataFrame, mag_col_name: str) -> pd.DataFrame:
                """Add numeric __mag__ column derived from mag_col_name."""
                if pdf.empty:
                    pdf["__mag__"] = pd.Series([], dtype="float64")
                    return pdf
                pdf = pdf.copy()
                pdf["__mag__"] = pd.to_numeric(pdf[mag_col_name], errors="coerce")
                return pdf

            ddf = ddf.map_partitions(
                _add_mag_column,
                mag_col_cfg,
                meta=meta_with_mag,
            )

            mag_col_internal = "__mag__"
            mag_min_cfg = getattr(algo, "mag_min", None)
            mag_max_cfg = getattr(algo, "mag_max", None)

            # Determine global min/max in __mag__ for automatic bounds.
            with diag_ctx("dask_mag_minmax"):
                mag_min_global_raw, mag_max_global_raw = dask_compute(
                    ddf[mag_col_internal].min(),
                    ddf[mag_col_internal].max(),
                )

            if mag_min_global_raw is None or mag_max_global_raw is None:
                raise ValueError(
                    "mag_global selection: unable to determine global magnitude "
                    "range (min/max returned None). Check the magnitude column."
                )

            mag_min_global_raw = float(mag_min_global_raw)
            mag_max_global_raw = float(mag_max_global_raw)

            if not np.isfinite(mag_min_global_raw) or not np.isfinite(mag_max_global_raw):
                raise ValueError(
                    "mag_global selection: global magnitude min/max are not finite. "
                    "Check the magnitude column values."
                )

            if mag_min_global_raw >= mag_max_global_raw:
                raise ValueError(
                    f"mag_global selection: invalid global magnitude range "
                    f"[{mag_min_global_raw}, {mag_max_global_raw}]."
                )

            # ------------------------------------------------------------------
            # Effective mag_min (automatic mode: clip to >= -2)
            # ------------------------------------------------------------------
            if mag_min_cfg is None:
                raw = mag_min_global_raw
                mag_min = max(raw, -2.0)
                if mag_min != raw:
                    _log(
                        "[mag_global] mag_min not provided; using global minimum "
                        f"{raw:.4f} clipped to {mag_min:.4f} (>= -2).",
                        always=True,
                    )
                else:
                    _log(
                        "[mag_global] mag_min not provided; using global minimum "
                        f"magnitude {mag_min:.4f}.",
                        always=True,
                    )
            else:
                mag_min = float(mag_min_cfg)

            # Upper bound used for automatic histogram range (to avoid high outliers).
            mag_upper_for_hist = min(mag_max_global_raw, 40.0)

            if mag_upper_for_hist <= mag_min:
                raise ValueError(
                    "mag_global selection: after applying automatic bounds, "
                    f"mag_min={mag_min:.4f} is not smaller than the upper histogram "
                    f"limit={mag_upper_for_hist:.4f}. Check magnitude values or "
                    "configured mag_min."
                )

            # ------------------------------------------------------------------
            # Effective mag_max (automatic mode: histogram peak, mag <= 40)
            # ------------------------------------------------------------------
            if mag_max_cfg is None:
                # Use a preliminary histogram in [mag_min, mag_upper_for_hist]
                # to estimate the peak and define mag_max as the corresponding
                # bin center (rounded to 0.01).
                with diag_ctx("dask_mag_hist_auto_max"):
                    hist_auto, edges_auto, n_tot_auto = compute_mag_histogram_ddf(
                        ddf_like=ddf,
                        mag_col=mag_col_internal,
                        mag_min=mag_min,
                        mag_max=mag_upper_for_hist,
                        nbins=algo.mag_hist_nbins,
                    )

                if n_tot_auto == 0:
                    raise ValueError(
                        "mag_global selection: no objects found when estimating "
                        "automatic mag_max. Check the magnitude column and range."
                    )

                peak_idx = int(np.argmax(hist_auto))
                bin_left = float(edges_auto[peak_idx])
                bin_right = float(edges_auto[peak_idx + 1])
                peak_center = 0.5 * (bin_left + bin_right)

                # Enforce the <= 40 constraint implicitly via mag_upper_for_hist.
                peak_center = min(peak_center, mag_upper_for_hist)
                mag_max = float(np.round(peak_center, 2))

                _log(
                    "[mag_global] mag_max not provided; using histogram peak at "
                    f"{mag_max:.2f} (bin center from [{bin_left:.4f}, "
                    f"{bin_right:.4f}], with mag < 40).",
                    always=True,
                )
            else:
                mag_max = float(mag_max_cfg)

            if mag_min >= mag_max:
                raise ValueError(
                    f"algorithm.mag_min ({mag_min}) must be strictly smaller than "
                    f"algorithm.mag_max ({mag_max}) for mag_global selection."
                )

            # Store effective values back in the config object so that they are
            # echoed consistently in arguments and logs.
            algo.mag_min = mag_min
            algo.mag_max = mag_max

            # Restrict to [mag_min, mag_max] once; all depths use the same pool.
            meta_sel = meta_with_mag.copy()

            def _filter_mag_window(
                pdf: pd.DataFrame,
                mag_min_val: float,
                mag_max_val: float,
            ) -> pd.DataFrame:
                """Keep only rows with __mag__ in [mag_min_val, mag_max_val]."""
                if pdf.empty:
                    return pdf
                m = pd.to_numeric(pdf[mag_col_internal], errors="coerce")
                mask = (m >= mag_min_val) & (m <= mag_max_val)
                return pdf.loc[mask]

            ddf_sel = ddf.map_partitions(
                _filter_mag_window,
                mag_min,
                mag_max,
                meta=meta_sel,
            )
        else:
            # ------------------------------------------------------------------
            # Coverage: either fixed HEALPix coverage or HATS partitions
            # ------------------------------------------------------------------
            use_hats_cov = is_hats and bool(getattr(cfg.algorithm, "use_hats_as_coverage", False))

            base_meta = _get_meta_df(ddf)
            meta_with_icov = base_meta.copy()
            meta_with_icov["__icov__"] = pd.Series([], dtype="int64")

            if use_hats_cov:
                # ==============================================================
                # HATS-specific coverage: one coverage cell per HATS partition
                # ==============================================================
                try:
                    hp_pixels = ddf.get_healpix_pixels()
                    n_hp_pixels = len(hp_pixels)
                except Exception:
                    hp_pixels = None
                    n_hp_pixels = None

                n_parts = ddf.npartitions

                msg = (
                    f"[coverage] Using HATS partitions as coverage cells (__icov__), "
                    f"n_partitions={n_parts}"
                )
                if n_hp_pixels is not None:
                    msg += f", get_healpix_pixels() returned {n_hp_pixels} pixels"
                _log(msg, always=True)

                part_ids = dd.from_pandas(
                    pd.Series(range(n_parts), dtype="int64"),
                    npartitions=n_parts,
                )

                def _assign_icov(pdf: pd.DataFrame, part_series: pd.Series) -> pd.DataFrame:
                    """Assign the partition id as __icov__ for all rows in the partition."""
                    if pdf.empty:
                        pdf["__icov__"] = pd.Series([], dtype="int64")
                        return pdf
                    cov_id = int(part_series.iloc[0])
                    pdf = pdf.copy()
                    pdf["__icov__"] = cov_id
                    return pdf

                ddf = ddf.map_partitions(
                    _assign_icov,
                    part_ids,
                    meta=meta_with_icov,
                )
            else:
                # ==============================================================
                # Default coverage: HEALPix cells at coverage_order (Oc)
                # ==============================================================
                Oc = int(cfg.algorithm.coverage_order)
                NSIDE_C = 1 << Oc

                def _add_icov(pdf: pd.DataFrame, ra_col: str, dec_col: str) -> pd.DataFrame:
                    """Add coverage cell index (__icov__) at coverage_order."""
                    if len(pdf) == 0:
                        pdf["__icov__"] = pd.Series([], dtype="int64")
                        return pdf

                    if is_hats:
                        idx_name = getattr(pdf.index, "name", None)
                        m = _HEALPIX_INDEX_RE.match(str(idx_name)) if idx_name else None

                        if m is not None:
                            base_order = int(m.group(1))
                            if Oc <= base_order:
                                ipix_base = pdf.index.to_numpy()
                                shift = 2 * (base_order - Oc)
                                icov = (ipix_base >> shift).astype(np.int64)
                                pdf = pdf.copy()
                                pdf["__icov__"] = icov
                                return pdf

                    theta = np.deg2rad(90.0 - pd.to_numeric(pdf[dec_col], errors="coerce").to_numpy())
                    phi = np.deg2rad((pd.to_numeric(pdf[ra_col], errors="coerce").to_numpy()) % 360.0)
                    icov = hp.ang2pix(NSIDE_C, theta, phi, nest=True).astype(np.int64)
                    pdf = pdf.copy()
                    pdf["__icov__"] = icov
                    return pdf

                ddf = ddf.map_partitions(
                    _add_icov,
                    RA_NAME,
                    DEC_NAME,
                    meta=meta_with_icov,
                )

            # For coverage-based selection, we work on the full catalog with __icov__.
            ddf_sel = ddf

        # ------------------------------------------------------------------
        # Densmaps 0..lM + FITS (computed in parallel)
        # ------------------------------------------------------------------
        depths = list(range(0, cfg.algorithm.level_limit + 1))
        densmaps: Dict[int, np.ndarray] = {}

        # Build delayed densmaps for all depths.
        delayed_maps = {d: densmap_for_depth_delayed(ddf_sel, RA_NAME, DEC_NAME, depth=d) for d in depths}

        with diag_ctx("dask_densmaps"):
            computed = dask_compute(*delayed_maps.values())

        # Fill dict and write FITS files (already NumPy/FITS).
        for d, dens in zip(delayed_maps.keys(), computed, strict=False):
            densmaps[d] = dens
            write_densmap_fits(out_dir, d, dens)

        # Coverage densmap used for the MOC.
        dens_lc = densmaps[cfg.algorithm.level_coverage]

        # MOC
        write_moc(out_dir, cfg.algorithm.level_coverage, dens_lc)

        # metadata.xml / Metadata.xml
        dtypes_map = ddf.dtypes.to_dict()
        cols: List[Tuple[str, str, Optional[str]]] = [
            (c, str(dtypes_map.get(c, "object")), None) for c in keep_cols
        ]
        ra_idx = keep_cols.index(RA_NAME)
        dec_idx = keep_cols.index(DEC_NAME)
        write_metadata_xml(out_dir, cols, ra_idx, dec_idx)

        # properties
        n_src_total = int(densmaps[0].sum())
        write_properties(
            out_dir,
            cfg.output,
            cfg.algorithm.level_limit,
            n_src_total,
            tile_format="tsv",
        )

        # arguments (config echo)
        arg_text = textwrap.dedent(
            f"""
            # Input/output
            Input files: {paths}
            Input type: {cfg.input.format}
            Output dir: {out_dir}
            # Input data parameters
            Catalogue name: {cfg.output.cat_name}
            RA column name: {RA_NAME}
            DE column name: {DEC_NAME}
            # Selection parameters
            level_limit(lM): {cfg.algorithm.level_limit}
            level_coverage(lC): {cfg.algorithm.level_coverage}
            coverage_order(Oc): {cfg.algorithm.coverage_order}
            order_desc: {cfg.algorithm.order_desc}
            selection_mode: {cfg.algorithm.selection_mode}
            mag_column: {cfg.algorithm.mag_column}
            mag_min: {cfg.algorithm.mag_min}
            mag_max: {cfg.algorithm.mag_max}
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

        # =====================================================================
        # Selection stage
        # =====================================================================
        remainder_ddf = ddf_sel
        algo = cfg.algorithm
        selection_mode = (getattr(algo, "selection_mode", "coverage") or "coverage").lower()

        if selection_mode == "mag_global":
            # --------------------------------------------------------------
            # Global magnitude-complete selection:
            #   1) Compute per-depth targets T_d ∝ #active tiles at that depth.
            #   2) Convert T_d into magnitude quantiles using a global histogram.
            #   3) For each depth, select all objects in the corresponding
            #      magnitude slice and write tiles.
            # --------------------------------------------------------------
            mag_col_internal = "__mag__"
            if algo.mag_min is None or algo.mag_max is None:
                raise RuntimeError(
                    "mag_global: internal error — mag_min/mag_max should have been set earlier."
                )
            mag_min = float(algo.mag_min)
            mag_max = float(algo.mag_max)

            depths_sel = list(range(1, cfg.algorithm.level_limit + 1))

            # 1) Histogram + CDF of magnitudes in [mag_min, mag_max].
            with diag_ctx("dask_mag_hist"):
                hist, mag_edges_hist, N_tot_mag = compute_mag_histogram_ddf(
                    remainder_ddf,
                    mag_col=mag_col_internal,
                    mag_min=mag_min,
                    mag_max=mag_max,
                    nbins=algo.mag_hist_nbins,
                )

            if N_tot_mag == 0:
                _log(
                    "[selection] mag_global: no objects found in the magnitude range "
                    f"[{mag_min}, {mag_max}] → nothing to select.",
                    always=True,
                )
                return

            cdf_hist = hist.cumsum().astype("float64")
            if cdf_hist[-1] > 0:
                cdf_hist /= float(cdf_hist[-1])
            else:
                cdf_hist[:] = 0.0

            # 2) Per-depth targets T_d from densmaps (weights ∝ #active tiles).
            #    Optionally honoring n_1, n_2, n_3.
            weights_list: List[float] = []
            for d in depths_sel:
                counts_d = densmaps[d]
                tiles_active = int((counts_d > 0).sum())
                weights_list.append(max(1, tiles_active))

            weights = np.asarray(weights_list, dtype="float64")  # ndarray float64

            # Start with zero targets for all depths.
            T = np.zeros_like(weights, dtype="float64")

            # Optional approximate fixed totals for depths 1, 2 and 3.
            fixed_targets: Dict[int, float] = {}
            for d, n_val in (
                (1, getattr(algo, "n_1", None)),
                (2, getattr(algo, "n_2", None)),
                (3, getattr(algo, "n_3", None)),
            ):
                if (d in depths_sel) and (n_val is not None):
                    if int(n_val) < 0:
                        raise ValueError(f"algorithm.n_{d} must be non-negative if provided (got {n_val}).")
                    fixed_targets[d] = float(int(n_val))

            sum_fixed = float(sum(fixed_targets.values()))

            # If fixed targets exceed total objects, rescale uniformly.
            if sum_fixed > float(N_tot_mag) and sum_fixed > 0.0:
                scale = float(N_tot_mag) / sum_fixed
                _log(
                    "[mag_global] Sum of fixed targets n_1/n_2/n_3 "
                    f"({int(sum_fixed)}) exceeds the total number of objects "
                    f"in the magnitude range ({int(N_tot_mag)}). "
                    f"Rescaling n_1/n_2/n_3 by a factor {scale:.3f}.",
                    always=True,
                )
                for d in list(fixed_targets.keys()):
                    fixed_targets[d] *= scale
                sum_fixed = float(N_tot_mag)

            # Assign fixed contributions to the corresponding depths.
            for d, val in fixed_targets.items():
                idx = depths_sel.index(d)
                T[idx] = val

            # Remaining objects are distributed across the other depths.
            N_rem = max(0.0, float(N_tot_mag) - sum_fixed)
            if N_rem > 0.0:
                free_mask = np.ones_like(weights, dtype=bool)
                for d in fixed_targets:
                    idx = depths_sel.index(d)
                    free_mask[idx] = False

                W_free = float(weights[free_mask].sum())
                if W_free <= 0.0:
                    # Fallback: spread uniformly across all free depths.
                    n_free = int(free_mask.sum())
                    if n_free > 0:
                        T[free_mask] += N_rem / float(n_free)
                else:
                    T[free_mask] += weights[free_mask] / W_free * N_rem

            # Convert per-depth totals into cumulative targets and quantiles.
            T_cum = np.cumsum(T)
            if T_cum[-1] > 0.0:
                Q = T_cum / float(N_tot_mag)
            else:
                Q = np.zeros_like(T_cum, dtype="float64")

            # Convert cumulative targets into magnitude edges per depth.
            level_edges: np.ndarray = np.empty(len(depths_sel) + 1, dtype="float64")
            level_edges[0] = mag_min
            for i, q in enumerate(Q, start=1):
                level_edges[i] = _quantile_from_histogram(cdf_hist, mag_edges_hist, q)

            # Ensure non-decreasing edges and clamp to [mag_min, mag_max].
            level_edges = np.maximum.accumulate(level_edges)
            level_edges[0] = mag_min
            level_edges[-1] = mag_max

            _log(
                "[selection] mag_global mode: per-depth magnitude slices:\n"
                + "\n".join(
                    f"  depth {d}: [{level_edges[i]:.6f}, {level_edges[i+1]:.6f}"
                    f"{')' if d != depths_sel[-1] else ']'}"
                    for i, d in enumerate(depths_sel)
                ),
                always=True,
            )

            header_line = build_header_line_from_keep(keep_cols)

            # 3) Loop over depths, select magnitude slice, write tiles.
            for i, depth in enumerate(depths_sel):
                depth_t0 = time.time()
                m_lo = level_edges[i]
                m_hi = level_edges[i + 1]

                with diag_ctx(f"dask_depth_mag_{depth:02d}"):
                    if depth != depths_sel[-1]:
                        mag_mask = (remainder_ddf[mag_col_internal] >= m_lo) & (
                            remainder_ddf[mag_col_internal] < m_hi
                        )
                    else:
                        # Last depth is inclusive on the upper bound.
                        mag_mask = (remainder_ddf[mag_col_internal] >= m_lo) & (
                            remainder_ddf[mag_col_internal] <= m_hi
                        )

                    depth_ddf = remainder_ddf[mag_mask]

                    selected_pdf = depth_ddf.compute()
                    _log_depth_stats(
                        _log,
                        depth,
                        "selected",
                        counts=densmaps[depth],
                        selected_len=len(selected_pdf),
                    )

                    if len(selected_pdf) == 0:
                        _log(
                            f"[DEPTH {depth}] mag_global: no rows in "
                            f"magnitude slice [{m_lo:.6f}, {m_hi:.6f}] → skipping.",
                            always=True,
                        )
                        _log(
                            f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
                            always=True,
                        )
                        continue

                    # Map selected rows to HEALPix pixels at this order.
                    ra_vals = pd.to_numeric(selected_pdf[RA_NAME], errors="coerce").to_numpy()
                    dec_vals = pd.to_numeric(selected_pdf[DEC_NAME], errors="coerce").to_numpy()

                    theta = np.deg2rad(90.0 - dec_vals)
                    phi = np.deg2rad(ra_vals % 360.0)

                    NSIDE_L = 1 << depth
                    ipixL = hp.ang2pix(NSIDE_L, theta, phi, nest=True).astype(np.int64)
                    selected_pdf["__ipix__"] = ipixL

                    counts = densmaps[depth]
                    allsky_needed = depth in (1, 2)

                    written_per_ipix, allsky_df = finalize_write_tiles(
                        out_dir=out_dir,
                        depth=depth,
                        header_line=header_line,
                        ra_col=RA_NAME,
                        dec_col=DEC_NAME,
                        counts=counts,
                        selected=selected_pdf,
                        order_desc=cfg.algorithm.order_desc,
                        allsky_collect=allsky_needed,
                    )
                    _log_depth_stats(
                        _log,
                        depth,
                        "written",
                        counts=densmaps[depth],
                        written=written_per_ipix,
                    )

                    # Allsky handling (same logic as coverage mode).
                    if allsky_needed and allsky_df is not None and len(allsky_df) > 0:
                        norder_dir = out_dir / f"Norder{depth}"
                        norder_dir.mkdir(parents=True, exist_ok=True)
                        tmp_allsky = norder_dir / ".Allsky.tsv.tmp"
                        final_allsky = norder_dir / "Allsky.tsv"

                        nsrc_tot = int(counts.sum())
                        nwritten_tot = int(sum(written_per_ipix.values())) if written_per_ipix else 0
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
                        os.replace(tmp_allsky, final_allsky)

                _log(
                    f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
                    always=True,
                )

        else:
            # =================================================================
            # Coverage-based selection: uniform expected k per coverage (__icov__)
            # =================================================================
            remainder_ddf = ddf_sel

            # If base expected density and per-level overrides are non-positive,
            # there is nothing to select.
            if float(cfg.algorithm.k_per_cov_initial) <= 0.0 and not (
                cfg.algorithm.k_per_cov_per_level or {}
            ):
                _log(
                    "[selection] k_per_cov_initial <= 0 and no per-level overrides → "
                    "nothing to select; finishing early.",
                    always=True,
                )
                return

            Tmap = cfg.algorithm.targets_total_per_level or {}

            # Pre-compute coverage density map (per __icov__) from densmaps.
            cov_order = int(cfg.algorithm.coverage_order)
            dens_cov: Dict[int, int] = {}
            if cov_order in densmaps:
                vec = densmaps[cov_order]
                dens_cov = {int(i): int(v) for i, v in enumerate(vec) if v > 0}
            else:
                dens_cov = {}

            from dask.distributed import wait

            for depth in range(1, cfg.algorithm.level_limit + 1):
                depth_t0 = time.time()

                with diag_ctx(f"dask_depth_{depth:02d}"):
                    # (a) compute k_desired for this depth.
                    algo = cfg.algorithm
                    mode = (getattr(algo, "density_mode", "constant") or "constant").lower()
                    use_total_profile = getattr(algo, "targets_total_initial", None) is not None

                    delta = max(0, depth - 1)
                    n_cov: Optional[int] = None
                    if use_total_profile or depth in Tmap:
                        try:
                            n_cov = int(remainder_ddf["__icov__"].dropna().nunique().compute())
                        except Exception:
                            n_cov = 0

                    if use_total_profile:
                        tti = getattr(algo, "targets_total_initial", None)
                        if use_total_profile:
                            if tti is None:
                                raise ValueError(
                                    "targets_total_initial must be set when using total-profile density mode."
                                )
                            T0 = float(tti)
                        if mode == "constant":
                            T_desired = T0
                        elif mode == "linear":
                            T_desired = T0 * float(1 + delta)
                        elif mode == "exp":
                            base = float(getattr(algo, "density_exp_base", 2.0))
                            if base <= 1.0:
                                base = 2.0
                            T_desired = T0 * (base ** float(delta))
                        elif mode == "log":
                            T_desired = T0 * math.log2(delta + 2.0)
                        else:
                            raise ValueError(f"Unknown density_mode: {algo.density_mode!r}")

                        if n_cov is None or n_cov <= 0:
                            k_desired = 0.0
                        else:
                            k_desired = T_desired / float(n_cov)

                        _log(
                            f"[DEPTH {depth}] start (density_mode={mode}, "
                            f"targets_total_initial={T0:.4f}, "
                            f"n_cov={n_cov}, k_desired_from_total={k_desired:.4f})",
                            always=True,
                        )
                    else:
                        k0 = float(algo.k_per_cov_initial)

                        if mode == "constant":
                            k_desired = k0
                        elif mode == "linear":
                            k_desired = k0 * float(1 + delta)
                        elif mode == "exp":
                            base = float(getattr(algo, "density_exp_base", 2.0))
                            if base <= 1.0:
                                base = 2.0
                            k_desired = k0 * (base ** float(delta))
                        elif mode == "log":
                            k_desired = k0 * math.log2(delta + 2.0)
                        else:
                            raise ValueError(f"Unknown density_mode: {algo.density_mode!r}")

                        _log(
                            f"[DEPTH {depth}] start (density_mode={mode}, " f"k_desired={k_desired:.4f})",
                            always=True,
                        )

                    if algo.k_per_cov_per_level and depth in algo.k_per_cov_per_level:
                        k_desired = float(algo.k_per_cov_per_level[depth])

                    k_desired = max(0.0, float(k_desired))

                    _log_depth_stats(_log, depth, "start", counts=densmaps[depth])

                    if depth in Tmap:
                        T_L = float(Tmap[depth])
                        if n_cov is None:
                            try:
                                n_cov = int(remainder_ddf["__icov__"].dropna().nunique().compute())
                            except Exception:
                                n_cov = 0

                        if n_cov > 0 and T_L > 0.0:
                            cap_per_cov = T_L / float(n_cov)
                            if cap_per_cov < k_desired:
                                _log(
                                    f"[DEPTH {depth}] applying total cap: T_L={int(T_L)}, "
                                    f"N_cov={n_cov} → cap_per_cov={cap_per_cov:.4f} "
                                    f"(before k_desired={k_desired:.4f})",
                                    always=True,
                                )
                                k_desired = cap_per_cov
                        else:
                            _log(
                                f"[DEPTH {depth}] cannot apply total cap: " f"T_L={T_L}, N_cov={n_cov}",
                                always=True,
                            )

                    if k_desired <= 0.0:
                        _log(
                            f"[DEPTH {depth}] k_desired <= 0 → skipping this depth",
                            always=True,
                        )
                        continue

                    # (b) density bias (coverage-density-based k).
                    bias_mode = (getattr(algo, "density_bias_mode", "none") or "none").lower()
                    use_bias = bias_mode in ("proportional", "inverse") and bool(dens_cov)

                    k_per_cov_for_selection: Any
                    k_per_cov_desired_map: Optional[Dict[int, float]] = None

                    if use_bias:
                        alpha = float(getattr(algo, "density_bias_exponent", 1.0))
                        alpha = abs(alpha)
                        eps = 1e-6

                        w_raw: Dict[int, float] = {}
                        for icov, cnt in dens_cov.items():
                            val = float(cnt) + eps
                            if bias_mode == "proportional":
                                w = val**alpha
                            else:
                                w = val ** (-alpha)
                            w_raw[int(icov)] = w

                        if not w_raw:
                            use_bias = False
                            k_per_cov_for_selection = int(max(1, math.ceil(k_desired)))
                        else:
                            w_vals = list(w_raw.values())
                            mean_w = float(np.mean(w_vals))
                            if mean_w <= 0.0 or not np.isfinite(mean_w):
                                use_bias = False
                                k_per_cov_for_selection = int(max(1, math.ceil(k_desired)))
                            else:
                                k_per_cov_desired: Dict[int, float] = {}
                                for icov, w in w_raw.items():
                                    w_norm = w / mean_w
                                    k_c = max(0.0, k_desired * w_norm)
                                    k_per_cov_desired[int(icov)] = k_c

                                k_per_cov_int: Dict[int, int] = {}
                                for icov, k_c in k_per_cov_desired.items():
                                    if k_c <= 0.0:
                                        continue
                                    k_per_cov_int[int(icov)] = max(1, int(math.ceil(k_c)))

                                if not k_per_cov_int:
                                    use_bias = False
                                    k_per_cov_for_selection = int(max(1, math.ceil(k_desired)))
                                else:
                                    k_per_cov_for_selection = k_per_cov_int
                                    k_per_cov_desired_map = k_per_cov_desired
                                    _log(
                                        f"[DEPTH {depth}] density bias active: "
                                        f"mode={bias_mode}, exponent={alpha}, "
                                        f"base_k_desired={k_desired:.4f}",
                                        always=True,
                                    )
                    else:
                        k_per_cov_for_selection = int(max(1, math.ceil(k_desired)))
                        k_per_cov_desired_map = None

                    # (c) candidate selection & exact reduction.
                    needed_cols = list(remainder_ddf.columns)
                    if cfg.columns.score not in needed_cols:
                        needed_cols.append(cfg.columns.score)
                    if "__icov__" not in needed_cols:
                        needed_cols.append("__icov__")
                    sel_ddf = remainder_ddf[needed_cols]

                    meta_cand = _get_meta_df(sel_ddf)

                    cand_ddf = sel_ddf.map_partitions(
                        _candidates_by_coverage_partition,
                        score_col=cfg.columns.score,
                        order_desc=cfg.algorithm.order_desc,
                        k_per_cov=k_per_cov_for_selection,
                        tie_buffer=int(cfg.algorithm.tie_buffer),
                        ra_col=RA_NAME,
                        dec_col=DEC_NAME,
                        meta=meta_cand,
                    )

                    target_parts = cfg.cluster.n_workers * cfg.cluster.threads_per_worker * 2

                    if (not is_hats) and hasattr(cand_ddf, "shuffle"):
                        cand_ddf = cand_ddf.shuffle(
                            "__icov__",
                            npartitions=max(target_parts, sel_ddf.npartitions),
                        )
                    elif not is_hats and hasattr(cand_ddf, "_ddf") and hasattr(cand_ddf._ddf, "shuffle"):
                        base_ddf = cand_ddf._ddf  # type: ignore[attr-defined]
                        cand_ddf = base_ddf.shuffle(
                            "__icov__",
                            npartitions=max(target_parts, sel_ddf.npartitions),
                        )
                    else:
                        _log(
                            f"[DEPTH {depth}] HATS / LSDB path or no shuffle available "
                            f"→ keeping native partitioning for __icov__",
                            always=True,
                        )

                    if avoid_computes:
                        selected_ddf = _reduce_coverage_exact_dask(
                            cand_ddf,
                            score_col=cfg.columns.score,
                            order_desc=cfg.algorithm.order_desc,
                            k_per_cov=k_per_cov_for_selection,
                            ra_col=RA_NAME,
                            dec_col=DEC_NAME,
                        )

                        selected_pdf = selected_ddf.compute()
                        _log_depth_stats(
                            _log,
                            depth,
                            "selected_before_fractional",
                            selected_len=len(selected_pdf),
                        )
                    else:
                        cand_pdf = cand_ddf.compute()
                        _log_depth_stats(
                            _log,
                            depth,
                            "candidates",
                            candidates_len=len(cand_pdf),
                        )

                        selected_pdf = _reduce_coverage_exact(
                            cand_pdf,
                            score_col=cfg.columns.score,
                            order_desc=cfg.algorithm.order_desc,
                            k_per_cov=k_per_cov_for_selection,
                            ra_col=RA_NAME,
                            dec_col=DEC_NAME,
                        )
                        _log_depth_stats(
                            _log,
                            depth,
                            "selected_before_fractional",
                            selected_len=len(selected_pdf),
                        )

                    selected_pdf, _dropped_pdf = apply_fractional_k_per_cov(
                        selected_pdf,
                        k_desired=k_desired,
                        score_col=cfg.columns.score,
                        order_desc=cfg.algorithm.order_desc,
                        mode=getattr(cfg.algorithm, "fractional_mode", "random"),
                        mode_logic=getattr(
                            cfg.algorithm,
                            "fractional_mode_logic",
                            "auto",
                        ),
                        ra_col=RA_NAME,
                        dec_col=DEC_NAME,
                        k_per_cov_desired_map=k_per_cov_desired_map,
                    )
                    _log_depth_stats(
                        _log,
                        depth,
                        "selected",
                        selected_len=len(selected_pdf),
                    )

                    if len(selected_pdf) > 0:
                        ra_vals = pd.to_numeric(
                            selected_pdf[RA_NAME],
                            errors="coerce",
                        ).to_numpy()
                        dec_vals = pd.to_numeric(
                            selected_pdf[DEC_NAME],
                            errors="coerce",
                        ).to_numpy()

                        theta = np.deg2rad(90.0 - dec_vals)
                        phi = np.deg2rad(ra_vals % 360.0)

                        NSIDE_L = 1 << depth
                        ipixL = hp.ang2pix(
                            NSIDE_L,
                            theta,
                            phi,
                            nest=True,
                        ).astype(np.int64)
                        selected_pdf["__ipix__"] = ipixL

                    header_line = build_header_line_from_keep(keep_cols)
                    counts = densmaps[depth]
                    allsky_needed = depth in (1, 2)

                    written_per_ipix, allsky_df = finalize_write_tiles(
                        out_dir=out_dir,
                        depth=depth,
                        header_line=header_line,
                        ra_col=RA_NAME,
                        dec_col=DEC_NAME,
                        counts=counts,
                        selected=selected_pdf,
                        order_desc=cfg.algorithm.order_desc,
                        allsky_collect=allsky_needed,
                    )
                    _log_depth_stats(
                        _log,
                        depth,
                        "written",
                        counts=densmaps[depth],
                        written=written_per_ipix,
                    )

                    if allsky_needed and allsky_df is not None and len(allsky_df) > 0:
                        norder_dir = out_dir / f"Norder{depth}"
                        norder_dir.mkdir(parents=True, exist_ok=True)
                        tmp_allsky = norder_dir / ".Allsky.tsv.tmp"
                        final_allsky = norder_dir / "Allsky.tsv"

                        nsrc_tot = int(counts.sum())
                        nwritten_tot = int(sum(written_per_ipix.values())) if written_per_ipix else 0
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
                        os.replace(tmp_allsky, final_allsky)

                    thr_cov = build_cov_thresholds(
                        selected_pdf,
                        score_col=cfg.columns.score,
                        order_desc=cfg.algorithm.order_desc,
                    )
                    if len(thr_cov) == 0:
                        _log(
                            f"[INFO] Depth {depth}: nothing selected; " "stopping selection loop.",
                            always=True,
                        )
                        break

                    remainder_meta = _get_meta_df(remainder_ddf)

                    remainder_ddf = remainder_ddf.map_partitions(
                        filter_remainder_by_coverage_partition,
                        score_expr=cfg.columns.score,
                        order_desc=cfg.algorithm.order_desc,
                        thr_cov=thr_cov,
                        ra_col=RA_NAME,
                        dec_col=DEC_NAME,
                        meta=remainder_meta,
                    )

                    if persist_ddfs:
                        remainder_ddf = remainder_ddf.persist()
                        wait(remainder_ddf)

                _log(
                    f"[DEPTH {depth}] done in {_fmt_dur(time.time() - depth_t0)}",
                    always=True,
                )

    try:
        if diagnostics_mode == "global":
            # Global diagnostics handled at cluster level.
            from dask.distributed import performance_report

            global_report = report_dir / "dask_global.html"
            with performance_report(filename=str(global_report)):
                _run_core_pipeline()
        else:
            # "per_step" or "off": diagnostics handled by diag_ctx or disabled.
            _run_core_pipeline()
    finally:
        # Graceful shutdown.
        try:
            shutdown_cluster(runtime)
        except Exception:
            pass

        t1 = time.time()
        elapsed = t1 - t0
        _log(
            f"END HiPS catalog pipeline. Elapsed {_fmt_dur(elapsed)} " f"({elapsed:.3f} s)",
            always=True,
        )

        # Persist log
        try:
            with (out_dir / "process.log").open("a", encoding="utf-8") as f:
                f.write("\n".join(log_lines) + "\n")
        except Exception as e:
            _log(
                f"{_ts()} | ERROR writing process.log: " f"{type(e).__name__}: {e}",
                always=True,
            )
