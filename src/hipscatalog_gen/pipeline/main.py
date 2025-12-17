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

# Standard library
import shutil
import time

# Internal modules
from contextlib import suppress
from pathlib import Path
from typing import List

from ..cluster.runtime import setup_cluster, shutdown_cluster
from ..config import Config
from ..coverage.pipeline import add_coverage_column, run_coverage_selection
from ..mag_global.pipeline import prepare_mag_global, run_mag_global_selection
from ..score_global.pipeline import prepare_score_global, run_score_global_selection
from ..utils import _mkdirs, _ts
from .common import (
    build_and_prepare_input,
    compute_and_write_densmaps,
    log_epilogue,
    log_prologue,
    write_common_static_products,
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
    t0 = time.time()
    log_lines: List[str] = []

    def _log(msg: str, always: bool = False) -> None:
        line = f"{_ts()} | {msg}"
        print(line)
        log_lines.append(line)

    overwrite = bool(getattr(cfg.output, "overwrite", False))
    if out_dir.exists():
        if overwrite:
            _log(f"[output] overwrite=True -> deleting existing contents under {out_dir}", True)
            if out_dir.is_file():
                out_dir.unlink()
            else:
                shutil.rmtree(out_dir)
        else:
            raise ValueError(
                f"output.out_dir already exists at {out_dir}. "
                "Set output.overwrite=true to delete it before writing a new catalog."
            )

    _mkdirs(out_dir)

    report_dir = out_dir / "dask_reports"
    _mkdirs(report_dir)

    log_prologue(cfg, out_dir, _log)

    if not (4 <= int(cfg.algorithm.level_limit) <= 11):
        raise ValueError("level_limit (lM) must be within [4, 11] to mirror the CDS tool.")

    if cfg.algorithm.level_coverage > cfg.algorithm.level_limit:
        cfg.algorithm.level_coverage = cfg.algorithm.level_limit
        _log("WARNING: level_coverage was > level_limit; set lC = lM", always=True)

    fmt_lower = str(cfg.input.format).lower()
    if fmt_lower != "hats" and getattr(cfg.algorithm, "use_hats_as_coverage", False):
        _log(
            "[config] algorithm.use_hats_as_coverage=True was requested, but "
            "input.format is not 'hats'. This option is only meaningful for "
            "HATS/LSDB catalogs and will be ignored for this run.",
            always=True,
        )

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

    runtime, diag_ctx = setup_cluster(cfg.cluster, report_dir, _log)
    persist_ddfs = runtime.persist_ddfs
    avoid_computes = runtime.avoid_computes
    diagnostics_mode = runtime.diagnostics_mode

    def _run_core_pipeline() -> None:
        ddf, RA_NAME, DEC_NAME, keep_cols, is_hats, paths = build_and_prepare_input(
            cfg, diag_ctx, _log, persist_ddfs
        )

        selection_mode = (getattr(cfg.algorithm, "selection_mode", "coverage") or "coverage").lower()

        if selection_mode == "mag_global":
            remainder_ddf = prepare_mag_global(ddf, cfg, diag_ctx, _log)
        elif selection_mode == "score_global":
            remainder_ddf = prepare_score_global(ddf, cfg, diag_ctx, _log)
        else:
            remainder_ddf = add_coverage_column(ddf, cfg, is_hats, RA_NAME, DEC_NAME, _log)

        densmaps = compute_and_write_densmaps(
            ddf_sel=remainder_ddf,
            ra_col=RA_NAME,
            dec_col=DEC_NAME,
            level_limit=cfg.algorithm.level_limit,
            out_dir=out_dir,
            diag_ctx=diag_ctx,
        )

        write_common_static_products(out_dir, cfg, densmaps, keep_cols, RA_NAME, DEC_NAME, paths, ddf)

        if selection_mode == "mag_global":
            run_mag_global_selection(
                remainder_ddf=remainder_ddf,
                densmaps=densmaps,
                keep_cols=keep_cols,
                ra_col=RA_NAME,
                dec_col=DEC_NAME,
                cfg=cfg,
                out_dir=out_dir,
                diag_ctx=diag_ctx,
                log_fn=_log,
            )
        elif selection_mode == "score_global":
            run_score_global_selection(
                remainder_ddf=remainder_ddf,
                densmaps=densmaps,
                keep_cols=keep_cols,
                ra_col=RA_NAME,
                dec_col=DEC_NAME,
                cfg=cfg,
                out_dir=out_dir,
                diag_ctx=diag_ctx,
                log_fn=_log,
            )
        else:
            run_coverage_selection(
                remainder_ddf=remainder_ddf,
                cfg=cfg,
                densmaps=densmaps,
                keep_cols=keep_cols,
                ra_col=RA_NAME,
                dec_col=DEC_NAME,
                out_dir=out_dir,
                diag_ctx=diag_ctx,
                log_fn=_log,
                persist_ddfs=persist_ddfs,
                avoid_computes=avoid_computes,
                is_hats=is_hats,
            )

    try:
        if diagnostics_mode == "global":
            from dask.distributed import performance_report

            global_report = report_dir / "dask_global.html"
            with performance_report(filename=str(global_report)):
                _run_core_pipeline()
        else:
            _run_core_pipeline()
    finally:
        with suppress(Exception):
            shutdown_cluster(runtime)

        log_epilogue(out_dir, log_lines, t0, _log)
