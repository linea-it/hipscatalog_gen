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

import json

# Standard library
import shutil
import time

# Internal modules
from contextlib import suppress
from pathlib import Path

from ..cluster.runtime import setup_cluster, shutdown_cluster
from ..config import Config
from ..io.output import write_properties
from ..utils import _mkdirs, _ts
from .common import (
    build_and_prepare_input,
    compute_and_write_densmaps,
    compute_input_total,
    log_epilogue,
    log_prologue,
    write_common_static_products,
    write_counts_summaries,
)
from .logging_utils import setup_structured_logger
from .modes import get_selection_mode
from .structure import PipelineContext, PipelineStage, run_stages
from .validation import validate_common_cfg

__all__ = ["run_pipeline"]


# =============================================================================
# Pipeline (per_cov-only)
# =============================================================================


def run_pipeline(cfg: Config, *, json_logs: bool = False) -> None:
    """Run the full HiPS catalog generation pipeline.

    Args:
        cfg: Parsed configuration object with input, algorithm, cluster, and output options.
        json_logs: Whether to also emit structured JSON lines to ``process.jsonl``.

    Raises:
        ValueError: If ``output.out_dir`` already exists without ``output.overwrite`` set.
        ValueError: If ``level_limit`` is outside the supported range [4, 11].
        ValueError: If the configured ``selection_mode`` is unsupported.
    """
    out_dir = Path(cfg.output.out_dir)
    t0 = time.time()

    overwrite = bool(getattr(cfg.output, "overwrite", False))
    if out_dir.exists():
        if overwrite:
            print(f"{_ts()} | [output] overwrite=True -> deleting existing contents under {out_dir}")
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

    log_ctx, _log = setup_structured_logger(
        out_dir,
        getattr(cfg.algorithm, "selection_mode", "mag_global"),
        json_logs=json_logs,
    )

    log_prologue(cfg, out_dir, _log)

    if not (4 <= int(cfg.algorithm.level_limit) <= 11):
        raise ValueError("level_limit (lM) must be within [4, 11] to mirror the CDS tool.")

    validate_common_cfg(cfg)

    selection_mode = (getattr(cfg.algorithm, "selection_mode", "mag_global") or "mag_global").lower()
    mode_entry = get_selection_mode(selection_mode)

    runtime, diag_ctx = setup_cluster(cfg.cluster, report_dir, _log)
    persist_ddfs = runtime.persist_ddfs
    avoid_computes = runtime.avoid_computes
    diagnostics_mode = runtime.diagnostics_mode

    ctx = PipelineContext(
        cfg=cfg,
        out_dir=out_dir,
        report_dir=report_dir,
        log_fn=_log,
        diag_ctx=diag_ctx,
        persist_ddfs=persist_ddfs,
        avoid_computes=avoid_computes,
        selection_mode=selection_mode,
        log_ctx=log_ctx,
    )

    def _stage_prepare_input(context: PipelineContext) -> PipelineContext:
        """Load inputs, validate RA/DEC, and set partitioning flags."""
        ddf, RA_NAME, DEC_NAME, keep_cols, is_hats, paths = build_and_prepare_input(
            context.cfg, context.diag_ctx, context.log_fn, context.persist_ddfs
        )
        return context.with_updates(
            ddf=ddf,
            RA_NAME=RA_NAME,
            DEC_NAME=DEC_NAME,
            keep_cols=keep_cols,
            is_hats=is_hats,
            paths=paths,
        )

    def _stage_count_input(context: PipelineContext) -> PipelineContext:
        """Compute total rows after RA/DEC validation."""
        if context.ddf is None:
            raise RuntimeError("Pipeline context missing input DDF.")  # pragma: no cover
        input_total = compute_input_total(
            context.ddf, context.diag_ctx, context.log_fn, context.avoid_computes
        )
        return context.with_updates(input_total=input_total)

    def _stage_prepare_selection(context: PipelineContext) -> PipelineContext:
        """Prepare remainder DDF for tile writing after selection normalization."""
        if context.ddf is None:
            raise RuntimeError("Pipeline context missing input DDF.")  # pragma: no cover
        if context.selection_params is None:
            raise RuntimeError("Pipeline context missing selection parameters.")  # pragma: no cover
        remainder_ddf = mode_entry.prepare_fn(
            context.ddf,
            context.cfg,
            context.diag_ctx,
            context.log_fn,
            context.selection_params,
            persist_ddfs=context.persist_ddfs,
            avoid_computes=context.avoid_computes,
        )
        return context.with_updates(remainder_ddf=remainder_ddf)

    def _stage_densmaps(context: PipelineContext) -> PipelineContext:
        """Compute density maps and write FITS outputs."""
        if context.remainder_ddf is None or context.RA_NAME is None or context.DEC_NAME is None:
            raise RuntimeError(
                "Pipeline context missing selection inputs for densmap computation."
            )  # pragma: no cover
        densmaps = compute_and_write_densmaps(
            ddf_sel=context.remainder_ddf,
            ra_col=context.RA_NAME,
            dec_col=context.DEC_NAME,
            level_limit=context.cfg.algorithm.level_limit,
            out_dir=context.out_dir,
            diag_ctx=context.diag_ctx,
        )
        return context.with_updates(densmaps=densmaps)

    def _stage_static_products(context: PipelineContext) -> PipelineContext:
        """Write static artifacts (arguments, metadata, densmap files)."""
        if (
            context.keep_cols is None
            or context.RA_NAME is None
            or context.DEC_NAME is None
            or context.paths is None
            or context.ddf is None
        ):
            raise RuntimeError(
                "Pipeline context missing data for static product writing."
            )  # pragma: no cover
        write_common_static_products(
            context.out_dir,
            context.cfg,
            context.densmaps,
            context.keep_cols,
            context.RA_NAME,
            context.DEC_NAME,
            context.paths,
            context.ddf,
        )
        return context

    def _stage_run_selection(context: PipelineContext) -> PipelineContext:
        """Run the configured selection mode and write tiles."""
        if (
            context.remainder_ddf is None
            or context.keep_cols is None
            or context.RA_NAME is None
            or context.DEC_NAME is None
        ):
            raise RuntimeError("Pipeline context missing selection inputs.")  # pragma: no cover
        mode_entry.run_fn(
            remainder_ddf=context.remainder_ddf,
            densmaps=context.densmaps,
            keep_cols=context.keep_cols,
            ra_col=context.RA_NAME,
            dec_col=context.DEC_NAME,
            cfg=context.cfg,
            out_dir=context.out_dir,
            diag_ctx=context.diag_ctx,
            log_fn=context.log_fn,
            avoid_computes=context.avoid_computes,
            params=context.selection_params,
        )
        return context

    def _stage_counts(context: PipelineContext) -> PipelineContext:
        """Write per-depth count summaries and store telemetry."""
        if context.input_total is None:
            raise RuntimeError("Pipeline context missing input totals.")  # pragma: no cover
        total_written, counts_payload = write_counts_summaries(
            context.out_dir, context.cfg.algorithm.level_limit, context.input_total, context.log_fn
        )
        telemetry = dict(context.telemetry)
        telemetry["output_counts"] = counts_payload
        return context.with_updates(total_written=total_written, telemetry=telemetry)

    def _stage_properties(context: PipelineContext) -> PipelineContext:
        """Write HiPS properties using the computed totals."""
        if context.total_written is None:
            raise RuntimeError("Pipeline context missing written counts.")  # pragma: no cover
        write_properties(
            context.out_dir,
            context.cfg.output,
            context.cfg.algorithm.level_limit,
            context.total_written,
            tile_format="tsv",
        )
        return context

    def _stage_normalize_selection(context: PipelineContext) -> PipelineContext:
        """Validate configuration and normalize selection parameters."""
        if context.ddf is None:
            raise RuntimeError("Pipeline context missing input DDF.")  # pragma: no cover
        mode_entry.validate_fn(context.cfg)
        normalized_ddf, params = mode_entry.normalize_fn(
            context.ddf,
            context.cfg,
            context.diag_ctx,
            context.log_fn,
            context.persist_ddfs,
            context.avoid_computes,
        )
        return context.with_updates(ddf=normalized_ddf, selection_params=params)

    pipeline_stages = [
        PipelineStage("prepare_input", _stage_prepare_input, diag_label="dask_prepare_input"),
        PipelineStage("input_total", _stage_count_input, diag_label="dask_input_total"),
        PipelineStage("normalize_selection", _stage_normalize_selection),
        PipelineStage(f"prepare_{selection_mode}", _stage_prepare_selection),
        PipelineStage("densmaps", _stage_densmaps, diag_label="dask_densmaps"),
        PipelineStage("static_products", _stage_static_products),
        PipelineStage(f"run_{selection_mode}", _stage_run_selection),
        PipelineStage("counts", _stage_counts),
        PipelineStage("properties", _stage_properties),
    ]

    def _run_core_pipeline() -> None:
        """Execute the ordered pipeline stages with telemetry updates."""
        final_ctx = run_stages(pipeline_stages, ctx)
        telemetry = dict(final_ctx.telemetry)
        telemetry["selection_mode"] = selection_mode
        telemetry["level_limit"] = cfg.algorithm.level_limit
        telemetry["moc_order"] = getattr(cfg.algorithm, "moc_order", cfg.algorithm.level_limit)
        telemetry["input_rows"] = final_ctx.input_total
        telemetry["output_rows"] = final_ctx.total_written
        telemetry["total_duration_s"] = time.time() - t0
        from contextlib import suppress

        with suppress(Exception):
            (out_dir / "telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")

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

        log_epilogue(out_dir, [], t0, _log, write_process_log=False)
