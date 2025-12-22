"""Pipeline data structures and execution harness."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Sequence

import numpy as np

from ..config import Config

LogFn = Callable[..., None]
DiagCtxFactory = Callable[[str], ContextManager[Any]]
StageFn = Callable[["PipelineContext"], "PipelineContext | None"]


@dataclass(slots=True)
class PipelineStage:
    """Represents a single ordered pipeline step."""

    name: str
    fn: StageFn
    diag_label: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Shared state passed between pipeline stages."""

    cfg: Config
    out_dir: Path
    report_dir: Path
    log_fn: LogFn
    diag_ctx: DiagCtxFactory
    persist_ddfs: bool
    avoid_computes: bool
    selection_mode: str
    log_ctx: Any | None = None
    ddf: Any | None = None
    RA_NAME: str | None = None
    DEC_NAME: str | None = None
    keep_cols: List[str] | None = None
    is_hats: bool = False
    paths: List[str] | None = None
    input_total: int | None = None
    remainder_ddf: Any | None = None
    densmaps: Dict[int, np.ndarray] = field(default_factory=dict)
    total_written: int | None = None
    selection_params: Any | None = None
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **kwargs: Any) -> PipelineContext:
        """Return a new context with updated fields."""
        return replace(self, **kwargs)


def run_stages(stages: Sequence[PipelineStage], ctx: PipelineContext) -> PipelineContext:
    """Execute ordered pipeline stages, threading a context object.

    Args:
        stages: Ordered sequence of ``PipelineStage`` objects.
        ctx: Initial pipeline context.

    Returns:
        Final context returned after the last stage.
    """
    for stage in stages:
        if ctx.log_ctx is not None:
            if hasattr(ctx.log_ctx, "stage"):
                ctx.log_ctx.stage = stage.name
            elif isinstance(ctx.log_ctx, dict):
                ctx.log_ctx["stage"] = stage.name
        diag_label = stage.diag_label or stage.name
        import time

        t0 = time.time()
        with ctx.diag_ctx(diag_label):
            result = stage.fn(ctx)
            if result is not None:
                ctx = result
        duration = time.time() - t0
        # Telemetry is kept mutable for low overhead.
        stage_metrics = ctx.telemetry.get("stages", {})
        stage_metrics[stage.name] = {"duration_s": duration}
        ctx.telemetry["stages"] = stage_metrics
        if ctx.log_ctx is not None:
            if hasattr(ctx.log_ctx, "stage"):
                ctx.log_ctx.stage = None
            elif isinstance(ctx.log_ctx, dict):
                ctx.log_ctx["stage"] = None
    return ctx
