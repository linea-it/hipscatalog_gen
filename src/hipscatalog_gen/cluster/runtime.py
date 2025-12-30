"""Dask cluster setup and teardown utilities."""

from __future__ import annotations

from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Tuple, cast

import dask
from dask.distributed import Client, LocalCluster, performance_report

from ..config import ClusterCfg

if TYPE_CHECKING:
    from dask_jobqueue import SLURMCluster as DaskSLURMCluster

SLURMCluster: type[Any] | None

try:
    # Optional for SLURM-based clusters
    from dask_jobqueue import SLURMCluster as _SLURMCluster

    SLURMCluster = cast(type["DaskSLURMCluster"], _SLURMCluster)
except Exception:  # pragma: no cover - optional dependency
    # When dask_jobqueue is not available, we keep SLURMCluster as a placeholder.
    SLURMCluster = None


__all__ = [
    "ClusterRuntime",
    "setup_cluster",
    "shutdown_cluster",
]


# Use task-based shuffle for DataFrame operations (original behaviour).
dask.config.set({"dataframe.shuffle.method": "tasks"})


@dataclass
class ClusterRuntime:
    """Runtime handles for the Dask cluster."""

    cluster: Any
    client: Client
    persist_ddfs: bool
    avoid_computes: bool
    diagnostics_mode: str


def setup_cluster(
    cfg: ClusterCfg,
    report_dir: Path,
    log_fn: Callable[[str, bool], None],
) -> Tuple[ClusterRuntime, Callable[[str], ContextManager[Any]]]:
    """Create and configure the Dask cluster and diagnostics context.

    Args:
        cfg: Cluster configuration (local or SLURM) with worker counts and memory limits.
        report_dir: Directory where per-step diagnostics reports are written.
        log_fn: Logging callback ``(message, always)``.

    Returns:
        Tuple[ClusterRuntime, Callable[[str], ContextManager[Any]]]: A pair
        ``(runtime, diag_ctx_factory)``, where:

        - ``runtime``: ``ClusterRuntime`` with cluster/client handles and flags.
        - ``diag_ctx_factory``: callable ``label -> context manager`` used as
          ``with diag_ctx_factory("step_name"):`` around pipeline steps.

    Raises:
        ImportError: If ``mode='slurm'`` is set but ``dask-jobqueue`` is not available.
    """
    # ------------------------------------------------------------------
    # Cluster creation (local or SLURM)
    # ------------------------------------------------------------------
    cluster: Any
    client: Client

    if cfg.mode == "slurm":
        if SLURMCluster is None:
            raise ImportError("dask-jobqueue is required for mode='slurm'")
        sl = cfg.slurm or {}
        job_directives = sl.get("job_extra_directives", sl.get("job_extra", []))

        cluster = SLURMCluster(
            queue=sl.get("queue", "cpu_dev"),
            account=sl.get("account", None),
            cores=cfg.threads_per_worker,
            processes=1,
            memory=cfg.memory_per_worker,
            job_extra_directives=job_directives,
        )
        cluster.scale(cfg.n_workers)
        client = Client(cluster)
    else:
        cluster = LocalCluster(
            n_workers=cfg.n_workers,
            threads_per_worker=cfg.threads_per_worker,
            memory_limit=cfg.memory_per_worker,
        )
        client = Client(cluster)

    log_fn(f"Dask dashboard: {client.dashboard_link}", True)

    # ------------------------------------------------------------------
    # Memory vs. compute policy
    # ------------------------------------------------------------------
    persist_ddfs = bool(getattr(cfg, "persist_ddfs", False))
    avoid_computes = bool(getattr(cfg, "avoid_computes_wherever_possible", True))

    if persist_ddfs:
        log_fn("[cluster] persist_ddfs=True → will persist large intermediates in memory", True)
    else:
        log_fn("[cluster] persist_ddfs=False (lower memory consumption)", True)

    if avoid_computes:
        log_fn(
            "[cluster] avoid_computes_wherever_possible=True → will try to avoid large .compute() calls "
            "whenever possible (using more Dask-native operations) (lower memory consumption)",
            True,
        )
    else:
        log_fn(
            "[cluster] avoid_computes_wherever_possible=False → keep standard behaviour for computes.",
            True,
        )

    diagnostics_mode = getattr(cfg, "diagnostics_mode", "per_step")

    # ------------------------------------------------------------------
    # Diagnostics context factory
    # ------------------------------------------------------------------
    def diag_ctx(label: str) -> ContextManager[Any]:
        """Return a diagnostics context for a labeled pipeline step.

        Modes:

        - ``per_step``: one HTML report per labeled step.
        - ``global``: handled by an outer global report (no-op here).
        - ``off``: diagnostics disabled (no-op).
        """
        if diagnostics_mode == "per_step":
            return performance_report(filename=str(report_dir / f"{label}.html"))
        # "global" or "off" → no per-step diagnostics
        return nullcontext()

    runtime = ClusterRuntime(
        cluster=cluster,
        client=client,
        persist_ddfs=persist_ddfs,
        avoid_computes=avoid_computes,
        diagnostics_mode=str(diagnostics_mode),
    )

    return runtime, diag_ctx


def shutdown_cluster(runtime: ClusterRuntime) -> None:
    """Gracefully close client and cluster.

    Args:
        runtime: ClusterRuntime with cluster and client handles.
    """
    with suppress(Exception):
        runtime.client.close()
    with suppress(Exception):
        runtime.cluster.close()
