"""Unit tests for Dask cluster runtime helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hipscatalog_gen.cluster import runtime
from hipscatalog_gen.config import ClusterCfg


@pytest.fixture
def log_capture():
    """Collect log messages emitted during setup/shutdown."""
    messages = []

    def _log_fn(msg: str, always: bool = False, **_: object) -> None:
        messages.append(msg)

    return messages, _log_fn


def test_setup_cluster_local_with_per_step_diagnostics(tmp_path, monkeypatch, log_capture):
    """Local cluster setup wires client/cluster, flags, and per-step diagnostics."""
    logs, log_fn = log_capture
    created = {}

    class FakeCluster:
        def __init__(self, **kwargs):
            created["local_cluster_kwargs"] = kwargs

        def close(self):
            created["cluster_closed"] = True

    class FakeClient:
        def __init__(self, cluster):
            created["client_cluster"] = cluster
            self.dashboard_link = "http://localhost:8787"

        def close(self):
            created["client_closed"] = True

    class DummyReportCtx:
        def __init__(self, filename):
            created["report_filename"] = filename

        def __enter__(self):
            created["report_entered"] = True
            return "report_ctx"

        def __exit__(self, exc_type, exc_val, exc_tb):
            created["report_exited"] = True

    monkeypatch.setattr(runtime, "LocalCluster", FakeCluster)
    monkeypatch.setattr(runtime, "Client", FakeClient)
    monkeypatch.setattr(runtime, "performance_report", lambda filename: DummyReportCtx(filename))

    cfg = ClusterCfg(
        mode="local",
        n_workers=2,
        threads_per_worker=3,
        memory_per_worker="4GB",
        persist_ddfs=True,
        avoid_computes_wherever_possible=False,
        diagnostics_mode="per_step",
    )

    runtime_obj, diag_ctx_factory = runtime.setup_cluster(cfg, tmp_path, log_fn)

    assert isinstance(runtime_obj.cluster, FakeCluster)
    assert isinstance(runtime_obj.client, FakeClient)
    assert runtime_obj.persist_ddfs is True
    assert runtime_obj.avoid_computes is False
    assert runtime_obj.diagnostics_mode == "per_step"

    assert created["local_cluster_kwargs"] == {
        "n_workers": 2,
        "threads_per_worker": 3,
        "memory_limit": "4GB",
    }
    assert created["client_cluster"] is runtime_obj.cluster
    assert any("Dask dashboard" in msg for msg in logs)
    assert any("persist_ddfs=True" in msg for msg in logs)
    assert any("avoid_computes_wherever_possible=False" in msg for msg in logs)

    with diag_ctx_factory("step1") as ctx:
        assert ctx == "report_ctx"
    assert tmp_path / "step1.html" == Path(created["report_filename"])
    assert created["report_entered"] and created["report_exited"]

    # shutdown_cluster closes client/cluster with no errors
    runtime.shutdown_cluster(runtime_obj)
    assert created["client_closed"] and created["cluster_closed"]


def test_diag_ctx_global_and_off(monkeypatch, log_capture, tmp_path):
    """Diagnostics mode global/off returns nullcontext (no performance reports)."""
    logs, log_fn = log_capture
    # Reuse simple fakes
    monkeypatch.setattr(runtime, "LocalCluster", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        runtime, "Client", lambda cluster: SimpleNamespace(cluster=cluster, dashboard_link="link")
    )

    for mode in ("global", "off"):
        cfg = ClusterCfg(
            mode="local",
            n_workers=1,
            threads_per_worker=1,
            memory_per_worker="1GB",
            diagnostics_mode=mode,
        )
        runtime_obj, diag_ctx_factory = runtime.setup_cluster(cfg, tmp_path, log_fn)
        with diag_ctx_factory("anything") as ctx:
            assert ctx is None  # nullcontext yields None
        runtime.shutdown_cluster(runtime_obj)


def test_setup_cluster_slurm_missing_dependency(monkeypatch, log_capture, tmp_path):
    """If dask-jobqueue is unavailable, mode='slurm' raises ImportError."""
    logs, log_fn = log_capture
    monkeypatch.setattr(runtime, "SLURMCluster", None)
    cfg = ClusterCfg(
        mode="slurm",
        n_workers=1,
        threads_per_worker=1,
        memory_per_worker="1GB",
    )
    with pytest.raises(ImportError):
        runtime.setup_cluster(cfg, tmp_path, log_fn)
    assert not logs  # nothing logged before failure


def test_setup_cluster_slurm_success(monkeypatch, tmp_path, log_capture):
    """SLURM branch scales cluster and constructs client with directives."""
    logs, log_fn = log_capture
    created = {}

    class FakeSLURMCluster:
        def __init__(self, **kwargs):
            created["slurm_kwargs"] = kwargs

        def scale(self, n):
            created["scaled"] = n

        def close(self):
            created["slurm_closed"] = True

    class FakeClient:
        def __init__(self, cluster):
            created["client_cluster"] = cluster
            self.dashboard_link = "slurm-link"

        def close(self):
            created["client_closed"] = True

    monkeypatch.setattr(runtime, "SLURMCluster", FakeSLURMCluster)
    monkeypatch.setattr(runtime, "Client", FakeClient)

    cfg = ClusterCfg(
        mode="slurm",
        n_workers=4,
        threads_per_worker=2,
        memory_per_worker="8GB",
        slurm={"queue": "q", "account": "acc", "job_extra_directives": ["--x"]},
    )

    runtime_obj, diag_ctx_factory = runtime.setup_cluster(cfg, tmp_path, log_fn)
    assert created["slurm_kwargs"]["queue"] == "q"
    assert created["slurm_kwargs"]["account"] == "acc"
    assert created["slurm_kwargs"]["cores"] == 2
    assert created["slurm_kwargs"]["processes"] == 1
    assert created["slurm_kwargs"]["memory"] == "8GB"
    assert created["slurm_kwargs"]["job_extra_directives"] == ["--x"]
    assert created["scaled"] == 4
    assert created["client_cluster"] is runtime_obj.cluster
    assert runtime_obj.diagnostics_mode in {"per_step", "global", "off"}
    assert any("Dask dashboard" in msg for msg in logs)

    # diag_ctx_factory should default to nullcontext for non-per_step mode (global default)
    with diag_ctx_factory("label") as ctx:
        assert ctx is None

    runtime.shutdown_cluster(runtime_obj)
    assert created["client_closed"]
    assert created["slurm_closed"]


def test_shutdown_cluster_swallows_exceptions():
    """shutdown_cluster ignores errors raised by close() on client/cluster."""

    class BadClose:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True
            raise RuntimeError("fail")

    bad_client = BadClose()
    bad_cluster = BadClose()
    runtime_obj = runtime.ClusterRuntime(
        cluster=bad_cluster,
        client=bad_client,
        persist_ddfs=False,
        avoid_computes=False,
        diagnostics_mode="off",
    )

    runtime.shutdown_cluster(runtime_obj)  # Should not raise
    assert bad_client.closed and bad_cluster.closed
