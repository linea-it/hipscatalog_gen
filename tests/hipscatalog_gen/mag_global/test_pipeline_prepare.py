from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest
from dask import dataframe as dd
from hipscatalog_gen.mag_global.pipeline import prepare_mag_global


@pytest.fixture
def diag_ctx():
    """Returns a no-op diagnostic context."""
    return lambda name: nullcontext()


@pytest.fixture
def log_capture():
    """Collects log messages emitted by the pipeline."""
    logs: list[str] = []

    def _log_fn(msg: str, always: bool = False) -> None:
        logs.append(msg)

    return logs, _log_fn


def _cfg(**algo_kwargs) -> SimpleNamespace:
    """Creates a minimal config namespace for mag_global tests."""
    algo_defaults = dict(
        mag_hist_nbins=4,
        mag_adaptive_range="complete",
        level_limit=2,
        order_desc=True,
        n_1=None,
        n_2=None,
        n_3=None,
        low_memory_mode=True,
        persist_ddfs=False,
        avoid_computes=True,
    )
    algo_defaults.update(algo_kwargs)
    cluster_defaults = dict(low_memory_mode=algo_defaults.pop("low_memory_mode", True))
    return SimpleNamespace(
        algorithm=SimpleNamespace(**algo_defaults),
        cluster=SimpleNamespace(**cluster_defaults),
    )


def test_prepare_with_mag_column_sets_bounds_and_filters(diag_ctx, log_capture):
    """Tests that mag_column path sets mag_min/mag_max and returns filtered DDF."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [18.0, 19.5, 21.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(mag_column="MAG")

    ddf_sel = prepare_mag_global(ddf, cfg, diag_ctx, log_fn)
    result = ddf_sel.compute()

    assert cfg.algorithm.mag_min == 18.0
    assert cfg.algorithm.mag_max == 21.0
    assert result["__mag__"].tolist() == [18.0, 19.5, 21.0]
    assert any("mag_adaptive_range=complete" in msg for msg in logs)


def test_prepare_with_flux_column_converts_and_respects_window(diag_ctx, log_capture):
    """Tests that flux_column path converts to magnitudes and applies provided bounds."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"FLUX": [10.0, 0.0, -1.0, 1.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(
        flux_column="FLUX",
        mag_offset=25.0,
        mag_min=22.0,
        mag_max=24.0,
    )

    ddf_sel = prepare_mag_global(ddf, cfg, diag_ctx, log_fn)
    result = ddf_sel.compute()

    # flux=10 -> mag=22.5, flux=1 -> mag=25 (filtered out), non-positive -> 99 (filtered out)
    assert result["__mag__"].tolist() == [22.5]
    assert cfg.algorithm.mag_min == 22.0
    assert cfg.algorithm.mag_max == 24.0


def test_prepare_hist_peak_computes_upper_from_histogram(diag_ctx, log_capture):
    """Tests that mag_adaptive_range=hist_peak derives missing bound from histogram mode."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [18.0, 18.5, 19.0, 24.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak", mag_min=18.0)

    ddf_sel = prepare_mag_global(ddf, cfg, diag_ctx, log_fn)
    result = ddf_sel.compute()

    assert cfg.algorithm.mag_min == 18.0
    assert cfg.algorithm.mag_max == 18.75  # peak centered on the first bin
    assert result["__mag__"].max() <= cfg.algorithm.mag_max
    assert any("hist_peak" in msg for msg in logs)


@pytest.mark.parametrize(
    "cfg_kwargs",
    [
        {"mag_column": "MAG", "flux_column": "FLUX"},
        {"mag_column": "MISSING"},
        {"flux_column": "MISSING", "mag_offset": 25.0},
    ],
)
def test_prepare_rejects_invalid_column_configuration(cfg_kwargs, diag_ctx, log_capture):
    """Tests that incompatible or missing column choices raise meaningful errors."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0], "FLUX": [10.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(**cfg_kwargs)

    with pytest.raises((ValueError, KeyError)):
        prepare_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_prepare_requires_mag_offset_with_flux(diag_ctx, log_capture):
    """Tests that flux_column path fails when mag_offset is missing."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"FLUX": [10.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(flux_column="FLUX")  # no mag_offset provided

    with pytest.raises(ValueError):
        prepare_mag_global(ddf, cfg, diag_ctx, log_fn)


@pytest.mark.parametrize(
    "values",
    [
        [5.0, 5.0, 5.0],  # degenerate global range
        [1.0, 2.0, 3.0],  # user-supplied inverted range
    ],
)
def test_prepare_raises_on_invalid_range(values, diag_ctx, log_capture):
    """Tests that invalid global ranges trigger ValueError early."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": values})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_kwargs = {"mag_column": "MAG"}
    if values != [5.0, 5.0, 5.0]:
        cfg_kwargs.update({"mag_min": 10.0, "mag_max": 5.0})

    cfg = _cfg(**cfg_kwargs)

    with pytest.raises(ValueError):
        prepare_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_prepare_hist_peak_raises_on_invalid_bounds(diag_ctx, log_capture):
    """Tests that hist_peak fails fast when histogram bounds are invalid."""
    _, log_fn = log_capture
    # Global range [-5, -4]; mag_min=0 forces upper < lower in histogram.
    pdf = pd.DataFrame({"MAG": [-5.0, -4.5, -4.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak", mag_min=0.0)

    with pytest.raises(ValueError):
        prepare_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_prepare_hist_peak_raises_when_histogram_empty(diag_ctx, log_capture):
    """Tests that hist_peak raises if no objects fall into the histogram window."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [50.0, 51.0, 52.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak")

    with pytest.raises(ValueError):
        prepare_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_prepare_rejects_unknown_adaptive_mode(diag_ctx, log_capture):
    """Tests that unsupported mag_adaptive_range values are rejected."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="invalid-mode")

    with pytest.raises(ValueError):
        prepare_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_prepare_complete_with_single_bound(diag_ctx, log_capture):
    """Tests that complete mode fills the missing bound with global min/max."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_min = _cfg(mag_column="MAG", mag_min=2.0, mag_adaptive_range="complete")
    ddf_min = prepare_mag_global(ddf, cfg_min, diag_ctx, log_fn)
    assert cfg_min.algorithm.mag_min == 2.0
    assert cfg_min.algorithm.mag_max == 3.0
    assert ddf_min.compute()["__mag__"].tolist() == [2.0, 3.0]
    assert any("mag_min provided" in msg for msg in logs)

    logs.clear()
    cfg_max = _cfg(mag_column="MAG", mag_max=2.0, mag_adaptive_range="complete")
    ddf_max = prepare_mag_global(ddf, cfg_max, diag_ctx, log_fn)
    assert cfg_max.algorithm.mag_min == 1.0
    assert cfg_max.algorithm.mag_max == 2.0
    assert ddf_max.compute()["__mag__"].tolist() == [1.0, 2.0]
    assert any("mag_max provided" in msg for msg in logs)


def test_prepare_hist_peak_from_mag_max_only(diag_ctx, log_capture):
    """Tests that hist_peak derives mag_min when only mag_max is provided."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [10.0, 11.0, 12.0, 30.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak", mag_max=12.5)

    ddf_sel = prepare_mag_global(ddf, cfg, diag_ctx, log_fn)
    result = ddf_sel.compute()

    assert cfg.algorithm.mag_max == 12.5
    assert cfg.algorithm.mag_min == 10.69  # peak bin center from lower bound scan (rounded)
    assert result["__mag__"].min() >= cfg.algorithm.mag_min


def test_prepare_hist_peak_without_bounds(diag_ctx, log_capture):
    """Tests that hist_peak with no bounds clips global min and derives max from histogram."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [-5.0, 0.0, 1.0, 1.5, 50.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak", mag_hist_nbins=3)

    ddf_sel = prepare_mag_global(ddf, cfg, diag_ctx, log_fn)
    result = ddf_sel.compute()

    assert cfg.algorithm.mag_min == -2.0  # clipped global minimum
    assert cfg.algorithm.mag_max <= 40.0  # clipped at upper bound
    assert result["__mag__"].between(cfg.algorithm.mag_min, cfg.algorithm.mag_max).all()


def test_prepare_respects_low_memory_mode_no_persist(diag_ctx, log_capture, monkeypatch):
    """Tests that low_memory_mode avoids persist."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", low_memory_mode=True)

    persisted = {"called": False}

    def fake_persist(self):
        persisted["called"] = True
        return self

    monkeypatch.setattr(type(ddf), "persist", fake_persist, raising=False)

    prepare_mag_global(ddf, cfg, diag_ctx, log_fn, avoid_computes=True)
    assert not persisted["called"]


def test_prepare_persists_when_low_memory_mode_false(diag_ctx, log_capture, monkeypatch):
    """Tests that low_memory_mode=False allows persist."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", low_memory_mode=False)
    cfg.cluster.low_memory_mode = False

    persisted = {"called": False}

    def fake_persist(self):
        persisted["called"] = True
        return self

    monkeypatch.setattr(type(ddf), "persist", fake_persist, raising=False)
    import dask.distributed as ddist

    monkeypatch.setattr(ddist, "wait", lambda *_args, **_kwargs: None, raising=False)

    prepare_mag_global(ddf, cfg, diag_ctx, log_fn, avoid_computes=False, persist_ddfs=True)
    assert persisted["called"]
    assert any("Persisting filtered DDF" in msg for msg in logs)
