"""Unit tests for mag_global prepare/normalize steps."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd
from hipscatalog_gen.mag_global.pipeline import normalize_mag_global, prepare_mag_global
from hipscatalog_gen.pipeline.params import MagGlobalParams


@pytest.fixture
def diag_ctx():
    """Returns a no-op diagnostic context."""
    return lambda name: nullcontext()


@pytest.fixture
def log_capture():
    """Collects log messages emitted by the pipeline."""
    logs: list[str] = []

    def _log_fn(msg: str, always: bool = False, **_: dict) -> None:
        """Capture log message into the local list for assertions."""
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
    """mag_column path sets params and returns filtered DDF."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [18.0, 19.5, 21.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(mag_column="MAG")

    ddf_norm, params = normalize_mag_global(ddf, cfg, diag_ctx, log_fn)
    ddf_sel = prepare_mag_global(ddf_norm, cfg, diag_ctx, log_fn, params)
    result = ddf_sel.compute()

    assert params.mag_min == 18.0
    assert params.mag_max == 21.0
    assert result["__mag__"].tolist() == [18.0, 19.5, 21.0]
    assert any("mag_adaptive_range=complete" in msg for msg in logs)


def test_prepare_with_flux_column_converts_and_respects_window(diag_ctx, log_capture):
    """flux_column path converts to magnitudes and applies provided bounds."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"FLUX": [10.0, 0.0, -1.0, 1.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(
        flux_column="FLUX",
        mag_offset=25.0,
        mag_min=22.0,
        mag_max=24.0,
    )

    ddf_norm, params = normalize_mag_global(ddf, cfg, diag_ctx, log_fn)
    ddf_sel = prepare_mag_global(ddf_norm, cfg, diag_ctx, log_fn, params)
    result = ddf_sel.compute()

    # flux=10 -> mag=22.5, flux=1 -> mag=25 (filtered out), non-positive -> 99 (filtered out)
    assert result["__mag__"].tolist() == [22.5]
    assert params.mag_min == 22.0
    assert params.mag_max == 24.0


def test_prepare_hist_peak_computes_upper_from_histogram(diag_ctx, log_capture):
    """hist_peak derives missing bound from histogram mode."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [18.0, 18.5, 19.0, 24.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak", mag_min=18.0)

    ddf_norm, params = normalize_mag_global(ddf, cfg, diag_ctx, log_fn)
    ddf_sel = prepare_mag_global(ddf_norm, cfg, diag_ctx, log_fn, params)
    result = ddf_sel.compute()

    assert params.mag_min == 18.0
    assert params.mag_max == 18.75  # peak centered on the first bin
    assert result["__mag__"].max() <= params.mag_max
    assert any("hist_peak" in msg for msg in logs)


@pytest.mark.parametrize(
    "cfg_kwargs",
    [
        {"mag_column": "MAG", "flux_column": "FLUX"},
        {"mag_column": "MISSING"},
        {"flux_column": "MISSING", "mag_offset": 25.0},
    ],
)
def test_normalize_rejects_invalid_column_configuration(cfg_kwargs, diag_ctx, log_capture):
    """Incompatible or missing column choices raise meaningful errors."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0], "FLUX": [10.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(**cfg_kwargs)

    with pytest.raises((ValueError, KeyError)):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_requires_mag_offset_with_flux(diag_ctx, log_capture):
    """flux_column path fails when mag_offset is missing."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"FLUX": [10.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(flux_column="FLUX")  # no mag_offset provided

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


@pytest.mark.parametrize(
    "values",
    [
        [5.0, 5.0, 5.0],  # degenerate global range
        [1.0, 2.0, 3.0],  # user-supplied inverted range
    ],
)
def test_normalize_raises_on_invalid_range(values, diag_ctx, log_capture):
    """Invalid global ranges trigger ValueError early."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": values})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_kwargs = {"mag_column": "MAG"}
    if values != [5.0, 5.0, 5.0]:
        cfg_kwargs.update({"mag_min": 10.0, "mag_max": 5.0})

    cfg = _cfg(**cfg_kwargs)

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_hist_peak_raises_on_invalid_bounds(diag_ctx, log_capture):
    """hist_peak fails fast when histogram bounds are invalid."""
    _, log_fn = log_capture
    # Global range [-5, -4]; mag_min=0 forces upper < lower in histogram.
    pdf = pd.DataFrame({"MAG": [-5.0, -4.5, -4.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak", mag_min=0.0)

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_hist_peak_raises_when_histogram_empty(diag_ctx, log_capture):
    """hist_peak raises if no objects fall into the histogram window."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [50.0, 51.0, 52.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak")

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_rejects_unknown_adaptive_mode(diag_ctx, log_capture):
    """Unsupported mag_adaptive_range values are rejected."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="invalid-mode")

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_rejects_nonfinite_global_range(diag_ctx, log_capture):
    """Non-finite global min/max are rejected when keep_invalid_values=False."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [float("nan"), float("inf")]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_keep_invalid_values=False)

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_rejects_missing_mag_and_flux(diag_ctx, log_capture):
    """Missing both mag_column and flux_column raises."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"OTHER": [1.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg()  # no mag/flux configured

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_keep_invalid_maps_to_sentinel(diag_ctx, log_capture):
    """keep_invalid_values maps NaN/Inf to a sentinel inside the resolved window."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [float("nan"), float("inf"), -float("inf"), 1.5]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(mag_column="MAG", mag_keep_invalid_values=True)

    ddf_norm, params = normalize_mag_global(ddf, cfg, diag_ctx, log_fn)
    result = ddf_norm.compute()

    assert params.sentinel is not None
    assert result["__mag__"].isna().sum() == 0
    assert params.mag_min <= params.sentinel <= params.mag_max
    assert any("keep_invalid_values=True" in msg for msg in logs)


def test_normalize_handles_empty_partition_mag(diag_ctx, log_capture):
    """Empty partitions in mag path are tolerated and filled with __mag__."""
    _, log_fn = log_capture
    empty = pd.DataFrame({"MAG": []})
    non_empty = pd.DataFrame({"MAG": [1.0, 2.0]})
    ddf = dd.from_delayed(
        [
            dd.from_pandas(empty, npartitions=1).to_delayed()[0],
            dd.from_pandas(non_empty, npartitions=1).to_delayed()[0],
        ],
        meta={"MAG": "f8"},
    )
    cfg = _cfg(mag_column="MAG")

    ddf_norm, params = normalize_mag_global(ddf, cfg, diag_ctx, log_fn)
    assert params.mag_min == 1.0 and params.mag_max == 2.0
    assert "__mag__" in ddf_norm.columns


def test_normalize_handles_empty_partition_flux(diag_ctx, log_capture):
    """Empty partitions in flux path are tolerated."""
    _, log_fn = log_capture
    empty = pd.DataFrame({"FLUX": []})
    non_empty = pd.DataFrame({"FLUX": [1.0, 10.0]})
    ddf = dd.from_delayed(
        [
            dd.from_pandas(empty, npartitions=1).to_delayed()[0],
            dd.from_pandas(non_empty, npartitions=1).to_delayed()[0],
        ],
        meta={"FLUX": "f8"},
    )
    cfg = _cfg(flux_column="FLUX", mag_offset=25.0)

    ddf_norm, _ = normalize_mag_global(ddf, cfg, diag_ctx, log_fn)
    assert "__mag__" in ddf_norm.columns


def test_normalize_minmax_none_raises(monkeypatch, diag_ctx, log_capture):
    """Guard against dask min/max returning None."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG")

    monkeypatch.setattr("hipscatalog_gen.mag_global.pipeline.dask_compute", lambda *_, **__: (None, None))

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_hist_peak_histogram_empty(diag_ctx, log_capture):
    """Histogram path raises when no rows fall into the histogram window."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [100.0, 101.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak")

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_hist_peak_mocked_empty_histogram(monkeypatch, diag_ctx, log_capture):
    """_histogram_peak raises when histogram reports zero total."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0, 2.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak", mag_max=5.0)

    def fake_histogram_ddf(**kwargs):
        return np.zeros(4, dtype="int64"), np.array([0, 1, 2, 3, 4], dtype="float64"), 0

    monkeypatch.setattr("hipscatalog_gen.mag_global.pipeline.compute_histogram_ddf", fake_histogram_ddf)

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_keep_invalid_hist_peak_not_allowed(diag_ctx, log_capture):
    """keep_invalid_values=True with hist_peak raises."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0, 2.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_keep_invalid_values=True, mag_adaptive_range="hist_peak")

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_keep_invalid_all_nan_inf_raises(diag_ctx, log_capture):
    """keep_invalid_values=True fails when all values are NaN/Inf."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [float("nan"), float("inf")]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_keep_invalid_values=True)

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_keep_invalid_finite_scan_returns_none(monkeypatch, diag_ctx, log_capture):
    """_finite_min_max returning None triggers guard."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [float("nan")]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_keep_invalid_values=True)

    monkeypatch.setattr("hipscatalog_gen.mag_global.pipeline._finite_min_max", lambda *_: (None, None))

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_keep_invalid_order_desc_false_expands_range(diag_ctx, log_capture):
    """Order ascending with keep_invalid_values expands mag_max to include sentinel."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0, 2.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_keep_invalid_values=True, order_desc=False, mg_order_desc=False)

    _, params = normalize_mag_global(ddf, cfg, diag_ctx, log_fn)
    assert params.sentinel is not None
    assert params.mag_max >= params.sentinel
    assert any("keep_invalid_values=True" in msg for msg in logs)


def test_normalize_complete_with_single_bound(diag_ctx, log_capture):
    """complete mode fills the missing bound with global min/max."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    cfg_min = _cfg(mag_column="MAG", mag_min=2.0, mag_adaptive_range="complete")
    _, params_min = normalize_mag_global(ddf, cfg_min, diag_ctx, log_fn)
    assert params_min.mag_min == 2.0
    assert params_min.mag_max == 3.0

    cfg_max = _cfg(mag_column="MAG", mag_max=2.0, mag_adaptive_range="complete")
    _, params_max = normalize_mag_global(ddf, cfg_max, diag_ctx, log_fn)
    assert params_max.mag_min == 1.0
    assert params_max.mag_max == 2.0

    cfg_full = _cfg(mag_column="MAG", mag_adaptive_range="complete")
    _, params_full = normalize_mag_global(ddf, cfg_full, diag_ctx, log_fn)
    assert params_full.mag_min == 1.0
    assert params_full.mag_max == 3.0


def test_normalize_hist_peak_with_mag_max_only(diag_ctx, log_capture):
    """hist_peak path deriving mag_min from histogram when only mag_max is given."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [10.0, 11.0, 12.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak", mag_max=20.0)

    _, params = normalize_mag_global(ddf, cfg, diag_ctx, log_fn)
    assert params.mag_min < params.mag_max <= 20.0
    assert any("mag_max provided" in msg for msg in logs)


def test_normalize_hist_peak_with_explicit_bounds_skips_histogram(diag_ctx, log_capture):
    """hist_peak with both bounds bypasses histogram computation."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak", mag_min=0.0, mag_max=5.0)

    _, params = normalize_mag_global(ddf, cfg, diag_ctx, log_fn)
    assert params.mag_min == 0.0
    assert params.mag_max == 5.0
    assert any("explicit mag_min/mag_max" in msg for msg in logs)


def test_normalize_hist_peak_without_bounds(diag_ctx, log_capture):
    """hist_peak path with no bounds uses clipped min and histogram-derived max."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [-1.0, 0.0, 0.5]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_adaptive_range="hist_peak")

    _, params = normalize_mag_global(ddf, cfg, diag_ctx, log_fn)
    assert params.mag_min >= -2.0
    assert params.mag_min < params.mag_max
    assert any("no bounds provided" in msg for msg in logs)


def test_normalize_raises_when_resolved_bounds_inverted(diag_ctx, log_capture):
    """Explicit mag_min/mag_max that invert after validation raise at the final check."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"MAG": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_column="MAG", mag_min=5.0, mag_max=5.0, mag_adaptive_range="complete")

    with pytest.raises(ValueError):
        normalize_mag_global(ddf, cfg, diag_ctx, log_fn)


def test_prepare_passes_empty_partitions(diag_ctx, log_capture):
    """prepare_mag_global returns empty partitions untouched."""
    _, log_fn = log_capture
    empty = pd.DataFrame({"__mag__": []})
    ddf = dd.from_delayed(
        [
            dd.from_pandas(empty, npartitions=1).to_delayed()[0],
            dd.from_pandas(pd.DataFrame({"__mag__": [1.0, 2.0]}), npartitions=1).to_delayed()[0],
        ],
        meta={"__mag__": "f8"},
    )
    cfg = _cfg(mag_column="MAG")
    params = MagGlobalParams(mag_min=0.0, mag_max=2.0, sentinel=None)

    ddf_sel = prepare_mag_global(ddf, cfg, diag_ctx, log_fn, params)
    result = ddf_sel.compute()
    assert set(result["__mag__"].tolist()) == {1.0, 2.0}
