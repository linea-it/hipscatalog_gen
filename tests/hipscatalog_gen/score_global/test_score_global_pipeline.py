"""Unit tests for score_global pipeline steps."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd
from hipscatalog_gen.score_global.pipeline import (
    normalize_score_global,
    prepare_score_global,
    run_score_global_selection,
)


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
    """Creates a minimal config namespace for score_global tests."""
    algo_defaults = dict(
        score_column="SCORE",
        score_adaptive_range="complete",
        score_hist_nbins=4,
        score_min=None,
        score_max=None,
        score_keep_invalid_values=False,
        sg_order_desc=False,
        order_desc=False,
        level_limit=2,
        score_tie_column=None,
        tie_column=None,
        score_n_1=None,
        score_n_2=None,
        score_n_3=None,
        score_k_1=None,
        score_k_2=None,
        score_k_3=None,
    )
    algo_defaults.update(algo_kwargs)
    cluster_defaults = dict(low_memory_mode=True)
    return SimpleNamespace(
        algorithm=SimpleNamespace(**algo_defaults),
        cluster=SimpleNamespace(**cluster_defaults),
    )


def test_normalize_requires_score_column(diag_ctx, log_capture):
    """Missing score_column raises early."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"SCORE": [1.0]}), npartitions=1)
    cfg = _cfg(score_column=None)

    with pytest.raises(ValueError):
        normalize_score_global(ddf, cfg, diag_ctx, log_fn)


def test_normalize_validates_mode_and_hist(diag_ctx, log_capture):
    """Reject invalid adaptive mode or histogram bin counts."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"SCORE": [1.0]}), npartitions=1)
    with pytest.raises(ValueError):
        normalize_score_global(ddf, _cfg(score_adaptive_range="invalid"), diag_ctx, log_fn)
    with pytest.raises(ValueError):
        normalize_score_global(ddf, _cfg(score_hist_nbins=0), diag_ctx, log_fn)


def test_normalize_keep_invalid_constraints(diag_ctx, log_capture):
    """keep_invalid_values only allowed in complete mode and requires finite values."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"SCORE": [1.0]}), npartitions=1)
    with pytest.raises(ValueError):
        normalize_score_global(
            ddf,
            _cfg(score_keep_invalid_values=True, score_adaptive_range="hist_peak"),
            diag_ctx,
            log_fn,
        )

    ddf_all_nan = dd.from_pandas(pd.DataFrame({"SCORE": [float("nan"), float("inf")]}), npartitions=1)
    with pytest.raises(ValueError):
        normalize_score_global(
            ddf_all_nan,
            _cfg(score_keep_invalid_values=True),
            diag_ctx,
            log_fn,
        )


def test_normalize_keep_invalid_sets_sentinel(diag_ctx, log_capture):
    """keep_invalid_values path sets sentinel and expands range appropriately."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"SCORE": [1.0, 2.0, float("nan")]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(score_keep_invalid_values=True, order_desc=False)

    ddf_norm, params = normalize_score_global(ddf, cfg, diag_ctx, log_fn)
    result = ddf_norm.compute()

    assert params.sentinel is not None
    assert params.score_min == 1.0
    assert params.score_max >= params.sentinel > 2.0
    assert "__score__" in result.columns
    assert any("keep_invalid_values=True" in msg for msg in logs)

    # Descending order adjusts min instead.
    cfg_desc = _cfg(score_keep_invalid_values=True, sg_order_desc=True)
    _, params_desc = normalize_score_global(ddf, cfg_desc, diag_ctx, log_fn)
    assert params_desc.score_min <= params_desc.sentinel <= params_desc.score_max


def test_normalize_resolve_value_range_called(diag_ctx, log_capture, monkeypatch):
    """When not keeping invalids, resolve_value_range is invoked with expected args."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"SCORE": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(score_min=None, score_max=None, score_adaptive_range="complete")

    captured: dict[str, object] = {}

    def fake_resolve_value_range(**kwargs):
        captured.update(kwargs)
        return (0.5, 2.5)

    monkeypatch.setattr("hipscatalog_gen.score_global.pipeline.resolve_value_range", fake_resolve_value_range)

    _, params = normalize_score_global(ddf, cfg, diag_ctx, log_fn)
    assert params.score_min == 0.5 and params.score_max == 2.5
    assert captured.get("value_col") == "__score__"


def test_prepare_filters_and_persists(monkeypatch, diag_ctx, log_capture):
    """prepare_score_global filters by score window and optionally persists."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"__score__": [0.0, 1.0, 2.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg()
    params = SimpleNamespace(score_min=0.5, score_max=1.5, sentinel=None)

    captured = {}

    def fake_maybe_persist(ddf_like, should_persist, diag_ctx, log_fn, **kwargs):
        captured["should_persist"] = should_persist
        captured["diag_label"] = kwargs.get("diag_label")
        return ddf_like

    monkeypatch.setattr("hipscatalog_gen.score_global.pipeline.maybe_persist_ddf", fake_maybe_persist)

    ddf_sel = prepare_score_global(
        ddf,
        cfg,
        diag_ctx,
        log_fn,
        params,
        persist_ddfs=True,
        avoid_computes=False,
    )
    result = ddf_sel.compute()
    assert set(result["__score__"].tolist()) == {1.0}
    assert captured.get("should_persist") is True
    assert captured.get("diag_label") == "dask_score_persist_filtered"


def test_prepare_handles_empty_partition(diag_ctx, log_capture):
    """Empty partitions remain empty after filtering."""
    _, log_fn = log_capture
    empty = pd.DataFrame({"__score__": []})
    ddf = dd.from_pandas(empty, npartitions=1)
    cfg = _cfg()
    params = SimpleNamespace(score_min=0.0, score_max=1.0, sentinel=None)

    ddf_sel = prepare_score_global(ddf, cfg, diag_ctx, log_fn, params)
    assert ddf_sel.compute().empty


def test_run_selection_requires_params(diag_ctx, log_capture):
    """run_score_global_selection fails fast when params are missing."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"__score__": [1.0]}), npartitions=1)
    cfg = _cfg(level_limit=1)

    with pytest.raises(RuntimeError):
        run_score_global_selection(
            remainder_ddf=ddf,
            densmaps={1: np.ones(1, dtype="int64")},
            keep_cols=["__score__"],
            ra_col="RA",
            dec_col="DEC",
            cfg=cfg,
            out_dir="/tmp/should_not_write",
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            params=None,
        )


def test_run_selection_converts_k_and_forwards(monkeypatch, diag_ctx, log_capture):
    """k_* targets are converted to fixed counts and forwarded to select_by_score_slices."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"__score__": [1.0]}), npartitions=1)
    densmaps = {1: np.array([1, 1], dtype="int64")}
    cfg = _cfg(level_limit=1, score_k_1=0.5, score_tie_column="SNR", sg_order_desc=True)
    params = SimpleNamespace(score_min=0.0, score_max=2.0, sentinel=None)

    captured: dict[str, object] = {}

    def fake_select_by_score_slices(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(
        "hipscatalog_gen.score_global.pipeline.select_by_score_slices",
        fake_select_by_score_slices,
    )

    run_score_global_selection(
        remainder_ddf=ddf,
        densmaps=densmaps,
        keep_cols=["__score__"],
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir="/tmp",
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        params=params,
    )

    assert captured.get("fixed_targets") == {1: 1.0}
    assert captured.get("tie_col") == "SNR"
    assert captured.get("order_desc") is True


def test_run_selection_rejects_both_n_and_k(monkeypatch, diag_ctx, log_capture):
    """Setting both n_* and k_* raises ValueError."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"__score__": [1.0]}), npartitions=1)
    densmaps = {1: np.array([1], dtype="int64")}
    cfg = _cfg(level_limit=1, score_n_1=1, score_k_1=0.5)
    params = SimpleNamespace(score_min=0.0, score_max=1.0, sentinel=None)

    with pytest.raises(ValueError):
        run_score_global_selection(
            remainder_ddf=ddf,
            densmaps=densmaps,
            keep_cols=["__score__"],
            ra_col="RA",
            dec_col="DEC",
            cfg=cfg,
            out_dir="/tmp",
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            params=params,
        )
