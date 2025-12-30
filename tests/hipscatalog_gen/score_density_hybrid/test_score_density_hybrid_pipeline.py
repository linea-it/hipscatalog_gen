"""Tests for the score_density_hybrid pipeline (normalize, prepare, run)."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import healpy as hp
import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd
from hipscatalog_gen.score_density_hybrid import pipeline as sdh_pipeline


@pytest.fixture
def diag_ctx():
    """Returns a no-op diagnostic context."""
    return lambda name: nullcontext()


@pytest.fixture
def log_capture():
    """Collects log messages emitted by the pipeline."""
    logs: list[str] = []

    def _log_fn(msg: str, always: bool = False, **_: dict) -> None:
        """Capture log message into the local list."""
        logs.append(msg)

    return logs, _log_fn


def _cfg(**overrides) -> SimpleNamespace:
    """Build a minimal config namespace for score_density_hybrid."""
    algo_defaults = dict(
        sdh_score_column="SCORE",
        sdh_score_adaptive_range="complete",
        sdh_score_hist_nbins=4,
        sdh_score_min=None,
        sdh_score_max=None,
        sdh_keep_invalid_values=False,
        sdh_order_desc=False,
        order_desc=False,
        level_limit=3,
        sdh_tie_column=None,
        tie_column=None,
        sdh_n_1=None,
        sdh_n_2=None,
        sdh_n_3=None,
        sdh_k_1=None,
        sdh_k_2=None,
        sdh_k_3=None,
        sdh_density_bias_n1=0.0,
        sdh_density_bias_n2=0.0,
        sdh_density_bias_n3=0.0,
    )
    algo_defaults.update(overrides)
    cluster_defaults = dict(low_memory_mode=True)
    return SimpleNamespace(
        algorithm=SimpleNamespace(**algo_defaults),
        cluster=SimpleNamespace(**cluster_defaults),
    )


def test_normalize_requires_score_column(diag_ctx, log_capture):
    """Missing score column raises."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"SCORE": [1.0]}), npartitions=1)
    with pytest.raises(ValueError):
        sdh_pipeline.normalize_score_density_hybrid(ddf, _cfg(sdh_score_column=None), diag_ctx, log_fn)


def test_normalize_validates_mode_and_hist(diag_ctx, log_capture):
    """Invalid adaptive mode or histogram bins rejected."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"SCORE": [1.0]}), npartitions=1)
    with pytest.raises(ValueError):
        sdh_pipeline.normalize_score_density_hybrid(
            ddf, _cfg(sdh_score_adaptive_range="bad"), diag_ctx, log_fn
        )
    with pytest.raises(ValueError):
        sdh_pipeline.normalize_score_density_hybrid(ddf, _cfg(sdh_score_hist_nbins=0), diag_ctx, log_fn)


def test_normalize_keep_invalid_path(diag_ctx, log_capture):
    """keep_invalid_values maps NaN/Inf to sentinel and adjusts range."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"SCORE": [1.0, 2.0, float("nan")]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(sdh_keep_invalid_values=True, sdh_order_desc=False)

    ddf_norm, params = sdh_pipeline.normalize_score_density_hybrid(ddf, cfg, diag_ctx, log_fn)
    result = ddf_norm.compute()

    assert params.sentinel is not None
    assert params.score_min == 1.0
    assert params.score_max >= params.sentinel > 2.0
    assert "__score__" in result.columns
    assert any("keep_invalid_values=True" in msg for msg in logs)

    cfg_desc = _cfg(sdh_keep_invalid_values=True, sdh_order_desc=True)
    _, params_desc = sdh_pipeline.normalize_score_density_hybrid(ddf, cfg_desc, diag_ctx, log_fn)
    assert params_desc.score_min <= params_desc.sentinel <= params_desc.score_max


def test_normalize_keep_invalid_constraints(diag_ctx, log_capture):
    """keep_invalid_values with hist_peak or all invalid raises."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"SCORE": [1.0]}), npartitions=1)
    with pytest.raises(ValueError):
        sdh_pipeline.normalize_score_density_hybrid(
            ddf,
            _cfg(sdh_keep_invalid_values=True, sdh_score_adaptive_range="hist_peak"),
            diag_ctx,
            log_fn,
        )

    ddf_all_nan = dd.from_pandas(pd.DataFrame({"SCORE": [float("nan"), float("inf")]}), npartitions=1)
    with pytest.raises(ValueError):
        sdh_pipeline.normalize_score_density_hybrid(
            ddf_all_nan,
            _cfg(sdh_keep_invalid_values=True),
            diag_ctx,
            log_fn,
        )


def test_normalize_resolve_value_range_called(diag_ctx, log_capture, monkeypatch):
    """resolve_value_range is invoked when not keeping invalids."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"SCORE": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(sdh_score_adaptive_range="complete")

    captured: dict[str, object] = {}

    def fake_resolve_value_range(**kwargs):
        captured.update(kwargs)
        return (0.5, 2.5)

    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.resolve_value_range", fake_resolve_value_range
    )

    _, params = sdh_pipeline.normalize_score_density_hybrid(ddf, cfg, diag_ctx, log_fn)
    assert (params.score_min, params.score_max) == (0.5, 2.5)
    assert captured.get("value_col") == "__score__"


def test_prepare_filters_and_attaches_ids(monkeypatch, diag_ctx, log_capture):
    """prepare_score_density_hybrid filters by score and attaches unique ids."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"__score__": [0.0, 1.0, 2.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg()
    params = SimpleNamespace(score_min=0.5, score_max=1.5, sentinel=None)

    captured = {}

    def fake_maybe_persist(ddf_like, should_persist, diag_ctx, log_fn, **kwargs):
        captured["persist"] = should_persist
        captured["diag_label"] = kwargs.get("diag_label")
        return ddf_like

    monkeypatch.setattr("hipscatalog_gen.score_density_hybrid.pipeline.maybe_persist_ddf", fake_maybe_persist)

    ddf_sel = sdh_pipeline.prepare_score_density_hybrid(
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
    assert "__sdh_id__" in result.columns
    assert captured["persist"] is True
    assert captured["diag_label"] == "dask_sdh_persist_filtered"


def test_attach_unique_id_and_drop_selected_ids():
    """Direct helper coverage for id attachment and dropping selected ids."""
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    with_ids = sdh_pipeline._attach_unique_id(pdf, partition_info={"number": 2})
    assert "__sdh_id__" in with_ids.columns
    assert with_ids["__sdh_id__"].dtype == np.int64
    assert (with_ids["__sdh_id__"].iloc[0] >> 32) == 2

    empty_ids = sdh_pipeline._attach_unique_id(pd.DataFrame({"a": []}))
    assert "__sdh_id__" in empty_ids.columns
    assert empty_ids.empty

    kept = sdh_pipeline._drop_selected_ids(with_ids, ids=with_ids["__sdh_id__"].iloc[:1])
    assert len(kept) == 2
    assert sdh_pipeline._drop_selected_ids(with_ids, ids=[]).equals(with_ids)
    assert sdh_pipeline._drop_selected_ids(pd.DataFrame({"__sdh_id__": []}), ids=[1]).empty


def test_filter_score_window_empty_partition():
    """Direct coverage for _filter_score_window with empty input."""
    empty = pd.DataFrame({"__score__": []})
    assert sdh_pipeline._filter_score_window(empty, 0.0, 1.0).empty


def test_distribute_by_weights_edge_and_remainder():
    """_distribute_by_weights covers zero/empty and remainder distribution."""
    assert sdh_pipeline._distribute_by_weights(0, {1: 1}) == {1: 0}
    assert sdh_pipeline._distribute_by_weights(5, {}) == {}
    assert sdh_pipeline._distribute_by_weights(3, {1: 0, 2: 0}) == {1: 0, 2: 0}
    dist_remainder = sdh_pipeline._distribute_by_weights(1, {1: 1, 2: 1})
    assert sum(dist_remainder.values()) == 1

    weights = {1: 1, 2: 2}
    dist = sdh_pipeline._distribute_by_weights(3, weights)
    assert sum(dist.values()) == 3
    assert set(dist.keys()) == {1, 2}


def test_targets_stage1_by_depth_branches(log_capture):
    """_targets_stage1_by_depth handles provided counts, clamping, and redistribution."""
    logs, log_fn = log_capture
    densmaps = {1: np.array([2, 0], dtype="int64"), 2: np.array([1, 1], dtype="int64")}

    # No base targets
    assert sdh_pipeline._targets_stage1_by_depth(densmaps, {}, 10, {}, log_fn) == {}

    base_targets = {1: 10, 2: 5}
    provided = {1: 1}
    totals = sdh_pipeline._targets_stage1_by_depth(densmaps, base_targets, 12, provided, log_fn)
    assert totals[1] >= 0 and totals[2] >= 0
    assert sum(totals.values()) == min(12, 12)  # total_target capped by n_tot_score
    assert any("clamping" in msg.lower() for msg in logs)


def test_run_selection_requires_params(diag_ctx, log_capture):
    """run_score_density_hybrid_selection without params raises RuntimeError."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"__score__": [1.0]}), npartitions=1)
    cfg = _cfg(level_limit=1)

    with pytest.raises(RuntimeError):
        sdh_pipeline.run_score_density_hybrid_selection(
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


def test_run_selection_rejects_both_n_and_k(monkeypatch, diag_ctx, log_capture):
    """Setting both n_* and k_* for a depth raises ValueError."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"__score__": [1.0]}), npartitions=1)
    cfg = _cfg(level_limit=1, sdh_n_1=1, sdh_k_1=0.5)
    params = SimpleNamespace(score_min=0.0, score_max=1.0, sentinel=None)

    with pytest.raises(ValueError):
        sdh_pipeline.run_score_density_hybrid_selection(
            remainder_ddf=ddf,
            densmaps={1: np.ones(1, dtype="int64")},
            keep_cols=["__score__"],
            ra_col="RA",
            dec_col="DEC",
            cfg=cfg,
            out_dir="/tmp",
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            params=params,
        )


def test_run_selection_histogram_empty(monkeypatch, diag_ctx, log_capture):
    """Histogram with zero total exits early."""
    logs, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"__score__": [1.0]}), npartitions=1)
    cfg = _cfg(level_limit=1)
    params = SimpleNamespace(score_min=0.0, score_max=1.0, sentinel=None)

    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.compute_score_histogram_ddf",
        lambda *_, **__: (np.zeros(1, dtype="int64"), np.array([0.0, 1.0]), 0),
    )

    sdh_pipeline.run_score_density_hybrid_selection(
        remainder_ddf=ddf,
        densmaps={1: np.ones(1, dtype="int64")},
        keep_cols=["__score__"],
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir="/tmp",
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        params=params,
    )

    assert any("nothing to select" in msg for msg in logs)


def test_run_selection_targets_but_no_candidates(monkeypatch, diag_ctx, log_capture):
    """Targets exist but targets_per_tile_map empty -> depth skip path."""
    logs, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"__score__": [1.0], "__sdh_id__": [1]}), npartitions=1)
    cfg = _cfg(level_limit=1)
    params = SimpleNamespace(score_min=0.0, score_max=1.0, sentinel=None)

    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.compute_score_histogram_ddf",
        lambda *_, **__: (np.array([1], dtype="int64"), np.array([0.0, 1.0]), 1),
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.assign_level_edges",
        lambda **kwargs: (np.array([0.0, 1.0]), np.array([1.0])),
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline._targets_stage1_by_depth",
        lambda **kwargs: {1: 1},
    )
    densmaps = {1: np.zeros(hp.nside2npix(2), dtype="int64")}  # no active tiles

    sdh_pipeline.run_score_density_hybrid_selection(
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

    assert any("no active tiles or zero targets" in msg for msg in logs)


def test_run_selection_depth_with_empty_selection(monkeypatch, diag_ctx, log_capture):
    """Depth has targets but reduce_topk yields empty selection -> skip writing."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "__score__": [0.5], "__sdh_id__": [1]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(level_limit=1)
    params = SimpleNamespace(score_min=0.0, score_max=1.0, sentinel=None)

    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.compute_score_histogram_ddf",
        lambda *_, **__: (np.array([1], dtype="int64"), np.array([0.0, 1.0]), 1),
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.assign_level_edges",
        lambda **kwargs: (np.array([0.0, 1.0]), np.array([1.0])),
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline._targets_stage1_by_depth",
        lambda **kwargs: {1: 1},
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.targets_per_tile",
        lambda counts, depth_total, bias: {0: depth_total},
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.add_ipix_column",
        lambda pdf, depth, ra_col, dec_col: pdf.assign(__ipix__=1),
    )
    # reduce_topk returns empty
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.reduce_topk_by_group_dask",
        lambda cand_ddf, **kwargs: cand_ddf.map_partitions(lambda pdf: pdf.iloc[0:0], meta=cand_ddf._meta),
    )

    sdh_pipeline.run_score_density_hybrid_selection(
        remainder_ddf=ddf,
        densmaps={1: np.ones(hp.nside2npix(2), dtype="int64")},
        keep_cols=["RA", "DEC", "__score__"],
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir="/tmp",
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        params=params,
    )

    assert any("no rows selected" in msg for msg in logs)


def test_run_selection_cdf_zero(monkeypatch, diag_ctx, log_capture):
    """cdf_hist zero branch is taken when histogram has zero counts but n_tot_score>0."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"__score__": [1.0]}), npartitions=1)
    cfg = _cfg(level_limit=1)
    params = SimpleNamespace(score_min=0.0, score_max=1.0, sentinel=None)

    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.compute_score_histogram_ddf",
        lambda *_, **__: (np.zeros(2, dtype="int64"), np.array([0.0, 0.5, 1.0]), 1),
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.assign_level_edges",
        lambda **kwargs: (np.array([0.0, 1.0]), np.array([0.0, 0.0])),
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline._targets_stage1_by_depth",
        lambda **kwargs: {1: 0},
    )

    sdh_pipeline.run_score_density_hybrid_selection(
        remainder_ddf=ddf,
        densmaps={1: np.ones(1, dtype="int64")},
        keep_cols=["__score__"],
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir="/tmp",
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        params=params,
    )


def test_run_selection_stage1_and_stage2(monkeypatch, diag_ctx, log_capture):
    """Stage1 selection writes tiles and Stage2 delegates to score slicing for deeper depths."""
    _, log_fn = log_capture
    pdf = pd.DataFrame(
        {
            "RA": [0.0, 10.0, 20.0],
            "DEC": [0.0, 5.0, -5.0],
            "__score__": [1.0, 2.0, 3.0],
            "__sdh_id__": [10, 11, 12],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(
        level_limit=4, sdh_k_1=0.5, sdh_order_desc=True, sdh_tie_column="TIE", sdh_n_2=None, sdh_k_2=None
    )
    densmaps = {
        1: np.ones(hp.nside2npix(2), dtype="int64"),
        2: np.ones(hp.nside2npix(4), dtype="int64"),
        3: np.ones(hp.nside2npix(8), dtype="int64"),
        4: np.ones(hp.nside2npix(16), dtype="int64"),
    }
    params = SimpleNamespace(score_min=0.0, score_max=3.0, sentinel=None)

    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.compute_score_histogram_ddf",
        lambda *_, **__: (np.array([1], dtype="int64"), np.array([0.0, 1.0]), 3),
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.assign_level_edges",
        lambda **kwargs: (np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1.0, 0.0, 0.0, 0.0])),
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline._targets_stage1_by_depth",
        lambda **kwargs: {1: 2, 2: 0, 3: 0},
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.targets_per_tile",
        lambda counts, depth_total, bias: {0: depth_total},
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.add_ipix_column",
        lambda pdf, depth, ra_col, dec_col: pdf.assign(__ipix__=0),
    )
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.reduce_topk_by_group_dask",
        lambda cand_ddf, **_: cand_ddf,
    )

    captured_writes: list[dict] = []
    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.write_tiles_with_allsky",
        lambda **kwargs: (captured_writes.append(kwargs) or ({0: len(kwargs["selected"])}, None)),
    )

    captured_stage2: list[dict] = []

    def fake_select_by_score_slices(**kwargs):
        captured_stage2.append(kwargs)
        return None

    monkeypatch.setattr(
        "hipscatalog_gen.score_density_hybrid.pipeline.select_by_score_slices", fake_select_by_score_slices
    )

    sdh_pipeline.run_score_density_hybrid_selection(
        remainder_ddf=ddf,
        densmaps=densmaps,
        keep_cols=["RA", "DEC", "__score__"],
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir="/tmp",
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        params=params,
    )

    assert captured_writes  # stage1 wrote something
    assert captured_writes[0]["order_desc"] is True
    assert captured_stage2 and captured_stage2[0]["depths_sel"] == [4]
