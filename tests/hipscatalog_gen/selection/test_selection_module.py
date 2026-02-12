"""Unit tests for selection helpers (histograms, ranges, slicing)."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import healpy as hp
import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd

from hipscatalog_gen.selection import common as selection_common
from hipscatalog_gen.selection import levels as selection_levels
from hipscatalog_gen.selection import score as selection_score
from hipscatalog_gen.selection import slicing as selection_slicing


@pytest.fixture
def diag_ctx():
    """Returns a no-op diagnostic context."""
    return lambda name: nullcontext()


@pytest.fixture
def log_capture():
    """Collects log messages emitted by helpers."""
    logs: list[str] = []

    def _log_fn(msg: str, always: bool = False, **_: Any) -> None:
        """Capture a log message into the local list."""
        logs.append(msg)

    return logs, _log_fn


def test_compute_histogram_ddf_handles_invalids():
    """Histogram helper drops invalids unless keep_invalid=True."""
    pdf = pd.DataFrame({"VAL": [0.0, 1.0, 2.0, float("nan"), float("inf")]})
    ddf = dd.from_pandas(pdf, npartitions=2)

    hist, edges, n_total = selection_score.compute_histogram_ddf(ddf, "VAL", 0.0, 2.0, 2)
    assert n_total == 5
    assert hist.tolist() == [1, 2]  # NaN/Inf dropped
    assert edges.tolist() == [0.0, 1.0, 2.0]

    hist_keep, _, _ = selection_score.compute_histogram_ddf(
        ddf,
        "VAL",
        0.0,
        2.0,
        2,
        keep_invalid=True,
        sentinel=0.0,
    )
    assert hist_keep.tolist() == [3, 2]  # NaN/Inf mapped to sentinel inside first bin


def test_finite_min_max_scans_only_finite_values():
    """_finite_min_max ignores NaN/Inf and returns None when nothing finite."""
    pdf_all_invalid = pd.DataFrame({"X": [float("nan"), float("inf"), -float("inf")]})
    ddf_all_invalid = dd.from_pandas(pdf_all_invalid, npartitions=1)
    assert selection_score._finite_min_max(ddf_all_invalid, "X") == (None, None)

    pdf_mixed = pd.DataFrame({"X": [float("nan"), -10.0, 5.0, float("inf")]})
    ddf_mixed = dd.from_pandas(pdf_mixed, npartitions=2)
    assert selection_score._finite_min_max(ddf_mixed, "X") == (-10.0, 5.0)


def test_sentinel_for_order_brackets_range():
    """Sentinel falls strictly outside the finite range respecting ordering."""
    sentinel_asc = selection_score._sentinel_for_order(1.2, 3.8, order_desc=False)
    sentinel_desc = selection_score._sentinel_for_order(1.2, 3.8, order_desc=True)
    sentinel_desc_int = selection_score._sentinel_for_order(1.0, 3.0, order_desc=True)
    sentinel_asc_int = selection_score._sentinel_for_order(1.0, 3.0, order_desc=False)

    assert sentinel_asc > 3.8
    assert sentinel_desc < 1.2
    assert sentinel_desc_int == 0.0  # floor equals min triggers decrement branch
    assert sentinel_asc_int == 4.0  # ceil equals max triggers increment branch


def test_map_invalid_to_sentinel_applies_extra_mask():
    """_map_invalid_to_sentinel replaces NaN/Inf and custom-masked values."""
    pdf = pd.DataFrame({"v": [1.0, float("nan"), float("inf"), -5.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)

    mapped = selection_score._map_invalid_to_sentinel(
        ddf,
        col="v",
        sentinel=0.0,
        extra_mask_fn=lambda vals: vals < 0,
    ).compute()

    assert mapped["v"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert mapped["v"].dtype == "float64"


def test_map_invalid_to_sentinel_handles_empty_partitions():
    """Empty partitions get the sentinel column with float dtype."""
    empty_ddf = dd.from_pandas(pd.DataFrame({"v": []}), npartitions=1)
    mapped = selection_score._map_invalid_to_sentinel(empty_ddf, col="v", sentinel=-1.0).compute()
    assert mapped["v"].dtype == "float64"
    assert mapped.empty


def test_compute_histogram_with_forced_empty_values(monkeypatch):
    """Force to_numeric to produce empty arrays to cover vals.size==0 branch."""
    pdf = pd.DataFrame({"VAL": [1.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    orig_to_numeric = selection_score.pd.to_numeric

    def fake_to_numeric(*args, **kwargs):
        return pd.Series([], dtype="float64")

    monkeypatch.setattr(selection_score.pd, "to_numeric", fake_to_numeric)
    hist, _, n_total = selection_score.compute_histogram_ddf(ddf, "VAL", 0.0, 1.0, 1)
    assert hist.tolist() == [0]
    assert n_total == 0

    monkeypatch.setattr(selection_score.pd, "to_numeric", orig_to_numeric)


def test_compute_histogram_handles_empty_partition():
    """compute_histogram_ddf tolerates empty partitions and wrapper forwards args."""
    empty = dd.from_pandas(pd.DataFrame({"VAL": []}), npartitions=1)
    hist, edges, n_total = selection_score.compute_histogram_ddf(empty, "VAL", 0.0, 1.0, 2)
    assert hist.tolist() == [0, 0]
    assert edges.tolist() == [0.0, 0.5, 1.0]
    assert n_total == 0

    # Wrapper covers compute_score_histogram_ddf path.
    hist2, edges2, n_total2 = selection_score.compute_score_histogram_ddf(empty, "VAL", 0.0, 1.0, 2)
    assert hist2.tolist() == [0, 0]
    assert edges2.tolist() == edges.tolist()
    assert n_total2 == n_total


def test_quantile_from_histogram_corners_and_plateau():
    """Quantile helper covers empty CDF, bounds clamping, and flat plateau."""
    edges = np.array([0.0, 1.0, 2.0], dtype="float64")
    assert selection_score._quantile_from_histogram(np.array([], dtype="float64"), edges, 0.5) == 0.0
    assert selection_score._quantile_from_histogram(np.array([0.2, 0.8]), edges, -1.0) == 0.0
    assert selection_score._quantile_from_histogram(np.array([0.2, 0.8]), edges, 2.0) == 2.0

    # Plateau at the end exercises the while-loop and fallback to the last edge.
    cdf_plateau = np.array([0.1, 0.1, 0.1], dtype="float64")
    edges_plateau = np.array([0.0, 1.0, 2.0, 3.0], dtype="float64")
    assert selection_score._quantile_from_histogram(cdf_plateau, edges_plateau, 0.15) == 3.0


def test_quantile_from_histogram_non_monotonic(monkeypatch):
    """Non-monotonic CDF still progresses through the while-loop branch."""
    cdf = np.array([0.5, 0.5, 0.4], dtype="float64")
    edges = np.array([0.0, 1.0, 2.0, 3.0], dtype="float64")
    monkeypatch.setattr(selection_score.np, "searchsorted", lambda *_args, **_kwargs: 1)
    assert selection_score._quantile_from_histogram(cdf, edges, 0.3) == 3.0

    cdf2 = np.array([0.5, 0.5, 0.6], dtype="float64")
    edges2 = np.array([0.0, 1.0, 2.0, 3.0], dtype="float64")
    assert selection_score._quantile_from_histogram(cdf2, edges2, 0.3) > 0.0


def test_add_score_column_supports_expressions_with_empty_partitions():
    """add_score_column handles expressions and preserves empty partitions."""
    empty = dd.from_pandas(pd.DataFrame({"A": [], "B": []}), npartitions=1)
    non_empty = dd.from_pandas(pd.DataFrame({"A": [1.0, 2.0], "B": [10.0, 20.0]}), npartitions=1)
    ddf = dd.from_delayed(empty.to_delayed() + non_empty.to_delayed(), meta={"A": "f8", "B": "f8"})

    scored = selection_score.add_score_column(ddf, "A + B", output_col="__score__").compute()

    non_empty_scores = scored.loc[scored["A"].notna(), "__score__"].tolist()
    assert non_empty_scores == [11.0, 22.0]
    assert "__score__" in scored.columns


def test_add_score_column_from_existing_column():
    """Column path uses existing column without evaluating expression."""
    pdf = pd.DataFrame({"SCORE": [1.0, np.inf, -np.inf]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    scored = selection_score.add_score_column(ddf, "SCORE", output_col="__score__").compute()
    assert scored["__score__"].isna().sum() == 2  # inf mapped to nan


def test_resolve_value_range_branches(diag_ctx, log_capture):
    """resolve_value_range covers complete and hist_peak paths."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"SCORE": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=2)

    # complete mode uses global min/max without calling histogram
    vmin, vmax = selection_score.resolve_value_range(
        ddf,
        value_col="SCORE",
        range_mode="complete",
        min_cfg=None,
        max_cfg=None,
        hist_nbins=4,
        compute_hist_fn=lambda *_args, **_kwargs: (_args, _kwargs),  # should not be invoked
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        label="score",
    )
    assert (vmin, vmax) == (1.0, 3.0)
    assert any("global range" in msg for msg in logs)

    # hist_peak derives the missing bound from a provided histogram
    def fake_hist_fn(ddf_like, value_col, lo, hi, nbins):
        assert value_col == "SCORE"
        return np.array([2, 1], dtype="int64"), np.array([1.0, 2.0, 3.0], dtype="float64"), 3

    vmin_hp, vmax_hp = selection_score.resolve_value_range(
        ddf,
        value_col="SCORE",
        range_mode="hist_peak",
        min_cfg=None,
        max_cfg=None,
        hist_nbins=2,
        compute_hist_fn=fake_hist_fn,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        label="score",
    )
    assert vmin_hp == 1.0
    assert vmax_hp == 1.5  # peak center of the first bin
    assert any("histogram peak" in msg for msg in logs)

    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf,
            value_col="SCORE",
            range_mode="complete",
            min_cfg=None,
            max_cfg=None,
            hist_nbins=0,
            compute_hist_fn=fake_hist_fn,
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )


def test_resolve_value_range_hist_peak_branches(diag_ctx, log_capture):
    """Exercise hist_peak with only min, only max, invalid mode, and empty histogram."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"SCORE": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    def fake_hist_fn(ddf_like, value_col, lo, hi, nbins):
        return np.array([1, 2], dtype="int64"), np.array([0.0, 1.0, 2.0], dtype="float64"), 3

    # Only max provided
    vmin, vmax = selection_score.resolve_value_range(
        ddf,
        value_col="SCORE",
        range_mode="hist_peak",
        min_cfg=None,
        max_cfg=5.0,
        hist_nbins=2,
        compute_hist_fn=fake_hist_fn,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        label="score",
    )
    assert vmin < vmax <= 5.0

    # Only min provided
    vmin2, vmax2 = selection_score.resolve_value_range(
        ddf,
        value_col="SCORE",
        range_mode="hist_peak",
        min_cfg=0.0,
        max_cfg=None,
        hist_nbins=2,
        compute_hist_fn=fake_hist_fn,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        label="score",
    )
    assert vmax2 == 1.5  # histogram peak used as max
    assert vmin2 < vmax2

    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf,
            value_col="SCORE",
            range_mode="invalid",
            min_cfg=None,
            max_cfg=None,
            hist_nbins=1,
            compute_hist_fn=fake_hist_fn,
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )

    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf,
            value_col="SCORE",
            range_mode="hist_peak",
            min_cfg=None,
            max_cfg=None,
            hist_nbins=1,
            compute_hist_fn=lambda *_: (np.zeros(2, dtype="int64"), np.array([0.0, 1.0, 2.0]), 0),
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )

    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            dd.from_pandas(pd.DataFrame({"SCORE": [5.0, 5.0]}), npartitions=1),
            value_col="SCORE",
            range_mode="complete",
            min_cfg=None,
            max_cfg=None,
            hist_nbins=1,
            compute_hist_fn=fake_hist_fn,
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )


def test_resolve_value_range_final_validation(diag_ctx, log_capture):
    """Non-finite global range and inverted resolved range raise errors."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"SCORE": [float("nan"), float("inf")]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf,
            value_col="SCORE",
            range_mode="complete",
            min_cfg=None,
            max_cfg=None,
            hist_nbins=1,
            compute_hist_fn=lambda *_: (None, None, None),
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )

    pdf2 = pd.DataFrame({"SCORE": [1.0, 2.0]})
    ddf2 = dd.from_pandas(pdf2, npartitions=1)
    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf2,
            value_col="SCORE",
            range_mode="complete",
            min_cfg=5.0,
            max_cfg=4.0,
            hist_nbins=1,
            compute_hist_fn=lambda *_: (None, None, None),
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )


def test_resolve_value_range_complete_single_bounds(diag_ctx, log_capture):
    """complete mode fills the missing bound while keeping the provided one."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"SCORE": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    vmin, vmax = selection_score.resolve_value_range(
        ddf,
        value_col="SCORE",
        range_mode="complete",
        min_cfg=None,
        max_cfg=2.5,
        hist_nbins=1,
        compute_hist_fn=lambda *_: (_ for _ in ()),  # should not be invoked
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        label="score",
    )
    assert vmin == 1.0 and vmax == 2.5
    assert any("min not provided" in msg for msg in logs)

    vmin2, vmax2 = selection_score.resolve_value_range(
        ddf,
        value_col="SCORE",
        range_mode="complete",
        min_cfg=0.0,
        max_cfg=None,
        hist_nbins=1,
        compute_hist_fn=lambda *_: (_ for _ in ()),
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        label="score",
    )
    assert vmin2 == 0.0 and vmax2 == 3.0
    assert any("max not provided" in msg for msg in logs)


def test_resolve_value_range_invalids(monkeypatch, diag_ctx, log_capture):
    """Guard rails: invalid globals, empty histogram, and non-finite peak."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"SCORE": [5.0, 5.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf,
            value_col="SCORE",
            range_mode="complete",
            min_cfg=None,
            max_cfg=None,
            hist_nbins=1,
            compute_hist_fn=lambda *_: (_ for _ in ()),
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )

    monkeypatch.setattr(selection_score, "dask_compute", lambda *_, **__: (None, None))
    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf,
            value_col="SCORE",
            range_mode="complete",
            min_cfg=None,
            max_cfg=None,
            hist_nbins=1,
            compute_hist_fn=lambda *_: (_ for _ in ()),
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )

    def hist_nan(*_args, **_kwargs):
        return np.array([1], dtype="int64"), np.array([np.nan, np.nan], dtype="float64"), 1

    assert hist_nan() is not None

    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            dd.from_pandas(pd.DataFrame({"SCORE": [1.0]}), npartitions=1),
            value_col="SCORE",
            range_mode="hist_peak",
            min_cfg=None,
            max_cfg=None,
            hist_nbins=1,
            compute_hist_fn=hist_nan,
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )

    def hist_empty(*_args, **_kwargs):
        return np.zeros(2, dtype="int64"), np.array([0.0, 1.0, 2.0], dtype="float64"), 0

    assert hist_empty() is not None

    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf,
            value_col="SCORE",
            range_mode="hist_peak",
            min_cfg=None,
            max_cfg=2.0,
            hist_nbins=2,
            compute_hist_fn=hist_empty,
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )

    def hist_inverted(*_args, **_kwargs):
        return np.array([1], dtype="int64"), np.array([0.0, 1.0], dtype="float64"), 1

    assert hist_inverted() is not None

    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf,
            value_col="SCORE",
            range_mode="hist_peak",
            min_cfg=5.0,
            max_cfg=4.0,
            hist_nbins=1,
            compute_hist_fn=hist_inverted,
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )


def test_resolve_value_range_hist_peak_errors(diag_ctx, log_capture):
    """hist_peak raises when histogram peak conflicts with provided bounds or is non-finite."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"SCORE": [1.0, 2.0, 3.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    def hist_peak_high(*_args, **_kwargs):
        return np.array([0, 1], dtype="int64"), np.array([0.0, 1.0, 10.0], dtype="float64"), 3

    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf,
            value_col="SCORE",
            range_mode="hist_peak",
            min_cfg=None,
            max_cfg=1.0,
            hist_nbins=2,
            compute_hist_fn=hist_peak_high,
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )

    def hist_peak_low(*_args, **_kwargs):
        return np.array([1], dtype="int64"), np.array([0.0, 1.0], dtype="float64"), 1

    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf,
            value_col="SCORE",
            range_mode="hist_peak",
            min_cfg=5.0,
            max_cfg=None,
            hist_nbins=1,
            compute_hist_fn=hist_peak_low,
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )

    def hist_nan_peak(*_args, **_kwargs):
        return np.array([1], dtype="int64"), np.array([np.nan, np.nan], dtype="float64"), 1

    with pytest.raises(ValueError):
        selection_score.resolve_value_range(
            ddf,
            value_col="SCORE",
            range_mode="hist_peak",
            min_cfg=None,
            max_cfg=None,
            hist_nbins=1,
            compute_hist_fn=hist_nan_peak,
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            label="score",
        )


def test_targets_per_tile_distributes_across_active_pixels():
    """targets_per_tile returns a per-ipix mapping summing to depth_total."""
    counts = np.array([10, 0, 20], dtype="int64")
    targets = selection_common.targets_per_tile(counts, depth_total=6, bias=0.5)

    assert set(targets.keys()) == {0, 2}
    assert sum(targets.values()) == 6


def test_targets_per_tile_edge_cases():
    """Edge cases for targets_per_tile: zero totals and no active pixels."""
    assert selection_common.targets_per_tile(np.array([0, 0], dtype="int64"), depth_total=0, bias=0.5) == {}
    assert selection_common.targets_per_tile(np.array([0, 0], dtype="int64"), depth_total=5, bias=0.5) == {}


def test_reduce_topk_by_group_dask_orders_with_tiebreakers():
    """reduce_topk_by_group_dask keeps top scores per group respecting tie_col."""
    pdf = pd.DataFrame(
        {
            "group": [1, 1, 2, 2],
            "score": [1.0, 0.5, 0.9, 0.9],
            "RA": [1.0, 0.0, 2.0, 1.0],
            "DEC": [0.0, 0.0, 0.0, 0.0],
            "TIE": [2.0, 1.0, 1.0, 2.0],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=2)

    result = selection_common.reduce_topk_by_group_dask(
        ddf_like=ddf,
        group_col="group",
        score_col="score",
        order_desc=True,
        k_per_group={1: 1, 2: 1},
        ra_col="RA",
        dec_col="DEC",
        tie_col="TIE",
    ).compute()

    # group 1 -> highest score (tie_col not used), group 2 -> tie broken by TIE ascending
    assert set(result["group"]) == {1, 2}
    assert result.loc[result["group"] == 1, "score"].iloc[0] == 1.0
    assert result.loc[result["group"] == 2, "RA"].iloc[0] == 2.0


def test_reduce_topk_by_group_dask_handles_empty_and_missing(monkeypatch):
    """reduce_topk_by_group_dask empty config, empty groups, and missing k mapping."""
    pdf = pd.DataFrame({"group": [1], "score": [1.0], "RA": [0.0], "DEC": [0.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    empty_result = selection_common.reduce_topk_by_group_dask(
        ddf_like=ddf,
        group_col="group",
        score_col="score",
        order_desc=True,
        k_per_group={},
        ra_col="RA",
        dec_col="DEC",
    ).compute()
    assert empty_result.empty

    groups = [
        pd.DataFrame({"group": [], "score": [], "RA": [], "DEC": []}),
        pd.DataFrame({"group": [2], "score": [1.0], "RA": [0.0], "DEC": [0.0]}),
    ]

    class FakeGroupby:
        def __init__(self, groups_in):
            self.groups_in = groups_in

        def __getitem__(self, _cols):
            return self

        def apply(self, fn, meta):
            frames = [fn(g.copy()) for g in self.groups_in]
            return pd.concat(frames, ignore_index=True)

    class FakeBase:
        def __init__(self, groups_in):
            self.groups_in = groups_in

        def groupby(self, *_args, **_kwargs):
            return FakeGroupby(self.groups_in)

    monkeypatch.setattr(selection_common, "_get_dask_base", lambda *_args, **_kwargs: FakeBase(groups))
    monkeypatch.setattr(selection_common, "_get_meta_df", lambda *_args, **_kwargs: groups[1].head(0))

    result = selection_common.reduce_topk_by_group_dask(
        ddf_like="ignored",
        group_col="group",
        score_col="score",
        order_desc=True,
        k_per_group={1: 1},  # group id 2 not present, so k=0 path used
        ra_col="RA",
        dec_col="DEC",
    )
    assert isinstance(result, pd.DataFrame)
    assert result.empty

    # When _get_dask_base returns an object without groupby, the input is returned untouched.
    monkeypatch.setattr(selection_common, "_get_dask_base", lambda *_args, **_kwargs: object())
    sentinel_obj = object()
    assert (
        selection_common.reduce_topk_by_group_dask(
            ddf_like=sentinel_obj,
            group_col="group",
            score_col="score",
            order_desc=True,
            k_per_group={1: 1},
            ra_col="RA",
            dec_col="DEC",
        )
        is sentinel_obj
    )


def test_add_ipix_column_handles_empty_and_nonempty():
    """add_ipix_column returns int64 __ipix__ and tolerates empty frames."""
    pdf = pd.DataFrame({"RA": [0.0, 90.0], "DEC": [0.0, 0.0]})
    with_ipix = selection_common.add_ipix_column(pdf, depth=1, ra_col="RA", dec_col="DEC")
    assert "__ipix__" in with_ipix.columns
    assert with_ipix["__ipix__"].dtype == np.int64

    empty = selection_common.add_ipix_column(
        pd.DataFrame({"RA": [], "DEC": []}), depth=1, ra_col="RA", dec_col="DEC"
    )
    assert "__ipix__" in empty.columns
    assert empty["__ipix__"].empty


def test_select_by_value_slices_skips_when_histogram_empty(tmp_path, diag_ctx, log_capture, monkeypatch):
    """select_by_value_slices exits early when histogram reports zero total."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "VAL": [5.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    densmaps = {1: np.ones(hp.nside2npix(2), dtype="int64")}
    calls: list[dict] = []

    monkeypatch.setattr(
        selection_slicing,
        "write_tiles_with_allsky",
        lambda **kwargs: (calls.append(kwargs) or ({}, None)),
    )

    def fake_hist_fn(*_, **__):
        return np.zeros(2, dtype="int64"), np.array([0.0, 1.0, 2.0], dtype="float64"), 0

    selection_slicing.select_by_value_slices(
        remainder_ddf=ddf,
        densmaps=densmaps,
        depths_sel=[1],
        keep_cols=["RA", "DEC", "VAL"],
        ra_col="RA",
        dec_col="DEC",
        value_col="VAL",
        order_desc=False,
        label="val",
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        compute_hist_fn=fake_hist_fn,
        value_min=0.0,
        value_max=1.0,
        hist_nbins=2,
    )

    assert calls == []
    assert any("no objects found" in msg for msg in logs)


def test_select_by_value_slices_raises_when_tie_column_missing(tmp_path, diag_ctx, log_capture):
    """Missing tie column raises before ordering."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "VAL": [0.5]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    densmaps = {1: np.ones(hp.nside2npix(2), dtype="int64")}
    level_edges = np.array([0.0, 1.0], dtype="float64")

    with pytest.raises(KeyError):
        selection_slicing.select_by_value_slices(
            remainder_ddf=ddf,
            densmaps=densmaps,
            depths_sel=[1],
            keep_cols=["RA", "DEC", "VAL"],
            ra_col="RA",
            dec_col="DEC",
            value_col="VAL",
            order_desc=False,
            label="val",
            out_dir=tmp_path,
            diag_ctx=diag_ctx,
            log_fn=log_fn,
            level_edges=level_edges,
            tie_col="SNR",  # not present in the dataframe
        )


def test_select_by_value_slices_full_flow(monkeypatch, tmp_path, diag_ctx, log_capture):
    """Exercise histogram branch, empty slice, ordering, and writing."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame(
        {
            "RA": [0.0, 10.0],
            "DEC": [0.0, 5.0],
            "VAL": [1.5, 1.7],
            "TIE": [2.0, 1.0],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=2)
    densmaps = {1: np.ones(hp.nside2npix(2), dtype="int64"), 2: np.ones(hp.nside2npix(4), dtype="int64")}

    captured_written: list[dict] = []

    monkeypatch.setattr(selection_slicing, "build_header_line_from_keep", lambda cols: "|".join(cols))
    monkeypatch.setattr(
        selection_slicing,
        "write_tiles_with_allsky",
        lambda **kwargs: (captured_written.append(kwargs) or ({0: len(kwargs["selected"])}, None)),
    )

    level_edges = np.array([0.0, 1.0, 2.0], dtype="float64")  # first slice empty, second slice populated

    selection_slicing.select_by_value_slices(
        remainder_ddf=ddf,
        densmaps=densmaps,
        depths_sel=[1, 2],
        keep_cols=["RA", "DEC", "VAL", "TIE"],
        ra_col="RA",
        dec_col="DEC",
        value_col="VAL",
        order_desc=True,
        label="val",
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        level_edges=level_edges,
        tie_col="TIE",
    )

    assert captured_written  # depth 2 writes
    assert any("no rows in slice" in msg for msg in logs)
    assert captured_written[0]["selected"]["VAL"].tolist() == [1.7, 1.5]  # order_desc with tie handled


def test_select_by_value_slices_uses_stream_path_without_allsky(monkeypatch, tmp_path, diag_ctx, log_capture):
    """Depths outside (1, 2) use streaming write path and log aggregated write stats."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "VAL": [0.5], "TIE": [1.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    densmaps = {3: np.ones(hp.nside2npix(8), dtype="int64")}
    level_edges = np.array([0.0, 1.0], dtype="float64")
    called: dict[str, Any] = {}

    def fake_stream(**kwargs):
        called.update(kwargs)
        return 7, 3, 7

    monkeypatch.setattr(selection_slicing, "_stream_write_depth_without_allsky", fake_stream)
    monkeypatch.setattr(
        selection_slicing,
        "write_tiles_with_allsky",
        lambda **_kwargs: pytest.fail("write_tiles_with_allsky should not be called directly in stream path"),
    )

    selection_slicing.select_by_value_slices(
        remainder_ddf=ddf,
        densmaps=densmaps,
        depths_sel=[3],
        keep_cols=["RA", "DEC", "VAL", "TIE"],
        ra_col="RA",
        dec_col="DEC",
        value_col="VAL",
        order_desc=False,
        label="val",
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        level_edges=level_edges,
        tie_col="TIE",
    )

    assert called["depth"] == 3
    assert called["tie_col"] == "TIE"
    assert any("[DEPTH 3] selected:" in msg and "selected=7" in msg for msg in logs)
    assert any("[DEPTH 3] written:" in msg and "tiles_written=3" in msg for msg in logs)


def test_stream_write_depth_uses_dask_submit_and_forces_compaction_off(monkeypatch, tmp_path, log_capture):
    """When a distributed client exists, bucket processing is submitted via client.submit."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [1.0, 2.0], "DEC": [1.0, 2.0], "VAL": [0.2, 0.8]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    counts = np.ones(hp.nside2npix(8), dtype="int64")

    class _FakeFuture:
        def __init__(self, value):
            self.value = value

    class _FakeClient:
        def __init__(self):
            self.submissions: list[dict[str, Any]] = []

        def scheduler_info(self):
            return {"workers": {"w1": {}, "w2": {}}}

        def scatter(self, value, broadcast=False):
            return value

        def submit(self, fn, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs.pop("pure", None)
            self.submissions.append(kwargs)
            return _FakeFuture(fn(*args, **kwargs))

        def gather(self, futures):
            return [f.value for f in futures]

    fake_client = _FakeClient()
    monkeypatch.setattr(selection_slicing, "_get_active_dask_client", lambda: fake_client)

    def fake_process_bucket_dir(**kwargs):
        return selection_slicing._BucketWriteStats(
            selected_len=1,
            tiles_written=1,
            rows_written=1,
            files_in=1,
            files_out=1,
        )

    monkeypatch.setattr(selection_slicing, "_process_bucket_dir", fake_process_bucket_dir)

    selected_len, tiles_written, rows_written = selection_slicing._stream_write_depth_without_allsky(
        depth_ddf=ddf,
        depth=3,
        value_col="VAL",
        order_desc=False,
        tie_col=None,
        ra_col="RA",
        dec_col="DEC",
        out_dir=tmp_path,
        header_line="RA\tDEC\tVAL\n",
        counts=counts,
        log_fn=log_fn,
    )

    assert selected_len == 2
    assert tiles_written >= 1
    assert rows_written >= 1
    assert fake_client.submissions
    assert all(s["compaction_mode"] == "off" for s in fake_client.submissions)
    assert any("dask bucket submit" in msg for msg in logs)


def test_stream_write_depth_requires_active_dask_client(monkeypatch, tmp_path, log_capture):
    """Streaming bucket merge fails fast when no distributed Client is active."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [1.0], "DEC": [1.0], "VAL": [0.2]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    counts = np.ones(hp.nside2npix(8), dtype="int64")

    monkeypatch.setattr(selection_slicing, "_get_active_dask_client", lambda: None)

    with pytest.raises(RuntimeError, match="No active dask.distributed Client found"):
        selection_slicing._stream_write_depth_without_allsky(
            depth_ddf=ddf,
            depth=3,
            value_col="VAL",
            order_desc=False,
            tie_col=None,
            ra_col="RA",
            dec_col="DEC",
            out_dir=tmp_path,
            header_line="RA\tDEC\tVAL\n",
            counts=counts,
            log_fn=log_fn,
        )


def test_process_bucket_dir_stream_merge_preserves_stable_order(tmp_path):
    """K-way merge keeps global order and stable tie ordering across files."""
    out_dir = tmp_path / "out"
    bucket_dir = tmp_path / "bucket_0000"
    bucket_dir.mkdir(parents=True, exist_ok=True)

    part0 = pd.DataFrame(
        {
            "__ipix__": [3, 3, 3],
            "VAL": [10.0, 8.0, 7.0],
            "RA": [1.0, 2.0, 5.0],
            "DEC": [0.0, 0.0, 0.0],
            "OBJID": [1, 2, 101],
        }
    )
    part1 = pd.DataFrame(
        {
            "__ipix__": [3, 3, 3],
            "VAL": [9.0, 8.0, 7.0],
            "RA": [0.0, 1.0, 5.0],
            "DEC": [0.0, 1.0, 0.0],
            "OBJID": [3, 4, 202],
        }
    )

    part0.to_parquet(bucket_dir / "part_00000000_a.parquet", index=False)
    part1.to_parquet(bucket_dir / "part_00000001_b.parquet", index=False)

    counts = np.full(hp.nside2npix(8), 100, dtype="int64")
    stats = selection_slicing._process_bucket_dir(
        bucket_dir=bucket_dir,
        depth=3,
        out_dir=out_dir,
        header_line="OBJID\tRA\tDEC\tVAL\n",
        counts=counts,
        sort_cols=["VAL", "RA", "DEC"],
        ascending=[False, True, True],
        compaction_mode="off",
        compaction_min_depth=8,
        compaction_min_files=4096,
        compaction_chunk_size=128,
        compaction_target_files=8,
    )

    assert stats.selected_len == 6
    tile_path = out_dir / "Norder3" / "Dir0" / "Npix3.tsv"
    assert tile_path.exists()

    with tile_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    data_rows = lines[2:]
    objids = [int(row.split("\t")[0]) for row in data_rows]
    assert objids == [1, 3, 4, 2, 101, 202]


def test_stream_compaction_auto_gating():
    """Auto compaction starts at high depth or when bucket fan-out is large."""
    # Below thresholds -> no compaction.
    assert (
        selection_slicing._should_compact_bucket(
            depth=7,
            files_in=1500,
            mode="auto",
            min_depth=8,
            min_files=4096,
        )
        is False
    )
    # Depth threshold reached -> compaction.
    assert (
        selection_slicing._should_compact_bucket(
            depth=8,
            files_in=1500,
            mode="auto",
            min_depth=8,
            min_files=4096,
        )
        is True
    )
    # File threshold reached -> compaction even at lower depth.
    assert (
        selection_slicing._should_compact_bucket(
            depth=6,
            files_in=5000,
            mode="auto",
            min_depth=8,
            min_files=4096,
        )
        is True
    )


def test_stream_compaction_mode_overrides():
    """Explicit on/off modes override adaptive thresholds."""
    assert (
        selection_slicing._should_compact_bucket(
            depth=4,
            files_in=2,
            mode="on",
            min_depth=8,
            min_files=4096,
        )
        is True
    )
    assert (
        selection_slicing._should_compact_bucket(
            depth=10,
            files_in=99999,
            mode="off",
            min_depth=8,
            min_files=4096,
        )
        is False
    )


def test_detected_cpu_count_prefers_affinity(monkeypatch):
    """CPU detection uses process affinity when available."""
    monkeypatch.setattr(selection_slicing.os, "sched_getaffinity", lambda _pid: {0, 1, 2})
    monkeypatch.setattr(selection_slicing.os, "cpu_count", lambda: 99)
    assert selection_slicing._detected_cpu_count() == 3


def test_detected_cpu_count_fallback(monkeypatch):
    """CPU detection falls back to os.cpu_count and then to 1."""
    monkeypatch.setattr(
        selection_slicing.os,
        "sched_getaffinity",
        lambda _pid: (_ for _ in ()).throw(OSError("x")),
    )
    monkeypatch.setattr(selection_slicing.os, "cpu_count", lambda: None)
    assert selection_slicing._detected_cpu_count() == 1

    monkeypatch.setattr(selection_slicing.os, "cpu_count", lambda: 4)
    assert selection_slicing._detected_cpu_count() == 4


def test_resolve_bucket_workers_default_is_adaptive(monkeypatch):
    """Default workers are capped by both detected CPUs and bucket count."""
    monkeypatch.delenv("HIPSCATALOG_STREAM_BUCKET_WORKERS", raising=False)
    monkeypatch.setattr(selection_slicing, "_detected_cpu_count", lambda: 2)
    workers, detected, from_env = selection_slicing._resolve_bucket_workers(16)
    assert (workers, detected, from_env) == (2, 2, False)

    monkeypatch.setattr(selection_slicing, "_detected_cpu_count", lambda: 64)
    workers2, detected2, from_env2 = selection_slicing._resolve_bucket_workers(3)
    assert (workers2, detected2, from_env2) == (3, 64, False)


def test_resolve_bucket_workers_env_allows_override(monkeypatch):
    """Explicit env override can exceed detected CPU count."""
    monkeypatch.setenv("HIPSCATALOG_STREAM_BUCKET_WORKERS", "8")
    monkeypatch.setattr(selection_slicing, "_detected_cpu_count", lambda: 2)
    workers, detected, from_env = selection_slicing._resolve_bucket_workers(16)
    assert (workers, detected, from_env) == (8, 2, True)


def test_select_by_value_slices_histogram_path(monkeypatch, tmp_path, diag_ctx, log_capture):
    """Histogram path with zeroed CDF still proceeds and calls assign_level_edges."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "VAL": [0.1]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    densmaps = {1: np.ones(hp.nside2npix(2), dtype="int64")}

    monkeypatch.setattr(selection_slicing, "build_header_line_from_keep", lambda cols: "|".join(cols))
    monkeypatch.setattr(
        selection_slicing,
        "write_tiles_with_allsky",
        lambda **kwargs: ({0: len(kwargs["selected"])}, None),
    )

    captured_args: list[np.ndarray] = []

    def fake_assign_level_edges(**kwargs):
        captured_args.append(kwargs["cdf_hist"])
        return np.array([0.0, 1.0], dtype="float64"), np.array([1.0])

    def fake_hist_fn(*_, **__):
        return np.zeros(1, dtype="int64"), np.array([0.0, 1.0], dtype="float64"), 1

    monkeypatch.setattr(selection_slicing, "assign_level_edges", fake_assign_level_edges)

    selection_slicing.select_by_value_slices(
        remainder_ddf=ddf,
        densmaps=densmaps,
        depths_sel=[1],
        keep_cols=["RA", "DEC", "VAL"],
        ra_col="RA",
        dec_col="DEC",
        value_col="VAL",
        order_desc=False,
        label="val",
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        compute_hist_fn=fake_hist_fn,
        value_min=0.0,
        value_max=1.0,
        hist_nbins=1,
    )

    assert captured_args and np.allclose(captured_args[0], [0.0])
    assert any("per-depth slices" in msg for msg in logs)


def test_select_by_value_slices_histogram_normalized(monkeypatch, tmp_path, diag_ctx, log_capture):
    """Histogram path where CDF normalizes (cdf_hist[-1] > 0)."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "VAL": [0.2]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    densmaps = {1: np.ones(hp.nside2npix(2), dtype="int64")}

    monkeypatch.setattr(selection_slicing, "build_header_line_from_keep", lambda cols: "|".join(cols))
    monkeypatch.setattr(
        selection_slicing,
        "write_tiles_with_allsky",
        lambda **kwargs: ({0: len(kwargs["selected"])}, None),
    )
    monkeypatch.setattr(selection_slicing, "assign_level_edges", lambda **_: (np.array([0.0, 1.0]), None))

    def fake_hist_fn(*_args, **_kwargs):
        return np.array([1], dtype="int64"), np.array([0.0, 1.0], dtype="float64"), 1

    selection_slicing.select_by_value_slices(
        remainder_ddf=ddf,
        densmaps=densmaps,
        depths_sel=[1],
        keep_cols=["RA", "DEC", "VAL"],
        ra_col="RA",
        dec_col="DEC",
        value_col="VAL",
        order_desc=False,
        label="val",
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        compute_hist_fn=fake_hist_fn,
        value_min=0.0,
        value_max=1.0,
        hist_nbins=1,
    )


def test_select_by_value_slices_requires_hist_params(tmp_path, diag_ctx, log_capture):
    """Missing histogram parameters raises ValueError when level_edges is absent."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "VAL": [0.1]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    densmaps = {1: np.ones(hp.nside2npix(2), dtype="int64")}

    with pytest.raises(ValueError):
        selection_slicing.select_by_value_slices(
            remainder_ddf=ddf,
            densmaps=densmaps,
            depths_sel=[1],
            keep_cols=["RA", "DEC", "VAL"],
            ra_col="RA",
            dec_col="DEC",
            value_col="VAL",
            order_desc=False,
            label="val",
            out_dir=tmp_path,
            diag_ctx=diag_ctx,
            log_fn=log_fn,
        )


def test_select_by_score_slices_delegates(monkeypatch, tmp_path, diag_ctx, log_capture):
    """select_by_score_slices forwards parameters to select_by_value_slices."""
    called: dict[str, Any] = {}

    def fake_select_by_value_slices(**kwargs):
        called.update(kwargs)
        return "ok"

    monkeypatch.setattr(selection_slicing, "select_by_value_slices", fake_select_by_value_slices)

    selection_slicing.select_by_score_slices(
        remainder_ddf="ddf",
        densmaps="densmaps",
        depths_sel=[1],
        keep_cols=["RA"],
        ra_col="RA",
        dec_col="DEC",
        score_col="SCORE",
        score_min=0.0,
        score_max=1.0,
        hist_nbins=2,
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_capture[1],
        label="score",
        order_desc=True,
        fixed_targets=None,
        hist_diag_ctx_name="diag",
        depth_diag_prefix="depth",
        tie_col="TIE",
    )

    assert called.get("value_col") == "SCORE"
    assert called.get("compute_hist_fn") == selection_slicing.compute_score_histogram_ddf


def test_assign_level_edges_branches(log_capture):
    """Cover fixed target validation, rescaling, and monotonic enforcement."""
    logs, log_fn = log_capture
    densmaps = {
        1: np.array([1, 0, 2], dtype="int64"),
        2: np.array([0, 0, 3], dtype="int64"),
    }
    depths_sel = [1, 2]
    cdf_hist = np.array([0.2, 1.0], dtype="float64")
    edges = np.array([0.0, 1.0, 2.0], dtype="float64")

    level_edges, targets = selection_levels.assign_level_edges(
        densmaps=densmaps,
        depths_sel=depths_sel,
        fixed_targets={1: 5, 2: 5},
        cdf_hist=cdf_hist,
        score_edges_hist=edges,
        score_min=0.0,
        score_max=2.0,
        n_tot_score=2.0,
        log_fn=log_fn,
        label="sel",
    )

    assert level_edges[0] == 0.0 and level_edges[-1] == 2.0
    assert np.all(np.diff(level_edges) >= 0)
    assert targets.sum() == pytest.approx(2.0)
    assert any("Rescaling" in msg for msg in logs)

    with pytest.raises(ValueError):
        selection_levels.assign_level_edges(
            densmaps=densmaps,
            depths_sel=depths_sel,
            fixed_targets={1: -1},
            cdf_hist=cdf_hist,
            score_edges_hist=edges,
            score_min=0.0,
            score_max=2.0,
            n_tot_score=2.0,
            log_fn=log_fn,
            label="sel",
        )


def test_assign_level_edges_free_and_monotonic(monkeypatch, log_capture):
    """Distribute remainder to free depths and enforce monotonic edges."""
    _, log_fn = log_capture
    densmaps = {1: np.array([1, 1], dtype="int64"), 2: np.array([1, 1], dtype="int64")}
    depths_sel = [1, 2]
    cdf_hist = np.array([0.0, 0.0], dtype="float64")
    edges = np.array([0.0, 0.5, 1.0], dtype="float64")

    level_edges, targets = selection_levels.assign_level_edges(
        densmaps=densmaps,
        depths_sel=depths_sel,
        fixed_targets={1: 1, 99: 3},  # depth 99 ignored
        cdf_hist=cdf_hist,
        score_edges_hist=edges,
        score_min=0.0,
        score_max=1.0,
        n_tot_score=4.0,
        log_fn=log_fn,
        label="sel",
    )

    assert targets.sum() == pytest.approx(4.0)
    assert np.all(np.diff(level_edges) >= 0)  # monotonic enforcement kicks in

    def fake_asarray(vals, dtype=None):
        return np.zeros(len(vals), dtype="float64")

    monkeypatch.setattr(selection_levels.np, "asarray", fake_asarray)
    level_edges_zero, targets_zero = selection_levels.assign_level_edges(
        densmaps=densmaps,
        depths_sel=depths_sel,
        fixed_targets={},
        cdf_hist=cdf_hist,
        score_edges_hist=edges,
        score_min=0.0,
        score_max=1.0,
        n_tot_score=2.0,
        log_fn=log_fn,
        label="sel",
    )

    assert targets_zero.sum() == pytest.approx(2.0)
    assert np.all(level_edges_zero >= 0.0)
