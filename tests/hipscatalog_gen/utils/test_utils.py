"""Unit tests for shared utilities in hipscatalog_gen.utils."""

from __future__ import annotations

import re
import sys
import types

import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd
from hipscatalog_gen import utils


@pytest.fixture
def log_capture():
    """Collects log messages emitted by helpers."""
    logs: list[str] = []

    def _log_fn(msg: str, always: bool = False, **_: dict) -> None:
        logs.append(msg)

    return logs, _log_fn


def test_mkdirs_and_write_text(tmp_path):
    """_mkdirs creates dirs and _write_text writes UTF-8 content."""
    target_dir = tmp_path / "a" / "b"
    utils._mkdirs(target_dir)
    assert target_dir.exists()

    out_file = target_dir / "file.txt"
    utils._write_text(out_file, "hello")
    assert out_file.read_text(encoding="utf-8") == "hello"


def test_detect_hats_catalog_root(tmp_path):
    """_detect_hats_catalog_root finds markers and returns None otherwise."""
    hats_dir = tmp_path / "nested"
    utils._mkdirs(hats_dir)
    marker = hats_dir / "hats.properties"
    marker.write_text("x", encoding="utf-8")

    # Should resolve to the directory containing the marker.
    root = utils._detect_hats_catalog_root([str(hats_dir / "file.dat")])
    assert root == hats_dir

    # No marker -> None.
    assert utils._detect_hats_catalog_root([str(tmp_path / "other")]) is None


def test_time_helpers_and_fmt_dur():
    """_now_str/_ts return formatted strings and _fmt_dur formats durations."""
    now = utils._now_str()
    ts = utils._ts()
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z", now)
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", ts)
    assert utils._fmt_dur(3661.789) == "01:01:01.789"


def test_stats_counts_and_log_depth_stats(log_capture):
    """_stats_counts aggregates totals and _log_depth_stats formats message."""
    logs, log_fn = log_capture
    counts = np.array([1, 0, 2], dtype="int64")
    total, nz = utils._stats_counts(counts)
    assert total == 3 and nz == 2

    utils._log_depth_stats(log_fn, depth=1, phase="selected", counts=counts, selected_len=3, written={0: 2})
    assert any("input_rows=3" in msg and "selected=3" in msg for msg in logs)

    utils._log_depth_stats(log_fn, depth=2, phase="done", candidates_len=5, remainder_len=1, written={})
    assert any("candidates=5" in msg and "remainder=1" in msg for msg in logs)


def test_log_depth_stats_minimal(log_capture):
    """_log_depth_stats also handles calls without optional fields."""
    logs, log_fn = log_capture
    utils._log_depth_stats(log_fn, depth=0, phase="start")
    assert logs == ["[DEPTH 0] start: "]


def test_get_dask_base_regular_and_missing(monkeypatch):
    """_get_dask_base returns dask-like objects or raises when missing methods."""

    class WithMap:
        def map_partitions(self):
            return self

    obj = WithMap()
    assert utils._get_dask_base(obj) is obj
    assert utils._get_dask_base(obj, require_map_partitions=True) is obj

    with pytest.raises(TypeError):
        utils._get_dask_base(object(), require_map_partitions=True)

    with pytest.raises(TypeError):
        utils._get_dask_base(object(), require_to_delayed=True)

    # LSDB catalog path with _ddf fallback.
    fake_base = types.SimpleNamespace(groupby=lambda self=None: None)

    class FakeCatalog:
        def __init__(self, base):
            self._ddf = base

    lsdb_mod = types.ModuleType("lsdb")
    catalog_mod = types.ModuleType("lsdb.catalog")
    catalog_mod.Catalog = FakeCatalog
    lsdb_mod.catalog = catalog_mod
    sys.modules["lsdb"] = lsdb_mod
    sys.modules["lsdb.catalog"] = catalog_mod

    catalog = FakeCatalog(fake_base)
    assert utils._get_dask_base(catalog, require_groupby=True) is fake_base

    catalog_no_base = FakeCatalog(base=None)
    with pytest.raises(TypeError):
        utils._get_dask_base(catalog_no_base)

    class WithToDelayed:
        def to_delayed(self):
            return "ok"

    assert utils._get_dask_base(WithToDelayed(), require_to_delayed=True) is not None

    catalog_bad_methods = FakeCatalog(base=object())
    with pytest.raises(TypeError):
        utils._get_dask_base(catalog_bad_methods, require_groupby=True)


def test_score_deps_and_resolve_col_name():
    """_score_deps parses identifiers and _resolve_col_name resolves names/indices."""
    deps = utils._score_deps("a + b * c", ["a", "c", "x"])
    assert deps == ["a", "c"]
    assert utils._score_deps("", ["a"]) == []

    pdf = pd.DataFrame({"A": [1], "B": [2]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    assert utils._resolve_col_name("A", ddf, header=True) == "A"
    with pytest.raises(KeyError):
        utils._resolve_col_name("C", ddf, header=True)

    assert utils._resolve_col_name("1", ddf, header=False) == "A"
    with pytest.raises(IndexError):
        utils._resolve_col_name("5", ddf, header=False)
    with pytest.raises(KeyError):
        utils._resolve_col_name("Z", ddf, header=False)

    assert utils._resolve_col_name("A", ddf, header=False) == "A"


def test_get_meta_df_with_meta_and_head(monkeypatch):
    """_get_meta_df prefers _meta, falls back to head(0), else empty DataFrame."""
    pdf = pd.DataFrame({"A": pd.Series([], dtype="float64")})
    ddf = dd.from_pandas(pdf, npartitions=1)
    meta = utils._get_meta_df(ddf)
    assert list(meta.columns) == ["A"]

    class FakeBase:
        def __init__(self):
            self._meta = None

        def head(self, n):
            return pd.DataFrame({"X": pd.Series([], dtype="int64")})

    monkeypatch.setattr(utils, "_get_dask_base", lambda *_: FakeBase())
    meta2 = utils._get_meta_df(object())
    assert list(meta2.columns) == ["X"]

    class FakeBad:
        def head(self, n):
            raise ValueError("fail")

    monkeypatch.setattr(utils, "_get_dask_base", lambda *_: FakeBad())
    meta3 = utils._get_meta_df(object())
    assert meta3.empty


def test_get_meta_df_to_pandas_error(monkeypatch):
    """_get_meta_df tolerates meta/head objects with broken to_pandas implementations."""

    class BadMeta:
        def to_pandas(self):
            raise RuntimeError("boom")

    class WeirdHead:
        def to_pandas(self):
            return ["not a dataframe"]

    class FakeBase:
        def __init__(self):
            self._meta = BadMeta()

        def head(self, n):
            return WeirdHead()

    monkeypatch.setattr(utils, "_get_dask_base", lambda *_: FakeBase())
    meta = utils._get_meta_df(object())
    assert meta.empty


def test_get_meta_df_to_pandas_wrapper(monkeypatch):
    """_get_meta_df supports meta objects that implement to_pandas()."""

    class MetaWrapper:
        def to_pandas(self):
            return pd.DataFrame({"W": pd.Series([], dtype="float64")})

    class FakeBase:
        def __init__(self):
            self._meta = MetaWrapper()

    monkeypatch.setattr(utils, "_get_dask_base", lambda *_: FakeBase())
    meta = utils._get_meta_df(object())
    assert list(meta.columns) == ["W"]


def test_validate_and_normalize_radec(log_capture):
    """_validate_and_normalize_radec checks ranges and shifts RA when needed."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [0.0, 90.0], "DEC": [0.0, 45.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    out = utils._validate_and_normalize_radec(ddf, "RA", "DEC", log_fn)
    assert out.compute()["RA"].tolist() == [0.0, 90.0]
    assert any("RA in [0.000000" in msg for msg in logs)

    # RA in [-180, 180] shifts to [0, 360]
    pdf2 = pd.DataFrame({"RA": [-180.0, 0.0, 180.0], "DEC": [0.0, 0.0, 0.0]})
    ddf2 = dd.from_pandas(pdf2, npartitions=1)
    out2 = utils._validate_and_normalize_radec(ddf2, "RA", "DEC", log_fn)
    ra_vals = out2.compute()["RA"].tolist()
    assert all(0.0 <= v <= 360.0 for v in ra_vals)

    # Invalid DEC range
    pdf_bad_dec = pd.DataFrame({"RA": [0.0], "DEC": [100.0]})
    with pytest.raises(ValueError):
        utils._validate_and_normalize_radec(dd.from_pandas(pdf_bad_dec, npartitions=1), "RA", "DEC", log_fn)

    # Invalid RA range
    pdf_bad_ra = pd.DataFrame({"RA": [500.0], "DEC": [0.0]})
    with pytest.raises(ValueError):
        utils._validate_and_normalize_radec(dd.from_pandas(pdf_bad_ra, npartitions=1), "RA", "DEC", log_fn)

    # Non-finite values
    pdf_nan = pd.DataFrame({"RA": [np.nan], "DEC": [0.0]})
    with pytest.raises(ValueError):
        utils._validate_and_normalize_radec(dd.from_pandas(pdf_nan, npartitions=1), "RA", "DEC", log_fn)


def test_validate_and_normalize_radec_empty_partition(log_capture):
    """Empty partitions are left untouched during RA shift."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [-180.0, 0.0], "DEC": [0.0, 0.0]})
    ddf = dd.concat(
        [
            dd.from_pandas(pdf, npartitions=1),
            dd.from_pandas(
                pd.DataFrame({"RA": pd.Series([], dtype="float64"), "DEC": pd.Series([], dtype="float64")}),
                npartitions=1,
            ),
        ],
        interleave_partitions=True,
    )
    out = utils._validate_and_normalize_radec(ddf, "RA", "DEC", log_fn)
    assert out.compute().shape[0] == 2
