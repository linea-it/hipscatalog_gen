"""Unit tests for hipscatalog_gen.io input and output helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd

import hipscatalog_gen
import hipscatalog_gen.io.input as io_input
import hipscatalog_gen.io.output as io_output
from hipscatalog_gen.config import AlgoOpts, ClusterCfg, ColumnsCfg, Config, InputCfg, OutputCfg
from hipscatalog_gen.io import (
    TSVTileWriter,
    _build_input_ddf,
    build_header_line_from_keep,
    compute_column_report_global,
    compute_column_report_sample,
    finalize_write_tiles,
    write_arguments,
    write_densmap_fits,
    write_index_html,
    write_metadata_xml,
    write_moc,
    write_properties,
)

# Ensure fallback stubs are exercised even when real packages are present.
sys.modules.pop("lsdb", None)
sys.modules.pop("lsdb.catalog", None)
sys.modules.pop("mocpy", None)

lsdb_mod = types.ModuleType("lsdb")
catalog_mod = types.ModuleType("lsdb.catalog")


class DummyCatalog:
    """Minimal lsdb catalog stub for tests."""


def _dummy_open_catalog(*args, **kwargs):
    return DummyCatalog()


lsdb_mod.catalog = catalog_mod
lsdb_mod.open_catalog = _dummy_open_catalog
catalog_mod.Catalog = DummyCatalog
sys.modules["lsdb"] = lsdb_mod
sys.modules["lsdb.catalog"] = catalog_mod

if "mocpy" not in sys.modules:
    mocpy_mod = types.ModuleType("mocpy")

    class DummyMOC:
        """Minimal mocpy MOC stub for tests."""

    mocpy_mod.MOC = DummyMOC
    sys.modules["mocpy"] = mocpy_mod


def _base_cfg(
    fmt: str,
    *,
    header: bool = True,
    keep: list[str] | None = None,
    selection_mode: str = "mag_global",
    ascii_format: str | None = None,
    ra: str = "RA",
    dec: str = "DEC",
    score_column: str | None = None,
    sdh_score_column: str | None = None,
) -> Config:
    algo = AlgoOpts(
        selection_mode=selection_mode,
        level_limit=1,
        moc_order=1,
        mag_column="MAG" if selection_mode == "mag_global" else None,
        score_column=score_column,
        sdh_score_column=sdh_score_column,
    )
    cluster = ClusterCfg(mode="local", n_workers=1, threads_per_worker=1, memory_per_worker="1GB")
    output = OutputCfg(out_dir=".", cat_name="cat", target="0 0")
    return Config(
        input=InputCfg(paths=[], format=fmt, header=header, ascii_format=ascii_format),
        columns=ColumnsCfg(ra=ra, dec=dec, keep=keep),
        algorithm=algo,
        cluster=cluster,
        output=output,
    )


# =============================================================================
# input.py
# =============================================================================


def test_build_input_ddf_validations():
    """_build_input_ddf guards empty paths and unsupported formats."""
    cfg = _base_cfg("parquet")
    with pytest.raises(ValueError):
        _build_input_ddf([], cfg)

    bad_cfg = _base_cfg("fits")
    with pytest.raises(ValueError):
        _build_input_ddf(["file.fits"], bad_cfg)


def test_build_input_ddf_hats_keep_all(monkeypatch):
    """HATS input keeps all columns when columns.keep is None."""
    pdf = pd.DataFrame({"RA": [1.0], "DEC": [2.0], "score": [0.5], "EXTRA": [7]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    call_args: dict[str, Any] = {}

    class FakeCatalog:
        def __init__(self, columns: list[str]):
            self.columns = columns

        def __getitem__(self, cols: list[str]):
            return ddf[cols]

    def fake_open(path: str, columns: list[str] | str | None = None):
        call_args["columns"] = columns
        cols = list(pdf.columns) if columns in (None, "all") else list(columns)
        return FakeCatalog(cols)

    monkeypatch.setattr(io_input.lsdb, "open_catalog", fake_open)

    cfg = _base_cfg("hats", selection_mode="score_density_hybrid", sdh_score_column="score")
    ddf_out, ra, dec, keep_cols = _build_input_ddf(["/tmp/cat.hats"], cfg)

    assert call_args["columns"] == "all"
    assert (ra, dec) == ("RA", "DEC")
    assert keep_cols == ["RA", "DEC", "score", "EXTRA"]
    assert list(ddf_out.columns) == keep_cols


def test_build_input_ddf_hats_keep_none_preserves_catalog_order(monkeypatch):
    """HATS input with keep=None preserves source catalog column order."""
    pdf = pd.DataFrame({"SCORE": [0.5], "DEC": [2.0], "RA": [1.0], "EXTRA": [7]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    call_args: dict[str, Any] = {}

    class FakeCatalog:
        def __init__(self, columns: list[str]):
            self.columns = columns

        def __getitem__(self, cols: list[str]):
            return ddf[cols]

    def fake_open(path: str, columns: list[str] | str | None = None):
        call_args["columns"] = columns
        cols = list(pdf.columns) if columns in (None, "all") else list(columns)
        return FakeCatalog(cols)

    monkeypatch.setattr(io_input.lsdb, "open_catalog", fake_open)

    cfg = _base_cfg("hats", selection_mode="score_global", score_column="SCORE")
    ddf_out, ra, dec, keep_cols = _build_input_ddf(["/tmp/cat.hats"], cfg)

    assert call_args["columns"] == "all"
    assert (ra, dec) == ("RA", "DEC")
    assert keep_cols == ["SCORE", "DEC", "RA", "EXTRA"]
    assert list(ddf_out.columns) == keep_cols


def test_build_input_ddf_hats_subset(monkeypatch):
    """HATS input requests explicit columns when keep is provided."""
    pdf = pd.DataFrame({"RA": [1.0], "DEC": [2.0], "SCORE": [3.0], "EXTRA": [4.0]})
    call_args: dict[str, Any] = {}

    class FakeCatalog:
        def __init__(self, cols: list[str], ddf_obj: dd.DataFrame):
            self.columns = cols
            self._ddf = ddf_obj

        def __getitem__(self, cols: list[str]):
            return self._ddf[cols]

    def fake_open(path: str, columns: list[str] | None = None):
        cols = list(columns) if columns is not None else list(pdf.columns)
        call_args["columns"] = columns
        ddf_obj = dd.from_pandas(pdf[cols], npartitions=1)
        return FakeCatalog(cols, ddf_obj)

    monkeypatch.setattr(io_input.lsdb, "open_catalog", fake_open)

    cfg = _base_cfg("hats", keep=["EXTRA"], selection_mode="score_global", score_column="SCORE")
    ddf_out, ra, dec, keep_cols = _build_input_ddf(["/tmp/cat.hats"], cfg)

    assert call_args["columns"] == ["RA", "DEC", "SCORE", "EXTRA"]
    assert (ra, dec) == ("RA", "DEC")
    assert keep_cols == ["RA", "DEC", "SCORE", "EXTRA"]
    assert list(ddf_out.columns) == keep_cols
    # Ensure stub open_catalog path is covered.
    assert _dummy_open_catalog() is not None


def test_build_input_ddf_hats_requires_single_path():
    """HATS input rejects multiple paths."""
    cfg = _base_cfg("hats")
    with pytest.raises(ValueError):
        _build_input_ddf(["a", "b"], cfg)


def test_build_input_ddf_hats_mag_and_flux(monkeypatch):
    """HATS input orders mag/flux when available."""
    pdf = pd.DataFrame({"RA": [1.0], "DEC": [2.0], "MAG": [3.0], "FLUX": [4.0]})
    call_args: dict[str, Any] = {}

    class FakeCatalog:
        def __init__(self, cols: list[str]):
            self.columns = cols

        def __getitem__(self, cols: list[str]):
            return dd.from_pandas(pdf[cols], npartitions=1)

    def fake_open(path: str, columns: list[str] | str | None = None):
        cols = list(pdf.columns) if columns in (None, "all") else list(columns)
        call_args["columns"] = columns
        return FakeCatalog(cols)

    monkeypatch.setattr(io_input.lsdb, "open_catalog", fake_open)

    cfg = _base_cfg("hats")
    cfg.algorithm.flux_column = "FLUX"
    ddf_out, ra, dec, keep_cols = _build_input_ddf(["/tmp/cat.hats"], cfg)

    assert call_args["columns"] == "all"  # keep_all_columns path
    assert keep_cols[:4] == ["RA", "DEC", "MAG", "FLUX"]
    assert list(ddf_out.columns) == keep_cols


def test_build_input_ddf_parquet_and_csv(tmp_path):
    """Parquet and CSV inputs respect keep_all and score dependencies."""
    pq_pdf = pd.DataFrame({"RA": [0.0], "DEC": [1.0], "flux": [2.0]})
    pq_path = tmp_path / "data.parquet"
    pq_pdf.to_parquet(pq_path, index=False)

    cfg_parquet = _base_cfg("parquet")
    ddf_parquet, ra, dec, keep_cols = _build_input_ddf([str(pq_path)], cfg_parquet)
    assert (ra, dec) == ("RA", "DEC")
    assert keep_cols == ["RA", "DEC", "flux"]
    assert ddf_parquet.compute().equals(pq_pdf[keep_cols])

    csv_pdf = pd.DataFrame({"RA": [1], "DEC": [2], "SCORE": [3]})
    csv_path = tmp_path / "data.csv"
    csv_pdf.to_csv(csv_path, sep="\t", index=False)

    cfg_csv = _base_cfg(
        "csv",
        keep=[],
        selection_mode="score_global",
        score_column="SCORE",
        ascii_format="TSV",
    )
    ddf_csv, ra_c, dec_c, keep_cols_csv = _build_input_ddf([str(csv_path)], cfg_csv)
    assert (ra_c, dec_c) == ("RA", "DEC")
    assert keep_cols_csv == ["RA", "DEC", "SCORE"]
    pd.testing.assert_frame_equal(ddf_csv.compute(), csv_pdf[keep_cols_csv], check_dtype=False)

    csv_std_path = tmp_path / "data_std.csv"
    csv_pdf.to_csv(csv_std_path, index=False)
    cfg_csv_default = _base_cfg("csv", keep=[], selection_mode="mag_global")
    ddf_csv_default, ra_d, dec_d, keep_cols_default = _build_input_ddf([str(csv_std_path)], cfg_csv_default)
    assert (ra_d, dec_d) == ("RA", "DEC")
    assert keep_cols_default == ["RA", "DEC"]
    assert list(ddf_csv_default.columns) == keep_cols_default

    # Parquet with mag/flux columns should include both when available.
    pq_pdf2 = pd.DataFrame({"RA": [1.0], "DEC": [2.0], "MAG": [3.0], "FLUX": [4.0]})
    pq_path2 = tmp_path / "data2.parquet"
    pq_pdf2.to_parquet(pq_path2, index=False)
    cfg_parquet2 = _base_cfg("parquet", selection_mode="mag_global")
    cfg_parquet2.algorithm.flux_column = "FLUX"
    ddf_parquet2, ra2, dec2, keep_cols2 = _build_input_ddf([str(pq_path2)], cfg_parquet2)
    assert keep_cols2 == ["RA", "DEC", "MAG", "FLUX"]
    assert ddf_parquet2.compute().equals(pq_pdf2[keep_cols2])


def test_build_input_ddf_keep_none_preserves_input_order(tmp_path):
    """keep=None preserves input order for non-HATS inputs."""
    pdf = pd.DataFrame({"ID": [9], "DEC": [1.0], "RA": [2.0], "SCORE": [3.0], "EXTRA": [4.0]})
    path = tmp_path / "ordered.parquet"
    pdf.to_parquet(path, index=False)

    cfg = _base_cfg("parquet", keep=None, selection_mode="score_global", score_column="SCORE")
    ddf_out, ra, dec, keep_cols = _build_input_ddf([str(path)], cfg)

    assert (ra, dec) == ("RA", "DEC")
    assert keep_cols == list(pdf.columns)
    pd.testing.assert_frame_equal(ddf_out.compute(), pdf[keep_cols], check_dtype=False)


def test_build_input_ddf_keep_non_empty_complete_follows_keep_order(tmp_path):
    """When keep includes all required columns, final order follows keep exactly."""
    pdf = pd.DataFrame({"RA": [2.0], "DEC": [1.0], "SCORE": [3.0], "EXTRA": [4.0]})
    path = tmp_path / "keep_complete.parquet"
    pdf.to_parquet(path, index=False)

    keep = ["EXTRA", "RA", "DEC", "SCORE"]
    cfg = _base_cfg("parquet", keep=keep, selection_mode="score_global", score_column="SCORE")
    ddf_out, _, _, keep_cols = _build_input_ddf([str(path)], cfg)

    assert keep_cols == keep
    pd.testing.assert_frame_equal(ddf_out.compute(), pdf[keep], check_dtype=False)


def test_build_input_ddf_keep_non_empty_missing_required_prepends_missing(tmp_path):
    """Missing required columns are prepended; missing RA/DEC lead that block."""
    pdf = pd.DataFrame({"EXTRA": [4.0], "SCORE": [3.0], "RA": [2.0], "DEC": [1.0]})
    path = tmp_path / "keep_missing.parquet"
    pdf.to_parquet(path, index=False)

    cfg = _base_cfg("parquet", keep=["EXTRA"], selection_mode="score_global", score_column="SCORE")
    ddf_out, _, _, keep_cols = _build_input_ddf([str(path)], cfg)

    assert keep_cols == ["RA", "DEC", "SCORE", "EXTRA"]
    pd.testing.assert_frame_equal(ddf_out.compute(), pdf[keep_cols], check_dtype=False)


def test_build_input_ddf_tsv_no_header(tmp_path):
    """ASCII input without header resolves 1-based indices and requested columns."""
    pdf = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["A", "B", "C"])
    path = tmp_path / "data.tsv"
    pdf.to_csv(path, sep="\t", header=False, index=False)

    cfg = _base_cfg(
        "tsv",
        header=False,
        keep=None,
        selection_mode="score_density_hybrid",
        sdh_score_column="X",
        ascii_format="PIPE",
        ra="1",
        dec="2",
    )
    ddf_out, ra, dec, keep_cols = _build_input_ddf([str(path)], cfg)

    assert (ra, dec) == (0, 1)
    assert keep_cols == [0, 1, 2]
    assert list(ddf_out.columns) == keep_cols


def test_compute_column_report_sample_branches():
    """Sampling handles TypeError paths and builds numeric/string stats."""
    pdf = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", None]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    report = compute_column_report_sample(ddf, sample_rows=200_000)
    assert report["columns"]["A"]["min"] == 1.0
    assert report["columns"]["B"]["example"] in {"x", "y"}

    class NoLen:
        def __len__(self):
            raise RuntimeError("no len")

    class WeirdSampler:
        def __init__(self, df: pd.DataFrame):
            self._df = df
            self.columns = NoLen()

        def sample(self, *, n: int | None = None, **_: Any):
            if n is None:
                raise TypeError("needs n")
            return self._df

    report2 = compute_column_report_sample(WeirdSampler(pdf), sample_rows=2)
    assert report2["columns"]["A"]["max"] == 2.0

    class AlwaysFailSampler:
        def __init__(self, df: pd.DataFrame):
            self._df = df
            self.columns = df.columns

        def sample(self, **kwargs: Any):
            if "frac" in kwargs:
                raise TypeError("bad frac")
            if "n" in kwargs:
                raise RuntimeError("bad n")
            return self._df

        def head(self, n: int, compute: bool | None = None):
            return self._df.head(n)

    report3 = compute_column_report_sample(AlwaysFailSampler(pdf), sample_rows=1)
    assert report3["columns"]["A"]["min"] == 1.0
    # Direct call to cover successful path.
    assert AlwaysFailSampler(pdf).sample() is pdf

    empty_pdf = pd.DataFrame({"Z": pd.Series([], dtype="float64")})
    report_empty = compute_column_report_sample(empty_pdf, sample_rows=5)
    assert np.isnan(report_empty["columns"]["Z"]["min"])
    assert np.isnan(report_empty["columns"]["Z"]["max"])
    assert np.isnan(report_empty["columns"]["Z"]["mean"])

    class HeadOnly:
        def __init__(self, df: pd.DataFrame):
            self._df = df

        def head(self, n: int, compute: bool | None = None):
            return self._df.head(n)

    report3 = compute_column_report_sample(HeadOnly(pdf), sample_rows=1)
    assert "A" in report3["columns"]


def test_compute_column_report_global():
    """Global report computes numeric aggregations and string examples."""
    pdf = pd.DataFrame({"X": [1.0, 2.0], "Y": ["a", "b"]})
    pdf["Y"] = pdf["Y"].apply(lambda x: x.encode())
    report = compute_column_report_global(pdf)
    cols = report["columns"]
    assert cols["X"]["min"] == 1.0 and cols["X"]["max"] == 2.0
    assert cols["Y"]["example"] in {"a", "b", "b'a'", "b'b'"}


def test_compute_column_report_global_example_error(monkeypatch):
    """compute_column_report_global tolerates bad example values."""

    class FakeSeries:
        def isna(self):
            return self

        def sum(self):
            return 0

        def min(self):
            return 0

        def max(self):
            return 0

        def mean(self):
            return 0

        def dropna(self):
            return self

        def head(self, n: int):
            return self

    class FakeDDF:
        def __init__(self):
            self.dtypes = types.SimpleNamespace(
                to_dict=lambda: {"X": np.dtype("float64"), "Y": np.dtype("object")}
            )

        def __getitem__(self, key: str):
            return FakeSeries()

    bad_value = types.SimpleNamespace()

    def fake_compute(*tasks: Any):
        return (
            0,  # X n_null
            1,  # X min
            2,  # X max
            1.5,  # X mean
            0,  # Y n_null
            bad_value,  # Y example (will fail)
        )

    monkeypatch.setattr(io_input, "dask_compute", fake_compute)

    def _iloc_fail():
        raise RuntimeError("boom")

    bad_value.iloc = types.SimpleNamespace(__getitem__=lambda self, idx: _iloc_fail())
    with pytest.raises(RuntimeError):
        _iloc_fail()

    report = compute_column_report_global(FakeDDF())
    assert report["columns"]["Y"]["example"] == ""


# =============================================================================
# output.py
# =============================================================================


def test_tsv_tile_writer_paths(tmp_path):
    """TSVTileWriter builds expected directory layout and file names."""
    writer = TSVTileWriter(tmp_path, depth=1, header_line="RA\tDEC\n")
    assert writer.norder_dir.exists()
    assert writer.allsky_tmp().name == ".Allsky.tsv"
    assert writer.allsky_path().name == "Allsky.tsv"
    assert writer.cell_tmp(42).name == ".Npix42.tsv"
    assert writer.cell_path(42).name == "Npix42.tsv"


def test_finalize_write_tiles_empty_returns(tmp_path):
    """Empty selections short-circuit with empty results."""
    counts = np.array([1, 2], dtype="int64")
    selected = pd.DataFrame(columns=["__ipix__", "RA", "DEC"])
    written, allsky_df = finalize_write_tiles(
        tmp_path,
        depth=1,
        header_line="RA\tDEC\n",
        ra_col="RA",
        dec_col="DEC",
        counts=counts,
        selected=selected,
        order_desc=False,
    )
    assert written == {} and allsky_df is None


def test_finalize_write_tiles_writes_and_collects(tmp_path):
    """Tiles are written with completeness header and strings sanitized."""
    counts = np.array([2, 1], dtype="int64")
    selected = pd.DataFrame(
        {
            "__ipix__": [0, 0, 1],
            "__score__": [1.0, 2.0, 3.0],
            "__icov__": [0, 0, 0],
            "RA": [10.0, 20.0, 30.0],
            "DEC": [0.0, 1.0, 2.0],
            "NAME": ["a\nb", "c\td", "e"],
        }
    )
    header_line = "RA\tDEC\tNAME\n"
    written, allsky_df = finalize_write_tiles(
        tmp_path,
        depth=1,
        header_line=header_line,
        ra_col="RA",
        dec_col="DEC",
        counts=counts,
        selected=selected,
        order_desc=False,
        allsky_collect=True,
    )

    tile0 = tmp_path / "Norder1" / "Dir0" / "Npix0.tsv"
    assert tile0.exists()
    content = tile0.read_text(encoding="utf-8").splitlines()
    assert content[0].startswith("# Completeness =")
    assert "a b" in content[-2] and "c d" in content[-1]
    assert written == {0: 2, 1: 1}
    assert allsky_df is not None and len(allsky_df) == 3


def test_finalize_write_tiles_handles_all_null_object_column(tmp_path):
    """All-null object columns should not raise during regex sanitization."""
    counts = np.array([2, 1], dtype="int64")
    selected = pd.DataFrame(
        {
            "__ipix__": [0, 0, 1],
            "RA": [10.0, 20.0, 30.0],
            "DEC": [0.0, 1.0, 2.0],
            "NAME": [None, None, "a\nb"],
        }
    )
    header_line = "RA\tDEC\tNAME\n"
    written, allsky_df = finalize_write_tiles(
        tmp_path,
        depth=1,
        header_line=header_line,
        ra_col="RA",
        dec_col="DEC",
        counts=counts,
        selected=selected,
        order_desc=False,
        allsky_collect=False,
    )

    tile0 = tmp_path / "Norder1" / "Dir0" / "Npix0.tsv"
    tile1 = tmp_path / "Norder1" / "Dir0" / "Npix1.tsv"
    assert tile0.exists() and tile1.exists()
    assert written == {0: 2, 1: 1}
    assert allsky_df is None
    assert "a b" in tile1.read_text(encoding="utf-8")


def test_finalize_write_tiles_skips_out_of_range(tmp_path):
    """Tiles with invalid ipix are skipped."""
    counts = np.array([1], dtype="int64")
    selected = pd.DataFrame({"__ipix__": [-1, 5], "RA": [0.0, 1.0], "DEC": [0.0, 1.0]})
    header_line = "RA\tDEC\n"
    written, allsky_df = finalize_write_tiles(
        tmp_path,
        depth=0,
        header_line=header_line,
        ra_col="RA",
        dec_col="DEC",
        counts=counts,
        selected=selected,
        order_desc=False,
        allsky_collect=False,
    )
    assert written == {}
    assert allsky_df is None


def test_build_header_line_from_keep():
    """Header builder joins names with tabs and newline."""
    assert build_header_line_from_keep(["A", "B"]) == "A\tB\n"


def test_write_properties_and_arguments(tmp_path):
    """Properties and arguments files are created with fallback target parsing."""
    out_cfg = OutputCfg(out_dir=str(tmp_path), cat_name="cat", target="123 456")
    write_properties(tmp_path, out_cfg, level_limit=5, n_src=10)
    props = (tmp_path / "properties").read_text(encoding="utf-8")
    assert "publisher_did   = ivo://PRIVATE_USER/cat" in props
    assert "hips_order      = 5" in props
    assert "hips_initial_ra = 123" in props
    assert f"hips_builder    = linea-it/hipscatalog_gen v{hipscatalog_gen.__version__}" in props

    out_cfg_bad = OutputCfg(out_dir=str(tmp_path), cat_name="cat", target="bad-target")
    write_properties(tmp_path, out_cfg_bad, level_limit=5, n_src=10)
    props_bad = (tmp_path / "properties").read_text(encoding="utf-8")
    assert "hips_initial_ra = 0" in props_bad

    write_arguments(tmp_path, "--input x")
    assert (tmp_path / "arguments").read_text(encoding="utf-8") == "--input x"


def test_write_index_html(tmp_path):
    """index.html is generated with basic links and catalog label."""
    out_cfg = OutputCfg(out_dir=str(tmp_path), cat_name="DES_DR2_sample", target="0 0")
    write_index_html(tmp_path, out_cfg)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "DES_DR2_sample HiPS catalogue" in html
    assert "metadata.xml" in html
    assert "Moc.fits" in html
    assert "Norder1/Allsky.tsv" in html


def test_write_metadata_xml(monkeypatch, tmp_path):
    """metadata.xml is written, using the TypeError fallback when needed."""
    calls: list[Path] = []

    def fake_writeto(votable, path):
        calls.append(Path(path))
        raise TypeError("force fallback")

    monkeypatch.setattr(io_output, "vot_writeto", fake_writeto)

    columns = [
        ("RA", "float64", None),
        ("DEC", "float64", None),
        ("NAME", "str", "meta.id;src"),
        ("ID", "int32", None),
    ]
    write_metadata_xml(tmp_path, columns, ra_idx=0, dec_idx=1)

    lower = tmp_path / "metadata.xml"
    upper = tmp_path / "Metadata.xml"
    assert lower.exists() and upper.exists()
    xml_text = lower.read_text(encoding="utf-8")
    assert "pos.eq.ra;meta.main" in xml_text and "pos.eq.dec;meta.main" in xml_text
    assert calls  # TypeError branch was hit


def test_write_moc_branches(monkeypatch, tmp_path):
    """write_moc covers empty, builder fallback, save fallbacks, and JSON types."""

    class FakeMOC:
        fail_next = False
        serialize_value: Any = {}
        save_raises: Exception | None = None
        from_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def __init__(self, payload: Any):
            self.payload = payload
            self.saved = []
            self.written = []

        @classmethod
        def empty(cls, order: int):
            return cls({"order": order})

        @classmethod
        def from_healpix_cells(cls, *args, **kwargs):
            cls.from_calls.append((args, kwargs))
            if cls.fail_next:
                cls.fail_next = False
                raise RuntimeError("boom")
            return cls({"args": args, "kwargs": kwargs})

        def save(self, path: str, format: str | None = None):
            if self.save_raises is not None:
                err = self.save_raises
                if isinstance(err, TypeError) and getattr(self, "_raised_once", False):
                    self.save_raises = None
                else:
                    self._raised_once = True
                    raise err
            if format is None:
                raise TypeError("need format kw")
            self.saved.append((path, format))
            Path(path).write_bytes(b"")

        def write(self, path: Path, overwrite: bool = False):
            self.written.append((path, overwrite))

        def serialize(self, format: str = "json"):
            return self.serialize_value

    monkeypatch.setattr(io_output, "MOC", FakeMOC)

    # Empty path uses MOC.empty and json.dump branch (dict)
    FakeMOC.serialize_value = {"empty": True}
    write_moc(tmp_path, moc_order=1, dens_counts=np.array([], dtype="int64"))
    assert (tmp_path / "Moc.json").exists()

    # Non-empty path exercises builder fallback, save fallback, and bytes serialization.
    FakeMOC.fail_next = True
    FakeMOC.serialize_value = b"data"
    FakeMOC.save_raises = TypeError("positional format unsupported")
    FakeMOC.from_calls = []

    write_moc(tmp_path, moc_order=1, dens_counts=np.array([0, 1], dtype="int64"))
    assert (tmp_path / "Moc.fits").exists()
    assert (tmp_path / "Moc.json").read_text(encoding="utf-8") == "data"
    assert FakeMOC.from_calls and {"ipix", "depth", "max_depth"} <= set(FakeMOC.from_calls[0][1])

    # Force outer save exception to hit write(...)
    FakeMOC.save_raises = ValueError("force write")
    FakeMOC.serialize_value = "text"
    write_moc(tmp_path, moc_order=1, dens_counts=np.array([0, 1], dtype="int64"))
    fits_path = tmp_path / "Moc.fits"
    assert fits_path.exists() and fits_path.stat().st_size >= 0

    # Direct call to cover format=None guard.
    FakeMOC.save_raises = None
    fake = FakeMOC({"dummy": True})
    with pytest.raises(TypeError):
        fake.save(str(tmp_path / "x"), None)

    # If all builders fail, a RuntimeError is raised.
    def always_fail(cls, *args, **kwargs):  # type: ignore[unused-arg]
        raise RuntimeError("no builders")

    monkeypatch.setattr(FakeMOC, "from_healpix_cells", classmethod(always_fail))
    with pytest.raises(RuntimeError):
        write_moc(tmp_path, moc_order=1, dens_counts=np.array([5], dtype="int64"))


def test_write_densmap_fits(tmp_path):
    """densmap files are only written for depths < 13."""
    counts = np.array([1, 2, 3], dtype="int64")
    write_densmap_fits(tmp_path, depth=1, counts=counts)
    assert (tmp_path / "densmap_o1.fits").exists()

    write_densmap_fits(tmp_path, depth=13, counts=counts)
    assert not (tmp_path / "densmap_o13.fits").exists()
