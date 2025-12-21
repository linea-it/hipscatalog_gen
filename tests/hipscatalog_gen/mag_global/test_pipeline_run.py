from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import healpy as hp
import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd
from hipscatalog_gen.mag_global import pipeline
from hipscatalog_gen.mag_global.pipeline import run_mag_global_selection


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


def _cfg(mag_min: float, mag_max: float, **overrides) -> SimpleNamespace:
    """Builds a minimal config namespace for mag_global selection."""
    algo_defaults = dict(
        mag_min=mag_min,
        mag_max=mag_max,
        mag_hist_nbins=4,
        level_limit=2,
        mg_order_desc=False,
        n_1=None,
        n_2=None,
        n_3=None,
    )
    algo_defaults.update(overrides)
    cluster_defaults = dict(low_memory_mode=True)
    return SimpleNamespace(
        algorithm=SimpleNamespace(**algo_defaults),
        cluster=SimpleNamespace(**cluster_defaults),
    )


def _densmaps_for_depths(depths: list[int]) -> dict[int, np.ndarray]:
    """Creates simple densmap arrays with positive counts per depth."""
    result: dict[int, np.ndarray] = {}
    for depth in depths:
        nside = 1 << depth
        npix = hp.nside2npix(nside)
        result[depth] = np.full(npix, 5, dtype="int64")
    return result


def _read_data_rows(tsv_path: Path) -> int:
    """Counts data rows in a TSV tile (ignoring the completeness and header lines)."""
    with tsv_path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return max(0, len(lines) - 2)


def test_run_selection_no_objects_creates_no_outputs(tmp_path, diag_ctx, log_capture, monkeypatch):
    """Tests that the selection exits early when no rows fall inside the range."""
    _, log_fn = log_capture
    pdf = pd.DataFrame({"RA": [], "DEC": [], "__mag__": []})
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_min=0.0, mag_max=1.0)
    densmaps = _densmaps_for_depths([1, 2])

    monkeypatch.setattr(
        pipeline,
        "assign_level_edges",
        lambda **_: (np.array([17.0, 18.5, 20.0, 20.0, 20.0]), None),
    )

    run_mag_global_selection(
        remainder_ddf=ddf,
        densmaps=densmaps,
        keep_cols=["RA", "DEC", "__mag__"],
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
        avoid_computes=True,
    )

    assert list(tmp_path.iterdir()) == []


def test_run_selection_writes_tiles_per_depth(monkeypatch, tmp_path, diag_ctx, log_capture):
    """Tests that selection writes TSV tiles and Allsky files per depth."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame(
        {
            "RA": [0.0, 10.0, 20.0],
            "DEC": [0.0, 5.0, -5.0],
            "__mag__": [18.0, 19.0, 22.0],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(mag_min=17.0, mag_max=22.0)
    densmaps = _densmaps_for_depths([1, 2])

    def fake_assign_targets(**kwargs):
        # Force deterministic magnitude slices: [17, 19) and [19, 22].
        return np.array([17.0, 19.0, 22.0], dtype="float64")

    monkeypatch.setattr(pipeline, "assign_level_edges", lambda **_: (fake_assign_targets(), None))

    run_mag_global_selection(
        remainder_ddf=ddf,
        densmaps=densmaps,
        keep_cols=["RA", "DEC", "__mag__"],
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
    )

    depth1_tiles = list((tmp_path / "Norder1").rglob("Npix*.tsv"))
    depth2_tiles = list((tmp_path / "Norder2").rglob("Npix*.tsv"))

    assert depth1_tiles and depth2_tiles
    assert (tmp_path / "Norder1" / "Allsky.tsv").exists()
    assert (tmp_path / "Norder2" / "Allsky.tsv").exists()

    # With slices [17, 19) and [19, 22], expect 1 row in depth1 and 2 rows in depth2.
    depth1_rows = sum(_read_data_rows(p) for p in depth1_tiles)
    depth2_rows = sum(_read_data_rows(p) for p in depth2_tiles)
    assert depth1_rows == 1
    assert depth2_rows == 2

    assert any("per-depth magnitude slices" in msg for msg in logs)


def test_run_selection_skips_empty_depth(monkeypatch, tmp_path, diag_ctx, log_capture):
    """Tests that a depth with no rows is skipped while others write tiles."""
    logs, log_fn = log_capture
    pdf = pd.DataFrame(
        {
            "RA": [0.0, 10.0],
            "DEC": [0.0, 5.0],
            "__mag__": [19.5, 21.0],  # only second slice populated
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(mag_min=18.0, mag_max=22.0)
    densmaps = _densmaps_for_depths([1, 2])

    def fake_assign_targets(**kwargs):
        # Define slices [18, 19) (empty) and [19, 22] (has data).
        return np.array([18.0, 19.0, 22.0], dtype="float64")

    monkeypatch.setattr(pipeline, "assign_level_edges", lambda **_: (fake_assign_targets(), None))

    run_mag_global_selection(
        remainder_ddf=ddf,
        densmaps=densmaps,
        keep_cols=["RA", "DEC", "__mag__"],
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
    )

    depth1_tiles = list((tmp_path / "Norder1").rglob("Npix*.tsv"))
    depth2_tiles = list((tmp_path / "Norder2").rglob("Npix*.tsv"))
    assert not depth1_tiles  # empty slice skipped
    assert depth2_tiles  # populated slice written
    assert any("no rows in magnitude slice" in msg for msg in logs)


def test_run_selection_depth_without_allsky(tmp_path, diag_ctx, log_capture):
    """Tests that depths >2 do not write Allsky.tsv while still writing tiles."""
    _, log_fn = log_capture
    pdf = pd.DataFrame(
        {
            "RA": [0.0, 10.0],
            "DEC": [0.0, 5.0],
            "__mag__": [18.0, 19.0],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_min=17.0, mag_max=20.0, level_limit=3)
    densmaps = _densmaps_for_depths([1, 2, 3])

    run_mag_global_selection(
        remainder_ddf=ddf,
        densmaps=densmaps,
        keep_cols=["RA", "DEC", "__mag__"],
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
    )

    assert (tmp_path / "Norder3").exists()
    assert not (tmp_path / "Norder3" / "Allsky.tsv").exists()


def test_run_selection_passes_order_desc(monkeypatch, tmp_path, diag_ctx, log_capture):
    """Tests that mg_order_desc is forwarded to write_tiles_with_allsky."""
    _, log_fn = log_capture
    pdf = pd.DataFrame(
        {
            "RA": [0.0, 10.0],
            "DEC": [0.0, 5.0],
            "__mag__": [18.0, 19.0],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1)
    cfg = _cfg(mag_min=17.0, mag_max=20.0, level_limit=1, mg_order_desc=True)
    densmaps = _densmaps_for_depths([1])

    calls: list[bool] = []

    def fake_write_tiles_with_allsky(**kwargs):
        calls.append(kwargs.get("order_desc"))
        # mimic return signature: (written_per_ipix, allsky_df)
        return {0: len(pdf)}, None

    monkeypatch.setattr(pipeline, "write_tiles_with_allsky", fake_write_tiles_with_allsky)

    run_mag_global_selection(
        remainder_ddf=ddf,
        densmaps=densmaps,
        keep_cols=["RA", "DEC", "__mag__"],
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
    )

    assert calls == [True]


def test_run_selection_allsky_only_depths_1_2(tmp_path, diag_ctx, log_capture, monkeypatch):
    """Tests that Allsky is written only for depths 1 and 2 even when deeper levels exist."""
    _, log_fn = log_capture
    pdf = pd.DataFrame(
        {
            "RA": [0.0, 10.0, 20.0, 30.0],
            "DEC": [0.0, 5.0, -5.0, 10.0],
            "__mag__": [18.0, 18.5, 19.5, 19.8],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=2)
    cfg = _cfg(mag_min=17.0, mag_max=20.0, level_limit=4)
    densmaps = _densmaps_for_depths([1, 2, 3, 4])

    # Force slices that allocate rows across all depths so tiles exist and Allsky is written for 1/2.
    monkeypatch.setattr(
        pipeline,
        "assign_level_edges",
        lambda **_: (np.array([17.0, 18.1, 19.1, 19.7, 20.0]), None),
    )

    run_mag_global_selection(
        remainder_ddf=ddf,
        densmaps=densmaps,
        keep_cols=["RA", "DEC", "__mag__"],
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
    )

    assert (tmp_path / "Norder1" / "Allsky.tsv").exists()
    assert (tmp_path / "Norder2" / "Allsky.tsv").exists()
    assert not (tmp_path / "Norder3" / "Allsky.tsv").exists()
    assert not (tmp_path / "Norder4" / "Allsky.tsv").exists()
    # tiles should exist for deeper levels if data present
    assert list((tmp_path / "Norder3").rglob("Npix*.tsv"))
    assert list((tmp_path / "Norder4").rglob("Npix*.tsv"))


def test_assign_targets_rescales_fixed_counts(log_capture):
    """Tests that fixed n_1/n_2 exceeding total are rescaled and monotonic edges returned."""
    logs, log_fn = log_capture
    densmaps = {
        1: np.ones(5, dtype="int64"),
        2: np.ones(3, dtype="int64"),
    }
    depths_sel = [1, 2]
    algo = SimpleNamespace(n_1=5, n_2=5, n_3=None)
    cdf_hist = np.array([0.4, 1.0], dtype="float64")
    mag_edges_hist = np.array([10.0, 20.0, 30.0], dtype="float64")

    level_edges, _ = pipeline.assign_level_edges(
        densmaps=densmaps,
        depths_sel=depths_sel,
        fixed_targets={d: getattr(algo, f"n_{d}", None) for d in depths_sel},
        cdf_hist=cdf_hist,
        score_edges_hist=mag_edges_hist,
        score_min=10.0,
        score_max=30.0,
        n_tot_score=2.0,
        log_fn=log_fn,
        label="mag_global",
    )

    assert level_edges.tolist() == [10.0, 20.0, 30.0]
    assert any("Rescaling" in msg for msg in logs)


def test_assign_targets_all_fixed_no_free_bins(log_capture):
    """Tests that when all depths are fixed, free_mask is empty and weights branch is skipped."""
    logs, log_fn = log_capture
    densmaps = {1: np.ones(2, dtype="int64"), 2: np.ones(2, dtype="int64")}
    depths_sel = [1, 2]
    algo = SimpleNamespace(n_1=1, n_2=1, n_3=None)
    cdf_hist = np.array([0.4, 1.0], dtype="float64")  # ensures q=0.5 crosses in second bin
    mag_edges_hist = np.array([0.0, 1.0, 2.0], dtype="float64")

    level_edges, _ = pipeline.assign_level_edges(
        densmaps=densmaps,
        depths_sel=depths_sel,
        fixed_targets={d: getattr(algo, f"n_{d}", None) for d in depths_sel},
        cdf_hist=cdf_hist,
        score_edges_hist=mag_edges_hist,
        score_min=0.0,
        score_max=2.0,
        n_tot_score=2.0,
        log_fn=log_fn,
        label="mag_global",
    )

    assert level_edges.tolist() == [0.0, 1.0, 2.0]
    assert not any("Rescaling" in msg for msg in logs)


def test_run_selection_smoke_with_sample_parquet(tmp_path, diag_ctx, log_capture):
    """Smoke test using a real small parquet sample to ensure pipeline tolerance to real columns."""
    _, log_fn = log_capture
    sample = Path(
        "tests/data/des_dr2_small_sample/des_dr2_spatial_cone_ra38.200_dec-36.000_r1.000_part0.parquet"
    )
    pdf = pd.read_parquet(sample).rename(columns={"MAG_AUTO_I_DERED": "__mag__"})
    ddf = dd.from_pandas(pdf.head(200), npartitions=4)  # keep it small and fast
    cfg = _cfg(mag_min=float(pdf["__mag__"].min()), mag_max=float(pdf["__mag__"].max()))
    densmaps = _densmaps_for_depths([1, 2])

    run_mag_global_selection(
        remainder_ddf=ddf,
        densmaps=densmaps,
        keep_cols=list(pdf.columns),
        ra_col="RA",
        dec_col="DEC",
        cfg=cfg,
        out_dir=tmp_path,
        diag_ctx=diag_ctx,
        log_fn=log_fn,
    )

    # When all mags are equal, only the last depth slice gets data.
    assert (tmp_path / "Norder2").exists()
