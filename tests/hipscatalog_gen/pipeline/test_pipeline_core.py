"""Unit tests for hipscatalog_gen.pipeline (structure, validation, orchestration)."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd

from hipscatalog_gen.pipeline import common as pipeline_common
from hipscatalog_gen.pipeline import logging_utils, main, modes, structure, validation


@pytest.fixture
def diag_ctx():
    """Returns a no-op diagnostic context."""
    return lambda name: nullcontext()


@pytest.fixture
def log_capture():
    """Collects log messages emitted by helpers."""
    logs: list[str] = []

    def _log_fn(msg: str, always: bool = False, **_: dict) -> None:
        logs.append(msg)

    return logs, _log_fn


def _cfg_pipeline(tmp_path: Path, **algo_overrides) -> SimpleNamespace:
    """Minimal Config-like namespace for pipeline tests."""
    algo_defaults = dict(
        selection_mode="mag_global",
        level_limit=4,
        moc_order=4,
        mag_column="MAG",
        mag_hist_nbins=4,
        flux_column=None,
        mag_offset=None,
        mag_min=None,
        mag_max=None,
        mag_adaptive_range="complete",
        n_1=None,
        n_2=None,
        n_3=None,
        score_column=None,
        score_min=None,
        score_max=None,
        score_adaptive_range="complete",
        score_hist_nbins=4,
        score_n_1=None,
        score_n_2=None,
        score_n_3=None,
        sdh_score_column=None,
        sdh_score_min=None,
        sdh_score_max=None,
        sdh_score_adaptive_range="complete",
        sdh_score_hist_nbins=4,
        sdh_n_1=None,
        sdh_n_2=None,
        sdh_n_3=None,
        sdh_density_bias_n1=0.0,
        sdh_density_bias_n2=0.0,
        sdh_density_bias_n3=0.0,
        mg_order_desc=False,
        sg_order_desc=False,
        sdh_order_desc=False,
    )
    algo_defaults.update(algo_overrides)
    return SimpleNamespace(
        algorithm=SimpleNamespace(**algo_defaults),
        cluster=SimpleNamespace(
            mode="local",
            n_workers=1,
            threads_per_worker=1,
            memory_per_worker="1GB",
            persist_ddfs=False,
            avoid_computes_wherever_possible=True,
            diagnostics_mode="",
            slurm=None,
        ),
        output=SimpleNamespace(
            out_dir=str(tmp_path),
            cat_name="test",
            target="foo",
            overwrite=True,
            creator_did="creator",
            obs_title="obs",
        ),
        input=SimpleNamespace(
            paths=[str(tmp_path / "dummy.parquet")], format="parquet", header=None, ascii_format=None
        ),
        columns=SimpleNamespace(keep=["RA", "DEC", "MAG"]),
    )


def test_modes_registry_and_get():
    """get_selection_mode resolves valid names and rejects unknown ones."""
    mode = modes.get_selection_mode("mag_global")
    assert mode.name == "mag_global"
    with pytest.raises(ValueError):
        modes.get_selection_mode("unknown")


def test_validation_helpers_errors():
    """Validation functions surface config issues."""
    cfg = SimpleNamespace(algorithm=SimpleNamespace(mag_column="A", flux_column="B", mag_hist_nbins=4))
    with pytest.raises(ValueError):
        validation.validate_mag_global_cfg(cfg)

    cfg_score = SimpleNamespace(algorithm=SimpleNamespace(score_column=None, score_hist_nbins=0))
    with pytest.raises(ValueError):
        validation.validate_score_global_cfg(cfg_score)
    cfg_mag_flux_offset = SimpleNamespace(
        algorithm=SimpleNamespace(
            mag_column=None, flux_column="F", mag_offset=None, mag_hist_nbins=1, n_1=None, k_1=None
        )
    )
    with pytest.raises(ValueError):
        validation.validate_mag_global_cfg(cfg_mag_flux_offset)
    cfg_score_bins = SimpleNamespace(algorithm=SimpleNamespace(score_column="S", score_hist_nbins=0))
    with pytest.raises(ValueError):
        validation.validate_score_global_cfg(cfg_score_bins)

    cfg_sdh = SimpleNamespace(algorithm=SimpleNamespace(sdh_score_column=None, sdh_score_hist_nbins=0))
    with pytest.raises(ValueError):
        validation.validate_score_density_hybrid_cfg(cfg_sdh)
    cfg_sdh_bins = SimpleNamespace(algorithm=SimpleNamespace(sdh_score_column="S", sdh_score_hist_nbins=0))
    with pytest.raises(ValueError):
        validation.validate_score_density_hybrid_cfg(cfg_sdh_bins)

    cfg_common = SimpleNamespace(
        algorithm=SimpleNamespace(level_limit=0, moc_order=1),
        cluster=SimpleNamespace(mode="bad", n_workers=None, threads_per_worker=None, memory_per_worker=None),
        output=SimpleNamespace(out_dir=None, cat_name="", target=None),
    )
    with pytest.raises(ValueError):
        validation.validate_common_cfg(cfg_common)

    # nk negative
    bad_nk = SimpleNamespace(algorithm=SimpleNamespace(n_1=-1, k_1=None, level_limit=4, moc_order=4))
    with pytest.raises(ValueError):
        validation.validate_mag_global_cfg(bad_nk)

    both_nk = SimpleNamespace(
        algorithm=SimpleNamespace(
            n_1=1, k_1=1, level_limit=4, moc_order=4, mag_hist_nbins=4, mag_column="MAG", flux_column=None
        )
    )
    with pytest.raises(ValueError):
        validation.validate_mag_global_cfg(both_nk)

    neg_k = SimpleNamespace(
        algorithm=SimpleNamespace(score_column="S", score_hist_nbins=4, score_k_1=-1, score_n_1=None)
    )
    with pytest.raises(ValueError):
        validation.validate_score_global_cfg(neg_k)
    neg_k2 = SimpleNamespace(algorithm=SimpleNamespace(score_column="S", score_hist_nbins=4, score_k_2=-1))
    with pytest.raises(ValueError):
        validation.validate_score_global_cfg(neg_k2)

    both_sdh = SimpleNamespace(
        algorithm=SimpleNamespace(sdh_score_column="S", sdh_score_hist_nbins=4, sdh_n_1=1, sdh_k_1=1)
    )
    with pytest.raises(ValueError):
        validation.validate_score_density_hybrid_cfg(both_sdh)

    neg_depth2 = SimpleNamespace(
        algorithm=SimpleNamespace(sdh_score_column="S", sdh_score_hist_nbins=4, sdh_n_2=-1, sdh_k_2=None)
    )
    with pytest.raises(ValueError):
        validation.validate_score_density_hybrid_cfg(neg_depth2)
    neg_k3 = SimpleNamespace(
        algorithm=SimpleNamespace(sdh_score_column="S", sdh_score_hist_nbins=4, sdh_k_3=-1)
    )
    with pytest.raises(ValueError):
        validation.validate_score_density_hybrid_cfg(neg_k3)

    cfg_common_fields = SimpleNamespace(
        algorithm=SimpleNamespace(level_limit=1, moc_order=1),
        cluster=SimpleNamespace(
            mode="local", n_workers=None, threads_per_worker=None, memory_per_worker=None
        ),
        output=SimpleNamespace(out_dir="x", cat_name="y", target="z"),
    )
    with pytest.raises(ValueError):
        validation.validate_common_cfg(cfg_common_fields)

    cfg_common_level = SimpleNamespace(
        algorithm=SimpleNamespace(level_limit=0, moc_order=0),
        cluster=SimpleNamespace(mode="local", n_workers=1, threads_per_worker=1, memory_per_worker="1GB"),
        output=SimpleNamespace(out_dir="x", cat_name="y", target="z"),
    )
    with pytest.raises(ValueError):
        validation.validate_common_cfg(cfg_common_level)

    cfg_common_mode = SimpleNamespace(
        algorithm=SimpleNamespace(level_limit=1, moc_order=1),
        cluster=SimpleNamespace(mode="bad", n_workers=1, threads_per_worker=1, memory_per_worker="1GB"),
        output=SimpleNamespace(out_dir="x", cat_name="y", target="z"),
    )
    with pytest.raises(ValueError):
        validation.validate_common_cfg(cfg_common_mode)

    cfg_common_output = SimpleNamespace(
        algorithm=SimpleNamespace(level_limit=1, moc_order=1),
        cluster=SimpleNamespace(mode="local", n_workers=1, threads_per_worker=1, memory_per_worker="1GB"),
        output=SimpleNamespace(out_dir=None, cat_name="", target=""),
    )
    with pytest.raises(ValueError):
        validation.validate_common_cfg(cfg_common_output)


def test_validation_success():
    """Validation functions pass with valid configs."""
    algo = SimpleNamespace(
        mag_column="MAG",
        flux_column=None,
        mag_offset=25.0,
        mag_hist_nbins=4,
        n_1=None,
        k_1=None,
        score_column="SC",
        score_hist_nbins=4,
        score_n_1=None,
        score_k_1=None,
        sdh_score_column="SC",
        sdh_score_hist_nbins=4,
        sdh_n_1=None,
        sdh_k_1=None,
        level_limit=4,
        moc_order=4,
    )
    cfg_ok = SimpleNamespace(
        algorithm=algo,
        cluster=SimpleNamespace(mode="local", n_workers=1, threads_per_worker=1, memory_per_worker="1GB"),
        output=SimpleNamespace(out_dir="out", cat_name="c", target="t"),
    )
    validation.validate_mag_global_cfg(cfg_ok)
    validation.validate_score_global_cfg(cfg_ok)
    validation.validate_score_density_hybrid_cfg(cfg_ok)
    validation.validate_common_cfg(cfg_ok)


def test_logging_utils_writes_files(tmp_path):
    """Structured logger writes to process.log and process.jsonl."""
    log_ctx, log_fn = logging_utils.setup_structured_logger(tmp_path, "mag_global", json_logs=True)
    log_ctx.stage = "stage1"
    log_ctx.depth = 2
    log_fn("hello", always=True)

    assert (tmp_path / "process.log").exists()
    assert (tmp_path / "process.jsonl").exists()
    assert "hello" in (tmp_path / "process.log").read_text()


def test_structure_run_stages_tracks_stage_and_telemetry(diag_ctx, log_capture):
    """run_stages updates log_ctx and telemetry durations."""
    logs, log_fn = log_capture
    log_ctx = logging_utils.LogContext()

    def make_ctx():
        return structure.PipelineContext(
            cfg=_cfg_pipeline(Path("/tmp")),
            out_dir=Path("/tmp"),
            report_dir=Path("/tmp"),
            log_fn=log_fn,
            diag_ctx=diag_ctx,
            persist_ddfs=False,
            avoid_computes=True,
            selection_mode="mag_global",
            log_ctx=log_ctx,
        )

    ctx = make_ctx()
    stages = [
        structure.PipelineStage("one", lambda c: c.with_updates(telemetry={"stages": {}})),
        structure.PipelineStage("two", lambda c: c),
    ]
    final = structure.run_stages(stages, ctx)
    assert final.telemetry["stages"]["one"]["duration_s"] >= 0.0
    assert log_ctx.stage is None


def test_structure_run_stages_none_result(diag_ctx):
    """Stages returning None keep context and still record telemetry."""
    ctx = structure.PipelineContext(
        cfg=_cfg_pipeline(Path("/tmp")),
        out_dir=Path("/tmp"),
        report_dir=Path("/tmp"),
        log_fn=lambda *_, **__: None,
        diag_ctx=diag_ctx,
        persist_ddfs=False,
        avoid_computes=True,
        selection_mode="mag_global",
        log_ctx=None,
    )
    stages = [structure.PipelineStage("noop", lambda c: None)]
    final = structure.run_stages(stages, ctx)
    assert "noop" in final.telemetry.get("stages", {})


def test_structure_run_stages_with_dict_log_ctx(diag_ctx):
    """Dict log_ctx path sets and resets stage."""
    log_ctx = {}
    ctx = structure.PipelineContext(
        cfg=_cfg_pipeline(Path("/tmp")),
        out_dir=Path("/tmp"),
        report_dir=Path("/tmp"),
        log_fn=lambda *_, **__: None,
        diag_ctx=diag_ctx,
        persist_ddfs=False,
        avoid_computes=True,
        selection_mode="mag_global",
        log_ctx=log_ctx,
    )
    structure.run_stages([structure.PipelineStage("x", lambda c: c)], ctx)
    assert log_ctx.get("stage") is None


def test_common_log_prologue_epilogue(tmp_path, log_capture):
    """Prologue/epilogue emit logs and process.log is optionally written."""
    logs, log_fn = log_capture
    cfg = _cfg_pipeline(tmp_path)
    pipeline_common.log_prologue(cfg, tmp_path, log_fn)
    pipeline_common.log_epilogue(tmp_path, ["line1"], 0.0, log_fn, write_process_log=True)
    assert any("START HiPS" in m for m in logs)
    assert any("END HiPS" in m for m in logs)
    assert (tmp_path / "process.log").exists()


def test_log_epilogue_handles_write_error(monkeypatch, tmp_path, log_capture):
    """log_epilogue logs errors when process.log cannot be written."""
    logs, log_fn = log_capture
    cfg = _cfg_pipeline(tmp_path)
    pipeline_common.log_prologue(cfg, tmp_path, log_fn)

    orig_open = Path.open

    def fake_open(self, *args, **kwargs):
        if self.name == "process.log":
            raise OSError("fail")
        return orig_open(self, *args, **kwargs)

    monkeypatch.setattr(pipeline_common.Path, "open", fake_open)
    pipeline_common.log_epilogue(tmp_path, ["x"], 0.0, log_fn, write_process_log=True)
    assert any("ERROR writing process.log" in m for m in logs)
    # Non-process.log path exercises happy branch.
    other = tmp_path / "other.log"
    with other.open("w", encoding="utf-8") as fh:
        fh.write("ok")
    assert fake_open(other) is not None


def test_common_maybe_persist_ddf(diag_ctx, log_capture):
    """maybe_persist_ddf returns input when should_persist=False and persists otherwise."""
    _, log_fn = log_capture
    # Stub dask.distributed.wait to avoid needing a client.
    sys.modules["dask.distributed"] = SimpleNamespace(wait=lambda *_: None)

    class Dummy:
        def __init__(self):
            self.persisted = False

        def persist(self):
            self.persisted = True
            return self

    dummy = Dummy()
    assert pipeline_common.maybe_persist_ddf(dummy, False, diag_ctx, log_fn, log_prefix="x") is dummy
    persisted = pipeline_common.maybe_persist_ddf(dummy, True, diag_ctx, log_fn, log_prefix="x")
    assert persisted.persisted


def test_maybe_persist_import_error(diag_ctx, log_capture, monkeypatch):
    """maybe_persist_ddf handles missing dask.distributed import."""
    _, log_fn = log_capture
    sys.modules["dask.distributed"] = SimpleNamespace(wait=None)

    class Dummy:
        def persist(self):
            return self

    res = pipeline_common.maybe_persist_ddf(Dummy(), True, diag_ctx, log_fn, log_prefix="x")
    assert res is not None


def test_build_and_prepare_input(monkeypatch, tmp_path, diag_ctx, log_capture):
    """build_and_prepare_input loads paths, validates RA/DEC, repartitions, and persists."""
    logs, log_fn = log_capture
    cfg = _cfg_pipeline(tmp_path)
    cfg.input.paths = ["foo*.parquet"]
    sys.modules["dask.distributed"] = SimpleNamespace(wait=lambda *_: None)

    monkeypatch.setattr(pipeline_common, "_collect_input_paths", lambda *_: ["p1"])
    monkeypatch.setattr(pipeline_common, "_warn_if_hats_mismatch", lambda *_, **__: None)

    pdf = pd.DataFrame({"RA": [0.0, 1.0], "DEC": [0.0, 1.0], "X": [1, 2]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    monkeypatch.setattr(
        pipeline_common, "_build_input_ddf", lambda *_, **__: (ddf, "RA", "DEC", ["RA", "DEC", "X"])
    )
    monkeypatch.setattr(pipeline_common, "_validate_and_normalize_radec", lambda ddf_like, **__: ddf_like)

    ddf_out, ra_name, dec_name, keep_cols, is_hats, paths = pipeline_common.build_and_prepare_input(
        cfg, diag_ctx, log_fn, persist_ddfs=True
    )

    assert ra_name == "RA" and dec_name == "DEC"
    assert keep_cols == ["RA", "DEC", "X"]
    assert is_hats is False
    assert paths == ["p1"]
    assert ddf_out.npartitions == 1


def test_compute_input_total(diag_ctx, log_capture):
    """compute_input_total sums rows via dask_compute."""
    _, log_fn = log_capture
    ddf = dd.from_pandas(pd.DataFrame({"A": [1, 2, 3]}), npartitions=2)
    total = pipeline_common.compute_input_total(ddf, diag_ctx, log_fn, avoid_computes=False)
    assert total == 3


def test_compute_input_total_map_partitions(monkeypatch, diag_ctx, log_capture):
    """compute_input_total map_partitions branch and len fallback."""
    _, log_fn = log_capture

    class DummyMap:
        def map_partitions(self, fn, meta=None):
            class DummySum:
                def sum(self):
                    return 5

            return DummySum()

    monkeypatch.setattr(pipeline_common, "_get_dask_base", lambda *_, **__: DummyMap())
    total_mp = pipeline_common.compute_input_total(object(), diag_ctx, log_fn, avoid_computes=True)
    assert total_mp == 5

    class DummyLen:
        def __len__(self):
            return 7

    monkeypatch.setattr(pipeline_common, "_get_dask_base", lambda *_, **__: DummyLen())
    total_len = pipeline_common.compute_input_total(object(), diag_ctx, log_fn, avoid_computes=True)
    assert total_len == 7

    class DummyEmpty:
        pass

    monkeypatch.setattr(pipeline_common, "_get_dask_base", lambda *_, **__: DummyEmpty())
    with pytest.raises(TypeError):
        pipeline_common.compute_input_total(object(), diag_ctx, log_fn, avoid_computes=True)


def test_collect_paths_and_warn(tmp_path, log_capture, monkeypatch):
    """_collect_input_paths expands globs and _warn_if_hats_mismatch logs warning."""
    logs, log_fn = log_capture
    p = tmp_path / "a.txt"
    p.write_text("x")
    cfg = SimpleNamespace(input=SimpleNamespace(paths=[str(tmp_path / "*.txt")], format="parquet"))

    paths = pipeline_common._collect_input_paths(cfg, log_fn)
    assert paths == [str(p)]
    monkeypatch.setattr(pipeline_common, "_detect_hats_catalog_root", lambda *_: tmp_path)
    pipeline_common._warn_if_hats_mismatch(paths, cfg, log_fn)
    assert any("HATS catalog layout" in m for m in logs)


def test_write_common_static_products(monkeypatch, tmp_path):
    """write_common_static_products forwards to writers."""
    calls = []
    monkeypatch.setattr(pipeline_common, "write_arguments", lambda *args, **kwargs: calls.append("args"))
    monkeypatch.setattr(pipeline_common, "write_metadata_xml", lambda *args, **kwargs: calls.append("meta"))
    monkeypatch.setattr(pipeline_common, "write_index_html", lambda *args, **kwargs: calls.append("index"))
    monkeypatch.setattr(pipeline_common, "write_moc", lambda *args, **kwargs: calls.append("moc"))
    pdf = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "MAG": [1.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    densmaps = {4: np.ones(1, dtype="int64")}
    pipeline_common.write_common_static_products(
        out_dir=tmp_path,
        cfg=_cfg_pipeline(tmp_path),
        densmaps=densmaps,
        keep_cols=["RA", "DEC", "MAG"],
        ra_col="RA",
        dec_col="DEC",
        paths=["p1"],
        ddf=ddf,
    )
    assert set(calls) == {"args", "meta", "index", "moc"}


def test_compute_and_write_densmaps(monkeypatch, tmp_path, diag_ctx):
    """compute_and_write_densmaps runs densmap calculation and writes FITS placeholders."""
    pdf = pd.DataFrame({"RA": [0.0], "DEC": [0.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    monkeypatch.setattr(
        pipeline_common,
        "densmap_for_depth_delayed",
        lambda *args, **kwargs: pipeline_common.dask_delayed(lambda: np.ones(1, dtype="int64"))(),
    )
    calls: list[int] = []
    monkeypatch.setattr(pipeline_common, "write_densmap_fits", lambda *args, **kwargs: calls.append(1))

    densmaps = pipeline_common.compute_and_write_densmaps(
        ddf, "RA", "DEC", level_limit=1, out_dir=tmp_path, diag_ctx=diag_ctx
    )
    assert 0 in densmaps and 1 in densmaps
    assert calls  # write_densmap_fits called


def test_write_counts_summaries(tmp_path, log_capture):
    """write_counts_summaries reads tile files and aggregates counts."""
    _, log_fn = log_capture
    depth_dir = tmp_path / "Norder1"
    depth_dir.mkdir()
    tile_path = depth_dir / "Npix0.tsv"
    tile_path.write_text("comp\nheader\n1\t2\n3\t4\n", encoding="utf-8")
    # invalid name branches
    (depth_dir / "badname.txt").write_text("x", encoding="utf-8")
    (depth_dir / "Npixbad.tsv").write_text("x", encoding="utf-8")

    total_written, payload = pipeline_common.write_counts_summaries(
        tmp_path, level_limit=2, input_total=5, log_fn=log_fn
    )
    assert total_written == 2
    assert payload["output"]["total"] == 2
    assert payload["input"]["total"] == 5
    assert any("Total rows written: 2" in m for m in log_capture[0])


def test_write_counts_summaries_uses_precomputed_depth_totals(tmp_path, log_capture):
    """Precomputed depth totals skip tile scanning and preserve totals payload."""
    logs, log_fn = log_capture
    # No tile files are required for this path.
    total_written, payload = pipeline_common.write_counts_summaries(
        tmp_path,
        level_limit=4,
        input_total=10,
        log_fn=log_fn,
        precomputed_depth_totals={"3": 7, "4": 11, "9": 99},  # depth 9 out of range -> ignored
    )

    assert total_written == 18
    assert payload["output"]["total"] == 18
    assert payload["output"]["depth_totals"] == {"3": 7, "4": 11}
    assert payload["output"]["depths"] == {}
    assert payload["input"]["total"] == 10
    assert any("Using precomputed output counts" in m for m in logs)


def test_write_common_static_products_arguments_include_all_input_keys(tmp_path):
    """arguments file includes all known YAML input keys, even when unset."""
    cfg = _cfg_pipeline(tmp_path)
    pdf = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "MAG": [1.0]})
    ddf = dd.from_pandas(pdf, npartitions=1)
    densmaps = {4: np.ones(1, dtype="int64")}

    pipeline_common.write_common_static_products(
        out_dir=tmp_path,
        cfg=cfg,
        densmaps=densmaps,
        keep_cols=["RA", "DEC", "MAG"],
        ra_col="RA",
        dec_col="DEC",
        paths=["p1"],
        ddf=ddf,
    )

    args_text = (tmp_path / "arguments").read_text(encoding="utf-8")
    assert "algorithm.selection_defaults.hist_nbins: null" in args_text
    assert "mag_global.k_1: null" in args_text
    assert "score_global.k_1: null" in args_text
    assert "score_density_hybrid.k_1: null" in args_text
    assert "cluster.low_memory_mode: null" in args_text
    assert "output.overwrite: true" in args_text


def test_write_tiles_with_allsky(monkeypatch, tmp_path, log_capture):
    """write_tiles_with_allsky forwards to finalize_write_tiles and optionally writes allsky."""
    _, log_fn = log_capture
    df = pd.DataFrame({"RA": [0.0], "DEC": [0.0]})
    monkeypatch.setattr(
        pipeline_common,
        "finalize_write_tiles",
        lambda **kwargs: ({"0": 1}, df.copy()),
    )
    captured = []
    monkeypatch.setattr(pipeline_common, "write_allsky", lambda *args, **kwargs: captured.append(1))

    written, allsky_df = pipeline_common.write_tiles_with_allsky(
        out_dir=tmp_path,
        depth=1,
        header_line="h\n",
        ra_col="RA",
        dec_col="DEC",
        counts=np.ones(1, dtype="int64"),
        selected=df,
        order_desc=False,
        allsky_needed=True,
        log_fn=log_fn,
    )
    assert written == {"0": 1}
    assert allsky_df is not None
    assert captured  # write_allsky called


def test_write_allsky_outputs(tmp_path):
    """write_allsky writes Allsky.tsv with completeness header."""
    counts = np.array([2, 0], dtype="int64")
    allsky_df = pd.DataFrame({"RA": [0.0, 1.0], "DEC": [0.0, 1.0], "NAME": ["a\t", "b\n"]})
    pipeline_common.write_allsky(
        tmp_path, depth=1, header_line="RA\tDEC\n", counts=counts, allsky_df=allsky_df, nwritten_tot=2
    )
    out_file = tmp_path / "Norder1" / "Allsky.tsv"
    assert out_file.exists()
    content = out_file.read_text()
    first_line = content.splitlines()[0]
    assert "Completeness" in first_line


def test_write_allsky_sanitizes_object_columns(tmp_path):
    """write_allsky replaces tabs/newlines in object columns."""
    counts = np.array([1], dtype="int64")
    allsky_df = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "TXT": ["foo\tbar"]})
    pipeline_common.write_allsky(
        tmp_path, depth=2, header_line="RA\tDEC\tTXT\n", counts=counts, allsky_df=allsky_df, nwritten_tot=1
    )
    content = (tmp_path / "Norder2" / "Allsky.tsv").read_text()
    assert "foo bar" in content


def test_run_pipeline_happy_path(monkeypatch, tmp_path, log_capture):
    """run_pipeline orchestrates stages with mocks."""
    logs, log_fn = log_capture
    cfg = _cfg_pipeline(tmp_path, selection_mode="mag_global")

    dummy_ddf = dd.from_pandas(pd.DataFrame({"RA": [0.0], "DEC": [0.0], "MAG": [1.0]}), npartitions=1)

    # Mock selection mode entry
    class DummyMode:
        def __init__(self):
            self.normalize_called = False
            self.prepare_called = False
            self.run_called = False

        def normalize_fn(self, ddf, cfg, diag_ctx, log_fn, persist_ddfs, avoid_computes):
            self.normalize_called = True
            return ddf, SimpleNamespace(mag_min=0.0, mag_max=2.0, sentinel=None)

        def prepare_fn(self, ddf, cfg, diag_ctx, log_fn, params, **kwargs):
            self.prepare_called = True
            return ddf

        def run_fn(self, **kwargs):
            self.run_called = True
            return {"depth_totals": {"1": 1}, "depth_tiles": {"1": 1}}

    dummy_mode = DummyMode()
    monkeypatch.setattr(
        main,
        "get_selection_mode",
        lambda name: SimpleNamespace(
            **{
                "validate_fn": lambda cfg: None,
                "normalize_fn": dummy_mode.normalize_fn,
                "prepare_fn": dummy_mode.prepare_fn,
                "run_fn": dummy_mode.run_fn,
            }
        ),
    )

    monkeypatch.setattr(main, "setup_structured_logger", lambda *args, **kwargs: (None, log_fn))
    monkeypatch.setattr(
        main,
        "setup_cluster",
        lambda cfg, report_dir, log_fn: (
            SimpleNamespace(persist_ddfs=False, avoid_computes=True, diagnostics_mode=""),
            lambda name: nullcontext(),
        ),
    )
    monkeypatch.setattr(main, "shutdown_cluster", lambda runtime: None)
    monkeypatch.setattr(main, "validate_common_cfg", lambda cfg: None)
    monkeypatch.setattr(
        main,
        "build_and_prepare_input",
        lambda *_, **__: (dummy_ddf, "RA", "DEC", ["RA", "DEC"], False, ["p"]),
    )
    monkeypatch.setattr(main, "compute_input_total", lambda *_, **__: 1)
    monkeypatch.setattr(main, "compute_and_write_densmaps", lambda *_, **__: {1: np.ones(1, dtype="int64")})
    monkeypatch.setattr(main, "write_common_static_products", lambda *_, **__: None)
    captured_counts_kwargs: dict[str, object] = {}

    def fake_write_counts_summaries(*args, **kwargs):
        captured_counts_kwargs.update(kwargs)
        return (1, {"output": {}, "input": {}})

    monkeypatch.setattr(main, "write_counts_summaries", fake_write_counts_summaries)
    monkeypatch.setattr(main, "write_properties", lambda *_, **__: None)

    main.run_pipeline(cfg)
    assert dummy_mode.normalize_called and dummy_mode.prepare_called and dummy_mode.run_called
    assert captured_counts_kwargs.get("precomputed_depth_totals") == {"1": 1}
    assert any("START HiPS" in m for m in logs)


def test_run_pipeline_overwrite_file(monkeypatch, tmp_path, log_capture):
    """overwrite=True with existing file path deletes it and proceeds."""
    logs, log_fn = log_capture
    out_file = tmp_path / "existing"
    out_file.write_text("x")
    cfg = _cfg_pipeline(out_file, selection_mode="mag_global")
    cfg.output.overwrite = True

    dummy_ddf = dd.from_pandas(pd.DataFrame({"RA": [0.0], "DEC": [0.0], "MAG": [1.0]}), npartitions=1)
    monkeypatch.setattr(
        main,
        "get_selection_mode",
        lambda name: SimpleNamespace(
            **{
                "validate_fn": lambda cfg: None,
                "normalize_fn": lambda *args, **kwargs: (
                    dummy_ddf,
                    SimpleNamespace(mag_min=0.0, mag_max=1.0, sentinel=None),
                ),
                "prepare_fn": lambda *args, **kwargs: dummy_ddf,
                "run_fn": lambda **kwargs: None,
            }
        ),
    )
    monkeypatch.setattr(main, "setup_structured_logger", lambda *args, **kwargs: (None, log_fn))
    monkeypatch.setattr(
        main,
        "setup_cluster",
        lambda cfg, report_dir, log_fn: (
            SimpleNamespace(persist_ddfs=False, avoid_computes=True, diagnostics_mode=""),
            lambda name: nullcontext(),
        ),
    )
    monkeypatch.setattr(main, "shutdown_cluster", lambda runtime: None)
    monkeypatch.setattr(main, "validate_common_cfg", lambda cfg: None)
    monkeypatch.setattr(
        main,
        "build_and_prepare_input",
        lambda *_, **__: (dummy_ddf, "RA", "DEC", ["RA", "DEC"], False, ["p"]),
    )
    monkeypatch.setattr(main, "compute_input_total", lambda *_, **__: 1)
    monkeypatch.setattr(main, "compute_and_write_densmaps", lambda *_, **__: {1: np.ones(1, dtype="int64")})
    monkeypatch.setattr(main, "write_common_static_products", lambda *_, **__: None)
    monkeypatch.setattr(main, "write_counts_summaries", lambda *_, **__: (1, {"output": {}, "input": {}}))
    monkeypatch.setattr(main, "write_properties", lambda *_, **__: None)

    main.run_pipeline(cfg)


def test_run_pipeline_diagnostics_global(monkeypatch, tmp_path, log_capture):
    """diagnostics_mode='global' uses performance_report context."""
    logs, log_fn = log_capture
    cfg = _cfg_pipeline(tmp_path, selection_mode="mag_global")
    dummy_ddf = dd.from_pandas(pd.DataFrame({"RA": [0.0], "DEC": [0.0], "MAG": [1.0]}), npartitions=1)

    monkeypatch.setattr(
        main,
        "get_selection_mode",
        lambda name: SimpleNamespace(
            **{
                "validate_fn": lambda cfg: None,
                "normalize_fn": lambda *args, **kwargs: (
                    dummy_ddf,
                    SimpleNamespace(mag_min=0.0, mag_max=1.0, sentinel=None),
                ),
                "prepare_fn": lambda *args, **kwargs: dummy_ddf,
                "run_fn": lambda **kwargs: None,
            }
        ),
    )
    monkeypatch.setattr(main, "setup_structured_logger", lambda *args, **kwargs: (None, log_fn))

    class DummyReport:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    sys.modules["dask.distributed"] = SimpleNamespace(performance_report=lambda filename: DummyReport())
    monkeypatch.setattr(
        main,
        "setup_cluster",
        lambda cfg, report_dir, log_fn: (
            SimpleNamespace(persist_ddfs=False, avoid_computes=True, diagnostics_mode="global"),
            lambda name: nullcontext(),
        ),
    )
    monkeypatch.setattr(main, "shutdown_cluster", lambda runtime: None)
    monkeypatch.setattr(main, "validate_common_cfg", lambda cfg: None)
    monkeypatch.setattr(
        main,
        "build_and_prepare_input",
        lambda *_, **__: (dummy_ddf, "RA", "DEC", ["RA", "DEC"], False, ["p"]),
    )
    monkeypatch.setattr(main, "compute_input_total", lambda *_, **__: 1)
    monkeypatch.setattr(main, "compute_and_write_densmaps", lambda *_, **__: {1: np.ones(1, dtype="int64")})
    monkeypatch.setattr(main, "write_common_static_products", lambda *_, **__: None)
    monkeypatch.setattr(main, "write_counts_summaries", lambda *_, **__: (1, {"output": {}, "input": {}}))
    monkeypatch.setattr(main, "write_properties", lambda *_, **__: None)

    main.run_pipeline(cfg)


def test_run_pipeline_overwrite_guard(tmp_path):
    """run_pipeline raises when out_dir exists and overwrite is False."""
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    cfg = _cfg_pipeline(out_dir, selection_mode="mag_global")
    cfg.output.overwrite = False
    with pytest.raises(ValueError):
        main.run_pipeline(cfg)


def test_run_pipeline_invalid_mode(tmp_path):
    """Unsupported selection_mode raises ValueError."""
    cfg = _cfg_pipeline(tmp_path, selection_mode="unknown")
    with pytest.raises(ValueError):
        main.run_pipeline(cfg)


def test_run_pipeline_level_limit_guard(tmp_path):
    """Invalid level_limit outside [4,11] raises."""
    cfg = _cfg_pipeline(tmp_path, selection_mode="mag_global", level_limit=2)
    with pytest.raises(ValueError):
        main.run_pipeline(cfg)
