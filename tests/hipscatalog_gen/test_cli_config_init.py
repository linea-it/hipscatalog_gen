"""Unit tests for hipscatalog_gen CLI, config parsing, and package init."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import runpy
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hipscatalog_gen import cli
from hipscatalog_gen.config import Config, display_available_configs, load_config, load_config_from_dict


def _base_cfg_dict(selection_mode: str = "mag_global") -> dict[str, Any]:
    """Build a minimal valid configuration mapping."""
    algo_block: dict[str, Any] = {
        "selection_mode": selection_mode,
        "level_limit": 3,
        "moc_order": 5,
    }
    if selection_mode == "mag_global":
        algo_block["mag_global"] = {"mag_column": "MAG"}
    elif selection_mode == "score_global":
        algo_block["score_global"] = {"score_column": "SCORE"}
    else:
        algo_block["score_density_hybrid"] = {"score_column": "SCORE"}

    return {
        "input": {"paths": ["/tmp/data.parquet"], "format": "parquet", "header": True},
        "columns": {"ra": "RA", "dec": "DEC"},
        "algorithm": algo_block,
        "cluster": {},
        "output": {"out_dir": "/tmp/out", "cat_name": "cat"},
    }


# =============================================================================
# Config helpers and validation
# =============================================================================


def test_display_available_configs_outputs(capsys):
    """display_available_configs prints help text."""
    display_available_configs()
    out = capsys.readouterr().out
    assert "input" in out and "algorithm" in out


def test_load_config_from_file(tmp_path):
    """load_config reads YAML and returns Config."""
    cfg_dict = _base_cfg_dict()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg_dict), encoding="utf-8")
    cfg = load_config(str(cfg_path))
    assert isinstance(cfg, Config)
    assert cfg.algorithm.moc_order == cfg.algorithm.level_limit == 3
    assert cfg.cluster.persist_ddfs is False
    assert cfg.cluster.avoid_computes_wherever_possible is True


def test_build_config_low_memory_overrides():
    """Cluster policy derived from low_memory_mode can be overridden."""
    cfg_dict = _base_cfg_dict()
    cfg_dict["cluster"] = {"low_memory_mode": False}
    cfg = load_config_from_dict(cfg_dict)
    assert cfg.cluster.low_memory_mode is False
    assert cfg.cluster.persist_ddfs is True
    assert cfg.cluster.avoid_computes_wherever_possible is False

    cfg_dict["cluster"]["persist_ddfs"] = False
    cfg = load_config_from_dict(cfg_dict)
    assert cfg.cluster.persist_ddfs is False


def test_build_config_moc_order_clamped():
    """moc_order greater than level_limit is clamped."""
    cfg_dict = _base_cfg_dict()
    cfg_dict["algorithm"]["moc_order"] = 9
    cfg = load_config_from_dict(cfg_dict)
    assert cfg.algorithm.moc_order == cfg.algorithm.level_limit == 3


def test_build_config_requires_selection_mode():
    """Missing or invalid selection_mode raises."""
    cfg_dict = _base_cfg_dict()
    del cfg_dict["algorithm"]["selection_mode"]
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    cfg_dict = _base_cfg_dict()
    cfg_dict["algorithm"]["selection_mode"] = "bad"
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)


def test_build_config_mag_global_validations():
    """mag_global specific validation errors."""
    # mag_column and flux_column are mutually exclusive
    cfg_dict = _base_cfg_dict()
    cfg_dict["algorithm"]["mag_global"]["flux_column"] = "FLUX"
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    # Missing mag/flux
    cfg_dict = _base_cfg_dict()
    cfg_dict["algorithm"]["mag_global"] = {}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    # flux_column without mag_offset
    cfg_dict = _base_cfg_dict()
    cfg_dict["algorithm"]["mag_global"] = {"flux_column": "FLUX"}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    # invalid adaptive_range
    cfg_dict = _base_cfg_dict()
    cfg_dict["algorithm"]["mag_global"] = {"mag_column": "MAG", "adaptive_range": "weird"}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    # out-of-order n_* values
    cfg_dict = _base_cfg_dict()
    cfg_dict["algorithm"]["mag_global"] = {"mag_column": "MAG", "n_2": 5}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    # negative k_1
    cfg_dict = _base_cfg_dict()
    cfg_dict["algorithm"]["mag_global"] = {"mag_column": "MAG", "k_1": -1}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)


def test_build_config_score_global_validations():
    """score_global validation errors."""
    cfg_dict = _base_cfg_dict("score_global")
    cfg_dict["algorithm"]["score_global"] = {}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    cfg_dict = _base_cfg_dict("score_global")
    cfg_dict["algorithm"]["score_global"] = {"score_column": "S", "hist_nbins": 0}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    cfg_dict = _base_cfg_dict("score_global")
    cfg_dict["algorithm"]["score_global"] = {"score_column": "S", "adaptive_range": "bad"}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)


def test_build_config_sdh_validations():
    """score_density_hybrid validation errors."""
    cfg_dict = _base_cfg_dict("score_density_hybrid")
    cfg_dict["algorithm"]["score_density_hybrid"] = {}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    cfg_dict = _base_cfg_dict("score_density_hybrid")
    cfg_dict["algorithm"]["score_density_hybrid"] = {"score_column": "S", "adaptive_range": "bad"}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    cfg_dict = _base_cfg_dict("score_density_hybrid")
    cfg_dict["algorithm"]["score_density_hybrid"] = {"score_column": "S", "hist_nbins": -1}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    cfg_dict = _base_cfg_dict("score_density_hybrid")
    cfg_dict["algorithm"]["score_density_hybrid"] = {"score_column": "S", "density_bias_n1": 1.5}
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    cfg_dict = _base_cfg_dict("score_density_hybrid")
    cfg_dict["algorithm"]["score_density_hybrid"] = {
        "score_column": "S",
        "adaptive_range": "hist_peak",
        "hist_nbins": 10,
        "density_bias_n1": 0.5,
    }
    cfg = load_config_from_dict(cfg_dict)
    assert cfg.algorithm.sdh_density_bias_n1 == 0.5


def test_build_config_numeric_fields_convert_and_raise():
    """_to_int_or_none handles string conversion and invalid values."""
    cfg_dict = _base_cfg_dict()
    cfg_dict["algorithm"]["mag_global"]["k_1"] = "abc"
    with pytest.raises(ValueError):
        load_config_from_dict(cfg_dict)

    cfg_dict = _base_cfg_dict()
    cfg_dict["algorithm"]["mag_global"]["k_1"] = "5"
    cfg = load_config_from_dict(cfg_dict)
    assert cfg.algorithm.k_1 == 5


# =============================================================================
# Package __init__.py
# =============================================================================


def test_version_fallback(monkeypatch):
    """__version__ falls back to local pyproject when metadata is missing."""

    def _raise(*_args, **_kwargs):
        raise importlib.metadata.PackageNotFoundError("x")

    monkeypatch.setattr("importlib.metadata.version", _raise)
    mod = importlib.reload(importlib.import_module("hipscatalog_gen"))
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as f:
        project_version = tomllib.load(f)["project"]["version"]
    assert getattr(mod, "__version__", "") == str(project_version)


def test_version_and_run_pipeline(monkeypatch):
    """__version__ is read from metadata and run_pipeline proxies to pipeline.main."""
    monkeypatch.setattr("importlib.metadata.version", lambda *_: "1.2.3")

    dummy_run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    dummy_pipeline = SimpleNamespace(run_pipeline=lambda *a, **k: dummy_run_calls.append((a, k)) or "ok")
    sys.modules["hipscatalog_gen.pipeline.main"] = dummy_pipeline

    mod = importlib.reload(importlib.import_module("hipscatalog_gen"))
    assert mod.__version__ == "1.2.3"
    result = mod.run_pipeline(1, test=True)
    assert result == "ok"
    assert dummy_run_calls == [((1,), {"test": True})]


# =============================================================================
# CLI
# =============================================================================


def test_cli_list_modes(monkeypatch, capsys):
    """--list-modes prints available modes and exits."""
    dummy_mode = SimpleNamespace(description="d1")
    monkeypatch.setattr("hipscatalog_gen.pipeline.modes.MODE_REGISTRY", {"x": dummy_mode, "y": dummy_mode})
    cli.main(["--list-modes"])
    out = capsys.readouterr().out
    assert "x: d1" in out and "y: d1" in out


def test_cli_check_config(monkeypatch, capsys):
    """--check-config validates config and prints success."""
    cfg = SimpleNamespace(algorithm=SimpleNamespace(selection_mode="score_global"))
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    called: list[str] = []

    def _mk(name: str):
        return lambda *_: called.append(name)

    for name in [
        "validate_common_cfg",
        "validate_mag_global_cfg",
        "validate_score_density_hybrid_cfg",
        "validate_score_global_cfg",
    ]:
        monkeypatch.setattr(f"hipscatalog_gen.pipeline.validation.{name}", _mk(name))

    cli.main(["--check-config", "cfg.yaml"])
    out = capsys.readouterr().out
    assert "Configuration is valid." in out
    assert "validate_common_cfg" in called
    assert "validate_score_global_cfg" in called


def test_cli_check_config_mag_global(monkeypatch):
    """check-config dispatches to mag_global validator."""
    cfg = SimpleNamespace(algorithm=SimpleNamespace(selection_mode="mag_global"))
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    called: list[str] = []

    def _mk(name: str):
        return lambda *_: called.append(name)

    monkeypatch.setattr("hipscatalog_gen.pipeline.validation.validate_common_cfg", _mk("common"))
    monkeypatch.setattr("hipscatalog_gen.pipeline.validation.validate_mag_global_cfg", _mk("mag"))

    cli.main(["--check-config", "cfg.yaml"])
    assert called == ["common", "mag"]


def test_cli_check_config_invalid_mode(monkeypatch):
    """Unsupported selection_mode triggers ValueError."""
    cfg = SimpleNamespace(algorithm=SimpleNamespace(selection_mode="unknown"))
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    with pytest.raises(ValueError):
        cli.main(["--check-config", "cfg.yaml"])


def test_cli_check_config_invalid_mode_after_validation(monkeypatch):
    """Unsupported selection_mode after validation path raises."""
    cfg = SimpleNamespace(algorithm=SimpleNamespace(selection_mode="weird"))
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    for name in [
        "validate_common_cfg",
        "validate_mag_global_cfg",
        "validate_score_density_hybrid_cfg",
        "validate_score_global_cfg",
    ]:
        monkeypatch.setattr(f"hipscatalog_gen.pipeline.validation.{name}", lambda *_: None)

    with pytest.raises(ValueError):
        cli.main(["--check-config", "cfg.yaml"])


def test_cli_check_config_score_density_hybrid(monkeypatch):
    """check-config dispatches to score_density_hybrid validator."""
    cfg = SimpleNamespace(algorithm=SimpleNamespace(selection_mode="score_density_hybrid"))
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    called: list[str] = []

    def _mk(name: str):
        return lambda *_: called.append(name)

    monkeypatch.setattr("hipscatalog_gen.pipeline.validation.validate_common_cfg", _mk("common"))
    monkeypatch.setattr("hipscatalog_gen.pipeline.validation.validate_score_density_hybrid_cfg", _mk("sdh"))

    cli.main(["--check-config", "cfg.yaml"])
    assert called == ["common", "sdh"]


def test_cli_telemetry(tmp_path, capsys):
    """--telemetry prints summary fields and top stages."""
    data = {
        "selection_mode": "mag_global",
        "input_rows": 10,
        "output_rows": 5,
        "total_duration_s": 1.5,
        "stages": {"a": {"duration_s": 2.0}, "b": {"duration_s": 1.0}},
    }
    tfile = tmp_path / "telemetry.json"
    tfile.write_text(json.dumps(data), encoding="utf-8")

    cli.main(["--telemetry", str(tfile)])
    out = capsys.readouterr().out
    assert "selection_mode: mag_global" in out
    assert "top_stages:" in out and "a: 2.0s" in out


def test_cli_run_pipeline(monkeypatch):
    """--config runs the pipeline with parsed options."""
    cfg_obj = SimpleNamespace()
    monkeypatch.setattr(cli, "load_config", lambda path: cfg_obj)

    calls: list[tuple[Any, Any]] = []

    def fake_run(cfg, json_logs=False):
        calls.append((cfg, json_logs))

    sys.modules["hipscatalog_gen.pipeline.main"] = SimpleNamespace(run_pipeline=fake_run)
    cli.main(["--config", "cfg.yaml", "--json-logs"])
    assert calls == [(cfg_obj, True)]


def test_cli_main_entrypoint(monkeypatch, capsys):
    """Module __main__ executes main() using sys.argv when argv is None."""
    dummy_mode = SimpleNamespace(description="d1")
    monkeypatch.setattr("hipscatalog_gen.pipeline.modes.MODE_REGISTRY", {"m": dummy_mode})
    monkeypatch.setattr(sys, "argv", ["hipscatalog_gen.cli", "--list-modes"])
    runpy.run_module("hipscatalog_gen.cli", run_name="__main__")
    out = capsys.readouterr().out
    assert "m: d1" in out


def test_cli_serve_dispatch(monkeypatch):
    """serve subcommand dispatches to local HTTP server helper."""
    calls: list[dict[str, Any]] = []

    def fake_serve(out_dir, host, port, open_browser):
        calls.append(
            {
                "out_dir": out_dir,
                "host": host,
                "port": port,
                "open_browser": open_browser,
            }
        )

    monkeypatch.setattr(cli, "_serve_output_dir", fake_serve)
    cli.main(["serve", "--out", "/tmp/out"])
    assert calls == [{"out_dir": "/tmp/out", "host": "127.0.0.1", "port": 8000, "open_browser": True}]


def test_cli_serve_dispatch_custom_flags(monkeypatch):
    """serve subcommand forwards host/port/no-browser options."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "_serve_output_dir",
        lambda out_dir, host, port, open_browser: calls.append(
            {"out_dir": out_dir, "host": host, "port": port, "open_browser": open_browser}
        ),
    )
    cli.main(["serve", "--out", "/tmp/out", "--host", "127.0.0.2", "--port", "9001", "--no-browser"])
    assert calls == [{"out_dir": "/tmp/out", "host": "127.0.0.2", "port": 9001, "open_browser": False}]


def test_serve_output_dir_rejects_missing_path(tmp_path):
    """_serve_output_dir requires an existing directory."""
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="existing directory"):
        cli._serve_output_dir(str(missing), open_browser=False)


def test_serve_output_dir_keyboard_interrupt_closes_server(monkeypatch, tmp_path, capsys):
    """Server helper closes HTTP server on Ctrl+C and prints status lines."""
    created: list[Any] = []

    class _FakeHTTPServer:
        def __init__(self, addr, handler):
            self.addr = addr
            self.handler = handler
            self.closed = False
            created.append(self)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            self.closed = True

    monkeypatch.setattr(cli, "ThreadingHTTPServer", _FakeHTTPServer)
    monkeypatch.setattr(
        cli.webbrowser, "open", lambda _url: (_ for _ in ()).throw(RuntimeError("no-browser"))
    )

    cli._serve_output_dir(str(tmp_path), host="127.0.0.1", port=8765, open_browser=True)

    out = capsys.readouterr().out
    assert "Serving" in out
    assert "Open: http://127.0.0.1:8765/index.html" in out
    assert "Server stopped." in out
    assert created and created[0].closed is True
