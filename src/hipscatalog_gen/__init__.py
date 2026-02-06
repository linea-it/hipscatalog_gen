"""Public entrypoints for the HiPS catalog generation library."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path


def _resolve_version() -> str:
    """Resolve package version from installed metadata or local pyproject."""
    for dist_name in ("hipscatalog-gen", "hipscatalog_gen"):
        try:
            return _pkg_version(dist_name)
        except PackageNotFoundError:
            continue

    try:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        return str(data.get("project", {}).get("version", "0.0.0"))
    except Exception:
        return "0.0.0"


# Expose package version for tests and users.
__version__ = _resolve_version()

from .config import Config, load_config  # noqa: E402


def run_pipeline(*args, **kwargs):
    """Lazily import and invoke the pipeline entrypoint."""
    from .pipeline.main import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


__all__ = [
    "Config",
    "load_config",
    "run_pipeline",
    "__version__",
]
