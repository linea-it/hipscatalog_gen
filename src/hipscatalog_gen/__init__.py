from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Expose package version for tests and users.
# When the package is not installed (editable/dev mode), fall back gracefully.
try:
    __version__ = _pkg_version("hipscatalog_gen")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .config import Config, load_config  # noqa: E402
from .pipeline import run_pipeline  # noqa: E402

__all__ = [
    "Config",
    "load_config",
    "run_pipeline",
    "__version__",
]
