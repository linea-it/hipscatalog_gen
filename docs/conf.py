# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
from importlib import import_module
from importlib.metadata import version
from unittest.mock import MagicMock

# Make project package importable for autodoc (src layout)
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Mocks/autosummary toggles:
# - We mock heavy/optional dependencies so autodoc/autosummary can import
#   modules without requiring full runtime deps (dask/healpy/lsdb/etc.).
# - Pre-commit sets SPHINX_AUTOSUMMARY=0 to avoid generating stubs in hooks;
#   normal builds keep autosummary enabled by default.
# ---------------------------------------------------------------------------
MOCK_MODULES = [
    "dask",
    "dask.array",
    "dask.dataframe",
    "dask.delayed",
    "dask.distributed",
    "dask_jobqueue",
    "lsdb",
    "lsdb.catalog",
    "healpy",
    "mocpy",
    "astropy",
    "astropy.io",
    "astropy.io.fits",
    "astropy.io.votable",
    "astropy.table",
    "numpy",
    "pandas",
]

for _mod in MOCK_MODULES:
    try:
        import_module(_mod)
    except Exception:
        sys.modules[_mod] = MagicMock()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "hipscatalog-gen"
copyright = "2025, Luigi Silva @ LIneA"
author = "Luigi Lucas de Carvalho Silva"
release = version("hipscatalog-gen")
# for example take major/minor
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

extensions.append("nbsphinx")

# -- sphinx-copybutton configuration ----------------------------------------
extensions.append("sphinx_copybutton")
## sets up the expected prompt text from console blocks, and excludes it from
## the text that goes into the clipboard.
copybutton_exclude = ".linenos, .gp"
copybutton_prompt_text = ">> "

## lets us suppress the copy button on select code blocks.
copybutton_selector = "div:not(.no-copybutton) > div.highlight > pre"

templates_path = []
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# This assumes that sphinx-build is called from the root directory
master_doc = "index"
# Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = False
# Remove namespaces from class/method signatures
add_module_names = False

html_theme = "sphinx_rtd_theme"

# Generate autosummary stubs unless disabled via env (useful for pre-commit).
autosummary_generate = os.environ.get("SPHINX_AUTOSUMMARY", "1") != "0"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
# Keep autodoc_mock_imports aligned with manual mocks above for safety.
autodoc_mock_imports = MOCK_MODULES
