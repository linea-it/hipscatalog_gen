# hipscatalog-gen

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/linea-it/hipscatalog_gen/smoke-test.yml)](https://github.com/linea-it/hipscatalog_gen/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/linea-it/hipscatalog_gen/branch/main/graph/badge.svg)](https://codecov.io/gh/linea-it/hipscatalog_gen)

This project was created following the LINCC Frameworks Python Project Template (https://lincc-ppt.readthedocs.io/en/latest/).

-------------------------------------------------------------------------------

## Overview

hipscatalog-gen is a Python package for building HiPS-compliant catalog hierarchies from large astronomical tables using Dask and LSDB. It is inspired by and extends the logic of the CDS *Hipsgen-cat.jar* tool, providing a scalable and parallelized Python implementation suitable for large-scale workflows. Documentation: https://linea-it.github.io/hipscatalog_gen/


The pipeline supports three selection modes, configured in the YAML file under algorithm.selection_mode:

- **mag_global**   — global magnitude-complete selection.
- **score_global** — global selection driven by an arbitrary score/expression.
- **score_density_hybrid** — density-driven depths 1–3 with score-based distribution afterwards.

-------------------------------------------------------------------------------

## Quick Start (PyPI)

Install from PyPI into a fresh environment and run with a config file:

    python -m venv .venv          # or use conda
    source .venv/bin/activate
    pip install hipscatalog-gen

Fetch the example template and adapt it to your catalog:

    curl -O https://raw.githubusercontent.com/linea-it/hipscatalog_gen/main/examples/configs/config.template.yaml
    cp config.template.yaml config.yaml

Run the pipeline:

    hipscatalog-gen --config config.yaml
    # or: python -m hipscatalog_gen.cli --config config.yaml

--------------------------------------------------------------------------------

## Developer Install

For local development (editable install + tooling):

    git clone https://github.com/linea-it/hipscatalog_gen.git
    cd hipscatalog_gen
    conda create -n hipscatalog-gen python=3.13
    conda activate hipscatalog-gen
    pip install -e .[dev]

Optionally expose the env as a Jupyter kernel:

    python -m ipykernel install --user --name hipscatalog-gen --display-name "hipscatalog-gen"

-------------------------------------------------------------------------------

## Configuration

The pipeline is fully configured through a YAML file.

A complete annotated template is provided in ./examples/configs folder as:

- config.template.yaml

When installed from PyPI, download the template directly:

    curl -O https://raw.githubusercontent.com/linea-it/hipscatalog_gen/main/examples/configs/config.template.yaml

To create your own configuration:

    cp config.template.yaml config.yaml

Then edit config.yaml to match your input catalog and selection preferences.
Additional examples are available under ./examples/configs/.

Selection modes live under ``algorithm.selection_mode``:

- ``mag_global``, ``score_global``, ``score_density_hybrid``.
Mode-specific parameters live inside blocks ``algorithm.mag_global``, ``algorithm.score_global``, and
``algorithm.score_density_hybrid`` (with optional shared defaults in ``algorithm.selection_defaults``).

-------------------------------------------------------------------------------

## Running

The pipeline can be executed either as a Python library or from the command line.

### Run as a library

    from hipscatalog_gen.config import load_config, load_config_from_dict, display_available_configs
    from hipscatalog_gen.pipeline.main import run_pipeline

    cfg = load_config("config.yaml")
    run_pipeline(cfg)

### Run from the command line

List available selection modes:

    hipscatalog-gen --list-modes

Run with a config file:

    hipscatalog-gen --config config.yaml
    # or: python -m hipscatalog_gen.cli --config config.yaml

Validate a config without running:

    hipscatalog-gen --check-config config.yaml

Enable JSON logs (process.jsonl) via CLI flag (when running the pipeline):

    hipscatalog-gen --config config.yaml --json-logs

Summarize an existing telemetry.json:

    hipscatalog-gen --telemetry /path/to/telemetry.json

## Output Structure

Each run generates a HiPS-compliant directory structure under output.out_dir:

- Norder*/Dir*/Npix*.tsv  → Per-depth tiles.
- Norder*/Allsky.tsv      → Optional all-sky tables.
- densmap_o<depth>.fits   → Density maps for all depths up to level_limit.
- Moc.fits / Moc.json     → Multi-Order Coverage maps.
- properties / metadata.xml → HiPS metadata descriptors.
- process.log / arguments  → Run logs and configuration snapshot (optional process.jsonl when `--json-logs`).
- telemetry.json          → Run summary with per-stage durations and input/output counts.
- Existing ``output.out_dir`` causes an error; set ``output.overwrite: true`` to clear it before writing.

-------------------------------------------------------------------------------

## Mode Summary

- **mag_global**: magnitude-complete slices across all depths.
- mag_global hist_peak default bounds: when `adaptive_range=hist_peak` and `mag_min`/`mag_max` are not provided, the histogram range clips the global min/max to [-2, 40] (mag_min clipped to >= -2; mag_max from the peak within [-2, min(global_max, 40)]).
- **score_global**: score-based slices across all depths.
- **score_density_hybrid**: density-driven tiles for depths 1–3, then score slices for deeper levels.
- Ordering and ties: `order_desc` controls ascending/descending (default ascending); optional `tie_column` breaks ties before falling back to RA/DEC.
- Invalids: `keep_invalid_values` (per mode or in `selection_defaults`) can map NaN/Inf to a sentinel when `adaptive_range=complete`, sending them to the last slice; rejected for `hist_peak`.

-------------------------------------------------------------------------------

## Development and Contributing

This project follows the LINCC Frameworks Python Project Template.

To set up a development environment:

    pip install -e .[dev]
    pre-commit install
    pytest

Contributions, bug reports, and pull requests are welcome via GitHub Issues: https://github.com/linea-it/hipscatalog_gen/issues

-------------------------------------------------------------------------------

## Acknowledgments

This project acknowledges the foundational work of the **CDS HiPS Catalog Tool** (Hipsgen-cat.jar) developed by the Strasbourg Astronomical Data Center (Unistra/CNRS, 2016), which inspired aspects of the software design.
More information: https://aladin.cds.unistra.fr/hips/HipsCat.gml.

The mag-global mode builds on an idea originally suggested by **Julia Gschwend**.

-------------------------------------------------------------------------------

## Citation

If you use this package in your research, please cite:

Silva, L. L. C., et al. (2025). *hipscatalog-gen: A Python HiPS Catalog Pipeline*.
LIneA – Laboratório Interinstitucional de e-Astronomia.
Available at: https://github.com/linea-it/hipscatalog_gen

-------------------------------------------------------------------------------

## License

This project is licensed under the MIT License. See the LICENSE file for details.
