# hipscatalog-gen

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/linea-it/hipscatalog_gen/smoke-test.yml)](https://github.com/linea-it/hipscatalog_gen/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/linea-it/hipscatalog_gen/branch/main/graph/badge.svg)](https://codecov.io/gh/linea-it/hipscatalog_gen)

This project was created following the LINCC Frameworks Python Project Template (https://lincc-ppt.readthedocs.io/en/latest/).

-------------------------------------------------------------------------------

## Overview

hipscatalog-gen is a Python package for building HiPS-compliant catalog hierarchies from large astronomical tables using Dask and LSDB. It is inspired by and extends the logic of the CDS *Hipsgen-cat.jar* tool, providing a scalable and parallelized Python implementation suitable for large-scale workflows.


The pipeline supports three selection modes, configured in the YAML file under algorithm.selection_mode:

- **mag_global**   — global magnitude-complete selection.
- **score_global** — global selection driven by an arbitrary score/expression.
- **coverage**     — coverage-based selection per HEALPix or HATS cell.

Mode-specific parameters in the YAML use the prefixes:
`mg_*` (mag_global), `sg_*` (score_global), and `cov_*` (coverage).

-------------------------------------------------------------------------------

## Quick Start

    git clone https://github.com/linea-it/hipscatalog_gen.git
    cd hipscatalog_gen
    pip install -e .[dev]

Then run:

    python -m hipscatalog_gen.cli --config config.yaml

For conda-based setups, see "Environment Setup" below.

-------------------------------------------------------------------------------

## Environment Setup

All required packages are specified in the file environment.yaml at the ./examples/scripts folder.

    conda env create -f environment.yaml
    conda activate hipscatalog_gen_env

-------------------------------------------------------------------------------

## Configuration

The pipeline is fully configured through a YAML file.

A complete annotated template is provided in ./examples/configs folder as:

- config.template.yaml

To create your own configuration:

    cp config.template.yaml config.yaml

Then edit config.yaml to match your input catalog and selection preferences.
Additional examples are available under ./examples/configs/.

-------------------------------------------------------------------------------

## Running

The pipeline can be executed either as a Python library or from the command line.

### Run as a library

    from hipscatalog_gen import load_config, run_pipeline
    cfg = load_config("config.yaml")
    run_pipeline(cfg)

### Run from the command line

    python -m hipscatalog_gen.cli --config config.yaml

-------------------------------------------------------------------------------

## SLURM Cluster Usage

To run on a SLURM cluster:

1. Configure the cluster section in config.yaml with:
   - cluster.mode: slurm
   - cluster.n_workers, cluster.threads_per_worker, cluster.memory_per_worker
   - SLURM options under cluster.slurm (queue, account, etc.)

2. Use ./examples/scripts/run_hips.sbatch as an example batch script.

3. Submit and monitor:
       sbatch run_hips.sbatch
       squeue -u $USER

-------------------------------------------------------------------------------

## Output Structure

Each run generates a HiPS-compliant directory structure under output.out_dir:

- Norder*/Dir*/Npix*.tsv  → Per-depth tiles.
- Norder*/Allsky.tsv      → Optional all-sky tables.
- densmap_o<depth>.fits   → Density maps for all depths up to level_limit.
- Moc.fits / Moc.json     → Multi-Order Coverage maps.
- properties / metadata.xml → HiPS metadata descriptors.
- process.log / arguments  → Run logs and configuration snapshot.

-------------------------------------------------------------------------------

## Mode Comparison (Summary)

| Feature | Mag Global Mode | Score Global Mode | Coverage Mode |
|----------|----------------|-------------------|---------------|
| Partition basis | Global sample | Global sample | HEALPix/HATS cells (__icov__) |
| Main metric | Magnitude column | Arbitrary score/expression | Score + density profile |
| Completeness goal | Magnitude completeness | Score window completeness | Spatial balance / density control |
| Depth behavior | Histogram-based (mag_hist_nbins, n_1/n_2/n_3) | Histogram-based (score_hist_nbins, score_n_1/n_2/n_3) | Profile-driven (k_per_cov_*, targets_total_*) |
| Bias options | Not applicable | Not applicable | density_bias_mode / exponent |
| Typical use | Magnitude-complete catalogs | Global ranking/quality score selections | Uniform or density-aware selection |

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
