hipscatalog-gen: HiPS catalog pipeline
========================================================================================

hipscatalog-gen builds HiPS-compliant catalog hierarchies from large astronomical tables using Dask and LSDB. It extends ideas from the CDS ``Hipsgen-cat.jar`` in a scalable Python pipeline suited for survey-scale workflows.

Overview
--------

- Three selection modes (``algorithm.selection_mode``):

  - ``mag_global``: magnitude-complete selection (see ``algorithm.mag_global.*``).
  - ``mag_global`` hist_peak defaults: when no ``mag_min``/``mag_max`` are provided, the histogram range clips global min/max to [-2, 40] (min clipped to >= -2; max from the peak within [-2, min(global_max, 40)]).
  - ``score_global``: selection driven by an arbitrary score/expression (see ``algorithm.score_global.*``).
  - ``score_density_hybrid``: density-driven depths 1..``density_up_to_depth`` (default 4) with score-driven remainder (see ``algorithm.score_density_hybrid.*``).

- Runs locally or on SLURM-backed Dask clusters; outputs full HiPS layouts (tiles, all-sky, MOC, metadata, density maps).

Quick start
-----------

If you do not have Conda yet, install it first using the official docs:

- Conda install guide: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
- Miniconda install guide: https://www.anaconda.com/docs/getting-started/miniconda/install

.. code-block:: console

   conda create -n hipscatalog-gen "python>=3.11"
   conda activate hipscatalog-gen
   pip install hipscatalog-gen

   curl -O https://raw.githubusercontent.com/linea-it/hipscatalog_gen/main/examples/configs/config.template.yaml
   cp config.template.yaml config.yaml

   hipscatalog-gen --config config.yaml

Developer install
-----------------

.. code-block:: console

   git clone https://github.com/linea-it/hipscatalog_gen.git
   cd hipscatalog_gen
   conda create -n hipscatalog-gen-dev "python>=3.11"
   conda activate hipscatalog-gen-dev
   pip install -e .[dev]

Optional: expose the env as a Jupyter kernel:

.. code-block:: console

   python -m ipykernel install --user --name hipscatalog-gen --display-name "hipscatalog-gen"

Configuration
-------------

- Start from ``examples/configs/config.template.yaml`` (copy to ``config.yaml``). Adjust input paths, column mapping, and selection parameters inside the per-mode blocks under ``algorithm``. More examples live under ``examples/configs/``.
- When installed from PyPI, fetch the template with ``curl -O https://raw.githubusercontent.com/linea-it/hipscatalog_gen/main/examples/configs/config.template.yaml``.
- Cluster memory policy is fixed: the pipeline does not persist large intermediate DataFrames and avoids early large computes whenever possible.
- ``cluster.low_memory_mode`` is deprecated (warning only, no effect). ``cluster.persist_ddfs`` and ``cluster.avoid_computes_wherever_possible`` are deprecated and ignored.
- Streamed stage-2 writes require an active ``dask.distributed`` client and execute bucket processing on workers (driver remains orchestration-only).
- Stage-2 stream merge uses bounded fan-in (auto-tuned from worker concurrency + ``RLIMIT_NOFILE``) to reduce ``EMFILE`` risk on large runs.

Run the pipeline
----------------

Library:

.. code-block:: python

   from hipscatalog_gen.config import load_config, load_config_from_dict, display_available_configs
   from hipscatalog_gen.pipeline.main import run_pipeline

   cfg = load_config("config.yaml")
   run_pipeline(cfg)

CLI:

.. code-block:: console

   hipscatalog-gen --config config.yaml
   # or: python -m hipscatalog_gen.cli --config config.yaml

No dedicated ``sbatch`` wrapper script is required. For HPC usage, set
``cluster.mode: slurm`` in the YAML and run the same command above.

Outputs (HiPS layout)
---------------------

- ``Norder*/Dir*/Npix*.tsv``: per-depth tiles; optional ``Norder*/Allsky.tsv``.
- ``densmap_o<depth>.fits``: density maps up to ``algorithm.level_limit``.
- ``Moc.fits`` / ``Moc.json``: MOC maps.
- ``properties`` and ``metadata.xml``: HiPS metadata descriptors.
- ``process.log`` and ``arguments``: logs and config snapshot.
- Existing ``output.out_dir`` causes an error; set ``output.overwrite: true`` to clear it before writing.

Navigation
----------

- Reference (generated from code, grouped by domain).
- Example notebooks, including an intro and a simple pipeline run.

.. toctree::
   :hidden:

   Home page <self>
   Reference <reference/index>
   Notebooks <notebooks>
