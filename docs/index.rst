hipscatalog-gen: HiPS catalog pipeline
========================================================================================

hipscatalog-gen builds HiPS-compliant catalog hierarchies from large astronomical tables using Dask and LSDB. It extends ideas from the CDS ``Hipsgen-cat.jar`` in a scalable Python pipeline suited for survey-scale workflows.

Overview
--------

- Three selection modes (``algorithm.selection_mode``):

  - ``mag_global``: magnitude-complete selection (see ``algorithm.mag_global.*``).
  - ``score_global``: selection driven by an arbitrary score/expression (see ``algorithm.score_global.*``).
  - ``score_density_hybrid``: density-driven depths 1–3 with score-driven remainder (see ``algorithm.score_density_hybrid.*``).

- Runs locally; outputs full HiPS layouts (tiles, all-sky, MOC, metadata, density maps).

Quick start
-----------

.. code-block:: console

   git clone https://github.com/linea-it/hipscatalog_gen.git
   cd hipscatalog_gen
   conda create -n hipscatalog-gen python=3.13
   conda activate hipscatalog-gen
   pip install -e .[dev]
   python -m hipscatalog_gen.cli --config config.yaml

Environment (conda)
-------------------

.. code-block:: console

   conda create -n hipscatalog-gen python=3.13
   conda activate hipscatalog-gen
   pip install -e .[dev]

Optional: expose the env as a Jupyter kernel:

.. code-block:: console

   python -m ipykernel install --user --name hipscatalog-gen --display-name "hipscatalog-gen"

Configuration
-------------

- Start from ``examples/configs/config.template.yaml`` (copy to ``config.yaml``). Adjust input paths, column mapping, and selection parameters inside the per-mode blocks under ``algorithm``. More examples live under ``examples/configs/``.

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

   python -m hipscatalog_gen.cli --config config.yaml

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

- API reference (generated from code).
- Example notebooks, including an intro and a simple pipeline run.

.. toctree::
   :hidden:

   Home page <self>
   API Reference <api/index>
   Notebooks <notebooks>
