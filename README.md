# HiPS Catalog Pipeline (`hipscatalog_gen`)

This package builds HiPS-compliant catalog hierarchies from large input tables using Dask and LSDB.  
It reproduces the general directory and metadata structure recognized by Aladin, following the logic of the CDS *Hipsgen-cat.jar* tool, but implemented in Python for large-scale, parallelized workflows.

The pipeline supports two main selection modes, configured in the YAML under `algorithm.selection_mode`:

- `coverage`   → coverage-based selection per HEALPix / HATS cell.
- `mag_global` → global magnitude-complete selection.

---

## 1. Environment Setup

All required packages are specified in the `environment.yaml` file at the repository root.

### 1.1 Create and activate the environment

From the repository root:

    conda env create -f environment.yaml
    conda activate hipsenv

(adjust the environment name if it differs in `environment.yaml`)

---

## 2. Using `hipscatalog_gen` from the source tree

There is no separate installation step required if you run from the source tree.

From the repository root (the directory that contains the `hipscatalog_gen/` package):

    export PYTHONPATH=$PWD:$PYTHONPATH

Now Python can import `hipscatalog_gen` directly from the source tree.

---

## 3. Configuration Files

The pipeline is driven by a YAML configuration file.

- A fully annotated template, kept in sync with the code, is provided as:

  - `config.template.yaml`

- To create your own configuration:

  1. Copy the template:

         cp config.template.yaml config.yaml

  2. Edit `config.yaml` to match your catalog and desired selection settings.

There may also be additional example configurations under subdirectories in the repository (for example `configs/`), illustrating typical use cases.

---

## 4. Running the Pipeline

The pipeline can be executed either as a library or via the command line interface.

### 4.1 Run as a library (Python)

Example:

    from hipscatalog_gen import load_config, run_pipeline

    cfg = load_config("config.yaml")  # your customized config
    run_pipeline(cfg)

This is the recommended approach when integrating `hipscatalog_gen` into notebooks or larger Python workflows.

### 4.2 Run from the command line

The CLI entry point lives in `hipscatalog_gen.cli` and accepts a single required argument: `--config`.

From the repository root:

    python -m hipscatalog_gen.cli --config config.yaml

This will:

- Read the configuration.
- Build the input collection (Parquet/CSV/TSV via Dask, or HATS via LSDB).
- Compute densmaps and MOC.
- Run the selection (coverage or mag_global).
- Write HiPS tiles and metadata under `output.out_dir`.

### 4.3 Running on a SLURM cluster

To run on a SLURM cluster:

1. Configure the `cluster` section in your `config.yaml` with:

   - `cluster.mode: slurm`
   - Appropriate values for `cluster.n_workers`, `cluster.threads_per_worker`, `cluster.memory_per_worker`
   - SLURM-specific options under `cluster.slurm` (queue, account, job directives, diagnostics mode)

2. Create a batch script. Use ```run_hips.sbatch```script as an example.

3. Submit and monitor:

       sbatch run_hips.sbatch
       squeue -u $USER

---

## 5. Output Structure

For a given `output.out_dir`, the pipeline creates a HiPS directory structure including:

- `Norder*/Dir*/Npix*.tsv`  
  Per-depth tiles (one TSV per HEALPix cell), containing the selected rows and the configured columns.

- `Norder*/Allsky.tsv`  
  Optional all-sky tables (for some depths), aggregating all tiles at a given order.

- `densmap_o<depth>.fits`  
  Density maps (per-depth HEALPix counts) for all depths `0..algorithm.level_limit`.

- `Moc.fits`, `Moc.json`  
  Multi-Order Coverage maps derived from the densmap at `algorithm.level_coverage`.

- `properties`, `metadata.xml`  
  HiPS metadata descriptors recognized by Aladin and other HiPS clients.

- `process.log`  
  Main log file for the run, including configuration echoing and progress messages.

- `arguments`  
  Text file containing a compact snapshot of the effective arguments and configuration, for reproducibility.

No extra YAML snapshot is written by the pipeline; configuration is documented via `arguments` and `process.log`.

---

## 6. Configuration Overview

All runtime configuration is controlled via a YAML file. The main top-level sections are:

- `input:`  
  How the input catalog(s) are read.
  - Supported formats: `parquet`, `csv`, `tsv`, `hats`.
  - Paths (with wildcards) defined in `input.paths`.

- `columns:`  
  How RA/DEC are mapped and how the score is defined.
  - `ra`, `dec` must be in degrees; the pipeline normalizes RA and validates DEC.
  - `score` is a Python expression (or a single column) used in coverage mode.
  - `keep` lists extra columns to preserve; if omitted, only the minimal required subset is kept (RA/DEC, score dependencies, and `mag_column` in mag_global mode).

- `algorithm:`  
  Selection mode and per-depth logic.
  - `selection_mode`: `coverage` or `mag_global`.
  - `level_limit`, `level_coverage`, `coverage_order` control HiPS depth and coverage maps.
  - Coverage mode–specific parameters include:
    - `use_hats_as_coverage`, `order_desc`
    - `density_mode`, `k_per_cov_initial`, `targets_total_initial`, `density_exp_base`
    - `density_bias_mode`, `density_bias_exponent`
    - `k_per_cov_per_level`, `targets_total_per_level`
    - `fractional_mode`, `fractional_mode_logic`, `tie_buffer`
  - Mag-global–specific parameters include:
    - `mag_column`, `mag_min`, `mag_max`
    - `mag_hist_nbins`
    - `n_1`, `n_2`, `n_3` (optional global targets per depth)

- `cluster:`  
  Dask cluster configuration.
  - `mode`: `local` or `slurm`.
  - `n_workers`, `threads_per_worker`, `memory_per_worker`.
  - `persist_ddfs` and `avoid_computes_wherever_possible` for memory vs. throughput trade-offs.
  - Under `slurm`, queue/account and SLURM directives, plus diagnostics options.

- `output:`  
  HiPS output directory and metadata.
  - `out_dir`: root directory for the HiPS hierarchy.
  - `cat_name`, `target`, `creator_did`, `obs_title` for HiPS metadata.

The full annotated configuration template, kept in sync with the current implementation, is:

- `config.template.yaml`

Use that file as the authoritative reference for per-parameter behavior and defaults.

---

## 7. Mode Comparison (Conceptual Summary)

| Feature | Coverage Mode (`coverage`) | Mag Global Mode (`mag_global`) |
|----------|-----------------------------|--------------------------------|
| Partition basis | HEALPix or HATS coverage cells (`__icov__`) | Global sample (no explicit coverage cells) |
| Main selection metric | Score (`columns.score`) + density profile | Magnitude (`algorithm.mag_column`) |
| Completeness goal | Spatial balance / density control | Magnitude completeness |
| Depth behavior | Profile-driven (`density_mode`, `k_per_cov_*`, `targets_total_*`) | Histogram-based (`mag_hist_nbins`, `n_1/n_2/n_3`) |
| Bias options | `density_bias_mode`, `density_bias_exponent` | Not applicable (no density bias) |
| Typical use | Uniform or density-aware spatial downsampling | Brightness-limited, magnitude-complete catalogs |


---

## 8. Acknowledgment

This work was inspired by the HiPS Catalog tools developed at the CDS,  
whose design and public documentation provided valuable guidance for this independent reimplementation.

References:

- CDS (Strasbourg Astronomical Data Center): https://cds.unistra.fr/
- HiPSgen-cat (official Java implementation): https://aladin.cds.unistra.fr/hips/Hipsgen-cat.gml
